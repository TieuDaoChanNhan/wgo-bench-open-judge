"""Download WGO-Bench at a pinned revision and write one mp4 per episode plus segments.jsonl.

    python -m wgo_eval.wgo_prepare --root ../data/wgo

Writes:
    <root>/videos/<subset>/<episode_id>.mp4
    <root>/segments.jsonl   one row per gold segment:
        segment_id, episode_id, subset, video, start_sec, end_sec, task, label
    <root>/manifest.json    revision, counts per subset

`label` is the gold label. The labeler never reads it (see label.py).
Adapted from EgoANT scripts/prepare_wgo_bench.py (MIT, see THIRD_PARTY.md).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import write_jsonl

DATASET = "macrodata/WGO-Bench"
REVISION = "1c86a62e5302"  # pinned dataset commit; earlier commits carry older HomER annotations
PARQUET = "data/annotations.parquet"


def subset_of(episode_id: str) -> str:
    """'homer_12' -> 'homer'."""
    head, sep, tail = episode_id.rpartition("_")
    return head.lower() if sep and head and tail.isdigit() else "unknown"


def parse_segments(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value)
    return [
        {"start_sec": float(s["start_sec"]), "end_sec": float(s["end_sec"]), "subtask": str(s["subtask"])}
        for s in value or []
    ]


def video_bytes(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, dict) and value.get("bytes") is not None:
        return bytes(value["bytes"])
    raise TypeError(f"unsupported video field: {type(value)!r}")


def segment_rows(episode_id: str, instruction: str, segments: list[dict[str, Any]], video: str) -> list[dict[str, Any]]:
    subset = subset_of(episode_id)
    return [
        {
            "segment_id": f"{episode_id}/{i:03d}",
            "episode_id": episode_id,
            "subset": subset,
            "video": video,
            "start_sec": s["start_sec"],
            "end_sec": s["end_sec"],
            "task": instruction,
            "label": s["subtask"],
        }
        for i, s in enumerate(segments)
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--parquet", type=Path, default=None, help="use a local copy instead of downloading")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    path = args.parquet
    if path is None:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(DATASET, PARQUET, repo_type="dataset", revision=args.revision))

    rows: list[dict[str, Any]] = []
    episodes: Counter[str] = Counter()
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=4, columns=["id", "video", "instruction", "segments"]):
        for rec in batch.to_pylist():
            episode_id = str(rec["id"])
            subset = subset_of(episode_id)
            rel = f"videos/{subset}/{episode_id}.mp4"
            out = args.root / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(video_bytes(rec["video"]))
            rows.extend(segment_rows(episode_id, str(rec.get("instruction") or ""), parse_segments(rec["segments"]), rel))
            episodes[subset] += 1

    write_jsonl(args.root / "segments.jsonl", rows)
    seg_counts = Counter(r["subset"] for r in rows)
    manifest = {
        "dataset": DATASET,
        "revision": args.revision,
        "episodes": dict(sorted(episodes.items())),
        "segments": dict(sorted(seg_counts.items())),
        "total_segments": len(rows),
    }
    (args.root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
