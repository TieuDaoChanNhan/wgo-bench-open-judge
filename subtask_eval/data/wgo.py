"""WGO-Bench as an eval set: download at a pinned revision, then load episodes with gold segments.

    python -m subtask_eval.data.wgo --root data/wgo

Writes:
    <root>/videos/<subset>/<episode_id>.mp4
    <root>/segments.jsonl   one row per gold segment:
        segment_id, episode_id, subset, video, start_sec, end_sec, task, label
    <root>/manifest.json    revision, counts per subset

Annotators get episodes through `model_view`, which drops the gold segments.
Download code adapted from EgoANT scripts/prepare_wgo_bench.py (MIT, see THIRD_PARTY.md).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..common import read_jsonl, write_jsonl

DATASET = "macrodata/WGO-Bench"
REVISION = "1c86a62e5302"  # pinned dataset commit; earlier commits carry older HomER annotations
PARQUET = "data/annotations.parquet"

# Which FineRobotics data category each subset stands for.
CATEGORY = {"homer": "ego", "galaxea": "robot", "robointer_droid": "robot", "droid": "robot"}


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


def load_episodes(root: Path, subset: str | None = None) -> list[dict[str, Any]]:
    """Episodes in id order: episode_id, subset, category, video (relative to root), task, gold."""
    episodes: dict[str, dict[str, Any]] = {}
    for s in sorted(read_jsonl(Path(root) / "segments.jsonl"), key=lambda s: s["segment_id"]):
        if subset and s["subset"] != subset:
            continue
        ep = episodes.setdefault(s["episode_id"], {
            "episode_id": s["episode_id"], "subset": s["subset"], "category": CATEGORY.get(s["subset"], "unknown"),
            "video": s["video"], "task": s.get("task", ""), "gold": []})
        ep["gold"].append({"start_sec": s["start_sec"], "end_sec": s["end_sec"], "label": s["label"]})
    return [episodes[k] for k in sorted(episodes)]


def model_view(episode: dict[str, Any]) -> dict[str, Any]:
    """The only fields an annotator may see. Gold segments are dropped here."""
    return {k: episode[k] for k in ("episode_id", "video", "task")}


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
