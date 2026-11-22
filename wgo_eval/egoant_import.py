"""Convert EgoANT segmentation outputs (one JSON per episode) into segpreds.jsonl.

    python -m wgo_eval.egoant_import --pred-dir <egoant pred dir> --root ../data/wgo \
        --out runs/egoant-ref/segpreds.jsonl

Used to check score_seg against the reference scorer: scoring EgoANT's released HomER
predictions must reproduce the numbers EgoANT reports (e.g. 79 matches / 308 predicted /
470 gold, Segment F1 0.2031 for the run named in its README).

Lookup per episode follows EgoANT score_segmentation.py: <id>.json, homer/<id>.json,
<id>/merge_judge_annotations.json. A file may hold {"segments": [...]} or a bare list.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, write_jsonl


def find_pred(pred_dir: Path, episode_id: str) -> Path | None:
    for rel in (f"{episode_id}.json", f"homer/{episode_id}.json", f"{episode_id}/merge_judge_annotations.json"):
        if (pred_dir / rel).exists():
            return pred_dir / rel
    return None


def load_segments(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    items = obj.get("segments", []) if isinstance(obj, dict) else obj
    out = []
    for s in items or []:
        a = float(s.get("start_sec", s.get("start", 0)))
        b = float(s.get("end_sec", s.get("end", 0)))
        out.append({"start_sec": a, "end_sec": b, "subtask": str(s.get("subtask") or s.get("label") or "")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--subset", default="homer")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ids = sorted({s["episode_id"] for s in read_jsonl(args.root / "segments.jsonl")
                  if not args.subset or s["subset"] == args.subset})
    rows, missing = [], []
    for ep in ids:
        path = find_pred(args.pred_dir, ep)
        if path is None:
            missing.append(ep)
            continue
        segs = load_segments(path)
        rows.append({"episode_id": ep, "segments": segs, "failed": not segs, "source": str(path)})
    write_jsonl(args.out, rows)
    print(f"EGOANT_IMPORT|episodes={len(ids)} converted={len(rows)} missing={len(missing)} {missing[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
