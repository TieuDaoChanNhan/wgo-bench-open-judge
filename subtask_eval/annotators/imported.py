"""Import segmentations produced elsewhere into the common preds.jsonl format.

    python -m subtask_eval.annotators.imported --format egoant --src <egoant pred dir> \
        --root data/wgo --subset homer --out runs/egoant-ref

Formats:
    egoant  one JSON per episode, looked up as EgoANT score_segmentation.py does:
            <id>.json, homer/<id>.json, <id>/merge_judge_annotations.json;
            each holds {"segments": [...]} or a bare list of {start_sec, end_sec, subtask}.
    jsonl   one row per episode: {"episode_id", "segments": [{start_sec, end_sec, label|subtask}]}.

Used to check the scorer against the reference: scoring EgoANT's released HomER predictions
must reproduce the counts EgoANT reports (79 matched / 308 predicted / 470 gold, F1 0.2031).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..common import read_jsonl, write_jsonl
from ..data.wgo import load_episodes
from .base import clean_segments, new_cost


def find_egoant(src: Path, episode_id: str) -> Path | None:
    for rel in (f"{episode_id}.json", f"homer/{episode_id}.json", f"{episode_id}/merge_judge_annotations.json"):
        if (src / rel).exists():
            return src / rel
    return None


def load_egoant(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    items = obj.get("segments", []) if isinstance(obj, dict) else obj
    return [{"start_sec": float(s.get("start_sec", s.get("start", 0))),
             "end_sec": float(s.get("end_sec", s.get("end", 0))),
             "label": str(s.get("subtask") or s.get("label") or "")} for s in items or []]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--format", choices=["egoant", "jsonl"], required=True)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--subset", default=None)
    ap.add_argument("--out", type=Path, required=True, help="run directory")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    ids = [e["episode_id"] for e in load_episodes(args.root, args.subset)]
    given = {r["episode_id"]: r for r in read_jsonl(args.src)} if args.format == "jsonl" else {}
    rows, missing = [], []
    for ep in ids:
        if args.format == "egoant":
            path = find_egoant(args.src, ep)
            segs = load_egoant(path) if path else None
        else:
            segs = given[ep].get("segments") if ep in given else None
        if segs is None:
            missing.append(ep)
            continue
        # EgoANT predictions are kept exactly as stored (including any zero-length segment), so the
        # counts reproduce the reference scorer; other sources get the common clean-up.
        if args.format != "egoant":
            segs = clean_segments(segs)
        rows.append({"episode_id": ep, "segments": segs, "failed": not segs, "cost": new_cost()})
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "preds.jsonl", rows)
    meta = {"config": {"annotator": f"imported:{args.format}", "name": args.name or args.out.name, "src": str(args.src)},
            "sessions": []}
    (args.out / "run.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"IMPORT|episodes={len(ids)} converted={len(rows)} missing={len(missing)} {missing[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
