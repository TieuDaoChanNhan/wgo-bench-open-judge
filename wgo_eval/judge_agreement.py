"""Check an open judge against Gemini-3.5-Flash verdicts released by EgoANT.

EgoANT ships 79 HomER label pairs judged by gemini-3.5-flash with the WGO rubric
(reference_outputs/wgo_homer_v5_1/scores/semantic_e2e_gemini35.json at commit b27ea59f5f).

    python -m wgo_eval.judge_agreement extract --egoant-json <that file> --out runs/agree/pairs.jsonl
    python -m wgo_eval.judge --pairs runs/agree/pairs.jsonl --out runs/agree/judged.jsonl --model <judge> ...
    python -m wgo_eval.judge_agreement report --pairs runs/agree/pairs.jsonl --judged runs/agree/judged.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, write_jsonl


def extract_pairs(egoant: dict[str, Any], tasks: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Judge pairs with the reference verdict. `tasks` maps episode_id to the episode instruction,
    which EgoANT gave its judge but did not store in the file."""
    tasks = tasks or {}
    rows = []
    for ep in egoant.get("per_episode", []):
        for p in ep.get("pairs", []):
            if not isinstance(p.get("semantic_match"), bool):
                continue
            rows.append({"id": f"{ep['id']}/p{p['pred_i']:03d}-g{p['gold_j']:03d}", "gold_label": p["gold_label"],
                         "pred_label": p["pred_label"], "task": tasks.get(ep["id"], ""),
                         "reference_match": p["semantic_match"], "reference_model": p.get("judge_model")})
    return rows


def agreement(reference: list[bool], ours: list[bool]) -> dict[str, Any]:
    """Raw agreement, Cohen's kappa and the 2x2 table (reference rows, ours columns)."""
    n = len(reference)
    tt = sum(r and o for r, o in zip(reference, ours))
    tf = sum(r and not o for r, o in zip(reference, ours))
    ft = sum((not r) and o for r, o in zip(reference, ours))
    ff = n - tt - tf - ft
    po = (tt + ff) / n if n else 0.0
    p_ref, p_ours = (tt + tf) / n if n else 0.0, (tt + ft) / n if n else 0.0
    pe = p_ref * p_ours + (1 - p_ref) * (1 - p_ours)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return {"n": n, "agreement": round(po, 4), "kappa": round(kappa, 4),
            "table": {"ref_true_ours_true": tt, "ref_true_ours_false": tf,
                      "ref_false_ours_true": ft, "ref_false_ours_false": ff}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--egoant-json", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--root", type=Path, default=None, help="wgo_prepare root, to fill in episode instructions")
    r = sub.add_parser("report")
    r.add_argument("--pairs", type=Path, required=True)
    r.add_argument("--judged", type=Path, required=True)
    args = ap.parse_args()

    if args.cmd == "extract":
        tasks = {s["episode_id"]: s["task"] for s in read_jsonl(args.root / "segments.jsonl")} if args.root else {}
        rows = extract_pairs(json.loads(args.egoant_json.read_text(encoding="utf-8")), tasks)
        write_jsonl(args.out, rows)
        print(f"AGREE|pairs={len(rows)} reference_true={sum(r['reference_match'] for r in rows)} "
              f"with_task={sum(bool(r['task']) for r in rows)}")
        return 0

    ref = {p["id"]: p["reference_match"] for p in read_jsonl(args.pairs)}
    ours = {j["id"]: bool(j["match"]) for j in read_jsonl(args.judged)}
    ids = sorted(set(ref) & set(ours))
    result = agreement([ref[i] for i in ids], [ours[i] for i in ids])
    result["missing"] = len(set(ref) - set(ours))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
