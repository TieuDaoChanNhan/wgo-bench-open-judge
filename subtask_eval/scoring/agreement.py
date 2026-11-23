"""How far two sets of label verdicts agree: judge vs judge, judge vs reference, judge vs human.

    # reference pairs: 79 HomER pairs judged by gemini-3.5-flash, released by EgoANT at b27ea59f5f
    # (reference_outputs/wgo_homer_v5_1/scores/semantic_e2e_gemini35.json)
    subtask-eval agree extract-egoant --egoant-json <file> --root data/wgo --out runs/judge-agree

    # judge those pairs (writes into the same directory), then compare any two verdict files
    subtask-eval judge --run runs/judge-agree --model Qwen/Qwen3-32B-AWQ --out runs/judge-agree/judged_awq.jsonl
    subtask-eval agree compare --a runs/judge-agree/pairs.jsonl --b runs/judge-agree/judged_awq.jsonl

A verdict file is JSONL with an "id" and a boolean in "match" (judge output), "reference_match"
(extracted reference) or "human_match" (manual review). Only ids present in both files are compared.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..common import read_jsonl, write_jsonl
from ..data.wgo import load_episodes
from .score import pair_id

VERDICT_KEYS = ("match", "reference_match", "human_match")


def verdicts(path: Path) -> dict[str, bool]:
    out = {}
    for r in read_jsonl(path):
        for k in VERDICT_KEYS:
            if isinstance(r.get(k), bool):
                out[r["id"]] = r[k]
                break
    return out


def agreement(a: list[bool], b: list[bool]) -> dict[str, Any]:
    """Raw agreement, Cohen's kappa and the 2x2 table (a rows, b columns)."""
    n = len(a)
    tt = sum(x and y for x, y in zip(a, b))
    tf = sum(x and not y for x, y in zip(a, b))
    ft = sum((not x) and y for x, y in zip(a, b))
    ff = n - tt - tf - ft
    po = (tt + ff) / n if n else 0.0
    pa, pb = ((tt + tf) / n, (tt + ft) / n) if n else (0.0, 0.0)
    pe = pa * pb + (1 - pa) * (1 - pb)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return {"n": n, "agreement": round(po, 4), "kappa": round(kappa, 4),
            "accept_rate_a": round(pa, 4), "accept_rate_b": round(pb, 4),
            "table": {"a_true_b_true": tt, "a_true_b_false": tf, "a_false_b_true": ft, "a_false_b_false": ff}}


def extract_egoant(egoant: dict[str, Any], tasks: dict[str, str]) -> list[dict[str, Any]]:
    """Judge pairs carrying EgoANT's reference verdict. EgoANT gave its judge the episode
    instruction but did not store it, so `tasks` (episode_id -> instruction) fills it in."""
    rows = []
    for ep in egoant.get("per_episode", []):
        for p in ep.get("pairs", []):
            if not isinstance(p.get("semantic_match"), bool):
                continue
            rows.append({"id": pair_id(ep["id"], p["pred_i"], p["gold_j"]), "gold_label": p["gold_label"],
                         "pred_label": p["pred_label"], "task": tasks.get(ep["id"], ""),
                         "reference_match": p["semantic_match"], "reference_model": p.get("judge_model")})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract-egoant")
    e.add_argument("--egoant-json", type=Path, required=True)
    e.add_argument("--root", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True, help="directory; pairs.jsonl is written there")
    c = sub.add_parser("compare")
    c.add_argument("--a", type=Path, required=True)
    c.add_argument("--b", type=Path, required=True)
    c.add_argument("--disagreements", type=Path, default=None, help="write the pairs the two disagree on")
    c.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.cmd == "extract-egoant":
        tasks = {e["episode_id"]: e["task"] for e in load_episodes(args.root)}
        rows = extract_egoant(json.loads(args.egoant_json.read_text(encoding="utf-8")), tasks)
        write_jsonl(args.out / "pairs.jsonl", rows)
        print(f"AGREE|pairs={len(rows)} reference_true={sum(r['reference_match'] for r in rows)} "
              f"with_task={sum(bool(r['task']) for r in rows)}")
        return 0

    va, vb = verdicts(args.a), verdicts(args.b)
    ids = sorted(set(va) & set(vb))
    result = {"a": str(args.a), "b": str(args.b), **agreement([va[i] for i in ids], [vb[i] for i in ids]),
              "only_in_a": len(set(va) - set(vb)), "only_in_b": len(set(vb) - set(va))}
    print(json.dumps(result, indent=2))
    if args.disagreements:
        labels = {r["id"]: r for r in read_jsonl(args.a) + read_jsonl(args.b) if "gold_label" in r}
        write_jsonl(args.disagreements, [{"id": i, "a": va[i], "b": vb[i],
                                          "gold_label": labels.get(i, {}).get("gold_label"),
                                          "pred_label": labels.get(i, {}).get("pred_label")}
                                         for i in ids if va[i] != vb[i]])
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
