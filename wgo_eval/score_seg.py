"""Segment F1 and end-to-end F1 of a segmenter, with episode-bootstrap intervals.

    # 1. segment F1, and the matched pairs for the label judge
    python -m wgo_eval.score_seg --root ../data/wgo --segpreds runs/x/segpreds.jsonl \
        --pairs-out runs/x/e2e_pairs.jsonl --out runs/x/seg_score.json [--subset homer]

    # 2. judge the matched pairs (text only, same judge and rubric as the labeling task)
    python -m wgo_eval.judge --pairs runs/x/e2e_pairs.jsonl --out runs/x/e2e_judged.jsonl --model <judge> ...

    # 3. segment F1 and end-to-end F1
    python -m wgo_eval.score_seg --root ../data/wgo --segpreds runs/x/segpreds.jsonl \
        --judged runs/x/e2e_judged.jsonl --out runs/x/seg_score.json

An episode without a prediction scores zero and its gold segments still count. Metric
definitions are in seg_metrics.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, write_jsonl
from .seg_metrics import IOU_THRESHOLD, aggregate, bootstrap, gold_by_episode, score_episode


def pair_id(episode_id: str, pred_i: int, gold_j: int) -> str:
    return f"{episode_id}|p{pred_i:03d}|g{gold_j:03d}"


def score_all(gold: dict[str, dict[str, Any]], preds: dict[str, dict[str, Any]], *,
              snap: bool = True, threshold: float = IOU_THRESHOLD,
              verdicts: dict[str, bool] | None = None) -> list[dict[str, Any]]:
    rows = []
    for ep_id, ep in sorted(gold.items()):
        pred = preds.get(ep_id, {}).get("segments") or []
        r = score_episode(pred, ep["gold"], snap=snap, threshold=threshold)
        r.update(episode_id=ep_id, subset=ep["subset"], task=ep["task"], has_pred=ep_id in preds)
        for p in r["pairs"]:
            p["id"] = pair_id(ep_id, p["pred_i"], p["gold_j"])
        if verdicts is not None:
            for p in r["pairs"]:
                p["semantic_match"] = bool(verdicts.get(p["id"], False))
            r["n_sem"] = sum(p["semantic_match"] for p in r["pairs"])
            r["unjudged"] = sum(p["id"] not in verdicts for p in r["pairs"])
        rows.append(r)
    return rows


def mean_lengths(gold: dict[str, dict[str, Any]], preds: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Mean segment duration and count per episode, predicted vs gold, per subset and overall.

    Granularity check: a segmenter much coarser or finer than gold loses F1 regardless of quality.
    """
    acc: dict[str, dict[str, list[float]]] = {}
    for ep_id, ep in gold.items():
        for key in (ep["subset"], "all"):
            cell = acc.setdefault(key, {"pred": [], "gold": [], "eps": []})
            cell["gold"].extend(g["end_sec"] - g["start_sec"] for g in ep["gold"])
            cell["pred"].extend(s["end_sec"] - s["start_sec"] for s in preds.get(ep_id, {}).get("segments") or [])
            cell["eps"].append(1.0)

    def mean(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    return {k: {"pred_sec": mean(v["pred"]), "gold_sec": mean(v["gold"]),
                "pred_per_episode": round(len(v["pred"]) / len(v["eps"]), 2),
                "gold_per_episode": round(len(v["gold"]) / len(v["eps"]), 2)}
            for k, v in sorted(acc.items())}


def judge_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows for `judge --pairs`: one per temporally matched pair."""
    return [{"id": p["id"], "gold_label": p["gold_label"], "pred_label": p["pred_label"], "task": r["task"],
             "pred_failed": not p["pred_label"].strip()}
            for r in rows for p in r["pairs"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--segpreds", type=Path, required=True, help="output of wgo_eval.segment")
    ap.add_argument("--judged", type=Path, default=None, help="judge output on the pairs; adds end-to-end F1")
    ap.add_argument("--pairs-out", type=Path, default=None, help="write the matched pairs for the judge")
    ap.add_argument("--subset", default=None)
    ap.add_argument("--iou", type=float, default=IOU_THRESHOLD)
    ap.add_argument("--no-outer-snap", action="store_true")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    segments = read_jsonl(args.root / "segments.jsonl")
    if args.subset:
        segments = [s for s in segments if s["subset"] == args.subset]
    gold = gold_by_episode(segments)
    preds = {r["episode_id"]: r for r in read_jsonl(args.segpreds)}
    verdicts = {r["id"]: bool(r["match"]) for r in read_jsonl(args.judged)} if args.judged else None

    rows = score_all(gold, preds, snap=not args.no_outer_snap, threshold=args.iou, verdicts=verdicts)
    if args.pairs_out:
        pairs = judge_pairs(rows)
        write_jsonl(args.pairs_out, pairs)
        print(f"SCORE_SEG|wrote {len(pairs)} matched pairs to {args.pairs_out}")

    sem_key = "n_sem" if verdicts is not None else None
    result: dict[str, Any] = {"iou_threshold": args.iou, "outer_snap": not args.no_outer_snap,
                              **aggregate(rows, sem_key), **bootstrap(rows, sem_key, args.draws)}
    result["mean_seg_sec"] = mean_lengths(gold, preds)
    result["missing_pred"] = sum(not r["has_pred"] for r in rows)
    result["failed_pred"] = sum(bool(preds.get(r["episode_id"], {}).get("failed")) for r in rows)
    if verdicts is not None:
        result["unjudged_pairs"] = sum(r["unjudged"] for r in rows)
        judged = read_jsonl(args.judged)
        result["judge_model"] = judged[0].get("judge_model") if judged else None
    result["per_episode"] = rows

    def line(name: str, b: dict[str, Any]) -> str:
        s = f"{name:10s} eps={b['episodes']:3d} pred={b['n_pred']:4d} gold={b['n_gold']:4d} match={b['n_match']:4d} " \
            f"P={b['precision']:.3f} R={b['recall']:.3f} F1={b['f1']:.3f}"
        if "e2e_f1" in b:
            s += f" | sem={b['n_sem']:4d} E2E-F1={b['e2e_f1']:.3f} label|match={b['label_acc_on_matches']:.3f}"
        return s

    for name, b in result["per_subset"].items():
        print(line(name, b))
    print(line("micro", result["micro"]) + f"  CI95 {result['f1_micro_ci95']}")
    print(f"{'subset avg':10s} F1={result['subset_avg']['f1']:.3f}  CI95 {result['f1_subset_avg_ci95']}"
          + (f"  E2E-F1={result['subset_avg']['e2e_f1']:.3f}  CI95 {result['e2e_f1_subset_avg_ci95']}" if sem_key else ""))
    if verdicts is not None and result["unjudged_pairs"]:
        print(f"WARNING: {result['unjudged_pairs']} matched pairs have no verdict and count as no match")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
