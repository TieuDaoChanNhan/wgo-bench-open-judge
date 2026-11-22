"""Accuracy from judge verdicts: pooled, per subset, subset average, with episode bootstrap intervals.

    python -m wgo_eval.score --root ../data/wgo --judged runs/qwen3vl8b/judged.jsonl \
        [--preds runs/qwen3vl8b/preds.jsonl] [--out runs/qwen3vl8b/score.json]

Pooled accuracy = matches / gold segments (the Macrodata blog reports this).
Subset average = mean of per-subset accuracies (the Flora paper reports this).
Gold segments without a verdict count as no match. The 95% interval resamples
episodes with replacement (2000 draws, fixed seed).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import read_jsonl


def accuracies(segments: list[dict[str, Any]], verdicts: dict[str, bool]) -> dict[str, Any]:
    by_subset: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for s in segments:
        cell = by_subset[s["subset"]]
        cell[0] += bool(verdicts.get(s["segment_id"], False))
        cell[1] += 1
    hits = sum(c[0] for c in by_subset.values())
    total = sum(c[1] for c in by_subset.values())
    per_subset = {k: {"match": c[0], "n": c[1], "acc": c[0] / c[1]} for k, c in sorted(by_subset.items())}
    return {
        "pooled": hits / total if total else 0.0,
        "subset_avg": sum(v["acc"] for v in per_subset.values()) / len(per_subset) if per_subset else 0.0,
        "match": hits,
        "n": total,
        "per_subset": per_subset,
    }


def bootstrap(segments: list[dict[str, Any]], verdicts: dict[str, bool], draws: int = 2000, seed: int = 0) -> dict[str, list[float]]:
    """95% intervals of pooled and subset-average accuracy, resampling episodes within each subset."""
    episodes: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for s in segments:
        episodes[s["subset"]][s["episode_id"]].append(s)
    rng = random.Random(seed)
    pooled, avg = [], []
    for _ in range(draws):
        sample = []
        for eps in episodes.values():
            ids = list(eps)
            for _ in ids:
                sample.extend(eps[rng.choice(ids)])
        a = accuracies(sample, verdicts)
        pooled.append(a["pooled"])
        avg.append(a["subset_avg"])

    def ci(xs: list[float]) -> list[float]:
        xs = sorted(xs)
        return [round(xs[int(0.025 * (len(xs) - 1))], 4), round(xs[int(0.975 * (len(xs) - 1))], 4)]

    return {"pooled_ci95": ci(pooled), "subset_avg_ci95": ci(avg)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--judged", type=Path, required=True)
    ap.add_argument("--preds", type=Path, default=None, help="adds JSON-validity and failure counts")
    ap.add_argument("--subset", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    segments = read_jsonl(args.root / "segments.jsonl")
    if args.subset:
        segments = [s for s in segments if s["subset"] == args.subset]
    judged = read_jsonl(args.judged)
    verdicts = {r["id"]: bool(r["match"]) for r in judged}
    result = accuracies(segments, verdicts)
    result.update(bootstrap(segments, verdicts))
    result["judge_model"] = judged[0].get("judge_model") if judged else None
    result["unjudged"] = sum(1 for s in segments if s["segment_id"] not in verdicts)
    if args.preds:
        ids = {s["segment_id"] for s in segments}
        preds = [p for p in read_jsonl(args.preds) if p["segment_id"] in ids]
        result["preds"] = {"n": len(preds), "failed": sum(bool(p.get("failed")) for p in preds),
                           "not_json": sum(not p.get("json_ok") for p in preds)}

    for name, v in result["per_subset"].items():
        print(f"{name:10s} {v['match']:4d}/{v['n']:<4d} {100 * v['acc']:5.1f}")
    print(f"{'pooled':10s} {result['match']:4d}/{result['n']:<4d} {100 * result['pooled']:5.1f}  CI95 {result['pooled_ci95']}")
    print(f"{'subset avg':10s} {'':9s} {100 * result['subset_avg']:5.1f}  CI95 {result['subset_avg_ci95']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
