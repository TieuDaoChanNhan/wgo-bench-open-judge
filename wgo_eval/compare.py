"""Paired comparison of two labelers judged on the same segments.

    python -m wgo_eval.compare --root ../data/wgo --a runs/zs-qwen36-27b/judged.jsonl \
        --b runs/zs-qwen3vl8b/judged.jsonl [--subset homer] [--out compare.json]

Reports accuracy(A) - accuracy(B), pooled and subset average, with a paired 95% interval:
each bootstrap draw resamples episodes (within each subset) and scores both labelers on the
same draw, so the shared difficulty of an episode cancels out. A difference is reported as
clear only when its interval excludes zero. Also prints how many segments each labeler gets
right that the other misses.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import read_jsonl
from .score import accuracies


def verdicts(path: Path) -> dict[str, bool]:
    return {r["id"]: bool(r["match"]) for r in read_jsonl(path)}


def paired_bootstrap(segments: list[dict[str, Any]], va: dict[str, bool], vb: dict[str, bool],
                     draws: int = 2000, seed: int = 0) -> dict[str, Any]:
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
        a, b = accuracies(sample, va), accuracies(sample, vb)
        pooled.append(a["pooled"] - b["pooled"])
        avg.append(a["subset_avg"] - b["subset_avg"])

    def summary(xs: list[float]) -> dict[str, Any]:
        xs = sorted(xs)
        lo, hi = xs[int(0.025 * (len(xs) - 1))], xs[int(0.975 * (len(xs) - 1))]
        return {"ci95": [round(lo, 4), round(hi, 4)], "clear": lo > 0 or hi < 0,
                "share_a_better": round(sum(x > 0 for x in xs) / len(xs), 4)}

    return {"pooled": summary(pooled), "subset_avg": summary(avg)}


def compare(segments: list[dict[str, Any]], va: dict[str, bool], vb: dict[str, bool],
            draws: int = 2000, seed: int = 0) -> dict[str, Any]:
    a, b = accuracies(segments, va), accuracies(segments, vb)
    ids = [s["segment_id"] for s in segments]
    only_a = sum(va.get(i, False) and not vb.get(i, False) for i in ids)
    only_b = sum(vb.get(i, False) and not va.get(i, False) for i in ids)
    boot = paired_bootstrap(segments, va, vb, draws, seed)
    return {
        "n": len(ids),
        "acc_a": {"pooled": a["pooled"], "subset_avg": a["subset_avg"]},
        "acc_b": {"pooled": b["pooled"], "subset_avg": b["subset_avg"]},
        "diff_pooled": a["pooled"] - b["pooled"],
        "diff_subset_avg": a["subset_avg"] - b["subset_avg"],
        "only_a_right": only_a,
        "only_b_right": only_b,
        "bootstrap": boot,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--a", type=Path, required=True, help="judged.jsonl of labeler A")
    ap.add_argument("--b", type=Path, required=True, help="judged.jsonl of labeler B")
    ap.add_argument("--subset", default=None)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    segments = read_jsonl(args.root / "segments.jsonl")
    if args.subset:
        segments = [s for s in segments if s["subset"] == args.subset]
    result = compare(segments, verdicts(args.a), verdicts(args.b), args.draws)
    result.update(a=str(args.a), b=str(args.b), subset=args.subset or "all")

    p, ci = result["diff_pooled"], result["bootstrap"]["pooled"]
    print(f"A {100 * result['acc_a']['pooled']:.1f}  B {100 * result['acc_b']['pooled']:.1f}  "
          f"A-B {100 * p:+.1f} pooled, CI95 [{100 * ci['ci95'][0]:+.1f}, {100 * ci['ci95'][1]:+.1f}] "
          f"{'clear' if ci['clear'] else 'not clear'}")
    print(f"only A right {result['only_a_right']}, only B right {result['only_b_right']}, n={result['n']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
