"""Score one run: segment F1, end-to-end F1, granularity, gaps and cost.

    # 1. segment F1, and the matched pairs for the label judge
    python -m subtask_eval.scoring.score --root data/wgo --run runs/x --write-pairs
    # 2. judge the matched pairs
    python -m subtask_eval.scoring.judge --run runs/x --model <judge> ...
    # 3. everything, now with end-to-end F1 (add --subset homer for one subset)
    python -m subtask_eval.scoring.score --root data/wgo --run runs/x

Reads <run>/preds.jsonl, <run>/run.json and, when present, <run>/judged.jsonl.
Writes <run>/score_<subset or all>.json and prints a summary. An episode without a
prediction scores zero and its gold segments still count. Metric definitions: segments.py.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..annotators.base import load_predictions
from ..common import read_jsonl, write_jsonl
from ..data.wgo import load_episodes
from .segments import IOU_THRESHOLD, aggregate, bootstrap, full_cover, gap_stats, score_episode


def pair_id(episode_id: str, pred_i: int, gold_j: int) -> str:
    return f"{episode_id}|p{pred_i:03d}|g{gold_j:03d}"


def score_rows(episodes: list[dict[str, Any]], preds: dict[str, dict[str, Any]], *, snap: bool = True,
               threshold: float = IOU_THRESHOLD, verdicts: dict[str, bool] | None = None) -> list[dict[str, Any]]:
    rows = []
    for ep in episodes:
        pred = preds.get(ep["episode_id"], {}).get("segments") or []
        r = score_episode(pred, ep["gold"], snap=snap, threshold=threshold)
        r.update(episode_id=ep["episode_id"], subset=ep["subset"], task=ep["task"], has_pred=ep["episode_id"] in preds)
        for p in r["pairs"]:
            p["id"] = pair_id(ep["episode_id"], p["pred_i"], p["gold_j"])
        if verdicts is not None:
            for p in r["pairs"]:
                p["semantic_match"] = bool(verdicts.get(p["id"], False))
            r["n_sem"] = sum(p["semantic_match"] for p in r["pairs"])
            r["unjudged"] = sum(p["id"] not in verdicts for p in r["pairs"])
        rows.append(r)
    return rows


def judge_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows for the judge: one per temporally matched pair."""
    return [{"id": p["id"], "gold_label": p["gold_label"], "pred_label": p["pred_label"], "task": r["task"]}
            for r in rows for p in r["pairs"]]


def granularity(episodes: list[dict[str, Any]], preds: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean segment length and segments per episode, predicted vs gold, per subset and overall.

    A segmenter much coarser or finer than gold loses F1 whatever its quality, so read F1 next to this.
    """
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"pred": [], "gold": [], "eps": []})
    for ep in episodes:
        for key in (ep["subset"], "all"):
            cell = acc[key]
            cell["gold"].extend(g["end_sec"] - g["start_sec"] for g in ep["gold"])
            cell["pred"].extend(s["end_sec"] - s["start_sec"] for s in preds.get(ep["episode_id"], {}).get("segments") or [])
            cell["eps"].append(1.0)

    def mean(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    return {k: {"pred_sec": mean(v["pred"]), "gold_sec": mean(v["gold"]),
                "pred_per_episode": round(len(v["pred"]) / len(v["eps"]), 2),
                "gold_per_episode": round(len(v["gold"]) / len(v["eps"]), 2)}
            for k, v in sorted(acc.items())}


def gap_report(episodes: list[dict[str, Any]], threshold: float = IOU_THRESHOLD) -> dict[str, dict[str, float]]:
    """Pauses in the gold segmentation, and the best segment F1 a gap-free output could reach.

    lerobot-annotate always returns a gap-free cover. Scoring the gold boundaries themselves after
    the same stitching gives the ceiling such an output can reach with perfect boundaries.
    """
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        r = score_episode(full_cover(ep["gold"]), ep["gold"], threshold=threshold)
        r.update(subset=ep["subset"], **gap_stats(ep["gold"]))
        by[ep["subset"]].append(r)
        by["all"].append(r)
    out = {}
    for k, rows in sorted(by.items()):
        extent = sum(r["extent_sec"] for r in rows)
        out[k] = {"n_gaps": sum(r["n_gaps"] for r in rows),
                  "gap_share": round(sum(r["gap_sec"] for r in rows) / extent, 4) if extent else 0.0,
                  "gap_free_ceiling_f1": round(aggregate(rows)["micro"]["f1"], 4)}
    return out


def cost_report(run_dir: Path, preds: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Run-level cost: wall time, GPU-hours and tokens per hour of video."""
    meta_path = run_dir / "run.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    video_sec = sum(float(p.get("duration") or 0) for p in preds.values())
    wall = sum(s.get("wall_sec", 0) for s in meta.get("sessions", []))
    gpus = float((meta.get("config") or {}).get("gpus") or 0)
    hours = video_sec / 3600
    tok_in = sum(p.get("cost", {}).get("tokens_in", 0) for p in preds.values())
    tok_out = sum(p.get("cost", {}).get("tokens_out", 0) for p in preds.values())
    calls = sum(p.get("cost", {}).get("n_calls", 0) for p in preds.values())
    return {
        "video_hours": round(hours, 4),
        "wall_sec": round(wall, 1),
        "gpus": gpus,
        "gpu_hours_per_video_hour": round(wall / 3600 * gpus / hours, 4) if hours and gpus else None,
        "tokens_in_per_video_hour": round(tok_in / hours) if hours else None,
        "tokens_out_per_video_hour": round(tok_out / hours) if hours else None,
        "calls_per_video_hour": round(calls / hours, 1) if hours else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True, help="run directory with preds.jsonl")
    ap.add_argument("--subset", default=None)
    ap.add_argument("--write-pairs", action="store_true", help="write <run>/pairs.jsonl for the judge (all subsets)")
    ap.add_argument("--iou", type=float, default=IOU_THRESHOLD)
    ap.add_argument("--no-outer-snap", action="store_true")
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args()

    preds = load_predictions(args.run)
    snap = not args.no_outer_snap
    if args.write_pairs:
        pairs = judge_pairs(score_rows(load_episodes(args.root), preds, snap=snap, threshold=args.iou))
        write_jsonl(args.run / "pairs.jsonl", pairs)
        print(f"SCORE|wrote {len(pairs)} matched pairs to {args.run / 'pairs.jsonl'}")

    episodes = load_episodes(args.root, args.subset)
    judged_path = args.run / "judged.jsonl"
    judged = read_jsonl(judged_path)
    verdicts = {r["id"]: bool(r["match"]) for r in judged} if judged else None
    rows = score_rows(episodes, preds, snap=snap, threshold=args.iou, verdicts=verdicts)
    sem_key = "n_sem" if verdicts is not None else None
    in_scope = {e["episode_id"] for e in episodes}
    scoped_preds = {k: v for k, v in preds.items() if k in in_scope}

    result: dict[str, Any] = {
        "run": str(args.run), "subset": args.subset or "all", "iou_threshold": args.iou, "outer_snap": snap,
        **aggregate(rows, sem_key), **bootstrap(rows, sem_key, args.draws),
        "granularity": granularity(episodes, scoped_preds),
        "gold_gaps": gap_report(episodes, args.iou),
        "cost": cost_report(args.run, scoped_preds),
        "missing_pred": sum(not r["has_pred"] for r in rows),
        "failed_pred": sum(bool(scoped_preds.get(r["episode_id"], {}).get("failed")) for r in rows),
    }
    if verdicts is not None:
        result["unjudged_pairs"] = sum(r["unjudged"] for r in rows)
        result["judge_model"] = judged[0].get("judge_model")
    result["per_episode"] = rows

    def line(name: str, b: dict[str, Any]) -> str:
        s = (f"{name:16s} eps={b['episodes']:3d} pred={b['n_pred']:4d} gold={b['n_gold']:4d} match={b['n_match']:4d} "
             f"P={b['precision']:.3f} R={b['recall']:.3f} F1={b['f1']:.3f}")
        if "e2e_f1" in b:
            s += f" | E2E-F1={b['e2e_f1']:.3f} label|match={b['label_acc_on_matches']:.3f}"
        return s

    for name, b in result["per_subset"].items():
        print(line(name, b))
    print(line("micro", result["micro"]) + f"  CI95 {result['f1_micro_ci95']}")
    g = result["granularity"]["all"]
    print(f"granularity      pred {g['pred_sec']:.2f} s/seg, {g['pred_per_episode']:.1f} seg/ep | "
          f"gold {g['gold_sec']:.2f} s/seg, {g['gold_per_episode']:.1f} seg/ep")
    print(f"gold gaps        {result['gold_gaps']['all']}")
    print(f"cost             {result['cost']}")
    if verdicts is None:
        print("no judged.jsonl yet: end-to-end F1 not computed")
    elif result["unjudged_pairs"]:
        print(f"WARNING: {result['unjudged_pairs']} matched pairs have no verdict and count as no match")

    out = args.run / f"score_{args.subset or 'all'}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
