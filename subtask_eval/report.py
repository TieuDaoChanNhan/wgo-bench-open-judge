"""Collect scored runs into one comparison table (Markdown).

    subtask-eval report runs/cs-qwen36-27b runs/studio-flash-next ... [--subset all] [--out report.md]

Each run directory needs score_<subset>.json (from `subtask-eval score`) and run.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_run(run_dir: Path, subset: str) -> dict[str, Any] | None:
    path = run_dir / f"score_{subset}.json"
    if not path.exists():
        return None
    score = json.loads(path.read_text(encoding="utf-8"))
    meta_path = run_dir / "run.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    cfg = meta.get("config") or {}
    score["name"] = cfg.get("name") or run_dir.name
    score["annotator"] = cfg.get("annotator", "?")
    score["model"] = cfg.get("model", "")
    return score


def _ci(score: dict[str, Any], key: str) -> str:
    lo, hi = score.get(key, [None, None])
    return f"[{lo:.3f}, {hi:.3f}]" if lo is not None else ""


def _fmt(x: Any, spec: str = ".3f") -> str:
    return "–" if x is None else format(x, spec)


def table(scores: list[dict[str, Any]]) -> str:
    head = ("| Run | Annotator | Model | Seg F1 | 95% CI | Label acc (matched) | E2E F1 | s/seg (gold) | "
            "seg/ep (gold) | GPU-h / video-h | Failed |")
    sep = "|" + "---|" * 11
    lines = [head, sep]
    for s in sorted(scores, key=lambda s: -s["micro"]["f1"]):
        m, g, c = s["micro"], s["granularity"]["all"], s["cost"]
        lines.append(
            f"| {s['name']} | {s['annotator']} | {s['model']} | {m['f1']:.3f} | {_ci(s, 'f1_micro_ci95')} | "
            f"{_fmt(m.get('label_acc_on_matches'))} | {_fmt(m.get('e2e_f1'))} | "
            f"{g['pred_sec']:.1f} ({g['gold_sec']:.1f}) | {g['pred_per_episode']:.1f} ({g['gold_per_episode']:.1f}) | "
            f"{_fmt(c.get('gpu_hours_per_video_hour'), '.2f')} | {s['failed_pred'] + s['missing_pred']} |")
    return "\n".join(lines)


def per_subset_table(scores: list[dict[str, Any]]) -> str:
    subsets = sorted({k for s in scores for k in s["per_subset"]})
    lines = ["| Run | " + " | ".join(f"{k} F1" for k in subsets) + " |", "|" + "---|" * (len(subsets) + 1)]
    for s in sorted(scores, key=lambda s: -s["micro"]["f1"]):
        cells = [f"{s['per_subset'][k]['f1']:.3f}" if k in s["per_subset"] else "–" for k in subsets]
        lines.append(f"| {s['name']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--subset", default="all")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    scores, missing = [], []
    for run_dir in args.runs:
        s = load_run(run_dir, args.subset)
        (scores.append(s) if s else missing.append(str(run_dir)))
    if not scores:
        print(f"REPORT|no score_{args.subset}.json in {missing}")
        return 1
    gaps = scores[0]["gold_gaps"].get("all", {})
    text = "\n\n".join([
        f"## Subtask annotation on WGO-Bench ({args.subset})",
        table(scores),
        "### Segment F1 per subset",
        per_subset_table(scores),
        f"Gold pauses: {gaps.get('n_gaps')} gaps, {100 * gaps.get('gap_share', 0):.1f}% of the gold extent. "
        f"A gap-free output with perfect boundaries reaches at most Seg F1 {gaps.get('gap_free_ceiling_f1')}.",
        f"Judge: {scores[0].get('judge_model') or 'not run'}. IoU threshold {scores[0]['iou_threshold']}, "
        f"outer snap {scores[0]['outer_snap']}.",
    ]) + "\n"
    if missing:
        text += f"\nNot scored yet: {', '.join(missing)}\n"
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
