"""Segment F1 and end-to-end F1 for WGO-Bench, matching the reference scorer.

Reference: EgoANT scripts/wgo_bench/wgo_metrics.py, score_segmentation.py and
scripts/wgo_gemini_rescore.py at commit b27ea59f5f (MIT, see THIRD_PARTY.md):

1. Outer snap: the first predicted start and the last predicted end are set to the gold
   episode extent (first gold start, last gold end). Interior boundaries are untouched.
2. Greedy one-to-one matching: every (pred, gold) pair is scored by temporal IoU, pairs are
   taken in descending IoU order while IoU >= 0.75, and a segment is used at most once.
   Labels play no part in matching.
3. Segment F1 (micro): matches / predicted segments and matches / gold segments, pooled over
   episodes. Macro F1 is the unweighted mean of per-episode F1.
4. End-to-end F1: a matched pair also has to pass the label judge. Precision and recall use
   the same denominators as segment F1 (all predicted, all gold segments).
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Iterable

IOU_THRESHOLD = 0.75

Seg = dict[str, Any]  # {"start_sec": float, "end_sec": float, "subtask"/"label": str}


def iou(a: Seg, b: Seg) -> float:
    overlap = max(0.0, min(a["end_sec"], b["end_sec"]) - max(a["start_sec"], b["start_sec"]))
    union = (a["end_sec"] - a["start_sec"]) + (b["end_sec"] - b["start_sec"]) - overlap
    return overlap / union if union > 1e-9 else 0.0


def snap_outer_boundaries(pred: list[Seg], gold: list[Seg]) -> list[Seg]:
    """Copy of pred, sorted by start, with its outer edges aligned to the gold episode extent."""
    out = [dict(s) for s in sorted(pred, key=lambda s: (s["start_sec"], s["end_sec"]))]
    if not out or not gold:
        return out
    out[0]["start_sec"] = gold[0]["start_sec"]
    out[-1]["end_sec"] = gold[-1]["end_sec"]
    for s in out:
        s["end_sec"] = max(s["start_sec"], s["end_sec"])
    return out


def greedy_match(pred: list[Seg], gold: list[Seg], threshold: float = IOU_THRESHOLD) -> list[tuple[int, int, float]]:
    """(pred_index, gold_index, iou) pairs, greedy by descending IoU, one-to-one."""
    candidates = [(i, j, iou(p, g)) for i, p in enumerate(pred) for j, g in enumerate(gold)]
    candidates.sort(key=lambda c: c[2], reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    pairs = []
    for i, j, score in candidates:
        if score < threshold:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        pairs.append((i, j, score))
    return pairs


def prf(match: int, n_pred: int, n_gold: int) -> dict[str, float]:
    p = match / n_pred if n_pred else 0.0
    r = match / n_gold if n_gold else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def score_episode(pred: list[Seg], gold: list[Seg], *, snap: bool = True,
                  threshold: float = IOU_THRESHOLD) -> dict[str, Any]:
    """Per-episode counts and the matched pairs (indices refer to the snapped, sorted prediction)."""
    pred = snap_outer_boundaries(pred, gold) if snap else sorted(pred, key=lambda s: s["start_sec"])
    pairs = greedy_match(pred, gold, threshold)
    return {
        "n_pred": len(pred),
        "n_gold": len(gold),
        "n_match": len(pairs),
        **prf(len(pairs), len(pred), len(gold)),
        "pairs": [
            {"pred_i": i, "gold_j": j, "iou": round(s, 4),
             "pred_label": str(pred[i].get("subtask") or pred[i].get("label") or ""),
             "gold_label": str(gold[j].get("label") or gold[j].get("subtask") or "")}
            for i, j, s in sorted(pairs, key=lambda x: x[1])
        ],
    }


def aggregate(episodes: Iterable[dict[str, Any]], sem_key: str | None = None) -> dict[str, Any]:
    """Micro and macro scores over per-episode rows (each with subset, n_pred, n_gold, n_match).

    With sem_key, also end-to-end scores from the per-episode count stored under that key.
    """
    episodes = list(episodes)

    def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
        tp, tg, tm = (sum(r[k] for r in rows) for k in ("n_pred", "n_gold", "n_match"))
        out = {"episodes": len(rows), "n_pred": tp, "n_gold": tg, "n_match": tm, **prf(tm, tp, tg),
               "macro_f1": sum(prf(r["n_match"], r["n_pred"], r["n_gold"])["f1"] for r in rows) / len(rows) if rows else 0.0}
        if sem_key:
            ts = sum(r[sem_key] for r in rows)
            e2e = prf(ts, tp, tg)
            out.update(n_sem=ts, e2e_precision=e2e["precision"], e2e_recall=e2e["recall"], e2e_f1=e2e["f1"],
                       label_acc_on_matches=ts / tm if tm else 0.0)
        return out

    by_subset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in episodes:
        by_subset[r["subset"]].append(r)
    per_subset = {k: block(v) for k, v in sorted(by_subset.items())}
    result = {"micro": block(episodes), "per_subset": per_subset}
    keys = ["f1"] + (["e2e_f1"] if sem_key else [])
    result["subset_avg"] = {k: sum(v[k] for v in per_subset.values()) / len(per_subset) if per_subset else 0.0
                            for k in keys}
    return result


def bootstrap(episodes: list[dict[str, Any]], sem_key: str | None = None, draws: int = 2000,
              seed: int = 0) -> dict[str, list[float]]:
    """95% intervals of micro and subset-average F1, resampling episodes within each subset."""
    by_subset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in episodes:
        by_subset[r["subset"]].append(r)
    rng = random.Random(seed)
    keys = ["f1"] + (["e2e_f1"] if sem_key else [])
    samples: dict[str, list[float]] = {f"{k}_{s}": [] for k in keys for s in ("micro", "subset_avg")}
    for _ in range(draws):
        sample = [rng.choice(rows) for rows in by_subset.values() for _ in rows]
        agg = aggregate(sample, sem_key)
        for k in keys:
            samples[f"{k}_micro"].append(agg["micro"][k])
            samples[f"{k}_subset_avg"].append(agg["subset_avg"][k])

    def ci(xs: list[float]) -> list[float]:
        xs = sorted(xs)
        return [round(xs[int(0.025 * (len(xs) - 1))], 4), round(xs[int(0.975 * (len(xs) - 1))], 4)]

    return {f"{name}_ci95": ci(xs) for name, xs in samples.items()}


def gold_by_episode(segments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """segments.jsonl rows grouped per episode, in segment order."""
    out: dict[str, dict[str, Any]] = {}
    for s in sorted(segments, key=lambda s: s["segment_id"]):
        ep = out.setdefault(s["episode_id"], {"episode_id": s["episode_id"], "subset": s["subset"],
                                              "video": s["video"], "task": s.get("task", ""), "gold": []})
        ep["gold"].append({"start_sec": s["start_sec"], "end_sec": s["end_sec"], "label": s["label"]})
    return out
