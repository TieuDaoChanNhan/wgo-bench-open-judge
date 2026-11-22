"""Judge predicted labels against gold labels with a text-only LLM and the WGO-Bench rubric.

    python -m wgo_eval.judge --root ../data/wgo --preds runs/qwen3vl8b/preds.jsonl \
        --out runs/qwen3vl8b/judged.jsonl --base-url http://127.0.0.1:8001/v1 --model <judge>

    python -m wgo_eval.judge --pairs pairs.jsonl --out judged.jsonl ...   (rows: id, gold_label, pred_label, task)

The judge sees the gold label, the predicted label and the episode task, never video.
Failed or empty predictions count as no match without a call. Verdicts are cached by
(model, prompt) in --cache, so reruns cost nothing.

The rubric is the label judge rubric of the Macrodata write-up
(https://macrodata.co/blog/annotating-robot-video-subtasks), as transcribed in EgoANT
scripts/wgo_gemini_rescore.py (MIT).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .common import append_jsonl, chat, parse_match, read_jsonl

RUBRIC = """You are judging whether a predicted subtask label matches a gold subtask label.

Gold label:
{gold}

Predicted label:
{pred}

Episode instruction:
{task}

Accept if:
- It describes the same manipulation event or world-state change.
- The main action is correct.
- The main manipulated object is correct.
- Source, destination, direction, or spatial relation is correct when central to the event.
- Wording can differ; synonyms are fine.
- It may be slightly less detailed than the gold label if it is still useful.
Reject if:
- The action is wrong.
- The main object is wrong.
- Source, destination, or direction is flipped or wrong.
- It describes a different event.
- It is too vague to identify the subtask.
- It hallucinates an important object or action.

Ignore:
- Grammar.
- Minor wording differences.
- Timing; timing is evaluated separately.

Return only JSON:
{{"match": true}}"""

_LOCK = threading.Lock()


def judge_prompt(gold: str, pred: str, task: str) -> str:
    return RUBRIC.format(gold=gold.strip(), pred=pred.strip(), task=(task or "").strip())


def cache_key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}\0{prompt}".encode("utf-8")).hexdigest()


def pairs_from_preds(segments: list[dict[str, Any]], preds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One judge pair per gold segment; a segment without a prediction gets an empty one."""
    by_id = {p["segment_id"]: p for p in preds}
    pairs = []
    for s in segments:
        p = by_id.get(s["segment_id"], {"label": "", "failed": True})
        pairs.append({"id": s["segment_id"], "gold_label": s["label"], "pred_label": p.get("label") or "",
                      "task": s.get("task", ""), "pred_failed": bool(p.get("failed")) or not p.get("label")})
    return pairs


def judge_pair(pair: dict[str, Any], cache: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = {"id": pair["id"], "gold_label": pair["gold_label"], "pred_label": pair["pred_label"], "judge_model": args.model}
    if pair.get("pred_failed"):
        return {**out, "match": False, "skipped": "failed prediction"}
    prompt = judge_prompt(pair["gold_label"], pair["pred_label"], pair.get("task", ""))
    key = cache_key(args.model, prompt)
    with _LOCK:
        hit = cache.get(key)
    if hit is not None:
        return {**out, "match": hit["match"], "raw": hit["raw"], "cached": True}
    reply = chat(args.base_url, args.model, [{"role": "user", "content": prompt}], api_key=args.api_key,
                 max_tokens=args.max_tokens, timeout=args.timeout, retries=args.retries,
                 disable_thinking=not args.thinking)
    match = parse_match(reply["text"])
    entry = {"match": match, "raw": reply["text"][:300]}
    with _LOCK:
        cache[key] = entry
        append_jsonl(args.cache, {"key": key, **entry})
    return {**out, **entry, "cached": False}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--preds", type=Path, help="labeler output; needs --root")
    src.add_argument("--pairs", type=Path, help="JSONL with id, gold_label, pred_label, task")
    ap.add_argument("--root", type=Path, help="output root of wgo_prepare")
    ap.add_argument("--subset", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("runs/judge_cache.jsonl"))
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="local")
    ap.add_argument("--thinking", action="store_true", help="let the judge reason before answering")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()
    if args.thinking:
        args.max_tokens = max(args.max_tokens, 2048)

    if args.preds:
        if not args.root:
            ap.error("--preds needs --root")
        segments = read_jsonl(args.root / "segments.jsonl")
        if args.subset:
            segments = [s for s in segments if s["subset"] == args.subset]
        pairs = pairs_from_preds(segments, read_jsonl(args.preds))
    else:
        pairs = read_jsonl(args.pairs)

    cache = {r["key"]: r for r in read_jsonl(args.cache)}
    if args.out.exists():
        args.out.unlink()
    matches = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(judge_pair, p, cache, args) for p in pairs]
        for fut in as_completed(futures):
            row = fut.result()
            with _LOCK:
                append_jsonl(args.out, row)
            matches += bool(row["match"])
    print(f"JUDGE|pairs={len(pairs)} match={matches} model={args.model} thinking={args.thinking}")
    print(json.dumps({"accuracy": round(matches / max(1, len(pairs)), 4)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
