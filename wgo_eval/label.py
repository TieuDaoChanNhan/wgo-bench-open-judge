"""Label every gold segment with a VLM served behind an OpenAI-compatible endpoint (e.g. vLLM).

    python -m wgo_eval.label --root ../data/wgo --out runs/qwen3vl8b/preds.jsonl \
        --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3-VL-8B-Instruct [--subset homer]

The model sees only the frames of the segment and the episode task, never the gold label.
Output rows: segment_id, label, json_ok, failed, raw. Reruns skip segment_ids already in --out.

Prompt `egoant_raw` reproduces EgoANT's fixed-boundary "raw" setting (5 frames, longest side
640 px, JPEG quality 85), adapted from scripts/wgo_bench/label_gold_segments.py (MIT).
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .common import append_jsonl, chat, parse_label, read_jsonl

PROMPTS: dict[str, dict[str, str]] = {
    "egoant_raw": {
        "system": (
            "You label egocentric manipulation subtasks. "
            "Describe the completed manipulation event in one short imperative phrase. "
            "Include the main object and destination/target when visible. "
            'Do not narrate the actor. Reply with ONLY JSON: {"label": "..."}.'
        ),
        "user": (
            "I will send ordered raw frames sampled uniformly from one gold segment. "
            "Label the CURRENT segment only.\n"
            "Episode task (context only): {task}\n"
            "Label the CURRENT segment only."
        ),
    },
}


def sample_indices(nframes: int, start_f: int, end_f: int, n: int) -> list[int]:
    """n frame indices spread evenly from start_f to end_f inclusive, clipped to the video."""
    s = max(0, min(nframes - 1, start_f))
    e = max(s, min(nframes - 1, end_f))
    if n <= 1:
        return [s]
    return [int(round(s + (e - s) * i / (n - 1))) for i in range(n)]


def segment_frames(video: Path, start_sec: float, end_sec: float, n: int, max_side: int, quality: int) -> list[bytes]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video.name}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        out = []
        for idx in sample_indices(nframes, int(round(start_sec * fps)), int(round(end_sec * fps)), n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            scale = max_side / max(h, w)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if ok:
                out.append(buf.tobytes())
        return out
    finally:
        cap.release()


def build_messages(prompt: str, task: str, frames: list[bytes]) -> list[dict[str, Any]]:
    p = PROMPTS[prompt]
    content: list[dict[str, Any]] = [{"type": "text", "text": p["user"].format(task=task)}]
    for b in frames:
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode("ascii")}})
    return [{"role": "system", "content": p["system"]}, {"role": "user", "content": content}]


def model_input(segment: dict[str, Any]) -> dict[str, Any]:
    """The only fields the labeler may use. The gold label is dropped here."""
    return {k: segment[k] for k in ("segment_id", "video", "start_sec", "end_sec", "task")}


def label_one(seg: dict[str, Any], root: Path, args: argparse.Namespace) -> dict[str, Any]:
    seg = model_input(seg)
    frames = segment_frames(root / seg["video"], seg["start_sec"], seg["end_sec"], args.frames, args.max_side, args.quality)
    if not frames:
        return {"segment_id": seg["segment_id"], "label": "", "json_ok": False, "failed": True, "raw": "no frames decoded"}
    try:
        reply = chat(args.base_url, args.model, build_messages(args.prompt, seg["task"], frames),
                     api_key=args.api_key, max_tokens=256, timeout=args.timeout, retries=args.retries)
    except RuntimeError as exc:
        return {"segment_id": seg["segment_id"], "label": "", "json_ok": False, "failed": True, "raw": str(exc)}
    label, json_ok = parse_label(reply["text"])
    return {"segment_id": seg["segment_id"], "label": label, "json_ok": json_ok, "failed": not label,
            "raw": reply["text"][:500]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="output root of wgo_prepare")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="local")
    ap.add_argument("--prompt", default="egoant_raw", choices=sorted(PROMPTS))
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--subset", default=None, help="e.g. homer; default all subsets")
    ap.add_argument("--limit", type=int, default=0, help="first N segments only (smoke test)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    segments = read_jsonl(args.root / "segments.jsonl")
    if args.subset:
        segments = [s for s in segments if s["subset"] == args.subset]
    if args.limit:
        segments = segments[: args.limit]
    done = {r["segment_id"] for r in read_jsonl(args.out)}
    todo = [s for s in segments if s["segment_id"] not in done]
    print(f"LABEL|segments={len(segments)} done={len(done)} todo={len(todo)} model={args.model} prompt={args.prompt}", flush=True)

    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(label_one, s, args.root, args) for s in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            row.update(model=args.model, prompt=args.prompt)
            append_jsonl(args.out, row)
            failed += row["failed"]
            if i % 50 == 0 or i == len(todo):
                print(f"LABEL|{i}/{len(todo)} failed={failed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
