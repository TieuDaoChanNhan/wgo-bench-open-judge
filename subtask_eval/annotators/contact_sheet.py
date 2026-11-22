"""Baseline annotator: one VLM call over the episode's timestamped contact sheets (Macrodata style).

    python -m subtask_eval.annotators.contact_sheet --root data/wgo --out runs/cs-qwen36-27b \
        --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3.6-27B [--config configs/x.json] [--subset homer]

Input follows the WGO-Bench write-up and EgoANT scripts/wgo_bench/contact_sheet_segment.py
(MIT, see THIRD_PARTY.md): one frame every 0.5 s, 224 px tiles with a yellow timestamp in the
top-left corner, 20 tiles per sheet in 5 columns, all sheets of an episode in one call. If that
call fails (for example the context is too long), the sheets are split into non-overlapping
windows of 2/3 the size until every window succeeds.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
from pathlib import Path
from typing import Any

from ..common import chat, parse_with_config
from ..data.wgo import load_episodes, model_view
from .base import add_usage, new_cost, run

SYSTEM = "You segment egocentric manipulation videos into completed atomic events. Reply with ONLY valid JSON."

PROMPTS: dict[str, str] = {
    # The GEPA-selected prompt of the write-up (EgoANT USER_PROMPT_GEPA), unchanged.
    "gepa_duration_prior_v1": """Reconstruct the sequence of manipulation events in this robot video from the timestamped contact sheets.

Context:
- Each tile is a frame sampled every {sample_sec:.2f}s. Yellow text on each tile is the timestamp in seconds.
- Tiles are ordered left-to-right, top-to-bottom; sheets are in time order.
- Visible time range in this call: [{t0:.2f}, {t1:.2f}] seconds.
- Episode instruction (context only): {instruction}

Return only JSON with this shape:
{{"segments":[{{"start_sec":0.0,"end_sec":1.0,"subtask":"short action description"}}]}}

Rules:
- Segment only completed robot manipulation events, not every visible movement.
- Good boundaries happen when a held object changes, an object is placed or released, a tool starts/stops changing a surface, a container/door/lid opens or closes, or contents move between containers.
- Do not split approach, grasp adjustment, small repositioning, and retreat unless the world state changes.
- Do not merge separate pick/place/open/close/pour/wipe events when they complete different states.
- Most segments should be 2-10 seconds. Shorter segments are okay only for fast pick, place, open, close, or release events.
- Use the visible timestamps for start_sec and end_sec.
- Ignore label wording quality; prioritize temporally correct boundaries.
- start_sec/end_sec must be within [{t0:.2f}, {t1:.2f}], non-overlapping, sorted by time.
""",
    # EgoANT USER_PROMPT_LEGACY, unchanged.
    "legacy": """These images are contact sheets from one continuous egocentric video.
Each tile is a frame sampled every {sample_sec:.2f}s. The yellow text on each tile is the timestamp in seconds.
Tiles are ordered left-to-right, top-to-bottom, and sheets are in time order.

Episode task (context only): {instruction}

List every COMPLETED manipulation event visible in these sheets.
Rules:
- An event ends when the manipulation is completed (object placed, lid closed, pour finished, etc.).
- Use the yellow timestamps as the time axis.
- start_sec/end_sec must be within [{t0:.2f}, {t1:.2f}] and non-overlapping, sorted by time.
- Prefer concise imperative labels (verb + object + target when visible).
- Do not invent events that are not visible.
- Cover the full span as well as you can; merge tiny flickers.

Return ONLY JSON:
{{"segments":[{{"start_sec":0.0,"end_sec":1.0,"subtask":"..."}}, ...]}}
""",
}


# ---------- frames and contact sheets ----------

def sample_times(duration: float, sample_sec: float) -> list[float]:
    n = max(1, int(math.floor(duration / sample_sec)) + 1)
    times = [min(duration, i * sample_sec) for i in range(n)]
    if times[-1] < duration - 1e-3:
        times.append(duration)
    return times


def make_tile(frame, t: float, tile: int):
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    scale = tile / max(h, w)
    resized = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((tile, tile, 3), 16, dtype=np.uint8)
    y0, x0 = (tile - resized.shape[0]) // 2, (tile - resized.shape[1]) // 2
    canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    cv2.rectangle(canvas, (2, 2), (90, 28), (0, 0, 0), -1)
    cv2.putText(canvas, f"{t:.1f}s", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def build_sheets(video: Path, sample_sec: float, tile: int, cols: int, per_sheet: int,
                 quality: int) -> tuple[list[bytes], list[tuple[float, float]], float]:
    """JPEG contact sheets, the (first, last) timestamp of each, and the video duration."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video.name}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = nframes / fps
        tiles: list[tuple[float, Any]] = []
        for t in sample_times(duration, sample_sec):
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(max(nframes - 1, 0), max(0, int(round(t * fps)))))
            ok, frame = cap.read()
            if ok:
                tiles.append((t, make_tile(frame, t, tile)))
    finally:
        cap.release()

    sheets, spans = [], []
    for k in range(0, len(tiles), per_sheet):
        chunk = tiles[k:k + per_sheet]
        rows = math.ceil(len(chunk) / cols)
        sheet = np.full((rows * tile, cols * tile, 3), 8, dtype=np.uint8)
        for i, (_, im) in enumerate(chunk):
            r, c = divmod(i, cols)
            sheet[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = im
        ok, buf = cv2.imencode(".jpg", sheet, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            sheets.append(buf.tobytes())
            spans.append((chunk[0][0], chunk[-1][0]))
    return sheets, spans, duration


# ---------- reply parsing and clean-up ----------

def parse_segments(text: str) -> list[dict[str, Any]]:
    """Segments from a reply: fenced JSON, then the outermost object, then salvaged items."""
    text = text or ""
    obj = None
    for pattern in (r"```(?:json)?\s*(\{.*?\})\s*```", r"\{.*\}"):
        m = re.search(pattern, text, re.S)
        if m:
            try:
                obj = json.loads(m.group(1) if m.groups() else m.group(0))
                break
            except json.JSONDecodeError:
                continue
    items = obj.get("segments", []) if isinstance(obj, dict) else []
    if obj is None:
        items = []
        for m in re.finditer(r"\{[^{}]*\}", text):
            try:
                items.append(json.loads(m.group(0)))
            except json.JSONDecodeError:
                pass
    out = []
    for s in items if isinstance(items, list) else []:
        if not isinstance(s, dict):
            continue
        try:
            a = float(s.get("start_sec", s.get("start", 0)))
            b = float(s.get("end_sec", s.get("end", 0)))
        except (TypeError, ValueError):
            continue
        if a > 10000 or b > 10000:  # milliseconds
            a, b = a / 1000, b / 1000
        if b <= a:
            continue
        label = str(s.get("subtask") or s.get("label") or "event").strip() or "event"
        out.append({"start_sec": a, "end_sec": b, "label": label})
    return sorted(out, key=lambda s: s["start_sec"])


def merge_adjacent(segs: list[dict[str, Any]], gap: float = 0.05) -> list[dict[str, Any]]:
    """Join touching segments with the same label; trim the start of a segment that overlaps the previous one."""
    out: list[dict[str, Any]] = []
    for s in sorted((dict(s) for s in segs), key=lambda s: s["start_sec"]):
        if out and s["start_sec"] <= out[-1]["end_sec"] + gap and s["label"] == out[-1]["label"]:
            out[-1]["end_sec"] = max(out[-1]["end_sec"], s["end_sec"])
        elif out and s["start_sec"] < out[-1]["end_sec"]:
            s["start_sec"] = out[-1]["end_sec"]
            if s["end_sec"] - s["start_sec"] > 1e-3:
                out.append(s)
        else:
            out.append(s)
    return out


def postprocess(segs: list[dict[str, Any]], duration: float, min_seg_sec: float) -> list[dict[str, Any]]:
    """Clamp to the video, resolve overlaps, fold segments shorter than min_seg_sec into the previous one."""
    clamped = []
    for s in segs:
        a = max(0.0, min(duration, s["start_sec"]))
        b = max(a, min(duration, s["end_sec"]))
        if b - a >= 1e-3:
            clamped.append({**s, "start_sec": round(a, 3), "end_sec": round(b, 3)})
    out: list[dict[str, Any]] = []
    for s in merge_adjacent(clamped):
        if min_seg_sec > 0 and out and s["end_sec"] - s["start_sec"] < min_seg_sec:
            out[-1]["end_sec"] = max(out[-1]["end_sec"], s["end_sec"])
        else:
            out.append(s)
    return out


# ---------- model calls ----------

def build_messages(prompt: str, instruction: str, sheets: list[bytes], t0: float, t1: float,
                   sample_sec: float) -> list[dict[str, Any]]:
    text = PROMPTS[prompt].format(sample_sec=sample_sec, t0=t0, t1=t1, instruction=instruction)
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for b in sheets:
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode("ascii")}})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def window_sizes(n: int, max_per_call: int) -> list[int]:
    """Sheets per call to try in order: the whole episode (or max_per_call), then 2/3 of that, down to 1."""
    s = n if max_per_call <= 0 else min(n, max_per_call)
    sizes = [s]
    while s > 1:
        s = max(1, (s * 2) // 3)
        sizes.append(s)
    return sizes


def annotate_episode(ep: dict[str, Any], root: Path, args: argparse.Namespace) -> dict[str, Any]:
    cost = new_cost()
    row: dict[str, Any] = {"episode_id": ep["episode_id"], "segments": [], "failed": True, "cost": cost}
    sheets, spans, duration = build_sheets(root / ep["video"], args.sample_sec, args.tile_px, args.cols,
                                           args.tiles_per_sheet, args.quality)
    row.update(duration=round(duration, 3), n_sheets=len(sheets))
    if not sheets:
        return {**row, "error": "no frames decoded"}
    errors = []
    for size in window_sizes(len(sheets), args.max_sheets_per_call):
        segs, raws, ok = [], [], True
        for k in range(0, len(sheets), size):
            messages = build_messages(args.prompt, ep["task"], sheets[k:k + size], spans[k][0],
                                      spans[min(k + size, len(spans)) - 1][1], args.sample_sec)
            try:
                reply = chat(args.base_url, args.model, messages, api_key=args.api_key,
                             max_tokens=args.max_tokens, timeout=args.timeout, retries=args.retries)
            except RuntimeError as exc:
                cost["n_calls"] += 1
                errors.append(f"size={size}: {exc}"[:200])
                ok = False
                break
            add_usage(cost, reply["usage"])
            window = parse_segments(reply["text"])
            raws.append(reply["text"][:2000])
            if not window and size > 1:
                errors.append(f"size={size}: no segments parsed")
                ok = False
                break
            segs.extend(window)
        if ok:
            return {**row, "segments": postprocess(segs, duration, args.min_seg_sec), "failed": False,
                    "sheets_per_call": size, "raw": raws, "errors": errors}
    return {**row, "error": "; ".join(errors)[:600]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None, help="JSON with defaults for any option below")
    ap.add_argument("--root", type=Path, required=True, help="output root of subtask_eval.data.wgo")
    ap.add_argument("--out", type=Path, required=True, help="run directory")
    ap.add_argument("--name", default="contact_sheet")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="local")
    ap.add_argument("--prompt", default="gepa_duration_prior_v1", choices=sorted(PROMPTS))
    ap.add_argument("--sample-sec", type=float, default=0.5)
    ap.add_argument("--tile-px", type=int, default=224)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--tiles-per-sheet", type=int, default=20)
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--max-sheets-per-call", type=int, default=0, help="0 = whole episode in one call")
    ap.add_argument("--min-seg-sec", type=float, default=0.25)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--subset", default=None, help="e.g. homer; default all subsets")
    ap.add_argument("--limit", type=int, default=0, help="first N episodes only (smoke test)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--retries", type=int, default=2)
    args = parse_with_config(ap)

    episodes = [model_view(e) for e in load_episodes(args.root, args.subset)]
    if args.limit:
        episodes = episodes[: args.limit]
    config = {"annotator": "contact_sheet", **{k: v for k, v in vars(args).items()
                                               if k not in ("config", "config_values", "root", "out")}}
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    failed = run(lambda ep: annotate_episode(ep, args.root, args), episodes, args.out,
                 workers=args.workers, config=config)
    return 0 if failed < len(episodes) or not episodes else 1


if __name__ == "__main__":
    raise SystemExit(main())
