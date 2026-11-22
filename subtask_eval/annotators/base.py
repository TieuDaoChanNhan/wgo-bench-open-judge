"""What every annotator produces, and the shared run loop.

An annotator turns one episode (video + task, never gold) into labeled segments:

    {"episode_id": "homer_1",
     "segments": [{"start_sec": 0.0, "end_sec": 4.2, "label": "pick up the cup"}, ...],
     "failed": false,
     "duration": 94.0,
     "cost": {"wall_sec": 12.3, "n_calls": 4, "tokens_in": 51234, "tokens_out": 812},
     ...annotator-specific fields}

One JSONL row per episode in <out_dir>/preds.jsonl. <out_dir>/run.json records the config
(including "gpus", the GPUs the annotator's server used) and the wall time of each session;
the report combines these with the episode durations into GPU-hours per video-hour.
Reruns skip episodes already in preds.jsonl.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from ..common import append_jsonl, read_jsonl

Annotate = Callable[[dict[str, Any]], dict[str, Any]]


def new_cost() -> dict[str, float]:
    return {"wall_sec": 0.0, "n_calls": 0, "tokens_in": 0, "tokens_out": 0}


def add_usage(cost: dict[str, float], usage: dict[str, Any]) -> None:
    cost["n_calls"] += 1
    cost["tokens_in"] += int(usage.get("prompt_tokens") or 0)
    cost["tokens_out"] += int(usage.get("completion_tokens") or 0)


def clean_segments(segs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Common output shape: sorted, start < end, a string label under "label"."""
    out = []
    for s in segs:
        a, b = float(s["start_sec"]), float(s["end_sec"])
        if b > a:
            out.append({"start_sec": round(a, 3), "end_sec": round(b, 3),
                        "label": str(s.get("label") or s.get("subtask") or "").strip()})
    return sorted(out, key=lambda s: (s["start_sec"], s["end_sec"]))


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    """preds.jsonl (or a directory holding it) keyed by episode_id; the last row of an episode wins."""
    path = Path(path)
    if path.is_dir():
        path = path / "preds.jsonl"
    return {r["episode_id"]: r for r in read_jsonl(path)}


def run(annotate: Annotate, episodes: list[dict[str, Any]], out_dir: Path, *, workers: int,
        config: dict[str, Any]) -> int:
    """Annotate every episode not yet in out_dir/preds.jsonl. Returns the number of failed episodes."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = out_dir / "preds.jsonl"
    done = set(load_predictions(preds))
    todo = [e for e in episodes if e["episode_id"] not in done]
    print(f"ANNOTATE|{config.get('name', '?')} episodes={len(episodes)} done={len(done)} todo={len(todo)}", flush=True)

    t0 = time.time()
    failed = 0

    def timed(ep: dict[str, Any]) -> dict[str, Any]:
        start = time.time()
        try:
            row = annotate(ep)
        except Exception as exc:  # one bad episode must not stop the run
            row = {"episode_id": ep["episode_id"], "segments": [], "failed": True,
                   "error": f"{type(exc).__name__}: {exc}"[:300]}
        row.setdefault("cost", new_cost())
        row["cost"]["wall_sec"] = round(time.time() - start, 3)
        row["segments"] = clean_segments(row.get("segments") or [])
        row["failed"] = bool(row.get("failed")) or not row["segments"]
        return row

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(timed, e) for e in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            append_jsonl(preds, row)
            failed += row["failed"]
            print(f"ANNOTATE|{i}/{len(todo)} {row['episode_id']} segs={len(row['segments'])} "
                  f"calls={row['cost']['n_calls']} failed={failed}", flush=True)

    meta_path = out_dir / "run.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"sessions": []}
    meta["config"] = {k: v for k, v in config.items() if k not in ("api_key",)}
    meta["sessions"].append({"started": t0, "wall_sec": round(time.time() - t0, 3), "episodes": len(todo),
                             "workers": workers})
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return failed
