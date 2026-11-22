"""Identify the robointer_droid rows in WGO-Bench and the RoboInter-Data episodes they came from.

    python -m wgo_eval.wgo_robointer_episodes

Reads WGO-Bench at the pinned revision (same dataset, same download call as wgo_prepare.py), filters
rows whose `id` starts with "robointer_droid", and for each pulls:

  id                                -> wgo_id, episode_index (WGO's own numbering, parsed from id)
  metadata (a JSON string column)   -> source_dataset, source_episode_id, source_family,
                                        source_camera_view, source_subset, source_video_member
  perception_state.source_revision  -> source_revision (the HF revision of source_dataset used)

Writes results/wgo_robointer_episodes.json: one entry per row, sorted by episode_index,
plus a top-level exclude_droid_episodes list of every distinct RoboInter-Data episode referenced --
the numeric id before the camera view in source_episode_id, e.g. "10010" from
"10010_exterior_image_2_left" (droid_episode_root below). This is the list to exclude by when sourcing
new RoboInter-Data episodes for training: one physical episode can appear under several camera views,
and only the one camera view pinned in these 50 gold rows would otherwise be caught.

Uses only WGO-Bench's own metadata; does not download RoboInter-Data itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from .wgo_prepare import DATASET, PARQUET, REVISION

OUT_PATH = Path(__file__).resolve().parent.parent / "results" / "wgo_robointer_episodes.json"


def droid_episode_root(source_episode_id: str, source_camera_view: str) -> str:
    """The RoboInter-Data episode id without its camera view suffix.

    "10010_exterior_image_2_left" with camera view "exterior_image_2_left" -> "10010". Falls back to
    splitting on the first underscore if the camera view isn't a suffix of the episode id (defensive;
    not expected to trigger on the data this was written for).
    """
    suffix = "_" + source_camera_view if source_camera_view else ""
    if suffix and source_episode_id.endswith(suffix):
        return source_episode_id[: -len(suffix)]
    return source_episode_id.split("_", 1)[0]


def exclude_list(episodes: list[dict[str, Any]]) -> list[str]:
    """Sorted, de-duplicated droid_episode_root values across `episodes`."""
    return sorted({e["source_droid_episode"] for e in episodes})


def build_episodes(parquet_path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    episodes: list[dict[str, Any]] = []
    for batch in pf.iter_batches(batch_size=4, columns=["id", "metadata", "perception_state"]):
        for rec in batch.to_pylist():
            if not rec["id"].startswith("robointer_droid"):
                continue
            md = json.loads(rec["metadata"])
            source_episode_id = md.get("source_episode_id")
            source_camera_view = md.get("source_camera_view")
            episodes.append(
                {
                    "wgo_id": rec["id"],
                    "episode_index": md.get("episode_index"),
                    "source_dataset": md.get("source_dataset"),
                    "source_revision": rec["perception_state"]["source_revision"],
                    "source_episode_id": source_episode_id,
                    "source_droid_episode": droid_episode_root(source_episode_id, source_camera_view),
                    "source_family": md.get("source_family"),
                    "source_camera_view": source_camera_view,
                    "source_subset": md.get("source_subset"),
                    "source_video_member": md.get("source_video_member"),
                }
            )
    episodes.sort(key=lambda e: e["episode_index"])
    return episodes


def main() -> int:
    path = Path(hf_hub_download(DATASET, PARQUET, repo_type="dataset", revision=REVISION))
    episodes = build_episodes(path)
    excl = exclude_list(episodes)

    out = {
        "purpose": (
            "Exclusion list: these are the RoboInter-Data (DROID) episodes already used as gold "
            "segments in WGO-Bench robointer_droid, identified from the dataset's own metadata. When "
            "sourcing new RoboInter-Data episodes for training, exclude every episode listed in "
            "exclude_droid_episodes -- drop all of that episode's camera views, not only the one "
            "camera view pinned in this benchmark's gold rows."
        ),
        "wgo_dataset": DATASET,
        "wgo_dataset_revision": REVISION,
        "n_robointer_droid_rows": len(episodes),
        "source_dataset": "InternRobotics/RoboInter-Data",
        "source_dataset_revision": episodes[0]["source_revision"] if episodes else None,
        "identification_fields": ["source_episode_id", "source_camera_view", "episode_index"],
        "exclude_droid_episodes": excl,
        "episodes": episodes,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"n_robointer_droid_rows": len(episodes), "n_distinct_source_droid_episodes": len(excl)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
