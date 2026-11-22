"""Shared fixtures: a tiny WGO-style eval set written to a temp root. No video, GPU or network."""

from __future__ import annotations

import pytest

from subtask_eval.common import write_jsonl


def seg_rows():
    return [
        {"segment_id": "homer_1/001", "episode_id": "homer_1", "subset": "homer", "video": "videos/homer/homer_1.mp4",
         "start_sec": 4.0, "end_sec": 8.0, "task": "t", "label": "b"},
        {"segment_id": "homer_1/000", "episode_id": "homer_1", "subset": "homer", "video": "videos/homer/homer_1.mp4",
         "start_sec": 0.0, "end_sec": 4.0, "task": "t", "label": "a"},
        {"segment_id": "galaxea_2/000", "episode_id": "galaxea_2", "subset": "galaxea", "video": "videos/galaxea/galaxea_2.mp4",
         "start_sec": 0.0, "end_sec": 3.0, "task": "u", "label": "c"},
    ]


@pytest.fixture
def wgo_root(tmp_path):
    write_jsonl(tmp_path / "segments.jsonl", seg_rows())
    return tmp_path


def S(a, b, label="x"):
    return {"start_sec": a, "end_sec": b, "label": label}
