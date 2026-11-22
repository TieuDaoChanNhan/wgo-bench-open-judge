"""Unit tests for wgo_eval/wgo_robointer_episodes.py. No network needed; uses fake rows."""

from __future__ import annotations

from wgo_eval.wgo_robointer_episodes import droid_episode_root, exclude_list


def test_droid_episode_root_strips_camera_view_suffix():
    assert droid_episode_root("10010_exterior_image_2_left", "exterior_image_2_left") == "10010"
    assert droid_episode_root("1001_exterior_image_1_left", "exterior_image_1_left") == "1001"


def test_droid_episode_root_falls_back_to_first_underscore_if_suffix_does_not_match():
    # camera_view that isn't actually a suffix of the episode id: fall back rather than raise
    assert droid_episode_root("10010_exterior_image_2_left", "something_else") == "10010"


def test_droid_episode_root_empty_camera_view():
    assert droid_episode_root("10010_exterior_image_2_left", "") == "10010"


def test_exclude_list_dedupes_and_sorts():
    episodes = [
        {"source_droid_episode": "10020"},
        {"source_droid_episode": "10010"},
        {"source_droid_episode": "10010"},  # same episode, different camera view row
        {"source_droid_episode": "1002"},
    ]
    # plain lexicographic string sort, same as the committed file's exclude_droid_episodes order
    assert exclude_list(episodes) == ["10010", "1002", "10020"]


def test_exclude_list_empty():
    assert exclude_list([]) == []
