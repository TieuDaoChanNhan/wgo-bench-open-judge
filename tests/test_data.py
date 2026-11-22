from subtask_eval.common import parse_with_config
from subtask_eval.data import wgo


def test_prepare_rows():
    assert wgo.subset_of("homer_12") == "homer"
    assert wgo.subset_of("robointer_droid_3") == "robointer_droid"
    segs = wgo.parse_segments('[{"start_sec": 0, "end_sec": 7.7, "subtask": "twist open the lid"}]')
    rows = wgo.segment_rows("homer_1", "pour water", segs, "videos/homer/homer_1.mp4")
    assert rows == [{"segment_id": "homer_1/000", "episode_id": "homer_1", "subset": "homer",
                     "video": "videos/homer/homer_1.mp4", "start_sec": 0.0, "end_sec": 7.7,
                     "task": "pour water", "label": "twist open the lid"}]


def test_load_episodes_groups_and_orders(wgo_root):
    eps = wgo.load_episodes(wgo_root)
    assert [e["episode_id"] for e in eps] == ["galaxea_2", "homer_1"]
    homer = eps[1]
    assert homer["category"] == "ego" and eps[0]["category"] == "robot"
    assert [g["label"] for g in homer["gold"]] == ["a", "b"]
    assert [e["episode_id"] for e in wgo.load_episodes(wgo_root, "homer")] == ["homer_1"]


def test_model_view_drops_gold(wgo_root):
    view = wgo.model_view(wgo.load_episodes(wgo_root)[1])
    assert set(view) == {"episode_id", "video", "task"}


def test_parse_with_config(tmp_path):
    import argparse

    cfg = tmp_path / "c.json"
    cfg.write_text('{"model": "m-from-config", "max-tokens": 99, "gpus": 2}', encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--model")
    ap.add_argument("--max-tokens", type=int, default=1)
    args = parse_with_config(ap, ["--config", str(cfg), "--max-tokens", "5"])
    assert args.model == "m-from-config" and args.max_tokens == 5 and args.gpus == 2


def test_parse_with_config_satisfies_required_options(tmp_path):
    import argparse

    import pytest

    cfg = tmp_path / "c.json"
    cfg.write_text('{"model": "m"}', encoding="utf-8")

    def parser():
        ap = argparse.ArgumentParser()
        ap.add_argument("--config")
        ap.add_argument("--model", required=True)
        return ap

    assert parse_with_config(parser(), ["--config", str(cfg)]).model == "m"
    with pytest.raises(SystemExit):
        parse_with_config(parser(), [])
