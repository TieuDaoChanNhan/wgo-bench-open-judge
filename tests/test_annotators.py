import json

from subtask_eval.annotators import base, contact_sheet as cs, imported

from .conftest import S


def test_sample_times():
    assert cs.sample_times(2.0, 0.5) == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert cs.sample_times(1.2, 0.5) == [0.0, 0.5, 1.0, 1.2]


def test_parse_segments_variants():
    assert cs.parse_segments('```json\n{"segments":[{"start_sec":1,"end_sec":2,"subtask":"a"}]}\n```') == [S(1.0, 2.0, "a")]
    text = 'ok {"segments":[{"start":3000,"end":20000,"label":"b"},{"start_sec":0,"end_sec":1,"subtask":"a"},{"start_sec":5,"end_sec":4}]}'
    assert cs.parse_segments(text) == [S(0.0, 1.0, "a"), S(3.0, 20.0, "b")]
    assert cs.parse_segments('{"segments":[{"start_sec":0,"end_sec":1,"subtask":"a"}, {"start_sec":1,') == [S(0.0, 1.0, "a")]
    assert cs.parse_segments("no json here") == []


def test_postprocess_clamps_merges_and_folds_short():
    segs = [S(-1, 2, "a"), S(2.02, 3, "a"), S(2.5, 6, "b"), S(6, 6.1, "c"), S(6.1, 20, "d")]
    assert cs.postprocess(segs, duration=10.0, min_seg_sec=0.25) == [S(0.0, 3.0, "a"), S(3.0, 6.1, "b"), S(6.1, 10.0, "d")]


def test_window_sizes():
    assert cs.window_sizes(9, 0) == [9, 6, 4, 2, 1]
    assert cs.window_sizes(9, 3) == [3, 2, 1]
    assert cs.window_sizes(1, 0) == [1]


def test_messages_carry_task_and_range():
    msgs = cs.build_messages("gepa_duration_prior_v1", "pour water", [b"\xff\xd8"], 0.0, 9.5, 0.5)
    text = msgs[1]["content"][0]["text"]
    assert "pour water" in text and "[0.00, 9.50]" in text and msgs[1]["content"][1]["type"] == "image_url"


def test_clean_segments():
    out = base.clean_segments([{"start_sec": 5, "end_sec": 6, "subtask": " b "}, S(0, 1, "a"), S(2, 2, "z")])
    assert out == [S(0.0, 1.0, "a"), S(5.0, 6.0, "b")]


def test_run_loop_resumes_and_survives_errors(tmp_path):
    episodes = [{"episode_id": "e1"}, {"episode_id": "e2"}, {"episode_id": "e3"}]

    def annotate(ep):
        if ep["episode_id"] == "e2":
            raise RuntimeError("boom")
        return {"episode_id": ep["episode_id"], "segments": [S(0, 1, "a")], "duration": 1.0}

    assert base.run(annotate, episodes[:2], tmp_path, workers=2, config={"name": "t"}) == 1
    assert base.run(annotate, episodes, tmp_path, workers=2, config={"name": "t"}) == 0  # only e3 is new
    preds = base.load_predictions(tmp_path)
    assert set(preds) == {"e1", "e2", "e3"} and preds["e2"]["failed"] and "boom" in preds["e2"]["error"]
    meta = json.loads((tmp_path / "run.json").read_text())
    assert len(meta["sessions"]) == 2 and meta["config"]["name"] == "t"


def test_import_egoant_keeps_segments_as_stored(tmp_path):
    (tmp_path / "homer").mkdir()
    (tmp_path / "homer" / "homer_1.json").write_text(json.dumps({"segments": [
        {"start_sec": 0, "end_sec": 2, "subtask": "a"}, {"start_sec": 2, "end_sec": 2, "subtask": "z"}]}))
    assert imported.find_egoant(tmp_path, "homer_1") == tmp_path / "homer" / "homer_1.json"
    assert len(imported.load_egoant(tmp_path / "homer" / "homer_1.json")) == 2
    assert imported.find_egoant(tmp_path, "homer_9") is None
