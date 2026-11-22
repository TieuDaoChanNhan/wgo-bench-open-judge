"""Unit tests for the WGO-Bench harness. No GPU, network or video needed."""

from __future__ import annotations

import argparse
import json

import pytest

from wgo_eval import common, compare, judge, judge_agreement, label, score, wgo_prepare


def test_parse_label_json_and_fallback():
    assert common.parse_label('{"label": "open the lid"}') == ("open the lid", True)
    assert common.parse_label('Sure. {"subtask": "place cup in sink"}') == ("place cup in sink", True)
    assert common.parse_label('"pour water into the glass"\nextra') == ("pour water into the glass", False)
    assert common.parse_label("") == ("", False)


def test_parse_match():
    assert common.parse_match('{"match": true}') is True
    assert common.parse_match('answer: {"match": false}') is False
    assert common.parse_match("TRUE") is True
    with pytest.raises(ValueError):
        common.parse_match("true or false")


def test_message_text_uses_reasoning_when_content_empty():
    assert common.message_text({"content": None, "reasoning_content": '{"match": true}'}) == '{"match": true}'


def test_sample_indices():
    assert label.sample_indices(100, 10, 50, 5) == [10, 20, 30, 40, 50]
    assert label.sample_indices(30, 25, 90, 3) == [25, 27, 29]
    assert label.sample_indices(30, 5, 5, 1) == [5]


def test_model_input_drops_gold_label():
    seg = {"segment_id": "homer_1/000", "video": "v.mp4", "start_sec": 0.0, "end_sec": 1.0,
           "task": "t", "label": "GOLD", "subset": "homer", "episode_id": "homer_1"}
    assert "label" not in label.model_input(seg)
    messages = label.build_messages("egoant_raw", "pour water", [b"\xff\xd8"])
    assert "GOLD" not in json.dumps(messages)
    assert "pour water" in messages[1]["content"][0]["text"]


def test_prepare_rows():
    assert wgo_prepare.subset_of("homer_12") == "homer"
    assert wgo_prepare.subset_of("droid_3") == "droid"
    segs = wgo_prepare.parse_segments('[{"start_sec": 0, "end_sec": 7.7, "subtask": "twist open the lid"}]')
    rows = wgo_prepare.segment_rows("homer_1", "pour water", segs, "videos/homer/homer_1.mp4")
    assert rows == [{"segment_id": "homer_1/000", "episode_id": "homer_1", "subset": "homer",
                     "video": "videos/homer/homer_1.mp4", "start_sec": 0.0, "end_sec": 7.7,
                     "task": "pour water", "label": "twist open the lid"}]


def test_judge_prompt_matches_rubric_slots():
    p = judge.judge_prompt(" gold ", "pred", "task")
    assert "Gold label:\ngold\n" in p and "Predicted label:\npred\n" in p and '{"match": true}' in p


def _segments():
    return [
        {"segment_id": "homer_1/000", "episode_id": "homer_1", "subset": "homer", "label": "a", "task": "t"},
        {"segment_id": "homer_1/001", "episode_id": "homer_1", "subset": "homer", "label": "b", "task": "t"},
        {"segment_id": "homer_2/000", "episode_id": "homer_2", "subset": "homer", "label": "c", "task": "t"},
        {"segment_id": "droid_1/000", "episode_id": "droid_1", "subset": "droid", "label": "d", "task": "t"},
    ]


def test_pairs_from_preds_marks_missing_and_failed():
    preds = [{"segment_id": "homer_1/000", "label": "x", "failed": False},
             {"segment_id": "homer_1/001", "label": "", "failed": True}]
    pairs = judge.pairs_from_preds(_segments(), preds)
    assert [p["pred_failed"] for p in pairs] == [False, True, True, True]


def test_failed_prediction_is_no_match_without_call(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("judge must not be called")

    monkeypatch.setattr(judge, "chat", boom)
    args = argparse.Namespace(model="m")
    row = judge.judge_pair({"id": "x", "gold_label": "a", "pred_label": "", "pred_failed": True}, {}, args)
    assert row["match"] is False


def test_judge_pair_uses_cache(tmp_path, monkeypatch):
    calls = []

    def fake_chat(base_url, model, messages, **kw):
        calls.append(messages)
        return {"text": '{"match": true}', "usage": {}}

    monkeypatch.setattr(judge, "chat", fake_chat)
    args = argparse.Namespace(model="m", base_url="u", api_key="k", max_tokens=64, timeout=1, retries=1,
                              thinking=False, cache=tmp_path / "cache.jsonl")
    cache: dict = {}
    pair = {"id": "x", "gold_label": "a", "pred_label": "b", "task": "t", "pred_failed": False}
    assert judge.judge_pair(pair, cache, args)["match"] is True
    assert judge.judge_pair(pair, cache, args)["cached"] is True
    assert len(calls) == 1
    assert len(common.read_jsonl(args.cache)) == 1


def test_accuracies_pooled_and_subset_average():
    verdicts = {"homer_1/000": True, "homer_1/001": False, "homer_2/000": True, "droid_1/000": True}
    a = score.accuracies(_segments(), verdicts)
    assert a["match"] == 3 and a["n"] == 4
    assert a["pooled"] == pytest.approx(0.75)
    assert a["per_subset"]["homer"]["acc"] == pytest.approx(2 / 3)
    assert a["subset_avg"] == pytest.approx((2 / 3 + 1.0) / 2)


def test_missing_verdict_counts_as_no_match():
    assert score.accuracies(_segments(), {})["pooled"] == 0.0


def test_bootstrap_is_deterministic_and_bounded():
    verdicts = {"homer_1/000": True, "homer_1/001": False, "homer_2/000": True, "droid_1/000": True}
    b1 = score.bootstrap(_segments(), verdicts, draws=200, seed=1)
    b2 = score.bootstrap(_segments(), verdicts, draws=200, seed=1)
    assert b1 == b2
    lo, hi = b1["pooled_ci95"]
    assert 0.0 <= lo <= hi <= 1.0


def test_agreement_and_kappa():
    r = judge_agreement.agreement([True, True, False, False], [True, False, False, False])
    assert r["agreement"] == 0.75
    assert r["table"] == {"ref_true_ours_true": 1, "ref_true_ours_false": 1,
                          "ref_false_ours_true": 0, "ref_false_ours_false": 2}
    assert r["kappa"] == 0.5
    assert judge_agreement.agreement([True, False], [True, False])["kappa"] == 1.0


def test_extract_pairs_fills_task():
    egoant = {"per_episode": [{"id": "homer_1", "pairs": [
        {"pred_i": 4, "gold_j": 6, "gold_label": "g", "pred_label": "p", "semantic_match": True, "judge_model": "gemini"},
        {"pred_i": 5, "gold_j": 7, "gold_label": "g2", "pred_label": "p2", "semantic_match": None},
    ]}]}
    rows = judge_agreement.extract_pairs(egoant, {"homer_1": "pour water"})
    assert len(rows) == 1
    assert rows[0]["id"] == "homer_1/p004-g006" and rows[0]["task"] == "pour water"
    assert rows[0]["reference_match"] is True


def test_compare_identical_labelers_is_not_clear():
    v = {"homer_1/000": True, "homer_1/001": False, "homer_2/000": True, "droid_1/000": True}
    r = compare.compare(_segments(), v, dict(v), draws=200)
    assert r["diff_pooled"] == 0 and r["only_a_right"] == 0 and r["only_b_right"] == 0
    assert r["bootstrap"]["pooled"]["clear"] is False


def test_compare_counts_and_direction():
    va = {s["segment_id"]: True for s in _segments()}
    vb = {"homer_1/000": True}
    r = compare.compare(_segments(), va, vb, draws=200)
    assert r["diff_pooled"] == pytest.approx(0.75)
    assert (r["only_a_right"], r["only_b_right"]) == (3, 0)
    lo, hi = r["bootstrap"]["pooled"]["ci95"]
    assert 0 < lo <= hi
