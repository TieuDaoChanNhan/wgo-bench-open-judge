import argparse

import pytest

from subtask_eval.common import message_text, read_jsonl
from subtask_eval.scoring import judge


def test_parse_match():
    assert judge.parse_match('{"match": true}') is True
    assert judge.parse_match('answer: {"match": false}') is False
    assert judge.parse_match("TRUE") is True
    with pytest.raises(ValueError):
        judge.parse_match("true or false")


def test_message_text_uses_reasoning_when_content_empty():
    assert message_text({"content": None, "reasoning_content": '{"match": true}'}) == '{"match": true}'


def test_judge_prompt_slots():
    p = judge.judge_prompt(" gold ", "pred", "task")
    assert "Gold label:\ngold\n" in p and "Predicted label:\npred\n" in p and '{"match": true}' in p


def test_empty_label_is_no_match_without_call(monkeypatch):
    monkeypatch.setattr(judge, "chat", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no call")))
    row = judge.judge_pair({"id": "x", "gold_label": "a", "pred_label": " "}, {}, argparse.Namespace(model="m"))
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
    pair = {"id": "x", "gold_label": "a", "pred_label": "b", "task": "t"}
    assert judge.judge_pair(pair, cache, args)["match"] is True
    assert judge.judge_pair(pair, cache, args)["cached"] is True
    assert len(calls) == 1 and len(read_jsonl(args.cache)) == 1
