"""Unit tests for wgo_eval/serve_hf.py's request conversion. No GPU, network or torch needed."""

from __future__ import annotations

import base64

import pytest

from wgo_eval import serve_hf


def test_decode_data_url_roundtrip():
    raw = b"\xff\xd8\xff\xe0not a real jpeg but bytes"
    url = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
    mime, data = serve_hf.decode_data_url(url)
    assert mime == "image/jpeg"
    assert data == raw


def test_decode_data_url_rejects_non_data_url():
    with pytest.raises(ValueError):
        serve_hf.decode_data_url("https://example.com/frame.jpg")


def test_extract_image_bytes_order_and_count():
    raw1, raw2 = b"frame-one", b"frame-two"

    def data_url(raw: bytes) -> str:
        return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")

    messages = [
        {"role": "system", "content": "You label segments."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Episode task: pour water"},
                {"type": "image_url", "image_url": {"url": data_url(raw1)}},
                {"type": "image_url", "image_url": {"url": data_url(raw2)}},
            ],
        },
    ]
    assert serve_hf.extract_image_bytes(messages) == [raw1, raw2]


def test_extract_image_bytes_ignores_string_content():
    messages = [{"role": "system", "content": "no images here"}]
    assert serve_hf.extract_image_bytes(messages) == []


@pytest.mark.parametrize(
    "body,cap,expected",
    [
        ({"max_tokens": 256}, 512, 256),
        ({"max_tokens": 4096}, 512, 512),
        ({"max_tokens": 0}, 512, 1),
        ({}, 512, 512),
        ({"max_tokens": "not a number"}, 512, 512),
    ],
)
def test_requested_max_tokens_is_clamped(body, cap, expected):
    assert serve_hf.requested_max_tokens(body, cap) == expected


@pytest.mark.parametrize(
    "generation_config_eos,extra_id,expected",
    [
        (248044, 248046, [248044, 248046]),  # the real flora-captioner case: wrong + right id
        ([248046], 248046, [248046]),  # already correct: no duplicate
        (None, 5, [5]),
        (None, None, []),
        ([1, 2], 2, [1, 2]),
    ],
)
def test_combine_eos_token_ids(generation_config_eos, extra_id, expected):
    assert serve_hf.combine_eos_token_ids(generation_config_eos, extra_id) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("hello\nworld", True),
        ("hello", False),
        ("", False),
        ("   ", False),
        ("\nhello", False),  # no non-whitespace content yet, leading newline doesn't count
        ("\n  hello\nworld", True),
        ("hello\n", True),
    ],
)
def test_should_stop_at_newline(text, expected):
    assert serve_hf.should_stop_at_newline(text) == expected


class _FakeTokenizer:
    """decode() just joins the given tokens; good enough to drive should_stop_at_newline."""

    def decode(self, ids, skip_special_tokens=True):
        return "".join(ids)


class _FakeInputIds:
    """Mimics a torch tensor's input_ids[0, start:] 2D indexing over a single row of fake tokens."""

    def __init__(self, row: list[str]):
        self.row = row

    def __getitem__(self, idx):
        row_idx, sl = idx
        assert row_idx == 0
        return self.row[sl]


def test_stop_at_newline_triggers_after_first_line():
    prompt = ["P", "P"]
    generated = ["h", "e", "l", "l", "o", "\n", "w", "o", "r", "l", "d"]
    criterion = serve_hf.StopAtNewline(_FakeTokenizer(), prompt_len=len(prompt))
    assert criterion(_FakeInputIds(prompt + generated), scores=None) is True


def test_stop_at_newline_false_before_any_newline():
    prompt = ["P"]
    generated = ["h", "e", "l", "l", "o"]
    criterion = serve_hf.StopAtNewline(_FakeTokenizer(), prompt_len=len(prompt))
    assert criterion(_FakeInputIds(prompt + generated), scores=None) is False


def test_stop_at_newline_ignores_prompt_tokens_before_prompt_len():
    # a newline inside the prompt itself (before prompt_len) must not count
    prompt = ["\n", "\n"]
    generated = ["h", "i"]
    criterion = serve_hf.StopAtNewline(_FakeTokenizer(), prompt_len=len(prompt))
    assert criterion(_FakeInputIds(prompt + generated), scores=None) is False
