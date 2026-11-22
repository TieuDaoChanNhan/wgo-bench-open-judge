"""Shared helpers: JSONL files, an OpenAI-compatible chat call, reply parsing."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

import requests


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def message_text(message: dict[str, Any]) -> str:
    """Text of a chat completion message; falls back to reasoning fields when content is empty."""
    content = message.get("content")
    if isinstance(content, list):
        content = "\n".join(str(c.get("text") or "") for c in content if isinstance(c, dict))
    if content:
        return str(content)
    for key in ("reasoning_content", "reasoning"):
        if message.get(key):
            return str(message[key])
    return ""


def chat(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    api_key: str = "local",
    max_tokens: int = 256,
    timeout: float = 300.0,
    retries: int = 3,
    disable_thinking: bool = True,
) -> dict[str, Any]:
    """One chat completion with temperature 0. Returns {"text", "usage"}; raises after the last retry."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if disable_thinking:
        # vLLM forwards this to the chat template of Qwen3-family models.
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = base_url.rstrip("/") + "/chat/completions"
    last = ""
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            return {"text": message_text(data["choices"][0]["message"]), "usage": data.get("usage") or {}}
        except Exception as exc:  # network, HTTP or malformed reply
            last = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(2 * attempt, 10))
    raise RuntimeError(last[:300])


_JSON_OBJECT = re.compile(r"\{[^{}]*\}", re.S)


def parse_label(text: str) -> tuple[str, bool]:
    """Label from a labeler reply and whether it came from valid JSON.

    Accepts {"label": ...} or {"subtask": ...}. When no JSON object holds a label,
    the first non-empty line is used (as EgoANT does) and flagged as not JSON.
    """
    for match in _JSON_OBJECT.finditer(text or ""):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            value = obj.get("label") or obj.get("subtask")
            if isinstance(value, str) and value.strip():
                return value.strip(), True
    for line in (text or "").splitlines():
        line = line.strip().strip('"').strip()
        if line:
            return line, False
    return "", False


def parse_match(text: str) -> bool:
    """Verdict from a judge reply: JSON {"match": bool}, else the first true/false found."""
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("match"), bool):
            return obj["match"]
    except json.JSONDecodeError:
        pass
    m = re.search(r'"match"\s*:\s*(true|false)', text, re.I)
    if m:
        return m.group(1).lower() == "true"
    lowered = text.lower()
    if "true" in lowered and "false" not in lowered:
        return True
    if "false" in lowered and "true" not in lowered:
        return False
    raise ValueError(f"cannot parse judge reply: {text[:200]}")
