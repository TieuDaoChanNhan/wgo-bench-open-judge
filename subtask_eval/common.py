"""Shared helpers: JSONL files, run configs, an OpenAI-compatible chat call."""

from __future__ import annotations

import argparse
import json
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


def parse_with_config(ap: argparse.ArgumentParser, argv: list[str] | None = None) -> argparse.Namespace:
    """Parse args; values in --config <file.json> become defaults that the command line can override.

    Config keys use the option names with dashes or underscores (e.g. "max_sheets_per_call").
    Keys that the parser does not know are kept on the namespace too (e.g. "name", "gpus").
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    known, _ = pre.parse_known_args(argv)
    cfg: dict[str, Any] = {}
    if known.config:
        cfg = {k.replace("-", "_"): v for k, v in json.loads(known.config.read_text(encoding="utf-8")).items()}
        ap.set_defaults(**cfg)
    args = ap.parse_args(argv)
    for k, v in cfg.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    args.config_values = cfg
    return args


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
