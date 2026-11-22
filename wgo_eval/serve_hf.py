"""Minimal OpenAI-compatible HTTP server for a model vLLM cannot load correctly.

    python -m wgo_eval.serve_hf --model-dir /path/to/flora-captioner \
        --served-model-name nvidia/flora-captioner --port 8010

Implements just enough of the OpenAI chat-completions API for this harness's
label.py to use unmodified: GET /v1/models and POST /v1/chat/completions,
accepting the same messages shape label.py sends (text + image_url content
items with a data:image/...;base64,... URL) and returning
{"choices": [{"message": {"content": ...}}], "usage": {...}}.

Written for `nvidia/flora-captioner`, whose checkpoint vLLM 0.34.0 fails to
load (see results/zeroshot-flora/run.md) and whose recommended
inference precision (BF16 weights, FP32 Gated DeltaNet A_log/dt_bias
parameters) vLLM does not reproduce even when it does load a same-family
model. Loading here follows transformers directly, with the upstream
NVlabs/Flora `_keep_in_fp32_modules_strict` patch applied before
`from_pretrained` (see THIRD_PARTY.md for the exact source and license).

Requests are handled one at a time (a lock around generate()); run label.py
with --workers 1 against this server.

The checkpoint's own generation_config.json has the wrong eos_token_id (see
combine_eos_token_ids' docstring), so generate() is always given the
tokenizer's real eos id (<|im_end|>) too. --stop-at-newline additionally
stops generation after the first line of content; see should_stop_at_newline.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

_DATA_URL_RE = re.compile(r"^data:([^;]+);base64,(.*)$", re.S)


def decode_data_url(url: str) -> tuple[str, bytes]:
    """(mime type, raw bytes) from a data: URL. Raises ValueError for anything else."""
    m = _DATA_URL_RE.match(url or "")
    if not m:
        raise ValueError(f"not a data: URL: {url[:40]!r}")
    mime, b64 = m.groups()
    return mime, base64.b64decode(b64)


def extract_image_bytes(messages: list[dict[str, Any]]) -> list[bytes]:
    """Ordered raw image bytes from every image_url content item, across all messages.

    Order matches how the model's chat template walks the same `messages` list to
    place one vision placeholder token per image_url item, so the two must agree.
    """
    out: list[bytes] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image_url":
                url = (item.get("image_url") or {}).get("url", "")
                if url:
                    _, data = decode_data_url(url)
                    out.append(data)
    return out


def requested_max_tokens(body: dict[str, Any], cap: int) -> int:
    """Caller's max_tokens, clamped to [1, cap] (cap: the model's recommended ceiling)."""
    want = body.get("max_tokens")
    try:
        want = int(want)
    except (TypeError, ValueError):
        want = cap
    return max(1, min(want, cap))


def combine_eos_token_ids(generation_config_eos: int | list[int] | None, extra_id: int | None) -> list[int]:
    """Deduplicated eos token ids: the checkpoint's own generation_config.eos_token_id plus one more.

    nvidia/flora-captioner's generation_config.json has eos_token_id=248044, which is the
    tokenizer's pad_token_id (<|endoftext|>), not its eos_token (<|im_end|>, id 248046) -- so
    plain generate() never recognizes the turn-end token the model actually emits. extra_id is
    meant to be that tokenizer eos id (<|im_end|>), added in rather than substituted, in case a
    future checkpoint fixes generation_config.json and the two already agree.
    """
    ids: set[int] = set()
    if isinstance(generation_config_eos, int):
        ids.add(generation_config_eos)
    elif generation_config_eos:
        ids.update(generation_config_eos)
    if extra_id is not None:
        ids.add(extra_id)
    return sorted(ids)


def should_stop_at_newline(text: str) -> bool:
    """Whether `text` holds a non-whitespace character followed later by a newline.

    Used by StopAtNewline below to end generation right after the first line of content:
    this harness only keeps the first non-empty line of a non-JSON reply
    (wgo_eval.common.parse_label's fallback) and decodes greedily, so stopping here cannot
    change the label the harness records, only shorten the (otherwise often repetitive or
    off-topic) text generated after it.
    """
    return "\n" in text.lstrip()


class StopAtNewline:
    """A transformers StoppingCriteria-shaped callable: stop once the newly generated text
    (decoded from the tokens after `prompt_len`) satisfies should_stop_at_newline(). Not a
    transformers.StoppingCriteria subclass, so this module needs no transformers import at
    load time; StoppingCriteriaList only calls it, it does not check its type.
    """

    def __init__(self, tokenizer: Any, prompt_len: int):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
        text = self.tokenizer.decode(input_ids[0, self.prompt_len :], skip_special_tokens=True)
        return should_stop_at_newline(text)


class FloraModel:
    """Loads nvidia/flora-captioner with transformers, serving one request at a time."""

    def __init__(
        self,
        model_dir: str,
        max_new_tokens_cap: int = 512,
        device: str = "cuda:0",
        stop_at_newline: bool = False,
    ):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5PreTrainedModel

        # NVlabs/Flora captioner/batch_utils/vendor/production_gdn_fix.py (see THIRD_PARTY.md):
        # transformers only keeps named modules in FP32 across a bf16 load when they are
        # listed here; Qwen3.5's own base class does not list the Gated DeltaNet decay/bias
        # parameters, so without this they load as bf16 like everything else.
        Qwen3_5PreTrainedModel._keep_in_fp32_modules_strict = sorted(
            set(getattr(Qwen3_5PreTrainedModel, "_keep_in_fp32_modules_strict", None) or ()) | {"A_log", "dt_bias"}
        )

        self._torch = torch
        self.device = device
        self.max_new_tokens_cap = max_new_tokens_cap
        self.stop_at_newline = stop_at_newline
        self.lock = threading.Lock()

        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_dir, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(device).eval()

        im_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.eos_token_ids = combine_eos_token_ids(self.model.generation_config.eos_token_id, im_end_id)

        self._report_dtypes()
        print(
            f"DTYPE|eos_token_ids\tgeneration_config={self.model.generation_config.eos_token_id}"
            f"\t<|im_end|>={im_end_id}\tused={self.eos_token_ids}",
            flush=True,
        )

    def _report_dtypes(self) -> None:
        """Print actual runtime dtype of A_log/dt_bias plus one ordinary weight, to confirm the patch."""
        gdn_seen = 0
        for name, p in self.model.named_parameters():
            if name.endswith((".A_log", ".dt_bias")):
                print(f"DTYPE|gdn\t{name}\t{p.dtype}\t{tuple(p.shape)}", flush=True)
                gdn_seen += 1
        for name, p in self.model.named_parameters():
            if name.endswith(".weight") and not name.endswith((".A_log", ".dt_bias")):
                print(f"DTYPE|base\t{name}\t{p.dtype}", flush=True)
                break
        print(f"DTYPE|gdn_params_seen\t{gdn_seen}", flush=True)

    def generate(self, messages: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
        import io

        from PIL import Image

        images = [Image.open(io.BytesIO(b)).convert("RGB") for b in extract_image_bytes(messages)]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=images or None, return_tensors="pt").to(self.model.device)
        new_tokens = requested_max_tokens({"max_tokens": max_tokens}, self.max_new_tokens_cap)

        gen_kwargs: dict[str, Any] = {"max_new_tokens": new_tokens, "do_sample": False, "eos_token_id": self.eos_token_ids}
        if self.stop_at_newline:
            from transformers import StoppingCriteriaList

            gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [StopAtNewline(self.processor.tokenizer, prompt_len=inputs["input_ids"].shape[1])]
            )

        with self.lock, self._torch.no_grad():
            t0 = time.time()
            out = self.model.generate(**inputs, **gen_kwargs)
            latency_s = time.time() - t0

        gen = out[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
        n_in, n_out = int(inputs["input_ids"].shape[1]), int(gen.shape[1])
        return {
            "text": text,
            "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out},
            "latency_s": latency_s,
        }


def make_handler(model: "FloraModel", served_name: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, obj: Any, code: int = 200) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (http.server naming)
            if self.path.rstrip("/") == "/v1/models":
                self._send_json({"object": "list", "data": [{"id": served_name, "object": "model"}]})
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send_json({"error": "not found"}, 404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            try:
                result = model.generate(body.get("messages") or [], max_tokens=body.get("max_tokens", 256))
            except Exception as exc:  # noqa: BLE001 (report any failure to the caller, not just expected ones)
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)
                return
            self._send_json({"choices": [{"message": {"content": result["text"]}}], "usage": result["usage"]})

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet default stderr access log -> stdout
            print("HTTP|" + (fmt % args), flush=True)

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--served-model-name", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=512, help="FLORA's recommended ceiling")
    ap.add_argument(
        "--stop-at-newline",
        action="store_true",
        default=False,
        help="stop generation right after the first line of content (see should_stop_at_newline docstring)",
    )
    args = ap.parse_args()

    model = FloraModel(args.model_dir, max_new_tokens_cap=args.max_new_tokens, stop_at_newline=args.stop_at_newline)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(model, args.served_model_name))
    print(f"READY|{args.served_model_name} on {args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
