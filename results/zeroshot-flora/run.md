# FLORA zero-shot: a full run under the harness's own input setting

This records adding `nvidia/flora-captioner` to the zero-shot labeler
comparison in `results/zeroshot-v1/`, using the harness's own input setting
throughout: prompt `egoant_raw`, 5 frames per segment, no BEFORE/AFTER
context. This is **not** FLORA's native input format, which uses BEFORE/
TARGET/AFTER temporal blocks sampled at up to 64 frames total; that native
format was not used here, by design, so the result is comparable with the
other labelers in this harness. FLORA's own published HomER number (64.0)
uses that native input and a Gemini judge, so it is **not** directly
comparable to anything below.

Getting a full run took three stages, in order: (1) vLLM could not serve the
checkpoint at all, so `serve_hf.py` was written to serve it with
`transformers` instead; (2) that worked but was far too slow for a 743
segment run within budget; (3) the slowness turned out to be a genuine bug
in the checkpoint's own `generation_config.json` (wrong end-of-turn token
id), not a kernel or hardware limit -- fixing it brought the run from an
estimated 11 hours down to under an hour, with every one of 20 sanity-checked
labels staying byte-for-byte identical to before the fix. Sections below
cover all three stages, since each is evidence for the next.

## Model

- `nvidia/flora-captioner`, revision `669b3f4924b29212da609b292babddfef690f579`
- 27B parameters, Qwen3.5-family hybrid architecture (`Qwen3_5ForConditionalGeneration`):
  linear-attention ("Gated DeltaNet") layers with a full-attention layer every
  4th block
- License: NVIDIA License, non-commercial / evaluation use only (see
  `THIRD_PARTY.md`)

## Why vLLM was not used

Two independent problems, found by actually trying to serve the model with
vLLM `0.34.0` (the same version used for every other labeler and the judge
in this harness):

1. **vLLM cannot load the checkpoint at all.** Serving it (`vllm serve`, the
   same flags used for every other model here) fails during weight loading:

   ```
   ValueError: There is no module or parameter named 'layers.0.mlp.down_proj._extra_state'
   in Qwen3_5Model. The available parameters belonging to layers.0.mlp.down_proj
   (RowParallelLinear) are: {'layers.0.mlp.down_proj.weight'}
   ```

   The checkpoint carries a `._extra_state` tensor on many layers (consistent
   with the `torchao`-related loader code referenced in the model's own
   repository), which this version of vLLM's `Qwen3_5Model` weight loader
   does not expect and refuses to skip.

2. **Even where vLLM can load a model from this same architecture family, it
   does not preserve the model card's recommended mixed precision.** The
   model card states: "Native inference preserves FP32 Gated DeltaNet
   parameters while using BF16 base weights. Preserve these mixed parameter
   dtypes when loading the weights." vLLM's own log for this model family
   reads the config and reports:

   ```
   INFO model.py:2557 Downcasting torch.float32 to torch.bfloat16.
   ```

   followed by a per-tensor warning for every float32-stored tensor it casts
   down, with no exception for the Gated DeltaNet decay (`A_log`) or bias
   (`dt_bias`) parameters. Reading vLLM's own source for this model family
   confirms why: the HF config's `mamba_ssm_dtype` field is only used to set
   the dtype of the runtime SSM state *cache*, not of the weight parameters.

Per the plan for this investigation, neither of these was worked around
(no patched checkpoint, no different vLLM version, no changed cast logic)
once found — the plan was to stop and report instead, which is what this
file is.

## Serving with `transformers` instead

`serve_hf.py` (in this repo, under `wgo_eval/`) is a small OpenAI-compatible
HTTP server that loads the model with `transformers` directly rather than
through vLLM, for exactly this situation. It applies the one-line patch from
NVIDIA's own loading code (see `THIRD_PARTY.md` for the source and license)
that keeps `A_log` and `dt_bias` in FP32 across an otherwise-BF16 load:

```python
Qwen3_5PreTrainedModel._keep_in_fp32_modules_strict = sorted(
    set(getattr(Qwen3_5PreTrainedModel, "_keep_in_fp32_modules_strict", None) or ()) | {"A_log", "dt_bias"}
)
```

### Actual dtype after loading (confirmed, not assumed)

The server prints the dtype of every Gated DeltaNet parameter and one
ordinary weight right after loading. A representative sample from a real
run:

```
DTYPE|gdn   model.language_model.layers.62.linear_attn.dt_bias   torch.float32   (48,)
DTYPE|gdn   model.language_model.layers.62.linear_attn.A_log     torch.float32   (48,)
DTYPE|base  model.visual.patch_embed.proj.weight                 torch.bfloat16
DTYPE|gdn_params_seen  96
```

96 Gated DeltaNet parameters (48 layers x `A_log` + `dt_bias`) were found and
are FP32; every other sampled weight is BF16. This matches the model card's
recommended mixed precision, which plain vLLM does not reproduce for this
model family (see above).

This confirms `transformers` loading is correct; it does not by itself
confirm that `serve_hf.py`'s chat-template and image handling make the model
behave the way FLORA's own native pipeline would, only that this harness's
20-frame egoant_raw-style request round-trips (see next section).

## Serving works, with plausible but non-JSON labels

A 20-segment HomER smoke test (`--subset homer --limit 20 --workers 1`,
unchanged prompt and frame sampling) completed with 0 failed requests. The
model does not reply with the `{"label": ...}` JSON this harness's prompt
asks for; it replies with plain text, often repeating itself or trailing
into unrelated tokens after a first, usually reasonable, sentence. The
harness's `not_json` fallback (first non-empty line) picks up a reasonable
label in every one of the 20 cases, for example:

| Segment | Extracted label (first line) |
|---|---|
| `homer_1/003` | place pitcher on table with right hand |
| `homer_1/006` | hold glass with left hand, pour water from pitcher into glass with right hand |
| `homer_1/008` | place glass on table with left hand, place pitcher on table with right hand |
| `homer_1/017` | Move the wine glass to the left side of the table with the left hand and move the tall glass to the right side of the table with the right hand. |

A full `raw` example (showing the repetition pattern that follows the
usable first line):

```
hold pitcher with right hand, close pitcher with lid in left hand, hold pitcher with left hand, tighten pitcher lid with right hand

close pitcher with lid in left hand, tighten pitcher lid with right hand
hold pitcher with right hand, close pitcher with lid in left hand, hold pitcher with left hand, tighten pitcher lid with right hand
...
```

This is plausible: the harness's 5-frame, no-context input is not the
BEFORE/TARGET/AFTER input FLORA was trained and evaluated on, and the model
is not told (via this harness's prompt) to stop after one sentence the way
its native prompt's "Output only the instruction, on one line" framing does
more explicitly.

## Speed before the fix: kernels did not help, because kernels were not the problem

GPU: one NVIDIA GH200 (120 GB), a full node's GPU dedicated to the job (not
shared with other users), `--workers 1` as required for this single-request
server.

| Setup | Segments | Wall time (label phase) | Per segment |
|---|---|---|---|
| `transformers` load, optional kernels not installed (reference PyTorch fallback for the SSM/conv ops) | 20 | not separately timed; full job (load + label) was 22m55s | — |
| `transformers` load, with `causal-conv1d` and `flash-linear-attention` installed and active (confirmed: no "falling back to its reference PyTorch implementation" warnings in the log) | 20 | 1086 s (timed around the label.py call only) | 54.3 s |

Installing the two optional kernels the model's dependencies recommend did
not meaningfully change the speed. At 54.3 s/segment, 743 segments would
have taken an estimated ~11.2 hours, far past the 3-hour budget -- so the
next step was to find out *why* it was this slow rather than just running
over budget.

## The real cause: a wrong end-of-turn token id in the checkpoint's own config

`nvidia/flora-captioner/generation_config.json` sets `eos_token_id=248044`.
That id is actually the tokenizer's `pad_token` (`<|endoftext|>`), not its
real `eos_token` (`<|im_end|>`, id `248046`) -- confirmed directly:

```
tokenizer.eos_token  <|im_end|>      id 248046
tokenizer.pad_token  <|endoftext|>   id 248044
model.generation_config.eos_token_id  248044   <- wrong: this is the pad id
```

Decoding a real generation with special tokens kept (`skip_special_tokens=False`)
confirms the model knows exactly when to stop -- it emits `<|im_end|>` after
one sentence -- but plain `generate()` never recognises that id as a stop
condition, so it keeps going until `max_new_tokens`, usually repeating the
same answer or drifting into a hallucinated fake chat turn:

```
segment homer_1/000: <|im_end|> at token position 26 of 512
  hold pitcher with right hand, close pitcher with lid in left hand,
  hold pitcher with left hand, tighten pitcher lid with right hand<|im_end|>
  <|im_end|>
  close pitcher with lid in left hand, tighten pitcher lid with right hand<|im_end|>
  hold pitcher with right hand, close pitcher ... (repeats to the token limit)

segment homer_1/001: <|im_end|> at token position 15 of 512
  pass pitcher from left hand to right hand, pick up pitcher with right hand<|im_end|>
  <|im_end|>
  <|im_end|>
  <|im_start|>user
  hold pitcher with right hand<|im_end|>
  ... (hallucinates a fake further conversation turn, then repeats)
```

### Fix

`serve_hf.py` now always passes `eos_token_id=[248044, 248046]` (the
checkpoint's own id, plus the tokenizer's real one -- added, not replaced, in
case a future checkpoint fixes this and the two already agree) to
`generate()`. It also adds an opt-in `--stop-at-newline` stopping criterion
(off by default) that ends generation right after the first line of content;
since this harness only ever keeps the first non-empty line of a non-JSON
reply and decodes greedily, this cannot change any label, only shorten the
unused text after it. Both were on for the run below.

### Equivalence check: identical labels, ~12x faster

20 HomER labels were regenerated with the fix and compared against the
pre-fix labels saved earlier, after `parse_label` (the fallback that keeps
only the first non-empty line): **20/20 identical.** The raw replies
themselves are also now clean -- no more repetition after the first
sentence, matching the extracted label exactly in every case.

| | Wall time (20 segments) | Per segment |
|---|---|---|
| Before fix (with kernels installed) | 1086 s | 54.3 s |
| After fix | 86 s | 4.3 s |

~12.6x faster, with identical output, confirming the slowdown really was
the missing-eos bug and not the Gated DeltaNet kernels.

## Full run

743 segments, judged with `Qwen/Qwen3-32B-AWQ` (the same judge and cache
used throughout this harness), on one GH200 node with FLORA and the judge on
separate GPUs.

| | Wall time | Per segment |
|---|---|---|
| Label (743 segments, `--workers 1`) | 2561 s (42.7 min) | 3.45 s |
| Judge (743 pairs) | 25 s | -- |

0 failed label requests; all 743 replies are non-JSON (expected -- FLORA's
output is plain text, not `{"label": ...}`), picked up by the harness's
first-non-empty-line fallback.

### Scores (judged by the open-weight judge, `Qwen/Qwen3-32B-AWQ`)

| | HomER pooled (n=470) | Full set pooled (n=743) | Full set subset-average |
|---|---|---|---|
| `flora-captioner` | 39.8% [32.95, 48.06] | 38.8% [33.82, 44.4] | 38.4% [32.57, 45.2] |

Per-subset on the full set: homer 39.8% (187/470), galaxea 45.5% (56/123),
robointer_droid 30.0% (45/150).

### Pairwise comparison with the other labelers

Paired bootstrap on accuracy difference (A-B), same segments, same episode
resampling for both labelers. "Clear" means the 95% interval excludes zero.

| A vs B | Subset | A-B pooled | CI95 | Clear? | only A right / only B right |
|---|---|---|---|---|---|
| `flora-captioner` vs `Qwen3.6-27B` | HomER | -7.9 | [-14.2, -0.2] | yes | 55 / 92 |
| `flora-captioner` vs `Qwen3.6-27B` | full | -9.3 | [-14.1, -3.9] | yes | 78 / 147 |
| `flora-captioner` vs `Qwen3-VL-8B-Instruct` | HomER | +3.4 | [-4.7, +13.8] | no | 91 / 75 |
| `flora-captioner` vs `Qwen3-VL-8B-Instruct` | full | +0.5 | [-5.4, +6.6] | no | 124 / 120 |

Under this harness's shared input setting (not FLORA's native one), FLORA
scores clearly below the `Qwen3.6-27B` baseline, and is statistically
indistinguishable from `Qwen3-VL-8B-Instruct`, the harness's main labeler.
