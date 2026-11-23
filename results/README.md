# Results

All results are on WGO-Bench, revision `1c86a62e5302` (100 episodes, 743 segments). They were
produced from a clean checkout. Each folder's `provenance.json` records the exact commit, packages,
GPUs and model revisions. Cluster paths in these files are shown as `<workdir>`.

Metrics are explained in the [main README](../README.md#metrics).

## `wgo-bench/scorer-check/`

Checks that our scorer matches the reference scorer. EgoANT's released HomER segmentations are
imported and scored. The reference counts are 308 predicted, 470 gold, 79 matched, F1 0.203, and
we reproduce them exactly.

| File | Content |
|---|---|
| `preds.jsonl` | EgoANT's segmentations, one row per episode |
| `score_homer.json` | Our score for them |
| `provenance.json` | How the check was run |

## `wgo-bench/judge-validation/`

Checks which open judge reproduces the `gemini-3.5-flash` verdicts used by published numbers. It
covers 79 HomER label pairs with Gemini verdicts, released by EgoANT. See the
[judge section](../README.md#label-judge).

| File | Content |
|---|---|
| `pairs.jsonl` | The 79 pairs: human label, predicted label, task, Gemini verdict (`reference_match`) |
| `judged_qwen3-32b-awq.jsonl`, `judged_qwen3.8-27b.jsonl` | Each judge's verdict per pair |
| `agreement_*.json` | Agreement, Cohen's kappa, accept rates and the 2×2 table |
| `disagreements_gemini_vs_*.jsonl` | The pairs where a judge differs from Gemini |
| `provenance.json` | How the judges were run |

## `wgo-bench/baseline-qwen3.6-27b/`

The baseline pipeline: `Qwen/Qwen3.6-27B` over timestamped contact sheets, with the GEPA prompt of
the WGO-Bench write-up and one call per episode (config: [`configs/cs-qwen36-27b.json`](../configs/cs-qwen36-27b.json)).
It is judged by `Qwen/Qwen3-32B-AWQ`, and also by `Qwen/Qwen3.8-27B` for comparison.

| File | Content |
|---|---|
| `preds.jsonl` | Predicted segments and labels per episode, with the raw model reply, tokens and time |
| `run.json` | Config and wall time |
| `pairs.jsonl` | The 110 predicted/human segment pairs that match (IoU ≥ 0.75) |
| `judged.jsonl` | Verdicts of `Qwen3-32B-AWQ` (default judge) |
| `score_all.json`, `score_<subset>.json` | All metrics, overall and per subset, with per-episode detail |
| `judged_qwen3.8-27b.jsonl`, `score_all_qwen3.8-27b.json` | The same, judged by `Qwen3.8-27B` |
| `agreement_*.json`, `disagreements_*.jsonl` | The two judges compared on these 110 pairs |
| `provenance.json` | Two entries: annotation and default judge, then the second judge |

**Known failure:** one episode, `robointer_droid_000008`, has no prediction. The model answers
`{"segments": []}` for it on every attempt, and it scores zero.
