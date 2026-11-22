# WGO-Bench zero-shot: run zeroshot-v1

Full zero-shot run of the WGO-Bench subtask-labeling benchmark (homer, galaxea,
robointer_droid subsets, 743 segments total) on a single node with 4 GPUs, one
server per GPU (judge + three labelers). Judge agreement with the released
Gemini verdicts was run separately on 1 GPU. The pairwise comparisons between
the three labelers were computed afterwards on CPU from the saved verdicts;
adding the comparison script did not change the labeling, judging or scoring
code. The dataset is at the pinned revision below.

## Versions and hardware

- Dataset: `macrodata/WGO-Bench`, revision `1c86a62e5302` (100 episodes, 743
  segments: homer 470, galaxea 123, robointer_droid 150)
- Judge-agreement reference pairs: EgoANT, at the version listed in
  [`THIRD_PARTY.md`](../../THIRD_PARTY.md)
- vLLM `0.34.0`, PyTorch `2.14.0+cu130`
- GPU: 4x NVIDIA GH200 120GB on one node, one vLLM server per GPU
- Models (served name @ Hugging Face revision):
  - labeler (under test): `Qwen/Qwen3-VL-8B-Instruct` @ `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
  - judge: `Qwen/Qwen3-32B-AWQ` @ `0499c3ac83fdef8810b907a23894ba91e95eddd8`
  - baseline labeler: `Qwen/Qwen3.6-27B` @ `1cda14dd71a3f47115850eb790f416f165da1cb5`
  - baseline labeler: `cyankiwi/Qwen3.8-27B-AWQ-INT4` @ `e3a700906e6a73ba4290b1873dc2ba8198527ab5`

## Throughput

Wall time to label all 743 segments, and to judge all 743 predictions
afterwards against the shared judge server:

| Labeler | Label wall time | Label throughput | Judge wall time |
|---|---|---|---|
| `Qwen3-VL-8B-Instruct` | 80 s | 9.29 seg/s | 59 s |
| `Qwen3.6-27B` | 145 s | 5.12 seg/s | 59 s |
| `cyankiwi/Qwen3.8-27B-AWQ-INT4` | 137 s | 5.42 seg/s | 59 s |

All three labelers: 0 failed requests and 0 non-JSON replies out of 743
segments each.

## Accuracy (judged by the open-weight judge, `Qwen/Qwen3-32B-AWQ`)

Pooled accuracy with a 95% episode-bootstrap interval; HomER is the main
subset, the full set adds galaxea and robointer_droid.

| Labeler | HomER pooled (n=470) | Full set pooled (n=743) | Full set subset-average |
|---|---|---|---|
| `Qwen3-VL-8B-Instruct` | 36.4% [28.9, 43.8] | 38.2% [32.7, 43.5] | 39.7% [33.8, 46.2] |
| `Qwen3.6-27B` | 47.7% [40.0, 56.0] | 48.0% [42.0, 53.9] | 48.7% [42.2, 55.8] |
| `cyankiwi/Qwen3.8-27B-AWQ-INT4` | 41.5% [32.5, 50.2] | 43.6% [36.9, 49.9] | 45.6% [39.0, 52.7] |

The published protocol scores `Qwen3.6-27B` zero-shot at 55.7 on HomER with a
`gemini-3.5-flash` judge. Under this harness, with the same model and prompt
but the open judge above, it scores 47.7 on HomER — about 8 points lower. The
judge-agreement result below shows why: the open judge is never more lenient
than Gemini on these data, only stricter.

## Judge agreement with Gemini

79 HomER (gold label, predicted label, Gemini verdict) pairs released by
EgoANT, re-judged with the open judge used throughout this run:

- Raw agreement: 89.9%
- Cohen's kappa: 0.758 (substantial agreement)
- 2x2 table: both say match 52, Gemini match / ours no-match 8, Gemini
  no-match / ours match 0, both say no-match 19

All 8 disagreements go the same direction — Gemini calls it a match, the open
judge does not — and none the other way:

| Gold label | Predicted label | Gemini | Open judge |
|---|---|---|---|
| scoop white powder from the clear container using an orange measuring spoon | scoop contents from large jar | match | no match |
| wash the glass mug with a sponge under running water | wash the glass | match | no match |
| pick up the beige container from the countertop | Pick up the jar | match | no match |
| tighten the screw on the wheel of the yellow toy excavator using a green screwdriver | Tighten screw on toy truck wheel | match | no match |
| use a green screwdriver to tighten the screw on the toy excavator's wheel | Screw the wheel onto the toy truck | match | no match |
| use a green screwdriver to tighten the screw on the toy excavator's wheel | Screw the wheel onto the toy truck | match | no match |
| push a black tire onto the left-front wheel hub of the yellow toy excavator | Attach wheel to toy truck | match | no match |
| draw a palm tree in the middle of hills with a pen | Draw palm tree on paper | match | no match |

Most of these are cases where the predicted label names a more generic object
or vehicle type ("jar" for a container, "toy truck" for an excavator) than
the gold label — the open judge treats that as a wrong object, Gemini
apparently does not always.

## Pairwise comparison between labelers

Paired bootstrap on accuracy difference (A − B), same segments, same episode
resampling for both labelers so shared per-episode difficulty cancels out.
"Clear" means the 95% interval excludes zero.

| A vs B | Subset | A−B pooled | CI95 | Clear? | only A right / only B right |
|---|---|---|---|---|---|
| `Qwen3.6-27B` vs `Qwen3-VL-8B-Instruct` | HomER | +11.3 | [+5.5, +18.6] | yes | 77 / 24 |
| `Qwen3.6-27B` vs `Qwen3-VL-8B-Instruct` | full | +9.8 | [+5.3, +14.5] | yes | 119 / 46 |
| `Qwen3.6-27B` vs `cyankiwi/Qwen3.8-27B-AWQ-INT4` | HomER | +6.2 | [+1.4, +12.1] | yes | 67 / 38 |
| `Qwen3.6-27B` vs `cyankiwi/Qwen3.8-27B-AWQ-INT4` | full | +4.4 | [+0.6, +8.8] | yes | 96 / 63 |
| `cyankiwi/Qwen3.8-27B-AWQ-INT4` vs `Qwen3-VL-8B-Instruct` | HomER | +5.1 | [-1.0, +10.9] | **no** | 64 / 40 |
| `cyankiwi/Qwen3.8-27B-AWQ-INT4` vs `Qwen3-VL-8B-Instruct` | full | +5.4 | [+0.9, +9.6] | yes | 106 / 66 |

`Qwen3.6-27B` is clearly ahead of both other labelers on every slice. The
8B-vs-AWQ gap is clear on the full set but not on HomER alone (470 segments is
not quite enough to resolve a ~5-point gap there).

## What had to change to run this on the cluster, and why

The benchmark harness itself (`wgo_eval/*.py`) needed no changes and all of
its unit tests passed unmodified (17/17). Everything below is about the
orchestration layer (Slurm job scripts, environment setup) written for this
specific cluster, kept in `slurm/` in this results directory with cluster-
specific paths replaced by `$W` (the working directory) and the account name
replaced by a placeholder. The scripts source a cluster-specific `$W/env.sh`,
not included here, which loads the cluster's GCC, Python 3.13 and CUDA 13
modules and exports `PY_VLLM` (the vLLM environment's Python), `REPO`,
`WGO_ROOT`, `MODELS_DIR`, `RUNS_DIR` and `HF_HOME`. Changes:

- A job script sourced its own environment file using a path computed from
  the script's own location (`dirname "${BASH_SOURCE[0]}"`). Slurm copies a
  submitted script into its own spool directory before running it, so that
  computed path pointed nowhere and the job failed immediately. Fixed by
  using a fixed path instead.
- The vLLM venv's `vllm` console script was missing on this install, even
  though pip's package metadata registered it; `python -m
  vllm.entrypoints.cli.main serve ...` was used instead. Also, naming an
  environment variable with a `VLLM_` prefix in the harness's own env file
  triggered vLLM's own environment-variable scanner, which logs a warning
  for any `VLLM_*` variable it does not recognize; renamed to avoid that
  prefix entirely.
- A liveness check used `pgrep -f "vllm serve"` to detect whether the server
  process was still running while it finished loading. Once the invocation
  changed to the `python -m vllm.entrypoints.cli.main serve` form above, the
  process's command line no longer contained that literal substring, so the
  check always reported the server as dead after the first failed readiness
  poll and aborted the job within seconds, well before the model had
  finished loading. Fixed by matching the new invocation string instead
  (the full zero-shot job script tracks the server's exact process ID
  instead of pattern-matching the command line, and was not affected).
- On the shared GPU used only for interactive syntax/logic smoke tests before
  trusting the job queue (never for a result in this report), a low
  `--gpu-memory-utilization` value had to be raised because the GPU was
  shared with other users' processes; this does not apply to the actual
  job runs above, each of which has a full GPU to itself and uses 0.90.

No change was made to the prompt, frame sampling, judge rubric, or dataset
revision.
