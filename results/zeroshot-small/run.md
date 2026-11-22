# WGO-Bench zero-shot: run zeroshot-small

Zero-shot run of two small VLMs (`Qwen3-VL-2B-Instruct`, `Qwen3-VL-4B-Instruct`) on the full
WGO-Bench subtask-labeling benchmark (homer, galaxea, robointer_droid subsets, 743 segments
total), to see how accuracy changes with model size within one family. Uses the
unmodified harness: same dataset revision, same `egoant_raw` prompt (5 frames,
longest side 640 px, JPEG quality 85), same judge model and shared judge cache as the earlier
`zeroshot-v1` run. The `Qwen3-VL-8B-Instruct` numbers below are not re-run here; they are read
directly from `zeroshot-v1/zs-qwen3vl8b/`.

## Versions and hardware

- Dataset: `macrodata/WGO-Bench`, revision `1c86a62e5302` (100 episodes, 743 segments: homer
  470, galaxea 123, robointer_droid 150)
- vLLM `0.34.0`, PyTorch `2.14.0+cu130`
- GPU: 3x NVIDIA GH200 120GB on one node (one server per model + judge)
- Models (served name @ Hugging Face revision):
  - `Qwen/Qwen3-VL-2B-Instruct` @ `89644892e4d85e24eaac8bacfd4f463576704203`
  - `Qwen/Qwen3-VL-4B-Instruct` @ `ebb281ec70b05090aa6165b016eac8ec08e71b17`
  - judge: `Qwen/Qwen3-32B-AWQ` @ `0499c3ac83fdef8810b907a23894ba91e95eddd8`
  - reference (not re-run): `Qwen/Qwen3-VL-8B-Instruct` @ `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
    (from `zeroshot-v1`)
- Server config matches `zeroshot-v1`'s labeler servers exactly, so throughput is comparable:
  `--max-model-len 16384 --limit-mm-per-prompt '{"image": 8}' --max-num-seqs 16`

## Throughput and peak VRAM

Wall time to label all 743 segments, and peak GPU memory (`nvidia-smi memory.used`, sampled
every 2s from server start through the end of labeling, so it includes the weight-loading
peak, not just steady-state inference):

| Labeler | Label wall time | Label throughput | Peak VRAM |
|---|---|---|---|
| `Qwen3-VL-2B-Instruct` | 44 s | 16.89 seg/s | 87,440 MiB |
| `Qwen3-VL-4B-Instruct` | 45 s | 16.51 seg/s | 87,358 MiB |
| `Qwen3-VL-8B-Instruct` (from `zeroshot-v1`) | 80 s | 9.29 seg/s | not measured in that run |

Both small models: **0 failed requests and 0 non-JSON replies** out of 743 segments each.

Peak VRAM for the two small models is nearly identical (87.4 GiB vs. 87.4 GiB) despite the 2x
parameter-count difference -- at `--gpu-memory-utilization 0.90` on a 120 GB GH200, vLLM's KV
cache pre-allocation dominates and essentially fills the budget regardless of model size; this
number reflects vLLM's allocation policy at this setting, not the models' own footprint. The
8B run's peak VRAM was not recorded (sampling was added for this run only), so it is not
directly comparable here.

## Accuracy (judged by `Qwen/Qwen3-32B-AWQ`)

Pooled accuracy with a 95% episode-bootstrap interval; HomER is the main subset, the full set
adds galaxea and robointer_droid.

| Labeler | HomER pooled (n=470) | Full set pooled (n=743) | Full set subset-average |
|---|---|---|---|
| `Qwen3-VL-2B-Instruct` | 34.9% [29.2, 41.2] | 34.3% [29.8, 39.4] | 33.9% [28.5, 40.0] |
| `Qwen3-VL-4B-Instruct` | 39.1% [30.3, 48.2] | 37.8% [31.7, 44.1] | 36.7% [30.8, 43.4] |
| `Qwen3-VL-8B-Instruct` (from `zeroshot-v1`) | 36.4% [28.9, 43.8] | 38.2% [32.7, 43.5] | 39.7% [33.8, 46.2] |

## Pairwise comparison

Paired bootstrap on accuracy difference (A - B), same segments, same episode resampling for
both sides so shared per-episode difficulty cancels out. "Clear" means the 95% interval
excludes zero.

| A vs B | Subset | A-B pooled | CI95 | Clear? | only A right / only B right |
|---|---|---|---|---|---|
| `Qwen3-VL-2B-Instruct` vs `Qwen3-VL-8B-Instruct` | HomER | -1.5 | [-8.4, +6.5] | no | 54 / 61 |
| `Qwen3-VL-2B-Instruct` vs `Qwen3-VL-8B-Instruct` | full | -3.9 | [-8.5, +1.0] | no | 80 / 109 |
| `Qwen3-VL-4B-Instruct` vs `Qwen3-VL-8B-Instruct` | HomER | +2.8 | [-2.8, +9.7] | no | 55 / 42 |
| `Qwen3-VL-4B-Instruct` vs `Qwen3-VL-8B-Instruct` | full | -0.4 | [-4.6, +4.2] | no | 81 / 84 |
| `Qwen3-VL-4B-Instruct` vs `Qwen3-VL-2B-Instruct` | HomER | +4.3 | [-2.4, +11.1] | no | 62 / 42 |

**None of the five comparisons are statistically clear at n=743** (every 95% interval crosses
zero) -- 2B, 4B and 8B are not distinguishable from each other at this sample size under this
harness and judge. The point estimates have 4B slightly ahead of both 2B and 8B on HomER and
roughly level with 8B on the full set, but none of that is backed by a clear interval.

## Model size

At zero-shot, going from 2B to 4B gives a ~4 point lift on HomER that is not statistically clear,
and both small models label much faster than 8B (16.5-16.9 seg/s vs. 9.3 seg/s). Going from 4B to
8B gives nothing further on this benchmark: the 4B point estimate is not below 8B on any slice,
and the full-set difference is -0.4 points. This says only that the smaller models are not
clearly worse without training; it says nothing about how they compare after fine-tuning.
