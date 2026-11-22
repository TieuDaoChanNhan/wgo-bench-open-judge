# WGO-Bench subtask labeling: evaluation harness

This repository evaluates video captioners that label manipulation subtasks. It runs the subtask labeling task of [WGO-Bench](https://huggingface.co/datasets/macrodata/WGO-Bench) with gold segment boundaries, scores the labels with an open-weight judge model, and compares labelers with paired statistics. Every model runs behind an OpenAI-compatible endpoint (vLLM), so any vision-language model can be evaluated without changing the code.

## Contents

1. [Task](#task)
2. [Protocol](#protocol)
3. [Judge](#judge)
4. [Results](#results)
5. [Running the benchmark](#running-the-benchmark)
6. [Evaluating another model](#evaluating-another-model)
7. [Code layout](#code-layout)
8. [Limitations](#limitations)
9. [Sources and licenses](#sources-and-licenses)

## Task

An episode is a manipulation video with a high-level instruction and a list of gold subtask segments. For each segment, the labeler sees frames from that segment only, together with the episode instruction, and writes one short imperative phrase describing the completed event.

```
episode instruction : "use pitcher to pour water into tall glass and wine glass"
segment             : 0.0 s – 7.7 s
labeler input       : 5 frames sampled evenly in [0.0, 7.7] + the instruction
labeler output      : {"label": "twist open the pitcher lid"}
gold label          : "twist open the pitcher lid"
judge verdict       : {"match": true}
```

Segment boundaries are given, so the task measures labeling quality alone, independent of temporal segmentation.

## Protocol

| Item | Setting |
|---|---|
| Dataset | `macrodata/WGO-Bench`, revision `1c86a62e5302` (pinned; earlier revisions carry older HomER annotations) |
| Main subset | HomER, egocentric human video: 25 episodes, 470 segments |
| Full set | 100 episodes, 743 segments: HomER, RoboInter DROID (50 episodes, 150 segments), RoboCOIN Galaxea (25 episodes, 123 segments) |
| Frames | 5 per segment, sampled evenly from the first to the last frame of the segment, longest side 640 px, JPEG quality 85 |
| Labeler prompt | `egoant_raw` (see `label.py`); asks for one imperative phrase with the main object and the destination when visible, returned as JSON |
| Decoding | temperature 0, at most 256 new tokens, thinking disabled for models that support it |
| Judge | text only: gold label, predicted label and episode instruction; never the video |
| Failed outputs | an empty reply or no decodable frames counts as no match; replies that are not valid JSON are kept (first line) and counted separately |
| Use of the data | evaluation only; no training, prompt tuning or model selection on WGO-Bench |

**Metrics.**

- **Pooled accuracy:** matched segments / gold segments. This is how the Macrodata write-up reports labeling accuracy.
- **Subset average:** the mean of the per-subset accuracies, so each source counts equally. This is how the Flora paper reports it.
- **95% intervals** come from an episode-level bootstrap (2,000 draws, resampling episodes within each subset). Segments of one episode are correlated, so resampling segments would understate the uncertainty.
- **Comparisons** between two labelers use a paired bootstrap: each draw scores both labelers on the same resampled episodes. A difference is called clear only when its 95% interval excludes zero.

## Judge

The reference protocol uses `gemini-3.5-flash` as judge. This harness uses an open-weight model served locally, with the same rubric, so that evaluation needs no API access and no data leaves the machine:

- Model: [`Qwen/Qwen3-32B-AWQ`](https://huggingface.co/Qwen/Qwen3-32B-AWQ), the 4-bit AWQ release published by the Qwen team.
- Settings: temperature 0, thinking disabled, at most 64 output tokens.
- Rubric: the label judge rubric from the Macrodata write-up, unchanged (`judge.py`). A label matches when the main action, the main object and, where central, the source or destination are right; synonyms and slightly less detail are accepted; vague or hallucinated labels are rejected.

**Agreement with the reference judge.** The EgoANT project released 79 HomER label pairs with `gemini-3.5-flash` verdicts. Re-judged by the open judge:

| | Gemini: match | Gemini: no match |
|---|---|---|
| Open judge: match | 52 | 0 |
| Open judge: no match | 8 | 19 |

Agreement is 89.9% and Cohen's kappa 0.76. Every disagreement goes the same way: the open judge rejects 8 of the 60 labels Gemini accepts and never accepts a label Gemini rejects. Scores from this harness are therefore stricter than Gemini-judged scores and should not be compared with them directly. Qwen3.6-27B, which EgoANT reports at 55.7% on HomER with Gemini, scores 47.7% here with the same prompt and frames, a shift consistent with the agreement table. With 79 pairs, the agreement estimate itself is uncertain by several points.

## Results

Zero-shot, open judge, revision `1c86a62e5302`, one NVIDIA GH200 per server. Every run labeled 743/743 segments with 0 failed replies. All labelers except Flora were served with vLLM 0.34 and returned valid JSON for every segment; Flora was served with `serve_hf.py` (vLLM cannot load it) and never returns JSON, so its first line is used as the label.

| Labeler | HomER pooled | Full set pooled | Full set subset avg. | Labeling speed |
|---|---|---|---|---|
| Qwen3.6-27B | **47.7** [40.0, 56.0] | **48.0** [42.0, 53.9] | **48.7** [42.2, 55.8] | 5.1 seg/s |
| Qwen3.8-27B (AWQ INT4) | 41.5 [32.5, 50.2] | 43.6 [36.9, 49.9] | 45.6 [39.0, 52.7] | 5.4 seg/s |
| nvidia/flora-captioner | 39.8 [33.0, 48.1] | 38.8 [33.8, 44.4] | 38.4 [32.6, 45.2] | 0.29 seg/s (transformers, 1 request at a time) |
| Qwen3-VL-4B-Instruct | 39.1 [30.3, 48.2] | 37.8 [31.7, 44.1] | 36.7 [30.8, 43.4] | 16.5 seg/s |
| Qwen3-VL-8B-Instruct | 36.4 [28.9, 43.8] | 38.2 [32.7, 43.5] | 39.7 [33.8, 46.2] | 9.3 seg/s |
| Qwen3-VL-2B-Instruct | 34.9 [29.2, 41.2] | 34.3 [29.8, 39.4] | 33.9 [28.5, 40.0] | 16.9 seg/s |

Flora is run with the same five frames and prompt as every other labeler, not with its native input (dense frames from the segment and its neighbours), so this row measures it under this protocol only.

Accuracy in %, 95% episode-bootstrap intervals in brackets. Paired comparisons on HomER (A − B, 95% paired interval): Qwen3.6-27B − Qwen3-VL-8B +11.3 [+5.5, +18.6]; Qwen3.6-27B − Qwen3.8-27B +6.2 [+1.4, +12.1]; Qwen3.8-27B − Qwen3-VL-8B +5.1 [−1.0, +10.9], not clear; Flora − Qwen3.6-27B −7.9 [−14.2, −0.2]; Flora − Qwen3-VL-8B +3.4 [−4.7, +13.8], not clear. No pair among Qwen3-VL 2B, 4B and 8B is clearly different. Per-segment predictions, verdicts, comparisons and run details are in [`results/zeroshot-v1/`](results/zeroshot-v1/run.md), [`results/zeroshot-small/`](results/zeroshot-small/run.md) and [`results/zeroshot-flora/`](results/zeroshot-flora/run.md).

For context, published numbers on HomER with the Gemini judge: Flora captioner 64.0 and GPT-6 Astra 64.0 (Flora paper, arXiv 2610.31779), Qwen3.6-27B 55.7 and Qwen3.5-397B 50.2 (EgoANT report). These use a different judge and are not directly comparable with the table above.

## Running the benchmark

Requirements: an NVIDIA GPU with enough memory for the labeler (the 8B model needs about 20 GB, the 27B models in bf16 about 60 GB) and the judge (about 20 GB), Python 3.10 or newer, and [vLLM](https://docs.vllm.ai) in its own environment. On a single 32 GB GPU the labeler and the judge are served one after the other; with more memory or several GPUs they can run side by side.

**1. Environment and data**

```bash
git clone https://github.com/TieuDaoChanNhan/wgo-bench-open-judge.git && cd wgo-bench-open-judge
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python -m venv .venv-vllm && .venv-vllm/bin/pip install vllm

.venv/bin/python -m wgo_eval.wgo_prepare --root data/wgo     # downloads the pinned revision
.venv/bin/python -m pytest -q tests                 # unit tests, no GPU needed
```

`data/wgo/manifest.json` should report 100 episodes and 743 segments.

**2. Label**

```bash
.venv-vllm/bin/vllm serve Qwen/Qwen3-VL-8B-Instruct --host 127.0.0.1 --port 8000 \
    --max-model-len 16384 --limit-mm-per-prompt '{"image": 8}' &

.venv/bin/python -m wgo_eval.label --root data/wgo --out runs/qwen3vl8b/preds.jsonl \
    --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3-VL-8B-Instruct
```

Add `--subset homer` to label HomER only, `--limit 20` for a smoke test. Interrupted runs resume where they stopped.

**3. Judge**

```bash
.venv-vllm/bin/vllm serve Qwen/Qwen3-32B-AWQ --host 127.0.0.1 --port 8001 --max-model-len 4096 &

.venv/bin/python -m wgo_eval.judge --root data/wgo --preds runs/qwen3vl8b/preds.jsonl \
    --out runs/qwen3vl8b/judged.jsonl --base-url http://127.0.0.1:8001/v1 \
    --model Qwen/Qwen3-32B-AWQ --cache runs/judge_cache.jsonl
```

Verdicts are cached by judge model and prompt, so judging the same pair twice costs nothing.

**4. Score and compare**

```bash
.venv/bin/python -m wgo_eval.score --root data/wgo --subset homer \
    --judged runs/qwen3vl8b/judged.jsonl --preds runs/qwen3vl8b/preds.jsonl --out runs/qwen3vl8b/score_homer.json

.venv/bin/python -m wgo_eval.compare --root data/wgo --subset homer \
    --a runs/qwen36-27b/judged.jsonl --b runs/qwen3vl8b/judged.jsonl
```

**5. Judge agreement (optional)**

```bash
git clone https://github.com/ShuaiZhou302/EgoANT.git egoant && git -C egoant checkout b27ea59f5f
.venv/bin/python -m wgo_eval.judge_agreement extract --root data/wgo --out runs/agree/pairs.jsonl \
    --egoant-json egoant/reference_outputs/wgo_homer_v5_1/scores/semantic_e2e_gemini35.json
.venv/bin/python -m wgo_eval.judge --pairs runs/agree/pairs.jsonl --out runs/agree/judged.jsonl \
    --base-url http://127.0.0.1:8001/v1 --model Qwen/Qwen3-32B-AWQ --cache runs/judge_cache.jsonl
.venv/bin/python -m wgo_eval.judge_agreement report --pairs runs/agree/pairs.jsonl --judged runs/agree/judged.jsonl
```

`scripts/` holds the same steps as shell scripts for a single-GPU Ubuntu VM (`setup.sh`, `serve.sh`, `run_zeroshot.sh`). On a Slurm cluster, download weights, data and packages on a node with internet access and set `HF_HUB_OFFLINE=1` in the job; the servers listen on `127.0.0.1`, so compute nodes need no outside network.

## Evaluating another model

Serve the model with any OpenAI-compatible server and pass its name and URL to `label.py`. Keep the prompt, the frame settings, the judge and the dataset revision unchanged; these define the benchmark. A new prompt can be added to `PROMPTS` in `label.py` under a new name, and results with it should be reported as a separate setting.

If vLLM cannot load a model correctly (wrong weight dtypes, or a checkpoint format it rejects outright), `serve_hf.py` is a minimal OpenAI-compatible server that loads the model with `transformers` instead and exposes the same two endpoints `label.py` needs. It serves one request at a time, so run `label.py` against it with `--workers 1`. It was written for `nvidia/flora-captioner`; see `results/zeroshot-flora/run.md` for why vLLM did not work there.

## Code layout

| File | Role |
|---|---|
| `wgo_eval/wgo_prepare.py` | download the pinned dataset; write one mp4 per episode and one JSONL row per gold segment |
| `wgo_eval/wgo_robointer_episodes.py` | identify the RoboInter-Data (DROID) episodes behind `robointer_droid`, for exclusion from training data |
| `wgo_eval/label.py` | sample frames, call the labeler, parse `{"label": ...}`; the gold label is removed before the model call |
| `wgo_eval/judge.py` | judge predicted against gold labels with the rubric; cached verdicts |
| `wgo_eval/score.py` | pooled and subset-average accuracy with episode-bootstrap intervals |
| `wgo_eval/compare.py` | paired bootstrap comparison of two labelers |
| `wgo_eval/judge_agreement.py` | agreement of the judge with the released Gemini verdicts |
| `wgo_eval/serve_hf.py` | OpenAI-compatible server over `transformers`, for a model vLLM cannot load correctly |
| `wgo_eval/common.py` | JSONL helpers, the chat call, reply parsing |
| `tests/` | unit tests (no GPU, network or video needed) |
| `scripts/` | the same steps as shell scripts for a single-GPU VM |
| `results/` | committed predictions, verdicts and scores of reported runs |

## Limitations

- **Judge.** Scores depend on the judge. The open judge is stricter than the reference Gemini judge; agreement is measured on 79 pairs only. No human evaluation of the judge has been done yet.
- **Benchmark size.** HomER has 470 segments from 25 episodes, so 95% intervals are about ±7 points for a single labeler. Differences of a few points are not distinguishable.
- **Domain.** HomER shows household tasks. Results may not carry over to other settings such as workshops or factories.
- **Fixed inputs.** Five frames per segment without neighbouring context. Longer segments may lose detail; other input choices are possible but are separate settings.

## Sources and licenses

- WGO-Bench data: [macrodata/WGO-Bench](https://huggingface.co/datasets/macrodata/WGO-Bench), CC-BY-NC-SA-4.0. Protocol and judge rubric: [Macrodata, *Segmenting Robot Video into Actionable Subtasks*](https://macrodata.co/blog/annotating-robot-video-subtasks).
- Data preparation, the `egoant_raw` prompt and the transcribed rubric are adapted from [EgoANT](https://github.com/ShuaiZhou302/EgoANT), MIT license; the exact version used is listed in [`THIRD_PARTY.md`](THIRD_PARTY.md).
- Reference numbers: Flora paper (arXiv 2610.31779) and the EgoANT report.

**This repository.** The code is MIT-licensed ([`LICENSE`](LICENSE)). The files under `results/` contain WGO-Bench content and are distributed under CC BY-NC-SA 4.0; Flora's labels are also subject to its non-commercial NVIDIA License ([`results/LICENSE.md`](results/LICENSE.md)). WGO-Bench videos are not redistributed here; `wgo_prepare.py` downloads them from Hugging Face.
