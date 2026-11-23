# wgo-bench-open-judge

An open evaluation harness for **robot-video subtask annotation**: splitting a manipulation video
into its steps and naming each step.

```text
episode video + task ("put the dishes away")
        │  annotation pipeline under test
        ▼
0.0–4.2 s  "pick up the cup"   │  4.2–9.8 s  "place the cup in the sink"   │  ...
        │  compared with human annotations
        ▼
segment F1 · label accuracy · end-to-end F1 · granularity · cost
```

The benchmark is [WGO-Bench](https://huggingface.co/datasets/macrodata/WGO-Bench) by Macrodata:
100 episodes with 743 human-annotated segments. An **episode** is one recorded demonstration of one
task: a single video from a single camera, with its task instruction. The episodes come from three
sources:

- HomER: egocentric human video, 25 episodes;
- DROID: robot arm, 50 episodes;
- Galaxea: robot head camera, 25 episodes.

Everything runs on open-weight models served locally, **including the judge** that grades the labels.

## Results

The baseline reads each episode as timestamped contact sheets (grids of frames) and asks a
vision-language model for the segments in one call. The method and prompt come from
[EgoANT](https://github.com/ShuaiZhou302/EgoANT). The model, `Qwen/Qwen3.6-27B`, is our choice: it
is the default model of LeRobot's `lerobot-annotate`. The judge is `Qwen/Qwen3-32B-AWQ`
(see [Label judge](#label-judge)). 95% confidence intervals are in brackets.

| Subset | Segment F1 | Label accuracy on matched segments | End-to-end F1 |
|---|---|---|---|
| DROID | 0.376 | 0.47 | 0.177 |
| Galaxea | 0.179 | 0.44 | 0.080 |
| HomER | 0.119 | 0.56 | 0.067 |
| **All 100 episodes** | **0.193** [0.153, 0.240] | **0.50** | **0.096** [0.069, 0.130] |

**Granularity**, predicted vs human:

| Subset | Segments per episode | Mean segment length |
|---|---|---|
| DROID | 2.6 vs 3.0 | 10.8 s vs 10.0 s |
| Galaxea | 3.1 vs 4.9 | 5.0 s vs 3.6 s |
| HomER | 7.5 vs 18.8 | 12.4 s vs 5.0 s |
| **All** | **4.0 vs 7.4** | **10.4 s vs 5.8 s** |

**Cost:** 0.042 GPU-hours per hour of video on one NVIDIA GH200 (1.24 hours of video in 187 s),
with 405k input tokens, 15k output tokens and 86 model calls per hour of video.

Reading the numbers:

- **The baseline segments too coarsely.** The gap is largest on HomER, where humans mark 19 steps
  per episode and the model finds 7.5. This is the main reason HomER scores lowest.
- **For reference:** the WGO-Bench write-up reports a best segment F1 of 0.306 with Gemini 3.5 Flash.
- Full predictions, verdicts, scores and provenance are in [`results/`](results/README.md).

## Metrics

The protocol follows the WGO-Bench write-up and the reference scorer released with EgoANT.

**Segment F1** asks whether the predicted boundaries are right.
1. A predicted segment *matches* a human segment when they overlap by at least 75%, measured as
   **IoU** (intersection over union: the shared duration divided by the combined duration).
2. Each segment can be matched once. Pairs are taken from the highest IoU down.
3. Before matching, the first predicted start and the last predicted end are aligned with the
   human annotation, because where an episode "really" starts and ends is ambiguous.
4. Two rates follow:
   - **precision** = matched / predicted segments (how many predictions are right);
   - **recall** = matched / human segments (how many human segments were found).
5. **F1** is their harmonic mean, so it is high only when both are. Counts are pooled over all episodes.

**Label accuracy on matched segments** asks whether the matched segments are named correctly. A
judge model reads the human label and the predicted label and answers match or no match, using the
Macrodata rubric: same action, same object, and same source or destination where it matters;
synonyms are fine.

**End-to-end F1** is segment F1 where a match must also pass the judge. It is the single number for
"right boundaries *and* right names".

**Granularity** is the number of segments per episode and their mean length, predicted vs human. A
pipeline that splits much coarser or finer than the annotators loses F1 even when its segments are
sensible, so read F1 together with granularity.

**Cost** is GPU-hours, tokens and model calls per hour of video.
- GPU-hours depend on the hardware: the same run takes fewer GPU-hours on a faster GPU. Compare
  GPU-hours only between runs on the same GPU type; every run records its GPUs in its provenance.
- Tokens and calls do not depend on the hardware, so they compare across machines.

All F1 values come with 95% intervals from an **episode bootstrap**: the episodes are resampled
2,000 times and the score recomputed each time. Episodes, not segments, are resampled, because
segments of one episode are not independent.

## Label judge

Published WGO-Bench numbers use `gemini-3.5-flash` as judge. To stay open, we replace it with a
local model and pick the one that best reproduces Gemini's verdicts. EgoANT released 79 label pairs
with Gemini's verdicts; we judged the same 79 pairs with two candidates.

| | `Qwen/Qwen3-32B-AWQ` (used) | `Qwen/Qwen3.8-27B` |
|---|---|---|
| Agreement with Gemini | **0.90** (71/79) | 0.70 (55/79) |
| Cohen's kappa | **0.76** | 0.42 |
| Gemini accepts, judge rejects | 8 | 24 |
| Gemini rejects, judge accepts | 0 | 0 |

`Qwen3-32B-AWQ` agrees with Gemini more often, so we use it. Neither judge accepts a label that
Gemini rejects; both are stricter than Gemini, `Qwen3.8-27B` much more so.

**Agreement** is the share of pairs on which the judge gives Gemini's verdict.

**Cohen's kappa** is agreement corrected for the agreement two judges would reach by chance:

```text
kappa = (p_o - p_e) / (1 - p_e)

p_o = observed agreement
p_e = P(A says match) · P(B says match) + P(A says no match) · P(B says no match)
```

Worked example for `Qwen3-32B-AWQ`, from its 2×2 table:

| | Judge: match | Judge: no match | Total |
|---|---|---|---|
| **Gemini: match** | 52 | 8 | 60 |
| **Gemini: no match** | 0 | 19 | 19 |
| **Total** | 52 | 27 | 79 |

```text
p_o   = (52 + 19) / 79                          = 0.899
p_e   = (60/79)(52/79) + (19/79)(27/79)         = 0.582
kappa = (0.899 - 0.582) / (1 - 0.582)           = 0.76
```

Kappa is 1 for perfect agreement and 0 for chance level. As a rule of thumb, 0.4–0.6 is moderate
and above 0.6 is substantial.

**What the choice changes:** segment F1 does not use the judge. With `Qwen3.8-27B` instead, the
baseline's label accuracy would drop from 0.50 to 0.42 and its end-to-end F1 from 0.096 to 0.081.
Because both open judges are stricter than Gemini, label and end-to-end scores here should be
compared with other runs of this harness, not with published tables.

## Reproducibility

Reproducible means that someone else can rerun the evaluation and get the same numbers, and can
see exactly what produced each committed result. Three checks cover this.

| What | How it was checked | Result |
|---|---|---|
| **Scorer** | Scored the HomER predictions EgoANT released, and compared with EgoANT's published counts | Identical: 308 predicted, 470 human, 79 matched, F1 0.203 |
| **Judges** | Judged the same 79 pairs twice | Identical verdicts, 79/79, for both judges |
| **Annotator** | Ran the whole baseline twice from scratch with the same code and models | 83 of 100 episodes identical; segment F1 0.191 vs 0.193; label accuracy and end-to-end F1 unchanged |

The annotator decodes greedily, so it is nearly but not exactly deterministic: vLLM batches
requests differently from run to run, which changes the rounding inside the model slightly.

### Provenance

**Provenance** is a record of where a result came from. Every job writes a `provenance.json` next
to its outputs, with:

- **code:** the git commit, and any uncommitted changes;
- **environment:** installed Python packages, vLLM version and the GPUs used;
- **models:** the exact model revisions served;
- **data:** the dataset revision and a list of the files used;
- **config:** the settings of the run.

With this file, a result can be traced back to the exact code, models and data, and rerun. The
committed results were produced from a clean checkout, with no uncommitted changes.

### Re-scoring the committed results

Scoring needs no GPU. `--out` writes into a new directory, so the committed files stay untouched:

```bash
subtask-eval prepare --root data/wgo
subtask-eval score --root data/wgo --run results/wgo-bench/baseline-qwen3.6-27b --out runs/rescore
```

## Install

Python 3.10 or newer:

```bash
git clone https://github.com/TieuDaoChanNhan/wgo-bench-open-judge.git
cd wgo-bench-open-judge
pip install -e ".[test]"
pytest                                   # no GPU, network or video needed
subtask-eval prepare --root data/wgo     # downloads WGO-Bench at a pinned revision
```

Models are served separately with [vLLM](https://docs.vllm.ai), through an OpenAI-compatible API.

## Running an evaluation

Each stage is a subcommand that reads and writes files in a run directory:

```bash
# 1. annotate
subtask-eval annotate contact_sheet --config configs/cs-qwen36-27b.json \
    --root data/wgo --out runs/my-run --base-url http://127.0.0.1:8000/v1

# 2. segment F1, and the matched pairs to judge
subtask-eval score --root data/wgo --run runs/my-run --write-pairs

# 3. judge the labels of the matched pairs
subtask-eval judge --run runs/my-run --model Qwen/Qwen3-32B-AWQ \
    --base-url http://127.0.0.1:8001/v1

# 4. all metrics (add --subset homer for one subset)
subtask-eval score --root data/wgo --run runs/my-run

# 5. comparison table over runs
subtask-eval report runs/*
```

| Stage | What it does |
|---|---|
| `prepare` | Downloads WGO-Bench and writes episodes with their human segments |
| `annotate <annotator>` | Runs the pipeline under test. Annotators see the video and the task, never the human segments |
| `score` | Matches segments and computes every metric, per subset and overall |
| `judge` | Grades the labels of matched segments, with cached verdicts |
| `agree` | Compares two sets of verdicts (judge vs judge, judge vs reference, judge vs human review) |
| `report` | Builds one Markdown table over several runs |

**Annotators:**
- `contact_sheet`: any vision-language model over timestamped contact sheets.
- `import`: segmentations produced elsewhere.

**New candidate:** a new model or prompt is a new file in `configs/`. Nothing is tuned on WGO-Bench.

### On a Slurm cluster

[Slurm](https://slurm.schedmd.com) is the job scheduler used on most GPU clusters. `slurm/` has
three example jobs. Each one serves the models it needs with vLLM, runs the stages above and records
provenance.

| Job | GPUs | What it runs |
|---|---|---|
| `check.sbatch` | none | **Unit tests**, the small automated tests in `tests/` that check each function on toy inputs. Then the **scorer check**: scoring EgoANT's released predictions and comparing with its published counts (see [Reproducibility](#reproducibility)) |
| `run.sbatch` | annotator + 1 | One config end to end: annotate, score, judge, score again |
| `judges.sbatch` | one per judge | Judge comparison against Gemini and against each other |

```bash
export W=/path/to/workdir
cp slurm/env.example.sh $W/env.sh        # then edit the paths
CONFIG=configs/cs-qwen36-27b.json sbatch -p <partition> -A <account> \
    -o $W/logs/%x-%j.log slurm/run.sbatch
```

## Layout

```text
subtask_eval/
  data/wgo.py               WGO-Bench download and loading
  annotators/               pipelines under test, common output format and run loop
  scoring/                  matching and F1, label judge, agreement, run scoring
  report.py                 comparison table over runs
  provenance.py             what each job records about itself
configs/                    one file per evaluated pipeline
slurm/                      example cluster jobs
results/                    committed results (CC BY-NC-SA 4.0)
tests/
```

## Sources and licenses

- **WGO-Bench:** Macrodata,
  [*Segmenting Robot Video into Actionable Subtasks*](https://macrodata.co/blog/annotating-robot-video-subtasks).
  The data is CC BY-NC-SA 4.0. Videos are downloaded, never redistributed.
- **EgoANT:** the reference scorer, data preparation and the contact-sheet annotator are adapted
  from [EgoANT](https://github.com/ShuaiZhou302/EgoANT) (MIT). See [`THIRD_PARTY.md`](THIRD_PARTY.md).
- **Code:** MIT, see [`LICENSE`](LICENSE).
- **`results/`:** contains WGO-Bench annotations, so it is CC BY-NC-SA 4.0, see
  [`results/LICENSE.md`](results/LICENSE.md).
