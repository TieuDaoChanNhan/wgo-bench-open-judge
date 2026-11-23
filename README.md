# wgo-bench-open-judge

An open evaluation harness for **robot-video subtask annotation**: splitting a manipulation video
into its steps and naming each step.

```
episode video + task ("put the dishes away")
        │  annotation pipeline under test
        ▼
0.0–4.2 s  "pick up the cup"   │  4.2–9.8 s  "place the cup in the sink"   │  ...
        │  compared with human annotations
        ▼
segment F1 · label accuracy · end-to-end F1 · granularity · cost
```

The benchmark is [WGO-Bench](https://huggingface.co/datasets/macrodata/WGO-Bench) by Macrodata:
100 episodes with 743 human-annotated segments, from three sources:

- HomER: egocentric human video, 25 episodes;
- DROID: robot arm, 50 episodes;
- Galaxea: robot head camera, 25 episodes.

Everything runs on open-weight models served locally, **including the judge** that grades the labels.

## Results

The baseline is `Qwen/Qwen3.6-27B` reading each episode as timestamped contact sheets, following
the method of the WGO-Bench write-up. The judge is `Qwen/Qwen3-32B-AWQ`. 95% confidence intervals
are in brackets.

| Subset | Segment F1 | Label accuracy on matched segments | End-to-end F1 |
|---|---|---|---|
| DROID | 0.376 | 0.47 | 0.177 |
| Galaxea | 0.179 | 0.44 | 0.080 |
| HomER | 0.119 | 0.56 | 0.067 |
| **All 100 episodes** | **0.193** [0.153, 0.240] | **0.50** | **0.096** [0.069, 0.130] |

- **The baseline segments too coarsely:** 4.0 segments per episode at 10.4 s each, against
  7.4 human segments at 5.8 s. On HomER it predicts 187 segments for 470 gold ones.
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

**Granularity** is the mean segment length and the number of segments per episode, predicted vs
human. A pipeline that splits much coarser or finer than the annotators loses F1 even when its
segments are sensible, so read F1 together with this.

**Cost** is GPU-hours, tokens and model calls per hour of video.

All F1 values come with 95% intervals from an **episode bootstrap**: the episodes are resampled
2,000 times and the score recomputed each time. Episodes, not segments, are resampled, because
segments of one episode are not independent.

## Label judge

Published WGO-Bench numbers use `gemini-3.5-flash` as judge. To stay fully open we use a local
model, chosen by how closely it reproduces Gemini's verdicts. EgoANT released 79 label pairs judged
by Gemini; we judged the same 79 pairs with two candidates.

| | `Qwen/Qwen3-32B-AWQ` (used) | `Qwen/Qwen3.8-27B` |
|---|---|---|
| Agreement with Gemini | **0.90** (71/79) | 0.70 (55/79) |
| Cohen's kappa | **0.76** | 0.42 |
| Accept rate (Gemini: 0.76) | 0.66 | 0.46 |
| Gemini accepts, judge rejects | 8 | 24 |
| Gemini rejects, judge accepts | 0 | 0 |

How to read the table:

- **Agreement** is the share of the 79 pairs on which the judge gives Gemini's verdict.
- **Cohen's kappa** is agreement corrected for chance. Two judges that both say "match" most of the
  time agree often by luck alone; kappa removes that part.
  - 1 means perfect agreement, and 0 means no better than chance.
  - As a rule of thumb, 0.4–0.6 is moderate and above 0.6 is substantial.
- **Accept rate** is the share of pairs the judge calls a match. It is lower than Gemini's for both
  judges, so both are stricter.
- **The last two rows** are the two ways to disagree. Neither judge ever accepts a label Gemini
  rejects. The newer `Qwen3.8-27B` also rejects many that Gemini accepts, for example "open green
  bottle" against "unscrew green cap".

We therefore use `Qwen3-32B-AWQ`.

- **Effect on scores:** on the baseline, switching the judge to `Qwen3.8-27B` lowers label accuracy
  from 0.50 to 0.42 and end-to-end F1 from 0.096 to 0.081. The two judges agree on 83% of the
  baseline's 110 matched pairs (kappa 0.65). Segment F1 does not depend on the judge.
- **Comparability:** because both open judges are stricter than Gemini, label and end-to-end
  scores here compare runs of this harness with each other, not with published tables.

## Reproducibility

- **Scorer.** It reproduces the reference scorer exactly. On EgoANT's released HomER predictions it
  gives the published counts: 308 predicted, 470 gold, 79 matched, F1 0.203.
- **Judges.** They are deterministic. Judging the same pairs twice gave identical verdicts (79/79
  for both judges).
- **Annotator.** The baseline was run twice from scratch with the same code and models.
  - 83 of 100 episodes came out identical.
  - Segment F1 was 0.191 and 0.193; label accuracy 0.50 both times; end-to-end F1 0.096 both times.
  - Decoding is greedy, so the remaining differences come from batching inside vLLM.
- **Provenance.** Every job writes `provenance.json`: the git commit and any local diff, installed
  packages, vLLM version, GPUs, model revisions, dataset manifest and config. The committed results
  were produced from a clean checkout.
- **Re-scoring.** Scoring the committed predictions and verdicts needs no GPU and gives the
  committed scores. `--out` writes into another directory, so the committed files stay untouched:
  ```bash
  subtask-eval score --root data/wgo --run results/wgo-bench/baseline-qwen3.6-27b --out runs/rescore
  ```

## Install

```bash
git clone https://github.com/TieuDaoChanNhan/wgo-bench-open-judge.git && cd wgo-bench-open-judge
pip install -e ".[test]"                 # Python >= 3.10
pytest                                   # no GPU, network or video needed
subtask-eval prepare --root data/wgo     # downloads WGO-Bench at a pinned revision
```

Models are served separately with [vLLM](https://docs.vllm.ai), through an OpenAI-compatible API.

## Running an evaluation

Each stage is a subcommand and reads and writes files in a run directory:

```bash
subtask-eval annotate contact_sheet --config configs/cs-qwen36-27b.json \
    --root data/wgo --out runs/my-run --base-url http://127.0.0.1:8000/v1   # 1. annotate
subtask-eval score --root data/wgo --run runs/my-run --write-pairs          # 2. segment F1, matched pairs
subtask-eval judge --run runs/my-run --model Qwen/Qwen3-32B-AWQ \
    --base-url http://127.0.0.1:8001/v1                                     # 3. judge the labels
subtask-eval score --root data/wgo --run runs/my-run [--subset homer]       # 4. all metrics
subtask-eval report runs/*                                                  # 5. comparison table
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

`slurm/` has three example jobs. Each one serves the models it needs with vLLM, runs the stages
above, and records provenance:

| Job | GPUs | What it runs |
|---|---|---|
| `check.sbatch` | none | Unit tests and the scorer check |
| `run.sbatch` | annotator + 1 | One config end to end |
| `judges.sbatch` | one per judge | Judge comparison |

```bash
export W=/path/to/workdir && cp slurm/env.example.sh $W/env.sh   # then edit the paths
CONFIG=configs/cs-qwen36-27b.json sbatch -p <partition> -A <account> -o $W/logs/%x-%j.log slurm/run.sbatch
```

## Layout

```
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
