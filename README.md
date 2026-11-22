# subtask-eval

Evaluation harness for **subtask annotation pipelines** in FineRobotics. A pipeline takes an
episode video and its task and returns labeled segments `{start_sec, end_sec, label}`; this
repository measures how well those segments and labels match human annotations, and at what cost.

The first eval set is [WGO-Bench](https://huggingface.co/datasets/macrodata/WGO-Bench)
(100 episodes, 743 human segments: HomER egocentric, DROID and Galaxea robot).

## Pipeline

```
eval set ──► annotator ──► preds.jsonl ──► score (segments) ──► pairs.jsonl ──► judge ──► score (end-to-end) ──► report
(episodes,   (system        (one row per                         (matched                                         (table over
 gold)        under test)    episode)                             segments)                                         runs)
```

| Stage | Module | Command |
|---|---|---|
| Eval set | `subtask_eval/data/wgo.py` | `subtask-eval prepare --root data/wgo` |
| Annotators | `subtask_eval/annotators/` | `subtask-eval annotate <name> --config configs/<run>.json --root data/wgo --out runs/<run>` |
| Segment scoring | `subtask_eval/scoring/score.py` | `subtask-eval score --root data/wgo --run runs/<run> --write-pairs` |
| Label judge | `subtask_eval/scoring/judge.py` | `subtask-eval judge --run runs/<run> --model Qwen/Qwen3-32B-AWQ` |
| End-to-end scoring | same | `subtask-eval score --root data/wgo --run runs/<run> [--subset homer]` |
| Report | `subtask_eval/report.py` | `subtask-eval report runs/* --out report.md` |

Annotators see the video and the task only; gold segments are removed in `data.wgo.model_view`.

| Annotator | What it measures |
|---|---|
| `contact_sheet` | any VLM over timestamped contact sheets, Macrodata-style (`configs/cs-*.json`) |
| `import` | segmentations produced elsewhere (EgoANT outputs, other pipelines, APIs) |

## Metrics

Scoring follows the WGO-Bench reference scorer as implemented in EgoANT (`scoring/segments.py`):

- **Segment F1.** The first predicted start and last predicted end are snapped to the gold extent. Predicted and gold segments are then matched greedily one-to-one by IoU, threshold 0.75. Precision and recall are pooled over episodes (micro).
- **Label accuracy on matches.** The share of matched pairs whose labels the judge accepts.
- **End-to-end F1.** Same as segment F1, but a matched pair must also pass the label judge.
- **Granularity.** Mean segment length and segments per episode, predicted vs gold.
- **Gold gaps.** Pauses between gold segments, and the best segment F1 a gap-free output can reach. lerobot-annotate always outputs a gap-free cover.
- **Cost.** GPU-hours, tokens and calls per hour of video.

All F1 values come with 95% episode-bootstrap intervals and are reported per subset.

The judge is an open-weight model, `Qwen/Qwen3-32B-AWQ`, using the Macrodata rubric, and it reads text only. It is stricter than the `gemini-3.5-flash` judge used in published numbers. Label and end-to-end scores therefore compare runs of this harness with each other, not with published tables.

## Install and test

```bash
pip install -e ".[test]"   # Python >= 3.10
pytest                     # no GPU, network or video needed
```

Models are served separately with vLLM (OpenAI-compatible API). On JUPITER, `slurm/` has the
jobs: `check.sbatch` (tests and scorer check against EgoANT) and `run.sbatch` (one config end to
end). Copy `slurm/env.example.sh` to `$W/env.sh` first. See `JUPITER_HANDOFF.md`.

## Adding a run

Add a config in `configs/`:
- `name` and `annotator`;
- `model`: the served name;
- annotator options;
- `gpus` and `serve`, which tell `run.sbatch` how to start vLLM.

Then run `CONFIG=configs/<file>.json sbatch slurm/run.sbatch`. A different prompt or setting is a
new config, reported as its own run. Nothing is tuned on WGO-Bench.

## Sources and licenses

- WGO-Bench data: CC BY-NC-SA 4.0. Videos are not redistributed; `prepare` downloads them.
- The protocol and judge rubric come from Macrodata, [*Segmenting Robot Video into Actionable Subtasks*](https://macrodata.co/blog/annotating-robot-video-subtasks).
- Data preparation, metrics and the contact-sheet annotator are adapted from [EgoANT](https://github.com/ShuaiZhou302/EgoANT) (MIT); see [`THIRD_PARTY.md`](THIRD_PARTY.md).
- This repository is MIT ([`LICENSE`](LICENSE)).
