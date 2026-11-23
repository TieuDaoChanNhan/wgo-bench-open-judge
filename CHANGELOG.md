# Changelog

## 0.2.0 (2026-11-23)

The repository was rewritten from a label-only harness into a subtask annotation evaluator.

- Episodes are scored end to end from one run:
  - segment F1 (WGO-Bench protocol);
  - label accuracy on matched segments;
  - end-to-end F1;
  - granularity, gold gaps and cost.
- Stages are separate subcommands of one CLI (`subtask-eval`): `prepare`, `annotate`, `score`,
  `judge`, `agree`, `report` and `provenance`.
- Annotators:
  - `contact_sheet`: any VLM over timestamped contact sheets;
  - `import`: existing segmentations.
- Judge comparison with `agree`. The default judge is `Qwen/Qwen3-32B-AWQ` (agreement 0.90 with
  Gemini on 79 reference pairs).
- Every job records provenance. Example Slurm jobs are included.
- Committed results in `results/`: scorer check, judge validation and the Qwen3.6-27B baseline.

## 0.1.0

The first release evaluated subtask labeling only, with gold boundaries given and an open judge.
