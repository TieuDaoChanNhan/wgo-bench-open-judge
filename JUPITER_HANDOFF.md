# Handoff: run on JUPITER

The repository was restructured into the `subtask_eval` package (see README). **None of the code
has been run yet**, not even the unit tests. Steps in order; stop and report at the first failure.

WGO-Bench is already prepared on JUPITER (`data/wgo` with `segments.jsonl`, `manifest.json`, `videos/`).
Its `segments.jsonl` format is unchanged, so it does not need to be prepared again. Check that
`manifest.json` reports revision `1c86a62e5302`, 100 episodes and 743 segments.

## 0. Environment

```bash
cd $W/repo && git pull
$W/.venv/bin/pip install -e ".[test]"      # replaces the old requirements.txt install
cp slurm/env.example.sh $W/env.sh          # then edit paths: WGO_ROOT must point at the prepared data
```

The old `wgo_eval` package is gone. Remove any stale `wgo_eval*.egg-info` in the venv if imports clash.

## 1. Tests and scorer check (CPU)

1. In `$EGOANT_DIR` (EgoANT at `b27ea59f5f`), find the HomER segmentation predictions its README
   scores with `scripts/wgo_bench/score_segmentation.py`. Expected micro counts: 79 matched /
   308 predicted / 470 gold, F1 0.2031. See `README.md`, `docs/wgo_bench_v5_1_results.md` and
   `reference_outputs/`.
2. Run `EGOANT_PRED=<that dir, relative to $EGOANT_DIR> sbatch slurm/check.sbatch`.
3. It must end with `CHECK|ok`.
4. On a mismatch, find the cause before going on: sorting, snapping, tie order, missing
   episodes, or zero-length segments. Report it.

## 2. Smoke test (GPU)

1. Run `CONFIG=configs/cs-qwen36-27b.json LIMIT=2 sbatch slurm/run.sbatch`.
2. Check the run directory `runs/cs-qwen36-27b-smoke2/`:
   - `preds.jsonl` has 2 rows with non-empty `segments`;
   - `sheets_per_call == n_sheets`, meaning the whole episode went in one call;
   - `run.json` exists;
   - `pairs.jsonl`, `judged.jsonl` and `score_*.json` are written;
   - the summary prints granularity, gold gaps and cost.
3. If vLLM rejects a request (image count or context length), report it and the fallback window
   size that was used.

## 3. Full run

1. Run `CONFIG=configs/cs-qwen36-27b.json sbatch slurm/run.sbatch`.
2. Then run `$W/.venv/bin/python -m subtask_eval report $W/runs/cs-qwen36-27b --out $W/runs/report.md`.

## 4. Report back

Send back:
- `report.md` and the `score_*.txt` files;
- `run.json`;
- vLLM and model revisions (`HF_REVISION.txt`);
- wall time;
- number of failed episodes;
- anything you had to change.

Do not commit run outputs into the repository unless asked.

## Rules

- Do not change the dataset revision, the prompts, the frame settings or the judge.
- Nothing is tuned on WGO-Bench.
- A new setting goes in a new config file.
