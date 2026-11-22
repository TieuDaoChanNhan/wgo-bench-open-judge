# Handoff: segmentation and end-to-end tasks (run on JUPITER)

The labeling task (gold boundaries) is done and reported in the README. This change adds
the two other WGO-Bench tasks. The new code has **not been run yet**, not even the unit tests.

| Task | Input to the model | Metric |
|---|---|---|
| Labeling (existing) | frames of one gold segment | label accuracy |
| **Segmentation (new)** | whole episode as timestamped contact sheets | Segment F1 @ IoU >= 0.75 |
| **End-to-end (new)** | same run as segmentation | E2E F1: a matched segment must also pass the label judge |

New files: `wgo_eval/seg_metrics.py` (metrics), `wgo_eval/segment.py` (segmenter),
`wgo_eval/score_seg.py` (scoring CLI), `wgo_eval/egoant_import.py` (scorer check),
`tests/test_segmentation.py`, `results/segmentation-v1/slurm/seg_full.sbatch`.
`scripts/serve.sh` gains a `vlseg` mode (65k context, up to 64 images per request).

The protocol follows the EgoANT reference scorer (commit `b27ea59f5f`):
- **Outer snap:** before matching, the first predicted start and the last predicted end are set to the gold extent.
- **Matching:** greedy one-to-one by descending IoU, threshold 0.75.
- **Segment F1:** micro, pooled over episodes.
- **E2E F1:** same denominators as segment F1, counting only matched pairs the judge accepts.
- **Judge:** the existing open judge (`Qwen/Qwen3-32B-AWQ`) and rubric, unchanged.

## Steps

Do them in order, and stop at the first failure.

**0. Unit tests (CPU).** Run `python -m pytest -q tests`. All tests must pass. Fix any failure in `tests/test_segmentation.py` or the new modules, and report what you changed.

**1. Scorer check against the reference (CPU, no model).**
1. In the EgoANT checkout (`git checkout b27ea59f5f`), find the released HomER segmentation predictions that its README scores with `scripts/wgo_bench/score_segmentation.py`. Look in `README.md`, `docs/wgo_bench_v5_1_results.md` and `reference_outputs/`.
2. Note the expected micro counts. The README quotes 79 matched / 308 predicted / 470 gold, F1 0.2031.
3. Run:
   ```bash
   python -m wgo_eval.egoant_import --pred-dir <that dir> --root $WGO_ROOT --out runs/egoant-ref/segpreds.jsonl
   python -m wgo_eval.score_seg --root $WGO_ROOT --subset homer --segpreds runs/egoant-ref/segpreds.jsonl
   ```
4. The `micro` line must reproduce the expected counts exactly.
5. If EgoANT also released E2E scores with Gemini verdicts, record its E2E numbers for later comparison. Our judge is stricter, so E2E will not match exactly.
6. If the counts differ, find out why before going further. Likely causes: sorting, snapping, tie order, or episodes without a prediction.

**2. Smoke test (GPU).** Run `LIMIT=2 sbatch results/segmentation-v1/slurm/seg_full.sbatch`. Check:
- every server comes up;
- `segpreds.jsonl` holds 2 rows per segmenter, with non-empty `segments`;
- `sheets_per_call` equals `n_sheets`, meaning the whole episode went in one call;
- `seg_score_all.txt` prints.

If vLLM rejects the request (image count or context length), the segmenter falls back to smaller windows. Report how often that happened.

**3. Full run.** Run `sbatch results/segmentation-v1/slurm/seg_full.sbatch`. This covers 100 episodes with 3 segmenters: Qwen3.6-27B, Qwen3.8-27B-AWQ and Qwen3-VL-8B.

**4. Write `results/segmentation-v1/run.md`**, in the same style as `results/zeroshot-v1/run.md`. Include:
- versions and model revisions;
- wall time;
- failed or fallback episodes;
- per segmenter: Segment F1 and E2E F1, micro and subset average with 95% CI, per subset and HomER only, mean predicted segment length vs gold (gold mean is 5.77 s);
- label accuracy on matched pairs.

Copy `segpreds.jsonl`, `e2e_judged.jsonl` and `seg_score_*.json` into `results/segmentation-v1/<name>/`.

## Reference numbers (different judge or model, for orientation only)

| Source | Setting | Segment F1 | E2E F1 |
|---|---|---|---|
| Macrodata write-up | Gemini 3.5 Flash, 100 episodes, base prompt | 0.290 | – |
| Macrodata write-up | best GEPA prompt | 0.306 | 0.154–0.168 (Gemini judge) |
| Macrodata write-up | per subset, best prompt | Galaxea 0.589, DROID 0.292, HomER 0.227 | – |
| EgoANT | HomER only, reference run | 0.203 (79/308/470) | – |

## Do not change

- Dataset revision, prompts, frame settings or judge.
- Do not tune anything on WGO-Bench; it is evaluation-only.
- A different prompt or setting goes in as a new named entry in `PROMPTS`, reported as a separate run.

## Open questions to report back

- Is there a published per-dataset breakdown we can compare with, beyond the table above?
- Did any segmenter output overlapping or unsorted segments often? Post-processing handles them, but how often it happens is informative.
