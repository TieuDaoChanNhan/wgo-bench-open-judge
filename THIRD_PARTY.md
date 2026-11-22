# Third-party code

Adapted from EgoANT, https://github.com/ShuaiZhou302/EgoANT, commit `b27ea59f5f`:

| This repository | EgoANT source |
|---|---|
| `subtask_eval/data/wgo.py` (download and segment rows) | `scripts/prepare_wgo_bench.py` |
| `subtask_eval/scoring/segments.py`, `scoring/score.py` (outer snap, greedy IoU >= 0.75 matching, micro/macro segment F1, end-to-end F1) | `scripts/wgo_bench/wgo_metrics.py`, `scripts/wgo_bench/score_segmentation.py`, `scripts/wgo_gemini_rescore.py` |
| `subtask_eval/scoring/judge.py` (rubric text, reply parsing) | `scripts/wgo_gemini_rescore.py` |
| `subtask_eval/annotators/contact_sheet.py` (contact sheets, `gepa_duration_prior_v1` and `legacy` prompts, parsing, post-processing) | `scripts/wgo_bench/contact_sheet_segment.py` |
| `subtask_eval/annotators/imported.py` (prediction lookup) | `scripts/wgo_bench/score_segmentation.py` |

The judge rubric itself comes from the Macrodata write-up
https://macrodata.co/blog/annotating-robot-video-subtasks.

EgoANT license:

```
MIT License

Copyright (c) 2026 EgoANT contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
