"""Unit tests for segmentation scoring and the contact-sheet segmenter. No GPU, network or video needed."""

from __future__ import annotations

import pytest

from wgo_eval import score_seg, seg_metrics as m, segment


def S(a, b, label="x"):
    return {"start_sec": a, "end_sec": b, "subtask": label}


def G(a, b, label="x"):
    return {"start_sec": a, "end_sec": b, "label": label}


def test_iou():
    assert m.iou(S(0, 10), S(0, 10)) == 1.0
    assert m.iou(S(0, 10), S(5, 15)) == pytest.approx(5 / 15)
    assert m.iou(S(0, 1), S(2, 3)) == 0.0
    assert m.iou(S(1, 1), S(1, 1)) == 0.0


def test_outer_snap_only_moves_outer_edges_and_sorts():
    gold = [G(0.5, 4), G(4, 9.5)]
    snapped = m.snap_outer_boundaries([S(4.2, 12), S(1.3, 4.2)], gold)
    assert [(s["start_sec"], s["end_sec"]) for s in snapped] == [(0.5, 4.2), (4.2, 9.5)]
    assert m.snap_outer_boundaries([], gold) == []


def test_greedy_match_is_one_to_one_and_thresholded():
    gold = [G(0, 10), G(10, 20)]
    # Both predictions overlap the first gold segment; only the better one may take it.
    pairs = m.greedy_match([S(0, 9.5), S(0.5, 10), S(10, 14)], gold)
    assert [(i, j) for i, j, _ in pairs] == [(1, 0)] or [(i, j) for i, j, _ in pairs] == [(0, 0)]
    assert len(pairs) == 1  # S(10,14) has IoU 0.4 with gold 1: below 0.75
    assert m.greedy_match([S(0, 10)], [G(0, 10)], threshold=0.75)[0][2] == 1.0


def test_greedy_match_prefers_highest_iou():
    gold = [G(0, 10)]
    pairs = m.greedy_match([S(0, 8), S(0, 9.9)], gold)
    assert pairs == [(1, 0, pytest.approx(0.99))]


def test_score_episode_counts_and_snap_effect():
    gold = [G(0, 5, "pick cup"), G(5, 10, "place cup")]
    pred = [S(0.8, 5, "pick up the cup"), S(5, 9.0, "put cup down")]
    snapped = m.score_episode(pred, gold)
    assert snapped["n_match"] == 2 and snapped["f1"] == 1.0
    assert snapped["pairs"][0]["gold_label"] == "pick cup" and snapped["pairs"][0]["pred_label"] == "pick up the cup"
    raw = m.score_episode(pred, gold, snap=False)
    assert raw["n_match"] == 2  # 4.2/5 = 0.84 and 4/5 = 0.8 still pass
    assert m.score_episode([S(2, 5), S(5, 7)], gold, snap=False)["n_match"] == 0


def test_empty_prediction_scores_zero_but_counts_gold():
    r = m.score_episode([], [G(0, 1), G(1, 2)])
    assert (r["n_pred"], r["n_gold"], r["n_match"], r["f1"]) == (0, 2, 0, 0.0)


def test_aggregate_micro_macro_and_e2e():
    rows = [
        {"subset": "homer", "n_pred": 4, "n_gold": 2, "n_match": 2, "n_sem": 1},
        {"subset": "droid", "n_pred": 2, "n_gold": 4, "n_match": 0, "n_sem": 0},
    ]
    agg = m.aggregate(rows, sem_key="n_sem")
    micro = agg["micro"]
    assert (micro["precision"], micro["recall"]) == (2 / 6, 2 / 6)
    assert micro["f1"] == pytest.approx(1 / 3)
    assert micro["macro_f1"] == pytest.approx((2 * 0.5 * 1 / 1.5 + 0) / 2)
    assert micro["e2e_precision"] == pytest.approx(1 / 6) and micro["e2e_recall"] == pytest.approx(1 / 6)
    assert micro["label_acc_on_matches"] == 0.5
    assert agg["subset_avg"]["f1"] == pytest.approx((2 / 3 + 0) / 2)


def test_bootstrap_interval_contains_point_estimate():
    rows = [{"subset": "homer", "n_pred": p, "n_gold": g, "n_match": k}
            for p, g, k in [(5, 5, 3), (4, 6, 1), (6, 5, 4), (3, 3, 0)]]
    ci = m.bootstrap(rows, draws=300)
    lo, hi = ci["f1_micro_ci95"]
    assert lo <= m.aggregate(rows)["micro"]["f1"] <= hi


def test_gold_by_episode_and_score_all_pairs():
    segs = [
        {"segment_id": "homer_1/001", "episode_id": "homer_1", "subset": "homer", "video": "v", "start_sec": 4.0, "end_sec": 8.0, "task": "t", "label": "b"},
        {"segment_id": "homer_1/000", "episode_id": "homer_1", "subset": "homer", "video": "v", "start_sec": 0.0, "end_sec": 4.0, "task": "t", "label": "a"},
        {"segment_id": "droid_2/000", "episode_id": "droid_2", "subset": "droid", "video": "w", "start_sec": 0.0, "end_sec": 3.0, "task": "u", "label": "c"},
    ]
    gold = m.gold_by_episode(segs)
    assert [g["label"] for g in gold["homer_1"]["gold"]] == ["a", "b"]
    preds = {"homer_1": {"segments": [S(0.3, 4, "A"), S(4, 8, "B")]}}
    rows = score_seg.score_all(gold, preds, verdicts={"homer_1|p000|g000": True})
    by_ep = {r["episode_id"]: r for r in rows}
    assert by_ep["droid_2"]["has_pred"] is False and by_ep["droid_2"]["n_gold"] == 1
    assert by_ep["homer_1"]["n_sem"] == 1 and by_ep["homer_1"]["unjudged"] == 1
    lengths = score_seg.mean_lengths(gold, preds)
    assert lengths["homer"] == {"pred_sec": 3.85, "gold_sec": 4.0, "pred_per_episode": 2.0, "gold_per_episode": 2.0}
    assert lengths["droid"]["pred_per_episode"] == 0.0 and lengths["all"]["gold_per_episode"] == 1.5
    pairs = score_seg.judge_pairs(rows)
    assert {p["id"] for p in pairs} == {"homer_1|p000|g000", "homer_1|p001|g001"}
    assert all(p["task"] == "t" for p in pairs)


# ---------- segmenter helpers ----------

def test_sample_times():
    assert segment.sample_times(2.0, 0.5) == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert segment.sample_times(1.2, 0.5) == [0.0, 0.5, 1.0, 1.2]


def test_parse_segments_variants():
    assert segment.parse_segments('```json\n{"segments":[{"start_sec":1,"end_sec":2,"subtask":"a"}]}\n```') == [S(1.0, 2.0, "a")]
    text = 'ok {"segments":[{"start":3000,"end":20000,"label":"b"},{"start_sec":0,"end_sec":1,"subtask":"a"},{"start_sec":5,"end_sec":4}]}'
    assert segment.parse_segments(text) == [S(0.0, 1.0, "a"), S(3.0, 20.0, "b")]
    salvaged = segment.parse_segments('{"segments":[{"start_sec":0,"end_sec":1,"subtask":"a"}, {"start_sec":1,')
    assert salvaged == [S(0.0, 1.0, "a")]
    assert segment.parse_segments("no json here") == []


def test_postprocess_clamps_merges_and_folds_short():
    segs = [S(-1, 2, "a"), S(2.02, 3, "a"), S(2.5, 6, "b"), S(6, 6.1, "c"), S(6.1, 20, "d")]
    out = segment.postprocess(segs, duration=10.0, min_seg_sec=0.25)
    assert out == [S(0.0, 3.0, "a"), S(3.0, 6.1, "b"), S(6.1, 10.0, "d")]


def test_window_sizes():
    assert segment.window_sizes(9, 0) == [9, 6, 4, 2, 1]
    assert segment.window_sizes(9, 3) == [3, 2, 1]
    assert segment.window_sizes(1, 0) == [1]


def test_segment_messages_never_contain_gold():
    msgs = segment.build_messages("gepa_duration_prior_v1", "pour water", [b"\xff\xd8"], 0.0, 9.5, 0.5)
    text = msgs[1]["content"][0]["text"]
    assert "pour water" in text and "[0.00, 9.50]" in text
    assert msgs[1]["content"][1]["type"] == "image_url"
