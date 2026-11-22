import pytest

from subtask_eval.scoring import segments as m

from .conftest import S


def G(a, b, label="x"):
    return {"start_sec": a, "end_sec": b, "label": label}


def test_iou():
    assert m.iou(S(0, 10), S(0, 10)) == 1.0
    assert m.iou(S(0, 10), S(5, 15)) == pytest.approx(5 / 15)
    assert m.iou(S(0, 1), S(2, 3)) == 0.0
    assert m.iou(S(1, 1), S(1, 1)) == 0.0


def test_outer_snap_only_moves_outer_edges_and_sorts():
    snapped = m.snap_outer_boundaries([S(4.2, 12), S(1.3, 4.2)], [G(0.5, 4), G(4, 9.5)])
    assert [(s["start_sec"], s["end_sec"]) for s in snapped] == [(0.5, 4.2), (4.2, 9.5)]
    assert m.snap_outer_boundaries([], [G(0, 1)]) == []


def test_greedy_match_one_to_one_threshold_and_best_first():
    gold = [G(0, 10), G(10, 20)]
    pairs = m.greedy_match([S(0, 9.5), S(0.5, 10), S(10, 14)], gold)
    assert len(pairs) == 1 and pairs[0][1] == 0  # S(10,14) has IoU 0.4 with gold 1
    assert m.greedy_match([S(0, 8), S(0, 9.9)], [G(0, 10)]) == [(1, 0, pytest.approx(0.99))]


def test_score_episode_counts_labels_and_snap():
    gold = [G(0, 5, "pick cup"), G(5, 10, "place cup")]
    pred = [S(0.8, 5, "pick up the cup"), S(5, 9.0, "put cup down")]
    r = m.score_episode(pred, gold)
    assert r["n_match"] == 2 and r["f1"] == 1.0
    assert (r["pairs"][0]["gold_label"], r["pairs"][0]["pred_label"]) == ("pick cup", "pick up the cup")
    assert m.score_episode([S(2, 5), S(5, 7)], gold, snap=False)["n_match"] == 0


def test_empty_prediction_scores_zero_but_counts_gold():
    r = m.score_episode([], [G(0, 1), G(1, 2)])
    assert (r["n_pred"], r["n_gold"], r["n_match"], r["f1"]) == (0, 2, 0, 0.0)


def test_aggregate_micro_macro_e2e_and_subset_avg():
    rows = [{"subset": "homer", "n_pred": 4, "n_gold": 2, "n_match": 2, "n_sem": 1},
            {"subset": "droid", "n_pred": 2, "n_gold": 4, "n_match": 0, "n_sem": 0}]
    agg = m.aggregate(rows, sem_key="n_sem")
    micro = agg["micro"]
    assert micro["f1"] == pytest.approx(1 / 3)
    assert micro["macro_f1"] == pytest.approx((2 / 3 + 0) / 2)
    assert micro["e2e_f1"] == pytest.approx(1 / 6) and micro["label_acc_on_matches"] == 0.5
    assert agg["subset_avg"]["f1"] == pytest.approx((2 / 3 + 0) / 2)


def test_bootstrap_contains_point_estimate():
    rows = [{"subset": "homer", "n_pred": p, "n_gold": g, "n_match": k}
            for p, g, k in [(5, 5, 3), (4, 6, 1), (6, 5, 4), (3, 3, 0)]]
    lo, hi = m.bootstrap(rows, draws=300)["f1_micro_ci95"]
    assert lo <= m.aggregate(rows)["micro"]["f1"] <= hi


def test_full_cover_and_gap_stats():
    gold = [G(0, 3), G(4, 6), G(6, 9)]
    assert [(s["start_sec"], s["end_sec"]) for s in m.full_cover(gold)] == [(0, 4), (4, 6), (6, 9)]
    st = m.gap_stats(gold)
    assert st == {"n_gaps": 1, "gap_sec": 1, "extent_sec": 9}
    # Stitching closes a gap by stretching the segment before it: a 1 s gap after a 3 s segment
    # gives IoU 3/4 = 0.75 (still a match), a 2 s gap gives 3/5 (no match).
    assert m.score_episode(m.full_cover(gold), gold)["n_match"] == 3
    assert m.score_episode(m.full_cover([G(0, 3), G(5, 6)]), [G(0, 3), G(5, 6)])["n_match"] == 1
