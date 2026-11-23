import json

from subtask_eval import report
from subtask_eval.common import write_jsonl
from subtask_eval.data.wgo import load_episodes
from subtask_eval.scoring import score

from .conftest import S


def _preds():
    return {"homer_1": {"episode_id": "homer_1", "segments": [S(0.3, 4, "A"), S(4, 8, "B")], "duration": 8.0,
                        "cost": {"n_calls": 2, "tokens_in": 100, "tokens_out": 10}}}


def test_score_rows_pairs_and_verdicts(wgo_root):
    rows = score.score_rows(load_episodes(wgo_root), _preds(), verdicts={"homer_1|p000|g000": True})
    by = {r["episode_id"]: r for r in rows}
    assert by["galaxea_2"]["has_pred"] is False and by["galaxea_2"]["n_gold"] == 1
    assert by["homer_1"]["n_sem"] == 1 and by["homer_1"]["unjudged"] == 1
    pairs = score.judge_pairs(rows)
    assert {p["id"] for p in pairs} == {"homer_1|p000|g000", "homer_1|p001|g001"}
    assert all(p["task"] == "t" for p in pairs)


def test_granularity(wgo_root):
    g = score.granularity(load_episodes(wgo_root), _preds())
    assert g["homer"] == {"pred_sec": 3.85, "gold_sec": 4.0, "pred_per_episode": 2.0, "gold_per_episode": 2.0}
    assert g["galaxea"]["pred_per_episode"] == 0.0 and g["all"]["gold_per_episode"] == 1.5


def test_cost_report(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"config": {"gpus": 2}, "sessions": [{"wall_sec": 36}]}))
    c = score.cost_report(tmp_path, {"e": {"duration": 3600, "cost": {"tokens_in": 1000, "tokens_out": 50, "n_calls": 4}}})
    assert c["gpu_hours_per_video_hour"] == 0.02 and c["tokens_in_per_video_hour"] == 1000


def test_cli_end_to_end_without_models(wgo_root, tmp_path, monkeypatch, capsys):
    run = tmp_path / "run"
    write_jsonl(run / "preds.jsonl", list(_preds().values()))
    (run / "run.json").write_text(json.dumps({"config": {"name": "r1", "annotator": "test", "gpus": 1},
                                              "sessions": [{"wall_sec": 10}]}))
    from subtask_eval import cli

    assert cli.main(["score", "--root", str(wgo_root), "--run", str(run), "--write-pairs", "--draws", "50"]) == 0
    assert len((run / "pairs.jsonl").read_text().splitlines()) == 2
    write_jsonl(run / "judged.jsonl", [{"id": "homer_1|p000|g000", "match": True, "judge_model": "j"},
                                       {"id": "homer_1|p001|g001", "match": False, "judge_model": "j"}])
    assert cli.main(["score", "--root", str(wgo_root), "--run", str(run), "--draws", "50"]) == 0
    s = json.loads((run / "score_all.json").read_text())
    assert s["micro"]["n_match"] == 2 and s["micro"]["n_sem"] == 1 and s["missing_pred"] == 1
    assert report.main([str(run)]) == 0
    assert "| r1 | test |" in capsys.readouterr().out

    # --out re-scores into another directory and leaves the run directory untouched
    before = (run / "score_all.json").read_text()
    other = tmp_path / "rescore"
    assert cli.main(["score", "--root", str(wgo_root), "--run", str(run), "--draws", "50", "--out", str(other)]) == 0
    assert (run / "score_all.json").read_text() == before
    assert json.loads((other / "score_all.json").read_text())["micro"] == s["micro"]
