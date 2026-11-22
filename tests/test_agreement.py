import json

from subtask_eval.common import write_jsonl
from subtask_eval.scoring import agreement as ag


def test_agreement_and_kappa():
    r = ag.agreement([True, True, False, False], [True, False, False, False])
    assert r["agreement"] == 0.75 and r["kappa"] == 0.5
    assert r["table"] == {"a_true_b_true": 1, "a_true_b_false": 1, "a_false_b_true": 0, "a_false_b_false": 2}
    assert ag.agreement([True, False], [True, False])["kappa"] == 1.0


def test_extract_egoant_fills_task_and_skips_unjudged():
    egoant = {"per_episode": [{"id": "homer_1", "pairs": [
        {"pred_i": 4, "gold_j": 6, "gold_label": "g", "pred_label": "p", "semantic_match": True, "judge_model": "gemini"},
        {"pred_i": 5, "gold_j": 7, "gold_label": "g2", "pred_label": "p2", "semantic_match": None}]}]}
    rows = ag.extract_egoant(egoant, {"homer_1": "pour water"})
    assert len(rows) == 1
    assert rows[0]["id"] == "homer_1|p004|g006" and rows[0]["task"] == "pour water" and rows[0]["reference_match"]


def test_verdicts_reads_any_verdict_key_and_compare_cli(tmp_path, capsys):
    write_jsonl(tmp_path / "a.jsonl", [{"id": "1", "reference_match": True, "gold_label": "g", "pred_label": "p"},
                                       {"id": "2", "reference_match": False}, {"id": "3", "reference_match": True}])
    write_jsonl(tmp_path / "b.jsonl", [{"id": "1", "match": True}, {"id": "2", "match": True}])
    assert ag.verdicts(tmp_path / "a.jsonl") == {"1": True, "2": False, "3": True}
    assert ag.main(["compare", "--a", str(tmp_path / "a.jsonl"), "--b", str(tmp_path / "b.jsonl"),
                    "--disagreements", str(tmp_path / "d.jsonl")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n"] == 2 and out["agreement"] == 0.5 and out["only_in_a"] == 1
    assert [json.loads(line)["id"] for line in (tmp_path / "d.jsonl").read_text().splitlines()] == ["2"]
