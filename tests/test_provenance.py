import json

from subtask_eval import provenance


def test_record_appends_entries_and_reads_model_revisions(tmp_path, monkeypatch):
    model = tmp_path / "models" / "m"
    model.mkdir(parents=True)
    (model / "HF_REVISION.txt").write_text("org/m@abc\n")
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    monkeypatch.delenv("VLLM", raising=False)
    out = tmp_path / "run"
    provenance.record(out, tmp_path, [model], note="first")
    provenance.record(out, tmp_path, [], note="second")
    hist = json.loads((out / "provenance.json").read_text())
    assert [h["note"] for h in hist] == ["first", "second"]
    assert hist[0]["models"] == {str(model): "org/m@abc"} and hist[0]["env"]["SLURM_JOB_ID"] == "42"
    assert set(hist[0]["code"]) == {"repo", "commit", "branch", "status", "diff"}


def test_archive_copies_whitelist_and_tails_logs(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for name in ("preds.jsonl", "run.json", "score_all.json", "judged_x.jsonl", "videos.mp4", "pairs.jsonl"):
        (run / name).write_text("{}\n")
    (run / "vllm-judge.log").write_text("\n".join(str(i) for i in range(1000)))
    dest = tmp_path / "dest"
    files = provenance.archive(run, dest, [])
    assert "videos.mp4" not in files and {"preds.jsonl", "judged_x.jsonl", "vllm-judge.log.tail"} <= set(files)
    tail = (dest / "vllm-judge.log.tail").read_text().splitlines()
    assert len(tail) == provenance.LOG_TAIL_LINES and tail[-1] == "999"
