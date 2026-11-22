"""Record how a job ran, and archive a run for review.

    # at the start of every job: code, environment, hardware, models, data
    subtask-eval provenance record --out runs/x --models $MODELS_DIR/Qwen3.6-27B $JUDGE_DIR

    # copy the reviewable files of a run into a results checkout (no videos)
    subtask-eval provenance archive --run runs/x --dest ../results/runs/x --logs subtask-eval-run-123.log

`record` appends one entry to <out>/provenance.json per call, so a run that is resumed or
re-judged keeps every session. It captures the git commit and any uncommitted diff, so a
local patch can never hide behind a commit hash.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ENV_KEYS = ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_NODELIST", "SLURM_JOB_PARTITION", "CUDA_VISIBLE_DEVICES",
            "WGO_ROOT", "MODELS_DIR", "RUNS_DIR", "JUDGE_DIR", "JUDGE_NAME", "CONFIG", "LIMIT", "JUDGES", "RUNS",
            "EGOANT_JSON", "EGOANT_PRED", "TASK_REPO", "JUDGE_CACHE", "HF_HUB_OFFLINE")
# Files a reviewer needs; everything else in a run directory is left out.
ARCHIVE_GLOBS = ("preds.jsonl", "run.json", "provenance.json", "pairs.jsonl", "judged*.jsonl", "score_*.json",
                 "score_*.txt", "agree_*.json", "disagree_*.jsonl", "check.txt", "report.md")
LOG_TAIL_LINES = 400


def _cmd(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=120)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def code_state(repo: Path) -> dict[str, Any]:
    if shutil.which("git") is None:
        # Seen on JUPITER compute nodes: git is only on the login nodes unless a module is loaded.
        print("PROVENANCE|WARNING git not found on PATH: commit and diff cannot be recorded", file=sys.stderr)
    return {
        "repo": str(repo),
        "commit": _cmd(["git", "rev-parse", "HEAD"], repo),
        "branch": _cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo),
        "status": _cmd(["git", "status", "--porcelain"], repo),
        "diff": _cmd(["git", "diff", "HEAD"], repo),  # uncommitted local changes, if any
    }


def model_revisions(dirs: list[Path]) -> dict[str, str | None]:
    out = {}
    for d in dirs:
        rev = d / "HF_REVISION.txt"
        out[str(d)] = rev.read_text(encoding="utf-8").strip() if rev.exists() else None
    return out


def record(out_dir: Path, repo: Path, models: list[Path], note: str = "") -> dict[str, Any]:
    entry: dict[str, Any] = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": note,
        "host": socket.gethostname(),
        "argv": sys.argv,
        "env": {k: os.environ[k] for k in ENV_KEYS if k in os.environ},
        "code": code_state(repo),
        "python": sys.version,
        "platform": platform.platform(),
        "pip_freeze": (_cmd([sys.executable, "-m", "pip", "freeze"]) or "").splitlines(),
        "vllm": _cmd([os.environ["VLLM"], "--version"]) if os.environ.get("VLLM") else None,
        "gpus": _cmd(["nvidia-smi", "-L"]),
        "models": model_revisions(models),
    }
    wgo = os.environ.get("WGO_ROOT")
    if wgo and (Path(wgo) / "manifest.json").exists():
        entry["dataset"] = json.loads((Path(wgo) / "manifest.json").read_text(encoding="utf-8"))
    cfg = os.environ.get("CONFIG")
    if cfg and Path(cfg).exists():
        entry["config_file"] = json.loads(Path(cfg).read_text(encoding="utf-8"))

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "provenance.json"
    history = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    history.append(entry)
    path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return entry


def archive(run_dir: Path, dest: Path, logs: list[Path]) -> list[str]:
    """Copy the reviewable files of run_dir to dest; logs are cut to their last lines."""
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for pattern in ARCHIVE_GLOBS:
        for f in sorted(run_dir.glob(pattern)):
            shutil.copy2(f, dest / f.name)
            copied.append(f.name)
    for log in [*logs, *sorted(run_dir.glob("vllm-*.log"))]:
        if log.exists():
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
            (dest / f"{log.name}.tail").write_text("\n".join(lines[-LOG_TAIL_LINES:]) + "\n", encoding="utf-8")
            copied.append(f"{log.name}.tail")
    return copied


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--repo", type=Path, default=Path.cwd())
    r.add_argument("--models", type=Path, nargs="*", default=[])
    r.add_argument("--note", default="")
    a = sub.add_parser("archive")
    a.add_argument("--run", type=Path, required=True)
    a.add_argument("--dest", type=Path, required=True)
    a.add_argument("--logs", type=Path, nargs="*", default=[])
    args = ap.parse_args(argv)

    if args.cmd == "record":
        e = record(args.out, args.repo, args.models, args.note)
        dirty = "DIRTY" if e["code"]["status"] else "clean"
        print(f"PROVENANCE|commit={e['code']['commit']} {dirty} job={e['env'].get('SLURM_JOB_ID')} -> {args.out}")
        if not e["code"]["commit"]:
            print("PROVENANCE|ERROR no git commit recorded (is git on PATH? is --repo a checkout?)", file=sys.stderr)
            return 1
        return 0
    files = archive(args.run, args.dest, args.logs)
    print(f"ARCHIVE|{len(files)} files -> {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
