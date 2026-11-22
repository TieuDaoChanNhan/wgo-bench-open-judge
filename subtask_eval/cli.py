"""Command-line entry point: `subtask-eval <command> ...` (or `python -m subtask_eval <command> ...`).

    subtask-eval prepare   --root data/wgo
    subtask-eval annotate  contact_sheet --config configs/cs-qwen36-27b.json --root data/wgo --out runs/cs-qwen36-27b
    subtask-eval annotate  import --format egoant --src <dir> --root data/wgo --out runs/egoant-ref
    subtask-eval score     --root data/wgo --run runs/cs-qwen36-27b --write-pairs
    subtask-eval judge     --run runs/cs-qwen36-27b --model Qwen/Qwen3-32B-AWQ
    subtask-eval agree     compare --a <verdicts> --b <verdicts>     (judge vs judge / reference / human)
    subtask-eval report    runs/* --out report.md
"""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "prepare": "subtask_eval.data.wgo",
    "score": "subtask_eval.scoring.score",
    "judge": "subtask_eval.scoring.judge",
    "agree": "subtask_eval.scoring.agreement",
    "report": "subtask_eval.report",
}
ANNOTATORS = {
    "contact_sheet": "subtask_eval.annotators.contact_sheet",
    "import": "subtask_eval.annotators.imported",
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("annotators:", ", ".join(sorted(ANNOTATORS)))
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "annotate":
        if not rest or rest[0] not in ANNOTATORS:
            print(f"usage: subtask-eval annotate <{'|'.join(sorted(ANNOTATORS))}> ...")
            return 2
        module, prog, rest = ANNOTATORS[rest[0]], f"subtask-eval annotate {rest[0]}", rest[1:]
    elif cmd in COMMANDS:
        module, prog = COMMANDS[cmd], f"subtask-eval {cmd}"
    else:
        print(f"unknown command {cmd!r}; run `subtask-eval --help`")
        return 2
    saved, sys.argv = sys.argv, [prog, *rest]
    try:
        return int(importlib.import_module(module).main() or 0)
    finally:
        sys.argv = saved


if __name__ == "__main__":
    raise SystemExit(main())
