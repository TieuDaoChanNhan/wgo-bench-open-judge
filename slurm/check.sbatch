#!/usr/bin/env bash
#SBATCH -p <partition>
#SBATCH -N 1
#SBATCH --time=00:30:00
#SBATCH --account=<budget>
#SBATCH --job-name=subtask-eval-check
#SBATCH --output=%x-%j.log
# CPU only. Unit tests, then the scorer check against EgoANT's released HomER predictions.
#   EGOANT_PRED=<dir inside $EGOANT_DIR> sbatch slurm/check.sbatch
set -euo pipefail
source "$W/env.sh"
cd "$REPO"

echo "##### unit tests"
"$PY" -m pytest -q tests

echo "##### scorer check (expect micro: pred=308 gold=470 match=79, F1 0.203)"
: "${EGOANT_PRED:?set EGOANT_PRED to the EgoANT prediction directory}"
OUT="$RUNS_DIR/egoant-ref"
"$PY" -m subtask_eval annotate import --format egoant --src "$EGOANT_DIR/$EGOANT_PRED" \
  --root "$WGO_ROOT" --subset homer --out "$OUT" --name egoant-ref
"$PY" -m subtask_eval score --root "$WGO_ROOT" --run "$OUT" --subset homer | tee "$OUT/check.txt"
grep -q "pred= 308 gold= 470 match=  79" "$OUT/check.txt" && echo "CHECK|ok" || { echo "CHECK|MISMATCH"; exit 1; }
