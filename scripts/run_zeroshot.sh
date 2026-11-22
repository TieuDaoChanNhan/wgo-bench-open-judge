#!/usr/bin/env bash
# Zero-shot run on one GPU VM: unit tests, label, judge agreement check, judge, score.
#   LABELER=Qwen/Qwen3-VL-8B-Instruct NAME=zs-qwen3vl8b SUBSET=homer LIMIT=0 bash scripts/run_zeroshot.sh
# LIMIT>0 labels only the first N segments (smoke test). Outputs go to $W/runs/$NAME/.
set -euo pipefail
W="${W:-/data/bench}"
LABELER="${LABELER:-Qwen/Qwen3-VL-8B-Instruct}"
JUDGE_REPO="${JUDGE_REPO:-Qwen/Qwen3-32B-AWQ}"
JUDGE_DIR="$W/models/$(basename "$JUDGE_REPO")"
NAME="${NAME:-zs-qwen3vl8b}"
SUBSET="${SUBSET:-homer}"
LIMIT="${LIMIT:-0}"
EGOANT_COMMIT=b27ea59f5f
PY="$W/.venv/bin/python"
OUT="$W/runs/$NAME"
WGO="$W/data/wgo"
mkdir -p "$OUT"
cd "$W/repo"

echo "##### unit tests"
"$PY" -m pytest -q tests

echo "##### label ($LABELER, subset=${SUBSET:-all}, limit=$LIMIT)"
bash scripts/serve.sh "$LABELER" 8000 vl
"$PY" -m wgo_eval.label --root "$WGO" --out "$OUT/preds.jsonl" --base-url http://127.0.0.1:8000/v1 \
  --model "$LABELER" ${SUBSET:+--subset "$SUBSET"} --limit "$LIMIT"
bash scripts/serve.sh stop

echo "##### judge ($JUDGE_REPO)"
for _ in $(seq 120); do [ -f "$JUDGE_DIR/.done" ] && break; sleep 30; done
[ -f "$JUDGE_DIR/.done" ] || { echo "RUN|judge weights missing"; exit 1; }
bash scripts/serve.sh "$JUDGE_DIR" 8001 text "$JUDGE_REPO"
JUDGE=(--base-url http://127.0.0.1:8001/v1 --model "$JUDGE_REPO" --cache "$W/runs/judge_cache.jsonl")

# Agreement with the gemini-3.5-flash verdicts released by EgoANT (79 HomER pairs).
if [ ! -d "$W/egoant" ]; then
  git clone -q https://github.com/ShuaiZhou302/EgoANT.git "$W/egoant"
  git -C "$W/egoant" checkout -q "$EGOANT_COMMIT"
fi
AG="$W/runs/judge-agreement"
"$PY" -m wgo_eval.judge_agreement extract --root "$WGO" --out "$AG/pairs.jsonl" \
  --egoant-json "$W/egoant/reference_outputs/wgo_homer_v5_1/scores/semantic_e2e_gemini35.json"
"$PY" -m wgo_eval.judge --pairs "$AG/pairs.jsonl" --out "$AG/judged.jsonl" "${JUDGE[@]}"
"$PY" -m wgo_eval.judge_agreement report --pairs "$AG/pairs.jsonl" --judged "$AG/judged.jsonl" | tee "$AG/report.json"

# Segments without a label (all but the first LIMIT in a smoke test) count as no match, without a judge call.
SCOPE=(--root "$WGO" ${SUBSET:+--subset "$SUBSET"})
"$PY" -m wgo_eval.judge --preds "$OUT/preds.jsonl" --out "$OUT/judged.jsonl" "${SCOPE[@]}" "${JUDGE[@]}"
bash scripts/serve.sh stop

echo "##### score"
"$PY" -m wgo_eval.score "${SCOPE[@]}" --judged "$OUT/judged.jsonl" --preds "$OUT/preds.jsonl" --out "$OUT/score.json"
echo "RUN|ok $NAME"
