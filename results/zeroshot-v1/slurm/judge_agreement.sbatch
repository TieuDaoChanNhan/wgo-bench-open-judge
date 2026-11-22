#!/usr/bin/env bash
#SBATCH -p <partition>
#SBATCH -N 1
#SBATCH --time=00:40:00
#SBATCH --account=<budget>
#SBATCH --job-name=wgo-agree
#SBATCH --output=%x-%j.log
set -euo pipefail

source "$W/env.sh"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

OUT="$RUNS_DIR/agree"
mkdir -p "$OUT"
cd "$REPO"

stop_vllm() {
  pkill -f "[v]llm.entrypoints.cli.main serve" || true
  for _ in $(seq 30); do pgrep -f "[v]llm.entrypoints.cli.main serve" >/dev/null || break; sleep 2; done
}
trap stop_vllm EXIT

wait_ready() {
  local port="$1" name="$2"
  for _ in $(seq 180); do
    sleep 10
    curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null && { echo "SERVE|ok ${name} on :${port}"; return 0; }
    pgrep -f "[v]llm.entrypoints.cli.main serve" >/dev/null || break
  done
  echo "SERVE|fail ${name}"
  return 1
}

echo "##### extract pairs (no GPU needed)"
"$HARNESS_PY" -m wgo_eval.judge_agreement extract --root "$WGO_ROOT" --out "$OUT/pairs.jsonl" \
  --egoant-json "$EGOANT_ROOT/reference_outputs/wgo_homer_v5_1/scores/semantic_e2e_gemini35.json"
echo "AGREE|pairs=$(wc -l < "$OUT/pairs.jsonl")"

echo "##### judge: Qwen3-32B-AWQ on :8001"
nohup "$PY_VLLM" -m vllm.entrypoints.cli.main serve "$MODELS_DIR/Qwen3-32B-AWQ" \
  --served-model-name Qwen/Qwen3-32B-AWQ --host 127.0.0.1 --port 8001 \
  --max-model-len 4096 --gpu-memory-utilization 0.90 --dtype auto \
  > "$OUT/vllm-8001.log" 2>&1 &
wait_ready 8001 judge

"$HARNESS_PY" -m wgo_eval.judge --pairs "$OUT/pairs.jsonl" --out "$OUT/judged.jsonl" \
  --base-url http://127.0.0.1:8001/v1 --model Qwen/Qwen3-32B-AWQ --cache "$RUNS_DIR/judge_cache.jsonl"

stop_vllm

echo "##### report"
"$HARNESS_PY" -m wgo_eval.judge_agreement report --pairs "$OUT/pairs.jsonl" --judged "$OUT/judged.jsonl" | tee "$OUT/report.json"

echo "RUN|agreement ok"
