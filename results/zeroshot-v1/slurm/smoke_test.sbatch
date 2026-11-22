#!/usr/bin/env bash
#SBATCH -p <partition>
#SBATCH -N 1
#SBATCH --time=01:00:00
#SBATCH --account=<budget>
#SBATCH --job-name=wgo-smoke
#SBATCH --output=%x-%j.log
set -euo pipefail

source "$W/env.sh"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

OUT="$RUNS_DIR/smoke"
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

echo "##### labeler: Qwen3-VL-8B-Instruct on :8000"
nohup "$PY_VLLM" -m vllm.entrypoints.cli.main serve "$MODELS_DIR/Qwen3-VL-8B-Instruct" \
  --served-model-name Qwen/Qwen3-VL-8B-Instruct --host 127.0.0.1 --port 8000 \
  --max-model-len 16384 --limit-mm-per-prompt '{"image": 8}' \
  --gpu-memory-utilization 0.90 --dtype auto > "$OUT/vllm-8000.log" 2>&1 &
wait_ready 8000 labeler

"$HARNESS_PY" -m wgo_eval.label --root "$WGO_ROOT" --out "$OUT/preds.jsonl" \
  --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3-VL-8B-Instruct \
  --subset homer --limit 20

stop_vllm

echo "##### judge: Qwen3-32B-AWQ on :8001"
nohup "$PY_VLLM" -m vllm.entrypoints.cli.main serve "$MODELS_DIR/Qwen3-32B-AWQ" \
  --served-model-name Qwen/Qwen3-32B-AWQ --host 127.0.0.1 --port 8001 \
  --max-model-len 4096 --gpu-memory-utilization 0.90 --dtype auto \
  > "$OUT/vllm-8001.log" 2>&1 &
wait_ready 8001 judge

"$HARNESS_PY" -m wgo_eval.judge --preds "$OUT/preds.jsonl" --root "$WGO_ROOT" --subset homer \
  --out "$OUT/judged.jsonl" --base-url http://127.0.0.1:8001/v1 --model Qwen/Qwen3-32B-AWQ \
  --cache "$RUNS_DIR/judge_cache.jsonl"

stop_vllm

echo "##### score"
"$HARNESS_PY" -m wgo_eval.score --root "$WGO_ROOT" --subset homer \
  --judged "$OUT/judged.jsonl" --preds "$OUT/preds.jsonl" --out "$OUT/score.json"

echo "RUN|smoke ok"
