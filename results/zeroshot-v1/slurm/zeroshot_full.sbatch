#!/usr/bin/env bash
#SBATCH -p <partition>
#SBATCH -N 1
#SBATCH --time=03:00:00
#SBATCH --account=<budget>
#SBATCH --job-name=wgo-zs-full
#SBATCH --output=%x-%j.log
set -uo pipefail   # no -e: a baseline labeler is allowed to fail without killing the job

source "$W/env.sh"
export HF_HUB_OFFLINE=1

OUT="$RUNS_DIR/zeroshot_full"
mkdir -p "$OUT"
cd "$REPO" || { echo "RUN|abort: cannot cd to REPO=$REPO"; exit 1; }

echo "##### environment"
echo "vllm: $("$PY_VLLM" -c 'import vllm; print(vllm.__version__)')"
echo "torch: $("$PY_VLLM" -c 'import torch; print(torch.__version__)')"
nvidia-smi -L
for f in "$MODELS_DIR"/*/HF_REVISION.txt; do cat "$f"; done

declare -A SERVER_PID
declare -A SERVER_READY

stop_all() {
  for name in "${!SERVER_PID[@]}"; do
    kill "${SERVER_PID[$name]}" 2>/dev/null || true
  done
  pkill -f "[v]llm.entrypoints.cli.main serve" || true
  for _ in $(seq 30); do pgrep -f "[v]llm.entrypoints.cli.main serve" >/dev/null || break; sleep 2; done
}
trap stop_all EXIT

# start_server <name> <gpu> <model_dir> <port> <served_name> <vl|text> <required 0|1>
start_server() {
  local name="$1" gpu="$2" model_dir="$3" port="$4" served="$5" kind="$6" required="$7"
  if [ "$required" = "0" ] && { [ ! -d "$model_dir" ] || [ ! -f "$model_dir/.done" ]; }; then
    echo "SERVE|skip $name: model not downloaded ($model_dir)"
    SERVER_READY[$name]=0
    return
  fi
  local extra=(--max-model-len 4096 --max-num-seqs 64)
  [ "$kind" = "vl" ] && extra=(--max-model-len 16384 --limit-mm-per-prompt '{"image": 8}' --max-num-seqs 16)
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY_VLLM" -m vllm.entrypoints.cli.main serve "$model_dir" \
    --served-model-name "$served" --host 127.0.0.1 --port "$port" \
    --gpu-memory-utilization 0.90 --dtype auto "${extra[@]}" \
    > "$OUT/vllm-$name.log" 2>&1 &
  SERVER_PID[$name]=$!
  echo "SERVE|starting $name (gpu=$gpu port=$port pid=${SERVER_PID[$name]} required=$required)"
}

# wait_ready <name> <port> <required 0|1>
wait_ready() {
  local name="$1" port="$2" required="$3"
  [ "${SERVER_READY[$name]:-}" = "0" ] && return
  for _ in $(seq 180); do
    sleep 10
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "SERVE|ready $name on :${port}"
      SERVER_READY[$name]=1
      return
    fi
    kill -0 "${SERVER_PID[$name]}" 2>/dev/null || break
  done
  echo "SERVE|FAILED $name (required=$required) -- see $OUT/vllm-$name.log"
  SERVER_READY[$name]=0
  if [ "$required" = "1" ]; then
    echo "RUN|abort: required server $name did not come up"
    exit 1
  fi
  kill "${SERVER_PID[$name]}" 2>/dev/null || true
}

echo "##### starting servers"
start_server judge     0 "$MODELS_DIR/Qwen3-32B-AWQ"          8001 Qwen/Qwen3-32B-AWQ          text 1
start_server main       1 "$MODELS_DIR/Qwen3-VL-8B-Instruct"   8000 Qwen/Qwen3-VL-8B-Instruct    vl   1
start_server base36     2 "$MODELS_DIR/Qwen3.6-27B"            8002 Qwen/Qwen3.6-27B             vl   0
start_server base38awq  3 "$MODELS_DIR/Qwen3.8-27B-AWQ-INT4"   8003 cyankiwi/Qwen3.8-27B-AWQ-INT4 vl   0

wait_ready judge    8001 1
wait_ready main     8000 1
wait_ready base36   8002 0
wait_ready base38awq 8003 0

# NAME -> port
declare -A LABELER_PORT=( [zs-qwen3vl8b]=8000 [zs-qwen36-27b]=8002 [zs-qwen38-27b]=8003 )
declare -A LABELER_SERVER=( [zs-qwen3vl8b]=main [zs-qwen36-27b]=base36 [zs-qwen38-27b]=base38awq )
declare -A LABELER_MODEL=( [zs-qwen3vl8b]="Qwen/Qwen3-VL-8B-Instruct" [zs-qwen36-27b]="Qwen/Qwen3.6-27B" [zs-qwen38-27b]="cyankiwi/Qwen3.8-27B-AWQ-INT4" )

echo "##### labeling (parallel across labelers that came up)"
declare -A LABEL_PID
for name in "${!LABELER_PORT[@]}"; do
  server="${LABELER_SERVER[$name]}"
  [ "${SERVER_READY[$server]:-0}" = "1" ] || { echo "LABEL|skip $name: server $server not ready"; continue; }
  mkdir -p "$OUT/$name"
  (
    t0=$(date +%s)
    "$HARNESS_PY" -m wgo_eval.label --root "$WGO_ROOT" --out "$OUT/$name/preds.jsonl" \
      --base-url "http://127.0.0.1:${LABELER_PORT[$name]}/v1" --model "${LABELER_MODEL[$name]}"
    t1=$(date +%s)
    n=$(wc -l < "$OUT/$name/preds.jsonl")
    echo "TIMING|$name wall_s=$((t1 - t0)) segments=$n segs_per_sec=$(awk -v n="$n" -v t="$((t1 - t0))" 'BEGIN{print (t>0)? n/t : "NA"}')" \
      | tee "$OUT/$name/timing.txt"
  ) > "$OUT/$name/label_run.log" 2>&1 &
  LABEL_PID[$name]=$!
done
for name in "${!LABEL_PID[@]}"; do wait "${LABEL_PID[$name]}"; done

echo "##### judging (shared judge server, shared cache)"
declare -A JUDGE_PID
for name in "${!LABELER_PORT[@]}"; do
  [ -f "$OUT/$name/preds.jsonl" ] || continue
  (
    t0=$(date +%s)
    "$HARNESS_PY" -m wgo_eval.judge --preds "$OUT/$name/preds.jsonl" --root "$WGO_ROOT" \
      --out "$OUT/$name/judged.jsonl" --base-url http://127.0.0.1:8001/v1 --model Qwen/Qwen3-32B-AWQ \
      --cache "$RUNS_DIR/judge_cache.jsonl"
    t1=$(date +%s)
    echo "TIMING|judge-$name wall_s=$((t1 - t0))" | tee -a "$OUT/$name/timing.txt"
  ) > "$OUT/$name/judge_run.log" 2>&1 &
  JUDGE_PID[$name]=$!
done
for name in "${!JUDGE_PID[@]}"; do wait "${JUDGE_PID[$name]}"; done

echo "##### scoring"
for name in "${!LABELER_PORT[@]}"; do
  [ -f "$OUT/$name/judged.jsonl" ] || continue
  "$HARNESS_PY" -m wgo_eval.score --root "$WGO_ROOT" --subset homer \
    --judged "$OUT/$name/judged.jsonl" --preds "$OUT/$name/preds.jsonl" --out "$OUT/$name/score_homer.json"
  "$HARNESS_PY" -m wgo_eval.score --root "$WGO_ROOT" \
    --judged "$OUT/$name/judged.jsonl" --preds "$OUT/$name/preds.jsonl" --out "$OUT/$name/score_all.json"
  echo "SCORE|$name homer:"; cat "$OUT/$name/score_homer.json"
  echo "SCORE|$name all:";   cat "$OUT/$name/score_all.json"
done

echo "RUN|zeroshot_full ok"
