#!/usr/bin/env bash
#SBATCH -p <partition>
#SBATCH -N 1
#SBATCH --time=02:00:00
#SBATCH --account=<account>
#SBATCH --job-name=wgo-zs-small
#SBATCH --output=%x-%j.log
set -uo pipefail

source $W/env.sh
export HF_HUB_OFFLINE=1

OUT="$RUNS_DIR/zeroshot_small"
mkdir -p "$OUT"
cd "$REPO" || { echo "RUN|abort: cannot cd to REPO=$REPO"; exit 1; }

echo "##### environment"
echo "vllm: $("$PY_VLLM" -c 'import vllm; print(vllm.__version__)')"
echo "torch: $("$PY_VLLM" -c 'import torch; print(torch.__version__)')"
nvidia-smi -L
cat "$MODELS_DIR/Qwen3-VL-2B-Instruct/HF_REVISION.txt"; echo
cat "$MODELS_DIR/Qwen3-VL-4B-Instruct/HF_REVISION.txt"; echo

declare -A SERVER_PID

stop_all() {
  for name in "${!SERVER_PID[@]}"; do
    kill "${SERVER_PID[$name]}" 2>/dev/null || true
  done
  pkill -f "[v]llm.entrypoints.cli.main serve" || true
  for _ in $(seq 30); do pgrep -f "[v]llm.entrypoints.cli.main serve" >/dev/null || break; sleep 2; done
}
trap stop_all EXIT

start_server() {
  local name="$1" gpu="$2" model_dir="$3" port="$4" served="$5"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY_VLLM" -m vllm.entrypoints.cli.main serve "$model_dir" \
    --served-model-name "$served" --host 127.0.0.1 --port "$port" \
    --gpu-memory-utilization 0.90 --dtype auto \
    --max-model-len 16384 --limit-mm-per-prompt '{"image": 8}' --max-num-seqs 16 \
    > "$OUT/vllm-$name.log" 2>&1 &
  SERVER_PID[$name]=$!
  echo "SERVE|starting $name (gpu=$gpu port=$port pid=${SERVER_PID[$name]})"
}

start_judge() {
  CUDA_VISIBLE_DEVICES=0 nohup "$PY_VLLM" -m vllm.entrypoints.cli.main serve "$MODELS_DIR/Qwen3-32B-AWQ" \
    --served-model-name Qwen/Qwen3-32B-AWQ --host 127.0.0.1 --port 8001 \
    --max-model-len 4096 --gpu-memory-utilization 0.90 --dtype auto \
    > "$OUT/vllm-judge.log" 2>&1 &
  SERVER_PID[judge]=$!
  echo "SERVE|starting judge (gpu=0 port=8001 pid=${SERVER_PID[judge]})"
}

wait_ready() {
  local name="$1" port="$2"
  for _ in $(seq 180); do
    sleep 10
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "SERVE|ready $name on :${port}"
      return 0
    fi
    kill -0 "${SERVER_PID[$name]}" 2>/dev/null || break
  done
  echo "RUN|abort: $name did not come up -- see $OUT/vllm-$name.log"
  exit 1
}

declare -A LABELER_PORT=( [zs-qwen3vl2b]=8010 [zs-qwen3vl4b]=8011 )
declare -A LABELER_MODEL=( [zs-qwen3vl2b]="Qwen/Qwen3-VL-2B-Instruct" [zs-qwen3vl4b]="Qwen/Qwen3-VL-4B-Instruct" )
declare -A LABELER_GPU=( [zs-qwen3vl2b]=1 [zs-qwen3vl4b]=2 )
declare -A VRAM_PID

echo "##### starting servers (VRAM sampling starts now too, to catch the load-time peak)"
start_judge
mkdir -p "$OUT/zs-qwen3vl2b" "$OUT/zs-qwen3vl4b"
for name in zs-qwen3vl2b zs-qwen3vl4b; do
  start_server "${name#zs-qwen3vl}" "${LABELER_GPU[$name]}" "$MODELS_DIR/${LABELER_MODEL[$name]#Qwen/}" \
    "${LABELER_PORT[$name]}" "${LABELER_MODEL[$name]}"
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${LABELER_GPU[$name]}"; sleep 2; done ) \
    > "$OUT/$name/vram_samples.txt" &
  VRAM_PID[$name]=$!
done

wait_ready judge  8001
wait_ready 2b 8010
wait_ready 4b 8011

echo "##### labeling (sequential, cleaner timing per model)"
for name in zs-qwen3vl2b zs-qwen3vl4b; do
  t0=$(date +%s)
  "$HARNESS_PY" -m wgo_eval.label --root "$WGO_ROOT" --out "$OUT/$name/preds.jsonl" \
    --base-url "http://127.0.0.1:${LABELER_PORT[$name]}/v1" --model "${LABELER_MODEL[$name]}"
  t1=$(date +%s)
  kill "${VRAM_PID[$name]}" 2>/dev/null || true
  n=$(wc -l < "$OUT/$name/preds.jsonl")
  peak=$(sort -n "$OUT/$name/vram_samples.txt" | tail -1)
  echo "TIMING|$name wall_s=$((t1 - t0)) segments=$n segs_per_sec=$(awk -v n="$n" -v t="$((t1 - t0))" 'BEGIN{print (t>0)? n/t : "NA"}') peak_vram_mib=$peak" \
    | tee "$OUT/$name/timing.txt"
done

echo "##### judging (shared judge server, shared cache)"
for name in zs-qwen3vl2b zs-qwen3vl4b; do
  t0=$(date +%s)
  "$HARNESS_PY" -m wgo_eval.judge --preds "$OUT/$name/preds.jsonl" --root "$WGO_ROOT" \
    --out "$OUT/$name/judged.jsonl" --base-url http://127.0.0.1:8001/v1 --model Qwen/Qwen3-32B-AWQ \
    --cache "$RUNS_DIR/judge_cache.jsonl"
  t1=$(date +%s)
  echo "TIMING|judge-$name wall_s=$((t1 - t0))" | tee -a "$OUT/$name/timing.txt"
done

echo "##### scoring"
for name in zs-qwen3vl2b zs-qwen3vl4b; do
  "$HARNESS_PY" -m wgo_eval.score --root "$WGO_ROOT" --subset homer \
    --judged "$OUT/$name/judged.jsonl" --preds "$OUT/$name/preds.jsonl" --out "$OUT/$name/score_homer.json"
  "$HARNESS_PY" -m wgo_eval.score --root "$WGO_ROOT" \
    --judged "$OUT/$name/judged.jsonl" --preds "$OUT/$name/preds.jsonl" --out "$OUT/$name/score_all.json"
  echo "SCORE|$name homer:"; cat "$OUT/$name/score_homer.json"
  echo "SCORE|$name all:";   cat "$OUT/$name/score_all.json"
done

echo "##### compare: 2b vs 8b, 4b vs 8b (homer+all), 4b vs 2b (homer)"
mkdir -p "$OUT/compare"
EIGHTB="$RUNS_DIR/zeroshot_full/zs-qwen3vl8b"
"$HARNESS_PY" -m wgo_eval.compare --root "$WGO_ROOT" --a "$OUT/zs-qwen3vl2b/judged.jsonl" --b "$EIGHTB/judged.jsonl" \
  --subset homer --out "$OUT/compare/compare_2b_vs_8b_homer.json"
"$HARNESS_PY" -m wgo_eval.compare --root "$WGO_ROOT" --a "$OUT/zs-qwen3vl2b/judged.jsonl" --b "$EIGHTB/judged.jsonl" \
  --out "$OUT/compare/compare_2b_vs_8b_all.json"
"$HARNESS_PY" -m wgo_eval.compare --root "$WGO_ROOT" --a "$OUT/zs-qwen3vl4b/judged.jsonl" --b "$EIGHTB/judged.jsonl" \
  --subset homer --out "$OUT/compare/compare_4b_vs_8b_homer.json"
"$HARNESS_PY" -m wgo_eval.compare --root "$WGO_ROOT" --a "$OUT/zs-qwen3vl4b/judged.jsonl" --b "$EIGHTB/judged.jsonl" \
  --out "$OUT/compare/compare_4b_vs_8b_all.json"
"$HARNESS_PY" -m wgo_eval.compare --root "$WGO_ROOT" --a "$OUT/zs-qwen3vl4b/judged.jsonl" --b "$OUT/zs-qwen3vl2b/judged.jsonl" \
  --subset homer --out "$OUT/compare/compare_4b_vs_2b_homer.json"

echo "RUN|zeroshot_small ok"
