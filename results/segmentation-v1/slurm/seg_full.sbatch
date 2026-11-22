#!/usr/bin/env bash
#SBATCH -p <partition>
#SBATCH -N 1
#SBATCH --time=04:00:00
#SBATCH --account=<budget>
#SBATCH --job-name=wgo-seg-full
#SBATCH --output=%x-%j.log
# Segmentation + end-to-end run on all 100 WGO-Bench episodes: one node, 4 GPUs,
# judge on GPU 0 and one segmenter per remaining GPU. Same env.sh as zeroshot_full.sbatch
# (REPO, RUNS_DIR, MODELS_DIR, PY_VLLM, HARNESS_PY, WGO_ROOT).
# LIMIT=2 sbatch ... runs a smoke test on the first 2 episodes per segmenter.
set -uo pipefail   # no -e: one segmenter may fail without killing the job

source "$W/env.sh"
export HF_HUB_OFFLINE=1
LIMIT="${LIMIT:-0}"
PROMPT="${PROMPT:-gepa_duration_prior_v1}"

OUT="$RUNS_DIR/segmentation_v1"
mkdir -p "$OUT"
cd "$REPO" || { echo "RUN|abort: cannot cd to REPO=$REPO"; exit 1; }

echo "##### environment"
echo "vllm: $("$PY_VLLM" -c 'import vllm; print(vllm.__version__)')"
nvidia-smi -L
for f in "$MODELS_DIR"/*/HF_REVISION.txt; do cat "$f"; done
"$HARNESS_PY" -m pytest -q tests || { echo "RUN|abort: unit tests failed"; exit 1; }

declare -A SERVER_PID SERVER_READY
stop_all() {
  for name in "${!SERVER_PID[@]}"; do kill "${SERVER_PID[$name]}" 2>/dev/null || true; done
  pkill -f "[v]llm.entrypoints.cli.main serve" || true
}
trap stop_all EXIT

# start_server <name> <gpu> <model_dir> <port> <served_name> <vlseg|text> <required 0|1>
start_server() {
  local name="$1" gpu="$2" model_dir="$3" port="$4" served="$5" kind="$6" required="$7"
  if [ "$required" = "0" ] && { [ ! -d "$model_dir" ] || [ ! -f "$model_dir/.done" ]; }; then
    echo "SERVE|skip $name: model not downloaded ($model_dir)"; SERVER_READY[$name]=0; return
  fi
  local extra=(--max-model-len 4096 --max-num-seqs 64)
  [ "$kind" = "vlseg" ] && extra=(--max-model-len 65536 --limit-mm-per-prompt '{"image": 64}' --max-num-seqs 8)
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY_VLLM" -m vllm.entrypoints.cli.main serve "$model_dir" \
    --served-model-name "$served" --host 127.0.0.1 --port "$port" \
    --gpu-memory-utilization 0.90 --dtype auto "${extra[@]}" > "$OUT/vllm-$name.log" 2>&1 &
  SERVER_PID[$name]=$!
  echo "SERVE|starting $name (gpu=$gpu port=$port)"
}

wait_ready() {
  local name="$1" port="$2" required="$3"
  [ "${SERVER_READY[$name]:-}" = "0" ] && return
  for _ in $(seq 180); do
    sleep 10
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "SERVE|ready $name on :${port}"; SERVER_READY[$name]=1; return
    fi
    kill -0 "${SERVER_PID[$name]}" 2>/dev/null || break
  done
  echo "SERVE|FAILED $name -- see $OUT/vllm-$name.log"; SERVER_READY[$name]=0
  [ "$required" = "1" ] && { echo "RUN|abort: required server $name did not come up"; exit 1; }
  kill "${SERVER_PID[$name]}" 2>/dev/null || true
}

echo "##### starting servers"
start_server judge     0 "$MODELS_DIR/Qwen3-32B-AWQ"         8001 Qwen/Qwen3-32B-AWQ            text  1
start_server seg36     1 "$MODELS_DIR/Qwen3.6-27B"           8002 Qwen/Qwen3.6-27B              vlseg 1
start_server seg38awq  2 "$MODELS_DIR/Qwen3.8-27B-AWQ-INT4"  8003 cyankiwi/Qwen3.8-27B-AWQ-INT4 vlseg 0
start_server seg8b     3 "$MODELS_DIR/Qwen3-VL-8B-Instruct"  8000 Qwen/Qwen3-VL-8B-Instruct     vlseg 0
wait_ready judge 8001 1; wait_ready seg36 8002 1; wait_ready seg38awq 8003 0; wait_ready seg8b 8000 0

declare -A PORT=( [seg-qwen36-27b]=8002 [seg-qwen38-27b]=8003 [seg-qwen3vl8b]=8000 )
declare -A SERVER=( [seg-qwen36-27b]=seg36 [seg-qwen38-27b]=seg38awq [seg-qwen3vl8b]=seg8b )
declare -A MODEL=( [seg-qwen36-27b]="Qwen/Qwen3.6-27B" [seg-qwen38-27b]="cyankiwi/Qwen3.8-27B-AWQ-INT4" [seg-qwen3vl8b]="Qwen/Qwen3-VL-8B-Instruct" )
JUDGE=(--base-url http://127.0.0.1:8001/v1 --model Qwen/Qwen3-32B-AWQ --cache "$RUNS_DIR/judge_cache.jsonl")

echo "##### segment -> match -> judge -> score, one pipeline per segmenter"
declare -A PID
for name in "${!PORT[@]}"; do
  [ "${SERVER_READY[${SERVER[$name]}]:-0}" = "1" ] || { echo "SEGMENT|skip $name"; continue; }
  D="$OUT/$name"; mkdir -p "$D"
  (
    t0=$(date +%s)
    "$HARNESS_PY" -m wgo_eval.segment --root "$WGO_ROOT" --out "$D/segpreds.jsonl" --prompt "$PROMPT" \
      --base-url "http://127.0.0.1:${PORT[$name]}/v1" --model "${MODEL[$name]}" --limit "$LIMIT"
    t1=$(date +%s)
    echo "TIMING|$name segment_wall_s=$((t1 - t0)) episodes=$(wc -l < "$D/segpreds.jsonl")" | tee "$D/timing.txt"
    "$HARNESS_PY" -m wgo_eval.score_seg --root "$WGO_ROOT" --segpreds "$D/segpreds.jsonl" --pairs-out "$D/e2e_pairs.jsonl"
    "$HARNESS_PY" -m wgo_eval.judge --pairs "$D/e2e_pairs.jsonl" --out "$D/e2e_judged.jsonl" "${JUDGE[@]}"
    for sub in all homer; do
      S=(); [ "$sub" = "homer" ] && S=(--subset homer)
      "$HARNESS_PY" -m wgo_eval.score_seg --root "$WGO_ROOT" "${S[@]}" --segpreds "$D/segpreds.jsonl" \
        --judged "$D/e2e_judged.jsonl" --out "$D/seg_score_$sub.json" | tee "$D/seg_score_$sub.txt"
    done
  ) > "$D/run.log" 2>&1 &
  PID[$name]=$!
done
for name in "${!PID[@]}"; do wait "${PID[$name]}"; echo "DONE|$name"; tail -n 6 "$OUT/$name/run.log"; done
echo "RUN|segmentation_v1 ok"
