#!/usr/bin/env bash
#SBATCH -p <partition>
#SBATCH -N 1
#SBATCH --time=02:00:00
#SBATCH --account=<budget>
#SBATCH --job-name=wgo-flora-full
#SBATCH --output=%x-%j.log
set -uo pipefail

source "$W/env.sh"
export HF_HUB_OFFLINE=1

OUT="$RUNS_DIR/zs-flora"
mkdir -p "$OUT"
cd "$REPO"

echo "##### GPU0: serve_hf.py (FLORA), GPU1: vLLM judge -- start both now, use judge later"
CUDA_VISIBLE_DEVICES=0 nohup "$PY_VLLM" -m wgo_eval.serve_hf --model-dir "$MODELS_DIR/flora-captioner" \
  --served-model-name nvidia/flora-captioner --host 127.0.0.1 --port 8010 \
  --stop-at-newline \
  > "$OUT/serve_hf.log" 2>&1 &
FPID=$!
echo "SERVE|flora pid=$FPID"

CUDA_VISIBLE_DEVICES=1 nohup "$PY_VLLM" -m vllm.entrypoints.cli.main serve "$MODELS_DIR/Qwen3-32B-AWQ" \
  --served-model-name Qwen/Qwen3-32B-AWQ --host 127.0.0.1 --port 8001 \
  --max-model-len 4096 --gpu-memory-utilization 0.90 --dtype auto \
  > "$OUT/vllm-judge.log" 2>&1 &
JPID=$!
echo "SERVE|judge pid=$JPID"

wait_ready() {
  local name="$1" port="$2" pid="$3"
  for _ in $(seq 180); do
    sleep 10
    curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 && { echo "SERVE|ready $name"; return 0; }
    kill -0 "$pid" 2>/dev/null || { echo "SERVE|$name process died"; return 1; }
  done
  echo "SERVE|FAILED $name"
  return 1
}

wait_ready flora 8010 "$FPID" || { echo "RUN|abort: flora did not come up"; exit 1; }
wait_ready judge 8001 "$JPID" || { echo "RUN|abort: judge did not come up"; exit 1; }

echo "##### label 743 segments (full set, no --subset/--limit), --workers 1"
t0=$(date +%s)
"$HARNESS_PY" -m wgo_eval.label --root "$WGO_ROOT" --out "$OUT/preds.jsonl" \
  --base-url http://127.0.0.1:8010/v1 --model nvidia/flora-captioner --workers 1
t1=$(date +%s)
n=$(wc -l < "$OUT/preds.jsonl")
echo "TIMING|label wall_s=$((t1 - t0)) segments=$n segs_per_sec=$(awk -v n="$n" -v t="$((t1 - t0))" 'BEGIN{print (t>0)? n/t : "NA"}')"

kill "$FPID" 2>/dev/null || true
pkill -f "[b]enchmark.serve_hf" || true

echo "##### judge all 743 predictions (shared cache)"
t0=$(date +%s)
"$HARNESS_PY" -m wgo_eval.judge --preds "$OUT/preds.jsonl" --root "$WGO_ROOT" \
  --out "$OUT/judged.jsonl" --base-url http://127.0.0.1:8001/v1 --model Qwen/Qwen3-32B-AWQ \
  --cache "$RUNS_DIR/judge_cache.jsonl"
t1=$(date +%s)
echo "TIMING|judge wall_s=$((t1 - t0))"

kill "$JPID" 2>/dev/null || true
pkill -f "[v]llm.entrypoints.cli.main serve" || true

echo "##### score"
"$HARNESS_PY" -m wgo_eval.score --root "$WGO_ROOT" --subset homer \
  --judged "$OUT/judged.jsonl" --preds "$OUT/preds.jsonl" --out "$OUT/score_homer.json"
"$HARNESS_PY" -m wgo_eval.score --root "$WGO_ROOT" \
  --judged "$OUT/judged.jsonl" --preds "$OUT/preds.jsonl" --out "$OUT/score_all.json"
echo "SCORE|homer:"; cat "$OUT/score_homer.json"
echo "SCORE|all:"; cat "$OUT/score_all.json"

echo "##### compare: flora vs zs-qwen36-27b, flora vs zs-qwen3vl8b (homer + all)"
OLDRUNS="$RUNS_DIR/zeroshot_full"
for other in zs-qwen36-27b zs-qwen3vl8b; do
  "$HARNESS_PY" -m wgo_eval.compare --root "$WGO_ROOT" --a "$OUT/judged.jsonl" --b "$OLDRUNS/$other/judged.jsonl" \
    --subset homer --out "$OUT/compare_flora_vs_${other}_homer.json"
  "$HARNESS_PY" -m wgo_eval.compare --root "$WGO_ROOT" --a "$OUT/judged.jsonl" --b "$OLDRUNS/$other/judged.jsonl" \
    --out "$OUT/compare_flora_vs_${other}_all.json"
done

echo "RUN|flora full run ok"
