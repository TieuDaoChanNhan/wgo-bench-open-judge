#!/usr/bin/env bash
#SBATCH -p <partition>
#SBATCH -N 1
#SBATCH --time=01:00:00
#SBATCH --account=<budget>
#SBATCH --job-name=wgo-flora-hf-speedtest
#SBATCH --output=%x-%j.log
set -uo pipefail

source "$W/env.sh"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

OUT="$RUNS_DIR/flora_hf_speedtest"
mkdir -p "$OUT"
cd "$REPO"

echo "##### kernels available"
"$PY_VLLM" -c "import causal_conv1d; print('causal_conv1d', causal_conv1d.__file__)"
"$PY_VLLM" -c "import fla; print('fla', fla.__file__)"

echo "##### starting serve_hf.py"
nohup "$PY_VLLM" -m wgo_eval.serve_hf --model-dir "$MODELS_DIR/flora-captioner" \
  --served-model-name nvidia/flora-captioner --host 127.0.0.1 --port 8011 \
  > "$OUT/serve_hf.log" 2>&1 &
SPID=$!
echo "SERVE|serve_hf pid=$SPID"

ready=0
for _ in $(seq 60); do
  sleep 10
  if curl -sf http://127.0.0.1:8011/v1/models >/dev/null 2>&1; then
    ready=1
    echo "SERVE|ready"
    break
  fi
  kill -0 "$SPID" 2>/dev/null || { echo "SERVE|serve_hf process died"; break; }
done

if [ "$ready" = "1" ]; then
  echo "##### checking for optimized-kernel fallback warnings"
  grep -i "falling back to its reference PyTorch" "$OUT/serve_hf.log" || echo "no fallback warnings"

  echo "##### label 20 homer segments, --workers 1 (timed)"
  t0=$(date +%s)
  "$HARNESS_PY" -m wgo_eval.label --root "$WGO_ROOT" --out "$OUT/preds.jsonl" \
    --base-url http://127.0.0.1:8011/v1 --model nvidia/flora-captioner \
    --subset homer --limit 20 --workers 1
  t1=$(date +%s)
  n=$(wc -l < "$OUT/preds.jsonl")
  echo "TIMING|wall_s=$((t1 - t0)) segments=$n segs_per_sec=$(awk -v n="$n" -v t="$((t1 - t0))" 'BEGIN{print (t>0)? n/t : "NA"}')"
  cat "$OUT/preds.jsonl"
else
  echo "RUN|serve_hf did not become ready, see $OUT/serve_hf.log"
fi

kill "$SPID" 2>/dev/null || true
pkill -f "[b]enchmark.serve_hf" || true
echo "RUN|flora hf speedtest done"
