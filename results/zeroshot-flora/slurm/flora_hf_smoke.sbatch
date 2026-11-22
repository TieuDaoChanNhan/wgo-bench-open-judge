#!/usr/bin/env bash
#SBATCH -p <partition>
#SBATCH -N 1
#SBATCH --time=01:00:00
#SBATCH --account=<budget>
#SBATCH --job-name=wgo-flora-hf-smoke
#SBATCH --output=%x-%j.log
set -uo pipefail

source "$W/env.sh"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

OUT="$RUNS_DIR/flora_hf_smoke"
mkdir -p "$OUT"
cd "$REPO"

echo "##### GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "##### starting serve_hf.py"
nohup "$PY_VLLM" -m wgo_eval.serve_hf --model-dir "$MODELS_DIR/flora-captioner" \
  --served-model-name nvidia/flora-captioner --host 127.0.0.1 --port 8010 \
  > "$OUT/serve_hf.log" 2>&1 &
SPID=$!
echo "SERVE|serve_hf pid=$SPID"

ready=0
for _ in $(seq 60); do
  sleep 10
  if curl -sf http://127.0.0.1:8010/v1/models >/dev/null 2>&1; then
    ready=1
    echo "SERVE|ready"
    break
  fi
  kill -0 "$SPID" 2>/dev/null || { echo "SERVE|serve_hf process died"; break; }
done

echo "##### dtype log so far"
grep "^DTYPE|" "$OUT/serve_hf.log" || echo "no DTYPE lines yet"

if [ "$ready" = "1" ]; then
  echo "##### label 20 homer segments, --workers 1"
  "$HARNESS_PY" -m wgo_eval.label --root "$WGO_ROOT" --out "$OUT/preds.jsonl" \
    --base-url http://127.0.0.1:8010/v1 --model nvidia/flora-captioner \
    --subset homer --limit 20 --workers 1
  echo "LABEL|exit=$?"
  cat "$OUT/preds.jsonl"
else
  echo "RUN|serve_hf did not become ready, see $OUT/serve_hf.log"
fi

kill "$SPID" 2>/dev/null || true
pkill -f "[b]enchmark.serve_hf" || true
echo "RUN|flora hf smoke done"
