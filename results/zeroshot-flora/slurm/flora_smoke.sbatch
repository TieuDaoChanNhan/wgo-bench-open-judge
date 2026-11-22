#!/usr/bin/env bash
#SBATCH -p <partition>
#SBATCH -N 1
#SBATCH --time=01:00:00
#SBATCH --account=<budget>
#SBATCH --job-name=wgo-flora-smoke
#SBATCH --output=%x-%j.log
set -uo pipefail

source "$W/env.sh"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

OUT="$RUNS_DIR/flora_smoke"
mkdir -p "$OUT"
cd "$REPO"

echo "##### GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "##### step 1: offline load + dtype introspection of Gated DeltaNet params"
"$PY_VLLM" - <<'PYEOF'
import os
import sys
try:
    from vllm import LLM
except Exception as e:
    print("IMPORT_FAIL", repr(e)); sys.exit(0)

try:
    llm = LLM(
        model=os.environ["MODELS_DIR"] + "/flora-captioner",
        dtype="auto",
        gpu_memory_utilization=0.90,
        max_model_len=16384,
        limit_mm_per_prompt={"image": 8},
        enforce_eager=True,
    )
    print("LOAD_OK")
except Exception as e:
    print("LOAD_FAIL", repr(e))
    sys.exit(0)

try:
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
except Exception as e:
    print("INTROSPECT_PATH_FAIL", repr(e))
    model = None

if model is not None:
    seen = set()
    for name, p in model.named_parameters():
        low = name.lower()
        if any(k in low for k in ("a_log", "dt_bias", "conv1d", "in_proj", "gdn", "mixer")):
            key = (p.dtype,)
            if key not in seen:
                seen.add(key)
            print("PARAM", name, p.dtype, tuple(p.shape))
    print("DTYPES_SEEN", seen)
PYEOF
echo "##### step 1 exit=$?"

echo "##### step 2: serve + label 20 homer segments (only if step 1 did not show LOAD_FAIL)"
nohup "$PY_VLLM" -m vllm.entrypoints.cli.main serve "$MODELS_DIR/flora-captioner" \
  --served-model-name nvidia/flora-captioner --host 127.0.0.1 --port 8010 \
  --max-model-len 16384 --limit-mm-per-prompt '{"image": 8}' \
  --gpu-memory-utilization 0.90 --dtype auto \
  > "$OUT/vllm-flora.log" 2>&1 &
VPID=$!
echo "SERVE|flora pid=$VPID"

ready=0
for _ in $(seq 180); do
  sleep 10
  if curl -sf http://127.0.0.1:8010/v1/models >/dev/null 2>&1; then
    ready=1
    echo "SERVE|ready flora"
    break
  fi
  kill -0 "$VPID" 2>/dev/null || { echo "SERVE|flora process died"; break; }
done

if [ "$ready" = "1" ]; then
  "$HARNESS_PY" -m wgo_eval.label --root "$WGO_ROOT" --out "$OUT/preds.jsonl" \
    --base-url http://127.0.0.1:8010/v1 --model nvidia/flora-captioner \
    --subset homer --limit 20
  echo "LABEL|exit=$?"
  cat "$OUT/preds.jsonl"
else
  echo "RUN|flora server did not become ready, see $OUT/vllm-flora.log"
fi

kill "$VPID" 2>/dev/null || true
pkill -f "[v]llm.entrypoints.cli.main serve" || true
echo "RUN|flora smoke done"
