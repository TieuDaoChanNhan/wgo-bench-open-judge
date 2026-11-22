#!/usr/bin/env bash
# Start an OpenAI-compatible vLLM server on 127.0.0.1 and wait until it answers.
#   bash scripts/serve.sh <model id or dir> <port> <vl|vlseg|text> [served name]
# vl: labeling (5 frames per request); vlseg: segmentation (all contact sheets of an episode in one request)
#   bash scripts/serve.sh stop
# One GPU holds one server at a time here, so callers stop the labeler before starting the judge.
set -euo pipefail
W="${W:-/data/bench}"

if [ "${1:-}" = "stop" ]; then
  pkill -f "[v]llm serve" || true
  for _ in $(seq 30); do pgrep -f "[v]llm serve" >/dev/null || break; sleep 2; done
  echo "SERVE|stopped"
  exit 0
fi

MODEL="$1"; PORT="$2"; KIND="$3"; NAME="${4:-$1}"
LOG="$W/runs/vllm-$PORT.log"
if [ "$KIND" = "vl" ]; then
  EXTRA=(--max-model-len 16384 --limit-mm-per-prompt '{"image": 8}' --max-num-seqs 16)
elif [ "$KIND" = "vlseg" ]; then
  EXTRA=(--max-model-len 65536 --limit-mm-per-prompt '{"image": 64}' --max-num-seqs 8)
else
  EXTRA=(--max-model-len 4096 --max-num-seqs 64)
fi
nohup "$W/.venv-vllm/bin/vllm" serve "$MODEL" --served-model-name "$NAME" --host 127.0.0.1 --port "$PORT" \
  --gpu-memory-utilization 0.90 --dtype auto "${EXTRA[@]}" > "$LOG" 2>&1 &

for _ in $(seq 180); do
  sleep 10
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
    echo "SERVE|ok $NAME on :$PORT"
    exit 0
  fi
  pgrep -f "[v]llm serve" >/dev/null || break
done
echo "SERVE|fail $NAME"
tail -n 40 "$LOG"
exit 1
