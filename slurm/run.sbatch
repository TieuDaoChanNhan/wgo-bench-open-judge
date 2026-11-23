#!/usr/bin/env bash
#SBATCH -p <partition>
#SBATCH -N 1
#SBATCH --time=04:00:00
#SBATCH --account=<budget>
#SBATCH --job-name=subtask-eval-run
#SBATCH --output=%x-%j.log
# One candidate end to end: serve it, annotate, match, judge, score.
#   CONFIG=configs/cs-qwen36-27b.json [LIMIT=2] sbatch slurm/run.sbatch
# The annotator's server uses GPUs 0..gpus-1 (tensor parallel), the judge the next GPU.
# JUDGE_DIR / JUDGE_NAME given on the command line override env.sh.
set -uo pipefail
_JD="${JUDGE_DIR:-}"; _JN="${JUDGE_NAME:-}"
: "${W:?set W to your work directory (it holds env.sh)}"
source "$W/env.sh"
JUDGE_DIR="${_JD:-$JUDGE_DIR}"; JUDGE_NAME="${_JN:-$JUDGE_NAME}"; export JUDGE_DIR JUDGE_NAME
cd "$REPO" || exit 1
: "${CONFIG:?set CONFIG to a file in configs/}"
LIMIT="${LIMIT:-0}"

cfg() {  # cfg <dotted key>: one value from the config file
  "$PY" -c "import functools, json, sys; print(functools.reduce(lambda d, k: d[k], sys.argv[2].split('.'), json.load(open(sys.argv[1]))))" "$CONFIG" "$1"
}
NAME="$(cfg name)"; ANNOTATOR="$(cfg annotator)"; SERVED="$(cfg model)"; GPUS="$(cfg gpus)"
[ "$LIMIT" != "0" ] && NAME="$NAME-smoke$LIMIT"
OUT="$RUNS_DIR/$NAME"; mkdir -p "$OUT"
echo "RUN|$NAME annotator=$ANNOTATOR model=$SERVED gpus=$GPUS limit=$LIMIT judge=$JUDGE_NAME"
"$PY" -m subtask_eval provenance record --out "$OUT" --repo "$REPO" --note "run.sbatch $CONFIG" \
  --models "$MODELS_DIR/$(cfg serve.model)" "$JUDGE_DIR"

PIDS=()
trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT

serve() {  # serve <gpus csv> <model dir> <served name> <port> <log> [extra vllm args...]
  local gpus="$1" dir="$2" served="$3" port="$4" log="$5"; shift 5
  CUDA_VISIBLE_DEVICES="$gpus" nohup "$VLLM" serve "$dir" --served-model-name "$served" --host 127.0.0.1 \
    --port "$port" --gpu-memory-utilization 0.90 --dtype auto "$@" > "$log" 2>&1 &
  PIDS+=($!)
}
wait_ready() {  # wait_ready <port> <log>
  for _ in $(seq 240); do
    sleep 10
    curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1 && { echo "SERVE|ready :$1"; return 0; }
  done
  echo "SERVE|FAILED :$1"; tail -n 40 "$2"; exit 1
}

ANN_GPUS="$(seq -s, 0 $((GPUS - 1)))"
serve "$ANN_GPUS" "$MODELS_DIR/$(cfg serve.model)" "$SERVED" 8000 "$OUT/vllm-annotator.log" \
  --tensor-parallel-size "$GPUS" --max-model-len "$(cfg serve.max_model_len)" \
  --limit-mm-per-prompt "{\"image\": $(cfg serve.limit_images)}" --max-num-seqs "$(cfg serve.max_num_seqs)"
serve "$GPUS" "$JUDGE_DIR" "$JUDGE_NAME" 8001 "$OUT/vllm-judge.log" --max-model-len 4096 --max-num-seqs 64
wait_ready 8000 "$OUT/vllm-annotator.log"
wait_ready 8001 "$OUT/vllm-judge.log"

set -e
"$PY" -m subtask_eval annotate "$ANNOTATOR" --config "$CONFIG" --root "$WGO_ROOT" --out "$OUT" \
  --base-url http://127.0.0.1:8000/v1 --limit "$LIMIT" --name "$NAME"
"$PY" -m subtask_eval score --root "$WGO_ROOT" --run "$OUT" --write-pairs > /dev/null
"$PY" -m subtask_eval judge --run "$OUT" --base-url http://127.0.0.1:8001/v1 --model "$JUDGE_NAME" \
  --cache "$RUNS_DIR/judge_cache.jsonl"
for sub in all homer galaxea robointer_droid; do
  S=(); [ "$sub" != "all" ] && S=(--subset "$sub")
  "$PY" -m subtask_eval score --root "$WGO_ROOT" --run "$OUT" "${S[@]}" | tee "$OUT/score_$sub.txt"
done
echo "RUN|ok $NAME -> $OUT"
