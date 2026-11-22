#!/usr/bin/env bash
#SBATCH -p <partition>
#SBATCH -N 1
#SBATCH --time=01:30:00
#SBATCH --account=<budget>
#SBATCH --job-name=subtask-eval-judges
#SBATCH --output=%x-%j.log
# Compare label judges. Serves the judges in JUDGES (one GPU each), then:
#   1. judges the 79 EgoANT pairs that carry gemini-3.5-flash verdicts, and compares each judge
#      with Gemini and with each other;
#   2. for every run directory in RUNS, judges its matched pairs with each judge, scores the run
#      per judge (score_<subset>_<tag>.json) and compares the judges' verdicts.
#   JUDGES="Qwen3-32B-AWQ:Qwen/Qwen3-32B-AWQ Qwen3.8-27B:Qwen/Qwen3.8-27B" \
#   RUNS="$RUNS_DIR/cs-qwen36-27b" EGOANT_JSON=<file in $EGOANT_DIR> sbatch slurm/judges.sbatch
# Each JUDGES entry is <dir under $MODELS_DIR>:<served name>; the tag is the dir name.
# RUNS may be empty (step 2 skipped). TASK_REPO and JUDGE_CACHE override the checkout and cache file.
set -uo pipefail
source "$W/env.sh"
REPO="${TASK_REPO:-$REPO}"                       # a separate checkout, to run beside another job
JUDGE_CACHE="${JUDGE_CACHE:-$RUNS_DIR/judge_cache.jsonl}"
cd "$REPO" || exit 1
: "${JUDGES:?}" "${EGOANT_JSON:?}"
RUNS="${RUNS:-}"
AG="$RUNS_DIR/judge-agree"; mkdir -p "$AG"

PIDS=(); trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT
declare -A PORT NAME
gpu=0
for j in $JUDGES; do
  tag="${j%%:*}"; NAME[$tag]="${j#*:}"; PORT[$tag]=$((8001 + gpu))
  CUDA_VISIBLE_DEVICES=$gpu nohup "$VLLM" serve "$MODELS_DIR/$tag" --served-model-name "${NAME[$tag]}" \
    --host 127.0.0.1 --port "${PORT[$tag]}" --gpu-memory-utilization 0.90 --dtype auto \
    --max-model-len 4096 --max-num-seqs 64 > "$AG/vllm-$tag.log" 2>&1 &
  PIDS+=($!); gpu=$((gpu + 1))
  cat "$MODELS_DIR/$tag/HF_REVISION.txt" 2>/dev/null || true
done
for tag in "${!PORT[@]}"; do
  for _ in $(seq 180); do sleep 10; curl -sf "http://127.0.0.1:${PORT[$tag]}/v1/models" >/dev/null && break; done
  curl -sf "http://127.0.0.1:${PORT[$tag]}/v1/models" >/dev/null || { echo "SERVE|FAILED $tag"; tail -n 40 "$AG/vllm-$tag.log"; exit 1; }
  echo "SERVE|ready $tag"
done
set -e

judge() {  # judge <tag> <run dir> <out file>
  "$PY" -m subtask_eval judge --run "$2" --out "$3" --base-url "http://127.0.0.1:${PORT[$1]}/v1" \
    --model "${NAME[$1]}" --cache "$JUDGE_CACHE"
}
compare_all() {  # compare_all <dir> <file prefix>; pairwise over the judges' verdict files
  local tags=("${!PORT[@]}")
  for ((i = 0; i < ${#tags[@]}; i++)); do for ((k = i + 1; k < ${#tags[@]}; k++)); do
    a="${tags[$i]}"; b="${tags[$k]}"
    "$PY" -m subtask_eval agree compare --a "$1/$2$a.jsonl" --b "$1/$2$b.jsonl" \
      --disagreements "$1/disagree_${a}_vs_${b}.jsonl" --out "$1/agree_${a}_vs_${b}.json"
  done; done
}

echo "##### 1. EgoANT pairs with gemini-3.5-flash verdicts"
"$PY" -m subtask_eval agree extract-egoant --egoant-json "$EGOANT_DIR/$EGOANT_JSON" --root "$WGO_ROOT" --out "$AG"
for tag in "${!PORT[@]}"; do
  judge "$tag" "$AG" "$AG/judged_$tag.jsonl"
  "$PY" -m subtask_eval agree compare --a "$AG/pairs.jsonl" --b "$AG/judged_$tag.jsonl" \
    --disagreements "$AG/disagree_gemini_vs_$tag.jsonl" --out "$AG/agree_gemini_vs_$tag.json"
done
compare_all "$AG" judged_

for run in $RUNS; do
  echo "##### 2. $run"
  [ -f "$run/pairs.jsonl" ] || "$PY" -m subtask_eval score --root "$WGO_ROOT" --run "$run" --write-pairs > /dev/null
  for tag in "${!PORT[@]}"; do
    judge "$tag" "$run" "$run/judged_$tag.jsonl"
    for sub in all homer; do
      S=(); [ "$sub" != "all" ] && S=(--subset "$sub")
      "$PY" -m subtask_eval score --root "$WGO_ROOT" --run "$run" "${S[@]}" --judged "$run/judged_$tag.jsonl" \
        --tag "$tag" | tail -n 4
    done
  done
  compare_all "$run" judged_
done
echo "RUN|judges ok"
