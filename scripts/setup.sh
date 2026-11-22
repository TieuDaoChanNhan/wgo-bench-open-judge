#!/usr/bin/env bash
# One-time setup of a GPU VM for the benchmark (Ubuntu 22.04 + NVIDIA driver). Idempotent.
# Expects the repo at $W/repo. Builds two venvs (vLLM server, harness), starts the judge
# model download in the background and writes WGO-Bench to $W/data/wgo.
#   W=/data/bench bash scripts/setup.sh
set -euo pipefail
W="${W:-/data/bench}"
JUDGE_REPO="${JUDGE_REPO:-Qwen/Qwen3-32B-AWQ}"
export PIP_DISABLE_PIP_VERSION_CHECK=1 DEBIAN_FRONTEND=noninteractive

sudo -E apt-get update -qq
# python3-dev + build-essential: Triton (used by vLLM) compiles a small C helper at startup.
sudo -E apt-get install -y -qq ffmpeg python3-venv python3-pip python3-dev build-essential git tmux
mkdir -p "$W"/{models,data,runs}

[ -x "$W/.venv-vllm/bin/python" ] || python3 -m venv "$W/.venv-vllm"
"$W/.venv-vllm/bin/pip" install -q -U pip uv huggingface_hub

JUDGE_DIR="$W/models/$(basename "$JUDGE_REPO")"
if [ ! -f "$JUDGE_DIR/.done" ] && ! pgrep -f "download $JUDGE_REPO" >/dev/null; then
  HF="$W/.venv-vllm/bin/hf"; [ -x "$HF" ] || HF="$W/.venv-vllm/bin/huggingface-cli"
  nohup env HF_HUB_OFFLINE=0 bash -c "$HF download $JUDGE_REPO --local-dir $JUDGE_DIR && touch $JUDGE_DIR/.done" \
    > "$W/runs/judge_download.log" 2>&1 &
  echo "SETUP|judge download started: $JUDGE_REPO"
fi

[ -x "$W/.venv/bin/python" ] || python3 -m venv "$W/.venv"
"$W/.venv/bin/pip" install -q -U pip
"$W/.venv/bin/pip" install -q -r "$W/repo/requirements.txt"

"$W/.venv-vllm/bin/uv" pip install -q --python "$W/.venv-vllm/bin/python" vllm --torch-backend=auto
"$W/.venv-vllm/bin/python" -c "import vllm; print('SETUP|vllm', vllm.__version__)"

if [ ! -f "$W/data/wgo/manifest.json" ]; then
  (cd "$W/repo" && HF_HUB_OFFLINE=0 HF_HOME="$W/hf" "$W/.venv/bin/python" -m wgo_eval.wgo_prepare --root "$W/data/wgo")
fi
cat "$W/data/wgo/manifest.json"
echo "SETUP|ok"
