# Copy to $W/env.sh and adjust. Sourced by every job in slurm/.
export REPO="$W/repo"                       # this repository
export WGO_ROOT="$W/data/wgo"               # output of `subtask-eval prepare`
export MODELS_DIR="$W/models"               # one directory per model, each with a .done marker
export RUNS_DIR="$W/runs"                   # run directories are created here
export PY="$W/.venv/bin/python"             # environment with this package installed (pip install -e .[test])
export VLLM="$W/.venv-vllm/bin/vllm"        # vLLM in its own environment
export JUDGE_DIR="$MODELS_DIR/Qwen3.8-27B"     # default judge (choice pending); run.sbatch accepts overrides
export JUDGE_NAME="Qwen/Qwen3.8-27B"
export EGOANT_DIR="$W/egoant"               # EgoANT checkout at b27ea59f5f (for check.sbatch)
export HF_HUB_OFFLINE=1
