# Example settings for the jobs in slurm/. Usage:
#   export W=/path/to/workdir          # holds env.sh, data, models, runs, logs
#   cp slurm/env.example.sh $W/env.sh  # then adjust the paths below
#   sbatch -p <partition> -A <account> -o $W/logs/%x-%j.log slurm/<job>.sbatch
# Load your cluster's modules here as well. git must be on PATH on the compute nodes:
# `provenance record` reads the commit with it and fails without it. On some clusters it is
# only on login nodes unless a module is loaded (e.g. `module load git`).
export REPO="$W/repo"                       # this repository
export WGO_ROOT="$W/data/wgo"               # output of `subtask-eval prepare`
export MODELS_DIR="$W/models"               # one directory per model, each with a .done marker
export RUNS_DIR="$W/runs"                   # run directories are created here
export PY="$W/.venv/bin/python"             # environment with this package installed (pip install -e .[test])
export VLLM="$W/.venv-vllm/bin/vllm"        # vLLM in its own environment
export JUDGE_DIR="$MODELS_DIR/Qwen3-32B-AWQ"   # default judge (see README, Label judge)
export JUDGE_NAME="Qwen/Qwen3-32B-AWQ"
export EGOANT_DIR="$W/egoant"               # EgoANT checkout at b27ea59f5f (for check.sbatch)
export HF_HUB_OFFLINE=1
