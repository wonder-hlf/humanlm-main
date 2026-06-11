#!/bin/bash
#SBATCH -J hlf-cps-state-data
#SBATCH -p A6000:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$HOME/workspace/hlf_workspace}"
ENV_DIR="$WORKSPACE_ROOT/.conda"
export PATH="$ENV_DIR/bin:$PATH"
export HUMANLM_REPO="$WORKSPACE_ROOT/humanlm-main"
export QWEN_LOCAL_MODEL="$WORKSPACE_ROOT/qwen_models/qwen/Qwen3-8B"
export CANDIDATES_PER_SAMPLE="${CANDIDATES_PER_SAMPLE:-3}"
export MINIMUM_SCORE="${MINIMUM_SCORE:-0.5}"

for variable in DSV4PRO_API_KEY DSV4PRO_BASE_URL DSV4PRO_MODEL; do
  if [ -z "${!variable:-}" ]; then
    echo "Missing required environment variable: $variable" >&2
    exit 1
  fi
done

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Qwen policy: $QWEN_LOCAL_MODEL"
nvidia-smi

cd "$WORKSPACE_ROOT"
bash humanlm-main/scripts/build_cps_state_first_sft.sh

# MARK_BASH #Don't delete this line!!!
