#!/bin/bash
#SBATCH -J hlf-cps-state-sft
#SBATCH -p A6000:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$HOME/workspace/hlf_workspace}"
ENV_DIR="$WORKSPACE_ROOT/.conda"

export PATH="$ENV_DIR/bin:$PATH"
export WORKSPACE_ROOT
export GPU_LIST="0,1,2,3"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Python: $(which python)"
nvidia-smi

cd "$WORKSPACE_ROOT"
bash humanlm-main/scripts/run_chuangzhi_cps_state_first_sft.sh

# MARK_BASH #Don't delete this line!!!
