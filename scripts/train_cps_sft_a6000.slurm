#!/bin/bash
#SBATCH -J hlf-cps-sft-smoke
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

echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "Workspace:     $WORKSPACE_ROOT"
echo "Python:        $(which python)"
echo "Python version: $(python --version)"

nvidia-smi
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('gpu_count:', torch.cuda.device_count())"
python -c "import verl; import verl.trainer.fsdp_sft_trainer; print('verl:', verl.__file__)"

cd "$WORKSPACE_ROOT"
bash humanlm-main/scripts/run_chuangzhi_cps_sft.sh

# MARK_BASH #Don't delete this line!!!
