#!/bin/bash
#SBATCH -J hlf-cps-state-gen
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
REPO="$WORKSPACE_ROOT/humanlm-main"
SOURCE_DIR="${SOURCE_DIR:-$REPO/data/cps_humanlm/v1/20p}"
CANDIDATE_DIR="${CANDIDATE_DIR:-$REPO/outputs/cps_state_first_raw_candidates/20p}"
QWEN_LOCAL_MODEL="${QWEN_LOCAL_MODEL:-$WORKSPACE_ROOT/humanlm_outputs/cps_qwen3_8b_sft_reviewed_smoke/merged_hf}"
CANDIDATES_PER_SAMPLE="${CANDIDATES_PER_SAMPLE:-3}"
LIMIT="${LIMIT:-0}"

export PATH="$ENV_DIR/bin:$PATH"
mkdir -p "$CANDIDATE_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Qwen policy: $QWEN_LOCAL_MODEL"
echo "Candidates per sample: $CANDIDATES_PER_SAMPLE"
nvidia-smi

for split in train val test; do
  python "$REPO/scripts/run_cps_dsv4pro_alignment.py" \
    --input "$SOURCE_DIR/$split.jsonl" \
    --output "$CANDIDATE_DIR/$split.jsonl" \
    --qwen-local-model "$QWEN_LOCAL_MODEL" \
    --candidates-per-sample "$CANDIDATES_PER_SAMPLE" \
    --limit "$LIMIT" \
    --generate-only \
    --append
done

# MARK_BASH #Don't delete this line!!!
