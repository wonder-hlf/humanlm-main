#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$DEFAULT_WORKSPACE_ROOT}"
TRAIN_REPO="${TRAIN_REPO:-$WORKSPACE_ROOT/verl-recipe-humanlm}"
MODEL_PATH="${MODEL_PATH:-$WORKSPACE_ROOT/qwen_models/qwen/Qwen3-8B}"
DATA_DIR="${DATA_DIR:-$WORKSPACE_ROOT/humanlm-main/data/cps_team_sft/sft/r_no_tag/20p}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$WORKSPACE_ROOT/humanlm_outputs}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-cps_qwen3_8b_sft_smoke}"

require_path() {
  if [ ! -e "$1" ]; then
    echo "Missing required path: $1" >&2
    exit 1
  fi
}

require_path "$TRAIN_REPO/humanlm/train_sft_humanlm.sh"
require_path "$TRAIN_REPO/humanlm/chat_templates/qwen3_multi_role_template_think.jinja"
require_path "$MODEL_PATH/config.json"
require_path "$DATA_DIR/train.parquet"
require_path "$DATA_DIR/val.parquet"

if ! python3 -c "import verl; import verl.trainer.fsdp_sft_trainer" >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Missing VERL core trainer.

The verl-recipe-humanlm repository contains HumanLM recipes, but its pinned
commit does not contain the verl core Python package. Install a compatible
VERL environment before training, then verify with:

  python3 -c "import verl; import verl.trainer.fsdp_sft_trainer; print(verl.__file__)"
EOF
  exit 1
fi

OUTPUT_DIR="$OUTPUT_ROOT/$EXPERIMENT_NAME"
CACHE_ROOT="$WORKSPACE_ROOT/hf_cache"
mkdir -p "$OUTPUT_DIR" "$CACHE_ROOT"

echo "Training repo: $TRAIN_REPO"
echo "Base model:    $MODEL_PATH"
echo "Dataset:       $DATA_DIR"
echo "Output dir:    $OUTPUT_DIR"
echo "GPUs:          $GPU_LIST"

cd "$TRAIN_REPO"

export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export HF_HOME="$CACHE_ROOT"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/hub"
export TRANSFORMERS_CACHE="$CACHE_ROOT/hub"
export HF_DATASETS_CACHE="$CACHE_ROOT/datasets"
export XDG_CACHE_HOME="$CACHE_ROOT"
export VLLM_DOWNLOAD_DIR="$CACHE_ROOT/hub"
export VERL_CACHE_DIR="$CACHE_ROOT/verl-cache"
export VLLM_USE_V1=1

NUM_GPUS=$(awk -F',' '{print NF}' <<< "$GPU_LIST")

# Call the VERL trainer directly. The pinned upstream shell script contains
# cluster-specific //llm_twin paths and an outdated chat-template path.
python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/val.parquet" \
  +data.kwargs.multirole_chat_template_path="$TRAIN_REPO/humanlm/chat_templates/qwen3_multi_role_template_think.jinja" \
  +data.apply_chat_template_kwargs.enable_thinking=false \
  data.multiturn.enable=false \
  data.max_length=8196 \
  +data.dataset=cps_team \
  data.truncation=right \
  data.train_batch_size=16 \
  data.micro_batch_size_per_gpu=1 \
  data.prompt_key=prompt \
  data.response_key=generation \
  model.partial_pretrain="$MODEL_PATH" \
  model.fsdp_config.model_dtype=bfloat16 \
  model.enable_gradient_checkpointing=true \
  optim.lr=1e-6 \
  optim.warmup_steps_ratio=0.1 \
  optim.lr_scheduler=cosine \
  +trainer.val_before_train=true \
  trainer.total_epochs=1 \
  trainer.project_name=humanlm_cps \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.save_freq=20 \
  trainer.test_freq=10 \
  trainer.n_gpus_per_node="$NUM_GPUS"
