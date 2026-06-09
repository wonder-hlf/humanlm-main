#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$DEFAULT_WORKSPACE_ROOT}"
HUMANLM_REPO="${HUMANLM_REPO:-$WORKSPACE_ROOT/humanlm-main}"
MODEL_PATH="${MODEL_PATH:-$WORKSPACE_ROOT/qwen_models/qwen/Qwen3-8B}"
DATA_DIR="${DATA_DIR:-$HUMANLM_REPO/data/cps_team_sft/sft/r_no_tag/20p}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$HUMANLM_REPO/user_study/templates/qwen3_multi_role_template_think.jinja}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$WORKSPACE_ROOT/humanlm_outputs}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-cps_qwen3_8b_sft_smoke}"

require_path() {
  if [ ! -e "$1" ]; then
    echo "Missing required path: $1" >&2
    exit 1
  fi
}

require_path "$CHAT_TEMPLATE"
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

echo "HumanLM repo:  $HUMANLM_REPO"
echo "Chat template: $CHAT_TEMPLATE"
echo "Base model:    $MODEL_PATH"
echo "Dataset:       $DATA_DIR"
echo "Output dir:    $OUTPUT_DIR"
echo "GPUs:          $GPU_LIST"

cd "$HUMANLM_REPO"

export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export HF_HOME="$CACHE_ROOT"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/hub"
export TRANSFORMERS_CACHE="$CACHE_ROOT/hub"
export HF_DATASETS_CACHE="$CACHE_ROOT/datasets"
export XDG_CACHE_HOME="$CACHE_ROOT"
export VLLM_DOWNLOAD_DIR="$CACHE_ROOT/hub"
export VERL_CACHE_DIR="$CACHE_ROOT/verl-cache"
export VLLM_USE_V1=1
export WANDB_MODE=disabled

NUM_GPUS=$(awk -F',' '{print NF}' <<< "$GPU_LIST")

# Call the installed VERL trainer directly. The external HumanLM recipe repo is
# only needed later for latent-state GRPO, not for this response-only SFT.
python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/val.parquet" \
  +data.kwargs.multirole_chat_template_path="$CHAT_TEMPLATE" \
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
  optim.lr_warmup_steps_ratio=0.1 \
  ++optim.lr_scheduler=cosine \
  +trainer.val_before_train=false \
  trainer.total_epochs=1 \
  trainer.logger='["console"]' \
  trainer.project_name=humanlm_cps \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.save_freq=5 \
  trainer.test_freq=5 \
  trainer.n_gpus_per_node="$NUM_GPUS"
