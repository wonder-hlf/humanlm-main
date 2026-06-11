#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$DEFAULT_WORKSPACE_ROOT}"
HUMANLM_REPO="${HUMANLM_REPO:-$WORKSPACE_ROOT/humanlm-main}"
MODEL_PATH="${MODEL_PATH:-$WORKSPACE_ROOT/humanlm_outputs/cps_qwen3_8b_sft_reviewed_smoke/merged_hf}"
DATA_DIR="${DATA_DIR:-$HUMANLM_REPO/data/cps_state_first_sft/20p}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$HUMANLM_REPO/user_study/templates/qwen3_multi_role_template_think.jinja}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$WORKSPACE_ROOT/humanlm_outputs}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-cps_qwen3_8b_state_first_sft_smoke}"

for path in "$MODEL_PATH/config.json" "$DATA_DIR/train.parquet" "$DATA_DIR/val.parquet" "$CHAT_TEMPLATE"; do
  if [ ! -e "$path" ]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
done

OUTPUT_DIR="$OUTPUT_ROOT/$EXPERIMENT_NAME"
CACHE_ROOT="$WORKSPACE_ROOT/hf_cache"
mkdir -p "$OUTPUT_DIR" "$CACHE_ROOT"

echo "Training-two model: $MODEL_PATH"
echo "State-first data:   $DATA_DIR"
echo "Output dir:         $OUTPUT_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export HF_HOME="$CACHE_ROOT"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/hub"
export TRANSFORMERS_CACHE="$CACHE_ROOT/hub"
export HF_DATASETS_CACHE="$CACHE_ROOT/datasets"
export XDG_CACHE_HOME="$CACHE_ROOT"
export WANDB_MODE=disabled

NUM_GPUS=$(awk -F',' '{print NF}' <<< "$GPU_LIST")

cd "$HUMANLM_REPO"
python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/val.parquet" \
  +data.kwargs.multirole_chat_template_path="$CHAT_TEMPLATE" \
  +data.apply_chat_template_kwargs.enable_thinking=false \
  data.multiturn.enable=false \
  data.max_length=8196 \
  +data.dataset=cps_state_first \
  data.truncation=right \
  data.train_batch_size=16 \
  data.micro_batch_size_per_gpu=1 \
  data.prompt_key=prompt \
  data.response_key=generation \
  model.partial_pretrain="$MODEL_PATH" \
  model.fsdp_config.model_dtype=bfloat16 \
  model.enable_gradient_checkpointing=true \
  optim.lr=5e-7 \
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
