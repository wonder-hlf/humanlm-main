#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HUMANLM_REPO="${HUMANLM_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SOURCE_DIR="${SOURCE_DIR:-$HUMANLM_REPO/data/cps_humanlm/v1/20p}"
ALIGNMENT_DIR="${ALIGNMENT_DIR:-$HUMANLM_REPO/outputs/cps_state_first_candidates_base_qwen/20p}"
OUTPUT_DIR="${OUTPUT_DIR:-$HUMANLM_REPO/data/cps_state_first_sft_base_qwen/20p}"
CANDIDATES_PER_SAMPLE="${CANDIDATES_PER_SAMPLE:-3}"
MINIMUM_SCORE="${MINIMUM_SCORE:-0.5}"
LIMIT="${LIMIT:-0}"
QWEN_LOCAL_MODEL="${QWEN_LOCAL_MODEL:-}"

for variable in DSV4PRO_API_KEY DSV4PRO_BASE_URL DSV4PRO_MODEL; do
  if [ -z "${!variable:-}" ]; then
    echo "Missing required environment variable: $variable" >&2
    exit 1
  fi
done
if [ -z "$QWEN_LOCAL_MODEL" ] && { [ -z "${QWEN_BASE_URL:-}" ] || [ -z "${QWEN_MODEL:-}" ]; }; then
  echo "Set QWEN_LOCAL_MODEL, or both QWEN_BASE_URL and QWEN_MODEL." >&2
  exit 1
fi

mkdir -p "$ALIGNMENT_DIR" "$OUTPUT_DIR"

for split in train val test; do
  echo "Generating and judging $split candidates"
  command=(
    python "$HUMANLM_REPO/scripts/run_cps_dsv4pro_alignment.py"
    --input "$SOURCE_DIR/$split.jsonl" \
    --output "$ALIGNMENT_DIR/$split.jsonl" \
    --candidates-per-sample "$CANDIDATES_PER_SAMPLE" \
    --limit "$LIMIT" \
    --append
  )
  if [ -n "$QWEN_LOCAL_MODEL" ]; then
    command+=(--qwen-local-model "$QWEN_LOCAL_MODEL")
  fi
  "${command[@]}"
done

python "$HUMANLM_REPO/scripts/prepare_cps_state_first_sft.py" \
  --source-dir "$SOURCE_DIR" \
  --alignment-dir "$ALIGNMENT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --minimum-score "$MINIMUM_SCORE"

echo "State-first SFT data written to: $OUTPUT_DIR"
