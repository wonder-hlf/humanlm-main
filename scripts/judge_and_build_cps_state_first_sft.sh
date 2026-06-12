#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${HUMANLM_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SOURCE_DIR="${SOURCE_DIR:-$REPO/data/cps_humanlm/v1/20p}"
CANDIDATE_DIR="${CANDIDATE_DIR:-$REPO/outputs/cps_state_first_raw_candidates_base_qwen/20p}"
ALIGNMENT_DIR="${ALIGNMENT_DIR:-$REPO/outputs/cps_state_first_candidates_base_qwen/20p}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/data/cps_state_first_sft_base_qwen/20p}"
MINIMUM_SCORE="${MINIMUM_SCORE:-0.5}"
SPLITS="${SPLITS:-train val test}"

for variable in DSV4PRO_API_KEY DSV4PRO_BASE_URL DSV4PRO_MODEL; do
  if [ -z "${!variable:-}" ]; then
    echo "Missing required environment variable: $variable" >&2
    exit 1
  fi
done

mkdir -p "$ALIGNMENT_DIR" "$OUTPUT_DIR"
for split in $SPLITS; do
  python "$REPO/scripts/judge_cps_state_candidates.py" \
    --source "$SOURCE_DIR/$split.jsonl" \
    --candidates "$CANDIDATE_DIR/$split.jsonl" \
    --output "$ALIGNMENT_DIR/$split.jsonl" \
    --append
done

if [ "$SPLITS" = "train val test" ]; then
  python "$REPO/scripts/prepare_cps_state_first_sft.py" \
    --source-dir "$SOURCE_DIR" \
    --alignment-dir "$ALIGNMENT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --minimum-score "$MINIMUM_SCORE"
fi
