#!/bin/bash
# Run post-gradient defense baselines on the official PEFTLeak CIFAR100 image runner.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAGER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$DAGER_ROOT" || exit 1

DEVICE="${DEVICE:-cuda:3}"
DATA_ROOT="${DATA_ROOT:-./data}"
IMG_LIST_PATH="${IMG_LIST_PATH:-PEFTLeak-main/PEFTLeak-main/img_list.npy}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/peftleak_official_image/baselines}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-42}"
PUBLIC_SPLIT="${PUBLIC_SPLIT:-test}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"

mkdir -p "$LOG_DIR"

run_variant() {
  local label="$1"
  shift
  local log_file="${LOG_DIR}/${label}.log"
  echo "[peftleak-official-image] running ${label}" >&2
  DEVICE="$DEVICE" \
  DATA_ROOT="$DATA_ROOT" \
  IMG_LIST_PATH="$IMG_LIST_PATH" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  BATCH_SIZE="$BATCH_SIZE" \
  SEED="$SEED" \
  PUBLIC_SPLIT="$PUBLIC_SPLIT" \
  bash scripts/peftleak_official_image_clean.sh "$@" 2>&1 | tee "$log_file"
}

run_variant "none" --defense none "$@"
run_variant "topk_0.1" --defense topk --defense_topk_ratio 0.1 "$@"
run_variant "topk_0.3" --defense topk --defense_topk_ratio 0.3 "$@"
run_variant "compression_8" --defense compression --defense_n_bits 8 "$@"
run_variant "compression_16" --defense compression --defense_n_bits 16 "$@"
run_variant "proj_only_0.5" --defense proj_only --defense_lrb_keep_ratio_sensitive 0.5 "$@"
run_variant "proj_only_0.75" --defense proj_only --defense_lrb_keep_ratio_sensitive 0.75 "$@"
run_variant "proj_only_0.9" --defense proj_only --defense_lrb_keep_ratio_sensitive 0.9 "$@"
run_variant "full_lrb_0.5" --defense full_lrb --defense_lrb_keep_ratio_sensitive 0.5 "$@"
