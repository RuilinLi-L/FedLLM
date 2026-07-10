#!/bin/bash
# Run one source-aligned PEFTLeak CIFAR100 image-adapter attack/defense point.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAGER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$DAGER_ROOT" || exit 1

PYTHON_BIN="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:3}"
DATA_ROOT="${DATA_ROOT:-./data}"
IMG_LIST_PATH="${IMG_LIST_PATH:-PEFTLeak-main/PEFTLeak-main/img_list.npy}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/peftleak_official_image}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-42}"

download_args=()
if [[ "${DOWNLOAD_CIFAR100:-0}" == "1" ]]; then
  download_args+=(--download)
fi

"$PYTHON_BIN" PEFTLeak-main/PEFTLeak-main/official_image_runner.py \
  --dataset cifar100 \
  --data_root "$DATA_ROOT" \
  --img_list_path "$IMG_LIST_PATH" \
  --public_split "${PUBLIC_SPLIT:-test}" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --defense none \
  --seed "$SEED" \
  --output_dir "$OUTPUT_DIR" \
  "${download_args[@]}" \
  "$@"
