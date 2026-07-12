#!/bin/bash
# Train one reportable CIFAR-100 ViT-B/16 Adapter utility point.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$ROOT" || exit 1

PYTHON_BIN="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:3}"
DATA_ROOT="${DATA_ROOT:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/peftleak_official_image/utility/single}"
SEED="${SEED:-101}"

download_args=()
if [[ "${DOWNLOAD_CIFAR100:-0}" == "1" ]]; then
  download_args+=(--download)
fi

amp_args=(--amp)
if [[ "${AMP:-1}" == "0" ]]; then
  amp_args=(--no-amp)
fi

"$PYTHON_BIN" train_peftleak_image_utility.py \
  --dataset cifar100 \
  --profile formal_vit_b16 \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --rng_seed "$SEED" \
  --split_seed "${SPLIT_SEED:-42}" \
  --adapter_bottleneck_dim "${ADAPTER_BOTTLENECK_DIM:-64}" \
  --num_epochs "${NUM_EPOCHS:-20}" \
  --batch_size "${BATCH_SIZE:-128}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-256}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --lr_adapter "${LR_ADAPTER:-0.001}" \
  --lr_head "${LR_HEAD:-0.001}" \
  --weight_decay "${WEIGHT_DECAY:-0.01}" \
  --warmup_epochs "${WARMUP_EPOCHS:-1}" \
  "${amp_args[@]}" \
  "${download_args[@]}" \
  "$@"
