#!/bin/bash
# Run the PEFTLeak-aligned fixed-batch image reproduction protocol.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAGER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$DAGER_ROOT" || exit 1

PYTHON_BIN="${PYTHON:-python}"
CACHE="${CACHE:-./models_cache}"
DATASET="${DATASET:-cifar100}"
N_CLASSES="${N_CLASSES:-100}"
N_IMAGES="${N_IMAGES:-32}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PUBLIC_SPLIT_SIZE="${PUBLIC_SPLIT_SIZE:-$N_IMAGES}"
SPLIT_SEED="${SPLIT_SEED:-${RNG_SEED:-101}}"
ATTACK_INDICES_PATH="${ATTACK_INDICES_PATH:-./PEFTLeak-main/PEFTLeak-main/img_list.npy}"
PUBLIC_INDICES_PATH="${PUBLIC_INDICES_PATH:-$ATTACK_INDICES_PATH}"

"$PYTHON_BIN" attack_peftleak_image.py \
  --mode official_vit_adapter \
  --peftleak_profile official_cifar32 \
  --dataset "$DATASET" \
  --data_root "$CACHE" \
  --cache_dir "$CACHE" \
  --n_classes "$N_CLASSES" \
  --n_images "$N_IMAGES" \
  --batch_size "$BATCH_SIZE" \
  --public_split_size "$PUBLIC_SPLIT_SIZE" \
  --vit_config "${VIT_CONFIG:-cifar_small}" \
  --adapter_layers "${ADAPTER_LAYERS:-first_n}" \
  --attack_rounds "${ATTACK_ROUNDS:-1}" \
  --adapter_bottleneck_dim "${ADAPTER_BOTTLENECK_DIM:-64}" \
  --official_grouping "${OFFICIAL_GROUPING:-tag}" \
  --metrics "${METRICS:-mse,psnr,ssim,lpips,patch_recovery}" \
  --rng_seed "${RNG_SEED:-101}" \
  --sample_strategy indices_file \
  --split_seed "$SPLIT_SEED" \
  --attack_indices_path "$ATTACK_INDICES_PATH" \
  --public_indices_path "$PUBLIC_INDICES_PATH" \
  --fail_on_synthetic_fallback \
  "$@"
