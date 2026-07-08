#!/bin/bash
# Run one PEFTLeak official-aligned image attack experiment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAGER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$DAGER_ROOT" || exit 1

PYTHON_BIN="${PYTHON:-python}"
SPLIT_SEED="${SPLIT_SEED:-${RNG_SEED:-101}}"
SAMPLE_STRATEGY="${SAMPLE_STRATEGY:-first_n}"
sample_args=(--sample_strategy "$SAMPLE_STRATEGY" --split_seed "$SPLIT_SEED")
if [[ -n "${ATTACK_INDICES_PATH:-}" ]]; then
  sample_args+=(--attack_indices_path "$ATTACK_INDICES_PATH")
fi
if [[ -n "${PUBLIC_INDICES_PATH:-}" ]]; then
  sample_args+=(--public_indices_path "$PUBLIC_INDICES_PATH")
fi

"$PYTHON_BIN" attack_peftleak_image.py \
  --mode official_vit_adapter \
  --peftleak_profile official_cifar32 \
  --dataset "${DATASET:-cifar100}" \
  --data_root "${DATA_ROOT:-./models_cache}" \
  --cache_dir "${CACHE_DIR:-./models_cache}" \
  --n_classes "${N_CLASSES:-100}" \
  --n_images "${N_IMAGES:-8}" \
  --batch_size "${BATCH_SIZE:-2}" \
  --public_split_size "${PUBLIC_SPLIT_SIZE:-64}" \
  --vit_config "${VIT_CONFIG:-cifar_small}" \
  --adapter_layers "${ADAPTER_LAYERS:-all}" \
  --attack_rounds "${ATTACK_ROUNDS:-1}" \
  --adapter_bottleneck_dim "${ADAPTER_BOTTLENECK_DIM:-64}" \
  --official_grouping "${OFFICIAL_GROUPING:-tag}" \
  --metrics "${METRICS:-mse,psnr,ssim,lpips,patch_recovery}" \
  --rng_seed "${RNG_SEED:-101}" \
  "${sample_args[@]}" \
  --fail_on_synthetic_fallback \
  "$@"
