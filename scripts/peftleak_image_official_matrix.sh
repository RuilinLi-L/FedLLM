#!/bin/bash
# Sweep PEFTLeak official-aligned v1 proxy settings.
# This matrix is a mechanism smoke test, not the source-aligned paper path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAGER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$DAGER_ROOT" || exit 1

echo "[peftleak-image] running the proxy v1 matrix; do not label it as a PEFTLeak source reproduction." >&2

DATASETS=( ${DATASETS:-cifar10 cifar100} )
BATCHES=( ${BATCHES:-1 2} )
LAYERS=( ${LAYERS:-all msa mlp} )
BOTTLENECKS=( ${BOTTLENECKS:-64} )
ROUNDS=( ${ROUNDS:-1} )
PYTHON_BIN="${PYTHON:-python}"
SAMPLE_STRATEGY="${SAMPLE_STRATEGY:-first_n}"
SPLIT_SEED="${SPLIT_SEED:-${RNG_SEED:-101}}"
sample_args=(--sample_strategy "$SAMPLE_STRATEGY" --split_seed "$SPLIT_SEED")
if [[ -n "${ATTACK_INDICES_PATH:-}" ]]; then
  sample_args+=(--attack_indices_path "$ATTACK_INDICES_PATH")
fi
if [[ -n "${PUBLIC_INDICES_PATH:-}" ]]; then
  sample_args+=(--public_indices_path "$PUBLIC_INDICES_PATH")
fi

for dataset in "${DATASETS[@]}"; do
  for batch in "${BATCHES[@]}"; do
    for layers in "${LAYERS[@]}"; do
      for bottleneck in "${BOTTLENECKS[@]}"; do
        for rounds in "${ROUNDS[@]}"; do
          echo "========== official_vit_adapter dataset=${dataset} batch=${batch} layers=${layers} bottleneck=${bottleneck} rounds=${rounds} =========="
          "$PYTHON_BIN" attack_peftleak_image.py \
            --mode official_vit_adapter \
            --peftleak_profile official_cifar32 \
            --dataset "$dataset" \
            --data_root "${DATA_ROOT:-./models_cache}" \
            --cache_dir "${CACHE_DIR:-./models_cache}" \
            --n_classes "${N_CLASSES:-100}" \
            --n_images "${N_IMAGES:-8}" \
            --batch_size "$batch" \
            --public_split_size "${PUBLIC_SPLIT_SIZE:-64}" \
            --vit_config "${VIT_CONFIG:-cifar_small}" \
            --adapter_layers "$layers" \
            --adapter_bottleneck_dim "$bottleneck" \
            --attack_rounds "$rounds" \
            --official_grouping "${OFFICIAL_GROUPING:-tag}" \
            --metrics "${METRICS:-mse,psnr,ssim,lpips,patch_recovery}" \
            --rng_seed "${RNG_SEED:-101}" \
            "${sample_args[@]}" \
            --fail_on_synthetic_fallback \
            "$@"
        done
      done
    done
  done
done
