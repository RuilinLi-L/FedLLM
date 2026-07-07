#!/bin/bash
# Sweep PEFTLeak official-aligned image attack settings.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAGER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$DAGER_ROOT" || exit 1

DATASETS=( ${DATASETS:-cifar10 cifar100} )
BATCHES=( ${BATCHES:-1 2} )
LAYERS=( ${LAYERS:-all msa mlp} )
BOTTLENECKS=( ${BOTTLENECKS:-64} )
ROUNDS=( ${ROUNDS:-1} )

for dataset in "${DATASETS[@]}"; do
  for batch in "${BATCHES[@]}"; do
    for layers in "${LAYERS[@]}"; do
      for bottleneck in "${BOTTLENECKS[@]}"; do
        for rounds in "${ROUNDS[@]}"; do
          echo "========== official_vit_adapter dataset=${dataset} batch=${batch} layers=${layers} bottleneck=${bottleneck} rounds=${rounds} =========="
          python attack_peftleak_image.py \
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
            --fail_on_synthetic_fallback \
            "$@"
        done
      done
    done
  done
done
