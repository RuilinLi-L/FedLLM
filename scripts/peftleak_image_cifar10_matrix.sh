#!/bin/bash
# Run the PEFTLeak image-side CIFAR10 official-aligned privacy matrix.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAGER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$DAGER_ROOT" || exit 1

PYTHON_BIN="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
CACHE="${CACHE:-./models_cache}"
OUT="${OUT:-./outputs/peftleak_image_cifar10}"
DATASET="${DATASET:-cifar10}"
N_CLASSES="${N_CLASSES:-10}"
N_IMAGES="${N_IMAGES:-32}"
BATCH_SIZE="${BATCH_SIZE:-2}"
PUBLIC_SPLIT_SIZE="${PUBLIC_SPLIT_SIZE:-64}"
VIT_CONFIG="${VIT_CONFIG:-cifar_small}"
ADAPTER_LAYERS="${ADAPTER_LAYERS:-first_n}"
ADAPTER_BOTTLENECK_DIM="${ADAPTER_BOTTLENECK_DIM:-64}"
ATTACK_ROUNDS="${ATTACK_ROUNDS:-1}"
OFFICIAL_GROUPING="${OFFICIAL_GROUPING:-tag}"
METRICS="${METRICS:-mse,psnr,ssim,lpips,patch_recovery}"
SAMPLE_STRATEGY="${SAMPLE_STRATEGY:-seeded_shuffle}"
ATTACK_INDICES_PATH="${ATTACK_INDICES_PATH:-}"
PUBLIC_INDICES_PATH="${PUBLIC_INDICES_PATH:-}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_ABLATIONS="${RUN_ABLATIONS:-0}"

read -r -a SEED_LIST <<< "${SEEDS:-101 202 303 404 505}"

mkdir -p "$CACHE" "$OUT/privacy" "$OUT/proxy_utility" "$OUT/tables"

if [[ "${DOWNLOAD_CIFAR10:-0}" == "1" ]]; then
  CACHE="$CACHE" "$PYTHON_BIN" - <<'PY'
import os
from torchvision.datasets import CIFAR10

root = os.environ.get("CACHE", "./models_cache")
CIFAR10(root=root, train=True, download=True)
CIFAR10(root=root, train=False, download=True)
print(f"CIFAR10 ready under {root}")
PY
fi

if [[ "$DATASET" == "cifar10" && ! -d "$CACHE/cifar-10-batches-py" ]]; then
  echo "CIFAR10 not found under $CACHE/cifar-10-batches-py."
  echo "Run with DOWNLOAD_CIFAR10=1, or download CIFAR10 before this matrix."
  exit 1
fi

run_img_privacy() {
  local tag="$1"
  local seed="$2"
  shift 2
  local sample_args=(--sample_strategy "$SAMPLE_STRATEGY" --split_seed "$seed")
  if [[ -n "$ATTACK_INDICES_PATH" ]]; then
    sample_args+=(--attack_indices_path "$ATTACK_INDICES_PATH")
  fi
  if [[ -n "$PUBLIC_INDICES_PATH" ]]; then
    sample_args+=(--public_indices_path "$PUBLIC_INDICES_PATH")
  fi
  echo "========== ${tag} seed=${seed} =========="
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
    --vit_config "$VIT_CONFIG" \
    --adapter_layers "$ADAPTER_LAYERS" \
    --adapter_bottleneck_dim "$ADAPTER_BOTTLENECK_DIM" \
    --attack_rounds "$ATTACK_ROUNDS" \
    --official_grouping "$OFFICIAL_GROUPING" \
    --metrics "$METRICS" \
    --device "$DEVICE" \
    --rng_seed "$seed" \
    "${sample_args[@]}" \
    --fail_on_synthetic_fallback \
    "$@" \
    | tee "$OUT/privacy/${tag}_seed${seed}.log"
}

if [[ "$RUN_SMOKE" == "1" ]]; then
  echo "========== smoke none seed=101 =========="
  "$PYTHON_BIN" attack_peftleak_image.py \
    --mode official_vit_adapter \
    --peftleak_profile official_cifar32 \
    --dataset "$DATASET" \
    --data_root "$CACHE" \
    --cache_dir "$CACHE" \
    --n_classes "$N_CLASSES" \
    --n_images 1 \
    --batch_size 1 \
    --public_split_size 4 \
    --vit_config "$VIT_CONFIG" \
    --adapter_layers first_n \
    --adapter_bottleneck_dim "$ADAPTER_BOTTLENECK_DIM" \
    --official_grouping "$OFFICIAL_GROUPING" \
    --metrics "$METRICS" \
    --defense none \
    --device "$DEVICE" \
    --rng_seed 101 \
    --sample_strategy first_n \
    --split_seed 101 \
    --fail_on_synthetic_fallback \
    | tee "$OUT/privacy/smoke_none_seed101.log"
fi

for seed in "${SEED_LIST[@]}"; do
  run_img_privacy "none" "$seed" --defense none
done

for p in 0.05 0.1 0.3 0.5; do
  for seed in "${SEED_LIST[@]}"; do
    run_img_privacy "topk_${p}" "$seed" --defense topk --defense_topk_ratio "$p"
  done
done

for b in 4 8 16; do
  for seed in "${SEED_LIST[@]}"; do
    run_img_privacy "compression_${b}" "$seed" --defense compression --defense_n_bits "$b"
  done
done

for k in 0.5 0.65 0.75 0.9; do
  for seed in "${SEED_LIST[@]}"; do
    run_img_privacy "proj_only_${k}" "$seed" \
      --defense lrb \
      --defense_lrb_preset proj_only \
      --defense_lrb_keep_ratio_sensitive "$k"
  done
done

for preset in proj_clip full_lrb; do
  for k in 0.5 0.65 0.75; do
    for seed in "${SEED_LIST[@]}"; do
      run_img_privacy "${preset}_${k}" "$seed" \
        --defense lrb \
        --defense_lrb_preset "$preset" \
        --defense_lrb_keep_ratio_sensitive "$k"
    done
  done
done

for sigma in 1e-5 1e-4 5e-4 1e-3; do
  for seed in "${SEED_LIST[@]}"; do
    run_img_privacy "noise_${sigma}" "$seed" --defense noise --defense_noise "$sigma"
    run_img_privacy "dpsgd_${sigma}" "$seed" \
      --defense dpsgd \
      --defense_noise "$sigma" \
      --defense_clip_norm 1.0
  done
done

for seed in "${SEED_LIST[@]}"; do
  run_img_privacy "soteria_60" "$seed" --defense soteria --defense_soteria_pruning_rate 60.0
  run_img_privacy "mixup_1.0" "$seed" --defense mixup --defense_mixup_alpha 1.0
done

for k in 0.5 0.65 0.75 0.9; do
  for seed in "${SEED_LIST[@]}"; do
    run_img_privacy "signed_bottleneck_${k}" "$seed" \
      --defense signed_bottleneck \
      --defense_lrb_keep_ratio_sensitive "$k"
  done
done

if [[ "$RUN_ABLATIONS" == "1" ]]; then
  for preset in identity_lrb clip_only proj_rule_only proj_empirical_only proj_uniform proj_no_empirical; do
    for seed in "${SEED_LIST[@]}"; do
      run_img_privacy "${preset}_0.5" "$seed" \
        --defense lrb \
        --defense_lrb_preset "$preset" \
        --defense_lrb_keep_ratio_sensitive 0.5
    done
  done
fi

"$PYTHON_BIN" scripts/peftleak_image_proxy_table.py \
  --log-dir "$OUT/privacy" \
  --output "$OUT/tables/image_privacy_proxy_utility.csv" \
  --markdown-output "$OUT/tables/image_privacy_proxy_utility.md"

grep -R "synthetic_fallback=1" "$OUT/privacy" && exit 1 || echo "No synthetic fallback"
grep -R "result_status=failed" "$OUT/privacy" && exit 1 || echo "No failed runs"
