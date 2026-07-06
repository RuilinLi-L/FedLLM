#!/bin/bash
# Smoke-test the official-aligned PEFTLeak image path with official CIFAR32-style defaults.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAGER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$DAGER_ROOT" || exit 1

python attack_peftleak_image.py \
  --mode official_vit_adapter \
  --peftleak_profile official_cifar32 \
  --dataset "${DATASET:-synthetic}" \
  --n_images "${N_IMAGES:-1}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --public_split_size "${PUBLIC_SPLIT_SIZE:-4}" \
  --adapter_layers "${ADAPTER_LAYERS:-first_n}" \
  --attack_rounds "${ATTACK_ROUNDS:-1}" \
  --defense "${DEFENSE:-none}" \
  "$@"
