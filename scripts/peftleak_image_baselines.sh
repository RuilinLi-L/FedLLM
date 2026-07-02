#!/bin/bash
# Sweep the lightweight PEFTLeak-style image smoke/defense sanity matrix.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAGER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$DAGER_ROOT" || exit 1

DEFENSES=( none noise dpsgd topk compression soteria mixup lrb lrbprojonly signed_bottleneck )

for defense in "${DEFENSES[@]}"; do
  echo "========== defense=${defense} =========="
  DEF_EXTRA=()
  case "$defense" in
    noise|dpsgd)
      DEF_EXTRA=( --defense_noise 1e-5 )
      ;;
    topk)
      DEF_EXTRA=( --defense_topk_ratio 0.1 )
      ;;
    compression)
      DEF_EXTRA=( --defense_n_bits 8 )
      ;;
    signed_bottleneck)
      DEF_EXTRA=( --defense_lrb_preset signed_bottleneck --defense_lrb_keep_ratio_sensitive 0.99 )
      ;;
  esac
  python attack_peftleak_image.py --mode vit_adapter --dataset synthetic --n_images 4 --batch_size 2 --defense "$defense" "${DEF_EXTRA[@]}" "$@"
done
