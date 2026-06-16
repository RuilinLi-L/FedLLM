#!/bin/bash
# Sweep FedLLM PEFT text defenses. DAGER is intentionally unsupported in the main matrix.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAGER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$DAGER_ROOT" || exit 1

DATASET="${1:-sst2}"
BATCH="${2:-2}"
MODEL="${3:-gpt2}"
N_INPUTS="${4:-2}"
EXTRA=()
if [ "$#" -gt 4 ]; then
  EXTRA=( "${@:5}" )
fi

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
    soteria)
      DEF_EXTRA=( --defense_soteria_pruning_rate 60 )
      ;;
    mixup)
      DEF_EXTRA=( --defense_mixup_alpha 1.0 )
      ;;
    lrbprojonly)
      DEF_EXTRA=( --defense_lrb_keep_ratio_sensitive 0.5 )
      ;;
    signed_bottleneck)
      DEF_EXTRA=( --defense_lrb_preset signed_bottleneck --defense_lrb_keep_ratio_sensitive 0.99 )
      ;;
  esac
  bash "$SCRIPT_DIR/peftleak_eval.sh" "$DATASET" "$BATCH" "$MODEL" "$N_INPUTS" \
    "${EXTRA[@]}" \
    --defense "$defense" \
    "${DEF_EXTRA[@]}"
done
