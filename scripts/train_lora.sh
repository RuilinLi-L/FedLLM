#!/bin/bash
# Train a LoRA sequence-classification checkpoint with the same CLI semantics as train.py.
# Usage:
#   ./scripts/train_lora.sh DATASET BATCH_SIZE MODEL_PATH LORA_R [extra train.py args...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAGER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$DAGER_ROOT" || exit 1

# shellcheck source=_dager_auto_log.sh
source "${SCRIPT_DIR}/_dager_auto_log.sh"
dager_auto_log_enable "train_lora" "$@"

if [ "$#" -lt 4 ]; then
  cat >&2 <<EOF
[dager] Usage:
[dager]   ./scripts/train_lora.sh DATASET BATCH_SIZE MODEL_PATH LORA_R [extra train.py args...]
[dager]
[dager] Example:
[dager]   ./scripts/train_lora.sh sst2 2 gpt2 16 --num_epochs 1 --output_dir ./models/gpt2_sst2_lora_r16
EOF
  exit 2
fi

DATASET="$1"
BATCH="$2"
MODEL="$3"
LORA_R="$4"
EXTRA=()
if [ "$#" -gt 4 ]; then
  EXTRA=( "${@:5}" )
fi

has_extra_flag() {
  local flag="$1"
  local arg
  for arg in "${EXTRA[@]}"; do
    if [[ "$arg" == "$flag" || "$arg" == "${flag}="* ]]; then
      return 0
    fi
  done
  return 1
}

if has_extra_flag "--num_epoch"; then
  echo "[dager] train.py uses --num_epochs, not --num_epoch." >&2
  exit 2
fi

python train.py \
  --dataset "$DATASET" \
  --task seq_class \
  --batch_size "$BATCH" \
  --model_path "$MODEL" \
  --train_method peft \
  --peft_method lora \
  --lora_r "$LORA_R" \
  "${EXTRA[@]}"
