#!/bin/bash
# Train a PEFT sequence-classification checkpoint with train.py semantics.
# Usage:
#   ./scripts/train_peft.sh DATASET BATCH_SIZE MODEL_PATH PEFT_METHOD [extra train.py args...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAGER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$DAGER_ROOT" || exit 1

# shellcheck source=_dager_auto_log.sh
source "${SCRIPT_DIR}/_dager_auto_log.sh"
dager_auto_log_enable "train_peft" "$@"

if [ "$#" -lt 4 ]; then
  cat >&2 <<EOF
[dager] Usage:
[dager]   ./scripts/train_peft.sh DATASET BATCH_SIZE MODEL_PATH PEFT_METHOD [extra train.py args...]
[dager]
[dager] Examples:
[dager]   ./scripts/train_peft.sh sst2 2 bert-base-uncased lora --lora_r 16 --num_epochs 1
[dager]   ./scripts/train_peft.sh sst2 2 bert-base-uncased ia3 --num_epochs 1
[dager]   ./scripts/train_peft.sh sst2 2 bert-base-uncased prefix --peft_num_virtual_tokens 20 --num_epochs 1
EOF
  exit 2
fi

DATASET="$1"
BATCH="$2"
MODEL="$3"
PEFT_METHOD="$4"
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

case "$PEFT_METHOD" in
  lora|ia3|prefix)
    ;;
  adapter)
    echo "[dager] --peft_method adapter is v2 planned and not part of v1 PEFT DAGER/partial eval." >&2
    exit 2
    ;;
  *)
    echo "[dager] PEFT_METHOD must be lora, ia3, or prefix." >&2
    exit 2
    ;;
esac

if [ "$PEFT_METHOD" = "lora" ] && ! has_extra_flag "--lora_r"; then
  echo "[dager] --peft_method lora requires --lora_r R." >&2
  exit 2
fi

python train.py \
  --dataset "$DATASET" \
  --task seq_class \
  --batch_size "$BATCH" \
  --model_path "$MODEL" \
  --train_method peft \
  --peft_method "$PEFT_METHOD" \
  "${EXTRA[@]}"
