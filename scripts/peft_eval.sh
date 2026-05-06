#!/bin/bash
# Run a single PEFT/LoRA evaluation through attack.py.
# Usage:
#   ./scripts/peft_eval.sh DATASET BATCH_SIZE MODEL_PATH N_INPUTS --finetuned_path PATH --lora_r R [extra attack args...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAGER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$DAGER_ROOT" || exit 1

# shellcheck source=_dager_auto_log.sh
source "${SCRIPT_DIR}/_dager_auto_log.sh"
dager_auto_log_enable "peft_eval" "$@"

if [ "$#" -lt 4 ]; then
  cat >&2 <<EOF
[dager] Usage:
[dager]   ./scripts/peft_eval.sh DATASET BATCH_SIZE MODEL_PATH N_INPUTS --finetuned_path PATH --lora_r R [extra attack args...]
EOF
  exit 2
fi

DATASET="$1"
BATCH="$2"
MODEL="$3"
N_INPUTS="$4"
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

if ! has_extra_flag "--finetuned_path"; then
  echo "[dager] peft_eval.sh requires --finetuned_path PATH to a PEFT adapter directory or LoRA .pt/.pth checkpoint." >&2
  exit 2
fi

if ! has_extra_flag "--lora_r"; then
  echo "[dager] peft_eval.sh requires --lora_r R." >&2
  exit 2
fi

python attack.py \
  --dataset "$DATASET" \
  --split val \
  --n_inputs "$N_INPUTS" \
  --batch_size "$BATCH" \
  --l1_filter all \
  --l2_filter non-overlap \
  --model_path "$MODEL" \
  --device cuda \
  --task seq_class \
  --cache_dir ./models_cache \
  "${EXTRA[@]}" \
  --train_method lora
