#!/bin/bash
# Run one non-DAGER FedLLM PEFT text attack.
# Usage:
#   ./scripts/peftleak_eval.sh DATASET BATCH_SIZE MODEL_PATH N_INPUTS --peft_method lora|ia3|adapter --finetuned_path PATH [extra args...]
# Examples:
#   ./scripts/peftleak_eval.sh sst2 1 bert-base-uncased 2 --peft_method adapter --finetuned_path PATH --peftleak_attack_mode ratio
#   ./scripts/peftleak_eval.sh sst2 1 gpt2 2 --peft_method adapter --finetuned_path PATH --peftleak_attack_mode both

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAGER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$DAGER_ROOT" || exit 1

if [ "$#" -lt 4 ]; then
  cat >&2 <<EOF
[peftleak] Usage:
[peftleak]   ./scripts/peftleak_eval.sh DATASET BATCH_SIZE MODEL_PATH N_INPUTS --peft_method lora|ia3|adapter --finetuned_path PATH [extra args...]
[peftleak]   pass --peftleak_attack_mode opt|ratio|both to select optimization, structured ratio, or both.
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

has_flag() {
  local flag="$1"
  local arg
  for arg in "${EXTRA[@]}"; do
    if [[ "$arg" == "$flag" || "$arg" == "${flag}="* ]]; then
      return 0
    fi
  done
  return 1
}

if ! has_flag "--finetuned_path"; then
  echo "[peftleak] --finetuned_path PATH is required." >&2
  exit 2
fi

if ! has_flag "--peft_method"; then
  EXTRA+=( --peft_method lora )
fi

python attack_peftleak.py \
  --dataset "$DATASET" \
  --split val \
  --n_inputs "$N_INPUTS" \
  --batch_size "$BATCH" \
  --model_path "$MODEL" \
  --device cuda \
  --task seq_class \
  --cache_dir ./models_cache \
  --train_method peft \
  "${EXTRA[@]}"
