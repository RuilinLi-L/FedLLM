#!/bin/bash
# Run a single PEFT evaluation through attack.py.
# Usage:
#   ./scripts/peft_eval.sh DATASET BATCH_SIZE MODEL_PATH N_INPUTS --peft_method lora --finetuned_path PATH [extra attack args...]

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
[dager]   ./scripts/peft_eval.sh DATASET BATCH_SIZE MODEL_PATH N_INPUTS --peft_method lora|ia3 --finetuned_path PATH [extra attack args...]
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
PEFT_METHOD="lora"

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

extra_value() {
  local flag="$1"
  local idx=0
  while [ "$idx" -lt "${#EXTRA[@]}" ]; do
    local arg="${EXTRA[$idx]}"
    if [ "$arg" = "$flag" ] && [ $((idx + 1)) -lt "${#EXTRA[@]}" ]; then
      printf '%s' "${EXTRA[$((idx + 1))]}"
      return 0
    fi
    if [[ "$arg" == "${flag}="* ]]; then
      printf '%s' "${arg#*=}"
      return 0
    fi
    idx=$((idx + 1))
  done
  return 1
}

if ! has_extra_flag "--finetuned_path"; then
  echo "[dager] peft_eval.sh requires --finetuned_path PATH to a PEFT adapter directory or LoRA .pt/.pth checkpoint." >&2
  exit 2
fi

if has_extra_flag "--peft_method"; then
  PEFT_METHOD="$(extra_value --peft_method)"
else
  EXTRA+=( --peft_method "$PEFT_METHOD" )
fi

case "$PEFT_METHOD" in
  lora|ia3)
    ;;
  prefix)
    echo "[dager] --peft_method prefix is trainable but not supported by DAGER span eval in v1." >&2
    exit 2
    ;;
  adapter)
    echo "[dager] --peft_method adapter is planned for v2 but not enabled in v1." >&2
    exit 2
    ;;
  *)
    echo "[dager] --peft_method must be lora or ia3 for DAGER PEFT eval." >&2
    exit 2
    ;;
esac

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
  --train_method peft
