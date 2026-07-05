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

flag_value() {
  local flag="$1"
  local idx=0
  local value=""
  local found=1
  while [ "$idx" -lt "${#EXTRA[@]}" ]; do
    local arg="${EXTRA[$idx]}"
    if [[ "$arg" == "$flag" ]]; then
      if [ "$((idx + 1))" -lt "${#EXTRA[@]}" ]; then
        value="${EXTRA[$((idx + 1))]}"
        found=0
        idx=$((idx + 2))
        continue
      fi
    elif [[ "$arg" == "${flag}="* ]]; then
      value="${arg#*=}"
      found=0
    fi
    idx=$((idx + 1))
  done
  if [ "$found" -eq 0 ]; then
    printf '%s\n' "$value"
    return 0
  fi
  return 1
}

sanitize_label() {
  local raw="$1"
  raw="${raw##*/}"
  raw="${raw// /_}"
  raw="${raw//[!A-Za-z0-9._-]/_}"
  printf '%s' "$raw"
}

defense_param_label() {
  local defense="$1"
  local value=""
  local preset=""
  case "$defense" in
    none)
      return 0
      ;;
    noise|dpsgd)
      if value="$(flag_value "--defense_noise")"; then
        sanitize_label "$value"
      fi
      ;;
    topk)
      if value="$(flag_value "--defense_topk_ratio")"; then
        sanitize_label "$value"
      fi
      ;;
    compression)
      if value="$(flag_value "--defense_n_bits")"; then
        sanitize_label "$value"
      fi
      ;;
    soteria)
      if value="$(flag_value "--defense_soteria_pruning_rate")"; then
        sanitize_label "$value"
      fi
      ;;
    mixup)
      if value="$(flag_value "--defense_mixup_alpha")"; then
        sanitize_label "$value"
      fi
      ;;
    lrb|lrbprojonly|signed_bottleneck)
      if preset="$(flag_value "--defense_lrb_preset")"; then
        if [ "$preset" != "custom" ] && [ "$preset" != "$defense" ]; then
          printf '%s' "$(sanitize_label "$preset")"
        fi
      fi
      if value="$(flag_value "--defense_lrb_keep_ratio_sensitive")"; then
        if [ -n "$preset" ] && [ "$preset" != "custom" ] && [ "$preset" != "$defense" ]; then
          printf '_'
        fi
        sanitize_label "$value"
      fi
      ;;
  esac
}

if ! has_flag "--finetuned_path"; then
  echo "[peftleak] --finetuned_path PATH is required." >&2
  exit 2
fi

if ! has_flag "--peft_method"; then
  EXTRA+=( --peft_method lora )
fi

if has_flag "--rng_seed"; then
  SEEDS=( "$(flag_value "--rng_seed")" )
else
  SEEDS_RAW="${PEFTLEAK_SEEDS:-101 202 303}"
  SEEDS_RAW="${SEEDS_RAW//,/ }"
  read -r -a SEEDS <<< "$SEEDS_RAW"
fi

if [ "${#SEEDS[@]}" -eq 0 ]; then
  echo "[peftleak] No seeds requested." >&2
  exit 2
fi

RUN_EXTRA=()
for ((idx = 0; idx < ${#EXTRA[@]}; idx++)); do
  arg="${EXTRA[$idx]}"
  if [[ "$arg" == "--rng_seed" ]]; then
    idx=$((idx + 1))
    continue
  fi
  if [[ "$arg" == "--rng_seed="* ]]; then
    continue
  fi
  RUN_EXTRA+=( "$arg" )
done

if PEFT_METHOD="$(flag_value "--peft_method")"; then
  :
else
  PEFT_METHOD="lora"
fi
if ATTACK_MODE="$(flag_value "--peftleak_attack_mode")"; then
  :
else
  ATTACK_MODE="opt"
fi
if DEFENSE="$(flag_value "--defense")"; then
  :
else
  DEFENSE="none"
fi

MODEL_LABEL="$(sanitize_label "$MODEL")"
PEFT_LABEL="$(sanitize_label "$PEFT_METHOD")"
ATTACK_LABEL="$(sanitize_label "$ATTACK_MODE")"
DEFENSE_LABEL="$(sanitize_label "$DEFENSE")"
PARAM_LABEL="$(defense_param_label "$DEFENSE")"
LOG_DIR="${PEFTLEAK_LOG_DIR:-log/peftleak_text_${DATASET}/privacy}"
mkdir -p "$LOG_DIR"

LOG_STEM="${MODEL_LABEL}_${PEFT_LABEL}_${ATTACK_LABEL}_${DEFENSE_LABEL}"
if [ -n "$PARAM_LABEL" ]; then
  LOG_STEM="${LOG_STEM}_${PARAM_LABEL}"
fi

overall_rc=0
for seed in "${SEEDS[@]}"; do
  log_file="${LOG_DIR}/${LOG_STEM}_seed${seed}.txt"
  echo "[peftleak] Running seed=${seed}; log=${log_file}" >&2
  set +e
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
    "${RUN_EXTRA[@]}" \
    --rng_seed "$seed" 2>&1 | tee "$log_file"
  rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -ne 0 ]; then
    overall_rc="$rc"
  fi
done

exit "$overall_rc"
