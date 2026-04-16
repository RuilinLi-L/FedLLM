#!/bin/bash
# Run proxy-only utility baselines:
# - reuse an existing clean anchor checkpoint when available
# - otherwise train a clean anchor first
# - evaluate only proxy utility for each configured defense

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAGER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$DAGER_ROOT" || exit 1

DATASET="${1:-sst2}"
BATCH="${2:-2}"
MODEL="${3:-gpt2}"
EPOCHS="${4:-1}"
RAW_EXTRA=()
if [ "$#" -gt 4 ]; then
  RAW_EXTRA=( "${@:5}" )
fi

ANCHOR_DIR=""
INCLUDE_SENSITIVITY=0
EXTRA=()

parse_script_args() {
  local idx=0
  while [ "$idx" -lt "${#RAW_EXTRA[@]}" ]; do
    local arg="${RAW_EXTRA[$idx]}"
    case "$arg" in
      --anchor_dir)
        ANCHOR_DIR="${RAW_EXTRA[$((idx + 1))]}"
        idx=$((idx + 2))
        ;;
      --anchor_dir=*)
        ANCHOR_DIR="${arg#*=}"
        idx=$((idx + 1))
        ;;
      --include_sensitivity)
        INCLUDE_SENSITIVITY=1
        idx=$((idx + 1))
        ;;
      *)
        EXTRA+=( "$arg" )
        idx=$((idx + 1))
        ;;
    esac
  done
}

slugify() {
  printf '%s' "$1" | tr -c 'a-zA-Z0-9._-' '_' | sed 's/__*/_/g' | sed 's/^_\|_$//g'
}

is_model_dir() {
  local dir="$1"
  [ -d "$dir" ] || return 1
  [ -f "$dir/config.json" ] || return 1

  if ! [ -f "$dir/model.safetensors" ] && \
     ! [ -f "$dir/model.safetensors.index.json" ] && \
     ! [ -f "$dir/pytorch_model.bin" ] && \
     ! [ -f "$dir/pytorch_model.bin.index.json" ]; then
    return 1
  fi
  return 0
}

resolve_existing_anchor() {
  local safe_model candidate
  safe_model="$(slugify "$(basename "$MODEL")")"
  local candidates=(
    "./models/${safe_model}-ft-clean/final"
    "./models/${safe_model}-ft-clean"
    "./models/${safe_model}-ft-rt/final"
    "./models/${safe_model}-ft-rt"
  )
  if [ -n "$ANCHOR_DIR" ]; then
    candidates=( "$ANCHOR_DIR/final" "$ANCHOR_DIR" "${candidates[@]}" )
  fi
  for candidate in "${candidates[@]}"; do
    if [ -n "$candidate" ] && is_model_dir "$candidate"; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  return 1
}

run_logged() {
  local label="$1"
  shift
  local logfile="${RUN_DIR}/${label}.txt"
  echo "[proxy-baselines] running ${label}" >&2
  set +e
  "$@" --log_file "$logfile"
  local rc=$?
  set -e
  printf '%s,%s\n' "$label" "$rc" >> "${RUN_DIR}/exit_codes.csv"
  return 0
}

run_variant() {
  local defense="$1"
  local param="$2"
  local tag="$3"
  local extra_flag=()

  case "$defense" in
    none)
      extra_flag=()
      ;;
    noise|dpsgd)
      extra_flag=( --defense_noise "$param" )
      ;;
    topk)
      extra_flag=( --defense_topk_ratio "$param" )
      ;;
    compression)
      extra_flag=( --defense_n_bits "$param" )
      ;;
    soteria)
      extra_flag=( --defense_soteria_pruning_rate "$param" )
      ;;
    mixup)
      extra_flag=( --defense_mixup_alpha "$param" )
      ;;
    lrb)
      extra_flag=( --defense_lrb_keep_ratio_sensitive "$param" )
      ;;
    *)
      echo "[proxy-baselines] unsupported defense ${defense}" >&2
      return 0
      ;;
  esac

  run_logged \
    "proxy_${tag}" \
    python3 scripts/proxy_utility.py \
    --dataset "$DATASET" \
    --task seq_class \
    --batch_size "$BATCH" \
    --model_path "$ANCHOR_MODEL_DIR" \
    --n_train_batches 100 \
    --val_size 256 \
    --eval_batch_size 16 \
    --train_method full \
    --defense "$defense" \
    "${extra_flag[@]}" \
    "${EXTRA[@]}"
}

parse_script_args

safe_model="$(slugify "$(basename "$MODEL")")"
safe_ds="$(slugify "$DATASET")"
stamp="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="log/runs/proxy_baselines_${safe_ds}_b${BATCH}_${safe_model}_${stamp}"
mkdir -p "$RUN_DIR"
printf 'label,exit_code\n' > "${RUN_DIR}/exit_codes.csv"

if ! ANCHOR_MODEL_DIR="$(resolve_existing_anchor)"; then
  if [ -z "$ANCHOR_DIR" ]; then
    ANCHOR_DIR="./models/${safe_model}_utility_anchor_${safe_ds}"
  fi
  echo "[proxy-baselines] no clean anchor found, training one at ${ANCHOR_DIR}" >&2
  run_logged \
    "anchor_train_none" \
    python3 train.py \
    --dataset "$DATASET" \
    --task seq_class \
    --batch_size "$BATCH" \
    --num_epochs "$EPOCHS" \
    --save_every 0 \
    --model_path "$MODEL" \
    --train_method full \
    --defense none \
    --rng_seed 101 \
    --output_dir "$ANCHOR_DIR" \
    "${EXTRA[@]}"
  ANCHOR_MODEL_DIR="${ANCHOR_DIR}/final"
fi

echo "[proxy-baselines] anchor model: ${ANCHOR_MODEL_DIR}" >&2

run_variant none n/a none
run_variant lrb 0.2 lrb_0.2
run_variant topk 0.1 topk_0.1
run_variant compression 8 compression_8
run_variant noise 5e-4 noise_5e-4
run_variant dpsgd 5e-4 dpsgd_5e-4
run_variant mixup 0.3 mixup_0.3
run_variant soteria 30 soteria_30

if [ "$INCLUDE_SENSITIVITY" -eq 1 ]; then
  run_variant lrb 0.35 lrb_0.35
  run_variant compression 16 compression_16
fi

python3 scripts/collect_experiment_logs.py \
  "$RUN_DIR" \
  -o "${RUN_DIR}/results.csv" \
  --markdown "${RUN_DIR}/results.md"

echo "[proxy-baselines] done. outputs live under ${RUN_DIR}" >&2
