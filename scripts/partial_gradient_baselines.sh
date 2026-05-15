#!/bin/bash
# Run partial-gradient transfer sweeps through attack.py.
# Usage:
#   ./scripts/partial_gradient_baselines.sh DATASET BATCH_SIZE MODEL_PATH N_INPUTS --exposure first2 --finetuned_path PATH [extra attack args...]
# Script-only flags:
#   --exposure <first2|last2|mid2|qkv_only|lora_only>
#   --train_method <full|lora|peft>
#   --baseline_defense <none|noise|dpsgd|topk|compression|soteria|mixup|dager|lrb|lrbprojonly>
#   --baseline_param <value>
#   --lrb_variants <comma-separated LRB presets>
#   --lrb_main_k <value>
#   --allow_unsupported_exposure

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAGER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$DAGER_ROOT" || exit 1

DATASET="${1:-sst2}"
BATCH="${2:-2}"
MODEL="${3:-gpt2}"
N_INPUTS="${4:-3}"
RAW_EXTRA=()
if [ "$#" -gt 4 ]; then
  RAW_EXTRA=( "${@:5}" )
fi

EXPOSURE=""
TRAIN_METHOD="full"
PEFT_METHOD=""
BASELINE_DEFENSE=""
BASELINE_PARAM=""
LRB_VARIANTS_RAW=""
LRB_MAIN_K="0.5"
ALLOW_UNSUPPORTED_EXPOSURE=0
PARTIAL_ATTACK_VARIANT="full_gradient_visible"
UNSUPPORTED_REASON="n/a"
EXTRA=()

ALL_FULL_DEFENSES=( none noise dpsgd topk compression soteria mixup lrb lrbprojonly )
ALL_PEFT_DEFENSES=( none noise dpsgd topk compression soteria mixup dager lrb lrbprojonly )
SUPPORTED_PEFT_DEFENSES=( none noise dpsgd topk compression soteria mixup lrb lrbprojonly )
KNOWN_LRB_PRESETS=(
  identity_lrb
  clip_only
  proj_only
  proj_clip
  full_lrb
  pool_full
  rule_only
  empirical_only
  uniform_all_sensitive
  proj_rule_only
  proj_empirical_only
  proj_uniform
  proj_no_empirical
)

parse_script_args() {
  local idx=0
  while [ "$idx" -lt "${#RAW_EXTRA[@]}" ]; do
    local arg="${RAW_EXTRA[$idx]}"
    case "$arg" in
      --exposure)
        EXPOSURE="${RAW_EXTRA[$((idx + 1))]:-}"
        idx=$((idx + 2))
        ;;
      --exposure=*)
        EXPOSURE="${arg#*=}"
        idx=$((idx + 1))
        ;;
      --train_method)
        TRAIN_METHOD="${RAW_EXTRA[$((idx + 1))]:-}"
        idx=$((idx + 2))
        ;;
      --train_method=*)
        TRAIN_METHOD="${arg#*=}"
        idx=$((idx + 1))
        ;;
      --peft_method)
        PEFT_METHOD="${RAW_EXTRA[$((idx + 1))]:-}"
        idx=$((idx + 2))
        ;;
      --peft_method=*)
        PEFT_METHOD="${arg#*=}"
        idx=$((idx + 1))
        ;;
      --baseline_defense)
        BASELINE_DEFENSE="${RAW_EXTRA[$((idx + 1))]:-}"
        idx=$((idx + 2))
        ;;
      --baseline_defense=*)
        BASELINE_DEFENSE="${arg#*=}"
        idx=$((idx + 1))
        ;;
      --baseline_param)
        BASELINE_PARAM="${RAW_EXTRA[$((idx + 1))]:-}"
        idx=$((idx + 2))
        ;;
      --baseline_param=*)
        BASELINE_PARAM="${arg#*=}"
        idx=$((idx + 1))
        ;;
      --lrb_variants)
        LRB_VARIANTS_RAW="${RAW_EXTRA[$((idx + 1))]:-}"
        idx=$((idx + 2))
        ;;
      --lrb_variants=*)
        LRB_VARIANTS_RAW="${arg#*=}"
        idx=$((idx + 1))
        ;;
      --lrb_main_k)
        LRB_MAIN_K="${RAW_EXTRA[$((idx + 1))]:-}"
        idx=$((idx + 2))
        ;;
      --lrb_main_k=*)
        LRB_MAIN_K="${arg#*=}"
        idx=$((idx + 1))
        ;;
      --allow_unsupported_exposure)
        ALLOW_UNSUPPORTED_EXPOSURE=1
        idx=$((idx + 1))
        ;;
      *)
        EXTRA+=( "$arg" )
        idx=$((idx + 1))
        ;;
    esac
  done
}

has_attack_extra_flag() {
  local flag="$1"
  local arg
  for arg in "${EXTRA[@]}"; do
    if [[ "$arg" == "$flag" || "$arg" == "${flag}="* ]]; then
      return 0
    fi
  done
  return 1
}

attack_extra_value() {
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

dager_param_slug() {
  printf '%s' "$1" | tr -c 'a-zA-Z0-9._-' '_' | sed 's/__*/_/g' | sed 's/^_\|_$//g'
}

is_supported_peft_defense() {
  local defense="$1"
  local item
  for item in "${SUPPORTED_PEFT_DEFENSES[@]}"; do
    if [ "$item" = "$defense" ]; then
      return 0
    fi
  done
  return 1
}

is_known_lrb_variant() {
  local variant="$1"
  local item
  for item in "${KNOWN_LRB_PRESETS[@]}"; do
    if [ "$item" = "$variant" ]; then
      return 0
    fi
  done
  return 1
}

dager_param_name() {
  case "$1" in
    noise|dpsgd)
      printf 'defense_noise'
      ;;
    topk)
      printf 'defense_topk_ratio'
      ;;
    compression)
      printf 'defense_n_bits'
      ;;
    soteria)
      printf 'defense_soteria_pruning_rate'
      ;;
    mixup)
      printf 'defense_mixup_alpha'
      ;;
    lrb)
      if [ -n "$LRB_VARIANTS_RAW" ]; then
        printf 'defense_lrb_preset'
      else
        printf 'defense_lrb_keep_ratio_sensitive'
      fi
      ;;
    lrbprojonly)
      printf 'defense_lrb_preset'
      ;;
    *)
      printf 'n/a'
      ;;
  esac
}

dager_set_param_values() {
  local defense="$1"
  case "$defense" in
    none|dager)
      param_vals=( "" )
      ;;
    noise|dpsgd)
      param_vals=( 1e-6 1e-5 1e-4 5e-4 1e-3 )
      ;;
    topk)
      param_vals=( 0.01 0.05 0.1 0.3 0.5 0.7 0.9 )
      ;;
    compression)
      param_vals=( 2 4 8 16 32 )
      ;;
    soteria)
      param_vals=( 10 30 50 70 90 )
      ;;
    mixup)
      param_vals=( 0.1 0.3 0.5 1.0 2.0 )
      ;;
    lrb|lrbprojonly)
      param_vals=( 0.05 0.1 0.2 0.35 0.5 )
      ;;
    *)
      echo "[partial-gradient] Unknown defense for sweep: ${defense}" >&2
      exit 2
      ;;
  esac
}

exposure_flags() {
  case "$EXPOSURE" in
    first2)
      EXPOSURE_EXTRA=( --gradient_layer_subset first2 --gradient_param_filter all )
      PARTIAL_ATTACK_VARIANT="dager_prefix_visible"
      UNSUPPORTED_REASON="n/a"
      ;;
    last2)
      EXPOSURE_EXTRA=( --gradient_layer_subset last2 --gradient_param_filter all )
      PARTIAL_ATTACK_VARIANT="unsupported_nonprefix_dager"
      UNSUPPORTED_REASON="nonprefix_layer_subset_requires_layer_aligned_decoder"
      ;;
    mid[0-9]*|middle[0-9]*)
      EXPOSURE_EXTRA=( --gradient_layer_subset "$EXPOSURE" --gradient_param_filter all )
      PARTIAL_ATTACK_VARIANT="unsupported_nonprefix_dager"
      UNSUPPORTED_REASON="nonprefix_layer_subset_requires_layer_aligned_decoder"
      ;;
    qkv_only)
      EXPOSURE_EXTRA=( --gradient_layer_subset all --gradient_param_filter qkv_only )
      PARTIAL_ATTACK_VARIANT="dager_qkv_visible"
      UNSUPPORTED_REASON="n/a"
      ;;
    lora_only)
      EXPOSURE_EXTRA=( --gradient_layer_subset all --gradient_param_filter lora_only )
      PARTIAL_ATTACK_VARIANT="peft_adapter_visible"
      UNSUPPORTED_REASON="n/a"
      ;;
    *)
      echo "[partial-gradient] --exposure must be one of: first2, last2, mid2, qkv_only, lora_only." >&2
      exit 2
      ;;
  esac
}

dager_has_result_summary() {
  local file="$1"
  grep -q '^===== RESULT SUMMARY START =====$' "$file" 2>/dev/null && \
    grep -q '^===== RESULT SUMMARY END =====$' "$file" 2>/dev/null
}

dager_result_summary_block_from_file() {
  local file="$1"
  awk '
    /^===== RESULT SUMMARY START =====$/ {
      in_block = 1
      block = $0 ORS
      next
    }
    in_block {
      block = block $0 ORS
    }
    /^===== RESULT SUMMARY END =====$/ && in_block {
      last = block
      in_block = 0
    }
    END {
      printf "%s", last
    }
  ' "$file"
}

dager_summary_value() {
  local file="$1"
  local key="$2"
  awk -F= -v key="$key" '
    /^===== RESULT SUMMARY START =====$/ {
      in_block = 1
      next
    }
    /^===== RESULT SUMMARY END =====$/ {
      in_block = 0
    }
    in_block && index($0, key "=") == 1 {
      value = substr($0, length(key) + 2)
    }
    END {
      print value
    }
  ' "$file"
}

dager_fallback_summary_block() {
  local defense="$1"
  local param="$2"
  local log_base="$3"
  local rc="$4"
  local t_start="$5"
  local t_end="$6"
  cat <<EOF
===== RESULT SUMMARY START =====
summary_version=2
result_status=failed
dataset=${DATASET}
split=val
task=seq_class
model_path=${MODEL}
finetuned_path=$(attack_extra_value --finetuned_path || printf 'n/a')
batch_size=${BATCH}
train_method=${TRAIN_METHOD}
peft_method=${PEFT_METHOD:-n/a}
defense=${defense}
defense_param_name=$(dager_param_name "$defense")
defense_param_value=${param:-n/a}
gradient_layer_subset=${GRADIENT_LAYER_SUBSET}
gradient_param_filter=${GRADIENT_PARAM_FILTER}
partial_filter_active=true
partial_attack_variant=${PARTIAL_ATTACK_VARIANT}
visible_grad_count=n/a
visible_matrix_grad_count=n/a
visible_param_names=n/a
dager_visible_candidate_count=n/a
dager_visible_param_names=n/a
selected_block_ids=n/a
unsupported_reason=${UNSUPPORTED_REASON}
n_inputs_requested=${N_INPUTS}
n_inputs_completed=0
last_rec_status=failed
rec_token_mean=n/a
rec_maxb_token_mean=n/a
error_type=runner_error
error_message=missing_result_summary_or_process_failed
script_variant=${log_base}
script_start_time=${t_start}
script_end_time=${t_end}
script_exit_code=${rc}
===== RESULT SUMMARY END =====
EOF
}

dager_unsupported_summary_block() {
  local defense="$1"
  local param="$2"
  cat <<EOF
===== RESULT SUMMARY START =====
summary_version=2
result_status=unsupported
dataset=${DATASET}
split=val
task=seq_class
model_path=${MODEL}
finetuned_path=$(attack_extra_value --finetuned_path || printf 'n/a')
batch_size=${BATCH}
train_method=${TRAIN_METHOD}
peft_method=${PEFT_METHOD:-n/a}
defense=${defense}
defense_param_name=$(dager_param_name "$defense")
defense_param_value=${param:-n/a}
gradient_layer_subset=${GRADIENT_LAYER_SUBSET}
gradient_param_filter=${GRADIENT_PARAM_FILTER}
partial_filter_active=true
partial_attack_variant=${PARTIAL_ATTACK_VARIANT}
visible_grad_count=n/a
visible_matrix_grad_count=n/a
visible_param_names=n/a
dager_visible_candidate_count=n/a
dager_visible_param_names=n/a
selected_block_ids=n/a
unsupported_reason=${UNSUPPORTED_REASON}
n_inputs_requested=${N_INPUTS}
n_inputs_completed=0
last_rec_status=unsupported
rec_token_mean=n/a
rec_maxb_token_mean=n/a
error_type=unsupported_defense
error_message=PEFT eval currently supports only defenses: none, noise, dpsgd, topk, compression, soteria, mixup, lrb, lrbprojonly
===== RESULT SUMMARY END =====
EOF
}

dager_unsupported_exposure_summary_block() {
  local defense="$1"
  local param="$2"
  cat <<EOF
===== RESULT SUMMARY START =====
summary_version=2
result_status=unsupported
dataset=${DATASET}
split=val
task=seq_class
model_path=${MODEL}
finetuned_path=$(attack_extra_value --finetuned_path || printf 'n/a')
batch_size=${BATCH}
train_method=${TRAIN_METHOD}
peft_method=${PEFT_METHOD:-n/a}
defense=${defense}
defense_param_name=$(dager_param_name "$defense")
defense_param_value=${param:-n/a}
gradient_layer_subset=${GRADIENT_LAYER_SUBSET}
gradient_param_filter=${GRADIENT_PARAM_FILTER}
partial_filter_active=true
partial_attack_variant=${PARTIAL_ATTACK_VARIANT}
visible_grad_count=n/a
visible_matrix_grad_count=n/a
visible_param_names=n/a
dager_visible_candidate_count=n/a
dager_visible_param_names=n/a
selected_block_ids=n/a
unsupported_reason=${UNSUPPORTED_REASON}
n_inputs_requested=${N_INPUTS}
n_inputs_completed=0
last_rec_status=unsupported
rec_token_mean=n/a
rec_maxb_token_mean=n/a
error_type=unsupported_partial_exposure
error_message=${UNSUPPORTED_REASON}
===== RESULT SUMMARY END =====
EOF
}

write_variant_summary_file() {
  local def_file="$1"
  local summary_block="$2"
  local defense="$3"
  local param="$4"
  local log_base="$5"
  local t_start="$6"
  local t_end="$7"
  local rc="$8"
  local tmpfile="${9:-}"

  {
    echo "===== VARIANT START exposure=${EXPOSURE} train_method=${TRAIN_METHOD} defense=${defense} param=${param:-n/a} dataset=${DATASET} batch=${BATCH} model=$(basename "$MODEL") start=${t_start} ====="
    printf '%s\n' "$summary_block"
    echo "===== VARIANT END end=${t_end} exit_code=${rc} ====="
    if [ -n "$tmpfile" ] && [ -f "$tmpfile" ] && [ "$rc" -ne 0 ]; then
      echo "--- last 25 lines from run output ---"
      tail -n 25 "$tmpfile"
    fi
  } >"$def_file"
}

parse_script_args

if [ -z "$EXPOSURE" ]; then
  echo "[partial-gradient] --exposure is required." >&2
  exit 2
fi

case "$TRAIN_METHOD" in
  full|lora|peft)
    ;;
  *)
    echo "[partial-gradient] --train_method must be full, lora, or peft." >&2
    exit 2
    ;;
esac

if [ -n "$PEFT_METHOD" ]; then
  case "$PEFT_METHOD" in
    lora|ia3)
      ;;
    prefix)
      echo "[partial-gradient] --peft_method prefix is trainable but not supported by DAGER span eval in v1." >&2
      exit 2
      ;;
    adapter)
      echo "[partial-gradient] --peft_method adapter is planned for v2 but not enabled in v1." >&2
      exit 2
      ;;
    *)
      echo "[partial-gradient] --peft_method must be lora or ia3 for DAGER PEFT eval." >&2
      exit 2
      ;;
  esac
fi

exposure_flags
GRADIENT_LAYER_SUBSET="${EXPOSURE_EXTRA[1]}"
GRADIENT_PARAM_FILTER="${EXPOSURE_EXTRA[3]}"

if [ "$EXPOSURE" = "lora_only" ] && [ "$TRAIN_METHOD" != "lora" ] && [ "$TRAIN_METHOD" != "peft" ]; then
  echo "[partial-gradient] --exposure lora_only requires --train_method lora or peft." >&2
  exit 2
fi

if ! has_attack_extra_flag "--finetuned_path"; then
  echo "[partial-gradient] pass --finetuned_path PATH for credible seq_class partial-gradient runs." >&2
  exit 2
fi

if [ -n "$BASELINE_DEFENSE" ]; then
  case "$BASELINE_DEFENSE" in
    none|noise|dpsgd|topk|compression|soteria|mixup|dager|lrb|lrbprojonly)
      ;;
    *)
      echo "[partial-gradient] Unsupported --baseline_defense: ${BASELINE_DEFENSE}" >&2
      exit 2
      ;;
  esac
fi

if [ -n "$BASELINE_PARAM" ] && [ -z "$BASELINE_DEFENSE" ]; then
  echo "[partial-gradient] --baseline_param requires --baseline_defense." >&2
  exit 2
fi

if [ "$BASELINE_DEFENSE" = "none" ] && [ -n "$BASELINE_PARAM" ]; then
  echo "[partial-gradient] --baseline_defense none cannot be combined with --baseline_param." >&2
  exit 2
fi

if [ -n "$LRB_VARIANTS_RAW" ] && [ "$BASELINE_DEFENSE" != "lrb" ]; then
  echo "[partial-gradient] --lrb_variants can only be used with --baseline_defense lrb." >&2
  exit 2
fi

if [ -n "$LRB_VARIANTS_RAW" ] && [ -n "$BASELINE_PARAM" ]; then
  echo "[partial-gradient] --lrb_variants cannot be combined with --baseline_param." >&2
  exit 2
fi

LRB_VARIANTS=()
if [ -n "$LRB_VARIANTS_RAW" ]; then
  IFS=',' read -r -a LRB_VARIANTS <<< "$LRB_VARIANTS_RAW"
  for variant in "${LRB_VARIANTS[@]}"; do
    if ! is_known_lrb_variant "$variant"; then
      echo "[partial-gradient] unknown LRB variant: ${variant}" >&2
      exit 2
    fi
  done
fi

if [ "$TRAIN_METHOD" = "lora" ]; then
  TRAIN_METHOD="peft"
  PEFT_METHOD="${PEFT_METHOD:-lora}"
fi

if [ "$TRAIN_METHOD" = "peft" ]; then
  PEFT_METHOD="${PEFT_METHOD:-lora}"
fi

if [ "$TRAIN_METHOD" = "peft" ]; then
  all_defenses=( "${ALL_PEFT_DEFENSES[@]}" )
else
  all_defenses=( "${ALL_FULL_DEFENSES[@]}" )
fi

selected_defenses=()
if [ -n "$BASELINE_DEFENSE" ]; then
  if [ "$BASELINE_DEFENSE" = "none" ]; then
    selected_defenses=( none )
  else
    selected_defenses=( none "$BASELINE_DEFENSE" )
  fi
else
  selected_defenses=( "${all_defenses[@]}" )
fi

stamp=$(date +%Y%m%d_%H%M%S)
log_root="${DAGER_LOG_DIR:-log/runs}"
mkdir -p "$log_root" || true
safe_ds=$(printf '%s' "$DATASET" | tr -c 'a-zA-Z0-9._-' '_')
safe_model=$(printf '%s' "$(basename "$MODEL")" | tr -c 'a-zA-Z0-9._-' '_')
focus_suffix=""
if [ -n "$BASELINE_DEFENSE" ]; then
  focus_suffix="_focus_${BASELINE_DEFENSE}"
  if [ -n "$BASELINE_PARAM" ]; then
    focus_suffix="${focus_suffix}_$(dager_param_slug "$BASELINE_PARAM")"
  fi
fi
run_dir="${log_root}/partial_gradient_${EXPOSURE}_${TRAIN_METHOD}_${safe_ds}_b${BATCH}_${safe_model}${focus_suffix}_${stamp}"
mkdir -p "$run_dir" || true
summary_path="${run_dir}/summary.txt"
results_csv="${run_dir}/results.csv"
results_md="${run_dir}/results.md"

header_line="===== run start $(date '+%Y-%m-%d %H:%M:%S') tag=partial_gradient_baselines argv: $* ====="
{
  echo "$header_line"
  echo "exposure=${EXPOSURE}"
  echo "gradient_layer_subset=${GRADIENT_LAYER_SUBSET}"
  echo "gradient_param_filter=${GRADIENT_PARAM_FILTER}"
  echo "partial_attack_variant=${PARTIAL_ATTACK_VARIANT}"
  echo "unsupported_reason=${UNSUPPORTED_REASON}"
  echo "allow_unsupported_exposure=${ALLOW_UNSUPPORTED_EXPOSURE}"
  echo "train_method=${TRAIN_METHOD}"
  echo "peft_method=${PEFT_METHOD:-n/a}"
  echo "focus_baseline_defense=${BASELINE_DEFENSE:-all}"
  echo "focus_baseline_param=${BASELINE_PARAM:-all}"
  echo "lrb_variants=${LRB_VARIANTS_RAW:-none}"
  echo "lrb_main_k=${LRB_MAIN_K}"
  echo "selected_defenses=${selected_defenses[*]}"
} >"${run_dir}/_run_header.txt"
cp "${run_dir}/_run_header.txt" "${summary_path}"
echo "[partial-gradient] Run directory: ${run_dir}" >&2
echo "[partial-gradient] Summary: ${summary_path}" >&2

BASE=(
  python attack.py
  --dataset "$DATASET"
  --split val
  --n_inputs "$N_INPUTS"
  --batch_size "$BATCH"
  --l1_filter all
  --l2_filter non-overlap
  --model_path "$MODEL"
  --device cuda
  --task seq_class
  --cache_dir ./models_cache
  "${EXPOSURE_EXTRA[@]}"
)

TRAIN_EXTRA=()
if [ "$TRAIN_METHOD" = "peft" ]; then
  TRAIN_EXTRA=( --train_method peft --peft_method "$PEFT_METHOD" )
fi

variant_files=()

run_variant() {
  local defense="$1"
  local log_base="$2"
  local param="$3"
  shift 3
  local def_extra=( "$@" )
  local def_file
  local summary_block
  local t_start
  local t_end
  local rc

  echo "---------- ${log_base} ----------"
  def_file="${run_dir}/${log_base}.txt"
  t_start=$(date '+%Y-%m-%d %H:%M:%S')

  if [[ "$PARTIAL_ATTACK_VARIANT" == unsupported_* && "$ALLOW_UNSUPPORTED_EXPOSURE" != "1" ]]; then
    summary_block="$(dager_unsupported_exposure_summary_block "$defense" "$param")"
    t_end=$(date '+%Y-%m-%d %H:%M:%S')
    write_variant_summary_file "$def_file" "$summary_block" "$defense" "$param" "$log_base" "$t_start" "$t_end" 0
    {
      echo ""
      echo "========== variant=${log_base} exposure=${EXPOSURE} defense=${defense} param=${param:-n/a} =========="
      printf '%s\n' "$summary_block"
    } >>"${summary_path}"
    variant_files+=( "$def_file" )
    return 0
  fi

  if [ "$TRAIN_METHOD" = "peft" ] && ! is_supported_peft_defense "$defense"; then
    summary_block="$(dager_unsupported_summary_block "$defense" "$param")"
    t_end=$(date '+%Y-%m-%d %H:%M:%S')
    write_variant_summary_file "$def_file" "$summary_block" "$defense" "$param" "$log_base" "$t_start" "$t_end" 0
    {
      echo ""
      echo "========== variant=${log_base} exposure=${EXPOSURE} defense=${defense} param=${param:-n/a} =========="
      printf '%s\n' "$summary_block"
    } >>"${summary_path}"
    variant_files+=( "$def_file" )
    return 0
  fi

  local tmpfile
  tmpfile=$(mktemp)
  set +e
  "${BASE[@]}" "${EXTRA[@]}" "${TRAIN_EXTRA[@]}" --defense "$defense" "${def_extra[@]}" 2>&1 | tee "$tmpfile"
  rc=${PIPESTATUS[0]}
  set -e
  t_end=$(date '+%Y-%m-%d %H:%M:%S')

  if dager_has_result_summary "$tmpfile"; then
    summary_block="$(dager_result_summary_block_from_file "$tmpfile")"
  else
    summary_block="$(dager_fallback_summary_block "$defense" "$param" "$log_base" "$rc" "$t_start" "$t_end")"
  fi

  write_variant_summary_file "$def_file" "$summary_block" "$defense" "$param" "$log_base" "$t_start" "$t_end" "$rc" "$tmpfile"
  {
    echo ""
    echo "========== variant=${log_base} exposure=${EXPOSURE} defense=${defense} param=${param:-n/a} =========="
    printf '%s\n' "$summary_block"
  } >>"${summary_path}"

  variant_files+=( "$def_file" )
  rm -f "$tmpfile"
}

for defense in "${selected_defenses[@]}"; do
  if [ "$defense" = "lrb" ] && [ "${#LRB_VARIANTS[@]}" -gt 0 ]; then
    for variant in "${LRB_VARIANTS[@]}"; do
      log_base="lrb_${variant}_k$(dager_param_slug "$LRB_MAIN_K")"
      param="${variant}@k=${LRB_MAIN_K}"
      DEF_EXTRA=(
        --defense_lrb_preset "$variant"
        --defense_lrb_keep_ratio_sensitive "$LRB_MAIN_K"
      )
      run_variant "$defense" "$log_base" "$param" "${DEF_EXTRA[@]}"
    done
    continue
  fi

  if [ "$defense" = "lrbprojonly" ] && [ "${#LRB_VARIANTS[@]}" -gt 0 ]; then
    echo "[partial-gradient] --lrb_variants applies to --baseline_defense lrb, not lrbprojonly." >&2
    exit 2
  fi

  if [ "$defense" = "none" ]; then
    param_vals=( "" )
  elif [ -n "$BASELINE_DEFENSE" ] && [ "$defense" = "$BASELINE_DEFENSE" ] && [ -n "$BASELINE_PARAM" ]; then
    param_vals=( "$BASELINE_PARAM" )
  else
    dager_set_param_values "$defense"
  fi

  for val in "${param_vals[@]}"; do
    DEF_EXTRA=()
    if [ "$defense" = "none" ] || [ "$defense" = "dager" ]; then
      log_base="$defense"
      param=""
    else
      slug="$(dager_param_slug "$val")"
      log_base="${defense}_${slug}"
      param="$val"
      case "$defense" in
        noise|dpsgd)
          DEF_EXTRA=( --defense_noise "$val" )
          ;;
        topk)
          DEF_EXTRA=( --defense_topk_ratio "$val" )
          ;;
        compression)
          DEF_EXTRA=( --defense_n_bits "$val" )
          ;;
        soteria)
          DEF_EXTRA=( --defense_soteria_pruning_rate "$val" )
          ;;
        mixup)
          DEF_EXTRA=( --defense_mixup_alpha "$val" )
          ;;
        lrb)
          DEF_EXTRA=( --defense_lrb_keep_ratio_sensitive "$val" )
          ;;
        lrbprojonly)
          param="lrbprojonly@k=${val}"
          DEF_EXTRA=( --defense_lrb_keep_ratio_sensitive "$val" )
          ;;
      esac
    fi
    run_variant "$defense" "$log_base" "$param" "${DEF_EXTRA[@]}"
  done
done

{
  echo ""
  echo "===== COMPARISON ====="
  printf "%-28s | %-10s | %-11s | %-12s | %-14s | %-12s | %-12s | %-12s | %-15s | %-12s | %s\n" \
    "variant" "exposure" "defense" "param" "rec_token" "rouge1_fm" "rouge2_fm" "r1+r2" "last_rec_status" "total_time" "status"
  local_file=""
  for local_file in "${variant_files[@]}"; do
    variant_name="$(basename "${local_file%.txt}")"
    defense_disp="$(dager_summary_value "$local_file" "defense")"
    param_disp="$(dager_summary_value "$local_file" "defense_param_value")"
    rec_tok="$(dager_summary_value "$local_file" "rec_token_mean")"
    r1="$(dager_summary_value "$local_file" "agg_rouge1_fm")"
    r2="$(dager_summary_value "$local_file" "agg_rouge2_fm")"
    rr="$(dager_summary_value "$local_file" "agg_r1fm_r2fm")"
    last_rec="$(dager_summary_value "$local_file" "last_rec_status")"
    total_time="$(dager_summary_value "$local_file" "last_total_time")"
    status="$(dager_summary_value "$local_file" "result_status")"
    printf "%-28s | %-10s | %-11s | %-12s | %-14s | %-12s | %-12s | %-12s | %-15s | %-12s | %s\n" \
      "${variant_name}" \
      "${EXPOSURE}" \
      "${defense_disp:-?}" \
      "${param_disp:-n/a}" \
      "${rec_tok:-n/a}" \
      "${r1:-n/a}" \
      "${r2:-n/a}" \
      "${rr:-n/a}" \
      "${last_rec:-n/a}" \
      "${total_time:-n/a}" \
      "${status:-unknown}"
  done
} >>"${summary_path}"

if [ "${#variant_files[@]}" -gt 0 ]; then
  python "${SCRIPT_DIR}/collect_experiment_logs.py" "${variant_files[@]}" -o "${results_csv}" --markdown "${results_md}"
  echo "[partial-gradient] Summary: ${summary_path}" >&2
  echo "[partial-gradient] CSV: ${results_csv}" >&2
  echo "[partial-gradient] Markdown: ${results_md}" >&2
fi
