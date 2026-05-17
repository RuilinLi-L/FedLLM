#!/bin/bash
# Run PEFT/LoRA baseline sweeps through attack.py while recording unsupported variants.
# LoRA direct-generation names are eval-only shorthand:
# - dpsgd: DP-SGD-style per-example clipping + Gaussian noise, no accountant.
# - soteria: Soteria-style representation masking, not training-time Soteria.
# - mixup: manifold MixUp-style representation interpolation.
# Usage:
#   ./scripts/peft_baselines.sh DATASET BATCH_SIZE MODEL_PATH N_INPUTS --finetuned_path PATH [--lora_r R] [extra attack args...]

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

BASELINE_DEFENSE=""
BASELINE_PARAM=""
LRB_VARIANTS_RAW=""
LRB_MAIN_K="0.5"
PEFT_METHOD="lora"
PEFT_EVAL_SCOPE="dager_eval"
ADAPTIVE_ATTACK_CHECK=0
EXTRA=()

ALL_DEFENSES=( none noise dpsgd topk compression soteria mixup dager lrb lrbprojonly )
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
      --baseline_defense)
        if [ $((idx + 1)) -ge "${#RAW_EXTRA[@]}" ]; then
          echo "[dager] --baseline_defense requires a value." >&2
          exit 2
        fi
        BASELINE_DEFENSE="${RAW_EXTRA[$((idx + 1))]}"
        idx=$((idx + 2))
        ;;
      --baseline_defense=*)
        BASELINE_DEFENSE="${arg#*=}"
        idx=$((idx + 1))
        ;;
      --baseline_param)
        if [ $((idx + 1)) -ge "${#RAW_EXTRA[@]}" ]; then
          echo "[dager] --baseline_param requires a value." >&2
          exit 2
        fi
        BASELINE_PARAM="${RAW_EXTRA[$((idx + 1))]}"
        idx=$((idx + 2))
        ;;
      --baseline_param=*)
        BASELINE_PARAM="${arg#*=}"
        idx=$((idx + 1))
        ;;
      --peft_method)
        if [ $((idx + 1)) -ge "${#RAW_EXTRA[@]}" ]; then
          echo "[dager] --peft_method requires a value." >&2
          exit 2
        fi
        PEFT_METHOD="${RAW_EXTRA[$((idx + 1))]}"
        idx=$((idx + 2))
        ;;
      --peft_method=*)
        PEFT_METHOD="${arg#*=}"
        idx=$((idx + 1))
        ;;
      --lrb_variants)
        if [ $((idx + 1)) -ge "${#RAW_EXTRA[@]}" ]; then
          echo "[dager] --lrb_variants requires a value." >&2
          exit 2
        fi
        LRB_VARIANTS_RAW="${RAW_EXTRA[$((idx + 1))]}"
        idx=$((idx + 2))
        ;;
      --lrb_variants=*)
        LRB_VARIANTS_RAW="${arg#*=}"
        idx=$((idx + 1))
        ;;
      --lrb_main_k)
        if [ $((idx + 1)) -ge "${#RAW_EXTRA[@]}" ]; then
          echo "[dager] --lrb_main_k requires a value." >&2
          exit 2
        fi
        LRB_MAIN_K="${RAW_EXTRA[$((idx + 1))]}"
        idx=$((idx + 2))
        ;;
      --lrb_main_k=*)
        LRB_MAIN_K="${arg#*=}"
        idx=$((idx + 1))
        ;;
      --adaptive_attack_check)
        ADAPTIVE_ATTACK_CHECK=1
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

dager_param_slug() {
  printf '%s' "$1" | tr -c 'a-zA-Z0-9._-' '_' | sed 's/__*/_/g' | sed 's/^_\|_$//g'
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
      echo "[dager] Unknown defense for sweep: ${defense}" >&2
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
train_method=peft
peft_method=${PEFT_METHOD}
peft_eval_scope=${PEFT_EVAL_SCOPE}
peft_type=n/a
peft_target_modules=$(attack_extra_value --lora_target_modules || printf 'n/a')
peft_feedforward_modules=n/a
peft_num_virtual_tokens=$(attack_extra_value --peft_num_virtual_tokens || printf 'n/a')
peft_checkpoint_type=n/a
peft_adapter_r=n/a
peft_adapter_target_modules=n/a
peft_adapter_feedforward_modules=n/a
peft_adapter_task_type=n/a
peft_adapter_base_model=n/a
peft_adapter_peft_type=n/a
lora_r=$(attack_extra_value --lora_r || printf 'n/a')
lora_target_modules=$(attack_extra_value --lora_target_modules || printf 'n/a')
lora_checkpoint_type=n/a
lora_adapter_r=n/a
lora_adapter_target_modules=n/a
lora_adapter_feedforward_modules=n/a
lora_adapter_task_type=n/a
lora_adapter_base_model=n/a
lora_adapter_peft_type=n/a
defense=${defense}
defense_param_name=$(dager_param_name "$defense")
defense_param_value=${param:-n/a}
n_inputs_requested=${N_INPUTS}
n_inputs_completed=0
last_input_idx=n/a
last_input_time=n/a
last_total_time=n/a
last_rec_status=failed
rec_l1_mean=n/a
rec_l1_maxb_mean=n/a
rec_l2_mean=n/a
rec_token_mean=n/a
rec_maxb_token_mean=n/a
error_type=runner_error
error_message=missing_result_summary_or_process_failed
adaptive_attack=$(if printf '%s' "$log_base" | grep -q '_adaptive'; then printf 'defense_aware'; else printf 'none'; fi)
adaptive_attack_profile=$(if printf '%s' "$log_base" | grep -q '_adaptive'; then case "$defense" in topk) printf 'topk_support' ;; compression) printf 'quantization_robust' ;; lrb|lrbprojonly) printf 'projection_span' ;; *) printf 'generic_ranked_span' ;; esac; else printf 'none'; fi)
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
train_method=peft
peft_method=${PEFT_METHOD}
peft_eval_scope=${PEFT_EVAL_SCOPE}
peft_type=n/a
peft_target_modules=$(attack_extra_value --lora_target_modules || printf 'n/a')
peft_feedforward_modules=n/a
peft_num_virtual_tokens=$(attack_extra_value --peft_num_virtual_tokens || printf 'n/a')
peft_checkpoint_type=n/a
peft_adapter_r=n/a
peft_adapter_target_modules=n/a
peft_adapter_feedforward_modules=n/a
peft_adapter_task_type=n/a
peft_adapter_base_model=n/a
peft_adapter_peft_type=n/a
lora_r=$(attack_extra_value --lora_r || printf 'n/a')
lora_target_modules=$(attack_extra_value --lora_target_modules || printf 'n/a')
lora_checkpoint_type=n/a
lora_adapter_r=n/a
lora_adapter_target_modules=n/a
lora_adapter_feedforward_modules=n/a
lora_adapter_task_type=n/a
lora_adapter_base_model=n/a
lora_adapter_peft_type=n/a
defense=${defense}
defense_param_name=$(dager_param_name "$defense")
defense_param_value=${param:-n/a}
n_inputs_requested=${N_INPUTS}
n_inputs_completed=0
last_input_idx=n/a
last_input_time=n/a
last_total_time=n/a
last_rec_status=unsupported
rec_l1_mean=n/a
rec_l1_maxb_mean=n/a
rec_l2_mean=n/a
rec_token_mean=n/a
rec_maxb_token_mean=n/a
error_type=unsupported_defense
error_message=PEFT eval currently supports only defenses: none, noise, dpsgd, topk, compression, soteria, mixup, lrb, lrbprojonly
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
    echo "===== VARIANT START defense=${defense} param=${param:-n/a} dataset=${DATASET} batch=${BATCH} model=$(basename "$MODEL") start=${t_start} ====="
    printf '%s\n' "$summary_block"
    echo "===== VARIANT END end=${t_end} exit_code=${rc} ====="
    if [ -n "$tmpfile" ] && [ -f "$tmpfile" ] && [ "$rc" -ne 0 ]; then
      echo "--- last 25 lines from run output ---"
      tail -n 25 "$tmpfile"
    fi
  } >"$def_file"
}

parse_script_args

case "$PEFT_METHOD" in
  lora|ia3)
    PEFT_EVAL_SCOPE="dager_eval"
    ;;
  prefix)
    echo "[dager] --peft_method prefix is training-only in v1 and excluded from DAGER/partial-gradient eval matrices." >&2
    exit 2
    ;;
  adapter)
    echo "[dager] --peft_method adapter is v2 planned and not part of v1 PEFT DAGER/partial eval." >&2
    exit 2
    ;;
  *)
    echo "[dager] --peft_method must be lora or ia3 for DAGER PEFT eval." >&2
    exit 2
    ;;
esac

if [ -n "$BASELINE_DEFENSE" ]; then
  case "$BASELINE_DEFENSE" in
    none|noise|dpsgd|topk|compression|soteria|mixup|dager|lrb|lrbprojonly)
      ;;
    *)
      echo "[dager] Unsupported --baseline_defense: ${BASELINE_DEFENSE}" >&2
      exit 2
      ;;
  esac
fi

if [ -n "$BASELINE_PARAM" ] && [ -z "$BASELINE_DEFENSE" ]; then
  echo "[dager] --baseline_param requires --baseline_defense." >&2
  exit 2
fi

if [ "$BASELINE_DEFENSE" = "none" ] && [ -n "$BASELINE_PARAM" ]; then
  echo "[dager] --baseline_defense none cannot be combined with --baseline_param." >&2
  exit 2
fi

if [ "$BASELINE_DEFENSE" = "dager" ] && [ -n "$BASELINE_PARAM" ]; then
  echo "[dager] --baseline_defense dager does not accept --baseline_param in peft_baselines.sh." >&2
  exit 2
fi

if [ -n "$LRB_VARIANTS_RAW" ] && [ "$BASELINE_DEFENSE" != "lrb" ]; then
  echo "[dager] --lrb_variants can only be used with --baseline_defense lrb." >&2
  exit 2
fi

if [ -n "$LRB_VARIANTS_RAW" ] && [ -n "$BASELINE_PARAM" ]; then
  echo "[dager] --lrb_variants cannot be combined with --baseline_param." >&2
  exit 2
fi

LRB_VARIANTS=()
if [ -n "$LRB_VARIANTS_RAW" ]; then
  IFS=',' read -r -a LRB_VARIANTS <<< "$LRB_VARIANTS_RAW"
  for variant in "${LRB_VARIANTS[@]}"; do
    if ! is_known_lrb_variant "$variant"; then
      echo "[dager] unknown LRB variant: ${variant}" >&2
      exit 2
    fi
  done
fi

if ! has_attack_extra_flag "--finetuned_path"; then
  echo "[dager] peft_baselines.sh requires --finetuned_path PATH to a PEFT adapter directory or LoRA .pt/.pth checkpoint." >&2
  exit 2
fi

selected_defenses=()
if [ -n "$BASELINE_DEFENSE" ]; then
  if [ "$BASELINE_DEFENSE" = "none" ]; then
    selected_defenses=( none )
  else
    selected_defenses=( none "$BASELINE_DEFENSE" )
  fi
else
  selected_defenses=( "${ALL_DEFENSES[@]}" )
fi

run_dir=""
summary_path=""
results_csv=""
results_md=""
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
run_dir="${log_root}/peft_baselines_${safe_ds}_b${BATCH}_${safe_model}${focus_suffix}_${stamp}"
mkdir -p "$run_dir" || true
summary_path="${run_dir}/summary.txt"
results_csv="${run_dir}/results.csv"
results_md="${run_dir}/results.md"
header_line="===== run start $(date '+%Y-%m-%d %H:%M:%S') tag=peft_baselines argv: $* ====="
{
  echo "$header_line"
  echo "focus_baseline_defense=${BASELINE_DEFENSE:-all}"
  echo "focus_baseline_param=${BASELINE_PARAM:-all}"
  echo "lrb_variants=${LRB_VARIANTS_RAW:-none}"
  echo "lrb_main_k=${LRB_MAIN_K}"
  echo "selected_defenses=${selected_defenses[*]}"
  echo "train_method=peft"
  echo "peft_method=${PEFT_METHOD}"
  echo "peft_eval_scope=${PEFT_EVAL_SCOPE}"
  echo "supported_peft_defenses=${SUPPORTED_PEFT_DEFENSES[*]}"
  echo "lora_eval_semantics=dpsgd=DP-SGD-style_no_accountant;soteria=Soteria-style_representation_masking_eval_only;mixup=manifold_MixUp-style_representation_interpolation"
} >"${run_dir}/_run_header.txt"
cp "${run_dir}/_run_header.txt" "${summary_path}"
echo "[dager] Run directory: ${run_dir}" >&2
echo "[dager] Variant summaries: ${run_dir}/<defense>_<param>.txt" >&2
echo "[dager] Summary: ${summary_path}" >&2

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
)

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

  if ! is_supported_peft_defense "$defense"; then
    summary_block="$(dager_unsupported_summary_block "$defense" "$param")"
    t_end=$(date '+%Y-%m-%d %H:%M:%S')
    write_variant_summary_file "$def_file" "$summary_block" "$defense" "$param" "$log_base" "$t_start" "$t_end" 0
    {
      echo ""
      echo "========== variant=${log_base} defense=${defense} param=${param:-n/a} =========="
      printf '%s\n' "$summary_block"
    } >>"${summary_path}"
    variant_files+=( "$def_file" )
    echo "[dager] Marked unsupported PEFT variant: ${def_file}" >&2
    return 0
  fi

  local tmpfile
  tmpfile=$(mktemp)
  set +e
  "${BASE[@]}" "${EXTRA[@]}" --train_method peft --peft_method "$PEFT_METHOD" --defense "$defense" "${def_extra[@]}" 2>&1 | tee "$tmpfile"
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
    echo "========== variant=${log_base} defense=${defense} param=${param:-n/a} =========="
    printf '%s\n' "$summary_block"
  } >>"${summary_path}"

  variant_files+=( "$def_file" )
  echo "[dager] Wrote variant summary: ${def_file}" >&2
  rm -f "$tmpfile"
}

for defense in "${selected_defenses[@]}"; do
  if [ -n "$BASELINE_DEFENSE" ] && [ "$defense" != "none" ]; then
    echo "========== defense=${defense} (focused) =========="
  else
    echo "========== defense=${defense} (sweep) =========="
  fi

  if [ "$defense" = "lrb" ] && [ "${#LRB_VARIANTS[@]}" -gt 0 ]; then
    for variant in "${LRB_VARIANTS[@]}"; do
      log_base="lrb_${variant}_k$(dager_param_slug "$LRB_MAIN_K")"
      param="${variant}@k=${LRB_MAIN_K}"
      DEF_EXTRA=(
        --defense_lrb_preset "$variant"
        --defense_lrb_keep_ratio_sensitive "$LRB_MAIN_K"
      )
      if [ "$ADAPTIVE_ATTACK_CHECK" = "1" ]; then
        log_base="${log_base}_adaptive"
        DEF_EXTRA+=( --adaptive_attack defense_aware )
      fi
      run_variant "$defense" "$log_base" "$param" "${DEF_EXTRA[@]}"
    done
    continue
  fi

  if [ "$defense" = "lrbprojonly" ] && [ "${#LRB_VARIANTS[@]}" -gt 0 ]; then
    echo "[dager] --lrb_variants applies to --baseline_defense lrb, not lrbprojonly." >&2
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
    if [ "$ADAPTIVE_ATTACK_CHECK" = "1" ] && [ "$defense" != "none" ] && [ "$defense" != "dager" ]; then
      log_base="${log_base}_adaptive"
      DEF_EXTRA+=( --adaptive_attack defense_aware )
      if [ "$defense" = "noise" ] || [ "$defense" = "dpsgd" ]; then
        DEF_EXTRA+=( --defense_adaptive_decoding )
      fi
    fi
    run_variant "$defense" "$log_base" "$param" "${DEF_EXTRA[@]}"
  done
done

{
  echo ""
  echo "===== COMPARISON ====="
  printf "%-28s | %-11s | %-10s | %-14s | %-14s | %-12s | %-12s | %-12s | %-15s | %-12s | %s\n" \
    "variant" "defense" "param" "rec_token" "rec_maxb_token" "rouge1_fm" "rouge2_fm" "r1+r2" "last_rec_status" "total_time" "status"
  local_file=""
  for local_file in "${variant_files[@]}"; do
    variant_name="$(basename "${local_file%.txt}")"
    defense_disp="$(dager_summary_value "$local_file" "defense")"
    param_disp="$(dager_summary_value "$local_file" "defense_param_value")"
    rec_tok="$(dager_summary_value "$local_file" "rec_token_mean")"
    rec_maxb="$(dager_summary_value "$local_file" "rec_maxb_token_mean")"
    r1="$(dager_summary_value "$local_file" "agg_rouge1_fm")"
    r2="$(dager_summary_value "$local_file" "agg_rouge2_fm")"
    rr="$(dager_summary_value "$local_file" "agg_r1fm_r2fm")"
    last_rec="$(dager_summary_value "$local_file" "last_rec_status")"
    total_time="$(dager_summary_value "$local_file" "last_total_time")"
    status="$(dager_summary_value "$local_file" "result_status")"
    printf "%-28s | %-11s | %-10s | %-14s | %-14s | %-12s | %-12s | %-12s | %-15s | %-12s | %s\n" \
      "${variant_name}" \
      "${defense_disp:-?}" \
      "${param_disp:-n/a}" \
      "${rec_tok:-n/a}" \
      "${rec_maxb:-n/a}" \
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
  echo "[dager] Summary: ${summary_path}" >&2
  echo "[dager] CSV: ${results_csv}" >&2
  echo "[dager] Markdown: ${results_md}" >&2
fi
