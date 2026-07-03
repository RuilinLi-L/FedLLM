#!/bin/bash
# Run partial Transformer gradient matching baselines through attack_partial_gradient.py.
# Usage:
#   ./scripts/ptg_baselines.sh DATASET BATCH_SIZE MODEL_PATH N_INPUTS --finetuned_path PATH [extra PTG args...]
# Focus one row:
#   ./scripts/ptg_baselines.sh sst2 1 bert-base-uncased 1 --exposure query_only --baseline_defense none --finetuned_path PATH
# Script-only flags:
#   --exposure <first2|mid2|last2|qkv_only|query_only|key_only|value_only|attn_out_only|attn_only|ffn_in_only|ffn_out_only|ffn_only|classifier_only|all>[,...]
#   --baseline_defense <none|proj_only|topk|compression|noise|dpsgd|soteria|mixup|lrbprojonly|signed_bottleneck>
#   --baseline_param <value>
#   --full_sweep
#   --lrb_main_k <value>
#   --python <python executable>

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR_PART="${SCRIPT_PATH%/*}"
if [ "$SCRIPT_DIR_PART" = "$SCRIPT_PATH" ]; then
  SCRIPT_DIR_PART="."
fi
SCRIPT_DIR="$(cd "$SCRIPT_DIR_PART" && pwd)"
FEDLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$FEDLLM_ROOT" || exit 1

DATASET="${1:-sst2}"
BATCH="${2:-1}"
MODEL="${3:-bert-base-uncased}"
N_INPUTS="${4:-1}"
RAW_EXTRA=()
if [ "$#" -gt 4 ]; then
  RAW_EXTRA=( "${@:5}" )
fi

EXPOSURE_RAW=""
BASELINE_DEFENSE=""
BASELINE_PARAM=""
FULL_SWEEP=0
LRB_MAIN_K="0.5"
PYTHON_BIN="${PYTHON:-python}"
EXTRA=()

ptg_slug() {
  printf '%s' "$1" | tr -c 'a-zA-Z0-9._-' '_' | sed 's/__*/_/g' | sed 's/^_\|_$//g'
}

has_extra_flag() {
  local flag="$1"
  local idx=0
  while [ "$idx" -lt "${#EXTRA[@]}" ]; do
    local arg="${EXTRA[$idx]}"
    if [ "$arg" = "$flag" ] || [[ "$arg" == "${flag}="* ]]; then
      return 0
    fi
    idx=$((idx + 1))
  done
  return 1
}

parse_script_args() {
  local idx=0
  while [ "$idx" -lt "${#RAW_EXTRA[@]}" ]; do
    local arg="${RAW_EXTRA[$idx]}"
    case "$arg" in
      --exposure)
        EXPOSURE_RAW="${RAW_EXTRA[$((idx + 1))]:-}"
        idx=$((idx + 2))
        ;;
      --exposure=*)
        EXPOSURE_RAW="${arg#*=}"
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
      --full_sweep)
        FULL_SWEEP=1
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
      --python)
        PYTHON_BIN="${RAW_EXTRA[$((idx + 1))]:-python}"
        idx=$((idx + 2))
        ;;
      --python=*)
        PYTHON_BIN="${arg#*=}"
        idx=$((idx + 1))
        ;;
      *)
        EXTRA+=( "$arg" )
        idx=$((idx + 1))
        ;;
    esac
  done
}

append_csv_values() {
  local raw="$1"
  local -n out_ref="$2"
  IFS=',' read -ra parts <<<"$raw"
  local item
  for item in "${parts[@]}"; do
    if [ -n "$item" ]; then
      out_ref+=( "$item" )
    fi
  done
}

exposure_args() {
  local exposure="$1"
  case "$exposure" in
    all)
      printf '%s\n' --gradient_layer_subset all --gradient_param_filter all
      ;;
    first2|mid2|last2)
      printf '%s\n' --gradient_layer_subset "$exposure" --gradient_param_filter all
      ;;
    qkv_only|query_only|key_only|value_only|attn_out_only|attn_only|ffn_in_only|ffn_out_only|ffn_only|classifier_only)
      printf '%s\n' --gradient_layer_subset all --gradient_param_filter "$exposure"
      ;;
    *)
      echo "[ptg] unsupported --exposure: ${exposure}" >&2
      return 2
      ;;
  esac
}

defense_args() {
  local defense="$1"
  local param="${2:-}"
  case "$defense" in
    none)
      printf '%s\n' --defense none
      ;;
    proj_only)
      param="${param:-$LRB_MAIN_K}"
      printf '%s\n' --defense lrb --defense_lrb_preset proj_only --defense_lrb_keep_ratio_sensitive "$param"
      ;;
    lrbprojonly)
      param="${param:-$LRB_MAIN_K}"
      printf '%s\n' --defense lrbprojonly --defense_lrb_keep_ratio_sensitive "$param"
      ;;
    topk)
      param="${param:-0.1}"
      printf '%s\n' --defense topk --defense_topk_ratio "$param"
      ;;
    compression)
      param="${param:-8}"
      printf '%s\n' --defense compression --defense_n_bits "$param"
      ;;
    noise|dpsgd)
      param="${param:-1e-5}"
      printf '%s\n' --defense "$defense" --defense_noise "$param"
      ;;
    soteria)
      param="${param:-60}"
      printf '%s\n' --defense soteria --defense_soteria_pruning_rate "$param"
      ;;
    mixup)
      param="${param:-1.0}"
      printf '%s\n' --defense mixup --defense_mixup_alpha "$param"
      ;;
    signed_bottleneck)
      param="${param:-0.99}"
      printf '%s\n' --defense signed_bottleneck --defense_lrb_preset signed_bottleneck --defense_lrb_keep_ratio_sensitive "$param"
      ;;
    *)
      echo "[ptg] unsupported --baseline_defense: ${defense}" >&2
      return 2
      ;;
  esac
}

defense_param_label() {
  local defense="$1"
  local param="${2:-}"
  case "$defense" in
    none) printf 'n/a' ;;
    proj_only) printf 'proj_only@k=%s' "${param:-$LRB_MAIN_K}" ;;
    lrbprojonly) printf 'lrbprojonly@k=%s' "${param:-$LRB_MAIN_K}" ;;
    topk) printf 'topk@%s' "${param:-0.1}" ;;
    compression) printf 'compression@%sbit' "${param:-8}" ;;
    noise|dpsgd) printf '%s@%s' "$defense" "${param:-1e-5}" ;;
    soteria) printf 'soteria@%s' "${param:-60}" ;;
    mixup) printf 'mixup@%s' "${param:-1.0}" ;;
    signed_bottleneck) printf 'signed_bottleneck@k=%s' "${param:-0.99}" ;;
    *) printf '%s@%s' "$defense" "${param:-n/a}" ;;
  esac
}

parse_script_args

if ! has_extra_flag "--finetuned_path"; then
  echo "[ptg] pass --finetuned_path PATH; attack_partial_gradient.py requires a seq_class checkpoint." >&2
  exit 2
fi

exposures=()
if [ -n "$EXPOSURE_RAW" ]; then
  append_csv_values "$EXPOSURE_RAW" exposures
elif [ "$FULL_SWEEP" = "1" ]; then
  exposures=( first2 qkv_only mid2 last2 query_only key_only value_only attn_out_only attn_only ffn_in_only ffn_out_only ffn_only classifier_only )
else
  exposures=( first2 qkv_only mid2 last2 )
fi

defenses=()
if [ -n "$BASELINE_DEFENSE" ]; then
  defenses=( "$BASELINE_DEFENSE" )
else
  defenses=( none proj_only topk compression )
fi

stamp=$(date +%Y%m%d_%H%M%S)
log_root="${FEDLLM_LOG_DIR:-log/runs}"
mkdir -p "$log_root" || true
safe_ds=$(ptg_slug "$DATASET")
model_base="${MODEL##*/}"
safe_model=$(ptg_slug "$model_base")
focus_suffix=""
if [ -n "$EXPOSURE_RAW" ]; then
  focus_suffix="${focus_suffix}_$(ptg_slug "$EXPOSURE_RAW")"
fi
if [ -n "$BASELINE_DEFENSE" ]; then
  focus_suffix="${focus_suffix}_focus_$(ptg_slug "$BASELINE_DEFENSE")"
  if [ -n "$BASELINE_PARAM" ]; then
    focus_suffix="${focus_suffix}_$(ptg_slug "$BASELINE_PARAM")"
  fi
fi
run_dir="${log_root}/ptg_${safe_ds}_b${BATCH}_${safe_model}${focus_suffix}_${stamp}"
mkdir -p "$run_dir" || true
summary_path="${run_dir}/summary.txt"
results_csv="${run_dir}/results.csv"
results_md="${run_dir}/results.md"

{
  echo "===== run start $(date '+%Y-%m-%d %H:%M:%S') tag=ptg_baselines argv: $* ====="
  echo "attack=partial_transformer_gradients"
  echo "partial_attack_variant=ptg_gradient_matching"
  echo "dataset=${DATASET}"
  echo "batch_size=${BATCH}"
  echo "model_path=${MODEL}"
  echo "n_inputs=${N_INPUTS}"
  echo "exposures=${exposures[*]}"
  echo "defenses=${defenses[*]}"
  echo "lrb_main_k=${LRB_MAIN_K}"
} >"$summary_path"

echo "[ptg] Run directory: ${run_dir}" >&2
echo "[ptg] Summary: ${summary_path}" >&2

BASE=(
  "$PYTHON_BIN" attack_partial_gradient.py
  --dataset "$DATASET"
  --split val
  --n_inputs "$N_INPUTS"
  --batch_size "$BATCH"
  --model_path "$MODEL"
  --task seq_class
  --cache_dir ./models_cache
  --device cuda
  --attn_implementation eager
)

variant_files=()
for exposure in "${exposures[@]}"; do
  exp_text="$(exposure_args "$exposure")" || exit 2
  mapfile -t EXP_ARGS <<<"$exp_text"
  for defense in "${defenses[@]}"; do
    def_text="$(defense_args "$defense" "$BASELINE_PARAM")" || exit 2
    mapfile -t DEF_ARGS <<<"$def_text"
    param_label="$(defense_param_label "$defense" "$BASELINE_PARAM")"
    log_base="$(ptg_slug "${exposure}_${defense}_${param_label}")"
    def_file="${run_dir}/${log_base}.txt"
    echo "---------- exposure=${exposure} defense=${defense} param=${param_label} ----------"
    {
      echo "===== VARIANT START exposure=${exposure} defense=${defense} param=${param_label} ====="
      echo "command=${BASE[*]} ${EXTRA[*]} ${EXP_ARGS[*]} ${DEF_ARGS[*]}"
    } >"$def_file"
    set +e
    "${BASE[@]}" "${EXTRA[@]}" "${EXP_ARGS[@]}" "${DEF_ARGS[@]}" 2>&1 | tee -a "$def_file"
    rc=${PIPESTATUS[0]}
    set -e
    echo "===== VARIANT END exposure=${exposure} defense=${defense} param=${param_label} exit_code=${rc} =====" >>"$def_file"
    {
      echo ""
      echo "========== exposure=${exposure} defense=${defense} param=${param_label} exit_code=${rc} =========="
      cat "$def_file"
    } >>"$summary_path"
    variant_files+=( "$def_file" )
  done
done

set +e
"$PYTHON_BIN" scripts/collect_experiment_logs.py "${variant_files[@]}" -o "$results_csv" --markdown "$results_md" >/dev/null 2>&1
collect_rc=$?
set -e
if [ "$collect_rc" -eq 0 ]; then
  echo "[ptg] Results CSV: ${results_csv}" >&2
  echo "[ptg] Results Markdown: ${results_md}" >&2
else
  echo "[ptg] Log collection failed with exit code ${collect_rc}; raw logs remain in ${run_dir}." >&2
fi

echo "[ptg] Completed ${#variant_files[@]} variants." >&2
