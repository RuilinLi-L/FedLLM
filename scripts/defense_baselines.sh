#!/bin/bash
# Run DAGER attack with the defense baselines listed in FL-LLM.md.
# Notes:
# - dpsgd uses faithful DP-SGD semantics: per-example clipping, mean aggregation, then noise.
# - soteria prunes the highest-scoring classifier-input representation dimensions.
# - Historical dpsgd/soteria logs from older revisions are not directly comparable.
# Usage:
#   ./scripts/defense_baselines.sh [DATASET] [BATCH_SIZE] [MODEL_PATH] [N_INPUTS] [extra attack args...]
# Examples:
#   ./scripts/defense_baselines.sh sst2 2 gpt2 3 --finetuned_path ./models/gpt2-ft-rt
#   ./scripts/defense_baselines.sh sst2 2 gpt2 3 --baseline_defense noise --finetuned_path ./models/gpt2-ft-rt
#   ./scripts/defense_baselines.sh sst2 2 gpt2 3 --baseline_defense soteria --baseline_param 60 --finetuned_path ./models/gpt2-ft-rt
#
# Script-only flags handled here and not forwarded to attack.py:
#   --baseline_defense <none|noise|dpsgd|topk|compression|soteria|mixup|lrb>
#   --baseline_param <value>
#
# Logging:
# - Creates a run directory under log/runs/ by default.
# - Writes one summary file per variant, plus summary.txt, results.csv, and results.md.
# - Full stdout/stderr still stream to the terminal.

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
EXTRA=()

ALL_DEFENSES=( none noise dpsgd topk compression soteria mixup lrb )

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
      printf 'defense_lrb_keep_ratio_sensitive'
      ;;
    *)
      printf 'n/a'
      ;;
  esac
}

dager_set_param_values() {
  local defense="$1"
  case "$defense" in
    none)
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
    lrb)
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
summary_version=1
result_status=failed
dataset=${DATASET}
split=val
task=seq_class
model_path=${MODEL}
finetuned_path=n/a
batch_size=${BATCH}
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
script_variant=${log_base}
script_start_time=${t_start}
script_end_time=${t_end}
script_exit_code=${rc}
===== RESULT SUMMARY END =====
EOF
}

parse_script_args

if [ -n "$BASELINE_DEFENSE" ]; then
  case "$BASELINE_DEFENSE" in
    none|noise|dpsgd|topk|compression|soteria|mixup|lrb)
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

if ! has_attack_extra_flag "--finetuned_path"; then
  cat >&2 <<EOF
[dager] defense_baselines.sh uses --task seq_class with backbone model ids such as ${MODEL}.
[dager] For credible baseline results, pass --finetuned_path <trained checkpoint>.
[dager] Example:
[dager]   ./scripts/defense_baselines.sh ${DATASET} ${BATCH} ${MODEL} ${N_INPUTS} --finetuned_path ./models/gpt2-ft-rt
EOF
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
if [ -z "${DAGER_NO_AUTO_LOG:-}" ]; then
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
  run_dir="${log_root}/defense_baselines_${safe_ds}_b${BATCH}_${safe_model}${focus_suffix}_${stamp}"
  mkdir -p "$run_dir" || true
  summary_path="${run_dir}/summary.txt"
  results_csv="${run_dir}/results.csv"
  results_md="${run_dir}/results.md"
  tag=$(printf '%s' "defense_baselines" | tr -c 'a-zA-Z0-9._-' '_')
  header_line="===== run start $(date '+%Y-%m-%d %H:%M:%S') tag=${tag} argv: $* ====="
  {
    echo "$header_line"
    echo "focus_baseline_defense=${BASELINE_DEFENSE:-all}"
    echo "focus_baseline_param=${BASELINE_PARAM:-all}"
    echo "selected_defenses=${selected_defenses[*]}"
  } >"${run_dir}/_run_header.txt"
  {
    echo "$header_line"
    echo "focus_baseline_defense=${BASELINE_DEFENSE:-all}"
    echo "focus_baseline_param=${BASELINE_PARAM:-all}"
    echo "selected_defenses=${selected_defenses[*]}"
  } >"${summary_path}"
  echo "[dager] Run directory: ${run_dir}" >&2
  echo "[dager] Variant summaries: ${run_dir}/<defense>_<param>.txt" >&2
  echo "[dager] Summary: ${summary_path}" >&2
fi

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

  echo "---------- ${log_base} ----------"

  if [ -n "$run_dir" ]; then
    local tmpfile
    local t_start
    local t_end
    local rc
    local def_file
    local summary_block

    tmpfile=$(mktemp)
    t_start=$(date '+%Y-%m-%d %H:%M:%S')
    set +e
    "${BASE[@]}" --defense "$defense" "${def_extra[@]}" "${EXTRA[@]}" 2>&1 | tee "$tmpfile"
    rc=${PIPESTATUS[0]}
    set -e
    t_end=$(date '+%Y-%m-%d %H:%M:%S')

    if dager_has_result_summary "$tmpfile"; then
      summary_block="$(dager_result_summary_block_from_file "$tmpfile")"
    else
      summary_block="$(dager_fallback_summary_block "$defense" "$param" "$log_base" "$rc" "$t_start" "$t_end")"
    fi

    def_file="${run_dir}/${log_base}.txt"
    {
      echo "===== VARIANT START defense=${defense} param=${param:-n/a} dataset=${DATASET} batch=${BATCH} model=$(basename "$MODEL") start=${t_start} ====="
      printf '%s\n' "$summary_block"
      echo "===== VARIANT END end=${t_end} exit_code=${rc} ====="
      if [ "$rc" -ne 0 ]; then
        echo "--- last 25 lines from run output ---"
        tail -n 25 "$tmpfile"
      fi
    } >"$def_file"

    {
      echo ""
      echo "========== variant=${log_base} defense=${defense} param=${param:-n/a} =========="
      printf '%s\n' "$summary_block"
    } >>"${summary_path}"

    variant_files+=( "$def_file" )
    echo "[dager] Wrote variant summary: ${def_file}" >&2
    rm -f "$tmpfile"
  else
    "${BASE[@]}" --defense "$defense" "${def_extra[@]}" "${EXTRA[@]}"
  fi
}

for defense in "${selected_defenses[@]}"; do
  if [ -n "$BASELINE_DEFENSE" ] && [ "$defense" != "none" ]; then
    echo "========== defense=${defense} (focused) =========="
  else
    echo "========== defense=${defense} (sweep) =========="
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
    if [ "$defense" = "none" ]; then
      log_base="none"
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
      esac
    fi
    run_variant "$defense" "$log_base" "$param" "${DEF_EXTRA[@]}"
  done
done

if [ -n "$run_dir" ]; then
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
fi
