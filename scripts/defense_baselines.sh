#!/bin/bash
# Run DAGER attack with each defense baseline (FL-LLM.md).
# Usage: ./scripts/defense_baselines.sh [DATASET] [BATCH_SIZE] [MODEL_PATH] [N_INPUTS] [extra python args...]
# Example: ./scripts/defense_baselines.sh sst2 2 gpt2 3
#
# Each defense (except none) sweeps a strength parameter so logs show a privacy–utility curve,
# not only a single aggressive default. Per-variant logs: {defense}_{param_slug}.txt
# For LLaMA + Soteria, add e.g. --defense_soteria_sample_dims 256 to extra args (applies to every run).
#
# Logging: by default creates a run directory under log/runs/ with per-variant metrics-only .txt files,
# a summary.txt (compatible with collect_experiment_logs.py), and _run_header.txt.
# Full stdout/stderr still go to the terminal. One variant failing (e.g. OOM) does not stop the rest.
# Set DAGER_NO_AUTO_LOG=1 to disable file logging. DAGER_LOG_DIR overrides the log root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAGER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$DAGER_ROOT" || exit 1

DATASET="${1:-sst2}"
BATCH="${2:-2}"
MODEL="${3:-gpt2}"
N_INPUTS="${4:-3}"
EXTRA=()
if [ "$#" -gt 4 ]; then
  EXTRA=( "${@:5}" )
fi

# Safe filename fragment from a parameter value (e.g. 1e-6 -> 1e-6, 0.05 -> 0_05)
dager_param_slug() {
  printf '%s' "$1" | tr -c 'a-zA-Z0-9._-' '_' | sed 's/__*/_/g' | sed 's/^_\|_$//g'
}

# Extract final Rec Status, final Rec L1, last [Aggregate metrics]: block, last timing line, Done with all. (attack.py output)
dager_extract_attack_metrics() {
  awk '
    /^Rec Status:/ { rec_status = $0 }
    /^Rec L1:/ { rec = $0 }
    /\[Aggregate metrics\]:/ { agg = ""; in_agg = 1 }
    in_agg { agg = agg $0 "\n" }
    /^r1fm\+r2fm/ { in_agg = 0 }
    /^input #[0-9]+ time:/ { timeln = $0 }
    /^Done with all/ { done = $0 }
    END {
      if (rec_status != "") print rec_status
      if (rec != "") print rec
      if (agg != "") printf "%s", agg
      if (timeln != "") print timeln
      if (done != "") print done
    }
  ' "$1"
}

run_dir=""
if [ -z "${DAGER_NO_AUTO_LOG:-}" ]; then
  stamp=$(date +%Y%m%d_%H%M%S)
  log_root="${DAGER_LOG_DIR:-log/runs}"
  mkdir -p "$log_root" || true
  safe_ds=$(printf '%s' "$DATASET" | tr -c 'a-zA-Z0-9._-' '_')
  safe_model=$(printf '%s' "$(basename "$MODEL")" | tr -c 'a-zA-Z0-9._-' '_')
  run_dir="${log_root}/defense_baselines_${safe_ds}_b${BATCH}_${safe_model}_${stamp}"
  mkdir -p "$run_dir" || true
  tag=$(printf '%s' "defense_baselines" | tr -c 'a-zA-Z0-9._-' '_')
  header_line="===== run start $(date '+%Y-%m-%d %H:%M:%S') tag=${tag} argv: $* ====="
  echo "$header_line" >"${run_dir}/_run_header.txt"
  echo "$header_line" >"${run_dir}/summary.txt"
  echo "[dager] Run directory: ${run_dir}" >&2
  echo "[dager] Per-variant metrics: ${run_dir}/<defense>_<param>.txt ; summary: ${run_dir}/summary.txt" >&2
  echo "[dager] CSV: python ${SCRIPT_DIR}/collect_experiment_logs.py ${run_dir}/summary.txt" >&2
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

# Record variant basename for comparison table (space-separated, appended each run)
comparison_variants=()

run_variant() {
  local defense="$1"
  local log_base="$2"
  local param="$3"
  shift 3
  local def_extra=( "$@" )
  local summary_title=""

  if [[ "$defense" == "none" ]]; then
    summary_title="========== defense=none =========="
  else
    summary_title="========== defense=${defense} param=${param} =========="
  fi

  echo "---------- ${log_base} ----------"

  if [ -n "$run_dir" ]; then
    comparison_variants+=( "$log_base" )
  fi

  if [ -n "$run_dir" ]; then
    local tmpfile
    local t_start
    local t_end
    local rc
    local def_file

    tmpfile=$(mktemp)
    t_start=$(date '+%Y-%m-%d %H:%M:%S')
    set +e
    "${BASE[@]}" --defense "$defense" "${def_extra[@]}" "${EXTRA[@]}" 2>&1 | tee "$tmpfile"
    rc=${PIPESTATUS[0]}
    set -e
    t_end=$(date '+%Y-%m-%d %H:%M:%S')

    def_file="${run_dir}/${log_base}.txt"
    {
      echo "===== defense=${defense} param=${param:-n/a} dataset=${DATASET} batch=${BATCH} model=$(basename "$MODEL") start=${t_start} ====="
      if [ "$rc" -eq 0 ]; then
        dager_extract_attack_metrics "$tmpfile"
      else
        echo "(run failed before completion; partial output below if any)"
        dager_extract_attack_metrics "$tmpfile"
      fi
      echo "===== end=${t_end} exit_code=${rc} ====="
      if [ "$rc" -ne 0 ]; then
        echo "--- last 25 lines from run output ---"
        tail -n 25 "$tmpfile"
      fi
    } >"$def_file"

    {
      echo ""
      echo "$summary_title"
      if [ "$rc" -eq 0 ] && grep -q '^Done with all' "$tmpfile" 2>/dev/null; then
        dager_extract_attack_metrics "$tmpfile"
      else
        echo "FAILED (exit_code=${rc})"
        tail -n 25 "$tmpfile"
      fi
    } >>"${run_dir}/summary.txt"

    rm -f "$tmpfile"
  else
    "${BASE[@]}" --defense "$defense" "${def_extra[@]}" "${EXTRA[@]}"
  fi
}

for defense in none noise dpsgd topk compression soteria mixup; do
  echo "========== defense=${defense} (sweep) =========="

  case "$defense" in
    none)
      param_vals=( "" ) # single run, no extra defense args
      ;;
    noise)
      param_vals=( 1e-6 1e-5 1e-4 5e-4 1e-3 )
      ;;
    dpsgd)
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
    *)
      param_vals=( "" )
      ;;
  esac

  for val in "${param_vals[@]}"; do
    DEF_EXTRA=()
    if [[ "$defense" == "none" ]]; then
      log_base="none"
      param=""
    else
      slug=$(dager_param_slug "$val")
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
      esac
    fi
    run_variant "$defense" "$log_base" "$param" "${DEF_EXTRA[@]}"
  done
done

echo "========== defense=dager (sweep) =========="
run_variant "dager" "dager_basis_1e-3" "basis_1e-3" \
  --defense_dager_basis_perturb --defense_dager_basis_noise_scale 0.001
run_variant "dager" "dager_basis_5e-3" "basis_5e-3" \
  --defense_dager_basis_perturb --defense_dager_basis_noise_scale 0.005
run_variant "dager" "dager_basis_1e-2" "basis_1e-2" \
  --defense_dager_basis_perturb --defense_dager_basis_noise_scale 0.01
run_variant "dager" "dager_slice_first_1" "slice_first_1" \
  --no_defense_dager_basis_perturb \
  --defense_dager_gradient_slicing --defense_dager_slice_first_n 1
run_variant "dager" "dager_slice_random_0.3" "slice_random_0.3" \
  --no_defense_dager_basis_perturb \
  --defense_dager_gradient_slicing --defense_dager_random_slice --defense_dager_slice_prob 0.3
run_variant "dager" "dager_combined_light" "combined_light" \
  --defense_dager_basis_perturb --defense_dager_basis_noise_scale 0.01 \
  --defense_dager_gradient_slicing --defense_dager_random_slice --defense_dager_slice_prob 0.5
run_variant "dager" "dager_combined_full" "combined_full" \
  --defense_dager_basis_perturb --defense_dager_basis_noise_scale 0.01 \
  --defense_dager_offset_embedding --defense_dager_offset_scale 0.005 \
  --defense_dager_gradient_slicing --defense_dager_random_slice --defense_dager_slice_prob 0.3 \
  --defense_dager_rank_limit

if [ -n "$run_dir" ]; then
  {
    echo ""
    echo "===== COMPARISON ====="
    printf "%-28s | %10s | %8s | %8s | %8s | %s\n" "variant" "param" "r1_fm" "r2_fm" "r1+r2" "status"
    for log_base in "${comparison_variants[@]}"; do
      f="${run_dir}/${log_base}.txt"
      status="FAILED"
      r1="-"
      r2="-"
      rr="-"
      param_disp="-"
      if [ -f "$f" ]; then
        param_disp=$(grep '^===== defense=' "$f" | head -1 | sed -n 's/.*param=\([^[:space:]]*\).*/\1/p')
        param_disp="${param_disp:-?}"
        ec=$(grep '^===== end=' "$f" | tail -1 | sed -n 's/.*exit_code=\([0-9]*\).*/\1/p')
        if [ "$ec" = "0" ] && grep -q '^Done with all' "$f" 2>/dev/null; then
          status="ok"
          r1=$(grep '^rouge1' "$f" | tail -1 | sed -n 's/.*fm:[[:space:]]*\([0-9.]*\).*/\1/p')
          r2=$(grep '^rouge2' "$f" | tail -1 | sed -n 's/.*fm:[[:space:]]*\([0-9.]*\).*/\1/p')
          rr=$(grep '^r1fm+r2fm' "$f" | tail -1 | sed -n 's/.*=[[:space:]]*\([0-9.]*\).*/\1/p')
          r1="${r1:--}"
          r2="${r2:--}"
          rr="${rr:--}"
        fi
      fi
      printf "%-28s | %10s | %8s | %8s | %8s | %s\n" "$log_base" "$param_disp" "$r1" "$r2" "$rr" "$status"
    done
  } >>"${run_dir}/summary.txt"
fi
