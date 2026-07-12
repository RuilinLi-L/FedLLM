#!/bin/bash
# Run the formal PEFTLeak image-side downstream utility matrix.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$ROOT" || exit 1

PROFILE="${PROFILE:-core}"
case "$PROFILE" in
  core|full) ;;
  *) echo "PROFILE must be core or full; got ${PROFILE}." >&2; exit 2 ;;
esac

read -r -a SEED_LIST <<< "${SEEDS:-101 202 303}"
RUN_ROOT="${RUN_ROOT:-outputs/peftleak_official_image/utility}"
LOG_ROOT="${LOG_ROOT:-${RUN_ROOT}/logs}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

failures=0

run_variant() {
  local label="$1"
  local seed="$2"
  shift 2
  local output_dir="${RUN_ROOT}/seed${seed}/${label}"
  local log_dir="${LOG_ROOT}/seed${seed}"
  local log_file="${log_dir}/${label}.log"
  mkdir -p "$output_dir" "$log_dir"

  if [[ "${FORCE:-0}" != "1" ]] && \
     grep -q '^result_status=ok$' "$log_file" 2>/dev/null && \
     grep -q '^reportable=true$' "$log_file" 2>/dev/null && \
     [[ -f "${output_dir}/best_adapter_head.pt" ]]; then
    echo "[peftleak-image-utility] skip completed ${label} seed=${seed}" >&2
    return 0
  fi

  echo "[peftleak-image-utility] running ${label} seed=${seed}" >&2
  if SEED="$seed" OUTPUT_DIR="$output_dir" \
     bash scripts/peftleak_official_image_utility.sh "$@" 2>&1 | tee "$log_file"; then
    return 0
  fi
  echo "[peftleak-image-utility] failed ${label} seed=${seed}" >&2
  failures=$((failures + 1))
}

for seed in "${SEED_LIST[@]}"; do
  run_variant head_only "$seed" --defense none --utility_control head_only --lr_adapter 0
  run_variant none "$seed" --defense none
  run_variant topk_0.1 "$seed" --defense topk --defense_topk_ratio 0.1
  if [[ "$PROFILE" == "full" ]]; then
    run_variant topk_0.3 "$seed" --defense topk --defense_topk_ratio 0.3
  fi
  run_variant compression_8 "$seed" --defense compression --defense_n_bits 8
  if [[ "$PROFILE" == "full" ]]; then
    run_variant compression_16 "$seed" --defense compression --defense_n_bits 16
  fi
  run_variant proj_only_0.5 "$seed" \
    --defense proj_only --defense_lrb_keep_ratio_sensitive 0.5
  run_variant proj_only_0.75 "$seed" \
    --defense proj_only --defense_lrb_keep_ratio_sensitive 0.75
  run_variant proj_only_0.9 "$seed" \
    --defense proj_only --defense_lrb_keep_ratio_sensitive 0.9
  run_variant full_lrb_0.5 "$seed" \
    --defense full_lrb --defense_lrb_keep_ratio_sensitive 0.5
done

if (( failures > 0 )); then
  echo "[peftleak-image-utility] ${failures} run(s) failed." >&2
  exit 1
fi

result_args=(
  --utility-log-dir "$LOG_ROOT"
  --output-dir "${RUN_ROOT}/tables"
  --expected-seeds "${SEED_LIST[@]}"
)
if [[ -n "${PRIVACY_LOG_ROOT:-}" ]]; then
  result_args+=(--privacy-log-dir "$PRIVACY_LOG_ROOT")
else
  echo "[peftleak-image-utility] PRIVACY_LOG_ROOT is unset; writing utility-only tables." >&2
fi
"${PYTHON:-python}" scripts/peftleak_image_results.py "${result_args[@]}"

echo "[peftleak-image-utility] complete: ${RUN_ROOT}" >&2
