#!/bin/bash
# Run and validate the GPT-2 PEFTLeak-style text Adapter ratio v2 privacy matrix.
#
# Usage:
#   RUN_ROOT=log/peftleak_text_sst2_v2/RUN_ID \
#     bash scripts/peftleak_text_v2_privacy.sh preflight|smoke|pilot|formal|all|validate [stage]

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$ROOT" || exit 1

ACTION="${1:-}"
case "$ACTION" in
  preflight|smoke|pilot|formal|all|validate) ;;
  *)
    echo "Usage: RUN_ROOT=PATH bash scripts/peftleak_text_v2_privacy.sh preflight|smoke|pilot|formal|all|validate [stage]" >&2
    exit 2
    ;;
esac

PYTHON_BIN="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
CACHE_DIR="${CACHE_DIR:-./models_cache}"
CHECKPOINT="${CHECKPOINT:-./outputs/peftleak_text_sst2/models/gpt2_adapter_clean/final_adapter}"
SMOKE_N="${SMOKE_N:-2}"
PILOT_N="${PILOT_N:-10}"
FORMAL_N="${FORMAL_N:-100}"
SMOKE_SEED="${SMOKE_SEED:-101}"
PILOT_SEED="${PILOT_SEED:-101}"
read -r -a SEED_LIST <<< "${SEEDS:-101 202 303}"

SMOKE_LABELS=(none proj_only_0.5)
MATRIX_LABELS=(none topk_0.1 compression_6 noise_1e-3 proj_only_0.5 proj_only_0.65 proj_only_0.75 proj_only_0.9)

failures=0

summary_value() {
  local key="$1"
  local log_file="$2"
  grep -m1 "^${key}=" "$log_file" 2>/dev/null | cut -d= -f2-
}

check_environment() {
  local missing=0
  if [[ ! -d "$CHECKPOINT" ]]; then
    echo "[peftleak-text-v2] missing Adapter checkpoint directory: ${CHECKPOINT}" >&2
    missing=1
  fi
  for filename in adapter_config.json fedllm_peft_metadata.json adapter_head.bin; do
    if [[ ! -f "${CHECKPOINT}/${filename}" ]]; then
      echo "[peftleak-text-v2] missing checkpoint file: ${CHECKPOINT}/${filename}" >&2
      missing=1
    fi
  done
  if [[ ! -f "${CHECKPOINT}/adapter_model.bin" && \
        ! -f "${CHECKPOINT}/adapter_model.safetensors" && \
        ! -f "${CHECKPOINT}/pytorch_adapter.bin" && \
        ! -f "${CHECKPOINT}/pytorch_adapter.safetensors" ]]; then
    echo "[peftleak-text-v2] missing an AdapterHub .bin/.safetensors payload in ${CHECKPOINT}" >&2
    missing=1
  fi
  (( missing == 0 )) || return 1

  mkdir -p "$CACHE_DIR"
  "$PYTHON_BIN" -c "import adapters, peft, torch; assert torch.cuda.is_available(), 'CUDA is unavailable'" || return 1
}

run_preflight() {
  check_environment || return 1
  "$PYTHON_BIN" -B test_peftleak_text_new_semantics.py
}

require_run_root() {
  if [[ -z "${RUN_ROOT:-}" ]]; then
    echo "[peftleak-text-v2] RUN_ROOT is required so smoke, pilot, and formal logs share one auditable run." >&2
    echo "[peftleak-text-v2] Example: export RUN_ROOT=./log/peftleak_text_sst2_v2/$(date +%Y%m%d_%H%M%S)" >&2
    return 1
  fi
  mkdir -p "$RUN_ROOT"/{smoke,pilot,formal}
}

validate_log() {
  local log_file="$1"
  local expected_n="$2"
  local require_clean_leakage="$3"
  local required_lines=(
    "result_status=ok"
    "attack=fedllm_peftleak_style_text_adapter_ratio_v2"
    "attack_variant=fixed_public_bins_registered_embedding_probe"
    "reproduction_level=peftleak_style_text_adaptation"
    "n_inputs_completed=${expected_n}"
    "probe_inventory_fixed=true"
    "probe_installed_before_private_data=true"
    "decoder_private_routing=false"
    "public_stats_source=test_disjoint_partition"
  )

  if [[ ! -f "$log_file" ]]; then
    echo "[peftleak-text-v2] missing log: ${log_file}" >&2
    return 1
  fi
  if [[ "$(grep -c '^===== RESULT SUMMARY START =====$' "$log_file")" -ne 1 || \
        "$(grep -c '^===== RESULT SUMMARY END =====$' "$log_file")" -ne 1 ]]; then
    echo "[peftleak-text-v2] incomplete or duplicate summary: ${log_file}" >&2
    return 1
  fi
  local required
  for required in "${required_lines[@]}"; do
    if ! grep -qx "$required" "$log_file"; then
      echo "[peftleak-text-v2] missing ${required@Q} in ${log_file}" >&2
      return 1
    fi
  done
  if grep -Eiq 'Traceback|CUDA out of memory|(^|[^[:alnum:]_])nan([^[:alnum:]_]|$)' "$log_file"; then
    echo "[peftleak-text-v2] failure marker found in ${log_file}" >&2
    return 1
  fi
  if ! grep -Eq '^shared_gradient_count=[1-9][0-9]*$' "$log_file" || \
     ! grep -Eq '^shared_gradient_names_sha256=[0-9a-f]{64}$' "$log_file" || \
     ! grep -Eq '^rec_token_mean=([0-9]+([.][0-9]+)?|[.][0-9]+)$' "$log_file" || \
     ! grep -Eq '^recovered_position_count_mean=([0-9]+([.][0-9]+)?|[.][0-9]+)$' "$log_file"; then
    echo "[peftleak-text-v2] missing a required numeric/hash field in ${log_file}" >&2
    return 1
  fi
  if [[ "$require_clean_leakage" == "1" ]]; then
    if ! awk -F= '$1 == "rec_token_mean" { found=1; if ($2 + 0 > 0) ok=1 } END { exit !(found && ok) }' "$log_file" || \
       ! awk -F= '$1 == "recovered_position_count_mean" { found=1; if ($2 + 0 > 0) ok=1 } END { exit !(found && ok) }' "$log_file"; then
      echo "[peftleak-text-v2] clean attack did not recover any private text in ${log_file}" >&2
      return 1
    fi
  fi
}

validate_stage() {
  local stage="$1"
  local expected_n="$2"
  local seed_string="$3"
  shift 3
  local labels=("$@")
  local seeds=()
  read -r -a seeds <<< "$seed_string"
  local expected_files=$(( ${#seeds[@]} * ${#labels[@]} ))
  local actual_files
  actual_files="$(find "${RUN_ROOT}/${stage}" -maxdepth 1 -type f -name '*.txt' | wc -l)"
  if [[ "$actual_files" -ne "$expected_files" ]]; then
    echo "[peftleak-text-v2] ${stage} expected ${expected_files} logs, found ${actual_files}." >&2
    return 1
  fi

  local inventory_hash=""
  local seed label log_file current_hash
  for seed in "${seeds[@]}"; do
    for label in "${labels[@]}"; do
      log_file="${RUN_ROOT}/${stage}/${label}_seed${seed}_n${expected_n}.txt"
      validate_log "$log_file" "$expected_n" "$([[ "$label" == "none" ]] && echo 1 || echo 0)" || return 1
      current_hash="$(summary_value shared_gradient_names_sha256 "$log_file")"
      if [[ -z "$inventory_hash" ]]; then
        inventory_hash="$current_hash"
      elif [[ "$current_hash" != "$inventory_hash" ]]; then
        echo "[peftleak-text-v2] shared gradient inventory changed in ${log_file}." >&2
        return 1
      fi
    done
  done
  echo "[peftleak-text-v2] validated ${stage}: ${expected_files} logs, inventory=${inventory_hash}" >&2
}

run_one() {
  local n_inputs="$1"
  local seed="$2"
  local stage="$3"
  local label="$4"
  shift 4
  local log_file="${RUN_ROOT}/${stage}/${label}_seed${seed}_n${n_inputs}.txt"

  if validate_log "$log_file" "$n_inputs" "$([[ "$label" == "none" ]] && echo 1 || echo 0)" 2>/dev/null; then
    echo "[peftleak-text-v2] skip completed ${stage}/${label} seed=${seed}" >&2
    return 0
  fi
  if [[ -e "$log_file" && "${FORCE:-0}" != "1" ]]; then
    echo "[peftleak-text-v2] refusing to overwrite incomplete log: ${log_file}" >&2
    echo "[peftleak-text-v2] use a new RUN_ROOT, or set FORCE=1 to replace this v2 log." >&2
    failures=$((failures + 1))
    return 1
  fi

  echo "[peftleak-text-v2] running ${stage}/${label} seed=${seed} n_inputs=${n_inputs}" >&2
  if "$PYTHON_BIN" -u attack_peftleak_new.py \
      --dataset sst2 --split val --batch_size 1 --n_inputs "$n_inputs" \
      --model_path gpt2 --finetuned_path "$CHECKPOINT" \
      --train_method peft --peft_method adapter --adapter_reduction_factor 16 \
      --peftleak_attack_mode ratio --peftleak_max_length 128 \
      --peftleak_ratio_bins 8 --peftleak_ratio_rows_per_bin 4 \
      --peftleak_ratio_public_n_inputs 64 \
      --precision full --pad right --algo sgd --attn_implementation eager \
      --text_metric_backend datasets --device "$DEVICE" --device_grad auto \
      --cache_dir "$CACHE_DIR" --grad_mode eval --rng_seed "$seed" \
      "$@" 2>&1 | tee "$log_file"; then
    if validate_log "$log_file" "$n_inputs" "$([[ "$label" == "none" ]] && echo 1 || echo 0)"; then
      return 0
    fi
  fi
  echo "[peftleak-text-v2] failed ${stage}/${label} seed=${seed}" >&2
  failures=$((failures + 1))
  return 1
}

run_matrix() {
  local n_inputs="$1"
  local seed="$2"
  local stage="$3"
  run_one "$n_inputs" "$seed" "$stage" none --defense none || true
  run_one "$n_inputs" "$seed" "$stage" topk_0.1 --defense topk --defense_topk_ratio 0.1 || true
  run_one "$n_inputs" "$seed" "$stage" compression_6 --defense compression --defense_n_bits 6 || true
  run_one "$n_inputs" "$seed" "$stage" noise_1e-3 --defense noise --defense_noise 1e-3 || true
  local keep_ratio
  for keep_ratio in 0.5 0.65 0.75 0.9; do
    run_one "$n_inputs" "$seed" "$stage" "proj_only_${keep_ratio}" \
      --defense lrb --defense_lrb_preset proj_only \
      --defense_lrb_keep_ratio_sensitive "$keep_ratio" || true
  done
}

run_smoke() {
  run_one "$SMOKE_N" "$SMOKE_SEED" smoke none --defense none || true
  run_one "$SMOKE_N" "$SMOKE_SEED" smoke proj_only_0.5 \
    --defense lrb --defense_lrb_preset proj_only \
    --defense_lrb_keep_ratio_sensitive 0.5 || true
  (( failures == 0 )) && validate_stage smoke "$SMOKE_N" "$SMOKE_SEED" "${SMOKE_LABELS[@]}"
}

run_pilot() {
  validate_stage smoke "$SMOKE_N" "$SMOKE_SEED" "${SMOKE_LABELS[@]}" || return 1
  run_matrix "$PILOT_N" "$PILOT_SEED" pilot
  (( failures == 0 )) && validate_stage pilot "$PILOT_N" "$PILOT_SEED" "${MATRIX_LABELS[@]}"
}

run_formal() {
  validate_stage pilot "$PILOT_N" "$PILOT_SEED" "${MATRIX_LABELS[@]}" || return 1
  local seed
  for seed in "${SEED_LIST[@]}"; do
    run_matrix "$FORMAL_N" "$seed" formal
  done
  (( failures == 0 )) && validate_stage formal "$FORMAL_N" "${SEED_LIST[*]}" "${MATRIX_LABELS[@]}"
}

run_preflight || exit 1
if [[ "$ACTION" == "preflight" ]]; then
  echo "[peftleak-text-v2] preflight passed." >&2
  exit 0
fi
require_run_root || exit 1

action_rc=0
case "$ACTION" in
  smoke)
    run_smoke || action_rc=$?
    ;;
  pilot)
    run_pilot || action_rc=$?
    ;;
  formal)
    run_formal || action_rc=$?
    ;;
  all)
    run_smoke && run_pilot && run_formal || action_rc=$?
    ;;
  validate)
    case "${2:-formal}" in
      smoke) validate_stage smoke "$SMOKE_N" "$SMOKE_SEED" "${SMOKE_LABELS[@]}" || action_rc=$? ;;
      pilot) validate_stage pilot "$PILOT_N" "$PILOT_SEED" "${MATRIX_LABELS[@]}" || action_rc=$? ;;
      formal) validate_stage formal "$FORMAL_N" "${SEED_LIST[*]}" "${MATRIX_LABELS[@]}" || action_rc=$? ;;
      *) echo "[peftleak-text-v2] validate stage must be smoke, pilot, or formal." >&2; exit 2 ;;
    esac
    ;;
esac

if (( failures > 0 || action_rc > 0 )); then
  echo "[peftleak-text-v2] failed: action_rc=${action_rc}, run_failures=${failures}." >&2
  exit 1
fi

echo "[peftleak-text-v2] complete: action=${ACTION} run_root=${RUN_ROOT}" >&2
