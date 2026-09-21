#!/usr/bin/env bash
# Run one formal GPT-2 / SST-2 causal-LM utility condition and write its named log.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAGER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$DAGER_ROOT"

CONDITION="${1:-}"
SEED="${2:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "$CONDITION" != "none" && "$CONDITION" != "slr_rho05" ]]; then
  echo "Usage: $0 {none|slr_rho05} {101|202|303}" >&2
  exit 2
fi
if [[ "$SEED" != "101" && "$SEED" != "202" && "$SEED" != "303" ]]; then
  echo "Usage: $0 {none|slr_rho05} {101|202|303}" >&2
  exit 2
fi

LOG_DIR="log/runs/ntp_sst2_utility"
if [[ "$CONDITION" == "none" ]]; then
  RUN_NAME="none_seed${SEED}"
  DEFENSE_ARGS=(--defense none)
else
  RUN_NAME="slr_rho05_seed${SEED}"
  DEFENSE_ARGS=(
    --defense lrbprojonly
    --defense_lrb_keep_ratio_sensitive 0.5
    --defense_lrb_seed_mode static
  )
fi

LOG_FILE="${LOG_DIR}/${RUN_NAME}.txt"
OUTPUT_DIR="outputs/ntp_sst2_gpt2/${RUN_NAME}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

{
  echo "[ntp-utility] run=${RUN_NAME} batch_size=1 output=${OUTPUT_DIR}"
  DAGER_NO_AUTO_LOG=1 "$PYTHON_BIN" train.py \
    --dataset sst2 \
    --task next_token_pred \
    --model_path gpt2 \
    --train_method full \
    --batch_size 1 \
    --num_epochs 2 \
    --save_every 0 \
    --rng_seed "$SEED" \
    --output_dir "$OUTPUT_DIR" \
    "${DEFENSE_ARGS[@]}"
} 2>&1 | tee "$LOG_FILE"
