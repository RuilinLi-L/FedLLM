#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_dager_auto_log.sh
source "${SCRIPT_DIR}/_dager_auto_log.sh"
dager_auto_log_enable "lora" "$@"

if [ "$#" -lt 2 ]; then
  cat >&2 <<EOF
[dager] Usage:
[dager]   ./scripts/lora.sh DATASET BATCH_SIZE [extra attack args...]
[dager] Notes:
[dager]   This is a compatibility wrapper around ./scripts/peft_eval.sh.
EOF
  exit 2
fi

DATASET="$1"
BATCH="$2"
EXTRA=()
if [ "$#" -gt 2 ]; then
  EXTRA=( "${@:3}" )
fi

DAGER_NO_AUTO_LOG=1 "${SCRIPT_DIR}/peft_eval.sh" \
  "$DATASET" \
  "$BATCH" \
  "meta-llama/Meta-Llama-3.1-8B" \
  "100" \
  --l1_span_thresh 0.05 \
  --l2_span_thresh 0.05 \
  --rank_tol 5e-9 \
  --finetuned_path ./models/lora_8530.pt \
  --lora_r 256 \
  --pad left \
  "${EXTRA[@]}"
