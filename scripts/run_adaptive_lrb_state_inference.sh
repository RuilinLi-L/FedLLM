#!/usr/bin/env bash
set -euo pipefail

# Dedicated runner for state_inference_v1.  It never writes to historical
# adaptive-matrix directories and records their pre/post hashes as evidence.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CHECKPOINT=""
PYTHON_BIN="${PYTHON:-python}"
DEVICE="cuda"
CACHE_DIR="./models_cache"
RUN_DIR=""
MODE="formal"
SKIP_EXISTING=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bash scripts/run_adaptive_lrb_state_inference.sh --checkpoint PATH [options]

Options:
  --checkpoint PATH    required GPT-2 SST-2 checkpoint
  --python PATH        Python executable (default: python)
  --device DEVICE      attack device (default: cuda)
  --cache-dir PATH     Hugging Face dataset cache (default: ./models_cache)
  --run-dir PATH       new, empty output root
  --mode smoke|formal  smoke uses 2 fit updates, 1 held-out update, and 2 public batches
  --skip-existing      resume only logs whose state summaries are complete
  --dry-run            print commands without executing
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --checkpoint=*) CHECKPOINT="${1#*=}"; shift ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --python=*) PYTHON_BIN="${1#*=}"; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --device=*) DEVICE="${1#*=}"; shift ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --cache-dir=*) CACHE_DIR="${1#*=}"; shift ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --run-dir=*) RUN_DIR="${1#*=}"; shift ;;
    --mode) MODE="$2"; shift 2 ;;
    --mode=*) MODE="${1#*=}"; shift ;;
    --skip-existing) SKIP_EXISTING=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[state-inference] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$CHECKPOINT" ] || [ ! -e "$CHECKPOINT" ]; then
  echo "[state-inference] --checkpoint must reference an existing checkpoint" >&2
  exit 2
fi
case "$MODE" in smoke|formal) ;; *) echo "[state-inference] invalid --mode: $MODE" >&2; exit 2 ;; esac

if [ "$DRY_RUN" -eq 0 ]; then
  if ! "$PYTHON_BIN" -c 'import torch, datasets, transformers; assert torch.cuda.is_available()' >/dev/null 2>&1; then
    echo "[state-inference] Python must provide CUDA torch, datasets, and transformers; activate the project dager environment or pass --python." >&2
    exit 2
  fi
  if ! HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 "$PYTHON_BIN" -c 'from datasets import load_dataset; import sys; split = load_dataset("glue", "sst2", cache_dir=sys.argv[1])["validation"]; assert len(split) > 0' "$CACHE_DIR"; then
    echo "[state-inference] SST-2 validation is not available in --cache-dir=$CACHE_DIR. Populate that cache or pass the correct --cache-dir before running." >&2
    exit 2
  fi
fi

if [ -z "$RUN_DIR" ]; then
  RUN_DIR="log/runs/adaptive_lrb_state_inference_sst2_official_validation_$(date +%Y%m%d_%H%M%S)"
fi
if [ -e "$RUN_DIR" ] && [ "$SKIP_EXISTING" -ne 1 ]; then
  echo "[state-inference] refusing to reuse existing run root: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR/logs"

LEGACY_ROOTS=(
  "log/同预算 white-box baselines/new/adaptive_lrb_matrix_sst2_official_validation_20260718_114719"
  "log/同预算 white-box baselines/new/adaptive_lrb_matrix_sst2_official_validation_20260718_114758"
)
manifest() {
  for root in "${LEGACY_ROOTS[@]}"; do
    if [ -d "$root" ]; then
      find "$root" -type f \( -name '*.txt' -o -name 'results.csv' \) -print0 | sort -z | xargs -0 sha256sum
    else
      printf 'MISSING  %s\n' "$root"
    fi
  done
}
manifest >"$RUN_DIR/legacy_inputs_before.sha256"

if [ "$MODE" = "smoke" ]; then
  TARGET_INPUTS=3
  FIT_END=2
  EVAL_START=2
  EVAL_COUNT=1
  M_VALUES=1
  BUDGETS=0,1
  CALIBRATION_BATCHES=2
  CANDIDATES=32
  SEEDS=(1101)
else
  TARGET_INPUTS=100
  FIT_END=64
  EVAL_START=80
  EVAL_COUNT=20
  M_VALUES=1,4,16,64
  BUDGETS=0,100,400
  CALIBRATION_BATCHES=512
  CANDIDATES=2048
  SEEDS=(1101 1202 1303)
fi

{
  printf 'protocol=state_inference_v1\n'
  printf 'mode=%s\n' "$MODE"
  printf 'target_inputs=%s\n' "$TARGET_INPUTS"
  printf 'fit_end=%s\n' "$FIT_END"
  printf 'eval_start=%s\n' "$EVAL_START"
  printf 'eval_count=%s\n' "$EVAL_COUNT"
  printf 'm_values=%s\n' "$M_VALUES"
  printf 'budgets=%s\n' "$BUDGETS"
  printf 'calibration_batches=%s\n' "$CALIBRATION_BATCHES"
  printf 'candidate_count=%s\n' "$CANDIDATES"
  printf 'cache_dir=%s\n' "$CACHE_DIR"
  printf 'seeds=%s\n' "${SEEDS[*]}"
} >"$RUN_DIR/run_manifest.txt"

printf 'seed,exit_code\n' >"$RUN_DIR/exit_codes.csv"
for seed in "${SEEDS[@]}"; do
  logfile="$RUN_DIR/logs/static_state_seed${seed}.txt"
  if [ -e "$logfile" ] && [ "$SKIP_EXISTING" -eq 1 ] && grep -q '^result_status=ok$' "$logfile"; then
    echo "[state-inference] skip complete seed $seed" >&2
    continue
  fi
  if [ -e "$logfile" ]; then
    echo "[state-inference] refusing to overwrite: $logfile" >&2
    exit 2
  fi
  command=(
    "$PYTHON_BIN" attack_adaptive_lrb_state_inference.py
    --dataset sst2 --split official_validation --n_inputs "$TARGET_INPUTS" --batch_size 2
    --l1_filter all --l2_filter non-overlap --model_path gpt2 --finetuned_path "$CHECKPOINT"
    --cache_dir "$CACHE_DIR"
    --device "$DEVICE" --task seq_class --algo sgd --n_layers 2
    --defense lrb --defense_lrb_preset proj_only --defense_lrb_keep_ratio_sensitive 0.5
    --defense_lrb_seed_mode static --defense_lrb_seed 700001
    --adaptive_attack defense_aware --adaptive_lrb_sign_source defense_device
    --adaptive_lrb_knowledge method_only --adaptive_lrb_attack_seed 900001 --adaptive_lrb_seed_samples 1
    --adaptive_candidate_multiplier 100 --rng_seed "$seed" --log_file "$logfile"
    --state-target-inputs "$TARGET_INPUTS" --state-fit-end "$FIT_END"
    --state-eval-start "$EVAL_START" --state-eval-count "$EVAL_COUNT"
    --state-m-values "$M_VALUES" --state-budgets "$BUDGETS"
    --state-calibration-batches "$CALIBRATION_BATCHES" --state-candidate-count "$CANDIDATES"
  )
  echo "[state-inference] run seed=$seed" >&2
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '%q ' "${command[@]}"; printf '\n'
    continue
  fi
  set +e
  "${command[@]}"
  rc=$?
  set -e
  printf '%s,%s\n' "$seed" "$rc" >>"$RUN_DIR/exit_codes.csv"
  if [ "$rc" -ne 0 ]; then
    exit "$rc"
  fi
done

manifest >"$RUN_DIR/legacy_inputs_after.sha256"
if ! cmp -s "$RUN_DIR/legacy_inputs_before.sha256" "$RUN_DIR/legacy_inputs_after.sha256"; then
  echo "[state-inference] legacy input hash mismatch; investigate before reporting results" >&2
  exit 3
fi

if [ "$DRY_RUN" -eq 0 ]; then
  "$PYTHON_BIN" scripts/collect_state_inference_results.py --run-root "$RUN_DIR"
fi
echo "[state-inference] complete: $RUN_DIR" >&2
