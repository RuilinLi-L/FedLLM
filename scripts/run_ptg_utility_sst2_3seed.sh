#!/bin/bash
# Run utility points that pair exactly with the SST-2 GPT-2 PTG first2 privacy sweep.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FEDLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$FEDLLM_ROOT" || exit 1

PROFILE="p0"
BASELINE=""
SEEDS_RAW="101,202,303"
BATCH_SIZE=1
EPOCHS=1
GPU_REQUEST="auto"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CHECKPOINT="./models/gpt2_sst2_clean_num_epochs_2/final"
RUN_DIR=""
PRIVACY_ROOT=""
DRY_RUN=0
KEEP_MODELS=0
NO_COLLECT=0
SKIP_CLEAN=0
FAILURES=0

P0_BASELINES=(
  none
  proj_only_0.2
  proj_only_0.5
  proj_only_0.75
  compression_2
  compression_4
  noise_5e-4
  dpsgd_5e-4
)

P1_BASELINES=(
  proj_only_0.65
  proj_only_0.9
  topk_0.05
  topk_0.1
  topk_0.3
  compression_8
  compression_16
)

usage() {
  cat <<'EOF'
Usage: bash scripts/run_ptg_utility_sst2_3seed.sh [options]

Options:
  --profile p0|p1|all       P0 tests the current privacy/utility question first
  --baseline NAME           Run clean plus one named baseline
  --seeds LIST              Comma-separated seeds (default: 101,202,303)
  --batch-size N            Utility training batch size (default: 1, matching PTG)
  --epochs N                Training epochs (default: 1)
  --gpu auto|all|ID[,ID]    Preserve, clear, or set CUDA_VISIBLE_DEVICES
  --python PATH             Python executable (default: python3)
  --checkpoint PATH         Fine-tuned GPT-2 checkpoint used as the utility anchor
  --privacy-root PATH       PTG privacy root to include in trade-off collection
  --log-dir PATH            Reuse a run directory; completed logs are skipped
  --keep-models             Save final model checkpoints (off by default)
  --skip-clean              Do not add the clean anchor to this worker
  --no-collect              Skip result collection after training
  --dry-run                 Print commands without running training
  -h, --help                Show this help

P0: none, proj_only@0.2/@0.5/@0.75, compression@2/@4,
    noise@5e-4, DP-SGD-style@5e-4.
P1: proj_only@0.65/@0.9, top-k@0.05/@0.1/@0.3, compression@8/@16.
EOF
}

all_baselines() {
  printf '%s\n' "${P0_BASELINES[@]}" "${P1_BASELINES[@]}"
}

is_valid_baseline() {
  local candidate="$1"
  local item
  while IFS= read -r item; do
    if [ "$item" = "$candidate" ]; then
      return 0
    fi
  done < <(all_baselines)
  return 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile)
      PROFILE="${2:?--profile requires a value}"
      shift 2
      ;;
    --profile=*) PROFILE="${1#*=}"; shift ;;
    --baseline)
      BASELINE="${2:?--baseline requires a value}"
      shift 2
      ;;
    --baseline=*) BASELINE="${1#*=}"; shift ;;
    --seeds)
      SEEDS_RAW="${2:?--seeds requires a value}"
      shift 2
      ;;
    --seeds=*) SEEDS_RAW="${1#*=}"; shift ;;
    --batch-size)
      BATCH_SIZE="${2:?--batch-size requires a value}"
      shift 2
      ;;
    --batch-size=*) BATCH_SIZE="${1#*=}"; shift ;;
    --epochs)
      EPOCHS="${2:?--epochs requires a value}"
      shift 2
      ;;
    --epochs=*) EPOCHS="${1#*=}"; shift ;;
    --gpu)
      GPU_REQUEST="${2:?--gpu requires a value}"
      shift 2
      ;;
    --gpu=*) GPU_REQUEST="${1#*=}"; shift ;;
    --python)
      PYTHON_BIN="${2:?--python requires a value}"
      shift 2
      ;;
    --python=*) PYTHON_BIN="${1#*=}"; shift ;;
    --checkpoint)
      CHECKPOINT="${2:?--checkpoint requires a value}"
      shift 2
      ;;
    --checkpoint=*) CHECKPOINT="${1#*=}"; shift ;;
    --privacy-root)
      PRIVACY_ROOT="${2:?--privacy-root requires a value}"
      shift 2
      ;;
    --privacy-root=*) PRIVACY_ROOT="${1#*=}"; shift ;;
    --log-dir)
      RUN_DIR="${2:?--log-dir requires a value}"
      shift 2
      ;;
    --log-dir=*) RUN_DIR="${1#*=}"; shift ;;
    --keep-models) KEEP_MODELS=1; shift ;;
    --skip-clean) SKIP_CLEAN=1; shift ;;
    --no-collect) NO_COLLECT=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[ptg-utility] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$PROFILE" in
  p0|p1|all) ;;
  *)
    echo "[ptg-utility] --profile must be p0, p1, or all." >&2
    exit 2
    ;;
esac

if [ -n "$BASELINE" ] && ! is_valid_baseline "$BASELINE"; then
  echo "[ptg-utility] Unsupported baseline: ${BASELINE}" >&2
  exit 2
fi

if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || ! [[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ptg-utility] --batch-size and --epochs must be positive integers." >&2
  exit 2
fi

IFS=',' read -r -a SEEDS <<< "$SEEDS_RAW"
if [ "${#SEEDS[@]}" -eq 0 ]; then
  echo "[ptg-utility] At least one seed is required." >&2
  exit 2
fi
for seed in "${SEEDS[@]}"; do
  if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
    echo "[ptg-utility] Invalid seed: ${seed}" >&2
    exit 2
  fi
done

case "$GPU_REQUEST" in
  auto) ;;
  all) unset CUDA_VISIBLE_DEVICES ;;
  *) export CUDA_VISIBLE_DEVICES="$GPU_REQUEST" ;;
esac

if [ -z "$PRIVACY_ROOT" ]; then
  if [ -d "log/runs/ptg_gpt2_first2_privacy_3seed_20260714" ]; then
    PRIVACY_ROOT="log/runs/ptg_gpt2_first2_privacy_3seed_20260714"
  else
    PRIVACY_ROOT="log/ptg_gpt2_first2_privacy_3seed_20260714"
  fi
fi

if [ -z "$RUN_DIR" ]; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  RUN_LABEL="${BASELINE:-$PROFILE}"
  RUN_DIR="log/runs/ptg_utility_sst2_b${BATCH_SIZE}_gpt2_3seed_${RUN_LABEL}_${STAMP}"
fi

EXIT_CODES="${RUN_DIR}/exit_codes.csv"
if [ "$DRY_RUN" -eq 0 ]; then
  if [ ! -f "${CHECKPOINT}/config.json" ]; then
    echo "[ptg-utility] Missing checkpoint: ${CHECKPOINT}" >&2
    exit 1
  fi
  mkdir -p "$RUN_DIR"
  if [ ! -f "$EXIT_CODES" ]; then
    printf 'tag,seed,exit_code\n' > "$EXIT_CODES"
  fi
  cat > "${RUN_DIR}/run_manifest.txt" <<EOF
script=scripts/run_ptg_utility_sst2_3seed.sh
profile=${PROFILE}
baseline=${BASELINE:-all_from_profile}
dataset=sst2
model=gpt2
checkpoint=${CHECKPOINT}
batch_size=${BATCH_SIZE}
num_epochs=${EPOCHS}
seeds=${SEEDS[*]}
gpu_request=${GPU_REQUEST}
privacy_root=${PRIVACY_ROOT}
keep_models=${KEEP_MODELS}
skip_clean=${SKIP_CLEAN}
p0=${P0_BASELINES[*]}
p1=${P1_BASELINES[*]}
EOF
fi

print_command() {
  printf '[ptg-utility] command:'
  printf ' %q' "$@"
  printf '\n'
}

run_variant() {
  local defense="$1"
  local param="$2"
  local tag="$3"
  local defense_args=()

  case "$defense" in
    none)
      defense_args=(--defense none)
      ;;
    proj_only)
      defense_args=(
        --defense lrb
        --defense_lrb_preset proj_only
        --defense_lrb_keep_ratio_sensitive "$param"
      )
      ;;
    topk)
      defense_args=(--defense topk --defense_topk_ratio "$param")
      ;;
    compression)
      defense_args=(--defense compression --defense_n_bits "$param")
      ;;
    noise)
      defense_args=(--defense noise --defense_noise "$param")
      ;;
    dpsgd)
      defense_args=(
        --defense dpsgd
        --defense_noise "$param"
        --defense_clip_norm 1.0
      )
      ;;
    *)
      echo "[ptg-utility] Unsupported defense: ${defense}" >&2
      exit 2
      ;;
  esac

  local seed
  for seed in "${SEEDS[@]}"; do
    local logfile="${RUN_DIR}/train_${tag}_seed${seed}.txt"
    local output_dir="${RUN_DIR}/models/${tag}_seed${seed}"

    if [ "$DRY_RUN" -eq 0 ] && [ -f "$logfile" ] && grep -q 'result_status=ok' "$logfile"; then
      echo "[ptg-utility] skip completed ${tag}/seed${seed}" >&2
      continue
    fi

    local cmd=(
      "$PYTHON_BIN" train.py
      --dataset sst2
      --task seq_class
      --batch_size "$BATCH_SIZE"
      --num_epochs "$EPOCHS"
      --save_every 0
      --model_path "$CHECKPOINT"
      --models_cache ./models_cache
      --train_method full
      --rng_seed "$seed"
      --device cuda
      --output_dir "$output_dir"
      "${defense_args[@]}"
      --log_file "$logfile"
    )
    if [ "$KEEP_MODELS" -eq 0 ]; then
      cmd+=(--skip_final_save)
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
      print_command "${cmd[@]}"
      continue
    fi

    echo "[ptg-utility] running ${tag}/seed${seed}" >&2
    set +e
    "${cmd[@]}"
    local rc=$?
    set -e
    printf '%s,%s,%s\n' "$tag" "$seed" "$rc" >> "$EXIT_CODES"
    if [ "$rc" -ne 0 ]; then
      FAILURES=$((FAILURES + 1))
    fi
  done
}

run_named_baseline() {
  case "$1" in
    none) run_variant none n/a none ;;
    proj_only_0.2) run_variant proj_only 0.2 proj_only_0.2 ;;
    proj_only_0.5) run_variant proj_only 0.5 proj_only_0.5 ;;
    proj_only_0.65) run_variant proj_only 0.65 proj_only_0.65 ;;
    proj_only_0.75) run_variant proj_only 0.75 proj_only_0.75 ;;
    proj_only_0.9) run_variant proj_only 0.9 proj_only_0.9 ;;
    topk_0.05) run_variant topk 0.05 topk_0.05 ;;
    topk_0.1) run_variant topk 0.1 topk_0.1 ;;
    topk_0.3) run_variant topk 0.3 topk_0.3 ;;
    compression_2) run_variant compression 2 compression_2 ;;
    compression_4) run_variant compression 4 compression_4 ;;
    compression_8) run_variant compression 8 compression_8 ;;
    compression_16) run_variant compression 16 compression_16 ;;
    noise_5e-4) run_variant noise 5e-4 noise_5e-4 ;;
    dpsgd_5e-4) run_variant dpsgd 5e-4 dpsgd_5e-4 ;;
  esac
}

run_profile() {
  local item
  case "$PROFILE" in
    p0)
      for item in "${P0_BASELINES[@]}"; do
        if [ "$item" != "none" ] || [ "$SKIP_CLEAN" -eq 0 ]; then run_named_baseline "$item"; fi
      done
      ;;
    p1)
      if [ "$SKIP_CLEAN" -eq 0 ]; then run_named_baseline none; fi
      for item in "${P1_BASELINES[@]}"; do run_named_baseline "$item"; done
      ;;
    all)
      for item in "${P0_BASELINES[@]}" "${P1_BASELINES[@]}"; do
        if [ "$item" != "none" ] || [ "$SKIP_CLEAN" -eq 0 ]; then run_named_baseline "$item"; fi
      done
      ;;
  esac
}

if [ -n "$BASELINE" ]; then
  if [ "$SKIP_CLEAN" -eq 0 ]; then run_named_baseline none; fi
  if [ "$BASELINE" != "none" ]; then
    run_named_baseline "$BASELINE"
  fi
else
  run_profile
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[ptg-utility] dry run complete; no training was started." >&2
  exit 0
fi

if [ "$NO_COLLECT" -eq 0 ]; then
  COLLECT_INPUTS=("$RUN_DIR")
  if [ -d "$PRIVACY_ROOT" ]; then
    while IFS= read -r -d '' privacy_log; do
      COLLECT_INPUTS+=("$privacy_log")
    done < <(find "$PRIVACY_ROOT" -type f -name '*.txt' ! -name 'summary.txt' -print0)
  else
    echo "[ptg-utility] warning: privacy root not found: ${PRIVACY_ROOT}" >&2
  fi

  "$PYTHON_BIN" scripts/collect_experiment_logs.py "${COLLECT_INPUTS[@]}" \
    -o "${RUN_DIR}/results.csv" \
    --markdown "${RUN_DIR}/results.md" \
    --utility-output "${RUN_DIR}/utility_results.csv" \
    --utility-markdown "${RUN_DIR}/utility_results.md" \
    --tradeoff-output "${RUN_DIR}/privacy_utility_tradeoff.csv" \
    --tradeoff-markdown "${RUN_DIR}/privacy_utility_tradeoff.md"
fi

if [ "$FAILURES" -ne 0 ]; then
  echo "[ptg-utility] ${FAILURES} training run(s) failed; inspect ${EXIT_CODES}." >&2
  exit 1
fi

echo "[ptg-utility] done: ${RUN_DIR}" >&2
