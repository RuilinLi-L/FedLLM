#!/usr/bin/env bash
# Run a three-seed, GPU full-gradient DAGER baseline sweep for one dataset.

set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage:" \
    "  bash scripts/run_dager_privacy_one_dataset_baselines.sh <sst2|cola|rotten_tomatoes> [options]" \
    "" \
    "Options:" \
    "  --gpu auto|all|ID     GPU visibility mode. Default: auto." \
    "  --log-dir PATH        Output log root. Default: log/runs/dager_privacy_<dataset>_baselines_3seed_<timestamp>." \
    "  --n-inputs N          Formal DAGER n_inputs. Default: 100." \
    "  --dry-run             Print commands without executing them." \
    "  --skip-main           Skip non-none baseline sweeps." \
    "  --run-adaptive        Also run defense-aware adaptive checks." \
    "  --no-collect          Skip final collect_experiment_logs.py aggregation." \
    "  -h, --help            Show this help." \
    "" \
    "This script intentionally fixes DAGER_SEEDS=\"101 202 303\" and never passes" \
    "--rng_seed to defense_baselines.sh. Non-none focused baselines are called with" \
    "--skip_anchor_none so the clean none anchor is run only once."
}

if [ "$#" -eq 0 ]; then
  usage >&2
  exit 2
fi

case "$1" in
  -h|--help)
    usage
    exit 0
    ;;
  -*)
    echo "[dager-one-dataset] dataset is required before options." >&2
    usage >&2
    exit 2
    ;;
  *)
    DATASET="$1"
    shift
    ;;
esac

GPU_REQUEST="auto"
GPU_MODE="auto"
LOG_DIR=""
N_INPUTS="100"
DRY_RUN=0
SKIP_MAIN=0
RUN_ADAPTIVE=0
NO_COLLECT=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpu)
      GPU_REQUEST="$2"; shift 2 ;;
    --gpu=*)
      GPU_REQUEST="${1#*=}"; shift ;;
    --log-dir)
      LOG_DIR="$2"; shift 2 ;;
    --log-dir=*)
      LOG_DIR="${1#*=}"; shift ;;
    --n-inputs)
      N_INPUTS="$2"; shift 2 ;;
    --n-inputs=*)
      N_INPUTS="${1#*=}"; shift ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    --skip-main)
      SKIP_MAIN=1; shift ;;
    --run-adaptive)
      RUN_ADAPTIVE=1; shift ;;
    --no-collect)
      NO_COLLECT=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[dager-one-dataset] unknown option: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR_PART="${SCRIPT_PATH%/*}"
if [ "$SCRIPT_DIR_PART" = "$SCRIPT_PATH" ]; then
  SCRIPT_DIR_PART="."
fi
SCRIPT_DIR="$(cd "$SCRIPT_DIR_PART" && pwd)"
DAGER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$DAGER_ROOT" || exit 1

case "$DATASET" in
  sst2)
    FINETUNED_PATH="./models/gpt2_sst2_clean_num_epochs_2/final"
    ;;
  cola)
    FINETUNED_PATH="./models/gpt2_cola_clean_num_epochs_2/final"
    ;;
  rotten_tomatoes)
    FINETUNED_PATH="./models/gpt2_rotten_tomatoes_clean_num_epochs_2/final"
    ;;
  *)
    echo "[dager-one-dataset] unsupported dataset: ${DATASET}" >&2
    usage >&2
    exit 2
    ;;
esac

export DAGER_SEEDS="101 202 303"

LRB_GRID=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.97 0.99)
TOPK_GRID=(0.01 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95)
COMPRESSION_GRID=(2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 40 48 56 64)
NOISE_GRID=(1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3 1e-2)
DPSGD_GRID=(1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3 1e-2)
DPSGD_OPACUS_GRID=(1e-5 3e-5 1e-4 3e-4 5e-4 1e-3 3e-3 1e-2)
SOTERIA_GRID=(10 20 30 50 70 90 95)
MIXUP_GRID=(0.05 0.10 0.20 0.50 1.0 2.0 5.0)

ADAPTIVE_LRBPROJONLY_GRID=(0.65 0.90 0.95 0.99)
ADAPTIVE_TOPK_GRID=(0.10 0.30 0.50)
ADAPTIVE_COMPRESSION_GRID=(8 16 24 32)
ADAPTIVE_DPSGD_OPACUS_GRID=(5e-4 1e-3 1e-2)

cuda_visible_devices_label() {
  if [ "${CUDA_VISIBLE_DEVICES+x}" = "x" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
      printf '%s' "$CUDA_VISIBLE_DEVICES"
    else
      printf 'empty'
    fi
  else
    printf 'unset'
  fi
}

configure_gpu_visibility() {
  case "$GPU_REQUEST" in
    ""|auto|AUTO)
      GPU_MODE="auto"
      ;;
    all|ALL)
      GPU_MODE="all"
      unset CUDA_VISIBLE_DEVICES
      ;;
    *)
      GPU_MODE="manual"
      export CUDA_VISIBLE_DEVICES="$GPU_REQUEST"
      ;;
  esac
}

configure_gpu_visibility

if [ -z "$LOG_DIR" ]; then
  printf -v STAMP '%(%Y%m%d_%H%M%S)T' -1
  safe_dataset=$(printf '%s' "$DATASET" | tr -c 'a-zA-Z0-9._-' '_')
  LOG_DIR="log/runs/dager_privacy_${safe_dataset}_baselines_3seed_${STAMP}"
fi
export DAGER_LOG_DIR="$LOG_DIR"
mkdir -p "$DAGER_LOG_DIR"

planned_main_configs() {
  printf '%s\n' "$((1 + ${#LRB_GRID[@]} + ${#LRB_GRID[@]} + ${#TOPK_GRID[@]} + ${#COMPRESSION_GRID[@]} + ${#NOISE_GRID[@]} + ${#DPSGD_GRID[@]} + ${#DPSGD_OPACUS_GRID[@]} + ${#SOTERIA_GRID[@]} + ${#MIXUP_GRID[@]}))"
}

planned_adaptive_configs() {
  printf '%s\n' "$((${#ADAPTIVE_LRBPROJONLY_GRID[@]} + ${#ADAPTIVE_TOPK_GRID[@]} + ${#ADAPTIVE_COMPRESSION_GRID[@]} + ${#ADAPTIVE_DPSGD_OPACUS_GRID[@]}))"
}

run_cmd() {
  printf '[dager-one-dataset] run:'
  printf ' %q' "$@"
  printf '\n'
  if [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  "$@"
}

write_manifest() {
  local created_at
  printf -v created_at '%(%Y-%m-%d %H:%M:%S)T' -1
  {
    printf 'script=%s\n' 'scripts/run_dager_privacy_one_dataset_baselines.sh'
    printf 'created_at=%s\n' "$created_at"
    printf 'dataset=%s\n' "$DATASET"
    printf 'model=%s\n' 'gpt2'
    printf 'batch_size=%s\n' '2'
    printf 'n_inputs=%s\n' "$N_INPUTS"
    printf 'seeds=%s\n' "$DAGER_SEEDS"
    printf 'finetuned_path=%s\n' "$FINETUNED_PATH"
    printf 'gpu_request=%s\n' "$GPU_REQUEST"
    printf 'gpu_mode=%s\n' "$GPU_MODE"
    printf 'cuda_visible_devices=%s\n' "$(cuda_visible_devices_label)"
    printf 'device=%s\n' 'attack.py --device cuda via scripts/defense_baselines.sh'
    printf 'threat_surface=%s\n' 'full_gradient_dager'
    printf 'main_configs=%s\n' "$(planned_main_configs)"
    printf 'adaptive_configs=%s\n' "$(planned_adaptive_configs)"
    printf 'run_adaptive=%s\n' "$RUN_ADAPTIVE"
    printf 'dpsgd_opacus=%s\n' 'included'
    printf 'formal_dp_claim=%s\n' 'false'
    printf 'lrb_grid=%s\n' "${LRB_GRID[*]}"
    printf 'topk_grid=%s\n' "${TOPK_GRID[*]}"
    printf 'compression_grid=%s\n' "${COMPRESSION_GRID[*]}"
    printf 'noise_grid=%s\n' "${NOISE_GRID[*]}"
    printf 'dpsgd_grid=%s\n' "${DPSGD_GRID[*]}"
    printf 'dpsgd_opacus_grid=%s\n' "${DPSGD_OPACUS_GRID[*]}"
    printf 'soteria_grid=%s\n' "${SOTERIA_GRID[*]}"
    printf 'mixup_grid=%s\n' "${MIXUP_GRID[*]}"
  } > "${DAGER_LOG_DIR}/run_manifest.txt"
}

check_environment() {
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dager-one-dataset] dry-run: skipping Python/CUDA environment checks." >&2
    return 0
  fi

  if [ ! -d "$FINETUNED_PATH" ]; then
    echo "[dager-one-dataset] missing finetuned checkpoint: ${FINETUNED_PATH}" >&2
    exit 2
  fi

  python3 -c "import torch; print('cuda_available=', torch.cuda.is_available()); raise SystemExit(0 if torch.cuda.is_available() else 1)"
  python3 -c "from utils.gpu import resolve_cuda_device; print('resolved_cuda_device=', resolve_cuda_device('cuda'))"

  if python3 -c "import opacus; print('opacus ok')" ; then
    :
  else
    echo "[dager-one-dataset] warning: opacus is unavailable; dpsgd_opacus rows may be logged as failures." >&2
  fi
}

run_focused_param_grid() {
  local defense="$1"
  shift
  local param
  for param in "$@"; do
    run_cmd bash scripts/defense_baselines.sh "$DATASET" 2 gpt2 "$N_INPUTS" \
      --baseline_defense "$defense" \
      --baseline_param "$param" \
      --skip_anchor_none \
      --finetuned_path "$FINETUNED_PATH"
  done
}

run_adaptive_param_grid() {
  local defense="$1"
  shift
  local param
  for param in "$@"; do
    run_cmd bash scripts/defense_baselines.sh "$DATASET" 2 gpt2 "$N_INPUTS" \
      --baseline_defense "$defense" \
      --baseline_param "$param" \
      --adaptive_attack_check \
      --skip_anchor_none \
      --finetuned_path "$FINETUNED_PATH"
  done
}

echo "[dager-one-dataset] root: ${DAGER_ROOT}" >&2
echo "[dager-one-dataset] dataset: ${DATASET}" >&2
echo "[dager-one-dataset] checkpoint: ${FINETUNED_PATH}" >&2
echo "[dager-one-dataset] log dir: ${DAGER_LOG_DIR}" >&2
echo "[dager-one-dataset] seeds: ${DAGER_SEEDS}" >&2
echo "[dager-one-dataset] planned main configs: $(planned_main_configs)" >&2
echo "[dager-one-dataset] planned adaptive configs: $(planned_adaptive_configs) (run_adaptive=${RUN_ADAPTIVE})" >&2
echo "[dager-one-dataset] gpu mode: ${GPU_MODE} (request=${GPU_REQUEST})" >&2
echo "[dager-one-dataset] CUDA_VISIBLE_DEVICES=$(cuda_visible_devices_label)" >&2

write_manifest
check_environment

echo "[dager-one-dataset] stage: none anchor" >&2
run_cmd bash scripts/defense_baselines.sh "$DATASET" 2 gpt2 "$N_INPUTS" \
  --baseline_defense none \
  --finetuned_path "$FINETUNED_PATH"

if [ "$SKIP_MAIN" -eq 0 ]; then
  echo "[dager-one-dataset] stage: main parameter sweeps" >&2
  run_focused_param_grid lrbprojonly "${LRB_GRID[@]}"
  run_focused_param_grid lrb "${LRB_GRID[@]}"
  run_focused_param_grid topk "${TOPK_GRID[@]}"
  run_focused_param_grid compression "${COMPRESSION_GRID[@]}"
  run_focused_param_grid noise "${NOISE_GRID[@]}"
  run_focused_param_grid dpsgd "${DPSGD_GRID[@]}"
  run_focused_param_grid dpsgd_opacus "${DPSGD_OPACUS_GRID[@]}"
  run_focused_param_grid soteria "${SOTERIA_GRID[@]}"
  run_focused_param_grid mixup "${MIXUP_GRID[@]}"
fi

if [ "$RUN_ADAPTIVE" -eq 1 ]; then
  echo "[dager-one-dataset] stage: adaptive checks" >&2
  run_adaptive_param_grid lrbprojonly "${ADAPTIVE_LRBPROJONLY_GRID[@]}"
  run_adaptive_param_grid topk "${ADAPTIVE_TOPK_GRID[@]}"
  run_adaptive_param_grid compression "${ADAPTIVE_COMPRESSION_GRID[@]}"
  run_adaptive_param_grid dpsgd_opacus "${ADAPTIVE_DPSGD_OPACUS_GRID[@]}"
fi

if [ "$NO_COLLECT" -eq 0 ]; then
  echo "[dager-one-dataset] stage: collect" >&2
  run_cmd python3 scripts/collect_experiment_logs.py "$DAGER_LOG_DIR" \
    -o "$DAGER_LOG_DIR/all_results.csv" \
    --markdown "$DAGER_LOG_DIR/all_results.md"
fi

echo "[dager-one-dataset] done: ${DAGER_LOG_DIR}" >&2
