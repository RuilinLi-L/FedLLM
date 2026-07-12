#!/usr/bin/env bash
# Run the full DAGER noise/DP-SGD privacy sweep for one dataset and one seed.

set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage:" \
    "  bash scripts/run_dager_privacy_noise_dpsgd_one_seed.sh \\" \
    "    --baselines <noise|dpsgd|LIST> \\" \
    "    --dataset <sst2|cola|rotten_tomatoes> \\" \
    "    --seed <NON_NEGATIVE_INTEGER> [options]" \
    "" \
    "Required:" \
    "  --baselines LIST      Run noise, dpsgd, or both." \
    "                        Comma-separated and space-separated values are accepted." \
    "  --dataset DATASET     One of: sst2, cola, rotten_tomatoes." \
    "  --seed SEED           One explicit non-negative integer seed." \
    "" \
    "Options:" \
    "  --log-dir PATH        Output log root. Default includes dataset, baselines, seed, and timestamp." \
    "  --dry-run             Print all scheduled commands without executing them." \
    "  --no-collect          Skip final collect_experiment_logs.py aggregation." \
    "  -h, --help            Show this help." \
    "" \
    "Examples:" \
    "  bash scripts/run_dager_privacy_noise_dpsgd_one_seed.sh --baselines noise,dpsgd --dataset sst2 --seed 101" \
    "  bash scripts/run_dager_privacy_noise_dpsgd_one_seed.sh --baselines noise --dataset cola --seed 202 --dry-run"
}

die_usage() {
  echo "[dager-noise-dpsgd] $1" >&2
  usage >&2
  exit 2
}

BASELINES_REQUEST=""
DATASET=""
SEED=""
LOG_DIR=""
DRY_RUN=0
NO_COLLECT=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --baselines)
      shift
      if [ "$#" -eq 0 ] || [[ "$1" == -* ]]; then
        die_usage "--baselines requires at least one value."
      fi
      BASELINES_REQUEST=""
      while [ "$#" -gt 0 ] && [[ "$1" != -* ]]; do
        BASELINES_REQUEST="${BASELINES_REQUEST}${BASELINES_REQUEST:+ }$1"
        shift
      done
      ;;
    --baselines=*)
      BASELINES_REQUEST="${1#*=}"
      shift
      ;;
    --dataset)
      if [ "$#" -lt 2 ] || [[ "$2" == -* ]]; then
        die_usage "--dataset requires a value."
      fi
      DATASET="$2"
      shift 2
      ;;
    --dataset=*)
      DATASET="${1#*=}"
      shift
      ;;
    --seed)
      if [ "$#" -lt 2 ]; then
        die_usage "--seed requires a value."
      fi
      SEED="$2"
      shift 2
      ;;
    --seed=*)
      SEED="${1#*=}"
      shift
      ;;
    --log-dir)
      if [ "$#" -lt 2 ] || [[ "$2" == -* ]]; then
        die_usage "--log-dir requires a value."
      fi
      LOG_DIR="$2"
      shift 2
      ;;
    --log-dir=*)
      LOG_DIR="${1#*=}"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --no-collect)
      NO_COLLECT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die_usage "unknown option: $1"
      ;;
  esac
done

if [ -z "$BASELINES_REQUEST" ]; then
  die_usage "--baselines is required."
fi
if [ -z "$DATASET" ]; then
  die_usage "--dataset is required."
fi
if [ -z "$SEED" ]; then
  die_usage "--seed is required."
fi
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
  die_usage "invalid seed: ${SEED}; --seed must be a non-negative integer."
fi

SELECT_NOISE=0
SELECT_DPSGD=0
raw_baselines="${BASELINES_REQUEST//,/ }"
for baseline in $raw_baselines; do
  case "$baseline" in
    noise)
      SELECT_NOISE=1
      ;;
    dpsgd)
      SELECT_DPSGD=1
      ;;
    *)
      die_usage "unsupported baseline: ${baseline}; allowed baselines are noise and dpsgd."
      ;;
  esac
done

SELECTED_BASELINES=()
if [ "$SELECT_NOISE" -eq 1 ]; then
  SELECTED_BASELINES+=(noise)
fi
if [ "$SELECT_DPSGD" -eq 1 ]; then
  SELECTED_BASELINES+=(dpsgd)
fi
if [ "${#SELECTED_BASELINES[@]}" -eq 0 ]; then
  die_usage "--baselines cannot be empty."
fi

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
    die_usage "unsupported dataset: ${DATASET}."
    ;;
esac

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR_PART="${SCRIPT_PATH%/*}"
if [ "$SCRIPT_DIR_PART" = "$SCRIPT_PATH" ]; then
  SCRIPT_DIR_PART="."
fi
SCRIPT_DIR="$(cd "$SCRIPT_DIR_PART" && pwd)"
DAGER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$DAGER_ROOT" || exit 1

MODEL="gpt2"
BATCH_SIZE="2"
N_INPUTS="100"
SWEEP_GRID=(
  1e-6 3e-6 1e-5 3e-5 1e-4 2e-4 3e-4
  5e-4 7e-4 1e-3 2e-3 3e-3 5e-3 1e-2
)

baseline_tag="${SELECTED_BASELINES[*]}"
baseline_tag="${baseline_tag// /-}"
if [ -z "$LOG_DIR" ]; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  LOG_DIR="log/runs/dager_privacy_${DATASET}_${baseline_tag}_seed${SEED}_${STAMP}"
fi
export DAGER_LOG_DIR="$LOG_DIR"
mkdir -p "$DAGER_LOG_DIR"

cuda_visible_devices_label() {
  if [ "${CUDA_VISIBLE_DEVICES+x}" = "x" ]; then
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
      printf '%s' "$CUDA_VISIBLE_DEVICES"
    else
      printf 'empty'
    fi
  else
    printf 'unset'
  fi
}

planned_configs() {
  printf '%s\n' "$((${#SELECTED_BASELINES[@]} * ${#SWEEP_GRID[@]}))"
}

write_manifest() {
  {
    printf 'runbook=%s\n' 'docs/DAGER_PRIVACY_SST2_COLA_RT_3SEED_RUNBOOK_ZH.md'
    printf 'script=%s\n' 'scripts/run_dager_privacy_noise_dpsgd_one_seed.sh'
    printf 'created_at=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
    printf 'selected_baselines=%s\n' "${SELECTED_BASELINES[*]}"
    printf 'dataset=%s\n' "$DATASET"
    printf 'seed=%s\n' "$SEED"
    printf 'model=%s\n' "$MODEL"
    printf 'batch_size=%s\n' "$BATCH_SIZE"
    printf 'n_inputs=%s\n' "$N_INPUTS"
    printf 'split=%s\n' 'val'
    printf 'task=%s\n' 'seq_class'
    printf 'finetuned_path=%s\n' "$FINETUNED_PATH"
    printf 'device=%s\n' 'attack.py --device cuda via scripts/defense_baselines.sh'
    printf 'cuda_visible_devices=%s\n' "$(cuda_visible_devices_label)"
    printf 'threat_surface=%s\n' 'full_gradient_dager'
    printf 'sweep_grid=%s\n' "${SWEEP_GRID[*]}"
    printf 'configs_per_baseline=%s\n' "${#SWEEP_GRID[@]}"
    printf 'planned_configs=%s\n' "$(planned_configs)"
    printf 'formal_dp_claim=%s\n' 'false'
  } > "${DAGER_LOG_DIR}/run_manifest.txt"
}

check_environment() {
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dager-noise-dpsgd] dry-run: skipping Python/CUDA/checkpoint environment checks." >&2
    return 0
  fi

  if [ ! -d "$FINETUNED_PATH" ]; then
    echo "[dager-noise-dpsgd] missing finetuned checkpoint: ${FINETUNED_PATH}" >&2
    exit 2
  fi

  python3 -c "import torch; print('cuda_available=', torch.cuda.is_available()); raise SystemExit(0 if torch.cuda.is_available() else 1)"
  python3 -c "from utils.gpu import resolve_cuda_device; print('resolved_cuda_device=', resolve_cuda_device('cuda'))"
}

run_cmd() {
  printf '[dager-noise-dpsgd] run:'
  printf ' %q' "$@"
  printf '\n'
  if [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  "$@"
}

echo "[dager-noise-dpsgd] root: ${DAGER_ROOT}" >&2
echo "[dager-noise-dpsgd] dataset: ${DATASET}" >&2
echo "[dager-noise-dpsgd] selected baselines: ${SELECTED_BASELINES[*]}" >&2
echo "[dager-noise-dpsgd] seed: ${SEED}" >&2
echo "[dager-noise-dpsgd] checkpoint: ${FINETUNED_PATH}" >&2
echo "[dager-noise-dpsgd] log dir: ${DAGER_LOG_DIR}" >&2
echo "[dager-noise-dpsgd] sweep points per baseline: ${#SWEEP_GRID[@]}" >&2
echo "[dager-noise-dpsgd] planned configs: $(planned_configs)" >&2
echo "[dager-noise-dpsgd] CUDA_VISIBLE_DEVICES=$(cuda_visible_devices_label)" >&2

write_manifest
check_environment

for baseline in "${SELECTED_BASELINES[@]}"; do
  echo "[dager-noise-dpsgd] stage: ${baseline} dense sweep" >&2
  for param in "${SWEEP_GRID[@]}"; do
    run_cmd bash scripts/defense_baselines.sh "$DATASET" "$BATCH_SIZE" "$MODEL" "$N_INPUTS" \
      --baseline_defense "$baseline" \
      --baseline_param "$param" \
      --skip_anchor_none \
      --finetuned_path "$FINETUNED_PATH" \
      --rng_seed "$SEED"
  done
done

if [ "$NO_COLLECT" -eq 0 ]; then
  echo "[dager-noise-dpsgd] stage: collect" >&2
  run_cmd python3 scripts/collect_experiment_logs.py "$DAGER_LOG_DIR" \
    -o "$DAGER_LOG_DIR/all_results.csv" \
    --markdown "$DAGER_LOG_DIR/all_results.md"
fi

echo "[dager-noise-dpsgd] done: ${DAGER_LOG_DIR}" >&2
