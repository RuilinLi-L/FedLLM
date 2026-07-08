#!/usr/bin/env bash
# Run the three-seed, GPU full-gradient DAGER privacy sweep for
# SST-2, CoLA, and Rotten Tomatoes.

set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage:" \
    "  bash scripts/run_dager_privacy_sst2_cola_rt_3seed.sh [options]" \
    "" \
    "Options:" \
    "  --gpu auto|all|ID     GPU visibility mode. Default: auto." \
    "  --log-dir PATH        Output log root. Default: log/runs/dager_privacy_sst2_cola_rt_3seed_<timestamp>." \
    "  --n-inputs N          Formal DAGER n_inputs. Default: 100." \
    "  --smoke-inputs N      Smoke-test n_inputs. Default: 2." \
    "  --dry-run             Print commands without executing them." \
    "  --skip-smoke          Skip smoke tests." \
    "  --skip-main           Skip built-in baseline sweeps." \
    "  --skip-extra          Skip extra boundary points." \
    "  --skip-adaptive       Skip defense-aware adaptive checks." \
    "  --no-collect          Skip final collect_experiment_logs.py aggregation." \
    "  -h, --help            Show this help." \
    "" \
    "This script intentionally fixes DAGER_SEEDS=\"101 202 303\" and never passes" \
    "--rng_seed to defense_baselines.sh. defense_baselines.sh calls attack.py with" \
    "--device cuda; bare cuda is resolved by utils.gpu.resolve_cuda_device()." \
    "" \
    "GPU modes:" \
    "  auto                 Preserve the current CUDA_VISIBLE_DEVICES and let attack.py choose an idle visible GPU." \
    "  all                  Unset CUDA_VISIBLE_DEVICES, then let attack.py choose from all GPUs." \
    "  0 / 1 / 0,1          Restrict CUDA_VISIBLE_DEVICES manually, then let attack.py choose within that range."
}

GPU_REQUEST="auto"
GPU_MODE="auto"
LOG_DIR=""
N_INPUTS="100"
SMOKE_INPUTS="2"
DRY_RUN=0
SKIP_SMOKE=0
SKIP_MAIN=0
SKIP_EXTRA=0
SKIP_ADAPTIVE=0
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
    --smoke-inputs)
      SMOKE_INPUTS="$2"; shift 2 ;;
    --smoke-inputs=*)
      SMOKE_INPUTS="${1#*=}"; shift ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    --skip-smoke)
      SKIP_SMOKE=1; shift ;;
    --skip-main)
      SKIP_MAIN=1; shift ;;
    --skip-extra)
      SKIP_EXTRA=1; shift ;;
    --skip-adaptive)
      SKIP_ADAPTIVE=1; shift ;;
    --no-collect)
      NO_COLLECT=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[dager-privacy] unknown option: $1" >&2
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

export DAGER_SEEDS="101 202 303"

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
  LOG_DIR="log/runs/dager_privacy_sst2_cola_rt_3seed_${STAMP}"
fi
export DAGER_LOG_DIR="$LOG_DIR"
mkdir -p "$DAGER_LOG_DIR"

run_cmd() {
  printf '[dager-privacy] run:'
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
    printf 'runbook=%s\n' 'docs/DAGER_PRIVACY_SST2_COLA_RT_3SEED_RUNBOOK_ZH.md'
    printf 'script=%s\n' 'scripts/run_dager_privacy_sst2_cola_rt_3seed.sh'
    printf 'created_at=%s\n' "$created_at"
    printf 'datasets=%s\n' 'sst2 cola rotten_tomatoes'
    printf 'model=%s\n' 'gpt2'
    printf 'batch_size=%s\n' '2'
    printf 'n_inputs=%s\n' "$N_INPUTS"
    printf 'smoke_inputs=%s\n' "$SMOKE_INPUTS"
    printf 'seeds=%s\n' "$DAGER_SEEDS"
    printf 'gpu_request=%s\n' "$GPU_REQUEST"
    printf 'gpu_mode=%s\n' "$GPU_MODE"
    printf 'cuda_visible_devices=%s\n' "$(cuda_visible_devices_label)"
    printf 'device=%s\n' 'attack.py --device cuda via scripts/defense_baselines.sh'
    printf 'threat_surface=%s\n' 'full_gradient_dager'
    printf 'dpsgd_opacus=%s\n' 'included'
    printf 'formal_dp_claim=%s\n' 'false'
  } > "${DAGER_LOG_DIR}/run_manifest.txt"
}

check_environment() {
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dager-privacy] dry-run: skipping Python/CUDA environment checks." >&2
    return 0
  fi

  python3 -c "import torch; print('cuda_available=', torch.cuda.is_available()); raise SystemExit(0 if torch.cuda.is_available() else 1)"
  python3 -c "from utils.gpu import resolve_cuda_device; print('resolved_cuda_device=', resolve_cuda_device('cuda'))"

  if python3 -c "import opacus; print('opacus ok')" ; then
    :
  else
    echo "[dager-privacy] warning: opacus is unavailable; dpsgd_opacus rows may be logged as failures." >&2
  fi
}

declare -A CKPT=(
  [sst2]="./models/gpt2_sst2_clean_num_epochs_2/final"
  [cola]="./models/gpt2_cola_clean_num_epochs_2/final"
  [rotten_tomatoes]="./models/gpt2_rotten_tomatoes_clean_num_epochs_2/final"
)

DATASETS=(sst2 cola rotten_tomatoes)
MAIN_DEFENSES=(lrbprojonly lrb topk compression noise dpsgd dpsgd_opacus mixup soteria)

declare -A EXTRA=(
  [lrbprojonly]="0.75 0.995 1.0"
  [lrb]="0.75 0.995 1.0"
  [topk]="0.03 0.15 0.25 0.35"
  [compression]="6 10 14"
  [dpsgd_opacus]="1e-4 3e-4 5e-4 1e-3 3e-3 1e-2"
)

declare -A ADAPTIVE=(
  [lrbprojonly]="0.65 0.90 0.95 0.99"
  [topk]="0.10 0.30 0.50"
  [compression]="8 16 24 32"
  [dpsgd_opacus]="5e-4 1e-3 1e-2"
)

echo "[dager-privacy] root: ${DAGER_ROOT}" >&2
echo "[dager-privacy] log dir: ${DAGER_LOG_DIR}" >&2
echo "[dager-privacy] seeds: ${DAGER_SEEDS}" >&2
echo "[dager-privacy] gpu mode: ${GPU_MODE} (request=${GPU_REQUEST})" >&2
echo "[dager-privacy] CUDA_VISIBLE_DEVICES=$(cuda_visible_devices_label)" >&2

write_manifest
check_environment

if [ "$SKIP_SMOKE" -eq 0 ]; then
  echo "[dager-privacy] stage: smoke" >&2
  for ds in "${DATASETS[@]}"; do
    run_cmd bash scripts/defense_baselines.sh "$ds" 2 gpt2 "$SMOKE_INPUTS" \
      --baseline_defense none \
      --finetuned_path "${CKPT[$ds]}"
  done
fi

if [ "$SKIP_MAIN" -eq 0 ]; then
  echo "[dager-privacy] stage: main sweeps" >&2
  for ds in "${DATASETS[@]}"; do
    for defense in "${MAIN_DEFENSES[@]}"; do
      run_cmd bash scripts/defense_baselines.sh "$ds" 2 gpt2 "$N_INPUTS" \
        --baseline_defense "$defense" \
        --finetuned_path "${CKPT[$ds]}"
    done
  done
fi

if [ "$SKIP_EXTRA" -eq 0 ]; then
  echo "[dager-privacy] stage: extra boundary points" >&2
  for ds in "${DATASETS[@]}"; do
    for defense in "${!EXTRA[@]}"; do
      for param in ${EXTRA[$defense]}; do
        run_cmd bash scripts/defense_baselines.sh "$ds" 2 gpt2 "$N_INPUTS" \
          --baseline_defense "$defense" \
          --baseline_param "$param" \
          --finetuned_path "${CKPT[$ds]}"
      done
    done
  done
fi

if [ "$SKIP_ADAPTIVE" -eq 0 ]; then
  echo "[dager-privacy] stage: adaptive checks" >&2
  for ds in "${DATASETS[@]}"; do
    for defense in "${!ADAPTIVE[@]}"; do
      for param in ${ADAPTIVE[$defense]}; do
        run_cmd bash scripts/defense_baselines.sh "$ds" 2 gpt2 "$N_INPUTS" \
          --baseline_defense "$defense" \
          --baseline_param "$param" \
          --adaptive_attack_check \
          --finetuned_path "${CKPT[$ds]}"
      done
    done
  done
fi

if [ "$NO_COLLECT" -eq 0 ]; then
  echo "[dager-privacy] stage: collect" >&2
  run_cmd python3 scripts/collect_experiment_logs.py "$DAGER_LOG_DIR" \
    -o "$DAGER_LOG_DIR/all_results.csv" \
    --markdown "$DAGER_LOG_DIR/all_results.md"
fi

echo "[dager-privacy] done: ${DAGER_LOG_DIR}" >&2
