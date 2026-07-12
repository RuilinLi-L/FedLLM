#!/usr/bin/env bash
# Run selected-seed, GPU full-gradient DAGER privacy baselines for
# SST-2, CoLA, and Rotten Tomatoes.

set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage:" \
    "  bash scripts/run_dager_privacy_selected_baselines_sst2_cola_rt_3seed.sh [options]" \
    "" \
    "Options:" \
    "  --baselines LIST      Baselines to run. Default: all." \
    "                        Accepts comma-separated or space-separated names." \
    "                        Allowed: all, none, lrbprojonly, lrb, topk, compression," \
    "                                 noise, dpsgd, dpsgd_opacus, mixup, soteria." \
    "  --seeds LIST          Seeds to run. Default: 101 202 303." \
    "                        Accepts comma-separated or space-separated integers." \
    "  --gpu auto|all|ID     GPU visibility mode. Default: auto." \
    "  --log-dir PATH        Output log root. Default includes the selected seed tag." \
    "  --n-inputs N          Formal DAGER n_inputs. Default: 100." \
    "  --dry-run             Print commands without executing them." \
    "  --skip-main           Skip non-none baseline sweeps." \
    "  --run-adaptive        Also run defense-aware adaptive checks." \
    "  --no-collect          Skip final collect_experiment_logs.py aggregation." \
    "  -h, --help            Show this help." \
    "" \
    "Examples:" \
    "  bash scripts/run_dager_privacy_selected_baselines_sst2_cola_rt_3seed.sh --baselines topk" \
    "  bash scripts/run_dager_privacy_selected_baselines_sst2_cola_rt_3seed.sh --baselines none,topk" \
    "  bash scripts/run_dager_privacy_selected_baselines_sst2_cola_rt_3seed.sh --baselines noise dpsgd mixup soteria" \
    "  bash scripts/run_dager_privacy_selected_baselines_sst2_cola_rt_3seed.sh --baselines topk --run-adaptive" \
    "  bash scripts/run_dager_privacy_selected_baselines_sst2_cola_rt_3seed.sh --baselines dpsgd_opacus --seeds 101" \
    "" \
    "This script exports the selected DAGER_SEEDS and never passes --rng_seed to" \
    "defense_baselines.sh. Non-none selected baselines are called with" \
    "--skip_anchor_none so selecting topk only runs topk, not repeated clean none anchors." \
    "Parameter grids are aligned with run_dager_privacy_one_dataset_baselines.sh." \
    "defense_baselines.sh calls attack.py with --device cuda; bare cuda is resolved" \
    "by utils.gpu.resolve_cuda_device()."
}

die_usage() {
  echo "[dager-selected] $1" >&2
  usage >&2
  exit 2
}

GPU_REQUEST="auto"
GPU_MODE="auto"
LOG_DIR=""
N_INPUTS="100"
DRY_RUN=0
SKIP_MAIN=0
RUN_ADAPTIVE=0
NO_COLLECT=0
BASELINES_REQUEST="all"
SEEDS_REQUEST="101 202 303"

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
      BASELINES_REQUEST="${1#*=}"; shift ;;
    --seeds)
      shift
      if [ "$#" -eq 0 ] || [[ "$1" == -* ]]; then
        die_usage "--seeds requires at least one value."
      fi
      SEEDS_REQUEST=""
      while [ "$#" -gt 0 ] && [[ "$1" != -* ]]; do
        SEEDS_REQUEST="${SEEDS_REQUEST}${SEEDS_REQUEST:+ }$1"
        shift
      done
      ;;
    --seeds=*)
      SEEDS_REQUEST="${1#*=}"; shift ;;
    --gpu)
      if [ "$#" -lt 2 ]; then die_usage "--gpu requires a value."; fi
      GPU_REQUEST="$2"; shift 2 ;;
    --gpu=*)
      GPU_REQUEST="${1#*=}"; shift ;;
    --log-dir)
      if [ "$#" -lt 2 ]; then die_usage "--log-dir requires a value."; fi
      LOG_DIR="$2"; shift 2 ;;
    --log-dir=*)
      LOG_DIR="${1#*=}"; shift ;;
    --n-inputs)
      if [ "$#" -lt 2 ]; then die_usage "--n-inputs requires a value."; fi
      N_INPUTS="$2"; shift 2 ;;
    --n-inputs=*)
      N_INPUTS="${1#*=}"; shift ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    --skip-main)
      SKIP_MAIN=1; shift ;;
    --run-adaptive)
      RUN_ADAPTIVE=1; shift ;;
    --skip-adaptive)
      RUN_ADAPTIVE=0; shift ;;
    --no-collect)
      NO_COLLECT=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      die_usage "unknown option: $1" ;;
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

DATASETS=(sst2 cola rotten_tomatoes)
ALL_BASELINES=(none lrbprojonly lrb topk compression noise dpsgd dpsgd_opacus mixup soteria)
MAIN_DEFENSES=(lrbprojonly lrb topk compression noise dpsgd dpsgd_opacus mixup soteria)

declare -A CKPT=(
  [sst2]="./models/gpt2_sst2_clean_num_epochs_2/final"
  [cola]="./models/gpt2_cola_clean_num_epochs_2/final"
  [rotten_tomatoes]="./models/gpt2_rotten_tomatoes_clean_num_epochs_2/final"
)

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

SELECTED_BASELINES=()
declare -A SELECTED_LOOKUP=()

baseline_is_allowed() {
  local candidate="$1"
  local baseline
  for baseline in "${ALL_BASELINES[@]}"; do
    if [ "$baseline" = "$candidate" ]; then
      return 0
    fi
  done
  return 1
}

baseline_selected() {
  local candidate="$1"
  [ "${SELECTED_LOOKUP[$candidate]+x}" = "x" ]
}

normalize_selected_baselines() {
  local raw
  local token
  local has_all=0
  declare -A requested=()

  raw="${BASELINES_REQUEST//,/ }"
  if [ -z "${raw// /}" ]; then
    die_usage "--baselines cannot be empty."
  fi

  for token in $raw; do
    if [ "$token" = "all" ]; then
      has_all=1
      continue
    fi
    if ! baseline_is_allowed "$token"; then
      die_usage "unsupported baseline: ${token}"
    fi
    requested[$token]=1
  done

  if [ "$has_all" -eq 1 ]; then
    SELECTED_BASELINES=( "${ALL_BASELINES[@]}" )
  else
    local baseline
    SELECTED_BASELINES=()
    for baseline in "${ALL_BASELINES[@]}"; do
      if [ "${requested[$baseline]+x}" = "x" ]; then
        SELECTED_BASELINES+=( "$baseline" )
      fi
    done
  fi

  if [ "${#SELECTED_BASELINES[@]}" -eq 0 ]; then
    die_usage "no baselines selected."
  fi

  SELECTED_LOOKUP=()
  for token in "${SELECTED_BASELINES[@]}"; do
    SELECTED_LOOKUP[$token]=1
  done
}

normalize_requested_seeds() {
  local raw
  local token
  declare -A seen=()

  raw="${SEEDS_REQUEST//,/ }"
  DAGER_SEEDS=""
  for token in $raw; do
    if ! [[ "$token" =~ ^[0-9]+$ ]]; then
      die_usage "invalid seed: ${token}; seeds must be non-negative integers."
    fi
    if [ "${seen[$token]+x}" != "x" ]; then
      DAGER_SEEDS="${DAGER_SEEDS}${DAGER_SEEDS:+ }${token}"
      seen[$token]=1
    fi
  done
  if [ -z "$DAGER_SEEDS" ]; then
    die_usage "--seeds cannot be empty."
  fi
  export DAGER_SEEDS
}

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

run_cmd() {
  printf '[dager-selected] run:'
  printf ' %q' "$@"
  printf '\n'
  if [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  "$@"
}

run_focused_param_grid() {
  local ds="$1"
  local defense="$2"
  shift 2
  local param
  for param in "$@"; do
    run_cmd bash scripts/defense_baselines.sh "$ds" 2 gpt2 "$N_INPUTS" \
      --baseline_defense "$defense" \
      --baseline_param "$param" \
      --skip_anchor_none \
      --finetuned_path "${CKPT[$ds]}"
  done
}

run_main_baseline() {
  local ds="$1"
  local defense="$2"
  case "$defense" in
    lrbprojonly|lrb)
      run_focused_param_grid "$ds" "$defense" "${LRB_GRID[@]}"
      ;;
    topk)
      run_focused_param_grid "$ds" "$defense" "${TOPK_GRID[@]}"
      ;;
    compression)
      run_focused_param_grid "$ds" "$defense" "${COMPRESSION_GRID[@]}"
      ;;
    noise)
      run_focused_param_grid "$ds" "$defense" "${NOISE_GRID[@]}"
      ;;
    dpsgd)
      run_focused_param_grid "$ds" "$defense" "${DPSGD_GRID[@]}"
      ;;
    dpsgd_opacus)
      run_focused_param_grid "$ds" "$defense" "${DPSGD_OPACUS_GRID[@]}"
      ;;
    soteria)
      run_focused_param_grid "$ds" "$defense" "${SOTERIA_GRID[@]}"
      ;;
    mixup)
      run_focused_param_grid "$ds" "$defense" "${MIXUP_GRID[@]}"
      ;;
    *)
      echo "[dager-selected] unsupported main baseline: ${defense}" >&2
      exit 2
      ;;
  esac
}

run_adaptive_param_grid() {
  local ds="$1"
  local defense="$2"
  shift 2
  local param
  for param in "$@"; do
    run_cmd bash scripts/defense_baselines.sh "$ds" 2 gpt2 "$N_INPUTS" \
      --baseline_defense "$defense" \
      --baseline_param "$param" \
      --adaptive_attack_check \
      --skip_anchor_none \
      --finetuned_path "${CKPT[$ds]}"
  done
}

run_adaptive_baseline() {
  local ds="$1"
  local defense="$2"
  case "$defense" in
    lrbprojonly)
      run_adaptive_param_grid "$ds" "$defense" "${ADAPTIVE_LRBPROJONLY_GRID[@]}"
      ;;
    topk)
      run_adaptive_param_grid "$ds" "$defense" "${ADAPTIVE_TOPK_GRID[@]}"
      ;;
    compression)
      run_adaptive_param_grid "$ds" "$defense" "${ADAPTIVE_COMPRESSION_GRID[@]}"
      ;;
    dpsgd_opacus)
      run_adaptive_param_grid "$ds" "$defense" "${ADAPTIVE_DPSGD_OPACUS_GRID[@]}"
      ;;
  esac
}

dpsgd_opacus_selected() {
  baseline_selected dpsgd_opacus
}

grid_count_for_baseline() {
  case "$1" in
    none)
      printf '1'
      ;;
    lrbprojonly|lrb)
      printf '%s' "${#LRB_GRID[@]}"
      ;;
    topk)
      printf '%s' "${#TOPK_GRID[@]}"
      ;;
    compression)
      printf '%s' "${#COMPRESSION_GRID[@]}"
      ;;
    noise)
      printf '%s' "${#NOISE_GRID[@]}"
      ;;
    dpsgd)
      printf '%s' "${#DPSGD_GRID[@]}"
      ;;
    dpsgd_opacus)
      printf '%s' "${#DPSGD_OPACUS_GRID[@]}"
      ;;
    soteria)
      printf '%s' "${#SOTERIA_GRID[@]}"
      ;;
    mixup)
      printf '%s' "${#MIXUP_GRID[@]}"
      ;;
    *)
      printf '0'
      ;;
  esac
}

planned_main_configs_per_dataset() {
  local total=0
  local baseline
  for baseline in "${SELECTED_BASELINES[@]}"; do
    if [ "$baseline" = "none" ] || [ "$SKIP_MAIN" -eq 0 ]; then
      total=$((total + $(grid_count_for_baseline "$baseline")))
    fi
  done
  printf '%s\n' "$total"
}

planned_adaptive_configs_per_dataset() {
  local total=0
  if [ "$RUN_ADAPTIVE" -eq 0 ]; then
    printf '0\n'
    return 0
  fi
  if baseline_selected lrbprojonly; then
    total=$((total + ${#ADAPTIVE_LRBPROJONLY_GRID[@]}))
  fi
  if baseline_selected topk; then
    total=$((total + ${#ADAPTIVE_TOPK_GRID[@]}))
  fi
  if baseline_selected compression; then
    total=$((total + ${#ADAPTIVE_COMPRESSION_GRID[@]}))
  fi
  if baseline_selected dpsgd_opacus; then
    total=$((total + ${#ADAPTIVE_DPSGD_OPACUS_GRID[@]}))
  fi
  printf '%s\n' "$total"
}

write_manifest() {
  local created_at
  printf -v created_at '%(%Y-%m-%d %H:%M:%S)T' -1
  {
    printf 'runbook=%s\n' 'docs/DAGER_PRIVACY_SST2_COLA_RT_3SEED_RUNBOOK_ZH.md'
    printf 'script=%s\n' 'scripts/run_dager_privacy_selected_baselines_sst2_cola_rt_3seed.sh'
    printf 'created_at=%s\n' "$created_at"
    printf 'datasets=%s\n' "${DATASETS[*]}"
    printf 'selected_baselines=%s\n' "${SELECTED_BASELINES[*]}"
    printf 'model=%s\n' 'gpt2'
    printf 'batch_size=%s\n' '2'
    printf 'n_inputs=%s\n' "$N_INPUTS"
    printf 'seeds=%s\n' "$DAGER_SEEDS"
    printf 'checkpoint_sst2=%s\n' "${CKPT[sst2]}"
    printf 'checkpoint_cola=%s\n' "${CKPT[cola]}"
    printf 'checkpoint_rotten_tomatoes=%s\n' "${CKPT[rotten_tomatoes]}"
    printf 'gpu_request=%s\n' "$GPU_REQUEST"
    printf 'gpu_mode=%s\n' "$GPU_MODE"
    printf 'cuda_visible_devices=%s\n' "$(cuda_visible_devices_label)"
    printf 'device=%s\n' 'attack.py --device cuda via scripts/defense_baselines.sh'
    printf 'threat_surface=%s\n' 'full_gradient_dager'
    printf 'skip_main=%s\n' "$SKIP_MAIN"
    printf 'run_adaptive=%s\n' "$RUN_ADAPTIVE"
    printf 'main_configs_per_dataset=%s\n' "$(planned_main_configs_per_dataset)"
    printf 'adaptive_configs_per_dataset=%s\n' "$(planned_adaptive_configs_per_dataset)"
    printf 'main_defenses=%s\n' "${MAIN_DEFENSES[*]}"
    printf 'lrb_grid=%s\n' "${LRB_GRID[*]}"
    printf 'topk_grid=%s\n' "${TOPK_GRID[*]}"
    printf 'compression_grid=%s\n' "${COMPRESSION_GRID[*]}"
    printf 'noise_grid=%s\n' "${NOISE_GRID[*]}"
    printf 'dpsgd_grid=%s\n' "${DPSGD_GRID[*]}"
    printf 'dpsgd_opacus_grid=%s\n' "${DPSGD_OPACUS_GRID[*]}"
    printf 'soteria_grid=%s\n' "${SOTERIA_GRID[*]}"
    printf 'mixup_grid=%s\n' "${MIXUP_GRID[*]}"
    printf 'adaptive_lrbprojonly_grid=%s\n' "${ADAPTIVE_LRBPROJONLY_GRID[*]}"
    printf 'adaptive_topk_grid=%s\n' "${ADAPTIVE_TOPK_GRID[*]}"
    printf 'adaptive_compression_grid=%s\n' "${ADAPTIVE_COMPRESSION_GRID[*]}"
    printf 'adaptive_dpsgd_opacus_grid=%s\n' "${ADAPTIVE_DPSGD_OPACUS_GRID[*]}"
    if dpsgd_opacus_selected; then
      printf 'dpsgd_opacus=%s\n' 'selected'
    else
      printf 'dpsgd_opacus=%s\n' 'not_selected'
    fi
    printf 'formal_dp_claim=%s\n' 'false'
  } > "${DAGER_LOG_DIR}/run_manifest.txt"
}

check_environment() {
  local ds
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dager-selected] dry-run: skipping Python/CUDA/checkpoint environment checks." >&2
    return 0
  fi

  for ds in "${DATASETS[@]}"; do
    if [ ! -d "${CKPT[$ds]}" ]; then
      echo "[dager-selected] missing finetuned checkpoint for ${ds}: ${CKPT[$ds]}" >&2
      exit 2
    fi
  done

  python3 -c "import torch; print('cuda_available=', torch.cuda.is_available()); raise SystemExit(0 if torch.cuda.is_available() else 1)"
  python3 -c "from utils.gpu import resolve_cuda_device; print('resolved_cuda_device=', resolve_cuda_device('cuda'))"

  if dpsgd_opacus_selected; then
    if python3 -c "import opacus; print('opacus ok')" ; then
      :
    else
      echo "[dager-selected] warning: opacus is unavailable; dpsgd_opacus rows may be logged as failures." >&2
    fi
  fi
}

normalize_requested_seeds
normalize_selected_baselines
configure_gpu_visibility

if [ -z "$LOG_DIR" ]; then
  printf -v STAMP '%(%Y%m%d_%H%M%S)T' -1
  if [ "$DAGER_SEEDS" = "101 202 303" ]; then
    SEED_TAG="3seed"
  else
    SEED_TAG="seeds_${DAGER_SEEDS// /-}"
  fi
  LOG_DIR="log/runs/dager_privacy_selected_baselines_sst2_cola_rt_${SEED_TAG}_${STAMP}"
fi
export DAGER_LOG_DIR="$LOG_DIR"
mkdir -p "$DAGER_LOG_DIR"

echo "[dager-selected] root: ${DAGER_ROOT}" >&2
echo "[dager-selected] log dir: ${DAGER_LOG_DIR}" >&2
echo "[dager-selected] datasets: ${DATASETS[*]}" >&2
echo "[dager-selected] selected baselines: ${SELECTED_BASELINES[*]}" >&2
echo "[dager-selected] seeds: ${DAGER_SEEDS}" >&2
echo "[dager-selected] planned main configs per dataset: $(planned_main_configs_per_dataset)" >&2
echo "[dager-selected] planned adaptive configs per dataset: $(planned_adaptive_configs_per_dataset) (run_adaptive=${RUN_ADAPTIVE})" >&2
echo "[dager-selected] gpu mode: ${GPU_MODE} (request=${GPU_REQUEST})" >&2
echo "[dager-selected] CUDA_VISIBLE_DEVICES=$(cuda_visible_devices_label)" >&2
echo "[dager-selected] checkpoints:" >&2
for ds in "${DATASETS[@]}"; do
  echo "[dager-selected]   ${ds}: ${CKPT[$ds]}" >&2
done

write_manifest
check_environment

if baseline_selected none; then
  echo "[dager-selected] stage: selected none anchor" >&2
  for ds in "${DATASETS[@]}"; do
    run_cmd bash scripts/defense_baselines.sh "$ds" 2 gpt2 "$N_INPUTS" \
      --baseline_defense none \
      --finetuned_path "${CKPT[$ds]}"
  done
fi

if [ "$SKIP_MAIN" -eq 0 ]; then
  echo "[dager-selected] stage: selected main parameter sweeps" >&2
  for ds in "${DATASETS[@]}"; do
    for defense in "${MAIN_DEFENSES[@]}"; do
      if baseline_selected "$defense"; then
        run_main_baseline "$ds" "$defense"
      fi
    done
  done
fi

if [ "$RUN_ADAPTIVE" -eq 1 ]; then
  echo "[dager-selected] stage: selected adaptive checks" >&2
  for ds in "${DATASETS[@]}"; do
    for defense in "${MAIN_DEFENSES[@]}"; do
      if baseline_selected "$defense"; then
        run_adaptive_baseline "$ds" "$defense"
      fi
    done
  done
fi

if [ "$NO_COLLECT" -eq 0 ]; then
  echo "[dager-selected] stage: collect" >&2
  run_cmd python3 scripts/collect_experiment_logs.py "$DAGER_LOG_DIR" \
    -o "$DAGER_LOG_DIR/all_results.csv" \
    --markdown "$DAGER_LOG_DIR/all_results.md"
fi

echo "[dager-selected] done: ${DAGER_LOG_DIR}" >&2
