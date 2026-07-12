#!/bin/bash
# Run the smallest useful 3-seed utility matrix selected from the completed
# CoLA and Rotten Tomatoes DAGER privacy sweeps.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAGER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$DAGER_ROOT" || exit 1

PROFILE="p0"
DATASETS="cola,rotten_tomatoes"
GPU_REQUEST="auto"
DRY_RUN=0
RUN_DIR=""
SEEDS=(101 202 303)
FAILURES=0

usage() {
  cat <<'EOF'
Usage: bash scripts/run_dager_utility_cola_rt_critical.sh [options]

Options:
  --profile p0|p1|all       p0: main-table points; p1: extra mechanism points
  --datasets LIST           Comma-separated: cola,rotten_tomatoes
  --gpu auto|all|ID[,ID]    Preserve, clear, or set CUDA_VISIBLE_DEVICES
  --log-dir PATH            Reuse a run directory (successful logs are skipped)
  --dry-run                 Print commands without running training
  -h, --help                Show this help

P0 per dataset: none, Projection-LRB@0.99, privacy-boundary top-k, compression@20.
P1 per dataset: Projection-LRB@0.90 and full LRB@0.50.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile)
      PROFILE="${2:?--profile requires a value}"
      shift 2
      ;;
    --profile=*)
      PROFILE="${1#*=}"
      shift
      ;;
    --datasets)
      DATASETS="${2:?--datasets requires a value}"
      shift 2
      ;;
    --datasets=*)
      DATASETS="${1#*=}"
      shift
      ;;
    --gpu)
      GPU_REQUEST="${2:?--gpu requires a value}"
      shift 2
      ;;
    --gpu=*)
      GPU_REQUEST="${1#*=}"
      shift
      ;;
    --log-dir)
      RUN_DIR="${2:?--log-dir requires a value}"
      shift 2
      ;;
    --log-dir=*)
      RUN_DIR="${1#*=}"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[utility-critical] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$PROFILE" in
  p0|p1|all) ;;
  *)
    echo "[utility-critical] --profile must be p0, p1, or all." >&2
    exit 2
    ;;
esac

case "$GPU_REQUEST" in
  auto)
    ;;
  all)
    unset CUDA_VISIBLE_DEVICES
    ;;
  *)
    export CUDA_VISIBLE_DEVICES="$GPU_REQUEST"
    ;;
esac

IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"
for dataset in "${DATASET_LIST[@]}"; do
  case "$dataset" in
    cola|rotten_tomatoes) ;;
    *)
      echo "[utility-critical] Unsupported dataset: $dataset" >&2
      exit 2
      ;;
  esac
done

if [ -z "$RUN_DIR" ]; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  RUN_DIR="log/runs/dager_utility_cola_rt_critical_${PROFILE}_${STAMP}"
fi

EXIT_CODES="${RUN_DIR}/exit_codes.csv"
if [ "$DRY_RUN" -eq 0 ]; then
  mkdir -p "$RUN_DIR"
  if [ ! -f "$EXIT_CODES" ]; then
    printf 'dataset,tag,seed,exit_code\n' > "$EXIT_CODES"
  fi

  cat > "${RUN_DIR}/run_manifest.txt" <<EOF
script=scripts/run_dager_utility_cola_rt_critical.sh
profile=${PROFILE}
datasets=${DATASETS}
batch_size=2
num_epochs=1
seeds=${SEEDS[*]}
gpu_request=${GPU_REQUEST}
threat_surface=full_gradient_dager_selected_utility
p0=none,lrbprojonly@0.99,topk@privacy_boundary,compression@20
p1=lrbprojonly@0.90,lrb@0.50
EOF
fi

checkpoint_for() {
  case "$1" in
    cola) printf './models/gpt2_cola_clean_num_epochs_2/final' ;;
    rotten_tomatoes) printf './models/gpt2_rotten_tomatoes_clean_num_epochs_2/final' ;;
  esac
}

topk_boundary_for() {
  case "$1" in
    cola) printf '0.45' ;;
    rotten_tomatoes) printf '0.55' ;;
  esac
}

print_command() {
  printf '[utility-critical] command:'
  printf ' %q' "$@"
  printf '\n'
}

run_variant() {
  local dataset="$1"
  local defense="$2"
  local param="$3"
  local tag="$4"
  local checkpoint
  checkpoint="$(checkpoint_for "$dataset")"

  if [ "$DRY_RUN" -eq 0 ] && [ ! -f "${checkpoint}/config.json" ]; then
    echo "[utility-critical] Missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi

  local defense_args=(--defense "$defense")
  case "$defense" in
    none)
      ;;
    lrbprojonly|lrb)
      defense_args+=(--defense_lrb_keep_ratio_sensitive "$param")
      ;;
    topk)
      defense_args+=(--defense_topk_ratio "$param")
      ;;
    compression)
      defense_args+=(--defense_n_bits "$param")
      ;;
    *)
      echo "[utility-critical] Unsupported defense: ${defense}" >&2
      exit 2
      ;;
  esac

  local dataset_dir="${RUN_DIR}/${dataset}"
  if [ "$DRY_RUN" -eq 0 ]; then
    mkdir -p "${dataset_dir}/models"
  fi

  for seed in "${SEEDS[@]}"; do
    local logfile="${dataset_dir}/train_${tag}_seed${seed}.txt"
    local output_dir="${dataset_dir}/models/${tag}_seed${seed}"

    if [ -f "$logfile" ] && grep -q 'result_status=ok' "$logfile"; then
      echo "[utility-critical] skip completed ${dataset}/${tag}/seed${seed}" >&2
      continue
    fi

    local cmd=(
      python3 train.py
      --dataset "$dataset"
      --task seq_class
      --batch_size 2
      --num_epochs 1
      --save_every 0
      --model_path "$checkpoint"
      --models_cache ./models_cache
      --train_method full
      --rng_seed "$seed"
      --device cuda
      --output_dir "$output_dir"
      "${defense_args[@]}"
      --log_file "$logfile"
    )

    if [ "$DRY_RUN" -eq 1 ]; then
      print_command "${cmd[@]}"
      continue
    fi

    echo "[utility-critical] running ${dataset}/${tag}/seed${seed}" >&2
    set +e
    "${cmd[@]}"
    local rc=$?
    set -e
    printf '%s,%s,%s,%s\n' "$dataset" "$tag" "$seed" "$rc" >> "$EXIT_CODES"
    if [ "$rc" -ne 0 ]; then
      FAILURES=$((FAILURES + 1))
    fi
  done
}

for dataset in "${DATASET_LIST[@]}"; do
  if [ "$PROFILE" = "p0" ] || [ "$PROFILE" = "all" ]; then
    run_variant "$dataset" none n/a none
    run_variant "$dataset" lrbprojonly 0.99 lrbprojonly_0.99
    topk_boundary="$(topk_boundary_for "$dataset")"
    run_variant "$dataset" topk "$topk_boundary" "topk_${topk_boundary}"
    run_variant "$dataset" compression 20 compression_20
  fi

  if [ "$PROFILE" = "p1" ] || [ "$PROFILE" = "all" ]; then
    run_variant "$dataset" lrbprojonly 0.90 lrbprojonly_0.90
    run_variant "$dataset" lrb 0.50 lrb_0.50
  fi
done

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[utility-critical] dry run complete; no training was started." >&2
  exit 0
fi

python3 scripts/collect_experiment_logs.py "$RUN_DIR" \
  -o "${RUN_DIR}/results.csv" \
  --markdown "${RUN_DIR}/results.md" \
  --utility-output "${RUN_DIR}/utility_results.csv" \
  --utility-markdown "${RUN_DIR}/utility_results.md"

if [ "$FAILURES" -ne 0 ]; then
  echo "[utility-critical] ${FAILURES} training run(s) failed; inspect ${EXIT_CODES}." >&2
  exit 1
fi

echo "[utility-critical] done: ${RUN_DIR}" >&2
