#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/run_uniform_oracle_resolution_sweep.sh [options]

Options:
  --checkpoint PATH   default: ./models/gpt2_sst2_clean_num_epochs_2/final
  --python PATH       default: python
  --device DEVICE     default: cuda
  --run_root PATH     default: log/runs/uniform_oracle_resolution_sst2_official_validation
  --dry_run           print the delegated commands without running attacks
  -h, --help

The script runs smoke checks first, then formal three-seed jobs for:
  - uniform signed reconstruction at r=0.65, 0.75, and 0.9
  - an undefended defense-aware DAGER anchor

The audited r=0.5 signed and unsigned controls are reused from the existing
20260718_114719 static matrix and are not rerun here.
EOF
}

CHECKPOINT="./models/gpt2_sst2_clean_num_epochs_2/final"
PYTHON_BIN="python"
DEVICE="cuda"
RUN_ROOT="log/runs/uniform_oracle_resolution_sst2_official_validation"
DRY_RUN=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --checkpoint=*) CHECKPOINT="${1#*=}"; shift ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --python=*) PYTHON_BIN="${1#*=}"; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --device=*) DEVICE="${1#*=}"; shift ;;
    --run_root) RUN_ROOT="$2"; shift 2 ;;
    --run_root=*) RUN_ROOT="${1#*=}"; shift ;;
    --dry_run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[uniform-oracle] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -e "$CHECKPOINT" ]; then
  echo "[uniform-oracle] checkpoint not found: $CHECKPOINT" >&2
  exit 2
fi

COMMON_ARGS=(
  --dataset sst2
  --checkpoint "$CHECKPOINT"
  --split official_validation
  --batch_size 2
  --knowledge oracle
  --sign_source defense_device
  --defense_seed_mode static
  --defense_seed 700001
  --seed_samples 1
  --reduce min
  --candidate_multiplier 100
  --device "$DEVICE"
  --python "$PYTHON_BIN"
  --skip_existing
)

if [ "$DRY_RUN" -eq 1 ]; then
  COMMON_ARGS+=( --dry_run )
fi

ratio_slug() {
  printf '%s' "$1" | tr '.' 'p'
}

run_ratio() {
  local mode="$1" ratio="$2" seeds="$3"
  local slug
  slug="$(ratio_slug "$ratio")"
  bash scripts/run_adaptive_lrb_matrix.sh \
    "${COMMON_ARGS[@]}" \
    --n_inputs 100 \
    --k "$ratio" \
    --variants proj_uniform \
    --seeds "$seeds" \
    --mode "$mode" \
    --run_dir "${RUN_ROOT}/${mode}/uniform_r${slug}"
}

run_none() {
  local mode="$1" seeds="$2"
  bash scripts/run_adaptive_lrb_matrix.sh \
    "${COMMON_ARGS[@]}" \
    --n_inputs 100 \
    --k 1.0 \
    --variants none \
    --seeds "$seeds" \
    --mode "$mode" \
    --run_dir "${RUN_ROOT}/${mode}/none_anchor"
}

for ratio in 0.65 0.75 0.9; do
  run_ratio smoke "$ratio" 101
done
run_none smoke 101

for ratio in 0.65 0.75 0.9; do
  run_ratio formal "$ratio" 101,202,303
done
run_none formal 101,202,303

echo "[uniform-oracle] complete: $RUN_ROOT" >&2
echo "[uniform-oracle] analyze with scripts/analyze_uniform_oracle_resolution.py" >&2
