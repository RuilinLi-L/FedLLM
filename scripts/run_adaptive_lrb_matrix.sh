#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$ROOT" || exit 1

DATASET="sst2"
CHECKPOINT=""
SPLIT="official_validation"
N_INPUTS=100
BATCH_SIZE=2
MODEL_PATH="gpt2"
MAIN_K="0.5"
VARIANTS_RAW="proj_only,proj_uniform"
KNOWLEDGE_RAW="oracle,method_only"
DEFENSE_SEED_MODE="static"
DEFENSE_SEED=""
ATTACK_SEED=""
RATIO_GRID="auto"
SEED_SAMPLES_RAW="16"
REDUCE_RAW="min"
SEEDS_RAW="101,202,303"
MODE="formal"
DEVICE="cuda"
CANDIDATE_MULTIPLIER=100
PYTHON_BIN="${PYTHON:-python}"
RUN_DIR=""
SKIP_EXISTING=0
DRY_RUN=0
REUSING_RUN_DIR=0
RUN_FAILURES=0

usage() {
  cat <<'EOF'
Usage: bash scripts/run_adaptive_lrb_matrix.sh --checkpoint PATH [options]

Required:
  --checkpoint PATH

Matrix options:
  --dataset NAME                 default: sst2
  --split val|test|official_validation
  --n_inputs N                  default: 100
  --batch_size N                default: 2
  --model_path PATH             default: gpt2
  --k FLOAT                     default: 0.5
  --variants CSV                LRB presets
  --knowledge CSV               oracle,ratio_hidden,signs_hidden,method_only
  --defense_seed_mode MODE      static|per_update
  --defense_seed N              optional independent LRB seed
  --attack_seed N               required when signs are hidden
  --ratio_grid auto|CSV
  --seed_samples CSV            default: 16
  --reduce CSV                  min,mean
  --seeds CSV                   default: 101,202,303

Execution:
  --mode smoke|formal           smoke forces n_inputs=2 and the first experiment seed
  --candidate_multiplier N      default: 100
  --device DEVICE               default: cuda
  --python PATH                 default: python
  --run_dir PATH                explicit output directory
  --skip_existing               reuse an explicit directory and skip complete logs
  --dry_run                     print commands without executing
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --dataset=*) DATASET="${1#*=}"; shift ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --checkpoint=*) CHECKPOINT="${1#*=}"; shift ;;
    --split) SPLIT="$2"; shift 2 ;;
    --split=*) SPLIT="${1#*=}"; shift ;;
    --n_inputs) N_INPUTS="$2"; shift 2 ;;
    --n_inputs=*) N_INPUTS="${1#*=}"; shift ;;
    --batch_size|--batch) BATCH_SIZE="$2"; shift 2 ;;
    --batch_size=*|--batch=*) BATCH_SIZE="${1#*=}"; shift ;;
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --model_path=*) MODEL_PATH="${1#*=}"; shift ;;
    --k) MAIN_K="$2"; shift 2 ;;
    --k=*) MAIN_K="${1#*=}"; shift ;;
    --variants) VARIANTS_RAW="$2"; shift 2 ;;
    --variants=*) VARIANTS_RAW="${1#*=}"; shift ;;
    --knowledge) KNOWLEDGE_RAW="$2"; shift 2 ;;
    --knowledge=*) KNOWLEDGE_RAW="${1#*=}"; shift ;;
    --defense_seed_mode) DEFENSE_SEED_MODE="$2"; shift 2 ;;
    --defense_seed_mode=*) DEFENSE_SEED_MODE="${1#*=}"; shift ;;
    --defense_seed) DEFENSE_SEED="$2"; shift 2 ;;
    --defense_seed=*) DEFENSE_SEED="${1#*=}"; shift ;;
    --attack_seed) ATTACK_SEED="$2"; shift 2 ;;
    --attack_seed=*) ATTACK_SEED="${1#*=}"; shift ;;
    --ratio_grid) RATIO_GRID="$2"; shift 2 ;;
    --ratio_grid=*) RATIO_GRID="${1#*=}"; shift ;;
    --seed_samples) SEED_SAMPLES_RAW="$2"; shift 2 ;;
    --seed_samples=*) SEED_SAMPLES_RAW="${1#*=}"; shift ;;
    --reduce) REDUCE_RAW="$2"; shift 2 ;;
    --reduce=*) REDUCE_RAW="${1#*=}"; shift ;;
    --seeds) SEEDS_RAW="$2"; shift 2 ;;
    --seeds=*) SEEDS_RAW="${1#*=}"; shift ;;
    --mode) MODE="$2"; shift 2 ;;
    --mode=*) MODE="${1#*=}"; shift ;;
    --candidate_multiplier) CANDIDATE_MULTIPLIER="$2"; shift 2 ;;
    --candidate_multiplier=*) CANDIDATE_MULTIPLIER="${1#*=}"; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --device=*) DEVICE="${1#*=}"; shift ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --python=*) PYTHON_BIN="${1#*=}"; shift ;;
    --run_dir) RUN_DIR="$2"; shift 2 ;;
    --run_dir=*) RUN_DIR="${1#*=}"; shift ;;
    --skip_existing) SKIP_EXISTING=1; shift ;;
    --dry_run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[adaptive-matrix] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$CHECKPOINT" ]; then
  echo "[adaptive-matrix] --checkpoint is required." >&2
  exit 2
fi
PYTHON_AVAILABLE=0
if command -v "$PYTHON_BIN" >/dev/null 2>&1 || [ -x "$PYTHON_BIN" ]; then
  PYTHON_AVAILABLE=1
elif [ "$DRY_RUN" -ne 1 ]; then
  echo "[adaptive-matrix] Python executable is unavailable: $PYTHON_BIN" >&2
  exit 2
fi
if [ ! -e "$CHECKPOINT" ]; then
  echo "[adaptive-matrix] checkpoint does not exist: $CHECKPOINT" >&2
  exit 2
fi
case "$SPLIT" in val|test|official_validation) ;; *) echo "[adaptive-matrix] invalid split: $SPLIT" >&2; exit 2 ;; esac
case "$DEFENSE_SEED_MODE" in static|per_update) ;; *) echo "[adaptive-matrix] invalid seed mode: $DEFENSE_SEED_MODE" >&2; exit 2 ;; esac
case "$MODE" in smoke|formal) ;; *) echo "[adaptive-matrix] invalid mode: $MODE" >&2; exit 2 ;; esac

IFS=',' read -r -a VARIANTS <<<"$VARIANTS_RAW"
IFS=',' read -r -a KNOWLEDGE <<<"$KNOWLEDGE_RAW"
IFS=',' read -r -a SEED_SAMPLES <<<"$SEED_SAMPLES_RAW"
IFS=',' read -r -a REDUCERS <<<"$REDUCE_RAW"
IFS=',' read -r -a SEEDS <<<"$SEEDS_RAW"

if [ "$MODE" = "smoke" ]; then
  N_INPUTS=2
  SEEDS=( "${SEEDS[0]}" )
fi

for variant in "${VARIANTS[@]}"; do
  case "$variant" in
    identity_lrb|sign_only|clip_only|proj_only|proj_clip|full_lrb|pool_full|rule_only|empirical_only|uniform_all_sensitive|proj_rule_only|proj_empirical_only|proj_uniform|proj_uniform_pool|proj_uniform_nearest|proj_uniform_stride|signed_bottleneck|proj_no_empirical) ;;
    *) echo "[adaptive-matrix] unsupported variant: $variant" >&2; exit 2 ;;
  esac
done
for knowledge in "${KNOWLEDGE[@]}"; do
  case "$knowledge" in oracle|ratio_hidden|signs_hidden|method_only) ;; *) echo "[adaptive-matrix] invalid knowledge: $knowledge" >&2; exit 2 ;; esac
  if { [ "$knowledge" = "signs_hidden" ] || [ "$knowledge" = "method_only" ]; } && [ -z "$ATTACK_SEED" ]; then
    echo "[adaptive-matrix] --attack_seed is required for $knowledge." >&2
    exit 2
  fi
done
for reducer in "${REDUCERS[@]}"; do
  case "$reducer" in min|mean) ;; *) echo "[adaptive-matrix] invalid reducer: $reducer" >&2; exit 2 ;; esac
done

slugify() { printf '%s' "$1" | tr -c 'a-zA-Z0-9._-' '_' | sed 's/__*/_/g' | sed 's/^_\|_$//g'; }

if [ -z "$RUN_DIR" ]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  RUN_DIR="log/runs/adaptive_lrb_matrix_$(slugify "$DATASET")_${SPLIT}_${stamp}"
elif [ -e "$RUN_DIR" ] && [ "$SKIP_EXISTING" -ne 1 ]; then
  echo "[adaptive-matrix] explicit run directory already exists; pass --skip_existing to reuse it: $RUN_DIR" >&2
  exit 2
elif [ -e "$RUN_DIR" ]; then
  REUSING_RUN_DIR=1
fi
if [ -e "$RUN_DIR" ] && [ "$REUSING_RUN_DIR" -ne 1 ]; then
  echo "[adaptive-matrix] generated run directory already exists; refusing to overwrite: $RUN_DIR" >&2
  exit 2
fi
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "$LOG_DIR"

session_stamp="$(date +%Y%m%d_%H%M%S)_$$"
MANIFEST_PATH="${RUN_DIR}/run_manifest.txt"
CHECKPOINT_HASH_PATH="${RUN_DIR}/checkpoint_sha256.txt"
EXIT_CODES_PATH="${RUN_DIR}/exit_codes.csv"
if [ "$REUSING_RUN_DIR" -eq 1 ]; then
  MANIFEST_PATH="${RUN_DIR}/run_manifest_resume_${session_stamp}.txt"
  CHECKPOINT_HASH_PATH="${RUN_DIR}/checkpoint_sha256_resume_${session_stamp}.txt"
  EXIT_CODES_PATH="${RUN_DIR}/exit_codes_resume_${session_stamp}.csv"
fi

{
  echo "created_at=$(date -Iseconds)"
  echo "dataset=$DATASET"
  echo "split=$SPLIT"
  echo "n_inputs=$N_INPUTS"
  echo "batch_size=$BATCH_SIZE"
  echo "model_path=$MODEL_PATH"
  echo "checkpoint=$CHECKPOINT"
  echo "k=$MAIN_K"
  echo "variants=$VARIANTS_RAW"
  echo "knowledge=$KNOWLEDGE_RAW"
  echo "defense_seed_mode=$DEFENSE_SEED_MODE"
  echo "defense_seed=${DEFENSE_SEED:-rng_seed}"
  echo "attack_seed=${ATTACK_SEED:-n/a}"
  echo "ratio_grid=$RATIO_GRID"
  echo "seed_samples=$SEED_SAMPLES_RAW"
  echo "reducers=$REDUCE_RAW"
  echo "seeds=$SEEDS_RAW"
  echo "mode=$MODE"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || printf n/a)"
  echo "git_dirty=$(if git diff --quiet --ignore-submodules HEAD 2>/dev/null; then printf false; else printf true; fi)"
  if [ "$PYTHON_AVAILABLE" -eq 1 ]; then
    echo "python=$($PYTHON_BIN --version 2>&1 || printf unavailable)"
    echo "torch=$($PYTHON_BIN -c 'import torch; print(torch.__version__)' 2>/dev/null || printf unavailable)"
    echo "transformers=$($PYTHON_BIN -c 'import transformers; print(transformers.__version__)' 2>/dev/null || printf unavailable)"
    echo "datasets=$($PYTHON_BIN -c 'import datasets; print(datasets.__version__)' 2>/dev/null || printf unavailable)"
  else
    echo "python=unavailable"
    echo "torch=unavailable"
    echo "transformers=unavailable"
    echo "datasets=unavailable"
  fi
  echo "gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | paste -sd ';' - || printf unavailable)"
} >"$MANIFEST_PATH"

if [ -d "$CHECKPOINT" ]; then
  find "$CHECKPOINT" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum >"$CHECKPOINT_HASH_PATH"
else
  sha256sum "$CHECKPOINT" >"$CHECKPOINT_HASH_PATH"
fi
printf 'label,exit_code\n' >"$EXIT_CODES_PATH"

run_one() {
  local variant="$1" knowledge="$2" sample_count="$3" reducer="$4" seed="$5"
  local label
  label="$(slugify "${variant}__${knowledge}__${DEFENSE_SEED_MODE}__m${sample_count}__${reducer}__seed${seed}")"
  local logfile="${LOG_DIR}/${label}.txt"
  if [ -e "$logfile" ]; then
    if [ "$SKIP_EXISTING" -eq 1 ] && grep -q '^result_status=ok$' "$logfile"; then
      echo "[adaptive-matrix] skip complete: $label" >&2
      return 0
    fi
    echo "[adaptive-matrix] refusing to overwrite existing incomplete log: $logfile" >&2
    return 2
  fi

  local command=(
    "$PYTHON_BIN" attack.py
    --dataset "$DATASET" --split "$SPLIT" --n_inputs "$N_INPUTS" --batch_size "$BATCH_SIZE"
    --l1_filter all --l2_filter non-overlap
    --model_path "$MODEL_PATH" --finetuned_path "$CHECKPOINT"
    --device "$DEVICE" --task seq_class --algo sgd
    --defense lrb --defense_lrb_preset "$variant"
    --defense_lrb_keep_ratio_sensitive "$MAIN_K"
    --defense_lrb_seed_mode "$DEFENSE_SEED_MODE"
    --adaptive_attack defense_aware
    --adaptive_candidate_multiplier "$CANDIDATE_MULTIPLIER"
    --adaptive_lrb_knowledge "$knowledge"
    --adaptive_lrb_ratio_grid "$RATIO_GRID"
    --adaptive_lrb_seed_samples "$sample_count"
    --adaptive_lrb_hypothesis_reduce "$reducer"
    --rng_seed "$seed" --log_file "$logfile"
  )
  if [ -n "$DEFENSE_SEED" ]; then command+=( --defense_lrb_seed "$DEFENSE_SEED" ); fi
  if [ -n "$ATTACK_SEED" ]; then command+=( --adaptive_lrb_attack_seed "$ATTACK_SEED" ); fi

  echo "[adaptive-matrix] run: $label" >&2
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    return 0
  fi
  set +e
  "${command[@]}"
  local rc=$?
  set -e
  printf '%s,%s\n' "$label" "$rc" >>"$EXIT_CODES_PATH"
  if [ "$rc" -ne 0 ]; then
    RUN_FAILURES=$((RUN_FAILURES + 1))
  fi
  return 0
}

for variant in "${VARIANTS[@]}"; do
  for knowledge in "${KNOWLEDGE[@]}"; do
    for reducer in "${REDUCERS[@]}"; do
      for sample_count in "${SEED_SAMPLES[@]}"; do
        if { [ "$knowledge" = "oracle" ] || [ "$knowledge" = "ratio_hidden" ]; } && [ "$sample_count" != "${SEED_SAMPLES[0]}" ]; then
          continue
        fi
        if [ "$knowledge" = "oracle" ] && { [ "$reducer" != "${REDUCERS[0]}" ] || [ "$sample_count" != "${SEED_SAMPLES[0]}" ]; }; then
          continue
        fi
        for seed in "${SEEDS[@]}"; do
          run_one "$variant" "$knowledge" "$sample_count" "$reducer" "$seed"
        done
      done
    done
  done
done

if [ "$DRY_RUN" -eq 0 ]; then
  mapfile -t LOG_FILES < <(find "$LOG_DIR" -maxdepth 1 -type f -name '*.txt' | sort)
  if [ "${#LOG_FILES[@]}" -gt 0 ]; then
    "$PYTHON_BIN" scripts/collect_experiment_logs.py "${LOG_FILES[@]}" \
      -o "${RUN_DIR}/results.csv" --markdown "${RUN_DIR}/results.md"
  fi
fi

echo "[adaptive-matrix] done: $RUN_DIR" >&2
if [ "$RUN_FAILURES" -ne 0 ]; then
  echo "[adaptive-matrix] failed experiments: $RUN_FAILURES" >&2
  exit 1
fi
