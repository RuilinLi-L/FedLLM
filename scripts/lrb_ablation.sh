#!/usr/bin/env bash
# Run publication-grade LRB ablations.
#
# The script evaluates each ablation variant at three levels:
#   1) DAGER attack-time privacy
#   2) one-step proxy utility
#   3) end-to-end training utility
#
# Example:
#   bash scripts/lrb_ablation.sh --lrb_main_k 0.5 --n_inputs 100 --mode all --skip_existing
#
# Quick smoke test:
#   bash scripts/lrb_ablation.sh --n_inputs 5 --mode privacy --variants none,identity_lrb,full_lrb

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAGER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$DAGER_ROOT" || exit 1

DATASET="sst2"
BATCH_SIZE="2"
MODEL_PATH="gpt2"
FINETUNED_PATH="./models/gpt2-ft-rt"
TOKENIZER_PATH=""
CACHE_DIR="./models_cache"
N_INPUTS="100"
EPOCHS="1"
LRB_MAIN_K="0.5"
RUN_DIR=""
SEEDS_RAW="101 202 303"
MODE="all"
VARIANTS_RAW="all"
DEVICE="cuda"
SKIP_EXISTING=0
DRY_RUN=0
NO_COLLECT=0
FAIL_FAST=0

ALL_VARIANTS=(
  none
  identity_lrb
  clip_only
  proj_only
  proj_clip
  full_lrb
  pool_full
  rule_only
  empirical_only
  uniform_all_sensitive
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/lrb_ablation.sh [options]

Main options:
  --dataset NAME              Dataset, default: sst2
  --batch_size N              Batch size, default: 2
  --model_path PATH           Backbone/model id for attack.py, default: gpt2
  --finetuned_path PATH       Fine-tuned checkpoint / utility anchor, default: ./models/gpt2-ft-rt
  --tokenizer_path PATH       Optional tokenizer path for proxy/train when needed
  --cache_dir PATH            Hugging Face cache for attack.py, default: ./models_cache
  --n_inputs N                DAGER inputs per seed, default: 100
  --epochs N                  End-to-end utility epochs, default: 1
  --lrb_main_k FLOAT          Main LRB sensitive keep ratio, default: 0.5
  --run_dir PATH              Output directory. If unset, an auto timestamped dir is used.
  --seeds "101 202 303"      Space-separated seeds, default: "101 202 303"

Execution control:
  --mode all|privacy|proxy|train|privacy,proxy
                              Which stages to run, default: all
  --variants LIST             Comma-separated variants, or all. Default: all
                              Variants: none,identity_lrb,clip_only,proj_only,proj_clip,
                                        full_lrb,pool_full,rule_only,empirical_only,
                                        uniform_all_sensitive
  --skip_existing             Skip a log if it already contains a result summary
  --dry_run                   Print commands without running
  --no_collect                Do not collect/summarize logs at the end
  --fail_fast                 Stop at the first failed command
  --device DEVICE             Device for Python stages, default: cuda. Bare cuda auto-selects an idle visible GPU.

Recommended formal run:
  tmux new -s lrb_ablation
  conda activate dager
  bash scripts/lrb_ablation.sh --lrb_main_k 0.5 --n_inputs 100 --mode all --skip_existing
EOF
}

slugify() {
  printf '%s' "$1" | tr -c 'a-zA-Z0-9._-' '_' | sed 's/__*/_/g' | sed 's/^_\|_$//g'
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset)
      DATASET="$2"; shift 2 ;;
    --dataset=*)
      DATASET="${1#*=}"; shift ;;
    --batch_size|--batch)
      BATCH_SIZE="$2"; shift 2 ;;
    --batch_size=*|--batch=*)
      BATCH_SIZE="${1#*=}"; shift ;;
    --model_path|--model)
      MODEL_PATH="$2"; shift 2 ;;
    --model_path=*|--model=*)
      MODEL_PATH="${1#*=}"; shift ;;
    --finetuned_path)
      FINETUNED_PATH="$2"; shift 2 ;;
    --finetuned_path=*)
      FINETUNED_PATH="${1#*=}"; shift ;;
    --tokenizer_path)
      TOKENIZER_PATH="$2"; shift 2 ;;
    --tokenizer_path=*)
      TOKENIZER_PATH="${1#*=}"; shift ;;
    --cache_dir|--models_cache)
      CACHE_DIR="$2"; shift 2 ;;
    --cache_dir=*|--models_cache=*)
      CACHE_DIR="${1#*=}"; shift ;;
    --n_inputs)
      N_INPUTS="$2"; shift 2 ;;
    --n_inputs=*)
      N_INPUTS="${1#*=}"; shift ;;
    --epochs|--num_epochs)
      EPOCHS="$2"; shift 2 ;;
    --epochs=*|--num_epochs=*)
      EPOCHS="${1#*=}"; shift ;;
    --lrb_main_k|--main_k)
      LRB_MAIN_K="$2"; shift 2 ;;
    --lrb_main_k=*|--main_k=*)
      LRB_MAIN_K="${1#*=}"; shift ;;
    --run_dir)
      RUN_DIR="$2"; shift 2 ;;
    --run_dir=*)
      RUN_DIR="${1#*=}"; shift ;;
    --seeds)
      SEEDS_RAW="$2"; shift 2 ;;
    --seeds=*)
      SEEDS_RAW="${1#*=}"; shift ;;
    --mode)
      MODE="$2"; shift 2 ;;
    --mode=*)
      MODE="${1#*=}"; shift ;;
    --variants)
      VARIANTS_RAW="$2"; shift 2 ;;
    --variants=*)
      VARIANTS_RAW="${1#*=}"; shift ;;
    --device)
      DEVICE="$2"; shift 2 ;;
    --device=*)
      DEVICE="${1#*=}"; shift ;;
    --skip_existing)
      SKIP_EXISTING=1; shift ;;
    --dry_run)
      DRY_RUN=1; shift ;;
    --no_collect)
      NO_COLLECT=1; shift ;;
    --fail_fast)
      FAIL_FAST=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[lrb-ablation] unknown option: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if [ -z "$RUN_DIR" ]; then
  safe_ds="$(slugify "$DATASET")"
  safe_model="$(slugify "$(basename "$MODEL_PATH")")"
  stamp="$(date +%Y%m%d_%H%M%S)"
  RUN_DIR="log/runs/lrb_ablation_${safe_ds}_b${BATCH_SIZE}_${safe_model}_k${LRB_MAIN_K}_${stamp}"
fi

mkdir -p "$RUN_DIR"/{privacy,proxy,train,models}
EXIT_CODES="${RUN_DIR}/exit_codes.csv"
printf 'stage,variant,seed,exit_code,log_file\n' > "$EXIT_CODES"

read -r -a SEEDS <<< "$SEEDS_RAW"

if [ "$VARIANTS_RAW" = "all" ]; then
  SELECTED_VARIANTS=( "${ALL_VARIANTS[@]}" )
else
  IFS=',' read -r -a SELECTED_VARIANTS <<< "$VARIANTS_RAW"
fi

is_known_variant() {
  local candidate="$1"
  local variant
  for variant in "${ALL_VARIANTS[@]}"; do
    if [ "$variant" = "$candidate" ]; then
      return 0
    fi
  done
  return 1
}

for variant in "${SELECTED_VARIANTS[@]}"; do
  if ! is_known_variant "$variant"; then
    echo "[lrb-ablation] unknown variant: ${variant}" >&2
    exit 2
  fi
done

mode_has() {
  local stage="$1"
  if [ "$MODE" = "all" ]; then
    return 0
  fi
  case ",${MODE}," in
    *",${stage},"*) return 0 ;;
    *) return 1 ;;
  esac
}

summary_exists() {
  local file="$1"
  [ -f "$file" ] || return 1
  grep -q 'SUMMARY END' "$file" 2>/dev/null
}

variant_args() {
  local variant="$1"
  DEFENSE="lrb"
  DEF_EXTRA=()

  case "$variant" in
    none)
      DEFENSE="none"
      DEF_EXTRA=()
      ;;
    identity_lrb)
      DEF_EXTRA=(
        --defense_lrb_keep_ratio_sensitive 1.0
        --defense_lrb_keep_ratio_other 1.0
        --defense_lrb_clip_scale_sensitive 1000000
        --defense_lrb_clip_scale_other 1000000
        --defense_lrb_noise_sensitive 0
        --defense_lrb_noise_other 0
        --defense_lrb_empirical_weight 0
        --defense_lrb_projection signed_pool
      )
      ;;
    clip_only)
      DEF_EXTRA=(
        --defense_lrb_keep_ratio_sensitive 1.0
        --defense_lrb_keep_ratio_other 1.0
        --defense_lrb_clip_scale_sensitive 0.5
        --defense_lrb_clip_scale_other 1.0
        --defense_lrb_noise_sensitive 0
        --defense_lrb_noise_other 0
        --defense_lrb_empirical_weight 0.6
        --defense_lrb_projection signed_pool
      )
      ;;
    proj_only)
      DEF_EXTRA=(
        --defense_lrb_keep_ratio_sensitive "$LRB_MAIN_K"
        --defense_lrb_keep_ratio_other 0.75
        --defense_lrb_clip_scale_sensitive 1000000
        --defense_lrb_clip_scale_other 1000000
        --defense_lrb_noise_sensitive 0
        --defense_lrb_noise_other 0
        --defense_lrb_empirical_weight 0.6
        --defense_lrb_projection signed_pool
      )
      ;;
    proj_clip)
      DEF_EXTRA=(
        --defense_lrb_keep_ratio_sensitive "$LRB_MAIN_K"
        --defense_lrb_keep_ratio_other 0.75
        --defense_lrb_clip_scale_sensitive 0.5
        --defense_lrb_clip_scale_other 1.0
        --defense_lrb_noise_sensitive 0
        --defense_lrb_noise_other 0
        --defense_lrb_empirical_weight 0.6
        --defense_lrb_projection signed_pool
      )
      ;;
    full_lrb)
      DEF_EXTRA=(
        --defense_lrb_keep_ratio_sensitive "$LRB_MAIN_K"
        --defense_lrb_keep_ratio_other 0.75
        --defense_lrb_clip_scale_sensitive 0.5
        --defense_lrb_clip_scale_other 1.0
        --defense_lrb_noise_sensitive 0.03
        --defense_lrb_noise_other 0.005
        --defense_lrb_empirical_weight 0.6
        --defense_lrb_projection signed_pool
      )
      ;;
    pool_full)
      DEF_EXTRA=(
        --defense_lrb_keep_ratio_sensitive "$LRB_MAIN_K"
        --defense_lrb_keep_ratio_other 0.75
        --defense_lrb_clip_scale_sensitive 0.5
        --defense_lrb_clip_scale_other 1.0
        --defense_lrb_noise_sensitive 0.03
        --defense_lrb_noise_other 0.005
        --defense_lrb_empirical_weight 0.6
        --defense_lrb_projection pool
      )
      ;;
    rule_only)
      DEF_EXTRA=(
        --defense_lrb_keep_ratio_sensitive "$LRB_MAIN_K"
        --defense_lrb_keep_ratio_other 0.75
        --defense_lrb_clip_scale_sensitive 0.5
        --defense_lrb_clip_scale_other 1.0
        --defense_lrb_noise_sensitive 0.03
        --defense_lrb_noise_other 0.005
        --defense_lrb_empirical_weight 0
        --defense_lrb_projection signed_pool
      )
      ;;
    empirical_only)
      DEF_EXTRA=(
        --defense_lrb_keep_ratio_sensitive "$LRB_MAIN_K"
        --defense_lrb_keep_ratio_other 0.75
        --defense_lrb_clip_scale_sensitive 0.5
        --defense_lrb_clip_scale_other 1.0
        --defense_lrb_noise_sensitive 0.03
        --defense_lrb_noise_other 0.005
        --defense_lrb_empirical_weight 1
        --defense_lrb_projection signed_pool
      )
      ;;
    uniform_all_sensitive)
      DEF_EXTRA=(
        --defense_lrb_keep_ratio_sensitive "$LRB_MAIN_K"
        --defense_lrb_keep_ratio_other "$LRB_MAIN_K"
        --defense_lrb_clip_scale_sensitive 0.5
        --defense_lrb_clip_scale_other 0.5
        --defense_lrb_noise_sensitive 0.03
        --defense_lrb_noise_other 0.03
        --defense_lrb_empirical_weight 0
        --defense_lrb_projection signed_pool
      )
      ;;
    *)
      echo "[lrb-ablation] internal error: unsupported variant ${variant}" >&2
      exit 2
      ;;
  esac
}

run_logged() {
  local stage="$1"
  local variant="$2"
  local seed="$3"
  local logfile="$4"
  shift 4
  local cmd=( "$@" )

  if [ "$SKIP_EXISTING" -eq 1 ] && summary_exists "$logfile"; then
    echo "[lrb-ablation] skip existing ${stage}/${variant}/seed${seed}: ${logfile}" >&2
    printf '%s,%s,%s,%s,%s\n' "$stage" "$variant" "$seed" "skipped" "$logfile" >> "$EXIT_CODES"
    return 0
  fi

  echo "[lrb-ablation] running ${stage}/${variant}/seed${seed}" >&2
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '[dry-run]'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    printf '%s,%s,%s,%s,%s\n' "$stage" "$variant" "$seed" "dry_run" "$logfile" >> "$EXIT_CODES"
    return 0
  fi

  set +e
  "${cmd[@]}"
  local rc=$?
  set -e
  printf '%s,%s,%s,%s,%s\n' "$stage" "$variant" "$seed" "$rc" "$logfile" >> "$EXIT_CODES"

  if [ "$rc" -ne 0 ]; then
    echo "[lrb-ablation] command failed (${rc}): ${stage}/${variant}/seed${seed}" >&2
    if [ "$FAIL_FAST" -eq 1 ]; then
      exit "$rc"
    fi
  fi
}

write_manifest() {
  cat > "${RUN_DIR}/manifest.txt" <<EOF
dataset=${DATASET}
batch_size=${BATCH_SIZE}
model_path=${MODEL_PATH}
finetuned_path=${FINETUNED_PATH}
tokenizer_path=${TOKENIZER_PATH}
cache_dir=${CACHE_DIR}
n_inputs=${N_INPUTS}
epochs=${EPOCHS}
lrb_main_k=${LRB_MAIN_K}
seeds=${SEEDS_RAW}
mode=${MODE}
variants=${SELECTED_VARIANTS[*]}
run_dir=${RUN_DIR}
created_at=$(date '+%Y-%m-%d %H:%M:%S')
EOF
}

collect_results() {
  echo "[lrb-ablation] collecting logs under ${RUN_DIR}" >&2
  python3 scripts/collect_experiment_logs.py \
    "${RUN_DIR}/privacy" \
    "${RUN_DIR}/proxy" \
    "${RUN_DIR}/train" \
    -o "${RUN_DIR}/raw_results.csv" \
    --markdown "${RUN_DIR}/raw_results.md"

  python3 - "$RUN_DIR" <<'PY'
from __future__ import annotations

import csv
import math
import re
import statistics
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
raw_csv = run_dir / "raw_results.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def variant_from_path(path: str) -> str:
    stem = Path(path).stem
    return re.sub(r"_seed\d+$", "", stem)


def seed_from_path(path: str) -> str:
    m = re.search(r"_seed(\d+)$", Path(path).stem)
    return m.group(1) if m else ""


def to_float(value: str | None) -> float | None:
    if value in (None, "", "n/a"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def time_to_seconds(value: str | None) -> float | None:
    if value in (None, "", "n/a"):
        return None
    parts = str(value).split(":")
    try:
        nums = [float(part) for part in parts]
    except ValueError:
        return None
    if len(nums) == 3:
        h, m, s = nums
    elif len(nums) == 2:
        h, m, s = 0.0, nums[0], nums[1]
    else:
        return None
    return h * 3600 + m * 60 + s


def fmt_float(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.6f}"


def summarize(rows: list[dict[str, str]], fields: list[str], *, time_fields: list[str] = []) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)

    out: list[dict[str, str]] = []
    for variant, items in sorted(grouped.items()):
        summary: dict[str, str] = {
            "variant": variant,
            "n_runs": str(len(items)),
            "seeds": " ".join(sorted({item.get("seed", "") for item in items if item.get("seed", "")})),
            "failed_runs": str(sum(1 for item in items if item.get("result_status") not in ("", "ok"))),
        }
        for field in fields:
            values = [to_float(item.get(field)) for item in items]
            values = [value for value in values if value is not None]
            if values:
                summary[field] = fmt_float(statistics.mean(values))
                summary[f"{field}_std"] = fmt_float(statistics.stdev(values) if len(values) > 1 else 0.0)
            else:
                summary[field] = ""
                summary[f"{field}_std"] = ""
        for field in time_fields:
            values = [time_to_seconds(item.get(field)) for item in items]
            values = [value for value in values if value is not None]
            if values:
                summary[f"{field}_seconds"] = fmt_float(statistics.mean(values))
                summary[f"{field}_seconds_std"] = fmt_float(statistics.stdev(values) if len(values) > 1 else 0.0)
            else:
                summary[f"{field}_seconds"] = ""
                summary[f"{field}_seconds_std"] = ""
        out.append(summary)
    return out


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_md(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    lines = [
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join("---" for _ in keys) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")).replace("|", "\\|") for key in keys) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


rows = read_rows(raw_csv)
for row in rows:
    row["variant"] = variant_from_path(row.get("log_path", ""))
    row["seed"] = row.get("seed") or seed_from_path(row.get("log_path", ""))

privacy = [row for row in rows if row.get("log_kind") == "attack_dager"]
proxy = [row for row in rows if row.get("log_kind") == "proxy_utility"]
train = [row for row in rows if row.get("log_kind") == "train"]

privacy_summary = summarize(
    privacy,
    ["rec_token_mean", "rec_maxb_token_mean", "agg_rouge1_fm", "agg_rouge2_fm", "agg_r1fm_r2fm"],
    time_fields=["last_total_time"],
)
proxy_summary = summarize(
    proxy,
    ["grad_cosine_mean", "norm_retention_mean", "delta_train_loss_mean", "delta_val_accuracy_mean", "delta_val_macro_f1_mean", "step_runtime_mean"],
)
utility_summary = summarize(
    train,
    ["eval_accuracy", "eval_macro_f1", "eval_loss", "final_train_loss"],
    time_fields=["total_train_time"],
)

privacy_by_variant = {row["variant"]: row for row in privacy_summary}
proxy_by_variant = {row["variant"]: row for row in proxy_summary}
utility_by_variant = {row["variant"]: row for row in utility_summary}
none_acc = to_float(utility_by_variant.get("none", {}).get("eval_accuracy"))

combined: list[dict[str, str]] = []
for variant in sorted(set(privacy_by_variant) | set(proxy_by_variant) | set(utility_by_variant)):
    p = privacy_by_variant.get(variant, {})
    x = proxy_by_variant.get(variant, {})
    u = utility_by_variant.get(variant, {})
    acc = to_float(u.get("eval_accuracy"))
    rec = to_float(p.get("rec_token_mean"))
    row = {
        "variant": variant,
        "privacy_runs": p.get("n_runs", "0"),
        "utility_runs": u.get("n_runs", "0"),
        "rec_token_mean": p.get("rec_token_mean", ""),
        "rec_token_mean_std": p.get("rec_token_mean_std", ""),
        "rouge1_fm": p.get("agg_rouge1_fm", ""),
        "rouge2_fm": p.get("agg_rouge2_fm", ""),
        "r1_plus_r2": p.get("agg_r1fm_r2fm", ""),
        "eval_accuracy": u.get("eval_accuracy", ""),
        "eval_accuracy_std": u.get("eval_accuracy_std", ""),
        "eval_macro_f1": u.get("eval_macro_f1", ""),
        "eval_loss": u.get("eval_loss", ""),
        "utility_drop": fmt_float(none_acc - acc) if none_acc is not None and acc is not None else "",
        "privacy_score": fmt_float(1.0 - rec) if rec is not None else "",
        "grad_cosine_mean": x.get("grad_cosine_mean", ""),
        "norm_retention_mean": x.get("norm_retention_mean", ""),
        "step_runtime_mean": x.get("step_runtime_mean", ""),
        "train_time_seconds": u.get("total_train_time_seconds", ""),
        "attack_time_seconds": p.get("last_total_time_seconds", ""),
    }
    combined.append(row)

for name, table in [
    ("ablation_privacy_summary", privacy_summary),
    ("ablation_proxy_summary", proxy_summary),
    ("ablation_utility_summary", utility_summary),
    ("ablation_combined_summary", combined),
]:
    write_csv(run_dir / f"{name}.csv", table)
    write_md(run_dir / f"{name}.md", table)

print(f"Wrote ablation summaries to {run_dir}")
PY
}

write_manifest

echo "[lrb-ablation] run dir: ${RUN_DIR}" >&2
echo "[lrb-ablation] variants: ${SELECTED_VARIANTS[*]}" >&2
echo "[lrb-ablation] seeds: ${SEEDS[*]}" >&2
echo "[lrb-ablation] mode: ${MODE}" >&2

TOKENIZER_ARGS=()
if [ -n "$TOKENIZER_PATH" ]; then
  TOKENIZER_ARGS=( --tokenizer_path "$TOKENIZER_PATH" )
fi

for variant in "${SELECTED_VARIANTS[@]}"; do
  variant_args "$variant"
  for seed in "${SEEDS[@]}"; do
    if mode_has privacy; then
      privacy_log="${RUN_DIR}/privacy/${variant}_seed${seed}.txt"
      run_logged privacy "$variant" "$seed" "$privacy_log" \
        python3 attack.py \
          --dataset "$DATASET" \
          --split val \
          --n_inputs "$N_INPUTS" \
          --batch_size "$BATCH_SIZE" \
          --l1_filter all \
          --l2_filter non-overlap \
          --model_path "$MODEL_PATH" \
          --device "$DEVICE" \
          --task seq_class \
          --cache_dir "$CACHE_DIR" \
          --finetuned_path "$FINETUNED_PATH" \
          --rng_seed "$seed" \
          --defense "$DEFENSE" \
          "${DEF_EXTRA[@]}" \
          --log_file "$privacy_log"
    fi

    if mode_has proxy; then
      proxy_log="${RUN_DIR}/proxy/${variant}_seed${seed}.txt"
      run_logged proxy "$variant" "$seed" "$proxy_log" \
        python3 scripts/proxy_utility.py \
          --dataset "$DATASET" \
          --task seq_class \
          --batch_size "$BATCH_SIZE" \
          --model_path "$FINETUNED_PATH" \
          "${TOKENIZER_ARGS[@]}" \
          --device "$DEVICE" \
          --models_cache "$CACHE_DIR" \
          --n_train_batches 100 \
          --val_size 256 \
          --eval_batch_size 16 \
          --train_method full \
          --rng_seed "$seed" \
          --defense "$DEFENSE" \
          "${DEF_EXTRA[@]}" \
          --log_file "$proxy_log"
    fi

    if mode_has train; then
      train_log="${RUN_DIR}/train/${variant}_seed${seed}.txt"
      run_logged train "$variant" "$seed" "$train_log" \
        python3 train.py \
          --dataset "$DATASET" \
          --task seq_class \
          --batch_size "$BATCH_SIZE" \
          --num_epochs "$EPOCHS" \
          --save_every 0 \
          --model_path "$FINETUNED_PATH" \
          "${TOKENIZER_ARGS[@]}" \
          --device "$DEVICE" \
          --models_cache "$CACHE_DIR" \
          --train_method full \
          --rng_seed "$seed" \
          --output_dir "${RUN_DIR}/models/${variant}_seed${seed}" \
          --defense "$DEFENSE" \
          "${DEF_EXTRA[@]}" \
          --log_file "$train_log"
    fi
  done
done

if [ "$NO_COLLECT" -eq 0 ] && [ "$DRY_RUN" -eq 0 ]; then
  collect_results
fi

echo "[lrb-ablation] done: ${RUN_DIR}" >&2
