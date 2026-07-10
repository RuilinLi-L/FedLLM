# PTG Partial-Gradient Runbook

## 1. 范围和前置条件

本手册运行 `scripts/ptg_baselines.sh`（入口为 `attack_partial_gradient.py`）的 PTG / LAMP-lite gradient-matching 攻击。正式设置固定为 `SST-2 + GPT-2 + batch_size=1`；`first2` 是主 exposure，`qkv_only` 是在主矩阵选定 Projection-LRB operating point 后的 transfer extension。

PTG 仅是 partial-gradient 场景的补充性 LAMP-lite 验证，不替代 full-gradient DAGER 的主要证据。命令在项目 Linux `dager` 环境运行，要求 `python`、`torch`、本地 SST-2 cache、本地 GPT-2 checkpoint 和严格 `datasets` ROUGE metric 可用。PowerShell 用户通过本机 Bash/Linux 环境运行。

## 2. 公共配置和指标契约

从项目根目录执行：

```bash
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET=sst2
BATCH=1
MODEL=gpt2
CKPT=./models/gpt2_sst2_clean_num_epochs_2/final
PTG_STEPS=80
PTG_LR=0.1
PTG_EMBED_NORM_WEIGHT=0.01
PTG_PARITY_MODE=fedllm
# 默认仅跑 seed 101；正式三 seed 时，在执行本段前设置 PTG_SEEDS="101 202 303"。
PTG_SEEDS="${PTG_SEEDS:-101}"
read -r -a SEEDS <<< "$PTG_SEEDS"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
export FEDLLM_LOG_DIR="log/runs/ptg_gpt2_${RUN_DATE}"

PTG_COMMON=(
  --finetuned_path "$CKPT"
  --ptg_steps "$PTG_STEPS"
  --ptg_lr "$PTG_LR"
  --ptg_embed_norm_weight "$PTG_EMBED_NORM_WEIGHT"
  --ptg_parity_mode "$PTG_PARITY_MODE"
  --rouge_backend datasets
  --python "$PYTHON_BIN"
)
```

所有调用默认仅运行 seed `101`；需要多 seed 时，在设置块之前执行 `export PTG_SEEDS="101 202 303"`。所有正式调用必须通过 `PTG_COMMON` 传递 `--rouge_backend datasets`，并显式传入一个 `--exposure` 与一个 `--baseline_defense`，避免脚本默认展开未计划的 variants。`simple_ngram` 仅限显式诊断（临时改为 `--rouge_backend simple_ngram` 的 smoke），不得进入正式表、跨 seed 聚合，也不能与 `datasets` ROUGE 比较。

GPT-2 runs use `PTG_PARITY_MODE=fedllm`; do not use `source` mode. Strict source parity is BERT-only because GPT-2 packs Q/K/V in `c_attn`.

每个成功的 raw variant log 必须同时有：

- `result_status=ok`。
- `ptg_parity_mode=fedllm`。
- `ptg_exposure_scope=partial`。
- `n_inputs_completed=n_inputs_requested`。
- `selected_gradient_count > 0`。
- `ptg_final_loss < ptg_initial_loss`。
- `ptg_rouge_backend_requested=datasets`。
- `rouge_backend=datasets`。

任一条件不满足时，该 run 只能作为实现状态、pilot 或负结果，不能进入正式 PTG 表。

## 3. Smoke：clean `first2`，`n_inputs=10`

主矩阵之前，先用 seed 101 运行 clean smoke：

```bash
python test_partial_transformer_gradients_semantics.py

./scripts/ptg_baselines.sh "$DATASET" "$BATCH" "$MODEL" 10 \
  "${PTG_COMMON[@]}" \
  --exposure first2 \
  --baseline_defense none \
  --rng_seed 101
```

检查 raw variant `*.txt`，不要检查 `summary.txt`：后者嵌入 variant 输出，会造成重复解析。只有满足第 2 节验收条件后，才进入正式实验。

## 4. `first2` 正式矩阵

设置 `N_INPUTS=100`。命令迭代当前 `SEEDS`，默认只有 `101`；要生成正式三 seed 矩阵，先设置 `PTG_SEEDS="101 202 303"`：

| defense | 参数 | 默认 runs（seed 101） | 三 seed 正式 runs |
|---|---:|---:|---:|
| `none` | `n/a` | 1 | 3 |
| `proj_only` | `k=0.5/0.65/0.75/0.9` | 4 | 12 |
| `topk` | `0.1` | 1 | 3 |
| `compression` | `8` bits | 1 | 3 |

不要将单个 `proj_only@k=0.9` 与 `none` 的差异当作结论。默认单 seed 用于快速筛查；用于结论时，必须完成全部 keep-ratio sweep 并按三 seed 聚合后选择 projection operating point。

```bash
N_INPUTS=100

# clean anchor
for seed in "${SEEDS[@]}"; do
  ./scripts/ptg_baselines.sh "$DATASET" "$BATCH" "$MODEL" "$N_INPUTS" \
    "${PTG_COMMON[@]}" \
    --exposure first2 \
    --baseline_defense none \
    --rng_seed "$seed"
done

# Projection-LRB / proj_only keep-ratio sweep
for k in 0.5 0.65 0.75 0.9; do
  for seed in "${SEEDS[@]}"; do
    ./scripts/ptg_baselines.sh "$DATASET" "$BATCH" "$MODEL" "$N_INPUTS" \
      "${PTG_COMMON[@]}" \
      --exposure first2 \
      --baseline_defense proj_only \
      --baseline_param "$k" \
      --rng_seed "$seed"
  done
done

# strong empirical baselines
for seed in "${SEEDS[@]}"; do
  ./scripts/ptg_baselines.sh "$DATASET" "$BATCH" "$MODEL" "$N_INPUTS" \
    "${PTG_COMMON[@]}" \
    --exposure first2 \
    --baseline_defense topk \
    --baseline_param 0.1 \
    --rng_seed "$seed"

  ./scripts/ptg_baselines.sh "$DATASET" "$BATCH" "$MODEL" "$N_INPUTS" \
    "${PTG_COMMON[@]}" \
    --exposure first2 \
    --baseline_defense compression \
    --baseline_param 8 \
    --rng_seed "$seed"
done
```

## 5. 选择 Projection-LRB 点并验证 `qkv_only`

完成第 4 节后，依据完整 `first2` sweep 的三 seed 隐私指标、matching loss 和同一 operating point 的 utility 共同选定 `BEST_K`。不得预设为 `0.9`；下方 `0.65` 仅是占位值。

然后以当前 `SEEDS` 重复 clean、选中的 `proj_only`、`topk@0.1` 和 `compression@8`。默认只运行 `101`；用于正式验证时，先设置 `PTG_SEEDS="101 202 303"`：

```bash
BEST_K=0.65  # 替换为 first2 完整聚合后的实际选择

for seed in "${SEEDS[@]}"; do
  ./scripts/ptg_baselines.sh "$DATASET" "$BATCH" "$MODEL" "$N_INPUTS" \
    "${PTG_COMMON[@]}" \
    --exposure qkv_only \
    --baseline_defense none \
    --rng_seed "$seed"

  ./scripts/ptg_baselines.sh "$DATASET" "$BATCH" "$MODEL" "$N_INPUTS" \
    "${PTG_COMMON[@]}" \
    --exposure qkv_only \
    --baseline_defense proj_only \
    --baseline_param "$BEST_K" \
    --rng_seed "$seed"

  ./scripts/ptg_baselines.sh "$DATASET" "$BATCH" "$MODEL" "$N_INPUTS" \
    "${PTG_COMMON[@]}" \
    --exposure qkv_only \
    --baseline_defense topk \
    --baseline_param 0.1 \
    --rng_seed "$seed"

  ./scripts/ptg_baselines.sh "$DATASET" "$BATCH" "$MODEL" "$N_INPUTS" \
    "${PTG_COMMON[@]}" \
    --exposure qkv_only \
    --baseline_defense compression \
    --baseline_param 8 \
    --rng_seed "$seed"
done
```

GPT-2 的 Q/K/V 位于 packed `c_attn` tensor，因此 `query_only`、`key_only`、`value_only` 在该 backbone 上会等价或接近 `qkv_only`，不属于本轮正式矩阵。

## 6. 可选 coverage baselines

`full_lrb@k=0.5` 是强防御/over-defense 对照，`noise`、`dpsgd`、`soteria` 和 `mixup` 只作为 coverage，均不纳入初始主矩阵。完成第 4、5 节后再补充，且仍使用严格 ROUGE 和单一 exposure/defense。`full_lrb` 默认对 `first2` 和 `qkv_only` 各运行 seed `101`；用于正式比较时，设置 `PTG_SEEDS="101 202 303"` 后再运行：

```bash
for seed in "${SEEDS[@]}"; do
  ./scripts/ptg_baselines.sh "$DATASET" "$BATCH" "$MODEL" "$N_INPUTS" \
    "${PTG_COMMON[@]}" \
    --exposure first2 \
    --baseline_defense full_lrb \
    --baseline_param 0.5 \
    --rng_seed "$seed"
done
```

将上面命令的 `--exposure first2` 改为 `--exposure qkv_only`，即可在 `qkv_only` 上运行同一 `full_lrb@k=0.5` 对照。其他可选 coverage baseline 的单 seed smoke 例如：

```bash
./scripts/ptg_baselines.sh "$DATASET" "$BATCH" "$MODEL" "$N_INPUTS" \
  "${PTG_COMMON[@]}" \
  --exposure first2 \
  --baseline_defense noise \
  --baseline_param 5e-4 \
  --rng_seed 101
```

将 `noise`/`5e-4` 分别替换为 `dpsgd`/`5e-4`、`soteria`/`30` 或 `mixup`/`0.3`。是否将这些 baseline 扩展为三 seed 或延伸到 `qkv_only`，在主矩阵稳定后按论文空间决定。

## 7. 收集、聚合和解释

`rec_token_mean` 是 PTG 的主隐私指标，越低表示恢复越弱；保留 `agg_r1fm_r2fm` 作为文本重叠补充。ROUGE 仅在相同 `rouge_backend` 内可比较，禁止把 `datasets` 与 `simple_ngram` 或 `simple_ngram_fallback` 放入同一结论。

默认单 seed 结果只用于筛查或 pilot。正式表将 `PTG_SEEDS` 设为 `101 202 303`，按三个 seeds 报告 `mean +/- std`，并同时保留 `rec_token_mean`、`agg_r1fm_r2fm`、`ptg_initial_loss` 和 `ptg_final_loss`，以区分恢复变弱与 gradient matching 未收敛。

收集器的目录输入不递归，且每个 PTG `summary.txt` 嵌入 raw variant 输出。因此必须只传递 raw `*.txt`，排除 `summary.txt`：

```bash
mapfile -t VARIANT_LOGS < <(
  find "$FEDLLM_LOG_DIR" -type f -name '*.txt' ! -name 'summary.txt' -print | sort
)

if [ "${#VARIANT_LOGS[@]}" -eq 0 ]; then
  echo "No raw PTG variant logs found under $FEDLLM_LOG_DIR" >&2
  exit 1
fi

"$PYTHON_BIN" scripts/collect_experiment_logs.py "${VARIANT_LOGS[@]}" \
  --output "$FEDLLM_LOG_DIR/ptg_raw.csv" \
  --markdown "$FEDLLM_LOG_DIR/ptg_raw.md" \
  --tradeoff-output "$FEDLLM_LOG_DIR/ptg_tradeoff.csv" \
  --tradeoff-markdown "$FEDLLM_LOG_DIR/ptg_tradeoff.md"
```

`ptg_tradeoff.*` 只有在同一次收集同时传入可匹配的 utility logs 时才会得到可解释的 join；PTG `batch_size=1` 不会自动匹配 batch 不同的 utility 结果。先用 `ptg_raw.*` 报告 PTG 隐私，再以同一 defense operating point 和兼容元数据补充 utility logs。

PTG 结果始终是 supplementary evidence，不替代 full-gradient DAGER 的主要证据。
