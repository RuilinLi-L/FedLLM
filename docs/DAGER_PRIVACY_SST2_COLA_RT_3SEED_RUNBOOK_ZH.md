# SST2/CoLA/Rotten Tomatoes 三种子 DAGER 隐私实验 Runbook

本文档用于在 Linux 上重跑统一口径的 full-gradient DAGER 隐私实验，并固定三件事：

- 三个数据集：`sst2`、`cola`、`rotten_tomatoes`
- 三个随机种子：`101 202 303`
- GPU 执行：`CUDA_VISIBLE_DEVICES=0` 且 wrapper 内部调用 `attack.py --device cuda`

本 runbook 是 privacy-only；utility 后续只对 Pareto 候选点补跑。

## 实验设置

| 项 | 设置 |
| --- | --- |
| 模型 | `gpt2` |
| threat surface | full gradients，不含 PEFT / partial-gradient |
| batch size | `2` |
| attack budget | `n_inputs=100` |
| split | `val` |
| task | `seq_class` |
| seeds | `101 202 303` |
| device | GPU，`--device cuda` |

Checkpoint 路径：

| 数据集 | checkpoint |
| --- | --- |
| `sst2` | `./models/gpt2_sst2_clean_num_epochs_2/final` |
| `cola` | `./models/gpt2_cola_clean_num_epochs_2/final` |
| `rotten_tomatoes` | `./models/gpt2_rotten_tomatoes_clean_num_epochs_2/final` |

## Baselines 和 Sweep

| 角色 | baseline | 参数 |
| --- | --- | --- |
| 泄露锚点 | `none` | 单点 |
| 主方法 | `lrbprojonly` / Projection-LRB | 内置 `0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.96,0.97,0.98,0.99`，额外补 `0.75,0.995,1.0` |
| 过防御参考 | `lrb` / full LRB | 同 Projection-LRB |
| 强 baseline | `topk` | 内置 `0.01,0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.97,0.98,0.99`，额外补 `0.03,0.15,0.25,0.35` |
| 强 baseline | `compression` | 内置 `2,4,8,12,16,17,18,19,20,21,22,23,24,25,26,27,28,30,32,36,40,44,48,56,64`，额外补 `6,10,14` |
| 噪声覆盖 | `noise` | `1e-6` 到 `1e-2` 的内置 dense sweep |
| DP-SGD-style | `dpsgd` | 同 `noise`，clip norm 默认 `1.0` |
| Opacus DP-SGD | `dpsgd_opacus` | 主点 `0.01`，额外补 `1e-4,3e-4,5e-4,1e-3,3e-3,1e-2` |
| appendix 覆盖 | `mixup`, `soteria` | 内置 dense sweep |

`dpsgd_opacus` 会报告 Opacus/RDP 相关字段。除非最终完整核验 accountant 与训练设置，否则论文里不要把它写成正式 DP 保证。

## 推荐执行方式

直接运行脚本：

```bash
cd /data/lrl/FedLLM
conda activate dager

bash scripts/run_dager_privacy_sst2_cola_rt_3seed.sh
```

默认行为：

- 固定 `DAGER_SEEDS="101 202 303"`。
- 固定 `CUDA_VISIBLE_DEVICES=0`。
- 创建日志根目录 `log/runs/dager_privacy_sst2_cola_rt_3seed_<timestamp>`。
- 依次运行 smoke、主 sweep、额外边界点、adaptive check、最终汇总。

可选参数：

```bash
bash scripts/run_dager_privacy_sst2_cola_rt_3seed.sh --gpu 1
bash scripts/run_dager_privacy_sst2_cola_rt_3seed.sh --dry-run
bash scripts/run_dager_privacy_sst2_cola_rt_3seed.sh --log-dir log/runs/my_run
bash scripts/run_dager_privacy_sst2_cola_rt_3seed.sh --skip-adaptive
```

`--dry-run` 只打印命令，不真正执行，适合先检查调度范围。

## 等价手动命令

如果不使用 runner，可以手动执行下面命令。

```bash
cd /data/lrl/FedLLM
conda activate dager

set -euo pipefail

# 三种子：不要在后续命令里再传 --rng_seed。
export DAGER_SEEDS="101 202 303"

# GPU：defense_baselines.sh 内部固定传 attack.py --device cuda。
export CUDA_VISIBLE_DEVICES=0

python3 -c "import torch; print('cuda_available=', torch.cuda.is_available())"
python3 -c "import opacus; print('opacus ok')" || echo "opacus unavailable; dpsgd_opacus rows may fail and be logged"

STAMP="$(date +%Y%m%d_%H%M%S)"
export DAGER_LOG_DIR="log/runs/dager_privacy_sst2_cola_rt_3seed_${STAMP}"
mkdir -p "$DAGER_LOG_DIR"

declare -A CKPT=(
  [sst2]="./models/gpt2_sst2_clean_num_epochs_2/final"
  [cola]="./models/gpt2_cola_clean_num_epochs_2/final"
  [rotten_tomatoes]="./models/gpt2_rotten_tomatoes_clean_num_epochs_2/final"
)

DATASETS=(sst2 cola rotten_tomatoes)
```

Smoke test：

```bash
for ds in "${DATASETS[@]}"; do
  bash scripts/defense_baselines.sh "$ds" 2 gpt2 2 \
    --baseline_defense none \
    --finetuned_path "${CKPT[$ds]}"
done
```

主 privacy sweep：

```bash
for ds in "${DATASETS[@]}"; do
  for defense in lrbprojonly lrb topk compression noise dpsgd dpsgd_opacus mixup soteria; do
    bash scripts/defense_baselines.sh "$ds" 2 gpt2 100 \
      --baseline_defense "$defense" \
      --finetuned_path "${CKPT[$ds]}"
  done
done
```

额外边界点：

```bash
declare -A EXTRA=(
  [lrbprojonly]="0.75 0.995 1.0"
  [lrb]="0.75 0.995 1.0"
  [topk]="0.03 0.15 0.25 0.35"
  [compression]="6 10 14"
  [dpsgd_opacus]="1e-4 3e-4 5e-4 1e-3 3e-3 1e-2"
)

for ds in "${DATASETS[@]}"; do
  for defense in "${!EXTRA[@]}"; do
    for param in ${EXTRA[$defense]}; do
      bash scripts/defense_baselines.sh "$ds" 2 gpt2 100 \
        --baseline_defense "$defense" \
        --baseline_param "$param" \
        --finetuned_path "${CKPT[$ds]}"
    done
  done
done
```

Defense-aware adaptive check：

```bash
declare -A ADAPTIVE=(
  [lrbprojonly]="0.65 0.90 0.95 0.99"
  [topk]="0.10 0.30 0.50"
  [compression]="8 16 24 32"
  [dpsgd_opacus]="5e-4 1e-3 1e-2"
)

for ds in "${DATASETS[@]}"; do
  for defense in "${!ADAPTIVE[@]}"; do
    for param in ${ADAPTIVE[$defense]}; do
      bash scripts/defense_baselines.sh "$ds" 2 gpt2 100 \
        --baseline_defense "$defense" \
        --baseline_param "$param" \
        --adaptive_attack_check \
        --finetuned_path "${CKPT[$ds]}"
    done
  done
done
```

汇总结果：

```bash
python3 scripts/collect_experiment_logs.py "$DAGER_LOG_DIR" \
  -o "$DAGER_LOG_DIR/all_results.csv" \
  --markdown "$DAGER_LOG_DIR/all_results.md"
```

## 验收标准

- 每个主表候选 row 必须有 `seed=101/202/303` 三条完成记录。
- 每条完成记录要求 `n_inputs_completed=100`。
- 主指标：`rec_token_mean`、`rec_maxb_token_mean`、`agg_rouge1_fm`、`agg_rouge2_fm`、`agg_r1fm_r2fm`。
- `compression@2` 或其他低 bit 点如果失败，只作为失败记录保留，不算稳定零恢复点。
- `dpsgd_opacus` 若因缺少 `opacus` 或 Opacus batch 问题失败，先不改代码，记录为环境/实现失败点；主表仍以 `dpsgd` 作为 DP-SGD-style coverage，`dpsgd_opacus` 放 appendix。

## 明确确认

- 是三种子：runner 会设置 `export DAGER_SEEDS="101 202 303"`，并且不会给 wrapper 传 `--rng_seed`。
- 是 GPU：`scripts/defense_baselines.sh` 内部调用 `attack.py --device cuda`，runner 额外用 `CUDA_VISIBLE_DEVICES=<gpu>` 固定 GPU。
- 不要把 `DAGER=0` 写成形式化隐私保证；它只表示当前 DAGER budget 和实现下未恢复。
