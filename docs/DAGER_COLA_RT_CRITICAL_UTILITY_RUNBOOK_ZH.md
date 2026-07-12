# CoLA / Rotten Tomatoes 关键效用实验 Runbook

## 结论先行

时间有限时，先跑 P0，不要照旧清单铺满所有参数。P0 每个数据集只有四个配置、每个配置三个 seed，共 `2 x 4 x 3 = 24` 次完整训练：

| 数据集 | clean | Projection-LRB | top-k | compression |
| --- | --- | --- | --- | --- |
| CoLA | `none` | `k=0.99` | `ratio=0.45` | `20 bit` |
| Rotten Tomatoes | `none` | `k=0.99` | `ratio=0.55` | `20 bit` |

这四类分别对应 clean utility 锚点、主方法、强稀疏化基线和强量化基线。参数来自 2026-07-09 的 full-gradient DAGER 3-seed 隐私日志，不是沿用 SST2 参数。

## 隐私日志依据

统一设置：GPT-2、batch size `2`、`n_inputs=100`、seeds `101/202/303`、validation split、full-gradient DAGER。

日志目录：

- CoLA：`log/runs/dager_privacy_cola_baselines_3seed_20260709_003349`
- Rotten Tomatoes：`log/runs/dager_privacy_rotten_tomatoes_baselines_3seed_20260709_003608`

关键边界如下。表中的 token recovery 和 R1+R2 均为三 seed 均值；所有行都有三个成功 seed，每个 seed 完成 100 个输入。

| 数据集 | 防御点 | token recovery | R1+R2 | 判断 |
| --- | --- | ---: | ---: | --- |
| CoLA | `none` | 0.999755 | 199.944272 | 严重泄露 |
| CoLA | `lrbprojonly@0.99` | 0 | 0 | 当前最高 keep ratio 仍为零恢复 |
| CoLA | `topk@0.45` | 0 | 0 | 最大稳定零恢复比例 |
| CoLA | `topk@0.50` | 0.003333 | 0.150000 | 已越过零恢复边界 |
| CoLA | `compression@20` | 0 | 0 | 最大稳定零恢复 bit 数 |
| CoLA | `compression@22` | 0.514786 | 23.367069 | 已明显恢复 |
| Rotten Tomatoes | `none` | 0.885739 | 164.658710 | 严重泄露 |
| Rotten Tomatoes | `lrbprojonly@0.99` | 0 | 0 | 当前最高 keep ratio 仍为零恢复 |
| Rotten Tomatoes | `topk@0.55` | 0 | 0 | 最大稳定零恢复比例 |
| Rotten Tomatoes | `topk@0.60` | 0.001667 | 0 | token 指标已非零 |
| Rotten Tomatoes | `compression@20` | 0 | 0 | 最大稳定零恢复 bit 数 |
| Rotten Tomatoes | `compression@22` | 0.163051 | 2.747972 | 已越过零恢复边界 |

`DAGER=0` 只表示当前攻击预算和实现下未恢复，不是形式化隐私保证。

## 已有 utility

目前只有 CoLA 的以下完整 3-seed utility，可以复用但不能替代 P0：

| 防御 | accuracy mean +/- std | macro-F1 mean +/- std |
| --- | --- | --- |
| `none` | 0.748801 +/- 0.007488 | 0.627794 +/- 0.022603 |
| `dpsgd@5e-4` | 0.745286 +/- 0.003082 | 0.600982 +/- 0.016047 |
| `full_lrb@0.5` | 0.758389 +/- 0.005832 | 0.640371 +/- 0.015756 |

Rotten Tomatoes 暂无已汇总的 full-training utility。CoLA 也没有 Projection-LRB、top-k 边界或 compression 边界的 utility。

## 执行

在 Linux 训练服务器上：

```bash
cd /data/lrl/FedLLM
conda activate dager

# 先检查将执行的 24 条训练命令，不启动训练
bash scripts/run_dager_utility_cola_rt_critical.sh --profile p0 --dry-run

# 开始 P0；默认保留当前 CUDA_VISIBLE_DEVICES，并由 train.py 选择空闲可见 GPU
bash scripts/run_dager_utility_cola_rt_critical.sh --profile p0 --gpu auto
```

手动限制 GPU：

```bash
bash scripts/run_dager_utility_cola_rt_critical.sh --profile p0 --gpu 2
bash scripts/run_dager_utility_cola_rt_critical.sh --profile p0 --gpu 0,1
```

如果中断，指定原日志目录即可续跑；包含 `result_status=ok` 的 seed 会跳过：

```bash
bash scripts/run_dager_utility_cola_rt_critical.sh \
  --profile p0 \
  --log-dir log/runs/dager_utility_cola_rt_critical_p0_YYYYMMDD_HHMMSS
```

只跑一个数据集：

```bash
bash scripts/run_dager_utility_cola_rt_critical.sh --profile p0 --datasets cola
bash scripts/run_dager_utility_cola_rt_critical.sh --profile p0 --datasets rotten_tomatoes
```

runner 与隐私实验保持一致：batch size `2`、epoch `1`、full fine-tuning、seeds `101/202/303`，并从各自 clean 2-epoch checkpoint 继续训练。与 `utility_baselines.sh --baseline_defense ...` 连续调用不同，它不会为每个候选点重复训练 `none`。

## P1：P0 跑完后再补

P1 每个数据集再加两个配置，共 12 次训练：

- `lrbprojonly@0.90`：保守的 Projection-LRB 固定点，用于判断 `0.99` 是否存在偶然 utility 优势。
- `full_lrb@0.50`：过防御/机制参照。CoLA 已有结果；时间更紧时只补 Rotten Tomatoes。

```bash
# 两个数据集都补 P1
bash scripts/run_dager_utility_cola_rt_critical.sh --profile p1

# 时间更紧：只补 RT；CoLA 的 full_lrb@0.5 已有三 seed
bash scripts/run_dager_utility_cola_rt_critical.sh \
  --profile p1 \
  --datasets rotten_tomatoes
```

不要在第一轮跑 `noise/dpsgd/mixup/soteria` 的 dense utility sweep。它们不是当前主表的关键竞争者；CoLA 已有 `dpsgd@5e-4` coverage。

## 验收

runner 结束后检查新目录中的：

- `exit_codes.csv`：所有目标行应为 `0`。
- `utility_results.csv`：每行 `result_status=ok`、`n_runs=3`、`seeds=101 202 303`。
- `utility_results.md`：重点比较 `eval_accuracy`、`eval_macro_f1`、`eval_loss` 和 `total_train_time_seconds` 的 mean/std。

P0 的决策规则：

1. 先确认 Projection-LRB@0.99 相对 `none` 的 accuracy 和 macro-F1 drop。
2. 再与各数据集的 top-k 零恢复边界及共同的 compression@20 比较。
3. 若 Projection-LRB@0.99 utility 最好或接近最好，再补 P1 和 adaptive DAGER；否则先检查 `k=0.90`，不要继续盲目铺参数。
