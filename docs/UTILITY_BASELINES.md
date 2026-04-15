# Utility 指标测试说明

本文说明如何测试 `LRB` 和各类 defense baselines 对 utility 的影响，并统一汇总出：

- `proxy utility` 快速对照结果
- 训练期 `end-to-end utility` 结果
- `privacy-utility tradeoff` 联表

当前主入口是 [`scripts/utility_baselines.sh`](../scripts/utility_baselines.sh)。

相关脚本还包括：

- [`scripts/proxy_utility.py`](../scripts/proxy_utility.py)
- [`train.py`](../train.py)
- [`scripts/collect_experiment_logs.py`](../scripts/collect_experiment_logs.py)
- [`DEFENSE_BASELINES.md`](./DEFENSE_BASELINES.md)

## 1. 当前 utility 测什么

这里的 utility 评估分成两层：

1. `proxy utility`
   - 不真正重训整轮模型
   - 只在 held-out train batches 上比较原始梯度 `g` 和 defended 梯度 `g'`
   - 做 one-step virtual update，看 utility 指标怎么变化
2. `end-to-end utility`
   - 把 defense 真正挂进 `train.py`
   - 跑完整训练
   - 输出最终验证集 `accuracy / macro-F1 / loss / train time`

这样做的目的，是先用 proxy utility 快速筛方法，再用训练期 utility 给最终结论。

## 2. 当前固定对照的 baselines

utility 主表默认比较下面这些 operating points：

| defense | 参数 |
|---|---|
| `none` | `n/a` |
| `lrb` | `0.2` |
| `topk` | `0.1` |
| `compression` | `8` |
| `noise` | `5e-4` |
| `dpsgd` | `5e-4` |
| `mixup` | `0.3` |
| `soteria` | `30` |

可选 sensitivity check：

| defense | 参数 |
|---|---|
| `lrb` | `0.35` |
| `compression` | `16` |

说明：

- 主表不默认放 `lrb@0.35` 和 `compression@16`。
- 如果想把它们也一起跑，使用 `--include_sensitivity`。

## 3. 运行前准备

utility 流程依赖一个 `clean anchor model`。

脚本会按下面顺序自动寻找 anchor：

1. `--anchor_dir`
2. `--anchor_dir/final`
3. `./models/<model>-ft-clean`
4. `./models/<model>-ft-clean/final`
5. `./models/<model>-ft-rt`
6. `./models/<model>-ft-rt/final`

如果都找不到，脚本会自动先训练一个 `none` 的 clean anchor。

因此你有两种使用方式：

### 3.1 已有 clean anchor

例如：

```bash
bash scripts/utility_baselines.sh sst2 2 gpt2 1 \
  --anchor_dir ./models/gpt2-ft-rt
```

### 3.2 没有 clean anchor，自动补训

例如：

```bash
bash scripts/utility_baselines.sh sst2 2 gpt2 1
```

这时脚本会先自动训练：

- `none`
- `seed=101`
- 输出到类似 `./models/gpt2_utility_anchor_sst2/`

## 4. 一次跑完整套 utility baselines

最常用命令：

```bash
bash scripts/utility_baselines.sh sst2 2 gpt2 1 \
  --anchor_dir ./models/gpt2-ft-rt \
  --privacy_logs log/runs/test
```

位置参数含义：

| 位置 | 含义 |
|---|---|
| `$1` | 数据集，如 `sst2`、`cola`、`rte`、`rotten_tomatoes` |
| `$2` | `batch_size` |
| `$3` | `model_path`，如 `gpt2`、`bert-base-uncased` |
| `$4` | `num_epochs`，训练期 utility 的训练轮数 |
| `$5+` | 透传给 `train.py` / `proxy_utility.py` 的额外参数 |

脚本额外支持的控制参数：

| 参数 | 含义 |
|---|---|
| `--anchor_dir <path>` | 指定 clean anchor 模型目录 |
| `--privacy_logs <path>` | 指定 attack 日志目录或文件，用于生成 tradeoff 联表 |
| `--include_sensitivity` | 额外加入 `lrb@0.35` 和 `compression@16` |

说明：

- 每个 baseline 都会先跑一次 `proxy utility`。
- 然后每个 baseline 再跑 3 个 seed 的训练期 utility：
  - `101`
  - `202`
  - `303`
- 最后统一调用 [`scripts/collect_experiment_logs.py`](../scripts/collect_experiment_logs.py) 汇总结果。

## 5. 只跑单个 utility 组件

如果你不想一次跑完整流程，也可以单独跑。

### 5.1 只跑 proxy utility

主入口是 [`scripts/proxy_utility.py`](../scripts/proxy_utility.py)。

例如测试 `lrb@0.2`：

```bash
python3 scripts/proxy_utility.py \
  --dataset sst2 \
  --task seq_class \
  --model_path ./models/gpt2-ft-rt \
  --batch_size 2 \
  --train_method full \
  --defense lrb \
  --defense_lrb_keep_ratio_sensitive 0.2 \
  --n_train_batches 100 \
  --val_size 256 \
  --eval_batch_size 16 \
  --learning_rate 5e-5 \
  --log_file log/runs/proxy_lrb_0.2.txt
```

例如测试 `topk@0.1`：

```bash
python3 scripts/proxy_utility.py \
  --dataset sst2 \
  --task seq_class \
  --model_path ./models/gpt2-ft-rt \
  --batch_size 2 \
  --train_method full \
  --defense topk \
  --defense_topk_ratio 0.1 \
  --n_train_batches 100 \
  --val_size 256 \
  --eval_batch_size 16 \
  --log_file log/runs/proxy_topk_0.1.txt
```

例如只跑对照组 `none`：

```bash
python3 scripts/proxy_utility.py \
  --dataset sst2 \
  --task seq_class \
  --model_path ./models/gpt2-ft-rt \
  --batch_size 2 \
  --train_method full \
  --defense none \
  --n_train_batches 100 \
  --val_size 256 \
  --eval_batch_size 16 \
  --log_file log/runs/proxy_none.txt
```

proxy utility 默认关注的指标包括：

- `grad_cosine_mean`
- `norm_retention_mean`
- `delta_train_loss_mean`
- `delta_val_loss_mean`
- `delta_val_accuracy_mean`
- `delta_val_macro_f1_mean`
- `step_runtime_mean`

### 5.2 只跑训练期 end-to-end utility

主入口是 [`train.py`](../train.py)。

例如测试 `lrb@0.2`：

```bash
python3 train.py \
  --dataset sst2 \
  --task seq_class \
  --batch_size 2 \
  --num_epochs 1 \
  --save_every 0 \
  --model_path ./models/gpt2-ft-rt \
  --train_method full \
  --rng_seed 101 \
  --defense lrb \
  --defense_lrb_keep_ratio_sensitive 0.2 \
  --output_dir ./models/utility_lrb_seed101 \
  --log_file log/runs/train_lrb_seed101.txt
```

例如测试 `noise@5e-4`：

```bash
python3 train.py \
  --dataset sst2 \
  --task seq_class \
  --batch_size 2 \
  --num_epochs 1 \
  --save_every 0 \
  --model_path ./models/gpt2-ft-rt \
  --train_method full \
  --rng_seed 101 \
  --defense noise \
  --defense_noise 5e-4 \
  --output_dir ./models/utility_noise_seed101 \
  --log_file log/runs/train_noise_seed101.txt
```

例如测试 `dpsgd@5e-4`：

```bash
python3 train.py \
  --dataset sst2 \
  --task seq_class \
  --batch_size 2 \
  --num_epochs 1 \
  --save_every 0 \
  --model_path ./models/gpt2-ft-rt \
  --train_method full \
  --rng_seed 101 \
  --defense dpsgd \
  --defense_noise 5e-4 \
  --defense_clip_norm 1.0 \
  --output_dir ./models/utility_dpsgd_seed101 \
  --log_file log/runs/train_dpsgd_seed101.txt
```

例如只跑 `none` 对照组：

```bash
python3 train.py \
  --dataset sst2 \
  --task seq_class \
  --batch_size 2 \
  --num_epochs 1 \
  --save_every 0 \
  --model_path ./models/gpt2-ft-rt \
  --train_method full \
  --rng_seed 101 \
  --defense none \
  --output_dir ./models/utility_none_seed101 \
  --log_file log/runs/train_none_seed101.txt
```

训练期 utility 默认输出的关键指标包括：

- `eval_accuracy`
- `eval_macro_f1`
- `eval_loss`
- `final_train_loss`
- `total_train_time`

## 6. 旧参数兼容说明

`train.py` 现在统一推荐使用 `--defense` 接口。

不过为了兼容老命令，下面两个旧参数依然可用：

- `--noise`
- `--pct_mask`

它们会自动映射到：

- `--defense_noise`
- `--defense_pct_mask`

但新实验不建议继续混用旧风格，统一使用 `--defense ...` 更清晰。

## 7. 日志与结果文件说明

默认情况下，`scripts/utility_baselines.sh` 会在 `log/runs/` 下创建一个 run 目录。

目录名类似：

```text
log/runs/utility_baselines_sst2_b2_gpt2_YYYYMMDD_HHMMSS/
```

目录内常见文件如下：

| 文件 | 作用 |
|---|---|
| `anchor_train_none.txt` | 自动补训 clean anchor 的日志 |
| `proxy_<variant>.txt` | 每个 baseline 的 proxy utility 日志 |
| `train_<variant>_seed101.txt` 等 | 每个 baseline 的训练期 utility 日志 |
| `exit_codes.csv` | 每条任务的退出码 |
| `results.csv` | 原始解析后的逐日志结果 |
| `results.md` | 原始解析后的 Markdown 表 |
| `utility_results.csv` | 训练期 utility 聚合表 |
| `utility_results.md` | 训练期 utility Markdown 表 |
| `privacy_utility_tradeoff.csv` | utility 与 privacy 联表 |
| `privacy_utility_tradeoff.md` | tradeoff Markdown 表 |

### 7.1 proxy utility 日志尾部

proxy 日志尾部会包含：

```text
===== PROXY UTILITY SUMMARY START =====
...
===== PROXY UTILITY SUMMARY END =====
```

### 7.2 训练期 utility 日志尾部

训练日志尾部会包含：

```text
===== TRAIN RESULT SUMMARY START =====
...
===== TRAIN RESULT SUMMARY END =====
```

这两个 summary block 都可以被 [`scripts/collect_experiment_logs.py`](../scripts/collect_experiment_logs.py) 直接解析。

## 8. 如何手动汇总结果

如果你已经有若干 utility 日志，也可以单独调用收表脚本。

例如：

```bash
python3 scripts/collect_experiment_logs.py log/runs/utility_baselines_sst2_b2_gpt2_20260415_120000 log/runs/test \
  -o log/runs/utility_baselines_sst2_b2_gpt2_20260415_120000/results.csv \
  --markdown log/runs/utility_baselines_sst2_b2_gpt2_20260415_120000/results.md \
  --utility-output log/runs/utility_baselines_sst2_b2_gpt2_20260415_120000/utility_results.csv \
  --utility-markdown log/runs/utility_baselines_sst2_b2_gpt2_20260415_120000/utility_results.md \
  --tradeoff-output log/runs/utility_baselines_sst2_b2_gpt2_20260415_120000/privacy_utility_tradeoff.csv \
  --tradeoff-markdown log/runs/utility_baselines_sst2_b2_gpt2_20260415_120000/privacy_utility_tradeoff.md
```

说明：

- 第一个输入目录一般是 utility run 目录。
- 第二个输入目录一般是现有 attack/privacy 日志目录。
- 最终 `tradeoff` 表会把 training utility 和 attack privacy anchor 拼到一起。

## 9. 如何理解最终结果

最终建议重点看两张表：

### 9.1 `utility_results.csv`

这张表主要看训练期效用：

- `eval_accuracy`
- `eval_macro_f1`
- `eval_loss`
- `final_train_loss`
- `total_train_time`
- `failed_runs`

它回答的问题是：

- 某个 defense 会不会把任务性能打坏
- 三个 seed 下结果是否稳定
- 训练代价是否明显增加

### 9.2 `privacy_utility_tradeoff.csv`

这张表会把 utility 和 privacy 合并在一起，重点看：

- `eval_accuracy`
- `utility_drop`
- `rec_token_mean`
- `privacy_score`
- `agg_rouge1_fm`
- `agg_rouge2_fm`
- `pareto_optimal`

其中：

- `utility_drop = acc_none - acc_method`
- `privacy_score = 1 - rec_token_mean`

因此：

- `utility_drop` 越小越好
- `privacy_score` 越大越好

如果某个方法在 tradeoff 表里被标成 `pareto_optimal=true`，说明它位于当前结果的 Pareto front 上。

## 10. 当前推荐跑法

如果你现在只想得到一套可直接写结论的 utility 结果，推荐：

```bash
bash scripts/utility_baselines.sh sst2 2 gpt2 1 \
  --anchor_dir ./models/gpt2-ft-rt \
  --privacy_logs log/runs/test
```

如果你还想顺手把敏感性点也一起补掉，推荐：

```bash
bash scripts/utility_baselines.sh sst2 2 gpt2 1 \
  --anchor_dir ./models/gpt2-ft-rt \
  --privacy_logs log/runs/test \
  --include_sensitivity
```

## 11. 结论书写建议

utility 结论不建议写成“谁 accuracy 最高”，更合适的口径是：

- 在相近 privacy 水平下，`LRB` 是否拥有更小的 `utility_drop`
- 在相近 utility drop 下，`LRB` 是否拥有更高的 `privacy_score`
- `LRB` 是否位于 `privacy-utility tradeoff` 的 Pareto front 上

如果满足这些条件，可以写：

> `LRB` 在当前设定下处于更优的 privacy-utility tradeoff 前沿，在保持较小 utility 损失的同时提供了更强的隐私保护。

如果不满足，也建议如实写成：

> `LRB` 在 privacy 上更强，但 utility 上与 `topk` / `compression` 处于同档，或存在一定代价。
