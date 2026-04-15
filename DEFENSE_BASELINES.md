# 防御 Baselines 测试说明

本文说明如何在 DAGER 攻击框架下测试防御 baselines，并统一查看结果日志。

当前支持的 baseline 为：

- `none`
- `noise`
- `dpsgd`
- `topk`
- `compression`
- `soteria`
- `mixup`
- `lrb`

主入口是 [`scripts/defense_baselines.sh`](/D:/code/Projects/FedLLM/scripts/defense_baselines.sh)。

## 1. 运行前准备

建议先准备好微调模型，并通过 `--finetuned_path` 传入；否则脚本会直接退出，避免得到不可信的 seq_class 结果。

示例：

```bash
bash scripts/defense_baselines.sh sst2 2 gpt2 3 --finetuned_path ./models/gpt2-ft-rt
```

位置参数含义：

| 位置 | 含义 |
|---|---|
| `$1` | 数据集，如 `sst2`、`cola`、`rte`、`rotten_tomatoes` |
| `$2` | `batch_size` |
| `$3` | `model_path`，如 `gpt2`、`bert-base-uncased` |
| `$4` | `n_inputs`，测试样本数 |
| `$5+` | 透传给 `attack.py` 的额外参数 |

脚本自己额外支持两个“只影响 baseline 选择、不透传给 attack.py”的参数：

| 参数 | 含义 |
|---|---|
| `--baseline_defense <name>` | 只测试某个 baseline |
| `--baseline_param <value>` | 只测试该 baseline 的某一个参数值 |

## 2. 一次跑完整套防御 baselines

不加 `--baseline_defense` 时，会跑完整 baseline 集合：

- `none`
- `noise`
- `dpsgd`
- `topk`
- `compression`
- `soteria`
- `mixup`
- `lrb`

命令：

```bash
bash scripts/defense_baselines.sh sst2 2 gpt2 100 \
  --finetuned_path ./models/gpt2-ft-rt
```

说明：

- 除 `none` 外，其余防御会自动按默认 sweep 参数表做多档测试。
- 适合正式对比实验。

默认 sweep 参数如下：

| defense | sweep 参数 | 默认测试值 |
|---|---|---|
| `none` | 无 | 单次 |
| `noise` | `--defense_noise` | `1e-6 1e-5 1e-4 5e-4 1e-3` |
| `dpsgd` | `--defense_noise` | `1e-6 1e-5 1e-4 5e-4 1e-3` |
| `topk` | `--defense_topk_ratio` | `0.01 0.05 0.1 0.3 0.5 0.7 0.9` |
| `compression` | `--defense_n_bits` | `2 4 8 16 32` |
| `soteria` | `--defense_soteria_pruning_rate` | `10 30 50 70 90` |
| `mixup` | `--defense_mixup_alpha` | `0.1 0.3 0.5 1.0 2.0` |
| `lrb` | `--defense_lrb_keep_ratio_sensitive` | `0.05 0.1 0.2 0.35 0.5` |

语义说明：
- `dpsgd` 现在是标准 DP-SGD 语义：逐样本裁剪，再求平均，再按 `sigma * clip_norm / batch_size` 加高斯噪声。
- `soteria` 现在按分类头真正使用的 representation 打分，并剪掉**最高分**的那部分维度。
- `lrb` 现在默认是 `LRB-v2` 风格：用“结构先验 + 当前梯度统计”的混合敏感度校准，并把梯度投影到一个 seeded `signed_pool` 公共子空间，再主要向残差方向加噪。
- 旧版本 `dpsgd` / `soteria` 结果与当前实现**不可直接横向比较**。

## 3. 只跑某一个 baseline

如果你只想看某一种防御，使用 `--baseline_defense`。

注意：

- 当 `--baseline_defense` 不是 `none` 时，脚本会自动额外跑一份 `none` 作为对照。
- 也就是说，单 baseline 测试默认是“`none + 目标 defense`”。

### 3.1 只跑一个 baseline 的整组 sweep

例如只跑 `noise`：

```bash
bash scripts/defense_baselines.sh sst2 2 gpt2 20 \
  --baseline_defense noise \
  --finetuned_path ./models/gpt2-ft-rt
```

这条命令会跑：

- `none`
- `noise` 的全部默认 sweep 档位

再例如只跑 `soteria`：

```bash
bash scripts/defense_baselines.sh sst2 2 gpt2 20 \
  --baseline_defense soteria \
  --finetuned_path ./models/gpt2-ft-rt
```

这条命令会跑：

- `none`
- `soteria` 的全部默认 sweep 档位

### 3.2 只跑一个 baseline 的某一个参数值

如果你已经知道想测哪一档参数，再加 `--baseline_param`。

例如只跑 `noise=1e-4`：

```bash
bash scripts/defense_baselines.sh sst2 2 gpt2 20 \
  --baseline_defense noise \
  --baseline_param 1e-4 \
  --finetuned_path ./models/gpt2-ft-rt
```

这条命令只会跑：

- `none`
- `noise@1e-4`

例如只跑 `soteria=60`：

```bash
bash scripts/defense_baselines.sh sst2 2 gpt2 20 \
  --baseline_defense soteria \
  --baseline_param 60 \
  --finetuned_path ./models/gpt2-ft-rt
```

这条命令只会跑：

- `none`
- `soteria@60`

例如只跑对照组 `none`：

```bash
bash scripts/defense_baselines.sh sst2 2 gpt2 20 \
  --baseline_defense none \
  --finetuned_path ./models/gpt2-ft-rt
```

注意：

- `--baseline_defense none` 不能和 `--baseline_param` 一起使用。
- `--baseline_param` 必须和 `--baseline_defense` 一起使用。

## 4. 直接单独调用 attack.py

如果你不想走 baseline 脚本，也可以直接跑 `attack.py`。

优点：

- 更适合你自己手动调参
- 可以直接指定日志文件
- 日志尾部会输出统一的 `RESULT SUMMARY` 结果块

### 4.1 直接跑某个 baseline，并写日志文件

例如：

```bash
python attack.py --dataset sst2 --split val --n_inputs 10 --batch_size 2 \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 \
  --device cuda --task seq_class --cache_dir ./models_cache \
  --finetuned_path ./models/gpt2-ft-rt \
  --defense topk --defense_topk_ratio 0.1 \
  --log_file log/runs/topk_single.txt
```

这个日志文件末尾会包含：

```text
===== RESULT SUMMARY START =====
...
===== RESULT SUMMARY END =====
```

你可以直接查看该结果块获取最终指标。

### 4.2 常见单独测试命令

`noise`：

```bash
python attack.py --dataset sst2 --split val --n_inputs 10 --batch_size 2 \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 \
  --device cuda --task seq_class --cache_dir ./models_cache \
  --finetuned_path ./models/gpt2-ft-rt \
  --defense noise --defense_noise 1e-4
```

`dpsgd`：

```bash
python attack.py --dataset sst2 --split val --n_inputs 10 --batch_size 2 \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 \
  --device cuda --task seq_class --cache_dir ./models_cache \
  --finetuned_path ./models/gpt2-ft-rt \
  --defense dpsgd --defense_noise 1e-4 --defense_clip_norm 1.0
```

这里的 `--defense_clip_norm` 是逐样本裁剪阈值 `C`，`--defense_noise` 是 noise multiplier `sigma`。

`topk`：

```bash
python attack.py --dataset sst2 --split val --n_inputs 10 --batch_size 2 \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 \
  --device cuda --task seq_class --cache_dir ./models_cache \
  --finetuned_path ./models/gpt2-ft-rt \
  --defense topk --defense_topk_ratio 0.1
```

`compression`：

```bash
python attack.py --dataset sst2 --split val --n_inputs 10 --batch_size 2 \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 \
  --device cuda --task seq_class --cache_dir ./models_cache \
  --finetuned_path ./models/gpt2-ft-rt \
  --defense compression --defense_n_bits 8
```

`soteria`：

```bash
python attack.py --dataset sst2 --split val --n_inputs 10 --batch_size 2 \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 \
  --device cuda --task seq_class --cache_dir ./models_cache \
  --finetuned_path ./models/gpt2-ft-rt \
  --defense soteria --defense_soteria_pruning_rate 60
```

`mixup`：

```bash
python attack.py --dataset sst2 --split val --n_inputs 10 --batch_size 2 \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 \
  --device cuda --task seq_class --cache_dir ./models_cache \
  --finetuned_path ./models/gpt2-ft-rt \
  --defense mixup --defense_mixup_alpha 1.0
```

`lrb`：

```bash
python attack.py --dataset sst2 --split val --n_inputs 10 --batch_size 2 \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 \
  --device cuda --task seq_class --cache_dir ./models_cache \
  --finetuned_path ./models/gpt2-ft-rt \
  --defense lrb \
  --defense_lrb_sensitive_n_layers 2 \
  --defense_lrb_keep_ratio_sensitive 0.2 \
  --defense_lrb_keep_ratio_other 0.75 \
  --defense_lrb_empirical_weight 0.6 \
  --defense_lrb_projection signed_pool \
  --defense_lrb_noise_sensitive 0.03 \
  --defense_lrb_noise_other 0.005
```

补充说明：

- `--defense_lrb_empirical_weight` 控制 LRB 的混合校准强度：
  - `0` 表示只用层级启发式先验
  - `1` 表示只用当前梯度统计
- `--defense_lrb_projection signed_pool` 是默认推荐设置，它比旧的坐标系 pooling 更接近“公共随机子空间”。
- 如果你想退回更接近旧版 LRB v1 的行为，可以显式设置：

```bash
--defense_lrb_empirical_weight 0 --defense_lrb_projection pool
```

## 5. 日志与结果文件说明

默认情况下，`defense_baselines.sh` 会在 `log/runs/` 下创建一个 run 目录。

目录内主要文件如下：

| 文件 | 作用 |
|---|---|
| `_run_header.txt` | 本次 run 的头信息 |
| `none.txt`、`noise_1e-4.txt` 等 | 每个变体的最终结果摘要 |
| `summary.txt` | 所有变体的汇总结果和 comparison 表 |
| `results.csv` | 结构化结果表，适合后续统计/画图 |
| `results.md` | Markdown 表格版结果 |

### 5.1 每个变体日志里有什么

每个变体文件都会包含一段统一结果块：

```text
===== RESULT SUMMARY START =====
summary_version=1
result_status=ok
dataset=...
defense=...
defense_param_name=...
defense_param_value=...
rec_token_mean=...
rec_maxb_token_mean=...
agg_rouge1_fm=...
agg_rouge2_fm=...
agg_r1fm_r2fm=...
...
===== RESULT SUMMARY END =====
```

你单独测试某个 baseline 时，直接看这个文件就能拿到最终指标。

### 5.2 summary.txt 看什么

`summary.txt` 里有两类内容：

- 每个变体的 `RESULT SUMMARY`
- 末尾一张 `COMPARISON` 表

`COMPARISON` 表至少包含：

- `variant`
- `defense`
- `param`
- `rec_token`
- `rec_maxb_token`
- `rouge1_fm`
- `rouge2_fm`
- `r1+r2`
- `last_rec_status`
- `total_time`
- `status`

### 5.3 results.csv 看什么

`results.csv` 适合你后续做筛选、画图、导入表格软件。

生成方式已经内置在 `defense_baselines.sh` 中，脚本跑完会自动生成，不需要你额外手动执行。

如果你想手动重新汇总，也可以：

```bash
python scripts/collect_experiment_logs.py log/runs/your_run_dir/*.txt -o log/runs/your_run_dir/results.csv
```

更推荐直接使用脚本自动生成的 `results.csv`。

## 6. 推荐终端操作流程

### 6.1 冒烟测试

先用很小的 `n_inputs` 确认流程正常：

```bash
bash scripts/defense_baselines.sh sst2 2 gpt2 1 \
  --finetuned_path ./models/gpt2-ft-rt
```

### 6.2 正式全量 baseline 实验

```bash
bash scripts/defense_baselines.sh sst2 2 gpt2 100 \
  --finetuned_path ./models/gpt2-ft-rt
```

### 6.3 只看一个 baseline 的 sweep

```bash
bash scripts/defense_baselines.sh sst2 2 gpt2 20 \
  --baseline_defense noise \
  --finetuned_path ./models/gpt2-ft-rt
```

### 6.4 只看一个 baseline 的某一个参数

```bash
bash scripts/defense_baselines.sh sst2 2 gpt2 20 \
  --baseline_defense noise \
  --baseline_param 1e-4 \
  --finetuned_path ./models/gpt2-ft-rt
```

### 6.5 推荐查看顺序

1. 先看终端最后打印出来的 run 目录路径
2. 再看该目录下的 `summary.txt`
3. 然后看目标变体对应的 `*.txt`
4. 最后用 `results.csv` 做整体对比或后处理

## 7. 注意事项

- `noise` 和 `dpsgd` 必须提供 `--defense_noise`。
- `mixup` 只有在 `batch_size >= 2` 时才有真实混合效果。
- `mixup` 只对 `seq_class` 任务有意义。
- `soteria` 在大模型上建议额外加 `--defense_soteria_sample_dims 256` 以减少开销。
- `soteria` 在 `train_method=lora` 下未集成。
- 如果你只是想对某一个 baseline 做单点对比，优先使用：
  `--baseline_defense <name> --baseline_param <value>`

## 8. 相关代码入口

- 防御实现入口：[`utils/defenses.py`](/D:/code/Projects/FedLLM/utils/defenses.py)
- 攻击主脚本：[`attack.py`](/D:/code/Projects/FedLLM/attack.py)
- baseline 脚本：[`scripts/defense_baselines.sh`](/D:/code/Projects/FedLLM/scripts/defense_baselines.sh)
- 结果汇总器：[`scripts/collect_experiment_logs.py`](/D:/code/Projects/FedLLM/scripts/collect_experiment_logs.py)
