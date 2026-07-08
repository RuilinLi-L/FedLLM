# Partial Gradient 中文 Runbook

本文档记录 FedLLM 项目中 partial-gradient 隐私与效用实验的推荐跑法。当前建议把 partial gradient 作为 AAAI 投稿中的补充鲁棒性实验；主结果仍以 full-gradient DAGER 为核心。

## 0. 结论先行

当前有两条 partial-gradient 隐私实验线：

| 路线 | 入口 | 定位 | 当前建议 |
|---|---|---|---|
| DAGER partial exposure | [`scripts/partial_gradient_baselines.sh`](../scripts/partial_gradient_baselines.sh) -> [`attack.py`](../attack.py) | 证据更稳，沿用 DAGER span/candidate decoding，只暴露部分层或部分参数梯度 | 优先跑，作为 partial-gradient 主证据 |
| PTG / LAMP-lite gradient matching | [`scripts/ptg_baselines.sh`](../scripts/ptg_baselines.sh) -> [`attack_partial_gradient.py`](../attack_partial_gradient.py) | 更接近 partial Transformer gradients 原文思想，用 dummy embedding gradient matching | 先做 smoke/pilot，结果稳定后再进表 |

推荐主设置：

```bash
DATASET=sst2
BATCH=2
MODEL=gpt2
N_INPUTS=100
CKPT=./models/gpt2_sst2_clean_num_epochs_2/final
EXPOSURE=first2
```

推荐主表先跑 `first2`，因为这是当前最稳的 partial-gradient exposure。随后可扩展 `mid2`、`last2`、`qkv_only`。效用实验不依赖 exposure；同一个 defense operating point 的 utility 可以和 DAGER partial 或 PTG partial 隐私结果拼表。

所有命令默认在项目根目录运行：

```bash
cd D:/code/Projects/FedLLM
```

在 Windows PowerShell 中可以显式加 `bash`：

```powershell
bash scripts/partial_gradient_baselines.sh ...
```

## 1. 隐私实验 A：DAGER-Based Partial Gradient

### 1.1 跑法定位

DAGER-based partial-gradient 实验通过 `attack.py` 运行，只让攻击者看到部分梯度：

- `first2`：前两层梯度可见，推荐作为主设置。
- `mid2`：中间两层梯度可见。
- `last2`：最后两层梯度可见。
- `qkv_only`：只暴露 Q/K/V 相关参数梯度。
- `lora_only`：只暴露 LoRA / PEFT adapter 梯度，需要 `--train_method lora` 或 `--train_method peft`。

输出目录形如：

```text
log/runs/partial_gradient_<exposure>_<train_method>_<dataset>_b<batch>_<model>_*/
```

主要看：

- `summary.txt`
- `results.csv`
- `results.md`
- 每个 variant 对应的 `*.txt`

关键字段：

| 字段 | 含义 |
|---|---|
| `result_status` | 该行是否成功完成 |
| `partial_attack_variant` | partial 攻击变体，例如 `dager_prefix_visible` |
| `gradient_layer_subset` | 可见层集合，例如 `first2` |
| `gradient_param_filter` | 可见参数过滤器，例如 `all` / `qkv_only` |
| `visible_grad_count` | 可见梯度 tensor 数量 |
| `selected_block_ids` | DAGER 实际选中的 block |
| `rec_token_mean` | token 恢复率，越低越隐私 |
| `agg_r1fm_r2fm` | ROUGE-1 + ROUGE-2 聚合指标，越低越隐私 |
| `last_total_time` | 攻击耗时 |

### 1.2 推荐主矩阵

正式 partial-gradient 隐私表建议先覆盖：

| 方法 | 参数 | 地位 |
|---|---|---|
| `none` | `n/a` | clean leakage 对照 |
| `proj_only` | `k=0.9` | Projection-LRB 主方法 |
| `proj_only` | `k=0.99` | 复现已有 partial 强防御点时可加跑 |
| `full_lrb` | `k=0.5` | over-defense 对照，不作为主方法 |
| `topk` | `0.1` | 强 empirical baseline |
| `compression` | `8` | 强 empirical baseline |
| `noise` | `5e-4` | coverage baseline |
| `dpsgd` | `5e-4` | coverage baseline；无 accountant 时不报告正式 DP 保证 |
| `soteria` | `30` | coverage baseline |
| `mixup` | `0.3` | coverage baseline |

### 1.3 Clean

```bash
./scripts/partial_gradient_baselines.sh sst2 2 gpt2 100 \
  --exposure first2 \
  --baseline_defense none \
  --finetuned_path ./models/gpt2_sst2_clean_num_epochs_2/final
```

### 1.4 Projection-LRB 主方法：`proj_only@k=0.9`

```bash
./scripts/partial_gradient_baselines.sh sst2 2 gpt2 100 \
  --exposure first2 \
  --baseline_defense lrb \
  --lrb_variants proj_only \
  --lrb_main_k 0.9 \
  --finetuned_path ./models/gpt2_sst2_clean_num_epochs_2/final
```

如果要复现已有 partial-gradient 强防御点，可额外跑 `proj_only@k=0.99`：

```bash
./scripts/partial_gradient_baselines.sh sst2 2 gpt2 100 \
  --exposure first2 \
  --baseline_defense lrb \
  --lrb_variants proj_only \
  --lrb_main_k 0.99 \
  --finetuned_path ./models/gpt2_sst2_clean_num_epochs_2/final
```

### 1.5 Full LRB over-defense 对照

```bash
./scripts/partial_gradient_baselines.sh sst2 2 gpt2 100 \
  --exposure first2 \
  --baseline_defense lrb \
  --lrb_variants full_lrb \
  --lrb_main_k 0.5 \
  --finetuned_path ./models/gpt2_sst2_clean_num_epochs_2/final
```

### 1.6 Top-k 强 baseline

推荐主点是 `topk@0.1`：

```bash
./scripts/partial_gradient_baselines.sh sst2 2 gpt2 100 \
  --exposure first2 \
  --baseline_defense topk \
  --baseline_param 0.1 \
  --finetuned_path ./models/gpt2_sst2_clean_num_epochs_2/final
```

如需与已有 partial 结果对齐，也可加跑 `topk@0.3`：

```bash
./scripts/partial_gradient_baselines.sh sst2 2 gpt2 100 \
  --exposure first2 \
  --baseline_defense topk \
  --baseline_param 0.3 \
  --finetuned_path ./models/gpt2_sst2_clean_num_epochs_2/final
```

### 1.7 Compression 强 baseline

推荐主点是 `compression@8`：

```bash
./scripts/partial_gradient_baselines.sh sst2 2 gpt2 100 \
  --exposure first2 \
  --baseline_defense compression \
  --baseline_param 8 \
  --finetuned_path ./models/gpt2_sst2_clean_num_epochs_2/final
```

如需与已有 partial 结果对齐，也可加跑更强的 `compression@20`：

```bash
./scripts/partial_gradient_baselines.sh sst2 2 gpt2 100 \
  --exposure first2 \
  --baseline_defense compression \
  --baseline_param 20 \
  --finetuned_path ./models/gpt2_sst2_clean_num_epochs_2/final
```

### 1.8 Coverage baselines

```bash
./scripts/partial_gradient_baselines.sh sst2 2 gpt2 100 \
  --exposure first2 \
  --baseline_defense noise \
  --baseline_param 5e-4 \
  --finetuned_path ./models/gpt2_sst2_clean_num_epochs_2/final
```

```bash
./scripts/partial_gradient_baselines.sh sst2 2 gpt2 100 \
  --exposure first2 \
  --baseline_defense dpsgd \
  --baseline_param 5e-4 \
  --finetuned_path ./models/gpt2_sst2_clean_num_epochs_2/final
```

```bash
./scripts/partial_gradient_baselines.sh sst2 2 gpt2 100 \
  --exposure first2 \
  --baseline_defense soteria \
  --baseline_param 30 \
  --finetuned_path ./models/gpt2_sst2_clean_num_epochs_2/final
```

```bash
./scripts/partial_gradient_baselines.sh sst2 2 gpt2 100 \
  --exposure first2 \
  --baseline_defense mixup \
  --baseline_param 0.3 \
  --finetuned_path ./models/gpt2_sst2_clean_num_epochs_2/final
```

### 1.9 扩展 exposure

确认 `first2` 跑通后，再把同一批 baseline 换成其他 exposure：

```bash
./scripts/partial_gradient_baselines.sh sst2 2 gpt2 100 \
  --exposure qkv_only \
  --baseline_defense lrb \
  --lrb_variants proj_only \
  --lrb_main_k 0.9 \
  --finetuned_path ./models/gpt2_sst2_clean_num_epochs_2/final
```

```bash
./scripts/partial_gradient_baselines.sh sst2 2 gpt2 100 \
  --exposure mid2 \
  --baseline_defense topk \
  --baseline_param 0.1 \
  --finetuned_path ./models/gpt2_sst2_clean_num_epochs_2/final
```

```bash
./scripts/partial_gradient_baselines.sh sst2 2 gpt2 100 \
  --exposure last2 \
  --baseline_defense compression \
  --baseline_param 8 \
  --finetuned_path ./models/gpt2_sst2_clean_num_epochs_2/final
```

扩展 exposure 不需要一开始全量跑完所有 coverage baselines。建议先跑 `none`、`proj_only@0.9`、`topk@0.1`、`compression@8`。

## 2. 隐私实验 B：PTG / LAMP-lite Gradient Matching

### 2.1 跑法定位

PTG / LAMP-lite 是更接近 partial Transformer gradients 原文思想的路径。它不调用 DAGER span decomposition 或 candidate decoding，而是：

1. 计算真实 batch 梯度。
2. 应用指定 defense。
3. 只保留选定层或模块的 partial gradients。
4. 优化 dummy input embeddings，使 dummy gradients 匹配可见 partial gradients。
5. 将优化后的 embeddings decode 到 token，并计算恢复指标。

当前代码目标是 mechanism parity，不是完整复现原文所有表格数值。论文写法建议称为：

> FedLLM LAMP-lite adaptation of partial Transformer gradient leakage.

输出目录形如：

```text
log/runs/ptg_<dataset>_b<batch>_<model>_*/
```

主要看：

- `summary.txt`
- `results.csv`
- `results.md`
- 每个 exposure/defense variant 的 `*.txt`

关键字段：

| 字段 | 含义 |
|---|---|
| `result_status` | 该行是否成功完成 |
| `selected_gradient_count` | 实际用于 matching 的梯度数量 |
| `ptg_initial_loss` | 初始 gradient matching loss |
| `ptg_final_loss` | 最终 gradient matching loss |
| `ptg_loss_reduction` | loss 下降量或比例 |
| `fixed_token_count` | 被固定的 special/pad token 数量 |
| `rec_token_mean` | token 恢复率 |
| `agg_r1fm_r2fm` | ROUGE-1 + ROUGE-2 聚合指标 |

PTG 结果进入论文表的最低门槛：

- `result_status=ok`。
- `selected_gradient_count > 0`。
- `ptg_final_loss < ptg_initial_loss`。
- decoded text 非空。
- `n_inputs=10` pilot 至少能稳定产出 summary。

如果不满足这些条件，PTG 只作为实现状态、pilot 或负结果说明，不放进主结果表。

### 2.2 GPT-2 pilot

先用小规模 GPT-2 pilot 检查流程：

```bash
./scripts/ptg_baselines.sh sst2 1 gpt2 10 \
  --exposure first2,qkv_only \
  --finetuned_path ./models/gpt2_sst2_clean_num_epochs_2/final \
  --ptg_steps 80 \
  --ptg_lr 0.1 \
  --ptg_embed_norm_weight 0.01
```

注意：GPT-2 的 Q/K/V 在 `c_attn` 中是 packed tensor，因此 `query_only`、`key_only`、`value_only` 在 GPT-2 上会等价或接近 `qkv_only`。如果要精确复现原文 selector，优先等 BERT fine-tuned checkpoint 准备好后再跑 BERT 版本。

### 2.3 PTG clean

```bash
./scripts/ptg_baselines.sh sst2 1 gpt2 10 \
  --exposure first2,qkv_only \
  --baseline_defense none \
  --finetuned_path ./models/gpt2_sst2_clean_num_epochs_2/final \
  --ptg_steps 80 \
  --ptg_lr 0.1 \
  --ptg_embed_norm_weight 0.01
```

### 2.4 PTG Projection-LRB

```bash
./scripts/ptg_baselines.sh sst2 1 gpt2 10 \
  --exposure first2,qkv_only \
  --baseline_defense proj_only \
  --baseline_param 0.9 \
  --finetuned_path ./models/gpt2_sst2_clean_num_epochs_2/final \
  --ptg_steps 80 \
  --ptg_lr 0.1 \
  --ptg_embed_norm_weight 0.01
```

### 2.5 PTG top-k

```bash
./scripts/ptg_baselines.sh sst2 1 gpt2 10 \
  --exposure first2,qkv_only \
  --baseline_defense topk \
  --baseline_param 0.1 \
  --finetuned_path ./models/gpt2_sst2_clean_num_epochs_2/final \
  --ptg_steps 80 \
  --ptg_lr 0.1 \
  --ptg_embed_norm_weight 0.01
```

### 2.6 PTG compression

```bash
./scripts/ptg_baselines.sh sst2 1 gpt2 10 \
  --exposure first2,qkv_only \
  --baseline_defense compression \
  --baseline_param 8 \
  --finetuned_path ./models/gpt2_sst2_clean_num_epochs_2/final \
  --ptg_steps 80 \
  --ptg_lr 0.1 \
  --ptg_embed_norm_weight 0.01
```

### 2.7 PTG coverage baselines

这些只建议在 clean / Projection-LRB / top-k / compression 跑通后补：

```bash
./scripts/ptg_baselines.sh sst2 1 gpt2 10 \
  --exposure first2,qkv_only \
  --baseline_defense noise \
  --baseline_param 5e-4 \
  --finetuned_path ./models/gpt2_sst2_clean_num_epochs_2/final \
  --ptg_steps 80 \
  --ptg_lr 0.1 \
  --ptg_embed_norm_weight 0.01
```

```bash
./scripts/ptg_baselines.sh sst2 1 gpt2 10 \
  --exposure first2,qkv_only \
  --baseline_defense dpsgd \
  --baseline_param 5e-4 \
  --finetuned_path ./models/gpt2_sst2_clean_num_epochs_2/final \
  --ptg_steps 80 \
  --ptg_lr 0.1 \
  --ptg_embed_norm_weight 0.01
```

```bash
./scripts/ptg_baselines.sh sst2 1 gpt2 10 \
  --exposure first2,qkv_only \
  --baseline_defense soteria \
  --baseline_param 30 \
  --finetuned_path ./models/gpt2_sst2_clean_num_epochs_2/final \
  --ptg_steps 80 \
  --ptg_lr 0.1 \
  --ptg_embed_norm_weight 0.01
```

```bash
./scripts/ptg_baselines.sh sst2 1 gpt2 10 \
  --exposure first2,qkv_only \
  --baseline_defense mixup \
  --baseline_param 0.3 \
  --finetuned_path ./models/gpt2_sst2_clean_num_epochs_2/final \
  --ptg_steps 80 \
  --ptg_lr 0.1 \
  --ptg_embed_norm_weight 0.01
```

### 2.8 BERT source-selector 版本

如果以后准备好 BERT fine-tuned checkpoint，例如 `./models/bert-sst2`，可以跑更贴近原文 selector 的版本：

```bash
./scripts/ptg_baselines.sh sst2 1 bert-base-uncased 10 \
  --exposure query_only,key_only,value_only,ffn_only \
  --finetuned_path ./models/bert-sst2 \
  --ptg_steps 80 \
  --ptg_lr 0.1 \
  --ptg_embed_norm_weight 0.01
```

进一步的 full module sweep：

```bash
./scripts/ptg_baselines.sh sst2 1 bert-base-uncased 10 \
  --full_sweep \
  --finetuned_path ./models/bert-sst2 \
  --ptg_steps 80 \
  --ptg_lr 0.1 \
  --ptg_embed_norm_weight 0.01
```

如果要打开 source parity mode，可参考 [`docs/PARTIAL_TRANSFORMER_GRADIENTS_REPRO.md`](./PARTIAL_TRANSFORMER_GRADIENTS_REPRO.md)。在没有稳定 pilot 前，不建议把 source parity 的长步数配置直接并入正式表。

## 3. 效用实验

### 3.1 跑法定位

效用实验入口是 [`scripts/utility_baselines.sh`](../scripts/utility_baselines.sh)。该脚本会做两件事：

1. 在 clean anchor checkpoint 上跑 proxy utility。
2. 用 `101`、`202`、`303` 三个 seed 跑 end-to-end training utility。

输出目录形如：

```text
log/runs/utility_baselines_<dataset>_b<batch>_<model>_*/
```

主要看：

- `utility_results.csv`
- `utility_results.md`
- `privacy_utility_tradeoff.csv`
- `privacy_utility_tradeoff.md`
- `results.csv`
- `results.md`

关键字段：

| 字段 | 含义 |
|---|---|
| `eval_accuracy` | 验证集 accuracy |
| `eval_macro_f1` | 验证集 macro-F1 |
| `eval_loss` | 验证集 loss |
| `final_train_loss` | 最终训练 loss |
| `total_train_time` | 训练时间 |
| `utility_drop` | 相对 clean 的 accuracy drop |
| `rec_token_mean` | 拼表后的隐私恢复率 |
| `privacy_score` | 通常为 `1 - rec_token_mean` |
| `pareto_optimal` | 当前表内是否在 Pareto front |

注意：utility 不依赖 partial exposure。只要 defense operating point 一致，就可以把同一份 utility 和 partial-gradient privacy 结果拼在一起。

### 3.2 Top-k utility

```bash
./scripts/utility_baselines.sh sst2 2 gpt2 1 \
  --anchor_dir ./models/gpt2_sst2_clean_num_epochs_2 \
  --baseline_defense topk \
  --baseline_param 0.1 \
  --privacy_logs <partial_run_dir>
```

其中 `<partial_run_dir>` 替换成对应的 `log/runs/partial_gradient_*` 或 `log/runs/ptg_*` 目录。

### 3.3 Compression utility

```bash
./scripts/utility_baselines.sh sst2 2 gpt2 1 \
  --anchor_dir ./models/gpt2_sst2_clean_num_epochs_2 \
  --baseline_defense compression \
  --baseline_param 8 \
  --privacy_logs <partial_run_dir>
```

### 3.4 Coverage utility baselines

```bash
./scripts/utility_baselines.sh sst2 2 gpt2 1 \
  --anchor_dir ./models/gpt2_sst2_clean_num_epochs_2 \
  --baseline_defense noise \
  --baseline_param 5e-4 \
  --privacy_logs <partial_run_dir>
```

```bash
./scripts/utility_baselines.sh sst2 2 gpt2 1 \
  --anchor_dir ./models/gpt2_sst2_clean_num_epochs_2 \
  --baseline_defense dpsgd \
  --baseline_param 5e-4 \
  --privacy_logs <partial_run_dir>
```

```bash
./scripts/utility_baselines.sh sst2 2 gpt2 1 \
  --anchor_dir ./models/gpt2_sst2_clean_num_epochs_2 \
  --baseline_defense soteria \
  --baseline_param 30 \
  --privacy_logs <partial_run_dir>
```

```bash
./scripts/utility_baselines.sh sst2 2 gpt2 1 \
  --anchor_dir ./models/gpt2_sst2_clean_num_epochs_2 \
  --baseline_defense mixup \
  --baseline_param 0.3 \
  --privacy_logs <partial_run_dir>
```

### 3.5 Projection-LRB utility

当前 `utility_baselines.sh` 的 `lrb` 简化接口不能稳定表达 `proj_only` preset。因此 `proj_only@k=0.9` 的 utility 建议：

1. 优先沿用已有 full-gradient utility 结果，如果 operating point 完全一致。
2. 或者直接用 `train.py` 显式指定 preset，并跑 `101`、`202`、`303` 三个 seed。

Seed 101：

```bash
python train.py \
  --dataset sst2 \
  --task seq_class \
  --batch_size 2 \
  --num_epochs 1 \
  --model_path ./models/gpt2_sst2_clean_num_epochs_2/final \
  --train_method full \
  --defense lrb \
  --defense_lrb_preset proj_only \
  --defense_lrb_keep_ratio_sensitive 0.9 \
  --rng_seed 101 \
  --output_dir log/runs/utility_proj_only_k0.9/models/seed101 \
  --log_file log/runs/utility_proj_only_k0.9/train_seed101.txt
```

Seed 202：

```bash
python train.py \
  --dataset sst2 \
  --task seq_class \
  --batch_size 2 \
  --num_epochs 1 \
  --model_path ./models/gpt2_sst2_clean_num_epochs_2/final \
  --train_method full \
  --defense lrb \
  --defense_lrb_preset proj_only \
  --defense_lrb_keep_ratio_sensitive 0.9 \
  --rng_seed 202 \
  --output_dir log/runs/utility_proj_only_k0.9/models/seed202 \
  --log_file log/runs/utility_proj_only_k0.9/train_seed202.txt
```

Seed 303：

```bash
python train.py \
  --dataset sst2 \
  --task seq_class \
  --batch_size 2 \
  --num_epochs 1 \
  --model_path ./models/gpt2_sst2_clean_num_epochs_2/final \
  --train_method full \
  --defense lrb \
  --defense_lrb_preset proj_only \
  --defense_lrb_keep_ratio_sensitive 0.9 \
  --rng_seed 303 \
  --output_dir log/runs/utility_proj_only_k0.9/models/seed303 \
  --log_file log/runs/utility_proj_only_k0.9/train_seed303.txt
```

如果要把 direct training 的 utility logs 和 privacy logs 拼表，可手动调用收集器：

```bash
python scripts/collect_experiment_logs.py \
  log/runs/utility_proj_only_k0.9 \
  <partial_run_dir> \
  -o log/runs/utility_proj_only_k0.9/results.csv \
  --markdown log/runs/utility_proj_only_k0.9/results.md \
  --utility-output log/runs/utility_proj_only_k0.9/utility_results.csv \
  --utility-markdown log/runs/utility_proj_only_k0.9/utility_results.md \
  --tradeoff-output log/runs/utility_proj_only_k0.9/privacy_utility_tradeoff.csv \
  --tradeoff-markdown log/runs/utility_proj_only_k0.9/privacy_utility_tradeoff.md
```

### 3.6 Full LRB utility

`full_lrb@k=0.5` 是 over-defense 对照。建议同样用 direct training 明确 preset：

```bash
python train.py \
  --dataset sst2 \
  --task seq_class \
  --batch_size 2 \
  --num_epochs 1 \
  --model_path ./models/gpt2_sst2_clean_num_epochs_2/final \
  --train_method full \
  --defense lrb \
  --defense_lrb_preset full_lrb \
  --defense_lrb_keep_ratio_sensitive 0.5 \
  --rng_seed 101 \
  --output_dir log/runs/utility_full_lrb_k0.5/models/seed101 \
  --log_file log/runs/utility_full_lrb_k0.5/train_seed101.txt
```

正式表同样跑 `101`、`202`、`303`。

## 4. 推荐执行顺序

### 4.1 最小可发表 partial-gradient 包

先跑 DAGER-based partial：

1. `none`
2. `proj_only@k=0.9`
3. `topk@0.1`
4. `compression@8`
5. `full_lrb@k=0.5`

然后跑对应 utility：

1. `topk@0.1`
2. `compression@8`
3. `proj_only@k=0.9`
4. `full_lrb@k=0.5`

最后补 coverage：

1. `noise@5e-4`
2. `dpsgd@5e-4`
3. `soteria@30`
4. `mixup@0.3`

### 4.2 PTG pilot 包

PTG 不建议一开始全量扫。先跑：

1. `none`
2. `proj_only@k=0.9`
3. `topk@0.1`
4. `compression@8`

Exposure 只用：

1. `first2`
2. `qkv_only`

`n_inputs=10` 结果稳定后，再考虑：

- 增加 `n_inputs`。
- 扩展 `mid2` / `last2`。
- 等 BERT checkpoint 就绪后跑 `query_only` / `key_only` / `value_only` / `ffn_only`。

## 5. 结果检查清单

### 5.1 DAGER partial

每个 run 完成后检查：

```text
log/runs/partial_gradient_*/results.csv
log/runs/partial_gradient_*/results.md
log/runs/partial_gradient_*/summary.txt
```

必须确认：

- `result_status=ok` 或明确记录失败原因。
- `partial_filter_active=true`。
- `gradient_layer_subset` 和 `gradient_param_filter` 与预期一致。
- `visible_grad_count` 不是空。
- `rec_token_mean` 和 `agg_r1fm_r2fm` 有数值。
- `last_total_time` 有数值。

### 5.2 PTG

每个 run 完成后检查：

```text
log/runs/ptg_*/results.csv
log/runs/ptg_*/results.md
log/runs/ptg_*/summary.txt
```

必须确认：

- `selected_gradient_count > 0`。
- `ptg_initial_loss` 和 `ptg_final_loss` 都存在。
- `ptg_final_loss < ptg_initial_loss`。
- decoded text 非空。
- `rec_token_mean` 和 `agg_r1fm_r2fm` 有数值。

### 5.3 Utility

每个 run 完成后检查：

```text
log/runs/utility_baselines_*/utility_results.csv
log/runs/utility_baselines_*/privacy_utility_tradeoff.csv
```

必须确认：

- 三个 seed 都完成，或 `failed_runs` 清楚标记。
- `eval_accuracy`、`eval_macro_f1`、`eval_loss` 都有数值。
- `privacy_utility_tradeoff.csv` 中 defense 名称和 privacy run 的 defense 名称能对上。
- `utility_drop` 的计算基准是同一批 run 中的 `none`。

## 6. 论文写法建议

建议写：

- Full-gradient DAGER 是主攻击设置和主结果。
- Partial-gradient 是 additional attack surface / robustness evaluation。
- DAGER partial exposure 说明在只暴露部分层或模块梯度时仍可评估文本泄漏。
- PTG / LAMP-lite 说明我们实现了 partial Transformer gradient leakage 的核心机制适配。
- Projection-LRB 是机制驱动的 recoverability bottleneck 防御，不是 formal privacy guarantee。
- `topk@0.1` 和 `compression@8` 是强 empirical baselines，必须公平比较。
- `noise`、`dpsgd`、`soteria`、`mixup` 是 coverage baselines。

避免写：

- 不要说 `DAGER=0` 等于安全或形式化隐私。
- 不要说 Projection-LRB 有 epsilon/delta DP 保证。
- 不要说 PTG 已完整复现原文所有表格。
- 不要说 Projection-LRB 在所有 partial-gradient 设置下都优于 top-k / compression。
- 没有 accountant 时，不要把当前 `dpsgd` 结果写成正式 DP 结果。

推荐英文表述：

```text
We implement a FedLLM LAMP-lite adaptation of partial Transformer gradient leakage, where the attacker observes only a selected layer/module subset of gradients and optimizes dummy input embeddings to match the visible gradients.
```

推荐中文表述：

```text
我们将 partial Transformer gradient leakage 的核心机制适配到 FedLLM 设置中：攻击者只观察选定层或模块的梯度，并通过优化 dummy input embeddings 来匹配这些可见梯度。该实验用于补充验证 Projection-LRB 在部分梯度暴露场景下的鲁棒性，而不声称完整复现原文全部数值表格。
```

## 7. 最终建议

如果时间有限，优先顺序是：

1. DAGER partial `first2` 的 `none / proj_only@0.9 / topk@0.1 / compression@8 / full_lrb@0.5`。
2. 与这些 operating points 对应的 utility。
3. PTG `first2,qkv_only` 的 pilot。
4. Coverage baselines。
5. 更多 exposure 和 BERT source-selector 版本。

这样最符合当前证据强度：full-gradient DAGER 仍是主结果，partial-gradient 用来证明方法不是只在 full update 暴露下有效。
