# LRB 实验任务清单与运行手册

这份文档把前面讨论过的路线整理成一个可直接执行的实验 runbook，目标是把当前 `LRB` 从“在 DAGER 下表现很强”推进成“结论更稳、证据更完整、可以扩展到 PEFT 与其他场景”的主方法。

相关背景材料：

- [FL-LLM.md](./FL-LLM.md)
- [DEFENSE_BASELINES.md](./DEFENSE_BASELINES.md)
- [实验运行指南.md](./实验运行指南.md)
- [utils/lrb_defense.py](../utils/lrb_defense.py)
- [utils/defenses.py](../utils/defenses.py)
- [attack.py](../attack.py)
- [scripts/defense_baselines.sh](../scripts/defense_baselines.sh)
- [scripts/lora.sh](../scripts/lora.sh)

## 1. 当前判断

基于现有结果，可以先给出一个谨慎但明确的判断：

- 在当前 `DAGER + GPT2 + seq_class + batch_size=2` 框架下，`LRB` 已经处在最强一档。
- 相比 `noise / dpsgd / mixup / soteria`，`LRB` 明显更强。
- 相比 `topk / compression`，`LRB` 不能草率地说“全面碾压”，但至少已经是同一档的强方法。
- 从研究叙事上看，`LRB` 比 `topk / compression` 更贴近 [FL-LLM.md](./FL-LLM.md) 里“通用防御”的方向，因为它不是简单的稀疏化或量化，而是在做结构化的梯度表示约束。
- 当前 ablation 已经说明，`LRB` 的主效应大概率主要来自 `projection / bottleneck`，而不是单纯的 clipping。

因此，下一步不应该继续无节制扫参，而应该转向：

1. 锁定默认配置
2. 做 strongest baselines 公平对照
3. 做机制消融
4. 扩展到更多数据集、backbone、PEFT
5. 最后补 utility 与更一般威胁模型

## 2. 默认实验假设

这份 runbook 默认先围绕下面这组主设置推进：

- 主方法：`LRB`
- 主评测框架：`DAGER`
- 主数据集：`sst2`
- 主模型：`gpt2`
- 任务：`seq_class`
- batch size：`2`
- 第一阶段优先比较的 `LRB` 主参数：
  - `defense_lrb_keep_ratio_sensitive=0.2`
  - 备选：`defense_lrb_keep_ratio_sensitive=0.35`

当前 strongest baselines 的参考档位可先固定为：

- `topk=0.1`
- `compression=8` 或 `16`
- `mixup=0.3`
- `noise=5e-4`
- `dpsgd=5e-4`
- `soteria=30`

说明：

- `soteria` 在当前 GPT2 设置下表现很差，保留它主要是为了完整 baseline coverage，而不是因为它还有竞争力。
- `dpsgd` 已按论文语义修复后，计算更重，也更容易受显存限制影响。

## 3. 总实验任务清单

| ID | 任务 | 目标 | 当前是否可直接执行 | 主要产出 |
|---|---|---|---|---|
| A1 | 锁定 `LRB` 默认配置 | 从“看起来强”变成“固定主配置” | 是 | 主配置与备选配置 |
| A2 | strongest baselines 公平对照 | 验证 `LRB` 是否属于最强一档 | 是 | DAGER 对照总表 |
| A3 | `LRB` 组件消融 | 验证主效应到底来自哪里 | 是 | ablation 表 |
| A4 | 扩展到更多 dataset / backbone | 验证不是单一 setting 偶然结果 | 是 | 跨数据/模型结果 |
| B1 | 进入 PEFT / LoRA 场景 | 对齐 PEFT leakage 威胁模型 | 是 | `LRB` 在 PEFT 下的结果 |
| B2 | 测 partial-gradient / layer-level 泄露 | 对齐更一般的攻击面 | 否，需先补入口 | layer-level leakage 结果 |
| C1 | 做 utility 评估 | 证明不是只会“打断攻击” | 否，需先补训练时 hook | accuracy / F1 / loss 表 |
| C2 | 汇总结论与论文叙事 | 把阶段结论固化成可写文稿的结果 | 是 | 一段可进论文的结论 |

## 4. A1 锁定 LRB 默认配置

### 4.1 目标

从当前 sweep 中选出：

- 一个主配置
- 一个备选配置

建议第一轮只比较两个关键值：

- `keep_ratio_sensitive=0.2`
- `keep_ratio_sensitive=0.35`

### 4.2 推荐默认完整参数

如果需要先固定一版默认配置，建议从下面这组开始：

```bash
--defense lrb \
--defense_lrb_sensitive_n_layers 2 \
--defense_lrb_keep_ratio_sensitive 0.2 \
--defense_lrb_keep_ratio_other 0.75 \
--defense_lrb_clip_scale_sensitive 0.5 \
--defense_lrb_clip_scale_other 1.0 \
--defense_lrb_noise_sensitive 0.03 \
--defense_lrb_noise_other 0.005 \
--defense_lrb_empirical_weight 0.6 \
--defense_lrb_projection signed_pool
```

### 4.3 命令

#### A1-1 主配置候选：0.2

```bash
bash scripts/defense_baselines.sh sst2 2 gpt2 100 \
  --baseline_defense lrb \
  --baseline_param 0.2 \
  --finetuned_path ./models/gpt2-ft-rt
```

#### A1-2 备选配置候选：0.35

```bash
bash scripts/defense_baselines.sh sst2 2 gpt2 100 \
  --baseline_defense lrb \
  --baseline_param 0.35 \
  --finetuned_path ./models/gpt2-ft-rt
```

### 4.4 预期产出

- 一个主配置，例如 `LRB@0.2`
- 一个备选配置，例如 `LRB@0.35`
- 对应的关键指标：
  - `rec_token_mean`
  - `agg_rouge1_fm`
  - `agg_rouge2_fm`
  - `last_total_time`
  - `last_rec_status`

### 4.5 成功标准

- `LRB` 结果稳定处于最强一档
- 运行时间没有异常爆炸
- 效果不依赖非常极端的参数点

## 5. A2 strongest baselines 公平对照

### 5.1 目标

用每个 baseline 当前最强的一档参数，与 `LRB` 做公平比较，而不是再做大范围 sweep。

### 5.2 命令

#### A2-1 `LRB`

```bash
bash scripts/defense_baselines.sh sst2 2 gpt2 100 \
  --baseline_defense lrb \
  --baseline_param 0.2 \
  --finetuned_path ./models/gpt2-ft-rt
```

#### A2-2 `topk`

```bash
bash scripts/defense_baselines.sh sst2 2 gpt2 100 \
  --baseline_defense topk \
  --baseline_param 0.1 \
  --finetuned_path ./models/gpt2-ft-rt
```

#### A2-3 `compression`

```bash
bash scripts/defense_baselines.sh sst2 2 gpt2 100 \
  --baseline_defense compression \
  --baseline_param 8 \
  --finetuned_path ./models/gpt2-ft-rt
```

如果 `compression=16` 之前也很强，可以补一条：

```bash
bash scripts/defense_baselines.sh sst2 2 gpt2 100 \
  --baseline_defense compression \
  --baseline_param 16 \
  --finetuned_path ./models/gpt2-ft-rt
```

#### A2-4 `mixup`

```bash
bash scripts/defense_baselines.sh sst2 2 gpt2 100 \
  --baseline_defense mixup \
  --baseline_param 0.3 \
  --finetuned_path ./models/gpt2-ft-rt
```

#### A2-5 `noise`

```bash
bash scripts/defense_baselines.sh sst2 2 gpt2 100 \
  --baseline_defense noise \
  --baseline_param 5e-4 \
  --finetuned_path ./models/gpt2-ft-rt
```

#### A2-6 `dpsgd`

```bash
bash scripts/defense_baselines.sh sst2 2 gpt2 100 \
  --baseline_defense dpsgd \
  --baseline_param 5e-4 \
  --finetuned_path ./models/gpt2-ft-rt
```

说明：

- 现在的 `dpsgd` 是论文对齐的逐样本裁剪版本，不是旧的 post-hoc noisy clipping。
- 如果显存紧张，可先把 `n_inputs` 降到 `20` 做 smoke test，再跑正式版。

#### A2-7 `soteria`

```bash
bash scripts/defense_baselines.sh sst2 2 gpt2 100 \
  --baseline_defense soteria \
  --baseline_param 30 \
  --finetuned_path ./models/gpt2-ft-rt
```

### 5.3 汇总命令

```bash
python scripts/collect_experiment_logs.py log/runs \
  -o log/runs/dager_stageA_results.csv \
  --markdown log/runs/dager_stageA_results.md
```

### 5.4 预期产出

一张 strongest baselines 总表，至少包含：

- `defense`
- `best param`
- `rec_token_mean`
- `agg_rouge1_fm`
- `agg_rouge2_fm`
- `last_total_time`

### 5.5 预期结论

- `LRB / topk / compression` 属于当前最强一档
- `LRB` 更适合作为主方法
- `mixup / noise / dpsgd` 明显弱一档
- `soteria` 在当前 GPT2 设置下几乎是失败 baseline

## 6. A3 LRB 组件消融

### 6.1 目标

验证 `LRB` 不是“看上去生效其实没走到”，并进一步确认主效应来自哪一部分。

### 6.2 命令

#### A3-1 identity-LRB

预期应当与 `none` 基本一致，用来证明分支确实被调用了。

```bash
python attack.py --dataset sst2 --split val --n_inputs 100 --batch_size 2 \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 \
  --device cuda --task seq_class --cache_dir ./models_cache \
  --finetuned_path ./models/gpt2-ft-rt \
  --defense lrb \
  --defense_lrb_sensitive_n_layers 2 \
  --defense_lrb_keep_ratio_sensitive 1.0 \
  --defense_lrb_keep_ratio_other 1.0 \
  --defense_lrb_clip_scale_sensitive 1000000 \
  --defense_lrb_clip_scale_other 1000000 \
  --defense_lrb_noise_sensitive 0 \
  --defense_lrb_noise_other 0 \
  --defense_lrb_empirical_weight 0 \
  --defense_lrb_projection signed_pool \
  --log_file log/runs/lrb_identity.txt
```

#### A3-2 projection-only

```bash
python attack.py --dataset sst2 --split val --n_inputs 100 --batch_size 2 \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 \
  --device cuda --task seq_class --cache_dir ./models_cache \
  --finetuned_path ./models/gpt2-ft-rt \
  --defense lrb \
  --defense_lrb_sensitive_n_layers 2 \
  --defense_lrb_keep_ratio_sensitive 0.2 \
  --defense_lrb_keep_ratio_other 0.75 \
  --defense_lrb_clip_scale_sensitive 1000000 \
  --defense_lrb_clip_scale_other 1000000 \
  --defense_lrb_noise_sensitive 0 \
  --defense_lrb_noise_other 0 \
  --defense_lrb_empirical_weight 0.6 \
  --defense_lrb_projection signed_pool \
  --log_file log/runs/lrb_projection_only.txt
```

#### A3-3 clip-only

```bash
python attack.py --dataset sst2 --split val --n_inputs 100 --batch_size 2 \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 \
  --device cuda --task seq_class --cache_dir ./models_cache \
  --finetuned_path ./models/gpt2-ft-rt \
  --defense lrb \
  --defense_lrb_sensitive_n_layers 2 \
  --defense_lrb_keep_ratio_sensitive 1.0 \
  --defense_lrb_keep_ratio_other 1.0 \
  --defense_lrb_clip_scale_sensitive 0.5 \
  --defense_lrb_clip_scale_other 1.0 \
  --defense_lrb_noise_sensitive 0 \
  --defense_lrb_noise_other 0 \
  --defense_lrb_empirical_weight 0 \
  --defense_lrb_projection signed_pool \
  --log_file log/runs/lrb_clip_only.txt
```

#### A3-4 `signed_pool` vs `pool`

```bash
python attack.py --dataset sst2 --split val --n_inputs 100 --batch_size 2 \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 \
  --device cuda --task seq_class --cache_dir ./models_cache \
  --finetuned_path ./models/gpt2-ft-rt \
  --defense lrb \
  --defense_lrb_sensitive_n_layers 2 \
  --defense_lrb_keep_ratio_sensitive 0.2 \
  --defense_lrb_keep_ratio_other 0.75 \
  --defense_lrb_clip_scale_sensitive 0.5 \
  --defense_lrb_clip_scale_other 1.0 \
  --defense_lrb_noise_sensitive 0.03 \
  --defense_lrb_noise_other 0.005 \
  --defense_lrb_empirical_weight 0.6 \
  --defense_lrb_projection pool \
  --log_file log/runs/lrb_pool.txt
```

#### A3-5 `empirical_weight=0` vs `0.6`

```bash
python attack.py --dataset sst2 --split val --n_inputs 100 --batch_size 2 \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 \
  --device cuda --task seq_class --cache_dir ./models_cache \
  --finetuned_path ./models/gpt2-ft-rt \
  --defense lrb \
  --defense_lrb_sensitive_n_layers 2 \
  --defense_lrb_keep_ratio_sensitive 0.2 \
  --defense_lrb_keep_ratio_other 0.75 \
  --defense_lrb_clip_scale_sensitive 0.5 \
  --defense_lrb_clip_scale_other 1.0 \
  --defense_lrb_noise_sensitive 0.03 \
  --defense_lrb_noise_other 0.005 \
  --defense_lrb_empirical_weight 0 \
  --defense_lrb_projection signed_pool \
  --log_file log/runs/lrb_emp0.txt
```

### 6.3 预期产出

一张 ablation 表，至少包含：

- `variant`
- `rec_token_mean`
- `agg_rouge1_fm`
- `agg_rouge2_fm`
- `last_total_time`

### 6.4 预期结论

- `identity-LRB` 约等于 `none`
- `projection-only` 已足以大幅压制 DAGER
- `clip-only` 效果明显弱
- 当前 `LRB` 的关键机制主要是 projection / bottleneck

## 7. A4 扩展到更多 dataset / backbone

### 7.1 目标

验证 `LRB` 的结果不是某一个数据集、某一个 backbone、某一个 prompt 格式下的偶然现象。

### 7.2 命令

#### A4-1 GPT2 + CoLA

```bash
bash scripts/defense_baselines.sh cola 2 gpt2 100 \
  --baseline_defense lrb \
  --baseline_param 0.2 \
  --finetuned_path ./models/gpt2-ft-rt
```

#### A4-2 GPT2 + RTE

```bash
bash scripts/defense_baselines.sh rte 2 gpt2 100 \
  --baseline_defense lrb \
  --baseline_param 0.2 \
  --finetuned_path ./models/gpt2-ft-rt
```

#### A4-3 BERT + SST2

```bash
bash scripts/defense_baselines.sh sst2 2 bert-base-uncased 100 \
  --baseline_defense lrb \
  --baseline_param 0.2 \
  --finetuned_path ./models/bert-base-uncased-ft-rt
```

### 7.3 预期产出

- 一个跨 dataset / backbone 的稳定性结果表
- 至少能回答“`LRB` 是否只在 GPT2+SST2 上有效”

### 7.4 成功标准

- `LRB` 不要求每个设置都第一，但应保持在强方法一档
- 排名不要出现灾难性崩溃

## 8. B1 进入 PEFT / LoRA 场景

### 8.1 目标

把结论从 full fine-tuning 下的完整梯度泄露，推进到 `PEFT / LoRA` 场景，因为这更接近实际 LLM 微调使用习惯。

### 8.2 命令

#### B1-1 LoRA + none

```bash
bash scripts/lora.sh sst2 2 gpt2 100 none 0
```

#### B1-2 LoRA + LRB

如果 `scripts/lora.sh` 已支持透传当前 defense 参数，可以直接跑：

```bash
bash scripts/lora.sh sst2 2 gpt2 100 lrb 0.2
```

如果 `scripts/lora.sh` 还没有透传全部 `LRB` 细参数，则建议直接改成显式命令，确保下面这些参数能传进去：

```bash
python attack.py --dataset sst2 --split val --n_inputs 100 --batch_size 2 \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 \
  --device cuda --task seq_class --cache_dir ./models_cache \
  --train_method lora \
  --finetuned_path ./models/gpt2-lora-ft-rt \
  --defense lrb \
  --defense_lrb_sensitive_n_layers 2 \
  --defense_lrb_keep_ratio_sensitive 0.2 \
  --defense_lrb_keep_ratio_other 0.75 \
  --defense_lrb_clip_scale_sensitive 0.5 \
  --defense_lrb_clip_scale_other 1.0 \
  --defense_lrb_noise_sensitive 0.03 \
  --defense_lrb_noise_other 0.005 \
  --defense_lrb_empirical_weight 0.6 \
  --defense_lrb_projection signed_pool
```

#### B1-3 LoRA + strongest baseline 对照

建议至少补：

```bash
bash scripts/lora.sh sst2 2 gpt2 100 topk 0.1
bash scripts/lora.sh sst2 2 gpt2 100 compression 8
```

### 8.3 预期产出

- 一张 `PEFT / LoRA` 场景下的 strongest baselines 对照表
- 一个关键判断：`LRB` 是否从 full-gradient setting 平滑迁移到 adapter-only gradient setting

### 8.4 成功标准

- `LRB` 在 LoRA 下依旧保持竞争力
- 如果排名变化，也能解释是因为攻击面改变，而不是方法彻底失效

## 9. B2 partial-gradient / layer-level 场景

### 9.1 当前状态

这一部分目前还不能直接按现有脚本完整跑通，因为还缺少显式的“只暴露部分层梯度”或“只暴露某个子模块梯度”的攻击入口。

### 9.2 建议补的接口

建议未来补两个参数：

- `--gradient_layer_subset`
- `--gradient_param_filter`

目标是让攻击端能明确只拿到：

- 最后一层
- 最后 N 层
- classifier / lm_head
- adapter / LoRA 层

### 9.3 实现后的目标命令示例

```bash
python attack.py --dataset sst2 --split val --n_inputs 100 --batch_size 2 \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 \
  --device cuda --task seq_class --cache_dir ./models_cache \
  --finetuned_path ./models/gpt2-ft-rt \
  --gradient_layer_subset last2 \
  --defense lrb \
  --defense_lrb_keep_ratio_sensitive 0.2
```

### 9.4 预期产出

- 一张 layer-level leakage 对照表
- 一个更强的结论：`LRB` 不仅在完整梯度下有效，在部分梯度泄露下也仍有意义

## 10. C1 utility 评估

### 10.1 当前状态

现在大多数结果都还是“attack-time defended gradient”视角，也就是主要在看它能不能阻断重构攻击。  
这对于安全论文是必要的，但不够，因为最终还需要回答：

- 它会不会严重伤害任务性能
- 它能不能在训练过程中稳定使用
- 它的代价是否可接受

### 10.2 当前可以直接做的 clean training 对照

先记录没有 `LRB` 的 clean performance：

```bash
python train.py --dataset sst2 --model_path gpt2 --task seq_class \
  --train_method full --epochs 3 --batch_size 8 \
  --output_dir ./models/gpt2-ft-clean
```

### 10.3 训练时 LRB hook 完成后的目标命令

等训练阶段支持 `LRB` 后，再做：

```bash
python train.py --dataset sst2 --model_path gpt2 --task seq_class \
  --train_method full --epochs 3 --batch_size 8 \
  --defense lrb \
  --defense_lrb_sensitive_n_layers 2 \
  --defense_lrb_keep_ratio_sensitive 0.2 \
  --defense_lrb_keep_ratio_other 0.75 \
  --defense_lrb_clip_scale_sensitive 0.5 \
  --defense_lrb_clip_scale_other 1.0 \
  --defense_lrb_noise_sensitive 0.03 \
  --defense_lrb_noise_other 0.005 \
  --defense_lrb_empirical_weight 0.6 \
  --defense_lrb_projection signed_pool \
  --output_dir ./models/gpt2-ft-lrb
```

### 10.4 预期产出

- 一个 utility 表：
  - accuracy
  - F1
  - train loss
  - train time
  - attack resistance

### 10.5 成功标准

- 性能下降可接受
- 训练开销没有大到完全不可用
- 能形成“privacy-utility tradeoff”叙事

## 11. C2 结果汇总与阶段性写作

### 11.1 汇总命令

```bash
python scripts/collect_experiment_logs.py log/runs \
  -o log/runs/final_lrb_summary.csv \
  --markdown log/runs/final_lrb_summary.md
```

### 11.2 建议最终固定的 3 张表

#### 表 1：DAGER strongest baselines

| defense | best param | rec_token_mean | rouge1 | rouge2 | runtime |
|---|---:|---:|---:|---:|---:|
| none | - |  |  |  |  |
| noise | 5e-4 |  |  |  |  |
| dpsgd | 5e-4 |  |  |  |  |
| mixup | 0.3 |  |  |  |  |
| soteria | 30 |  |  |  |  |
| topk | 0.1 |  |  |  |  |
| compression | 8/16 |  |  |  |  |
| lrb | 0.2 |  |  |  |  |

#### 表 2：LRB ablations

| variant | rec_token_mean | rouge1 | rouge2 | runtime |
|---|---:|---:|---:|---:|
| none |  |  |  |  |
| identity-lrb |  |  |  |  |
| projection-only |  |  |  |  |
| clip-only |  |  |  |  |
| pool |  |  |  |  |
| signed_pool |  |  |  |  |
| empirical_weight=0 |  |  |  |  |
| empirical_weight=0.6 |  |  |  |  |

#### 表 3：transfer results

| setting | defense | rec_token_mean | rouge1 | rouge2 | note |
|---|---|---:|---:|---:|---|
| GPT2 + SST2 | lrb |  |  |  | main |
| GPT2 + CoLA | lrb |  |  |  | dataset transfer |
| GPT2 + RTE | lrb |  |  |  | dataset transfer |
| BERT + SST2 | lrb |  |  |  | backbone transfer |
| GPT2 + SST2 + LoRA | lrb |  |  |  | PEFT transfer |

### 11.3 预期可写进文稿的结论

可以把现阶段结论写成下面这种口径：

`LRB` 在当前 DAGER 攻击框架下稳定进入最强防御方法一档。与经典随机扰动类基线相比，它表现出明显更强的重构抑制能力；与 `topk` 和 `compression` 相比，它至少达到了同档竞争力，同时提供了更贴近“结构化通用防御”的方法叙事。消融结果进一步表明，`LRB` 的核心收益主要来自对敏感梯度子空间的投影与瓶颈约束，而不是单纯的裁剪。下一步需要在 PEFT、partial-gradient 和训练时 utility 三个维度继续补全，以把“在 DAGER 上效果强”推进为“更一般威胁模型下也成立”的结论。`

## 12. 预期产出总表

| 阶段 | 你最终应该拿到什么 |
|---|---|
| A1 | `LRB` 主配置与备选配置 |
| A2 | strongest baselines 总表 |
| A3 | `LRB` ablation 总表 |
| A4 | dataset / backbone transfer 表 |
| B1 | PEFT / LoRA 对照表 |
| B2 | partial-gradient leakage 表 |
| C1 | utility 表 |
| C2 | 一段可进入论文或汇报文稿的阶段性结论 |

## 13. 推荐执行顺序

建议按下面顺序推进：

1. 先做 A1，锁定主配置，不再来回扫很多参数。
2. 再做 A2，拿 strongest baselines 公平对照。
3. 然后做 A3，补足机制解释，不然方法叙事不够稳。
4. 接着做 A4 和 B1，证明不是单点 setting 有效。
5. 再补 B2，让威胁模型更完整。
6. 最后做 C1，把 utility 和 privacy tradeoff 补齐。
7. 做完后统一汇总到 C2，用表格和一段正式结论收口。

## 14. 当前最重要的现实提醒

有三点最好始终记住：

1. 目前可以较有把握地说，`LRB` 在当前 DAGER 框架下比大多数经典 baseline 强，而且已经是最强一档。
2. 目前还不能不加限定地说，`LRB` 在所有场景下都“远超全部 baseline”，因为 `topk / compression` 在当前框架里也很强。
3. 现在最值得投入时间的，不是再做很多小 sweep，而是补“泛化性证据”和“utility 证据”。
