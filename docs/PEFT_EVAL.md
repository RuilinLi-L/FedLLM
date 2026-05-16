# PEFT / LoRA Evaluation And Training Notes

> Current training-defense status: LoRA/IA3 PEFT training supports
> `none / noise / topk / compression / lrb / lrbprojonly` plus
> `dpsgd / soteria / mixup / dager`. `dpsgd` is DP-SGD-style clipping plus
> Gaussian noise without a privacy accountant; `soteria` is a
> representation-masking style baseline; `mixup` is a manifold MixUp-style
> baseline and falls back to ordinary gradients when `batch_size < 2`.
> Prefix PEFT training still supports only post-gradient defenses; prefix
> direct-generation and DAGER training defenses remain unsupported. PEFT
> adapter-only gradients usually do not include position-embedding tensors, so
> `defense_dager_offset_embedding` is typically a no-op for LoRA/IA3 training
> unless position embeddings are trainable/shared.

本文档说明当前仓库中 PEFT 路线的定位、支持范围、训练/评测入口和实验矩阵。当前版本是 v1：主线仍是 Projection-LRB，但已经把 LoRA-only 入口扩展为通用 PEFT 入口。

## 1. 当前定位

当前框架支持：

- PEFT DAGER eval：`lora / ia3`
- PEFT training：`lora / ia3 / prefix`，支持训练期 post-gradient `none / noise / topk / compression / lrb / lrbprojonly`；LoRA/IA3 额外支持训练期 `dpsgd / soteria / mixup / dager`
- BERT PEFT：`bert-base-uncased` 可用于 LoRA/IA3/Prefix 的 seq_class 路线
- GPT-2 LoRA 主线保持兼容，Llama LoRA 主线保持兼容
- representation-side bottleneck v1：在 seq_class 的 classifier-input representation 上执行 `mask / dropout / projection`
- legacy LoRA：旧的 `--train_method lora` 会自动映射到 `--train_method peft --peft_method lora`

当前暂不支持：

- 完整 Houlsby-style Adapter。`--peft_method adapter` 会明确报错：`adapter is planned but not enabled in v1`
- Prefix 训练期 DP-SGD-style / Soteria-style / MixUp-style / DAGER defense。LoRA/IA3 已接通训练期 baseline，但 `dpsgd` 没有 privacy accountant，不声明 formal DP guarantee
- Prefix DAGER span eval。Prefix 可以训练/smoke，但当前 DAGER eval 入口会拒绝 `--peft_method prefix`
- Llama Prefix。v1 只把 Prefix smoke 路线放在 BERT/GPT-2

## 2. CLI 语义

新入口：

```bash
--train_method peft
--peft_method lora|ia3|prefix
```

兼容入口：

```bash
--train_method lora
```

会自动等价为：

```bash
--train_method peft --peft_method lora
```

DAGER PEFT eval 入口当前只支持：

```bash
--train_method peft
--peft_method lora|ia3
```

LoRA 仍需要：

```bash
--lora_r 16
```

Prefix 可选：

```bash
--peft_num_virtual_tokens 20
```

representation bottleneck：

```bash
--defense_rep_bottleneck none|mask|dropout|projection
--defense_rep_keep_ratio 0.5
--defense_rep_dropout_p 0.1
```

## 3. PEFT 默认模块

LoRA 默认 target modules：

| Model | Default |
|---|---|
| `gpt2`, `openai-community/gpt2-large` | `c_attn` |
| `bert-base-uncased` | `query,value` |
| Llama family | `q_proj` |

IA3 默认 target/feedforward modules：

| Model | Target modules | Feedforward modules |
|---|---|---|
| GPT-2 | `c_attn,c_fc` | `c_fc` |
| BERT | `query,value,intermediate.dense` | `intermediate.dense` |
| Llama | `q_proj,v_proj,down_proj` | `down_proj` |

Prefix 默认：

| Model | Default |
|---|---|
| GPT-2 | `num_virtual_tokens=20` |
| BERT | `num_virtual_tokens=20` |
| Llama | v2 planned |

## 4. Training

BERT LoRA smoke：

```bash
python train.py \
  --dataset sst2 \
  --task seq_class \
  --model_path bert-base-uncased \
  --train_method peft \
  --peft_method lora \
  --lora_r 16 \
  --batch_size 2 \
  --num_epochs 1 \
  --models_cache ./models_cache \
  --output_dir ./models/bert_sst2_lora_r16
```

BERT IA3 smoke：

```bash
python train.py \
  --dataset sst2 \
  --task seq_class \
  --model_path bert-base-uncased \
  --train_method peft \
  --peft_method ia3 \
  --batch_size 2 \
  --num_epochs 1 \
  --models_cache ./models_cache \
  --output_dir ./models/bert_sst2_ia3
```

BERT Prefix smoke：

```bash
python train.py \
  --dataset sst2 \
  --task seq_class \
  --model_path bert-base-uncased \
  --train_method peft \
  --peft_method prefix \
  --peft_num_virtual_tokens 20 \
  --batch_size 2 \
  --num_epochs 1 \
  --models_cache ./models_cache \
  --output_dir ./models/bert_sst2_prefix
```

训练期 Projection-LRB：

```bash
python train.py \
  --dataset sst2 \
  --task seq_class \
  --model_path bert-base-uncased \
  --train_method peft \
  --peft_method lora \
  --lora_r 16 \
  --defense lrbprojonly \
  --defense_lrb_keep_ratio_sensitive 0.5 \
  --batch_size 2 \
  --num_epochs 1
```

representation-side bottleneck：

```bash
python train.py \
  --dataset sst2 \
  --task seq_class \
  --model_path bert-base-uncased \
  --train_method peft \
  --peft_method lora \
  --lora_r 16 \
  --defense none \
  --defense_rep_bottleneck projection \
  --defense_rep_keep_ratio 0.5 \
  --batch_size 2 \
  --num_epochs 1
```

脚本入口：

```bash
bash scripts/train_peft.sh sst2 2 bert-base-uncased lora --lora_r 16 --num_epochs 1
bash scripts/train_peft.sh sst2 2 bert-base-uncased ia3 --num_epochs 1
bash scripts/train_peft.sh sst2 2 bert-base-uncased prefix --peft_num_virtual_tokens 20 --num_epochs 1
```

## 5. Evaluation

单次 PEFT DAGER eval：

```bash
bash scripts/peft_eval.sh sst2 2 bert-base-uncased 1 \
  --peft_method lora \
  --finetuned_path ./models/bert_sst2_lora_r16/final_adapter \
  --defense none
```

`--peft_method` 省略时默认是 `lora`。DAGER PEFT eval 当前只支持 `lora / ia3`；`prefix` adapter 可以训练，但当前 span eval 会明确拒绝。PEFT adapter 目录会读取 `adapter_config.json`，并校验：

- `peft_type`
- `target_modules`
- `feedforward_modules`
- `num_virtual_tokens`
- `task_type`
- `base_model_name_or_path`

LoRA legacy `.pt/.pth` checkpoint 仍支持，但只适用于 LoRA，并且必须显式提供 `--lora_r`。

## 6. 推荐 v1 实验矩阵

最小矩阵：

```text
dataset: sst2
model: bert-base-uncased
peft_method: lora / ia3
```

每组建议跑：

```text
none
topk@0.1
compression@8
lrbprojonly@0.5
lrb full_lrb@0.5
rep_projection@0.5
rep_projection@0.5 + lrbprojonly@0.5
```

LoRA/GPT-2 主线继续保留，用来和已有 DAGER/Projection-LRB 结果对齐；BERT 用于证明 encoder PEFT 泛化。

## 7. 结果字段

训练和 eval summary 会记录：

- `train_method=peft`
- `peft_method`
- `peft_type`
- `peft_target_modules`
- `peft_feedforward_modules`
- `peft_num_virtual_tokens`
- `lora_r`
- `lora_target_modules`
- `rep_bottleneck_type`
- `rep_keep_ratio`
- `rep_dropout_p`
- `rep_bottleneck_with_lrb`

旧字段仍保留：

- `lora_checkpoint_type`
- `lora_adapter_r`
- `lora_adapter_target_modules`
- `lora_adapter_task_type`
- `lora_adapter_base_model`
- `lora_adapter_peft_type`

这样旧日志聚合不会坏，同时新 PEFT 方法不会被混进 LoRA-only 行。

## 8. 论文表述边界

建议论文中这样写：

- Projection-LRB 是主方法
- IA3/BERT 是 DAGER PEFT eval 泛化实验；Prefix 保留为训练/smoke 路线，当前不声明 DAGER span eval 结果
- representation-side bottleneck 是 forward-side ablation，不声明 formal DP
- LoRA/IA3 training-side `dpsgd / soteria / mixup` 是 baseline-style implementation，不声明 formal DP 或原论文完整复现
- PEFT adapter-only gradients usually do not include position-embedding tensors, so `defense_dager_offset_embedding` is typically a no-op for LoRA/IA3 training unless position embeddings are trainable/shared.
- Adapter 是 v2 planned，不作为当前实验结果声称
