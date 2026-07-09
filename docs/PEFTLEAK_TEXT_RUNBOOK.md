# PEFTLeak Text-Side 隐私-效用实验运行手册

本文档用于规划和执行 `sst2` 上的 PEFT leakage 文本侧实验。实验目标不是声称完整复现原始 image-side PEFTLeak，而是评估一个 **PEFTLeak-style text-side adaptation**。

核心定位：

- `adapter`：使用 malicious adapter probe + gradient-ratio recovery 的结构化 ratio attack。
- `lora` / `ia3`：使用 optimization-based PEFT gradient-matching attack 作为 PEFT leakage baseline。
- `oracle` routing 只用于 sanity check，不进入论文主表。
- 正式表格里每一个 privacy 点都必须有同模型、同 PEFT 方法、同 defense、同 defense 参数的 utility 点。

## 基础设定

建议从服务器仓库根目录运行：

```bash
cd /data/lrl/FedLLM
```

如果服务器路径不同，只需要改 `cd`。下面所有命令都使用仓库相对路径。

建议先开一个 `tmux` session，避免长实验断开：

```bash
tmux new -s peftleak_text
```

通用变量：

```bash
export DEVICE=cuda
export CACHE=./models_cache
export OUT=./outputs/peftleak_text_sst2

export EPOCHS=3
export N_PRIV=100
export B_PRIV=1
export PEFTLEAK_LOG_DIR="$OUT/privacy"

mkdir -p "$CACHE" "$OUT/models" "$OUT/privacy" "$OUT/utility" "$OUT/tables"
```

如果只想先做 pilot，把 `N_PRIV` 改小：

```bash
export N_PRIV=10
```

基础矩阵：

| 项目 | 设置 |
|---|---|
| 数据集 | `sst2` |
| 主 backbone | `gpt2` 默认；`bert-base-uncased` 可作为补充 |
| PEFT 方法 | `adapter`, `lora`, `ia3` |
| Adapter privacy attack | `--peftleak_attack_mode ratio` |
| LoRA/IA3 privacy attack | `--peftleak_attack_mode opt` |
| Ratio attack batch size | 第一版先用 `batch_size=1` |
| Privacy 样本数 | pilot: `2/10`; formal: `100` |
| Utility seeds | `101`, `202`, `303` |

## 当前实验定位

论文中建议这样写：

```text
We evaluate a PEFTLeak-style text-side adaptation based on malicious adapter
probes and gradient-ratio recovery, together with optimization-based LoRA/IA3
gradient-matching baselines.
```

不要写成：

```text
We fully reproduce the original PEFTLeak attack on text.
```

原因是原始 PEFTLeak 是 image-side ViT+adapter 攻击；这里是文本侧机制迁移，攻击核心对齐 gradient-ratio recovery，但模态、routing 和 backbone 都已经改变。

## 检查点路径约定

本文档默认会训练以下 clean checkpoints：

```text
GPT2_ADAPTER=$OUT/models/gpt2_adapter_clean/final_adapter
GPT2_LORA=$OUT/models/gpt2_lora_r16_clean/final_adapter
GPT2_IA3=$OUT/models/gpt2_ia3_clean/final_adapter

BERT_ADAPTER=$OUT/models/bert_adapter_clean/final_adapter  # optional
BERT_LORA=$OUT/models/bert_lora_r16_clean/final_adapter    # optional
BERT_IA3=$OUT/models/bert_ia3_clean/final_adapter          # optional
```

如果已经有训练好的 PEFT checkpoint，可以直接替换这些路径，跳过 clean checkpoint 训练。

## 冒烟测试

先跑语义测试，确认代码环境没有明显问题：

```bash
python test_peftleak_text_semantics.py
python test_peft_eval_semantics.py
```

再训练一个最小 GPT2 Adapter checkpoint：

```bash
python train.py \
  --dataset sst2 \
  --task seq_class \
  --batch_size 8 \
  --num_epochs 1 \
  --model_path gpt2 \
  --train_method peft \
  --peft_method adapter \
  --adapter_reduction_factor 16 \
  --defense none \
  --rng_seed 101 \
  --models_cache "$CACHE" \
  --output_dir "$OUT/models/gpt2_adapter_clean_smoke" \
  --log_file "$OUT/utility/gpt2_adapter_clean_smoke.txt"
```

Adapter ratio attack 冒烟：

```bash
PEFTLEAK_LOG_DIR="$OUT/privacy/smoke" bash scripts/peftleak_eval.sh sst2 1 gpt2 2 \
  --finetuned_path "$OUT/models/gpt2_adapter_clean_smoke/final_adapter" \
  --peft_method adapter \
  --peftleak_attack_mode ratio \
  --peftleak_ratio_route public_bins \
  --peftleak_ratio_bins 8 \
  --peftleak_ratio_public_n_inputs 16 \
  --defense none \
  --device "$DEVICE" \
  --cache_dir "$CACHE"
```

检查 summary，至少应该看到：

```text
result_status=ok
attack_variant=text_ratio
ratio_reportable=true
ratio_recovered_hidden_count > 0
ratio_collision_rate=...
rec_token_mean=...
ratio_rec_token_mean=...
```

如果这里失败，不要继续跑正式矩阵。

## 训练 Clean PEFT Checkpoints

### GPT2 Adapter

```bash
python train.py \
  --dataset sst2 \
  --task seq_class \
  --batch_size 8 \
  --num_epochs "$EPOCHS" \
  --model_path gpt2 \
  --train_method peft \
  --peft_method adapter \
  --adapter_reduction_factor 16 \
  --defense none \
  --rng_seed 101 \
  --models_cache "$CACHE" \
  --output_dir "$OUT/models/gpt2_adapter_clean" \
  --log_file "$OUT/utility/gpt2_adapter_clean.txt"
```

### GPT2 LoRA

```bash
python train.py \
  --dataset sst2 \
  --task seq_class \
  --batch_size 8 \
  --num_epochs "$EPOCHS" \
  --model_path gpt2 \
  --train_method peft \
  --peft_method lora \
  --lora_r 16 \
  --defense none \
  --rng_seed 101 \
  --models_cache "$CACHE" \
  --output_dir "$OUT/models/gpt2_lora_r16_clean" \
  --log_file "$OUT/utility/gpt2_lora_r16_clean.txt"
```

### GPT2 IA3

```bash
python train.py \
  --dataset sst2 \
  --task seq_class \
  --batch_size 8 \
  --num_epochs "$EPOCHS" \
  --model_path gpt2 \
  --train_method peft \
  --peft_method ia3 \
  --defense none \
  --rng_seed 101 \
  --models_cache "$CACHE" \
  --output_dir "$OUT/models/gpt2_ia3_clean" \
  --log_file "$OUT/utility/gpt2_ia3_clean.txt"
```

### BERT Adapter / LoRA / IA3

BERT optional commands mirror the GPT2 defaults; replace:

```text
--model_path bert-base-uncased
```

并把输出目录改成：

```text
$OUT/models/bert_adapter_clean
$OUT/models/bert_lora_r16_clean
$OUT/models/bert_ia3_clean
```

如果 GPT2 显存不够，先把训练 `--batch_size` 降到 `2` 或 `4`。

## Privacy 实验参数表

第一轮建议不要把所有弱 baseline 铺满，优先扫能进主表或 Pareto 图的点。

| baseline | privacy sweep 参数 |
|---|---|
| `none` | clean 对照 |
| `topk` | `0.01 0.05 0.1 0.3 0.5` |
| `compression` | `2 4 8 16` |
| `proj_only` | `0.5 0.65 0.75 0.9` |
| `proj_clip` | `0.5 0.65 0.75` |
| `full_lrb` | `0.5 0.65` |
| `noise` | `1e-5 1e-4 5e-4 1e-3` |
| `dpsgd` | `1e-5 1e-4 5e-4 1e-3` |
| ablation | `identity_lrb clip_only proj_rule_only proj_empirical_only proj_uniform proj_no_empirical` |

## Adapter Ratio Privacy

Adapter 的主攻击是 ratio：

```text
--peftleak_attack_mode ratio
--peftleak_ratio_route public_bins
```

### Clean / None

```bash
bash scripts/peftleak_eval.sh sst2 "$B_PRIV" gpt2 "$N_PRIV" \
  --finetuned_path "$OUT/models/gpt2_adapter_clean/final_adapter" \
  --peft_method adapter \
  --peftleak_attack_mode ratio \
  --peftleak_ratio_route public_bins \
  --peftleak_ratio_bins 8 \
  --peftleak_ratio_public_n_inputs 64 \
  --defense none \
  --device "$DEVICE" \
  --cache_dir "$CACHE"
```

### Top-k Sweep

```bash
for p in 0.01 0.05 0.1 0.3 0.5; do
bash scripts/peftleak_eval.sh sst2 "$B_PRIV" gpt2 "$N_PRIV" \
  --finetuned_path "$OUT/models/gpt2_adapter_clean/final_adapter" \
  --peft_method adapter \
  --peftleak_attack_mode ratio \
  --peftleak_ratio_route public_bins \
  --peftleak_ratio_public_n_inputs 64 \
  --defense topk \
  --defense_topk_ratio "$p" \
  --device "$DEVICE" \
  --cache_dir "$CACHE"
done
```

### Compression Sweep

```bash
for b in 2 4 8 16; do
bash scripts/peftleak_eval.sh sst2 "$B_PRIV" gpt2 "$N_PRIV" \
  --finetuned_path "$OUT/models/gpt2_adapter_clean/final_adapter" \
  --peft_method adapter \
  --peftleak_attack_mode ratio \
  --peftleak_ratio_route public_bins \
  --peftleak_ratio_public_n_inputs 64 \
  --defense compression \
  --defense_n_bits "$b" \
  --device "$DEVICE" \
  --cache_dir "$CACHE"
done
```

### Projection-LRB / `proj_only`

```bash
for k in 0.5 0.65 0.75 0.9; do
bash scripts/peftleak_eval.sh sst2 "$B_PRIV" gpt2 "$N_PRIV" \
  --finetuned_path "$OUT/models/gpt2_adapter_clean/final_adapter" \
  --peft_method adapter \
  --peftleak_attack_mode ratio \
  --peftleak_ratio_route public_bins \
  --peftleak_ratio_public_n_inputs 64 \
  --defense lrb \
  --defense_lrb_preset proj_only \
  --defense_lrb_keep_ratio_sensitive "$k" \
  --device "$DEVICE" \
  --cache_dir "$CACHE"
done
```

### `proj_clip` / `full_lrb`

```bash
for preset in proj_clip full_lrb; do
  for k in 0.5 0.65; do
  bash scripts/peftleak_eval.sh sst2 "$B_PRIV" gpt2 "$N_PRIV" \
    --finetuned_path "$OUT/models/gpt2_adapter_clean/final_adapter" \
    --peft_method adapter \
    --peftleak_attack_mode ratio \
    --peftleak_ratio_route public_bins \
    --peftleak_ratio_public_n_inputs 64 \
    --defense lrb \
    --defense_lrb_preset "$preset" \
    --defense_lrb_keep_ratio_sensitive "$k" \
    --device "$DEVICE" \
    --cache_dir "$CACHE"
  done
done
```

### Noise / DP-SGD-style

```bash
for sigma in 1e-5 1e-4 5e-4 1e-3; do
bash scripts/peftleak_eval.sh sst2 "$B_PRIV" gpt2 "$N_PRIV" \
  --finetuned_path "$OUT/models/gpt2_adapter_clean/final_adapter" \
  --peft_method adapter \
  --peftleak_attack_mode ratio \
  --peftleak_ratio_route public_bins \
  --peftleak_ratio_public_n_inputs 64 \
  --defense noise \
  --defense_noise "$sigma" \
  --device "$DEVICE" \
  --cache_dir "$CACHE"

bash scripts/peftleak_eval.sh sst2 "$B_PRIV" gpt2 "$N_PRIV" \
  --finetuned_path "$OUT/models/gpt2_adapter_clean/final_adapter" \
  --peft_method adapter \
  --peftleak_attack_mode ratio \
  --peftleak_ratio_route public_bins \
  --peftleak_ratio_public_n_inputs 64 \
  --defense dpsgd \
  --defense_noise "$sigma" \
  --defense_clip_norm 1.0 \
  --device "$DEVICE" \
  --cache_dir "$CACHE"
done
```

## LoRA / IA3 Privacy

LoRA 和 IA3 不报告 ratio 结构化攻击，正式结果使用 opt attack：

```text
--peftleak_attack_mode opt
--peftleak_steps 100
--peftleak_restarts 3
--peftleak_match_loss normalized_mse
```

### LoRA Clean / None

```bash
bash scripts/peftleak_eval.sh sst2 1 gpt2 "$N_PRIV" \
  --finetuned_path "$OUT/models/gpt2_lora_r16_clean/final_adapter" \
  --peft_method lora \
  --peftleak_attack_mode opt \
  --peftleak_steps 100 \
  --peftleak_restarts 3 \
  --peftleak_match_loss normalized_mse \
  --defense none \
  --device "$DEVICE" \
  --cache_dir "$CACHE"
```

### IA3 Clean / None

```bash
bash scripts/peftleak_eval.sh sst2 1 gpt2 "$N_PRIV" \
  --finetuned_path "$OUT/models/gpt2_ia3_clean/final_adapter" \
  --peft_method ia3 \
  --peftleak_attack_mode opt \
  --peftleak_steps 100 \
  --peftleak_restarts 3 \
  --peftleak_match_loss normalized_mse \
  --defense none \
  --device "$DEVICE" \
  --cache_dir "$CACHE"
```

LoRA/IA3 的其他 defense 点复用 Adapter 的 sweep 参数，只需要替换：

```text
--peft_method lora
--finetuned_path "$OUT/models/gpt2_lora_r16_clean/final_adapter"
```

或：

```text
--peft_method ia3
--finetuned_path "$OUT/models/gpt2_ia3_clean/final_adapter"
```

## Utility 实验命令

PEFT utility 建议直接用 `train.py --train_method peft`，不要用 full-gradient 默认的 `scripts/utility_baselines.sh` 作为主入口。

utility 不需要像 privacy 一样铺得非常密，第一轮优先跑可能进入主表和 Pareto 图的点。

推荐优先级：

| 角色 | defense 点 |
|---|---|
| clean reference | `none` |
| 主方法 | `proj_only@0.5`, `0.65`, `0.75`, `0.9` |
| 强 baseline | `topk@0.1`, `0.3`; `compression@4`, `8`, `16` |
| over-defense | `full_lrb@0.5` |
| coverage baseline | `dpsgd@5e-4`, `dpsgd@1e-3`, `noise@1e-3` |

### Adapter Utility: None

```bash
for seed in 101 202 303; do
python train.py \
  --dataset sst2 \
  --task seq_class \
  --batch_size 8 \
  --num_epochs "$EPOCHS" \
  --model_path gpt2 \
  --train_method peft \
  --peft_method adapter \
  --adapter_reduction_factor 16 \
  --defense none \
  --rng_seed "$seed" \
  --models_cache "$CACHE" \
  --output_dir "$OUT/models/utility_gpt2_adapter_none_seed${seed}" \
  --log_file "$OUT/utility/gpt2_adapter_none_seed${seed}.txt"
done
```

### Adapter Utility: Top-k

```bash
for p in 0.1 0.3; do
for seed in 101 202 303; do
python train.py \
  --dataset sst2 \
  --task seq_class \
  --batch_size 8 \
  --num_epochs "$EPOCHS" \
  --model_path gpt2 \
  --train_method peft \
  --peft_method adapter \
  --adapter_reduction_factor 16 \
  --defense topk \
  --defense_topk_ratio "$p" \
  --rng_seed "$seed" \
  --models_cache "$CACHE" \
  --output_dir "$OUT/models/utility_gpt2_adapter_topk_${p}_seed${seed}" \
  --log_file "$OUT/utility/gpt2_adapter_topk_${p}_seed${seed}.txt"
done
done
```

### Adapter Utility: Compression

```bash
for b in 4 8 16; do
for seed in 101 202 303; do
python train.py \
  --dataset sst2 \
  --task seq_class \
  --batch_size 8 \
  --num_epochs "$EPOCHS" \
  --model_path gpt2 \
  --train_method peft \
  --peft_method adapter \
  --adapter_reduction_factor 16 \
  --defense compression \
  --defense_n_bits "$b" \
  --rng_seed "$seed" \
  --models_cache "$CACHE" \
  --output_dir "$OUT/models/utility_gpt2_adapter_compression_${b}_seed${seed}" \
  --log_file "$OUT/utility/gpt2_adapter_compression_${b}_seed${seed}.txt"
done
done
```

### Adapter Utility: Projection-LRB

```bash
for k in 0.5 0.65 0.75 0.9; do
for seed in 101 202 303; do
python train.py \
  --dataset sst2 \
  --task seq_class \
  --batch_size 8 \
  --num_epochs "$EPOCHS" \
  --model_path gpt2 \
  --train_method peft \
  --peft_method adapter \
  --adapter_reduction_factor 16 \
  --defense lrb \
  --defense_lrb_preset proj_only \
  --defense_lrb_keep_ratio_sensitive "$k" \
  --rng_seed "$seed" \
  --models_cache "$CACHE" \
  --output_dir "$OUT/models/utility_gpt2_adapter_proj_only_${k}_seed${seed}" \
  --log_file "$OUT/utility/gpt2_adapter_proj_only_${k}_seed${seed}.txt"
done
done
```

### LoRA / IA3 Utility

LoRA utility 把 Adapter flags 替换成：

```text
--peft_method lora --lora_r 16
```

IA3 utility 替换成：

```text
--peft_method ia3
```

BERT utility 把：

```text
--model_path gpt2
```

替换成：

```text
--model_path bert-base-uncased
```

如果显存不够，降低 `--batch_size`。

## Projection-LRB 消融实验

消融建议至少覆盖：

```text
identity_lrb
clip_only
proj_only
proj_clip
full_lrb
proj_rule_only
proj_empirical_only
proj_uniform
proj_no_empirical
```

Adapter ratio privacy 消融：

```bash
for preset in identity_lrb clip_only proj_only proj_clip full_lrb proj_rule_only proj_empirical_only proj_uniform proj_no_empirical; do
bash scripts/peftleak_eval.sh sst2 "$B_PRIV" gpt2 "$N_PRIV" \
  --finetuned_path "$OUT/models/gpt2_adapter_clean/final_adapter" \
  --peft_method adapter \
  --peftleak_attack_mode ratio \
  --peftleak_ratio_route public_bins \
  --peftleak_ratio_public_n_inputs 64 \
  --defense lrb \
  --defense_lrb_preset "$preset" \
  --defense_lrb_keep_ratio_sensitive 0.5 \
  --device "$DEVICE" \
  --cache_dir "$CACHE"
done
```

对应 utility 每个 preset 跑 `101/202/303` 三个 seed。

## 结果汇总

所有 privacy 和 utility 跑完后：

```bash
python scripts/collect_experiment_logs.py "$OUT" \
  -o "$OUT/tables/raw_results.csv" \
  --markdown "$OUT/tables/raw_results.md" \
  --utility-output "$OUT/tables/utility_results.csv" \
  --utility-markdown "$OUT/tables/utility_results.md" \
  --tradeoff-output "$OUT/tables/privacy_utility_tradeoff.csv" \
  --tradeoff-markdown "$OUT/tables/privacy_utility_tradeoff.md"
```

重点看 privacy 字段：

```text
rec_token_mean
ratio_rec_token_mean
opt_rec_token_mean
agg_rouge1_fm
agg_rouge2_fm
agg_r1fm_r2fm
ratio_collision_rate
ratio_recovered_hidden_count
ratio_reportable
```

重点看 utility 字段：

```text
eval_accuracy
eval_macro_f1
eval_loss
final_train_loss
utility_drop
```

`privacy_utility_tradeoff.csv` 默认用 `rec_token_mean` join privacy。ratio 模式下 `rec_token_mean` 应该就是 ratio primary 结果；如果要单独比较 `opt` 和 `ratio`，看 raw CSV 里的 `opt_rec_token_mean` 和 `ratio_rec_token_mean`。

## 推荐论文主表

第一版主表不要铺太多点，建议：

| PEFT/Attack | 主表点 |
|---|---|
| Adapter ratio | `none`, `topk@0.1`, `compression@8`, `proj_only@0.5`, `proj_only@0.65`, `proj_only@0.75`, `full_lrb@0.5` |
| LoRA opt | `none`, `topk@0.1`, `compression@8`, `proj_only@0.5`, `full_lrb@0.5` |
| IA3 opt | `none`, `topk@0.1`, `compression@8`, `proj_only@0.5` |

消融表：

```text
identity_lrb
clip_only
proj_only
proj_clip
full_lrb
proj_rule_only
proj_empirical_only
proj_uniform
proj_no_empirical
```

## 推荐运行顺序

1. 先跑 `test_peftleak_text_semantics.py` 和 Adapter ratio smoke。
2. 训练 GPT2 Adapter clean checkpoint。
3. 跑 Adapter ratio privacy：`none`、`proj_only`、`topk`、`compression` 优先。
4. 看 `rec_token_mean`、`ratio_collision_rate`、`ratio_recovered_hidden_count`，确认 attack 不是空跑。
5. 补 Adapter utility：`none`、`proj_only@0.5/0.65/0.75/0.9`、`topk@0.1/0.3`、`compression@8/16`。
6. 跑 GPT2 LoRA/IA3 opt privacy 和 utility。
7. 如果 GPT2 结果稳定，再扩到 BERT。
8. 最后补消融和弱 baseline：`noise`、`dpsgd`、`soteria`、`mixup`。

## 最低验收标准

正式写进论文前，至少满足：

```text
1. test_peftleak_text_semantics.py 在服务器 torch 环境通过。
2. GPT2 Adapter ratio smoke 通过。
3. BERT Adapter ratio smoke 作为补充通过。
4. clean/none 在 reportable attack 下有非零恢复。
5. 每个 privacy 点都有匹配 utility 点。
6. 主表所有 ratio 行都有 ratio_reportable=true。
7. oracle routing 不进入主表。
8. DP-SGD-style 只写作 clipping + Gaussian noise baseline，不声明 formal DP。
```

简短结论：privacy 要先多扫，找到恢复边界；utility 不需要全铺满，但必须对所有可能进入主表和 Pareto 图的点做多 seed。
