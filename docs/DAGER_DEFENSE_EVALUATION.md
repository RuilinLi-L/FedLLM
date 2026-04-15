# DAGER防御方法评估指南

本文档介绍如何测试和评估新实现的DAGER防御方法的效果。

## 评估指标

DAGER攻击的评估主要使用以下指标：
- **ROUGE-1**: 恢复文本的单词级召回率
- **ROUGE-2**: 恢复文本的双词级召回率  
- **Exact Match**: 完全匹配的序列比例

防御效果越好，这些指标的值应该越低（接近0）。

## 测试步骤

### 1. 基准测试（无防御）

首先运行无防御的基准测试，了解DAGER在无防御情况下的攻击效果：

```bash
python attack.py --dataset sst2 --split val --n_inputs 10 --batch_size 2 \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 \
  --device cuda --task seq_class --cache_dir ./models_cache \
  --defense none
```

### 2. 测试DAGER防御方法

#### 2.1 测试动态基底扰动（默认启用）

```bash
python attack.py --dataset sst2 --split val --n_inputs 10 --batch_size 2 \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 \
  --device cuda --task seq_class --cache_dir ./models_cache \
  --defense dager --defense_dager_basis_perturb --defense_dager_basis_noise_scale 0.01
```

#### 2.2 测试梯度切片（省略前两层）

```bash
python attack.py --dataset sst2 --split val --n_inputs 10 --batch_size 2 \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 \
  --device cpu --task seq_class --cache_dir ./models_cache \
  --defense dager --defense_dager_gradient_slicing --defense_dager_slice_first_n 1
```

#### 2.3 测试组合防御

```bash
python attack.py --dataset sst2 --split val --n_inputs 10 --batch_size 2 \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 \
  --device cuda --task seq_class --cache_dir ./models_cache \
  --defense dager \
  --defense_dager_basis_perturb --defense_dager_basis_noise_scale 0.01 \
  --defense_dager_gradient_slicing --defense_dager_random_slice --defense_dager_slice_prob 0.5
```

#### 2.4 测试所有防御策略

```bash
python attack.py --dataset sst2 --split val --n_inputs 10 --batch_size 2 \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 \
  --device cuda --task seq_class --cache_dir ./models_cache \
  --defense dager \
  --defense_dager_basis_perturb --defense_dager_basis_noise_scale 0.01 \
  --defense_dager_offset_embedding --defense_dager_offset_scale 0.005 \
  --defense_dager_gradient_slicing --defense_dager_random_slice --defense_dager_slice_prob 0.3 \
  --defense_dager_rank_limit
```

### 3. 与其他防御方法比较

可以与现有的防御基线进行比较：

```bash
# DP-SGD
python attack.py --dataset sst2 --split val --n_inputs 10 --batch_size 2 \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 \
  --device cuda --task seq_class --cache_dir ./models_cache \
  --defense dpsgd --defense_noise 0.001 --defense_clip_norm 1.0

# Top-k
python attack.py --dataset sst2 --split val --n_inputs 10 --batch_size 2 \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 \
  --device cuda --task seq_class --cache_dir ./models_cache \
  --defense topk --defense_topk_ratio 0.1

# Soteria
python attack.py --dataset sst2 --split val --n_inputs 10 --batch_size 2 \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 \
  --device cuda --task seq_class --cache_dir ./models_cache \
  --defense soteria --defense_soteria_pruning_rate 60.0
```

## 结果解读

### 输出示例

攻击脚本的输出会包含类似以下的结果：

```
ROUGE-1: 0.95, ROUGE-2: 0.92, Exact Match: 0.80
```

### 结果分析

1. **无防御情况**：DAGER通常能达到很高的ROUGE分数（接近1.0）
2. **有效防御**：ROUGE分数应该显著降低
3. **防御效果评估**：
   - ROUGE-1 < 0.1：防御效果很好
   - ROUGE-1 < 0.3：防御效果中等
   - ROUGE-1 > 0.5：防御效果有限

### 参数调优建议

1. **噪声尺度 (`--defense_dager_basis_noise_scale`)**
   - 范围：0.001 - 0.1
   - 太小可能无效，太大会影响模型训练

2. **梯度切片概率 (`--defense_dager_slice_prob`)**
   - 范围：0.1 - 0.9
   - 值越小，防御越强，但对训练影响越大

3. **偏移尺度 (`--defense_dager_offset_scale`)**
   - 范围：0.001 - 0.05
   - 专门针对位置编码的防御

## 批量测试

可以使用现有的批量测试脚本：

```bash
# 修改 defense_baselines.sh 或创建新的测试脚本
# 添加DAGER防御的测试配置
```

## 注意事项

1. **计算资源**：DAGER防御会增加一些计算开销，但通常可以接受
2. **模型兼容性**：防御方法与GPT-2、BERT、LLaMA等主流模型兼容
3. **训练影响**：在评估防御效果时，也需要考虑对模型训练效果的影响
4. **随机性**：由于防御中包含随机成分，建议多次运行取平均值

## 预期效果

根据防御设计原理，DAGER防御方法应该能够：

1. **破坏Span Check**：通过动态基底扰动，使DAGER的token恢复失效
2. **干扰位置编码**：通过随机偏移嵌入，破坏位置信息的确定性
3. **切断恢复链**：通过梯度切片，阻止DAGER使用关键层的梯度
4. **打破低秩假设**：通过秩限制防御，破坏DAGER的理论基础

这些防御策略的组合应该能够显著降低DAGER攻击的成功率。