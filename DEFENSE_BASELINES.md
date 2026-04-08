# 防御 Baselines 测试说明

本文说明如何在 DAGER 攻击框架下批量或单独测试 `FL-LLM.md` 中列出的防御 baselines。

## 方式一：一键跑全部配置

已有脚本 `scripts/defense_baselines.sh`，会依次运行 `none` 与各防御；除 `none` 外，**每种防御会对一个强度参数做多档 sweep**（弱→强），便于观察隐私–效用曲线，而不是只用一档过强参数。

| 防御 | Sweep 参数 | 取值（示例） |
|------|------------|--------------|
| `none` | — | 单次 |
| `noise` / `dpsgd` | `--defense_noise` | 1e-6 … 1e-3 |
| `topk` | `--defense_topk_ratio` | 0.01 … 0.9 |
| `compression` | `--defense_n_bits` | 2 … 32 |
| `soteria` | `--defense_soteria_pruning_rate` | 10 … 90 |
| `mixup` | `--defense_mixup_alpha` | 0.1 … 2.0 |

日志目录下每个变体一个文件：`none.txt`、`noise_1e-6.txt`、`topk_0_9.txt` 等；`summary.txt` 末尾 **COMPARISON** 表含 `variant` 与 `param` 列。

**调参提示：** 若在很弱的 `topk` / `compression` 下 ROUGE 仍长期为 0，可能是精确 span 检验对结构化扰动过严；可再尝试放宽 `--l1_span_thresh`，或评估是否对这类防御启用 `attack.py` 中与噪声梯度配套的 outlier 解码路径（见 `uses_noisy_gradient_decoding`）。

```bash
# 基本用法（GPT-2, sst2, batch_size=2, 3 个样本）
bash scripts/defense_baselines.sh sst2 2 gpt2 100
```

### 参数说明

| 位置 | 含义 |
|------|------|
| `$1` | 数据集：`sst2`、`cola`、`rte`、`rotten_tomatoes` 等 |
| `$2` | `batch_size` |
| `$3` | `model_path`：`gpt2`、`bert-base-uncased`、`openai-community/gpt2-large` 等 |
| `$4` | `n_inputs`（测试样本数） |
| `$5+` | 额外传给 `attack.py` 的参数 |

### BERT 示例

```bash
bash scripts/defense_baselines.sh sst2 2 bert-base-uncased 3
```

---

## 方式二：单独测试某一种防御

直接调用 `attack.py`，通过 `--defense` 指定防御类型。

### 1. 无防御（基准）

```bash
python attack.py --dataset sst2 --split val --n_inputs 3 --batch_size 2 \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 \
  --device cuda --task seq_class --cache_dir ./models_cache \
  --defense none
```

### 2. Noise Injection

需同时指定 `--defense_noise`。

```bash
python attack.py --dataset sst2 --split val --n_inputs 10 --batch_size 2 \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 \
  --device cuda --task seq_class --cache_dir ./models_cache \
  --defense noise --defense_noise 0.001
```

### 3. DP-SGD

需 `--defense_noise`；可选 `--defense_clip_norm`。

```bash
python attack.py --dataset sst2 --split val --n_inputs 10 --batch_size 2 \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 \
  --device cuda --task seq_class --cache_dir ./models_cache \
  --defense dpsgd --defense_noise 0.001 --defense_clip_norm 1.0
```

### 4. Top-k Sparsification

可调 `--defense_topk_ratio`。

```bash
python attack.py --dataset cola --split val --n_inputs 10 --batch_size 1 \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 \
  --device cuda --task seq_class --cache_dir ./models_cache \
  --defense topk --defense_topk_ratio 0.1
```

### 5. Gradient Compression

可调 `--defense_n_bits`。

```bash
python attack.py --dataset sst2 --split val --n_inputs 10 --batch_size 1 \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 \
  --device cuda --task seq_class --cache_dir ./models_cache \
  --defense compression --defense_n_bits 8
```

### 6. Soteria

可调 `--defense_soteria_pruning_rate`；大模型建议加 `--defense_soteria_sample_dims` 以加速。

```bash
python attack.py --dataset sst2 --split val --n_inputs 3 --batch_size 2 \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 \
  --device cuda --task seq_class --cache_dir ./models_cache \
  --defense soteria --defense_soteria_pruning_rate 60.0
```

仅采样 256 个隐层维度打分的加速示例：

```bash
python attack.py --dataset sst2 --split val --n_inputs 3 --batch_size 2 \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 \
  --device cuda --task seq_class --cache_dir ./models_cache \
  --defense soteria --defense_soteria_pruning_rate 60.0 --defense_soteria_sample_dims 256
```

### 7. MixUp

可调 `--defense_mixup_alpha`；仅对 `--task seq_class` 有意义。

```bash
python attack.py --dataset sst2 --split val --n_inputs 3 --batch_size 2 \
  --l1_filter all --l2_filter non-overlap --model_path gpt2 \
  --device cuda --task seq_class --cache_dir ./models_cache \
  --defense mixup --defense_mixup_alpha 1.0
```

---

## 关键 CLI 参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--defense` | `none` | 防御类型 |
| `--defense_noise` | `None` | `noise` / `dpsgd` 的高斯噪声标准差 |
| `--defense_clip_norm` | `1.0` | `dpsgd` 的 L2 梯度裁剪范数（堆叠梯度整体裁剪） |
| `--defense_topk_ratio` | `0.1` | Top-k：每个张量按 \|g\| 保留的比例（0.1 即保留 10%） |
| `--defense_n_bits` | `8` | 压缩：均匀量化位数 |
| `--defense_soteria_pruning_rate` | `60.0` | Soteria：低于该百分位得分的列被剪枝 |
| `--defense_soteria_sample_dims` | `None` | Soteria：若设置则只随机采样这么多隐维打分 |
| `--defense_mixup_alpha` | `1.0` | MixUp：`Beta(alpha, alpha)` 混合系数分布 |

---

## 建议的测试顺序

1. **小规模冒烟**（确认流程与依赖无报错）：

   ```bash
   bash scripts/defense_baselines.sh sst2 2 gpt2 1
   ```

   `batch_size = 1` 时 MixUp 会回退到与普通 `compute_grads` 相同的梯度（不做嵌入混合）；要测真正的 MixUp，请使用 `batch_size >= 2`。

2. **正式对比实验**（例如 100 条）：

   ```bash
   bash scripts/defense_baselines.sh sst2 2 gpt2 100
   ```

3. **消融 / 调参**：单独调用 `attack.py`，修改 `--defense_noise`、`--defense_topk_ratio` 等即可。

---

## 注意事项

- `noise` 与 `dpsgd` 必须同时提供 `--defense_noise`，否则会报错。
- `mixup` 仅对 `--task seq_class` 生效；`next_token_pred` 会回退到普通 `compute_grads`。
- `soteria` 在 `train_method=lora` 下未集成，会抛出 `NotImplementedError`。
- 在 LLaMA 等大模型上跑 Soteria 时，建议加 `--defense_soteria_sample_dims 256`（或更小）以降低单次打分开销。

---

## 相关文件

- 防御实现与入口：`utils/defenses.py`（`apply_defense`）
- MixUp 梯度：`utils/models.py`（`compute_grads_mixup`）
- 参数定义：`args_factory.py`
- 攻击主脚本：`attack.py`（`reconstruct` 中调用 `apply_defense`）
- 背景与 baseline 列表：`FL-LLM.md`
