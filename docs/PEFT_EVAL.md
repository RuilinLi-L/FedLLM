# PEFT / LoRA 框架说明

这份文档说明当前仓库里 **PEFT / LoRA 评测框架** 的定位、结构、支持范围和使用方式。  
如果你现在想回答两个问题：

- “这个框架现在已经做到哪一步了？”
- “我该怎么实际跑起来？”

优先看这份文档。

---

## 1. 当前框架是什么

当前这套框架的定位是：

- 一个 **PEFT / LoRA 场景下的攻击与防御评测框架**
- 不是完整的 PEFT 训练平台
- 也不是训练期防御已经全部接通的框架

更具体地说，当前版本已经支持：

- 加载已有的 LoRA checkpoint，包括 PEFT adapter 目录和兼容 `.pt/.pth`
- 在 `attack.py` 路径下对 LoRA 更新做攻击评测
- 在 LoRA 评测场景下比较 `none / noise / dpsgd / topk / compression / soteria / mixup / lrb`
  - 其中 `dpsgd / soteria / mixup` 是 attack-time eval baseline 的代码名，论文表述应分别写成 DP-SGD-style、Soteria-style、manifold MixUp-style。
- 在 LoRA 训练期使用 post-gradient `none / noise / topk / compression / lrb`
- 对结果做统一日志记录和结果汇总

当前版本**还不支持**：

- LoRA 训练期完整 `DP-SGD / Soteria / MixUp / DAGER defense`
- BERT 的 PEFT 路线
- Adapter / IA3 / Prefix Tuning 等其他 PEFT 方法

所以可以把它理解成：

> **LoRA eval-first 的最小可用研究框架**

---

## 2. 方法定位

这版框架的方法主线按 [`FL-LLM.md`](./FL-LLM.md) 组织。

### 2.1 `lrb` 在这个框架里的角色

`lrb` 不再只是“又一个 baseline”，而是当前项目的**主方法原型**。

文档叙事上，它对应：

- `LRB-v2`
- `post-gradient HLRB`
- `backward-side recoverability bottleneck` 的首版实现

也就是说，当前版本的重点不是“把梯度变乱”，而是：

- 尽量保留任务相关的低恢复性结构
- 优先破坏更容易泄露样本细节的方向
- 在攻击时共享更不容易被重构的更新

### 2.2 这版到底实现了 `HLRB` 的哪一部分

当前只落地了 `FL-LLM.md` 里的 **backward-side** 部分，也就是：

- attack-time / post-gradient defense

当前**没有**落地的部分：

- forward-side representation bottleneck
- LoRA 训练期 direct-generation defense（完整 DP-SGD / Soteria / MixUp / DAGER）
- 完整 calibration pipeline

研究演进位可以按下面理解：

- v1：`post-gradient HLRB` on PEFT eval and LoRA training
- v2：加入显式 layer-wise calibration profile
- v3：加入 representation-side bottleneck 和 direct-generation training defense

---

## 3. 框架结构

从“怎么跑起来”的角度看，现在这套 PEFT 框架主要有 4 层。

### 3.1 入口层

标准入口有两个：

- `scripts/peft_eval.sh`
- `scripts/peft_baselines.sh`

兼容入口保留：

- `scripts/lora.sh`

其中：

- `peft_eval.sh` 负责单次实验
- `peft_baselines.sh` 负责批量 baseline sweep
- `lora.sh` 只是历史兼容包装层，内部会转到新的 PEFT 入口

### 3.2 参数与校验层

LoRA 相关参数通过 `attack.py` 统一进入，然后在参数解析阶段做早期校验：

- `--train_method lora`
- `--finetuned_path`
- `--lora_r`
- `--lora_target_modules`
- `--defense`

当前的校验会提前拒绝：

- 不支持的模型族
- 缺失的 `--finetuned_path`
- legacy `.pt/.pth` checkpoint 缺失 `--lora_r`
- adapter 目录里的 rank / target_modules / task_type 与 CLI 不一致
- 非 PEFT adapter 目录、且不是 `.pt/.pth` 的 checkpoint
- LoRA 下不支持的 defense

这样错误会在启动早期暴露，而不是跑到中途才失败。

### 3.3 共享 PEFT 加载层

仓库现在已经有一层共享的 LoRA helper，统一负责：

- 判断模型族
- 决定 LoRA target modules
- 读取 PEFT adapter 目录中的 `adapter_config.json`
- 自动回填或校验 `lora_r`、`target_modules`、`task_type`、`base_model_name_or_path`
- 包装 Hugging Face `PeftModel` / `get_peft_model`
- 加载 PEFT adapter 目录或本地 `.pt/.pth` `state_dict`

默认 target modules 保持向后兼容：

- GPT-2 系：`c_attn`
- Llama 系：`q_proj`

也可以通过 `--lora_target_modules` 显式指定：

- GPT-2：`c_attn` / `all-linear`
- Llama：`q_proj` / `qv` / `qkvo` / `all-linear`

### 3.4 结果记录与汇总层

`attack.py` 和 `train.py` 现在的结果块会写入 PEFT 关键字段：

- `train_method`
- `lora_r`
- `lora_target_modules`
- `lora_checkpoint_type`
- `lora_adapter_r`
- `lora_adapter_target_modules`
- `lora_adapter_task_type`
- `lora_adapter_base_model`
- `lora_adapter_peft_type`

这样 `scripts/collect_experiment_logs.py` 在聚合结果时，可以把：

- `train_method=full`
- `train_method=lora`

稳定分开，避免 full fine-tuning 和 LoRA 结果混表。
同时，`scripts/collect_experiment_logs.py` 会把 `lora_target_modules` 放进聚合分组键，避免 `q_proj`、`qkvo`、`all-linear` 结果混在同一行。

---

## 4. 当前支持矩阵

### 4.1 模型族

当前支持：

- `gpt2`
- `openai-community/gpt2-large`
- 仓库当前支持的 `meta-llama/*` 模型

当前不支持：

- `bert-base-uncased` 的 PEFT
- 其他未接入的 backbone

### 4.2 PEFT 类型

当前只支持：

- LoRA

当前不支持：

- Adapter
- IA3
- Prefix Tuning
- Prompt Tuning

当前代码里已经有通用 PEFT 元数据 plumbing，但运行时只把 LoRA 作为可用路径；不要把其他 PEFT 方法写成已支持结果。

### 4.3 checkpoint 格式

当前支持：

- PEFT 原生 adapter 目录
  - 会读取 `adapter_config.json`
  - 可以自动推断 `lora_r` 和 `target_modules`
  - 如果 CLI 显式传入的 rank / target_modules / task_type 不一致，会 fail-fast
- 本地 `.pt`
- 本地 `.pth`
- 内容为当前仓库兼容格式的 LoRA `state_dict`
  - 仍必须显式传 `--lora_r`
  - `--lora_target_modules` 未传时使用模型族默认值

当前不支持：

- 非 PEFT adapter 的普通目录 checkpoint

### 4.4 LoRA 下的 defense

当前 eval 正式支持：

- `none`
- `noise`
- `dpsgd`
- `topk`
- `compression`
- `soteria`
- `mixup`
- `lrb`

论文与报告中的严谨命名：

| 代码名 | 推荐论文名 | 当前 LoRA eval 语义 | 不应声称 |
|---|---|---|---|
| `dpsgd` | DP-SGD-style clipping and Gaussian noise | 对当前 batch 的 LoRA 可训练参数做逐样本梯度、L2 裁剪、平均、加高斯噪声 | 完整 DP-SGD、训练期 DP、形式化 privacy guarantee、已报告 privacy cost |
| `soteria` | Soteria-style representation masking | 对 `seq_class` 的 classifier-input representation 打分并 mask，再重算 LoRA 可训练参数梯度 | 原始 Soteria 的严格复现、LoRA 训练期 Soteria 已实现 |
| `mixup` | Manifold MixUp-style representation interpolation | 混合 hidden/classifier-input representation 与标签，再生成 LoRA 可训练参数梯度 | 原始 input-level MixUp、训练期 MixUp defense 已实现 |

当前 eval 不支持：

- `dager`

当前 LoRA 训练期支持 post-gradient：

- `none`
- `noise`
- `topk`
- `compression`
- `lrb`

其中：

- 单次运行会直接 fail-fast
- `peft_baselines.sh` 会把不支持项记录为 `unsupported`

---

## 5. 从 LoRA 训练到 PEFT 实验

这一节是当前最推荐的可执行流程。核心原则是：

> 先用 `train.py` 训练并保存一个真实存在的 PEFT adapter 目录，再把这个目录传给 `peft_eval.sh` 或 `peft_baselines.sh`。兼容 `.pt/.pth` 仍然保留。

不要把文档里的示例占位符直接传给脚本。例如 `path/to/lora_checkpoint.pt` 不是有效路径，会触发：

```text
ValueError: LoRA checkpoint file does not exist: 'path/to/lora_checkpoint.pt'.
```

### 5.1 使用前准备

在仓库根目录执行：

```bash
conda env create -f environment.yml -n dager
conda activate dager
mkdir -p models models_cache log/runs
```

如果你跑 Llama，需要先配置 Hugging Face 权限：

```bash
export HF_TOKEN=your_huggingface_token
```

### 5.2 训练并保存 GPT-2 LoRA checkpoint

推荐先用 `SST2 + GPT2` 做最小闭环：

```bash
python train.py \
  --dataset sst2 \
  --task seq_class \
  --model_path gpt2 \
  --train_method lora \
  --lora_r 16 \
  --batch_size 2 \
  --num_epochs 1 \
  --models_cache ./models_cache \
  --output_dir ./models/gpt2_sst2_lora_r16
```

训练结束后，当前项目会保存：

```text
./models/gpt2_sst2_lora_r16/final_adapter/
./models/gpt2_sst2_lora_r16/final_adapter.pt
./models/gpt2_sst2_lora_r16/final_adapter_tokenizer/
```

后续 PEFT/LoRA eval 应优先使用这个 PEFT adapter 目录：

```bash
LORA_CKPT=./models/gpt2_sst2_lora_r16/final_adapter
```

注意：

- 使用 PEFT adapter 目录时，评测可从 `adapter_config.json` 自动读取 rank 和 target modules；显式传入 `--lora_r` 或 `--lora_target_modules` 时必须与 adapter 配置一致。
- 使用 legacy `.pt/.pth` 时，评测仍必须显式传 `--lora_r`；`--lora_target_modules` 未传时使用模型族默认值。
- `--model_path` 必须和训练 checkpoint 的 backbone 一致；上面训练是 `gpt2`，评测也要传 `gpt2`。
- 当前 eval 支持 PEFT adapter 目录，也继续支持本地 `.pt/.pth` `state_dict` 文件。
- 中途 `--save_every` 保存的 LoRA adapter 目录也可用于 eval，但论文主表建议使用最终的 `final_adapter`。

### 5.3 快速检查 checkpoint

正式跑实验前，可以先确认 adapter 目录存在且包含配置和权重：

```bash
test -d "$LORA_CKPT" && ls -lh "$LORA_CKPT"

python - <<'PY'
import torch
from pathlib import Path
path = Path("./models/gpt2_sst2_lora_r16/final_adapter")
print(path)
print("adapter_config =", (path / "adapter_config.json").is_file())
print("adapter_weights =", any((path / name).is_file() for name in ["adapter_model.bin", "adapter_model.safetensors"]))
PY
```

如果这里失败，先修 checkpoint 路径或训练流程，不要直接进入长实验。

### 5.4 单次 smoke test

先用 `n_inputs=1` 和 `defense=none` 验证加载链路：

```bash
bash scripts/peft_eval.sh sst2 2 gpt2 1 \
  --finetuned_path "$LORA_CKPT" \
  --defense none
```

这一步通过后，再扩大到 `20` 或 `100` 个样本。

---

## 6. 标准实验命令

### 6.1 Projection-LRB / LRB variant 对照

当前 LRB 消融后的主候选是 `proj_only@0.5`，但 LoRA/PEFT 下不能直接假设它必然最优，所以建议同时跑：

```text
none
proj_only@0.5
proj_clip@0.5
full_lrb@0.5
```

命令：

```bash
bash scripts/peft_baselines.sh sst2 2 gpt2 100 \
  --finetuned_path "$LORA_CKPT" \
  --baseline_defense lrb \
  --lrb_variants proj_only,proj_clip,full_lrb \
  --lrb_main_k 0.5
```

`peft_baselines.sh` 在聚焦某个 baseline 时会自动带上 `none`，所以这条命令会同时产出 clean LoRA baseline 和三个 LRB variant。

### 6.2 强 baseline：topk

```bash
bash scripts/peft_baselines.sh sst2 2 gpt2 100 \
  --finetuned_path "$LORA_CKPT" \
  --baseline_defense topk \
  --baseline_param 0.1
```

### 6.3 强 baseline：compression

```bash
bash scripts/peft_baselines.sh sst2 2 gpt2 100 \
  --finetuned_path "$LORA_CKPT" \
  --baseline_defense compression \
  --baseline_param 8
```

### 6.4 direct-generation eval baselines：DP-SGD-style / Soteria-style / Manifold MixUp-style

这三类现在在 LoRA eval 路径中可跑，但含义是“攻击评测时生成或改写当前 batch 的 LoRA 共享梯度”，不是 LoRA 训练期 defense 已完整接通。尤其是 `dpsgd` 没有 privacy accountant，不能作为 formal DP result 报告。

论文建议写法：

> We include attack-time LoRA baselines adapted from standard defense ideas: DP-SGD-style per-example clipping with Gaussian noise, Soteria-style representation masking, and manifold MixUp-style representation interpolation. These baselines are used for leakage evaluation only and do not imply formal DP guarantees or complete training-time implementations.

```bash
bash scripts/peft_baselines.sh sst2 2 gpt2 100 \
  --finetuned_path "$LORA_CKPT" \
  --baseline_defense dpsgd \
  --baseline_param 1e-4

bash scripts/peft_baselines.sh sst2 2 gpt2 100 \
  --finetuned_path "$LORA_CKPT" \
  --baseline_defense soteria \
  --baseline_param 60

bash scripts/peft_baselines.sh sst2 2 gpt2 100 \
  --finetuned_path "$LORA_CKPT" \
  --baseline_defense mixup \
  --baseline_param 1.0
```

### 6.5 完整支持项 sweep

如果算力允许，可以不指定 `--baseline_defense`，让脚本跑完整 baseline sweep：

```bash
bash scripts/peft_baselines.sh sst2 2 gpt2 100 \
  --finetuned_path "$LORA_CKPT"
```

这会：

- 正常运行 LoRA eval 下支持的 `none / noise / dpsgd / topk / compression / soteria / mixup / lrb`
  - `dpsgd / soteria / mixup` 在结果表中应按上面的 style baseline 命名解释
- 将 LoRA eval 下仍未定义公平语义的 `dager` 记录为 `unsupported`
- 输出统一的 `summary.txt / results.csv / results.md`

### 6.6 单次评测入口

如果只想跑一个 defense 点，可以直接用 `peft_eval.sh`：

```bash
bash scripts/peft_eval.sh sst2 2 gpt2 20 \
  --finetuned_path "$LORA_CKPT" \
  --defense lrb \
  --defense_lrb_preset proj_only \
  --defense_lrb_keep_ratio_sensitive 0.5
```

### 6.7 Llama 路线

Llama 路线同样要求传入真实存在的 PEFT adapter 目录或 `.pt/.pth` LoRA checkpoint。示例：

```bash
LLAMA_LORA_CKPT=./models/llama_sst2_lora_r256/final_adapter

bash scripts/peft_eval.sh sst2 2 meta-llama/Meta-Llama-3.1-8B 20 \
  --finetuned_path "$LLAMA_LORA_CKPT" \
  --defense compression \
  --defense_n_bits 8 \
  --pad left
```

Llama 通常还需要：

- `HF_TOKEN`
- 适当的 `--pad left`
- 更谨慎的 `rank_tol / span_thresh`
- 更多显存和更长运行时间

### 6.8 历史兼容入口

```bash
bash scripts/lora.sh sst2 2
```

这个入口仍然保留，但语义是固定的：

- 模型默认是 `meta-llama/Meta-Llama-3.1-8B`
- `n_inputs=100`
- `--finetuned_path ./models/lora_8530.pt`
- `--lora_r 256`

如果你要做新实验，优先用 `train.py` 准备 checkpoint，再用 `peft_eval.sh` 或 `peft_baselines.sh` 跑评测。

---

## 7. 参数说明

### 7.1 `peft_eval.sh` 的位置参数

```bash
scripts/peft_eval.sh DATASET BATCH_SIZE MODEL_PATH N_INPUTS [额外参数...]
```

含义：

| 位置 | 含义 |
|---|---|
| `$1` | `DATASET` |
| `$2` | `BATCH_SIZE` |
| `$3` | `MODEL_PATH` |
| `$4` | `N_INPUTS` |
| `$5+` | 透传给 `attack.py` 的额外参数 |

### 7.2 必需额外参数

LoRA 路线下必须给：

- `--finetuned_path PATH`

如果 `PATH` 是 PEFT adapter 目录，`--lora_r` 和 `--lora_target_modules` 可由 `adapter_config.json` 自动推断。
如果 `PATH` 是 legacy `.pt/.pth`，必须额外给：

- `--lora_r R`

可选但建议记录清楚：

- `--lora_target_modules PRESET`

脚本内部会自动补上：

- `--train_method lora`

### 7.3 推荐常用参数

常见会一起用的参数有：

- `--defense lrb`
- `--defense_topk_ratio`
- `--defense_n_bits`
- `--defense_noise`
- `--lora_target_modules`
- `--rank_tol`
- `--l1_span_thresh`
- `--l2_span_thresh`
- `--pad left`

---

## 8. 输出什么结果

### 8.1 单次运行

单次运行的日志末尾会有统一结果块：

```text
===== RESULT SUMMARY START =====
...
===== RESULT SUMMARY END =====
```

现在这个结果块里会包含：

- `train_method`
- `lora_r`
- `lora_target_modules`
- `lora_checkpoint_type`
- `defense`
- `defense_param_name`
- `defense_param_value`
- `rec_token_mean`
- `agg_rouge1_fm`
- `agg_rouge2_fm`
- `last_total_time`
- `result_status`

### 8.2 baseline sweep

`peft_baselines.sh` 会创建一个 run 目录，里面通常有：

- `_run_header.txt`
- `none.txt`
- `topk_0.1.txt`
- `lrb_0.2.txt`
- `summary.txt`
- `results.csv`
- `results.md`

其中：

- `summary.txt` 适合快速看整体对比
- `results.csv` 适合后续做表格和统计
- `results.md` 适合直接读

### 8.3 聚合时的关键区别

因为现在 summary 里写入了 `train_method` 和 `lora_r`，所以后续汇总时：

- full 和 LoRA 不会被混成一行
- 不同 LoRA rank 的结果也能区分
- 不同 LoRA target modules 的结果也会分开聚合

---

## 9. 常见报错与含义

### 9.1 缺少 `--finetuned_path`

含义：

- 你声明了 `--train_method lora`
- 但没有给 LoRA checkpoint

处理：

- 补上 PEFT adapter 目录或本地 `.pt/.pth` checkpoint 路径

### 9.2 legacy checkpoint 缺少 `--lora_r`

含义：

- 你传的是 legacy `.pt/.pth`
- 这类文件没有 `adapter_config.json`，框架无法自动知道 LoRA rank

处理：

- 补上 `--lora_r`
- 或改用 PEFT adapter 目录

### 9.3 模型族不支持

含义：

- 当前 LoRA eval 只支持 GPT-2 和 Llama

处理：

- 改用支持的模型
- 或者后续扩展 PEFT loader

### 9.4 defense 不支持

含义：

- 你在 LoRA eval 下用了当前还没接通的 defense，比如 `dager`

处理：

- eval 使用 `none / noise / dpsgd / topk / compression / soteria / mixup / lrb`
  - 写论文时将 `dpsgd / soteria / mixup` 分别解释为 DP-SGD-style、Soteria-style、manifold MixUp-style eval baseline
- 训练期 LoRA defense 仍只使用 post-gradient 路线：`none / noise / topk / compression / lrb`

### 9.5 checkpoint 格式不支持

含义：

- 给了目录
- 或给了非 `.pt/.pth`

处理：

- 改成 PEFT adapter 目录，或当前仓库可识别的本地 LoRA `.pt/.pth` `state_dict`

### 9.6 adapter 元数据冲突

含义：

- `adapter_config.json` 中的 `r`、`target_modules` 或 `task_type` 与 CLI 参数不一致

处理：

- 以 adapter 目录为准，删除冲突的 CLI 参数
- 或确认是否拿错了 checkpoint / backbone

---

## 10. 推荐使用流程

如果你现在是第一次用这个框架，建议按这个顺序：

1. 先训练或确认一个真实 checkpoint

```bash
python train.py \
  --dataset sst2 \
  --task seq_class \
  --model_path gpt2 \
  --train_method lora \
  --lora_r 16 \
  --batch_size 2 \
  --num_epochs 1 \
  --models_cache ./models_cache \
  --output_dir ./models/gpt2_sst2_lora_r16

LORA_CKPT=./models/gpt2_sst2_lora_r16/final_adapter
```

2. 再做最小 smoke test

```bash
bash scripts/peft_eval.sh sst2 2 gpt2 1 \
  --finetuned_path "$LORA_CKPT" \
  --lora_r 16 \
  --defense none
```

3. 再跑 LoRA/PEFT 主对照

```bash
bash scripts/peft_baselines.sh sst2 2 gpt2 100 \
  --finetuned_path "$LORA_CKPT" \
  --lora_r 16 \
  --baseline_defense lrb \
  --lrb_variants proj_only,proj_clip,full_lrb \
  --lrb_main_k 0.5
```

4. 最后再扩到 Llama 路线

这样能比较稳地定位问题：

- 是 checkpoint 有问题
- 是模型权限有问题
- 还是 defense / 参数组合有问题

---

## 11. 当前版本一句话总结

当前这个框架已经是：

> **一个面向 GPT-2 / Llama 的 LoRA 攻击与防御评测框架，支持已有 LoRA checkpoint 的统一加载、LoRA 场景下的 baseline 对比，以及按 `FL-LLM` 主线组织的 `lrb` 主方法评测。**

但它还不是：

> **完整的 PEFT 训练与训练期防御平台。**

如果你后面继续扩展，最自然的下一步是：

- 补 `representation-side bottleneck`
- 补训练期 defense
- 补更完整的 calibration pipeline
