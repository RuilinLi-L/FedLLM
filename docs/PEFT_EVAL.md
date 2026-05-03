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

- 加载已有的 LoRA checkpoint
- 在 `attack.py` 路径下对 LoRA 更新做攻击评测
- 在 LoRA 场景下比较 `none / noise / topk / compression / lrb`
- 对结果做统一日志记录和结果汇总

当前版本**还不支持**：

- LoRA 训练期 `LRB / DP-SGD / Soteria / MixUp / DAGER defense`
- BERT 的 PEFT 路线
- Adapter / IA3 / Prefix Tuning 等其他 PEFT 方法
- PEFT 原生 adapter 目录格式加载

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
- 完整训练期 defense
- 完整 calibration pipeline

研究演进位可以按下面理解：

- v1：`post-gradient HLRB` on PEFT eval
- v2：加入显式 layer-wise calibration profile
- v3：加入 representation-side bottleneck 和训练期 defense

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
- `--defense`

当前的校验会提前拒绝：

- 不支持的模型族
- 缺失的 `--finetuned_path`
- 缺失的 `--lora_r`
- 非 `.pt/.pth` 的 checkpoint
- LoRA 下不支持的 defense

这样错误会在启动早期暴露，而不是跑到中途才失败。

### 3.3 共享 PEFT 加载层

仓库现在已经有一层共享的 LoRA helper，统一负责：

- 判断模型族
- 决定 LoRA target modules
- 包装 `peft.LoraModel`
- 加载本地 `.pt/.pth` `state_dict`

当前 target modules 固定为：

- GPT-2 系：`c_attn`
- Llama 系：`q_proj`

### 3.4 结果记录与汇总层

`attack.py` 现在的结果块会多写两个 PEFT 关键字段：

- `train_method`
- `lora_r`

这样 `scripts/collect_experiment_logs.py` 在聚合结果时，可以把：

- `train_method=full`
- `train_method=lora`

稳定分开，避免 full fine-tuning 和 LoRA 结果混表。

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

### 4.3 checkpoint 格式

当前只支持：

- 本地 `.pt`
- 本地 `.pth`
- 内容为当前仓库风格的 `state_dict`

当前不支持：

- PEFT 原生 adapter 目录
- 其他目录式 checkpoint

### 4.4 LoRA 下的 defense

当前正式支持：

- `none`
- `noise`
- `topk`
- `compression`
- `lrb`

当前不支持：

- `dpsgd`
- `mixup`
- `soteria`
- `dager`

其中：

- 单次运行会直接 fail-fast
- `peft_baselines.sh` 会把不支持项记录为 `unsupported`

---

## 5. 你应该怎么用

### 5.1 使用前准备

至少准备好下面几样东西：

1. 环境

推荐继续沿用仓库原本环境：

```bash
conda env create -f environment.yml -n dager
conda activate dager
```

2. 目录

```bash
mkdir -p models models_cache
```

3. LoRA checkpoint

你需要准备一个本地 LoRA checkpoint，例如：

```text
./models/gpt2_lora.pt
./models/llama_lora.pt
```

注意：

- 必须是本地文件
- 必须是 `.pt` 或 `.pth`
- 当前不能直接给一个 PEFT adapter 目录

4. Llama 路线的 Hugging Face 权限

如果你跑 Llama，需要先配置 `HF_TOKEN`。

---

## 6. 标准用法

### 6.1 单次评测：GPT-2 路线

```bash
bash scripts/peft_eval.sh sst2 2 gpt2 20 \
  --finetuned_path ./models/gpt2_lora.pt \
  --lora_r 16 \
  --defense lrb \
  --defense_lrb_keep_ratio_sensitive 0.2
```

这条命令的意思是：

- 数据集：`sst2`
- batch size：`2`
- backbone：`gpt2`
- 评测样本数：`20`
- 加载已有 LoRA checkpoint
- 用 `lrb` 做 defended gradient eval

### 6.2 单次评测：Llama 路线

```bash
bash scripts/peft_eval.sh sst2 2 meta-llama/Meta-Llama-3.1-8B 20 \
  --finetuned_path ./models/llama_lora.pt \
  --lora_r 256 \
  --defense compression \
  --defense_n_bits 8 \
  --pad left
```

Llama 路线通常还需要：

- `HF_TOKEN`
- 适当的 `--pad left`
- 更谨慎的 `rank_tol / span_thresh`

### 6.3 批量 baseline sweep

```bash
bash scripts/peft_baselines.sh sst2 2 gpt2 20 \
  --finetuned_path ./models/gpt2_lora.pt \
  --lora_r 16
```

这会：

- 跑 LoRA 场景下的 baseline sweep
- 对支持的 baseline 正常执行
- 对不支持的 baseline 记成 `unsupported`
- 输出统一汇总文件

### 6.4 聚焦某个 baseline

```bash
bash scripts/peft_baselines.sh sst2 2 gpt2 20 \
  --finetuned_path ./models/gpt2_lora.pt \
  --lora_r 16 \
  --baseline_defense lrb \
  --baseline_param 0.2
```

这类命令适合：

- 做单点对比
- 快速 smoke test
- 固定主配置跑复现实验

### 6.5 历史兼容入口

```bash
bash scripts/lora.sh sst2 2
```

这个入口仍然保留，但要知道它的语义是固定的：

- 模型默认是 `meta-llama/Meta-Llama-3.1-8B`
- `n_inputs=100`
- `--finetuned_path ./models/lora_8530.pt`
- `--lora_r 256`

所以：

- 如果你只是复现仓库历史 LoRA 路线，可以继续用它
- 如果你想认真做新实验，优先用 `peft_eval.sh` 和 `peft_baselines.sh`

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
- `--lora_r R`

脚本内部会自动补上：

- `--train_method lora`

### 7.3 推荐常用参数

常见会一起用的参数有：

- `--defense lrb`
- `--defense_topk_ratio`
- `--defense_n_bits`
- `--defense_noise`
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

---

## 9. 常见报错与含义

### 9.1 缺少 `--finetuned_path`

含义：

- 你声明了 `--train_method lora`
- 但没有给 LoRA checkpoint

处理：

- 补上本地 `.pt/.pth` checkpoint 路径

### 9.2 缺少 `--lora_r`

含义：

- 当前框架需要显式知道 LoRA rank

处理：

- 补上 `--lora_r`

### 9.3 模型族不支持

含义：

- 当前 LoRA eval 只支持 GPT-2 和 Llama

处理：

- 改用支持的模型
- 或者后续扩展 PEFT loader

### 9.4 defense 不支持

含义：

- 你在 LoRA 下用了当前还没接通的 defense，比如 `soteria`

处理：

- 首版只用 `none / noise / topk / compression / lrb`

### 9.5 checkpoint 格式不支持

含义：

- 给了目录
- 或给了非 `.pt/.pth`

处理：

- 改成当前仓库可识别的本地 LoRA `state_dict`

---

## 10. 推荐使用流程

如果你现在是第一次用这个框架，建议按这个顺序：

1. 先做一个最小 smoke test

```bash
bash scripts/peft_eval.sh sst2 2 gpt2 1 \
  --finetuned_path ./models/gpt2_lora.pt \
  --lora_r 16 \
  --defense none
```

2. 再做单点 `lrb` 对比

```bash
bash scripts/peft_eval.sh sst2 2 gpt2 20 \
  --finetuned_path ./models/gpt2_lora.pt \
  --lora_r 16 \
  --defense lrb \
  --defense_lrb_keep_ratio_sensitive 0.2
```

3. 再跑 baseline sweep

```bash
bash scripts/peft_baselines.sh sst2 2 gpt2 20 \
  --finetuned_path ./models/gpt2_lora.pt \
  --lora_r 16
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
