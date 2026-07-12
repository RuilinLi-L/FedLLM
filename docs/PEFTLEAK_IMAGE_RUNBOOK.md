# PEFTLeak 图像侧实验 Runbook（Source-Aligned CIFAR-100）

本文档用于在 Linux 服务器上运行论文侧 PEFTLeak 图像/Adapter privacy 实验。当前唯一推荐入口是：

```bash
bash scripts/peftleak_official_image_clean.sh
```

该入口调用 `PEFTLeak-main/PEFTLeak-main/official_image_runner.py`，贴近 upstream `Adapter_attack.ipynb` 的 CIFAR-100 恶意 ViT/Adapter 路径，并支持在完整 Adapter update 上施加 post-gradient defense。

它是 **source-aligned CIFAR-100 port**，但不是 upstream 仓库的 byte-for-byte 完整复现。论文中建议写：

```text
We evaluate Projection-LRB on a source-aligned CIFAR-100 implementation of
the PEFTLeak image/Adapter leakage mechanism as supplementary cross-modal evidence.
```

不要写：

```text
We fully reproduce PEFTLeak.
Projection-LRB provides formal privacy guarantees for PEFT.
```

图像侧只承担补充机制证据。DAGER 仍是本文的主攻击证据面。

## 1. 正式协议

默认论文协议如下：

| 项目 | 配置 |
| --- | --- |
| victim dataset | CIFAR-100 train |
| victim indices | `PEFTLeak-main/PEFTLeak-main/img_list.npy` 的前32项 |
| public statistics | CIFAR-100 test |
| batch size | 32 |
| image / patch size | 32 / 16 |
| patches per image | 4 |
| target patch count | 128 |
| model | upstream-style malicious custom ViT with Adapters |
| shared update | 完整96个 Adapter weight/bias gradient tensors |
| attack-visible subset | 官方恢复公式使用的20对 weight/bias gradients |
| defense location | 完整 Adapter update 生成后、patch recovery 前 |
| evaluation unit | 单个固定 victim batch、单轮 post-gradient privacy evaluation |

runner 先对完整96个 Adapter 梯度张量施加 defense，再从 defended update 中选出官方恢复机制使用的20对梯度。不能只防御攻击者最终读取的20对梯度。

默认 `PUBLIC_SPLIT=test`，public statistics 与 victim train split 天然分离。若显式设置 `PUBLIC_SPLIT=train`，runner 会先从 public set 中排除 `img_list.npy` 对应的 victim indices，并通过 `public_excluded_attack_count` 报告排除数量。

当前协议不是：

- FedAvg 或多轮联邦训练；
- 多客户端聚合攻击；
- 真实图像任务 utility 训练；
- formal DP evaluation。

## 2. 环境和数据

从服务器仓库根目录运行：

```bash
cd /data/lrl/FedLLM
```

首次创建环境：

```bash
conda env create -f environment-peftleak.yml
conda activate fedllm-peftleak
```

已有环境时只需：

```bash
conda activate fedllm-peftleak
```

`environment-peftleak.yml` 已包含正式指标所需的 PyTorch、torchvision、SciPy、scikit-image、LPIPS 和 k-means-constrained。先检查 runner 能否加载：

```bash
python PEFTLeak-main/PEFTLeak-main/official_image_runner.py --help
```

统一变量：

```bash
export DEVICE=cuda:3
export DATA_ROOT=./data
export IMG_LIST_PATH=PEFTLeak-main/PEFTLeak-main/img_list.npy
export OUT=outputs/peftleak_official_image
export BATCH_SIZE=32
export SEED=42
export PUBLIC_SPLIT=test

mkdir -p "$OUT/smoke" "$OUT/clean" "$OUT/manual"
set -o pipefail
```

确认 victim index 文件存在：

```bash
test -f "$IMG_LIST_PATH" && echo "img_list ready"
```

第一次运行时设置 `DOWNLOAD_CIFAR100=1`。runner 会下载 train 和 test split；后续运行不再需要该变量。

## 3. 严格 Smoke Test

先运行 batch=1 smoke：

```bash
DOWNLOAD_CIFAR100=1 \
BATCH_SIZE=1 \
SEED=42 \
OUTPUT_DIR="$OUT/smoke/seed42" \
bash scripts/peftleak_official_image_clean.sh \
  2>&1 | tee "$OUT/smoke/none_seed42.log"
```

该 smoke 会通过 wrapper 自动启用：

```text
--require_reportable_metrics
```

因此它会同时验证：

- CIFAR-100 和 `img_list.npy` 能否加载；
- 官方恶意 ViT/Adapter 路径能否完成前向和反向；
- 完整 Adapter update 能否收集；
- 官方恢复公式与本地恢复实现是否一致；
- patch clustering 和 SciPy Hungarian matching 是否可用；
- SSIM 和 LPIPS 是否能成功计算。

batch=1 只用于环境与机制检查，不能作为正式论文结果。smoke 至少应满足：

```text
result_status=ok
dataset=cifar100
batch_size=1
attack_index_count=1
require_reportable_metrics=true
defense=none
defense_applied_to=adapter_weight_bias_gradients
defended_gradient_count=96
patch_count=4
peftleak_target_patch_count=4
reconstruct_parity_status=ok
image_metric_scope=clustered_full_images_one_to_one_all_reconstructions
image_matching_method=hungarian
reconstructed_image_count=1
image_match_count=1
mse_status=ok
ssim_status=ok
lpips_status=ok
```

如果 smoke 失败，不要继续运行正式矩阵。先处理 summary 中的 `error_type` 和 `error_message`。

## 4. Canonical Clean

source-aligned canonical clean 使用固定 batch=32 和 seed=42：

```bash
BATCH_SIZE=32 \
SEED=42 \
OUTPUT_DIR="$OUT/clean/seed42" \
bash scripts/peftleak_official_image_clean.sh \
  --defense none \
  2>&1 | tee "$OUT/clean/none_seed42.log"
```

正式 clean 日志应满足第8节的完整验收标准。不要用 batch=1 smoke 替代 batch=32 clean。

## 5. Defense Baselines

### 5.1 一键 baseline 矩阵

运行：

```bash
OUTPUT_DIR="$OUT/baselines/seed42" \
LOG_DIR="$OUT/baselines/seed42/logs" \
BATCH_SIZE=32 \
SEED=42 \
bash scripts/peftleak_official_image_baselines.sh
```

当前脚本实际覆盖：

```text
none
topk@0.1
topk@0.3
compression@8
compression@16
proj_only@0.5
proj_only@0.75
proj_only@0.9
full_lrb@0.5
```

方法角色必须保持一致：

| 方法 | 角色 |
| --- | --- |
| `none` | clean leakage anchor |
| `topk` | 强通信稀疏化 baseline |
| `compression` | 强量化 baseline |
| `proj_only` | Projection-LRB / LRB-lite，本文主方法 |
| `full_lrb` | projection + clipping + residual-space noise 的强防御/过防御参照 |

`full_lrb` 不是最终主方法，不能替代 `proj_only` 成为论文方法身份。

### 5.2 单独运行核心点

定义统一日志函数：

```bash
run_official_image () {
  label="$1"
  shift
  mkdir -p "$OUT/manual/seed${SEED}"
  OUTPUT_DIR="$OUT/manual/seed${SEED}/${label}" \
  bash scripts/peftleak_official_image_clean.sh "$@" \
    2>&1 | tee "$OUT/manual/seed${SEED}/${label}.log"
}
```

Clean：

```bash
run_official_image "none" \
  --defense none
```

Top-k：

```bash
run_official_image "topk_0.1" \
  --defense topk \
  --defense_topk_ratio 0.1
```

Compression：

```bash
run_official_image "compression_8" \
  --defense compression \
  --defense_n_bits 8
```

Projection-LRB：

```bash
run_official_image "proj_only_0.5" \
  --defense proj_only \
  --defense_lrb_keep_ratio_sensitive 0.5
```

完整 LRB 强防御参照：

```bash
run_official_image "full_lrb_0.5" \
  --defense full_lrb \
  --defense_lrb_keep_ratio_sensitive 0.5
```

若不使用辅助函数，`full_lrb@0.5` 的直接命令是：

```bash
bash scripts/peftleak_official_image_clean.sh \
  --defense full_lrb \
  --defense_lrb_keep_ratio_sensitive 0.5
```

### 5.3 可选 PSNR

wrapper 默认请求：

```text
mse,ssim,lpips,patch_recovery
```

默认没有请求 PSNR，因此 `psnr_status=not_requested` 是正常的。若表格需要 PSNR，显式运行：

```bash
OUTPUT_DIR="$OUT/baselines_psnr/seed42" \
LOG_DIR="$OUT/baselines_psnr/seed42/logs" \
bash scripts/peftleak_official_image_baselines.sh \
  --metrics mse,psnr,ssim,lpips,patch_recovery
```

不要把包含 PSNR 和不包含 PSNR 的日志误判为不同攻击协议；它们只是在请求的报告指标上不同。

## 6. 多 Seed 稳健性

使用独立输出和日志目录，避免不同 seed 相互覆盖：

```bash
for seed in 101 202 303; do
  SEED="$seed" \
  OUTPUT_DIR="$OUT/baselines/seed${seed}" \
  LOG_DIR="$OUT/baselines/seed${seed}/logs" \
  BATCH_SIZE=32 \
  PUBLIC_SPLIT=test \
  bash scripts/peftleak_official_image_baselines.sh
done
```

这里所有 seed 都使用同一个 `img_list.npy`，因此 victim batch 保持固定。seed 改变模型初始化以及包含随机性的 defense/reconstruction 过程，不代表三个独立 victim batches。

公平比较要求：

- 同一 seed 下所有 defense 使用相同 `IMG_LIST_PATH`、`BATCH_SIZE` 和 `PUBLIC_SPLIT`；
- 每个 seed 使用独立 `LOG_DIR`；
- 不把 seed=42 canonical 点与 seed=101/202/303 的 mean/std 混成四 seed 统计，除非论文明确这样定义；
- 多 seed 结果只能说明固定 victim batch 下对随机初始化/随机过程的稳健性。

## 7. 指标口径

### 7.1 Patch recovery

主 patch 指标是：

```text
patch_recovery_rate
peftleak_recovered_patch_rate
```

两者当前指向同一个协议：在 denormalized image space 中，以默认 MSE 阈值 `0.005` 做 same-position、one-to-one GT verification。每个 target patch 最多匹配一次，因此 recovery rate 必须位于 `[0, 1]`。

默认同时输出 `0.001/0.005/0.010` 三个阈值的诊断字段，但主字段使用 `0.005`。不要用 `candidate_patch_count` 代替 recovered patch count。

### 7.2 完整图像质量

顶层指标：

```text
mse / mse_std
ssim / ssim_std
lpips / lpips_std
psnr / psnr_std       # 仅显式请求时
```

计算流程是：

1. 只使用恢复候选本身做无 GT clustering；
2. 每个 cluster 内缺失的 patch position 用中灰 patch 填充；
3. 始终构造与 victim batch 等量的完整图像；
4. 仅在最终评价时使用 Hungarian matching 将重建图像与 references 做一对一 permutation-invariant 匹配；
5. 在整个重建 batch 上计算顶层质量指标。

正式日志必须包含：

```text
image_metric_scope=clustered_full_images_one_to_one_all_reconstructions
image_matching_method=hungarian
```

这些顶层指标不会先按 recovery threshold 筛选“恢复成功”的 patch 或图像。即使 defense 完全阻断候选，runner 也会对中灰填充的完整图像计算有限指标。

指标方向：

| 指标 | 更差重建 / 更强 privacy 的方向 |
| --- | --- |
| patch recovery rate | 越低 |
| MSE | 越高 |
| PSNR | 越低 |
| SSIM | 越低 |
| LPIPS | 越高 |

### 7.3 诊断字段

`matched_patch_*` 是满足 `0.005` 阈值的一对一 patch matches 上的条件诊断，只用于解释恢复误差，不能替代顶层完整图像 MSE/SSIM/LPIPS。

以下字段描述重建结构，而不是运行是否有效：

```text
candidate_patch_count
clustered_image_count
complete_reconstructed_image_rate
reconstructed_patch_coverage_rate
duplicate_cluster_position_count
```

它们可以因 defense 变强而下降，不应设置固定成功阈值。

### 7.4 严格指标模式

`scripts/peftleak_official_image_clean.sh` 默认传入 `--require_reportable_metrics`。在该模式下，以下情况会使运行直接失败：

- 官方恢复公式 parity 失败；
- patch clustering 依赖或聚类过程失败；
- SciPy Hungarian matching 不可用；
- 请求的 SSIM 或 LPIPS 无法计算；
- `--metrics` 包含未知指标。

正式结果不接受请求指标的 `n/a`、`unavailable` 或 `failed:*` 状态。零候选不是依赖失败，仍会通过中灰图像完成完整评价。

## 8. 正式日志验收

batch=32、`PUBLIC_SPLIT=test` 的每个论文点至少满足：

```text
result_status=ok
attack=peftleak_official_image
dataset=cifar100
model=official_custom_vit
batch_size=32
attack_index_count=32
public_split=test
public_sample_count=10000
public_attack_overlap_count=0
public_excluded_attack_count=0
require_reportable_metrics=true
defense_applied_to=adapter_weight_bias_gradients
defended_gradient_count=96
patch_count=128
peftleak_target_patch_count=128
peftleak_recovered_patch_status=ok
reconstruct_parity_status=ok
image_metric_scope=clustered_full_images_one_to_one_all_reconstructions
image_matching_method=hungarian
reconstructed_image_count=32
image_match_count=32
mse_status=ok
ssim_status=ok
lpips_status=ok
```

若显式请求 PSNR，还必须满足：

```text
psnr_status=ok
```

批量检查失败日志：

```bash
grep -R "result_status=failed" "$OUT" || echo "No failed runs"
grep -R -E "mse_status=(failed|unavailable)|ssim_status=(failed|unavailable)|lpips_status=(failed|unavailable)" \
  "$OUT" || echo "No unavailable requested metrics"
```

对于每个 defense，还应确认：

```text
defense=<expected defense>
defense_param_name=<expected parameter>
defense_param_value=<expected value>
```

`reconstructed_patch_coverage_rate`、`complete_reconstructed_image_rate`、candidate 数量和 recovery rate 是攻击/防御结果，不是有效性门槛。

runner 的 RESULT SUMMARY 通过 stdout 输出。baseline wrapper 使用 `tee` 保存每个点的日志，但 `output_dir` 本身不会自动生成论文 CSV 或 Markdown 表格。

## 9. Utility 和论文表述边界

runner 输出的：

```text
loss
loss_scope=pre_defense_attack_batch
```

只是恶意 ViT/Adapter victim batch 在 defense 前的 cross-entropy loss。由于当前 defense 是 post-gradient transform，它不能作为训练后 utility、CIFAR-100 test accuracy 或 privacy-utility tradeoff。

图像侧当前可以支持：

- PEFTLeak 图像/Adapter 泄露机制在 source-aligned CIFAR-100 路径上的补充验证；
- Projection-LRB、top-k、compression 和 full LRB 对 Adapter update recoverability 的比较；
- projection bottleneck 是否跨文本/图像攻击机制保持有效的机制证据。

图像侧当前不能单独支持：

- 完整图像任务 utility 结论；
- 多轮 FL 或 FedAvg 结论；
- 正式 DP 保证；
- Projection-LRB 对所有 PEFT/LoRA 设置都有效；
- Projection-LRB 已普遍优于 top-k 或 compression。

建议论文方法角色：

```text
Main method: Projection-LRB / proj_only
Strong empirical baselines: topk, compression
Strong-defense / over-defense reference: full_lrb
Primary attack evidence: DAGER
Supplementary cross-modal mechanism evidence: PEFTLeak image/Adapter
```

## 10. 图像侧真实 Utility

当前 source-aligned runner 仍只负责固定 victim batch 的隐私重建评估。真实 CIFAR-100 下游效用使用独立的预训练 ViT-B/16 Adapter 训练轨道，完整命令、矩阵和验收标准见：

- [PEFTLeak 图像侧下游效用 Runbook](./PEFTLEAK_IMAGE_UTILITY_RUNBOOK.md)

两条轨道按 12 层双 Adapter、bottleneck=64、96 个共享 Adapter 梯度以及 defense hyperparameter 对齐。分类 head 是本地参数，不进入 PEFTLeak 共享更新，并通过 `head_only` 对照量化其独立贡献。该结果只能称为 supplementary cross-protocol comparison，不能称为同一恶意模型的严格 Pareto。`batch_top1_acc` 和 `loss_scope=pre_defense_attack_batch` 仍然只是攻击诊断，不得作为下游 utility。

## 11. 代码和文档验证

在 `fedllm-peftleak` 环境中执行：

```bash
python PEFTLeak-main/PEFTLeak-main/official_image_runner.py --help
python test_peftleak_image_semantics.py
python test_peftleak_image_utility_semantics.py
python test_lrb_defense_simple.py
python test_defense_baselines_semantics.py
```

正式开跑前再次确认 baseline wrapper 包含预期矩阵：

```bash
grep -n "run_variant" scripts/peftleak_official_image_baselines.sh
```

## Appendix A. Legacy CIFAR-10 / Official-Aligned v1 Proxy

以下是历史 `official-aligned v1 proxy` 路径：

```bash
python attack_peftleak_image.py --mode official_vit_adapter
bash scripts/peftleak_image_cifar10_matrix.sh
```

该路径可用于 smoke、debug 和历史兼容，但不是当前 paper-facing source-aligned runner。它的模型结构、采样协议、指标字段和 baseline 支持与本 runbook 正文不同。

Legacy v1 结果不得：

- 合并进 source-aligned CIFAR-100 主表；
- 冒充 upstream PEFTLeak 完整复现；
- 与当前完整图像顶层指标直接混合计算 mean/std；
- 作为图像侧真实 utility 结果。
