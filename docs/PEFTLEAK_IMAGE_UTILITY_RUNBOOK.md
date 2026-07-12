# PEFTLeak 图像侧下游效用 Runbook

本文档用于在 Linux 服务器上评估共享 Adapter 梯度经过防御后，CIFAR-100 分类任务的真实下游效用。该轨道与 `official_image_runner.py` 的 source-aligned 隐私轨道按 Adapter 结构和防御参数对齐，但不声称在同一个恶意模型上完成端到端训练。

## 1. 实验口径

正式配置：

| 项目 | 配置 |
| --- | --- |
| dataset | CIFAR-100 |
| backbone | ImageNet `ViT-B/16` (`IMAGENET1K_V1`) |
| PEFT structure | 12 blocks，每层 attention/MLP 各一个 bottleneck Adapter |
| bottleneck | 64 |
| shared update | 96 个 Adapter weight/bias gradient tensors |
| local update | classification head，不上传且不施加共享梯度防御 |
| utility control | `head_only`：Adapter LR=0，仅训练本地 classification head |
| split | 45k train / 5k validation，`split_seed=42` |
| test | 官方 10k test，仅在最佳 validation checkpoint 上评估 |
| optimizer | AdamW，Adapter/head LR `1e-3`，weight decay `0.01` |
| schedule | 1 epoch warmup + cosine，20 epochs |
| seeds | 101 / 202 / 303 |

主指标为 test top-1 accuracy、macro-F1、cross-entropy 和相对同 seed `none` 的 accuracy drop。重建 MSE、SSIM、LPIPS 和 patch recovery 属于隐私指标，不是 utility。

## 2. 环境准备

```bash
cd /data/lrl/FedLLM
conda activate fedllm-peftleak
export DEVICE=cuda:3
export DATA_ROOT=./data
```

第一次运行允许下载 CIFAR-100：

```bash
export DOWNLOAD_CIFAR100=1
```

ViT-B/16 预训练权重由 torchvision 下载到其标准缓存。正式模式在数据、权重或 CUDA 不可用时直接失败，不回退到随机模型或 synthetic 数据。

## 3. Debug Smoke

本地/服务器 CPU smoke 不下载数据：

```bash
python train_peftleak_image_utility.py \
  --profile debug_tiny \
  --device cpu \
  --num_epochs 1 \
  --batch_size 4 \
  --eval_batch_size 4 \
  --num_workers 0 \
  --adapter_bottleneck_dim 8 \
  --debug_train_size 8 \
  --debug_validation_size 4 \
  --debug_test_size 4 \
  --no_pretrained \
  --no-amp \
  --defense proj_only \
  --defense_lrb_keep_ratio_sensitive 0.5 \
  --output_dir .runtime/peftleak_image_utility_debug
```

验收字段：

```text
result_status=ok
reportable=false
shared_scope=adapter_only
local_scope=classification_head
```

debug 结果不得进入正式表。

## 4. 四点 Pilot

先用 seed 42 跑四个点，确认正常收敛和效用量级：

```bash
mkdir -p outputs/peftleak_official_image/utility/pilot

for spec in \
  "none --defense none" \
  "topk_0.1 --defense topk --defense_topk_ratio 0.1" \
  "compression_8 --defense compression --defense_n_bits 8" \
  "proj_only_0.5 --defense proj_only --defense_lrb_keep_ratio_sensitive 0.5"
do
  label="${spec%% *}"
  args="${spec#* }"
  SEED=42 OUTPUT_DIR="outputs/peftleak_official_image/utility/pilot/${label}" \
    bash scripts/peftleak_official_image_utility.sh $args \
    2>&1 | tee "outputs/peftleak_official_image/utility/pilot/${label}.log"
done
```

只有 `none` 能稳定学习、各 defense 没有 non-finite gradient，并且 checkpoint/test 指标完整后，才启动正式矩阵。不要根据 test accuracy 为每个 defense 单独调参。

## 5. 正式 Utility 矩阵

核心 7 个防御点加 1 个 head-only 对照、三 seed，共 24 次 utility 训练：

```bash
PROFILE=core \
SEEDS="101 202 303" \
RUN_ROOT=outputs/peftleak_official_image/utility/formal \
LOG_ROOT=outputs/peftleak_official_image/utility/formal/logs \
bash scripts/peftleak_official_image_utility_matrix.sh
```

核心矩阵：

```text
head_only control
none
topk@0.1
compression@8
proj_only@0.5
proj_only@0.75
proj_only@0.9
full_lrb@0.5
```

`head_only` 使用 `defense=none` 和 `lr_adapter=0`，用于量化本地分类 head 单独贡献的效用，不进入 privacy 对比。设置 `PROFILE=full` 会额外运行 `topk@0.3` 和 `compression@16`。脚本默认跳过已有成功 summary 且 checkpoint 存在的点；设置 `FORCE=1` 才会覆盖重跑。

如果尚未完成三 seed privacy，保持 `PRIVACY_LOG_ROOT` 未设置；矩阵脚本只生成 utility 表，不再自动读取历史 seed42。privacy 完成后再按第7节显式合并。

## 6. 对齐的 Privacy 补跑

现有 seed 42 结果继续保留。正式 mean/std 使用独立的 101/202/303 三 seed，不把 seed 42 混入统计：

```bash
for seed in 101 202 303; do
  PROFILE=core \
  SEED="$seed" \
  OUTPUT_DIR="outputs/peftleak_official_image/privacy/seed${seed}" \
  LOG_DIR="outputs/peftleak_official_image/privacy/seed${seed}/logs" \
  bash scripts/peftleak_official_image_baselines.sh
done
```

`official_image_runner.py` 将 `seed` 同时用于模型随机性和 LRB 的 `rng_seed`，summary 中的 `defense_rng_seed` 用于审计。

## 7. 汇总表

```bash
python scripts/peftleak_image_results.py \
  --privacy-log-dir outputs/peftleak_official_image/privacy \
  --utility-log-dir outputs/peftleak_official_image/utility/formal/logs \
  --expected-seeds 101 202 303 \
  --output-dir outputs/peftleak_official_image/tables
```

输出：

```text
image_privacy.csv
image_utility.csv
image_cross_protocol_comparison.csv
image_cross_protocol_comparison.md
```

聚合器按完整协议签名、defense、参数和 seed 去重；签名包含 batch/profile/backbone/bottleneck/split/optimizer 等字段。完全重复日志只计一次，配置相同但指标冲突的重复日志直接失败。它只接收 `result_status=ok` 且 `reportable=true` 的 utility，并要求 privacy/utility 的 seed 集与 `--expected-seeds` 完全一致。Accuracy drop 先在同 seed 内配对，再计算 mean/std。

不要把同时包含历史非完整日志和 canonical 日志的父目录作为输入。例如仓库当前 `log/peftleak_official_image/baselines` 同时包含旧 `logs/` 与完整 `seed42/logs/`，同配置指标冲突时聚合器会按设计拒绝执行。若只检查现有 canonical seed42，应明确使用 `log/peftleak_official_image/baselines/seed42/logs`。

该合并表的 `comparison_scope` 固定为 `cross_protocol_supplementary`。它不计算或宣称严格的同系统 Pareto optimality，因为 privacy 使用 source-aligned 恶意 CIFAR32 custom ViT，而 utility 使用正常的 ImageNet 预训练 ViT-B/16。

## 8. 正式验收

每个 utility 点至少满足：

```text
result_status=ok
reportable=true
profile=formal_vit_b16
pretrained_weights=IMAGENET1K_V1
shared_scope=adapter_only
local_scope=classification_head
utility_control=standard
adapter_bottleneck_dim=64
shared_gradient_tensor_count=96
eval_accuracy=<finite>
eval_macro_f1=<finite>
eval_loss=<finite>
final_model_path=<existing checkpoint>
```

论文中建议写：

```text
As a supplementary cross-protocol study, we evaluate downstream utility using
ImageNet-pretrained ViT-B/16 Adapter fine-tuning on CIFAR-100, while privacy is
measured with the source-aligned PEFTLeak image/Adapter reconstruction path at
matched defense hyperparameters.
```

不要写成 exact malicious-model end-to-end utility、完整复现 PEFTLeak 或形式化隐私保证。

## 9. 代码验证

```bash
python test_peftleak_image_utility_semantics.py
python -c "import test_peftleak_image_semantics as t; t.test_official_defense_summary_reports_effective_rng_seed()"
bash -n scripts/peftleak_official_image_utility.sh
bash -n scripts/peftleak_official_image_utility_matrix.sh
```
