# PEFTLeak 图像侧实验 Runbook（CIFAR10 / Official-Aligned v1）

本文档用于在 Linux 服务器上运行 PEFTLeak 图像侧 privacy 实验。它参考 `docs/PEFTLEAK_TEXT_RUNBOOK.md` 的组织方式，但图像侧当前定位不同：这里是 **PEFTLeak-style image-side adapter leakage with an official-aligned v1 configuration**，不是官方 `info-ucr/PEFTLeak` 的 byte-for-byte 完整复现。

论文中建议写：

```text
We evaluate Projection-LRB on a PEFTLeak-style image-side adapter leakage
mechanism with an official-aligned configuration.
```

不要写：

```text
We fully reproduce PEFTLeak.
```

当前图像侧入口是：

```bash
python attack_peftleak_image.py
```

## 1. 基础准备

建议从服务器仓库根目录运行：

```bash
cd /data/lrl/FedLLM
conda activate base-llm   # 或替换成你的 torch/torchvision 环境
```

通用变量：

```bash
export DEVICE=cuda
export CACHE=./models_cache
export OUT=./outputs/peftleak_image_cifar10

export DATASET=cifar10
export N_CLASSES=10
export N_IMAGES=32          # pilot 可改成 8；最终可加到 100
export BATCH_SIZE=2
export PUBLIC_SPLIT_SIZE=64
export SEEDS="101 202 303 404 505"
export SAMPLE_STRATEGY=seeded_shuffle

mkdir -p "$CACHE" "$OUT/privacy" "$OUT/proxy_utility" "$OUT/tables"
```

正式实验必须带：

```bash
--fail_on_synthetic_fallback
```

否则 CIFAR10 缺失时会退回 synthetic smoke 数据，并在 summary 中标记 `synthetic_fallback=1`。这类结果不能进入论文表格。

## 2. CIFAR10 数据集下载

当前代码中 CIFAR10/CIFAR100 使用 `download=False`，所以需要先显式下载：

```bash
python - <<'PY'
from torchvision.datasets import CIFAR10
CIFAR10(root="./models_cache", train=True, download=True)
CIFAR10(root="./models_cache", train=False, download=True)
PY
```

确认数据存在：

```bash
test -d ./models_cache/cifar-10-batches-py && echo "CIFAR10 ready"
```

也可以让矩阵脚本负责下载：

```bash
DOWNLOAD_CIFAR10=1 bash scripts/peftleak_image_cifar10_matrix.sh
```

## 3. Smoke Test

先跑一个最小 official-aligned smoke：

```bash
python attack_peftleak_image.py \
  --mode official_vit_adapter \
  --peftleak_profile official_cifar32 \
  --dataset cifar10 \
  --data_root "$CACHE" \
  --cache_dir "$CACHE" \
  --n_classes 10 \
  --n_images 1 \
  --batch_size 1 \
  --public_split_size 4 \
  --adapter_layers first_n \
  --adapter_bottleneck_dim 64 \
  --official_grouping tag \
  --metrics mse,psnr,ssim,lpips,patch_recovery \
  --defense none \
  --device "$DEVICE" \
  --rng_seed 101 \
  --sample_strategy first_n \
  --split_seed 101 \
  --fail_on_synthetic_fallback \
  | tee "$OUT/privacy/smoke_none_seed101.log"
```

检查 summary 至少应看到：

```text
result_status=ok
dataset=cifar10
synthetic_fallback=0
attack_variant=official_vit_adapter
reproduction_level=peftleak_official_aligned_v1
primary_metric_source=direct 或 clustered
mse=...
psnr=...
patch_recovery_rate=...
sample_strategy=first_n
peftleak_protocol=legacy_first_n
```

如果 smoke 失败，不要继续跑正式矩阵。

### PEFTLeak-aligned fixed-batch reproduction

Use this only to align with the PEFTLeak image-side fixed victim batch protocol.
It is not a multi-seed defense table and should not be reported as mean/std.

```bash
bash scripts/peftleak_image_official_fixed_batch.sh
```

Defaults:

```text
DATASET=cifar100
N_CLASSES=100
N_IMAGES=32
BATCH_SIZE=32
PUBLIC_SPLIT_SIZE=$N_IMAGES
SAMPLE_STRATEGY=indices_file
ATTACK_INDICES_PATH=./PEFTLeak-main/PEFTLeak-main/img_list.npy
```

If `PUBLIC_INDICES_PATH` is not set, the fixed-batch script reuses the same
numeric indices on the public split and defaults `PUBLIC_SPLIT_SIZE` to
`N_IMAGES` so the official victim index file is sufficient. For paper runs,
prefer an explicit non-overlapping public index file or a saved public-stat file.

## 4. 一键运行 CIFAR10 Privacy 矩阵

推荐直接使用：

```bash
bash scripts/peftleak_image_cifar10_matrix.sh
```

脚本默认使用：

```text
DATASET=cifar10
N_CLASSES=10
N_IMAGES=32
BATCH_SIZE=2
PUBLIC_SPLIT_SIZE=64
SEEDS="101 202 303 404 505"
SAMPLE_STRATEGY=seeded_shuffle
```

Reportable multi-seed defense matrices must use `sample_strategy=seeded_shuffle`
or explicit index files. Re-running `first_n` with different `rng_seed` values
is only a fixed-split sanity check.

常见覆盖方式：

```bash
N_IMAGES=8 bash scripts/peftleak_image_cifar10_matrix.sh
N_IMAGES=100 PUBLIC_SPLIT_SIZE=128 bash scripts/peftleak_image_cifar10_matrix.sh
RUN_ABLATIONS=1 bash scripts/peftleak_image_cifar10_matrix.sh
```

脚本会运行所有核心 privacy baseline，并在最后生成：

```text
$OUT/tables/image_privacy_proxy_utility.csv
$OUT/tables/image_privacy_proxy_utility.md
```

## 5. 手动运行 Privacy Baselines

如果需要分批跑，可以先定义统一函数：

```bash
run_img_privacy () {
  tag="$1"; seed="$2"; shift 2
  echo "========== ${tag} seed=${seed} =========="
  python attack_peftleak_image.py \
    --mode official_vit_adapter \
    --peftleak_profile official_cifar32 \
    --dataset "$DATASET" \
    --data_root "$CACHE" \
    --cache_dir "$CACHE" \
    --n_classes "$N_CLASSES" \
    --n_images "$N_IMAGES" \
    --batch_size "$BATCH_SIZE" \
    --public_split_size "$PUBLIC_SPLIT_SIZE" \
    --vit_config cifar_small \
    --adapter_layers first_n \
    --adapter_bottleneck_dim 64 \
    --attack_rounds 1 \
    --official_grouping tag \
    --metrics mse,psnr,ssim,lpips,patch_recovery \
    --device "$DEVICE" \
    --rng_seed "$seed" \
    --sample_strategy "${SAMPLE_STRATEGY:-seeded_shuffle}" \
    --split_seed "$seed" \
    --fail_on_synthetic_fallback \
    "$@" \
    | tee "$OUT/privacy/${tag}_seed${seed}.log"
}
```

### Clean / None

```bash
for seed in $SEEDS; do
  run_img_privacy "none" "$seed" --defense none
done
```

### Top-k

```bash
for p in 0.05 0.1 0.3 0.5; do
  for seed in $SEEDS; do
    run_img_privacy "topk_${p}" "$seed" \
      --defense topk \
      --defense_topk_ratio "$p"
  done
done
```

### Compression

```bash
for b in 4 8 16; do
  for seed in $SEEDS; do
    run_img_privacy "compression_${b}" "$seed" \
      --defense compression \
      --defense_n_bits "$b"
  done
done
```

### Projection-LRB / `proj_only`

这是图像侧最贴近主方法的矩阵。

```bash
for k in 0.5 0.65 0.75 0.9; do
  for seed in $SEEDS; do
    run_img_privacy "proj_only_${k}" "$seed" \
      --defense lrb \
      --defense_lrb_preset proj_only \
      --defense_lrb_keep_ratio_sensitive "$k"
  done
done
```

### `proj_clip` / `full_lrb`

```bash
for preset in proj_clip full_lrb; do
  for k in 0.5 0.65 0.75; do
    for seed in $SEEDS; do
      run_img_privacy "${preset}_${k}" "$seed" \
        --defense lrb \
        --defense_lrb_preset "$preset" \
        --defense_lrb_keep_ratio_sensitive "$k"
    done
  done
done
```

### Noise / DP-SGD-style

```bash
for sigma in 1e-5 1e-4 5e-4 1e-3; do
  for seed in $SEEDS; do
    run_img_privacy "noise_${sigma}" "$seed" \
      --defense noise \
      --defense_noise "$sigma"

    run_img_privacy "dpsgd_${sigma}" "$seed" \
      --defense dpsgd \
      --defense_noise "$sigma" \
      --defense_clip_norm 1.0
  done
done
```

这里的 `dpsgd` 只能写成 clipping + Gaussian noise baseline；当前没有 DP accountant，不能声明 formal DP。

### Soteria-like / MixUp-like

```bash
for seed in $SEEDS; do
  run_img_privacy "soteria_60" "$seed" \
    --defense soteria \
    --defense_soteria_pruning_rate 60.0

  run_img_privacy "mixup_1.0" "$seed" \
    --defense mixup \
    --defense_mixup_alpha 1.0
done
```

`soteria` 和 `mixup` 是 patch/image 空间的 approximate coverage baselines，不是完整原方法复现。

### Signed Bottleneck

```bash
for k in 0.5 0.65 0.75 0.9; do
  for seed in $SEEDS; do
    run_img_privacy "signed_bottleneck_${k}" "$seed" \
      --defense signed_bottleneck \
      --defense_lrb_keep_ratio_sensitive "$k"
  done
done
```

### 可选 LRB 消融

```bash
for preset in identity_lrb clip_only proj_rule_only proj_empirical_only proj_uniform proj_no_empirical; do
  for seed in $SEEDS; do
    run_img_privacy "${preset}_0.5" "$seed" \
      --defense lrb \
      --defense_lrb_preset "$preset" \
      --defense_lrb_keep_ratio_sensitive 0.5
  done
done
```

## 6. 代理效用与真实效用边界

当前可直接得到的图像侧“代理效用”来自同一批 privacy 日志：

```text
vit_adapter_loss
batch_top1_acc
```

它们只能作为 malicious ViT-adapter attack batch 的 sanity metric，不能等价于完整 CIFAR10 训练效用。AAAI 正文不要把 `batch_top1_acc` 写成 utility accuracy。

生成代理效用表：

```bash
python scripts/peftleak_image_proxy_table.py \
  --log-dir "$OUT/privacy" \
  --output "$OUT/tables/image_privacy_proxy_utility.csv" \
  --markdown-output "$OUT/tables/image_privacy_proxy_utility.md"
```

真实图像效用实验目前还没有仓库入口。论文前应补一个 CIFAR/ViT 训练脚本，例如：

```bash
python train_peftleak_image_utility.py \
  --dataset cifar10 \
  --data_root "$CACHE" \
  --model_path torchvision_vit_small \
  --image_size 32 \
  --patch_size 16 \
  --n_classes 10 \
  --train_method adapter \
  --adapter_bottleneck_dim 64 \
  --defense <baseline> \
  --rng_seed <101|202|303> \
  --output_dir "$OUT/utility/<tag>_seed<seed>" \
  --log_file "$OUT/utility/<tag>_seed<seed>.log"
```

真实效用应报告：

```text
test_accuracy
test_macro_f1
test_loss
train_time
defense
defense_param_name
defense_param_value
rng_seed
```

在该训练脚本实现前，图像侧只能作为 privacy/sanity 补充证据，不应承担主表 utility 结论。

## 7. 汇总检查

正式进表前检查没有 synthetic fallback：

```bash
grep -R "synthetic_fallback=1" "$OUT/privacy" || echo "No synthetic fallback"
grep -R "result_status=failed" "$OUT/privacy" || echo "No failed runs"
```

重点看 privacy 字段：

```text
mse
psnr
ssim
lpips
patch_recovery_rate
primary_metric_source
candidate_patch_count
recovered_patch_count
collision_patch_count
unresolved_patch_count
```

主表优先放：

```text
none
topk@0.1
compression@8
proj_only@0.5
proj_only@0.65
proj_only@0.75
proj_only@0.9
full_lrb@0.5
```

补充表放：

```text
noise
dpsgd-style
soteria-like
mixup-like
proj_clip
signed_bottleneck
LRB ablations
```

## 8. 最低验收标准

正式写进论文前，至少满足：

```text
1. 每个实验点都有 seed 101/202/303。
2. 所有正式日志 synthetic_fallback=0。
3. 所有正式日志 result_status=ok。
4. oracle_* 字段不进入主表。
5. topk 和 compression 与 Projection-LRB 使用相同 batch/n_images/public_split_size。
6. soteria/mixup 标注为 approximate coverage baselines。
7. DP-SGD-style 只写 clipping + Gaussian noise baseline，不声明 formal DP。
8. 图像侧整体写作定位为 supplementary cross-modal evidence。
```
### Updated sampling/protocol acceptance criteria

```text
1. Fixed-batch reproduction uses peftleak_protocol=official_fixed_batch and is not reported as mean/std.
2. Multi-seed defense tables use sample_strategy=seeded_shuffle or explicit index files.
3. Every reportable defense point has seeds 101/202/303/404/505, synthetic_fallback=0, and result_status=ok.
4. Same-seed defense comparisons share attack_indices_hash and public_indices_hash.
5. oracle_* fields stay out of primary tables.
6. topk/compression/Projection-LRB use the same batch_size/n_images/public_split_size/sample protocol.
7. soteria/mixup are labeled approximate coverage baselines.
8. DP-SGD-style is reported as clipping + Gaussian noise only, with no formal DP claim.
9. Image-side results are framed as supplementary cross-modal evidence.
```
