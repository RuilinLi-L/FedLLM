# LRB Ablation Runbook

本文说明如何使用 `scripts/lrb_ablation.sh` 运行 LRB 机制消融实验，并生成可用于论文表格的结果。

## 1. 当前推荐主配置

基于 `utility260426` 的结果，当前 LRB 消融主配置建议使用：

```text
defense_lrb_keep_ratio_sensitive = 0.5
defense_lrb_keep_ratio_other     = 0.75
projection                       = signed_pool
empirical_weight                 = 0.6
noise_sensitive                  = 0.03
noise_other                      = 0.005
clip_sensitive                   = 0.5
clip_other                       = 1.0
```

原因是 `lrb@0.5` 在已完成的 LRB operating points 中 utility 最好：

| point | eval_accuracy | utility_drop |
| --- | ---: | ---: |
| `lrb@0.2` | `0.821865` | `0.091361` |
| `lrb@0.35` | `0.868119` | `0.045107` |
| `lrb@0.5` | `0.892584` | `0.020642` |

因此，论文消融应优先围绕 `lrb@0.5` 解释机制，而不是继续使用过重的 `0.2` 或 `0.35`。

## 2. 脚本位置

```bash
scripts/lrb_ablation.sh
```

脚本会自动运行三类实验：

| stage | 内容 | 主要指标 |
| --- | --- | --- |
| `privacy` | DAGER attack-time privacy | `rec_token_mean`, `rouge1`, `rouge2` |
| `proxy` | one-step proxy utility | `grad_cosine_mean`, `norm_retention_mean`, `step_runtime_mean` |
| `train` | end-to-end utility | `eval_accuracy`, `eval_macro_f1`, `eval_loss`, `total_train_time` |

默认 `--mode all` 会三类都跑。

## 3. 消融 variants

默认会跑以下 variants：

| variant | 作用 |
| --- | --- |
| `none` | clean anchor，不使用防御 |
| `identity_lrb` | LRB 分支恒等映射，检查代码路径本身是否引入变化 |
| `clip_only` | 只保留 layer-wise clipping，不做 projection / noise |
| `proj_only` | 只保留 low-resolution projection，不做 clipping / noise |
| `proj_clip` | projection + clipping，不加 residual noise |
| `full_lrb` | 完整 LRB 主配置 |
| `pool_full` | 用 `pool` 替换默认 `signed_pool` |
| `rule_only` | `empirical_weight=0`，只用结构先验 |
| `empirical_only` | `empirical_weight=1`，只用当前梯度校准 |
| `uniform_all_sensitive` | 所有层使用同等强度，检验 layer-wise 分配是否必要 |

论文里的机制结论主要依赖：

- `identity_lrb` 是否接近 `none`
- `proj_only` 是否已能显著降低 DAGER 恢复
- `proj_clip` 和 `full_lrb` 的差距是否说明 residual noise 有增益
- `pool_full` 与 `full_lrb` 的差距是否支持 `signed_pool` 设计
- `rule_only / empirical_only / full_lrb` 的差距是否支持混合敏感度校准
- `uniform_all_sensitive` 与 `full_lrb` 的差距是否支持 layer-wise bottleneck

## 4. 冒烟测试

先用少量输入确认环境、路径和日志都正常：

```bash
cd /data/lrl/FedLLM
conda activate dager

bash scripts/lrb_ablation.sh \
  --n_inputs 5 \
  --mode privacy \
  --variants none,identity_lrb,full_lrb \
  --skip_existing
```

如果这一步完成，检查输出目录：

```bash
ls log/runs/lrb_ablation_*
```

冒烟输出应至少包含：

```text
privacy/
raw_results.csv
raw_results.md
ablation_privacy_summary.csv
ablation_privacy_summary.md
ablation_combined_summary.csv
ablation_combined_summary.md
```

## 5. 正式论文实验

推荐在 `tmux` 中运行，防止 SSH 断开导致任务中止：

```bash
cd /data/lrl/FedLLM
conda activate dager
tmux new -s lrb_ablation
```

正式命令：

```bash
bash scripts/lrb_ablation.sh \
  --lrb_main_k 0.5 \
  --n_inputs 100 \
  --mode all \
  --skip_existing
```

默认设置：

```text
dataset       = sst2
batch_size    = 2
model_path    = gpt2
finetuned     = ./models/gpt2-ft-rt
epochs        = 1
seeds         = 101 202 303
variants      = all
```

如果模型路径不同，显式指定：

```bash
bash scripts/lrb_ablation.sh \
  --model_path gpt2 \
  --finetuned_path ./models/gpt2-ft-rt \
  --lrb_main_k 0.5 \
  --n_inputs 100 \
  --mode all \
  --skip_existing
```

## 6. 分阶段运行

如果机器资源紧张，建议分三步跑。

先跑 privacy：

```bash
bash scripts/lrb_ablation.sh \
  --lrb_main_k 0.5 \
  --n_inputs 100 \
  --mode privacy \
  --skip_existing
```

再跑 proxy：

```bash
bash scripts/lrb_ablation.sh \
  --lrb_main_k 0.5 \
  --n_inputs 100 \
  --mode proxy \
  --skip_existing \
  --run_dir log/runs/<上一步的lrb_ablation目录>
```

最后跑 train：

```bash
bash scripts/lrb_ablation.sh \
  --lrb_main_k 0.5 \
  --n_inputs 100 \
  --mode train \
  --skip_existing \
  --run_dir log/runs/<同一个lrb_ablation目录>
```

注意：分阶段续跑时，要把 `--run_dir` 指向同一个目录，否则结果会分散到多个 run 目录。

## 7. 只跑部分 variants

如果要先确认主机制，可以只跑核心 variants：

```bash
bash scripts/lrb_ablation.sh \
  --lrb_main_k 0.5 \
  --n_inputs 100 \
  --mode all \
  --variants none,identity_lrb,proj_only,proj_clip,full_lrb \
  --skip_existing
```

如果要专门看 sensitivity calibration：

```bash
bash scripts/lrb_ablation.sh \
  --lrb_main_k 0.5 \
  --n_inputs 100 \
  --mode all \
  --variants none,full_lrb,rule_only,empirical_only \
  --skip_existing
```

如果要专门看 projection basis：

```bash
bash scripts/lrb_ablation.sh \
  --lrb_main_k 0.5 \
  --n_inputs 100 \
  --mode all \
  --variants none,full_lrb,pool_full \
  --skip_existing
```

## 8. 输出文件

每次运行会生成一个目录，例如：

```text
log/runs/lrb_ablation_sst2_b2_gpt2_k0.5_YYYYMMDD_HHMMSS/
```

目录结构：

```text
privacy/                         DAGER attack logs
proxy/                           proxy utility logs
train/                           end-to-end utility logs
models/                          training outputs
manifest.txt                     本次实验配置
exit_codes.csv                   每个 stage/variant/seed 的退出码
raw_results.csv                  原始汇总
raw_results.md
ablation_privacy_summary.csv     privacy 按 variant 聚合
ablation_privacy_summary.md
ablation_proxy_summary.csv       proxy 按 variant 聚合
ablation_proxy_summary.md
ablation_utility_summary.csv     utility 按 variant 聚合
ablation_utility_summary.md
ablation_combined_summary.csv    论文主表推荐入口
ablation_combined_summary.md
```

最常看的文件是：

```bash
cat log/runs/<run_dir>/ablation_combined_summary.md
cat log/runs/<run_dir>/exit_codes.csv
```

## 9. 论文表格建议

主消融表建议使用 `ablation_combined_summary.md`，列：

| column | 含义 |
| --- | --- |
| `variant` | 消融版本 |
| `rec_token_mean` | token 恢复率，越低越好 |
| `rouge1_fm`, `rouge2_fm` | DAGER 文本恢复指标，越低越好 |
| `eval_accuracy` | 训练后验证准确率，越高越好 |
| `utility_drop` | 相对 `none` 的准确率下降，越低越好 |
| `grad_cosine_mean` | defended gradient 与 clean gradient 的方向相似度 |
| `norm_retention_mean` | 梯度范数保留比例 |
| `step_runtime_mean` | proxy 单步运行时间 |
| `train_time_seconds` | 完整训练平均耗时 |

推荐论文表结构：

| variant | rec token | R1+R2 | acc | drop | grad cos | runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none |  |  |  |  |  |  |
| identity-lrb |  |  |  |  |  |  |
| clip-only |  |  |  |  |  |  |
| projection-only |  |  |  |  |  |  |
| projection+clip |  |  |  |  |  |  |
| full LRB |  |  |  |  |  |  |
| pool full |  |  |  |  |  |  |
| rule-only |  |  |  |  |  |  |
| empirical-only |  |  |  |  |  |  |
| uniform all-sensitive |  |  |  |  |  |  |

## 10. 结果解释口径

理想情况下，希望看到：

1. `identity_lrb` 接近 `none`  
   说明 LRB 代码路径本身不是导致 privacy 变化的原因。

2. `clip_only` 弱于 `proj_only`  
   说明主效应不是单纯 clipping。

3. `proj_only` 已经显著降低 DAGER 恢复  
   支持 “recoverability bottleneck / projection” 是核心机制。

4. `full_lrb` 优于 `proj_clip` 或在相近 privacy 下更稳  
   支持 residual-space noise 有额外增益。

5. `full_lrb` 优于 `pool_full`  
   支持 `signed_pool` 公共随机子空间比普通 pooling 更合理。

6. `full_lrb` 优于 `rule_only` 和 `empirical_only`  
   支持结构先验 + 当前梯度校准的混合敏感度估计。

7. `full_lrb` 优于 `uniform_all_sensitive`  
   支持 layer-wise 分配，而不是所有层同等强度处理。

如果结果与预期不同，也很有价值。例如：

- 如果 `proj_only` 已经等于 `full_lrb`，说明可以简化 LRB，减少 noise 和 clipping。
- 如果 `rule_only` 接近 `full_lrb`，说明 empirical calibration 对当前设置帮助有限，可以作为 runtime 优化方向。
- 如果 `uniform_all_sensitive` 明显更好，说明当前 layer-wise sensitivity 估计可能需要重做。

## 11. 常见问题

### 11.1 中途断了怎么办

使用同一个 `--run_dir` 加 `--skip_existing` 继续跑：

```bash
bash scripts/lrb_ablation.sh \
  --run_dir log/runs/<run_dir> \
  --lrb_main_k 0.5 \
  --n_inputs 100 \
  --mode all \
  --skip_existing
```

### 11.2 只想重新汇总怎么办

目前脚本没有单独的 `collect-only` 参数。可以直接用内置汇总依赖的原始命令：

```bash
python3 scripts/collect_experiment_logs.py \
  log/runs/<run_dir>/privacy \
  log/runs/<run_dir>/proxy \
  log/runs/<run_dir>/train \
  -o log/runs/<run_dir>/raw_results.csv \
  --markdown log/runs/<run_dir>/raw_results.md
```

如果需要重新生成 `ablation_*_summary`，建议重新运行脚本并加 `--skip_existing`，它会跳过已有日志并重新汇总：

```bash
bash scripts/lrb_ablation.sh \
  --run_dir log/runs/<run_dir> \
  --lrb_main_k 0.5 \
  --mode all \
  --skip_existing
```

### 11.3 正式结果用几个 seed

当前主表保持 `101 / 202 / 303` 三个 seed，与已有 utility 实验一致。  
如果论文篇幅允许，可以把更多 seed 放到 appendix，但主表先不要改变 seed 设置，否则和现有结果不可直接比较。

## 12. 推荐执行顺序

1. 跑冒烟测试：`n_inputs=5`, `mode=privacy`。
2. 跑正式 privacy：`n_inputs=100`, `mode=privacy`。
3. 检查 `ablation_privacy_summary.md`，确认 `identity_lrb` 和 `full_lrb` 行合理。
4. 跑 proxy 和 train，或直接 `mode=all`。
5. 检查 `exit_codes.csv` 是否全部为 `0` 或 `skipped`。
6. 使用 `ablation_combined_summary.md` 写论文消融表。

