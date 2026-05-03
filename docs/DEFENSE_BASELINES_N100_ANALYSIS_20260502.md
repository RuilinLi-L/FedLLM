# n_inputs=100 防御基线结果分析（2026-05-02）

本文基于这次正式防御基线 sweep：

```text
log/runs/defense_baselines_sst2_b2_gpt2_20260501_010024/defense_baselines_sst2_b2_gpt2_20260501_010024
```

运行设置：

- dataset: `sst2`
- model: `gpt2`
- finetuned_path: `./models/gpt2-ft-rt`
- batch_size: `2`
- n_inputs: `100`
- defenses: `none / noise / dpsgd / topk / compression / soteria / mixup / lrb`

## 1. 核心结论

这组 `n_inputs=100` 结果强化了一个判断：在当前 `SST2 + GPT2 + batch=2` 的 DAGER attack-time 评测里，真正进入 strong privacy baseline 的只有三类：

- `lrb@0.05~0.5`
- `topk@0.01~0.3`
- `compression@4/8/16`

但结合已完成 utility 后，privacy-utility tradeoff 仍然是：

| 方法 | privacy | utility | 当前判断 |
| --- | --- | --- | --- |
| `topk@0.1` | DAGER=0 | `acc=0.912462`, drop `0.000764` | 当前最强经验点 |
| `compression@8` | DAGER=0 | `acc=0.911315`, drop `0.001911` | 次强经验点 |
| `topk@0.3` | DAGER=0 | `acc=0.910933`, drop `0.002293` | 已补，没超过 `topk@0.1` |
| `compression@16` | DAGER=0 | `acc=0.909021`, drop `0.004205` | 已补，没超过 `compression@8` |
| `lrb@0.5` | DAGER=0 | `acc=0.892584`, drop `0.020642` | LRB 当前最佳 utility 点，但仍弱于 topk/compression |
| `lrb@0.35` | DAGER=0 | `acc=0.868119`, drop `0.045107` | 比 `0.2` 好，但仍偏重 |
| `lrb@0.2` | DAGER=0 | `acc=0.821865`, drop `0.091361` | privacy 强，过防御 |

因此，论文叙事不能写成“LRB 已在 SST2/GPT2 上全面优于 topk/compression”。更稳的说法是：

> LRB 在 DAGER recoverability 抑制上非常强，且 `0.5` 明显改善了 utility；但在当前 full-gradient DAGER 单一攻击面上，topk/compression 仍是更强的经验 tradeoff。LRB 要作为主方法，需要依靠机制消融、PEFT/LoRA、partial-gradient 等更广攻击面的泛化证据。

## 2. n_inputs=100 privacy 结果

| defense | 参数区间 / 关键点 | rec_token_mean | R1+R2 | 结论 |
| --- | --- | ---: | ---: | --- |
| `none` | n/a | `0.833506` | `141.710856` | clean 泄露严重 |
| `lrb` | `0.05/0.1/0.2/0.35/0.5` | `0.000000` | `0.000000` | 全部打到 0 |
| `topk` | `0.01/0.05/0.1/0.3` | `0.000000` | `0.000000` | 安全区到 0.3 |
| `topk` | `0.5 / 0.7 / 0.9` | `0.047395 / 0.161375 / 0.381985` | `1.492857 / 8.767857 / 35.730123` | ratio 越大泄露越多 |
| `compression` | `4/8/16` | `0.000000` | `0.000000` | 这三档有效 |
| `compression` | `2` | `0.000000` | `0.000000` | 只跑到 56/100，SVD 失败，不应算正式成功点 |
| `compression` | `32` | `0.828508` | `141.137409` | 基本退化到 none |
| `noise` | `1e-6 -> 1e-3` | `0.953319 -> 0.173061` | `19.180338 -> 5.523918` | ROUGE 大降，但 token 仍有残留 |
| `dpsgd` | `1e-6 -> 1e-3` | `0.966569 -> 0.077848` | `15.737140 -> 0.583333` | privacy 随噪声增强，但 utility 风险很大 |
| `mixup` | `0.1~2.0` | `0.882837~0.929118` | `154.743067~168.820917` | 比 none 更差，不是 privacy defense |
| `soteria` | `10~90` | `0.996217~1.000000` | `189.384179~193.500000` | 显著恶化 |

需要特别改口径的一点是 `compression@2`：旧文档里常把 `2/4/8/16` 一起写成 DAGER=0，但这次 `n_inputs=100` 中 `compression@2` 在第 56 个样本处 `_LinAlgError` 失败，所以正式结论应只把 `4/8/16` 计入稳定成功点。

## 3. 已完成但旧文档仍写“待补”的实验

以下 utility 点已经有结果，不应再写成最高优先级待补：

| defense | param | eval_accuracy | utility_drop |
| --- | ---: | ---: | ---: |
| `lrb` | `0.35` | `0.868119` | `0.045107` |
| `lrb` | `0.5` | `0.892584` | `0.020642` |
| `topk` | `0.3` | `0.910933` | `0.002293` |
| `compression` | `16` | `0.909021` | `0.004205` |

这些结果改变了下一步优先级：现在不是先补这些 utility 点，而是应该把它们汇总成正式 privacy-utility Pareto 表，并把重点转向 LRB 消融和跨攻击面验证。

## 4. 需要更正的文档口径

### `docs/UTILITY_RESULTS_ANALYSIS_20260426.md`

需要标注为旧版 `n_inputs≈20` 口径，并在顶部加入本次 `n_inputs=100` 更新：

- clean baseline 改为 `rec_token_mean=0.833506`, `R1+R2=141.710856`。
- `topk@0.5/0.7/0.9` 的新值分别为 `0.047395/0.161375/0.381985`。
- `compression@2` 不能再算稳定成功点，因为本次失败在 `56/100`。
- `lrb@0.35/0.5`、`topk@0.3`、`compression@16` utility 已完成。

### `docs/CURRENT_WORK_STATUS_ANALYSIS_20260427.md`

需要从“待补公平 utility 点”改成“公平 utility 点已补，结论是 topk/compression 仍占优，LRB@0.5 是当前 LRB 最佳点”。同时把 `compression@2` 从稳定成功点中移除。

### `docs/LRB_实验任务清单.md`

需要把默认主配置从 `lrb@0.2` 倾向调整为：

- privacy-only 对照可保留 `0.2/0.35/0.5`；
- 论文消融和 utility-facing 主配置优先用 `lrb@0.5`；
- `topk@0.3` 和 `compression@16` 已补，不再是待补项。

### `docs/LRB_方法详解.md`

当前文档仍写“`train.py` 还没有 `--defense lrb` 的训练时 hook”。这已经过时。当前准确口径是：

- full training 下 `train.py` 已支持 `--defense lrb`；
- `train_method != full` 时，`lrb/dpsgd/mixup/soteria/dager` 训练期 defense 仍未支持；
- forward-side representation bottleneck 仍未实现。

### `docs/实验运行指南.md`

防御表中漏了 `lrb`。应补 `--defense lrb` 及 `--defense_lrb_keep_ratio_sensitive` 等关键参数。

## 5. 现在真正需要补的实验

### P0：LRB 机制消融

`log/runs/lrb_ablation_sst2_b2_gpt2_k0.5_20260430_133224` 当前只有 `none / identity_lrb / full_lrb` 的 privacy 结果，且 `utility_runs=0`。正式论文消融还没完成。

最小需要补：

- `clip_only`
- `proj_only`
- `proj_clip`
- `pool_full`
- `rule_only`
- `empirical_only`
- `uniform_all_sensitive`
- proxy + train utility 汇总

建议以 `lrb@0.5` 为主配置。

### P1：PEFT / LoRA 攻击面

最小结果集：

- LoRA + `none`
- LoRA + `lrb@0.5`
- LoRA + `topk@0.1` 和/或 `topk@0.3`
- LoRA + `compression@8` 和/或 `compression@16`

目的不是再证明 DAGER=0，而是看压缩类 baseline 是否在 LoRA 更新攻击面下仍然强，以及 LRB 是否有跨攻击面的结构性优势。

### P1：partial-gradient / layer-level leakage

当前还缺入口。建议先补：

- `--gradient_layer_subset`
- `--gradient_param_filter`

最小设置：

- first block only
- first 2 blocks
- last 2 blocks
- attention q/k/v only
- LoRA params only

### P2：跨数据集 / backbone

最小组合：

- `cola + gpt2`
- `rte + gpt2`
- `sst2 + bert-base-uncased`

每组至少跑 `none / lrb@0.5 / topk@0.1 / compression@8`，资源允许再加 `topk@0.3 / compression@16`。

### P2：LRB runtime 分析

`lrb@0.5` utility 已改善到 `acc=0.892584`，但训练时间仍约 `01:56:47`，明显慢于 none，也慢于普通训练路径。需要排查：

- signed mask / projection 是否每步重复生成；
- calibration sampling 是否过重；
- 是否能缓存 layer-wise shape plan；
- `defense_lrb_calibration_samples` 是否可降；
- `lrb@0.35` 训练时间方差很大，需查异常 seed 路径。

### P3：附录型实验

这些不应压过 P0/P1：

- `compression@2` 失败复跑或修 SVD 稳定性；
- `noise@1e-3` / `dpsgd@1e-3` utility，用于展示更强扰动下的代价；
- `soteria` utility 作为失败/迁移不顺 baseline 附录。

## 6. 推荐下一步一句话

当前主线应从“补公平 utility 点”更新为：

> 公平 utility 点已经补齐，topk/compression 在 full-gradient DAGER 上仍是更强经验 tradeoff；接下来应以 `lrb@0.5` 为主配置完成机制消融，并补 PEFT/LoRA 与 partial-gradient 攻击面，验证 LRB 的结构性泛化价值。
