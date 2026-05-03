# Utility 实验结果整理与分析（2026-04-26）

> 2026-05-02 更新：已有一组正式 `n_inputs=100` DAGER 防御基线结果，见 [DEFENSE_BASELINES_N100_ANALYSIS_20260502.md](./DEFENSE_BASELINES_N100_ANALYSIS_20260502.md)。本文后续章节仍保留 2026-04-26 的阶段性口径，其中部分 `n_inputs≈20` privacy 数值和“待补实验”已经过期。当前应以 `none rec_token_mean=0.833506, R1+R2=141.710856`、`compression@4/8/16` 稳定 DAGER=0、`compression@2` 在 56/100 处失败、以及已完成的 `lrb@0.35/0.5`、`topk@0.3`、`compression@16` utility 为准。

## 1. 文档目的

本文档用于把当前 `utility` 实验、`proxy utility` 实验、以及 `DAGER defense` 结果统一整理起来，回答三个问题：

1. 当前各个 defense 在 `utility` 上到底表现如何。
2. 当前固定 operating point 下，哪些方法的 privacy-utility tradeoff 最好。
3. 这些结果对后续主方法 `LRB` 的研究路线有什么启示。

本文的整理范围主要基于以下文件：

- `docs/FL-LLM.md`
- `docs/UTILITY_BASELINES.md`
- `docs/DEFENSE_BASELINES.md`
- `log/runs/test/summary_lrb.txt`
- `log/runs/test/results_topk.md`
- `log/runs/test/results_compression.md`
- `log/runs/test/results_mixup.md`
- `log/runs/test/results_soteria.md`
- `log/runs/test/summary_noise_table.txt`
- `log/runs/test/summary_dpsgd_table.txt`
- `log/runs/utility260426/utility_results_none.md`
- `log/runs/utility260426/utility_results_lrb.md`
- `log/runs/utility260426/utility_results_topk.md`
- `log/runs/utility260426/utility_results_compression.md`
- `log/runs/utility260426/utility_results_mixup.md`
- `log/runs/utility260426/utility_results_noise.md`
- `log/runs/utility260426/utility_results_dpsgd.md`
- `log/runs/utility260426/results_lrb.md`
- `log/runs/proxy_baselines_sst2_b2_gpt2_20260416_200409/results.md`

截至 `2026-04-26`，`soteria` 的 utility 训练结果还没有完整汇总出来，因此本文对 `soteria` 的分析以已有 privacy/proxy 结果为主，不把它纳入已完成的 utility 主表。

---

## 2. 当前实验设定

根据 `UTILITY_BASELINES.md` 和现有日志，当前这组结果的核心设定可以概括为：

- 数据集：`sst2`
- 任务：`seq_class`
- 模型：`gpt2-ft-rt`
- batch size：`2`
- 训练方式：`full`
- utility 训练 epoch：`1`
- utility seed：`101 / 202 / 303`
- proxy utility：`100` 个 held-out train batches
- DAGER privacy 评估：本文主体沿用 2026-04-26 阶段性 `20` 输入样本口径；2026-05-02 后正式对照应使用 `n_inputs=100` 结果

当前 utility 主表使用的固定 operating points 为：

| defense | parameter |
| --- | --- |
| `none` | `n/a` |
| `lrb` | `0.2` |
| `topk` | `0.1` |
| `compression` | `8` |
| `noise` | `5e-4` |
| `dpsgd` | `5e-4` |
| `mixup` | `0.3` |
| `soteria` | `30` |

需要特别注意：这只是“每个方法选了一个固定点”的对比，不等于“每个方法都已经调到自己的最优点”。

---

## 3. End-to-End Utility 结果总表

下面这张表是当前最重要的 utility 主结果。`utility_drop` 直接来自 `privacy_utility_tradeoff_*.md`，表示相对同次运行中的 `none` 基线，最终验证准确率下降了多少。

`time_ratio_vs_none` 是用每个 defense 对应文件中同时记录的 `none` 训练时间做近似比值，因此更适合看相对量级，不适合做特别精细的跨文件比较。

| defense | param | final_train_loss | eval_accuracy | eval_macro_f1 | eval_loss | utility_drop | total_train_time | time_ratio_vs_none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| none | n/a | 0.189247 | 0.913226 | 0.913184 | 0.246637 | 0.000000 | 00:42:51 | 1.00x |
| topk | 0.1 | 0.178671 | 0.912462 | 0.912430 | 0.243324 | 0.000764 | 06:03:08 | 6.40x |
| compression | 8 | 0.161670 | 0.911315 | 0.911290 | 0.263210 | 0.001911 | 07:07:37 | 8.12x |
| mixup | 0.3 | 0.196403 | 0.910933 | 0.910906 | 0.239469 | 0.002293 | 00:48:40 | 0.91x |
| lrb | 0.2 | 0.320997 | 0.821865 | 0.821360 | 0.441765 | 0.091361 | 08:52:41 | 10.30x |
| noise | 5e-4 | 0.502573 | 0.715979 | 0.715617 | 0.552434 | 0.197247 | 01:54:29 | 1.51x |
| dpsgd | 5e-4 | 2.102079 | 0.504205 | 0.366347 | 2.612275 | 0.409021 | 03:29:31 | 2.82x |
| soteria | 30 | pending | pending | pending | pending | pending | pending | pending |

### 3.1 直接观察

从 utility 主表可以先得到几个非常直接的结论：

- `topk@0.1` 的 utility 几乎和 `none` 一样，准确率只掉了 `0.000764`。
- `compression@8` 也非常强，准确率只掉了 `0.001911`。
- `mixup@0.3` 的 utility 也很好，准确率只掉了 `0.002293`，而且训练时间几乎不增加。
- `lrb@0.2` 当前 utility 明显偏差，准确率下降了 `0.091361`。
- `noise@5e-4` 和 `dpsgd@5e-4` 都需要较大代价，尤其 `dpsgd` 最终准确率只有 `0.504205`，已经接近不可用。

如果只按“固定点 utility”排序，当前大致是：

`none ≈ topk ≈ compression ≈ mixup >> lrb >> noise >> dpsgd`

### 3.2 当前 utility 最关键的一点

当前最重要的信息不是“某个 defense 有没有把 DAGER 打掉”，而是：

> 在多个已经能把 DAGER 打到接近 0 的方法里，utility 差异非常大。

也就是说，**“privacy 成功”并不自动意味着“是好的 defense”**。真正有区分度的，是在相近 privacy 水平下，谁保留了更多 utility。

---

## 4. Proxy Utility 结果总表

下面整理的是 `proxy utility` 的固定点结果，来源是 `log/runs/proxy_baselines_sst2_b2_gpt2_20260416_200409/results.md`。

需要特别说明：

- `utility260426/results_proxy.md` 指向的也是这组独立的 `proxy_baselines_sst2_b2_gpt2_20260416_200409` 结果。
- 它**不是** `utility260426/results_lrb.md`、`results_topk.md`、`results_compression.md` 等文件内部 proxy 行的逐项拷贝。
- 也就是说，`results_proxy.md` 应理解为一份单独的统一 proxy 汇总，而 `results_*.md` 中的 proxy 行来自各自 focused utility run，自身数值可能不同。

| defense | param | grad_cosine_mean | norm_retention_mean | delta_train_loss_mean | delta_val_loss_mean | delta_val_accuracy_mean | delta_val_macro_f1_mean | step_runtime_mean | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| none | n/a | 1.000000 | 1.000000 | -0.031331 | 0.000083 | 0.000273 | 0.000273 | 1.065636 | ok |
| compression | 8 | 0.969237 | 1.034058 | -0.031344 | 0.000084 | 0.000273 | 0.000273 | 1.108627 | ok |
| topk | 0.1 | 0.945117 | 0.945117 | -0.030731 | 0.000078 | 0.000273 | 0.000273 | 1.679956 | ok |
| mixup | 0.3 | 0.525846 | 22.249394 | 0.000806 | 0.000192 | 0.000195 | 0.000197 | 1.023529 | ok |
| noise | 5e-4 | 0.490102 | 12.943462 | -0.031332 | 0.000084 | 0.000273 | 0.000273 | 1.011835 | ok |
| lrb | 0.2 | 0.264516 | 0.083383 | -0.023003 | 0.000000 | 0.000000 | 0.000000 | 8.347359 | ok |
| dpsgd | 5e-4 | 0.111590 | 6.456979 | 0.011906 | 0.000008 | 0.000000 | 0.000000 | 1.176916 | ok |
| soteria | 30 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | failed |

### 4.1 对 proxy 结果的解释

这里至少能读出四件事：

1. `topk` 和 `compression` 对一步更新的破坏很小。  
   它们的 `grad_cosine_mean` 仍然很高，说明更新方向大体还保住了。

2. `lrb` 和 `dpsgd` 对梯度结构改动更大。  
   尤其 `lrb@0.2` 的 `grad_cosine_mean=0.264516`，`norm_retention_mean=0.083383`，说明它对原始梯度做了很强的变换。

3. `lrb` 的每步开销很高。  
   `step_runtime_mean=8.347359`，远高于其他固定点，这和后面 end-to-end utility 里的长训练时间是对应的。

4. `proxy utility` 明显不能代替 `end-to-end utility`。  
   在 proxy 里，`lrb/noise/dpsgd` 对验证精度的即时影响接近 0，但完整训练后它们的最终精度分别下降了 `9.1`、`19.7`、`40.9` 个点。

### 4.2 一个重要结论

当前结果说明：

> proxy utility 更适合做“初筛”和“排除明显坏点”，但不适合拿来替代真正的训练 utility。

特别是对于像 `LRB` 这种会系统性改变训练轨迹的 defense，proxy 很容易低估长期优化代价。

---

## 5. 与 Utility 对齐的 Privacy 结果

下面把 utility 主表对应的固定点，与 DAGER privacy 结果对齐整理。这里的 privacy 主要用：

- `rec_token_mean`：越低越好
- `agg_r1fm_r2fm`：越低越好
- `last_rec_status`：是否还能正常恢复出候选

### 5.1 固定点对齐表

| defense | param | rec_token_mean | agg_r1fm_r2fm | last_rec_status | 说明 |
| --- | ---: | ---: | ---: | --- | --- |
| none | n/a | 0.833506 | 141.710856 | ok | 泄露非常严重 |
| lrb | 0.2 | 0.000000 | 0.000000 | no_l1_candidates | DAGER 完全失效 |
| topk | 0.1 | 0.000000 | 0.000000 | no_l1_candidates | DAGER 完全失效 |
| compression | 8 | 0.000000 | 0.000000 | no_l1_candidates | DAGER 完全失效 |
| mixup | 0.3 | 0.929118 | 168.820917 | ok | 比 none 更差 |
| noise | 5e-4 | 0.237136 | 5.324657 | no_l1_candidates | 明显下降，但 token 未归零 |
| dpsgd | 5e-4 | 0.182152 | 1.500000 | no_l1_candidates | 明显下降，但 utility 很差 |
| soteria | 30 | 1.000000 | 192.676990 | ok | 当前结果明显恶化 |

### 5.2 当前固定点的综合排序

如果只看当前这一组固定点，那么可以把方法分成三层：

第一层：**privacy 和 utility 同时都比较强**

- `topk@0.1`
- `compression@8`

第二层：**privacy 很强，但 utility 代价过大**

- `lrb@0.2`

第三层：**要么 utility 不行，要么 privacy 不行**

- `noise@5e-4`
- `dpsgd@5e-4`
- `mixup@0.3`
- `soteria@30`

换句话说，**在当前已完成的固定点里，最强的经验 baseline 其实是 `topk@0.1`，其次是 `compression@8`。**

### 5.3 结果解释时的可比性注意事项

`docs/DEFENSE_BASELINES.md` 里明确提到：

> 旧版本 `dpsgd / soteria` 结果与当前实现**不可直接横向比较**。

因此本文对这两类结果的使用原则是：

- 对 `dpsgd`：主要把它当作“理论标准 baseline 的经验趋势参考”，即当前设置下要想把攻击压下去，utility 代价很大。
- 对 `soteria`：主要把它当作“当前迁移状态不稳定”的参考，而不是精确和 `lrb/topk/compression` 做逐项公平比较。

这不会影响本文的主结论，因为当前真正决定研究主线的，主要是：

- `LRB / topk / compression` 之间的 fixed-point utility 对比
- `LRB / topk / compression` 在 sweep 中暴露出的安全区间和优化空间

---

## 6. Sweep 视角下的更细结论

固定点表很有用，但还不够。真正更关键的是把 sweep 信息一起看，因为它会告诉我们“当前固定点是不是过保守了”。

### 6.1 LRB：DAGER 已经饱和，当前 utility 点偏重

`log/runs/test/summary_lrb.txt` 显示：

- `lrb@0.05` -> `rec_token_mean=0`
- `lrb@0.1` -> `rec_token_mean=0`
- `lrb@0.2` -> `rec_token_mean=0`
- `lrb@0.35` -> `rec_token_mean=0`
- `lrb@0.5` -> `rec_token_mean=0`

这说明对 DAGER 来说，`LRB` 在 `0.05 ~ 0.5` 的整个 sweep 区间都已经把攻击打到了 `no_l1_candidates`。

这件事的含义非常重要：

- 当前 `lrb@0.2` 并不是“刚好够用”的点。
- 对 DAGER 而言，`LRB` 的 privacy 早就饱和了。
- 后面真正需要优化的是 utility，而不是继续追求更低的 DAGER 指标。

也就是说，`lrb@0.2` 当前更像一个**过防御点**。

### 6.2 Top-k：安全区间很宽，当前 fixed point 仍偏保守

`log/runs/test/results_topk.md` 显示：

- `topk@0.01` -> `rec_token_mean=0`
- `topk@0.05` -> `rec_token_mean=0`
- `topk@0.1` -> `rec_token_mean=0`
- `topk@0.3` -> `rec_token_mean=0`
- `topk@0.5` -> `rec_token_mean=0.047395`
- `topk@0.7` -> `rec_token_mean=0.161375`
- `topk@0.9` -> `rec_token_mean=0.381985`

这说明：

- 对 DAGER 来说，`topk` 在 `0.1` 甚至 `0.3` 附近仍处于强安全区。
- 当前 utility 跑的是 `topk@0.1`，这未必是它 utility 最好的点。
- 如果后面补 `topk@0.3` 的 utility，很可能还能更接近 `none`，同时保持 DAGER=0。

因此，当前 `topk@0.1` 已经很强，但它可能还不是 `topk` 自己的最优点。

### 6.3 Compression：8-bit 很强，16-bit 很值得补跑 utility

`log/runs/test/results_compression.md` 显示：

- `compression@2` -> `rec_token_mean=0`，但本次只完成 `56/100` 后 SVD 失败，不能算稳定成功点
- `compression@4` -> `rec_token_mean=0`
- `compression@8` -> `rec_token_mean=0`
- `compression@16` -> `rec_token_mean=0`
- `compression@32` -> 基本退化为 `none`

这说明：

- `compression` 在 `4/8/16` 都能把当前 DAGER 打掉；`2bit` 需要先解决稳定性问题。
- 当前 utility 只跑了 `compression@8`。
- 从隐私 sweep 看，`compression@16` 是非常值得补跑的，因为它很可能能在保持 DAGER=0 的同时，进一步提升 utility。

因此，`compression@8` 现在虽然已经很强，但也不是它唯一值得看的点。

### 6.4 Noise：需要很大扰动才有效，代价明显

`log/runs/test/summary_noise_table.txt` 显示：

- `1e-6` -> `rec_token_mean=0.953319`
- `1e-5` -> `rec_token_mean=0.716408`
- `1e-4` -> `rec_token_mean=0.419720`
- `5e-4` -> `rec_token_mean=0.237136`
- `1e-3` -> `rec_token_mean=0.173061`

可见：

- 小噪声几乎没有真正解决泄露。
- 要到 `5e-4` 这个量级，DAGER 的 Rouge 才基本归零。
- 但这个点对应的 utility 已经下降到 `0.715979`。

因此 `noise` 更像是一个“能生效，但代价粗暴”的 baseline。

### 6.5 DP-SGD：理论强，但当前 utility 几乎不可接受

`log/runs/test/summary_dpsgd_table.txt` 显示：

- `1e-6` -> `rec_token_mean=0.966569`
- `1e-5` -> `rec_token_mean=0.823412`
- `1e-4` -> `rec_token_mean=0.488569`
- `5e-4` -> `rec_token_mean=0.182152`
- `1e-3` -> `rec_token_mean=0.077848`

而固定点 utility 结果是：

- `eval_accuracy=0.504205`
- `utility_drop=0.409021`

这说明：

- DP-SGD 在当前设置下不是“没效果”，而是“要到有效时 utility 代价太大”。
- 它仍然是一个重要理论 baseline，但很难成为这篇工作的经验最优点。

### 6.6 Mixup：utility 友好，但不是当前场景下的有效 privacy defense

`log/runs/test/results_mixup.md` 显示：

- `mixup@0.1` -> `rec_token_mean=0.882837`
- `mixup@0.3` -> `rec_token_mean=0.929118`
- `mixup@0.5` -> `rec_token_mean=0.891860`
- `mixup@1.0` -> `rec_token_mean=0.908417`
- `mixup@2.0` -> `rec_token_mean=0.885879`

与 `none=0.833506` 相比，`mixup` 在所有这些点上都更差。

因此当前可以明确说：

- `mixup` 在这套 FedSGD + DAGER + SST2 + GPT2 设定里，不适合作为强 privacy baseline。
- 它保住了 utility，但没有保住 privacy。

### 6.7 Soteria：当前结果不支持它作为主 baseline

当前 `soteria` 有三类信息：

1. `proxy` 结果在 `proxy_baselines_sst2_b2_gpt2_20260416_200409/results.md` 中报错：
   - `RuntimeError: inconsistent tensor size ...`

2. DAGER 结果在 `log/runs/test/results_soteria.md` 中明显恶化：
   - `soteria@10` -> `rec_token_mean=0.996217`
   - `soteria@30` -> `rec_token_mean=1.000000`
   - `soteria@50/70/90` -> `rec_token_mean=1.000000`

3. 用户当前反馈中，`soteria` 的 utility 训练明显比 `lrb` 更慢，而且截至整理时还没完整出结果。

因此，至少在当前阶段，可以保守地说：

- `soteria` 在这套 LLM 设置里迁移并不顺。
- 它既没有展现稳定的 proxy 行为，也没有展现好的 DAGER privacy。
- 即使 utility 之后跑完，也更像是“补 baseline 完整性”，而不是最值得投入的主方向。

---

## 7. 通过 Utility 可以得到的核心结论

这一部分是本文档最关键的结论整理。

### 7.1 结论一：当前 clean baseline 的泄露非常严重

`none` 的 DAGER 结果为：

- `rec_token_mean=0.833506`
- `agg_r1fm_r2fm=141.710856`

这说明在当前设定下，不加 defense 的 FedSGD 训练确实存在很严重的文本恢复风险，这一点与 `FL-LLM.md` 里的总体动机完全一致。

### 7.2 结论二：并不是所有“把攻击打掉”的 defense 都一样好

当前至少有三个方法在固定点上把 DAGER 打到了 `0`：

- `lrb@0.2`
- `topk@0.1`
- `compression@8`

但这三者的 utility 完全不同：

- `topk@0.1`：`acc=0.912462`
- `compression@8`：`acc=0.911315`
- `lrb@0.2`：`acc=0.821865`

因此，utility 告诉我们最重要的一件事是：

> “防住攻击”只是第一步，真正能区分 defense 质量的是在相近 privacy 下保留多少训练效用。

### 7.3 结论三：当前 `LRB@0.2` 是有效的，但显然偏重

对 `LRB` 来说，现在最明确的结论不是“它不行”，而是：

- 它对 DAGER 是非常有效的。
- 但 `0.2` 这个 operating point 明显过重。
- 它把太多可用于任务学习的信号也一起压掉了。

换句话说，当前问题不在“有没有 privacy”，而在“privacy 有余，utility 不足”。

### 7.4 结论四：当前最强经验对手不是 noise / dpsgd，而是 topk / compression

从已完成的 fixed-point utility 对比看，真正需要认真对待的强对手是：

- `topk@0.1`
- `compression@8`

如果后面论文只证明 `LRB` 比 `noise/dpsgd/mixup` 强，说服力是不够的。  
因为当前最强的经验 baseline 已经很明确，尤其是 `topk`。

### 7.5 结论五：proxy utility 有参考价值，但不足以做最终结论

当前 `proxy utility` 给人的直觉会是：

- `lrb` 虽然改得很猛，但验证精度即时变化不大；
- `dpsgd/noise` 的一步结果也似乎问题不大。

但 end-to-end utility 证明并不是这样。  
因此这组实验给出的直接启示是：

- proxy 更适合做快速筛选；
- 真正的结论必须靠完整训练 utility。

### 7.6 结论六：当前 fixed-point utility 还不是公平的“最优点对最优点”比较

这是很容易被忽略、但非常重要的一点。

从 sweep 看：

- `LRB` 在 `0.05~0.5` 都是 DAGER=0
- `topk` 在 `0.01~0.3` 都是 DAGER=0
- `compression` 在 `4/8/16` 都是 DAGER=0，`2bit` 本次失败不计入稳定点

但 utility 只跑了：

- `lrb@0.2`
- `topk@0.1`
- `compression@8`

所以当前 utility 对比本质上是：

> 每个方法各挑了一个固定点做对比，而不是每个方法都先各自调到“同等 privacy 下的最佳 utility”再比较。

这意味着：

- 现在可以得出“当前固定点谁更强”的结论；
- 但还不能得出“方法本质上谁一定更强”的最终结论。

---

## 8. 对后续研究的具体启示

### 8.1 对 LRB 的最直接启示：下一步不该继续追 DAGER=0，而该追回 utility

因为 `LRB` 整个 sweep 都已经是 `DAGER=0`，所以当前最该做的不是更强地压攻击，而是：

- 让 `LRB` 保留更多任务相关信号；
- 把 utility 拉回到接近 `topk/compression` 的水平；
- 同时保持跨攻击面的泛化能力。

这与 `FL-LLM.md` 中“recoverability bottleneck 而不是粗暴扰动”的主线是对齐的。

### 8.2 当前最应该做的是汇总公平 utility 点，而不是继续补同一批点

2026-05-02 时，原先最值得补跑的点已经完成：

- `lrb@0.35`：`eval_accuracy=0.868119`
- `lrb@0.5`：`eval_accuracy=0.892584`
- `topk@0.3`：`eval_accuracy=0.910933`
- `compression@16`：`eval_accuracy=0.909021`

新的直接结论是：

- `lrb@0.5` 明显优于 `lrb@0.2/0.35`，是当前 LRB utility-facing 最佳点；
- `topk@0.3` 没有超过 `topk@0.1`；
- `compression@16` 没有超过 `compression@8`；
- full-gradient DAGER 这一个攻击面上，topk/compression 的经验 tradeoff 仍强于 LRB。

换句话说，下一阶段最该做的是把这些结果正式整理成 Pareto 表/图，并把 LRB 的说服力转向机制消融和跨攻击面泛化。

### 8.3 如果 LRB 要作为主方法，必须证明的不只是 DAGER=0，而是跨攻击泛化

现在 `topk` 和 `compression` 的问题在于：

- 它们在当前 DAGER 上很强；
- 但它们并不是以 recoverability 结构建模为目标设计出来的。

而 `LRB` 的研究价值在于：

- 它更接近 `FL-LLM.md` 里提出的“通用 recoverability bottleneck”叙事；
- 理论上更有机会跨 `full-gradient / PEFT / partial-gradient` 三类泄露面起作用。

所以接下来如果要证明 `LRB` 值得做主方法，最关键的不是再证明一次它对 DAGER 有效，而是要补：

- PEFT leakage 结果
- partial-gradient leakage 结果
- 更公平 operating point 下的 utility 结果

### 8.4 LRB 当前还有明显的工程优化空间

`LRB` 当前表现出的工程问题包括：

- proxy `step_runtime_mean=8.347359`，明显偏高；
- `lrb@0.2` end-to-end 训练时间达到 `08:52:41`；
- `lrb@0.35` utility 训练时间方差很大；
- `lrb@0.5` 已改善到 `01:56:47`，但仍明显慢于 none。

这说明 `LRB` 当前除了“防御强度偏重”之外，还存在“实现开销偏高、稳定性一般”的问题。

因此后面除了调参，还建议检查：

- 敏感层统计是否每步重复做了过多计算
- 投影/池化实现是否有额外同步或低效张量操作
- 随机投影和噪声注入是否可以做缓存或预计算
- 是否存在某些 seed 下的异常性能路径

### 8.5 当前 baseline 结论对论文叙事的影响

如果按现在的数据写论文，最自然的叙事不是：

> “LRB 已经在所有方面都最好。”

而应该更诚实地写成：

> “当前 LRB 在 recoverability 抑制上非常强，`0.5` 已明显改善 utility，但在 full-gradient DAGER 这个单一攻击面上仍弱于 topk/compression；因此，后续核心问题不是继续调同一批 DAGER 点，而是通过机制消融、PEFT/LoRA 和 partial-gradient 证明 LRB 的结构性泛化价值。”

这会比直接宣称 `LRB` 已经最优更稳，也更符合现有数据。

---

## 9. 现阶段最值得保留的结论表述

如果要把这组结果压缩成几句比较适合汇报或写阶段总结的话，可以用下面这版：

### 9.1 一句话版本

当前 utility 结果表明：`LRB` 在 DAGER 上已经能稳定打到 `0` 恢复，且 `lrb@0.5` 已把 utility drop 从 `0.091361` 降到 `0.020642`；但在现有已完成的固定点中，`topk@0.1` 和 `compression@8` 仍是最强经验基线。LRB 下一阶段的重点应从“继续压 DAGER/继续补同类 utility 点”转向“机制消融与跨攻击泛化验证”。

### 9.2 稍详细版本

在 `SST2 + GPT2 + batch=2` 这组实验中，clean baseline 的泄露非常严重；`LRB@0.2/0.35/0.5`、`topk@0.1/0.3`、`compression@8/16` 都能把当前 DAGER 攻击打到 `0` 恢复，但它们的 utility 差异很大。`topk@0.1` 和 `compression@8` 几乎不损失准确率，`lrb@0.5` 虽明显好于 `0.2/0.35`，但仍有约 `2.1` 个点 accuracy drop。因此下一步最重要的是把这些点整理成 Pareto 表/图，并验证 `LRB` 是否能在 PEFT 和 partial-gradient 等更广攻击面上体现出比压缩类 baseline 更强的泛化优势。

---

## 10. 建议的下一步实验优先级

建议按下面顺序推进：

1. 把已完成的 `lrb@0.35/0.5`、`topk@0.3`、`compression@16` 整理成“同等 DAGER privacy 下的 utility Pareto 图”。
2. 跑 LRB 机制消融，至少补 `clip_only / proj_only / proj_clip / pool_full / rule_only / empirical_only / uniform_all_sensitive`。
3. 再补 `LRB / topk / compression` 在 PEFT leakage 和 partial-gradient attack 下的结果。
4. 单独排查 `LRB` 的运行时开销和方差来源。
5. `soteria` utility 跑完后补入附录，但不建议把它作为当前研究主线。

---

## 11. 最终判断

基于当前所有已完成结果，可以给出一个相对稳妥的阶段性判断：

- `LRB` 方向本身是成立的，因为它已经非常强地压低了 recoverability。
- `LRB@0.2` 不是当前最好的 utility operating point，`LRB@0.5` 已经把 drop 降到 `0.020642`。
- 在目前已完成的固定点里，`topk@0.1` 是最强经验点，`compression@8` 次之；`topk@0.3` 和 `compression@16` 没有反超。
- 真正值得做的下一步，不是继续增加更多弱 baseline，而是以 `lrb@0.5` 为主配置完成机制消融，并证明它在更广攻击面上的结构性优势。
