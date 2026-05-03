# 当前工作现状、问题分析与下一步计划（2026-04-27）

> 2026-05-02 更新：`n_inputs=100` 的完整 defense baseline sweep 已完成，详见 [DEFENSE_BASELINES_N100_ANALYSIS_20260502.md](./DEFENSE_BASELINES_N100_ANALYSIS_20260502.md)。本文中的部分 privacy 数值来自旧的 `log/runs/test/` 阶段性结果；最新口径是 `none rec_token_mean=0.833506, R1+R2=141.710856`，`lrb@0.05~0.5`、`topk@0.01~0.3`、`compression@4/8/16` 可稳定打到 0，`compression@2` 本次因 SVD 在 56/100 处失败。`lrb@0.35/0.5`、`topk@0.3`、`compression@16` utility 已补齐，下一步重心应改为 Pareto 汇总、LRB 消融、PEFT/LoRA 和 partial-gradient。

本文基于当前 `docs/`、`docs/参考/` 和 `log/runs/` 中的材料，对 `FedLLM / FL-LLM` 方向做一次阶段性梳理。重点不是重新复述已有文档，而是把初始 idea、utility 结果、DAGER/PEFT/LRB 文档、服务器日志和参考图中的论文推进标准统一到一条研究主线上。

## 1. 结论先行

当前工作已经从“复现 DAGER 攻击 + 枚举 baseline”推进到了一个更清晰的阶段：

- 研究问题已经成立：在 `SST2 + GPT2 + batch=2 + full fine-tuning` 设置下，不加防御时 DAGER 的文本恢复非常严重；最新 `n_inputs=100` 结果为 `rec_token_mean=0.833506`，`ROUGE-1 + ROUGE-2=141.710856`。
- 防御评测框架已经初步打通：`none / noise / dpsgd / topk / compression / mixup / soteria / lrb` 的 DAGER attack-time 对照、proxy utility、end-to-end utility 大多已经有日志。
- 当前最强的固定点经验 baseline 是 `topk@0.1` 和 `compression@8`，它们在 DAGER 上都把恢复打到 `0`，同时 utility 几乎不掉。
- `LRB@0.2` 的 privacy 很强，但 fixed-point utility 明显偏重：`eval_accuracy=0.821865`，相对 `none` 掉 `0.091361`，训练时间和方差也偏大。
- 因此，现在不能写成“LRB 已经全面超过所有 baseline”。更稳的表述是：`LRB` 证明了结构化 recoverability bottleneck 路线有效，但当前 operating point 不是最优，需要补公平 utility 点和跨攻击泛化证据。
- 下一步的核心不是继续加弱 baseline，而是围绕 `LRB / topk / compression` 做同等 privacy 下的 utility Pareto 对比，并把证据扩展到 PEFT leakage 与 partial-gradient leakage。

一句话概括：

> 这个方向值得继续做，但论文主线要从“我能把 DAGER 打掉”升级为“我能解释并优化中间更新的 recoverability，在相近隐私水平下取得更好的泛化性和可接受 utility”。

## 2. 当前研究主线是否清楚

### 2.1 初始 idea 的主问题

`docs/FL-LLM.md` 和 `docs/1.md` 的核心问题是一致的：

> 大模型联邦训练中，服务器能否从客户端上传的梯度/更新中恢复原始文本？如果能，能否设计一种通用机制，从根本上降低中间更新信息的可恢复性？

这个问题的价值来自三个点：

1. 任务有现实场景：联邦 LLM / PEFT 微调中共享梯度或更新，而不是共享原始数据。
2. 能力瓶颈明确：不是一般隐私担忧，而是 gradient/update 中的结构性 recoverability。
3. 防御目标明确：不只防一个攻击代码，而是降低“中间信息 -> 原始样本”的恢复能力。

这与参考图 1 的五阶段逻辑基本对齐：

- 阶段 1：任务价值是 LLM/FedSGD/PEFT 训练中的数据恢复风险。
- 阶段 2：能力卡在梯度/更新空间泄露了 token 级几何结构。
- 阶段 3：旧机制绕不开，是因为 noise / DP / compression 大多不是以 recoverability 为目标。
- 阶段 4：新机制落在 `LRB / HLRB`，尤其是 layer-wise sensitivity、low-resolution public-subspace reconstruction、residual-space noise。
- 阶段 5：当前还没最终交付“方法全面优于 baseline”，但已经有一个很强的阶段性结果和清晰补证路径。

### 2.2 威胁模型覆盖

初始文档规划了三类攻击：

| 攻击面 | 代表方法 | 当前状态 |
| --- | --- | --- |
| Full-gradient inversion | DAGER / LAMP | DAGER 已成为当前主实验；LAMP 目前更多是背景 baseline，还未成为当前实证主表核心 |
| PEFT leakage | PEFTLeak / ReCIT / LoRA 更新恢复 | 框架已有 `PEFT / LoRA eval-first` 能力，但缺少完整结果表 |
| Partial-gradient / layer-level leakage | Partial Transformer Gradients | 文档已有威胁建模，但还缺显式攻击入口和结果 |

所以当前证据主要支撑第一类攻击面。第二、三类攻击面是下一阶段论文说服力的关键。

## 3. 当前系统和工程框架状态

### 3.1 DAGER defense baseline 框架

已经具备：

- `scripts/defense_baselines.sh`：跑 DAGER attack-time baseline sweep。
- `utils/defenses.py`：统一接入 `none / noise / dpsgd / topk / compression / soteria / mixup / dager / lrb`。
- `utils/lrb_defense.py`：实现当前 LRB。
- `scripts/collect_experiment_logs.py`：汇总 attack、proxy utility、training utility 和 tradeoff。
- `log/runs/test/` 与 `log/runs/defense_baselines_sst2_b2_gpt2_20260501_010024/`：保存了整理后的 DAGER privacy 结果。

当前 DAGER 主结果的最重要事实是：

| defense | param | DAGER 恢复状态 |
| --- | ---: | --- |
| none | n/a | `rec_token_mean=0.833506`，`R1+R2=141.710856`，泄露严重 |
| lrb | 0.05-0.5 | 全部 `rec_token_mean=0` |
| topk | 0.01-0.3 | 全部 `rec_token_mean=0` |
| compression | 4/8/16 | 全部 `rec_token_mean=0` |
| compression | 2 | 指标为 0，但本次只完成 56/100 后 SVD 失败，不算稳定成功点 |
| noise | 5e-4/1e-3 | ROUGE 基本归零，但 token 恢复仍有残留 |
| dpsgd | 5e-4 | 明显降低，但未归零 |
| mixup | 0.1-2.0 | 比 none 更差，不适合当 privacy defense |
| soteria | 10-90 | 明显恶化，不适合当前主线 |

### 3.2 Utility 框架

已经具备：

- `scripts/proxy_utility.py`：one-step proxy utility。
- `scripts/proxy_baselines.sh`：批量 proxy。
- `scripts/utility_baselines.sh`：end-to-end training utility。
- `train.py` 训练侧已经能通过统一 `--defense` 路径跑 full training defense。

需要注意一个文档同步问题：

- `docs/LRB_方法详解.md` 中仍写着当前 LRB 不是训练时 hook。
- 但 `docs/LRB_实验任务清单.md`、`train.py`、`scripts/utility_baselines.sh` 和 2026-04-26 的 utility 日志表明，full training utility 已经可以跑 `lrb/topk/compression/noise/dpsgd/mixup`。
- 更准确的当前状态应写为：full training defense 已接通并产出结果；LoRA 训练期 defense 未完整接通；LRB 的 forward-side representation bottleneck 也未接通。

### 3.3 PEFT / LoRA 框架

`docs/PEFT_EVAL.md` 说明当前是一个 LoRA eval-first 框架：

已经支持：

- 加载已有本地 `.pt/.pth` LoRA checkpoint。
- GPT-2 / Llama 家族 LoRA 攻击评测。
- LoRA 下比较 `none / noise / topk / compression / lrb`。
- 结果中记录 `train_method` 和 `lora_r`，避免 full 和 LoRA 混表。

尚不支持：

- LoRA 训练期防御。
- BERT PEFT。
- Adapter / IA3 / Prefix Tuning。
- LoRA 下 `dpsgd / mixup / soteria / dager`。

这意味着 PEFT 是下一阶段最自然的扩展，但当前还不是已完成证据。

## 4. 当前实验结果分析

### 4.1 End-to-End Utility 主表

当前最重要的 utility 结果来自 `log/runs/utility260426/utility_results_*.md`，设置为：

- 数据集：`sst2`
- 模型：`gpt2-ft-rt`
- batch size：`2`
- 训练方式：`full`
- epoch：`1`
- seeds：`101 / 202 / 303`

| defense | param | eval_accuracy | eval_macro_f1 | utility_drop | total_train_time | 结论 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| none | n/a | 0.913226 | 0.913184 | 0.000000 | 00:42:51 | clean anchor |
| topk | 0.1 | 0.912462 | 0.912430 | 0.000764 | 06:03:08 | 当前 fixed-point 最强 |
| compression | 8 | 0.911315 | 0.911290 | 0.001911 | 07:07:37 | 很强，仅略低于 topk |
| mixup | 0.3 | 0.910933 | 0.910906 | 0.002293 | 00:48:40 | utility 好，但 privacy 失败 |
| lrb | 0.2 | 0.821865 | 0.821360 | 0.091361 | 08:52:41 | privacy 强，但过防御 |
| noise | 5e-4 | 0.715979 | 0.715617 | 0.197247 | 01:54:29 | 粗暴有效，utility 代价大 |
| dpsgd | 5e-4 | 0.504205 | 0.366347 | 0.409021 | 03:29:31 | 当前设置几乎不可用 |

这里最关键的发现不是谁 accuracy 高，而是：

> `LRB@0.2`、`topk@0.1`、`compression@8` 都能把 DAGER 打到 `0`，但 utility 差异很大。

这说明“privacy success”不足以支撑方法优越性。论文必须做 privacy-utility Pareto。

### 4.2 Proxy Utility 的启示

`proxy_baselines_sst2_b2_gpt2_20260416_200409/results.md` 显示：

| defense | param | grad_cosine_mean | norm_retention_mean | step_runtime_mean | proxy 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| none | n/a | 1.000000 | 1.000000 | 1.065636 | clean |
| compression | 8 | 0.969237 | 1.034058 | 1.108627 | 几乎保留更新方向 |
| topk | 0.1 | 0.945117 | 0.945117 | 1.679956 | 更新方向保留很好 |
| mixup | 0.3 | 0.525846 | 22.249394 | 1.023529 | proxy 不坏，但 privacy 不行 |
| noise | 5e-4 | 0.490102 | 12.943462 | 1.011835 | proxy 难以预测长期训练 |
| lrb | 0.2 | 0.264516 | 0.083383 | 8.347359 | 改动强，开销大 |
| dpsgd | 5e-4 | 0.111590 | 6.456979 | 1.176916 | 改动强，长期 utility 更差 |
| soteria | 30 | n/a | n/a | n/a | proxy 失败，tensor size 不一致 |

这说明 proxy utility 只能初筛，不能替代完整训练：

- `lrb/noise/dpsgd` 在 proxy 里验证精度即时变化接近 0，但完整训练后分别掉约 9.1、19.7、40.9 个点。
- 对会持续改变训练轨迹的 defense，proxy 会低估长期代价。

### 4.3 DAGER Privacy 与 Utility 对齐

当前 fixed-point privacy-utility 分层如下：

第一层：privacy 和 utility 同时强

- `topk@0.1`
- `compression@8`

第二层：privacy 强但 utility 代价过大

- `lrb@0.2`

第三层：privacy 或 utility 不成立

- `mixup@0.3`：utility 好，privacy 失败。
- `noise@5e-4`：privacy 有效但 utility 大幅下降。
- `dpsgd@5e-4`：privacy 有效但 utility 几乎不可接受。
- `soteria@30`：当前实现下 privacy 反而更差。

### 4.4 Sweep 结果说明 fixed point 不公平

当前 utility 对比还不是“每个方法最优点对最优点”，因为 DAGER sweep 显示：

- `LRB` 在 `0.05 / 0.1 / 0.2 / 0.35 / 0.5` 全部 DAGER=0。
- `topk` 在 `0.01 / 0.05 / 0.1 / 0.3` 全部 DAGER=0，`0.5` 仍然很低。
- `compression` 在 `2 / 4 / 8 / 16` 全部 DAGER=0，`32` 退化为 none。

但当前 utility 只跑了：

- `lrb@0.2`
- `topk@0.1`
- `compression@8`

因此现在只能说：

> 当前 fixed points 下，topk/compression 比 LRB 更优；还不能说方法本质上 topk/compression 一定强于 LRB。

下一步必须补：

- `lrb@0.35`
- `lrb@0.5`
- `topk@0.3`
- `compression@16`

并画同等 DAGER privacy 下的 utility Pareto 图。

## 5. 对 LRB 当前状态的判断

### 5.1 方法方向成立

LRB 的方向与原始 idea 高度一致：

- 原始 idea 不是要做“加噪 baseline”，而是要降低中间更新信息的可恢复性。
- 当前 LRB 正是一个 layer-wise recoverability bottleneck。
- 它通过敏感度估计、按层裁剪、低分辨率公共表示、残差方向加噪来抑制样本细节。

相比 `noise/dpsgd/topk/compression`，它的叙事优势是：

- `noise/dpsgd` 是扰动/隐私标准系；
- `topk/compression` 是通信压缩系；
- `LRB` 是显式以 recoverability 为目标的结构化防御。

### 5.2 当前证据不足以证明它已最优

当前 `LRB@0.2` 的问题也很明确：

- Utility 明显掉：`0.913226 -> 0.821865`。
- 训练时间明显增加：平均 `08:52:41`，且 seed 303 跑到 `16:50:52`。
- proxy runtime 也高：`step_runtime_mean=8.347359`。
- 当前只在 DAGER full-gradient setting 上有强证据，PEFT 和 partial-gradient 证据不足。

因此，当前论文表述要避免：

> LRB 全面 SOTA。

更适合写成：

> LRB 在 recoverability 抑制上非常强，但固定点过重；它的真正价值需要通过更公平 operating point 和跨攻击面泛化来证明。

### 5.3 当前最强竞争者是谁

不是 `noise / dpsgd / mixup / soteria`，而是：

- `topk@0.1`
- `compression@8`
- 已补的 `topk@0.3 / compression@16` 没有超过上述两个点，但仍应放进 Pareto 表

这对论文叙事非常重要。如果论文只和 noise/dpsgd 比，就会显得避开了最强对手。

## 6. 对照参考图的论文缺口诊断

### 6.1 五阶段逻辑表

| 阶段 | 当前完成度 | 主要缺口 |
| --- | --- | --- |
| 1. 这件事凭什么值得做 | 已基本完成 | 需要补一篇任务价值文献和一篇性能/泄露基线文献作为 introduction anchor |
| 2. 能力卡在哪个环节 | 已有 DAGER 机制解释 | 需要把“梯度子空间暴露 token span”进一步转成我们自己的 problem statement |
| 3. 别人为什么绕不开 | 部分完成 | 需要更明确说明 topk/compression 为什么在 DAGER 上强但不构成通用防御 |
| 4. 新机制如何落地 | 已有 LRB 代码和文档 | 需要把当前代码机制与 HLRB 愿景分清楚，避免写过头 |
| 5. 最终交付什么 | 未完成 | 需要方法名、量化提升、机制性结论、Pareto 图和跨攻击泛化证据 |

### 6.2 图表类型清单

结合参考图 2，目前最该准备的图表是：

1. Main Results 对比表  
   内容：`none / noise / dpsgd / mixup / soteria / topk / compression / lrb` 的 DAGER privacy + utility 联表。

2. Privacy-Utility Pareto 图  
   横轴：`utility_drop` 或 `eval_accuracy`；纵轴：`rec_token_mean` / `privacy_score` / `ROUGE-1+2`。  
   重点标出 `LRB / topk / compression` 的多个 operating points。

3. LRB Ablation 表  
   至少包括：identity-LRB、clip-only、projection-only、projection+noise、pool vs signed_pool、empirical_weight=0 vs 0.6。

4. 参数敏感性曲线  
   对 `lrb keep_ratio_sensitive`、`topk ratio`、`compression bits` 分别画 privacy 和 utility。

5. Complexity / runtime 图  
   因为当前 `LRB` 开销是实质问题，必须正面展示并优化。

6. Transfer 表  
   数据集/backbone/PEFT/partial-gradient 结果。

### 6.3 Gap-Novelty-Contribution 映射

当前可以构造三条论文贡献映射：

| 映射 | Gap | Novelty | Contribution |
| --- | --- | --- | --- |
| 主问题 | 现有防御多是加噪、DP 或通信压缩，不直接建模更新的 recoverability | 提出 layer-wise recoverability bottleneck | 在 DAGER full-gradient 场景下显著压低恢复 |
| 衍生问题 | 同样 privacy 下 utility 差异巨大，固定点比较不公平 | 用 Pareto/operating point 视角重新评估防御 | 证明 privacy success 不等于好 defense，找出公平对比点 |
| 分析贡献 | topk/compression 在 DAGER 上异常强，但机制不是隐私中心 | 分析压缩类方法在 DAGER 下的有效边界，并与 LRB 对比 | 为 LRB 的跨攻击泛化实验提供必要动机 |

当前第三条尤其关键：如果 topk/compression 已经很强，LRB 必须证明自己在“更一般攻击面”上有价值。

## 7. 下一步计划

### 7.1 最高优先级：汇总公平 utility 点

目标：把已经补齐的 fixed-point 和 sensitivity utility 结果整理成同等 privacy 下的 Pareto 对比。

已完成的关键点包括：

- `lrb@0.35`：`eval_accuracy=0.868119`
- `lrb@0.5`：`eval_accuracy=0.892584`
- `topk@0.3`：`eval_accuracy=0.910933`
- `compression@16`：`eval_accuracy=0.909021`

成功标准：

- 画出 `LRB / topk / compression` 的 Pareto 图。
- 明确写出：`lrb@0.5` 已把 utility drop 从 `0.091361` 降到 `0.020642`，但当前仍未超过 `topk@0.1` / `compression@8`。

### 7.2 第二优先级：做 LRB 机制消融

必须补，不然方法像“调参黑箱”。

建议实验：

- identity-LRB：证明代码分支没有引入额外偏差。
- clip-only：验证裁剪本身是否足够。
- projection-only：验证主效应是否来自 bottleneck。
- projection + noise：验证残差噪声增益。
- `pool` vs `signed_pool`：验证公共随机子空间叙事。
- `empirical_weight=0` vs `0.6`：验证动态校准是否有用。

预期最好能证明：

> LRB 的主效应来自低恢复性投影/瓶颈，而不是单纯裁剪或加噪。

### 7.3 第三优先级：PEFT / LoRA

PEFT 是原始 idea 里很重要的现实场景，目前框架已经具备 eval-first 能力，但结果未补。

最小结果集：

- LoRA + none
- LoRA + lrb
- LoRA + topk
- LoRA + compression

重点问题：

> 在只共享 LoRA 更新时，LRB 是否还能保持比压缩类方法更稳定的 recoverability 抑制？

如果 LoRA 下 topk/compression 也很强，LRB 的价值仍然要靠跨攻击面和机制解释来支撑。

### 7.4 第四优先级：partial-gradient / layer-level leakage

当前还缺入口。建议补：

- `--gradient_layer_subset`
- `--gradient_param_filter`

最小设置：

- first block only
- first 2 blocks
- last 2 blocks
- attention q/k/v only
- LoRA params only

这部分一旦补上，可以直接对应原始 idea 的第三类威胁。

### 7.5 第五优先级：优化 LRB runtime

当前 LRB 的运行开销会被审稿人关注。建议排查：

- 每 step 是否重复生成 signed mask。
- pooling/interpolation 是否有不必要 CPU/GPU 同步。
- calibration sampling 是否每层每步过多。
- 是否能缓存 layer-wise mask、shape plan、随机符号。
- 是否能减少 `defense_lrb_calibration_samples` 或做低频校准。

目标不是立刻做到最快，而是至少解释开销来自哪里，并给出可接受版本。

## 8. 建议论文叙事

当前最稳的叙事不是“我们提出一个已经完胜所有方法的防御”，而是：

1. 先证明问题严重：DAGER 在 clean FedSGD/LLM setting 下恢复严重。
2. 再证明普通 defense 不够：noise/dpsgd utility 代价大；mixup/soteria 不稳定；topk/compression 虽强但不是以隐私 recoverability 为中心。
3. 提出 LRB：一个 layer-wise recoverability bottleneck，显式针对最可恢复层和细节方向。
4. 展示 LRB 能强力阻断 DAGER，但承认 fixed point 过重。
5. 通过 Pareto、消融、PEFT、partial-gradient 证明它的结构性价值。

可以使用的阶段性表述：

> In the current SST2 + GPT2 FedSGD setting, LRB strongly suppresses DAGER recoverability, reducing token recovery to zero across a wide parameter range. The `0.5` operating point improves utility substantially over `0.2/0.35`, but topk/compression still offer stronger empirical tradeoffs on full-gradient DAGER. The next step is therefore to use `lrb@0.5` for mechanism ablation and verify whether the recoverability bottleneck generalizes beyond full-gradient DAGER to PEFT and partial-gradient leakage.

中文版本：

> 当前结果说明，LRB 方向是成立的，因为它已经在 DAGER 上稳定压低 recoverability；`0.5` 是当前 LRB 最好的 utility 点，但尚不能证明它在 full-gradient DAGER 的隐私-效用权衡上优于 topk/compression。下一阶段应从“继续压 DAGER”转向“以 `lrb@0.5` 做机制消融，并验证 PEFT 与 partial-gradient 下的结构性泛化优势”。

## 9. 近期执行清单

建议按下面顺序推进：

1. 统一文档口径：更新 `LRB_方法详解.md` 中关于训练期 hook 的过时描述。
2. 汇总已完成的 `lrb@0.35/0.5`、`topk@0.3`、`compression@16` 到 Pareto 表和 Pareto 图。
3. 跑 LRB ablation：identity、clip-only、projection-only、pool/signed_pool、empirical_weight，并补 proxy/train utility。
4. 跑 LoRA eval-first：`none / lrb@0.5 / topk / compression`。
5. 设计 partial-gradient 接口。
6. 排查 LRB runtime 和 seed 方差。
7. 将结果整理成三张主表：main results、ablation、transfer。
8. 再决定是否推进 forward-side HLRB。

## 10. 最终阶段判断

当前工作的现状可以概括为：

- 方向成立：问题真实，DAGER 泄露严重，LRB 能有效降低 recoverability。
- 证据初步但不完整：DAGER full-gradient 证据强，utility 证据已开始暴露真实 tradeoff，PEFT/partial-gradient 证据不足。
- 主方法有潜力但不能过度宣称：`LRB@0.5` 是当前 LRB 最佳 utility 点，但仍需通过 Pareto、消融和跨攻击面验证来证明结构性价值。
- 最强对手明确：`topk/compression` 是真正需要认真比较的 strong baseline。
- 下一步路线清晰：fair utility points -> LRB ablation -> PEFT/partial-gradient -> runtime -> paper tables。

如果把这个阶段作为科研进展汇报，可以说：

> 我们已经完成了 DAGER 主攻击和多类防御 baseline 的评测闭环，并发现 clean FedSGD 设置下泄露严重。LRB 作为结构化 recoverability bottleneck 能稳定阻断 DAGER，但当前默认点 utility 代价偏大；同时 topk 和 compression 在 DAGER 上构成强经验基线。因此下一阶段将聚焦同等 privacy 下的 utility Pareto、LRB 机制消融，以及 PEFT/partial-gradient 的泛化验证，从而把当前“DAGER 上有效”的观察推进为“通用中间更新防御”的论文主线。
