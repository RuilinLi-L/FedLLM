# FedLLM / Projection-LRB 阶段性工作进展与论文整理稿

基于现有实验、初步思路模板与 agent 建议整理；不包含 Overleaf / LaTeX 生成

| 项目 | 内容 |
| --- | --- |
| 整理日期 | 2026-05-05 / 2026-05-06 更新 |
| 文档定位 | 阶段性研究进展与论文前期整理稿，不生成 Overleaf / LaTeX |
| 主数据集 / 模型 | SST2 / GPT-2 fine-tuned checkpoint |
| 主攻击面 | Full-gradient DAGER, batch=2, n_inputs=100 |
| 主方法身份 | Projection-LRB / LRB-lite, 由 low-resolution signed projection 消融证据支持 |
| 主要日志来源 | defense_baselines_sst2_b2_gpt2_20260501_010024; utility260426; lrb_ablation_sst2_b2_gpt2_k0.5_20260501_004737 |

| 核心结论先行 当前工作已经形成 DAGER 攻击、防御 baseline、utility、LRB 机制消融的闭环证据，但强证据边界主要是 SST2+GPT2+batch=2 的 full-gradient DAGER。 Clean FedSGD 泄露严重；LRB、topk、compression 均可在若干点上把当前 DAGER 恢复打到 0，但这不等价于形式化隐私保证。 仅看 full-gradient DAGER，topk@0.1 与 compression@8 仍是更强经验 tradeoff；full_lrb@0.5 已改善 utility，但仍偏重。 消融显示 projection bottleneck 是主效应：proj_only@0.5 在单次消融运行中达到 DAGER=0 并保持 clean-level utility，因此下一阶段应把主方法收束为 Projection-LRB。 |
| --- |

## 一、文档定位与当前工作状态

本文将现有阶段性汇总进一步整理成可继续写论文的研究稿。它保留已有实验事实，但把主线从泛称 LRB 调整为 Projection-LRB / LRB-lite，并按照初步思路模板补上关键的 Gap - Novelty - Contribution - Evidence 映射。

截至 2026-05-05，项目已经从复现 DAGER 与枚举 baseline 推进到更清晰的阶段：full-gradient DAGER 风险已被量化，多个 defense 的 privacy/utility 已经有结果，LRB 的主要组件也通过完整消融被拆开。当前最重要的变化是：主线不再是证明 DAGER 可以被打掉，而是证明哪一种 recoverability bottleneck 能在相近 privacy 下保留更多效用。

| 审稿口径下的证据边界 已经完成并可较强支撑：SST2 + GPT2 fine-tuned checkpoint + batch=2 + n_inputs=100 的 full-gradient DAGER attack-time privacy；full training utility fixed points；一次完整 LRB ablation。 尚不能直接外推：LoRA/PEFT 训练期 defense、partial-gradient attack、跨数据集/backbone、多 seed 统计显著性、adaptive attack 与严格 DP accounting。 因此当前论文主张应写成：Projection-LRB 是由 full-gradient DAGER 与消融结果支持的主方法候选，而不是已经在所有 FL-LLM 场景全面优于压缩类 baseline。 |
| --- |

## 二、背景与动机

大模型微调正在从集中式数据汇聚转向联邦学习、跨机构协作训练和 PEFT adapter 共享。这样的设置看起来天然适合隐私场景：医疗文本、金融评论、企业客服、用户输入日志等原始文本不离开本地，只上传梯度、模型更新或轻量参数更新。但梯度反演与文本恢复攻击表明，不上传原文并不等价于原文不可恢复。

本项目当前 clean DAGER 结果给出了直接动机：在 SST2 + GPT2 + batch=2 + n_inputs=100 的设置下，未防御 FedSGD 梯度的 rec_token_mean 达到 0.833506，ROUGE-1+ROUGE-2 达到 141.710856。这说明攻击者并不是只能恢复模糊主题，而是能够恢复大量 token 级文本信息。

进一步的动机来自 privacy-utility 冲突：简单把 DAGER 打到 0 并不困难，困难在于保留训练效用。当前实验显示，完整 LRB 能稳定抑制 DAGER，但 utility 代价偏高；topk/compression 在 full-gradient DAGER 下很强，却更像通信压缩副作用而不是针对文本 recoverability 的机制解释。

## 三、前人工作局限性与创新位置

已有工作为本项目提供了攻击、隐私和压缩三类重要参照，但各自都留下了适合本工作切入的缺口。当前实验也印证了这些局限：强扰动方法通常牺牲 utility，通信压缩方法在当前 DAGER 下表现强但机制目标不完全对齐，表示扰动和训练侧混合方法在 GPT2/FedSGD 设置中并不稳定。

| 方向 | 代表思路 | 当前局限 | 对本工作的启发 |
| --- | --- | --- | --- |
| 梯度/文本恢复攻击 | DLG, Inverting Gradients, TAG, LAMP, DAGER | 支撑 shared gradients / LLM updates 可能泄露文本；DAGER 是当前主攻击框架。 |   |
| 通用隐私/扰动 | DP-SGD, Gaussian noise | 理论或简单扰动 baseline；当前实验显示 utility 代价较大，且没有在本文中做完整 DP accounting。 |   |
| 通信压缩 baseline | Top-k sparsification, Deep Gradient Compression, QSGD | 当前 full-gradient DAGER 下的强经验对手；应正面比较，但不要把它们写成隐私中心设计。 |   |
| 表示/训练侧防御 | Soteria, mixup | 提供覆盖性 baseline；当前 GPT2/FedSGD 设置下 privacy 表现不稳定或失败。 |   |
| PEFT / partial-gradient 场景 | LoRA, Gradient Inversion Attacks on PEFT, ReCIT, Partial Transformer Gradients | 用于动机和后续泛化实验，不写成已完成结果。 |   |

## 四、关键映射表：Gap - Novelty - Contribution - Evidence

这是模板中最关键、此前缺失的映射表。它把研究瓶颈、创新动作、贡献证据和当前实验事实对应起来，便于后续直接转成 Introduction 的 contribution 段或论文主表叙事。

| Gap / 瓶颈 | Novelty / 创新动作 | Contribution / 贡献 | Evidence / 已有证据 |
| --- | --- | --- | --- |
| FL-LLM 中间更新存在文本 recoverability 风险 | 将防御目标从 generic perturbation 转为 suppress recoverability bottleneck | 把问题从是否共享原文推进到共享更新是否携带可恢复文本结构 | clean DAGER rec_token_mean=0.833506, R1+R2=141.710856 |
| topk/compression 在 full-gradient DAGER 下很强，但机制目标偏通信压缩 | Projection-LRB 用 layer-wise low-resolution signed projection 建立机制性瓶颈 | 提供区别于少传/量化的隐私机制叙事，并保留与强 baseline 的正面对比 | topk@0.1 和 compression@8 当前 tradeoff 很强；proj_only@0.5 显示新方法空间 |
| full_lrb 虽能强防御，但 clipping/noise 不是当前主效应且 utility 偏重 | 通过 ablation 将主方法从 full_lrb 收束为 projection-only / Projection-LRB | 避免把过防御配置写成最终方法，使论文主张与消融证据一致 | clip_only 几乎不阻断恢复；proj_only@0.5 DAGER=0 且 clean-level utility；full_lrb@0.5 drop=0.020642 |

## 五、希望解决的问题与应用意义

本工作希望解决的不是单一实验点上的攻击是否成功问题，而是 FL-LLM 中间更新如何在可训练与不可恢复之间取得平衡的问题。研究问题应拆成风险成立、防御有效、效用可接受、机制清楚和场景泛化五个层次。

| 问题层次 | 要回答的问题 | 当前已有依据 | 下一步补强 |
| --- | --- | --- | --- |
| 风险是否成立 | 客户端只上传梯度/更新时，攻击者是否仍能恢复文本？ | clean DAGER rec_token_mean=0.833506, R1+R2=141.710856 | 跨数据集/backbone 复核 |
| 防御是否有效 | 哪些方法能把 token recovery 和 ROUGE recovery 压到接近 0？ | LRB, topk, compression 多个点可达到 DAGER=0 | 说明 DAGER=0 的边界，不等同形式化安全 |
| 效用是否可接受 | 同等 privacy 下哪个 defense 保留最多 task accuracy？ | topk@0.1 与 compression@8 当前 fixed points 很强 | 把 Projection-LRB 纳入 main result 和多 seed |
| 机制是否清楚 | 防御效果来自 clipping, projection, noise 还是 layer sensitivity？ | proj_only@0.5 DAGER=0 且 accuracy=0.915520 | keep-ratio sweep 与 projection-only 细消融 |
| 场景是否泛化 | 是否适用于 PEFT/LoRA, partial-gradient 和真实联邦微调？ | PEFT eval-first 框架已有，partial-gradient 威胁已建模 | 补 LoRA 训练期 defense 和 partial-gradient 攻击入口 |

该问题对应的真实场景是多方拥有敏感文本、但又需要共同微调或适配语言模型的协作训练，例如医院之间联合训练临床文本分类模型、金融机构联合优化舆情或风控文本模型、企业之间共享客服/工单语义能力，以及移动端或边缘端更新个性化语言模型。

## 六、威胁模型与评价对象

本工作关注大模型联邦训练或分布式微调过程中由中间更新信息导致的数据泄露。客户端不上传原始文本，但会上传梯度、模型更新或 PEFT adapter 更新；攻击者可能据此恢复训练样本。关键不在于数据有没有直接共享，而在于共享更新本身是否携带足以恢复原始文本的高分辨率信息。

| 威胁模型要素 | 当前实验默认假设 | 审稿风险 / 后续补充 |
| --- | --- | --- |
| 攻击者能力 | 半诚实服务端或聚合方可观察单轮 full-gradient 更新，并运行 DAGER 类白盒恢复攻击 | 不是 secure aggregation 后只见总和的场景；多客户端聚合需另设实验 |
| 攻击者知识 | 知道模型结构、当前权重、tokenizer、任务形式和 batch size | label / batch size / client sampling 未知时需敏感性分析 |
| 被保护对象 | 客户端本地 batch 中的原始文本 token / sequence | rec_token_mean 与 ROUGE 衡量文本内容恢复，不是 DP privacy loss |
| 当前主攻击面 | SST2 + GPT2 fine-tuned checkpoint + batch=2 + n_inputs=100 的 full-gradient DAGER | PEFT/partial/backbone transfer 不应写成已完成结论 |
| 防御目标 | 降低中间更新 recoverability，同时保持 downstream accuracy / macro-F1 | 与 topk/compression 在相同 attack budget 和 privacy target 下比较 |

## 七、Projection-LRB 方法机制

根据当前消融证据，文档中的主方法应从泛称 LRB 收束为 Projection-LRB：一个面向 recoverability 的 layer-wise gradient projection defense。full_lrb 仍作为强防御 / 过防御对照保留，用来说明 clipping 与 residual-space noise 并不是当前 full-gradient DAGER 设置下的必要主效应。

更适合论文的方法表述是：对每层梯度 g_l，先根据结构先验与当前 batch 梯度统计估计敏感度 s_l，再用 keep ratio k_l 控制低分辨率 signed_pool 投影 P_{k_l}(g_l)。Projection-LRB 的核心更新可写作 \tilde{g}_l=P_{k_l}(g_l)，其中高敏感层使用更低分辨率 bottleneck；full_lrb 则在此基础上额外加入 clipping 与 residual-space noise。

| 模块 | 当前实现 | 核心作用 | 消融关注点 |
| --- | --- | --- | --- |
| Layer-wise sensitivity | 结构先验 + 当前 batch 梯度统计，默认 empirical_weight=0.6 | 识别更可能泄露 token 细节的层 | rule_only / empirical_only / uniform_all_sensitive |
| Low-resolution signed_pool projection | 随机符号翻转后做 pooling / interpolation 低分辨率重建 | Projection-LRB 主体；保留粗结构，移除高分辨率样本细节 | proj_only / proj_clip / pool_full |
| Layer-wise clipping | 按所有层范数中位数设置每层裁剪阈值 | full_lrb 附加约束；限制个别层大幅信息流 | clip_only / proj_clip |
| Residual-space noise | 噪声主要加在低分辨率表示之外的残差方向 | full_lrb 强防御组件；进一步污染攻击依赖的细节方向 | full_lrb vs proj_only |

| 推荐当前方法表述 Projection-LRB is a layer-wise adaptive gradient projection defense that suppresses recoverability by mapping sensitive-layer gradients into a low-resolution signed pooling subspace. full_lrb 可作为包含 clipping 与 residual-space noise 的强防御对照；当前不建议把 full_lrb 写成最终主方法。 |
| --- |

## 八、实验设置与指标

| 实验块 | 日志目录 / 文档 | 关键设置 | 用途 |
| --- | --- | --- | --- |
| DAGER defense baselines | log/runs/defense_baselines_sst2_b2_gpt2_20260501_010024 | sst2, gpt2, ./models/gpt2-ft-rt, batch=2, n_inputs=100 | 量化 clean 泄露与各 defense attack-time privacy |
| Utility focused runs | log/runs/utility260426 | full training, epoch=1, seeds=101/202/303 | 评估 defense 对 accuracy/F1/loss/runtime 的影响 |
| LRB ablation | log/runs/lrb_ablation_sst2_b2_gpt2_k0.5_20260501_004737 | variants=none/identity/clip/proj/full/pool/rule/empirical/uniform, n_inputs=100 | 拆解 LRB 主效应来自哪个模块 |
| 分析文档 | docs/*.md; docs/参考/*; docs/聊天记录20260505.md | 当前工作状态、方法详解、utility、defense、消融、论文图表建议 | 统一论文叙事和下一步路线 |

| 指标 | 含义 | 解读注意 |
| --- | --- | --- |
| rec_token_mean | DAGER 恢复文本与真实文本在 token 级别的平均恢复程度，越低越好 | 0 表示当前攻击和阈值下未恢复 token，不等于形式化安全 |
| ROUGE-1 + ROUGE-2 | 恢复文本与真实文本的 n-gram overlap 汇总，越低越好 | 与 token recovery 一起说明当前攻击是否失效 |
| eval_accuracy / macro-F1 | defense 后完整训练或评估在 SST2 分类任务上的 utility | 小幅高于 clean 的结果只能写成 clean-level utility |
| utility_drop | 相对 clean accuracy 的下降幅度，越低越好 | 负值通常反映随机波动或训练方差，不作为显著提升证据 |
| train_time / attack_time | defense 训练和攻击评估耗时 | 用于复杂度与实用性分析，尤其比较 Projection-LRB 与 full_lrb 开销 |

## 九、DAGER Defense Baselines：privacy 结果

正式 n_inputs=100 结果强化了一个判断：在 SST2+GPT2+batch=2 的 full-gradient DAGER 设置下，clean FedSGD 泄露非常严重。真正进入 strong privacy baseline 的主要是 LRB、Top-k 和 Compression 三类，但 DAGER=0 只表示当前攻击未恢复，不等价于严格隐私保证。

| defense | 参数区间 / 关键点 | rec_token_mean | R1+R2 | 结论 |
| --- | --- | --- | --- | --- |
| none | n/a | 0.833506 | 141.711 | clean 更新泄露严重，作为风险锚点 |
| lrb | 0.05/0.1/0.2/0.35/0.5 | 0.000000 | 0.000 | 全部 DAGER=0，privacy 已饱和 |
| topk | 0.01/0.05/0.1/0.3 | 0.000000 | 0.000 | 安全区到 0.3；0.5/0.7/0.9 开始泄露 |
| compression | 4/8/16 | 0.000000 | 0.000 | 稳定成功点；2bit 在 56/100 SVD 失败，不计入稳定点 |
| noise | 1e-6 -> 1e-3 | 0.953319 -> 0.173061 | 19.180 -> 5.524 | ROUGE 降低但 token 仍有残留 |
| dpsgd | 1e-6 -> 1e-3 | 0.966569 -> 0.077848 | 15.737 -> 0.583 | privacy 随噪声增强，但 utility 风险大 |
| mixup | 0.1~2.0 | 0.882837~0.929118 | 154.743~168.821 | 比 none 更差，不是 privacy defense |
| soteria | 10~90 | 0.996217~1.000000 | 189.384~193.500 | 显著恶化，适合作弱 baseline/附录对照 |

## 十、End-to-End Utility 与 Pareto 判断

Utility 结果说明，能够把 DAGER 打到 0 并不自动意味着 defense 好。当前 fixed-point 下，topk@0.1 与 compression@8 在保持 DAGER=0 的同时几乎不损失 accuracy；full_lrb@0.5 明显改善了 utility，却仍弱于 topk/compression 的强经验点。

| defense | param | accuracy | macro-F1 | loss | utility_drop | train_time |
| --- | --- | --- | --- | --- | --- | --- |
| none | n/a | 0.913226 | 0.913184 | 0.246637 | 0.000000 | 00:42:51 |
| topk | 0.1 | 0.912462 | 0.912430 | 0.243324 | 0.000764 | 06:03:08 |
| compression | 8 | 0.911315 | 0.911290 | 0.263210 | 0.001911 | 07:07:37 |
| mixup | 0.3 | 0.910933 | 0.910906 | 0.239469 | 0.002293 | 00:48:40 |
| lrb | 0.2 | 0.821865 | 0.821360 | 0.441765 | 0.091361 | 08:52:41 |
| noise | 5e-4 | 0.715979 | 0.715617 | 0.552434 | 0.197247 | 01:54:29 |
| dpsgd | 5e-4 | 0.504205 | 0.366347 | 2.612275 | 0.409021 | 03:29:31 |
| lrb | 0.35 | 0.868119 | 0.868010 | 0.356615 | 0.045107 | 05:29:49 |
| lrb | 0.5 | 0.892584 | 0.892472 | 0.321702 | 0.020642 | 01:56:47 |
| topk | 0.3 | 0.910933 | 0.910913 | 0.256373 | 0.002293 | 01:16:24 |
| compression | 16 | 0.909021 | 0.909012 | 0.245559 | 0.004205 | 02:03:04 |

| 层级 | 方法 | 事实依据 | 当前判断 |
| --- | --- | --- | --- |
| 第一层 | topk@0.1; compression@8 | DAGER=0; accuracy=0.912462 / 0.911315 | full-gradient DAGER 下当前最强经验 tradeoff |
| 第二层 | topk@0.3; compression@16 | DAGER=0; accuracy=0.910933 / 0.909021 | 已补齐但未反超强点 |
| 候选主方法 | proj_only@0.5 / Projection-LRB | Ablation: DAGER=0; accuracy=0.915520; drop=-0.002294 | 单次消融显示 clean-level utility，需纳入正式 main result 和多 seed |
| 第三层 | full_lrb@0.5 | DAGER=0; accuracy=0.892584; drop=0.020642 | 完整 LRB 当前最好 utility 点，但仍偏重 |
| 第四层 | lrb@0.2/0.35 | DAGER=0; drop=0.091361 / 0.045107 | privacy 饱和，utility 不足 |
| 失败/附录层 | mixup/noise/dpsgd/soteria | 要么 privacy 不成立，要么 utility 代价过大 | 作为 baseline coverage 保留，不宜做主竞争点 |

## 十一、LRB 消融：projection 是主效应

完整消融结果是当前最有价值的新证据。它显示 full_lrb 并不是当前最优形式：真正的主效应来自低分辨率 signed_pool projection。proj_only@0.5 在不加 residual noise、不做额外 clipping 的情况下已经把 DAGER rec_token_mean 和 ROUGE 恢复全部降到 0，并在该次运行中保持 clean-level accuracy。

| variant | rec_token | R1+R2 | accuracy | drop | train_time | attack_time | 机制结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none | 0.855333 | 148.260 | 0.913226 | 0.000000 | 00:20:02 | 00:05:19 | clean 锚点，泄露严重 |
| identity_lrb | 0.855333 | 148.260 | 0.913226 | 0.000000 | 00:53:14 | 00:07:18 | 证明 LRB 管线本身不改变结果 |
| clip_only | 0.854076 | 147.956 | 0.918196 | -0.004970 | 00:54:11 | 00:07:21 | clipping 几乎不能阻断恢复 |
| proj_only | 0.000000 | 0.000 | 0.915520 | -0.002294 | 01:25:34 | 00:08:50 | 主效应线索：DAGER=0 且 clean-level utility |
| proj_clip | 0.000000 | 0.000 | 0.913226 | 0.000000 | 01:25:53 | 00:08:12 | DAGER=0, utility 接近 clean |
| full_lrb | 0.000000 | 0.000 | 0.892584 | 0.020642 | 01:57:04 | 00:14:11 | DAGER=0, 但完整配置过重 |
| pool_full | 0.000000 | 0.000 | 0.870031 | 0.043195 | 01:41:12 | 00:17:52 | 普通 pool 弱于 signed_pool |
| rule_only | 0.000000 | 0.000 | 0.868884 | 0.044342 | 01:54:33 | 00:12:01 | 只用规则/完整强防御 utility 代价大 |
| empirical_only | 0.000000 | 0.000 | 0.902523 | 0.010703 | 04:01:50 | 00:12:15 | 能防住但有 utility 和开销代价 |
| uniform_all_sensitive | 0.000000 | 0.000 | 0.842890 | 0.070336 | 01:55:45 | 00:10:22 | 一刀切 layer-wise 设计不佳 |

| 消融给出的关键改口 主方法身份建议明确为 Projection-LRB / LRB-lite：layer-wise low-resolution signed_pool projection；full_lrb@0.5 不再写成最终主方法。 full_lrb 可保留为强防御 / 过防御对照，用来说明 residual noise 和完整配置并非当前 full-gradient DAGER 的必要主效应。 proj_only@0.5 的负 utility_drop 不应写成显著提升，只能写成该次运行达到 clean-level utility；正式论文主表需要多 seed 均值与标准差。 |
| --- |

## 十二、AAAI / 审稿口径下的 Claims 与 Non-Claims

这部分用于控制论文叙事边界。审稿人最敏感的是 claims 与 evidence 是否匹配，因此要正面承认当前证据边界，而不是把 DAGER=0 或单次消融结果写成通用安全结论。

| 类型 | 建议表述 |
| --- | --- |
| 可以主张 | clean full-gradient FedSGD 在当前 SST2/GPT2/batch=2 下泄露严重；Projection bottleneck 是当前消融中的 dominant effective component。 |
| 可以主张 | Projection-LRB 是比 full_lrb 更适合继续收束的主方法候选；full_lrb 更适合做 strong-defense / over-defense reference。 |
| 不能主张 | Projection-LRB 已全面优于 topk/compression。当前 full-gradient DAGER 下 topk@0.1 和 compression@8 仍是强经验 tradeoff。 |
| 不能主张 | DAGER=0 等价于形式化隐私安全，或 Projection-LRB 提供 epsilon/delta DP 保证。当前没有 DP accountant 和 adaptive attack 证明。 |
| 不能主张 | PEFT/LoRA 或 partial-gradient 攻击面已经被完整验证。当前只能写作下一阶段扩展和待补强证据。 |

## 十三、主文图表最小集合

根据 agent 建议，后续论文主文不应堆满所有 baseline sweep，而应保留能闭环主论点的最小图表集合：风险锚点、威胁模型、方法框架、主结果、消融，以及一个 transfer/runtime 预留位。

| 编号 | 图表名称 | 论文位置 | 当前状态 / 用途 |
| --- | --- | --- | --- |
| 表 1 | Threat Model / Setup Summary | Problem Setting | 已具备；用于一眼说明证据边界和攻击假设 |
| 表 2 | Experimental Setup Summary | Experimental Setup | 已具备；整理 runs、模型、数据集、metric |
| 表 3 | Main Results under Full-Gradient DAGER | Main Results | 已有 none/topk/compression/full_lrb 等；需把 Projection-LRB 正式纳入同台多 seed |
| 表 4 | Ablation Results | Ablation | 已具备；证明 projection 是 dominant effective component |
| 表 5 | Transfer Results / Runtime Cost | Transfer / Analysis | 当前缺口；PEFT/partial-gradient 与 runtime 需要补实验或补图 |
| 图 1 | Motivation / Problem Overview | Introduction | 待绘制；展示 clean updates -> text recovery -> recoverability bottleneck |
| 图 2 | Threat Model | Problem Setting | 待绘制；client -> defended update -> attacker reconstruction |
| 图 3 | Projection-LRB Framework | Method | 待绘制；layer gradients -> sensitivity -> signed projection -> defended update |
| 图 4 | Privacy-Utility Pareto | Main Results | 已有初版思路；需正式化并补 Projection-LRB sweep |
| 图 5 | Ablation / Keep-Ratio Sweep | Ablation | 当前可先放 ablation bar；补完 calibration 后换 keep-ratio sweep |

## 十四、下一步优先级

| 优先级 | 任务 | 目标 | 建议输出 |
| --- | --- | --- | --- |
| P0 | 统一主方法身份为 Projection-LRB / proj_only@0.5 | 让方法叙事与消融证据一致 | 主方法描述、算法伪代码、main table 新行 |
| P0 | proj_only keep-ratio sweep + 多 seed | 找 DAGER=0 时 utility 最高的最宽松 bottleneck，并估计方差 | k=0.5/0.65/0.75/0.9 曲线；mean±std 表 |
| P0 | projection-only 细消融 | 拆开 rule / empirical / uniform / no_empirical 作用 | proj_rule_only、proj_empirical_only、proj_uniform、proj_no_empirical 表 |
| P1 | 重做主结果和 Pareto 表 | topk/compression/Projection-LRB/full_lrb 同台比较 | main results + Pareto 图 |
| P1 | LoRA/PEFT 对照 | 验证跨实际轻量更新攻击面 | none/proj_only/proj_clip/full_lrb/topk/compression 的 PEFT 表 |
| P2 | partial-gradient / layer-level leakage | 验证局部更新泄露下的结构性防御价值 | first block / qkv / last layers 等攻击面表 |
| P2 | 跨数据集/backbone | 避免单一 SST2/GPT2 偶然性 | cola/rte/gpt2; sst2/bert 等结果 |

## 十五、可以直接用于汇报的阶段性结论

当前 SST2+GPT2+batch=2 的 full-gradient DAGER 实验表明，clean FedSGD 梯度存在严重文本恢复风险。full_lrb 完整配置能稳定把 DAGER 恢复压到 0，但其 clipping 和 residual noise 带来了明显 utility 代价。消融进一步显示，低分辨率 signed_pool projection 是当前 LRB 有效性的主因：proj_only@0.5 在不加噪、不做完整裁剪的情况下即可实现 DAGER=0，并在单次运行中保持 clean-level accuracy。因此，下一阶段应把主方法身份统一为 Projection-LRB / LRB-lite，并通过 keep-ratio sweep、多 seed、LoRA/PEFT、partial-gradient 和跨数据集实验验证其泛化价值。

In the current SST2/GPT2 full-gradient DAGER setting, the dominant effective component is the low-resolution signed projection bottleneck. While the full LRB configuration suppresses token recovery to zero, its clipping and residual-space noise introduce unnecessary utility loss. The projection-only variant achieves zero token and ROUGE recovery with clean-level utility in the current ablation run, making Projection-LRB the most promising method candidate pending multi-seed and cross-attack validation.

## 十六、材料来源清单

| 类别 | 路径 |
| --- | --- |
| 模板 | D:\\code\\Projects\\FedLLM\\docs\\初步思路模版.docx |
| Agent 聊天记录 | D:\\code\\Projects\\FedLLM\\docs\\聊天记录20260505.md |
| 现有成稿 | D:\\code\\Projects\\FedLLM\\output\\doc\\FedLLM_LRB_阶段性工作进展与实验结果汇总1.docx |
| DAGER baseline 分析 | D:\\code\\Projects\\FedLLM\\docs\\DEFENSE_BASELINES_N100_ANALYSIS_20260502.md |
| Utility 分析 | D:\\code\\Projects\\FedLLM\\docs\\UTILITY_RESULTS_ANALYSIS_20260426.md |
| LRB 消融分析 | D:\\code\\Projects\\FedLLM\\docs\\LRB_ABLATION_ANALYSIS_20260503.md |
| LRB 方法详解 | D:\\code\\Projects\\FedLLM\\docs\\LRB_方法详解.md |
| DAGER runs | D:\\code\\Projects\\FedLLM\\log\\runs\\defense_baselines_sst2_b2_gpt2_20260501_010024 |
| Utility runs | D:\\code\\Projects\\FedLLM\\log\\runs\\utility260426 |
| Ablation runs | D:\\code\\Projects\\FedLLM\\log\\runs\\lrb_ablation_sst2_b2_gpt2_k0.5_20260501_004737 |
