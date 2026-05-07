FedLLM / LRB 阶段性工作进展与实验结果汇总

基于现有分析文档、DAGER defense baselines、utility runs 与 LRB ablation runs

| 项目 | 内容 |
| --- | --- |
| 整理日期 | 2026-05-05 |
| 主数据集 / 模型 | SST2 / GPT-2 fine-tuned checkpoint |
| 主攻击面 | Full-gradient DAGER |
| 主方法候选 | Projection-LRB / LRB-lite（由消融结果支持） |
| 主要日志来源 | defense_baselines_sst2_b2_gpt2_20260501_010024；utility260426；lrb_ablation_sst2_b2_gpt2_k0.5_20260501_004737 |

| 核心结论先行 当前工作已经形成 DAGER 攻击、防御 baseline、utility、LRB 机制消融的闭环证据，但强证据边界主要是 SST2+GPT2+batch=2 的 full-gradient DAGER。 Clean FedSGD 泄露严重；LRB、topk、compression 均可在若干点上把当前 DAGER 恢复打到 0，但这不等价于形式化隐私保证。 仅看 full-gradient DAGER，topk@0.1 与 compression@8 仍是更强经验 tradeoff；完整 full_lrb@0.5 已改善 utility，但仍偏重。 最新消融显示 projection bottleneck 是主效应：proj_only@0.5 在单次消融运行中达到 DAGER=0 并保持 clean-level utility，因此下一阶段应把主方法收束为 Projection-LRB。 |
| --- |

## 一、文档定位与当前工作状态

本文参考 docs/初步思路模版.docx 的研究计划结构，将当前 FedLLM / FL-LLM 方向整理为一份阶段性研究进展文档。它不是重新粘贴已有 Markdown，而是把任务价值、文献与 baseline 位置、LRB 方法机制、已完成实验事实、当前判断和后续计划串成一条可继续写论文的主线。

截至 2026-05-05，项目已经从“复现 DAGER + 枚举 baseline”推进到更清晰的阶段：full-gradient DAGER 风险已被量化，多个 defense 的 privacy/utility 已经有结果，LRB 的主要组件也通过完整消融开始被拆开。当前最重要的变化是，主线不再是继续证明“DAGER 可以被打掉”，而是要证明“哪一种 recoverability bottleneck 能在相近 privacy 下保留更多效用，并且能泛化到 PEFT / partial-gradient 等更广攻击面”。

| 审稿口径下的证据边界 已经完成并可较强支撑的证据：SST2 + GPT2 fine-tuned checkpoint + batch=2 + n_inputs=100 的 full-gradient DAGER attack-time privacy；full training utility fixed points；一次完整 LRB ablation。 尚不能直接外推的证据：LoRA/PEFT 训练期 defense、partial-gradient attack、跨数据集/backbone、多 seed 统计显著性、adaptive attack 与严格 DP accounting。 因此当前论文主张应写成：Projection-LRB 是由 full-gradient DAGER 与消融结果支持的主方法候选，而不是已经在所有 FL-LLM 场景全面优于压缩类 baseline。 |
| --- |

| 当前状态一句话 DAGER clean 泄露严重，问题成立；topk/compression 是 full-gradient DAGER 下必须认真比较的强经验 baseline；full_lrb 能强力抑制恢复但过防御；消融显示低分辨率 projection bottleneck 才是当前最值得收束为主方法的机制。 |
| --- |

## 二、背景与动机

大模型微调正在从集中式数据汇聚转向联邦学习、跨机构协作训练和 PEFT adapter 共享。这样的设置看起来天然适合隐私场景：医疗文本、金融评论、企业客服、用户输入日志等原始文本不离开本地，只上传梯度、模型更新或轻量参数更新。但梯度反演与文本恢复攻击表明，“不上传原文”并不等价于“原文不可恢复”。在语言模型中，token embedding、layer-wise gradient 和序列损失共同形成了足以暴露样本内容的高分辨率信号。

本项目当前的 clean DAGER 结果给出了直接动机：在 SST2 + GPT2 + batch=2 + n_inputs=100 的设置下，未防御 FedSGD 梯度的 rec_token_mean 达到 0.833506，ROUGE-1+ROUGE-2 达到 141.710856。这说明攻击者并不是只能恢复模糊主题，而是能够恢复大量 token 级文本信息。因此，本工作不是为了在一个 toy setting 中展示攻击存在，而是要面向真实 FL-LLM / PEFT 协作训练中的中间更新泄露问题，寻找可落地的防御机制。

进一步的动机来自 privacy-utility 冲突：简单把 DAGER 打到 0 并不困难，困难在于保留训练效用。当前实验已经显示，完整 LRB 能稳定抑制 DAGER，但 utility 代价偏高；topk/compression 在 full-gradient DAGER 下很强，却更像通信压缩副作用而不是针对文本 recoverability 的机制解释。论文真正需要回答的是：能否设计一种面向 layer-wise recoverability 的 bottleneck，使更新仍可用于学习任务信号，但不再保留可被 DAGER 等攻击恢复的高分辨率文本细节。

## 三、前人工作局限性

已有工作为本项目提供了攻击、隐私和压缩三类重要参照，但各自都留下了适合本工作切入的缺口。当前实验也印证了这些局限：强扰动方法通常牺牲 utility，通信压缩方法在当前 DAGER 下表现强但机制目标不完全对齐，表示扰动和训练侧混合方法在 GPT2/FedSGD 设置中并不稳定。

| 方向 | 代表思路 | 主要局限性 | 对本工作的启发 |
| --- | --- | --- | --- |
| 梯度反演攻击 | DAGER / LAMP 等从梯度恢复文本 | 多数工作强调攻击能力，较少给出适合 FL-LLM 训练流程的低损耗防御；对 PEFT、partial-gradient 和 defense 后 utility 的系统闭环不足。 | 需要把攻击复现、defense、end-to-end utility 和消融放在同一评价链路里。 |
| 噪声与 DP-SGD | 对梯度裁剪后加噪，获得形式化 DP 语义 | 理论上强，但在大模型梯度和小 batch 文本任务中 utility 代价明显；本项目 noise/DP-SGD 结果也显示较难兼顾 privacy 与 accuracy。 | 不能只追求“加噪后看不见”，而要保留任务梯度中的可学习结构。 |
| 通信压缩 | Top-k sparsification、SVD/quantization compression | 当前 full-gradient DAGER 下是强 baseline，但核心目标是少传或压缩，不是解释哪些 layer/direction 泄露文本；跨 PEFT/partial-gradient 攻击面仍未验证。 | 主方法必须正面比较 topk/compression，同时说明 recoverability bottleneck 的机制差异。 |
| 表示扰动与训练增强 | Soteria、mixup 等表示层或训练侧防御 | 迁移到 GPT2/FedSGD 当前设置后不稳定：soteria 几乎完全泄露，mixup 多点 privacy 失败。 | 需要面向 Transformer 梯度结构重新设计，而不是直接搬用 CV/表示防御经验。 |
| PEFT/局部梯度泄露 | LoRA/adapter 更新反演、partial transformer gradient 攻击 | 更贴近真实部署，但当前仓库只有 eval-first 框架和威胁建模，尚缺完整 defense 结果。 | 这是下一阶段证明泛化意义的关键实验面。 |

## 四、希望解决的问题

本工作希望解决的不是单一实验点上的“攻击是否成功”问题，而是 FL-LLM 中间更新如何在可训练与不可恢复之间取得平衡的问题。具体来说，需要把研究问题拆成四个层次：先证明 clean 更新确实泄露，再建立强 baseline 对照，然后设计可解释的 recoverability bottleneck，最后验证该 bottleneck 是否能跨训练形式和攻击面泛化。

| 问题层次 | 要回答的问题 | 当前已有依据 | 下一步需要补强 |
| --- | --- | --- | --- |
| 风险是否成立 | 客户端只上传梯度/更新时，攻击者是否仍能恢复文本？ | clean DAGER rec_token_mean=0.833506，ROUGE-1+ROUGE-2=141.710856。 | 跨数据集/backbone 复核，避免单一 SST2/GPT2 偶然性。 |
| 防御是否有效 | 哪些方法能把 token 恢复和 ROUGE 恢复压到接近 0？ | LRB@0.05~0.5、topk@0.01~0.3、compression@4/8/16 在 DAGER 下表现强。 | 补齐稳定点说明；DAGER=0 只能表示当前攻击未恢复，不等价于严格隐私保证。 |
| 效用是否可接受 | 同等 privacy 下，哪个 defense 保留最多 task accuracy？ | topk@0.1 和 compression@8 是当前 full-gradient DAGER 的强经验 tradeoff；full_lrb utility 偏弱。 | 把 Projection-LRB 作为主方法纳入 main result，与强 baseline 做同等设置和多 seed 比较。 |
| 机制是否清楚 | 防御效果来自 clipping、projection、noise 还是 layer sensitivity？ | ablation 显示 proj_only@0.5 在单次运行中 DAGER=0 且 accuracy=0.915520，是当前最强机制线索。 | 做 projection-only keep-ratio sweep、多 seed 与细消融，确认最宽松 bottleneck 和方差。 |
| 场景是否泛化 | 该机制是否适用于 PEFT/LoRA、partial-gradient 和更真实的联邦微调？ | PEFT eval-first 框架已有，partial-gradient 威胁已建模。 | 补 LoRA 训练期 defense、partial-gradient 攻击入口和跨任务结果。 |

## 五、问题对应的场景与意义

该问题对应的真实场景是多方拥有敏感文本、但又需要共同微调或适配语言模型的协作训练。典型例子包括医院之间联合训练临床文本分类模型、金融机构联合优化舆情或风控文本模型、企业之间共享客服/工单语义能力，以及移动端或边缘端在本地数据上更新个性化语言模型。此时服务端、聚合方或半诚实参与方即使拿不到原文，也可能通过梯度或 adapter 更新进行文本恢复。

| 应用场景 | 为什么会共享更新 | 泄露风险 | 本工作意义 |
| --- | --- | --- | --- |
| 跨机构文本联邦学习 | 机构间数据不能集中，但希望共享模型能力 | 聚合方或参与方可从 batch 梯度推断训练文本 | 为 FL-LLM 提供 privacy audit 与 defense pipeline。 |
| PEFT / LoRA 协作微调 | 只上传 adapter 或低秩参数更新以降低训练成本 | 轻量更新仍可能携带样本级 token 线索 | 验证 defense 是否能贴近未来更常用的轻量微调流程。 |
| 边缘端或个性化模型更新 | 设备端保留用户输入，周期性上传更新 | 短文本、小 batch 和高重复 token 会放大恢复风险 | 提升用户侧文本数据保护能力。 |
| 科研与合规评估 | 需要定量说明某种训练协议是否安全 | 仅报告 accuracy 无法说明更新是否泄露文本 | 建立 privacy-utility-Pareto 和机制消融证据，支撑论文与合规报告。 |

| 本工作的实际意义 从工程上，它给 FedLLM / PEFT 训练流程提供可复现实验链路：攻击复现、defense sweep、utility 训练、消融和后续 PEFT/partial-gradient 扩展。 从论文上，它把问题从“某个 defense 是否让 DAGER 失败”推进到“中间更新的 recoverability 应该如何被结构性压缩，并在相近 privacy 下保留更多任务信号”。 从当前结果看，Projection-LRB / LRB-lite 是最值得继续推进的主方法候选：它直接对应场景中的低损耗隐私更新需求，也与消融证据一致。 |
| --- |

## 六、威胁模型与评价对象

本工作关注大模型联邦训练或分布式微调过程中由中间更新信息导致的数据泄露。客户端不上传原始文本，但会上传梯度、模型更新或 PEFT adapter 更新；攻击者可能据此恢复训练样本。这个问题的关键不在于“数据有没有直接共享”，而在于“共享更新本身是否携带足以恢复原始文本的高分辨率信息”。

| 攻击面 | 代表方法 / 文献 | 当前工程状态 | 论文作用 |
| --- | --- | --- | --- |
| Full-gradient inversion | DAGER / LAMP | DAGER 已成为主实验；n_inputs=100 defense sweep 已完成 | 证明 clean FedSGD 梯度存在严重文本恢复风险 |
| PEFT leakage | Gradient Inversion Attacks on PEFT / ReCIT | LoRA eval-first 框架已有，缺完整结果表 | 验证 LRB 是否对实际轻量微调更新也有价值 |
| Partial-gradient / layer-level leakage | Partial Transformer Gradients | 威胁已建模，显式攻击入口仍待补 | 证明防御不是只对 full-gradient 攻击有效 |

| 威胁模型要素 | 当前实验默认假设 | 审稿风险 / 后续补充 |
| --- | --- | --- |
| 攻击者能力 | 半诚实服务端或聚合方可观察客户端上传的单轮 full-gradient 更新，并运行 DAGER 类白盒恢复攻击。 | 需要明确不是 secure aggregation 后的只见总和场景；若考虑多客户端聚合，应另设实验。 |
| 攻击者知识 | 知道模型结构、当前权重、tokenizer、任务形式和 batch size；可使用同一 fine-tuned checkpoint 进行恢复。 | 若实际场景中 label、batch size 或客户端采样未知，需要补鲁棒性或敏感性分析。 |
| 被保护对象 | 客户端本地 batch 中的原始文本 token / sequence，而不是仅保护标签或成员关系。 | 需要在指标定义里说明 rec_token_mean 与 ROUGE 衡量的是文本内容恢复，不是 DP 隐私损失。 |
| 当前主要攻击面 | SST2 + GPT2 fine-tuned checkpoint + batch=2 + n_inputs=100 的 full-gradient DAGER。 | PEFT/LoRA 更新、partial-gradient 和跨 backbone 目前是待验证外推，不应写成已有结论。 |
| 防御目标 | 降低中间更新的 recoverability，同时尽量保持 downstream accuracy / macro-F1。 | 需要与 topk/compression 在相同训练配置、相同 attack budget 和相同 privacy target 下比较。 |

## 七、已有 baseline 与当前创新位置

参考 docs/FL-LLM.md 和 baseline 对比材料，当前 baseline 可以按防御逻辑分成扰动系、隐私标准系、通信压缩系、表示扰动系和训练正则系。它们都能提供必要对照，但大多不是直接以“最小化中间更新的 recoverability”为目标。LRB 的创新位置正在这里：不是粗暴减少数值精度，也不是平均加噪，而是按层和方向建立恢复瓶颈。

| baseline | 主要优势 | 主要不足 | 更适合扮演的角色 |
| --- | --- | --- | --- |
| noise | 实现简单，最小干预 | 需要较大噪声才明显有效，utility 快速下降 | sanity-check baseline |
| DP-SGD | 有明确 DP 语义 | 当前大模型/FedSGD 设置下 utility 代价很大 | 理论标准 baseline |
| topk | 计算/通信成本低，当前 DAGER 下很强 | 不是隐私中心设计，泛化到其他攻击面仍需验证 | 强通信压缩 baseline |
| compression | 兼顾通信效率和信息损失，工程接入容易 | 更多回答少传多少，不直接建模少泄露多少 | 强量化 baseline |
| soteria | 直接触及表示泄露 | 迁移到 GPT2/LLM 当前设置后稳定性差，privacy 反而恶化 | 表示层 baseline / 附录对照 |
| mixup | utility 友好 | 当前 DAGER 下 privacy 明显失败 | 训练侧弱防御 baseline |
| Projection-LRB | 显式针对 layer-wise recoverability bottleneck；消融显示 projection 是主效应 | 多 seed、keep-ratio sweep、PEFT/partial-gradient 证据未补齐 | 当前应收束的主方法候选 |
| full_lrb | 包含 projection、clipping、residual noise，privacy 很强 | 完整配置当前过防御，runtime 偏高 | 强防御 / 过防御消融对照 |

## 八、Projection-LRB 方法机制

根据当前消融证据，文档中的主方法应从泛称 LRB 收束为 Projection-LRB：一个面向 recoverability 的 layer-wise gradient projection defense。full_lrb 仍作为强防御 / 过防御对照保留，用来说明 clipping 与 residual-space noise 并不是当前 full-gradient DAGER 设置下的必要主效应。该实现已经在 train_method=full 的训练 utility 路径中接通，但 LoRA 训练期 defense、forward-side representation bottleneck 和完整 calibration pipeline 仍未落地。

更适合论文的方法表述是：对每层梯度 g_l，先根据结构先验与当前 batch 梯度统计估计敏感度 s_l，再用 keep ratio k_l 控制低分辨率 signed_pool 投影 P_{k_l}(g_l)。Projection-LRB 的核心更新可写作 \tilde{g}_l=P_{k_l}(g_l)，其中高敏感层使用更低分辨率 bottleneck；full_lrb 则在此基础上额外加入 clipping 与 residual-space noise。

| 模块 | 当前实现 | 核心作用 | 消融关注点 |
| --- | --- | --- | --- |
| Layer-wise sensitivity | 结构先验 + 当前 batch 梯度统计；默认 empirical_weight=0.6 | 识别哪些层更可能泄露 token 细节 | rule_only / empirical_only / uniform_all_sensitive |
| Low-resolution signed_pool projection | 先随机符号翻转，再 pooling/interpolation 低分辨率重建 | Projection-LRB 主体；保留粗结构，移除高分辨率样本细节 | proj_only / proj_clip / pool_full |
| Layer-wise clipping | 按所有层范数中位数设置每层裁剪阈值 | full_lrb 中的附加约束；限制个别层的大幅度信息流 | clip_only / proj_clip |
| Residual-space noise | 噪声主要加在低分辨率表示之外的残差方向 | full_lrb 中的强防御组件；进一步污染攻击依赖的细节方向 | full_lrb vs proj_only |

| 推荐当前方法表述 Projection-LRB is a layer-wise adaptive gradient projection defense that suppresses recoverability by mapping sensitive-layer gradients into a low-resolution signed pooling subspace. full_lrb 可作为包含 clipping 与 residual-space noise 的强防御对照；当前不建议把 full_lrb 写成最终主方法。 |
| --- |

## 九、实验设置与数据来源

| 实验块 | 日志目录 / 文档 | 关键设置 | 用途 |
| --- | --- | --- | --- |
| DAGER defense baselines | log/runs/defense_baselines_sst2_b2_gpt2_20260501_010024 | sst2, gpt2, ./models/gpt2-ft-rt, batch=2, n_inputs=100 | 量化 clean 泄露与各 defense 的 attack-time privacy |
| Utility focused runs | log/runs/utility260426 | full training, epoch=1, seeds=101/202/303 | 评估 defense 对 accuracy/F1/loss/runtime 的影响 |
| LRB ablation | log/runs/lrb_ablation_sst2_b2_gpt2_k0.5_20260501_004737 | variants=none/identity/clip/proj/full/pool/rule/empirical/uniform, n_inputs=100 | 拆解 LRB 主效应来自哪个模块 |
| 分析文档 | docs/*.md; docs/参考/* | 当前工作状态、方法详解、utility、defense、消融、参考图表要求 | 统一论文叙事、图表清单和下一步路线 |

| 比较口径 | 当前做法 | 需要在论文中明确的点 |
| --- | --- | --- |
| 模型与数据 | 主结果使用 SST2、GPT2 fine-tuned checkpoint、batch=2；clean checkpoint 与 defense runs 对齐。 | 这是当前强证据边界；跨数据集/backbone 是后续泛化实验。 |
| 攻击预算 | DAGER defense baseline sweep 使用 n_inputs=100；同一攻击脚本和同一恢复指标汇总。 | 不同 defense 的 DAGER=0 只表示在该攻击预算下未恢复，不是不可攻击证明。 |
| 训练效用 | utility runs 使用 full training，报告 accuracy、macro-F1、loss、training time。 | 需要补多 seed 均值/标准差；当前个别负 utility_drop 应按随机波动或 clean-level utility 表述。 |
| baseline 调参 | topk、compression、LRB/full_lrb 均有 sweep 或补点；当前最强 fixed points 是 topk@0.1 与 compression@8。 | 主表应按同等 privacy target 选各方法最佳 utility 点，避免固定点不公平。 |
| DP 口径 | dpsgd 采用逐样本裁剪 + 高斯噪声的 DP-SGD-style 实现。 | 未做 privacy accountant 时不能声称 epsilon/delta 形式化 DP 保证。 |

| 指标 | 含义 | 解读注意 |
| --- | --- | --- |
| rec_token_mean | DAGER 恢复文本与真实文本在 token 级别的平均恢复程度；越低表示当前攻击恢复出的 token 越少。 | rec_token_mean=0 表示当前攻击和阈值下未恢复 token，不等价于形式化隐私安全。 |
| ROUGE-1 + ROUGE-2 | 恢复文本与真实文本的 n-gram overlap 汇总；用于补充 token-level recovery 的文本相似度视角。 | ROUGE 归零与 token 恢复归零共同说明当前攻击失败，但仍需 adaptive attack / 其他攻击面验证。 |
| eval_accuracy / macro-F1 | defense 后完整训练或评估在 SST2 分类任务上的 utility。 | 小幅高于 clean 的结果应谨慎写作 clean-level utility，需多 seed 支撑显著提升。 |
| utility_drop | 相对 clean accuracy 的下降幅度；越低越好。 | 负值通常反映随机波动或训练方差，不应直接作为方法提升 accuracy 的证据。 |
| train_time / attack_time | defense 训练和攻击评估耗时。 | 用于复杂度与实用性分析，尤其要解释 Projection-LRB 与 full_lrb 的额外开销。 |

## 十、DAGER Defense Baselines：privacy 结果

正式 n_inputs=100 结果强化了一个判断：在 SST2+GPT2+batch=2 的 full-gradient DAGER 设置下，clean FedSGD 泄露非常严重，rec_token_mean=0.833506，ROUGE-1+ROUGE-2=141.710856。真正进入 strong privacy baseline 的主要是 LRB、Top-k 和 Compression 三类。

| defense | 参数区间 / 关键点 | rec_token_mean | R1+R2 | 结论 |
| --- | --- | --- | --- | --- |
| none | n/a | 0.833506 | 141.711 | clean 更新泄露严重，作为风险锚点 |
| lrb | 0.05/0.1/0.2/0.35/0.5 | 0.000000 | 0.000 | 全部 DAGER=0，privacy 已饱和 |
| topk | 0.01/0.05/0.1/0.3 | 0.000000 | 0.000 | 安全区到 0.3；更大比例开始泄露：0.5->0.047; 0.7->0.161; 0.9->0.382 |
| compression | 4/8/16 | 0.000000 | 0.000 | 4/8/16 稳定；2bit 在 56/100 SVD 失败；32bit 退化 |
| noise | 1e-6 -> 1e-3 | 0.953 -> 0.173 | 19.180 -> 5.524 | 需要较大扰动才有效，token 仍有残留 |
| dpsgd | 1e-6 -> 1e-3 | 0.967 -> 0.078 | 15.737 -> 0.583 | 理论基线重要，但当前 utility 代价大 |
| mixup | 0.1~2.0 | 0.883~0.929 | 154.743~168.821 | utility 友好但 privacy 恶化，不适合作为隐私防御 |
| soteria | 10~90 | 0.996~1.000 | 189.384~193.500 | 迁移到当前 LLM 设置后明显恶化 |


| privacy 口径注意 compression@2 虽然前 56 个样本指标为 0，但本次 n_inputs=100 在第 56/100 处 SVD 失败，正式稳定成功点只应计入 compression@4/8/16。 mixup 与 soteria 在当前设置下不是有效 privacy defense：mixup 多点 rec_token_mean 高于 clean，soteria 几乎完全泄露。 |
| --- |

## 十一、End-to-End Utility：训练效用结果

Utility 结果说明，能够把 DAGER 打到 0 并不自动意味着 defense 好。当前 fixed-point 下，topk@0.1 与 compression@8 在保持 DAGER=0 的同时几乎不损失 accuracy；LRB@0.2 也能 DAGER=0，但明显过防御。后来补齐的 full_lrb@0.5 明显改善了 utility，却仍弱于 topk/compression 的强经验点。需要注意的是，单次运行中略高于 clean 的 accuracy 应按 clean-level utility 或训练方差处理，不能直接声称显著提升。

| method | accuracy | macro-F1 | loss | utility_drop | train_time | source |
| --- | --- | --- | --- | --- | --- | --- |
| none | 0.913226 | 0.913184 | 0.246637 | 0.000000 | 00:42:51 | UTILITY_RESULTS_ANALYSIS_20260426.md |
| topk@0.1 | 0.912462 | 0.912430 | 0.243324 | 0.000764 | 06:03:08 | sst2_b2_gpt2_topk_0.1 |
| compression@8 | 0.911315 | 0.911290 | 0.263210 | 0.001911 | 07:07:37 | sst2_b2_gpt2_compression_8 |
| mixup@0.3 | 0.910933 | 0.910906 | 0.239469 | 0.002293 | 00:48:40 | sst2_b2_gpt2_mixup_0.3 |
| lrb@0.2 | 0.821865 | 0.821360 | 0.441765 | 0.091361 | 08:52:41 | sst2_b2_gpt2_lrb_0.2 |
| noise@5e-4 | 0.715979 | 0.715617 | 0.552434 | 0.197247 | 01:54:29 | sst2_b2_gpt2_noise_5e-4 |
| dpsgd@5e-4 | 0.504205 | 0.366347 | 2.612275 | 0.409021 | 03:29:31 | sst2_b2_gpt2_dpsgd_5e-4 |
| lrb@0.35 | 0.868119 | 0.868010 | 0.356615 | 0.045107 | 05:29:49 | sst2_b2_gpt2_lrb_0.35 |
| lrb@0.5 | 0.892584 | 0.892472 | 0.321702 | 0.020642 | 01:56:47 | sst_b2_gpt2_lrb_0.5 |
| topk@0.3 | 0.910933 | 0.910913 | 0.256373 | 0.002293 | 01:16:24 | sst2_b2_gpt2_topk_0.3 |
| compression@16 | 0.909021 | 0.909012 | 0.245559 | 0.004205 | 02:03:04 | sst2_b2_gpt2_compression_16 |


## 十二、Privacy-Utility Tradeoff 与当前 Pareto 判断

把 privacy 和 utility 对齐后，当前 full-gradient DAGER 的经验结论很清楚：topk@0.1 与 compression@8 是当前最强 fixed-point baseline；topk@0.3 与 compression@16 已补齐 utility，但没有超过它们；full_lrb@0.5 是当前完整 LRB 配置中最好的 utility 点，但仍有约 2.1 个 accuracy 点的下降。Projection-LRB 的 ablation 点显示了新的主方法空间，但还需要在 main result 中按同等 privacy target 与 topk/compression 同台、多 seed 比较。


| 层级 | 方法 | 事实依据 | 当前判断 |
| --- | --- | --- | --- |
| 第一层 | topk@0.1; compression@8 | DAGER=0；accuracy 分别为 0.912462 / 0.911315 | full-gradient DAGER 下当前最强经验 tradeoff |
| 第二层 | topk@0.3; compression@16 | DAGER=0；accuracy=0.910933 / 0.909021 | 已补齐但未反超强点 |
| 候选主方法 | proj_only@0.5 / Projection-LRB | Ablation: DAGER=0；accuracy=0.915520；drop=-0.002294 | 单次消融显示 clean-level utility，需纳入正式 main result 和多 seed |
| 第三层 | full_lrb@0.5 | DAGER=0；accuracy=0.892584；drop=0.020642 | 完整 LRB 当前最好 utility 点，但仍偏重 |
| 第四层 | lrb@0.2/0.35 | DAGER=0；drop=0.091361 / 0.045107 | privacy 饱和，utility 不足 |
| 失败/附录层 | mixup/noise/dpsgd/soteria | 要么 privacy 不成立，要么 utility 代价过大 | 作为 baseline coverage 保留，不宜做主竞争点 |

## 十三、LRB 消融：机制证据

完整消融结果是当前最有价值的新证据。它显示 full_lrb 并不是当前最优形式：真正的主效应来自低分辨率 signed_pool projection。proj_only@0.5 在不加 residual noise、不做额外 clipping 的情况下已经把 DAGER rec_token_mean 和 ROUGE 恢复全部降到 0，并在该次运行中保持 clean-level accuracy。这里应谨慎表述为机制线索和主方法候选，而不是已经证明 Projection-LRB 在统计意义上优于所有 baseline。

| variant | rec_token | R1+R2 | accuracy | drop | train_time | attack_time | 机制结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none | 0.855333 | 148.260 | 0.913226 | 0.000000 | 00:20:02 | 00:05:19 | clean 锚点，泄露严重 |
| identity_lrb | 0.855333 | 148.260 | 0.913226 | 0.000000 | 00:53:14 | 00:07:18 | 证明 LRB 管线本身不改变结果 |
| clip_only | 0.854076 | 147.956 | 0.918196 | -0.004970 | 00:54:11 | 00:07:21 | clipping 几乎不能阻断恢复 |
| proj_only | 0.000000 | 0.000 | 0.915520 | -0.002294 | 01:25:34 | 00:08:50 | 主效应线索：DAGER=0 且 clean-level utility |
| proj_clip | 0.000000 | 0.000 | 0.913226 | 0.000000 | 01:25:53 | 00:08:12 | DAGER=0，utility 接近 clean |
| full_lrb | 0.000000 | 0.000 | 0.892584 | 0.020642 | 01:57:04 | 00:14:11 | DAGER=0，但 residual noise/完整配置过重 |
| pool_full | 0.000000 | 0.000 | 0.870031 | 0.043195 | 01:41:12 | 00:17:52 | 普通 pool 弱于 signed_pool |
| rule_only | 0.000000 | 0.000 | 0.868884 | 0.044342 | 01:54:33 | 00:12:01 | 只用规则/完整强防御 utility 代价大 |
| empirical_only | 0.000000 | 0.000 | 0.902523 | 0.010703 | 04:01:50 | 00:12:15 | 能防住但有 utility 代价和开销 |
| uniform_all_sensitive | 0.000000 | 0.000 | 0.842890 | 0.070336 | 01:55:45 | 00:10:22 | 一刀切 layer-wise 设计不佳 |


| 消融给出的关键改口 主方法身份建议明确为 Projection-LRB / LRB-lite：layer-wise low-resolution signed_pool projection；full_lrb@0.5 不再写成最终主方法。 full_lrb 可保留为强防御 / 过防御对照，用来说明 residual noise 和完整配置并非当前 full-gradient DAGER 的必要主效应。 proj_only@0.5 的负 utility_drop 不应写成显著提升，只能写成该次运行达到 clean-level utility；正式论文主表需要多 seed 均值与标准差。 |
| --- |

## 十四、工程进度与已接通能力

| 模块 | 当前状态 | 已完成能力 | 主要缺口 |
| --- | --- | --- | --- |
| DAGER baseline | 已接通并有正式结果 | defense_baselines.sh；collect_experiment_logs.py；n_inputs=100 sweep | 跨数据集/backbone 仍待补 |
| Full training utility | 已接通 | train.py 支持 full training --defense lrb/topk/compression/noise/dpsgd/mixup | LRB runtime 和 seed 方差需要优化 |
| LRB preset / ablation | 已完成一轮完整消融 | identity/clip/proj/full/pool/rule/empirical/uniform | proj_only keep-ratio sweep 与 projection-only 细消融待补 |
| PEFT / LoRA | eval-first 最小框架已具备 | 加载本地 .pt/.pth LoRA checkpoint；支持 none/noise/topk/compression/lrb | 训练期 LoRA defense、BERT PEFT、Adapter/IA3/Prefix 未接通 |
| Partial-gradient | 文档建模完成，代码入口不足 | 已有威胁模型和实验计划 | 需要 gradient_layer_subset / gradient_param_filter 等入口 |

## 十五、论文逻辑与图表规划

参考用户提供的三张图，当前材料可以组织成五阶段逻辑：任务价值、机制缺陷、替代机制、技术落地、最终交付。现阶段最需要的是把实验事实转换成一组清晰图表，并避免过度宣称。

| 阶段 | 核心问题 | 当前实质内容 | 还缺什么证据 |
| --- | --- | --- | --- |
| 1 | 这件事凭什么值得做？ | FedSGD/LLM clean 梯度可被 DAGER 严重恢复；rec_token=0.833506 | Introduction 中补任务价值与强攻击文献 anchor |
| 2 | 能力卡在哪个环节？ | 泄露来自梯度/更新中的高分辨率 recoverability 结构 | 把 gradient 子空间暴露 token 信息讲成明确 problem statement |
| 3 | 别人为什么绕不开？ | noise/DP utility 代价大；topk/compression 强但不是隐私中心设计 | 解释压缩类方法在 full-gradient DAGER 上强但泛化性未证 |
| 4 | 新机制如何落地？ | Projection-LRB：layer-wise sensitivity -> signed_pool bottleneck -> defended update | keep-ratio sweep 与 PEFT/partial-gradient 验证 |
| 5 | 最终交付什么？ | 当前可交付方法候选、消融结论、主结果表雏形 | 跨攻击面主表、复杂度分析、稳定结论 |

| 图表类型 | 论文位置 | 核心功能 | 当前状态 |
| --- | --- | --- | --- |
| Main Results 对比表 | Experiments - Main Results | 比较 none/topk/compression/Projection-LRB/full_lrb 等 | 已有数据，需纳入 projection 点、多 seed 和强 baseline |
| Ablation Study 表 | Experiments - Ablation | 证明 projection 是主效应 | 已具备完整结果 |
| Privacy-Utility Pareto 图 | Experiments - Analysis | 同等 privacy 下比较效用 | 已有初版，需补 proj_only sweep |
| Framework Overview 图 | Method | 30 秒看懂 Projection-LRB 数据流 | 需绘制正式论文图 |
| 复杂度分析图 | Experiments - Analysis | 展示 accuracy vs train/attack/proxy runtime | 已有时间数据，需整理 |
| Transfer / PEFT 表 | Experiments - Transfer | 证明跨攻击面泛化 | 尚待实验 |

## 十六、当前不应过度声称的内容

- 不能写成 LRB/Projection-LRB 已在 SST2/GPT2 上全面优于 topk/compression。当前 full-gradient DAGER 下，topk@0.1 与 compression@8 的经验 tradeoff 更强。

- 不能写成 full_lrb 是最终主方法。消融已经显示 proj_only@0.5 更像当前主候选，应收束为 Projection-LRB。

- 不能说 Projection-LRB 在 LoRA/PEFT 或 partial-gradient 下必然最优。这些攻击面还没有完整结果。

- 不能把 full_lrb 或 Projection-LRB 当成严格 DP 方法。它们有裁剪/投影/加噪组件，但没有完整 epsilon/delta 证明。

- 不能把 proj_only@0.5 的负 utility_drop 写成显著提升；在多 seed 前只能写成 clean-level utility。

- 不能把 compression@2 算入稳定成功点。本次 n_inputs=100 在 56/100 处失败。

## 十七、下一步优先级

| 优先级 | 任务 | 目标 | 建议输出 |
| --- | --- | --- | --- |
| P0 | 把主方法身份统一为 Projection-LRB / proj_only@0.5 | 让论文方法与消融证据一致 | 主方法描述、算法伪代码、main table 新行 |
| P0 | proj_only keep-ratio sweep + 多 seed | 找 DAGER=0 时 utility 最高的最宽松 bottleneck，并估计方差 | k=0.5/0.65/0.75/0.9 曲线；mean±std 表 |
| P0 | projection-only 细消融 | 拆开 rule / empirical / uniform / no_empirical 的作用 | proj_rule_only、proj_empirical_only、proj_uniform、proj_no_empirical 表 |
| P1 | 重做主结果和 Pareto 表 | 把 topk/compression/Projection-LRB/full_lrb 同台比较 | main results + Pareto 图 |
| P1 | LoRA/PEFT 对照 | 验证跨实际轻量更新攻击面 | none/proj_only/proj_clip/full_lrb/topk/compression 的 PEFT 表 |
| P2 | partial-gradient / layer-level leakage | 验证局部更新泄露下的结构性防御价值 | first block / qkv / last layers 等攻击面表 |
| P2 | 跨数据集/backbone | 避免单一 SST2/GPT2 偶然性 | cola/rte/gpt2; sst2/bert 等结果 |
| P2 | runtime 优化 | 降低 Projection-LRB 和 full_lrb 实用开销 | train time / attack time / proxy runtime 分析图 |

## 十八、可以直接用于汇报的阶段性结论

| 中文结论 当前 SST2+GPT2+batch=2 的 full-gradient DAGER 实验表明，clean FedSGD 梯度存在严重文本恢复风险。full_lrb 完整配置能稳定把 DAGER 恢复压到 0，但其 clipping 和 residual noise 带来了明显 utility 代价。消融进一步显示，低分辨率 signed_pool projection 是当前 LRB 有效性的主因：proj_only@0.5 在不加噪、不做完整裁剪的情况下即可实现 DAGER=0，并在单次运行中保持 clean-level accuracy。因此，下一阶段应把主方法身份统一为 Projection-LRB / LRB-lite，并通过 keep-ratio sweep、多 seed、LoRA/PEFT、partial-gradient 和跨数据集实验验证其泛化价值。 |
| --- |

| English statement In the current SST2/GPT2 full-gradient DAGER setting, the dominant effective component is the low-resolution signed projection bottleneck. While the full LRB configuration suppresses token recovery to zero, its clipping and residual-space noise introduce unnecessary utility loss. The projection-only variant achieves zero token and ROUGE recovery with clean-level utility in the current ablation run, making Projection-LRB the most promising method candidate pending multi-seed and cross-attack validation. |
| --- |

## 十九、参考文献与 baseline 对应关系

本节先按当前工作实际用途整理参考文献：第一类用于支撑 DAGER 等文本恢复攻击和实验框架，第二类对应已跑的 defense baselines，第三类支撑 FL-LLM、PEFT/LoRA 与 partial-gradient 等问题场景。后续写论文时可再统一转换为 BibTeX 或 GB/T 7714 格式。

| 类别 | 参考文献 | 与本文工作的关系 |
| --- | --- | --- |
| 主攻击框架 | Petrov et al. DAGER: Exact Gradient Inversion for Large Language Models. NeurIPS 2024. | 当前 privacy 实验的主攻击与主评测框架；用于证明 full-gradient LLM/FedSGD 设置下文本可被高精度恢复。 |
| 文本梯度泄露 | Balunovic et al. LAMP: Extracting Text from Gradients with Language Model Priors. NeurIPS 2022. | DAGER 之前的重要文本恢复攻击；用于铺垫语言模型先验可显著增强梯度反演。 |
| Transformer 梯度攻击 | Deng et al. TAG: Gradient Attack on Transformer-based Language Models. Findings of EMNLP 2021. | 说明 Transformer/NLP 模型梯度泄露并非偶然现象，是 DAGER/LLM 攻击线的早期代表。 |
| 通用梯度反演 | Zhu, Liu, and Han. Deep Leakage from Gradients. NeurIPS 2019. | 通用 gradient inversion 起点；用于介绍 federated learning 中梯度本身可能泄露训练样本。 |
| 通用优化式反演 | Geiping et al. Inverting Gradients: How easy is it to break privacy in federated learning? NeurIPS 2020. | 补充传统优化式反演背景，说明该问题不局限于文本任务。 |

| 当前 baseline / 扩展点 | 参考文献 | 放入文档时的定位 |
| --- | --- | --- |
| DP-SGD / dpsgd | Abadi et al. Deep Learning with Differential Privacy. ACM CCS 2016. | 对应逐样本裁剪 + 高斯噪声的理论隐私 baseline；当前实验中 utility 代价较高。 |
| noise | Abadi et al. Deep Learning with Differential Privacy. ACM CCS 2016; Gaussian noise baseline as a simple perturbation control. | 作为最朴素的梯度扰动对照，不单独声称具备 DP 保证。 |
| topk | Aji and Heafield. Sparse Communication for Distributed Gradient Descent. EMNLP 2017; Lin et al. Deep Gradient Compression. ICLR 2018. | 定位为通信压缩/稀疏化强经验 baseline；当前 full-gradient DAGER 下 tradeoff 很强，但不是隐私中心设计。 |
| compression | Alistarh et al. QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding. NeurIPS 2017. | 对应当前代码中的 QSGD-style stochastic quantization；用于量化压缩类 baseline。 |
| soteria | Sun et al. Soteria: Provable Defense Against Privacy Leakage in Federated Learning from Representation Perspective. CVPR 2021. | 表示层隐私防御 baseline；当前 GPT2/DAGER 设置下效果不好，应作为对照而非主竞争点。 |
| mixup | Zhang et al. mixup: Beyond Empirical Risk Minimization. ICLR 2018. | 训练侧增强/混合样本 baseline；当前结果 utility 友好但 privacy 失败。 |

| 用途 | 参考文献 | 建议写法 |
| --- | --- | --- |
| 联邦学习场景 | McMahan et al. Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS 2017. | 用于介绍 FedAvg/FedSGD 和去中心化数据训练背景。 |
| LoRA / PEFT 场景 | Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022. | 用于说明只共享 LoRA/adapter 更新的现实训练流程。 |
| PEFT 梯度反演 | Sami et al. Gradient Inversion Attacks on Parameter-Efficient Fine-Tuning. CVPR 2025. | 用于支撑 PEFT 更新同样可能泄露数据；建议放在相关工作或未来扩展。 |
| LLM PEFT 私有数据恢复 | Xie et al. ReCIT: Reconstructing Full Private Data from Gradient in Parameter-Efficient Fine-Tuning of Large Language Models. arXiv 2025. | 预印本，适合作为 PEFT leakage 风险补充，不建议写成已发表顶会结论。 |
| Partial-gradient 泄露 | Li et al. Seeing the Forest through the Trees: Data Leakage from Partial Transformer Gradients. EMNLP 2024. | 支撑下一步 partial-gradient 泛化实验，说明只暴露部分 Transformer 梯度也可能泄露数据。 |

| 引用口径建议 DAGER、LAMP、TAG、DLG 是攻击背景的主线；DAGER 应作为当前实验框架的核心引用。 topk/compression 应明确写作通信压缩类强 baseline，而不是传统隐私防御；它们在当前 DAGER 下强，正好构成本文必须正面比较的经验对手。 ReCIT 目前按 arXiv 预印本处理；PEFT/partial-gradient 文献更适合放在动机、相关工作和未来验证部分。 |
| --- |

## 二十、材料来源清单

| 类别 | 路径 |
| --- | --- |
| 模板 | D:\\code\\Projects\\FedLLM\\docs\\初步思路模版.docx |
| 当前工作分析 | D:\\code\\Projects\\FedLLM\\docs\\CURRENT_WORK_STATUS_ANALYSIS_20260427.md |
| DAGER baseline 分析 | D:\\code\\Projects\\FedLLM\\docs\\DEFENSE_BASELINES_N100_ANALYSIS_20260502.md |
| Utility 分析 | D:\\code\\Projects\\FedLLM\\docs\\UTILITY_RESULTS_ANALYSIS_20260426.md |
| LRB 消融分析 | D:\\code\\Projects\\FedLLM\\docs\\LRB_ABLATION_ANALYSIS_20260503.md |
| LRB 方法详解 | D:\\code\\Projects\\FedLLM\\docs\\LRB_方法详解.md |
| FL-LLM 原始/整理思路 | D:\\code\\Projects\\FedLLM\\docs\\FL-LLM.md |
| PEFT 框架 | D:\\code\\Projects\\FedLLM\\docs\\PEFT_EVAL.md |
| 参考材料 | D:\\code\\Projects\\FedLLM\\docs\\参考 |
| DAGER runs | D:\\code\\Projects\\FedLLM\\log\\runs\\defense_baselines_sst2_b2_gpt2_20260501_010024 |
| Utility runs | D:\\code\\Projects\\FedLLM\\log\\runs\\utility260426 |
| Ablation runs | D:\\code\\Projects\\FedLLM\\log\\runs\\lrb_ablation_sst2_b2_gpt2_k0.5_20260501_004737 |
