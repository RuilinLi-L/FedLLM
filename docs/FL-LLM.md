**FL-LLM**

**总体目标**

这个工作关注**大模型训练过程中“中间信息导致的数据泄露问题”**。已有研究表明，在 Transformer/LLM 训练中，特别是在 FedSGD 这种设定下，服务器是有可能通过客户端上传的梯度或更新信息，**几乎完整恢复出本地训练文本**的，实现真实训练数据的恢复。

因此，我们的目标不只是验证某一种攻击，而是想回答一个更核心的问题：**训练过程中暴露的这些中间信息，到底会泄露多少数据？能不能设计一种方法，从根本上减少这种泄露？**

整体思路是，先在一个可复现、算力可控的训练框架下，把目前大模型中最主要的数据恢复攻击系统性复现出来，同时复现相关防御方法的baseline；在此基础上，再设计一个**尽可能通用的防御方法**，去同时抑制多种攻击，同时尽量不影响模型训练效果。当前阶段的重点不是马上确定防御方法（需要不断调试以达到可以同时防御多种攻击类型并SOTA），而是先把攻击和评测体系搭完整。

**框架以及攻击方法**

在实验设定上，我们以 FedSGD 的联邦大模型训练作为主要场景，因为目前最强的数据恢复攻击基本都建立在这个设定上。模型规模控制在 7B 以下开源 LLM，并结合 PEFT（如 LoRA / Adapter） 来降低算力开销。训练框架方面主要基于已有的攻击框架来实现。

在攻击选择上，我们不再简单堆叠所有已有方法，而是按照**中间信息如何泄露数据**来进行分类，并重点覆盖三类最有代表性的攻击：

第一类是 **基于梯度的恢复攻击（Gradient Inversion）**，包括 DAGER 这类可以在大模型中几乎精确恢复文本的攻击，以及类似 LAMP 这类较早的文本梯度泄露方法。这类攻击主要利用梯度与输入数据之间的直接映射关系。

**代表文章：DAGER: Exact Gradient Inversion for Large Language Models（主要）LAMP（次要）**

第二类是 **PEFT 相关的数据泄露（PEFT Leakage）**，包括 PEFTLeak 和 ReCIT 等工作。这类攻击说明，即使只共享 LoRA 或 Adapter 这样的轻量更新，仍然可能恢复出训练数据，因此对实际大模型训练更具现实意义。

**代表文章：Gradient Inversion Attacks on Parameter-Efficient Fine-Tuning（主要）ReCIT（无代码，次要）**

第三类是 **层级或局部信息泄露（Layer-level Leakage）**，即只利用部分层的梯度或更新信息，就可以恢复数据。这类攻击说明，数据泄露不一定依赖完整梯度，局部信息同样可能是危险的。

**代表文章：Seeing the Forest through the Trees: Data Leakage from Partial Transformer Gradients**

这三类攻击覆盖了当前大模型训练中主要的泄露路径，也构成了本工作的核心威胁模型（**实验中，就可以用上面的三个主要文章作为我们的攻击方法，都是有完整源码的**）。

| 论文 | 类别 | 核心方法 | 优点 | 缺点/局限 | 是否开源 |
| --- | --- | --- | --- | --- | --- |
| DAGER: Exact Gradient Inversion for Large Language Models | Gradient Inversion | 将梯度反演建模为精确优化问题，直接恢复token序列 | 当前最强攻击之一，可近乎精确恢复文本 | 依赖完整梯度，计算开销大 | ✅ |
| LAMP: Extracting Text from Gradients with Language Model Priors | Gradient Inversion | 结合语言模型先验辅助梯度反演 | 相比早期方法恢复效果更好，思路通用 | 恢复精度不如DAGER，偏早期方法 | ✅ |
| Gradient Inversion Attacks on Parameter-Efficient Fine-Tuning | PEFT Leakage | 从LoRA/Adapter更新中恢复训练数据 | 证明轻量更新同样泄露数据，现实意义强 | 依赖特定PEFT结构 | ✅ |
| ReCIT: Reconstruction of Training Data from Incremental Training | PEFT Leakage | 利用增量训练更新进行数据重构 | 不依赖完整梯度，更贴近实际训练 | 复现难度高，无公开代码 | ❌ |
| Seeing the Forest through the Trees: Data Leakage from Partial Transformer Gradients | Layer-level Leakage | 仅利用部分层梯度恢复数据 | 证明局部信息也会泄露，威胁更广泛 | 恢复精度低于全梯度攻击 | ✅ |

可参考综述：Analysis of Privacy Leakage in Federated Large Language Models

LLM in the middle: A systematic review of threats and mitigations to real-world LLM-based systems

**Baselines**

| 论文 | 类别 | 核心方法 | 优点 | 缺点/局限 | 是否开源 |
| --- | --- | --- | --- | --- | --- |
| Deep Learning with Differential Privacy (DP-SGD) | Defense (DP) | 梯度裁剪+加噪声实现差分隐私 | 理论最完善，标准方法 | 对模型性能影响明显 | ✅ |
| Soteria: Provable Defense against Privacy Leakage in Federated Learning | Defense (Representation) | 对中间表示进行扰动，降低可恢复性 | 专门针对梯度泄露设计 | 对复杂模型稳定性有限 | ✅ |
| Gradient Compression for Communication-Efficient FL | Defense (Compression) | 对梯度进行稀疏化或量化压缩 | 降低通信同时减少信息量 | 不是专为隐私设计，防护有限 | ✅ |
| Top-k Gradient Sparsification | Defense (Compression) | 仅保留梯度中最大k个分量 | 简单高效，易实现 | 信息仍可能被恢复 | ✅ |
| MixUp / Data Augmentation-based Defense | Defense (Data-level) | 通过数据混合降低样本可识别性 | 实现简单，无需改模型结构 | 对攻击的针对性较弱 | ✅ |
| Noise Injection on Gradients | Defense (Noise) | 直接对梯度添加随机扰动 | 简单直接，易于控制强度 | 隐私-效用权衡难调 | ✅ |

**对当前 baselines 的判断**

如果结合本文的三类核心威胁模型来看，现有 baselines 的价值更多是“提供有代表性的对照组”，而不是直接成为最终方法。更具体地说：

| baseline | 主要优势 | 主要不足 | 更适合扮演的角色 |
| --- | --- | --- | --- |
| `noise` | 最通用、最便宜、实现最简单，适合作为最小干预基线 | 只是在输出端做随机扰动，不改变泄露结构；往往需要较大噪声才有效，效用下降快 | 最基础的 sanity-check baseline |
| `dpsgd` | 唯一有明确差分隐私语义的标准 baseline，论文说服力强 | 训练和显存开销大；在大模型/FedSGD 场景下隐私-效用权衡通常不理想；也没有利用 Transformer 泄露的结构性 | 理论上界/标准合规 baseline |
| `topk` | 计算与通信成本低，容易在大模型上跑大规模 sweep | 保留下来的往往正是最显著、最可能携带关键信息的分量；不是为隐私设计 | 通信压缩对照组 |
| `compression` | 兼顾通信效率与一定信息损失，工程上容易接入 | 更多是在“少传多少”而不是“少泄露多少”上有效；对强攻击的针对性有限 | 压缩类 baseline |
| `soteria` | 直接触及 representation leakage，本质上比纯梯度后处理更接近泄露根因 | 更适合特定任务和表示路径；迁移到 LLM/PEFT 后稳定性和通用性有限；开销也不低 | 表示层 baseline |
| `mixup` | 对训练效用通常较友好，也可能降低样本唯一性 | 不是专门为梯度泄露设计；在小 batch 的 FL 设定里，混合后的梯度仍可能泄露较强信息 | 训练侧弱防御 baseline |
| `lrb`（当前版本） | 最接近本文“通用防御”目标，显式利用层级差异和恢复瓶颈来做结构化抑制 | 目前还是启发式 v1：敏感层识别主要靠规则，投影基底较固定，也还没有 forward-side 的表示约束 | 最有希望发展成主方法的原型 |

总结起来，`noise / dpsgd` 更像“通用扰动系”基线，`topk / compression` 更像“通信压缩系”基线，`soteria` 是“表示扰动系”基线，`mixup` 是“训练正则系”基线。真正和本文目标最一致的，是从 `lrb` 继续往前推，因为它开始显式针对“哪些中间信息更可恢复”这个问题本身，而不只是粗暴减少数值精度。

**防御方法设计**

在防御设计上，我们的目标不是针对某一个具体攻击去做优化，而是尝试设计一种**通用的防御机制**，能够整体降低“中间更新信息 → 原始数据”的泄露能力。换句话说，我们希望防御的不是某个攻击算法，而是**数据从梯度或更新中被恢复出来这件事本身，在实验层面就是，同时在上面的三种类型攻击当中用我们的防御算法达到最优的隐私-效用权衡**。

从直觉上看，大模型训练中涉及离散 token 到连续表示的映射，这一过程本身就提供了很多可以干预的空间，比如在更新信息中引入扰动、改变表示方式，或者限制信息表达能力等。因此后续可以从多个方向探索防御方法，例如对梯度结构进行修改、对更新信息进行扰动或压缩，或者在训练过程中限制信息泄露。

当前阶段不会提前固定某一种具体方案，而是先在统一的实验框架下验证一个核心问题：**是否存在一种方法，可以在基本不影响训练效果的前提下，同时降低多类数据恢复攻击的成功率**。在这个基础上，再逐步收敛到具体的防御设计。**（这个是当前需要考虑的重点，也就是复现攻击方法的基础上，同时尝试复现其自带或自己加入一些baseline，然后再设计我们自己的算法，看能不能打过这打过这些baseline）**

**建议的具体方法设计：LRB-v2 / HLRB**

如果沿着本文目标继续优化，我不建议完全推翻 `lrb`，而是建议把它升级成一个更完整的两阶段通用防御：**HLRB（Hierarchical Layer-wise Recoverability Bottleneck）**。它可以理解为 `lrb` 的研究版 v2。

核心思想不是去“骗过某个攻击”，而是主动限制训练过程中最容易被恢复的那部分信息，让共享更新只保留完成任务所必需、但不利于恢复原始样本的成分。

HLRB 可以分成两个层面：

1. **forward-side representation bottleneck**

- 在真正被下游头部或 PEFT 模块消费的表示上加入轻量瓶颈，而不是只在最终梯度上后处理。
- 对 `seq_class` 任务，可以像 `Soteria` 一样从“分类头真正使用的表示”出发，但不做一次性硬剪枝，而是做更平滑的低维投影、随机子空间掩码或受控 dropout。
- 对 PEFT 场景，可以把 bottleneck 放在 LoRA/Adapter 的输入或瓶颈激活上，直接限制可恢复信息进入可训练模块。

2. **backward-side recoverability bottleneck**

- 对每一层共享梯度 `G_l`，先估计一个泄露敏感度 `s_l`，再按敏感度决定该层的保留比例、裁剪强度和噪声强度。
- 不是像当前 `lrb` v1 那样主要依赖层名规则，而是引入一个**校准步骤**：在公开数据或 warmup batch 上统计每层的梯度能量、谱集中度、早层/embedding 重要性、以及对攻击 proxy 的可恢复性。
- 对于每层梯度，做分解  
  `G_l = P_l(G_l) + R_l(G_l)`  
  其中 `P_l` 是允许保留的低恢复性公共子空间，`R_l` 是更可能泄露样本细节的残差子空间。
- 最终共享的是  
  `\tilde{G}_l = clip(P_l(G_l)) + \xi_l`，其中 `\xi_l` 主要加在残差方向或与保留子空间正交的方向上。  
  这样做的直觉是：尽量保留任务相关的低频/稳定结构，优先破坏易恢复的高频/局部/样本特异信息。

相对当前 `lrb` v1，我建议的关键升级点是：

- **敏感度估计从规则改为校准**  
  当前版本主要依赖“embedding 和前几层更敏感”的启发式，这个方向是对的，但还不够稳。更好的做法是为每种模型和训练方式预先跑一次 calibration，得到 layer-wise sensitivity profile。

- **投影基底从固定 pooling 改为公共低恢复性子空间**  
  当前的 adaptive pooling 很便宜，但带有较强坐标系假设。更稳的方案是用固定随机正交基、Hadamard 风格基，或者由公开数据估计出的低秩公共子空间来做投影。

- **噪声加在“被丢弃的方向”而不是平均乱加**  
  这比全空间加噪更节省效用预算，也更符合“重点打断恢复路径”的设计目标。

- **兼容 full fine-tuning 与 PEFT**  
  对 full 模型，重点保护 embedding 和早层 attention 相关参数；对 LoRA/Adapter，重点保护 bottleneck 输入输出和前几层适配器，而不是只把 PEFT 看成“小参数量所以应该更安全”。

- **把方法目标写成“最小化 recoverability，而不是最大化扰动”**  
  这点很重要。研究叙事上，HLRB 的目标不是把梯度弄得越乱越好，而是在给定效用损失预算下，最大限度压低跨攻击面的 recoverability。

**为什么这个方向比单一 baseline 更适合作为主方法**

- 它和本文威胁模型更一致。本文不是只打 DAGER，也不是只打某个 PEFT 攻击，而是要同时面对 full-gradient、PEFT、partial-gradient 三类泄露。
- 它比 `noise/dpsgd` 更结构化，比 `soteria` 更通用，比 `topk/compression` 更以隐私为中心。
- 它保留了 `lrb` 当前实现的工程优势：可以先从 attack-time transform 做 v1/v2，对比结果出来后，再决定是否推进到训练时版本。

**现阶段最推荐的研究路线**

如果只选择一条主线，我建议：

- 保留 `noise / dpsgd / topk / compression / soteria / mixup` 作为完整 baseline 套件。
- 把 `lrb` 明确定位为“我们的主方法原型”，后续升级为 `LRB-v2 / HLRB`。
- 先做 `post-gradient HLRB`，验证是否能同时压低三类攻击。
- 如果结果成立，再进一步实现 `representation-side HLRB`，把 forward bottleneck 也纳入。

这样做的好处是：研究主线清晰，实验对照充分，而且方法演进路径自然，不需要突然从 baseline 跳到一个完全不同、难以解释的新方法。

**工作性质判断**

整体来看，这项工作是一个以**实践和问题为导向的研究型工作**，而不是偏理论研究。主要关注点在于真实联邦大模型训练场景下隐私风险是否存在、风险有多严重，以及在现实算力和系统约束下能否有效缓解这些风险。理论分析更多作为对实验现象和防御机制的解释和支撑，而不是核心目标。
