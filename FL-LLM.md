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

**防御方法设计**

在防御设计上，我们的目标不是针对某一个具体攻击去做优化，而是尝试设计一种**通用的防御机制**，能够整体降低“中间更新信息 → 原始数据”的泄露能力。换句话说，我们希望防御的不是某个攻击算法，而是**数据从梯度或更新中被恢复出来这件事本身，在实验层面就是，同时在上面的三种类型攻击当中用我们的防御算法达到最优的隐私-效用权衡**。

从直觉上看，大模型训练中涉及离散 token 到连续表示的映射，这一过程本身就提供了很多可以干预的空间，比如在更新信息中引入扰动、改变表示方式，或者限制信息表达能力等。因此后续可以从多个方向探索防御方法，例如对梯度结构进行修改、对更新信息进行扰动或压缩，或者在训练过程中限制信息泄露。

当前阶段不会提前固定某一种具体方案，而是先在统一的实验框架下验证一个核心问题：**是否存在一种方法，可以在基本不影响训练效果的前提下，同时降低多类数据恢复攻击的成功率**。在这个基础上，再逐步收敛到具体的防御设计。**（这个是当前需要考虑的重点，也就是复现攻击方法的基础上，同时尝试复现其自带或自己加入一些baseline，然后再设计我们自己的算法，看能不能打过这打过这些baseline）**

**具体方法设计（待补充）**

此处可填写之前设计的防御算法简介

**工作性质判断**

整体来看，这项工作是一个以**实践和问题为导向的研究型工作**，而不是偏理论研究。主要关注点在于真实联邦大模型训练场景下隐私风险是否存在、风险有多严重，以及在现实算力和系统约束下能否有效缓解这些风险。理论分析更多作为对实验现象和防御机制的解释和支撑，而不是核心目标。