# Projection-LRB: A Unified Update-Space Framework for Mitigating Text Reconstruction across Full, Adapter, and Partial Gradients

> 定位：AAAI 2027 投稿中文全文底稿，后续各章节统一在本文件续写，并用于改写英文正文。  
> 方法名称：Projection-LRB。  
> 说明：文末“非正文写作附注”用于锁定引用键和证据边界，不属于论文正文。

## Abstract

Federated fine-tuning keeps raw text local, yet shared updates can reveal client inputs. We study whether one client-side update transform can reduce text recoverability across heterogeneous gradient exposures. Projection-LRB is a shape-preserving reconstruction bottleneck: pooling maps each gradient to a lower-dimensional intermediate grid, interpolation restores the public tensor shape but not the removed degrees of freedom, and optional signed modulation changes the realized reconstruction coordinates. The core contribution is this operator; layer-wise ratio allocation is only an optional instantiation. We evaluate the same transform under full-gradient DAGER, a controlled PEFTLeak-style Adapter ratio probe, and partial-gradient matching. On three GPT-2 classification tasks, an adaptive-ratio instantiation drives standard-DAGER token and ROUGE recovery to zero over three seeds while retaining clean-level utility at the main operating point. It also reduces recovery under the evaluated Adapter and partial-gradient protocols, although stronger baselines remain in individual settings. Mechanism experiments show that reconstruction, rather than clipping or the Hybrid allocator, accounts for the standard-attack endpoint. Under oracle DAGER, however, uniform ratios from 0.5 to 0.9 form a non-monotonic R1+R2 plateau (101.18--104.57), and signed versus unsigned reconstruction at 0.5 is indistinguishable; the evidence therefore supports transformed-span mismatch, not monotonic resolution-dependent protection. An official-aligned PEFTLeak-style CIFAR-100/ViT-Adapter study provides single-seed cross-modal evidence that the same operator reduces image-patch recovery, but is not a general PEFT or cross-modal robustness claim. Overall, Projection-LRB is an empirical mitigation framework rather than a formal privacy mechanism: oracle state knowledge restores substantial text, and no result establishes irreversible deletion or universal adaptive robustness.

## 1. Introduction

在医疗、金融和跨机构协作等隐私敏感场景中，参与方往往希望利用分散保存的临床记录、客户对话或内部文档共同微调大语言模型，同时避免将原始数据集中上传。联邦学习通过交换本地计算的梯度或模型更新实现协同训练，参数高效微调（parameter-efficient fine-tuning, PEFT）则进一步将通信与训练限制在 LoRA、Adapter 等少量可训练参数上 \cite{mcmahan2017fedavg,hu2022lora}。然而，数据保留在本地并不意味着其语义内容也停留在本地：在单客户端更新可见或聚合不足以隐藏个体贡献时，服务器虽然无法直接读取客户端文本，却能够观察由这些文本产生的完整梯度、局部梯度或 adapter updates。因而，联邦微调所提供的是数据位置上的隔离，而非共享更新天然不可逆的隐私保证。

近期研究不断扩大这种更新泄露的已知边界。早期优化式梯度反演已表明，输入可以通过匹配观测梯度进行近似重建 \cite{zhu2019deep,balunovic2022lamp}；DAGER 进一步利用自注意力梯度的低秩结构和 token embedding 的离散性，在 honest-but-curious 的完整梯度设置下恢复大批量文本 \cite{petrov2024dager}。与此同时，PEFTLeak 揭示了精简 adapter gradients 仍可携带可解析的输入结构，ReCIT 则在包含恶意预训练和记忆增强的更强攻击模型中恢复上下文与个人可识别信息 \cite{sami2025peftleak,xie2025recit}。部分 Transformer 梯度攻击还表明，攻击者无需获得全部参数梯度，也可能从特定层或模块中重建文本 \cite{li2024partial}。这些工作采用的攻击知识、可见更新和主动能力并不相同，但共同提出了一个尚未充分回答的问题：不同更新暴露面是否包含可被统一抑制的可恢复结构？

现有防护机制主要沿三条路径展开。第一类以差分隐私和噪声注入为代表，通过裁剪与随机扰动限制单个样本对共享更新的影响 \cite{abadi2016dpsgd}；第二类采用 top-k sparsification 或 quantization 压缩更新，其主要目标是降低分布式训练的通信开销 \cite{aji2017sparse,alistarh2017qsgd}；第三类在输入或中间表示上进行掩码和混合，以降低样本特征的直接暴露 \cite{sun2021soteria,verma2019manifold}。这些机制提供了不同的 privacy-utility trade-off，其中 top-k 和 quantization 在本文评估中也是强经验基线。然而，它们通常不以文本可恢复性为直接设计对象，也没有回答哪些层、方向和分辨率承载了攻击所依赖的样本级结构。这一缺口并不意味着压缩方法无效，而是说明通信压缩率与文本可恢复性之间缺少可解释的逐层联系。因此，仅比较扰动幅度或压缩率，难以解释防御何时抑制恢复、何时又会被更强攻击绕过。

本文关注一个比梯度幅值更直接的对象：攻击所利用的更新坐标关系。我们研究能否在客户端上传前插入同一个更新变换，分别缓解完整梯度、Adapter 梯度关系和部分梯度上的文本恢复，而无需为每种攻击设计独立防御。为此，我们提出 Projection-LRB，一种逐层 reconstruction bottleneck。给定 keep ratios，算子执行 signed modulation、pooling 与 interpolation；interpolation 恢复共享接口要求的张量形状，但不会恢复 pooling 删除的自由度。最简单的实现对所有层使用统一 ratio，结构先验与更新统计只是一种可选 allocator。完整梯度、Adapter 参数子集或部分层梯度的暴露均发生在这一变换之后，因此方法的统一性来自插入点和算子的复用，而不依赖 Hybrid allocator 或特定攻击的解码规则。

三类文本实验承担互补角色。第一，在 SST-2、CoLA 与 Rotten Tomatoes 的 full-gradient DAGER 评估中，Projection-LRB (adaptive ratios, k=0.5) 在三个随机种子上均将 standard-attack token 与 ROUGE 恢复降至 0，并保持接近 clean 的下游效用。第二，在 controlled PEFTLeak-style Adapter ratio probe 中，同一 instantiation 将单次 100 条文本运行的 token recovery 从 `0.760` 降至 `0.202`；更严格的 fixed-probe v2 得到 `0.685±0.020`，说明 Legacy 协议高估了抑制幅度。第三，在 PTG `first2` 三种子协议中，k=0.2 相对 paired clean 将 token recovery 平均降低 24.3%，但 compression@2、noise 与 DP-SGD-style 在该协议中达到更低恢复。作为附录扩展，official-aligned PEFTLeak-style CIFAR-100/ViT-Adapter 实验还观察到 image-patch recovery 从 `0.109375` 降至 0。该图像结果是 single-seed、privacy-only 的跨模态机制证据，不等价于真实 PEFT utility 或一般 PEFT/LoRA 稳健性。

机制实验进一步收紧了方法归因。Projection-only 而非 clipping 达到 standard-DAGER 零恢复；Hybrid、rule-only、empirical-only 与 uniform schedules 在 standard 和 oracle 设置中均不可区分。Uniform oracle sweep 的 R1+R2 在 \(r=0.5--0.9\) 间为 `101.18--104.57`，没有形成单调曲线；signed 与 unsigned reconstruction@0.5 也基本相同。因此，我们不再把经验抑制归因于 Hybrid sensitivity、ratio 的连续单调效应或 signed 本身。更符合证据的解释是：standard decoder 仍用原始 token coordinates 检验已经被 reconstruction 改写的梯度 span，从而产生 transformed-span mismatch。Oracle 知道 realized state 后可恢复大量文本，而隐藏状态时真实 token 仍留在候选池中，失败主要发生在排序、过滤和序列组装。

本文的主要贡献如下：

1. **统一的更新空间视角。** 我们将完整梯度 DAGER、PEFTLeak-style Adapter 比值恢复与 PTG 部分梯度匹配置于同一“防御先作用于客户端更新、攻击再观察暴露子空间”的接口下，并在不混排跨协议绝对指标的前提下联合考察可恢复性、任务效用与计算成本。
2. **Projection-LRB。** 我们提出一种保持共享张量形状、但经 pooling 后具有秩上界的 reconstruction operator。该算子可使用统一或逐层 ratios，并可复用于 full-model、Adapter 与 partial-gradient 暴露路径；Hybrid ratio allocation 和 signed state 均不作为必要有效成分。
3. **跨暴露证据与机制边界。** 三任务 DAGER、controlled Adapter ratio probe 和 PTG `first2` 表明同一变换在三类文本协议内均可降低恢复，图像侧实验提供单种子的跨模态补充证据。Oracle ratio sweep、signed/unsigned control 和知识分层进一步表明，当前证据支持 transformed-span mismatch，而不支持单调分辨率效应、不可逆信息删除或一般白盒稳健性。

## 2. Related Work

### 2.1 Gradient Inversion in Language Models

梯度反演最初主要通过优化虚拟输入，使其产生的梯度与观测梯度一致 \cite{zhu2019deep}。文本的离散性与庞大搜索空间使这一思路比图像重建更困难，LAMP 因而引入语言模型先验和面向离散 token 的优化策略，在小批量文本恢复上取得进展 \cite{balunovic2022lamp}。DAGER 转而利用 Transformer 自注意力投影梯度的低秩结构，先判断 token embedding 是否位于观测梯度张成的子空间，再重建 token 集合与序列，从而在无需数据先验的 honest-but-curious 场景中扩展到更大批量和更长序列 \cite{petrov2024dager}。这一路线与允许服务器修改下发模型或训练过程的 malicious-server attacks 不同：其风险来自被动观察正常训练产生的更新，因此更直接对应本文的 full-gradient 主威胁模型。本文以 DAGER 作为完整梯度主攻击，不将不同假设下的攻击结果作绝对横向排序；我们的目标也不是改进反演算法，而是约束这类解析式攻击依赖的逐层高分辨率结构。

### 2.2 Leakage from PEFT and Partial Updates

只共享少量可训练参数并不会自动消除输入泄露。PEFTLeak 通过对 PEFT 模块进行攻击性设计，从 adapter gradients 中解析并聚合图像 patch，说明低维更新仍可能保留细粒度样本信息 \cite{sami2025peftleak}。ReCIT 面向文本 PEFT gradients，将过滤式 token 提取与记忆增强结合，在恶意预训练及 Personal Notes 假设下恢复上下文和个人可识别信息 \cite{xie2025recit}。前者关注 adapter 梯度中的解析关系，后者还借助模型记忆能力，因而二者与被动观察标准 PEFT 更新的攻击不能视为同一威胁模型。与它们不同，Seeing the Forest through the Trees 延续优化式文本反演路线，系统考察仅暴露单层、多个层以及 attention/FFN 子模块梯度时的数据恢复风险 \cite{li2024partial}。本文的 Adapter ratio 实验是受 PEFTLeak 梯度比值机制启发的 text-side controlled probe，并非对原始图像攻击的文字复现，也不代表标准 PEFT/LoRA 部署攻击；PTG 实验则用于检验 Projection-LRB 向局部梯度匹配攻击的有限迁移。由于二者的可见更新、攻击流程和指标口径不同，本文只讨论机制迁移，不直接比较跨协议的绝对恢复率。

### 2.3 Defenses and Gradient Compression

现有缓解方法可按机制分为随机化、通信压缩和表示干预。标准 DP-SGD 通过逐样本梯度裁剪、加噪及 privacy accountant 提供可量化的差分隐私语义 \cite{abadi2016dpsgd}；一般噪声注入可以作为低成本经验基线，但没有裁剪、采样分析和会计量时不能单独推出同等保证。本文代码中的 DP-SGD-style baseline 只实现裁剪与高斯噪声变换，未提供会计量，因此不能继承正式的隐私保证。Top-k sparsification、gradient dropping 与 QSGD 类量化主要为减少通信量而设计，但信息丢失也可能经验性降低恢复率 \cite{aji2017sparse,alistarh2017qsgd}；它们在本文的 standard DAGER 评估中是必须正面对比的强基线，而不是弱化处理的辅助方法，通信效率收益也不能直接解释为隐私保证。Soteria 从中间表示敏感性出发屏蔽高风险特征，Manifold Mixup 的原始目标则是通过混合隐藏状态学习更平滑的表示 \cite{sun2021soteria,verma2019manifold}；本文只将相应实现作为 Soteria-style 和 manifold Mixup-style coverage baselines。与这些方法相比，Projection-LRB 不声称提供更强的形式化保证或普遍优于压缩，而是将逐层更新的可恢复分辨率作为一等设计对象，并通过消融和跨攻击面评估检验这一结构性定位。

## 3. Problem Setup and Threat Model

### 3.1 Federated Update Exposure

考虑由服务器协调的联邦语言模型微调。客户端 \(i\) 持有不离开本地的数据集 \(D_i=\{(x_{ij},y_{ij})\}_{j=1}^{n_i}\)，在服务器下发的模型参数 \(\theta\) 上最小化经验损失 \(\mathcal{L}(\theta;D_i)\)。为统一描述梯度上传与一步模型更新，记防御前的单客户端更新为

\[
G_i=\nabla_{\theta}\mathcal{L}(\theta;D_i),
\qquad
\widetilde G_i=T_{\phi}(G_i),
\]

其中 \(T_{\phi}\) 是由参数 \(\phi\) 控制的更新变换；对 Projection-LRB 而言，\(\phi\) 包括逐层 keep ratio 及相应的 signed-pool reconstruction 规则。本文的主部署场景假设服务器能够在聚合前观察一个客户端的 \(\widetilde G_i\)。攻击者不能读取 \(D_i\)、防御前的 \(G_i\) 或客户端本地中间状态。Secure aggregation、加密聚合以及多个客户端更新混合后能否重新分离个体贡献均不在本文讨论范围内。

上述定义将防御插入点固定在客户端上传之前，并把变换后的更新作为唯一攻击输入；服务器不额外获得本地优化器状态、逐样本梯度或中间激活。若实际系统上传多步训练后的参数差值而非当前实验中的梯度，必须重新验证攻击与防御语义，本文不默认二者等价。

我们定义两类部署暴露。完整梯度暴露为 \(\mathcal{O}_{\mathrm{full}}(\widetilde G_i)=\widetilde G_i\)，对应 standard 与 knowledge-stratified DAGER；部分层暴露为 \(\mathcal{O}_{S}(\widetilde G_i)=\{\widetilde G_i^{(\ell)}:\ell\in S\}\)，对应只共享或只泄露参数子集的情形。PTG `first2` 中，防御首先作用于完整更新，随后才选择 GPT-2 前两个 Transformer blocks 的 24 个可见梯度，因此攻击者观察的是 \(\mathcal{O}_{S}(T_{\phi}(G_i))\)，而不是先截取原始局部梯度再施加防御。

此外，我们单独保留一个 controlled Adapter ratio probe 作为机制诊断：恶意 probe 在 weight/bias gradients 经 \(T_{\phi}\) 处理后，利用二者的比值关系恢复 token。Legacy 协议会针对当前私有 batch 动态生成 slot inventory，故 probe 结构本身依赖待攻击样本。该设置用于检验低分辨率重建是否破坏 Adapter 梯度比值结构，不属于上述部署暴露，也不作为现实服务器攻击协议。

### 3.2 Attacker Knowledge and Capabilities

服务器知道共享训练所必需的模型与协议信息，但不同评估对防御状态、主动能力和辅助信息的假设并不相同。Standard DAGER 不显式反演防御；knowledge-stratified DAGER 则固定攻击算法与候选预算，只改变攻击者是否获得当前更新实际使用的 keep ratios 和 Rademacher signs。四种知识设置构成一个偏序而非严格线性层级：oracle 同时知道二者，method-only 同时隐藏二者，ratio-hidden 与 signs-hidden 分别只隐藏其中一项，因而二者不可按攻击强弱直接排序。

| Evaluation | Attacker capability | Side information | Paper role |
| --- | --- | --- | --- |
| Standard DAGER | Honest-but-curious 地观察完整单客户端更新；解码时不显式反演 defense | 已知模型、checkpoint、tokenizer、任务与协议元数据 | 主攻击 |
| Knowledge-stratified DAGER | 观察完整防御更新，并使用 state-aware candidate decoding；按 oracle、ratio-hidden、signs-hidden、method-only 改变对 realized state 的访问 | Standard DAGER 信息、公开方法；部分设置额外获得实际 ratios 或 signs | 稳健性扩展 |
| PTG `first2` | 被动观察部分梯度并执行 80 steps 的 gradient matching | Batch size 1、known label、前两个 blocks、24 个可见梯度 | 探索性迁移 |
| Legacy Adapter ratio probe | Known-label、sample-adaptive malicious probe；对处理后的 weight/bias gradients 做比值恢复 | Batch size 1、动态 probe inventory、legacy public-bin statistics | 受控机制分析 |

因此，Legacy Adapter ratio probe 不能表述为 original PEFTLeak reproduction、标准 PEFT/LoRA 攻击或第三种等价部署威胁。ReCIT 所需的恶意预训练与记忆增强能力也未授予本文的被动服务器。除 knowledge-stratified DAGER 外，攻击者不会为已知防御变换专门修改恢复算法；相应结果只衡量给定攻击协议下的经验可恢复性。Knowledge-stratified evaluation 也不是“知道/不知道防御”的二元对照：四种设置均知道 Projection-LRB 方法，区别只在于攻击者能否获得当前更新的实例级重建状态。

### 3.3 Objectives and Metrics

对攻击 \(A\) 与固定协议 \(P\)，本文的目标是在给定任务效用预算内降低防御更新的经验可恢复性，可概括为

\[
\min_{\phi}\;\mathcal{R}_{A,P}\!\left(\mathcal{O}(T_{\phi}(G_i))\right)
\quad\text{s.t.}\quad
\Delta\mathcal{U}(T_{\phi})\leq\epsilon_{U},
\]

其中 \(\mathcal{R}_{A,P}\) 由 token recovery 与 ROUGE-1/2 衡量，\(\Delta\mathcal{U}\) 由下游 accuracy、macro-F1 和 loss 相对 clean 的变化刻画，CoLA 另报告 Matthews correlation coefficient（MCC）；训练时间与攻击时间作为计算成本另行报告。该目标不是对任意攻击取最大值，也不蕴含安全证明。

恢复指标只在相同攻击协议内比较。DAGER、PTG、Legacy probe 与 fixed-probe v2 的可见更新、先验和解码过程不同，其绝对恢复率不得横向排序。在 standard DAGER、Adapter probe 与 PTG 中，`rec_token_mean` 表示相应协议最终恢复的 token 比例；在 knowledge-stratified DAGER 中，它记录真实 token 是否仍位于扩大后的候选集合，`rec_maxb_token_mean` 进一步衡量 Top-B 排序后的候选召回，`rec_l1_mean` 与 `rec_l2_mean` 分别记录后续两阶段过滤恢复，最终序列质量由 ROUGE 报告。因而，高 candidate recall 与低 ROUGE 可以同时出现，不能仅凭一个 token 指标宣称信息被删除。类似地，本文的 DP-SGD-style baseline 仅包含裁剪和高斯噪声，没有 privacy accountant，不能等同于正式 DP-SGD。全文将结论限定为“在给定任务效用预算下降低当前协议下的经验可恢复性”，不主张对未知或自适应攻击的普遍保证。

## 4. A Unified Update-Space Defense: Projection-LRB

### 4.1 Motivation and Overview

梯度反演依赖的不只是更新幅值，还包括可将单个 token 或局部序列与参数方向对应起来的细粒度结构。Projection-LRB 因而不以求解形式化隐私目标为出发点，而是在客户端上传前，对每个有效梯度张量施加低分辨率 signed reconstruction。给定按模型参数顺序排列的梯度元组 \(\{G_l\}\) 和逐层 keep ratios \(\{r_l\}\)，方法直接计算

\[
\widetilde G_l=T_{r_l,q_l}(G_l),
\]

其中 \(q_l\) 是逐层 reconstruction seed。最简单的 uniform instantiation 令所有有效层满足 \(r_l=r\)；其他 ratio schedules 只决定每层使用何种分辨率，不改变核心算子。该变换输出与输入保持相同顺序、形状和数据类型，可以直接替换原上传梯度，不需要可学习打分器、额外优化或客户端与服务器之间的新交互。

这里的“Projection”是方法命名，不表示算子具有正交性、幂等性或信息论不可逆性；其作用是构造一个可控的低分辨率经验瓶颈。该瓶颈的设计假设是，降低更新的局部分辨率可能削弱当前反演器使用的样本级结构，同时仍保留下游优化可利用的较粗更新趋势。本文只通过攻击与效用实验检验这一假设，不由算子形式推出安全结论。

### 4.2 Intrinsic Resolution of Signed Reconstruction

先考虑长度为 \(n\) 的一维梯度。令 \(P_r\in\mathbb{R}^{q\times n}\) 表示 adaptive average pooling，\(U_r\in\mathbb{R}^{n\times q}\) 表示 linear interpolation，其中 \(q=\max(1,\operatorname{round}(nr))\)；令 \(D_s\) 为由 Rademacher signs 构成的对角矩阵。固定 ratio 与 signs 后，代码中的算子可写为

\[
T_{r,s}=D_sU_rP_rD_s.
\]

因此

\[
\operatorname{rank}(T_{r,s})\leq q,
\qquad
\dim\ker(T_{r,s})\geq n-q.
\]

这一秩界对应一个直接的 many-to-one 直觉。对任意 \(z\in\ker(T_{r,s})\)，都有

\[
T_{r,s}(x)=T_{r,s}(x+z).
\]

因此，原空间中沿核空间方向不同的更新会产生相同 reconstruction signature。该等价类解释说明 interpolation 无法唯一恢复 pooling 前的更新，但它本身既不是隐私证明，也不保证攻击一定失败。

Interpolation 将输出恢复到长度 \(n\)，但输出仍限制在至多 \(q\) 维的 reconstruction subspace 中，因而恢复外部形状不等于恢复被 pooling 删除的自由度。对二维 \(m\times n\) 梯度，行列 pooling 与 bilinear interpolation 是可分离的，可写为 \(T(G)=L_rG R_r^\top\)，其中 \(\operatorname{rank}(L_r)\leq q_m\)、\(\operatorname{rank}(R_r)\leq q_n\)，且 \(q_m\approx rm\)、\(q_n\approx rn\)。因此中间网格的元素数约为原矩阵的 \(r^2\)，并有

\[
\operatorname{rank}(T(G))
\leq \min\{\operatorname{rank}(G),q_m,q_n\}.
\]

因为 \(D_s^\top D_s=I\)，signed 与 unsigned operator 具有相同的 rank 和 singular-value spectrum；signs 不增加 pooling 的代数降维量，而是改变被保留 subspace 相对于原坐标的方向。Static signs 是确定性实验状态而非密码学密钥；攻击者知道 realized signs 时可以将同一变换纳入候选检验，因此本文不把符号调制本身解释为不可逆的信息删除。

### 4.3 How the Operator Changes DAGER's Span Test

对一个线性层，batch gradient 可表示为

\[
G_l=\sum_{i=1}^{B}\delta_{l,i}h_{l,i}^{\top},
\]

其中 \(h_{l,i}\) 是该层输入表示，\(\delta_{l,i}\) 是反向误差信号。对二维 signed reconstruction，变换后有

\[
T(G_l)=L_lG_lR_l^\top
=\sum_{i=1}^{B}(L_l\delta_{l,i})(R_lh_{l,i})^\top.
\]

Standard DAGER 使用候选 token representation \(h(v,p)\) 与观测梯度的 row/column span 进行匹配；经过 Projection-LRB 后，直接与观测 span 对齐的对象变为 \(R_lh(v,p)\)，而不是原始 \(h(v,p)\)。未显式建模防御的 standard decoder 因此面临 transformed-span mismatch。Knowledge-aware oracle 可以对候选施加相同的 \(R_l\)，从而恢复显著的 token 和序列信息。需要强调的是，不同 ratio 产生的 pooling/interpolation ranges 未必严格嵌套，DAGER 的候选排序与序列组装也不是 rank 的单调函数；因此 rank bound 不推出“ratio 越低，恢复越差”。本文用 standard/oracle 分层定位 decoder mismatch，再用 uniform oracle sweep 直接检验剩余 ratio 效应。实验没有观察到有序曲线，故机制结论限定为 operator-induced span transformation。

### 4.4 Ratio Schedules and Reconstruction Algorithm

Uniform schedule 令所有有效层使用同一 ratio \(r\)，是隔离 reconstruction resolution 的最简单实现。现有跨暴露主结果使用 Appendix A.2 定义的 adaptive-ratio instantiation；该 allocator 只产生 \(\{r_l\}\)，不属于下述 reconstruction operator 的必要步骤。Standard 与 oracle 消融均未显示 adaptive、rule-only、empirical-only 和 uniform schedules 之间存在可辨识的隐私优势，因此本文不把 adaptive allocation 列为贡献或必要组件。

对一维张量，算子使用取值为 \(\{-1,+1\}\) 的 Rademacher signs 逐坐标调制，将长度从 \(n\) 池化到 \(\max(1,\operatorname{round}(nr))\)，经 linear interpolation 恢复到 \(n\)，再乘相同 signs。对二维张量，代码分别生成可分离的行、列 signs，将两个维度池化后用 bilinear interpolation 恢复。高于二维的张量按一维展平路径处理。所有重建先转为 float 计算，最终恢复原 shape 和 dtype。

**Algorithm 1: Projection-LRB**

**Input:** gradients \(\{G_l\}\), keep ratios \(\{r_l\}\), base seed \(q\).
**Output:** defended gradients \(\{\widetilde G_l\}\).

1. 对每个非空 \(G_l\)，将外部 schedule 给出的 \(r_l\) 限制到 \([10^{-4},1]\)，并令 \(q_l=q+1009(l+1)\)。
2. 若 \(G_l\) 为标量或 \(r_l\geq0.999\)，令 \(\widetilde G_l=G_l\) 的副本。
3. 对一维或展平张量，生成逐坐标 signs，依次执行 sign modulation、average pooling、linear interpolation 和 inverse sign modulation。
4. 对二维张量，生成可分离的行列 signs，依次执行 sign modulation、二维 average pooling、bilinear interpolation 和 inverse sign modulation。
5. 恢复原 shape 和 dtype，并返回保持原梯度顺序的 \(\{\widetilde G_l\}\)。

### 4.5 Optional Full-LRB Extensions and Cost

Projection-LRB 的方法身份是上述 signed low-resolution operator，不包含裁剪或噪声。用于消融的 Full-LRB 与 adaptive-ratio instantiation 共用 Appendix A.2 的敏感度 \(s_l\)，并先计算所有有效层梯度范数的中位数 \(M\)。其裁剪尺度由 \(1.0\) 与 \(0.5\) 按 \(s_l\) 插值，即每层阈值为 \(M[(1-s_l)\cdot1.0+s_l\cdot0.5]\)，并在重建之前执行。随后采样高斯噪声 \(Z_l\)，构造 \(Z_l-T_{r_l}(Z_l)\)，将该残差归一到相对于裁剪后梯度范数的预设尺度后加到重建结果；噪声尺度同样按敏感度在 \(0.005\) 与 \(0.03\) 之间插值。该随机项与重建 signs 由相同的逐层 seed 规则控制，但二者通过各自的生成过程产生。

由于 \(T\) 不具备严格的正交算子性质，上述附加项仅称为 residual-space noise，不赋予严格正交性。核心 reconstruction 的符号调制、pooling 和 interpolation 按输入规模执行；optional adaptive allocator 还需要扫描完整梯度计算范数与尖峰统计，并为每层处理至多 4096 个校准元素。两种 instantiations 的总体时间复杂度均随梯度元素数线性增长，但常数开销不同。当前实现仍物化并输出同形状的 dense tensors，没有编码稀疏索引或低分辨率系数，因而不减少上传张量的通信字节数；实验中的时间指标用于记录这一额外本地变换成本，而非推断通信收益。

## 5. Experimental Setup

### 5.1 Data, Models, and Training

我们评估 SST-2、CoLA 与 Rotten Tomatoes 三个文本分类任务。SST-2 源自 Stanford Sentiment Treebank \cite{socher2013recursive}，并与 CoLA 一同采用 GLUE 的任务定义 \cite{wang2019glue,warstadt2019neural}；Rotten Tomatoes 使用电影评论情感数据 \cite{pang2005seeing}。所有主实验采用 GPT-2 sequence classification model \cite{radford2019language}。效用评估使用数据集官方 train/validation splits，并为每个任务从对应的、已训练 2 epochs 的 clean GPT-2 checkpoint 开始：full-gradient DAGER 对应的 full-model utility 在同一任务上继续训练 1 epoch、batch size 为 2；PTG 配对 utility 使用同一 SST-2 checkpoint 继续训练 1 epoch、batch size 为 1，以匹配 PTG privacy 的更新语义。两者均使用 AdamW、学习率 \(5\times10^{-5}\)、无 warmup 的 linear scheduler，并报告 seeds `101/202/303`。各防御与 clean 采用相同的继续训练入口和种子集合，以便在同一协议内比较下游指标。

Adapter utility 单独在 SST-2/GPT-2 Adapter 上运行，reduction factor 为 16，batch size 为 8，训练 3 epochs。它只为 Legacy 与 v2 中相同防御配置提供独立的任务效用参照；恶意 ratio probe 的 privacy 运行使用 batch size 1，且不会复用这次正常 Adapter 训练所上传的更新。因此两类日志属于 cross-protocol 汇总，不构成同一客户端、同一样本或同一上传更新上的严格 Pareto 比较。

Standard DAGER 主表中的 `split=val` 是脚本内部名称，并非官方 validation 或模型训练外 held-out set。数据加载器先随机排列原始 training split，将前 1000 条作为内部 `test`，再从其余样本中按长度分层构造攻击用 `val` subset。第 10 节的 knowledge-stratified evaluation 则单独使用 SST-2 official validation split，并在正式运行中记录 checkpoint digest、代码 commit、Python/PyTorch/Transformers/Datasets 版本及 GPU 型号。因而，两组 DAGER 结果的数据来源不同，只能在各自协议内解释，不能以 `val` 名称作直接横向比较。仓库提供 DAGER 与 PEFT 两套 Conda 配置，但本文仍不据环境文件反推其他历史正式运行的实际依赖版本。

### 5.2 Attack Protocols

| Evaluation | Data / exposure | Batch and scale | Optimization / knowledge |
| --- | --- | --- | --- |
| Standard DAGER | 三任务，完整梯度 | batch 2；每个点 100 个单客户端更新实例，共 200 条文本；3 seeds | 已知模型与协议，不显式反演 defense |
| Static knowledge-stratified DAGER | SST-2 official validation；完整防御梯度 | batch 2；每个 seed 100 个更新；3 seeds | oracle、ratio-hidden、signs-hidden、method-only；有限 ratio/sign hypotheses；`min` 聚合 |
| Per-update sign stress test | SST-2 official validation；每次更新重采样 signs | batch 2；每个 seed 100 个更新；3 seeds | oracle、signs-hidden、method-only；1/4/16/64 sign hypotheses；`min` 与 `mean` 分开报告 |
| Legacy Adapter ratio probe | SST-2 Adapter weight/bias gradients | batch 1；单次正式 privacy 运行含 100 条文本 | known label；sample-adaptive malicious probe；legacy public-bin statistics |
| Fixed-probe v2 | SST-2 Adapter weight/bias gradients | batch 1；每个 seed 100 条文本；3 seeds | fixed inventory；私有数据到达前安装；严格 disjoint public statistics |
| PTG `first2` | SST-2；前两个 GPT-2 blocks 的 24 个 gradient tensors，其中 8 个为 matrix gradients | batch 1；每个 seed 100 条文本；3 seeds | known label/padding；单 restart；Adam 80 steps，lr 0.1；cosine matching；embedding-norm weight 0.01 |

表中“batch”指生成一个可观察客户端更新所含的文本数，而“scale”指每个配置和种子的攻击样本量。Standard 与 knowledge-stratified DAGER 的 100 个观测单位因此是 100 个 batch size 2 的客户端更新，而不是 100 条文本；Adapter 与 PTG 的 batch size 为 1，观测单位与文本数相同。Static knowledge matrix 对每种 variant--knowledge 组合使用 seeds `101/202/303`，主攻击采用 `min` hypothesis reduction。Per-update 实验中的 `min` 是主攻击结果，`mean` 只用于 EOT-style sensitivity analysis，两者不得聚合为同一行。PTG 中防御先作用于完整更新，再选择 24 个可见梯度，并在单次初始化上执行 80 步匹配。七个正文 operating points 已具有同 checkpoint、batch size 1、1 epoch 和 seeds `101/202/303` 的配置对齐 utility；其余 PTG sweep 仍只作为 privacy extension。Fixed-probe v2 作为 Appendix A 的严格性检查，Legacy probe 则保留为主文机制实验。

### 5.3 Baselines

Standard DAGER Main Table 1 报告 Projection-LRB (adaptive ratios, k=0.5)，以及在 CoLA 和 Rotten Tomatoes 上具有完整配对 utility 的补充点 k=0.9；压缩对照统一列出 top-k@0.7/@0.9 与 compression@22/@32。它们用于展示高保留率或高 bit 区间的 privacy--utility 变化，不把 top-k sparsity、quantization bits 与 projection keep ratio 解释成相同的信息预算。更低保留率的 top-k 与更低 bit 的 compression 在独立 full sweep 中包含零恢复端点，但不与 Main Table 1 的高保留率点混称为同一 operating point。PTG Main Table 4 固定使用 none、Projection-LRB (adaptive ratios, k=0.2/0.5)、top-k@0.1、compression@8、noise@5e-4 和 DP-SGD-style@5e-4；这些点在 PTG 协议内单独解释，而不是按完整 PTG sweep 事后选择最优参数。Full-LRB、noise、DP-SGD-style、Soteria-style 与 mixup-style 仅用于组件消融或防御覆盖，不据此宣称统一的形式化隐私语义。尤其是 DP-SGD-style 没有 privacy accountant，不能视为正式 DP-SGD。

### 5.4 Metrics and Statistics

隐私指标为 token recovery 与 ROUGE-1/2；knowledge-stratified DAGER 另报告 candidate recall、Top-B recall 及 L1/L2 stage recovery，以定位攻击失败发生在哪个恢复阶段。效用指标为 accuracy、macro-F1 和 loss，CoLA 另报告 MCC；成本指标为训练时间与攻击时间。三种子结果先将每个正常完成的 seed-level aggregate 视为一个独立观测，再报告均值与 sample standard deviation（`ddof=1`）；不把 300 个样本混池，也不由旧表中已四舍五入的标准差反推。汇总排除失败、不完整、smoke 和重复日志，不用缺失项替代为零，也不把同一日志的重复副本计作独立种子。所有恢复指标只在攻击假设、可见更新、batch size 和优化预算一致的协议内部比较；DAGER、Adapter probe、fixed-probe v2 与 PTG 的绝对恢复率不做跨协议排序。`rec_token_mean=0` 只表示当前攻击在给定预算内未恢复 token，不能推出可量化的差分隐私结论。

## 6. Full-Gradient DAGER Results

### 6.1 Clean Updates Leak Severe Text Information

在完整梯度暴露下，未防御更新在三个任务上均表现出高可恢复性。Main Table 1 中，SST-2 的 `rec_token_mean` 为 `0.906254±0.029349`，R1+R2 为 `157.608202±7.120220`；Rotten Tomatoes 分别为 `0.885739±0.037738` 和 `164.658710±8.058300`。CoLA 的恢复进一步接近饱和，两个指标达到 `0.999755±0.000425` 和 `199.944272±0.096523`。这些统计来自每个种子 100 个单客户端更新实例；由于 batch size 为 2，每个 seed 实际覆盖 200 条文本。由此，clean leakage 并非 SST-2 单一任务上的偶然现象，而是在当前 GPT-2 sequence classification 设置中跨任务重复出现。

### 6.2 Standard-Attack Privacy and Utility

**Main Table 1. Full-gradient DAGER privacy and utility.** 所有数值均为 seeds `101/202/303` 的 mean ± sample SD；高保留率压缩基线的 12 个目标 utility 点均已满足三种子准入。`--` 表示该任务不使用 MCC。

| Dataset | Method | `rec_token_mean` | R1+R2 | Accuracy | Macro-F1 | MCC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SST-2 | none | `0.906254±0.029349` | `157.608202±7.120220` | `0.912462±0.008608` | `0.912433±0.008621` | -- |
| SST-2 | Projection-LRB (adaptive ratios, k=0.5) | `0.000000±0.000000` | `0.000000±0.000000` | `0.915137±0.004135` | `0.915086±0.004127` | -- |
| SST-2 | top-k@0.7 | `0.164238±0.025347` | `9.212706±1.141808` | `0.907110±0.007162` | `0.907084±0.007164` | -- |
| SST-2 | top-k@0.9 | `0.372119±0.033464` | `28.907796±4.370453` | `0.911315±0.007284` | `0.911291±0.007286` | -- |
| SST-2 | compression@22 | `0.380649±0.028887` | `18.443061±1.936717` | `0.912079±0.004027` | `0.912053±0.004017` | -- |
| SST-2 | compression@32 | `0.906078±0.028851` | `157.292912±7.157049` | `0.913609±0.006521` | `0.913577±0.006519` | -- |
| SST-2 | noise@1e-6 | `0.953844±0.014366` | `17.851501±1.417217` | `0.912080±0.002387` | `0.912058±0.002402` | -- |
| SST-2 | DP-SGD-style@1e-5 | `0.835020±0.018759` | `14.053037±1.427354` | `0.905199±0.005885` | `0.905172±0.005878` | -- |
| CoLA | none | `0.999755±0.000425` | `199.944272±0.096523` | `0.746564±0.005780` | `0.621441±0.020083` | `0.337219±0.020261` |
| CoLA | Projection-LRB (adaptive ratios, k=0.5) | `0.000000±0.000000` | `0.000000±0.000000` | `0.758709±0.009793` | `0.643373±0.028780` | `0.377492±0.030254` |
| CoLA | Projection-LRB (adaptive ratios, k=0.9) | `0.000000±0.000000` | `0.000000±0.000000` | `0.754874±0.007446` | `0.637656±0.018592` | `0.364403±0.024162` |
| CoLA | top-k@0.7 | `0.050093±0.008821` | `2.400210±1.241236` | `0.753915±0.005280` | `0.634566±0.012737` | `0.361955±0.017972` |
| CoLA | top-k@0.9 | `0.210144±0.012435` | `11.791838±1.232605` | `0.728987±0.034374` | `0.551285±0.121471` | `0.227149±0.203659` |
| CoLA | compression@22 | `0.514786±0.016399` | `23.367069±0.924317` | `0.744967±0.005337` | `0.621065±0.014293` | `0.332858±0.018412` |
| CoLA | compression@32 | `0.999755±0.000425` | `199.944272±0.096523` | `0.723234±0.027943` | `0.538217±0.112291` | `0.207395±0.180117` |
| CoLA | noise@1e-6 | `1.000000±0.000000` | `56.555866±4.261990` | `0.723874±0.028231` | `0.537808±0.111819` | `0.209653±0.181565` |
| CoLA | DP-SGD-style@1e-5 | `1.000000±0.000000` | `11.921810±5.212777` | `0.725471±0.003630` | `0.557900±0.011202` | `0.258227±0.014689` |
| Rotten Tomatoes | none | `0.885739±0.037738` | `164.658710±8.058300` | `0.862414±0.004434` | `0.862345±0.004484` | -- |
| Rotten Tomatoes | Projection-LRB (adaptive ratios, k=0.5) | `0.000000±0.000000` | `0.000000±0.000000` | `0.860225±0.002481` | `0.860136±0.002529` | -- |
| Rotten Tomatoes | Projection-LRB (adaptive ratios, k=0.9) | `0.000000±0.000000` | `0.000000±0.000000` | `0.860538±0.009764` | `0.860474±0.009721` | -- |
| Rotten Tomatoes | top-k@0.7 | `0.021667±0.015275` | `0.193630±0.171610` | `0.863665±0.008717` | `0.863622±0.008728` | -- |
| Rotten Tomatoes | top-k@0.9 | `0.063647±0.011004` | `1.693354±0.398930` | `0.862102±0.005706` | `0.862057±0.005686` | -- |
| Rotten Tomatoes | compression@22 | `0.163051±0.024694` | `2.747972±1.140052` | `0.870231±0.006655` | `0.870205±0.006650` | -- |
| Rotten Tomatoes | compression@32 | `0.883667±0.037650` | `164.304292±8.432649` | `0.860850±0.009396` | `0.860805±0.009426` | -- |
| Rotten Tomatoes | noise@1e-6 | `0.998180±0.000893` | `17.674292±0.271596` | `0.862727±0.005167` | `0.862681±0.005127` | -- |
| Rotten Tomatoes | DP-SGD-style@1e-5 | `0.995462±0.000999` | `8.744707±1.390282` | `0.845841±0.006246` | `0.845839±0.006248` | -- |

Projection-LRB 的 adaptive-ratio k=0.5 instantiation 在三个任务和三个种子上均将 standard DAGER 的两个恢复指标降为 0，同时效用均值保持在 clean 附近。SST-2 上其 accuracy 与 clean 分别为 `0.915137` 和 `0.912462`；CoLA 上 accuracy、macro-F1 与 MCC 的均值均高于 clean，但当前三个种子不足以支持统计显著提升，只能说明未观察到效用损失。Rotten Tomatoes 上 accuracy 的均值下降 `0.002189`，约为 0.22 percentage points。Adaptive-ratio k=0.9 的 privacy 在三个任务上也均为零恢复，但其严格配对 utility 当前只覆盖 CoLA 与 Rotten Tomatoes，因此 SST-2 k=0.9 不进入 combined table。

高保留率 top-k 与高 bit compression 呈现出非零、且随任务和参数明显变化的 standard-DAGER 恢复区间。Top-k@0.7 的 `rec_token_mean` 在 SST-2、CoLA 和 Rotten Tomatoes 上分别为 `0.164238`、`0.050093` 和 `0.021667`，提高到 @0.9 后分别为 `0.372119`、`0.210144` 和 `0.063647`。Compression@22 仍能降低恢复，而 @32 在三个任务上均接近 clean，符合高精度量化趋近原更新的预期。12/12 个目标点现均有配对的三种子 utility；SST-2 compression@32 与 Rotten Tomatoes compression@22 的 accuracy 分别为 `0.913609±0.006521` 和 `0.870231±0.006655`。SST-2 和 Rotten Tomatoes 的目标点整体接近各自 clean 均值；CoLA top-k@0.9 与 compression@32 的 accuracy 分别为 `0.728987±0.034374` 和 `0.723234±0.027943`，且 macro-F1/MCC 的种子间波动较大。

DP-SGD-style@1e-5 在 CoLA 与 Rotten Tomatoes 上的 token recovery 仍接近 clean，在 SST-2 上也仅降至 `0.835020`，同时三个任务的效用均低于各自 clean 均值。该实现只包含逐样本裁剪与高斯噪声，没有 privacy accountant，因而不提供可报告的 \(\epsilon,\delta\)-DP 保证。运行日志还显示，当前 Projection-LRB 实现的训练时间显著高于 none 和已有压缩基线；由于正式日志未完整记录跨机器硬件环境，本文不做跨批次绝对时间排名。输出张量仍为 dense 且形状不变，因此该方法当前也没有通信字节收益。上述 standard-attack 结果只说明当前解码器未能利用变换后的更新；第 10 节通过知识分层评估进一步区分“候选 token 仍然存在”与“攻击能否完成排序和序列组装”。

## 7. What Disrupts Standard Recovery?

### 7.1 Component Ablation

**Main Table 2. Projection-LRB mechanism analysis.** Panel A 来自独立的 SST-2、batch 2、`./models/gpt2-ft-rt` 三种子批次，与 Main Table 1 的 2026-07 checkpoints 不同。Panel B 使用 SST-2 official validation、batch 2、static defense seed `700001` 和 oracle DAGER；所有三种子项均报告 sample SD。

**Panel A: component ablation under standard DAGER.**

| Variant | `rec_token_mean` | R1+R2 | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| none | `0.855333±0.031263` | `148.260102±7.775494` | `0.913226±0.006521` | `0.913184±0.006501` |
| identity | `0.855333±0.031263` | `148.260102±7.775494` | `0.913226±0.006521` | `0.913184±0.006501` |
| clip-only | `0.854076±0.033792` | `147.955623±7.998571` | `0.918196±0.002885` | `0.918149±0.002897` |
| projection-only (adaptive ratios) | `0.000000±0.000000` | `0.000000±0.000000` | `0.915520±0.003686` | `0.915499±0.003696` |
| projection+clipping (adaptive ratios) | `0.000000±0.000000` | `0.000000±0.000000` | `0.913226±0.005171` | `0.913197±0.005154` |
| Full-LRB (adaptive ratios) | `0.000000±0.000000` | `0.000000±0.000000` | `0.892584±0.013439` | `0.892472±0.013547` |

**Panel B: ratio and signed controls under oracle DAGER.**

| Configuration | Candidate recall | Top-B recall | R1+R2 |
| --- | ---: | ---: | ---: |
| adaptive ratios@0.5 | `0.919333±0.019112` | `0.914969±0.019117` | `101.872623±3.734154` |
| uniform signed@0.5 | `0.921943±0.020668` | `0.916526±0.018586` | `101.933598±2.938826` |
| uniform signed@0.65 | `0.930479±0.024997` | `0.925149±0.023091` | `101.176730±3.467582` |
| uniform signed@0.75 | `0.927909±0.021159` | `0.922875±0.018806` | `104.101123±3.511430` |
| uniform signed@0.9 | `0.931242±0.021103` | `0.926690±0.019512` | `104.571362±3.670613` |
| uniform unsigned@0.5 | `0.907101±0.017703` | `0.903459±0.015739` | `101.143199±4.638276` |
| undefended anchor | `0.976223±0.010324` | `0.970630±0.005882` | `122.556760±5.062032` |

Identity 与 none 的 privacy 和 utility 完全一致，排除了仅由防御路由器或调用路径造成结果变化的解释。Clip-only 的 `rec_token_mean=0.854076` 与 none 的 `0.855333` 接近，表明中位范数裁剪本身没有破坏 standard DAGER 所利用的结构。相反，projection-only 已将两个恢复指标降为 0，并保持与 clean 接近的 accuracy；加入 clipping 后仍为同一隐私终点，但未带来可辨识的额外收益。Full-LRB 同样为零恢复，却将 accuracy 均值降至 `0.892584`，比 none 低约 2.06 percentage points。该消融将低分辨率重建识别为当前 standard-attack 结果的主要有效组件，也支持将 Full-LRB 视为 over-defense 参照而非主方法。

### 7.2 The Ratio Allocator Is Not a Necessary Component

Appendix Table A2 进一步比较 rule-only、empirical-only、uniform 与 no-empirical variants。在同一个 k=0.5 批次中，这些变体均达到 standard DAGER 零恢复，accuracy 均值位于 `0.914373` 至 `0.917049`。Panel B 又显示 adaptive@0.5 与 uniform signed@0.5 的 oracle R1+R2 分别为 `101.872623` 和 `101.933598`，同样没有可辨识差异。因此，现有实验不能证明 hybrid layer sensitivity 是达到隐私端点或提高 defense-aware suppression 的必要条件；本文将 ratio allocation 从核心算子中分离，并把 Hybrid 规则降级为 Appendix A.2 的可选 instantiation。

### 7.3 Uniform Reconstruction Resolution under Oracle DAGER

为避免 standard DAGER 的零恢复饱和掩盖分辨率效应，我们在 SST-2 official validation 上固定 uniform ratios，使用知道 realized ratios 与 signs 的 oracle DAGER 比较 \(r=0.5/0.65/0.75/0.9\)，并以同协议 undefended updates 作为独立 anchor。每个配置使用 seeds `101/202/303`、100 个 batch-size-2 updates、static defense seed `700001` 和 100 倍候选扩展。完整 L1/L2 与 paired cluster bootstrap 结果见 Appendix A.4。

Uniform signed 的 R1+R2 在 \(r=0.5/0.65/0.75/0.9\) 上分别为 `101.933598`、`101.176730`、`104.101123` 和 `104.571362`，形成非单调平台，而非“ratio 越低、oracle recovery 越低”的有序曲线。相邻差异中，`0.5→0.65` 与 `0.75→0.9` 的 95% CI 均跨 0；`0.65→0.75` 虽为正，但与整体单调假设不一致。Unsigned@0.5 的 R1+R2 为 `101.143199±4.638276`，也没有显示 signed modulation 在 oracle 下带来额外抑制。相反，undefended anchor 达到 `122.556760±5.062032`，与 uniform@0.5/@0.9 的配对差异 CI 均不跨 0。因而预设结论落在 `span_mismatch_only` 分支：当前证据支持 reconstruction-induced span transformation 相对未防御更新改变可恢复性，但不支持把 standard 零恢复解释为单调的 resolution effect，亦不支持 signed 在状态公开时优于 unsigned pooling。

## 8. Adapter-Gradient Leakage: PEFTLeak-Style Ratio Recovery

### 8.1 Mechanistic Evaluation with a PEFTLeak-Style Adapter Ratio Probe

我们使用 Legacy Adapter ratio probe 检查低分辨率重建是否会破坏恶意 Adapter probe 所依赖的 weight/bias gradient ratios。该评估是 batch 1、known-label、sample-adaptive 的受控机制实验：probe inventory 随当前私有样本构造，攻击利用 legacy public-bin statistics 进行路由。因此，它既不是原始图像侧 PEFTLeak 的文字复现，也不是对正常 Adapter 或 LoRA 更新的被动部署攻击。

**Main Table 3. Controlled PEFTLeak-style Adapter ratio probe.** Privacy 为每配置单次 100/100 正式运行，不报告虚构标准差；utility 为对应防御配置下 benign Adapter training 的 seeds `101/202/303` mean ± sample SD。两者属于 cross-protocol comparison。

| Method | `rec_token_mean` | R1+R2 | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| none | `0.760000` | `145.000000` | `0.916667±0.003504` | `0.916589±0.003491` |
| Projection-LRB (adaptive ratios, k=0.5) | `0.202490` | `12.036481` | `0.909786±0.005657` | `0.909698±0.005691` |
| Projection-LRB (adaptive ratios, k=0.65) | `0.212765` | `13.693024` | `0.914373±0.004634` | `0.914304±0.004650` |
| Projection-LRB (adaptive ratios, k=0.9) | `0.234016` | `13.137760` | `0.913226±0.006720` | `0.913150±0.006736` |
| top-k@0.05 | `0.431431` | `48.125711` | `0.903670±0.001148` | `0.903515±0.001182` |
| compression@4 | `0.115571` | `7.733200` | `0.901758±0.003686` | `0.901622±0.003706` |
| Full-LRB@0.5 | `0.015222` | `0.461551` | `0.850535±0.020811` | `0.850337±0.020861` |

Projection-LRB (adaptive ratios, k=0.5) 将 token recovery 从 `0.760000` 降至 `0.202490`，并将 R1+R2 从 `145.000000` 降至 `12.036481`；其 accuracy 均值为 `0.909786`，相对 benign none 低 `0.006881`。在这一受控协议内，它同时比 top-k@0.05 获得更低恢复和更高效用。Compression@4 与 Full-LRB@0.5 的恢复进一步降低，但 accuracy 分别降至 `0.901758` 和 `0.850535`。这些结果支持 low-resolution reconstruction 会扰乱该恶意 probe 使用的梯度比值关系，但不能把单次 privacy 运行与三种子 utility 解释为同一上传更新上的严格 Pareto front，也不能据此推出对一般 Adapter/LoRA updates 的防御。

### 8.2 Strictness Check and Scope

Appendix Table A1 的 fixed-probe v2 在私有样本到达前安装覆盖 128 个 token positions、包含 256 个 weight/bias gradient tensors 的固定 inventory，并严格隔离 public statistics。在该协议中，Projection-LRB (adaptive ratios, k=0.5) 的三种子 token recovery 为 `0.685368±0.019729`，明显高于 Legacy probe 的单次 `0.202490`；noise@1e-3 则达到更低的 `0.358894±0.010977`。由于两个 probe 协议并不相同，这一对照不能作为逐点统计比较，但足以说明 Legacy 的 sample-adaptive 设计高估了 Projection-LRB 的实际抑制幅度。本文因此只将 Main Table 3 作为 Adapter ratio mechanism evidence，不外推到 LoRA、一般 PEFT updates 或现实部署攻击。

### 8.3 Supplementary Image-Side Transfer

为检验同一 reconstruction mechanism 是否只适用于文本 GPT-2，我们还沿 PEFTLeak 原工作框架实施了一项 official-aligned、source-aligned 的图像侧补充研究。协议使用 CIFAR-100、batch size 32、upstream-style malicious ViT with Adapters 和 seed 42；防御先作用于完整 96 个 Adapter gradient tensors，攻击再读取 20 对 weight/bias gradients。未防御更新的 patch recovery 为 `0.109375`，而 Projection-LRB 的 adaptive-ratio k=0.5/0.75/0.9 instantiations 均为 0。对应 full-image MSE 从 none 的 `0.034469` 上升至 `0.260880--0.283339`，SSIM 从 `0.753896` 降至 `0.020130--0.029032`，LPIPS 从 `0.245784` 上升至 `0.687894--0.694709`。

这一结果表明，同一个低分辨率 reconstruction operator 可以从文本 GPT-2 转移到图像 ViT/Adapter 的已评估攻击路径，因而提供了补充性的跨模态机制证据。它不是逐字节复现，也不是多客户端 FedAvg 实验；更重要的是，top-k、compression 和 Full-LRB 在同一 fixed victim batch 上同样达到零 patch recovery。因此，该实验不支持 Projection-LRB 优于强基线，不证明普遍跨模态泛化，也不能替代真实部署中的多种子 privacy--utility 评估。完整九个配置及指标定义见 Appendix A.6。

## 9. Partial-Gradient Leakage under PTG

### 9.1 PTG `first2` Protocol

PTG 评估在 SST-2/GPT-2 上仅暴露前两个 Transformer blocks。防御先作用于完整更新，随后 selector 保留 24 个 gradient tensors，其中 8 个为 matrix gradients；攻击者已知 batch size 1 的标签与 padding，并以单次初始化执行 80 个 Adam steps。每个配置在 seeds `101/202/303` 上各攻击 100 条文本。该 gradient-matching 协议的恢复率与 DAGER 或 Adapter ratio probe 不可直接比较。

### 9.2 Repeatable but Limited Privacy–Utility Transfer

未防御 `first2` 更新的 `rec_token_mean` 为 `0.160194±0.016123`，R1+R2 为 `19.772638±1.583317`。Projection-LRB (adaptive ratios, k=0.2) 将二者降至 `0.121290±0.018283` 和 `14.616599±3.928618`。按相同 seed 配对，token recovery 的绝对下降为 `0.038904±0.028534`；由聚合均值计算，相对下降为 24.3%。三个 seed 的 token recovery 与 R1+R2 均沿相同方向下降，说明该变化在当前样本抽取下可重复，但幅度远未达到 full-gradient standard DAGER 中的零恢复端点。

**Main Table 4. PTG `first2` selected privacy–utility results.** 每行的 privacy 与 utility 均按 seeds `101/202/303` 计算 mean ± sample SD。Utility 使用与 PTG 相同的 SST-2 checkpoint、batch size 1 和 defense operating point 继续训练 1 epoch；训练时间为单次 utility run 的 wall-clock hours。

| Method | `rec_token_mean` | R1+R2 | Accuracy | Macro-F1 | Loss | Train time (h) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | `0.160194±0.016123` | `19.772638±1.583317` | `0.900230±0.004135` | `0.900178±0.004143` | `0.272405±0.004148` | `0.946±0.024` |
| Projection-LRB (adaptive ratios, k=0.2) | `0.121290±0.018283` | `14.616599±3.928618` | `0.913226±0.002388` | `0.913171±0.002409` | `0.258121±0.004659` | `10.871±1.136` |
| Projection-LRB (adaptive ratios, k=0.5) | `0.125698±0.013819` | `16.236530±1.106815` | `0.918960±0.002885` | `0.918945±0.002887` | `0.255761±0.012472` | `14.086±3.558` |
| top-k@0.1 | `0.154005±0.025120` | `18.806127±4.648958` | `0.911315±0.007807` | `0.911270±0.007847` | `0.265506±0.021022` | `3.323±0.852` |
| compression@8 | `0.149767±0.014262` | `19.011232±2.390093` | `0.911697±0.001987` | `0.911668±0.001999` | `0.248106±0.005608` | `4.526±0.551` |
| noise@5e-4 | `0.078178±0.004606` | `7.138664±2.085052` | `0.909404±0.008028` | `0.909391±0.008036` | `0.253409±0.020606` | `1.226±0.080` |
| DP-SGD-style@5e-4 | `0.070653±0.003690` | `6.569028±1.760894` | `0.920107±0.001751` | `0.920081±0.001744` | `0.633798±0.017393` | `1.725±0.026` |

Projection-LRB 的 adaptive-ratio k=0.2 与 k=0.5 instantiations 均降低当前 PTG 恢复指标，并在三种子均值上未观察到 accuracy 或 macro-F1 损失；这只支持 clean-level utility，不构成统计显著提升。代价是两点的平均训练时间分别达到 clean 的约 11.5 倍和 14.9 倍。Top-k@0.1 与 compression@8 在该攻击下仅带来较小恢复变化；noise 与 DP-SGD-style 的恢复率更低，但 DP-SGD-style 的 eval loss 升至 `0.633798±0.017393`，且没有 privacy accountant。完整 sweep 中 compression@2 的 token recovery 为 `0.055496±0.008613`，仍低于表内所有防御。因而，该表支持 Projection-LRB 向 `first2` gradient matching 的有限迁移，但不支持其在 PTG 下优于压缩或噪声基线。

### 9.3 Exposure Boundary

当前 partial-gradient 证据只覆盖 `first2`，因此所有结果和结论均限定于这一可见梯度设置，不能据此外推到其他 partial-gradient exposures。Appendix Table A3 报告未进入七点正文表的八个 `first2` full-sweep privacy 点，以避免把不同调参覆盖误读为同一完整 trade-off。

## 10. Mechanistic Implications and Limitations

三类正文协议回答同一个客户端侧 transform 能否在不同更新暴露路径中降低当前攻击恢复；oracle 与状态隐藏实验则检验这种抑制依赖哪些攻击知识。在 SST-2 official validation 上，知道 realized ratios 与 signs 的 oracle DAGER 对 adaptive-ratio@0.5 恢复 `101.872623±3.734154` 的 R1+R2，远高于 standard DAGER 的零恢复。这一结果排除了“pooling 已从共享更新中删除所有 token 信息”的解释。结合 uniform-oracle 非单调平台，更符合证据的机制是 operator-induced span transformation，而不是 ratio 单调决定 recoverability。

当 ratios 或 signs 未直接提供时，candidate recall 仍为 `0.978630--0.984620`，但 L1/L2 均为 0，最终 R1+R2 仅为 `6.45--9.42`。真实 token 因而大多仍进入扩展候选池；攻击失败主要发生在候选排序、过滤和序列组装。Signed modulation 在这里应视为隐藏时的 nuisance state：它改变 realized reconstruction coordinates，但 signed@0.5 与 unsigned@0.5 在 oracle 下不可区分，不能声称 signs 本身产生额外的代数降维或保留优化方向。完整 static matrix 与 102/102 per-update finite-hypothesis stress test 见 Appendix A.5。后者只测量有限 seed sampling 对 `min`/`mean` 聚合的敏感性，不等价于一般 EOT、直接状态估计、隐变量优化或跨更新学习。

实验范围仍有五项主要限制。第一，文本主实验只覆盖 GPT-2 sequence classifiers 和三个分类任务；除机制实验使用 official validation 外，历史 standard-DAGER 主表使用来自 training split 的内部 `val` subset。第二，正文 Adapter 部分是 controlled PEFTLeak-style ratio probe，privacy 与 benign utility 属于 cross-protocol comparison；fixed-probe v2 也显示 Legacy probe 高估了抑制幅度。第三，PTG 只覆盖 `first2` exposure，且 Projection-LRB 的抑制有限、训练成本约为 clean 的 11.5--14.9 倍，compression/noise 具有更强经验端点。第四，图像侧结果仅为 fixed victim batch、single seed、privacy-only、non-FedAvg 的 official-aligned study，不提供图像任务效用或普遍跨模态结论。第五，当前实现输出同形状 dense tensors，不减少通信字节数；DP-SGD-style baseline 也没有 privacy accountant。

因此，本文只主张 attack-specific empirical mitigation 与同一更新接口的可复用性，不主张 formal privacy、一般白盒稳健性、Hybrid allocator 的必要性、signed oracle advantage，或 Projection-LRB 普遍优于 top-k、compression 与 noise。更大模型、生成任务、真实 LoRA/Adapter updates、其他 partial-gradient exposures、多客户端聚合、non-IID sampling、FedAvg 多轮本地更新、跨更新状态学习和 secure aggregation 均属于后续外部有效性验证。

## 11. Conclusion

本文探索一个面向多种文本重建攻击的统一更新空间防御。Projection-LRB 在客户端上传前对更新施加同一个低分辨率 signed reconstruction bottleneck，完整梯度、Adapter 参数子集与部分层梯度的暴露均发生在该变换之后。由此，防御不依赖 DAGER、PEFTLeak-style ratio recovery 或 PTG 的特定解码流程，而是直接干预三者共同读取的更新结构。

实验表明，Projection-LRB 的 adaptive-ratio instantiation 在三个 GPT-2 文本分类任务的 standard DAGER 下将当前 token 与 ROUGE 恢复指标降至零并保持 clean-level utility，在受控 Adapter ratio probe 和 PTG `first2` 中也分别降低恢复；单种子图像侧研究提供了补充性的 ViT/Adapter 转移证据。组件消融将低分辨率 reconstruction 识别为主要有效部分，并未显示 adaptive allocator 相对 uniform schedule 的必要性。Uniform-oracle sweep 呈现非单调恢复平台，知识分层又显示 oracle 获得 realized state 后可恢复大量序列，而隐藏状态时真实 token 仍位于候选池中，失败主要发生在候选排序、过滤和序列组装阶段。

因此，本文的贡献是提出并实验探索一个可跨三类文本更新暴露复用、并在一个图像侧协议中显示补充转移的经验防御框架，而不是证明某个 operating point 对所有攻击都最优。当前证据不支持不可逆信息删除、形式化隐私、layer-wise adaptivity 的必要性、signed oracle advantage、单调分辨率效应或一般白盒稳健性。真实 PEFT update 攻击、更多 PTG exposures、跨更新状态学习以及多模型多数据协议，是将这一框架推进为更强部署结论所需的下一步。

# Appendix

## Appendix A.1 Fixed-Probe Adapter Ratio Strictness Check

为检验 Legacy probe 的严格性，我们在 SST-2/GPT-2 Adapter 设置中固定一个覆盖 128 个 token positions、包含 256 个 weight/bias gradient tensors 的 probe inventory，并在观察任何私有样本之前完成安装；public statistics 来自与私有评估严格不相交的分区。攻击通过任务梯度观测标签，使用 seeds `101/202/303`，每个配置、每个种子攻击 100 条文本。表 A1 报告三种子的均值与 sample SD。Privacy 与 utility 使用配置对齐但彼此独立的攻击和训练运行，属于 cross-protocol 汇总，不构成逐样本或同一更新上的 Pareto 比较。

| Method | `rec_token_mean` | R1+R2 | Accuracy | Macro-F1 | Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | `1.000000±0.000000` | `193.666667±1.154700` | `0.916667±0.003504` | `0.916589±0.003491` | `0.240148±0.008295` |
| Projection-LRB (adaptive ratios, k=0.5) | `0.685368±0.019729` | `98.674391±2.675286` | `0.909786±0.005657` | `0.909698±0.005691` | `0.252664±0.004481` |
| Projection-LRB (adaptive ratios, k=0.65) | `0.761991±0.020442` | `117.764104±1.432629` | `0.914373±0.004634` | `0.914304±0.004650` | `0.250359±0.005149` |
| Projection-LRB (adaptive ratios, k=0.75) | `0.836252±0.014610` | `136.816461±1.963298` | `0.911315±0.007635` | `0.911226±0.007657` | `0.246870±0.006813` |
| Projection-LRB (adaptive ratios, k=0.9) | `0.837218±0.014158` | `137.105283±1.633826` | `0.913226±0.006720` | `0.913150±0.006736` | `0.246265±0.006243` |
| top-k@0.1 | `1.000000±0.000000` | `193.666667±1.154700` | `0.907875±0.005419` | `0.907739±0.005460` | `0.248322±0.004294` |
| compression@6 | `0.999444±0.000963` | `193.433333±0.750556` | `0.909786±0.004028` | `0.909692±0.004042` | `0.246968±0.003666` |
| noise@1e-3 | `0.358894±0.010977` | `58.458704±0.805420` | `0.912844±0.003034` | `0.912731±0.003050` | `0.243639±0.007508` |

在这一更严格协议中，Projection-LRB (adaptive ratios, k=0.5) 仍降低当前探针的恢复指标，但幅度明显小于 Legacy 机制实验；noise@1e-3 的恢复率更低，是该协议下更强的经验隐私基线。这一结果不支持将旧 probe 的抑制幅度外推到标准 Adapter、LoRA 或一般 PEFT 更新。

## Appendix A.2 Optional Adaptive Ratio Allocation and Checks

本文跨暴露主结果使用一个可选的 adaptive-ratio instantiation。该 allocator 不改变 Algorithm 1，只为每层产生 \(r_l\)。对每个参数名，结构先验 \(p_l\) 按固定分支顺序取值：embedding 与前两个 Transformer blocks 的 attention 参数为 \(1.0\)，前两个 blocks 的其余参数为 \(0.7\)，其他 attention 参数为 \(0.45\)，尚未命中的 classifier/lm-head、bias 或 LayerNorm 参数为 \(0.15\)，其余参数为 \(0.25\)。分支顺序属于实现定义，因此 early-block bias 和 LayerNorm 先命中 early-layer 规则。

经验分数使用当前更新的范数、低分辨率校准残差和尖峰统计：

\[
m_l^{\mathrm{norm}}=\log(1+\lVert G_l\rVert_2),\qquad
m_l^{\mathrm{res}}=
\frac{\lVert S_l-T_{\rho_{\mathrm{cal}}}(S_l)\rVert_2}
{\lVert S_l\rVert_2+\epsilon},
\]

\[
m_l^{\mathrm{spike}}=
\log\!\left(1+\frac{\max |G_l|}{\operatorname{RMS}(G_l)+\epsilon}\right).
\]

其中 \(S_l\) 是最多包含 4096 个元素的确定性 stride sample。三个统计量分别在有效梯度层之间做 min-max normalization；若某一项跨层为常数，则归一化值统一设为 \(0.5\)。实现随后计算

\[
e_l=0.45\widehat m_l^{\mathrm{norm}}
+0.40\widehat m_l^{\mathrm{res}}
+0.15\widehat m_l^{\mathrm{spike}},
\qquad
s_l=0.4p_l+0.6e_l,
\]

并通过

\[
r_l=(1-s_l)\cdot0.75+s_l\cdot k,
\qquad
\rho_{\mathrm{cal}}=\frac{0.75+k}{2}
\]

得到逐层 ratio。该参数化只有在 \(k<0.75\) 时才使较高 \(s_l\) 对应更低分辨率；\(k>0.75\) 时方向反转。上述常数是实现选择而非由理论推出，以下消融专门检验该 allocator 是否带来可辨识收益。

表 A2(a) 使用 k=0.5 的三种子 fine-ablation 批次。`no-empirical` 的原始运行三条均失败，故使用同协议完成的 resume 批次；其余变体来自同一主批次。所有 privacy 行均正常完成 100/100，utility 均包含 seeds `101/202/303`。

| Variant | `rec_token_mean` | R1+R2 | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| hybrid allocator (projection-only) | `0.000000±0.000000` | `0.000000±0.000000` | `0.915137±0.004135` | `0.915086±0.004127` |
| rule-only | `0.000000±0.000000` | `0.000000±0.000000` | `0.914373±0.002387` | `0.914339±0.002410` |
| empirical-only | `0.000000±0.000000` | `0.000000±0.000000` | `0.917049±0.001324` | `0.917003±0.001330` |
| uniform | `0.000000±0.000000` | `0.000000±0.000000` | `0.917049±0.004028` | `0.917002±0.004025` |
| no-empirical | `0.000000±0.000000` | `0.000000±0.000000` | `0.914373±0.002387` | `0.914339±0.002410` |

表 A2(b) 只报告具有完整三种子 privacy 的 Projection-LRB keep-ratio sweep。k=0.5 与 k=0.75 的对应 utility 各有一个训练日志缺少正常 result status，因此不将两种子 utility 与其他三种子行混排。

| Sensitive keep ratio k | `rec_token_mean` | R1+R2 |
| ---: | ---: | ---: |
| 0.5 | `0.000000±0.000000` | `0.000000±0.000000` |
| 0.65 | `0.000000±0.000000` | `0.000000±0.000000` |
| 0.75 | `0.000000±0.000000` | `0.000000±0.000000` |
| 0.9 | `0.000000±0.000000` | `0.000000±0.000000` |

这些结果表明 standard DAGER 在该批次中对多种 sensitivity 设计和 keep ratio 都出现饱和式攻击失败，因而不能由零恢复进一步识别 sensitivity 设计的必要性。特别地，k=0.9 位于 sensitivity-to-keep-ratio 方向反转的一侧，不代表对敏感层施加了更窄瓶颈。结合 Appendix A.5 中各 allocator 的 oracle R1+R2 均约为 101--103，本文将该机制保留为可选实例化，不把它列为 Projection-LRB 的必要成分或独立贡献。

## Appendix A.3 PTG `first2` Full-Sweep Privacy Extension

表 A3 保留未进入七点 Main Table 4 的八个三种子 privacy 点。协议与第 9 节完全相同，所有行均为 3×`ok` 且完成 3×100/100；本表用于公开完整参数 sweep，不将缺少同表 utility 的行用于完整 trade-off 排名。

| Method | `rec_token_mean` | R1+R2 |
| --- | ---: | ---: |
| Projection-LRB (adaptive ratios, k=0.65) | `0.143678±0.013432` | `17.769638±1.314082` |
| Projection-LRB (adaptive ratios, k=0.75) | `0.125995±0.011764` | `17.722520±0.296722` |
| Projection-LRB (adaptive ratios, k=0.9) | `0.149386±0.009948` | `17.798912±1.789991` |
| top-k@0.05 | `0.145112±0.005394` | `18.035789±0.614641` |
| top-k@0.3 | `0.157531±0.005060` | `20.273679±0.635447` |
| compression@2 | `0.055496±0.008613` | `6.346535±1.159283` |
| compression@4 | `0.116038±0.005164` | `13.742571±3.370739` |
| compression@16 | `0.161157±0.007200` | `18.846869±1.798701` |

## Appendix A.4 Uniform Reconstruction Resolution under Oracle DAGER

本实验使用 SST-2 official validation、GPT-2、batch size 2、100 updates/seed、seeds `101/202/303`、static defense seed `700001` 和 candidate multiplier 100。`none` 是独立的 undefended anchor，不被解释为算子族中的连续 \(r=1\) 点。三种子列为 mean ± sample SD。

| Configuration | Candidate recall | Top-B recall | L1 | L2 | R1+R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| uniform signed@0.5 | `0.921943±0.020668` | `0.916526±0.018586` | `0.433333±0.035119` | `0.791667±0.023094` | `101.933598±2.938826` |
| uniform signed@0.65 | `0.930479±0.024997` | `0.925149±0.023091` | `0.438333±0.040104` | `0.785000±0.022913` | `101.176730±3.467582` |
| uniform signed@0.75 | `0.927909±0.021159` | `0.922875±0.018806` | `0.446667±0.033292` | `0.798333±0.012583` | `104.101123±3.511430` |
| uniform signed@0.9 | `0.931242±0.021103` | `0.926690±0.019512` | `0.448333±0.030139` | `0.791667±0.018930` | `104.571362±3.670613` |
| undefended anchor | `0.976223±0.010324` | `0.970630±0.005882` | `0.553333±0.025658` | `0.798333±0.017559` | `122.556760±5.062032` |
| uniform unsigned@0.5 | `0.907101±0.017703` | `0.903459±0.015739` | `0.430000±0.040927` | `0.800000±0.018028` | `101.143199±4.638276` |

相同 seed 与样本上的 paired cluster bootstrap 以 R1+R2 为统计量：

| Comparison | Mean delta (high minus low) | Paired 95% CI |
| --- | ---: | ---: |
| `0.5→0.65` | `-0.767660` | `[-2.986876, 1.138422]` |
| `0.65→0.75` | `2.922980` | `[0.348377, 5.994435]` |
| `0.75→0.9` | `0.504543` | `[-1.414867, 2.208970]` |
| `0.9→none` | `17.951263` | `[13.885826, 22.530650]` |
| `0.5→none` | `20.611127` | `[15.915719, 25.487489]` |

四个 uniform ratios 没有形成有序曲线，预设分析分支因此为 `span_mismatch_only`。该结果反驳 evaluated range 内的单调分辨率效应，但仍支持 reconstruction operator 相对 undefended anchor 改变 defense-aware recoverability。Signed@0.5 与 unsigned@0.5 也不可区分，故不主张 signed oracle advantage。

## Appendix A.5 Attacker Knowledge and Per-Update Sign Stress Test

Static matrix 使用 SST-2 official validation、static defense seed `700001`、seeds `101/202/303`、100 updates/seed 和 candidate multiplier 100。60/60 正式运行均为 `ok`、100/100 且日志唯一。表 A5(a) 报告完整的 5 variants × 4 knowledge settings；所有值为 mean ± sample SD。

| Variant | Knowledge | Candidate | Top-B | L1 | L2 | R1+R2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| adaptive | oracle | `0.919333±0.019112` | `0.914969±0.019117` | `0.436667±0.038837` | `0.795000±0.017321` | `101.872623±3.734154` |
| adaptive | ratio-hidden | `0.984620±0.002225` | `0.947560±0.015939` | `0±0` | `0±0` | `9.421112±0.517537` |
| adaptive | signs-hidden | `0.978630±0.014517` | `0.574774±0.064957` | `0±0` | `0±0` | `6.453098±0.561383` |
| adaptive | method-only | `0.977448±0.013724` | `0.699022±0.050986` | `0±0` | `0±0` | `7.265339±0.630267` |
| rule-only | oracle | `0.923843±0.022118` | `0.918678±0.020299` | `0.430000±0.036056` | `0.790000±0.018028` | `100.989458±2.804796` |
| rule-only | ratio-hidden | `0.923843±0.022118` | `0.918678±0.020299` | `0.430000±0.036056` | `0.790000±0.018028` | `100.989458±2.804796` |
| rule-only | signs-hidden | `0.978033±0.014003` | `0.593826±0.026685` | `0±0` | `0±0` | `6.100656±0.056769` |
| rule-only | method-only | `0.978824±0.012634` | `0.650588±0.031263` | `0±0` | `0±0` | `7.067853±0.656427` |
| empirical-only | oracle | `0.922438±0.020934` | `0.914863±0.024203` | `0.440000±0.042720` | `0.796667±0.027538` | `102.583330±4.219645` |
| empirical-only | ratio-hidden | `0.972842±0.024164` | `0.936969±0.032780` | `0.010000±0.010000` | `0.006667±0.005774` | `9.406133±0.656485` |
| empirical-only | signs-hidden | `0.977448±0.013724` | `0.670878±0.033771` | `0±0` | `0±0` | `6.912298±0.345520` |
| empirical-only | method-only | `0.977448±0.013724` | `0.724945±0.036454` | `0±0` | `0±0` | `7.477649±0.335845` |
| uniform signed | oracle | `0.921943±0.020668` | `0.916526±0.018586` | `0.433333±0.035119` | `0.791667±0.023094` | `101.933598±2.938826` |
| uniform signed | ratio-hidden | `0.921943±0.020668` | `0.916526±0.018586` | `0.433333±0.035119` | `0.791667±0.023094` | `101.933598±2.938826` |
| uniform signed | signs-hidden | `0.978033±0.014003` | `0.600805±0.015668` | `0±0` | `0±0` | `6.113934±0.078551` |
| uniform signed | method-only | `0.978824±0.012634` | `0.650588±0.031263` | `0±0` | `0±0` | `7.067853±0.656427` |
| uniform unsigned | oracle | `0.907101±0.017703` | `0.903459±0.015739` | `0.430000±0.040927` | `0.800000±0.018028` | `101.143199±4.638276` |
| uniform unsigned | ratio-hidden | `0.907101±0.017703` | `0.903459±0.015739` | `0.430000±0.040927` | `0.800000±0.018028` | `101.143199±4.638276` |
| uniform unsigned | signs-hidden | `0.907101±0.017703` | `0.903459±0.015739` | `0.430000±0.040927` | `0.800000±0.018028` | `101.143199±4.638276` |
| uniform unsigned | method-only | `0.907101±0.017703` | `0.903459±0.015739` | `0.430000±0.040927` | `0.800000±0.018028` | `101.143199±4.638276` |

Per-update matrix 在每个更新上重采样 signs。102/102 正式运行均为 `ok`、100/100 且日志唯一。下表完整列出 `min` 主攻击的 18 个三种子条件；所有 hidden-state 行的 L1/L2 均为 0。

| Variant | Knowledge | Hypotheses | Candidate | Top-B | R1+R2 (`min`) |
| --- | --- | ---: | ---: | ---: | ---: |
| adaptive | oracle | 1 | `0.920003±0.021971` | `0.915241±0.021171` | `101.989353±5.036116` |
| uniform | oracle | 1 | `0.921123±0.019568` | `0.915648±0.017309` | `101.759123±4.259130` |
| adaptive | signs-hidden | 1/4/16/64 | `0.966330±0.023425` / `0.977562±0.012712` / `0.976194±0.010758` / `0.976348±0.012154` | `0.346305±0.053066` / `0.458853±0.070114` / `0.552412±0.073616` / `0.579818±0.102584` | `6.308804±0.054059` / `7.495886±0.538045` / `7.183832±0.407909` / `6.967052±0.256386` |
| adaptive | method-only | 1/4/16/64 | `0.976348±0.012154` / `0.976348±0.012154` / `0.975556±0.011977` / `0.976348±0.012154` | `0.426183±0.072045` / `0.556859±0.100756` / `0.594753±0.059910` / `0.659367±0.033050` | `6.629681±0.332293` / `7.828483±0.241387` / `7.427559±0.403119` / `7.375314±0.421741` |
| uniform | signs-hidden | 1/4/16/64 | `0.969401±0.020414` / `0.977424±0.012948` / `0.976005±0.011395` / `0.975112±0.015924` | `0.302125±0.049794` / `0.473686±0.054549` / `0.576041±0.079900` / `0.575660±0.054237` | `5.707280±0.096611` / `6.806477±0.528819` / `6.640623±0.404104` / `6.610561±0.170276` |
| uniform | method-only | 1/4/16/64 | `0.969401±0.020414` / `0.977424±0.012948` / `0.976005±0.011395` / `0.975112±0.015924` | `0.302125±0.049794` / `0.473686±0.054549` / `0.576041±0.079900` / `0.575660±0.054237` | `5.707280±0.096611` / `6.806477±0.528819` / `6.640623±0.404104` / `6.610561±0.170276` |

`mean` 仅作为有限 seed-hypothesis 的 EOT-style sensitivity 汇总，不与 `min` 合并。下表完整覆盖 16 个 hidden-state 条件；每个单元按 hypotheses `1/4/16/64` 排列，所有 L1/L2 仍为 0。

| Variant | Knowledge | Candidate (`mean`) | Top-B (`mean`) | R1+R2 (`mean`) |
| --- | --- | ---: | ---: | ---: |
| adaptive | signs-hidden | `0.966330±0.023425` / `0.976348±0.012154` / `0.976637±0.012538` / `0.976348±0.012154` | `0.346305±0.053066` / `0.581246±0.035605` / `0.695711±0.047048` / `0.719293±0.010319` | `6.308804±0.054059` / `7.592659±0.262484` / `8.624421±0.251092` / `8.946655±0.271156` |
| adaptive | method-only | `0.976348±0.012154` / `0.976348±0.012154` / `0.976348±0.012154` / `0.976348±0.012154` | `0.397079±0.048719` / `0.624052±0.040537` / `0.679285±0.052120` / `0.711599±0.050606` | `6.250211±0.103797` / `7.775644±0.309779` / `8.622847±0.355025` / `8.993867±0.358746` |
| uniform | signs-hidden | `0.969401±0.020414` / `0.977562±0.012712` / `0.977851±0.013039` / `0.976602±0.015164` | `0.302125±0.049794` / `0.574004±0.054786` / `0.658866±0.058814` / `0.677730±0.069160` | `5.707280±0.096611` / `7.749452±0.152894` / `8.716969±0.373046` / `8.936966±0.262887` |
| uniform | method-only | `0.969401±0.020414` / `0.977562±0.012712` / `0.977851±0.013039` / `0.976602±0.015164` | `0.302125±0.049794` / `0.574004±0.054786` / `0.658866±0.058814` / `0.677730±0.069160` | `5.707280±0.096611` / `7.749452±0.152894` / `8.716969±0.373046` / `8.936966±0.262887` |

增加有限 seed hypotheses 并未接近 oracle；但该压力测试只说明当前 hypothesis family 与预算不足以消除状态不确定性，不构成一般 EOT robustness，也不能排除直接状态估计、连续隐变量优化或跨更新学习。

## Appendix A.6 PEFTLeak-Style Image-Side Cross-Modal Study

该 official-aligned/source-aligned path 使用 CIFAR-100、batch size 32、upstream-style malicious ViT with Adapters、seed 42 和 fixed victim batch。防御作用于完整 96 个 Adapter gradient tensors，攻击随后读取 20 对 weight/bias gradients。全部 9 个 canonical `seed42/logs` 均完成且最终指标状态为 `ok`。这是 single-seed、privacy-only、non-FedAvg、non-multi-client study。

| Defense | Patch recovery | Full-image MSE | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: |
| none | `0.109375` | `0.034469` | `0.753896` | `0.245784` |
| Projection-LRB (adaptive ratios, k=0.5) | `0.000000` | `0.283339` | `0.021863` | `0.687894` |
| Projection-LRB (adaptive ratios, k=0.75) | `0.000000` | `0.283288` | `0.020130` | `0.694709` |
| Projection-LRB (adaptive ratios, k=0.9) | `0.000000` | `0.260880` | `0.029032` | `0.688288` |
| top-k@0.1 | `0.000000` | `0.106257` | `0.125249` | `0.702562` |
| top-k@0.3 | `0.000000` | `0.115692` | `0.118767` | `0.645214` |
| compression@8 | `0.000000` | `0.071519` | `0.141476` | `0.816451` |
| compression@16 | `0.000000` | `0.071519` | `0.141476` | `0.816451` |
| Full-LRB@0.5 | `0.000000` | `0.300366` | `0.008401` | `0.697790` |

Full-image MSE、SSIM 和 LPIPS 在 clustered full images 上计算，采用 Hungarian one-to-one matching 并纳入全部重建样本。`strict_normalized_patch_recovery_rate` 在包括 none 在内的九个配置中均为 0，缺乏区分力，故只作为诊断状态而不进入表格。攻击 batch loss 不是图像任务效用，本文也不声称该表建立了图像侧 privacy--utility trade-off。由于 top-k、compression 与 Full-LRB 同样达到零 patch recovery，本表只支持单种子跨模态转移证据，不支持强基线优势或普遍跨模态泛化。

# 非正文写作附注

## 引用键对照

| Citation key | 论文 | 来源/年份 |
| --- | --- | --- |
| `mcmahan2017fedavg` | McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data* | AISTATS 2017 |
| `hu2022lora` | Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* | ICLR 2022 |
| `zhu2019deep` | Zhu, Liu, and Han, *Deep Leakage from Gradients* | NeurIPS 2019 |
| `balunovic2022lamp` | Balunovic et al., *LAMP: Extracting Text from Gradients with Language Model Priors* | NeurIPS 2022; DOI: `10.52202/068431-0555` |
| `petrov2024dager` | Petrov et al., *DAGER: Exact Gradient Inversion for Large Language Models* | NeurIPS 2024; arXiv: `2405.15586` |
| `sami2025peftleak` | Sami et al., *Gradient Inversion Attacks on Parameter-Efficient Fine-Tuning* | CVPR 2025; DOI: `10.1109/CVPR52734.2025.00956` |
| `xie2025recit` | Xie et al., *ReCIT: Reconstructing Full Private Data from Gradient in Parameter-Efficient Fine-Tuning of Large Language Models* | arXiv: `2504.20570`, 2025 |
| `li2024partial` | Li, Xu, and Dras, *Seeing the Forest through the Trees: Data Leakage from Partial Transformer Gradients* | EMNLP 2024; DOI: `10.18653/v1/2024.emnlp-main.275` |
| `abadi2016dpsgd` | Abadi et al., *Deep Learning with Differential Privacy* | ACM CCS 2016; DOI: `10.1145/2976749.2978318` |
| `sun2021soteria` | Sun et al., *Soteria: Provable Defense against Privacy Leakage in Federated Learning from Representation Perspective* | CVPR 2021; DOI: `10.1109/CVPR46437.2021.00919` |
| `aji2017sparse` | Aji and Heafield, *Sparse Communication for Distributed Gradient Descent* | EMNLP 2017; DOI: `10.18653/v1/D17-1045` |
| `alistarh2017qsgd` | Alistarh et al., *QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding* | NeurIPS 2017 |
| `verma2019manifold` | Verma et al., *Manifold Mixup: Better Representations by Interpolating Hidden States* | ICML 2019, PMLR 97 |
| `radford2019language` | Radford et al., *Language Models are Unsupervised Multitask Learners* | OpenAI Technical Report, 2019 |
| `wang2019glue` | Wang et al., *GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding* | ICLR 2019 |
| `warstadt2019neural` | Warstadt, Singh, and Bowman, *Neural Network Acceptability Judgments* | TACL 7, 2019, pp. 625-641; DOI: `10.1162/tacl_a_00290` |
| `socher2013recursive` | Socher et al., *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank* | EMNLP 2013 |
| `pang2005seeing` | Pang and Lee, *Seeing Stars: Exploiting Class Relationships for Sentiment Categorization with Respect to Rating Scales* | ACL 2005 |

## 证据与待确认项

- **Standard DAGER 主结果：** 数值来自 `AAAI2027_EVIDENCE_TABLES.md` 第 2.1、2.2 节。设置为 GPT-2 sequence classification、batch size 2、每个配置和种子攻击 100 个单客户端更新实例，即共 200 条文本；seeds 为 `101/202/303`。`rec_token=0` 只表示当前攻击未恢复 token。SST-2 adaptive-ratio k=0.9 缺少配对 utility；三任务 noise@5e-4 与 DP-SGD-style@5e-4 coverage 也未形成完整日志，因此不进入 Main Table 1。
- **机制消融：** “projection 是主要组件、clipping 单独无效”来自证据表第 3 节。该消融使用 `./models/gpt2-ft-rt`，与 2026-07 三数据集主表不是同一次 sweep。Appendix A2 的 fine-ablation 来自 2026-05-10 批次；keep-ratio 表因 k=0.5/@0.75 utility 不完整而只报告 privacy。
- **Legacy Adapter ratio 机制实验：** 第 8.1 节使用 `Mechanistic Evaluation with a PEFTLeak-Style Adapter Ratio Probe`。旧协议的 privacy 来自 `log/peftleak_text_sst2/privacy/` 中无 seed 后缀的单次 100 条文本正式日志，utility 来自 `log/peftleak_text_sst2/utility_/` 的三个独立训练种子，故两类指标是 cross-protocol comparison。该协议使用 known label、sample-adaptive malicious probe 和 legacy public-bin statistics，只覆盖 Adapter 梯度比值机制。
- **Fixed-probe v2 严格性检查：** 附录数据来自 `log/peftleak_text_sst2_v2/20260714_024606/formal/` 的 24 个正式运行；协议使用覆盖 128 个 token positions、包含 256 个 weight/bias gradient tensors 的固定 probe inventory，在私有数据到达前安装，并采用严格 disjoint public partition。证据表第 6 节已经与 Appendix A1 同步。
- **PTG 有限迁移：** 来自证据表第 5 节。协议为 SST-2、GPT-2、batch size 1、first two Transformer blocks 的 24 个 gradient tensors（8 个 matrix gradients）、known label/padding、single restart、80 个 Adam attack steps、学习率 0.1、cosine matching、embedding-norm weight 0.01，每个种子攻击 100 条文本。Main Table 4 的七个预先选定点均已完成三种子 privacy 和配置对齐 utility；汇总排除了旧的未完成 adaptive-ratio k=0.5/seed101 与重复的 k=0.2/seed303 副本。
- **静态知识分层矩阵：** Appendix A.5 来自 `log/同预算 white-box baselines/new/adaptive_lrb_matrix_sst2_official_validation_20260718_114719/`。该批次 60/60 正式运行均为 `ok`、100/100 且日志唯一，覆盖 5 个 projection variants、4 种 attacker-knowledge settings 与 seeds `101/202/303`；三种子统计从 `results.csv` 逐 seed 行计算 sample SD。
- **Per-update/EOT-style 状态：** Appendix A.5 的 per-update stress test 来自 `adaptive_lrb_matrix_sst2_official_validation_20260718_114758/`。102/102 正式运行均为 `ok`、100/100 且日志唯一；`min` 主攻击与 `mean` sensitivity 分表报告。Finite seed hypotheses 不能排除直接状态估计、隐变量优化或跨更新推断。
- **Uniform oracle：** Appendix A.4 使用 `uniform_oracle_resolution_sst2_official_validation/formal/`、`uniform_oracle_resolution_summary.csv` 与 `uniform_oracle_resolution_paired_bootstrap.csv`。六个三种子条件和五个 paired comparisons 已完成审计，预设结论为 `span_mismatch_only`。
- **图像侧补充：** Appendix A.6 来自 `log/peftleak_official_image/baselines/seed42/logs/` 的九个 canonical logs。该协议是 CIFAR-100、ViT-Adapter、batch 32、single seed、fixed victim batch、privacy-only、non-FedAvg；不能表述为一般 PEFT 或普遍跨模态防御。
- **强基线定位：** Standard DAGER Main Table 1 的高保留率 top-k 与高 bit compression 为非零恢复，独立 full sweep 中更强压缩点包含零恢复端点。PTG 七点表中 noise 与 DP-SGD-style 的恢复率低于 Projection-LRB，完整 privacy sweep 中 compression@2 进一步达到更低恢复；因此即使配对 utility 已完成，也不作 Projection-LRB 普遍优于这些基线的排序。
- **攻击数据来源：** Standard DAGER 主表中的 `split=val` 来自原始 training split，而非官方 validation 或未参与模型训练的 held-out set。`TextDataset` 先随机排列 training split，以前 1000 条构造内部 `test`，再从余下样本按长度分层抽取内部 `val`；知识分层实验则使用 SST-2 official validation。最终英文稿必须分别说明，不能沿用同一个 `val` 名称。
- **实现边界：** `signed_pool` 是降采样后插值恢复算子，不具备严格的正交算子性质。主实验与静态矩阵按同一设置复用 signs；per-update 扩展按更新重采样。当 \(k>0.75\) 时 sensitivity-to-keep-ratio 的方向反转。输出仍为同形状 dense tensors，当前实现不降低通信字节数。
- **运行环境待确认：** 114719/114758 manifests 已记录知识分层批次的 GPU、代码 commit、checkpoint digest 和主要软件版本，但其他历史正式日志仍未完整记录这些字段。仓库中的 Conda 配置不能替代逐运行 provenance，投稿归档前仍需补齐主表、Adapter 与 PTG 证据链。
- **引用状态：** 上表用于锁定中文草稿中的 citation keys，不替代正式 BibTeX。GPT-2 与新增数据集条目的题名、作者和发表信息已按原论文或官方来源核对；本轮访问 GLUE 的 OpenReview 页面时遇到浏览器验证限制，`wang2019glue` 暂按通行的 ICLR 2019 元数据锁定。转写英文稿时仍需从官方页面导入条目，并复核 GLUE 条目以及 ReCIT 的最终发表状态、会议页码与完整作者信息。
- **本轮文件边界：** 本轮只修改中文论文草稿与证据索引；未修改实验代码、脚本或原始日志，也未补跑实验。
