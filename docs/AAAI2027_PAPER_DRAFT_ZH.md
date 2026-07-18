# Projection-LRB: A Unified Update-Space Framework for Mitigating Text Reconstruction across Full, Adapter, and Partial Gradients

> 定位：AAAI 2027 投稿中文全文底稿，后续各章节统一在本文件续写，并用于改写英文正文。  
> 方法名称：Projection-LRB。  
> 说明：文末“非正文写作附注”用于锁定引用键和证据边界，不属于论文正文。

## Abstract

Federated fine-tuning keeps raw text local, yet shared gradients and parameter updates can reveal client inputs. Existing defenses are commonly evaluated on a single leakage surface and provide limited evidence about which update structures make text recoverable. We introduce Projection-LRB, an update-space bottleneck that applies layer-wise low-resolution signed reconstruction without changing the shared tensor interface. This common transform is evaluated across full-gradient DAGER, a PEFTLeak-style Adapter ratio probe, and partial-gradient matching. On three GPT-2 text-classification tasks, Projection-LRB drives token and ROUGE recovery to zero under standard DAGER in three-seed evaluations while retaining clean-level utility at the main operating point; it also reduces recovery within the Adapter and partial-gradient protocols, although stronger baselines remain in individual settings. Component ablations identify low-resolution reconstruction, rather than clipping or residual noise, as the dominant factor under standard DAGER. As an additional robustness extension, a knowledge-stratified evaluation on the official SST-2 validation split shows that recovery depends strongly on the realized transformation state: exact oracle knowledge restores substantial sequence recovery (R1+R2=101.87), whereas hiding the realized ratios or signs leaves candidate recall above 0.97 but reduces R1+R2 below 9.5 under the evaluated hypothesis budget. These results support a shared empirical recoverability bottleneck across heterogeneous update exposures and locate its knowledge-dependent boundary in candidate ranking and sequence assembly; they do not establish irreversible information removal, formal privacy, or universal robustness to adaptive attacks.

## 1. Introduction

在医疗、金融和跨机构协作等隐私敏感场景中，参与方往往希望利用分散保存的临床记录、客户对话或内部文档共同微调大语言模型，同时避免将原始数据集中上传。联邦学习通过交换本地计算的梯度或模型更新实现协同训练，参数高效微调（parameter-efficient fine-tuning, PEFT）则进一步将通信与训练限制在 LoRA、Adapter 等少量可训练参数上 \cite{mcmahan2017fedavg,hu2022lora}。然而，数据保留在本地并不意味着其语义内容也停留在本地：在单客户端更新可见或聚合不足以隐藏个体贡献时，服务器虽然无法直接读取客户端文本，却能够观察由这些文本产生的完整梯度、局部梯度或 adapter updates。因而，联邦微调所提供的是数据位置上的隔离，而非共享更新天然不可逆的隐私保证。

近期研究不断扩大这种更新泄露的已知边界。早期优化式梯度反演已表明，输入可以通过匹配观测梯度进行近似重建 \cite{zhu2019deep,balunovic2022lamp}；DAGER 进一步利用自注意力梯度的低秩结构和 token embedding 的离散性，在 honest-but-curious 的完整梯度设置下恢复大批量文本 \cite{petrov2024dager}。与此同时，PEFTLeak 揭示了精简 adapter gradients 仍可携带可解析的输入结构，ReCIT 则在包含恶意预训练和记忆增强的更强攻击模型中恢复上下文与个人可识别信息 \cite{sami2025peftleak,xie2025recit}。部分 Transformer 梯度攻击还表明，攻击者无需获得全部参数梯度，也可能从特定层或模块中重建文本 \cite{li2024partial}。这些工作采用的攻击知识、可见更新和主动能力并不相同，但共同提出了一个尚未充分回答的问题：不同更新暴露面是否包含可被统一抑制的可恢复结构？

现有防护机制主要沿三条路径展开。第一类以差分隐私和噪声注入为代表，通过裁剪与随机扰动限制单个样本对共享更新的影响 \cite{abadi2016dpsgd}；第二类采用 top-k sparsification 或 quantization 压缩更新，其主要目标是降低分布式训练的通信开销 \cite{aji2017sparse,alistarh2017qsgd}；第三类在输入或中间表示上进行掩码和混合，以降低样本特征的直接暴露 \cite{sun2021soteria,verma2019manifold}。这些机制提供了不同的 privacy-utility trade-off，其中 top-k 和 quantization 在本文评估中也是强经验基线。然而，它们通常不以文本可恢复性为直接设计对象，也没有回答哪些层、方向和分辨率承载了攻击所依赖的样本级结构。这一缺口并不意味着压缩方法无效，而是说明通信压缩率与文本可恢复性之间缺少可解释的逐层联系。因此，仅比较扰动幅度或压缩率，难以解释防御何时抑制恢复、何时又会被更强攻击绕过。

本文据此提出一个结构性假设：共享更新同时包含支撑下游优化的粗粒度任务信号，以及支持 token 识别与序列重建的高分辨率、样本特定结构；防御的关键不是一味缩小梯度范数，而是在尽量保留前者的同时限制后者的可恢复分辨率。我们的核心研究问题是，能否在客户端上传前只插入一次通用更新变换，便同时缓解依赖完整梯度、Adapter 梯度关系和部分梯度的三类文本重建，而无需针对每个攻击重写防御。基于这一视角，我们提出 Projection-LRB，一种逐层 recoverability bottleneck。该方法结合结构先验与更新统计估计层级敏感度，为不同层分配 keep ratio，并通过低分辨率 signed-pool reconstruction 对更新进行降采样和插值恢复。这里的 keep ratio 控制重建分辨率，不等同于按梯度幅值选择坐标的 top-k 稀疏率。Projection-LRB 不改变共享张量的接口；完整梯度、Adapter 参数子集或部分层梯度的选择均发生在该统一变换之后。由此，统一性来自防御插入点和算子的复用，而不依赖某一种攻击的内部解码规则。

我们在三类互补的更新暴露面上使用同一防御接口。首先，在 SST-2、CoLA 与 Rotten Tomatoes 的 full-gradient DAGER 评估中，Projection-LRB@0.5 在三个随机种子上均将当前 token 与 ROUGE 恢复指标降至 0，并保持接近 clean 的下游效用；组件消融进一步将低分辨率 reconstruction 识别为主导因素，而 clipping 单独不能阻断恢复。其次，在受控的 PEFTLeak-style Adapter 梯度比值探针中，Projection-LRB@0.5 将单次 100 条文本运行的 token recovery 从 `0.760` 降至 `0.202`，但更严格的 fixed-probe v2 得到 `0.685±0.016`，说明旧协议高估了抑制幅度。最后，在 PTG `first2` 三种子协议中，Projection-LRB@0.2 相对 paired clean 将 token recovery 平均降低 24.3%，但 compression@2、noise 与 DP-SGD-style 在该协议中达到更低恢复率。上述结果共同支持同一更新空间变换能够跨完整梯度、Adapter 梯度关系和部分梯度产生可测的抑制作用，而不支持 Projection-LRB 在每个攻击面上都优于专门基线。

作为对这一统一框架的稳健性扩展，我们进一步将攻击者对 Projection-LRB 的知识拆分为 oracle、ratio-hidden、signs-hidden 和 method-only 四种设置。在官方 SST-2 validation 上，oracle 知道每次更新实际使用的 ratios 与 signs 时，Projection-LRB@0.5 的 R1+R2 恢复至 `101.872623`；隐藏当前更新实际使用的 ratios 或 signs 后，候选集合中的真实 token recall 仍高于 `0.97`，但 R1+R2 分别降至 `9.421112` 和 `6.453098`。这一指标分裂表明，当前防御主要干扰候选排序、过滤和序列组装，而不是不可逆地删除 token 信息。与此同时，各 sensitivity 变体在 oracle 下的 R1+R2 均约为 101--103，不能证明 hybrid layer-wise allocation 优于 uniform reconstruction。本文因此将 Projection-LRB 定位为跨攻击面的机制驱动经验缓解框架，并将知识分层用于刻画其边界，而不宣称形式化隐私、普遍的白盒稳健性或对压缩基线的统一优势。

本文的主要贡献如下：

1. **统一的更新空间视角。** 我们将完整梯度 DAGER、PEFTLeak-style Adapter 比值恢复与 PTG 部分梯度匹配置于同一“防御先作用于客户端更新、攻击再观察暴露子空间”的接口下，并在不混排跨协议绝对指标的前提下联合考察可恢复性、任务效用与计算成本。
2. **Projection-LRB。** 我们提出一种保持共享张量形状的低分辨率 signed reconstruction bottleneck。该方法将 recoverability-critical update structure 作为直接干预对象，并可在 full-model、Adapter 与 partial-gradient 暴露路径中复用同一变换。
3. **跨暴露证据与知识边界。** 三任务 DAGER、Adapter ratio probe 和 PTG `first2` 实验表明同一变换在三类协议内均可降低恢复；组件消融识别出 reconstruction 的主导作用。进一步的知识分层实验表明，恢复依赖攻击者能否获得 realized ratios 与 signs，并揭示攻击失败发生在候选排序和序列组装阶段，而非 token 信息的不可逆删除。

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

梯度反演依赖的不只是更新幅值，还包括可将单个 token 或局部序列与参数方向对应起来的细粒度结构。Projection-LRB 因而不以求解一个形式化隐私目标为出发点，而是在客户端上传前，对每个有效梯度张量施加逐层的低分辨率更新变换。给定按模型参数顺序排列的梯度元组 \(\{G_l\}\)，方法先以参数名导出的结构先验和当前更新的统计量估计敏感度 \(s_l\)，再分配 keep ratio \(r_l\)，最后应用 signed-pool reconstruction operator \(T_{r_l}\)。该算子对梯度先做确定性符号调制、再降采样并插值恢复，输出与输入保持相同形状和数据类型，从而可以直接替换原上传梯度。整个过程是一次前向式变换：敏感度不参与训练，不维护跨批次状态，也不存在可学习打分器、额外优化求解或客户端与服务器之间的新交互。

这里的“Projection”是方法命名，不表示算子具有正交性、幂等性或信息论不可逆性；其作用是构造一个可控的低分辨率经验瓶颈。该瓶颈的设计假设是，降低更新的局部分辨率可能削弱当前反演器使用的样本级结构，同时仍保留下游优化可利用的较粗更新趋势。本文只通过攻击与效用实验检验这一假设，不由算子形式推出安全结论。

### 4.2 Hybrid Layer Sensitivity

对每个参数名，代码按固定分支顺序赋予结构先验 \(p_l\)：embedding 参数为 \(1.0\)；前两个 Transformer blocks 中的 attention 参数为 \(1.0\)；前两个 blocks 的其余参数为 \(0.7\)；其他 attention 参数为 \(0.45\)；尚未命中的 classifier/lm-head、bias 或 LayerNorm 参数为 \(0.15\)；其余参数为 \(0.25\)。分支优先级属于定义的一部分，因此前两个 blocks 内的 bias 和 LayerNorm 先命中 early-layer 规则，其先验为 \(0.7\)，而非 \(0.15\)。结构分数完全来自名称规则和解析出的 block index，不读取激活、样本 token 或标签；这一规则也意味着，参数命名无法匹配已知模式时会落入“其他参数”分支，而不会由实现自动学习新的类别。

经验敏感度由三个当前梯度统计量构成。首先，范数项衡量层更新规模；其次，校准残差衡量低分辨率重建对该层抽样向量的改变；最后，尖峰项衡量最大坐标相对于均方根幅值的突出程度：

\[
m_l^{\mathrm{norm}}=\log(1+\lVert G_l\rVert_2),\qquad
m_l^{\mathrm{res}}=\frac{\lVert S_l-T_{\rho_{\mathrm{cal}}}(S_l)\rVert_2}{\lVert S_l\rVert_2+\epsilon},
\]

\[
m_l^{\mathrm{spike}}=\log\!\left(1+\frac{\max |G_l|}{\operatorname{RMS}(G_l)+\epsilon}\right).
\]

其中 \(S_l\) 由展平梯度进行确定性 stride subsampling 得到，最多保留 4096 个元素；当梯度不超过该上限时保留全部元素，否则步长取“元素数除以上限”的向上取整，起点由该层 seed 对步长取模确定。校准向量已经是一维表示，因此残差项始终调用一维重建路径，而范数项和尖峰项仍在完整 \(G_l\) 上计算。三个统计量分别在所有非空梯度张量之间做 min-max normalization；空梯度不进入统计，输出时继续保持为空。若某一统计量在各层完全相同，代码将该项的归一化结果统一设为 \(0.5\)。这些量针对当前梯度元组即时计算，不使用历史滑动平均或额外校准数据。由此得到

\[
e_l=0.45\widehat m_l^{\mathrm{norm}}+0.40\widehat m_l^{\mathrm{res}}+0.15\widehat m_l^{\mathrm{spike}},
\qquad s_l=0.4p_l+0.6e_l.
\]

实现先收集全部有效层的三个原始统计量，再分别归一化，最后才进入逐层重建。因此，经验分数是当前客户端更新内部的相对排序，而不是某一参数张量固定不变的属性：即使一层自身梯度未变，其他层的取值范围变化也可能改变其归一化分数。结构先验为这种批次依赖的相对标度提供固定参照，但代码不对 \(s_l\) 做跨更新平滑。完成融合后，\(s_l\) 被限制到 \([0,1]\)，随后用于同一梯度元组的 keep-ratio 分配。

### 4.3 Adaptive Keep Ratio

Projection-LRB preset 将一般层的基准 keep ratio 固定为 \(0.75\)，将实验参数 \(k\) 作为敏感端点。经验分数和结构分数均位于 \([0,1]\)，融合结果在实现中再次限制到该区间，因此下面的线性插值以 \(0.75\) 和 \(k\) 为两个端点：

\[
r_l=(1-s_l)\cdot0.75+s_l\cdot k,
\qquad \rho_{\mathrm{cal}}=\frac{0.75+k}{2}
\]

分配逐层分辨率和校准分辨率。\(\rho_{\mathrm{cal}}\) 仅用于计算敏感度中的重建残差，并不覆盖最终的逐层 \(r_l\)。主 operating point 为 Projection-LRB@0.5，@0.9 作为补充点。需要注意，只有当 \(k<0.75\) 时，较高 \(s_l\) 才对应更小的 keep ratio 和更强的低分辨率瓶颈；当 \(k>0.75\) 时方向反转，较高敏感度反而得到更大的 keep ratio。因此，@0.9 是按代码定义评估的经验 operating point，不能解释为“敏感层压缩更强”。实现将 keep ratio 限制在 \([10^{-4},1]\) 内；标量梯度或 \(r_l\geq0.999\) 时直接返回原张量副本。

### 4.4 Signed-Pool Reconstruction

对一维张量，\(T_r\) 使用取值为 \(\{-1,+1\}\) 的 Rademacher signs 对坐标逐元素调制，将长度从 \(n\) 自适应平均池化到 \(\max(1,\operatorname{round}(nr))\)，经 linear interpolation 恢复到 \(n\)，再乘相同 signs。对二维张量，代码分别生成可分离的行、列 signs，将两个维度池化到 \(\max(1,\operatorname{round}(mr))\) 与 \(\max(1,\operatorname{round}(nr))\)，再用 bilinear interpolation 恢复。因而二维情形中的 \(r\) 是每个轴的分辨率比例，而不是保留坐标总数的比例；池化网格的元素数约随 \(r^2\) 变化。高于二维的张量先展平，随后按一维路径处理。池化输出尺寸至少为 1，linear 与 bilinear interpolation 均采用 `align_corners=False`。所有重建先转为 float 表示计算，最终恢复原 shape 和 dtype。

每层使用 \(q_l=\texttt{rng\_seed}+1009(l+1)\) 生成 signs；二维张量的列 signs 在生成器中顺接行 signs 的随机数序列。同一 seed、设备和形状对应的 signs 会被缓存并重复使用。除第 10 节的 per-update 扩展外，正文主实验采用 static seed mode：同一设置重复使用相同 signs。该状态是确定的实验参数而非密码学密钥；knowledge-stratified evaluation 明确区分攻击者是否获知每次更新实际使用的 signs。Per-update 扩展则按可观察更新重新采样 signs，用于检验有限 seed hypotheses 下的敏感性，而不改变算子本身。符号调制、池化和插值的组合未被代码验证为正交算子，因此不能把“重建残差”解释成严格的正交补空间。

**Algorithm 1: Projection-LRB**

**Input:** gradients \(\{G_l\}\), parameter names \(\{n_l\}\), sensitive endpoint \(k\), base seed \(q\).  
**Output:** defended gradients \(\{\widetilde G_l\}\).

1. 按 embedding、early attention、early remaining、later attention、head/bias/LayerNorm、other 的分支顺序，由 \(n_l\) 计算 \(p_l\)。
2. 对每个非空 \(G_l\)，以 \(q_l=q+1009(l+1)\) 做最多 4096 元素的确定性 stride subsampling，计算 \(m_l^{\mathrm{norm}}\)、\(m_l^{\mathrm{res}}\) 和 \(m_l^{\mathrm{spike}}\)。
3. 对三个统计量分别做跨层 min-max normalization；常数项置为 \(0.5\)，再计算 \(e_l\) 与 \(s_l=0.4p_l+0.6e_l\)。
4. 令 \(r_l=(1-s_l)0.75+s_lk\)，并限制到 \([10^{-4},1]\)。
5. 若 \(G_l\) 为空则保留空项；若其为标量或 \(r_l\geq0.999\)，令 \(\widetilde G_l=G_l\) 的副本；否则令 \(\widetilde G_l=T_{r_l}(G_l;q_l)\)。
6. 返回保持原梯度顺序、形状与数据类型的 \(\{\widetilde G_l\}\)。

### 4.5 Full-LRB and Cost

主文方法 Projection-LRB 只启用上述敏感度分配和 signed-pool reconstruction，不包含裁剪或噪声。用于消融的 Full-LRB 在此基础上先计算所有有效层梯度范数的中位数 \(M\)。其裁剪尺度由 \(1.0\) 与 \(0.5\) 按 \(s_l\) 插值，即每层阈值为 \(M[(1-s_l)\cdot1.0+s_l\cdot0.5]\)，并在重建之前执行。随后采样高斯噪声 \(Z_l\)，构造 \(Z_l-T_{r_l}(Z_l)\)，将该残差归一到相对于裁剪后梯度范数的预设尺度后加到重建结果；噪声尺度同样按敏感度在 \(0.005\) 与 \(0.03\) 之间插值。该随机项与重建 signs 由相同的逐层 seed 规则控制，但二者通过各自的生成过程产生。

由于 \(T\) 不具备严格的正交算子性质，上述附加项仅称为 residual-space noise，不赋予严格正交性。范数和尖峰统计需要扫描完整梯度，校准抽样至多处理每层 4096 个元素，池化、插值和逐层变换也按输入规模执行，因此总体时间复杂度随梯度元素数线性增长。当前实现仍物化并输出同形状的 dense tensors，没有编码稀疏索引或低分辨率系数，因而不减少上传张量的通信字节数；实验中的时间指标用于记录这一额外本地变换成本，而非推断通信收益。

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

Standard DAGER Main Table 1 报告 Projection-LRB@0.5，以及在 CoLA 和 Rotten Tomatoes 上具有完整配对 utility 的补充点 @0.9；压缩对照统一列出 top-k@0.7/@0.9 与 compression@22/@32。它们用于展示高保留率或高 bit 区间的 privacy--utility 变化，不把 top-k sparsity、quantization bits 与 projection keep ratio 解释成相同的信息预算。更低保留率的 top-k 与更低 bit 的 compression 在独立 full sweep 中包含零恢复端点，但不与 Main Table 1 的高保留率点混称为同一 operating point。PTG Main Table 4 固定使用 none、Projection-LRB@0.2/@0.5、top-k@0.1、compression@8、noise@5e-4 和 DP-SGD-style@5e-4；这些点在 PTG 协议内单独解释，而不是按完整 PTG sweep 事后选择最优参数。Full-LRB、noise、DP-SGD-style、Soteria-style 与 mixup-style 仅用于组件消融或防御覆盖，不据此宣称统一的形式化隐私语义。尤其是 DP-SGD-style 没有 privacy accountant，不能视为正式 DP-SGD。

### 5.4 Metrics and Statistics

隐私指标为 token recovery 与 ROUGE-1/2；knowledge-stratified DAGER 另报告 candidate recall、Top-B recall 及 L1/L2 stage recovery，以定位攻击失败发生在哪个恢复阶段。效用指标为 accuracy、macro-F1 和 loss，CoLA 另报告 MCC；成本指标为训练时间与攻击时间。三种子结果先对每个正常完成的正式运行提取同一口径指标，再计算均值与 population standard deviation（`ddof=0`）。汇总时排除失败、不完整、smoke 和重复日志，不用缺失项替代为零，也不把同一日志的重复副本计作独立种子。所有恢复指标只在攻击假设、可见更新、batch size 和优化预算一致的协议内部比较；DAGER、Adapter probe、fixed-probe v2 与 PTG 的绝对恢复率不做跨协议排序。`rec_token_mean=0` 只表示当前攻击在给定预算内未恢复 token，不能推出可量化的差分隐私结论。

## 6. Full-Gradient DAGER Results

### 6.1 Clean Updates Leak Severe Text Information

在完整梯度暴露下，未防御更新在三个任务上均表现出高可恢复性。Main Table 1 中，SST-2 的 `rec_token_mean` 为 `0.906254±0.023963`，R1+R2 为 `157.608202±5.813635`；Rotten Tomatoes 分别为 `0.885739±0.030813` 和 `164.658710±6.579574`。CoLA 的恢复进一步接近饱和，两个指标达到 `0.999755±0.000347` 和 `199.944272±0.078811`。这些统计来自每个种子 100 个单客户端更新实例；由于 batch size 为 2，每个 seed 实际覆盖 200 条文本。由此，clean leakage 并非 SST-2 单一任务上的偶然现象，而是在当前 GPT-2 sequence classification 设置中跨任务重复出现。

### 6.2 Standard-Attack Privacy and Utility

**Main Table 1. Full-gradient DAGER privacy and utility.** 所有数值均为 seeds `101/202/303` 的 mean ± population std；高保留率压缩基线的 12 个目标 utility 点均已满足三种子准入。`--` 表示该任务不使用 MCC。

| Dataset | Method | `rec_token_mean` | R1+R2 | Accuracy | Macro-F1 | MCC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SST-2 | none | `0.906254±0.023963` | `157.608202±5.813635` | `0.912462±0.007028` | `0.912433±0.007039` | -- |
| SST-2 | Projection-LRB@0.5 | `0.000000±0.000000` | `0.000000±0.000000` | `0.915137±0.003376` | `0.915086±0.003370` | -- |
| SST-2 | top-k@0.7 | `0.164238±0.020696` | `9.212706±0.932282` | `0.907110±0.005848` | `0.907084±0.005849` | -- |
| SST-2 | top-k@0.9 | `0.372119±0.027324` | `28.907796±3.568460` | `0.911315±0.005947` | `0.911291±0.005949` | -- |
| SST-2 | compression@22 | `0.380649±0.023586` | `18.443061±1.581323` | `0.912079±0.003288` | `0.912053±0.003280` | -- |
| SST-2 | compression@32 | `0.906078±0.023557` | `157.292912±5.843706` | `0.913609±0.005324` | `0.913577±0.005323` | -- |
| SST-2 | noise@1e-6 | `0.953844±0.011730` | `17.851501±1.157153` | `0.912080±0.001949` | `0.912058±0.001961` | -- |
| SST-2 | DP-SGD-style@1e-5 | `0.835020±0.015316` | `14.053037±1.165430` | `0.905199±0.004805` | `0.905172±0.004799` | -- |
| CoLA | none | `0.999755±0.000347` | `199.944272±0.078811` | `0.746564±0.004719` | `0.621441±0.016398` | `0.337219±0.016543` |
| CoLA | Projection-LRB@0.5 | `0.000000±0.000000` | `0.000000±0.000000` | `0.758709±0.007996` | `0.643373±0.023499` | `0.377492±0.024702` |
| CoLA | Projection-LRB@0.9 | `0.000000±0.000000` | `0.000000±0.000000` | `0.754874±0.006080` | `0.637656±0.015180` | `0.364403±0.019728` |
| CoLA | top-k@0.7 | `0.050093±0.007203` | `2.400210±1.013465` | `0.753915±0.004311` | `0.634566±0.010400` | `0.361955±0.014674` |
| CoLA | top-k@0.9 | `0.210144±0.010154` | `11.791838±1.006418` | `0.728987±0.028066` | `0.551285±0.099181` | `0.227149±0.166287` |
| CoLA | compression@22 | `0.514786±0.013390` | `23.367069±0.754702` | `0.744967±0.004358` | `0.621065±0.011670` | `0.332858±0.015033` |
| CoLA | compression@32 | `0.999755±0.000347` | `199.944272±0.078811` | `0.723234±0.022815` | `0.538217±0.091685` | `0.207395±0.147065` |
| CoLA | noise@1e-6 | `1.000000±0.000000` | `56.555866±3.479900` | `0.723874±0.023051` | `0.537808±0.091300` | `0.209653±0.148247` |
| CoLA | DP-SGD-style@1e-5 | `1.000000±0.000000` | `11.921810±4.256214` | `0.725471±0.002964` | `0.557900±0.009146` | `0.258227±0.011994` |
| Rotten Tomatoes | none | `0.885739±0.030813` | `164.658710±6.579574` | `0.862414±0.003620` | `0.862345±0.003661` | -- |
| Rotten Tomatoes | Projection-LRB@0.5 | `0.000000±0.000000` | `0.000000±0.000000` | `0.860225±0.002026` | `0.860136±0.002065` | -- |
| Rotten Tomatoes | Projection-LRB@0.9 | `0.000000±0.000000` | `0.000000±0.000000` | `0.860538±0.007972` | `0.860474±0.007937` | -- |
| Rotten Tomatoes | top-k@0.7 | `0.021667±0.012472` | `0.193630±0.140119` | `0.863665±0.007117` | `0.863622±0.007126` | -- |
| Rotten Tomatoes | top-k@0.9 | `0.063647±0.008985` | `1.693354±0.325725` | `0.862102±0.004659` | `0.862057±0.004643` | -- |
| Rotten Tomatoes | compression@22 | `0.163051±0.020162` | `2.747972±0.930849` | `0.870231±0.005434` | `0.870205±0.005430` | -- |
| Rotten Tomatoes | compression@32 | `0.883667±0.030741` | `164.304292±6.885229` | `0.860850±0.007672` | `0.860805±0.007696` | -- |
| Rotten Tomatoes | noise@1e-6 | `0.998180±0.000729` | `17.674292±0.221757` | `0.862727±0.004218` | `0.862681±0.004186` | -- |
| Rotten Tomatoes | DP-SGD-style@1e-5 | `0.995462±0.000816` | `8.744707±1.135160` | `0.845841±0.005100` | `0.845839±0.005102` | -- |

Projection-LRB@0.5 在三个任务和三个种子上均将 standard DAGER 的两个恢复指标降为 0，同时效用均值保持在 clean 附近。SST-2 上其 accuracy 与 clean 分别为 `0.915137` 和 `0.912462`；CoLA 上 accuracy、macro-F1 与 MCC 的均值均高于 clean，但当前三个种子不足以支持统计显著提升，只能说明未观察到效用损失。Rotten Tomatoes 上 accuracy 的均值下降 `0.002189`，约为 0.22 percentage points。Projection-LRB@0.9 的 privacy 在三个任务上也均为零恢复，但其严格配对 utility 当前只覆盖 CoLA 与 Rotten Tomatoes，因此 SST-2 @0.9 不进入 combined table。

高保留率 top-k 与高 bit compression 呈现出非零、且随任务和参数明显变化的 standard-DAGER 恢复区间。Top-k@0.7 的 `rec_token_mean` 在 SST-2、CoLA 和 Rotten Tomatoes 上分别为 `0.164238`、`0.050093` 和 `0.021667`，提高到 @0.9 后分别为 `0.372119`、`0.210144` 和 `0.063647`。Compression@22 仍能降低恢复，而 @32 在三个任务上均接近 clean，符合高精度量化趋近原更新的预期。12/12 个目标点现均有配对的三种子 utility；SST-2 compression@32 与 Rotten Tomatoes compression@22 的 accuracy 分别为 `0.913609±0.005324` 和 `0.870231±0.005434`。SST-2 和 Rotten Tomatoes 的目标点整体接近各自 clean 均值；CoLA top-k@0.9 与 compression@32 的 accuracy 分别为 `0.728987±0.028066` 和 `0.723234±0.022815`，且 macro-F1/MCC 的种子间波动较大。更低保留率或更低 bit 的完整 sweep 仍包含零恢复端点，也不能据此声称 Projection-LRB 普遍优于压缩基线。

DP-SGD-style@1e-5 在 CoLA 与 Rotten Tomatoes 上的 token recovery 仍接近 clean，在 SST-2 上也仅降至 `0.835020`，同时三个任务的效用均低于各自 clean 均值。该实现只包含逐样本裁剪与高斯噪声，没有 privacy accountant，因而不提供可报告的 \(\epsilon,\delta\)-DP 保证。运行日志还显示，当前 Projection-LRB 实现的训练时间显著高于 none 和已有压缩基线；由于正式日志未完整记录跨机器硬件环境，本文不做跨批次绝对时间排名。输出张量仍为 dense 且形状不变，因此该方法当前也没有通信字节收益。上述 standard-attack 结果只说明当前解码器未能利用变换后的更新；第 10 节通过知识分层评估进一步区分“候选 token 仍然存在”与“攻击能否完成排序和序列组装”。

## 7. What Disrupts Standard Recovery?

### 7.1 Component Ablation

**Main Table 2. Projection-LRB component ablation.** 数据来自独立的 SST-2、batch 2、`./models/gpt2-ft-rt` 三种子批次，与 Main Table 1 的 2026-07 checkpoints 不同；所有标准差均从 `raw_results.csv` 的逐 seed 行重算。

| Variant | `rec_token_mean` | R1+R2 | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| none | `0.855333±0.025526` | `148.260102±6.348664` | `0.913226±0.005324` | `0.913184±0.005308` |
| identity | `0.855333±0.025526` | `148.260102±6.348664` | `0.913226±0.005324` | `0.913184±0.005308` |
| clip-only | `0.854076±0.027591` | `147.955623±6.530806` | `0.918196±0.002356` | `0.918149±0.002365` |
| projection-only | `0.000000±0.000000` | `0.000000±0.000000` | `0.915520±0.003010` | `0.915499±0.003018` |
| projection+clipping | `0.000000±0.000000` | `0.000000±0.000000` | `0.913226±0.004222` | `0.913197±0.004208` |
| Full-LRB | `0.000000±0.000000` | `0.000000±0.000000` | `0.892584±0.010973` | `0.892472±0.011061` |

Identity 与 none 的 privacy 和 utility 完全一致，排除了仅由防御路由器或调用路径造成结果变化的解释。Clip-only 的 `rec_token_mean=0.854076` 与 none 的 `0.855333` 接近，表明中位范数裁剪本身没有破坏 standard DAGER 所利用的结构。相反，projection-only 已将两个恢复指标降为 0，并保持与 clean 接近的 accuracy；加入 clipping 后仍为同一隐私终点，但未带来可辨识的额外收益。Full-LRB 同样为零恢复，却将 accuracy 均值降至 `0.892584`，比 none 低约 2.06 percentage points。该消融将低分辨率重建识别为当前 standard-attack 结果的主要有效组件，也支持将 Full-LRB 视为 over-defense 参照而非主方法。

### 7.2 Sensitivity and Keep-Ratio Checks

Appendix Table A2 进一步比较 rule-only、empirical-only、uniform 与 no-empirical variants。在同一个 k=0.5 批次中，这些变体均达到 standard DAGER 零恢复，accuracy 均值位于 `0.914373` 至 `0.917049`。因此，现有实验不能证明 hybrid layer sensitivity 是达到该隐私端点的必要条件；更稳妥的解释是，signed low-resolution reconstruction 本身在这一攻击协议下已经占主导。第 10.2 节的 oracle 知识矩阵给出一致证据：adaptive、rule-only、empirical-only 与 uniform variants 的最终恢复均约为 101--103，仍不能区分 layer-wise allocation 的稳健性优势。独立 keep-ratio privacy sweep 中，k=0.5、0.65、0.75 与 0.9 也均为三种子零恢复，但 k=0.5 和 k=0.75 的 utility 各缺一个正常 summary，故附录不混入不完整效用。尤其当 k>0.75 时，敏感度越高对应的 keep ratio 反而越大，@0.9 只能作为方向反转的经验 operating point，不能支持“敏感层始终压缩更强”或更强攻击同样失败的结论。

## 8. Adapter-Gradient Leakage: PEFTLeak-Style Ratio Recovery

### 8.1 Mechanistic Evaluation with a PEFTLeak-Style Adapter Ratio Probe

我们使用 Legacy Adapter ratio probe 检查低分辨率重建是否会破坏恶意 Adapter probe 所依赖的 weight/bias gradient ratios。该评估是 batch 1、known-label、sample-adaptive 的受控机制实验：probe inventory 随当前私有样本构造，攻击利用 legacy public-bin statistics 进行路由。因此，它既不是原始图像侧 PEFTLeak 的文字复现，也不是对正常 Adapter 或 LoRA 更新的被动部署攻击。

**Main Table 3. PEFTLeak-style Adapter ratio probe.** Privacy 为每配置单次 100/100 正式运行，不报告虚构标准差；utility 为对应防御配置下 benign Adapter training 的 seeds `101/202/303` mean ± population std。两者属于 cross-protocol comparison。

| Method | `rec_token_mean` | R1+R2 | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| none | `0.760000` | `145.000000` | `0.916667±0.002861` | `0.916589±0.002850` |
| Projection-LRB@0.5 | `0.202490` | `12.036481` | `0.909786±0.004619` | `0.909698±0.004647` |
| Projection-LRB@0.65 | `0.212765` | `13.693024` | `0.914373±0.003784` | `0.914304±0.003797` |
| Projection-LRB@0.9 | `0.234016` | `13.137760` | `0.913226±0.005487` | `0.913150±0.005500` |
| top-k@0.05 | `0.431431` | `48.125711` | `0.903670±0.000937` | `0.903515±0.000965` |
| compression@4 | `0.115571` | `7.733200` | `0.901758±0.003010` | `0.901622±0.003026` |
| Full-LRB@0.5 | `0.015222` | `0.461551` | `0.850535±0.016992` | `0.850337±0.017033` |

Projection-LRB@0.5 将 token recovery 从 `0.760000` 降至 `0.202490`，并将 R1+R2 从 `145.000000` 降至 `12.036481`；其 accuracy 均值为 `0.909786`，相对 benign none 低 `0.006881`。在这一受控协议内，它同时比 top-k@0.05 获得更低恢复和更高效用。Compression@4 与 Full-LRB@0.5 的恢复进一步降低，但 accuracy 分别降至 `0.901758` 和 `0.850535`。这些结果支持 signed reconstruction 会扰乱该恶意 probe 的梯度比值结构，但不能把单次 privacy 运行与三种子 utility 解释为同一上传更新上的严格 Pareto front。

### 8.2 Strictness Check and Scope

Appendix Table A1 的 fixed-probe v2 在私有样本到达前安装覆盖 128 个 token positions、包含 256 个 weight/bias gradient tensors 的固定 inventory，并严格隔离 public statistics。在该协议中，Projection-LRB@0.5 的三种子 token recovery 为 `0.685368±0.016109`，明显高于 Legacy probe 的单次 `0.202490`；noise@1e-3 则达到更低的 `0.358894±0.008963`。由于两个 probe 协议并不相同，这一对照不能作为逐点统计比较，但足以说明 Legacy 的 sample-adaptive 设计高估了 Projection-LRB 的实际抑制幅度。本文因此只将 Main Table 3 作为 Adapter ratio mechanism evidence，不外推到 LoRA、一般 PEFT updates 或现实部署攻击。

## 9. Partial-Gradient Leakage under PTG

### 9.1 PTG `first2` Protocol

PTG 评估在 SST-2/GPT-2 上仅暴露前两个 Transformer blocks。防御先作用于完整更新，随后 selector 保留 24 个 gradient tensors，其中 8 个为 matrix gradients；攻击者已知 batch size 1 的标签与 padding，并以单次初始化执行 80 个 Adam steps。每个配置在 seeds `101/202/303` 上各攻击 100 条文本。该 gradient-matching 协议的恢复率与 DAGER 或 Adapter ratio probe 不可直接比较。

### 9.2 Repeatable but Limited Privacy–Utility Transfer

未防御 `first2` 更新的 `rec_token_mean` 为 `0.160194±0.013164`，R1+R2 为 `19.772638±1.292773`。Projection-LRB@0.2 将二者降至 `0.121290±0.014928` 和 `14.616599±3.207703`。按相同 seed 配对，token recovery 的绝对下降为 `0.038904±0.023298`；由聚合均值计算，相对下降为 24.3%。三个 seed 的 token recovery 与 R1+R2 均沿相同方向下降，说明该变化在当前样本抽取下可重复，但幅度远未达到 full-gradient standard DAGER 中的零恢复端点。

**Main Table 4. PTG `first2` selected privacy–utility results.** 每行的 privacy 与 utility 均按 seeds `101/202/303` 计算 mean ± population standard deviation。Utility 使用与 PTG 相同的 SST-2 checkpoint、batch size 1 和 defense operating point 继续训练 1 epoch；训练时间为单次 utility run 的 wall-clock hours。

| Method | `rec_token_mean` | R1+R2 | Accuracy | Macro-F1 | Loss | Train time (h) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | `0.160194±0.013164` | `19.772638±1.292773` | `0.900230±0.003376` | `0.900178±0.003383` | `0.272405±0.003387` | `0.946±0.020` |
| Projection-LRB@0.2 | `0.121290±0.014928` | `14.616599±3.207703` | `0.913226±0.001950` | `0.913171±0.001967` | `0.258121±0.003804` | `10.871±0.927` |
| Projection-LRB@0.5 | `0.125698±0.011283` | `16.236530±0.903711` | `0.918960±0.002356` | `0.918945±0.002357` | `0.255761±0.010183` | `14.086±2.905` |
| top-k@0.1 | `0.154005±0.020510` | `18.806127±3.795858` | `0.911315±0.006374` | `0.911270±0.006407` | `0.265506±0.017164` | `3.323±0.696` |
| compression@8 | `0.149767±0.011645` | `19.011232±1.951503` | `0.911697±0.001622` | `0.911668±0.001632` | `0.248106±0.004579` | `4.526±0.450` |
| noise@5e-4 | `0.078178±0.003761` | `7.138664±1.702438` | `0.909404±0.006555` | `0.909391±0.006561` | `0.253409±0.016825` | `1.226±0.065` |
| DP-SGD-style@5e-4 | `0.070653±0.003013` | `6.569028±1.437764` | `0.920107±0.001430` | `0.920081±0.001424` | `0.633798±0.014201` | `1.725±0.021` |

Projection-LRB@0.2 与 @0.5 均降低当前 PTG 恢复指标，并在三种子均值上未观察到 accuracy 或 macro-F1 损失；这只支持 clean-level utility，不构成统计显著提升。代价是两点的平均训练时间分别达到 clean 的约 11.5 倍和 14.9 倍。Top-k@0.1 与 compression@8 在该攻击下仅带来较小恢复变化；noise 与 DP-SGD-style 的恢复率更低，但 DP-SGD-style 的 eval loss 升至 `0.633798±0.014201`，且没有 privacy accountant。完整 sweep 中 compression@2 的 token recovery 为 `0.055496±0.007032`，仍低于表内所有防御。因而，该表支持 Projection-LRB 向 `first2` gradient matching 的有限迁移，但不支持其在 PTG 下优于压缩或噪声基线。

### 9.3 Exposure Boundary

当前 partial-gradient 证据只覆盖 `first2`，因此所有结果和结论均限定于这一可见梯度设置，不能据此外推到其他 partial-gradient exposures。Appendix Table A3 报告未进入七点正文表的八个 `first2` full-sweep privacy 点，以避免把不同调参覆盖误读为同一完整 trade-off。

## 10. Knowledge-Aware Robustness Extension

### 10.1 Attacker-Knowledge Model

前三组实验回答同一个更新空间变换能否迁移到不同泄露协议；本节进一步回答，当 full-gradient attacker 了解防御实现但只掌握不同程度的实例级状态时，这种抑制能否维持。我们在 SST-2 official validation、GPT-2、batch size 2 上评估 Projection-LRB@0.5。每个配置使用 seeds `101/202/303`，每个 seed 包含 100 个单客户端更新。Static matrix 固定 defense seed 为 `700001`，攻击 seed 为 `900001`，使用 100 倍候选扩展，并以 `min` 作为有限 hypotheses 的主攻击聚合规则。

四种知识设置均知道 Projection-LRB 的形式和超参数，但对当前更新实际使用的状态拥有不同访问权限。Oracle 获得逐层 ratios 与 realized signs；ratio-hidden 获得精确 signs，但只能在 `0.2/0.35/0.5/0.65/0.75/0.9` 网格上枚举 ratios；signs-hidden 获得精确 ratios，但只能评估 16 个 sign-seed hypotheses；method-only 同时隐藏二者，因此评估 6×16 个组合。Ratio-hidden 与 signs-hidden 分别缺失不同状态，不能解释为一条严格递进的“知识等级”。这些设置衡量的是给定候选预算下的经验攻击能力，而不是密码学密钥泄露模型。

### 10.2 Static Knowledge-Stratified Evaluation

**Main Table 5. Static knowledge-stratified DAGER on Projection-LRB@0.5.** 数值为 seeds `101/202/303` 的 mean ± population std。Candidate recall 对应扩大候选集合中的真实 token recall，Top-B 为排序并截断后的候选 recall；L1/L2 是 DAGER 后续恢复阶段，R1+R2 衡量最终序列质量。

| Attacker knowledge | Candidate recall | Top-B recall | L1 | L2 | R1+R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| oracle | `0.919333±0.015605` | `0.914969±0.015609` | `0.436667±0.031710` | `0.795000±0.014142` | `101.872623±3.048924` |
| ratio-hidden | `0.984620±0.001817` | `0.947560±0.013014` | `0.000000±0.000000` | `0.000000±0.000000` | `9.421112±0.422567` |
| signs-hidden | `0.978630±0.011853` | `0.574774±0.053037` | `0.000000±0.000000` | `0.000000±0.000000` | `6.453098±0.458368` |
| method-only | `0.977448±0.011205` | `0.699022±0.041630` | `0.000000±0.000000` | `0.000000±0.000000` | `7.265339±0.514611` |

Oracle access restores substantial text recovery, showing that standard DAGER 的零恢复不能解释为 token 信息已从防御更新中不可逆地消失。更关键的是，隐藏 ratios 或 signs 时，candidate recall 仍保持在 `0.978630--0.984620`，但 L1/L2 恢复降为 0，R1+R2 降至 `6.45--9.42`。Signs-hidden 首先显著降低 Top-B recall；ratio-hidden 即使保持 `0.947560` 的 Top-B recall，最终序列仍然失败。两种路径共同表明，实例级状态主要影响候选排序、后续过滤和序列组装，而不是简单地把真实 token 排除在扩大候选池之外。

这一矩阵同时不支持更强的 layer-adaptivity 结论。Projection-LRB、rule-only、empirical-only 与 uniform reconstruction 在 oracle 下的 R1+R2 均约为 101--103，没有可辨识的稳健性差异。对 rule-only 与 uniform variants，ratio-hidden 与 oracle 完全相同，说明实际攻击 ratio 已固定；只有依赖实例级经验 ratios 的变体对 ratio hiding 敏感。进一步地，不使用 signed modulation 的 `proj_uniform_pool` 在四种知识设置下均约为 candidate recall `0.907`、R1+R2 `101.14`，method-only 与 oracle 完全相同。因此，当前证据支持 signed transformation state 是知识敏感性的主要来源，但不支持 hybrid sensitivity allocation 比 uniform reconstruction 更安全或不可替代。

### 10.3 Per-Update Random Signs and Finite-Hypothesis Stress Test

Static signs 可能允许攻击者跨更新学习并复用同一状态。为评估更频繁的随机化，我们进一步在每个可观察更新上重新采样 signs，并比较 oracle、signs-hidden 与 method-only。隐藏状态的攻击分别使用 `1/4/16/64` 个 sign-seed hypotheses；`min` 作为主攻击结果，`mean` 仅作为 EOT-style 敏感性分析，二者始终分行报告。Oracle 获得每次更新 realized signs，用于检验随机化本身是否在状态公开时仍能抑制恢复；有限 hypotheses 则只衡量 seed sampling 的预算敏感性，不能排除直接估计 signs、联合优化隐变量或跨更新推断等更强攻击。

> 非正文写作注：per-update matrix 当前日志快照为 81/102 个正式运行完成，尚未满足整表准入。本节最终数值、曲线和关于 `min`/`mean` 的结果性表述须在 102/102 完成、排除 partial/duplicate logs 并完成三种子审计后写入；不得将当前中间快照作为投稿结果。

单更新压力测试本身也不能证明 per-update signs 优于 static hidden seed。要区分两者，后续还需要专门的跨更新攻击：攻击者先从多次 static-sign 更新估计状态，再将其迁移到新的更新，并与 per-update resampling 下的同预算攻击比较。因而，即使有限 seed hypotheses 未接近 oracle，也只能报告为 EOT-style seed-sampling sensitivity，而不能写成一般 EOT robustness。

### 10.4 Implications and Limitations

三类主协议与知识分层回答的是两个不同问题。DAGER、Adapter ratio recovery 与 PTG 表明，Projection-LRB 可以作为同一个客户端侧更新变换部署在不同暴露路径之前，并在每个已评估协议内降低恢复；知识分层则揭示该统一框架的保护效果依赖攻击者能否获得 realized transformation state。这一发现强化了“统一防御接口”的贡献，但不把它提升为适用于任意攻击者的通用隐私保证。特别地，candidate recall 在 hidden-state regimes 中接近 1，说明当前机制更准确的描述是干扰可恢复结构的利用过程，而非删除全部 token 信息。

实验范围仍有四项主要限制。第一，主实验只覆盖 GPT-2 sequence classifiers 和三个文本分类任务；除知识分层使用 official validation 外，历史 standard DAGER 主表使用来自 training split 的内部 `val` subset。第二，Adapter 部分仍是 PEFTLeak-style controlled ratio probe，privacy 与 benign utility 属于 cross-protocol comparison，不能替代真实 LoRA/Adapter update 上的完整攻击评估；fixed-probe v2 也表明 Legacy probe 高估了抑制幅度。第三，PTG 只覆盖 `first2` exposure，尚不能外推到其他层、模块或可见梯度组合。第四，当前实现输出同形状 dense tensors，不减少通信字节数，并在 PTG utility 中带来约 11.5--14.9 倍训练时间；DP-SGD-style baseline 也没有 privacy accountant。

因此，本文不主张 Projection-LRB 普遍优于 top-k、compression 或 noise，也不主张 layer-wise adaptive allocation 已被证明必要。后续评估应检验跨更新状态学习，补充真实 PEFT updates 与至少一种额外 PTG exposure，并将同一固定 Projection-LRB operating point 用于三类攻击，以进一步隔离“统一接口”与“按协议调参”的影响。更大模型、生成任务、多客户端聚合、non-IID sampling、FedAvg 多轮本地更新和 secure aggregation 则属于进一步外部有效性验证。

## 11. Conclusion

本文探索一个面向多种文本重建攻击的统一更新空间防御。Projection-LRB 在客户端上传前对更新施加同一个低分辨率 signed reconstruction bottleneck，完整梯度、Adapter 参数子集与部分层梯度的暴露均发生在该变换之后。由此，防御不依赖 DAGER、PEFTLeak-style ratio recovery 或 PTG 的特定解码流程，而是直接干预三者共同读取的更新结构。

实验表明，这一统一变换在三个 GPT-2 文本分类任务的 standard DAGER 下将当前 token 与 ROUGE 恢复指标降至零并保持 clean-level utility，在受控 Adapter ratio probe 和 PTG `first2` 中也分别降低恢复。组件消融将低分辨率 reconstruction 识别为主要有效部分。知识分层扩展进一步显示，oracle 获得 realized ratios 与 signs 后可恢复大量序列，而隐藏实例级状态时真实 token 仍位于候选池中，失败主要发生在候选排序、过滤和序列组装阶段。

因此，本文的贡献是提出并实验探索一个可跨三类更新暴露复用的防御框架，而不是证明某个 operating point 对所有攻击都最优。当前证据不支持不可逆信息删除、形式化隐私、layer-wise adaptivity 的必要性或一般白盒稳健性。完成 per-update/EOT-style 审计、真实 PEFT update 攻击和更多 PTG exposures，是将这一统一框架推进为更强部署结论所需的下一步。

# Appendix

## Appendix A.1 Fixed-Probe Adapter Ratio Strictness Check

为检验 Legacy probe 的严格性，我们在 SST-2/GPT-2 Adapter 设置中固定一个覆盖 128 个 token positions、包含 256 个 weight/bias gradient tensors 的 probe inventory，并在观察任何私有样本之前完成安装；public statistics 来自与私有评估严格不相交的分区。攻击通过任务梯度观测标签，使用 seeds `101/202/303`，每个配置、每个种子攻击 100 条文本。表 A1 报告三种子的均值与 population standard deviation。Privacy 与 utility 使用配置对齐但彼此独立的攻击和训练运行，属于 cross-protocol 汇总，不构成逐样本或同一更新上的 Pareto 比较。

| Method | `rec_token_mean` | R1+R2 | Accuracy | Macro-F1 | Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | `1.000000±0.000000` | `193.666667±0.942809` | `0.916667±0.002861` | `0.916589±0.002850` | `0.240148±0.006773` |
| Projection-LRB@0.5 | `0.685368±0.016109` | `98.674391±2.184362` | `0.909786±0.004619` | `0.909698±0.004647` | `0.252664±0.003659` |
| Projection-LRB@0.65 | `0.761991±0.016691` | `117.764104±1.169737` | `0.914373±0.003784` | `0.914304±0.003797` | `0.250359±0.004204` |
| Projection-LRB@0.75 | `0.836252±0.011929` | `136.816461±1.603026` | `0.911315±0.006234` | `0.911226±0.006252` | `0.246870±0.005563` |
| Projection-LRB@0.9 | `0.837218±0.011560` | `137.105283±1.334013` | `0.913226±0.005487` | `0.913150±0.005500` | `0.246265±0.005097` |
| top-k@0.1 | `1.000000±0.000000` | `193.666667±0.942809` | `0.907875±0.004425` | `0.907739±0.004458` | `0.248322±0.003506` |
| compression@6 | `0.999444±0.000786` | `193.433333±0.612826` | `0.909786±0.003289` | `0.909692±0.003300` | `0.246968±0.002993` |
| noise@1e-3 | `0.358894±0.008963` | `58.458704±0.657623` | `0.912844±0.002477` | `0.912731±0.002490` | `0.243639±0.006130` |

在这一更严格协议中，Projection-LRB@0.5 仍降低当前探针的恢复指标，但幅度明显小于 Legacy 机制实验；noise@1e-3 的恢复率更低，是该协议下更强的经验隐私基线。这一结果不支持将旧 probe 的抑制幅度外推到标准 Adapter、LoRA 或一般 PEFT 更新。

## Appendix A.2 Projection Sensitivity and Keep-Ratio Checks

表 A2(a) 使用 k=0.5 的三种子 fine-ablation 批次。`no-empirical` 的原始运行三条均失败，故使用同协议完成的 resume 批次；其余变体来自同一主批次。所有 privacy 行均正常完成 100/100，utility 均包含 seeds `101/202/303`。

| Variant | `rec_token_mean` | R1+R2 | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| projection-only | `0.000000±0.000000` | `0.000000±0.000000` | `0.915137±0.003376` | `0.915086±0.003370` |
| rule-only | `0.000000±0.000000` | `0.000000±0.000000` | `0.914373±0.001949` | `0.914339±0.001968` |
| empirical-only | `0.000000±0.000000` | `0.000000±0.000000` | `0.917049±0.001081` | `0.917003±0.001086` |
| uniform | `0.000000±0.000000` | `0.000000±0.000000` | `0.917049±0.003289` | `0.917002±0.003286` |
| no-empirical | `0.000000±0.000000` | `0.000000±0.000000` | `0.914373±0.001949` | `0.914339±0.001968` |

表 A2(b) 只报告具有完整三种子 privacy 的 Projection-LRB keep-ratio sweep。k=0.5 与 k=0.75 的对应 utility 各有一个训练日志缺少正常 result status，因此不将两种子 utility 与其他三种子行混排。

| Sensitive keep ratio k | `rec_token_mean` | R1+R2 |
| ---: | ---: | ---: |
| 0.5 | `0.000000±0.000000` | `0.000000±0.000000` |
| 0.65 | `0.000000±0.000000` | `0.000000±0.000000` |
| 0.75 | `0.000000±0.000000` | `0.000000±0.000000` |
| 0.9 | `0.000000±0.000000` | `0.000000±0.000000` |

这些结果表明 standard DAGER 在该批次中对多种 sensitivity 设计和 keep ratio 都出现饱和式攻击失败，因而不能由零恢复进一步识别 sensitivity 设计的必要性。特别地，k=0.9 位于 sensitivity-to-keep-ratio 方向反转的一侧，不代表对敏感层施加了更窄瓶颈。

## Appendix A.3 PTG `first2` Full-Sweep Privacy Extension

表 A3 保留未进入七点 Main Table 4 的八个三种子 privacy 点。协议与第 9 节完全相同，所有行均为 3×`ok` 且完成 3×100/100；本表用于公开完整参数 sweep，不将缺少同表 utility 的行用于完整 trade-off 排名。

| Method | `rec_token_mean` | R1+R2 |
| --- | ---: | ---: |
| Projection-LRB@0.65 | `0.143678±0.010967` | `17.769638±1.072943` |
| Projection-LRB@0.75 | `0.125995±0.009605` | `17.722520±0.242273` |
| Projection-LRB@0.9 | `0.149386±0.008123` | `17.798912±1.461522` |
| top-k@0.05 | `0.145112±0.004404` | `18.035789±0.501852` |
| top-k@0.3 | `0.157531±0.004132` | `20.273679±0.518841` |
| compression@2 | `0.055496±0.007032` | `6.346535±0.946550` |
| compression@4 | `0.116038±0.004216` | `13.742571±2.752197` |
| compression@16 | `0.161157±0.005879` | `18.846869±1.468634` |

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

- **Standard DAGER 主结果：** 数值来自 `AAAI2027_EVIDENCE_TABLES.md` 第 2.1、2.2 节。设置为 GPT-2 sequence classification、batch size 2、每个配置和种子攻击 100 个单客户端更新实例，即共 200 条文本；seeds 为 `101/202/303`。`rec_token=0` 只表示当前攻击未恢复 token。SST-2 Projection-LRB@0.9 缺少配对 utility；三任务 noise@5e-4 与 DP-SGD-style@5e-4 coverage 也未形成完整日志，因此不进入 Main Table 1。
- **机制消融：** “projection 是主要组件、clipping 单独无效”来自证据表第 3 节。该消融使用 `./models/gpt2-ft-rt`，与 2026-07 三数据集主表不是同一次 sweep。Appendix A2 的 fine-ablation 来自 2026-05-10 批次；keep-ratio 表因 k=0.5/@0.75 utility 不完整而只报告 privacy。
- **Legacy Adapter ratio 机制实验：** 第 8.1 节使用 `Mechanistic Evaluation with a PEFTLeak-Style Adapter Ratio Probe`。旧协议的 privacy 来自 `log/peftleak_text_sst2/privacy/` 中无 seed 后缀的单次 100 条文本正式日志，utility 来自 `log/peftleak_text_sst2/utility_/` 的三个独立训练种子，故两类指标是 cross-protocol comparison。该协议使用 known label、sample-adaptive malicious probe 和 legacy public-bin statistics，只覆盖 Adapter 梯度比值机制。
- **Fixed-probe v2 严格性检查：** 附录数据来自 `log/peftleak_text_sst2_v2/20260714_024606/formal/` 的 24 个正式运行；协议使用覆盖 128 个 token positions、包含 256 个 weight/bias gradient tensors 的固定 probe inventory，在私有数据到达前安装，并采用严格 disjoint public partition。证据表第 6 节已经与 Appendix A1 同步。
- **PTG 有限迁移：** 来自证据表第 5 节。协议为 SST-2、GPT-2、batch size 1、first two Transformer blocks 的 24 个 gradient tensors（8 个 matrix gradients）、known label/padding、single restart、80 个 Adam attack steps、学习率 0.1、cosine matching、embedding-norm weight 0.01，每个种子攻击 100 条文本。Main Table 4 的七个预先选定点均已完成三种子 privacy 和配置对齐 utility；汇总排除了旧的未完成 Projection-LRB@0.5/seed101 与重复的 Projection-LRB@0.2/seed303 副本。
- **静态知识分层矩阵：** 第 10.1--10.2 节来自 `log/同预算 white-box baselines/new/adaptive_lrb_matrix_sst2_official_validation_20260718_114719/`。该批次 60/60 正式运行均为 `ok` 且完成 100/100，覆盖 5 个 projection variants、4 种 attacker-knowledge settings 与 seeds `101/202/303`；数据为 SST-2 official validation，checkpoint digest、git commit、软件版本和 GPU 信息随 manifest 归档。Projection-LRB@0.5 的三种子均值与 population std 已由 `results.csv` 逐 seed 行复算。
- **Per-update/EOT-style 状态：** 第 10.3 节对应 `adaptive_lrb_matrix_sst2_official_validation_20260718_114758/`。当前本地快照为 81/102 个 exit-code-0 运行，另有一个 partial log，尚未生成完整 `results.csv`，因此正文不冻结中间数值。最终汇总必须将 `min` 主攻击与 `mean` sensitivity 分行，并明确 finite seed hypotheses 不能排除直接状态估计、隐变量优化或跨更新推断。
- **强基线定位：** Standard DAGER Main Table 1 的高保留率 top-k 与高 bit compression 为非零恢复，独立 full sweep 中更强压缩点包含零恢复端点。PTG 七点表中 noise 与 DP-SGD-style 的恢复率低于 Projection-LRB，完整 privacy sweep 中 compression@2 进一步达到更低恢复；因此即使配对 utility 已完成，也不作 Projection-LRB 普遍优于这些基线的排序。
- **攻击数据来源：** Standard DAGER 主表中的 `split=val` 来自原始 training split，而非官方 validation 或未参与模型训练的 held-out set。`TextDataset` 先随机排列 training split，以前 1000 条构造内部 `test`，再从余下样本按长度分层抽取内部 `val`；知识分层实验则使用 SST-2 official validation。最终英文稿必须分别说明，不能沿用同一个 `val` 名称。
- **实现边界：** `signed_pool` 是降采样后插值恢复算子，不具备严格的正交算子性质。主实验与静态矩阵按同一设置复用 signs；per-update 扩展按更新重采样。当 \(k>0.75\) 时 sensitivity-to-keep-ratio 的方向反转。输出仍为同形状 dense tensors，当前实现不降低通信字节数。
- **运行环境待确认：** 114719/114758 manifests 已记录知识分层批次的 GPU、代码 commit、checkpoint digest 和主要软件版本，但其他历史正式日志仍未完整记录这些字段。仓库中的 Conda 配置不能替代逐运行 provenance，投稿归档前仍需补齐主表、Adapter 与 PTG 证据链。
- **引用状态：** 上表用于锁定中文草稿中的 citation keys，不替代正式 BibTeX。GPT-2 与新增数据集条目的题名、作者和发表信息已按原论文或官方来源核对；本轮访问 GLUE 的 OpenReview 页面时遇到浏览器验证限制，`wang2019glue` 暂按通行的 ICLR 2019 元数据锁定。转写英文稿时仍需从官方页面导入条目，并复核 GLUE 条目以及 ReCIT 的最终发表状态、会议页码与完整作者信息。
- **本轮文件边界：** 本轮只修改中文论文草稿中的定位、知识分层章节和证据备注；未修改实验代码、原始日志、证据表或实验 TODO。Main Table 5 的静态知识结果直接由 114719 `results.csv` 的三种子正式行复算，per-update 结果保持待定。
