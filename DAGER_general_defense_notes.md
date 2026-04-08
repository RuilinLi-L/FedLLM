下面把 [FL-LLM.md](./FL-LLM.md) 里的任务设定和 [2405.15586v2.pdf](./2405.15586v2.pdf) 里的 DAGER 统一整理成一版“可以直接拿去写开题/方法设计”的说明。为了清楚，我分成两部分：

1. DAGER 到底做了什么，公式怎么理解。
2. 一个我认为更适合你这个课题的“通用防御机制”初稿，给出完整数学形式、变量定义、直觉解释、为什么可能同时压制三类攻击。

**一、DAGER：它到底做了什么**

**1. 问题设置**

DAGER 的攻击场景是标准的“honest-but-curious server”：

- 客户端本地有私有文本数据。
- 客户端不上传原始数据，只上传梯度或更新。
- 服务器表面上正常聚合，但实际上想从这些中间信息里把原始文本恢复出来。

DAGER 的核心结论是：

- 对 Transformer/LLM，尤其是 decoder-only 模型，前几层自注意力里的梯度不仅“有泄露”，而且常常足够做近乎精确的文本恢复。

**2. 符号与量的定义**

下面这些符号，基本就是 DAGER 的数学骨架。

- \(B\)：batch size，也就是一个客户端这次上传对应多少条序列。
- \(P\)：模型允许的最大上下文长度。
- \(n_j\)：第 \(j\) 条序列的真实长度。
- \(n = \max_j n_j\)：这个 batch 里最长序列的长度。
- \(b = \sum_{j=1}^{B} n_j\)：这个 batch 里所有非 padding token 的总数。
- \(d\)：模型隐藏维度，也可以理解成 embedding / hidden state 的维度。
- \(\mathcal V\)：词表集合。
- \(V = |\mathcal V|\)：词表大小。
- \(v\)：某个 token 在词表中的编号。
- \(i\)：token 在序列中的位置。
- \(f^0(v,i)\in \mathbb R^d\)：第 0 层 embedding 函数，把“词表 token + 位置”映射到一个 \(d\) 维向量。
- \(z_i^{(j)} = f^0(v_i^{(j)},i)\)：第 \(j\) 条样本第 \(i\) 个 token 的 embedding。
- \(Z_1 \in \mathbb R^{b\times d}\)：把整个 batch 的所有 token embedding 堆起来形成的矩阵。
- \(Z_l \in \mathbb R^{b\times d}\)：第 \(l\) 个 self-attention 层的输入表示。
- \(W_l^Q, W_l^K, W_l^V \in \mathbb R^{d\times d}\)：第 \(l\) 层 self-attention 的 query/key/value 投影矩阵。
- \(Q_l = Z_l W_l^Q\)，\(K_l = Z_l W_l^K\)，\(V_l = Z_l W_l^V\)：self-attention 三个线性投影后的表示。
- \(M\)：attention mask。对于 decoder，它是 causal mask，保证当前位置只能看前面的 token。
- \(L\)：训练损失。
- \(f_i^l(s)\in \mathbb R^d\)：给定一条 token 序列 \(s\)，在第 \(l\) 层 self-attention 输入处，第 \(i\) 个位置的表示向量。
- \(G_l^Q = \nabla_{W_l^Q}L\in \mathbb R^{d\times d}\)：第 \(l\) 层 query 投影矩阵的梯度。
- \(\Delta_l^Q = \nabla_{Q_l}L\in \mathbb R^{b\times d}\)：损失对 \(Q_l\) 的梯度。
- \(\tau_l\)：第 \(l\) 层做 span check 时用的接受阈值。
- \(T_i^\*\)：DAGER 恢复出来的“第 \(i\) 个位置可能出现过的 token 集合”。
- \(S_i^\*\)：DAGER 恢复出来的“长度为 \(i\) 的候选前缀序列集合”。

**3. DAGER 最核心的公式**

先看一个一般线性层：

如果某一层写成

\[
Y = XW,
\]

其中

- \(X\in\mathbb R^{b\times n}\) 是输入，
- \(W\in\mathbb R^{n\times m}\) 是权重，
- \(Y\in\mathbb R^{b\times m}\) 是输出，

那么对权重的梯度可以写成

\[
\nabla_W L = X^\top \nabla_Y L.
\]

把 \(\nabla_Y L\) 记作 \(\Delta\)，就变成

\[
\nabla_W L = X^\top \Delta.
\]

这一步非常关键，因为它意味着：

- 权重梯度不是“神秘的黑盒”。
- 它本质上是“输入矩阵”和“反向误差信号”的乘积。

因此有

\[
\operatorname{rank}(\nabla_W L)\le b.
\]

因为这个矩阵最多由 \(b\) 个样本/token 的信息叠加出来。

把这个结论代入 self-attention 的 query 投影层，就有

\[
G_l^Q = \nabla_{W_l^Q}L = Z_l^\top \Delta_l^Q.
\]

其中

- \(Z_l\in\mathbb R^{b\times d}\) 是该层输入表示，
- \(\Delta_l^Q\in\mathbb R^{b\times d}\) 是对 query 输出的梯度。

如果满足

\[
b < d
\]

并且 \(\Delta_l^Q\) 满秩到 \(b\)，那么论文给出的关键结论可以理解为：

\[
\operatorname{rowspan}(Z_l)=\operatorname{colspan}(G_l^Q).
\]

这里

- \(\operatorname{rowspan}(Z_l)\) 表示 \(Z_l\) 所有行向量张成的子空间。
- \(\operatorname{colspan}(G_l^Q)\) 表示梯度矩阵 \(G_l^Q\) 的列空间。

通俗解释就是：

- 客户端这批 token 在该层产生的那些表示向量，
- 它们张成的方向空间，
- 几乎就等于服务器看到的梯度张成的方向空间。

所以服务器其实拿到了一个“能刻画输入 token 表示所在方向”的空间。

**4. DAGER 的 span check**

DAGER 不直接“猜一句文本然后优化”，而是做一个成员检查：

“某个候选 token 的 embedding，是否落在梯度张成的子空间里？”

定义候选向量 \(z\in\mathbb R^d\) 到这个子空间的投影距离：

\[
d(z,l)=\left\|z-\operatorname{proj}(z,\operatorname{colspan}(G_l^Q))\right\|_2.
\]

其中

- \(\operatorname{proj}(z,\operatorname{colspan}(G_l^Q))\) 是把 \(z\) 投影到梯度列空间上，
- \(\|\cdot\|_2\) 是欧氏范数。

如果

\[
d(z,l) < \tau_l,
\]

就认为这个候选向量“很可能确实来自真实输入”。

通俗理解：

- 如果一个候选 token 真出现在客户端文本里，它的 embedding 往往会和梯度空间“贴得很近”。
- 如果它没出现过，一般就离得远。

所以 DAGER 是在做“几何验真”。

**5. 第一步：恢复每个位置可能有哪些 token**

因为 embedding 函数 \(f^0(v,i)\) 是已知的，服务器可以遍历词表和位置，对每个 \((v,i)\) 都算一次 embedding，再做 span check。

于是第 \(i\) 个位置上的候选 token 集合定义为

\[
T_i^\*=\left\{v\in\mathcal V \mid d(f^0(v,i),1)<\tau_1\right\}.
\]

这里的意思是：

- 对词表里每个 token \(v\)，都把它放到位置 \(i\) 上，
- 计算它在第 1 层对应的 embedding，
- 看它是不是和第一层 query 梯度空间足够接近。

如果接近，就把它加入 \(T_i^\*\)。

通俗理解：

- 第一步不是恢复整句话，
- 而是先恢复“每个位置可能出现过哪些词”。

这一步像是先把题目的选项缩到很小。

**6. 第二步：把 token 拼成真正的句子**

如果只是知道每个位置有哪些候选 token，还不够，因为排列组合仍然很多。

DAGER 的第二步用的是第二层 self-attention。

对于 decoder-only 模型，因为有 causal mask，第 \(i\) 个位置的表示只依赖前缀 \(s_{1:i}\)，不依赖后面的 token。这个性质非常重要。

先定义

\[
S_0^\*=\{\epsilon\},
\]

其中 \(\epsilon\) 是空序列。

然后递推地构造：

\[
S_i^\*=
\left\{
(s_{1:i-1}, v)\;\middle|\;
s_{1:i-1}\in S_{i-1}^\*,\;
v\in T_i^\*,\;
d(f_i^1(s_{1:i-1},v),2)<\tau_2
\right\}.
\]

这里

- \(S_{i-1}^\*\) 是长度为 \(i-1\) 的候选前缀集合，
- 把第 \(i\) 位候选 token \(v\) 接上去，
- 用第二层表示 \(f_i^1\) 再做一次 span check，
- 通过检查的前缀才保留。

通俗解释：

- 第一步找到“每个位置可能有什么词”。
- 第二步是“从左往右接龙”，每次只保留真正能和第二层梯度对上的前缀。
- 因为 decoder 的第 \(i\) 个位置只看前面，所以可以贪心地往前推进。

这就是为什么 DAGER 对 decoder 很强。

**7. encoder 为什么更难**

对 encoder 来说，没有 causal mask。第 \(i\) 个位置会看见整句，所以不能像 decoder 一样从左到右贪心恢复。

理论上可以定义整句候选集合

\[
S = T_1^\* \times T_2^\* \times \cdots \times T_P^\*
\]

然后筛选出真正满足

\[
S^\*=
\left\{
s\in S \mid d(f_i^1(s),2)<\tau_2,\ \forall i
\right\}.
\]

但这个搜索空间太大，所以论文里对 encoder 只能靠启发式：

- 先猜 EOS 位置来估计每句长度。
- 每个位置只保留离 span 最近的前 \(B\) 个 token。
- 只搜索有限数量的候选序列。

这也是为什么 DAGER 对 encoder 依然强，但没有 decoder 那么夸张。

**8. 为什么 DAGER 比以前强很多**

本质上，旧方法更像：

- “我猜一句文本，然后用优化让它的梯度尽量像真的梯度。”

DAGER 更像：

- “我先根据梯度判定哪些 token 确实出现过，再用结构性质把它们拼起来。”

区别在于：

- 旧方法是连续优化，文本是离散对象，所以很难。
- DAGER 直接利用“文本是离散 token”这件事，把问题变成了搜索和验真。

所以它不再是“近似恢复”，而是“精确恢复”。

**9. 为什么大模型反而更危险**

DAGER 的关键条件是

\[
b < d.
\]

其中

- \(b\) 是这一批里总 token 数，
- \(d\) 是隐藏维度。

如果模型更大，通常 \(d\) 更大，那么这个条件更容易成立。也就是说：

- 隐藏维度越大，
- 梯度空间越能容纳更多 token 的几何信息，
- 攻击能恢复的 batch 和序列长度也越大。

这也是论文里一个很重要、也很危险的结论：

- 更大的 LLM 不一定更安全，
- 反而可能泄露得更多。

**10. DAGER 的实验结果怎么读**

**先看 decoder，也就是 GPT-2 / LLaMA 这种。**

在 GPT-2 上，DAGER 对小到中等 batch 几乎是“满分恢复”：

- `CoLA` 上，\(B=1,2,4,8\) 时，ROUGE-1/2 基本都是 \(100/100\)。
- `SST-2` 上，ROUGE-1 都是 \(100\)，ROUGE-2 大约 \(86.0, 89.5, 92.8, 92.9\)。论文说明这里有一部分是度量工具对单词句子的 artifact，不是真恢复失败。
- `Rotten Tomatoes` 上，\(B=1,2\) 是 \(100/100\)，\(B=4,8\) 也接近 \(99\%-100\%\)。

和 TAG / LAMP 对比时，差距很大：

- TAG 往往只有个位数到几十的 ROUGE-1。
- LAMP 比 TAG 强，但随着 batch 增大很快掉下去。
- DAGER 基本一路接近满分。

**再看 encoder，也就是 BERT。**

DAGER 对 BERT 也明显领先：

- `CoLA` 上，\(B=1,2\) 基本 \(100/100\)，\(B=4\) 还有 \(94.0/89.9\)，\(B=8\) 还有 \(67.8/48.8\)。
- `SST-2` 上，\(B=8\) 还能到 \(74.1/59.8\)。
- `Rotten Tomatoes` 上，\(B=8\) 是 \(37.1/11.4\)，虽然不如 decoder，但仍明显高于基线。

也就是说：

- encoder 上它不是“总能精确恢复”，
- 但仍然比以前的攻击强很多。

**再看更大的 LLM。**

在 `LLaMa-2 7B` 上，结果最惊人：

- `CoLA`，\(B=128\) 时仍然大约 \(99.5/99.3\)。
- `SST-2`，\(B=128\) 时仍然大约 \(98.2/97.8\)。
- `Rotten Tomatoes`，\(B=128\) 时仍然大约 \(99.7/99.7\)。

这说明：

- 大模型隐藏维度大，
- DAGER 的低秩/span 机制在更大 batch 下依然成立，
- 泄露风险非常严重。

相对地，GPT-2 因为 \(d=768\) 比较小，到了 \(B=64,128\) 就开始明显掉。

**FedAvg 也不安全。**

论文还测了 FedAvg，不只是单步 FedSGD：

- 在 GPT-2 + Rotten Tomatoes + batch 16 上，
- 即使有多轮本地训练、小 batch、本地学习率等变化，
- DAGER 仍能拿到 \(95\%\) 甚至接近 \(100\%\) 的 ROUGE。

这很重要，因为这说明：

- 不是只有“最理想化的 FedSGD”才会泄露，
- 更实际的联邦训练协议也会。

**LoRA 也不安全。**

论文在 LoRA 上报告：

- 当 LoRA rank \(r=256\) 时，
- 仍能达到大约 \(94.8\) 的 ROUGE-1，
- 和 \(94.2\) 左右的 ROUGE-2。

所以“只传 PEFT 更新会更安全”这件事，并不能直接成立。

**长序列也不安全。**

论文还做了一个长序列实验：

- `ECHR` 数据集，
- 截断到 512 tokens，
- \(B=1\)，
- GPT-2 上 DAGER 的 ROUGE-1/2 是 \(100/100\)。

而 LAMP 在同设置下 ROUGE-1 只有约 \(10.1\)。

这表明 DAGER 不是只能搞短句子。

**速度也更快。**

论文给的典型结果是：

- GPT-2 + Rotten Tomatoes + \(B=8\) 时，
- 100 个 batch 的总攻击时间：
- DAGER 约 \(3.5\) 小时，
- TAG 约 \(10\) 小时，
- LAMP 约 \(50\) 小时。

所以它不只是更准，而且还更实用。

---

**二、一个更适合你课题的通用防御机制**

下面这部分不是 DAGER 论文原文，而是基于你在 [FL-LLM.md](./FL-LLM.md) 里的目标，我给出的一个“初步但可落地”的统一方案。

它的核心思想不是：

- “专门防 DAGER”
- 或者“专门防某一个攻击代码”。

而是：

- 直接降低“更新里能被恢复成原始数据”的能力本身。

我给它起一个名字：

**层级恢复性瓶颈防御**
**Layer-wise Recoverability Bottleneck, LRB**

你可以把它理解成：

- 我们不让服务器看到“高清、可逆、样本级细节很强”的更新，
- 只让它看到“足够训练，但不够还原文本”的更新。

**1. 先写出理想目标**

设：

- \(X_c\)：客户端 \(c\) 的本地原始数据。
- \(Y_c\)：客户端标签或训练目标。
- \(\theta_t\)：第 \(t\) 轮全局模型参数。
- \(U_c\)：客户端原始梯度或本地训练后得到的原始更新。
- \(F_\phi\)：带参数 \(\phi\) 的防御机制。
- \(\widetilde U_c = F_\phi(U_c)\)：客户端真正上传的、经过防御后的更新。

我们真正想优化的是：

\[
\min_\phi \; \mathbb E\big[\mathcal L_{\text{task}}(\theta_t-\eta \widetilde U_c)\big]
+ \lambda \, I(X_c;\widetilde U_c).
\]

这里

- \(\mathcal L_{\text{task}}\)：任务损失，比如分类 loss 或 language modeling loss。
- \(\eta\)：学习率。
- \(I(X_c;\widetilde U_c)\)：原始数据 \(X_c\) 和上传更新 \(\widetilde U_c\) 之间的互信息。
- \(\lambda\)：平衡隐私和效用的权重。

通俗解释：

- 第一项希望模型还能学得好。
- 第二项希望上传信息里尽量少带原始数据内容。

这就是“通用防御”的本质目标。

**2. 为什么这个目标比“防某个攻击”更对**

如果你只针对 DAGER 调参，那很可能出现：

- DAGER 掉了，
- 但 PEFTLeak 还可以恢复；
- 或者 partial-gradient attack 还可以恢复。

所以你的目标应当写成“跨攻击族”的：

设攻击集合

\[
\mathcal A=
\{\text{DAGER},\ \text{PEFTLeak/ReCIT},\ \text{PartialGrad}\}.
\]

定义经验风险：

\[
P(\phi)=\sum_{a\in\mathcal A} w_a \cdot \operatorname{ASR}_a(\phi),
\]

其中

- \(w_a\ge 0\) 是第 \(a\) 个攻击族的权重，
- \(\operatorname{ASR}_a(\phi)\) 是防御参数为 \(\phi\) 时，第 \(a\) 类攻击的成功度量。

这里的 \(\operatorname{ASR}_a\) 可以用下列任一种或组合：

\[
\operatorname{ASR}_a
=
\alpha_1 \cdot \text{ROUGE-1}_a
+\alpha_2 \cdot \text{ROUGE-2}_a
+\alpha_3 \cdot \text{ExactMatch}_a.
\]

其中

- \(\alpha_1,\alpha_2,\alpha_3\ge 0\)，通常可令它们和为 1。
- ExactMatch 表示完全恢复的比例。

于是最终你真正要找的是 Pareto 最优点：

\[
\phi^\*
=
\arg\min_\phi P(\phi)
\quad
\text{s.t.}
\quad
U(\phi)\ge U_{\text{base}}-\epsilon.
\]

这里

- \(U(\phi)\)：模型效用，比如验证准确率/F1/ROUGE/perplexity。
- \(U_{\text{base}}\)：无防御 baseline 的效用。
- \(\epsilon\)：允许的精度损失，比如 1% 到 3%。

通俗解释：

- 在模型效果别掉太多的前提下，
- 让三类攻击整体最难恢复。

这正是你课题想做的事。

---

**三、LRB 防御机制的完整形式**

**1. 基本对象**

对某一层 \(l\)，记客户端 \(c\) 在第 \(t\) 轮第 \(e\) 个本地 micro-step 的原始梯度为

\[
G_{c,t,l}^{(e)} \in \mathbb R^{m_l\times n_l}.
\]

这里

- \(m_l,n_l\) 是第 \(l\) 层参数矩阵的尺寸。
- 对 attention / MLP 权重，通常它们就是该层权重矩阵的行数和列数。
- 对向量参数可以看成特殊矩阵，或令 \(n_l=1\)。

如果采用 \(E\) 个本地 micro-step 再上传，那么先做本地混合：

\[
\bar G_{c,t,l}
=
\frac{1}{E}
\sum_{e=1}^{E}
G_{c,t,l}^{(e)}.
\]

这里

- \(E\)：客户端本地混合的步数。
- \(\bar G_{c,t,l}\)：第 \(l\) 层混合后的平均梯度。

通俗解释：

- 不让服务器看到每个很小 batch 的“清晰快照”，
- 而让它看到几步平均后的“模糊快照”。

这会天然削弱样本级可恢复性。

**2. 按层决定“谁更敏感”**

不是所有层都一样危险。你在课题里已经明确有一类攻击就是 partial-layer leakage，所以防御必须分层设计。

定义攻击敏感度分数：

\[
\kappa_l
=
\frac{1}{\sum_{a\in\mathcal A}w_a}
\sum_{a\in\mathcal A}
w_a
\cdot
\frac{\operatorname{ASR}_{a,l}}{\max_j \operatorname{ASR}_{a,j}}.
\]

这里

- \(\operatorname{ASR}_{a,l}\)：只给攻击算法 \(a\) 第 \(l\) 层的更新时，它的恢复成功率。
- \(\kappa_l\in[0,1]\)：越大表示这一层越容易泄露数据。

通俗解释：

- 先做一次离线 profiling。
- 看三类攻击最喜欢吃哪几层。
- 泄露最强的层，后面就给它更重的保护。

通常高敏感层会是：

- token embedding 层，
- 前 \(K\) 个 transformer block，
- 以及这些层上挂着的 LoRA / Adapter 模块。

记这些高敏感层集合为

\[
\mathcal S = \{\text{embedding},\text{block }1,\dots,\text{block }K,\text{their PEFT params}\}.
\]

**3. 先做按层裁剪**

对每层先做 Frobenius 范数裁剪：

\[
G_{c,t,l}^{\text{clip}}
=
\bar G_{c,t,l}
\cdot
\min\left(
1,\frac{C_l}{\|\bar G_{c,t,l}\|_F}
\right).
\]

这里

- \(\|\cdot\|_F\)：矩阵的 Frobenius 范数。
- \(C_l\)：第 \(l\) 层的裁剪阈值。

可以让 \(C_l\) 随敏感度变化：

\[
C_l
=
C_{\max}
-
\kappa_l\,(C_{\max}-C_{\min}).
\]

这表示：

- 越敏感的层，
- 裁剪越狠。

通俗解释：

- 裁剪就是限制“一个客户端一次能带出多少私有信息”。
- 它像给每层更新装上了流量阀门。

**4. 核心步骤：投影到公共低维子空间**

这是我认为最关键的一步，也是它和单纯 DP/noise baseline 最大的不同。

先为每层准备两个公共基：

\[
P_l \in \mathbb R^{m_l\times r_l},
\qquad
Q_l \in \mathbb R^{n_l\times c_l},
\]

满足

\[
P_l^\top P_l = I_{r_l},
\qquad
Q_l^\top Q_l = I_{c_l}.
\]

这里

- \(r_l\ll m_l\)，\(c_l\ll n_l\) 对高敏感层通常成立。
- \(P_l,Q_l\) 不是从当前客户端私有数据上学出来的，
- 而是从公共数据或历史大规模聚合更新统计中提取的“公共、任务相关子空间”。

一种可行做法是先准备公共校准更新 \(H_{l,1},\dots,H_{l,M}\)，然后算协方差：

\[
\Sigma_l^{\text{row}}
=
\frac{1}{M}
\sum_{m=1}^M
H_{l,m}H_{l,m}^\top,
\]

\[
\Sigma_l^{\text{col}}
=
\frac{1}{M}
\sum_{m=1}^M
H_{l,m}^\top H_{l,m}.
\]

然后取它们最大的特征向量作为基：

- \(P_l\)：\(\Sigma_l^{\text{row}}\) 的前 \(r_l\) 个特征向量。
- \(Q_l\)：\(\Sigma_l^{\text{col}}\) 的前 \(c_l\) 个特征向量。

之后把第 \(l\) 层裁剪后的更新投影进去：

\[
G_{c,t,l}^{\text{proj}}
=
P_lP_l^\top
G_{c,t,l}^{\text{clip}}
Q_lQ_l^\top.
\]

这里

- 左乘 \(P_lP_l^\top\) 表示只保留“行空间”里落在公共子空间的部分。
- 右乘 \(Q_lQ_l^\top\) 表示只保留“列空间”里落在公共子空间的部分。

还可以把投影维度设计成与敏感度相关：

\[
r_l
=
\max\left(r_{\min},\left\lfloor (1-\rho\kappa_l)m_l \right\rfloor\right),
\]

\[
c_l
=
\max\left(c_{\min},\left\lfloor (1-\rho\kappa_l)n_l \right\rfloor\right).
\]

这里

- \(\rho\in[0,1]\) 是压缩强度。
- \(\kappa_l\) 越大，\(r_l,c_l\) 越小。

通俗解释：

- 这一步强迫更新只能沿着“公共、粗粒度、任务共有”的方向上传。
- 而不是沿着“这次这几个 token 精确带出来的私有方向”上传。

这是为什么它对 DAGER 特别有希望。

因为 DAGER 依赖的是：

\[
\operatorname{rowspan}(Z_l)
=
\operatorname{colspan}(G_l^Q).
\]

也就是“私有 token 表示空间”和“共享梯度空间”高度一致。

而经过投影后，上传的是

\[
G_{c,t,l}^{\text{proj}}
=
P_lP_l^\top G_{c,t,l}^{\text{clip}} Q_lQ_l^\top,
\]

因此

\[
\operatorname{colspan}(G_{c,t,l}^{\text{proj}})
\subseteq
\operatorname{colspan}(P_l).
\]

也就是说：

- 服务器看到的列空间，主要由公共基 \(P_l\) 决定，
- 而不再是当前客户端这批 token 的私有 span 直接决定。

DAGER 的 token membership check 之所以强，是因为“真 token embedding 在这个空间里，假 token 不在”。  
但现在这个空间变成了公共低维空间，很多 token 投影后会互相混叠，区分度会明显下降。

**5. 再加噪声**

在投影之后，对每层加高斯噪声：

\[
N_{c,t,l}\sim \mathcal N(0,\sigma_l^2 C_l^2 I),
\]

\[
\widetilde G_{c,t,l}
=
G_{c,t,l}^{\text{proj}} + N_{c,t,l}.
\]

这里

- \(N_{c,t,l}\) 表示每个元素独立采样的高斯噪声矩阵。
- \(\sigma_l\) 是第 \(l\) 层噪声倍率。

同样可以让噪声随敏感度变化：

\[
\sigma_l
=
\sigma_{\min}
+
\kappa_l(\sigma_{\max}-\sigma_{\min}).
\]

这表示：

- 越敏感的层，噪声越大。
- 越不敏感的层，噪声越小。

通俗解释：

- 投影负责“改结构”，
- 噪声负责“抹细节”。

单独加噪声有时很难调，因为噪声要足够大才能防住攻击，但太大又会伤精度。  
而这里先用投影把真正危险的高分辨率方向砍掉，再用较温和的噪声收尾，通常更容易拿到好的隐私-效用平衡。

**6. 服务器聚合与更新**

客户端上传的是 \(\widetilde G_{c,t,l}\)。服务器按联邦平均聚合：

\[
\widetilde G_{t,l}
=
\sum_{c=1}^{N}
p_c \widetilde G_{c,t,l},
\]

其中

\[
p_c=\frac{|D_c|}{\sum_{c'}|D_{c'}|}.
\]

最后更新模型：

\[
W_{t+1,l}
=
W_{t,l}
-
\eta_t \widetilde G_{t,l}.
\]

这里

- \(N\)：参与本轮训练的客户端数。
- \(|D_c|\)：客户端 \(c\) 的数据量。
- \(\eta_t\)：第 \(t\) 轮学习率。

---

**四、LRB 为什么可能同时压三类攻击**

**1. 对 DAGER / LAMP 这类全梯度恢复攻击**

DAGER 的强点在于：

- 第一层用 span check 恢复 token 集，
- 第二层再恢复前缀结构。

它最依赖的是：

- 早期层更新的几何结构很“干净”，
- 当前私有样本的 token/前缀信息直接体现在梯度子空间里。

LRB 对它的打击有三层：

\[
\bar G \rightarrow G^{\text{clip}} \rightarrow G^{\text{proj}} \rightarrow \widetilde G.
\]

含义分别是：

- 本地混合让单样本痕迹不那么尖锐。
- 裁剪限制单次可泄露幅度。
- 公共子空间投影破坏“私有 token span = 共享梯度 span”。
- 噪声让边缘判定更不稳定。

所以 DAGER 那个

\[
d(z,l)=\|z-\operatorname{proj}(z,\operatorname{colspan}(G_l^Q))\|_2
\]

在防御后会失去区分力。  
因为服务器见到的不是原始 \(G_l^Q\)，而是被改写过空间结构的 \(\widetilde G_l\)。

**2. 对 PEFTLeak / ReCIT 这类 PEFT 更新恢复攻击**

LoRA 的有效更新通常写成

\[
\Delta W_l = A_l B_l,
\]

其中

- \(A_l\in\mathbb R^{d\times r}\),
- \(B_l\in\mathbb R^{r\times d}\),
- \(r\) 是 LoRA rank。

虽然这是低秩更新，但恰恰因为它结构化、参数少、贴近训练信号，所以也可能泄露。

在 LRB 里，可以对 LoRA 的有效更新直接做同样的防御：

\[
\widetilde{\Delta W_l}
=
F_\phi(\Delta W_l).
\]

如果最后还需要用 LoRA 形式回写，可以再做截断 SVD 重新分解：

\[
\widetilde{\Delta W_l}
\approx
\widetilde A_l \widetilde B_l.
\]

通俗解释：

- PEFTLeak 吃的是“LoRA 里那点精炼但私密的信息”。
- LRB 则是让 LoRA 更新只能保留公共子空间里的粗粒度方向，外加裁剪和噪声。
- 这样 LoRA 里那种“低秩但很私密”的结构就被削弱了。

**3. 对 partial-layer leakage 这类局部层攻击**

这类攻击的危险在于：

- 它甚至不需要完整更新，
- 只靠前几层或者一部分层的梯度就能恢复。

所以如果你的防御是“全局统一弱处理”，很可能没用。因为攻击者只盯着最敏感那几层。

LRB 的关键正是分层：

- 用 \(\kappa_l\) 找到最危险的层。
- 对这些层用更强的压缩、更小的阈值、更大的噪声。
- 对后面层则保留更多训练信息。

也就是说，它不是平均用力，而是“精准削掉最泄露的层”。

---

**五、这个方案和 baselines 的关系**

你 md 里列的 baselines，其实都能看成 LRB 的特例或局部组件。

**1. DP-SGD**

标准形式是

\[
g_i^{\text{clip}}
=
g_i\cdot
\min\left(1,\frac{C}{\|g_i\|_2}\right),
\]

\[
\widetilde g
=
\frac{1}{B}\sum_{i=1}^{B}g_i^{\text{clip}}
+
\mathcal N(0,\sigma^2C^2I).
\]

它的优点是：

- 理论最规范，
- 可以做正式 \((\varepsilon,\delta)\)-DP 记账。

缺点是：

- 全层统一加噪常常太伤效果，
- 它没有显式利用“哪些层最危险”这个结构。

如果在 LRB 里取

- \(P_l=I\),
- \(Q_l=I\),
- 没有分层压缩，
- 只有裁剪和加噪，

那它就很接近 DP baseline。

**2. Top-k / 压缩**

Top-k 可以写成

\[
\operatorname{TopK}(g)_j
=
g_j \cdot \mathbf 1\{j\in \mathcal K_k(g)\},
\]

其中 \(\mathcal K_k(g)\) 是绝对值最大的 \(k\) 个坐标集合。

问题在于：

- 它只是“少传一些数”，
- 但保留下来的往往反而是最强、最有辨识度的信息。

LRB 的不同在于：

- 不是简单删坐标，
- 而是强制更新只能落在公共低维子空间里，
- 这更像“改空间结构”，而不只是“砍坐标数”。

**3. 纯噪声注入**

标准形式很简单：

\[
\widetilde g = g + \epsilon,
\qquad
\epsilon\sim\mathcal N(0,\sigma^2I).
\]

问题是：

- 如果原始梯度结构一点没变，
- DAGER 这种几何型攻击依然可能从剩余结构里恢复。

LRB 先投影再加噪，通常比“只加噪”更合理。

**4. Soteria 类表示防御**

这类方法更接近：

- 在中间表示或特征层面做保护，
- 让输入难以从表示反推回去。

LRB 的精神和它有些相通，但重点不同：

- Soteria 更偏表示层。
- LRB 更偏“上传更新本身”的 recoverability。

对你的题目来说，LRB 更贴合“中间更新信息 \(\rightarrow\) 原始数据”的主线。

---

**六、为什么我觉得 LRB 是更像“通用防御”的路线**

一句话概括：

- DAGER 利用的是“私有 token span 直接暴露在梯度空间里”。
- PEFTLeak 利用的是“低秩更新仍保留样本特异结构”。
- Partial-layer 攻击利用的是“前几层已经足够泄露”。

而 LRB 的三个核心组件恰好分别对应这三件事：

- 本地混合：削弱单次上传的样本级清晰度。
- 公共子空间投影：破坏“私有几何结构直接可见”。
- 分层裁剪与噪声：针对最敏感层重点保护。

所以它不是“给某个攻击写补丁”，而是在直接压缩“更新可恢复性”。

---

**七、如果要把它落成实验，建议怎么做**

**1. 先做无防御基线**

在与你的 [FL-LLM.md](./FL-LLM.md) 一致的统一框架下，先跑：

- DAGER / LAMP
- PEFTLeak / ReCIT
- Partial Transformer Gradients

记录：

- ROUGE-1
- ROUGE-2
- Exact Match
- task accuracy / F1 / perplexity

**2. 先做层敏感度 profiling**

对每层单独暴露，算

\[
\kappa_l.
\]

你大概率会看到：

- embedding 和最前面的 1 到 2 个 block 最危险，
- 挂在这些层上的 LoRA/Adapter 也危险。

**3. 以 LRB 为主方法做消融**

建议逐步加组件：

- `Only Clip`
- `Clip + Noise`
- `Clip + Projection`
- `Clip + Projection + Noise`
- `Local Mixing + Clip + Projection + Noise`

这样你可以看清：

- 真正拉开差距的是不是“公共子空间投影”这一步，
- 还是本地混合最有用，
- 或者两者叠加最好。

**4. 用 Pareto 曲线选点**

最终不要只给一个点，而要给一条曲线：

- 横轴：任务效果下降量
- 纵轴：综合攻击成功率 \(P(\phi)\)

最优点不是“隐私最强”，而是：

- 在效用还能接受时，
- 对三类攻击整体最好。

---

**八、这套方案的局限**

这部分也应该写清楚，不然会显得太理想化。

- 它目前是“有强直觉、有清楚实现路径”的初步方案，不是现成定理保证的最终答案。
- 公共子空间 \(P_l,Q_l\) 选得不好，会伤训练效果。
- 如果压缩过猛，模型适配能力会掉，尤其在 PEFT 小参数场景更明显。
- 如果服务器长期看到多轮同一客户端更新，跨轮信息累积仍可能带来泄露。
- 如果想要严格理论保证，最好结合正式 DP accountant；如果只做经验防御，则需要更多实验支撑。

---

**九、最精炼的结论**

DAGER 的本质是：

\[
\text{私有 token 的表示空间}
\approx
\text{共享梯度的子空间}.
\]

它先用第一层梯度找 token 集，再用第二层结构恢复顺序，所以能把文本几乎精确重建出来。更大的 LLM 往往更危险，因为 \(d\) 更大，更容易满足 \(b<d\)。

我建议你的“通用防御”不要只想着“让 DAGER 失败”，而要直接改成：

\[
\text{原始更新}
\;\xrightarrow{\text{混合+裁剪+公共子空间投影+分层噪声}}\;
\text{低可恢复更新}.
\]

也就是 LRB 这条路线。  
它最有希望同时压制：

- 全梯度恢复，
- PEFT 更新恢复，
- 局部层恢复，

并且在实验上形成一条清晰的隐私-效用 Pareto 曲线。

如果你愿意，我下一步可以直接把这一套整理成“论文方法章节”的写法，包括：

- 方法名和整体框架图
- Threat model
- Method 部分的正式写法
- Algorithm 伪代码
- 实验章节怎么写
- baseline 和 ablation 怎么安排
