# LRB 消融实验结果分析与下一步路线

> 更新时间：2026-05-03  
> 正式结果目录：`log/runs/lrb_ablation_sst2_b2_gpt2_k0.5_20260501_004737`  
> 历史半成品目录：`log/runs/lrb_ablation_sst2_b2_gpt2_k0.5_20260430_133224`

本文基于当前完整的 LRB 消融实验，对 `full_lrb` 方法、各消融变体、实验结果和下一步论文推进路线做一次集中分析。核心目的不是只复述表格，而是回答三个问题：

1. 当前 `full_lrb` 到底做了什么；
2. 每个消融 variant 分别去掉或替换了什么机制；
3. 现有结果说明下一步应该把方法和实验推进到哪里。

## 1. 数据来源与结果口径

本文的正式分析以完整消融结果为准：

```text
log/runs/lrb_ablation_sst2_b2_gpt2_k0.5_20260501_004737
```

该目录的运行设置为：

| 项目 | 设置 |
| --- | --- |
| dataset | `sst2` |
| model | `gpt2` |
| finetuned_path | `./models/gpt2-ft-rt` |
| batch_size | `2` |
| n_inputs | `100` |
| epochs | `1` |
| lrb_main_k | `0.5` |
| seeds | `101 / 202 / 303` |
| variants | `none / identity_lrb / clip_only / proj_only / proj_clip / full_lrb / pool_full / rule_only / empirical_only / uniform_all_sensitive` |
| stages | privacy + proxy utility + end-to-end utility |

用户列出的早期目录：

```text
log/runs/lrb_ablation_sst2_b2_gpt2_k0.5_20260430_133224
```

只包含 `none / identity_lrb / full_lrb` 三个 privacy 结果，且 `ablation_proxy_summary.md` 和 `ablation_utility_summary.md` 为空。因此它只能作为历史半成品说明，不能作为正式消融结论来源。

相关背景文档包括：

- `docs/FL-LLM.md`
- `docs/LRB_方法详解.md`
- `docs/LRB_ABLATION_RUNBOOK.md`
- `docs/DEFENSE_BASELINES_N100_ANALYSIS_20260502.md`
- `docs/UTILITY_RESULTS_ANALYSIS_20260426.md`
- `docs/PEFT_EVAL.md`
- `docs/LRB_实验任务清单.md`

## 2. 当前研究问题：为什么需要 LRB

当前项目关注的是联邦大模型训练中的数据恢复风险。已有实验说明，在 `SST2 + GPT2 + batch=2` 的 full-gradient DAGER 设置下，不加防御时文本恢复非常严重。本次完整消融里的 clean baseline 为：

| variant | rec_token_mean | R1+R2 | eval_accuracy |
| --- | ---: | ---: | ---: |
| `none` | `0.855333` | `148.260102` | `0.913226` |

这说明攻击者能从客户端上传的梯度中恢复大量 token。项目真正想解决的不是“让某一次 DAGER 失败”，而是降低共享更新中可被恢复成原始文本的细节信息，即降低 recoverability。

因此 LRB 的研究假设是：

> 梯度中并不是所有信息都同等危险。任务学习主要需要相对稳定的粗结构，而数据恢复攻击更依赖高分辨率、样本特异、层级敏感的细节信息。若能对这些细节方向建立 bottleneck，就可能在保持训练效用的同时降低恢复能力。

这也解释了为什么 LRB 不应被写成普通加噪方法。它的目标不是“把梯度弄乱”，而是“让共享梯度只保留足够训练用、但不利于还原样本的表示”。

## 3. `full_lrb` 当前方法如何做

当前仓库中的 `full_lrb` 是一个 post-gradient transform。它不是改模型结构，也不是改 loss，而是在客户端算出原始梯度之后、上传给服务器或交给攻击者之前，对每个可训练参数张量的梯度做一次结构化变换。

设客户端本地 batch 产生的原始梯度为：

```text
G = (G_1, G_2, ..., G_L)
```

其中 `G_l` 表示第 `l` 个可训练参数张量的梯度，例如 embedding 权重、attention 权重、MLP 权重、classifier 权重等。`full_lrb` 会把每个 `G_l` 变成一个防御后的梯度：

```text
G_l -> \tilde{G}_l
```

整体流程是：

```text
1. 收集所有可训练参数的原始梯度 G_l
2. 为每一层估计隐私敏感度 s_l
3. 根据 s_l 决定该层的 keep_ratio、clip_scale、noise_scale
4. 对该层梯度做 layer-wise clipping
5. 对裁剪后的梯度做 low-resolution signed_pool projection
6. 构造 residual-space noise
7. 输出 projected gradient + residual noise
```

可以把 `full_lrb` 理解成下面这个公式：

```text
\tilde{G}_l =
  P_l( Clip_l(G_l) ) + N_l^{residual}
```

其中：

- `Clip_l` 是这一层自己的梯度范数裁剪；
- `P_l` 是这一层自己的低分辨率投影；
- `N_l^{residual}` 是主要落在投影残差空间里的噪声；
- 每一层的裁剪强度、投影保留比例、噪声强度都由 sensitivity `s_l` 决定。

### 3.1 第一步：估计每层 sensitivity

`sensitivity` 是 `full_lrb` 的调度信号。它回答的问题是：

> 这一层梯度有多可能携带可恢复原始文本的细节？

代码中每层 sensitivity 记为 `s_l`，范围在 `[0, 1]`。数值越接近 `1`，表示越敏感；数值越接近 `0`，表示越不敏感。敏感层会被更强处理：更低分辨率、更严格裁剪、更强残差噪声。

当前实现用两类信息估计 `s_l`：

1. 结构先验 `prior_l`；
2. 当前 batch 梯度的经验分数 `empirical_l`。

最终混合为：

```text
s_l = (1 - empirical_weight) * prior_l
    + empirical_weight * empirical_l
```

本次 `full_lrb` 中 `empirical_weight=0.6`，因此 sensitivity 有 `40%` 来自结构规则，`60%` 来自当前梯度统计。

### 3.2 结构先验：按层名判断哪些层天然更敏感

结构先验来自参数名。代码会检查参数名里是否包含 embedding、attention、q/k/v、classifier、bias、LayerNorm 等模式，并给出一个初始敏感度。

大致规则如下：

| 参数类型 | 先验 sensitivity | 为什么这么设 |
| --- | ---: | --- |
| token / position embedding | `1.0` | 最接近离散 token，梯度常直接暴露词表位置和输入细节 |
| 前 `sensitive_n_layers` 层的 attention | `1.0` | 早层 attention 与 token 交互最直接，通常更容易泄露文本结构 |
| 前 `sensitive_n_layers` 层的非 attention 参数 | `0.7` | 早层仍承载大量输入局部信息，但不如 attention 直接 |
| 其他 attention 参数 | `0.45` | attention 仍与 token 依赖相关，但后层更抽象 |
| classifier / lm_head / bias / LayerNorm | `0.15` | 更偏输出或归一化统计，通常不直接决定 token 恢复 |
| 其他普通参数 | `0.25` | 默认中低敏感 |

这里的 `sensitive_n_layers` 默认是 `2`，表示前两个 transformer block 被视为更需要保护。

这一步体现的是 LRB 的基本假设：

> embedding 和早层 attention 比后层、bias、LayerNorm 更可能泄露原始输入细节，所以应该分层处理，而不是所有梯度一刀切。

### 3.3 经验校准：当前梯度本身看起来有多危险

只靠层名规则可能不够，因为同一层在不同 batch、不同任务、不同训练阶段的梯度形态可能不同。因此代码还会从当前梯度中计算经验分数。

对每个有效梯度 `G_l`，代码计算三类指标。

第一类是梯度范数：

```text
norm_metric_l = log(1 + ||G_l||_2)
```

它衡量这一层更新能量有多大。能量越大，说明这一层对当前 batch 的响应越强，可能携带更多样本信息。

第二类是低分辨率投影残差比例。代码会从梯度中抽样一部分元素，对这个样本做一次低分辨率投影，然后看原样本和投影版本差多少：

```text
residual_ratio_l =
  || sample(G_l) - P(sample(G_l)) ||_2
  / ( || sample(G_l) ||_2 + eps )
```

这个值越大，说明该层梯度中有越多内容无法被低分辨率结构表示。LRB 把这类内容视为更可能包含高频、局部、样本特异细节，因此更危险。

第三类是尖峰性：

```text
spikiness_l = log(1 + max(|G_l|) / RMS(G_l))
```

它衡量梯度是否集中在少数坐标上。尖峰很强的梯度可能暴露某些离散 token 或局部特征。

这三类指标会先在当前所有层之间做 min-max 归一化，再合成经验分数：

```text
empirical_l =
  0.45 * norm_score_l
+ 0.40 * residual_score_l
+ 0.15 * spikiness_score_l
```

这里 `0.40` 给了 residual score 很高权重，因为它最接近 LRB 的核心问题：哪些层包含低分辨率 projection 难以保留的细节。

### 3.4 第二步：由 sensitivity 生成每层参数

`full_lrb` 不是给所有层使用同一组参数，而是用 sensitivity 在“非敏感层配置”和“敏感层配置”之间插值。

插值公式是：

```text
value_l =
  value_other * (1 - s_l)
+ value_sensitive * s_l
```

对每层都会生成三个实际参数：

```text
keep_ratio_l = mix(keep_other, keep_sensitive, s_l)
clip_scale_l = mix(clip_other, clip_sensitive, s_l)
noise_scale_l = mix(noise_other, noise_sensitive, s_l)
```

本次 `full_lrb` 的两端配置是：

| 参数 | 非敏感层端点 | 敏感层端点 | sensitivity 越高会怎样 |
| --- | ---: | ---: | --- |
| `keep_ratio` | `0.75` | `0.5` | 保留分辨率更低，投影更强 |
| `clip_scale` | `1.0` | `0.5` | 裁剪阈值更低，梯度幅值压得更强 |
| `noise_scale` | `0.005` | `0.03` | 残差噪声更强 |

举例来说，如果某层 `s_l=0`，它就是低敏感层：

```text
keep_ratio_l = 0.75
clip_scale_l = 1.0
noise_scale_l = 0.005
```

如果某层 `s_l=1`，它就是高敏感层：

```text
keep_ratio_l = 0.5
clip_scale_l = 0.5
noise_scale_l = 0.03
```

如果某层 `s_l=0.6`，它会得到中间强度：

```text
keep_ratio_l = 0.75 * 0.4 + 0.5 * 0.6 = 0.60
clip_scale_l = 1.0 * 0.4 + 0.5 * 0.6 = 0.70
noise_scale_l = 0.005 * 0.4 + 0.03 * 0.6 = 0.020
```

这就是 `Layer-wise` 的真正含义：不是只把层分成“保护”和“不保护”，而是每层都有连续的防御强度。

### 3.5 第三步：layer-wise clipping

接下来 `full_lrb` 先对每层做 clipping。

代码会先计算所有有效梯度张量范数的中位数：

```text
median_norm = median( ||G_1||, ||G_2||, ..., ||G_L|| )
```

然后每层的裁剪阈值为：

```text
max_norm_l = median_norm * clip_scale_l
```

若该层梯度范数没有超过阈值，则不变；若超过阈值，则按比例缩小：

```text
G_l^{clip} =
  G_l * max_norm_l / ( ||G_l||_2 + eps )
```

这一步主要限制某层梯度的整体能量。它不会改变张量形状，也不会选择性删除某些坐标，而是把整个张量整体缩放。

直觉上，clipping 可以防止某些敏感层通过特别大的梯度幅值暴露 batch 信息。但本次消融显示，`clip_only` 几乎不能降低 DAGER 恢复，因此 clipping 在当前设置下不是主效应。

### 3.6 第四步：low-resolution signed_pool projection

裁剪后，`full_lrb` 对每层梯度做低分辨率 projection：

```text
G_l^{proj} = P_l( G_l^{clip}; keep_ratio_l )
```

这里的 `keep_ratio_l` 不是保留多少个最大元素，而是低分辨率重建比例。

对 1D 梯度，过程是：

```text
长度 n
  -> adaptive_avg_pool1d 到 k = round(n * keep_ratio_l)
  -> interpolate 回长度 n
```

对 2D 梯度矩阵，过程是：

```text
形状 m x n
  -> adaptive_avg_pool2d 到 km x kn
  -> interpolate 回 m x n
```

输出形状与原梯度完全相同，但高分辨率细节被低分辨率近似替代。

`signed_pool` 在普通 pooling 前后增加了固定随机正负号：

```text
G_l^{signed} = D_l * G_l^{clip}
pooled = PoolInterpolate(G_l^{signed})
G_l^{proj} = D_l * pooled
```

其中 `D_l` 可以理解成由 `+1/-1` 组成的固定随机符号矩阵或向量，由 `rng_seed` 和 layer index 决定。这样做的效果是：投影不完全依赖原始坐标顺序，而更像共享的随机低分辨率子空间。

这一步是目前实验中最关键的部分。`proj_only` 只保留这一步，就已经把 DAGER 恢复打到 `0`。

### 3.7 第五步：residual-space noise

`full_lrb` 最后加 residual-space noise。它不是直接做：

```text
G_l + GaussianNoise
```

而是先生成一个随机噪声张量 `Z_l`，再把这个噪声也投影到同一个低分辨率空间：

```text
Z_l^{proj} = P_l(Z_l; keep_ratio_l)
```

然后只保留噪声中无法被 projection 表示的残差部分：

```text
Z_l^{residual} = Z_l - Z_l^{proj}
```

接着把这个残差噪声缩放到目标范数：

```text
target_norm_l = ||G_l^{clip}||_2 * noise_scale_l
N_l^{residual} =
  Z_l^{residual} / ||Z_l^{residual}||_2 * target_norm_l
```

最终输出：

```text
\tilde{G}_l = G_l^{proj} + N_l^{residual}
```

这一步的设计直觉是：

> projection 保留低分辨率粗结构；residual noise 主要扰动 projection 丢弃的细节空间。这样希望保留任务需要的粗信息，同时进一步破坏攻击恢复所需的高分辨率细节。

但本次消融显示，在当前 full-gradient DAGER 设置下，`proj_only` 已经足够把恢复打到 `0`，额外 residual noise 没有带来可见 privacy 增益，反而降低了 utility。因此 residual-space noise 更适合暂时作为“强防御版本”的可选组件，而不是当前主配置的必要模块。

### 3.8 `full_lrb` 的完整伪代码

把上面的步骤合起来，`full_lrb` 可以写成：

```text
Input:
  gradients G_1...G_L
  layer names name_1...name_L
  sensitive config:
    keep=0.5, clip=0.5, noise=0.03
  other config:
    keep=0.75, clip=1.0, noise=0.005
  empirical_weight=0.6
  projection=signed_pool

Step 1: compute median_norm over all gradient tensors

Step 2: for each layer l:
  prior_l = rule_based_sensitivity(name_l)
  empirical_l = gradient_based_score(G_l)
  s_l = 0.4 * prior_l + 0.6 * empirical_l

Step 3: for each layer l:
  keep_l  = mix(0.75, 0.5, s_l)
  clip_l  = mix(1.0, 0.5, s_l)
  noise_l = mix(0.005, 0.03, s_l)

  max_norm_l = median_norm * clip_l
  G_clip = clip_by_norm(G_l, max_norm_l)

  G_proj = signed_pool_projection(G_clip, keep_l)

  Z = random_noise_like(G_l)
  Z_proj = signed_pool_projection(Z, keep_l)
  Z_res = Z - Z_proj
  N_res = normalize(Z_res) * ||G_clip|| * noise_l

  output_l = G_proj + N_res

Return:
  defended gradients output_1...output_L
```

### 3.9 full_lrb 的实验配置

本次消融中，`full_lrb` 的主要参数为：

| 参数 | 值 | 含义 |
| --- | ---: | --- |
| `defense_lrb_sensitive_n_layers` | `2` | 前两个 transformer block 在结构先验中更敏感 |
| `defense_lrb_keep_ratio_sensitive` | `0.5` | 高敏感层的低分辨率保留比例 |
| `defense_lrb_keep_ratio_other` | `0.75` | 低敏感层的低分辨率保留比例 |
| `defense_lrb_clip_scale_sensitive` | `0.5` | 高敏感层裁剪阈值更低 |
| `defense_lrb_clip_scale_other` | `1.0` | 低敏感层裁剪更松 |
| `defense_lrb_noise_sensitive` | `0.03` | 高敏感层残差噪声更强 |
| `defense_lrb_noise_other` | `0.005` | 低敏感层残差噪声更弱 |
| `defense_lrb_empirical_weight` | `0.6` | 结构先验与经验校准混合，且更偏经验校准 |
| `defense_lrb_calibration_samples` | `4096` | 每层最多抽样 4096 个元素做经验校准 |
| `defense_lrb_projection` | `signed_pool` | 默认低分辨率投影模式 |

### 3.10 如何直观理解 full_lrb

如果用一句话解释：

> `full_lrb` 对每层梯度先判断“这层有多容易泄露”，再把高风险层压到更低分辨率、裁得更紧，并在被投影丢弃的细节空间里加噪。

如果用信息流解释：

```text
原始梯度 = 任务相关粗结构 + 样本特异细节

full_lrb 希望：
  保留任务相关粗结构
  降低样本特异细节的可恢复性
```

如果用本次实验结果解释：

```text
projection 是真正打断 DAGER 的关键；
clipping 和 residual noise 在当前设置下不是必要条件；
full_lrb 是一个完整但偏重的版本；
proj_only 是当前更合适的主方法候选。
```

## 4. 消融实验如何设计

消融实验不是随机列举变体，而是在逐一回答：LRB 到底靠什么起作用？

| variant | 机制配置 | 它验证的问题 |
| --- | --- | --- |
| `none` | 不做任何防御 | clean 泄露有多严重 |
| `identity_lrb` | 走 LRB 分支，但 `keep=1`、几乎不裁剪、`noise=0` | LRB 代码路径本身是否引入偏差 |
| `clip_only` | 只做 layer-wise clipping，不投影、不加噪 | 单纯限制梯度范数是否足够防御 |
| `proj_only` | 只做 low-resolution `signed_pool` projection，不裁剪、不加噪 | projection bottleneck 是否是主效应 |
| `proj_clip` | projection + clipping，不加 residual noise | clipping 是否能在 projection 外提供额外收益 |
| `full_lrb` | projection + clipping + residual-space noise | 完整 LRB 配置是否最优 |
| `pool_full` | 完整 LRB，但把 `signed_pool` 换成 `pool` | `signed_pool` 是否优于普通坐标 pooling |
| `rule_only` | 完整 LRB，但 sensitivity 只用层名规则 | 结构先验是否足够 |
| `empirical_only` | 完整 LRB，但 sensitivity 只用当前梯度统计 | 经验校准是否足够 |
| `uniform_all_sensitive` | 所有层都按同等强度处理 | 是否真的需要 layer-wise 分配 |

下面逐个解释每个 variant。

### 4.1 `none`

`none` 是不加防御的 clean baseline。它直接把原始客户端梯度交给 DAGER 攻击。

它的作用不是提出方法，而是给出风险下界：如果没有任何防御，当前设置下到底泄露到什么程度。

### 4.2 `identity_lrb`

`identity_lrb` 走 LRB 的代码路径，但把所有实际防御关闭：

- `keep_ratio_sensitive=1.0`
- `keep_ratio_other=1.0`
- clip 阈值设到极大；
- noise 设为 `0`；
- projection 模式仍保留为 `signed_pool`，但因为 `keep>=0.999`，代码会直接近似返回原梯度。

它的意义是做 sanity check。如果 `identity_lrb` 与 `none` 不一致，就说明 LRB 代码路径、参数枚举或日志汇总可能有问题。

本次结果里 `identity_lrb` 与 `none` 完全一致，因此消融管线可信。

### 4.3 `clip_only`

`clip_only` 只保留 layer-wise clipping：

- 不做低分辨率 projection；
- 不加 residual noise；
- 敏感层 clipping 更强，非敏感层 clipping 较弱；
- `keep_ratio=1.0`，所以不会降低分辨率。

它验证的是：

> DAGER 泄露是否主要来自梯度范数过大？

如果 clipping 有效，说明只要限制每层幅值就能降低恢复。但本次结果显示 `clip_only` 基本和 clean 一样泄露严重，所以当前 DAGER 并不主要靠梯度幅值异常来恢复文本。

### 4.4 `proj_only`

`proj_only` 只保留 low-resolution `signed_pool` projection：

- 敏感层目标 `keep_ratio=0.5`；
- 非敏感层目标 `keep_ratio=0.75`；
- 实际每层 `keep_ratio` 根据 sensitivity 在二者之间插值；
- 不做 clipping；
- 不加 residual noise。

它验证的是：

> 只限制梯度分辨率，是否已经足以打断 DAGER 的恢复路径？

本次结果说明答案是肯定的。在当前设置下，`proj_only` 已经把 DAGER 恢复打到 `0`，同时保持了最好的 utility。

### 4.5 `proj_clip`

`proj_clip` 在 `proj_only` 基础上加入 layer-wise clipping，但仍不加 residual noise。

它验证的是：

> projection 已经有效时，clipping 是否还能进一步提升 privacy 或 utility？

结果显示 `proj_clip` privacy 同样为 `0`，但 utility 不优于 `proj_only`。这说明当前 full-gradient DAGER 设置下，clipping 不是必要组件。

### 4.6 `full_lrb`

`full_lrb` 是当前文档和脚本中的完整 LRB 主配置：

- layer-wise sensitivity；
- layer-wise clipping；
- low-resolution `signed_pool` projection；
- residual-space noise。

它验证的是完整启发式方案是否优于各个简化组件。

本次结果显示，`full_lrb` 的 privacy 很强，但 utility 明显弱于 `proj_only` 和 `proj_clip`。因此在当前设置下，`full_lrb` 更像一个过防御配置，而不是最优 operating point。

### 4.7 `pool_full`

`pool_full` 与 `full_lrb` 几乎相同，唯一主要区别是：

```text
defense_lrb_projection = pool
```

也就是用普通坐标系 pooling 替代默认的 `signed_pool`。

它验证的是：

> `signed_pool` 这种随机符号低分辨率投影，是否比普通 pooling 更合理？

结果显示 `pool_full` utility 明显更差，支持当前默认使用 `signed_pool`。

### 4.8 `rule_only`

`rule_only` 使用完整 LRB 的 projection、clipping、noise，但 sensitivity 只来自层名规则：

```text
empirical_weight = 0
```

这意味着 embedding、早层 attention 等会被规则判定为高敏感，而不参考当前 batch 的实际梯度统计。

它验证的是：

> 只靠结构先验能否做出好的层级防御分配？

结果显示 `rule_only` utility 较差，说明纯规则先验可能保护过重，尤其可能把任务学习需要的有效信号也压掉。

### 4.9 `empirical_only`

`empirical_only` 使用完整 LRB 的 projection、clipping、noise，但 sensitivity 只来自当前梯度统计：

```text
empirical_weight = 1
```

它验证的是：

> 不依赖层名规则，只看当前梯度范数、投影残差和尖峰性，是否足够？

结果显示 `empirical_only` 明显好于 `rule_only` 和 `full_lrb`，但仍不如 `proj_only`。这说明经验校准在当前设置下比纯规则更稳，但完整配置里的 clipping/noise 仍然带来额外 utility 代价。

### 4.10 `uniform_all_sensitive`

`uniform_all_sensitive` 把所有层都当作同样敏感：

- 所有层 `keep_ratio=0.5`；
- 所有层 clip scale 相同；
- 所有层 noise scale 相同；
- 不再区分敏感层和非敏感层。

它验证的是：

> 是否需要 layer-wise 分配，还是所有层一刀切强防御就可以？

结果显示该变体 utility 最差，说明所有层同等强度处理会严重伤害训练效用。LRB 的 layer-wise 思想仍然必要，只是当前 full 配置的强度需要优化。

## 5. 实验结果总览

完整结果如下：

| variant | DAGER rec_token | R1+R2 | eval_accuracy | utility_drop | grad_cosine | norm_retention | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `none` | `0.855333` | `148.260102` | `0.913226` | `0.000000` | `1.000000` | `1.000000` | clean 泄露严重 |
| `identity_lrb` | `0.855333` | `148.260102` | `0.913226` | `0.000000` | `1.000000` | `1.000000` | 与 none 完全一致，代码路径可信 |
| `clip_only` | `0.854076` | `147.955623` | `0.918196` | `-0.004970` | `0.548411` | `0.191669` | utility 好，但 privacy 失败 |
| `proj_only` | `0.000000` | `0.000000` | `0.915520` | `-0.002294` | `0.520456` | `0.409748` | 当前最佳消融点 |
| `proj_clip` | `0.000000` | `0.000000` | `0.913226` | `0.000000` | `0.289518` | `0.092119` | privacy 成功，但不优于 proj_only |
| `full_lrb` | `0.000000` | `0.000000` | `0.892584` | `0.020642` | `0.289385` | `0.092158` | privacy 成功，但当前过防御 |
| `pool_full` | `0.000000` | `0.000000` | `0.870031` | `0.043195` | `0.255474` | `0.092119` | 普通 pool 弱于 signed_pool |
| `rule_only` | `0.000000` | `0.000000` | `0.868884` | `0.044342` | `0.284244` | `0.094546` | 纯规则 sensitivity 过重 |
| `empirical_only` | `0.000000` | `0.000000` | `0.902523` | `0.010703` | `0.286941` | `0.092449` | 好于 rule/full，但仍不如 proj_only |
| `uniform_all_sensitive` | `0.000000` | `0.000000` | `0.842890` | `0.070336` | `0.275230` | `0.059342` | 一刀切最伤 utility |

## 6. 机制性结论

### 6.1 `identity_lrb` 证明消融管线可信

`identity_lrb` 与 `none` 的 privacy 和 utility 完全一致：

| variant | rec_token_mean | R1+R2 | eval_accuracy |
| --- | ---: | ---: | ---: |
| `none` | `0.855333` | `148.260102` | `0.913226` |
| `identity_lrb` | `0.855333` | `148.260102` | `0.913226` |

这说明 LRB 分支本身没有因为代码路径、参数读取、日志汇总等问题引入额外差异。后续消融结果可以解释为机制差异，而不是管线偏差。

### 6.2 clipping 不是当前主效应

`clip_only` 的 `rec_token_mean=0.854076`，几乎等于 `none` 的 `0.855333`。虽然它的 `eval_accuracy=0.918196` 较高，但它没有防住 DAGER。

这说明：

> 当前 DAGER 恢复并不主要依赖梯度范数大小；只压缩每层梯度幅值不足以破坏恢复路径。

因此论文里不能把 LRB 的有效性解释成“裁剪梯度就够了”。裁剪最多是辅助组件，不是核心创新点。

### 6.3 projection bottleneck 是当前核心有效组件

`proj_only` 的结果非常关键：

| variant | rec_token_mean | R1+R2 | eval_accuracy | utility_drop |
| --- | ---: | ---: | ---: | ---: |
| `proj_only` | `0.000000` | `0.000000` | `0.915520` | `-0.002294` |

它没有 clipping，也没有 residual noise，只做低分辨率 `signed_pool` projection，却已经将 DAGER 恢复打到 `0`。

这给出当前最重要的机制性结论：

> Projection bottleneck is the main effective component.

更具体地说，DAGER 的恢复路径似乎强依赖原始梯度中的高分辨率细节；一旦这些细节被低分辨率公共子空间替代，即使不额外加噪，攻击也难以继续恢复 token。

### 6.4 full_lrb 当前过防御

`full_lrb` 同样能把 DAGER 打到 `0`，但 accuracy 降到 `0.892584`，utility drop 为 `0.020642`。与 `proj_only` 对比：

| variant | rec_token_mean | eval_accuracy | utility_drop |
| --- | ---: | ---: | ---: |
| `proj_only` | `0.000000` | `0.915520` | `-0.002294` |
| `full_lrb` | `0.000000` | `0.892584` | `0.020642` |

既然 `proj_only` 已经 privacy 成功，`full_lrb` 额外加入 clipping 和 residual noise 并没有带来可见 privacy 增益，反而伤害 utility。

因此当前不应继续把 `full_lrb@0.5` 写成最优主配置。更稳的表述是：

> `full_lrb` 证明完整 recoverability bottleneck 框架可行，但在当前 full-gradient DAGER 设置下配置过重；更合适的主方法候选是简化的 projection-only LRB。

### 6.5 signed_pool 比普通 pool 更合理

`pool_full` 与 `full_lrb` 只在 projection 模式上不同，但 `pool_full` accuracy 只有 `0.870031`，比 `full_lrb` 的 `0.892584` 更低。

这支持当前方法文档中关于 `signed_pool` 的判断：

> 直接在原始坐标系上 pooling 会过度依赖坐标顺序；signed random pooling 更接近公共随机低分辨率子空间，能更好保留任务相关结构。

### 6.6 layer-wise 分配仍然必要

`uniform_all_sensitive` 的 accuracy 为 `0.842890`，是所有 privacy 成功变体中最差之一。这说明所有层用同等强度处理并不可取。

因此需要区分两个结论：

- 当前 `full_lrb` 的配置过重；
- 但 layer-wise sensitivity 的思想并没有被否定。

更准确的判断是：

> LRB 需要 layer-wise allocation，但当前的规则先验、经验校准和 clipping/noise 强度还不是最优组合。

### 6.7 rule-only 与 empirical-only 的启示

`rule_only` 的 accuracy 为 `0.868884`，`empirical_only` 为 `0.902523`。这说明纯结构规则比纯经验校准更容易过防御。

但 `empirical_only` 仍不如 `proj_only`，原因不是 sensitivity 一定无用，而是该变体仍包含 clipping 和 residual noise。为了进一步判断 sensitivity 在 projection-only 路线下是否必要，下一步需要补：

```text
proj_rule_only
proj_empirical_only
proj_uniform
proj_no_empirical
```

当前实验还不能直接回答 projection-only 中最优 sensitivity 设计是什么。

## 7. 与现有 baseline 结论的关系

此前 `DEFENSE_BASELINES_N100_ANALYSIS_20260502.md` 和 `UTILITY_RESULTS_ANALYSIS_20260426.md` 的核心判断是：

- `topk@0.1` 与 `compression@8` 是 full-gradient DAGER 下很强的经验 baseline；
- `full_lrb@0.5` 明显好于 `lrb@0.2/0.35`，但 utility 仍弱于 topk/compression；
- LRB 要成为主方法，需要靠机制消融和跨攻击面泛化证明结构性价值。

本次消融使这个判断发生了一个重要更新：

> 如果只看 `full_lrb@0.5`，LRB 确实弱于 topk/compression；但 `proj_only@0.5` 已经表现出与 strong baselines 同量级甚至更好的 full-gradient DAGER tradeoff。

当前需要重新整理一张主结果表，把下面方法放在同一口径下比较：

| 方法 | 当前作用 |
| --- | --- |
| `none` | clean 泄露基线 |
| `topk@0.1` | 当前最强经验压缩 baseline 之一 |
| `compression@8` | 当前最强经验量化 baseline 之一 |
| `full_lrb@0.5` | 完整 LRB，但当前过重 |
| `proj_only@0.5` | 当前最强 LRB 变体 |
| `proj_clip@0.5` | projection + clipping 对照 |

如果 `proj_only` 在统一主表中仍保持 `DAGER=0` 且 utility 不低于 topk/compression，那么论文叙事就可以从：

> LRB 很强，但 full-gradient DAGER 上 tradeoff 还不如 topk/compression。

更新为：

> LRB 的完整初版配置过重，但消融发现 projection bottleneck 是关键有效组件；简化后的 Projection-LRB 在 full-gradient DAGER 上达到强隐私且几乎无 utility 损失。

## 8. 论文推进逻辑

结合组会图中给出的五阶段问题，当前结果可以组织成下面的论文逻辑。

### 阶段 1：这件事凭什么值得做

任务场景是联邦大模型或 PEFT 微调中共享梯度/更新，而不是共享原始数据。实验上，`none` 的 `rec_token_mean=0.855333`、`R1+R2=148.260102`，说明 clean FedSGD 更新在 DAGER 下存在严重文本恢复风险。

可写成：

> In federated LLM fine-tuning, shared gradients expose high-fidelity textual information under exact gradient inversion attacks.

### 阶段 2：能力卡在哪个具体环节

现有防御的缺陷不是单纯“没有加噪”，而是缺少对梯度 recoverability 结构的控制。普通 clipping 无法防住 DAGER；`clip_only` 的 `rec_token_mean=0.854076`，几乎等于 clean。

可写成：

> Merely bounding gradient magnitudes does not remove the reconstructive structure exploited by DAGER.

### 阶段 3：别人为什么绕不开，你凭什么绕开

topk/compression 虽然在当前 DAGER 下很强，但它们是通信压缩方法，不显式建模 recoverability。LRB 的差异是把共享更新看成“可恢复性受限的表示”，通过低分辨率公共子空间打断高分辨率细节。

本次消融给出关键证据：`proj_only` 不加噪、不裁剪，已经 DAGER=0。

可写成：

> The dominant leakage channel is not gradient magnitude but high-resolution reconstructive detail; a low-resolution projection bottleneck is sufficient to suppress exact recovery in the current setting.

### 阶段 4：新机制如何落地

当前可落地的模块是 Projection-LRB：

```text
layer-wise sensitivity
  -> low-resolution signed_pool projection
  -> defended gradient
```

完整 `full_lrb` 可以作为扩展版本或强防御版本，但当前主配置应优先考虑 `proj_only@0.5`。

推荐命名候选：

| 名称 | 含义 |
| --- | --- |
| `Projection-LRB` | 突出 projection bottleneck 是核心 |
| `LRB-Lite` | 突出比 full_lrb 更简洁、更高效 |
| `LRB-Proj` | 与现有消融名称对齐 |

当前最稳的是 `Projection-LRB`，因为它最能直接表达机制贡献。

### 阶段 5：最终交付什么

当前阶段可以交付三类内容：

1. 一个新方法候选：`Projection-LRB`；
2. 一组量化结果：`rec_token_mean=0`、`R1+R2=0`、`eval_accuracy=0.915520`；
3. 一条机制性结论：低分辨率 projection bottleneck 是当前 LRB 有效性的主因。

适合写进阶段总结的一句话：

> The ablation shows that the main effective component of LRB is the low-resolution projection bottleneck rather than layer-wise clipping or residual noise. In the current SST2/GPT2 full-gradient DAGER setting, `proj_only@0.5` reduces token recovery to zero while preserving clean-level utility, whereas `full_lrb@0.5` is over-defensive.

## 9. 下一步应该做什么

### P0：把主方法候选切换到 `proj_only@0.5`

短期内不要继续围绕 `full_lrb` 调 residual noise。当前 evidence 已经说明完整配置过重，应把论文主方法候选调整为：

```text
Projection-LRB = layer-wise low-resolution signed_pool projection
```

`full_lrb` 保留为强防御/过防御对照，用于说明并非所有组件都必要。

### P0：补 `proj_only` keep-ratio sweep

当前只验证了 `lrb_main_k=0.5`。下一步应检查 projection 可以放松到什么程度：

```text
proj_only keep_ratio_sensitive = 0.5 / 0.65 / 0.75 / 0.9
```

目标不是继续把 privacy 压得更低，而是找到：

> DAGER 仍为 0 时 utility 最高的最宽松 bottleneck。

推荐输出表：

| keep_ratio_sensitive | rec_token_mean | R1+R2 | eval_accuracy | utility_drop | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `0.5` | 当前已有 | 当前已有 | 当前已有 | 当前已有 | 当前默认 |
| `0.65` | 待补 | 待补 | 待补 | 待补 | 检查是否仍安全 |
| `0.75` | 待补 | 待补 | 待补 | 待补 | 检查 utility 上限 |
| `0.9` | 待补 | 待补 | 待补 | 待补 | 检查泄露边界 |

### P0：补 projection-only 细消融

当前 `rule_only / empirical_only / uniform_all_sensitive` 都带有 full_lrb 的 clipping/noise，因此不能单独说明 projection 路线下 sensitivity 的最佳设计。

建议新增：

| 新 variant | 目的 |
| --- | --- |
| `proj_rule_only` | projection-only 下只用结构规则 |
| `proj_empirical_only` | projection-only 下只用经验校准 |
| `proj_uniform` | projection-only 下所有层同等 keep ratio |
| `proj_no_empirical` | 检查是否可以去掉校准开销 |

这组实验将回答：

> Projection-LRB 是否真的需要复杂 layer-wise sensitivity，还是固定 projection 已经足够？

### P1：重做主结果表和 Pareto 表

需要把 `proj_only` 纳入主结果，而不是只比较 `full_lrb`。

建议主表包含：

```text
none
topk@0.1
compression@8
full_lrb@0.5
proj_only@0.5
proj_clip@0.5
```

指标至少包括：

- `rec_token_mean`
- `R1+R2`
- `eval_accuracy`
- `eval_macro_f1`
- `utility_drop`
- `train_time_seconds`
- `attack_time_seconds`

这张表用于回答：

> 在同等 DAGER privacy 下，Projection-LRB 是否能与 strong compression baselines 公平竞争？

### P1：跑 LoRA/PEFT 对照

不能直接假设 `proj_only` 在 LoRA/PEFT 下仍然最好。LoRA 的共享更新只来自少量 adapter 参数，梯度形态与 full fine-tuning 不同：

- GPT-2 LoRA 当前 target module 是 `c_attn`；
- Llama LoRA 当前 target module 是 `q_proj`；
- PEFT 框架当前是 eval-first，支持 `none / noise / topk / compression / lrb`；
- 现有 `peft_baselines.sh` 默认只扫 LRB 的 `keep_ratio`，还不能直接区分 `proj_only / proj_clip / full_lrb`。

最小 PEFT 实验组合应为：

```text
none
proj_only@0.5
proj_clip@0.5
full_lrb@0.5
topk@0.1
compression@8
```

这里要特别检验两种可能：

1. `proj_only` 仍然最好：说明 projection bottleneck 对 LoRA 更新也足够；
2. `full_lrb` 更好：说明 LoRA 的低秩更新更容易被攻击利用，额外 clipping/noise 在 PEFT 下有必要。

因此文档和论文中应避免过度宣称：

> `proj_only` 是当前 full-gradient DAGER 设置下最好的 LRB 变体，但不是所有攻击面下必然最优。

### P2：做 partial-gradient / layer-level leakage

当前 DAGER 是 full-gradient attack。论文若要证明 LRB 的通用价值，需要补 partial-gradient 或 layer-level leakage。

建议最小攻击面：

- first block only；
- first 2 blocks；
- last 2 blocks；
- attention q/k/v only；
- LoRA params only。

对比方法：

```text
none
topk@0.1
compression@8
proj_only@0.5
full_lrb@0.5
```

目标是回答：

> 压缩类 baseline 在 partial-gradient 下是否仍然强？Projection-LRB 是否更稳定？

### P2：跨数据集和 backbone

当前结果只在 `SST2 + GPT2 + batch=2` 上成立。建议补最小泛化组合：

```text
cola + gpt2
rte + gpt2
sst2 + bert-base-uncased
```

每组至少跑：

```text
none
topk@0.1
compression@8
proj_only@0.5
full_lrb@0.5
```

### P2：runtime 优化

`proj_only` 的训练时间约 `5134s`，仍明显高于 `none` 的 `1202s`。虽然它比 `full_lrb` 更好，但如果要写成实用方法，仍需分析开销。

优先排查：

- sensitivity calibration 是否每步重复计算；
- `signed_pool` 的随机符号是否可以缓存；
- projection shape plan 是否可以预计算；
- `defense_lrb_calibration_samples` 是否可以降低；
- projection-only 是否可以去掉 empirical calibration。

如果 `proj_no_empirical` 能保持结果，就可以把方法进一步简化为低开销版本。

## 10. 建议图表与论文材料

结合参考图中的论文图表清单，当前最应准备以下材料。

### 10.1 Main Results 对比表

位置：Experiments - Main Results

功能：证明 Projection-LRB 与 strongest baselines 公平对比。

必须包含：

- 至少 `none / topk / compression / full_lrb / proj_only`；
- privacy 指标：`rec_token_mean`、`R1+R2`；
- utility 指标：accuracy / macro-F1 / loss；
- runtime 指标：train time / attack time。

### 10.2 Ablation Study 表

位置：Experiments - Ablation

功能：证明每个模块的作用，并给出机制结论。

当前已有表可以支持：

- `identity_lrb` 证明代码路径可信；
- `clip_only` 证明 clipping 不是主效应；
- `proj_only` 证明 projection 是主效应；
- `full_lrb` 证明完整配置过重；
- `pool_full` 支持 `signed_pool`；
- `uniform_all_sensitive` 支持 layer-wise allocation。

### 10.3 参数敏感性曲线

位置：Experiments - Analysis

优先画：

```text
X轴：proj_only keep_ratio_sensitive
Y轴1：rec_token_mean / R1+R2
Y轴2：eval_accuracy / utility_drop
```

目标是证明 Projection-LRB 不依赖单一偶然参数点。

### 10.4 Framework Overview 图

位置：Method

建议画成：

```text
Client gradient
  -> layer-wise sensitivity estimator
  -> low-resolution signed public subspace
  -> shared defended update
  -> server aggregation / attacker reconstruction
```

如果最终主方法改为 Projection-LRB，图中不要把 residual noise 画成核心模块，可以放在 optional full variant 中。

### 10.5 复杂度分析图

位置：Experiments - Analysis

需要比较：

- `none`
- `topk@0.1`
- `compression@8`
- `full_lrb@0.5`
- `proj_only@0.5`

指标：

- train time；
- attack time；
- proxy step runtime；
- accuracy。

目标是证明 `proj_only` 不仅效果好，也比 full_lrb 更轻。

## 11. 现阶段最稳的结论表述

### 中文版本

在当前 `SST2 + GPT2 + batch=2` 的 full-gradient DAGER 设置下，clean 梯度存在严重文本恢复风险。完整 LRB 可以把 DAGER 恢复打到 0，但其 clipping 和 residual noise 组合带来明显 utility 代价。消融结果显示，真正起主效应的是低分辨率 `signed_pool` projection：`proj_only@0.5` 在不加噪、不裁剪的情况下将 `rec_token_mean` 和 `R1+R2` 都降为 0，同时保持 clean-level accuracy。因此，下一阶段应把主方法候选从 `full_lrb@0.5` 调整为 `Projection-LRB / LRB-lite`，并通过 keep-ratio sweep、LoRA/PEFT、partial-gradient 和跨数据集实验验证其泛化性。

### English version

In the current SST2/GPT2 full-gradient DAGER setting, the ablation shows that the dominant effective component of LRB is the low-resolution signed projection bottleneck. While the full LRB configuration suppresses token recovery to zero, its clipping and residual-space noise introduce unnecessary utility loss. The projection-only variant achieves zero token recovery and zero ROUGE recovery while preserving clean-level accuracy. This suggests that the next method candidate should be a simplified Projection-LRB, with further validation under LoRA/PEFT, partial-gradient leakage, and cross-dataset/backbone settings.

## 12. 当前不应过度声称的内容

为了避免论文叙事写过头，当前不建议写：

1. `full_lrb` 已经是最优方法；
2. LRB 在所有场景下都超过 topk/compression；
3. `proj_only` 在 LoRA/PEFT 下也必然最好；
4. residual noise 已被证明必要；
5. 当前 sensitivity 设计已经最优。

更稳的说法是：

1. LRB 方向成立；
2. projection bottleneck 是当前 full-gradient DAGER 下的主效应；
3. full_lrb 当前过防御；
4. Projection-LRB 是新的主方法候选；
5. LoRA/PEFT 与 partial-gradient 是下一阶段最关键验证。

## 13. 代码入口与运行命令

本轮代码推进后，LRB 不再需要在脚本里手写一长串底层参数。统一入口是：

```bash
--defense lrb \
--defense_lrb_preset proj_only \
--defense_lrb_keep_ratio_sensitive 0.5
```

其中 `--defense_lrb_preset custom` 是默认值，表示完全保留旧行为：用户直接指定 `keep_ratio / clip_scale / noise / empirical_weight / projection` 等底层参数，代码不会自动覆盖。只要 `--defense lrb` 且 preset 不是 `custom`，preset 会覆盖对应 LRB 参数；`--defense_lrb_keep_ratio_sensitive` 作为主 keep ratio，也就是文档里记作 `k` 的低分辨率保留比例。

### 13.1 已支持的 preset

当前入口支持：

```text
identity_lrb
clip_only
proj_only
proj_clip
full_lrb
pool_full
rule_only
empirical_only
uniform_all_sensitive
proj_rule_only
proj_empirical_only
proj_uniform
proj_no_empirical
```

推荐把 `proj_only@0.5` 作为下一轮主候选，但 PEFT/LoRA 下必须同时跑 `proj_clip` 和 `full_lrb`，不能直接假设 projection-only 仍然最好。

### 13.2 P0：proj_only keep-ratio sweep

目标是验证 `proj_only` 不是只在 `k=0.5` 单点偶然有效，并找 privacy-utility Pareto 边界。

```bash
bash scripts/lrb_ablation.sh \
  --mode all \
  --variants proj_only \
  --lrb_main_k 0.5 \
  --n_inputs 100 \
  --skip_existing
```

建议分别跑：

```text
k = 0.5 / 0.65 / 0.75 / 0.9
```

如果算力紧张，先用 `--mode privacy,train` 跑主指标；proxy 可以随后补。

### 13.3 P0：projection-only 细消融

目标是拆开 projection-only 中的 sensitivity 分配来源：结构规则、经验校准、全层统一 keep ratio。

```bash
bash scripts/lrb_ablation.sh \
  --mode all \
  --variants proj_only,proj_rule_only,proj_empirical_only,proj_uniform,proj_no_empirical \
  --lrb_main_k 0.5 \
  --n_inputs 100 \
  --skip_existing
```

如果 `proj_no_empirical` 或 `proj_rule_only` 接近 `proj_only`，说明 empirical calibration 可以弱化或去掉，方法会更简单、运行也更轻。如果 `proj_uniform` 明显变差，则继续保留 layer-wise sensitivity allocation。

### 13.4 P1：LoRA/PEFT 对照

目标是回答关键边界问题：`proj_only` 在 full-gradient DAGER 下最好，不代表 LoRA/PEFT 下也必然最好。LoRA 梯度空间更低秩、更结构化，`full_lrb` 的 clipping 或 residual-space noise 有可能重新变得有价值。

第一步先准备真实的 LoRA checkpoint。不要直接使用 `path/to/lora_checkpoint.pt` 这类占位符；当前 `attack.py` 会在参数解析阶段检查文件是否存在，路径不存在会直接失败。GPT-2 + SST2 推荐先用：

```bash
python train.py \
  --dataset sst2 \
  --task seq_class \
  --model_path gpt2 \
  --train_method lora \
  --lora_r 16 \
  --batch_size 2 \
  --num_epochs 1 \
  --models_cache ./models_cache \
  --output_dir ./models/gpt2_sst2_lora_r16
```

训练结束后使用最终保存的 `.pt` 文件：

```bash
export LORA_CKPT=./models/gpt2_sst2_lora_r16/final.pt
```

这里的 `--lora_r` 必须和训练时一致。上面训练用的是 `16`，下面评测也统一用 `16`。正式长跑前先做一次 smoke test：

```bash
bash scripts/peft_eval.sh sst2 2 gpt2 1 \
  --finetuned_path "$LORA_CKPT" \
  --lora_r 16 \
  --defense none
```

smoke test 通过后，再跑 LoRA/PEFT 的 LRB variant 对照：

```bash
bash scripts/peft_baselines.sh sst2 2 gpt2 100 \
  --finetuned_path "$LORA_CKPT" \
  --lora_r 16 \
  --baseline_defense lrb \
  --lrb_variants proj_only,proj_clip,full_lrb \
  --lrb_main_k 0.5
```

为了和强 baseline 公平比较，还需要分别跑：

```bash
bash scripts/peft_baselines.sh sst2 2 gpt2 100 \
  --finetuned_path "$LORA_CKPT" \
  --lora_r 16 \
  --baseline_defense topk \
  --baseline_param 0.1

bash scripts/peft_baselines.sh sst2 2 gpt2 100 \
  --finetuned_path "$LORA_CKPT" \
  --lora_r 16 \
  --baseline_defense compression \
  --baseline_param 8
```

这里的 PEFT/LoRA 仍然是 eval-first 攻击评估入口，不等价于“训练期 LoRA defense 已经完整支持所有 variant”。当前代码目标是先把 `none / proj_only / proj_clip / full_lrb / topk@0.1 / compression@8` 的 LoRA 对照跑通。完整的 LoRA 使用细节以 `docs/PEFT_EVAL.md` 为准。

### 13.5 Dry-run 与回归检查命令

正式跑长实验前，建议先做 dry-run：

```bash
bash scripts/lrb_ablation.sh \
  --mode privacy \
  --n_inputs 1 \
  --variants proj_only,proj_rule_only,proj_empirical_only \
  --dry_run
```

同时检查旧入口仍可用：

```bash
bash scripts/lrb_ablation.sh --variants full_lrb --dry_run
```

预期 dry-run 命令里应出现 `--defense_lrb_preset <variant>` 和 `--defense_lrb_keep_ratio_sensitive <k>`。旧的直接底层参数命令仍可以通过 `--defense_lrb_preset custom` 或不传 preset 保持兼容。
