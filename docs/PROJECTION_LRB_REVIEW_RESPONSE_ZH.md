# Projection-LRB 机制与自适应分配审稿回复

## Reviewer Concern

审稿人指出，Hybrid Layer Sensitivity 包含多个人工权重和结构先验，但 standard DAGER 与 oracle 消融均未显示其相对 uniform allocation 的可辨识优势。因此，现有证据不足以支持 adaptive keep-ratio allocation 是 Projection-LRB 的必要组件或独立创新。同时，原稿没有清楚解释低分辨率 reconstruction 为什么会影响 DAGER 的恢复过程。

## Response

感谢审稿人指出这一问题。我们同意，原稿对 hybrid ratio allocator 的方法学地位表述过强。修订稿将 Projection-LRB 的核心重新定义为 signed low-resolution reconstruction operator，并将结构先验、经验统计及其融合权重移至附录，作为一种可选的 adaptive-ratio instantiation。Algorithm 1 现在只接收外部给定的逐层 ratios 并执行 reconstruction；最简单的 uniform schedule 与 adaptive schedule 使用相同算子。现有 standard DAGER 消融中，uniform、rule-only、empirical-only 和 hybrid schedules 均达到零恢复；在 knowledge-aware oracle 设置中，各 schedule 的 R1+R2 同样约为 101--103。因此，我们不再声称 hybrid allocation 优于 uniform、不可替代或构成核心贡献。

修订稿进一步增加算子层面的机制解释。对固定 ratio 和 signs，一维 reconstruction 可写为 \(T_r=D U_rP_rD\)，其中 pooling 将长度 \(n\) 降到 \(q=\operatorname{round}(rn)\)，因此 \(\operatorname{rank}(T_r)\le q\)。Interpolation 恢复输出形状，但不恢复 pooling 删除的自由度。对线性层梯度 \(G_l=\sum_i\delta_{l,i}h_{l,i}^{\top}\)，二维变换可写为 \(T(G_l)=L_lG_lR_l^\top=\sum_i(L_l\delta_{l,i})(R_lh_{l,i})^\top\)。Standard DAGER 仍使用未变换的候选表示 \(h(v,p)\) 做 span test，因此面临 transformed-span mismatch；oracle attacker 对候选应用相同变换后能够恢复显著文本，这与修订稿的 knowledge-aware 结果一致。该分析用于解释当前攻击行为，不构成形式化隐私或不可逆信息删除证明。

为检验 reconstruction resolution 是否在 decoder mismatch 之外仍影响 defense-aware recoverability，我们将补充 uniform-ratio oracle sweep：在 SST-2 official validation、GPT-2、batch size 2、seeds `101/202/303` 下比较 \(r=0.5/0.65/0.75/0.9\)，并加入同协议 undefended anchor 与 unsigned pooling@0.5 控制。实验将报告 Candidate recall、Top-B、L1、L2、R1+R2、sample SD 和 paired bootstrap 95% CI。最终回复与正文结论必须在服务器日志通过准入后再定稿：只有 oracle recovery 随分辨率降低而一致下降时，才将剩余抑制归因于低分辨率瓶颈；否则仅保留 transformed-span mismatch 的解释。

## Revised Contribution Statement

> Projection-LRB is a signed low-resolution reconstruction operator that changes the intrinsic resolution and span coordinates of shared gradients while preserving their external tensor interface. Layer-wise ratio allocation is an optional instantiation; our current evidence does not establish an advantage over a uniform schedule.

## Evidence Boundary

- `DAGER=0` 只表示 standard attack 在当前协议下未恢复文本。
- Signed modulation 不额外降低 pooling operator 的 algebraic rank；它改变被保留 subspace 相对原坐标的方向。
- 当前 interpolation operator 不是正交投影，不能声称严格幂等、正交或保证保留优化方向。
- Exact oracle recovery 表明 token 信息并未被不可逆删除。
- Projection-LRB 不提供差分隐私或未知 adaptive attacks 下的普遍安全保证。
