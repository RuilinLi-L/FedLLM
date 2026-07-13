# GPT-2 PEFTLeak v2 Privacy Runbook

本手册运行 `attack_peftleak_new.py` 的 GPT-2 Adapter ratio 路径。实验定位是 **PEFTLeak-style malicious embedding-probe gradient-ratio attack**，不是原始图像 PEFTLeak 的完整文本复现。

## 1. 实验矩阵

旧 Adapter utility 结果可以复用。按三种子平均 accuracy 筛选以下效用相近的 privacy 点：

| 配置 | Accuracy | 角色 |
|---|---:|---|
| `none` | 0.9167 | 无防御攻击上界 |
| `noise@1e-3` | 0.9128 | 近 clean coverage baseline |
| `topk@0.1` | 0.9079 | 强稀疏化基线 |
| `compression@6` | 0.9098 | 与 `proj_only@0.5` 效用匹配 |
| `proj_only@0.5` | 0.9098 | 强隐私主方法点 |
| `proj_only@0.65` | 0.9144 | 最佳 utility 主方法点 |
| `proj_only@0.75` | 0.9113 | Projection 曲线 |
| `proj_only@0.9` | 0.9132 | Projection 曲线 |

正式矩阵为 8 个配置乘 `101/202/303` 三个攻击种子，共 24 个任务。暂不正式跑 `full_lrb@0.5` 和 DP-SGD，因为已有 utility 明显更低。

所有 privacy 配置必须使用同一个 clean Adapter checkpoint：

```text
./outputs/peftleak_text_sst2/models/gpt2_adapter_clean/final_adapter
```

不要改用各防御训练出的 utility checkpoint，否则会同时改变模型和上传防御，无法归因 privacy 差异。

## 2. 环境准备

在服务器仓库根目录运行：

```bash
cd /data/lrl/FedLLM
conda activate fedllm-peftleak
tmux new -s peftleak_text_v2

export DEVICE=cuda:7
export CHECKPOINT=./outputs/peftleak_text_sst2/models/gpt2_adapter_clean/final_adapter
export CACHE_DIR=./models_cache
export RUN_ID=$(date +%Y%m%d_%H%M%S)
export RUN_ROOT=./log/peftleak_text_sst2_v2/$RUN_ID
```

`DEVICE` 按服务器空闲 GPU 调整。`RUN_ROOT` 必须保留在同一个 shell/tmux session 中，smoke、pilot 和 formal 的门控依赖同一目录。

当前 `scripts/peftleak_eval.sh` 仍属于旧攻击入口。本轮只使用：

```text
scripts/peftleak_text_v2_privacy.sh
```

## 3. 分阶段运行

首先检查 CUDA、PEFT/AdapterHub 环境、checkpoint 完整性，并运行 10 项 v2 语义测试：

```bash
bash scripts/peftleak_text_v2_privacy.sh preflight
```

按顺序运行。runner 会在每阶段结束时自动验收；前一阶段不合格时不会启动下一阶段。

```bash
# 2 个配置、n_inputs=2
bash scripts/peftleak_text_v2_privacy.sh smoke

# 8 个配置、seed=101、n_inputs=10
bash scripts/peftleak_text_v2_privacy.sh pilot

# 8 个配置、3 seeds、n_inputs=100，共 24 个任务
bash scripts/peftleak_text_v2_privacy.sh formal
```

也可以一次执行完整流程：

```bash
bash scripts/peftleak_text_v2_privacy.sh all
```

已完成且通过验收的日志会自动跳过。遇到不完整日志时，runner 默认拒绝覆盖；优先更换 `RUN_ROOT`，仅在确认要替换本轮 v2 日志时使用 `FORCE=1`。

## 4. 验收与检查

可随时重新验收某一阶段：

```bash
bash scripts/peftleak_text_v2_privacy.sh validate smoke
bash scripts/peftleak_text_v2_privacy.sh validate pilot
bash scripts/peftleak_text_v2_privacy.sh validate formal
```

自动验收包括：

- `result_status=ok` 且 summary 唯一完整；
- `n_inputs_completed` 与阶段请求一致；
- clean `none` 的 `rec_token_mean` 和 `recovered_position_count_mean` 均大于零；
- `probe_inventory_fixed=true`、`probe_installed_before_private_data=true`；
- `decoder_private_routing=false`；
- `public_stats_source=test_disjoint_partition`；
- 所有配置的 `shared_gradient_names_sha256` 完全一致；
- 无 traceback、CUDA OOM、NaN 或缺失关键指标；
- formal 目录恰好包含 24 个日志。

快速查看正式结果：

```bash
rg -n '^(seed|defense|defense_param_value|rec_token_mean|recovered_position_count_mean|agg_r1fm_r2fm)=' "$RUN_ROOT/formal"
```

## 5. 论文口径

旧 utility 来自正常 Adapter 训练；新版 privacy 防御和观察的是额外注册的恶意 probe 梯度。因此二者合并时标记为：

```text
cross-protocol supplementary privacy-utility comparison
```

不要称为严格的同一 `Adapter + probe` 上传更新 Pareto，也不要写成 original PEFTLeak/ReCIT reproduction。
