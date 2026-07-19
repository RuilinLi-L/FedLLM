# Adaptive-LRB 单客户端实验运行手册

## 协议边界

- 本手册只覆盖单客户端、聚合前梯度可见、`algo=sgd`。
- 旧 `--split val/test` 继续表示从 training partition 构造的 `legacy_internal` 协议。
- `--split official_validation` 才表示训练外 held-out 协议。
- 旧命令默认 `--adaptive_lrb_sign_source legacy_cpu`，用于逐项复现既有 defense-aware DAGER。
- 本手册的新 matrix 使用 `defense_device`，使 `oracle/ratio_hidden` 在 GPU 上精确重放防御实际 signs。
- 固定但隐藏 signs 只是攻击知识消融。`per_update` 私有 signs 才是新的随机化协议，但仍不构成形式化隐私。

## 环境与冒烟

```bash
cd /data/lrl/FedLLM
conda activate dager

bash scripts/run_adaptive_lrb_matrix.sh \
  --dataset sst2 \
  --checkpoint ./models/gpt2_sst2_clean_num_epochs_2/final \
  --split official_validation --k 0.5 \
  --variants proj_only,proj_uniform \
  --knowledge oracle,ratio_hidden,signs_hidden,method_only \
  --sign_source defense_device \
  --defense_seed_mode static --defense_seed 700001 \
  --attack_seed 900001 --ratio_grid 0.2,0.35,0.5,0.65,0.75,0.9 \
  --seed_samples 2 --reduce min --seeds 101 --mode smoke
```

冒烟必须满足：所有目标日志只有一个完整 summary、`result_status=ok`、`n_inputs_completed=2`；`oracle/ratio_hidden` 为 `sign_source=defense_device`、`sign_knowledge=exact`，`method_only` 为 `ratio_knowledge=hidden`、`sign_knowledge=hidden`。

## 正式知识分层矩阵

```bash
bash scripts/run_adaptive_lrb_matrix.sh \
  --dataset sst2 \
  --checkpoint ./models/gpt2_sst2_clean_num_epochs_2/final \
  --split official_validation --n_inputs 100 --batch_size 2 --k 0.5 \
  --variants proj_only,proj_rule_only,proj_empirical_only,proj_uniform,proj_uniform_pool \
  --knowledge oracle,ratio_hidden,signs_hidden,method_only \
  --sign_source defense_device \
  --defense_seed_mode static --defense_seed 700001 \
  --attack_seed 900001 --ratio_grid 0.2,0.35,0.5,0.65,0.75,0.9 \
  --seed_samples 16 --reduce min --seeds 101,202,303 --mode formal
```

## Uniform Oracle Resolution Sweep（服务器执行）

本实验只补充 Projection-LRB 的 uniform-ratio oracle 曲线，不重跑已有的 adaptive allocator 消融或 per-update matrix。现有 `r=0.5` signed/unsigned oracle 三种子结果复用 `adaptive_lrb_matrix_sst2_official_validation_20260718_114719`；新任务包括 `r=0.65/0.75/0.9` 的 uniform signed reconstruction，以及同协议 `none` anchor。

先在服务器上检查所有委托命令：

```bash
bash -n scripts/run_adaptive_lrb_matrix.sh
bash -n scripts/run_uniform_oracle_resolution_sweep.sh
bash scripts/run_uniform_oracle_resolution_sweep.sh \
  --checkpoint ./models/gpt2_sst2_clean_num_epochs_2/final \
  --python /data2/ky/envs/dager/bin/python \
  --run_root log/runs/uniform_oracle_resolution_sst2_official_validation \
  --dry_run
```

确认 checkpoint、Python 和输出路径后执行；脚本会先完成四个 2-input smoke，再顺序执行 12 个正式运行，并通过 `--skip_existing` 支持从已完成日志继续：

```bash
bash scripts/run_uniform_oracle_resolution_sweep.sh \
  --checkpoint ./models/gpt2_sst2_clean_num_epochs_2/final \
  --python /data2/ky/envs/dager/bin/python \
  --run_root log/runs/uniform_oracle_resolution_sst2_official_validation
```

正式运行完成后执行准入、三种子 sample SD 和 paired cluster bootstrap：

```bash
/data2/ky/envs/dager/bin/python scripts/analyze_uniform_oracle_resolution.py \
  --existing-r05 'log/同预算 white-box baselines/new/adaptive_lrb_matrix_sst2_official_validation_20260718_114719' \
  --sweep-root log/runs/uniform_oracle_resolution_sst2_official_validation \
  --output-dir analysis/uniform_oracle_resolution
```

分析脚本要求每个条件恰好包含 seeds `101/202/303`、`result_status=ok`、100/100 inputs 和唯一日志，否则直接失败。输出包括：

- `uniform_oracle_resolution_summary.csv`：Candidate recall、Top-B、L1、L2、R1+R2 的 mean 与 sample SD；
- `uniform_oracle_resolution_paired_bootstrap.csv`：相邻 ratios 及 `r=0.5 -> none` 的 paired cluster bootstrap 95% CI；
- `uniform_oracle_resolution_results.md`：表格、审计日志与预注册解释分支。

只有输出 `interpretation_branch=resolution_supported` 时，正文才写“resolution reduction contributes to defense-aware suppression”；否则必须采用 `span_mismatch_only` 分支。

## 每更新随机 signs 与 EOT-style 压力测试

```bash
bash scripts/run_adaptive_lrb_matrix.sh \
  --dataset sst2 \
  --checkpoint ./models/gpt2_sst2_clean_num_epochs_2/final \
  --split official_validation --n_inputs 100 --batch_size 2 --k 0.5 \
  --variants proj_only,proj_uniform \
  --knowledge oracle,signs_hidden,method_only \
  --sign_source defense_device \
  --defense_seed_mode per_update --defense_seed 700001 \
  --attack_seed 900001 --seed_samples 1,4,16,64 \
  --reduce min,mean --seeds 101,202,303 --mode formal
```

`min` 是主攻击结果；`mean` 只作为 EOT-style 敏感性分析。两者不得聚合为同一行。

## 同预算 exact white-box 强基线

以下命令显式启用 defense-device signs，并固定使用与知识分层矩阵相同的
`official_validation` held-out 协议，用于补齐同预算 white-box baseline。
不得删除 `--split official_validation`，否则 `defense_baselines.sh` 会回退到
`legacy_internal` 的 `val` 协议，结果不能与本手册前述矩阵直接比较。

```bash
export CKPT=./models/gpt2_sst2_clean_num_epochs_2/final
export DAGER_SEEDS="101 202 303"

run_adaptive () {
  bash scripts/defense_baselines.sh sst2 2 gpt2 100 \
    --baseline_defense "$1" --baseline_param "$2" \
    --adaptive_attack_check --skip_anchor_none \
    --split official_validation \
    --adaptive_lrb_sign_source defense_device \
    --finetuned_path "$CKPT" --algo sgd \
    --adaptive_candidate_multiplier 100
}

run_adaptive lrbprojonly 0.5
run_adaptive topk 0.1
run_adaptive compression 8
run_adaptive noise 5e-4
```

## PTG `qkv_only`

```bash
run_ptg () {
  local seed="$1"
  local defense="$2"
  shift 2
  FEDLLM_LOG_DIR="log/ptg_qkv_only/seed_${seed}" \
    bash scripts/ptg_baselines.sh sst2 1 gpt2 100 \
      --exposure qkv_only --baseline_defense "$defense" \
      --finetuned_path "$CKPT" --rng_seed "$seed" \
      --ptg_steps 80 --ptg_restarts 1 --ptg_lr 0.1 \
      --ptg_match_loss cosine --ptg_embed_norm_weight 0.01 \
      --ptg_label_mode known --rouge_backend datasets "$@"
}

for seed in 101 202 303; do
  run_ptg "$seed" none
  run_ptg "$seed" proj_only --baseline_param 0.2
  run_ptg "$seed" proj_only --baseline_param 0.5
  run_ptg "$seed" topk --baseline_param 0.1
  run_ptg "$seed" compression --baseline_param 2
  run_ptg "$seed" compression --baseline_param 8
  run_ptg "$seed" noise --baseline_param 5e-4
  run_ptg "$seed" dpsgd --baseline_param 5e-4
done
```

## 结果准入

- 正式 privacy 行要求 `result_status=ok`、`100/100`、seeds `101/202/303`。
- 不混合 `legacy_internal` 与 `official_validation`。
- 不混合 `oracle/ratio_hidden/signs_hidden/method_only`。
- 不混合 `legacy_cpu/defense_device` sign source。
- 不混合 `static/per_update` 或 `min/mean`。
- 旧日志继续作为 legacy protocol 证据，不覆盖、不删除、不重新命名。
