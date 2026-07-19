# AAAI 2027 证据表与可追溯索引

更新日期：2026-07-19  
用途：为论文正文、附录和结果句提供单一数值来源。本文件只登记本地实际存在的日志；计划实验和缺失结果不得写成已完成证据。

## 1. 统计、配对与纳入规则

- 三种子结果固定使用 seeds `101/202/303`。每个 seed 的 100-input/update 聚合值作为一个独立观测，再跨 seed 计算均值与 sample standard deviation（`ddof=1`）；不把 300 个样本直接混池。
- 正式 privacy 行必须满足 `result_status=ok`、`n_inputs_completed=n_inputs_requested=100`，且三个规范化 `log_path` 互不重复。Utility 行必须有三个正常结束的训练日志和唯一 seed。
- 排除 smoke、失败、不完整、重复收集结果以及没有最终 summary 的日志。缺失项不按 0 处理，也不使用其他 operating point 代替。
- Privacy–utility 配对要求 dataset、checkpoint 语义、batch size、训练轮数、defense 与 parameter 一致。Legacy Adapter probe 的 privacy 与 utility 明确属于 cross-protocol comparison，是唯一例外。
- DAGER、Legacy Adapter probe、fixed-probe v2、PTG `first2` 与图像侧 patch recovery 的绝对恢复率不得跨协议排序。
- 表中 `mean ± SD` 均由逐 seed 正式记录重算为 sample SD，不采用旧 collector 的混池统计，也不由已四舍五入的 population SD 反推。

### 1.1 论文表格准入状态

| Table | Candidate rows | Current admission | Blocking evidence |
| --- | --- | --- | --- |
| Main Table 1 | DAGER 三任务 privacy–utility | 26/26 rows admitted | SST-2 Projection-LRB (adaptive ratios, k=0.9) 缺 utility，故不进入 26 行 |
| Main Table 2A | 六项 Projection-LRB 组件消融 | 6/6 rows admitted | none |
| Main Table 2B | adaptive/uniform/signed controls under oracle DAGER | 7/7 rows admitted | none |
| Main Table 3 | Legacy Adapter ratio probe | 7/7 rows admitted；privacy 为单次运行 | privacy 多种子不属于当前协议 |
| Main Table 4 | PTG `first2` selected privacy–utility | 7/7 rows admitted；每行 privacy/utility 均为三种子 | none |
| Appendix Table A1 | fixed-probe v2 | 8/8 rows admitted | none |
| Appendix Table A2 | sensitivity 与 keep-ratio | sensitivity 5/5；keep-ratio privacy 4/4 | k=0.5/@0.75 utility 各缺一个正常 summary，因此 keep-ratio 只报 privacy |
| Appendix Table A3 | PTG `first2` full-sweep privacy extension | 8/8 rows admitted | 仅作完整参数 sweep，不用于七点正文表的总体优劣排序 |
| Appendix Table A4 | uniform-oracle resolution 与 paired bootstrap | 6/6 rows；5/5 comparisons admitted | none |
| Appendix Table A5 | static knowledge 与 per-update stress test | static 60/60；per-update 102/102 | none |
| Appendix Table A6 | PEFTLeak-style image-side study | 9/9 canonical logs admitted | single seed、privacy-only，不补 utility |

## 2. Main Table 1: Full-Gradient DAGER

设置：GPT-2 sequence classification、batch size 2、每配置每 seed 100 个单客户端更新，即 200 条文本。所有值为 seeds `101/202/303` 的 mean ± sample SD；`--` 表示该任务不使用 MCC。

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

Privacy sources：三任务主 roots 为 `log/runs/dager_privacy_{sst2,cola,rotten_tomatoes}_baselines_3seed_20260709_*`；high-retention top-k/compression 使用相应配置的 `results.csv`。CoLA/Rotten Tomatoes noise@1e-6 的 seed101 来自 `dager_privacy_{dataset}_noise_seed101_20260713_*`，seed202/303 来自 `log/dager_defense/defense_baselines_{dataset}_b2_gpt2_focus_noise_1e-6_20260715_*`。SST-2 的 seed202/303 同样来自 `log/dager_defense`；seed101 来自完整 100/100 的历史正式日志 `log/runs/defense_baselines_sst2_b2_gpt2_20260509_115431/noise_1e-6.txt`。该旧 summary 未显式写 seed，但 `_run_header.txt` 未传 `--rng_seed`，且运行日期前最近归档 commit `0c9af631` 的 `args_factory.py` 默认 `rng_seed=101`，据此识别为 seed101。DP-SGD-style@1e-5 来自 `log/dager_defense/defense_baselines_{dataset}_b2_gpt2_focus_dpsgd_1e-5_20260715_*` 及其 completed seed subruns。Utility sources：none/Projection-LRB 与 corrected high-retention baselines 使用原 utility roots；noise 与 DP-SGD-style 使用 `log/dager_utility/dager_utility_cola_rt_critical_{noise_1e-6,dpsgd_1e-5}_{dataset}_20260715_*` 的逐 seed 训练日志。

准入边界：SST-2 Projection-LRB (adaptive ratios, k=0.9) 只有三种子 privacy，没有严格配对 utility，故不进入 combined Main Table 1。更低保留率 top-k 与更低 bit compression 的完整 sweep 包含零恢复点，但它们不是当前 26 行表中的同一 operating points。DP-SGD-style 没有 privacy accountant，不得标为形式化 DP。

## 3. Main Table 2: Projection-LRB Mechanism Analysis

### 3.1 Panel A: Component ablation

唯一来源：`log/runs/old/lrb_ablation_sst2_b2_gpt2_k0.5_20260501_004737/raw_results.csv`。Privacy 只取 `log_kind=attack_dager`，utility 只取 `log_kind=train`；proxy 行排除。Privacy seed 从文件名解析，utility seed 使用 summary 字段。所有行均为 3×`ok`，privacy 为 3×100/100。

| Variant | `rec_token_mean` | R1+R2 | Accuracy | Macro-F1 | Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | `0.855333 ± 0.031263` | `148.260102 ± 7.775494` | `0.913226 ± 0.006521` | `0.913184 ± 0.006501` | `0.246637 ± 0.005213` |
| identity | `0.855333 ± 0.031263` | `148.260102 ± 7.775494` | `0.913226 ± 0.006521` | `0.913184 ± 0.006501` | `0.246637 ± 0.005213` |
| clip-only | `0.854076 ± 0.033792` | `147.955623 ± 7.998571` | `0.918196 ± 0.002885` | `0.918149 ± 0.002897` | `0.236540 ± 0.014272` |
| projection-only (adaptive ratios) | `0.000000 ± 0.000000` | `0.000000 ± 0.000000` | `0.915520 ± 0.003686` | `0.915499 ± 0.003696` | `0.255493 ± 0.008061` |
| projection+clipping (adaptive ratios) | `0.000000 ± 0.000000` | `0.000000 ± 0.000000` | `0.913226 ± 0.005171` | `0.913197 ± 0.005154` | `0.259752 ± 0.009504` |
| Full-LRB (adaptive ratios) | `0.000000 ± 0.000000` | `0.000000 ± 0.000000` | `0.892584 ± 0.013439` | `0.892472 ± 0.013547` | `0.321702 ± 0.028850` |

该批次记录的 checkpoint 为 `./models/gpt2-ft-rt`，与 2026-07 三任务主 DAGER 批次不同。允许主张：在这一 standard-DAGER 消融中，projection-only 已达到当前攻击的零恢复端点，而 clip-only 与 none 的恢复接近；Full-LRB 具有更大的效用代价。不得由此证明 hybrid sensitivity 必要或推出 adaptive robustness。

### 3.2 Panel B: Oracle ratio and signed controls

Sources：adaptive@0.5 与 uniform unsigned@0.5 来自 `log/同预算 white-box baselines/new/adaptive_lrb_matrix_sst2_official_validation_20260718_114719/results.csv`；uniform signed@0.5 复用同一 static matrix；@0.65/@0.75/@0.9 与 undefended anchor 来自 `log/同预算 white-box baselines/new/uniform_oracle_resolution_sst2_official_validation/formal/*/results.csv`。全部为 seeds `101/202/303`、3×`ok`、3×100/100、日志唯一。

| Configuration | Candidate | Top-B | R1+R2 |
| --- | ---: | ---: | ---: |
| adaptive ratios@0.5 | `0.919333±0.019112` | `0.914969±0.019117` | `101.872623±3.734154` |
| uniform signed@0.5 | `0.921943±0.020668` | `0.916526±0.018586` | `101.933598±2.938826` |
| uniform signed@0.65 | `0.930479±0.024997` | `0.925149±0.023091` | `101.176730±3.467582` |
| uniform signed@0.75 | `0.927909±0.021159` | `0.922875±0.018806` | `104.101123±3.511430` |
| uniform signed@0.9 | `0.931242±0.021103` | `0.926690±0.019512` | `104.571362±3.670613` |
| uniform unsigned@0.5 | `0.907101±0.017703` | `0.903459±0.015739` | `101.143199±4.638276` |
| undefended anchor | `0.976223±0.010324` | `0.970630±0.005882` | `122.556760±5.062032` |

允许结论：adaptive 与 uniform 在当前 oracle 协议下不可区分；signed@0.5 与 unsigned@0.5 不支持 signed oracle advantage；undefended anchor 不是 operator 的连续 `r=1` 点。

## 4. Main Table 3: Legacy Adapter Ratio Probe

设置：SST-2、GPT-2 Adapter、reduction factor 16。Privacy 使用 batch 1、known label、sample-adaptive malicious probe、legacy public-bin statistics，每行是无 seed 后缀的单次 100/100 正式日志。Utility 使用 batch 8、3 epochs、seeds `101/202/303` 的 benign Adapter training，属于 cross-protocol comparison。

| Method | `rec_token_mean` (single run) | R1+R2 (single run) | Accuracy | Macro-F1 | Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | `0.760000` | `145.000000` | `0.916667 ± 0.003504` | `0.916589 ± 0.003491` | `0.240148 ± 0.008295` |
| Projection-LRB (adaptive ratios, k=0.5) | `0.202490` | `12.036481` | `0.909786 ± 0.005657` | `0.909698 ± 0.005691` | `0.252664 ± 0.004481` |
| Projection-LRB (adaptive ratios, k=0.65) | `0.212765` | `13.693024` | `0.914373 ± 0.004634` | `0.914304 ± 0.004650` | `0.250359 ± 0.005149` |
| Projection-LRB (adaptive ratios, k=0.9) | `0.234016` | `13.137760` | `0.913226 ± 0.006720` | `0.913150 ± 0.006736` | `0.246265 ± 0.006243` |
| top-k@0.05 | `0.431431` | `48.125711` | `0.903670 ± 0.001148` | `0.903515 ± 0.001182` | `0.255337 ± 0.003274` |
| compression@4 | `0.115571` | `7.733200` | `0.901758 ± 0.003686` | `0.901622 ± 0.003706` | `0.261210 ± 0.000647` |
| Full-LRB@0.5 | `0.015222` | `0.461551` | `0.850535 ± 0.020811` | `0.850337 ± 0.020861` | `0.363690 ± 0.039182` |

Privacy source files：`log/peftleak_text_sst2/privacy/gpt2_adapter_ratio_{none,proj_only_0.5,proj_only_0.65,proj_only_0.9,topk_0.05,compression_4,full_lrb_0.5}.txt`。Utility source files：`log/peftleak_text_sst2/utility_/gpt2_adapter_{none,proj_only_0.5,proj_only_0.65,proj_only_0.9,topk_0.05,compression_4,full_lrb_0.5}_seed{101,202,303}.txt`。旧证据表中的 `privacy1/` 与 `utility/` 路径无效，已更正。

## 5. PTG `first2` Evidence

设置：SST-2、GPT-2、batch 1、每配置每 seed 100 inputs、seeds `101/202/303`、known label/padding、single restart、80 Adam steps、lr 0.1、cosine matching、embedding-norm weight 0.01。防御先作用于完整更新，再选择前两个 blocks 的 24 个 gradient tensors，其中 8 个为 matrix gradients。全部 45 个日志为 `ok` 且完成 100/100。

Source root：`log/ptg_gpt2_first2_privacy_3seed_20260714/`。每行唯一映射为 `seed_<seed>/ptg_sst2_b1_gpt2_first2_focus_<config>_*/results.csv`。

### 5.1 Full privacy sweep

| Profile | Method | `rec_token_mean` | R1+R2 |
| --- | --- | ---: | ---: |
| P0 | none | `0.160194 ± 0.016123` | `19.772638 ± 1.583317` |
| P0 | Projection-LRB (adaptive ratios, k=0.2) | `0.121290 ± 0.018283` | `14.616599 ± 3.928618` |
| P0 | Projection-LRB (adaptive ratios, k=0.5) | `0.125698 ± 0.013819` | `16.236530 ± 1.106815` |
| P1 | Projection-LRB (adaptive ratios, k=0.65) | `0.143678 ± 0.013432` | `17.769638 ± 1.314082` |
| P0 | Projection-LRB (adaptive ratios, k=0.75) | `0.125995 ± 0.011764` | `17.722520 ± 0.296722` |
| P1 | Projection-LRB (adaptive ratios, k=0.9) | `0.149386 ± 0.009948` | `17.798912 ± 1.789991` |
| P1 | top-k@0.05 | `0.145112 ± 0.005394` | `18.035789 ± 0.614641` |
| P1 | top-k@0.1 | `0.154005 ± 0.025120` | `18.806127 ± 4.648958` |
| P1 | top-k@0.3 | `0.157531 ± 0.005060` | `20.273679 ± 0.635447` |
| P0 | compression@2 | `0.055496 ± 0.008613` | `6.346535 ± 1.159283` |
| P0 | compression@4 | `0.116038 ± 0.005164` | `13.742571 ± 3.370739` |
| P1 | compression@8 | `0.149767 ± 0.014262` | `19.011232 ± 2.390093` |
| P1 | compression@16 | `0.161157 ± 0.007200` | `18.846869 ± 1.798701` |
| P0 | noise@5e-4 | `0.078178 ± 0.004606` | `7.138664 ± 2.085052` |
| P0 | DP-SGD-style@5e-4 | `0.070653 ± 0.003690` | `6.569028 ± 1.760894` |

Projection-LRB (adaptive ratios, k=0.2) 相对同 seed clean 的绝对 token-recovery 下降分别为 `0.071785/0.020631/0.024296`，均值为 `0.038904 ± 0.028534`；用聚合均值计算的相对下降为 `(0.160194-0.121290)/0.160194=24.3%`。三个 seed 的 token recovery 与 R1+R2 方向均下降。

### 5.2 Main Table 4: selected paired privacy–utility

正文固定报告七个配置：none、Projection-LRB (adaptive ratios, k=0.2/0.5)、top-k@0.1、compression@8、noise@5e-4 和 DP-SGD-style@5e-4。Adaptive-ratio k=0.2 是当前 `first2` sweep 中恢复率最低的 projection 点，k=0.5 与 full-gradient DAGER 和 Adapter 机制实验的主点对齐；top-k@0.1 与 compression@8 沿用 SST-2 full-gradient 主协议的代表性 operating points；noise 和 DP-SGD-style 作为 coverage baselines。该选择不是 PTG 全 sweep 的逐方法最优调参，完整 15 点 privacy 仍由第 5.1 节保留。

Utility source root：`log/ptg_utility/`。七行共纳入 21 个唯一的 `result_status=ok` 训练日志，均为 SST-2、`./models/gpt2_sst2_clean_num_epochs_2/final`、batch 1、1 epoch、seeds `101/202/303`。Projection-LRB (adaptive ratios, k=0.5)/seed101 使用完成目录 `ptg_utility_sst2_b1_proj_0.5_20260716_164508_seed101`；旧目录 `...20260714_145129-seed101` 只运行约 65% 且没有最终 summary，已排除。`ptg_utility_sst2_b1_proj_0.2_20260715_131728-202/train_seed303.txt` 与 `...131728_303seed/train_seed303.txt` 的 SHA-256 均为 `3194A3540C2EB46616A92EF28B88008A112C82D009071B91FB378BA65DD81BF2`，属于同一 seed 303 日志的重复副本，只计一次。

| Method | `rec_token_mean` | R1+R2 | Accuracy | Macro-F1 | Loss | Train time (h) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | `0.160194 ± 0.016123` | `19.772638 ± 1.583317` | `0.900230 ± 0.004135` | `0.900178 ± 0.004143` | `0.272405 ± 0.004148` | `0.946 ± 0.024` |
| Projection-LRB (adaptive ratios, k=0.2) | `0.121290 ± 0.018283` | `14.616599 ± 3.928618` | `0.913226 ± 0.002388` | `0.913171 ± 0.002409` | `0.258121 ± 0.004659` | `10.871 ± 1.136` |
| Projection-LRB (adaptive ratios, k=0.5) | `0.125698 ± 0.013819` | `16.236530 ± 1.106815` | `0.918960 ± 0.002885` | `0.918945 ± 0.002887` | `0.255761 ± 0.012472` | `14.086 ± 3.558` |
| top-k@0.1 | `0.154005 ± 0.025120` | `18.806127 ± 4.648958` | `0.911315 ± 0.007807` | `0.911270 ± 0.007847` | `0.265506 ± 0.021022` | `3.323 ± 0.852` |
| compression@8 | `0.149767 ± 0.014262` | `19.011232 ± 2.390093` | `0.911697 ± 0.001987` | `0.911668 ± 0.001999` | `0.248106 ± 0.005608` | `4.526 ± 0.551` |
| noise@5e-4 | `0.078178 ± 0.004606` | `7.138664 ± 2.085052` | `0.909404 ± 0.008028` | `0.909391 ± 0.008036` | `0.253409 ± 0.020606` | `1.226 ± 0.080` |
| DP-SGD-style@5e-4 | `0.070653 ± 0.003690` | `6.569028 ± 1.760894` | `0.920107 ± 0.001751` | `0.920081 ± 0.001744` | `0.633798 ± 0.017393` | `1.725 ± 0.026` |

Projection-LRB 的 adaptive-ratio k=0.2/0.5 instantiations 在当前三种子均值上未观察到 accuracy 或 macro-F1 损失，但不能据此声称统计显著提升；其平均训练时间分别为 clean 的约 `11.5×` 和 `14.9×`，是七个正文点中开销最高的方案。Noise 和 DP-SGD-style 的恢复指标更低，其中 DP-SGD-style 的 eval loss 明显升高，且该实现没有 privacy accountant。完整 sweep 中 compression@2 的 `rec_token_mean=0.055496 ± 0.008613` 仍低于七点正文表中的所有防御，因此 Main Table 4 不支持 Projection-LRB 在 PTG 下普遍优于压缩或噪声基线。

## 6. Appendix Table A1: Fixed-Probe Adapter Ratio v2

Source root：`log/peftleak_text_sst2_v2/20260714_024606/formal/`，24 个正式运行（8 configs × 3 seeds），均为 batch 1、每 seed 100 条文本。Probe inventory 覆盖 128 个 token positions，包含 256 个 weight/bias gradient tensors，在私有数据到达前安装；public statistics 使用严格 disjoint partition。Utility 复用第 4 节对应的三种子 benign Adapter logs。

| Method | `rec_token_mean` | R1+R2 | Accuracy | Macro-F1 | Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | `1.000000 ± 0.000000` | `193.666667 ± 1.154700` | `0.916667 ± 0.003504` | `0.916589 ± 0.003491` | `0.240148 ± 0.008295` |
| Projection-LRB (adaptive ratios, k=0.5) | `0.685368 ± 0.019729` | `98.674391 ± 2.675286` | `0.909786 ± 0.005657` | `0.909698 ± 0.005691` | `0.252664 ± 0.004481` |
| Projection-LRB (adaptive ratios, k=0.65) | `0.761991 ± 0.020442` | `117.764104 ± 1.432629` | `0.914373 ± 0.004634` | `0.914304 ± 0.004650` | `0.250359 ± 0.005149` |
| Projection-LRB (adaptive ratios, k=0.75) | `0.836252 ± 0.014610` | `136.816461 ± 1.963298` | `0.911315 ± 0.007635` | `0.911226 ± 0.007657` | `0.246870 ± 0.006813` |
| Projection-LRB (adaptive ratios, k=0.9) | `0.837218 ± 0.014158` | `137.105283 ± 1.633826` | `0.913226 ± 0.006720` | `0.913150 ± 0.006736` | `0.246265 ± 0.006243` |
| top-k@0.1 | `1.000000 ± 0.000000` | `193.666667 ± 1.154700` | `0.907875 ± 0.005419` | `0.907739 ± 0.005460` | `0.248322 ± 0.004294` |
| compression@6 | `0.999444 ± 0.000963` | `193.433333 ± 0.750556` | `0.909786 ± 0.004028` | `0.909692 ± 0.004042` | `0.246968 ± 0.003666` |
| noise@1e-3 | `0.358894 ± 0.010977` | `58.458704 ± 0.805420` | `0.912844 ± 0.003034` | `0.912731 ± 0.003050` | `0.243639 ± 0.007508` |

## 7. Appendix Table A2: Projection Sensitivity and Keep-Ratio

### 7.1 Sensitivity variants at k=0.5

主来源：`log/runs/lrb_ablation_sst2_b2_gpt2_k0.5_20260510_010044/raw_results.csv`；`proj_no_empirical` 使用完成的恢复批次 `..._resume_no_empirical/raw_results.csv`，原批次中的三条 failed 记录排除。

| Variant | `rec_token_mean` | R1+R2 | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| hybrid allocator (projection-only) | `0.000000 ± 0.000000` | `0.000000 ± 0.000000` | `0.915137 ± 0.004135` | `0.915086 ± 0.004127` |
| rule-only | `0.000000 ± 0.000000` | `0.000000 ± 0.000000` | `0.914373 ± 0.002387` | `0.914339 ± 0.002410` |
| empirical-only | `0.000000 ± 0.000000` | `0.000000 ± 0.000000` | `0.917049 ± 0.001324` | `0.917003 ± 0.001330` |
| uniform | `0.000000 ± 0.000000` | `0.000000 ± 0.000000` | `0.917049 ± 0.004028` | `0.917002 ± 0.004025` |
| no-empirical | `0.000000 ± 0.000000` | `0.000000 ± 0.000000` | `0.914373 ± 0.002387` | `0.914339 ± 0.002410` |

### 7.2 Keep-ratio privacy-only sweep

| k | `rec_token_mean` | R1+R2 | Source raw results |
| ---: | ---: | ---: | --- |
| 0.5 | `0.000000 ± 0.000000` | `0.000000 ± 0.000000` | `lrb_ablation_sst2_b2_gpt2_k0.5_20260510_182031` |
| 0.65 | `0.000000 ± 0.000000` | `0.000000 ± 0.000000` | `lrb_ablation_sst2_b2_gpt2_k0.65_20260510_142233` |
| 0.75 | `0.000000 ± 0.000000` | `0.000000 ± 0.000000` | `lrb_ablation_sst2_b2_gpt2_k0.75_20260510_182130` |
| 0.9 | `0.000000 ± 0.000000` | `0.000000 ± 0.000000` | `lrb_ablation_sst2_b2_gpt2_k0.9_20260510_182257` |

四个 privacy 点均为 3×`ok`、3×100/100。k=0.5 与 k=0.75 的 utility 各只有两个 `result_status=ok`，故本表不混入不完整 utility。k>0.75 时 sensitivity-to-keep-ratio 方向反转，因此该 sweep 不能表述为“敏感层始终压缩更强”。

## 8. Historical Defense-Aware DAGER Boundary

Source：`log/runs/adaptive_lrb_keypoints_m100_sst2_b2_gpt2_20260614_223021/results.csv`。

| Variant | `rec_token_mean` | R1+R2 | Runs |
| --- | ---: | ---: | ---: |
| Projection-LRB (adaptive ratios, k=0.5) | `0.931875 ± 0.043807` | `143.967226 ± 6.766789` | 3 |
| projection-uniform@0.5 | `0.933369 ± 0.043342` | `144.413365 ± 6.004369` | 3 |
| Full-LRB@0.5 | `0.950914 ± 0.018713` | `17.788297 ± 1.608086` | 3 |

公开、确定性的 projection 可被 defense-aware decoding 绕过。Full-LRB 的 token 与 ROUGE 指标明显不一致，正文不选择性引用该行。

## 9. Appendix A4: Uniform-Oracle Resolution

Analysis artifacts：

- summary：`log/同预算 white-box baselines/new/uniform_oracle_resolution/uniform_oracle_resolution_summary.csv`
- paired bootstrap：`log/同预算 white-box baselines/new/uniform_oracle_resolution/uniform_oracle_resolution_paired_bootstrap.csv`
- formal logs：`log/同预算 white-box baselines/new/uniform_oracle_resolution_sst2_official_validation/formal/`

| Condition | Candidate | Top-B | L1 | L2 | R1+R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| uniform signed@0.5 | `0.921943±0.020668` | `0.916526±0.018586` | `0.433333±0.035119` | `0.791667±0.023094` | `101.933598±2.938826` |
| uniform signed@0.65 | `0.930479±0.024997` | `0.925149±0.023091` | `0.438333±0.040104` | `0.785000±0.022913` | `101.176730±3.467582` |
| uniform signed@0.75 | `0.927909±0.021159` | `0.922875±0.018806` | `0.446667±0.033292` | `0.798333±0.012583` | `104.101123±3.511430` |
| uniform signed@0.9 | `0.931242±0.021103` | `0.926690±0.019512` | `0.448333±0.030139` | `0.791667±0.018930` | `104.571362±3.670613` |
| undefended anchor | `0.976223±0.010324` | `0.970630±0.005882` | `0.553333±0.025658` | `0.798333±0.017559` | `122.556760±5.062032` |
| uniform unsigned@0.5 | `0.907101±0.017703` | `0.903459±0.015739` | `0.430000±0.040927` | `0.800000±0.018028` | `101.143199±4.638276` |

| Comparison | Mean delta | Paired 95% CI |
| --- | ---: | ---: |
| `0.5→0.65` | `-0.767660` | `[-2.986876, 1.138422]` |
| `0.65→0.75` | `2.922980` | `[0.348377, 5.994435]` |
| `0.75→0.9` | `0.504543` | `[-1.414867, 2.208970]` |
| `0.9→none` | `17.951263` | `[13.885826, 22.530650]` |
| `0.5→none` | `20.611127` | `[15.915719, 25.487489]` |

预设分支：`span_mismatch_only`。Monotonic resolution effect over `r=0.5--0.9` 被反驳；operator-induced span transformation 相对 undefended anchor 得到机制支持。

## 10. Appendix A5: Static Knowledge and Per-Update Stress Test

Static root：`log/同预算 white-box baselines/new/adaptive_lrb_matrix_sst2_official_validation_20260718_114719/`。审计：60 rows、60 unique logs、seeds `101/202/303`、60×`ok`、60×100/100。覆盖 adaptive/rule-only/empirical-only/uniform signed/uniform unsigned × oracle/ratio-hidden/signs-hidden/method-only。正文使用的 adaptive 结果：

| Knowledge | Candidate | Top-B | L1 | L2 | R1+R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| oracle | `0.919333±0.019112` | `0.914969±0.019117` | `0.436667±0.038837` | `0.795000±0.017321` | `101.872623±3.734154` |
| ratio-hidden | `0.984620±0.002225` | `0.947560±0.015939` | `0±0` | `0±0` | `9.421112±0.517537` |
| signs-hidden | `0.978630±0.014517` | `0.574774±0.064957` | `0±0` | `0±0` | `6.453098±0.561383` |
| method-only | `0.977448±0.013724` | `0.699022±0.050986` | `0±0` | `0±0` | `7.265339±0.630267` |

Per-update root：`log/同预算 white-box baselines/new/adaptive_lrb_matrix_sst2_official_validation_20260718_114758/`。审计：102 rows、102 unique logs、seeds `101/202/303`、102×`ok`、102×100/100。覆盖 adaptive/uniform、oracle 及 signs-hidden/method-only 的 1/4/16/64 hypotheses，并将 `min` 与 `mean` 分开。关键 endpoints：oracle adaptive `101.989353±5.036116`、oracle uniform `101.759123±4.259130`；uniform hidden `mean`@1/4/16/64 为 `5.707280±0.096611` / `7.749452±0.152894` / `8.716969±0.373046` / `8.936966±0.262887`，对应 `min` 为 `5.707280±0.096611` / `6.806477±0.528819` / `6.640623±0.404104` / `6.610561±0.170276`。完整 34 个三种子条件由论文 Appendix A.5 登记。有限 seed sampling 不等价于一般 EOT、直接状态估计或跨更新推断。

## 11. Appendix A6: PEFTLeak-Style Image-Side Evidence

设置：CIFAR-100、upstream-style malicious ViT with Adapters、batch 32、seed 42、fixed victim batch、privacy-only。防御先作用于完整 96 个 Adapter gradient tensors，攻击读取 20 对 weight/bias gradients。Source：`log/peftleak_official_image/baselines/seed42/logs/{none,proj_only_0.5,proj_only_0.75,proj_only_0.9,topk_0.1,topk_0.3,compression_8,compression_16,full_lrb_0.5}.log`。九个 canonical logs 的最终指标状态均为 `ok`。

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

MSE/SSIM/LPIPS 的 scope 为 `clustered_full_images_one_to_one_all_reconstructions`，使用 Hungarian one-to-one matching 并纳入全部重建样本。所有配置的 `strict_normalized_patch_recovery_rate` 均为 0，没有区分力，只登记为诊断状态。该实验是 single seed、fixed victim batch、privacy-only、non-FedAvg、non-multi-client、official-aligned/source-aligned path；攻击 batch loss 不是任务效用，不建立图像侧 privacy--utility trade-off。

## 12. Claim-to-Evidence Matrix

| Planned claim | Status | Boundary |
| --- | --- | --- |
| Clean full-gradient updates leak severe text | Supported | 三任务最新 standard-DAGER clean rows |
| Projection is the dominant standard-DAGER component | Supported within one ablation batch | 不证明 sensitivity 必要，不推广到 adaptive attack |
| Projection-LRB (adaptive ratios, k=0.5) preserves clean-level utility across three tasks | Supported empirically | 不使用“significant improvement” |
| Projection-LRB (adaptive ratios, k=0.9) has paired utility across all three tasks | Unsupported | SST-2 utility missing |
| Projection-LRB disrupts the controlled Adapter ratio probe | Supported only as mechanism evidence | Legacy privacy single-run；v2 显示旧协议高估抑制幅度，不外推到一般 Adapter/LoRA updates |
| Projection-LRB mitigates PTG `first2` leakage | Supported as limited, attack-specific transfer | 七个正文点 privacy/utility 三种子完成；仅覆盖 `first2`，且 compression@2、noise 与 DP-SGD-style 的恢复率更低 |
| Hybrid sensitivity is necessary | Unsupported | standard 与 oracle 下 allocator variants 不可区分 |
| Signed reconstruction has an oracle advantage over unsigned pooling | Unsupported | signed@0.5 与 unsigned@0.5 的 oracle 结果不可区分 |
| Lower ratio monotonically reduces oracle recoverability | Contradicted over `r=0.5--0.9` | non-monotonic plateau；预设分支为 `span_mismatch_only` |
| Transformed-span mismatch explains standard DAGER suppression | Supported as an attack-mechanism explanation | standard/oracle gap、uniform sweep 与 span analysis；不是隐私证明 |
| Projection-LRB is robust to adaptive attacks | Contradicted as a general claim | oracle knowledge restores substantial text |
| Image-side cross-modal transfer | Supported as single-seed supplementary evidence | CIFAR-100/ViT-Adapter fixed batch；strong baselines 同样达到零 patch recovery |
| General PEFT/LoRA robustness | Unsupported | controlled text probe 与 single-seed image path 不覆盖一般 deployments |
| Universal cross-modal generalization | Unsupported | 仅一个图像数据集、模型、victim batch 和 seed |
| Projection-LRB provides formal privacy | Unsupported | 无 accountant 或证明 |
