# SST-2 Results Summary (model-matched logs, updated 2026-05-17)

Source: `log/runs/*` excluding `log/runs/old`, plus the high-`k` Projection-LRB follow-ups from 2026-05-17. Main setting: SST-2 validation split, GPT-2, full-gradient DAGER with `n_inputs=100`, checkpoint `./models/gpt2_sst2_clean_num_epochs_2/final`.

Utility values are 3-seed means over seeds `101/202/303` unless a row is explicitly marked partial. Lower DAGER RecToken and R1+R2 are better. Negative accuracy drop means the defense is slightly above the clean mean in this run set; treat it as clean-level utility rather than a significant improvement.

## Main Privacy-Utility Table

This is the current SST-2 main table. The Projection-LRB row now uses the best complete keep-ratio sweep point, `proj_only@k=0.9`.

| Method | Setting | DAGER RecToken | DAGER R1+R2 | Attack status | Accuracy (%) | Macro-F1 (%) | Acc. drop (pp) | Eval loss | Train time | Role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Clean FedSGD | none | 0.8966 | 158.996 | ok | 91.25 +/- 0.86 | 91.24 +/- 0.86 | +0.00 | 0.2479 | 00:41:20 | Leakage anchor |
| Gaussian noise | sigma=5e-4 | 0.2655 | 3.797 | no_l1_candidates | 91.63 +/- 0.98 | 91.63 +/- 0.98 | -0.38 | 0.2429 | 01:11:09 | Reduces but does not eliminate recovery |
| DP-SGD-style noise | sigma=5e-4 | 0.2499 | 2.800 | no_l1_candidates | 92.20 +/- 0.50 | 92.20 +/- 0.50 | -0.96 | 0.6539 | 01:30:04 | DP-style baseline, no formal accounting |
| Top-k sparsification | ratio=0.1 | 0.0000 | 0.000 | no_l1_candidates | 91.82 +/- 0.86 | 91.82 +/- 0.86 | -0.57 | 0.2362 | 02:43:41 | Strong empirical baseline |
| Quantization | 8 bits | 0.0000 | 0.000 | no_l1_candidates | 92.13 +/- 0.40 | 92.12 +/- 0.40 | -0.88 | 0.2413 | 04:38:14 | Strong empirical baseline |
| Projection-LRB (proj-only) | k=0.9 | 0.0000 | 0.000 | no_l1_candidates | 91.63 +/- 0.70 | 91.62 +/- 0.70 | -0.38 | 0.2475 | 11:41:01 | Best complete sweep point |
| Full LRB | k=0.5 | 0.0000 | 0.000 | no_l1_candidates | 88.38 +/- 1.37 | 88.37 +/- 1.38 | +2.87 | 0.3354 | 14:38:34 | Over-defense reference |

## Projection-LRB Keep-Ratio Sweep

`complete` means privacy, utility, and proxy stages all finished for seeds `101/202/303`. The high-`k` rows are useful boundary probes, but should not be treated as stable operating points.

| Projection keep ratio | Privacy ok | Utility ok | Proxy ok | DAGER RecToken | DAGER R1+R2 | Accuracy (%) | Macro-F1 (%) | Acc. drop (pp) | Eval loss | Train time | Attack time | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| k=0.65 | 3/3 | 3/3 | 3/3 | 0.0000 | 0.000 | 91.48 +/- 0.43 | 91.47 +/- 0.43 | -0.23 | 0.2508 | 08:20:54 | 00:56:56 | complete |
| k=0.9 | 3/3 | 3/3 | 3/3 | 0.0000 | 0.000 | 91.63 +/- 0.70 | 91.62 +/- 0.70 | -0.38 | 0.2475 | 11:41:01 | 00:52:46 | complete |
| k=0.5 | 3/3 | 2/3 | 3/3 | 0.0000 | 0.000 | 91.46 +/- 0.57 | 91.45 +/- 0.57 | -0.21 | 0.2427 | 12:03:18 | 01:11:54 | partial |
| k=0.75 | 3/3 | 2/3 | 3/3 | 0.0000 | 0.000 | 91.11 +/- 0.24 | 91.11 +/- 0.25 | +0.14 | 0.2396 | 07:14:49 | 01:13:26 | partial |
| k=0.95 | 2/3 | 1/3 | 1/3 | 0.0000 | 0.000 | 91.05 +/- 0.00 | 91.05 +/- 0.00 | +0.20 | 0.2530 | 11:10:43 | 01:22:57 | partial boundary probe |
| k=0.97 | 1/3 | 1/3 | 1/3 | 0.0000 | 0.000 | 91.51 +/- 0.00 | 91.51 +/- 0.00 | -0.26 | 0.2518 | 11:55:19 | 02:21:57 | partial boundary probe |
| k=1.0 | 1/3 | 1/3 | 1/3 | 0.0000 | 0.000 | 92.20 +/- 0.00 | 92.20 +/- 0.00 | -0.95 | 0.2438 | 11:56:36 | 01:46:09 | partial boundary probe |

### Proxy Signals

| Projection keep ratio | Grad cosine | Norm retention | Proxy step runtime (s) | Interpretation |
| --- | --- | --- | --- | --- |
| k=0.5 | 0.5193 | 0.4109 | 1.465 | Legacy anchor; strict sweep has one killed train seed |
| k=0.65 | 0.5579 | 0.4118 | 2.660 | Conservative complete point |
| k=0.75 | 0.7579 | 0.5074 | 1.986 | Privacy complete, utility partial |
| k=0.9 | 0.7009 | 0.5525 | 1.354 | Best complete privacy-utility point |
| k=0.95 | 0.7005 | 0.5677 | 2.396 | Boundary probe only |
| k=0.97 | 0.7145 | 0.5569 | 2.604 | Boundary probe only |
| k=1.0 | 0.7121 | 0.5552 | 2.173 | Boundary probe only |

## Utility-Only Summary

This table combines the main utility baselines and the complete Projection-LRB sweep points. The old dedicated `proj_only@k=0.5` utility baseline remains useful as a legacy anchor, but `k=0.9` is the cleaner main-row choice because its sweep privacy, utility, and proxy stages are all complete.

| Method | Setting | Accuracy (%) | Macro-F1 (%) | Acc. drop (pp) | Eval loss | Train time | Runs / status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Clean FedSGD | none | 91.25 +/- 0.86 | 91.24 +/- 0.86 | +0.00 | 0.2479 | 00:41:20 | 3 |
| Gaussian noise | sigma=5e-4 | 91.63 +/- 0.98 | 91.63 +/- 0.98 | -0.38 | 0.2429 | 01:11:09 | 3 |
| DP-SGD-style noise | sigma=5e-4 | 92.20 +/- 0.50 | 92.20 +/- 0.50 | -0.96 | 0.6539 | 01:30:04 | 3 |
| Top-k sparsification | ratio=0.1 | 91.82 +/- 0.86 | 91.82 +/- 0.86 | -0.57 | 0.2362 | 02:43:41 | 3 |
| Quantization | 8 bits | 92.13 +/- 0.40 | 92.12 +/- 0.40 | -0.88 | 0.2413 | 04:38:14 | 3 |
| Projection-LRB (proj-only) | k=0.65 | 91.48 +/- 0.43 | 91.47 +/- 0.43 | -0.23 | 0.2508 | 08:20:54 | complete sweep |
| Projection-LRB (proj-only) | k=0.9 | 91.63 +/- 0.70 | 91.62 +/- 0.70 | -0.38 | 0.2475 | 11:41:01 | complete sweep |
| Projection-LRB (proj-only) | k=0.5 | 91.51 +/- 0.41 | 91.51 +/- 0.41 | -0.27 | 0.2461 | 14:41:28 | legacy dedicated utility baseline |
| Full LRB | k=0.5 | 88.38 +/- 1.37 | 88.37 +/- 1.38 | +2.87 | 0.3354 | 14:38:34 | 3 |
| Mixup | alpha=0.3 | 90.98 +/- 0.07 | 90.98 +/- 0.07 | +0.27 | 0.2332 | 01:02:50 | 3 |

## DAGER Attack Sweep, Batch Size 2

These rows are useful for appendix material and for explaining why top-k/quantization must be treated as strong baselines.

### Noise / DP-SGD

| Defense | Param | Status | RecToken | R1+R2 |
| --- | --- | --- | --- | --- |
| noise | 1e-6 | ok | 0.9373 | 19.455 |
| noise | 1e-5 | ok | 0.7525 | 9.271 |
| noise | 1e-4 | ok | 0.4290 | 9.270 |
| Gaussian noise | sigma=5e-4 | no_l1_candidates | 0.2655 | 3.797 |
| noise | 1e-3 | no_l1_candidates | 0.2254 | 3.543 |
| dpsgd | 1e-6 | ok | 0.9780 | 12.384 |
| dpsgd | 1e-5 | ok | 0.8240 | 13.419 |
| dpsgd | 1e-4 | ok | 0.4953 | 10.127 |
| DP-SGD-style noise | sigma=5e-4 | no_l1_candidates | 0.2499 | 2.800 |
| dpsgd | 1e-3 | no_l1_candidates | 0.1344 | 1.933 |

### Top-k / Quantization

| Defense | Param | Status | RecToken | R1+R2 |
| --- | --- | --- | --- | --- |
| topk | 0.01 | no_l1_candidates | 0.0000 | 0.000 |
| topk | 0.05 | no_l1_candidates | 0.0000 | 0.000 |
| Top-k sparsification | ratio=0.1 | no_l1_candidates | 0.0000 | 0.000 |
| topk | 0.3 | no_l1_candidates | 0.0000 | 0.000 |
| topk | 0.5 | no_l1_candidates | 0.0419 | 1.752 |
| topk | 0.7 | no_l1_candidates | 0.1829 | 10.500 |
| topk | 0.9 | no_l1_candidates | 0.3455 | 33.900 |
| compression | 2 bits | no_l1_candidates | 0.0000 | 0.000 |
| compression | 4 bits | no_l1_candidates | 0.0000 | 0.000 |
| Quantization | 8 bits | no_l1_candidates | 0.0000 | 0.000 |
| compression | 16 bits | no_l1_candidates | 0.0000 | 0.000 |
| compression | 32 bits | ok | 0.8946 | 158.826 |

### Projection-LRB / Full LRB

| Defense | Param | Status | RecToken | R1+R2 | Evidence type |
| --- | --- | --- | --- | --- | --- |
| Projection-LRB (proj-only) | k=0.05 | no_l1_candidates | 0.0000 | 0.000 | attack-only |
| Projection-LRB (proj-only) | k=0.1 | no_l1_candidates | 0.0000 | 0.000 | attack-only |
| Projection-LRB (proj-only) | k=0.2 | no_l1_candidates | 0.0000 | 0.000 | attack-only |
| Projection-LRB (proj-only) | k=0.35 | no_l1_candidates | 0.0000 | 0.000 | attack-only |
| Projection-LRB (proj-only) | k=0.5 | no_l1_candidates | 0.0000 | 0.000 | attack-only + utility baseline |
| Projection-LRB (proj-only) | k=0.65 | no_l1_candidates | 0.0000 | 0.000 | complete sweep |
| Projection-LRB (proj-only) | k=0.9 | no_l1_candidates | 0.0000 | 0.000 | complete sweep |
| Full LRB | k=0.05 | no_l1_candidates | 0.0000 | 0.000 | attack-only |
| Full LRB | k=0.1 | no_l1_candidates | 0.0000 | 0.000 | attack-only |
| Full LRB | k=0.2 | no_l1_candidates | 0.0000 | 0.000 | attack-only |
| Full LRB | k=0.35 | no_l1_candidates | 0.0000 | 0.000 | attack-only |
| Full LRB | k=0.5 | no_l1_candidates | 0.0000 | 0.000 | full utility baseline |

### Mixup / Soteria

| Defense | Param | Status | RecToken | R1+R2 | Interpretation |
| --- | --- | --- | --- | --- | --- |
| mixup | alpha=0.1 | ok | 0.9244 | 164.838 | Not a privacy win here |
| mixup | alpha=0.3 | ok | 0.9478 | 174.831 | Not a privacy win here |
| mixup | alpha=0.5 | ok | 0.9590 | 172.559 | Not a privacy win here |
| mixup | alpha=1.0 | ok | 0.9348 | 169.245 | Not a privacy win here |
| mixup | alpha=2.0 | ok | 0.9336 | 169.480 | Not a privacy win here |
| soteria | 30 | ok | 1.0000 | 193.252 | Worse than clean under this DAGER setting |
| soteria | 50 | ok | 1.0000 | 193.439 | Worse than clean under this DAGER setting |
| soteria | 70 | ok | 1.0000 | 193.479 | Worse than clean under this DAGER setting |

## Batch-Size Auxiliary Evidence

These rows should not be merged into the main table because the attack surface changes with batch size, and batch 32 has many failed runs.

| Batch | Clean RecToken | Clean R1+R2 | Useful observations | Paper use |
| --- | --- | --- | --- | --- |
| 2 | 0.8966 | 158.996 | Main results are complete; Projection-LRB k=0.9 is the current complete frontier point | Main paper |
| 8 | 0.4093 | 35.378 | topk@0.01-0.3, full LRB@0.05-0.5, and lrbprojonly@0.05-0.5 all reach zero recovery; mixup@2.0 still leaks strongly | Appendix / robustness note |
| 32 | 0.0703 | 2.893 | topk@0.01-0.3 reaches zero recovery, but many rows fail or stop early; clean leakage is already much lower | Not main-table ready |

## Takeaways

- Clean SST-2 FedSGD updates leak strongly: RecToken `0.8966`, ROUGE-1+2 `158.996`.
- Projection-LRB/proj-only is not a `k=0.5` fluke. The complete `k=0.65` and `k=0.9` sweep points both reach `DAGER=0` with clean-level utility.
- The recommended main Projection-LRB row is now `k=0.9`: it is complete, has zero current DAGER recovery, and gives `91.63 +/- 0.70%` accuracy.
- Top-k@0.1 and quantization@8 bits remain strong empirical baselines with zero current DAGER recovery and slightly higher SST-2 utility than Projection-LRB. The paper claim should be mechanism-grounded effectiveness, not universal dominance.
- Full LRB@0.5 is best described as an over-defense reference: privacy is strong, but accuracy drops by `+2.87 pp` and runtime is high.
- Mixup and Soteria are poor privacy baselines in this setting.

## Files

- Integrated main tables: `docs/new_experiment_tables_20260516.md`
- Keep-ratio sweep detail: `docs/proj_only_keep_ratio_sweep_20260517.md`
- Main sweep CSV: `tmp/proj_only_keep_ratio_sweep_summary.csv`
- Strict train rows: `tmp/proj_only_keep_ratio_sweep_strict_train_rows.csv`
- Privacy rows: `tmp/proj_only_keep_ratio_sweep_privacy_rows.csv`
