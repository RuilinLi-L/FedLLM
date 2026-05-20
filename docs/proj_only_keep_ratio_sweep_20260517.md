# Projection-LRB proj_only keep-ratio sweep (SST-2)

Source: `log/runs/lrb_ablation_sst2_b2_gpt2_k0.*_20260510_*` and the new high-k follow-ups `k=0.95/0.97/1.0` from `20260517`. Setting: SST-2, GPT-2, batch size 2, `n_inputs=100`, full-gradient DAGER, checkpoint `./models/gpt2_sst2_clean_num_epochs_2/final`, seeds `101/202/303`.

## Main sweep table

Lower DAGER recovery is better; higher accuracy/macro-F1 is better. `complete` means all three seeds finished successfully for the three stages in that directory. `partial` means this is only a follow-up probe and should not be treated as a stable operating point.

| Projection keep ratio | Privacy ok | Utility ok | Proxy ok | DAGER RecToken | DAGER R1+R2 | Accuracy (%) | Macro-F1 (%) | Acc. drop (pp) | Eval loss | Train time | Attack time | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| k=0.65 | 3/3 | 3/3 | 3/3 | 0.0000 | 0.000 | 91.48 +/- 0.43 | 91.47 +/- 0.43 | -0.23 | 0.2508 | 08:20:54 | 00:56:56 | complete |
| k=0.9 | 3/3 | 3/3 | 3/3 | 0.0000 | 0.000 | 91.63 +/- 0.70 | 91.62 +/- 0.70 | -0.38 | 0.2475 | 11:41:01 | 00:52:46 | complete |
| k=0.5 | 3/3 | 2/3 | 3/3 | 0.0000 | 0.000 | 91.46 +/- 0.57 | 91.45 +/- 0.57 | -0.21 | 0.2427 | 12:03:18 | 01:11:54 | partial |
| k=0.75 | 3/3 | 2/3 | 3/3 | 0.0000 | 0.000 | 91.11 +/- 0.24 | 91.11 +/- 0.25 | +0.14 | 0.2396 | 07:14:49 | 01:13:26 | partial |
| k=0.95 | 2/3 | 1/3 | 1/3 | 0.0000 | 0.000 | 91.05 +/- 0.00 | 91.05 +/- 0.00 | +0.20 | 0.2530 | 11:10:43 | 01:22:57 | partial |
| k=0.97 | 1/3 | 1/3 | 1/3 | 0.0000 | 0.000 | 91.51 +/- 0.00 | 91.51 +/- 0.00 | -0.26 | 0.2518 | 11:55:19 | 02:21:57 | partial |
| k=1.0 | 1/3 | 1/3 | 1/3 | 0.0000 | 0.000 | 92.20 +/- 0.00 | 92.20 +/- 0.00 | -0.95 | 0.2438 | 11:56:36 | 01:46:09 | partial |

## Proxy and practical role

| Projection keep ratio | Grad cosine | Norm retention | Proxy step runtime (s) | Interpretation |
| --- | --- | --- | --- | --- |
| k=0.5 | 0.5193 | 0.4109 | 1.465 | legacy anchor; privacy robust, but utility evidence is 2/3 because one train run was killed |
| k=0.65 | 0.5579 | 0.4118 | 2.660 | best complete point among the original full sweep; clean-level utility |
| k=0.75 | 0.7579 | 0.5074 | 1.986 | complete privacy, but one train failure and lower utility than 0.65/0.9 |
| k=0.9 | 0.7009 | 0.5525 | 1.354 | best complete privacy-utility point in the tested grid; current frontier endpoint |
| k=0.95 | 0.7005 | 0.5677 | 2.396 | high-k follow-up; partial run, useful as a boundary probe only |
| k=0.97 | 0.7145 | 0.5569 | 2.604 | high-k follow-up; partial run, useful as a boundary probe only |
| k=1.0 | 0.7121 | 0.5552 | 2.173 | high-k follow-up; partial run, useful as a boundary probe only |

## Integrated SST2 privacy-utility table

This table folds the new sweep back into the current SST2 story. The Projection-LRB row now uses the best complete point from the sweep (`k=0.9`).

| Method | Setting | DAGER RecToken | DAGER R1+R2 | Accuracy (%) | Macro-F1 (%) | Acc. drop (pp) | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Clean FedSGD | none | 0.8966 | 158.996 | 91.25 +/- 0.86 | 91.24 +/- 0.86 | +0.00 | from previous model-matched SST2 table |
| Gaussian noise | sigma=5e-4 | 0.2655 | 3.797 | 91.63 +/- 0.98 | 91.63 +/- 0.98 | -0.38 | strong empirical baseline |
| DP-SGD-style noise | sigma=5e-4 | 0.2499 | 2.800 | 92.20 +/- 0.50 | 92.20 +/- 0.50 | -0.96 | strong empirical baseline |
| Top-k sparsification | ratio=0.1 | 0.0000 | 0.000 | 91.82 +/- 0.86 | 91.82 +/- 0.86 | -0.57 | strong empirical baseline |
| Quantization | 8 bits | 0.0000 | 0.000 | 92.13 +/- 0.40 | 92.12 +/- 0.40 | -0.88 | strong empirical baseline |
| Projection-LRB (proj-only) | k=0.9 | 0.0000 | 0.000 | 91.63 +/- 0.70 | 91.62 +/- 0.70 | -0.38 | best complete point in the keep-ratio sweep |
| Full LRB | k=0.5 | 0.0000 | 0.000 | 88.38 +/- 1.37 | 88.37 +/- 1.38 | +2.87 | over-defense reference |

## Utility-only table

| Method | Setting | Accuracy (%) | Macro-F1 (%) | Acc. drop (pp) | Eval loss | Train time | Runs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Clean FedSGD | none | 91.25 +/- 0.86 | 91.24 +/- 0.86 | +0.00 | 0.2479 | 00:41:20 | 3 |
| Gaussian noise | sigma=5e-4 | 91.63 +/- 0.98 | 91.63 +/- 0.98 | -0.38 | 0.2429 | 01:11:09 | 3 |
| DP-SGD-style noise | sigma=5e-4 | 92.20 +/- 0.50 | 92.20 +/- 0.50 | -0.96 | 0.6539 | 01:30:04 | 3 |
| Top-k sparsification | ratio=0.1 | 91.82 +/- 0.86 | 91.82 +/- 0.86 | -0.57 | 0.2362 | 02:43:41 | 3 |
| Quantization | 8 bits | 92.13 +/- 0.40 | 92.12 +/- 0.40 | -0.88 | 0.2413 | 04:38:14 | 3 |
| Projection-LRB (proj-only) | k=0.9 | 91.63 +/- 0.70 | 91.62 +/- 0.70 | -0.38 | 0.2475 | 11:41:01 | 3 |
| Full LRB | k=0.5 | 88.38 +/- 1.37 | 88.37 +/- 1.38 | +2.87 | 0.3354 | 14:38:34 | 3 |
| Mixup | alpha=0.3 | 90.98 +/- 0.07 | 90.98 +/- 0.07 | +0.27 | 0.2332 | 01:02:50 | 3 |

## Conclusions

- The new sweep shows that `proj_only` is not a `k=0.5` fluke: the complete points `k=0.65` and `k=0.9` both achieve `DAGER=0` with clean-level utility, and the full complete sweep remains zero-recovery across `k=0.5/0.65/0.75/0.9`.
- Among the complete runs, `k=0.9` is the current privacy-utility frontier endpoint. The higher-`k` follow-ups (`0.95/0.97/1.0`) are only partial probes, so they do not yet move the frontier.
- `k=0.65` is the cleanest conservative point if you want a more cautious operating point in the paper text; `k=0.9` is the best utility among the complete runs.
- The correct paper claim is mechanism-grounded robustness across a wide projection keep-ratio range, not superiority over strong compression baselines. Top-k@0.1 and quantization@8 bits still have zero DAGER recovery and slightly higher SST2 utility in the current model-matched table.

## High-k follow-up rows

| k | Successful seed(s) | DAGER RecToken | Accuracy (%) | Train time | Attack time | Status |
| --- | --- | --- | --- | --- | --- | --- |
| k=0.95 | 101, 202 | 0.0000 | 91.05 +/- 0.00 | 11:10:43 | 01:22:57 | partial |
| k=0.97 | 101 | 0.0000 | 91.51 +/- 0.00 | 11:55:19 | 02:21:57 | partial |
| k=1.0 | 101 | 0.0000 | 92.20 +/- 0.00 | 11:56:36 | 01:46:09 | partial |

## Per-seed privacy rows

| k | Seed | Exit | Status | Rec status | N | RecToken | R1+R2 | Attack time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| k=0.5 | 101 | 0 | ok | no_l1_candidates | 100 | 0.0000 | n/a | 1:21:25 |
| k=0.5 | 202 | 0 | ok | no_l1_candidates | 100 | 0.0000 | n/a | 1:03:41 |
| k=0.5 | 303 | 0 | ok | no_l1_candidates | 100 | 0.0000 | n/a | 1:10:36 |
| k=0.65 | 101 | 0 | ok | no_l1_candidates | 100 | 0.0000 | n/a | 0:45:11 |
| k=0.65 | 202 | 0 | ok | no_l1_candidates | 100 | 0.0000 | n/a | 0:56:55 |
| k=0.65 | 303 | 0 | ok | no_l1_candidates | 100 | 0.0000 | n/a | 1:08:41 |
| k=0.75 | 101 | 0 | ok | no_l1_candidates | 100 | 0.0000 | n/a | 0:50:20 |
| k=0.75 | 202 | 0 | ok | no_l1_candidates | 100 | 0.0000 | n/a | 1:12:51 |
| k=0.75 | 303 | 0 | ok | no_l1_candidates | 100 | 0.0000 | n/a | 1:37:08 |
| k=0.9 | 101 | 0 | ok | no_l1_candidates | 100 | 0.0000 | n/a | 0:49:35 |
| k=0.9 | 202 | 0 | ok | no_l1_candidates | 100 | 0.0000 | n/a | 0:51:01 |
| k=0.9 | 303 | 0 | ok | no_l1_candidates | 100 | 0.0000 | n/a | 0:57:41 |
| k=0.95 | 101 | 0 | ok | no_l1_candidates | 100 | 0.0000 | n/a | 1:49:19 |
| k=0.95 | 202 | 0 | ok | no_l1_candidates | 100 | 0.0000 | n/a | 0:56:35 |
| k=0.95 | 303 | 1 | failed | n/a | 0 | n/a | n/a | 0:00:43 |
| k=0.97 | 101 | 0 | ok | no_l1_candidates | 100 | 0.0000 | n/a | 2:21:57 |
| k=0.97 | 202 | 1 | failed | n/a | 0 | n/a | n/a | 0:00:51 |
| k=0.97 | 303 | 1 | failed | n/a | 0 | n/a | n/a | 0:00:47 |
| k=1.0 | 101 | 0 | ok | no_l1_candidates | 100 | 0.0000 | n/a | 1:46:09 |
| k=1.0 | 202 | 143 | failed | n/a | n/a | 0.0000 | n/a | n/a |
| k=1.0 | 303 | 1 | failed | n/a | 0 | n/a | n/a | 0:00:42 |

## Per-seed utility rows

| k | Seed | Exit | Status | Steps | Accuracy (%) | Macro-F1 (%) | Eval loss | Train time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| k=0.5 | 101 | 0 | ok | 33675 | 91.05 | 91.05 | 0.2507 | 18:03:17 |
| k=0.5 | 202 | 137 | failed | n/a | n/a | n/a | n/a | n/a |
| k=0.5 | 303 | 0 | ok | 33675 | 91.86 | 91.85 | 0.2348 | 06:03:19 |
| k=0.65 | 101 | 0 | ok | 33675 | 91.28 | 91.28 | 0.2620 | 08:53:28 |
| k=0.65 | 202 | 0 | ok | 33675 | 91.97 | 91.97 | 0.2445 | 10:01:38 |
| k=0.65 | 303 | 0 | ok | 33675 | 91.17 | 91.17 | 0.2460 | 06:07:36 |
| k=0.75 | 101 | 137 | failed | n/a | n/a | n/a | n/a | n/a |
| k=0.75 | 202 | 0 | ok | 33675 | 91.28 | 91.28 | 0.2397 | 06:19:29 |
| k=0.75 | 303 | 0 | ok | 33675 | 90.94 | 90.94 | 0.2394 | 08:10:09 |
| k=0.9 | 101 | 0 | ok | 33675 | 91.28 | 91.28 | 0.2653 | 18:25:22 |
| k=0.9 | 202 | 0 | ok | 33675 | 91.17 | 91.16 | 0.2484 | 08:31:35 |
| k=0.9 | 303 | 0 | ok | 33675 | 92.43 | 92.43 | 0.2288 | 08:06:05 |
| k=0.95 | 101 | 0 | ok | 33675 | 91.05 | 91.05 | 0.2530 | 11:10:43 |
| k=0.95 | 202 | 1 | failed | 0 | n/a | n/a | n/a | n/a |
| k=0.95 | 303 | 1 | failed | 0 | n/a | n/a | n/a | n/a |
| k=0.97 | 101 | 0 | ok | 33675 | 91.51 | 91.51 | 0.2518 | 11:55:19 |
| k=0.97 | 202 | 1 | failed | 0 | n/a | n/a | n/a | n/a |
| k=0.97 | 303 | 1 | failed | 0 | n/a | n/a | n/a | n/a |
| k=1.0 | 101 | 0 | ok | 33675 | 92.20 | 92.20 | 0.2438 | 11:56:36 |
| k=1.0 | 202 | 143 | failed | n/a | n/a | n/a | n/a | n/a |
| k=1.0 | 303 | 1 | failed | 0 | n/a | n/a | n/a | n/a |

## Generated files

- `tmp/proj_only_keep_ratio_sweep_summary.csv`
- `tmp/proj_only_keep_ratio_sweep_strict_train_rows.csv`
- `tmp/proj_only_keep_ratio_sweep_privacy_rows.csv`
