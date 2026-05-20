# Paper Main Tables from Model-Matched Logs (updated 2026-05-17)

This file reorganizes the current non-`old` experiment logs according to the main-table requirements in `docs/聊天记录20260505.md`.

Primary inclusion rule:

- Main paper rows use model-matched checkpoints, especially `./models/gpt2_sst2_clean_num_epochs_2/final` for SST-2 and `./models/gpt2_cola_clean_num_epochs_2/final` for CoLA.
- Main SST-2 threat model is GPT-2, full-gradient DAGER, batch size 2, `n_inputs=100`.
- Utility numbers are 3-seed means when available. Projection keep-ratio sweep rows are marked `complete` only when privacy, utility, and proxy stages all finished for seeds `101/202/303`.
- `DAGER=0` means no recovery under the current attack budget and implementation. It is not a formal privacy guarantee.

## Table 1: Threat Model / Setup Summary

| Role | Dataset / model | Update / attack surface | Batch | Attack budget | Checkpoint | Utility seeds | Evidence status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Main setting | SST-2 / GPT-2 | Full gradients attacked by DAGER | 2 | `n_inputs=100` | `./models/gpt2_sst2_clean_num_epochs_2/final` | 101, 202, 303 | Complete for main SST-2 table |
| Cross-dataset check | CoLA / GPT-2 | Full gradients attacked by DAGER | 2 | `n_inputs=100` | `./models/gpt2_cola_clean_num_epochs_2/final` | 101, 202, 303 where available | Privacy mostly available; utility incomplete |
| Batch-size auxiliary | SST-2 / GPT-2 | Full gradients attacked by DAGER | 8 | `n_inputs=100` | `./models/gpt2_sst2_clean_num_epochs_2/final` | not included | Attack-only appendix evidence |
| Batch-size auxiliary | SST-2 / GPT-2 | Full gradients attacked by DAGER | 32 | `n_inputs=100` | `./models/gpt2_sst2_clean_num_epochs_2/final` | not included | Many failures; not main-table ready |
| Transfer target | PEFT / LoRA | Adapter/update leakage | n/a | n/a | n/a | n/a | Not verified in this batch |
| Transfer target | Partial gradients | Layer/block-level gradient leakage | n/a | n/a | n/a | n/a | Not verified in this batch |

## Table 2: Main Results under Full-Gradient DAGER

SST-2 / GPT-2 / batch size 2. Lower DAGER recovery is better. Negative accuracy drop means the run is slightly above the clean mean and should be described as clean-level utility, not a significant improvement.

| Method | Setting | DAGER RecToken | DAGER R1+R2 | Attack status | Accuracy (%) | Macro-F1 (%) | Acc. drop (pp) | Eval loss | Train time | Paper role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Clean FedSGD | none | 0.8966 | 158.996 | ok | 91.25 +/- 0.86 | 91.24 +/- 0.86 | +0.00 | 0.2479 | 00:41:20 | Leakage anchor |
| Gaussian noise | sigma=5e-4 | 0.2655 | 3.797 | no_l1_candidates | 91.63 +/- 0.98 | 91.63 +/- 0.98 | -0.38 | 0.2429 | 01:11:09 | Weak/noisy baseline |
| DP-SGD-style noise | sigma=5e-4 | 0.2499 | 2.800 | no_l1_candidates | 92.20 +/- 0.50 | 92.20 +/- 0.50 | -0.96 | 0.6539 | 01:30:04 | DP-style coverage, no accounting |
| Top-k sparsification | ratio=0.1 | 0.0000 | 0.000 | no_l1_candidates | 91.82 +/- 0.86 | 91.82 +/- 0.86 | -0.57 | 0.2362 | 02:43:41 | Strong empirical baseline |
| Quantization | 8 bits | 0.0000 | 0.000 | no_l1_candidates | 92.13 +/- 0.40 | 92.12 +/- 0.40 | -0.88 | 0.2413 | 04:38:14 | Strong empirical baseline |
| Projection-LRB (proj-only) | k=0.9 | 0.0000 | 0.000 | no_l1_candidates | 91.63 +/- 0.70 | 91.62 +/- 0.70 | -0.38 | 0.2475 | 11:41:01 | Best complete keep-ratio point |
| Full LRB | k=0.5 | 0.0000 | 0.000 | no_l1_candidates | 88.38 +/- 1.37 | 88.37 +/- 1.38 | +2.87 | 0.3354 | 14:38:34 | Over-defense reference |

### CoLA available evidence

CoLA supports the cross-dataset leakage story, but it is not yet a complete main table because utility is missing for top-k, quantization, Projection-LRB/proj-only, noise, mixup, and soteria.

| Method | Setting | DAGER RecToken | DAGER R1+R2 | Attack status | Accuracy (%) | Macro-F1 (%) | MCC (%) | Acc. drop (pp) | Paper role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Clean FedSGD | none | 0.9985 | 199.598 | ok | 74.88 +/- 0.75 | 62.78 +/- 2.26 | 34.44 +/- 2.54 | +0.00 | Cross-dataset leakage anchor |
| DP-SGD-style noise | sigma=5e-4 | 0.0859 | 0.167 | no_l1_candidates | 74.53 +/- 0.31 | 60.10 +/- 1.60 | 33.50 +/- 0.86 | +0.35 | Partial utility baseline |
| Full LRB | k=0.5 | 0.0000 | 0.000 | no_l1_candidates | 75.84 +/- 0.58 | 64.04 +/- 1.58 | 37.66 +/- 1.90 | -0.96 | Complete CoLA defense point |
| Top-k sparsification | ratio=0.1 | 0.0000 | 0.000 | no_l1_candidates | n/a | n/a | n/a | n/a | Privacy-only |
| Quantization | 8 bits | 0.0000 | 0.000 | no_l1_candidates | n/a | n/a | n/a | n/a | Privacy-only |
| Projection-LRB (proj-only) | k=0.5 | 0.0000 | 0.000 | no_l1_candidates | n/a | n/a | n/a | n/a | Privacy-only; utility needed |
| Gaussian noise | sigma=5e-4 | 0.8337 | 9.315 | ok | n/a | n/a | n/a | n/a | Weak privacy baseline |

## Table 3: Ablation Results

The new ablation evidence is a Projection-LRB/proj-only keep-ratio sweep. The useful paper claim is that `proj_only` is not a single-point accident at `k=0.5`: the complete points `k=0.65` and `k=0.9` also reach zero DAGER recovery with clean-level utility.

| Projection keep ratio | Privacy ok | Utility ok | Proxy ok | DAGER RecToken | DAGER R1+R2 | Accuracy (%) | Macro-F1 (%) | Acc. drop (pp) | Grad cosine | Norm retention | Status / role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| k=0.65 | 3/3 | 3/3 | 3/3 | 0.0000 | 0.000 | 91.48 +/- 0.43 | 91.47 +/- 0.43 | -0.23 | 0.5579 | 0.4118 | Complete, conservative operating point |
| k=0.9 | 3/3 | 3/3 | 3/3 | 0.0000 | 0.000 | 91.63 +/- 0.70 | 91.62 +/- 0.70 | -0.38 | 0.7009 | 0.5525 | Complete, current frontier endpoint |
| k=0.5 | 3/3 | 2/3 | 3/3 | 0.0000 | 0.000 | 91.46 +/- 0.57 | 91.45 +/- 0.57 | -0.21 | 0.5193 | 0.4109 | Partial in strict sweep; legacy anchor |
| k=0.75 | 3/3 | 2/3 | 3/3 | 0.0000 | 0.000 | 91.11 +/- 0.24 | 91.11 +/- 0.25 | +0.14 | 0.7579 | 0.5074 | Partial; one train failure |
| k=0.95 | 2/3 | 1/3 | 1/3 | 0.0000 | 0.000 | 91.05 +/- 0.00 | 91.05 +/- 0.00 | +0.20 | 0.7005 | 0.5677 | Boundary probe only |
| k=0.97 | 1/3 | 1/3 | 1/3 | 0.0000 | 0.000 | 91.51 +/- 0.00 | 91.51 +/- 0.00 | -0.26 | 0.7145 | 0.5569 | Boundary probe only |
| k=1.0 | 1/3 | 1/3 | 1/3 | 0.0000 | 0.000 | 92.20 +/- 0.00 | 92.20 +/- 0.00 | -0.95 | 0.7121 | 0.5552 | Boundary probe only |

Attack-only low-k checks from `defense_baselines_sst2_b2_gpt2_focus_lrbprojonly_20260514_143549` also show `lrbprojonly@k=0.05/0.1/0.2/0.35/0.5` all reaching `DAGER RecToken=0` and `R1+R2=0`, but those rows do not include matched 3-seed utility.

## Table 4: Transfer Results Across Attack Surfaces

This is the honest current transfer table. It should stay in the paper plan, but not be overstated until PEFT and partial-gradient rows are actually run.

| Attack surface | Dataset / model | Available evidence | Best current privacy observation | Utility status | Paper status |
| --- | --- | --- | --- | --- | --- |
| Full gradients | SST-2 / GPT-2 / batch 2 | Complete main table | Projection-LRB k=0.9, top-k@0.1, quantization@8, and full LRB@0.5 all reach `DAGER=0` | 3-seed utility available | Main evidence |
| Full gradients | CoLA / GPT-2 / batch 2 | Partial table | Clean leaks heavily; full LRB and attack-only Projection-LRB/top-k/quantization reach `DAGER=0` | Utility incomplete for Projection-LRB/top-k/quantization | Cross-dataset partial evidence |
| Full gradients | SST-2 / GPT-2 / batch 8 | Attack-only auxiliary | Clean RecToken drops to 0.4093; top-k@0.1 and lrbprojonly k=0.05-0.5 reach zero recovery; mixup@2.0 still leaks | No utility table | Appendix-only |
| Full gradients | SST-2 / GPT-2 / batch 32 | Noisy auxiliary | Clean RecToken is 0.0703; top-k@0.01-0.3 reaches zero recovery; many other rows fail | No utility table | Not main-table ready |
| PEFT / LoRA updates | n/a | No verified new logs | n/a | n/a | Required next experiment |
| Partial gradients | n/a | No verified new logs | n/a | n/a | Required next experiment |

## Table 5: Runtime / Cost Analysis

Times are wall-clock values from the current logs and implementation. They are useful for cost discussion, but should not be framed as optimized systems numbers.

| Method | Setting | Train time mean | DAGER attack time | Proxy step runtime | Cost interpretation |
| --- | --- | --- | --- | --- | --- |
| Clean FedSGD | none | 00:41:20 | 00:54:43 | n/a | Fast baseline, severe leakage |
| Gaussian noise | sigma=5e-4 | 01:11:09 | 03:00:45 | n/a | Moderate training overhead, incomplete privacy |
| DP-SGD-style noise | sigma=5e-4 | 01:30:04 | 03:51:13 | n/a | Higher loss and no formal accounting here |
| Top-k sparsification | ratio=0.1 | 02:43:41 | 01:58:27 | n/a | Strong empirical baseline with lower train cost than LRB |
| Quantization | 8 bits | 04:38:14 | 01:04:43 | n/a | Strong empirical baseline with good utility |
| Projection-LRB (proj-only) | k=0.9 | 11:41:01 | 00:52:46 | 1.354 s | Clean-level utility, but current training path is costly |
| Full LRB | k=0.5 | 14:38:34 | 05:27:36 | n/a | Strong privacy, but utility and cost make it an over-defense reference |

## Main Conclusions for the Paper

- Clean SST-2 and CoLA updates leak heavily under model-matched full-gradient DAGER.
- On SST-2, Projection-LRB/proj-only is no longer a `k=0.5` single-point result: complete `k=0.65` and `k=0.9` runs both reach zero current DAGER recovery with clean-level utility.
- `k=0.9` is the best complete Projection-LRB point in the current sweep and should replace `k=0.5` as the main Projection-LRB row in the SST-2 main table.
- Top-k@0.1 and quantization@8 remain strong empirical baselines. Current evidence supports a mechanism-grounded Projection-LRB claim, not universal dominance over compression/sparsification.
- Full LRB remains a useful over-defense reference: it blocks recovery but gives worse SST-2 utility and high cost.
- Transfer to PEFT/LoRA and partial-gradient leakage is still missing and should be presented as the next required evidence, not as a solved result.

## Remaining Gaps Before Camera-Ready Claims

- Run CoLA utility for Projection-LRB/proj-only, top-k, quantization, noise, mixup, and soteria.
- Re-run or stabilize incomplete high-k Projection-LRB probes (`k=0.95/0.97/1.0`) if the paper needs a finer Pareto boundary.
- Add PEFT/LoRA defense results and partial-gradient attack-surface results.
- Normalize runtime measurements if making strong systems-efficiency claims.
- Add at least one adaptive or transfer-attack check before claiming robustness beyond the current DAGER budget.

## Generated / Supporting Files

- `docs/proj_only_keep_ratio_sweep_20260517.md`
- `docs/sst2_results_table_20260516.md`
- `tmp/proj_only_keep_ratio_sweep_summary.csv`
- `tmp/proj_only_keep_ratio_sweep_strict_train_rows.csv`
- `tmp/proj_only_keep_ratio_sweep_privacy_rows.csv`
- `tmp/paper_table_sst2_privacy_utility.csv` (legacy main table with `proj_only@k=0.5`; superseded by the `k=0.9` row above)
- `tmp/paper_table_cola_privacy_utility_partial.csv`
