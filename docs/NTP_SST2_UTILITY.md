# GPT-2 / SST-2 Next-Token Prediction Utility

This experiment measures whether Projection-LRB/SLR preserves causal language-modeling utility. It is separate from the completed full-gradient DAGER privacy experiment; `attack.py` and DAGER reconstruction are not part of this protocol.

## Fixed protocol

- Every formal run independently loads the original Hugging Face pretrained `gpt2` checkpoint with `AutoModelForCausalLM.from_pretrained("gpt2")`.
- Classification checkpoints, the existing two-epoch classifier, NTP-finetuned checkpoints, and checkpoints from other runs are never used for initialization.
- The data are the official GLUE SST-2 `train` and `validation` partitions. SST-2 sentiment labels are removed and ignored; no class text, class IDs, or prompts are added.
- Training is standard causal language modeling. Labels copy `input_ids`, padding targets become `-100`, and the Hugging Face model performs its own causal shift.
- Training lasts 2 epochs with AdamW, learning rate `5e-5`, a linear schedule, and zero warmup steps.
- The utility-training minibatch size is **1**, matching the direct `train.py` default. This is independent of the DAGER attack batch size.
- The formal utility metrics are validation NLL and perplexity. NLL is weighted by the number of valid shifted target tokens, and `PPL = exp(NLL)` with overflow mapped to infinity.
- The untouched pretrained GPT-2 validation NLL/PPL is evaluated before fine-tuning as a single reference. Epoch-1 values remain in each run log; epoch-2 values are used for aggregation.
- None and SLR each use seeds 101, 202, and 303. Matched runs use identical pretrained initialization, tokenizer, data-shuffle seed, optimizer, schedule, batch size, and epoch count.
- SLR uses the existing `lrbprojonly` defense with the layer-wise scheduler endpoint `rho*=0.5` and `static` seed mode. It is not `proj_uniform`. The transformed gradients are written back before every `optimizer.step()` in both epochs.

## Six independent formal runs

Each command runs one experiment, saves the final checkpoint below `outputs/ntp_sst2_gpt2/`, and writes complete stdout/stderr to the named file below `log/runs/ntp_sst2_utility/`.

```bash
bash scripts/run_ntp_sst2_utility.sh none 101
bash scripts/run_ntp_sst2_utility.sh none 202
bash scripts/run_ntp_sst2_utility.sh none 303
bash scripts/run_ntp_sst2_utility.sh slr_rho05 101
bash scripts/run_ntp_sst2_utility.sh slr_rho05 202
bash scripts/run_ntp_sst2_utility.sh slr_rho05 303
```

The logs are respectively:

```text
log/runs/ntp_sst2_utility/none_seed101.txt
log/runs/ntp_sst2_utility/none_seed202.txt
log/runs/ntp_sst2_utility/none_seed303.txt
log/runs/ntp_sst2_utility/slr_rho05_seed101.txt
log/runs/ntp_sst2_utility/slr_rho05_seed202.txt
log/runs/ntp_sst2_utility/slr_rho05_seed303.txt
```

## Smoke tests

These commands exercise four examples from each source partition, use CPU when CUDA is unavailable, skip final checkpoint saving, and do not alter the formal outputs.

```bash
python train.py --dataset sst2 --task next_token_pred --model_path gpt2 --train_method full --batch_size 1 --num_epochs 1 --save_every 0 --rng_seed 101 --defense none --max_train_samples 4 --max_eval_samples 4 --skip_final_save --output_dir outputs/ntp_sst2_gpt2_smoke/none_seed101 --log_file log/runs/ntp_sst2_utility_smoke/none_seed101.txt

python train.py --dataset sst2 --task next_token_pred --model_path gpt2 --train_method full --batch_size 1 --num_epochs 1 --save_every 0 --rng_seed 101 --defense lrbprojonly --defense_lrb_keep_ratio_sensitive 0.5 --defense_lrb_seed_mode static --max_train_samples 4 --max_eval_samples 4 --skip_final_save --output_dir outputs/ntp_sst2_gpt2_smoke/slr_rho05_seed101 --log_file log/runs/ntp_sst2_utility_smoke/slr_rho05_seed101.txt
```

## Aggregation

Collect the six logs with:

```bash
python scripts/collect_experiment_logs.py log/runs/ntp_sst2_utility --utility-output log/runs/ntp_sst2_utility/utility_results.csv --utility-markdown log/runs/ntp_sst2_utility/utility_results.md
```

Utility aggregation keys include `task`, so `seq_class` and `next_token_pred` cannot share an aggregate row. The three epoch-2 `val_nll` and `val_perplexity` values are summarized with their mean and sample standard deviation (`ddof=1`). The pretrained reference is retained as one deterministic reference value rather than treated as three training measurements.
