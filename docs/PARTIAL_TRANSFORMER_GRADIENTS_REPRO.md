# Partial Transformer Gradients Reproduction

This document records the FedLLM LAMP-lite reproduction of the core attack idea in
Seeing the Forest through the Trees: Data Leakage from Partial Transformer Gradients.
The goal is mechanism parity, not full paper-table parity.

References:

- ACL Anthology: https://aclanthology.org/2024.emnlp-main.275/
- Official code: https://github.com/weijun-l/partial-gradients-leakage

## PTG vs DAGER Partial Exposure

`attack.py` remains the DAGER span-check / candidate-decoding path. Its partial-gradient
variants should be described as `dager_prefix_visible`, `dager_nonprefix_visible`,
`dager_qkv_visible`, or `peft_adapter_visible`.

`attack_partial_gradient.py` is the independent PTG optimization path:

- `attack=partial_transformer_gradients`
- `partial_attack_variant=ptg_gradient_matching`
- `attack_surface=partial_gradient` in collected tables

PTG computes the real batch gradients, applies the selected FedLLM defense, keeps only
the requested Transformer layer/module gradients, and optimizes dummy input embeddings
so candidate gradients match those visible partial gradients. It does not call DAGER
span decomposition or DAGER candidate decoding.

PTG-only filters such as `query_only`, `key_only`, `value_only`, `attn_out_only`, and
`ffn_*` are intentionally rejected by the DAGER entrypoint and should be run through
`attack_partial_gradient.py`.

## Source-Parity Mechanics

`attack_partial_gradient.py` now has two modes:

- `--ptg_parity_mode fedllm`: the original FedLLM LAMP-lite adaptation, kept as the default for compatibility.
- `--ptg_parity_mode source`: source-code parity mode for the official implementation, including raw BERT word-embedding optimization, source selector names, source-style initialization/defaults, source loss names, optimizer/scheduler knobs, optional perplexity logging/evaluation, token swaps, padding-knowledge control, and ROUGE-1 batch matching.

The source selector aliases are:

- `--grad_type all_layers|encoder|layer_encoder|attn_qkv|attn_query|attn_key|attn_value|attn_output|ffn_fc|ffn_output|word_emb`
- `--attack_layer all|0,1,2`

For GPT-2, `attn_query`, `attn_key`, and `attn_value` all expose the packed `c_attn` tensor, so they are effectively `attn_qkv`.

Source-style attack knobs include:

- `--ptg_init random|lm`, `--ptg_init_candidates`, `--ptg_init_size`, `--ptg_init_permutation_trials`
- `--ptg_match_loss cos|dlg|tag|cosine|normalized_mse`
- `--ptg_optimizer adam|bfgs|bert-adam`
- `--ptg_lr_decay_type none|StepLR|LambdaLR`, `--ptg_lr_decay`, `--ptg_lr_max_it`, `--ptg_grad_clip`
- `--ptg_lm_model_path`, `--ptg_lm_prior_weight`
- `--ptg_use_swaps`, `--ptg_swap_steps`, `--ptg_swap_burnin`, `--ptg_swap_every`, `--ptg_use_swaps_at_end`
- `--ptg_know_padding` / `--no_ptg_know_padding`

The official-source aliases `--loss cos|dlg|tag`, `--n_steps`, `--init_candidates`, `--init`, `--init_size`, `--opt_alg`, `--lr`, `--lr_decay_type`, `--lr_decay`, `--tag_factor`, `--grad_clip`, `--lr_max_it`, and `--print_every` are accepted and mapped to the corresponding PTG knobs.

In `source` mode, unspecified knobs follow the official defaults: `n_steps=2000`, `init_candidates=500`, `init=random`, `use_swaps=True`, `init_size=1.4`, `lr=0.01`, `coeff_perplexity=0.1`, `coeff_reg=0.1`, `lr_decay_type=StepLR`, `print_every=50`, and `attack_layer=0`.

DP-SGD has two explicit modes:

- `--defense dpsgd --ptg_dpsgd_mode dpsgd_style`: FedLLM's per-example clipping + Gaussian-noise gradient transform.
- `--defense dpsgd --ptg_dpsgd_mode source_opacus`: official-source style Opacus training loop. This constructs a train dataloader, wraps the training model with `PrivacyEngine.make_private`, backpropagates each batch, extracts visible partial gradients from the private model's live `.grad` tensors, syncs weights into the attack model, and then runs reconstruction.

Use `--noise_multiplier` / `--clipping_bound` for source Opacus parity; these are also mirrored to `--defense_noise` / `--defense_clip_norm` in summaries. Without a privacy accountant, do not report a formal epsilon/delta DP guarantee.

## LAMP-lite Mechanics

The current reproduction implements:

- partial Transformer gradient selection by layer and module family;
- dummy input embedding optimization with per-tensor cosine gradient matching by default;
- optional `normalized_mse` gradient matching for debugging;
- fixed sequence length and attention mask from the real batch;
- fixed special-token and padding positions by default;
- nearest-token decoding with ignored special/pad tokens excluded from recovery metrics;
- embedding norm regularization and TV regularization;
- known-label mode and bounded label-search mode.

Language-model perplexity support and token-swap search are implemented for source-parity
runs. Following the official attack loop, the source main optimization closure uses
gradient reconstruction plus embedding-norm regularization; perplexity is used for
source-style logging/evaluation and swap/candidate scoring when an LM is supplied. These
extras remain disabled by default in `fedllm` mode unless their weights/flags are explicitly
supplied.

## CLI

Minimal BERT parity smoke:

```bash
python attack_partial_gradient.py \
  --dataset sst2 \
  --split val \
  --task seq_class \
  --batch_size 1 \
  --n_inputs 1 \
  --model_path bert-base-uncased \
  --finetuned_path ./models/bert-sst2 \
  --gradient_layer_subset all \
  --gradient_param_filter query_only \
  --ptg_steps 80 \
  --ptg_lr 0.1 \
  --ptg_restarts 1 \
  --ptg_match_loss cosine \
  --ptg_label_mode known \
  --ptg_embed_norm_weight 0.01 \
  --attn_implementation eager
```

Supported PTG exposure controls:

- `--gradient_layer_subset all|firstN|lastN|midN`
- `--gradient_param_filter all|qkv_only|query_only|key_only|value_only|attn_out_only|attn_only|ffn_in_only|ffn_out_only|ffn_only|classifier_only`

GPT-2 packs Q/K/V in `c_attn`, so `query_only`, `key_only`, and `value_only` expose the
same packed tensor and the summary reports `effective_gradient_param_filter=qkv_only`.
BERT/LLaMA split projections use precise module selectors.

PTG attack knobs:

- `--ptg_steps`, `--ptg_lr`, `--ptg_restarts`
- `--ptg_match_loss cosine|normalized_mse`
- `--ptg_label_mode known|search`
- `--ptg_decode_metric cos|l2`
- `--ptg_tv_weight`, `--ptg_embed_norm_weight`, `--ptg_entropy_weight`
- `--ptg_fix_special_tokens` / `--no_ptg_fix_special_tokens`
- reserved: `--ptg_lm_prior_weight`, `--ptg_swap_steps`

## Baseline Wrapper

Default matrix:

```bash
./scripts/ptg_baselines.sh sst2 1 bert-base-uncased 10 \
  --finetuned_path ./models/bert-sst2 \
  --ptg_steps 80 \
  --ptg_lr 0.1 \
  --ptg_embed_norm_weight 0.01
```

This runs:

- exposures: `first2`, `qkv_only`, `mid2`, `last2`
- defenses: `none`, `proj_only`, `topk`, `compression`

Focused row:

```bash
./scripts/ptg_baselines.sh sst2 1 bert-base-uncased 1 \
  --exposure query_only \
  --baseline_defense none \
  --finetuned_path ./models/bert-sst2 \
  --ptg_steps 40
```

Full module sweep:

```bash
./scripts/ptg_baselines.sh sst2 1 bert-base-uncased 10 \
  --full_sweep \
  --finetuned_path ./models/bert-sst2 \
  --ptg_steps 80
```

Outputs go to `log/runs/ptg_*`, with one log per variant plus `results.csv` and
`results.md`.

## Acceptance Fields

Every PTG result summary should include:

- `result_status`
- `selected_gradient_names`, `selected_gradient_count`
- `ptg_initial_loss`, `ptg_final_loss`, `ptg_loss_reduction`
- `fixed_token_count`
- `rec_token_mean`
- `agg_rouge1_fm`, `agg_rouge2_fm`, `agg_r1fm_r2fm`
- runtime fields such as `last_input_time` and `last_total_time`

Pilot acceptance:

- `n_inputs=1` smoke emits a complete summary.
- `n_inputs=10` pilot records how often `ptg_final_loss < ptg_initial_loss` and whether decoded text is non-empty.
- Formal tables should cover at least `none/proj_only/topk/compression` under `first2` and `qkv_only`.

## Paper Wording

Before full parity:

> We implement a FedLLM LAMP-lite adaptation of partial Transformer gradient leakage,
> where the attacker observes only a selected layer/module subset of gradients and
> optimizes dummy input embeddings to match the visible gradients.

After BERT parity smoke:

> This reproduces the core mechanism of partial-gradient leakage: local Transformer
> gradients alone can drive gradient-matching reconstruction, without requiring
> full-model gradients.

Avoid claiming full paper-number reproduction, causal-LM next-token coverage, or that
Projection-LRB universally beats top-k/compression in all partial-gradient settings.
