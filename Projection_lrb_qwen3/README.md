# Qwen3 SST-2 preregistration scaffold

This directory is the isolated workspace for the Qwen3-1.7B-Base SST-2
Projection-LRB study.  It contains a narrow, defense-unaware Qwen3/RoPE DAGER
implementation for `defense=none` only; there is still **no LRB application,
defended-gradient path, training loop, PEFT path, or federated aggregation
code**.

Implemented capabilities are deterministic preregistration, one strict Qwen3
single-sample classification gradient diagnostic, and a manifest-only
`dager_qwen3_rope_defense_unaware` attack for the unmodified gradient.

## Scope and protocol

- Model: `models/Qwen3-1.7B-Base`, verified as a local Qwen3 model.
- Dataset: a local `datasets.DatasetDict` saved at `dataset_path`; the runner
  reads `DatasetDict["validation"]` only and never calls `load_dataset` or a
  network backend.  The split must contain `idx`, `sentence`, and `label`.
- Tokenization: Qwen3 tokenizer with `add_special_tokens=False`; text tokens
  are truncated to `max_length - 1`, then one EOS token is explicitly
  appended.  Version 1 requires `max_length == 32`.
- Eligibility: blank text and examples with fewer than
  `min_effective_token_length` non-EOS tokens are excluded.
- Ordering: eligible examples are sorted by
  `SHA256("glue|sst2|validation|{original_index}|{sentence}|{label}")`.
- Split: first 20 calibration, next 5 smoke, next 20 final.  The three sets
  must be disjoint.
- A gradient diagnostic creates a random classification head only with an
  explicit seed and `N(0, 1e-3)` initialization.  Its output is
  attack-transfer diagnostics only, never utility evidence.

The protocol is empirical and attack-specific.  A future zero-recovery result
must not be described as information deletion, formal privacy, or general
adaptive robustness.

## Configuration

Edit `configs/experiment.json` before running.  In particular,
`calibration_head_seed` and `smoke_head_seed` are intentionally `null` in the
checked-in template.  The command fails until both are explicitly filled with
integers distinct from each other and from `[101, 202, 303]`; it never invents
seeds.

All paths are resolved from the repository root.  `output_root` is required to
resolve to this directory's `outputs/` folder, so Qwen3 artifacts cannot write
into legacy GPT-2 locations.

## Run

From the repository root, after filling the two seeds and ensuring the local
model and the GLUE SST-2 validation split are available:

```powershell
python Projection_lrb_qwen3/scripts/preregister_experiment.py --config Projection_lrb_qwen3/configs/experiment.json
```

The equivalent Linux command is identical.  Use `--help` for CLI details.
The command is recoverable: if a prior manifest has the same deterministic
preregistration identity, matching files are retained; any inconsistency is a
hard error.

It creates or verifies:

```text
manifests/preregistration.json
manifests/calibration.jsonl
manifests/smoke.jsonl
manifests/final.jsonl
outputs/calibration/
outputs/smoke/
outputs/final/
outputs/utility/
```

`preregistration.json` contains the full configuration, its canonical SHA-256,
model/tokenizer key-file hashes, per-stage sample-list hashes, intersection
checks, creation time, Git commit, and Python/torch/transformers versions.
Its `preregistration_sha256` deliberately excludes the creation timestamp, so
two equivalent runs have the same protocol identity.

## Qwen3 gradient diagnostic

This CUDA-only command loads the local Qwen3 model through
`AutoModelForSequenceClassification`, uses BF16 by default (or true FP32 with
`--dtype float32`),
initializes `model.score` from the required explicit head seed, and runs one
unpadded batch-size-one step.  It hooks the true inputs to exactly
`model.model.layers[0|1].self_attn.q_proj`, writes the canonical
`named_parameters()` gradient manifest, captures q_proj output gradients, and
checks the FP32 linear-layer identity `G = Delta.T @ H`.

```bash
python Projection_lrb_qwen3/scripts/check_qwen3_gradient.py \
  --sentence "a moving and funny film" --label 1 --head-seed 404
```

The primary row-space basis is always the right-singular basis of raw `G`.
`G.T` is computed only as a fixed negative control and is never selected as a
repair.  `--rank-tol` is a relative tolerance, while `--rank-atol` is recorded
only as a BF16-noise diagnostic.  Per-token residual acceptance is restricted
to positions active under the predeclared Delta-norm relative rule; inactive
positions remain reported but cannot be used to force a direction failure.

The attack-adjacent runners use fixed active-token diagnostic residual
tolerances of `2e-4` for FP32 and `7.5e-3` for BF16.  The BF16 value is fixed
by the immutable 20-sample structural profile recorded in
`prereg_amendments/bf16_gradient_gate_pre_attack_075/`; it is not derived from
Layer-1 candidate selection, Layer-2 decoding, ROUGE, or LRB.  These
tolerances are solely for architecture and numerical-precision diagnostics;
they are not DAGER `tau1`/`tau2` values and do not take part in
attack-configuration calibration.  The standalone `check_qwen3_gradient.py`
keeps its explicit `--max-relative-residual` CLI control so the fixed profile
can be reproduced at each candidate gate.

Any failed identity, relative-rank-cap, active-token residual, finite-value,
or fixed-negative-control check writes the complete diagnostic JSON with
`status="failed_gradient_diagnostic"` and exits nonzero; it does not continue
to attack code.

## None-only Qwen3/RoPE DAGER

`scripts/run_none_attack.py` accepts no free-text argument.  It selects one
immutable sample by `stage` and `sample_key` from the expected preregistration
JSONL, verifies the configuration and sample-list hashes, and requires a head
seed registered for that stage.  Its fixed attack label is
`dager_qwen3_rope_defense_unaware`; `--defense` only accepts `none`.

The first layer uses the raw Qwen3 `nn.Linear` q_proj gradient shape
`[d_out, d_in]` directly.  Its DAGER basis has the legacy `[rank, feature]`
layout and is obtained from the raw gradient's right-singular space; it never
applies the GPT-2 `Conv1D` transpose.  Token candidates are the actual inputs
to `model.model.layers[0].self_attn.q_proj`, obtained from the local embedding
table and Qwen3's native RMSNorm.  The vocabulary is scanned in bounded chunks
whose size comes from the predeclared `attack_budget.parallel` field.
For Stage 4, the existing configuration key `rank_tol` is explicitly a
**relative rank tolerance**.  For each raw, untransposed q_proj gradient, the
attack computes `torch.linalg.svdvals(gradient.detach().float())`, sets
`relative_threshold = largest_singular_value * rank_tol`, and counts singular
values greater than or equal to that threshold.  The requested shared rank is
`max(q0_effective_rank, q1_effective_rank)`; it never uses captured `H`,
`Delta`, true text, or a theoretical diagnostic rank cap.

The legacy `feature_dim - rank_cutoff` and matrix-dimension limits may still
constrain basis extraction.  Such a constraint is not silent: every JSONL row
records `requested_shared_rank`, `applied_shared_rank`, `rank_cap`,
`rank_was_capped`, and `cap_reason`, alongside the per-layer relative
thresholds and effective ranks.  `get_layer_decomp` receives only that explicit
applied rank to extract the legacy `[rank, feature]` basis.

For layer 2, prefix representations are produced by the native Qwen3 first
decoder-layer forward with the model's own causal-mask helper, `position_ids`,
and native RoPE position embeddings.  It does not approximate RoPE.  The
decoder exhaustively retains only prefixes whose every layer-1 q_proj input
passes the existing DAGER span threshold.  No language-model prior, beam,
semantic filter, ground-truth-based choice, or defense-specific branch is
introduced.  `attack_budget.max_ids` is the existing distance-sorted decoder
candidate cap; `attack_budget.maxC` is the fixed prefix-search budget.

Example (the sample key must be copied from the preregistered manifest):

```bash
python Projection_lrb_qwen3/scripts/run_none_attack.py \
  --stage smoke \
  --sample-key '<64-character sample_key>' \
  --head-seed '<registered smoke seed>' \
  --defense none \
  --device cuda \
  --dtype bfloat16 \
  --output Projection_lrb_qwen3/outputs/smoke/qwen3_none_attack.jsonl
```

Before resolving the config or loading a model, the runner performs a legacy
ROUGE preflight.  It force-sets `HF_DATASETS_OFFLINE=1` and
`HF_HUB_OFFLINE=1`, requires `datasets.load_metric` to exist, and loads only
the cached `datasets.load_metric('rouge')` metric.  A fixed exact-match
self-test must return ROUGE-1 and ROUGE-2 of exactly `1.0`; any missing cache,
API, metric script, or self-test failure exits nonzero.  It never downloads a
metric and never falls back to `evaluate` or a custom ROUGE implementation.

The single JSONL record includes immutable sample/head provenance, true and
reconstructed token ids/text, aligned token recovery, exact recovery, the same
legacy ROUGE-1/ROUGE-2 definition as existing DAGER scripts, and
`legacy_rouge_backend` provenance (backend identifier, `datasets` version,
cached metric-script SHA-256, forced-offline values, and self-test result), as
well as empty-reconstruction state, layer-1 candidate counts, both DAGER
ranks, chunk timing, fixed thresholds/budgets, gradient diagnostics, and the
final search status.  Ground truth is read only after decoding to report
metrics; it is not passed to layer filtering or sequence recovery.

Run the isolated tests with third-party pytest autoload disabled:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q Projection_lrb_qwen3/tests
```

## Tests

The tests use a fake tokenizer and a temporary `DatasetDict.save_to_disk()`
artifact; they never download the model or dataset.

```powershell
python -m unittest discover -s Projection_lrb_qwen3/tests -v
```

On the Linux experiment server, use the pytest plugin-isolation setting for
the same no-network tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q Projection_lrb_qwen3/tests
```

## Future boundary

Future attack code must first apply formal LRB to the complete canonical
named-gradient tuple and only then select the two named gradients
`model.layers.0.self_attn.q_proj.weight` and
`model.layers.1.self_attn.q_proj.weight`.  It must retain tuple-index-derived
LRB seeds and use defense-unaware decoding for the standard attack.
