# Qwen3 SST-2 preregistration scaffold

This directory is the isolated workspace for the Qwen3-1.7B-Base SST-2
Projection-LRB study.  It deliberately contains **no DAGER implementation,
attack run, training loop, PEFT path, or federated aggregation code** yet.

Implemented capabilities are deterministic preregistration and one strict
Qwen3 single-sample classification gradient diagnostic.  The diagnostic is a
model-structure and row-space validation gate, not DAGER or an attack.

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

Any failed identity, relative-rank-cap, active-token residual, finite-value,
or fixed-negative-control check writes the complete diagnostic JSON with
`status="failed_gradient_diagnostic"` and exits nonzero; it does not continue
to attack code.

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
