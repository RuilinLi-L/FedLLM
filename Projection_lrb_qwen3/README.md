# Qwen3 SST-2 preregistration scaffold

This directory is the isolated workspace for the Qwen3-1.7B-Base SST-2
Projection-LRB study.  It deliberately contains **no DAGER implementation,
attack run, training loop, PEFT path, or federated aggregation code** yet.

The only implemented capability is deterministic preregistration of the
official GLUE SST-2 validation examples before any attack is allowed to run.

## Scope and protocol

- Model: `models/Qwen3-1.7B-Base`, verified as a local Qwen3 model.
- Dataset: `load_dataset("glue", "sst2", split="validation")` only.
- Tokenization: Qwen3 tokenizer with `add_special_tokens=False`; text tokens
  are truncated to `max_length - 1`, then one EOS token is explicitly
  appended.  Version 1 requires `max_length == 32`.
- Eligibility: blank text and examples with fewer than
  `min_effective_token_length` non-EOS tokens are excluded.
- Ordering: eligible examples are sorted by
  `SHA256("glue|sst2|validation|{original_index}|{sentence}|{label}")`.
- Split: first 20 calibration, next 5 smoke, next 20 final.  The three sets
  must be disjoint.
- Classification-head seeds are preregistered metadata only.  No random head
  is created by this scaffold, and such a head must never be used as utility
  evidence.

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

## Tests

The tests use a fake tokenizer and fake SST-2 loader; they do not download the
model or dataset.

```powershell
python -m unittest discover -s Projection_lrb_qwen3/tests -v
```

## Future boundary

Future attack code must first apply formal LRB to the complete canonical
named-gradient tuple and only then select the two named gradients
`model.layers.0.self_attn.q_proj.weight` and
`model.layers.1.self_attn.q_proj.weight`.  It must retain tuple-index-derived
LRB seeds and use defense-unaware decoding for the standard attack.
