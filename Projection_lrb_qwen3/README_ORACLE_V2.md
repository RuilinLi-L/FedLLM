# Qwen3 oracle v2

This is an isolated SST-2 full-parameter training and exact-state Projection-LRB
experiment. Historical Qwen/GPT-2 controls and output directories are unchanged.
The implementation uses the current raw `nn.Linear` q_proj orientation, absolute
rank tolerance, and native Qwen prefix forward. It is an empirical evaluation of
this attack, not a formal privacy guarantee.

## Environment and artifacts

Run on A100 using `/home/mcxu/miniconda3/envs/qwen3-lrb-dager/bin/python`.
`configs/oracle_v2.json` names the existing offline model/dataset and the new
`/data/mcxu/FedLLM-qwen-oracle-v2/Projection_lrb_qwen3/outputs/oracle_v2` directory.
The existing seed22 checkpoint is used only for a load/save compatibility test.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q Projection_lrb_qwen3/tests
python Projection_lrb_qwen3/scripts/run_oracle_v2.py --preregister
python Projection_lrb_qwen3/scripts/run_oracle_v2_pipeline.py --gpus 4,5,6 --stop-after smoke
python Projection_lrb_qwen3/scripts/run_oracle_v2_pipeline.py --gpus 4,5,6
python Projection_lrb_qwen3/scripts/summarize_oracle_v2.py
```

If matplotlib is absent, install its plotting dependencies into the isolated
repository directory with `python -m pip install --target .plot_runtime matplotlib`.
This directory is used only by reporting; the shared training environment is unchanged.

The pipeline skips only completed jobs whose specifications and JSON artifacts
match their completion receipt. Failed/interrupted jobs are never retried
implicitly. An approved new attempt must use a new output identity; retain the
failure artifact. Never delete failure records to make a resume pass.

The controller writes `pipeline_status.json`; every job has a specification,
progress record, log, provenance, and terminal artifact. Attack deadlines are
enforced by an external process watchdog, including when the worker is inside
CUDA. A Python timer ends an interruptible arm five seconds before the hard
deadline, records it as timeout, and permits subsequent independent arms.
Timeout and OOM are not converted to zero recovery. Training is never
silently retried with a smaller batch or different precision.
Each worker also checks active compute PIDs immediately before allocating the
model and waits for an idle GPU; sufficient free VRAM alone is not admission.
For an explicitly supervised shared-GPU run, set
`QWEN_V2_SHARED_MIN_FREE_MIB` to a measured capacity threshold. Admission then
records existing compute PIDs and free MiB. The default remains exclusive.

## Interfaces and numerical semantics

`DefendedQwenUpdate` keeps the complete canonical gradient tuple, names and
realized defense metadata, including absent gradients. q0/q1 independently use
`QwenColumnTransform`: for `G_tilde = L G R^T`, candidate row features become
`h R^T`. Signs are reproduced on the defense device with the original two-draw
RNG order; the column signs follow the row-sign draw. The transform uses the
same pooling bins and align_corners=False interpolation as the actual defense.
It does not assume an orthogonal projector.

Only registered noise-free signed_pool presets are supported. Uniform rho=1
is a real identity; nominal proj_only rho=1 generally is not. Feature capacity
is the column resolution `q = round(d_in * realized_rho)` (minimum 1). Both the
pool and interpolation ranks are verified. B is the actual shared DAGER basis
width selected with absolute atol=0.001 and rank_cutoff=20. Per-layer effective
ranks and any extra shared-B null-space directions are reported explicitly.
The v2 paired path computes one deterministic FP32 full SVD per tensor and
shares its basis between standard/oracle; historical randomized decomposition
is untouched.

The layer1 scanner and layer2 decoder accept an optional candidate transform;
legacy calls remain unchanged. The v2 decoder accepts no truth tokens, label or
sample. Truth is consulted only by post-scan diagnostics and post-search metrics.
Ranks break exact ties by token ID and also report tie-rank intervals. Position
averages, unique-token averages, and EOS flags are distinct.

The old decoder reports the first surviving prefix, often a single token. v2
keeps its historical metrics, and reports a fixed alternate reconstruction rule:
longest surviving prefix, then smallest mean residual, then lexicographic token
IDs. Since the RoPE candidate provider removes EOS, v2 text exact/position
recovery excludes the explicit terminal EOS, without claiming EOS generation.
The public length limit is 31 text tokens plus the explicit EOS; v2 prefix
search therefore caps text at 31. Resource-limited searches expose partial
metrics separately and use null primary recovery metrics.

`load_local_qwen3_sequence_classifier(mode='trained_checkpoint')` refuses
missing/mismatched model weights and preserves score.weight. Default
`random_head_diagnostic` retains the historical explicit-seed behavior.
Training uses FP32 parameters/AdamW state and BF16 autocast; attacks use the
registered BF16 diagnostic precision. Train and eval pooling use native Qwen
last-non-pad semantics, including when EOS and pad share an ID.

## Registered phases

1. P0: mathematical/regression tests, five random-head smoke samples, trained
   checkpoint roundtrip and three-step clean/proj_only/uniform training smokes.
2. P1: seed11 clean pilot, full 20-sample rho scan, deterministic operating-point
   selection, defended seed11 pilots, and checkpoint-family-specific calibration.
3. P2: paired seed101/202/303 full training; preserve init/order hashes and every
   optimizer update's defense coverage.
4. P3: new final20 on trained checkpoints, including same-clean-checkpoint
   mechanism controls and matched-training conditions.

New final20 excludes all historical45 and follows the original sample hash
ordering. The original20 calibration and smoke5 are retained byte-for-byte.
Final examples are never used to choose rho, tau or model checkpoints.
Validation utility is reported after fixed training and never selects a model.

Standard arms share the undefended tau control of the relevant checkpoint
family. Oracle tau1 independently follows the existing 95% active-position
recall/nonempty20 rule. tau2 follows the fixed three-candidate recovery/cost rule.
No qualifying configuration produces an explicit calibration_failed record and
blocks that reconstruction arm. Controls retain their input receipts.

## Deliverables

- Versioned sample/model/data/config provenance and frozen controls.
- Complete training checkpoints with weights/tokenizer hashes, utility metadata,
  paired initialization/order checks, and reload verification.
- Per-sample scan/attack JSON and consolidated `summary/per_sample.jsonl`, status-separated privacy CSV, utility CSV,
  checkpoint manifest and B/q versus token-rank figure.
- Summary ROUGE fields are fractions in `[0,1]`; `r1_r2_raw` is their sum
  in `[0,2]`, and `r1_r2_pct = 100 * r1_r2_raw` is the paper-facing
  `[0,200]` scale. Failed searches have null values for both sums.
- A success means the registered experiment completed with honest coverage;
  candidate separability collapse is a hypothesis, never a stopping criterion.
