#!/usr/bin/env python3
"""Run fixed Qwen3 Layer-1 DAGER baseline smokes from one captured gradient tuple.

This entrypoint is deliberately limited to three defense-unaware baseline arms.
It is a Qwen3-1.7B-Base, fixed-random-head mechanism smoke, not a checkpoint,
utility, formal privacy, adaptive, white-box, oracle, or complete-state run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
for import_root in (REPOSITORY_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import torch

from utils.defenses import gradient_compression, noise_injection, topk_sparsification

from src.config import load_experiment_config
from src.dager_qwen3.diagnostics import load_none_attack_controls, load_registered_sample, registered_head_seed
from src.dager_qwen3.frozen_tau1_control import verify_frozen_tau1_control
from src.dager_qwen3.metrics import preflight_legacy_dager_rouge_backend
from src.dager_qwen3.none_attack_core import (
    GradientDiagnosticFailure,
    NoneAttackCoreControls,
    NoneAttackCoreError,
    execute_dager_from_observed_q_gradients,
    q_canonical_indices,
    q_projection_observations_from_canonical_tuple,
)
from src.hashing import sha256_text
from src.result_schema import ResultSchemaError, write_or_verify_jsonl


SMOKE_STAGE = "smoke"
SMOKE_SAMPLE_KEY = "082ff67b9b082ecd2a3fcca8424d0ef3460c9e05aba05a7f5a6c677edd680322"
SMOKE_HEAD_SEED = 22
SMOKE_DTYPE = "bfloat16"
DEFENSE_RNG_SEED = 700001
DEFENSE_SEED_MODE = "static"
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.json"
FROZEN_TAU1_CONTROL_PATH = PROJECT_ROOT / "frozen_controls" / "qwen3_none_tau1_calibration.json"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "smoke" / "layer1_baselines_v1"
OUTPUT_PATH = OUTPUT_ROOT / "paired_smoke.jsonl"
ALL_SMOKE_OUTPUT_PATH = OUTPUT_ROOT / "paired_smoke_all.jsonl"


class Layer1BaselineSmokeError(RuntimeError):
    """Raised when the fixed Layer-1 baseline protocol is violated."""


@dataclass(frozen=True)
class BaselineArm:
    defense: str
    preset: str
    defense_param_name: str
    defense_param_value: float | int
    topk_keep_ratio: float | None
    compression_bits: int | None
    noise_sigma: float | None
    defense_rng_seed: int | None
    defense_seed_mode: str


BASELINE_ARMS: tuple[BaselineArm, ...] = (
    BaselineArm("topk", "topk", "defense_topk_ratio", 0.1, 0.1, None, None, None, "not_applicable"),
    BaselineArm("compression", "compression", "defense_n_bits", 8, None, 8, None, DEFENSE_RNG_SEED, DEFENSE_SEED_MODE),
    BaselineArm("noise", "noise", "defense_noise", 1e-6, None, None, 1e-6, DEFENSE_RNG_SEED, DEFENSE_SEED_MODE),
)


def _standard_controls(controls: Any) -> NoneAttackCoreControls:
    return NoneAttackCoreControls(
        tau1=controls.l1_span_threshold,
        tau2=controls.l2_span_threshold,
        rank_tolerance=controls.rank_tolerance,
        rank_cutoff=controls.rank_cutoff,
        max_search_candidates=controls.max_search_candidates,
        max_candidate_ids=controls.max_candidate_ids,
        parallel=controls.decode_batch_size,
        max_sequence_length=controls.max_sequence_length,
    )


def _all_registered_smoke_samples(config: Any) -> tuple[Any, ...]:
    """Load every preregistered smoke sample through the normal validator."""
    manifest_path = config.project_root / "manifests" / "smoke.jsonl"
    try:
        rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as error:
        raise Layer1BaselineSmokeError(f"Unable to read preregistered smoke manifest: {error}") from error
    sample_keys = [row.get("sample", {}).get("sample_key") for row in rows if isinstance(row, dict)]
    if len(sample_keys) != 5 or any(not isinstance(key, str) or len(key) != 64 for key in sample_keys):
        raise Layer1BaselineSmokeError("The fixed Layer-1 baseline smoke requires exactly five valid manifest samples.")
    if len(set(sample_keys)) != len(sample_keys):
        raise Layer1BaselineSmokeError("Smoke manifest contains duplicate preregistered sample keys.")
    return tuple(load_registered_sample(config=config, stage=SMOKE_STAGE, sample_key=key) for key in sample_keys)


def _canonical_gradient_tuple(model: Any) -> tuple[tuple[str, ...], tuple[Any | None, ...], dict[str, Any]]:
    """Clone one complete current named-gradient tuple without dropping None slots."""
    from src.gradient_capture import build_canonical_gradient_manifest

    manifest = build_canonical_gradient_manifest(model)
    named_parameters = tuple(model.named_parameters())
    names = tuple(name for name, _parameter in named_parameters)
    manifest_names = tuple(str(entry["name"]) for entry in manifest["entries"])
    if names != manifest_names:
        raise Layer1BaselineSmokeError("Canonical gradient manifest no longer matches model.named_parameters() order.")
    gradients = tuple(parameter.grad.detach().clone() if parameter.grad is not None else None for _name, parameter in named_parameters)
    if len(gradients) != len(names):
        raise Layer1BaselineSmokeError("Canonical gradient tuple construction lost a named parameter position.")
    return names, gradients, manifest


def _clone_canonical_tuple(gradients: tuple[Any | None, ...]) -> tuple[Any | None, ...]:
    return tuple(gradient.detach().clone() if gradient is not None else None for gradient in gradients)


def apply_baseline_to_canonical_tuple(
    *, arm: BaselineArm, canonical_gradients: tuple[Any | None, ...], canonical_parameter_names: tuple[str, ...]
) -> tuple[Any | None, ...]:
    """Apply one shared baseline helper to a fresh full canonical tuple."""
    if len(canonical_gradients) != len(canonical_parameter_names):
        raise Layer1BaselineSmokeError("Baseline input must retain one slot for every canonical parameter name.")
    raw_clone = _clone_canonical_tuple(canonical_gradients)
    if arm.defense == "topk":
        defended = tuple(topk_sparsification(raw_clone, float(arm.topk_keep_ratio)))
    elif arm.defense == "compression":
        defended = tuple(gradient_compression(raw_clone, int(arm.compression_bits), seed=int(arm.defense_rng_seed)))
    elif arm.defense == "noise":
        defended = tuple(noise_injection(raw_clone, float(arm.noise_sigma), seed=int(arm.defense_rng_seed)))
    else:
        raise Layer1BaselineSmokeError(f"Unsupported fixed baseline arm {arm.defense!r}.")
    if len(defended) != len(canonical_gradients):
        raise Layer1BaselineSmokeError(f"{arm.defense} changed the canonical gradient tuple length.")
    for index, raw_gradient in enumerate(canonical_gradients):
        if raw_gradient is None and defended[index] is not None:
            raise Layer1BaselineSmokeError(f"{arm.defense} filled canonical None gradient position {index}.")
    return defended


def _decode_baseline_arms(
    *,
    canonical_gradients: tuple[Any | None, ...],
    canonical_parameter_names: tuple[str, ...],
    q_parameter_names: tuple[str, str],
    canonical_q_indices: dict[str, int],
    decode_arm: Callable[[BaselineArm, tuple[Any, Any]], dict[str, Any]],
) -> tuple[tuple[BaselineArm, dict[str, Any]], ...]:
    """Transform each arm independently and decode only its observed q_proj pair."""
    decoded: list[tuple[BaselineArm, dict[str, Any]]] = []
    for arm in BASELINE_ARMS:
        defended = apply_baseline_to_canonical_tuple(
            arm=arm,
            canonical_gradients=canonical_gradients,
            canonical_parameter_names=canonical_parameter_names,
        )
        observations = q_projection_observations_from_canonical_tuple(
            canonical_gradients=defended,
            canonical_parameter_names=canonical_parameter_names,
            q_parameter_names=q_parameter_names,
            q_canonical_indices=canonical_q_indices,
        )
        decoded.append((arm, decode_arm(arm, observations)))
    return tuple(decoded)


def _arm_record(
    *, arm: BaselineArm, sample: Any, canonical_q_indices: dict[str, int], core: dict[str, Any], canonical_manifest: dict[str, Any], frozen_tau1: dict[str, Any], config_sha256: str
) -> dict[str, Any]:
    identity = sha256_text(
        "|".join(("qwen3_dager_layer1_baseline_smoke_v1", arm.defense, str(arm.defense_param_value), sample.preregistration_sha256, sample.sample_key, str(SMOKE_HEAD_SEED), SMOKE_DTYPE, str(frozen_tau1["frozen_control_identity_sha256"]), config_sha256))
    )
    return {
        "schema_version": 1,
        "record_type": "qwen3_dager_layer1_baseline_smoke",
        "result_identity_sha256": identity,
        "result_status": core["status"],
        "stage": SMOKE_STAGE,
        "sample_key": sample.sample_key,
        "model": "Qwen3-1.7B-Base",
        "task": "sst2",
        "batch_size": 1,
        "gradient_steps": 1,
        "dtype": SMOKE_DTYPE,
        "head_seed": SMOKE_HEAD_SEED,
        "defense": arm.defense,
        "preset": arm.preset,
        "defense_param_name": arm.defense_param_name,
        "defense_param_value": arm.defense_param_value,
        "topk_keep_ratio": arm.topk_keep_ratio,
        "compression_bits": arm.compression_bits,
        "noise_sigma": arm.noise_sigma,
        "defense_rng_seed": arm.defense_rng_seed,
        "defense_seed_mode": arm.defense_seed_mode,
        "defense_awareness": "defense_unaware_observed_q_proj_only",
        "attack_semantics": "defense_unaware_observed_q_proj_only",
        "canonical_q_proj_indices": dict(canonical_q_indices),
        "canonical_gradient_summary": {"parameter_tensor_count": canonical_manifest["parameter_tensor_count"], "gradient_tensor_count": canonical_manifest["gradient_tensor_count"], "none_gradient_position_count": sum(not entry["grad_present"] for entry in canonical_manifest["entries"])},
        "tau1": core["tau1"],
        "tau2": core["tau2"],
        "frozen_tau1_control_identity_sha256": frozen_tau1["frozen_control_identity_sha256"],
        "rank": {"layer_1": core["layer_1_rank"], "layer_2": core["layer_2_rank"], "requested_shared": core["requested_shared_rank"], "applied_shared": core["applied_shared_rank"]},
        "candidate_count": core["layer_1_candidate_count"],
        "decoder_candidate_count": core["layer_1_decoder_candidate_count"],
        "legacy_l1_token_membership": core["legacy_l1_token_membership"],
        "legacy_l1_token_membership_semantics": "real_non_eos_token_fraction_covered_by_dager_layer1_candidate_set",
        "layer_2_survivor_count": core["layer_2_survivor_count"],
        "token_recovery": core["token_recovery"],
        "termination_reason": core["termination_reason"],
        "attack_time_seconds": core["attack_time_seconds"],
    }


def _run_baseline_smoke_samples(args: argparse.Namespace, *, config: Any, samples: tuple[Any, ...], output_path: Path) -> list[dict[str, Any]]:
    from src.dager_qwen3.gradient_gate import decode_token_texts, diagnostic_thresholds
    from src.dager_qwen3.model_adapter import Qwen3RoPEDagerAdapter
    from src.gradient_capture import capture_single_example_gradients
    from src.qwen3_classifier import load_local_qwen3_sequence_classifier
    from src.span_diagnostics import diagnose_two_q_projections

    registered_head_seed(config, stage=SMOKE_STAGE, requested_seed=SMOKE_HEAD_SEED)
    frozen_tau1 = verify_frozen_tau1_control(project_root=PROJECT_ROOT, control_path=FROZEN_TAU1_CONTROL_PATH)
    controls = _standard_controls(load_none_attack_controls(config, frozen_tau1=float(frozen_tau1["selected_tau1"])))
    rouge_backend = preflight_legacy_dager_rouge_backend()
    if not samples:
        raise Layer1BaselineSmokeError("Layer-1 baseline execution requires at least one preregistered smoke sample.")
    torch.manual_seed(SMOKE_HEAD_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SMOKE_HEAD_SEED)
    bundle: Any | None = None
    try:
        bundle = load_local_qwen3_sequence_classifier(config.model_path, head_seed=SMOKE_HEAD_SEED, device=args.device, dtype=SMOKE_DTYPE)
        adapter = Qwen3RoPEDagerAdapter(bundle.model, bundle.tokenizer)
        records: list[dict[str, Any]] = []
        for sample in samples:
            if getattr(bundle.tokenizer, "eos_token_id", None) != sample.eos_token_id:
                raise Layer1BaselineSmokeError("Manifest EOS id differs from the loaded Qwen3 tokenizer EOS id.")
            input_ids = torch.tensor([sample.input_ids], dtype=torch.long, device=bundle.device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=bundle.device)
            labels = torch.tensor([sample.label], dtype=torch.long, device=bundle.device)
            captured = capture_single_example_gradients(bundle.model, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            diagnostic = diagnose_two_q_projections(q_inputs=captured.q_inputs, q_output_gradients=captured.q_output_gradients, q_gradients=captured.q_gradients, q_parameter_names=captured.q_parameter_names, token_ids=sample.input_ids, token_texts=decode_token_texts(bundle.tokenizer, sample.input_ids), eos_token_id=sample.eos_token_id, **diagnostic_thresholds(SMOKE_DTYPE))
            if diagnostic.get("passed") is not True:
                raise GradientDiagnosticFailure(diagnostic, diagnostic_thresholds(SMOKE_DTYPE))
            canonical_names, raw_canonical_gradients, canonical_manifest = _canonical_gradient_tuple(bundle.model)
            canonical_q_indices = q_canonical_indices(canonical_manifest, captured.q_parameter_names)
            raw_observations = q_projection_observations_from_canonical_tuple(
                canonical_gradients=raw_canonical_gradients,
                canonical_parameter_names=canonical_names,
                q_parameter_names=captured.q_parameter_names,
                q_canonical_indices=canonical_q_indices,
            )
            if any(
                not torch.equal(observed, captured_gradient)
                for observed, captured_gradient in zip(raw_observations, captured.q_gradients)
            ):
                raise Layer1BaselineSmokeError(
                    "Raw q_proj observations do not match the one captured complete canonical-gradient tuple."
                )

            def decode_arm(_arm: BaselineArm, observations: tuple[Any, Any]) -> dict[str, Any]:
                return execute_dager_from_observed_q_gradients(adapter=adapter, tokenizer=bundle.tokenizer, sample=sample, observed_q_gradients=observations, q_parameter_names=captured.q_parameter_names, q_canonical_indices=canonical_q_indices, controls=controls, rouge_backend=rouge_backend)

            for arm, core in _decode_baseline_arms(canonical_gradients=raw_canonical_gradients, canonical_parameter_names=canonical_names, q_parameter_names=captured.q_parameter_names, canonical_q_indices=canonical_q_indices, decode_arm=decode_arm):
                records.append(_arm_record(arm=arm, sample=sample, canonical_q_indices=canonical_q_indices, core=core, canonical_manifest=canonical_manifest, frozen_tau1=frozen_tau1, config_sha256=config.config_sha256))
        write_or_verify_jsonl(output_path, records)
        return records
    except (NoneAttackCoreError, ResultSchemaError) as error:
        raise Layer1BaselineSmokeError(str(error)) from error
    finally:
        if bundle is not None:
            del bundle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_baseline_smoke(args: argparse.Namespace) -> list[dict[str, Any]]:
    config = load_experiment_config(CONFIG_PATH, require_dataset_path=False)
    sample = load_registered_sample(config=config, stage=SMOKE_STAGE, sample_key=SMOKE_SAMPLE_KEY)
    return _run_baseline_smoke_samples(args, config=config, samples=(sample,), output_path=OUTPUT_PATH)


def run_all_baseline_smoke(args: argparse.Namespace) -> list[dict[str, Any]]:
    config = load_experiment_config(CONFIG_PATH, require_dataset_path=False)
    return _run_baseline_smoke_samples(args, config=config, samples=_all_registered_smoke_samples(config), output_path=ALL_SMOKE_OUTPUT_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed Qwen3 Layer-1 baseline smoke: topk@0.1, compression@8, and Gaussian noise@1e-6 only.")
    parser.add_argument("--device", default="cuda", help="Qwen3 execution device; all Layer-1 protocol controls remain fixed.")
    parser.add_argument("--all-smoke", action="store_true", help="Run all five preregistered smoke samples and write the immutable aggregate JSONL.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        records = run_all_baseline_smoke(args) if args.all_smoke else run_baseline_smoke(args)
    except Exception as error:
        print(json.dumps({"record_type": "qwen3_dager_layer1_baseline_smoke_error", "result_status": "error", "error_type": type(error).__name__, "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(records, sort_keys=True))
    return 0 if all(record["result_status"] == "ok" for record in records) else 3


if __name__ == "__main__":
    raise SystemExit(main())
