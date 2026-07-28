#!/usr/bin/env python3
"""Run the preregistered Qwen3 Layer-1 token-set mechanism final lattice.

This isolated runner evaluates 11 fixed defense-unaware observed-q_proj arms
on the 20 registered SST-2 validation samples.  It is not a utility,
formal-privacy, DP, adaptive, oracle, white-box, or complete-state experiment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
for import_root in (REPOSITORY_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import torch

from utils.defenses import gradient_compression, noise_injection, topk_sparsification
from utils.lrb_defense import apply_lrb_defense
from utils.lrb_presets import apply_lrb_preset

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


FINAL_STAGE = "final"
FINAL_DTYPE = "bfloat16"
FINAL_HEAD_SEEDS = (101, 202, 303)
DEFENSE_BASE_SEED = 700001
DEFENSE_SEED_MODE = "static"
LRB_PRESET = "proj_only"
COMPRESSION_SEMANTICS = "per_tensor_l2_norm_scaled_qsgd_style_stochastic_quantization"
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.json"
FROZEN_TAU1_CONTROL_PATH = PROJECT_ROOT / "frozen_controls" / "qwen3_none_tau1_calibration.json"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "final" / "layer1_all_arms_v1"
Q0 = "model.layers.0.self_attn.q_proj.weight"
Q1 = "model.layers.1.self_attn.q_proj.weight"


class Layer1FinalAllArmsError(RuntimeError):
    """Raised when the fixed final Layer-1 mechanism protocol is violated."""


@dataclass(frozen=True)
class FinalArm:
    label: str
    defense: str
    preset: str
    defense_param_name: str
    defense_param_value: float | int | None
    topk_keep_ratio: float | None
    compression_bits: int | None
    noise_sigma: float | None
    defense_rng_seed: int | None
    defense_seed_mode: str
    lrb_seed: int | None
    lrb_seed_mode: str


FINAL_ARMS: tuple[FinalArm, ...] = (
    FinalArm("none", "none", "none", "none", None, None, None, None, None, "not_applicable", None, "not_applicable"),
    FinalArm("lrbprojonly@0.2", "lrbprojonly", LRB_PRESET, "keep_ratio", 0.2, None, None, None, DEFENSE_BASE_SEED, DEFENSE_SEED_MODE, DEFENSE_BASE_SEED, DEFENSE_SEED_MODE),
    FinalArm("lrbprojonly@0.5", "lrbprojonly", LRB_PRESET, "keep_ratio", 0.5, None, None, None, DEFENSE_BASE_SEED, DEFENSE_SEED_MODE, DEFENSE_BASE_SEED, DEFENSE_SEED_MODE),
    FinalArm("lrbprojonly@0.65", "lrbprojonly", LRB_PRESET, "keep_ratio", 0.65, None, None, None, DEFENSE_BASE_SEED, DEFENSE_SEED_MODE, DEFENSE_BASE_SEED, DEFENSE_SEED_MODE),
    FinalArm("topk@0.1", "topk", "topk", "defense_topk_ratio", 0.1, 0.1, None, None, None, "not_applicable", None, "not_applicable"),
    FinalArm("topk@0.7", "topk", "topk", "defense_topk_ratio", 0.7, 0.7, None, None, None, "not_applicable", None, "not_applicable"),
    FinalArm("topk@0.9", "topk", "topk", "defense_topk_ratio", 0.9, 0.9, None, None, None, "not_applicable", None, "not_applicable"),
    FinalArm("compression@8", "compression", "compression", "defense_n_bits", 8, None, 8, None, DEFENSE_BASE_SEED, DEFENSE_SEED_MODE, None, "not_applicable"),
    FinalArm("compression@16", "compression", "compression", "defense_n_bits", 16, None, 16, None, DEFENSE_BASE_SEED, DEFENSE_SEED_MODE, None, "not_applicable"),
    FinalArm("compression@32", "compression", "compression", "defense_n_bits", 32, None, 32, None, DEFENSE_BASE_SEED, DEFENSE_SEED_MODE, None, "not_applicable"),
    FinalArm("noise@1e-6", "noise", "noise", "defense_noise", 1e-6, None, None, 1e-6, DEFENSE_BASE_SEED, DEFENSE_SEED_MODE, None, "not_applicable"),
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


def _all_registered_final_samples(config: Any) -> tuple[Any, ...]:
    """Load exactly the 20 immutable final samples through the normal validator."""
    manifest_path = config.project_root / "manifests" / "final.jsonl"
    try:
        rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as error:
        raise Layer1FinalAllArmsError(f"Unable to read preregistered final manifest: {error}") from error
    sample_keys = [row.get("sample", {}).get("sample_key") for row in rows if isinstance(row, dict)]
    if len(sample_keys) != 20 or any(not isinstance(key, str) or len(key) != 64 for key in sample_keys):
        raise Layer1FinalAllArmsError("The final Layer-1 protocol requires exactly 20 valid manifest samples.")
    if len(set(sample_keys)) != 20:
        raise Layer1FinalAllArmsError("Final manifest contains duplicate preregistered sample keys.")
    return tuple(load_registered_sample(config=config, stage=FINAL_STAGE, sample_key=key) for key in sample_keys)


def _canonical_gradient_tuple(model: Any) -> tuple[tuple[str, ...], tuple[Any | None, ...], dict[str, Any]]:
    """Clone one complete named-gradient tuple while retaining canonical None slots."""
    from src.gradient_capture import build_canonical_gradient_manifest

    manifest = build_canonical_gradient_manifest(model)
    named_parameters = tuple(model.named_parameters())
    names = tuple(name for name, _parameter in named_parameters)
    manifest_names = tuple(str(entry["name"]) for entry in manifest["entries"])
    if names != manifest_names:
        raise Layer1FinalAllArmsError("Canonical gradient manifest no longer matches model.named_parameters() order.")
    gradients = tuple(parameter.grad.detach().clone() if parameter.grad is not None else None for _name, parameter in named_parameters)
    if len(gradients) != len(names):
        raise Layer1FinalAllArmsError("Canonical gradient tuple construction lost a named parameter position.")
    return names, gradients, manifest


def _clone_canonical_tuple(gradients: tuple[Any | None, ...]) -> tuple[Any | None, ...]:
    return tuple(gradient.detach().clone() if gradient is not None else None for gradient in gradients)


def _lrb_arguments(*, keep_ratio: float, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        defense="lrbprojonly",
        defense_lrb_preset=LRB_PRESET,
        defense_lrb_keep_ratio_sensitive=keep_ratio,
        defense_lrb_seed=seed,
        defense_lrb_seed_mode=DEFENSE_SEED_MODE,
        rng_seed=seed,
    )


def _validate_defended_tuple(
    *, arm: FinalArm, raw: tuple[Any | None, ...], defended: tuple[Any | None, ...]
) -> tuple[Any | None, ...]:
    if len(defended) != len(raw):
        raise Layer1FinalAllArmsError(f"{arm.label} changed the canonical gradient tuple length.")
    for index, raw_gradient in enumerate(raw):
        if raw_gradient is None and defended[index] is not None:
            raise Layer1FinalAllArmsError(f"{arm.label} filled canonical None gradient position {index}.")
    return defended


def apply_final_arm_to_canonical_tuple(
    *, arm: FinalArm, canonical_gradients: tuple[Any | None, ...], canonical_parameter_names: tuple[str, ...]
) -> tuple[Any | None, ...]:
    """Apply one arm to an independent full-tuple clone before q_proj selection."""
    if len(canonical_gradients) != len(canonical_parameter_names):
        raise Layer1FinalAllArmsError("Every arm requires the complete canonical named-gradient tuple.")
    raw_clone = _clone_canonical_tuple(canonical_gradients)
    if arm.defense == "none":
        defended = raw_clone
    elif arm.defense == "topk":
        defended = tuple(topk_sparsification(raw_clone, float(arm.topk_keep_ratio)))
    elif arm.defense == "compression":
        defended = tuple(gradient_compression(raw_clone, int(arm.compression_bits), seed=int(arm.defense_rng_seed)))
    elif arm.defense == "noise":
        defended = tuple(noise_injection(raw_clone, float(arm.noise_sigma), seed=int(arm.defense_rng_seed)))
    elif arm.defense == "lrbprojonly":
        args = _lrb_arguments(keep_ratio=float(arm.defense_param_value), seed=int(arm.lrb_seed))
        apply_lrb_preset(args)
        defended = tuple(apply_lrb_defense(raw_clone, args, layer_names=list(canonical_parameter_names)))
    else:
        raise Layer1FinalAllArmsError(f"Unsupported fixed final arm {arm.label!r}.")
    return _validate_defended_tuple(arm=arm, raw=canonical_gradients, defended=defended)


def _decode_final_arms(
    *,
    canonical_gradients: tuple[Any | None, ...],
    canonical_parameter_names: tuple[str, ...],
    q_parameter_names: tuple[str, str],
    canonical_q_indices: dict[str, int],
    decode_arm: Callable[[FinalArm, tuple[Any, Any]], dict[str, Any]],
) -> tuple[tuple[FinalArm, dict[str, Any]], ...]:
    """Independently transform the full tuple for every arm, then observe q_proj."""
    decoded: list[tuple[FinalArm, dict[str, Any]]] = []
    for arm in FINAL_ARMS:
        defended = apply_final_arm_to_canonical_tuple(
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
    *, arm: FinalArm, sample: Any, head_seed: int, canonical_q_indices: dict[str, int], core: dict[str, Any], canonical_manifest: dict[str, Any], frozen_tau1: dict[str, Any], config_sha256: str
) -> dict[str, Any]:
    identity = sha256_text(
        "|".join(("qwen3_sst2_layer1_final_all_arms_v1", arm.label, sample.preregistration_sha256, sample.sample_key, str(head_seed), FINAL_DTYPE, str(frozen_tau1["frozen_control_identity_sha256"]), config_sha256))
    )
    return {
        "schema_version": 1,
        "record_type": "qwen3_dager_layer1_final_all_arms",
        "result_identity_sha256": identity,
        "result_status": core["status"],
        "stage": FINAL_STAGE,
        "sample_key": sample.sample_key,
        "head_seed": head_seed,
        "model": "Qwen3-1.7B-Base",
        "task": "sst2",
        "batch_size": 1,
        "gradient_steps": 1,
        "dtype": FINAL_DTYPE,
        "defense": arm.defense,
        "preset": arm.preset,
        "defense_param_name": arm.defense_param_name,
        "defense_param_value": arm.defense_param_value,
        "topk_keep_ratio": arm.topk_keep_ratio,
        "compression_bits": arm.compression_bits,
        "compression_semantics": COMPRESSION_SEMANTICS if arm.defense == "compression" else "not_applicable",
        "noise_sigma": arm.noise_sigma,
        "defense_rng_seed": arm.defense_rng_seed,
        "defense_seed_mode": arm.defense_seed_mode,
        "lrb_seed": arm.lrb_seed,
        "lrb_seed_mode": arm.lrb_seed_mode,
        "attack_semantics": "defense_unaware_observed_q_proj_only",
        "canonical_q_proj_indices": dict(canonical_q_indices),
        "canonical_gradient_summary": {
            "parameter_tensor_count": canonical_manifest["parameter_tensor_count"],
            "gradient_tensor_count": canonical_manifest["gradient_tensor_count"],
            "none_gradient_position_count": sum(not entry["grad_present"] for entry in canonical_manifest["entries"]),
        },
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


def _run_final_samples(
    args: argparse.Namespace, *, config: Any, samples: tuple[Any, ...], head_seed: int, output_path: Path
) -> list[dict[str, Any]]:
    """Capture each final example once and decode its independent fixed arms."""
    from src.dager_qwen3.gradient_gate import decode_token_texts, diagnostic_thresholds
    from src.dager_qwen3.model_adapter import Qwen3RoPEDagerAdapter
    from src.gradient_capture import capture_single_example_gradients
    from src.qwen3_classifier import load_local_qwen3_sequence_classifier
    from src.span_diagnostics import diagnose_two_q_projections

    registered_head_seed(config, stage=FINAL_STAGE, requested_seed=head_seed)
    frozen_tau1 = verify_frozen_tau1_control(project_root=PROJECT_ROOT, control_path=FROZEN_TAU1_CONTROL_PATH)
    controls = _standard_controls(load_none_attack_controls(config, frozen_tau1=float(frozen_tau1["selected_tau1"])))
    rouge_backend = preflight_legacy_dager_rouge_backend()
    if len(samples) != 20:
        raise Layer1FinalAllArmsError("Final execution requires exactly 20 preregistered samples.")

    torch.manual_seed(head_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(head_seed)
    bundle: Any | None = None
    try:
        bundle = load_local_qwen3_sequence_classifier(config.model_path, head_seed=head_seed, device=args.device, dtype=FINAL_DTYPE)
        adapter = Qwen3RoPEDagerAdapter(bundle.model, bundle.tokenizer)
        diagnostic_controls = diagnostic_thresholds(FINAL_DTYPE)
        records: list[dict[str, Any]] = []
        for sample in samples:
            if getattr(bundle.tokenizer, "eos_token_id", None) != sample.eos_token_id:
                raise Layer1FinalAllArmsError("Manifest EOS id differs from the loaded Qwen3 tokenizer EOS id.")
            input_ids = torch.tensor([sample.input_ids], dtype=torch.long, device=bundle.device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=bundle.device)
            labels = torch.tensor([sample.label], dtype=torch.long, device=bundle.device)
            captured = capture_single_example_gradients(bundle.model, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            diagnostic = diagnose_two_q_projections(
                q_inputs=captured.q_inputs,
                q_output_gradients=captured.q_output_gradients,
                q_gradients=captured.q_gradients,
                q_parameter_names=captured.q_parameter_names,
                token_ids=sample.input_ids,
                token_texts=decode_token_texts(bundle.tokenizer, sample.input_ids),
                eos_token_id=sample.eos_token_id,
                **diagnostic_controls,
            )
            if diagnostic.get("passed") is not True:
                raise GradientDiagnosticFailure(diagnostic, diagnostic_controls)
            canonical_names, raw_canonical_gradients, canonical_manifest = _canonical_gradient_tuple(bundle.model)
            canonical_q_indices = q_canonical_indices(canonical_manifest, captured.q_parameter_names)
            raw_observations = q_projection_observations_from_canonical_tuple(
                canonical_gradients=raw_canonical_gradients,
                canonical_parameter_names=canonical_names,
                q_parameter_names=captured.q_parameter_names,
                q_canonical_indices=canonical_q_indices,
            )
            if any(not torch.equal(observed, captured_gradient) for observed, captured_gradient in zip(raw_observations, captured.q_gradients)):
                raise Layer1FinalAllArmsError("Raw q_proj observations do not match the one captured complete canonical-gradient tuple.")

            def decode_arm(_arm: FinalArm, observations: tuple[Any, Any]) -> dict[str, Any]:
                return execute_dager_from_observed_q_gradients(
                    adapter=adapter, tokenizer=bundle.tokenizer, sample=sample, observed_q_gradients=observations,
                    q_parameter_names=captured.q_parameter_names, q_canonical_indices=canonical_q_indices,
                    controls=controls, rouge_backend=rouge_backend,
                )

            for arm, core in _decode_final_arms(
                canonical_gradients=raw_canonical_gradients, canonical_parameter_names=canonical_names,
                q_parameter_names=captured.q_parameter_names, canonical_q_indices=canonical_q_indices, decode_arm=decode_arm,
            ):
                records.append(_arm_record(arm=arm, sample=sample, head_seed=head_seed, canonical_q_indices=canonical_q_indices, core=core, canonical_manifest=canonical_manifest, frozen_tau1=frozen_tau1, config_sha256=config.config_sha256))
        if len(records) != 20 * len(FINAL_ARMS) or any(record["result_status"] != "ok" for record in records):
            raise Layer1FinalAllArmsError("Final seed lattice must contain exactly 220 result_status=ok records.")
        write_or_verify_jsonl(output_path, records)
        return records
    except (NoneAttackCoreError, ResultSchemaError) as error:
        raise Layer1FinalAllArmsError(str(error)) from error
    finally:
        if bundle is not None:
            del bundle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_final_head_seed(args: argparse.Namespace, *, head_seed: int) -> list[dict[str, Any]]:
    config = load_experiment_config(CONFIG_PATH, require_dataset_path=False)
    if tuple(config.final_head_seeds) != FINAL_HEAD_SEEDS:
        raise Layer1FinalAllArmsError("experiment.json final_head_seeds no longer match the fixed final protocol.")
    samples = _all_registered_final_samples(config)
    return _run_final_samples(args, config=config, samples=samples, head_seed=head_seed, output_path=OUTPUT_ROOT / f"seed{head_seed}" / "paired_final.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed Qwen3 Layer-1 final all-arm token-set evaluation over the preregistered final SST-2 manifest.")
    parser.add_argument("--device", default="cuda", help="Qwen3 execution device; all protocol controls remain fixed.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--head-seed", type=int, choices=FINAL_HEAD_SEEDS, default=101, help="One registered fixed random classifier-head seed.")
    group.add_argument("--all-head-seeds", action="store_true", help="Run all three registered fixed classifier-head seeds.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = FINAL_HEAD_SEEDS if args.all_head_seeds else (args.head_seed,)
    try:
        records = [record for head_seed in seeds for record in run_final_head_seed(args, head_seed=head_seed)]
    except Exception as error:
        print(json.dumps({"record_type": "qwen3_dager_layer1_final_all_arms_error", "result_status": "error", "error_type": type(error).__name__, "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"record_type": "qwen3_dager_layer1_final_all_arms_complete", "result_status": "ok", "record_count": len(records), "head_seeds": list(seeds)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
