#!/usr/bin/env python3
"""Run fixed preregistered Qwen3/SST-2 none vs Projection-LRB DAGER smokes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
for import_root in (REPOSITORY_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import torch

from utils.lrb_defense import apply_lrb_defense
from utils.lrb_presets import apply_lrb_preset

from src.config import load_experiment_config
from src.dager_qwen3.diagnostics import load_none_attack_controls, load_registered_sample, registered_head_seed
from src.dager_qwen3.frozen_tau1_control import verify_frozen_tau1_control
from src.dager_qwen3.metrics import preflight_legacy_dager_rouge_backend
from src.dager_qwen3.none_attack_core import (
    GradientDiagnosticFailure,
    NoneAttackCoreError,
    NoneAttackCoreControls,
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
LRB_SEED = 700001
LRB_SEED_MODE = "static"
LRB_PRESET = "proj_only"
LRB_KEEP_RATIO = 0.5
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.json"
FROZEN_TAU1_CONTROL_PATH = PROJECT_ROOT / "frozen_controls" / "qwen3_none_tau1_calibration.json"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "smoke" / "minimal_projonly_pair_l1metrics_v1"
OUTPUT_PATH = OUTPUT_ROOT / "paired_smoke.jsonl"
ALL_SMOKE_OUTPUT_PATH = OUTPUT_ROOT / "paired_smoke_all.jsonl"


class PairSmokeError(RuntimeError):
    """Raised when the fixed paired smoke protocol is violated."""


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
        raise PairSmokeError(f"Unable to read preregistered smoke manifest: {error}") from error
    if len(rows) < 2:
        raise PairSmokeError("All-smoke paired execution requires at least two preregistered smoke samples.")
    sample_keys: list[str] = []
    for row in rows:
        sample = row.get("sample") if isinstance(row, dict) else None
        key = sample.get("sample_key") if isinstance(sample, dict) else None
        if not isinstance(key, str) or len(key) != 64:
            raise PairSmokeError("Smoke manifest contains an invalid preregistered sample key.")
        sample_keys.append(key)
    if len(set(sample_keys)) != len(sample_keys):
        raise PairSmokeError("Smoke manifest contains duplicate preregistered sample keys.")
    return tuple(
        load_registered_sample(config=config, stage=SMOKE_STAGE, sample_key=sample_key)
        for sample_key in sample_keys
    )


def _canonical_gradient_tuple(model: Any) -> tuple[tuple[str, ...], tuple[Any | None, ...], dict[str, Any]]:
    """Clone the complete current named-gradient tuple without dropping None positions."""
    from src.gradient_capture import build_canonical_gradient_manifest

    manifest = build_canonical_gradient_manifest(model)
    named_parameters = tuple(model.named_parameters())
    names = tuple(name for name, _parameter in named_parameters)
    manifest_names = tuple(str(entry["name"]) for entry in manifest["entries"])
    if names != manifest_names:
        raise PairSmokeError("Canonical gradient manifest no longer matches model.named_parameters() order.")
    gradients = tuple(
        parameter.grad.detach().clone() if parameter.grad is not None else None
        for _name, parameter in named_parameters
    )
    if len(gradients) != len(names):
        raise PairSmokeError("Canonical gradient tuple construction lost a named parameter position.")
    return names, gradients, manifest


def _lrb_arguments(*, defense: str, preset: str, keep_ratio: float, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        defense=defense,
        defense_lrb_preset=preset,
        defense_lrb_keep_ratio_sensitive=keep_ratio,
        defense_lrb_seed=seed,
        defense_lrb_seed_mode=LRB_SEED_MODE,
        rng_seed=seed,
    )


def apply_lrb_to_canonical_tuple(
    *,
    canonical_gradients: tuple[Any | None, ...],
    canonical_parameter_names: tuple[str, ...],
    defense: str = "lrbprojonly",
    preset: str = LRB_PRESET,
    keep_ratio: float = LRB_KEEP_RATIO,
    seed: int = LRB_SEED,
) -> tuple[Any | None, ...]:
    """Apply existing LRB once to the whole canonical tuple, preserving holes."""
    if len(canonical_gradients) != len(canonical_parameter_names):
        raise PairSmokeError("LRB input must retain one gradient slot for every canonical parameter name.")
    args = _lrb_arguments(defense=defense, preset=preset, keep_ratio=keep_ratio, seed=seed)
    apply_lrb_preset(args)
    defended = tuple(
        apply_lrb_defense(canonical_gradients, args, layer_names=list(canonical_parameter_names))
    )
    if len(defended) != len(canonical_gradients):
        raise PairSmokeError("Existing LRB changed the canonical gradient tuple length.")
    for index, raw_gradient in enumerate(canonical_gradients):
        if raw_gradient is None and defended[index] is not None:
            raise PairSmokeError(f"Existing LRB filled canonical None gradient position {index}.")
    return defended


def _decode_observed_arm(
    *,
    adapter: Any,
    tokenizer: Any,
    sample: Any,
    observed_q_gradients: tuple[Any, Any],
    q_parameter_names: tuple[str, str],
    canonical_q_indices: dict[str, int],
    controls: NoneAttackCoreControls,
    rouge_backend: Any,
) -> dict[str, Any]:
    """Invoke the shared decoder without exposing LRB state to it."""
    return execute_dager_from_observed_q_gradients(
        adapter=adapter,
        tokenizer=tokenizer,
        sample=sample,
        observed_q_gradients=observed_q_gradients,
        q_parameter_names=q_parameter_names,
        q_canonical_indices=canonical_q_indices,
        controls=controls,
        rouge_backend=rouge_backend,
    )


def _arm_record(
    *,
    defense: str,
    preset: str,
    keep_ratio: float | None,
    lrb_seed: int | None,
    sample: Any,
    canonical_q_indices: dict[str, int],
    core: dict[str, Any],
    raw_gradient_diagnostic: dict[str, Any],
    canonical_manifest: dict[str, Any],
    frozen_tau1: dict[str, Any],
    config_sha256: str,
) -> dict[str, Any]:
    identity = sha256_text(
        "|".join(
            (
                "qwen3_sst2_minimal_projonly_pair_smoke_v1",
                defense,
                preset,
                str(keep_ratio),
                sample.preregistration_sha256,
                sample.sample_key,
                str(SMOKE_HEAD_SEED),
                SMOKE_DTYPE,
                str(frozen_tau1["frozen_control_identity_sha256"]),
                config_sha256,
            )
        )
    )
    return {
        "schema_version": 1,
        "record_type": "qwen3_dager_minimal_projonly_pair_smoke",
        "result_identity_sha256": identity,
        "model": "Qwen3-1.7B-Base",
        "task": "sst2",
        "batch_size": 1,
        "gradient_steps": 1,
        "dtype": SMOKE_DTYPE,
        "defense": defense,
        "preset": preset,
        "keep_ratio": keep_ratio,
        "lrb_seed": lrb_seed,
        "lrb_seed_mode": LRB_SEED_MODE if lrb_seed is not None else "none",
        "defense_awareness": "defense_unaware_observed_q_proj_only",
        "stage": SMOKE_STAGE,
        "sample_key": sample.sample_key,
        "head_seed": SMOKE_HEAD_SEED,
        "canonical_q_proj_indices": dict(canonical_q_indices),
        "canonical_gradient_summary": {
            "parameter_tensor_count": canonical_manifest["parameter_tensor_count"],
            "gradient_tensor_count": canonical_manifest["gradient_tensor_count"],
            "none_gradient_position_count": sum(
                not entry["grad_present"] for entry in canonical_manifest["entries"]
            ),
        },
        "tau1": core["tau1"],
        "tau2": core["tau2"],
        "frozen_tau1_control_identity_sha256": frozen_tau1["frozen_control_identity_sha256"],
        "rank": {
            "layer_1": core["layer_1_rank"],
            "layer_2": core["layer_2_rank"],
            "requested_shared": core["requested_shared_rank"],
            "applied_shared": core["applied_shared_rank"],
        },
        "candidate_count": core["layer_1_candidate_count"],
        "decoder_candidate_count": core["layer_1_decoder_candidate_count"],
        "reconstructed_token_ids": core["reconstructed_token_ids"],
        "token_recovery": core["token_recovery"],
        "token_recovery_semantics": core["token_recovery_semantics"],
        "legacy_l1_token_membership": core["legacy_l1_token_membership"],
        "legacy_l1_token_membership_semantics": core["legacy_l1_token_membership_semantics"],
        "layer_2_survivor_count": core["layer_2_survivor_count"],
        "exact_recovery": core["exact_recovery"],
        "rouge_1": core["rouge_1"],
        "rouge_2": core["rouge_2"],
        "termination_reason": core["termination_reason"],
        "attack_time_seconds": core["attack_time_seconds"],
        "result_status": core["status"],
        "raw_gradient_diagnostic_stage": "pre_lrb_raw_complete_gradient_capture",
        "raw_gradient_diagnostic": raw_gradient_diagnostic,
    }


def _run_paired_smoke_samples(
    args: argparse.Namespace,
    *,
    config: Any,
    samples: tuple[Any, ...],
    output_path: Path,
) -> list[dict[str, Any]]:
    """Capture each fixed update and decode raw and full-tuple Projection-LRB arms."""
    from src.dager_qwen3.gradient_gate import decode_token_texts, diagnostic_thresholds
    from src.dager_qwen3.model_adapter import Qwen3RoPEDagerAdapter
    from src.gradient_capture import capture_single_example_gradients
    from src.qwen3_classifier import load_local_qwen3_sequence_classifier
    from src.span_diagnostics import diagnose_two_q_projections

    registered_head_seed(config, stage=SMOKE_STAGE, requested_seed=SMOKE_HEAD_SEED)
    frozen_tau1 = verify_frozen_tau1_control(
        project_root=PROJECT_ROOT,
        control_path=FROZEN_TAU1_CONTROL_PATH,
    )
    controls = _standard_controls(load_none_attack_controls(config, frozen_tau1=float(frozen_tau1["selected_tau1"])))
    rouge_backend = preflight_legacy_dager_rouge_backend()
    if not samples:
        raise PairSmokeError("Paired smoke execution requires at least one preregistered sample.")

    torch.manual_seed(SMOKE_HEAD_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SMOKE_HEAD_SEED)
    bundle: Any | None = None
    try:
        bundle = load_local_qwen3_sequence_classifier(
            config.model_path,
            head_seed=SMOKE_HEAD_SEED,
            device=args.device,
            dtype=SMOKE_DTYPE,
        )
        diagnostic_controls = diagnostic_thresholds(SMOKE_DTYPE)
        adapter = Qwen3RoPEDagerAdapter(bundle.model, bundle.tokenizer)
        records: list[dict[str, Any]] = []
        for sample in samples:
            if getattr(bundle.tokenizer, "eos_token_id", None) != sample.eos_token_id:
                raise PairSmokeError("Manifest EOS id differs from the loaded Qwen3 tokenizer EOS id.")
            input_ids = torch.tensor([sample.input_ids], dtype=torch.long, device=bundle.device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=bundle.device)
            labels = torch.tensor([sample.label], dtype=torch.long, device=bundle.device)
            captured = capture_single_example_gradients(
                bundle.model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            raw_diagnostic = diagnose_two_q_projections(
                q_inputs=captured.q_inputs,
                q_output_gradients=captured.q_output_gradients,
                q_gradients=captured.q_gradients,
                q_parameter_names=captured.q_parameter_names,
                token_ids=sample.input_ids,
                token_texts=decode_token_texts(bundle.tokenizer, sample.input_ids),
                eos_token_id=sample.eos_token_id,
                **diagnostic_controls,
            )
            if raw_diagnostic.get("passed") is not True:
                raise GradientDiagnosticFailure(raw_diagnostic, diagnostic_controls)

            canonical_names, raw_canonical_gradients, canonical_manifest = _canonical_gradient_tuple(bundle.model)
            canonical_q_indices = q_canonical_indices(canonical_manifest, captured.q_parameter_names)
            none_observations = q_projection_observations_from_canonical_tuple(
                canonical_gradients=raw_canonical_gradients,
                canonical_parameter_names=canonical_names,
                q_parameter_names=captured.q_parameter_names,
                q_canonical_indices=canonical_q_indices,
            )
            if any(
                not torch.equal(observed, captured_gradient)
                for observed, captured_gradient in zip(none_observations, captured.q_gradients)
            ):
                raise PairSmokeError("Raw q_proj observations do not match the canonical complete-gradient tuple.")
            lrb_canonical_gradients = apply_lrb_to_canonical_tuple(
                canonical_gradients=raw_canonical_gradients,
                canonical_parameter_names=canonical_names,
            )
            projonly_observations = q_projection_observations_from_canonical_tuple(
                canonical_gradients=lrb_canonical_gradients,
                canonical_parameter_names=canonical_names,
                q_parameter_names=captured.q_parameter_names,
                q_canonical_indices=canonical_q_indices,
            )
            none_core = _decode_observed_arm(
                adapter=adapter,
                tokenizer=bundle.tokenizer,
                sample=sample,
                observed_q_gradients=none_observations,
                q_parameter_names=captured.q_parameter_names,
                canonical_q_indices=canonical_q_indices,
                controls=controls,
                rouge_backend=rouge_backend,
            )
            projonly_core = _decode_observed_arm(
                adapter=adapter,
                tokenizer=bundle.tokenizer,
                sample=sample,
                observed_q_gradients=projonly_observations,
                q_parameter_names=captured.q_parameter_names,
                canonical_q_indices=canonical_q_indices,
                controls=controls,
                rouge_backend=rouge_backend,
            )
            records.extend(
                (
                    _arm_record(
                        defense="none",
                        preset="none",
                        keep_ratio=None,
                        lrb_seed=None,
                        sample=sample,
                        canonical_q_indices=canonical_q_indices,
                        core=none_core,
                        raw_gradient_diagnostic=raw_diagnostic,
                        canonical_manifest=canonical_manifest,
                        frozen_tau1=frozen_tau1,
                        config_sha256=config.config_sha256,
                    ),
                    _arm_record(
                        defense="lrbprojonly",
                        preset=LRB_PRESET,
                        keep_ratio=LRB_KEEP_RATIO,
                        lrb_seed=LRB_SEED,
                        sample=sample,
                        canonical_q_indices=canonical_q_indices,
                        core=projonly_core,
                        raw_gradient_diagnostic=raw_diagnostic,
                        canonical_manifest=canonical_manifest,
                        frozen_tau1=frozen_tau1,
                        config_sha256=config.config_sha256,
                    ),
                )
            )
        write_or_verify_jsonl(output_path, records)
        return records
    except (NoneAttackCoreError, ResultSchemaError) as error:
        raise PairSmokeError(str(error)) from error
    finally:
        if bundle is not None:
            del bundle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_pair_smoke(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run the original fixed one-sample paired smoke without changing its artifact."""
    config = load_experiment_config(CONFIG_PATH, require_dataset_path=False)
    sample = load_registered_sample(config=config, stage=SMOKE_STAGE, sample_key=SMOKE_SAMPLE_KEY)
    return _run_paired_smoke_samples(args, config=config, samples=(sample,), output_path=OUTPUT_PATH)


def run_all_smoke_pairs(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run every preregistered smoke sample into a separate immutable artifact."""
    config = load_experiment_config(CONFIG_PATH, require_dataset_path=False)
    samples = _all_registered_smoke_samples(config)
    return _run_paired_smoke_samples(args, config=config, samples=samples, output_path=ALL_SMOKE_OUTPUT_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed preregistered Qwen3 none vs proj_only@0.5 DAGER mechanism smokes."
    )
    parser.add_argument("--device", default="cuda", help="Qwen3 execution device; protocol fields remain fixed.")
    parser.add_argument(
        "--all-smoke",
        action="store_true",
        help="Run every preregistered smoke sample and write a separate immutable aggregate JSONL.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        records = run_all_smoke_pairs(args) if args.all_smoke else run_pair_smoke(args)
    except Exception as error:
        print(
            json.dumps(
                {
                    "record_type": "qwen3_dager_minimal_projonly_pair_smoke_error",
                    "result_status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(records, sort_keys=True))
    return 0 if all(record["result_status"] == "ok" for record in records) else 3


if __name__ == "__main__":
    raise SystemExit(main())
