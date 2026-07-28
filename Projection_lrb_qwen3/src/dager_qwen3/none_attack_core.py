"""The single shared none-only Qwen3/RoPE DAGER execution path.

Both the one-sample runner and Stage-5 calibration call this module.  It owns
no protocol selection, no LRB transformation, and no result-file writing.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, Protocol


class NoneAttackCoreError(RuntimeError):
    """Raised when the shared standard DAGER computation cannot complete."""


class GradientDiagnosticFailure(NoneAttackCoreError):
    """Explicitly carries the diagnostic needed by the standalone sidecar."""

    def __init__(self, diagnostic: Mapping[str, Any], thresholds: Mapping[str, float]) -> None:
        super().__init__("Qwen3 gradient orientation diagnostic failed.")
        self.diagnostic = diagnostic
        self.thresholds = thresholds


class SampleProtocol(Protocol):
    """Fields shared by preregistered attack and calibration samples."""

    input_ids: tuple[int, ...]
    label: int
    eos_token_id: int


@dataclass(frozen=True)
class NoneAttackCoreControls:
    """Complete standard-DAGER controls, all supplied by the caller's protocol."""

    tau1: float
    tau2: float
    rank_tolerance: float
    rank_cutoff: int
    max_search_candidates: int
    max_candidate_ids: int
    parallel: int
    max_sequence_length: int


def _qproj_input_identity_diagnostic(
    *,
    adapter: Any,
    input_ids: Any,
    attention_mask: Any,
    captured: Any,
) -> dict[str, Any]:
    """Fail closed unless candidate forwards reproduce the captured q0/q1 inputs."""
    import torch

    expected_q0 = adapter.layer0_qproj_inputs_for_token_ids(input_ids.reshape(-1)).reshape_as(captured.q_inputs[0])
    cpu_state_after_capture = torch.random.get_rng_state()
    cuda_state_after_capture = torch.cuda.get_rng_state(input_ids.device)
    try:
        torch.random.set_rng_state(captured.cpu_rng_state_before_forward)
        torch.cuda.set_rng_state(captured.cuda_rng_state_before_forward, input_ids.device)
        expected_q1 = adapter.layer1_qproj_inputs_from_prefixes(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    finally:
        torch.random.set_rng_state(cpu_state_after_capture)
        torch.cuda.set_rng_state(cuda_state_after_capture, input_ids.device)

    tolerance = 5e-4 if captured.compute_dtype == torch.bfloat16 else 1e-4
    result: dict[str, Any] = {
        "record_type": "qwen3_qproj_input_identity",
        "source": "same_gradient_capture_forward_rng_state_replay",
        "tolerance": tolerance,
        "layers": {},
    }
    passed = True
    for name, expected, observed in zip(("q0", "q1"), (expected_q0, expected_q1), captured.q_inputs):
        if expected.shape != observed.shape:
            raise NoneAttackCoreError(
                f"{name} adapter input shape {tuple(expected.shape)} differs from captured shape {tuple(observed.shape)}."
            )
        expected_fp32 = expected.detach().float()
        observed_fp32 = observed.detach().float()
        absolute = (expected_fp32 - observed_fp32).abs()
        max_abs = float(absolute.max().item())
        max_rel = float((absolute / observed_fp32.abs().clamp_min(1e-12)).max().item())
        layer_passed = bool(torch.allclose(expected_fp32, observed_fp32, rtol=0.0, atol=tolerance))
        result["layers"][name] = {
            "shape": list(observed.shape),
            "dtype": str(observed.dtype),
            "max_abs_error": max_abs,
            "max_relative_error": max_rel,
            "passed": layer_passed,
        }
        passed = passed and layer_passed
    result["passed"] = passed
    if not passed:
        raise NoneAttackCoreError("Qwen3 adapter q0/q1 inputs do not match the gradient-capture forward.")
    return result


def q_canonical_indices(manifest: Mapping[str, Any], names: tuple[str, str]) -> dict[str, int]:
    """Resolve the two structural q-projection positions in canonical order."""
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise NoneAttackCoreError("Canonical gradient manifest lacks entries.")
    result: dict[str, int] = {}
    for entry in entries:
        if isinstance(entry, Mapping) and entry.get("name") in names:
            index = entry.get("canonical_index")
            if isinstance(index, int) and not isinstance(index, bool):
                result[str(entry["name"])] = index
    if set(result) != set(names):
        raise NoneAttackCoreError("Canonical gradient manifest does not contain both structural q_proj names.")
    return result


def q_projection_observations_from_canonical_tuple(
    *,
    canonical_gradients: tuple[Any | None, ...],
    canonical_parameter_names: tuple[str, ...],
    q_parameter_names: tuple[str, str],
    q_canonical_indices: Mapping[str, int],
) -> tuple[Any, Any]:
    """Read the two q-projection observations only through canonical indices.

    The paired Projection-LRB smoke keeps the complete named-gradient tuple,
    including ``None`` positions, through its transform.  This helper makes the
    subsequent q-projection selection auditable and prevents a q-only defense
    shortcut.
    """
    if len(canonical_gradients) != len(canonical_parameter_names):
        raise NoneAttackCoreError("Canonical gradient tuple and name tuple have different lengths.")
    observations: list[Any] = []
    for name in q_parameter_names:
        index = q_canonical_indices.get(name)
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(canonical_gradients):
            raise NoneAttackCoreError(f"Canonical q_proj index is invalid for {name!r}.")
        if canonical_parameter_names[index] != name:
            raise NoneAttackCoreError(
                f"Canonical q_proj index {index} resolves to {canonical_parameter_names[index]!r}, not {name!r}."
            )
        gradient = canonical_gradients[index]
        if gradient is None:
            raise NoneAttackCoreError(f"Canonical q_proj gradient is None for {name!r}.")
        observations.append(gradient)
    return observations[0], observations[1]


def execute_none_only_dager(
    *,
    model_path: Any,
    sample: SampleProtocol,
    controls: NoneAttackCoreControls,
    head_seed: int,
    device: str,
    dtype: str,
    rouge_backend: Any,
    true_prefix_diagnostic: bool = False,
) -> dict[str, Any]:
    """Execute standard defense-unaware DAGER once, without writing artifacts."""
    try:
        import torch

        from .gradient_gate import decode_token_texts, diagnostic_thresholds
        from .model_adapter import Qwen3RoPEDagerAdapter
        from src.gradient_capture import build_canonical_gradient_manifest, capture_single_example_gradients
        from src.qwen3_classifier import load_local_qwen3_sequence_classifier
        from src.span_diagnostics import diagnose_two_q_projections
    except Exception as error:
        raise NoneAttackCoreError(f"Qwen3 standard-DAGER dependencies are unavailable: {error}") from error
    torch.manual_seed(head_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(head_seed)
    bundle: Any | None = None
    try:
        bundle = load_local_qwen3_sequence_classifier(model_path, head_seed=head_seed, device=device, dtype=dtype)
        if getattr(bundle.tokenizer, "eos_token_id", None) != sample.eos_token_id:
            raise NoneAttackCoreError("Manifest EOS id differs from the loaded Qwen3 tokenizer EOS id.")
        input_ids = torch.tensor([sample.input_ids], dtype=torch.long, device=bundle.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=bundle.device)
        labels = torch.tensor([sample.label], dtype=torch.long, device=bundle.device)
        capture_started = perf_counter()
        captured = capture_single_example_gradients(bundle.model, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        capture_seconds = perf_counter() - capture_started
        diagnostic_controls = diagnostic_thresholds(dtype)
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
        adapter = Qwen3RoPEDagerAdapter(bundle.model, bundle.tokenizer)
        qproj_input_identity = _qproj_input_identity_diagnostic(
            adapter=adapter,
            input_ids=input_ids,
            attention_mask=attention_mask,
            captured=captured,
        )
        canonical = build_canonical_gradient_manifest(bundle.model)
        parameter_names = tuple(str(entry["name"]) for entry in canonical["entries"])
        canonical_gradients = tuple(
            parameter.grad.detach().clone() if parameter.grad is not None else None
            for _, parameter in bundle.model.named_parameters()
        )
        canonical_indices = q_canonical_indices(canonical, captured.q_parameter_names)
        observed_gradients = q_projection_observations_from_canonical_tuple(
            canonical_gradients=canonical_gradients,
            canonical_parameter_names=parameter_names,
            q_parameter_names=captured.q_parameter_names,
            q_canonical_indices=canonical_indices,
        )
        if any(not torch.equal(observed, captured_gradient) for observed, captured_gradient in zip(observed_gradients, captured.q_gradients)):
            raise NoneAttackCoreError("Canonical q_proj observations differ from the captured q_proj gradients.")
        result = execute_dager_from_observed_q_gradients(
            adapter=adapter,
            tokenizer=bundle.tokenizer,
            sample=sample,
            observed_q_gradients=observed_gradients,
            q_parameter_names=captured.q_parameter_names,
            q_canonical_indices=canonical_indices,
            controls=controls,
            rouge_backend=rouge_backend,
            true_prefix_diagnostic=true_prefix_diagnostic,
        )
        return {
            **result,
            "gradient_capture_time_seconds": capture_seconds,
            "loss": captured.loss,
            "gpu_peak_memory_bytes": captured.gpu_peak_memory_bytes,
            "canonical_gradient_summary": {
                "gradient_tensor_count": canonical["gradient_tensor_count"],
                "gradient_numel": canonical["gradient_numel"],
            },
            "diagnostic_status": "ok",
            "gradient_diagnostic": diagnostic,
            "qproj_input_identity": qproj_input_identity,
            "adapter": {
                "execution_path": adapter.metadata.execution_path,
                "hidden_size": adapter.metadata.hidden_size,
                "vocab_size": adapter.metadata.vocab_size,
            },
        }
    finally:
        if bundle is not None:
            del bundle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def execute_dager_from_observed_q_gradients(
    *,
    adapter: Any,
    tokenizer: Any,
    sample: SampleProtocol,
    observed_q_gradients: tuple[Any, Any],
    q_parameter_names: tuple[str, str],
    q_canonical_indices: Mapping[str, int],
    controls: NoneAttackCoreControls,
    rouge_backend: Any,
    true_prefix_diagnostic: bool = False,
) -> dict[str, Any]:
    """Decode two observed q-projection gradients with standard DAGER only.

    This core deliberately accepts only the two observed tensors and normal
    DAGER controls.  It receives no LRB seed, preset, layer metadata, or
    transform state, so both the raw and Projection-LRB branches are decoded
    through the exact same defense-unaware path.
    """
    if len(observed_q_gradients) != 2:
        raise NoneAttackCoreError("Standard DAGER requires exactly two observed q_proj gradients.")
    try:
        from .candidate_provider import RoPECandidateProvider
        from .gradient_decomposition import (
            decompose_qwen3_qproj_gradient,
            shared_dager_rank_for_qwen3_qproj_gradients,
        )
        from .layer1_filter import filter_qwen3_vocab_layer1
        from .layer2_decoder import Layer2DecoderConfig, decode_qwen3_rope_prefixes, layer2_audit_json_fields
        from .metrics import compute_attack_metrics
    except Exception as error:
        raise NoneAttackCoreError(f"Qwen3 standard-DAGER decoder dependencies are unavailable: {error}") from error
    attack_started = perf_counter()
    q0_gradient, q1_gradient = observed_q_gradients
    shared_rank = shared_dager_rank_for_qwen3_qproj_gradients(
        (q0_gradient, q1_gradient),
        feature_dim=adapter.metadata.hidden_size,
        rank_tolerance=controls.rank_tolerance,
        rank_cutoff=controls.rank_cutoff,
        decomposition_device=adapter.device,
    )
    q0_span = decompose_qwen3_qproj_gradient(
        q0_gradient,
        feature_dim=adapter.metadata.hidden_size,
        rank_tolerance=controls.rank_tolerance,
        rank_cutoff=controls.rank_cutoff,
        decomposition_device=adapter.device,
        shared_truncated_rank=shared_rank.applied_shared_rank,
    )
    q1_span = decompose_qwen3_qproj_gradient(
        q1_gradient,
        feature_dim=adapter.metadata.hidden_size,
        rank_tolerance=controls.rank_tolerance,
        rank_cutoff=controls.rank_cutoff,
        decomposition_device=adapter.device,
        shared_truncated_rank=shared_rank.applied_shared_rank,
    )
    layer1 = filter_qwen3_vocab_layer1(
        adapter=adapter,
        span=q0_span,
        threshold=controls.tau1,
        vocab_chunk_size=controls.parallel,
        distance_norm="l2",
    )
    candidate_provider = RoPECandidateProvider.from_layer1_result(
        layer1,
        eos_token_id=sample.eos_token_id,
        max_ids=controls.max_candidate_ids,
    )
    true_prefix_result: dict[str, Any] | None = None
    if true_prefix_diagnostic:
        try:
            from .true_prefix_diagnostic import diagnose_true_prefix
        except Exception as error:
            raise NoneAttackCoreError(
                f"Qwen3 true-prefix diagnostic dependencies are unavailable: {error}"
            ) from error
        true_prefix_result = diagnose_true_prefix(
            adapter=adapter,
            sample=sample,
            layer1=layer1,
            candidate_provider=candidate_provider,
            layer2_span=q1_span,
            threshold=controls.tau2,
            distance_norm="l2",
        )
    layer2 = decode_qwen3_rope_prefixes(
        adapter=adapter,
        span=q1_span,
        candidate_provider=candidate_provider,
        config=Layer2DecoderConfig(
            max_sequence_length=controls.max_sequence_length,
            threshold=controls.tau2,
            distance_norm="l2",
            search_budget=controls.max_search_candidates,
            decode_batch_size=controls.parallel,
        ),
    )
    metrics = compute_attack_metrics(
        tokenizer=tokenizer,
        ground_truth_token_ids=sample.input_ids,
        reconstructed_token_ids=layer2.selected_token_ids,
        layer1_candidate_token_ids=layer1.token_ids,
        eos_token_id=sample.eos_token_id,
        rouge_metric=rouge_backend.metric,
    )
    result = {
        "status": "search_budget_exhausted" if layer2.search_budget_exhausted else "ok",
        "tau1": controls.tau1,
        "tau2": controls.tau2,
        "ground_truth_token_ids": list(sample.input_ids),
        "ground_truth_token_text": list(metrics.ground_truth_token_text),
        "ground_truth_text": metrics.ground_truth_text,
        "reconstructed_token_ids": list(layer2.selected_token_ids),
        "reconstructed_token_text": list(metrics.reconstructed_token_text),
        "reconstructed_text": metrics.reconstructed_text,
        "token_recovery": metrics.token_recovery,
        "token_recovery_semantics": "first_layer2_survivor_position_match_for_reporting_only",
        "legacy_l1_token_membership": metrics.legacy_l1_token_membership,
        "legacy_l1_token_membership_semantics": "root_attack_rec_token_excluding_terminal_eos",
        "exact_recovery": metrics.exact_recovery,
        "rouge_1": metrics.rouge_1,
        "rouge_2": metrics.rouge_2,
        "legacy_rouge_backend": rouge_backend.json_metadata(),
        "empty_reconstruction": metrics.empty_reconstruction,
        "layer_1_candidate_count": layer1.candidate_count,
        "layer_1_decoder_candidate_count": len(candidate_provider.token_ids),
        "layer_1_rank": q0_span.truncated_rank,
        "layer_1_effective_rank": shared_rank.q0_effective_rank,
        "layer_2_rank": q1_span.truncated_rank,
        "layer_2_effective_rank": shared_rank.q1_effective_rank,
        "rank_definition": shared_rank.rank_definition,
        "rank_atol": shared_rank.rank_atol,
        "q0_effective_rank": shared_rank.q0_effective_rank,
        "q1_effective_rank": shared_rank.q1_effective_rank,
        "requested_shared_rank": shared_rank.requested_shared_rank,
        "applied_shared_rank": shared_rank.applied_shared_rank,
        "rank_was_capped": shared_rank.rank_was_capped,
        "rank_cap": shared_rank.rank_cap,
        "cap_reason": shared_rank.cap_reason,
        "attack_time_seconds": perf_counter() - attack_started,
        "thresholds": {
            "l1_span_thresh": controls.tau1,
            "l2_span_thresh": controls.tau2,
            "rank_atol": controls.rank_tolerance,
            "rank_definition": "absolute_matrix_rank_atol_rtol_zero",
            "rank_cutoff": controls.rank_cutoff,
            "distance_norm": "l2",
        },
        "search_budget": {
            "maxC": controls.max_search_candidates,
            "parallel": controls.parallel,
            "vocab_chunk_size": controls.parallel,
            "max_ids": controls.max_candidate_ids,
            "max_length": controls.max_sequence_length,
            "evaluated_prefix_count": layer2.evaluated_prefix_count,
            "search_budget_exhausted": layer2.search_budget_exhausted,
            "per_length_survivor_counts": [list(value) for value in layer2.per_length_survivor_counts],
        },
        "layer_2_completed_prefix_count": len(layer2.completed_prefixes),
        "selected_survivor_rule": "first_threshold_passing_prefix_in_enumeration_order",
        "selected_layer_2_mean_span_distance": layer2.selected_mean_span_distance,
        **layer2_audit_json_fields(layer2),
        "q_proj": {
            "parameter_names": list(q_parameter_names),
            "canonical_indices": dict(q_canonical_indices),
            "gradient_shapes": [list(gradient.shape) for gradient in observed_q_gradients],
            "orientation": "raw_qwen3_nn_linear_gradient_right_singular_vectors",
        },
        "adapter": {
            "execution_path": adapter.metadata.execution_path,
            "hidden_size": adapter.metadata.hidden_size,
            "vocab_size": adapter.metadata.vocab_size,
        },
    }
    if true_prefix_result is not None:
        result["true_prefix_diagnostic"] = true_prefix_result
    return result
