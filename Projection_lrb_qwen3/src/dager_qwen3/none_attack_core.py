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


def _q_canonical_indices(manifest: Mapping[str, Any], names: tuple[str, str]) -> dict[str, int]:
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


def execute_none_only_dager(
    *,
    model_path: Any,
    sample: SampleProtocol,
    controls: NoneAttackCoreControls,
    head_seed: int,
    device: str,
    dtype: str,
    rouge_backend: Any,
) -> dict[str, Any]:
    """Execute standard defense-unaware DAGER once, without writing artifacts."""
    try:
        import torch

        from .candidate_provider import RoPECandidateProvider
        from .gradient_decomposition import (
            decompose_qwen3_qproj_gradient,
            shared_dager_rank_for_qwen3_qproj_gradients,
        )
        from .gradient_gate import decode_token_texts, diagnostic_thresholds
        from .layer1_filter import filter_qwen3_vocab_layer1
        from .layer2_decoder import Layer2DecoderConfig, decode_qwen3_rope_prefixes, layer2_audit_json_fields
        from .model_adapter import Qwen3RoPEDagerAdapter
        from src.gradient_capture import build_canonical_gradient_manifest, capture_single_example_gradients
        from src.qwen3_classifier import load_local_qwen3_sequence_classifier
        from src.span_diagnostics import diagnose_two_q_projections
        from .metrics import compute_attack_metrics
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
        attack_started = perf_counter()
        adapter = Qwen3RoPEDagerAdapter(bundle.model, bundle.tokenizer)
        shared_rank = shared_dager_rank_for_qwen3_qproj_gradients(
            captured.q_gradients,
            feature_dim=adapter.metadata.hidden_size,
            rank_tolerance=controls.rank_tolerance,
            rank_cutoff=controls.rank_cutoff,
            decomposition_device=bundle.device,
        )
        q0_span = decompose_qwen3_qproj_gradient(
            captured.q_gradients[0], feature_dim=adapter.metadata.hidden_size,
            rank_tolerance=controls.rank_tolerance, rank_cutoff=controls.rank_cutoff,
            decomposition_device=bundle.device, shared_truncated_rank=shared_rank.applied_shared_rank,
        )
        q1_span = decompose_qwen3_qproj_gradient(
            captured.q_gradients[1], feature_dim=adapter.metadata.hidden_size,
            rank_tolerance=controls.rank_tolerance, rank_cutoff=controls.rank_cutoff,
            decomposition_device=bundle.device, shared_truncated_rank=shared_rank.applied_shared_rank,
        )
        layer1 = filter_qwen3_vocab_layer1(
            adapter=adapter, span=q0_span, threshold=controls.tau1,
            vocab_chunk_size=controls.parallel, distance_norm="l2",
        )
        candidate_provider = RoPECandidateProvider.from_layer1_result(
            layer1, eos_token_id=sample.eos_token_id, max_ids=controls.max_candidate_ids
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
        attack_seconds = perf_counter() - attack_started
        metrics = compute_attack_metrics(
            tokenizer=bundle.tokenizer,
            ground_truth_token_ids=sample.input_ids,
            reconstructed_token_ids=layer2.selected_token_ids,
            eos_token_id=sample.eos_token_id,
            rouge_metric=rouge_backend.metric,
        )
        canonical = build_canonical_gradient_manifest(bundle.model)
        return {
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
            "rank_rtol": shared_rank.rank_rtol,
            "q0_effective_rank": shared_rank.q0_effective_rank,
            "q1_effective_rank": shared_rank.q1_effective_rank,
            "q0_relative_threshold": shared_rank.q0_relative_threshold,
            "q1_relative_threshold": shared_rank.q1_relative_threshold,
            "requested_shared_rank": shared_rank.requested_shared_rank,
            "applied_shared_rank": shared_rank.applied_shared_rank,
            "rank_was_capped": shared_rank.rank_was_capped,
            "rank_cap": shared_rank.rank_cap,
            "cap_reason": shared_rank.cap_reason,
            "attack_time_seconds": attack_seconds,
            "gradient_capture_time_seconds": capture_seconds,
            "loss": captured.loss,
            "gpu_peak_memory_bytes": captured.gpu_peak_memory_bytes,
            "thresholds": {
                "l1_span_thresh": controls.tau1,
                "l2_span_thresh": controls.tau2,
                "rank_rtol": controls.rank_tolerance,
                "rank_definition": "relative_svd_threshold",
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
            "selected_layer_2_mean_span_distance": layer2.selected_mean_span_distance,
            **layer2_audit_json_fields(layer2),
            "q_proj": {
                "parameter_names": list(captured.q_parameter_names),
                "canonical_indices": _q_canonical_indices(canonical, captured.q_parameter_names),
                "gradient_shapes": [list(gradient.shape) for gradient in captured.q_gradients],
                "orientation": "raw_qwen3_nn_linear_gradient_right_singular_vectors",
            },
            "canonical_gradient_summary": {
                "gradient_tensor_count": canonical["gradient_tensor_count"],
                "gradient_numel": canonical["gradient_numel"],
            },
            "diagnostic_status": "ok",
            "gradient_diagnostic": diagnostic,
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
