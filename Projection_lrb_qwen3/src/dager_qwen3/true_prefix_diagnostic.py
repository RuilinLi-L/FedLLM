"""Read-only structural diagnosis for a registered Qwen3 DAGER prefix.

This module is deliberately separate from candidate enumeration.  It may be
called only after the standard DAGER spans, Layer-1 candidate provider, and
Layer-2 threshold are all fixed.  The registered token ids are then used to
explain an already-fixed attack outcome; they never enter attack search,
ranking, threshold selection, or defense selection.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

import torch


class TruePrefixDiagnosticError(RuntimeError):
    """Raised when a requested structural diagnostic cannot be evaluated."""


class PrefixSampleProtocol(Protocol):
    """The registered ground-truth fields read by the diagnostic only."""

    input_ids: tuple[int, ...]


DistanceNorm = Literal["l1", "l2"]


def _span_distances(*, basis: torch.Tensor, representations: torch.Tensor, norm: DistanceNorm) -> torch.Tensor:
    """Defer the shared Layer-1 helper import until this optional path runs."""
    from .layer1_filter import span_distances

    return span_distances(basis=basis, representations=representations, norm=norm)


def _ordered_missing(token_ids: tuple[int, ...], candidates: tuple[int, ...]) -> list[int]:
    candidate_set = set(candidates)
    missing: list[int] = []
    seen: set[int] = set()
    for token_id in token_ids:
        if token_id not in candidate_set and token_id not in seen:
            missing.append(token_id)
            seen.add(token_id)
    return missing


def _validate_token_ids(token_ids: tuple[int, ...]) -> None:
    if not token_ids:
        raise TruePrefixDiagnosticError("True-prefix diagnostic requires at least one registered token id.")
    if any(isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0 for token_id in token_ids):
        raise TruePrefixDiagnosticError("Registered token ids must be non-negative integers.")


def diagnose_true_prefix(
    *,
    adapter: Any,
    sample: PrefixSampleProtocol,
    layer1: Any,
    candidate_provider: Any,
    layer2_span: Any,
    threshold: float,
    distance_norm: DistanceNorm,
) -> dict[str, Any]:
    """Report whether the registered prefix survives fixed DAGER predicates.

    This is an evaluation-only oracle diagnostic.  In particular, callers must
    not consume its output to admit candidates, choose ranks or thresholds,
    rank reconstructions, or select a defense.  Layer-2's predicate is exactly
    the production decoder's position-wise strict ``distance < threshold``.
    """
    token_ids = tuple(sample.input_ids)
    _validate_token_ids(token_ids)
    if distance_norm not in ("l1", "l2"):
        raise TruePrefixDiagnosticError(f"Unsupported DAGER distance norm {distance_norm!r}.")
    if not isinstance(threshold, (int, float)) or not bool(torch.isfinite(torch.tensor(float(threshold)))) or threshold <= 0.0:
        raise TruePrefixDiagnosticError(f"Layer-2 threshold must be finite and positive, got {threshold!r}.")

    layer1_token_ids = tuple(getattr(layer1, "token_ids", ()))
    decoder_token_ids = tuple(getattr(candidate_provider, "token_ids", ()))
    if any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in layer1_token_ids):
        raise TruePrefixDiagnosticError("Layer-1 candidate list contains a non-integer token id.")
    if any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in decoder_token_ids):
        raise TruePrefixDiagnosticError("Layer-1 decoder candidate list contains a non-integer token id.")

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=adapter.device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=adapter.device)
    try:
        q1_inputs = adapter.layer1_qproj_inputs_from_prefixes(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    except Exception as error:
        raise TruePrefixDiagnosticError(
            f"Native Qwen3 Layer-2 true-prefix representation failed: {type(error).__name__}: {error}"
        ) from error
    if not isinstance(q1_inputs, torch.Tensor) or q1_inputs.ndim != 3:
        raise TruePrefixDiagnosticError("Native Qwen3 Layer-2 true-prefix representation must be rank-3.")
    expected_shape = (1, len(token_ids), int(adapter.metadata.hidden_size))
    if tuple(q1_inputs.shape) != expected_shape:
        raise TruePrefixDiagnosticError(
            f"Native Qwen3 Layer-2 true-prefix shape {tuple(q1_inputs.shape)} differs from {expected_shape}."
        )

    try:
        distances = _span_distances(
            basis=layer2_span.basis,
            representations=q1_inputs.reshape(-1, q1_inputs.shape[-1]),
            norm=distance_norm,
        )
    except Exception as error:
        raise TruePrefixDiagnosticError(
            f"Layer-2 true-prefix span distance failed: {type(error).__name__}: {error}"
        ) from error
    if distances.ndim != 1 or int(distances.numel()) != len(token_ids):
        raise TruePrefixDiagnosticError(
            f"Layer-2 true-prefix distance shape {tuple(distances.shape)} differs from ({len(token_ids)},)."
        )
    if not bool(torch.isfinite(distances).all()):
        raise TruePrefixDiagnosticError("Layer-2 true-prefix distances contain non-finite values.")
    distance_values = [float(value) for value in distances.detach().cpu().tolist()]
    passes = [distance < float(threshold) for distance in distance_values]
    positions = [
        {
            "position": index,
            "token_id": token_id,
            "distance": distance,
            "passes_threshold": passed,
        }
        for index, (token_id, distance, passed) in enumerate(zip(token_ids, distance_values, passes))
    ]
    layer1_set = set(layer1_token_ids)
    decoder_set = set(decoder_token_ids)
    return {
        "record_type": "qwen3_true_prefix_structural_diagnostic",
        "ground_truth_use": "post_span_structural_diagnostic_only",
        "affects_candidate_search": False,
        "affects_rank_selection": False,
        "affects_threshold_selection": False,
        "affects_attack_result": False,
        "layer_1": {
            "threshold_candidate_count": len(layer1_token_ids),
            "decoder_candidate_count": len(decoder_token_ids),
            "first_token_id": token_ids[0],
            "first_token_in_threshold_candidate_set": token_ids[0] in layer1_set,
            "first_token_in_decoder_candidate_set": token_ids[0] in decoder_set,
            "all_input_tokens_in_threshold_candidate_set": all(token_id in layer1_set for token_id in token_ids),
            "all_input_tokens_in_decoder_candidate_set": all(token_id in decoder_set for token_id in token_ids),
            "missing_input_token_ids_from_threshold_candidate_set": _ordered_missing(token_ids, layer1_token_ids),
            "missing_input_token_ids_from_decoder_candidate_set": _ordered_missing(token_ids, decoder_token_ids),
        },
        "layer_2": {
            "threshold": float(threshold),
            "distance_norm": distance_norm,
            "position_count": len(token_ids),
            "first_position_passes_threshold": passes[0],
            "all_positions_pass_threshold": all(passes),
            "position_results": positions,
            "distance_min": min(distance_values),
            "distance_mean": sum(distance_values) / len(distance_values),
            "distance_max": max(distance_values),
        },
    }
