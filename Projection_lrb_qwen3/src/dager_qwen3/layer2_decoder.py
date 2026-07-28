"""Threshold-filtered Qwen3/RoPE sequence recovery using the second q_proj span."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from time import perf_counter
from typing import TYPE_CHECKING, Any

import torch

from .gradient_decomposition import GradientSpan
from .layer1_filter import DistanceNorm, Layer1FilterError, span_distances

if TYPE_CHECKING:
    from .candidate_provider import RoPECandidateProvider
    from .model_adapter import Qwen3RoPEDagerAdapter


class Layer2DecoderError(RuntimeError):
    """Raised when threshold-based Qwen3 prefix recovery cannot proceed."""


@dataclass(frozen=True)
class Layer2DecoderConfig:
    """Fixed DAGER decoder budget; no beam, semantic filter, or LM prior."""

    max_sequence_length: int
    threshold: float
    distance_norm: DistanceNorm
    search_budget: int
    decode_batch_size: int


@dataclass(frozen=True)
class DecodedPrefix:
    """One fully threshold-passing candidate prefix in deterministic enumeration order."""

    token_ids: tuple[int, ...]
    mean_span_distance: float


@dataclass(frozen=True)
class Layer2LengthDistanceAudit:
    """Read-only statistics from one already-evaluated decoder length.

    ``distance_*`` values are derived from the existing per-prefix mean span
    distances returned by :func:`_evaluate_prefix_batch`; they are never a
    second span computation.  ``passing_count`` is the count of the existing
    position-wise DAGER predicate verdicts, so this object cannot change
    candidate retention or enumeration order.
    """

    prefix_length: int
    evaluated_count: int
    finite_distance_count: int
    nonfinite_distance_count: int
    passing_count: int
    rejected_count: int
    threshold: float
    distance_min: float | None
    distance_median: float | None
    distance_max: float | None


@dataclass(frozen=True)
class Layer2DecodeResult:
    """Results of exhaustive, threshold-filtered prefix expansion up to its budget."""

    selected_token_ids: tuple[int, ...]
    selected_mean_span_distance: float | None
    completed_prefixes: tuple[DecodedPrefix, ...]
    survivor_prefixes: tuple[DecodedPrefix, ...]
    evaluated_prefix_count: int
    search_budget_exhausted: bool
    per_length_survivor_counts: tuple[tuple[int, int], ...]
    per_length_distance_audit: tuple[Layer2LengthDistanceAudit, ...]
    termination_reason: str
    elapsed_seconds: float


def _validate_config(config: Layer2DecoderConfig) -> None:
    if isinstance(config.max_sequence_length, bool) or config.max_sequence_length < 1:
        raise Layer2DecoderError("max_sequence_length must be at least one.")
    if not isinstance(config.threshold, (int, float)) or not torch.isfinite(torch.tensor(float(config.threshold))) or config.threshold <= 0:
        raise Layer2DecoderError(f"l2 threshold must be finite and positive, got {config.threshold!r}.")
    if config.distance_norm not in ("l2", "l1"):
        raise Layer2DecoderError(f"Unsupported DAGER distance norm {config.distance_norm!r}.")
    if isinstance(config.search_budget, bool) or config.search_budget <= 0:
        raise Layer2DecoderError("search_budget must be a positive integer.")
    if isinstance(config.decode_batch_size, bool) or config.decode_batch_size <= 0:
        raise Layer2DecoderError("decode_batch_size must be a positive integer.")


def _evaluate_prefix_batch(
    *,
    adapter: Qwen3RoPEDagerAdapter,
    span: GradientSpan,
    prefixes: list[tuple[int, ...]],
    threshold: float,
    distance_norm: DistanceNorm,
) -> tuple[list[bool], list[float]]:
    if not prefixes:
        return [], []
    lengths = {len(prefix) for prefix in prefixes}
    if len(lengths) != 1:
        raise Layer2DecoderError("DAGER prefix batch must contain one sequence length at a time.")
    token_batch = torch.tensor(prefixes, dtype=torch.long, device=adapter.device)
    attention_mask = torch.ones_like(token_batch, dtype=torch.long, device=adapter.device)
    q1_inputs = adapter.layer1_qproj_inputs_from_prefixes(
        input_ids=token_batch,
        attention_mask=attention_mask,
    )
    flattened = q1_inputs.reshape(-1, q1_inputs.shape[-1])
    try:
        distances = span_distances(basis=span.basis, representations=flattened, norm=distance_norm)
    except Layer1FilterError as error:
        raise Layer2DecoderError(str(error)) from error
    per_prefix = distances.reshape(token_batch.shape[0], token_batch.shape[1])
    # ``span_distances`` normally rejects non-finite values before this point.
    # Keep that numerical safety explicit here as well: invalid distances can
    # never become survivors, while the finite-value predicate is unchanged.
    passes = torch.isfinite(per_prefix).all(dim=1) & (per_prefix < threshold).all(dim=1)
    means = per_prefix.mean(dim=1)
    return [bool(value) for value in passes.detach().cpu().tolist()], [
        float(value) for value in means.detach().cpu().tolist()
    ]


def _summarize_length_distances(
    *,
    prefix_length: int,
    passes: list[bool],
    mean_distances: list[float],
    threshold: float,
) -> Layer2LengthDistanceAudit:
    """Create audit metadata from one existing decoder-length evaluation.

    This accepts the values already computed for the legacy decoder's
    threshold predicate and does not call the adapter or span-distance helper.
    In particular, the production predicate remains the existing
    position-wise strict DAGER test in :func:`_evaluate_prefix_batch`.
    """
    if len(passes) != len(mean_distances):
        raise Layer2DecoderError(
            "Layer-2 distance audit received inconsistent pass and mean-distance counts."
        )
    finite_distances = [float(value) for value in mean_distances if math.isfinite(float(value))]
    passing_count = sum(bool(value) for value in passes)
    evaluated_count = len(mean_distances)
    return Layer2LengthDistanceAudit(
        prefix_length=prefix_length,
        evaluated_count=evaluated_count,
        finite_distance_count=len(finite_distances),
        nonfinite_distance_count=evaluated_count - len(finite_distances),
        passing_count=passing_count,
        rejected_count=evaluated_count - passing_count,
        threshold=float(threshold),
        distance_min=None if not finite_distances else min(finite_distances),
        distance_median=None if not finite_distances else float(median(finite_distances)),
        distance_max=None if not finite_distances else max(finite_distances),
    )


def _termination_reason(
    *,
    candidate_provider: RoPECandidateProvider,
    survivors: list[DecodedPrefix],
    exhausted: bool,
    distance_audit: list[Layer2LengthDistanceAudit],
    max_sequence_length: int,
) -> str:
    """Report why the existing decoder loop ended, without changing it."""
    if not candidate_provider.token_ids:
        return "no_layer1_candidates"
    if exhausted:
        return "search_budget_exhausted"
    if survivors:
        return "layer2_survivors_reported"
    for item in distance_audit:
        if item.passing_count == 0:
            return f"no_layer2_survivor_at_length_{item.prefix_length}"
    if distance_audit and distance_audit[-1].prefix_length >= max_sequence_length:
        return "max_length_reached"
    raise Layer2DecoderError("Layer-2 decoder ended without a reportable termination reason.")


def layer2_audit_json_fields(result: Layer2DecodeResult) -> dict[str, Any]:
    """Return JSON-ready audit fields from an already completed decode result."""
    return {
        "layer_2_distance_audit": [
            {
                "prefix_length": item.prefix_length,
                "evaluated_count": item.evaluated_count,
                "finite_distance_count": item.finite_distance_count,
                "nonfinite_distance_count": item.nonfinite_distance_count,
                "passing_count": item.passing_count,
                "rejected_count": item.rejected_count,
                "threshold": item.threshold,
                "distance_min": item.distance_min,
                "distance_median": item.distance_median,
                "distance_max": item.distance_max,
            }
            for item in result.per_length_distance_audit
        ],
        "termination_reason": result.termination_reason,
        "layer_2_survivor_count": len(result.survivor_prefixes),
        "layer_2_survivors": [
            {
                "token_ids": list(item.token_ids),
                "mean_span_distance": item.mean_span_distance,
            }
            for item in result.survivor_prefixes
        ],
    }


def decode_qwen3_rope_prefixes(
    *,
    adapter: Qwen3RoPEDagerAdapter,
    span: GradientSpan,
    candidate_provider: RoPECandidateProvider,
    config: Layer2DecoderConfig,
) -> Layer2DecodeResult:
    """Report all standard-DAGER threshold-passing prefixes in enumeration order.

    Each prefix is formed by the shared RoPE candidate vocabulary, evaluated by
    native Qwen3 layer-0 forward, and retained only when *every* q1 input lies
    in the layer-2 DAGER span.  Enumeration follows the legacy decoder's
    candidate order and performs no beam pruning or candidate re-ranking.
    """
    _validate_config(config)
    if span.feature_dim != adapter.metadata.hidden_size:
        raise Layer2DecoderError(
            f"Layer-2 span feature dimension {span.feature_dim} does not match Qwen3 hidden "
            f"size {adapter.metadata.hidden_size}."
        )
    started = perf_counter()
    survivors: list[DecodedPrefix] = []
    completed: list[DecodedPrefix] = []
    reportable_survivors: list[DecodedPrefix] = []
    evaluated = 0
    exhausted = False
    survivor_counts: list[tuple[int, int]] = []
    distance_audit: list[Layer2LengthDistanceAudit] = []
    for sequence_length in range(1, config.max_sequence_length + 1):
        if sequence_length > 1 and not survivors:
            survivor_counts.append((sequence_length, 0))
            break
        next_survivors: list[DecodedPrefix] = []
        if sequence_length == 1:
            prefix_iter = ((token_id,) for token_id in candidate_provider.candidates_for_position(0))
        else:
            prefix_iter = (
                prefix.token_ids + (token_id,)
                for prefix in survivors
                for token_id in candidate_provider.candidates_for_position(sequence_length - 1)
            )
        batch: list[tuple[int, ...]] = []
        length_passes: list[bool] = []
        length_mean_distances: list[float] = []
        for prefix in prefix_iter:
            if evaluated + len(batch) >= config.search_budget:
                exhausted = True
                break
            batch.append(prefix)
            if len(batch) < config.decode_batch_size:
                continue
            passes, scores = _evaluate_prefix_batch(
                adapter=adapter,
                span=span,
                prefixes=batch,
                threshold=float(config.threshold),
                distance_norm=config.distance_norm,
            )
            evaluated += len(batch)
            length_passes.extend(passes)
            length_mean_distances.extend(scores)
            for token_ids, passed, score in zip(batch, passes, scores):
                if not passed:
                    continue
                accepted = DecodedPrefix(token_ids, score)
                reportable_survivors.append(accepted)
                if accepted.token_ids[-1] == candidate_provider.eos_token_id:
                    completed.append(accepted)
                elif sequence_length < config.max_sequence_length:
                    next_survivors.append(accepted)
            batch = []
        if batch:
            passes, scores = _evaluate_prefix_batch(
                adapter=adapter,
                span=span,
                prefixes=batch,
                threshold=float(config.threshold),
                distance_norm=config.distance_norm,
            )
            evaluated += len(batch)
            length_passes.extend(passes)
            length_mean_distances.extend(scores)
            for token_ids, passed, score in zip(batch, passes, scores):
                if not passed:
                    continue
                accepted = DecodedPrefix(token_ids, score)
                reportable_survivors.append(accepted)
                if accepted.token_ids[-1] == candidate_provider.eos_token_id:
                    completed.append(accepted)
                elif sequence_length < config.max_sequence_length:
                    next_survivors.append(accepted)
        distance_audit.append(
            _summarize_length_distances(
                prefix_length=sequence_length,
                passes=length_passes,
                mean_distances=length_mean_distances,
                threshold=float(config.threshold),
            )
        )
        survivor_counts.append((sequence_length, len(next_survivors)))
        if exhausted or sequence_length >= config.max_sequence_length:
            break
        # This is exhaustive threshold-filtered expansion, not beam search:
        # the next iteration streams every accepted prefix/token pair in chunks.
        survivors = next_survivors
    # This field is only a deterministic reporting convenience.  The complete
    # survivor list is serialized for attacker-side evaluation; absence of EOS
    # no longer turns a non-empty survivor set into an empty reconstruction.
    selected = reportable_survivors[0] if reportable_survivors else None
    return Layer2DecodeResult(
        selected_token_ids=() if selected is None else selected.token_ids,
        selected_mean_span_distance=None if selected is None else selected.mean_span_distance,
        completed_prefixes=tuple(completed),
        survivor_prefixes=tuple(reportable_survivors),
        evaluated_prefix_count=evaluated,
        search_budget_exhausted=exhausted,
        per_length_survivor_counts=tuple(survivor_counts),
        per_length_distance_audit=tuple(distance_audit),
        termination_reason=_termination_reason(
            candidate_provider=candidate_provider,
            survivors=reportable_survivors,
            exhausted=exhausted,
            distance_audit=distance_audit,
            max_sequence_length=config.max_sequence_length,
        ),
        elapsed_seconds=perf_counter() - started,
    )
