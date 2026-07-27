"""Threshold-filtered Qwen3/RoPE sequence recovery using the second q_proj span."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from .candidate_provider import RoPECandidateProvider
from .gradient_decomposition import GradientSpan
from .layer1_filter import DistanceNorm, Layer1FilterError, span_distances
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
class Layer2DecodeResult:
    """Results of exhaustive, threshold-filtered prefix expansion up to its budget."""

    selected_token_ids: tuple[int, ...]
    selected_mean_span_distance: float | None
    completed_prefixes: tuple[DecodedPrefix, ...]
    evaluated_prefix_count: int
    search_budget_exhausted: bool
    per_length_survivor_counts: tuple[tuple[int, int], ...]
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
    passes = (per_prefix < threshold).all(dim=1)
    means = per_prefix.mean(dim=1)
    return [bool(value) for value in passes.detach().cpu().tolist()], [
        float(value) for value in means.detach().cpu().tolist()
    ]


def decode_qwen3_rope_prefixes(
    *,
    adapter: Qwen3RoPEDagerAdapter,
    span: GradientSpan,
    candidate_provider: RoPECandidateProvider,
    config: Layer2DecoderConfig,
) -> Layer2DecodeResult:
    """Recover EOS-terminated prefixes through standard DAGER span thresholding.

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
    evaluated = 0
    exhausted = False
    survivor_counts: list[tuple[int, int]] = []
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
            for token_ids, passed, score in zip(batch, passes, scores):
                if not passed:
                    continue
                accepted = DecodedPrefix(token_ids, score)
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
            for token_ids, passed, score in zip(batch, passes, scores):
                if not passed:
                    continue
                accepted = DecodedPrefix(token_ids, score)
                if accepted.token_ids[-1] == candidate_provider.eos_token_id:
                    completed.append(accepted)
                elif sequence_length < config.max_sequence_length:
                    next_survivors.append(accepted)
        survivor_counts.append((sequence_length, len(next_survivors)))
        if exhausted or sequence_length >= config.max_sequence_length:
            break
        # This is exhaustive threshold-filtered expansion, not beam search:
        # the next iteration streams every accepted prefix/token pair in chunks.
        survivors = next_survivors
    selected = completed[0] if completed else None
    return Layer2DecodeResult(
        selected_token_ids=() if selected is None else selected.token_ids,
        selected_mean_span_distance=None if selected is None else selected.mean_span_distance,
        completed_prefixes=tuple(completed),
        evaluated_prefix_count=evaluated,
        search_budget_exhausted=exhausted,
        per_length_survivor_counts=tuple(survivor_counts),
        elapsed_seconds=perf_counter() - started,
    )
