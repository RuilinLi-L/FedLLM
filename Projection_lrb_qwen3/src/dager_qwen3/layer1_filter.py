"""Chunked first-layer DAGER filtering over the Qwen3 tokenizer vocabulary."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import torch

from utils.functional import check_if_in_span

from .gradient_decomposition import GradientSpan
from .model_adapter import Qwen3RoPEDagerAdapter


class Layer1FilterError(RuntimeError):
    """Raised when a first-layer vocabulary scan violates the DAGER contract."""


DistanceNorm = Literal["l2", "l1"]


@dataclass(frozen=True)
class VocabularyChunkDiagnostic:
    """One bounded Qwen3 vocabulary scan chunk."""

    start_token_id: int
    end_token_id_exclusive: int
    elapsed_seconds: float
    passing_candidate_count: int


@dataclass(frozen=True)
class Layer1FilterResult:
    """All threshold-passing layer-0 candidate tokens in standard DAGER order."""

    token_ids: tuple[int, ...]
    distances: tuple[float, ...]
    threshold: float
    distance_norm: DistanceNorm
    chunk_size: int
    chunk_diagnostics: tuple[VocabularyChunkDiagnostic, ...]

    @property
    def candidate_count(self) -> int:
        return len(self.token_ids)


def span_distances(*, basis: torch.Tensor, representations: torch.Tensor, norm: DistanceNorm) -> torch.Tensor:
    """Use the repository's DAGER normalized projection-residual distance exactly."""
    if norm not in ("l2", "l1"):
        raise Layer1FilterError(f"Unsupported DAGER distance norm {norm!r}; expected 'l2' or 'l1'.")
    if basis.ndim != 2 or representations.ndim != 2:
        raise Layer1FilterError(
            f"DAGER span distance requires rank-2 basis/representations, got "
            f"{tuple(basis.shape)} and {tuple(representations.shape)}."
        )
    if basis.shape[1] != representations.shape[1]:
        raise Layer1FilterError(
            f"DAGER basis feature dimension {basis.shape[1]} does not match candidate dimension "
            f"{representations.shape[1]}."
        )
    if not bool(torch.isfinite(basis).all()) or not bool(torch.isfinite(representations).all()):
        raise Layer1FilterError("DAGER span distance requires finite basis and candidate representations.")
    # ``check_if_in_span`` normalizes in-place; clone makes the legacy distance
    # definition reusable without corrupting native Qwen3 candidate tensors.
    distances = check_if_in_span(
        basis.to(device=representations.device, dtype=torch.float32),
        representations.detach().to(dtype=torch.float32).clone(),
        norm=norm,
    )
    if distances.ndim != 1 or distances.shape[0] != representations.shape[0]:
        raise Layer1FilterError(f"DAGER distance returned unexpected shape {tuple(distances.shape)}.")
    if not bool(torch.isfinite(distances).all()):
        raise Layer1FilterError("DAGER span distance returned non-finite values.")
    return distances


def filter_qwen3_vocab_layer1(
    *,
    adapter: Qwen3RoPEDagerAdapter,
    span: GradientSpan,
    threshold: float,
    vocab_chunk_size: int,
    distance_norm: DistanceNorm = "l2",
) -> Layer1FilterResult:
    """Scan the vocabulary in bounded chunks using actual layer-0 q_proj inputs."""
    if (
        not isinstance(threshold, (int, float))
        or not torch.isfinite(torch.tensor(float(threshold)))
        or not float(threshold) > 0.0
    ):
        raise Layer1FilterError(f"l1 threshold must be finite and positive, got {threshold!r}.")
    if isinstance(vocab_chunk_size, bool) or not isinstance(vocab_chunk_size, int) or vocab_chunk_size <= 0:
        raise Layer1FilterError(f"vocab_chunk_size must be a positive integer, got {vocab_chunk_size!r}.")
    if span.feature_dim != adapter.metadata.hidden_size:
        raise Layer1FilterError(
            f"Layer-0 DAGER span dimension {span.feature_dim} differs from Qwen3 hidden size "
            f"{adapter.metadata.hidden_size}."
        )
    token_ids: list[int] = []
    distances_out: list[float] = []
    diagnostics: list[VocabularyChunkDiagnostic] = []
    for start in range(0, adapter.metadata.vocab_size, vocab_chunk_size):
        end = min(start + vocab_chunk_size, adapter.metadata.vocab_size)
        started = perf_counter()
        ids = torch.arange(start, end, device=adapter.device, dtype=torch.long)
        representations = adapter.layer0_qproj_inputs_for_token_ids(ids)
        distances = span_distances(basis=span.basis, representations=representations, norm=distance_norm)
        passed = distances < float(threshold)
        selected_ids = ids[passed]
        selected_distances = distances[passed]
        token_ids.extend(int(value) for value in selected_ids.detach().cpu().tolist())
        distances_out.extend(float(value) for value in selected_distances.detach().cpu().tolist())
        diagnostics.append(
            VocabularyChunkDiagnostic(
                start_token_id=start,
                end_token_id_exclusive=end,
                elapsed_seconds=perf_counter() - started,
                passing_candidate_count=int(selected_ids.numel()),
            )
        )
    # Legacy get_top_B_in_span sorts passing candidates by their span distance.
    ordered = sorted(zip(token_ids, distances_out), key=lambda item: (item[1], item[0]))
    return Layer1FilterResult(
        token_ids=tuple(token_id for token_id, _ in ordered),
        distances=tuple(distance for _, distance in ordered),
        threshold=float(threshold),
        distance_norm=distance_norm,
        chunk_size=vocab_chunk_size,
        chunk_diagnostics=tuple(diagnostics),
    )
