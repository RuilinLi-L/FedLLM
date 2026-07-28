"""Chunked first-layer DAGER filtering over the Qwen3 tokenizer vocabulary."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Literal

import torch

from utils.functional import check_if_in_span

from .gradient_decomposition import GradientSpan
if TYPE_CHECKING:
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
class VocabularyDistanceChunkDiagnostic:
    """One bounded, threshold-free Qwen3 vocabulary-distance scan chunk."""

    start_token_id: int
    end_token_id_exclusive: int
    elapsed_seconds: float
    scanned_token_count: int


@dataclass(frozen=True)
class Layer1DistanceScanResult:
    """Actual FP32 DAGER distances for every Qwen3 vocabulary token.

    ``token_ids`` remain in the native ascending vocabulary order.  Threshold
    selection intentionally happens outside this object so calibration and the
    formal attack use one candidate-representation and span-distance path.
    """

    token_ids: torch.Tensor
    distances: torch.Tensor
    distance_norm: DistanceNorm
    chunk_size: int
    chunk_diagnostics: tuple[VocabularyDistanceChunkDiagnostic, ...]

    def __post_init__(self) -> None:
        if self.token_ids.ndim != 1 or self.distances.ndim != 1:
            raise Layer1FilterError("Layer-1 distance scan tensors must both be one-dimensional.")
        if self.token_ids.shape != self.distances.shape:
            raise Layer1FilterError("Layer-1 distance scan token ids and distances must have equal shapes.")
        if self.token_ids.dtype != torch.long or self.distances.dtype != torch.float32:
            raise Layer1FilterError(
                "Layer-1 distance scan must retain CPU torch.long ids and CPU torch.float32 distances."
            )
        if self.token_ids.device.type != "cpu" or self.distances.device.type != "cpu":
            raise Layer1FilterError("Layer-1 distance scan tensors must be materialized on CPU.")
        if self.token_ids.numel() == 0 or not bool(torch.isfinite(self.distances).all()):
            raise Layer1FilterError("Layer-1 distance scan must contain finite distances for at least one token.")
        expected_ids = torch.arange(self.token_ids.numel(), dtype=torch.long)
        if not torch.equal(self.token_ids, expected_ids):
            raise Layer1FilterError(
                "Layer-1 distance scan token ids must be the complete native vocabulary in ascending order."
            )

    @property
    def scanned_token_count(self) -> int:
        return int(self.token_ids.numel())


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


def scan_qwen3_vocab_layer1_distances(
    *,
    adapter: Qwen3RoPEDagerAdapter,
    span: GradientSpan,
    vocab_chunk_size: int,
    distance_norm: DistanceNorm = "l2",
) -> Layer1DistanceScanResult:
    """Scan all native layer-0 candidates once without applying a threshold.

    This is the sole source of full-vocabulary Layer-1 distances for both the
    DAGER filter and the calibration observer.  It deliberately accepts no
    ground-truth tokens, threshold, or decoder controls.
    """
    if isinstance(vocab_chunk_size, bool) or not isinstance(vocab_chunk_size, int) or vocab_chunk_size <= 0:
        raise Layer1FilterError(f"vocab_chunk_size must be a positive integer, got {vocab_chunk_size!r}.")
    if span.feature_dim != adapter.metadata.hidden_size:
        raise Layer1FilterError(
            f"Layer-0 DAGER span dimension {span.feature_dim} differs from Qwen3 hidden size "
            f"{adapter.metadata.hidden_size}."
        )

    token_id_chunks: list[torch.Tensor] = []
    distance_chunks: list[torch.Tensor] = []
    diagnostics: list[VocabularyDistanceChunkDiagnostic] = []
    for start in range(0, adapter.metadata.vocab_size, vocab_chunk_size):
        end = min(start + vocab_chunk_size, adapter.metadata.vocab_size)
        started = perf_counter()
        ids = torch.arange(start, end, device=adapter.device, dtype=torch.long)
        representations = adapter.layer0_qproj_inputs_for_token_ids(ids)
        distances = span_distances(basis=span.basis, representations=representations, norm=distance_norm)
        token_id_chunks.append(ids.detach().to(device="cpu", dtype=torch.long))
        distance_chunks.append(distances.detach().to(device="cpu", dtype=torch.float32))
        diagnostics.append(
            VocabularyDistanceChunkDiagnostic(
                start_token_id=start,
                end_token_id_exclusive=end,
                elapsed_seconds=perf_counter() - started,
                scanned_token_count=end - start,
            )
        )

    token_ids = torch.cat(token_id_chunks, dim=0)
    all_distances = torch.cat(distance_chunks, dim=0)
    expected = torch.arange(adapter.metadata.vocab_size, dtype=torch.long)
    if not torch.equal(token_ids, expected):
        raise Layer1FilterError("Layer-1 distance scan did not cover the native vocabulary exactly once in order.")
    return Layer1DistanceScanResult(
        token_ids=token_ids,
        distances=all_distances,
        distance_norm=distance_norm,
        chunk_size=vocab_chunk_size,
        chunk_diagnostics=tuple(diagnostics),
    )


def filter_qwen3_layer1_distance_scan(
    scan: Layer1DistanceScanResult,
    *,
    threshold: float,
) -> Layer1FilterResult:
    """Apply one finite threshold to a shared, already-scanned distance vector."""
    if (
        not isinstance(threshold, (int, float))
        or not torch.isfinite(torch.tensor(float(threshold)))
        or not float(threshold) > 0.0
    ):
        raise Layer1FilterError(f"l1 threshold must be finite and positive, got {threshold!r}.")
    # Root DAGER's ``get_top_B_in_span`` uses a strict predicate.  Keeping the
    # same boundary here makes calibration and reconstruction comparable.
    passed = scan.distances < float(threshold)
    selected_ids = scan.token_ids[passed]
    selected_distances = scan.distances[passed]
    diagnostics: list[VocabularyChunkDiagnostic] = []
    for item in scan.chunk_diagnostics:
        diagnostics.append(
            VocabularyChunkDiagnostic(
                start_token_id=item.start_token_id,
                end_token_id_exclusive=item.end_token_id_exclusive,
                elapsed_seconds=item.elapsed_seconds,
                passing_candidate_count=int(
                    torch.count_nonzero(
                        passed[item.start_token_id : item.end_token_id_exclusive]
                    ).item()
                ),
            )
        )
    # Legacy get_top_B_in_span returns threshold-passing candidates ordered by
    # normalized span distance.  The token-id tie break makes that order stable.
    ordered = sorted(
        zip(selected_ids.tolist(), selected_distances.tolist()),
        key=lambda item: (item[1], item[0]),
    )
    return Layer1FilterResult(
        token_ids=tuple(int(token_id) for token_id, _ in ordered),
        distances=tuple(float(distance) for _, distance in ordered),
        threshold=float(threshold),
        distance_norm=scan.distance_norm,
        chunk_size=scan.chunk_size,
        chunk_diagnostics=tuple(diagnostics),
    )


def filter_qwen3_vocab_layer1(
    *,
    adapter: Qwen3RoPEDagerAdapter,
    span: GradientSpan,
    threshold: float,
    vocab_chunk_size: int,
    distance_norm: DistanceNorm = "l2",
) -> Layer1FilterResult:
    """Scan the vocabulary in bounded chunks using actual layer-0 q_proj inputs."""
    scan = scan_qwen3_vocab_layer1_distances(
        adapter=adapter,
        span=span,
        vocab_chunk_size=vocab_chunk_size,
        distance_norm=distance_norm,
    )
    return filter_qwen3_layer1_distance_scan(scan, threshold=threshold)
