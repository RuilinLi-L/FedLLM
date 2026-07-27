"""Position-independent RoPE candidate provider with no language-model prior."""

from __future__ import annotations

from dataclasses import dataclass

from .layer1_filter import Layer1FilterResult


class CandidateProviderError(RuntimeError):
    """Raised when the threshold-filtered candidate vocabulary is invalid."""


@dataclass(frozen=True)
class RoPECandidateProvider:
    """Reuse the same layer-1 candidate set at every Qwen3/RoPE position.

    This matches the existing RoPE DAGER rule: layer-1 token filtering has no
    absolute-position embedding branch, so one threshold-filtered vocabulary is
    supplied to sequence recovery.  The provider neither scores language nor
    introduces a beam or semantic filter.
    """

    token_ids: tuple[int, ...]
    distances: tuple[float, ...]
    eos_token_id: int

    @classmethod
    def from_layer1_result(
        cls, result: Layer1FilterResult, *, eos_token_id: int, max_ids: int = -1
    ) -> "RoPECandidateProvider":
        if not result.token_ids:
            raise CandidateProviderError("Layer-1 DAGER filtering produced no candidate tokens.")
        if len(result.token_ids) != len(result.distances):
            raise CandidateProviderError("Layer-1 DAGER candidate ids and distances have inconsistent lengths.")
        if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise CandidateProviderError(f"eos_token_id must be one non-negative integer, got {eos_token_id!r}.")
        if len(set(result.token_ids)) != len(result.token_ids):
            raise CandidateProviderError("Layer-1 DAGER candidate list contains duplicate token ids.")
        if isinstance(max_ids, bool) or not isinstance(max_ids, int) or max_ids == 0 or max_ids < -1:
            raise CandidateProviderError(f"max_ids must be -1 or one positive integer, got {max_ids!r}.")
        # This is the existing DAGER decoder cap: layer-1 candidates are already
        # sorted by span distance, and only then may the configured max_ids bound
        # the sequence decoder's Cartesian expansion.
        kept = len(result.token_ids) if max_ids < 0 else min(len(result.token_ids), max_ids)
        return cls(
            token_ids=result.token_ids[:kept],
            distances=result.distances[:kept],
            eos_token_id=eos_token_id,
        )

    def candidates_for_position(self, position: int) -> tuple[int, ...]:
        if isinstance(position, bool) or not isinstance(position, int) or position < 0:
            raise CandidateProviderError(f"position must be a non-negative integer, got {position!r}.")
        return self.token_ids
