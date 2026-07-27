"""Relative-SVD DAGER span decomposition for native Qwen3 ``nn.Linear`` gradients."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from utils.functional import get_layer_decomp


class GradientDecompositionError(RuntimeError):
    """Raised when a q_proj gradient cannot define an honest DAGER span."""


RANK_DEFINITION = "relative_svd_threshold"


@dataclass(frozen=True)
class RelativeSvdRank:
    """The predeclared effective rank of one raw Qwen3 q_proj gradient."""

    effective_rank: int
    relative_threshold: float
    largest_singular_value: float


@dataclass(frozen=True)
class RankApplication:
    """One explicit application of unavoidable decomposition caps."""

    requested_rank: int
    applied_rank: int
    rank_cap: int
    rank_was_capped: bool
    cap_reason: str | None


@dataclass(frozen=True)
class SharedDagerRank:
    """Shared two-layer DAGER rank selected before any span basis is constructed."""

    rank_definition: str
    rank_rtol: float
    q0_effective_rank: int
    q1_effective_rank: int
    q0_relative_threshold: float
    q1_relative_threshold: float
    requested_shared_rank: int
    applied_shared_rank: int
    rank_cap: int
    rank_was_capped: bool
    cap_reason: str | None


@dataclass(frozen=True)
class GradientSpan:
    """A fixed-orientation q_proj row-space basis plus relative-rank metadata."""

    basis: torch.Tensor
    effective_rank: int
    relative_threshold: float
    largest_singular_value: float
    requested_rank: int
    applied_rank: int
    rank_cap: int
    rank_was_capped: bool
    cap_reason: str | None
    feature_dim: int
    gradient_shape: tuple[int, int]
    rank_rtol: float
    rank_cutoff: int
    orientation: str
    decomposition_device: str

    @property
    def truncated_rank(self) -> int:
        """Compatibility alias for callers that name the applied DAGER basis width B."""
        return self.applied_rank


def _validate_rank_controls(*, rank_tolerance: float, rank_cutoff: int) -> None:
    if (
        isinstance(rank_tolerance, bool)
        or not isinstance(rank_tolerance, (int, float))
        or not torch.isfinite(torch.tensor(float(rank_tolerance)))
        or float(rank_tolerance) <= 0.0
    ):
        raise GradientDecompositionError(
            f"rank_tolerance must be one finite positive relative tolerance, got {rank_tolerance!r}."
        )
    if isinstance(rank_cutoff, bool) or not isinstance(rank_cutoff, int) or rank_cutoff < 0:
        raise GradientDecompositionError(f"rank_cutoff must be a non-negative integer, got {rank_cutoff!r}.")


def _validate_qwen3_linear_orientation(gradient: torch.Tensor, *, feature_dim: int, layer_name: str) -> int:
    if not isinstance(gradient, torch.Tensor) or gradient.ndim != 2:
        raise GradientDecompositionError(f"{layer_name} gradient must be one rank-2 tensor.")
    if not gradient.is_floating_point() or not bool(torch.isfinite(gradient).all()):
        raise GradientDecompositionError(f"{layer_name} gradient must be finite and floating point.")
    if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim <= 0:
        raise GradientDecompositionError(f"feature_dim must be positive, got {feature_dim!r}.")
    if int(gradient.shape[1]) != feature_dim:
        raise GradientDecompositionError(
            f"{layer_name} q_proj raw gradient must retain nn.Linear orientation [d_out, d_in]; "
            f"got shape {tuple(gradient.shape)} and expected d_in={feature_dim} in the final dimension. "
            "GPT-2 Conv1D-style gradient transpose is forbidden."
        )
    return min(int(gradient.shape[0]), int(gradient.shape[1]))


def relative_svd_rank(gradient: torch.Tensor, *, rank_tolerance: float) -> RelativeSvdRank:
    """Compute one effective rank from ``gradient.detach().float()`` exactly once."""
    _validate_rank_controls(rank_tolerance=rank_tolerance, rank_cutoff=0)
    if not isinstance(gradient, torch.Tensor) or gradient.ndim != 2:
        raise GradientDecompositionError("Relative SVD rank requires one rank-2 gradient tensor.")
    matrix = gradient.detach().float()
    if not bool(torch.isfinite(matrix).all()):
        raise GradientDecompositionError("Relative SVD rank requires finite gradient values.")
    try:
        singular_values = torch.linalg.svdvals(matrix)
    except Exception as error:
        raise GradientDecompositionError(
            f"FP32 torch.linalg.svdvals failed: {type(error).__name__}: {error}"
        ) from error
    if singular_values.numel() == 0 or not bool(torch.isfinite(singular_values).all()):
        raise GradientDecompositionError("FP32 singular-value computation returned no finite singular values.")
    largest = float(singular_values[0].item())
    if largest <= 0.0:
        raise GradientDecompositionError("A zero q_proj gradient has no positive relative-SVD DAGER rank.")
    threshold = largest * float(rank_tolerance)
    effective_rank = int(torch.count_nonzero(singular_values >= threshold).item())
    if effective_rank <= 0:
        raise GradientDecompositionError(
            f"Relative-SVD effective rank is zero at threshold={threshold} with rank_rtol={rank_tolerance}."
        )
    return RelativeSvdRank(
        effective_rank=effective_rank,
        relative_threshold=threshold,
        largest_singular_value=largest,
    )


def _apply_rank_cap(
    *, requested_rank: int,
    feature_dim: int,
    rank_cutoff: int,
    matrix_rank_cap: int,
) -> RankApplication:
    if requested_rank <= 0:
        raise GradientDecompositionError(f"requested rank must be positive, got {requested_rank}.")
    feature_cap = feature_dim - rank_cutoff
    if feature_cap <= 0:
        raise GradientDecompositionError(
            f"feature_dim-rank_cutoff must be positive, got {feature_dim}-{rank_cutoff}."
        )
    if matrix_rank_cap <= 0:
        raise GradientDecompositionError(f"matrix rank cap must be positive, got {matrix_rank_cap}.")
    rank_cap = min(feature_cap, matrix_rank_cap)
    applied_rank = min(requested_rank, rank_cap)
    reasons: list[str] = []
    if requested_rank > feature_cap:
        reasons.append("feature_dim_minus_rank_cutoff")
    if requested_rank > matrix_rank_cap:
        reasons.append("matrix_dimension")
    return RankApplication(
        requested_rank=requested_rank,
        applied_rank=applied_rank,
        rank_cap=rank_cap,
        rank_was_capped=bool(reasons),
        cap_reason="+".join(reasons) if reasons else None,
    )


def decompose_qwen3_qproj_gradient(
    gradient: torch.Tensor,
    *,
    feature_dim: int,
    rank_tolerance: float,
    rank_cutoff: int,
    decomposition_device: torch.device,
    shared_truncated_rank: int | None = None,
) -> GradientSpan:
    """Build raw-``G`` right-singular DAGER basis with relative-SVD rank semantics.

    Qwen3 ``q_proj`` uses ``nn.Linear(d_in, d_out)`` and PyTorch reports
    ``G`` as ``[d_out, d_in]``.  The final dimension must therefore remain
    ``d_in``; no GPT-2 Conv1D transpose is ever attempted.  The existing
    ``get_layer_decomp`` helper remains only a basis extractor and always
    receives the explicitly selected applied rank.
    """
    _validate_rank_controls(rank_tolerance=rank_tolerance, rank_cutoff=rank_cutoff)
    matrix_rank_cap = _validate_qwen3_linear_orientation(gradient, feature_dim=feature_dim, layer_name="q_proj")
    relative = relative_svd_rank(gradient, rank_tolerance=rank_tolerance)
    requested_rank = relative.effective_rank if shared_truncated_rank is None else shared_truncated_rank
    if isinstance(requested_rank, bool) or not isinstance(requested_rank, int) or requested_rank <= 0:
        raise GradientDecompositionError(
            f"shared_truncated_rank must be absent or a positive integer, got {shared_truncated_rank!r}."
        )
    application = _apply_rank_cap(
        requested_rank=requested_rank,
        feature_dim=feature_dim,
        rank_cutoff=rank_cutoff,
        matrix_rank_cap=matrix_rank_cap,
    )
    try:
        # Upcast before the legacy basis helper so its output remains FP32.  B
        # is already fixed above; this call must not choose rank on its own.
        _rank, basis = get_layer_decomp(
            gradient.detach().float(),
            B=application.applied_rank,
            tol=None,
            upcast=False,
            device=decomposition_device,
        )
    except Exception as error:
        raise GradientDecompositionError(
            f"DAGER basis decomposition failed on {decomposition_device}: {type(error).__name__}: {error}"
        ) from error
    if basis.ndim != 2 or tuple(basis.shape) != (application.applied_rank, feature_dim):
        raise GradientDecompositionError(
            f"Legacy DAGER basis shape {tuple(basis.shape)} differs from expected "
            f"({application.applied_rank}, {feature_dim})."
        )
    if not bool(torch.isfinite(basis).all()):
        raise GradientDecompositionError("DAGER basis contains non-finite values.")
    return GradientSpan(
        basis=basis.detach().to(device=gradient.device, dtype=torch.float32),
        effective_rank=relative.effective_rank,
        relative_threshold=relative.relative_threshold,
        largest_singular_value=relative.largest_singular_value,
        requested_rank=application.requested_rank,
        applied_rank=application.applied_rank,
        rank_cap=application.rank_cap,
        rank_was_capped=application.rank_was_capped,
        cap_reason=application.cap_reason,
        feature_dim=feature_dim,
        gradient_shape=(int(gradient.shape[0]), int(gradient.shape[1])),
        rank_rtol=float(rank_tolerance),
        rank_cutoff=rank_cutoff,
        orientation="raw_qwen3_nn_linear_gradient_right_singular_vectors",
        decomposition_device=str(decomposition_device),
    )


def shared_dager_rank_for_qwen3_qproj_gradients(
    gradients: tuple[torch.Tensor, torch.Tensor],
    *,
    feature_dim: int,
    rank_tolerance: float,
    rank_cutoff: int,
    decomposition_device: torch.device,
) -> SharedDagerRank:
    """Select q0/q1's shared rank by the predeclared relative-SVD rule only.

    Captured activations, output gradients, text, and diagnostic rank caps are
    intentionally absent from this function.  Matrix dimensions and the legacy
    rank cutoff can only constrain the requested rank after it is recorded.
    """
    del decomposition_device  # Basis extraction, not rank selection, uses this device.
    _validate_rank_controls(rank_tolerance=rank_tolerance, rank_cutoff=rank_cutoff)
    if len(gradients) != 2:
        raise GradientDecompositionError("Shared Qwen3 DAGER rank requires exactly q0 and q1 gradients.")
    q0_matrix_cap = _validate_qwen3_linear_orientation(gradients[0], feature_dim=feature_dim, layer_name="q0")
    q1_matrix_cap = _validate_qwen3_linear_orientation(gradients[1], feature_dim=feature_dim, layer_name="q1")
    q0 = relative_svd_rank(gradients[0], rank_tolerance=rank_tolerance)
    q1 = relative_svd_rank(gradients[1], rank_tolerance=rank_tolerance)
    requested_shared_rank = max(q0.effective_rank, q1.effective_rank)
    application = _apply_rank_cap(
        requested_rank=requested_shared_rank,
        feature_dim=feature_dim,
        rank_cutoff=rank_cutoff,
        matrix_rank_cap=min(q0_matrix_cap, q1_matrix_cap),
    )
    return SharedDagerRank(
        rank_definition=RANK_DEFINITION,
        rank_rtol=float(rank_tolerance),
        q0_effective_rank=q0.effective_rank,
        q1_effective_rank=q1.effective_rank,
        q0_relative_threshold=q0.relative_threshold,
        q1_relative_threshold=q1.relative_threshold,
        requested_shared_rank=requested_shared_rank,
        applied_shared_rank=application.applied_rank,
        rank_cap=application.rank_cap,
        rank_was_capped=application.rank_was_capped,
        cap_reason=application.cap_reason,
    )
