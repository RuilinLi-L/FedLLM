"""Legacy-faithful DAGER span decomposition for native Qwen3 ``nn.Linear`` gradients."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from utils.functional import get_layer_decomp, torch_matrix_rank


class GradientDecompositionError(RuntimeError):
    """Raised when a q_proj gradient cannot define an honest DAGER span."""


RANK_DEFINITION = "absolute_matrix_rank_atol_rtol_zero"


@dataclass(frozen=True)
class AbsoluteMatrixRank:
    """One FP32 legacy effective rank measured with an absolute tolerance."""

    effective_rank: int
    absolute_tolerance: float


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
    rank_atol: float
    q0_effective_rank: int
    q1_effective_rank: int
    requested_shared_rank: int
    applied_shared_rank: int
    rank_cap: int
    rank_was_capped: bool
    cap_reason: str | None


@dataclass(frozen=True)
class GradientSpan:
    """A fixed-orientation q_proj row-space basis plus absolute-rank metadata."""

    basis: torch.Tensor
    effective_rank: int
    absolute_tolerance: float
    requested_rank: int
    applied_rank: int
    rank_cap: int
    rank_was_capped: bool
    cap_reason: str | None
    feature_dim: int
    gradient_shape: tuple[int, int]
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
            f"rank_tolerance must be one finite positive absolute tolerance, got {rank_tolerance!r}."
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


def absolute_matrix_rank(gradient: torch.Tensor, *, rank_tolerance: float, device: torch.device) -> AbsoluteMatrixRank:
    """Replicate the legacy ``torch.linalg.matrix_rank(..., atol, rtol=0)`` rule in FP32."""
    _validate_rank_controls(rank_tolerance=rank_tolerance, rank_cutoff=0)
    if not isinstance(gradient, torch.Tensor) or gradient.ndim != 2:
        raise GradientDecompositionError("Absolute DAGER rank requires one rank-2 gradient tensor.")
    matrix = gradient.detach().float()
    if not bool(torch.isfinite(matrix).all()):
        raise GradientDecompositionError("Absolute DAGER rank requires finite gradient values.")
    try:
        effective_rank = torch_matrix_rank(
            matrix,
            tol=float(rank_tolerance),
            device=device,
            upcast=False,
        )
    except Exception as error:
        raise GradientDecompositionError(
            f"FP32 absolute matrix-rank computation failed: {type(error).__name__}: {error}"
        ) from error
    if effective_rank <= 0:
        raise GradientDecompositionError(
            f"Absolute DAGER effective rank is zero at atol={rank_tolerance}."
        )
    return AbsoluteMatrixRank(effective_rank=effective_rank, absolute_tolerance=float(rank_tolerance))


def _apply_rank_cap(
    *, requested_rank: int, feature_dim: int, rank_cutoff: int, matrix_rank_cap: int
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
    """Build raw-``G`` right-singular DAGER basis with legacy absolute-rank semantics.

    Qwen3 ``q_proj`` uses ``nn.Linear(d_in, d_out)`` and PyTorch reports ``G``
    as ``[d_out, d_in]``.  The final dimension remains ``d_in``; no GPT-2
    Conv1D transpose is applied.  ``B`` is selected by the same absolute rank
    rule as the root DAGER implementation before extracting the basis.
    """
    _validate_rank_controls(rank_tolerance=rank_tolerance, rank_cutoff=rank_cutoff)
    matrix_rank_cap = _validate_qwen3_linear_orientation(gradient, feature_dim=feature_dim, layer_name="q_proj")
    absolute = absolute_matrix_rank(
        gradient,
        rank_tolerance=rank_tolerance,
        device=decomposition_device,
    )
    requested_rank = absolute.effective_rank if shared_truncated_rank is None else shared_truncated_rank
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
        effective_rank=absolute.effective_rank,
        absolute_tolerance=absolute.absolute_tolerance,
        requested_rank=application.requested_rank,
        applied_rank=application.applied_rank,
        rank_cap=application.rank_cap,
        rank_was_capped=application.rank_was_capped,
        cap_reason=application.cap_reason,
        feature_dim=feature_dim,
        gradient_shape=(int(gradient.shape[0]), int(gradient.shape[1])),
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
    """Select the root-DAGER shared ``B=max(rank(q0), rank(q1))`` in FP32."""
    _validate_rank_controls(rank_tolerance=rank_tolerance, rank_cutoff=rank_cutoff)
    if len(gradients) != 2:
        raise GradientDecompositionError("Shared Qwen3 DAGER rank requires exactly q0 and q1 gradients.")
    q0_matrix_cap = _validate_qwen3_linear_orientation(gradients[0], feature_dim=feature_dim, layer_name="q0")
    q1_matrix_cap = _validate_qwen3_linear_orientation(gradients[1], feature_dim=feature_dim, layer_name="q1")
    q0 = absolute_matrix_rank(gradients[0], rank_tolerance=rank_tolerance, device=decomposition_device)
    q1 = absolute_matrix_rank(gradients[1], rank_tolerance=rank_tolerance, device=decomposition_device)
    requested_shared_rank = max(q0.effective_rank, q1.effective_rank)
    application = _apply_rank_cap(
        requested_rank=requested_shared_rank,
        feature_dim=feature_dim,
        rank_cutoff=rank_cutoff,
        matrix_rank_cap=min(q0_matrix_cap, q1_matrix_cap),
    )
    return SharedDagerRank(
        rank_definition=RANK_DEFINITION,
        rank_atol=float(rank_tolerance),
        q0_effective_rank=q0.effective_rank,
        q1_effective_rank=q1.effective_rank,
        requested_shared_rank=requested_shared_rank,
        applied_shared_rank=application.applied_rank,
        rank_cap=application.rank_cap,
        rank_was_capped=application.rank_was_capped,
        cap_reason=application.cap_reason,
    )
