"""DAGER span decomposition for the native Qwen3 ``nn.Linear`` orientation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from utils.functional import get_layer_decomp, torch_matrix_rank


class GradientDecompositionError(RuntimeError):
    """Raised when a q_proj gradient cannot define an honest DAGER span."""


@dataclass(frozen=True)
class GradientSpan:
    """One fixed-orientation q_proj row-space basis and its DAGER rank metadata."""

    basis: torch.Tensor
    raw_rank: int
    truncated_rank: int
    feature_dim: int
    gradient_shape: tuple[int, int]
    rank_tolerance: float | None
    rank_cutoff: int
    orientation: str
    decomposition_device: str


def _validate_rank_controls(*, rank_tolerance: float | None, rank_cutoff: int) -> None:
    if rank_tolerance is not None and (
        isinstance(rank_tolerance, bool)
        or not isinstance(rank_tolerance, (int, float))
        or float(rank_tolerance) < 0.0
        or not torch.isfinite(torch.tensor(float(rank_tolerance)))
    ):
        raise GradientDecompositionError(
            f"rank_tolerance must be None or one finite non-negative float, got {rank_tolerance!r}."
        )
    if isinstance(rank_cutoff, bool) or not isinstance(rank_cutoff, int) or rank_cutoff < 0:
        raise GradientDecompositionError(f"rank_cutoff must be a non-negative integer, got {rank_cutoff!r}.")


def decompose_qwen3_qproj_gradient(
    gradient: torch.Tensor,
    *,
    feature_dim: int,
    rank_tolerance: float | None,
    rank_cutoff: int,
    decomposition_device: torch.device,
    shared_truncated_rank: int | None = None,
) -> GradientSpan:
    """Build the raw ``G`` row-space basis without a GPT-2 Conv1D transpose.

    Qwen3 q_proj is ``nn.Linear(d_in, d_out)`` and its PyTorch gradient has
    shape ``[d_out, d_in]``.  Therefore the DAGER candidate representation is
    tested against the *right* singular vectors of raw ``G``.  The legacy
    ``torch_matrix_rank`` and ``get_layer_decomp`` helpers retain the existing
    absolute-tolerance/rank-cutoff semantics.
    """
    _validate_rank_controls(rank_tolerance=rank_tolerance, rank_cutoff=rank_cutoff)
    if not isinstance(gradient, torch.Tensor) or gradient.ndim != 2:
        raise GradientDecompositionError(
            f"Qwen3 q_proj gradient must be rank 2, got {type(gradient).__name__} "
            f"with shape {getattr(gradient, 'shape', None)}."
        )
    if not gradient.is_floating_point() or not bool(torch.isfinite(gradient).all()):
        raise GradientDecompositionError("Qwen3 q_proj gradient must be finite and floating point.")
    if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim <= 0:
        raise GradientDecompositionError(f"feature_dim must be positive, got {feature_dim!r}.")
    if int(gradient.shape[1]) != feature_dim:
        raise GradientDecompositionError(
            "Qwen3 q_proj raw gradient must retain nn.Linear orientation [d_out, d_in]; "
            f"got shape {tuple(gradient.shape)} and expected d_in={feature_dim} in the final dimension. "
            "Do not transpose it as GPT-2 Conv1D gradients are transposed."
        )
    rank_cap = min(int(gradient.shape[0]), int(gradient.shape[1]))
    try:
        raw_rank = torch_matrix_rank(
            gradient,
            tol=None if rank_tolerance is None else float(rank_tolerance),
            device=decomposition_device,
            upcast=True,
        )
    except Exception as error:
        raise GradientDecompositionError(
            f"DAGER rank computation failed on {decomposition_device}: {type(error).__name__}: {error}"
        ) from error
    if shared_truncated_rank is None:
        truncated_rank = min(raw_rank, feature_dim - rank_cutoff, rank_cap)
    else:
        if (
            isinstance(shared_truncated_rank, bool)
            or not isinstance(shared_truncated_rank, int)
            or shared_truncated_rank <= 0
        ):
            raise GradientDecompositionError(
                f"shared_truncated_rank must be one positive integer, got {shared_truncated_rank!r}."
            )
        truncated_rank = shared_truncated_rank
        if truncated_rank > min(feature_dim - rank_cutoff, rank_cap):
            raise GradientDecompositionError(
                f"shared_truncated_rank={truncated_rank} exceeds this layer's cap "
                f"min(feature_dim-rank_cutoff={feature_dim-rank_cutoff}, rank_cap={rank_cap})."
            )
    if truncated_rank <= 0:
        raise GradientDecompositionError(
            "DAGER rank became non-positive after the legacy rank cutoff: "
            f"raw_rank={raw_rank}, feature_dim={feature_dim}, rank_cutoff={rank_cutoff}, rank_cap={rank_cap}."
        )
    try:
        # Upcast before invoking the legacy helper, so its documented return cast
        # remains FP32 rather than returning a BF16 basis for span distances.
        _rank, basis = get_layer_decomp(
            gradient.detach().to(dtype=torch.float32),
            B=truncated_rank,
            tol=None if rank_tolerance is None else float(rank_tolerance),
            upcast=False,
            device=decomposition_device,
        )
    except Exception as error:
        raise GradientDecompositionError(
            f"DAGER SVD decomposition failed on {decomposition_device}: {type(error).__name__}: {error}"
        ) from error
    # ``utils.functional.get_layer_decomp`` returns the legacy DAGER layout
    # ``[rank, feature]``.  Its paired ``check_if_in_span`` einsum consumes
    # exactly this layout, so preserve it rather than transposing the basis.
    if basis.ndim != 2 or tuple(basis.shape) != (truncated_rank, feature_dim):
        raise GradientDecompositionError(
            f"Legacy DAGER decomposition returned basis shape {tuple(basis.shape)}, expected "
            f"({truncated_rank}, {feature_dim})."
        )
    if not bool(torch.isfinite(basis).all()):
        raise GradientDecompositionError("DAGER basis contains non-finite values.")
    return GradientSpan(
        basis=basis.detach().to(device=gradient.device, dtype=torch.float32),
        raw_rank=raw_rank,
        truncated_rank=truncated_rank,
        feature_dim=feature_dim,
        gradient_shape=(int(gradient.shape[0]), int(gradient.shape[1])),
        rank_tolerance=None if rank_tolerance is None else float(rank_tolerance),
        rank_cutoff=rank_cutoff,
        orientation="raw_qwen3_nn_linear_gradient_right_singular_vectors",
        decomposition_device=str(decomposition_device),
    )


def shared_dager_rank_for_qwen3_qproj_gradients(
    gradients: tuple[torch.Tensor, torch.Tensor],
    *,
    feature_dim: int,
    rank_tolerance: float | None,
    rank_cutoff: int,
    decomposition_device: torch.device,
) -> tuple[int, tuple[int, int]]:
    """Apply legacy DAGER's shared ``B=max(rank_l)`` rule to q0/q1 gradients."""
    _validate_rank_controls(rank_tolerance=rank_tolerance, rank_cutoff=rank_cutoff)
    raw_ranks: list[int] = []
    rank_cap: int | None = None
    for layer_index, gradient in enumerate(gradients):
        if not isinstance(gradient, torch.Tensor) or gradient.ndim != 2:
            raise GradientDecompositionError(f"q{layer_index} gradient must be a rank-2 tensor.")
        if int(gradient.shape[1]) != feature_dim:
            raise GradientDecompositionError(
                f"q{layer_index} gradient has final dimension {gradient.shape[1]}, expected d_in={feature_dim}; "
                "its GPT-2-style transpose is forbidden."
            )
        if not bool(torch.isfinite(gradient).all()):
            raise GradientDecompositionError(f"q{layer_index} gradient contains non-finite values.")
        try:
            rank = torch_matrix_rank(
                gradient,
                tol=None if rank_tolerance is None else float(rank_tolerance),
                device=decomposition_device,
                upcast=True,
            )
        except Exception as error:
            raise GradientDecompositionError(
                f"DAGER rank computation failed for q{layer_index}: {type(error).__name__}: {error}"
            ) from error
        raw_ranks.append(rank)
        current_cap = min(int(gradient.shape[0]), int(gradient.shape[1]))
        rank_cap = current_cap if rank_cap is None else min(rank_cap, current_cap)
    if rank_cap is None:
        raise GradientDecompositionError("No q_proj gradients were supplied for shared DAGER rank selection.")
    shared_rank = min(max(raw_ranks), feature_dim - rank_cutoff, rank_cap)
    if shared_rank <= 0:
        raise GradientDecompositionError(
            f"Shared DAGER rank became non-positive: raw_ranks={raw_ranks}, feature_dim={feature_dim}, "
            f"rank_cutoff={rank_cutoff}, rank_cap={rank_cap}."
        )
    return shared_rank, (raw_ranks[0], raw_ranks[1])
