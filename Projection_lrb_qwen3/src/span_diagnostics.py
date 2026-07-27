"""FP32 raw-gradient row-space diagnostics for Qwen3 q_proj inputs."""

from __future__ import annotations

from typing import Any, Literal

import torch


class SpanDiagnosticsError(RuntimeError):
    """Raised when a requested gradient orientation cannot define the input row space."""


GradientOrientation = Literal["gradient", "gradient.T"]


def _summary(values: torch.Tensor) -> dict[str, float | int]:
    flat = values.detach().float().reshape(-1)
    if flat.numel() == 0:
        raise SpanDiagnosticsError("Cannot summarize an empty tensor.")
    return {
        "count": int(flat.numel()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "mean": float(flat.mean().item()),
        "median": float(flat.median().item()),
    }


def _svd_summary(singular_values: torch.Tensor, *, rank: int) -> dict[str, Any]:
    count = int(singular_values.numel())
    top_count = min(8, count)
    return {
        "count": count,
        "rank": rank,
        "largest": float(singular_values[0].item()) if count else 0.0,
        "smallest": float(singular_values[-1].item()) if count else 0.0,
        "top_values": [float(value) for value in singular_values[:top_count].tolist()],
    }


def diagnose_raw_gradient_row_space(
    *,
    q_input: torch.Tensor,
    raw_gradient: torch.Tensor,
    rank_tol: float,
    max_relative_residual: float,
    gradient_orientation: GradientOrientation = "gradient",
) -> dict[str, Any]:
    """Measure q_proj inputs against the FP32 row space of one explicit orientation.

    ``gradient.T`` is never selected automatically.  The right singular vectors
    of the explicitly selected matrix form the tested row-space basis.
    """
    if gradient_orientation not in ("gradient", "gradient.T"):
        raise SpanDiagnosticsError(f"Unsupported gradient orientation: {gradient_orientation!r}.")
    if rank_tol <= 0.0:
        raise SpanDiagnosticsError(f"rank_tol must be positive, got {rank_tol}.")
    if max_relative_residual < 0.0:
        raise SpanDiagnosticsError(
            f"max_relative_residual must be non-negative, got {max_relative_residual}."
        )
    if raw_gradient.ndim != 2 or q_input.ndim < 1:
        raise SpanDiagnosticsError(
            f"Expected a 2-D gradient and tensor input, got grad={tuple(raw_gradient.shape)}, "
            f"input={tuple(q_input.shape)}."
        )
    input_fp32 = q_input.detach().to(dtype=torch.float32)
    gradient_fp32 = raw_gradient.detach().to(dtype=torch.float32)
    oriented_gradient = gradient_fp32 if gradient_orientation == "gradient" else gradient_fp32.transpose(0, 1)
    feature_dim = int(input_fp32.shape[-1])
    if int(oriented_gradient.shape[1]) != feature_dim:
        raise SpanDiagnosticsError(
            "Selected SVD row-space feature dimension does not match q_proj input: "
            f"orientation={gradient_orientation}, oriented_gradient={tuple(oriented_gradient.shape)}, "
            f"input_feature_dim={feature_dim}."
        )
    _u, singular_values, vh = torch.linalg.svd(oriented_gradient, full_matrices=False)
    rank = int(torch.count_nonzero(singular_values > rank_tol).item())
    flat_input = input_fp32.reshape(-1, feature_dim)
    if rank == 0:
        projected = torch.zeros_like(flat_input)
    else:
        basis = vh[:rank, :].transpose(0, 1)
        projected = (flat_input @ basis) @ basis.transpose(0, 1)
    residual = flat_input - projected
    relative_residual = residual.norm(dim=-1) / flat_input.norm(dim=-1).clamp_min(1e-12)
    basis_direction = (
        "right_singular_vectors_of_gradient"
        if gradient_orientation == "gradient"
        else "right_singular_vectors_of_gradient.T"
    )
    residual_summary = _summary(relative_residual)
    return {
        "working_dtype": "torch.float32",
        "raw_gradient_dtype": str(raw_gradient.dtype),
        "q_input_dtype": str(q_input.dtype),
        "gradient_orientation": gradient_orientation,
        "svd_basis_direction": basis_direction,
        "rank_threshold": float(rank_tol),
        "matrix_shape": [int(value) for value in oriented_gradient.shape],
        "q_input_shape": [int(value) for value in q_input.shape],
        "numerical_rank": rank,
        "singular_value_summary": _svd_summary(singular_values, rank=rank),
        "relative_residual": residual_summary,
        "max_relative_residual": float(max_relative_residual),
        "passes_relative_residual": bool(residual_summary["max"] <= max_relative_residual),
    }


def diagnose_two_q_projections(
    *,
    q_inputs: tuple[torch.Tensor, torch.Tensor],
    q_gradients: tuple[torch.Tensor, torch.Tensor],
    q_parameter_names: tuple[str, str],
    rank_tol: float,
    max_relative_residual: float,
    gradient_orientation: GradientOrientation,
) -> dict[str, Any]:
    """Run the same explicit raw-row-space diagnostic for q0 and q1."""
    layers: dict[str, dict[str, Any]] = {}
    for index, (q_input, gradient, name) in enumerate(zip(q_inputs, q_gradients, q_parameter_names)):
        diagnostic = diagnose_raw_gradient_row_space(
            q_input=q_input,
            raw_gradient=gradient,
            rank_tol=rank_tol,
            max_relative_residual=max_relative_residual,
            gradient_orientation=gradient_orientation,
        )
        diagnostic["parameter_name"] = name
        diagnostic["gradient_shape"] = [int(value) for value in gradient.shape]
        layers[f"q{index}"] = diagnostic
    return {
        "raw_gradient_space": True,
        "gradient_orientation": gradient_orientation,
        "rank_threshold": float(rank_tol),
        "max_relative_residual": float(max_relative_residual),
        "layers": layers,
        "passed": all(layer["passes_relative_residual"] for layer in layers.values()),
    }
