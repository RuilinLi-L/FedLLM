"""FP32 linear-gradient and active-token row-space diagnostics for Qwen3 q_proj."""

from __future__ import annotations

from typing import Any, Sequence

import torch


class SpanDiagnosticsError(RuntimeError):
    """Raised when q_proj gradient diagnostics cannot be evaluated honestly."""


def _finite(*tensors: torch.Tensor) -> bool:
    return all(bool(torch.isfinite(tensor).all()) for tensor in tensors)


def _summary(values: torch.Tensor) -> dict[str, float | int | None]:
    flat = values.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": int(flat.numel()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "mean": float(flat.mean().item()),
        "median": float(flat.median().item()),
    }


def _validate_positive_finite(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or not float(value) > 0.0 or not torch.isfinite(torch.tensor(value)):
        raise SpanDiagnosticsError(f"{name} must be finite and positive, got {value!r}.")


def _relative_rank(singular_values: torch.Tensor, rank_rtol: float) -> tuple[int, float]:
    if singular_values.numel() == 0 or float(singular_values[0].item()) == 0.0:
        return 0, 0.0
    threshold = float(singular_values[0].item()) * rank_rtol
    return int(torch.count_nonzero(singular_values > threshold).item()), threshold


def _spectral_gap_suggestion(singular_values: torch.Tensor) -> dict[str, float | int | None]:
    if singular_values.numel() < 2:
        return {"suggested_rank": None, "largest_gap_ratio": None, "note": "fewer than two singular values"}
    ratios = singular_values[:-1] / singular_values[1:].clamp_min(1e-30)
    index = int(torch.argmax(ratios).item())
    return {
        "suggested_rank": index + 1,
        "largest_gap_ratio": float(ratios[index].item()),
        "note": "reporting_only_not_used_to_change_any_configuration",
    }


def _row_space_residuals(
    *,
    matrix: torch.Tensor,
    flattened_input: torch.Tensor,
    rank_rtol: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float]:
    """Return residuals and SVD factors for the row space of one explicit matrix."""
    if matrix.ndim != 2 or flattened_input.ndim != 2:
        raise SpanDiagnosticsError("Row-space diagnostic requires two-dimensional matrices.")
    if matrix.shape[1] != flattened_input.shape[1]:
        raise SpanDiagnosticsError(
            f"Gradient row-space feature dimension {matrix.shape[1]} does not match input feature dimension "
            f"{flattened_input.shape[1]}."
        )
    _u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    rank, threshold = _relative_rank(singular_values, rank_rtol)
    if rank == 0:
        projected = torch.zeros_like(flattened_input)
    else:
        basis = vh[:rank, :].transpose(0, 1)
        projected = (flattened_input @ basis) @ basis.transpose(0, 1)
    residuals = (flattened_input - projected).norm(dim=-1) / flattened_input.norm(dim=-1).clamp_min(1e-12)
    return residuals, singular_values, vh, rank, threshold


def linear_gradient_identity(
    *,
    q_input: torch.Tensor,
    delta: torch.Tensor,
    raw_gradient: torch.Tensor,
) -> dict[str, Any]:
    """Check ``G = Delta.T @ H`` in FP32 and record ``G.T`` as a negative control."""
    if q_input.ndim != 3 or delta.ndim != 3 or raw_gradient.ndim != 2:
        raise SpanDiagnosticsError(
            f"Expected H/Delta rank 3 and G rank 2, got H={tuple(q_input.shape)}, "
            f"Delta={tuple(delta.shape)}, G={tuple(raw_gradient.shape)}."
        )
    if q_input.shape[:2] != delta.shape[:2]:
        raise SpanDiagnosticsError(
            f"H and Delta must share batch/sequence dimensions, got H={tuple(q_input.shape)}, Delta={tuple(delta.shape)}."
        )
    d_in = int(q_input.shape[-1])
    d_out = int(delta.shape[-1])
    if tuple(raw_gradient.shape) != (d_out, d_in):
        raise SpanDiagnosticsError(
            f"G shape {tuple(raw_gradient.shape)} must equal (d_out, d_in)=({d_out}, {d_in})."
        )
    h_fp32 = q_input.detach().to(dtype=torch.float32)
    delta_fp32 = delta.detach().to(dtype=torch.float32)
    gradient_fp32 = raw_gradient.detach().to(dtype=torch.float32)
    if not _finite(h_fp32, delta_fp32, gradient_fp32):
        raise SpanDiagnosticsError("H, Delta, and G must be finite before the FP32 identity diagnostic.")
    reconstructed = delta_fp32.reshape(-1, d_out).transpose(0, 1) @ h_fp32.reshape(-1, d_in)
    if not _finite(reconstructed):
        raise SpanDiagnosticsError("G_reconstructed is non-finite in the FP32 linear-gradient identity check.")
    positive_error = float(
        (gradient_fp32 - reconstructed).norm().item() / gradient_fp32.norm().clamp_min(1e-12).item()
    )
    transpose_comparable = tuple(gradient_fp32.transpose(0, 1).shape) == tuple(reconstructed.shape)
    transpose_error: float | None
    if transpose_comparable:
        transpose_error = float(
            (gradient_fp32.transpose(0, 1) - reconstructed).norm().item()
            / reconstructed.norm().clamp_min(1e-12).item()
        )
    else:
        transpose_error = None
    if not torch.isfinite(torch.tensor(positive_error)) or (
        transpose_error is not None and not torch.isfinite(torch.tensor(transpose_error))
    ):
        raise SpanDiagnosticsError("Linear-gradient identity produced a non-finite relative error.")
    return {
        "working_dtype": "torch.float32",
        "h_shape": [int(value) for value in q_input.shape],
        "delta_shape": [int(value) for value in delta.shape],
        "gradient_shape": [int(value) for value in raw_gradient.shape],
        "h_dtype": str(q_input.dtype),
        "delta_dtype": str(delta.dtype),
        "gradient_dtype": str(raw_gradient.dtype),
        "reconstructed_gradient_shape": [int(value) for value in reconstructed.shape],
        "gradient_relative_error": positive_error,
        "gradient_t_relative_error": transpose_error,
        "gradient_t_comparable": transpose_comparable,
        "finite": True,
    }


def _negative_control_worse(
    *,
    positive_identity_error: float,
    transpose_identity_error: float | None,
    positive_active_residual: dict[str, float | int | None],
    transpose_active_residual: dict[str, float | int | None] | None,
    factor: float,
) -> tuple[bool, dict[str, bool]]:
    identity_worse = (
        transpose_identity_error is not None
        and transpose_identity_error >= positive_identity_error * factor
        and transpose_identity_error > positive_identity_error
    )
    positive_max = positive_active_residual["max"]
    transpose_max = None if transpose_active_residual is None else transpose_active_residual["max"]
    residual_worse = (
        isinstance(positive_max, float)
        and isinstance(transpose_max, float)
        and transpose_max >= positive_max * factor
        and transpose_max > positive_max
    )
    transpose_not_comparable = transpose_identity_error is None
    checks = {
        "identity_error_worse": bool(identity_worse),
        "active_residual_worse": bool(residual_worse),
        "transpose_not_comparable": transpose_not_comparable,
    }
    return any(checks.values()), checks


def diagnose_q_projection_layer(
    *,
    q_input: torch.Tensor,
    delta: torch.Tensor,
    raw_gradient: torch.Tensor,
    token_ids: Sequence[int],
    token_texts: Sequence[str],
    eos_token_id: int,
    rank_atol: float,
    rank_rtol: float,
    delta_rtol: float,
    identity_error_tol: float,
    max_active_relative_residual: float,
    negative_control_factor: float,
) -> dict[str, Any]:
    """Diagnose one q_proj layer using the raw ``G`` row space and fixed ``G.T`` control."""
    for name, value in (
        ("rank_rtol", rank_rtol),
        ("delta_rtol", delta_rtol),
        ("identity_error_tol", identity_error_tol),
        ("max_active_relative_residual", max_active_relative_residual),
        ("negative_control_factor", negative_control_factor),
    ):
        _validate_positive_finite(name, value)
    if rank_atol < 0.0 or not torch.isfinite(torch.tensor(rank_atol)):
        raise SpanDiagnosticsError(f"rank_atol must be finite and non-negative, got {rank_atol!r}.")
    identity = linear_gradient_identity(q_input=q_input, delta=delta, raw_gradient=raw_gradient)
    h_fp32 = q_input.detach().to(dtype=torch.float32)
    delta_fp32 = delta.detach().to(dtype=torch.float32)
    gradient_fp32 = raw_gradient.detach().to(dtype=torch.float32)
    sequence_length = int(h_fp32.shape[1])
    if len(token_ids) != sequence_length or len(token_texts) != sequence_length:
        raise SpanDiagnosticsError(
            f"Token diagnostics require {sequence_length} ids/texts, got ids={len(token_ids)}, texts={len(token_texts)}."
        )
    flat_input = h_fp32.reshape(-1, h_fp32.shape[-1])
    residuals, singular_values, _vh, relative_rank, relative_threshold = _row_space_residuals(
        matrix=gradient_fp32,
        flattened_input=flat_input,
        rank_rtol=rank_rtol,
    )
    absolute_rank = int(torch.count_nonzero(singular_values > rank_atol).item())
    delta_flat = delta_fp32.reshape(-1, delta_fp32.shape[-1])
    _du, delta_singular_values, _dvh = torch.linalg.svd(delta_flat, full_matrices=False)
    delta_rank, delta_relative_threshold = _relative_rank(delta_singular_values, rank_rtol)
    d_in = int(h_fp32.shape[-1])
    d_out = int(delta_fp32.shape[-1])
    theoretical_rank_cap = min(sequence_length, delta_rank, d_in, d_out)
    delta_norms = delta_flat.norm(dim=-1)
    max_delta_norm = float(delta_norms.max().item()) if delta_norms.numel() else 0.0
    delta_active_threshold = max_delta_norm * delta_rtol
    active_mask = (delta_norms >= delta_active_threshold) & (delta_norms > 0.0)
    active_residual = _summary(residuals[active_mask])
    all_residual = _summary(residuals)
    inactive_positions = [int(position) for position, active in enumerate(active_mask.tolist()) if not active]
    per_token = [
        {
            "position": position,
            "token_id": int(token_ids[position]),
            "token_text": str(token_texts[position]),
            "is_eos": bool(token_ids[position] == eos_token_id),
            "delta_l2_norm": float(delta_norms[position].item()),
            "relative_row_space_residual": float(residuals[position].item()),
            "active_by_delta": bool(active_mask[position].item()),
        }
        for position in range(sequence_length)
    ]
    transpose_active_residual: dict[str, float | int | None] | None = None
    transpose_row_space_error: str | None = None
    try:
        transpose_residuals, _ts, _tvh, _trank, _tthreshold = _row_space_residuals(
            matrix=gradient_fp32.transpose(0, 1),
            flattened_input=flat_input,
            rank_rtol=rank_rtol,
        )
        transpose_active_residual = _summary(transpose_residuals[active_mask])
    except SpanDiagnosticsError as error:
        transpose_row_space_error = str(error)
    negative_control_worse, negative_control_checks = _negative_control_worse(
        positive_identity_error=float(identity["gradient_relative_error"]),
        transpose_identity_error=identity["gradient_t_relative_error"],
        positive_active_residual=active_residual,
        transpose_active_residual=transpose_active_residual,
        factor=negative_control_factor,
    )
    active_residual_max = active_residual["max"]
    checks = {
        "gradient_identity": bool(identity["gradient_relative_error"] <= identity_error_tol),
        "relative_rank_within_theoretical_cap": bool(relative_rank <= theoretical_rank_cap),
        "active_tokens_present": bool(active_residual["count"] > 0),
        "active_token_residual": bool(
            isinstance(active_residual_max, float) and active_residual_max <= max_active_relative_residual
        ),
        "all_numeric_values_finite": _finite(
            h_fp32,
            delta_fp32,
            gradient_fp32,
            singular_values,
            delta_singular_values,
            delta_norms,
            residuals,
        ),
        "orientation_is_fixed_gradient": True,
        "gradient_t_negative_control_worse": negative_control_worse,
    }
    return {
        "working_dtype": "torch.float32",
        "gradient_orientation": "gradient",
        "svd_basis_direction": "right_singular_vectors_of_gradient",
        "identity": identity,
        "rank": {
            "singular_values": [float(value) for value in singular_values.tolist()],
            "absolute_threshold": float(rank_atol),
            "absolute_threshold_rank": absolute_rank,
            "relative_threshold": float(relative_threshold),
            "relative_threshold_rank": relative_rank,
            "rank_rtol": float(rank_rtol),
            "delta_relative_threshold": float(delta_relative_threshold),
            "delta_relative_rank": delta_rank,
            "theoretical_rank_cap": theoretical_rank_cap,
            "relative_rank_exceeds_theoretical_cap": bool(relative_rank > theoretical_rank_cap),
            "spectral_gap_suggestion": _spectral_gap_suggestion(singular_values),
        },
        "delta_activity": {
            "delta_rtol": float(delta_rtol),
            "max_delta_l2_norm": max_delta_norm,
            "active_threshold": delta_active_threshold,
            "active_token_count": int(active_mask.sum().item()),
            "inactive_token_positions": inactive_positions,
        },
        "per_token": per_token,
        "row_space_residual": {
            "all_tokens": all_residual,
            "active_tokens": active_residual,
            "max_active_relative_residual": float(max_active_relative_residual),
        },
        "gradient_t_negative_control": {
            "gradient_t_active_token_residual": transpose_active_residual,
            "gradient_t_row_space_error": transpose_row_space_error,
            "comparison_factor": float(negative_control_factor),
            "worse_checks": negative_control_checks,
            "is_obviously_worse": negative_control_worse,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def diagnose_two_q_projections(
    *,
    q_inputs: tuple[torch.Tensor, torch.Tensor],
    q_output_gradients: tuple[torch.Tensor, torch.Tensor],
    q_gradients: tuple[torch.Tensor, torch.Tensor],
    q_parameter_names: tuple[str, str],
    token_ids: Sequence[int],
    token_texts: Sequence[str],
    eos_token_id: int,
    rank_atol: float,
    rank_rtol: float,
    delta_rtol: float,
    identity_error_tol: float,
    max_active_relative_residual: float,
    negative_control_factor: float,
) -> dict[str, Any]:
    """Run the fixed positive-gradient and negative-transpose diagnostics for q0 and q1."""
    layers: dict[str, dict[str, Any]] = {}
    for index, (q_input, delta, gradient, name) in enumerate(
        zip(q_inputs, q_output_gradients, q_gradients, q_parameter_names)
    ):
        diagnostic = diagnose_q_projection_layer(
            q_input=q_input,
            delta=delta,
            raw_gradient=gradient,
            token_ids=token_ids,
            token_texts=token_texts,
            eos_token_id=eos_token_id,
            rank_atol=rank_atol,
            rank_rtol=rank_rtol,
            delta_rtol=delta_rtol,
            identity_error_tol=identity_error_tol,
            max_active_relative_residual=max_active_relative_residual,
            negative_control_factor=negative_control_factor,
        )
        diagnostic["parameter_name"] = name
        layers[f"q{index}"] = diagnostic
    return {
        "raw_gradient_space": True,
        "gradient_orientation": "gradient",
        "gradient_t_is_negative_control": True,
        "layers": layers,
        "passed": all(layer["passed"] for layer in layers.values()),
    }
