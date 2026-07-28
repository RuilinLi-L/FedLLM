"""Shared Qwen3 q_proj diagnostic gate for attack-adjacent runners."""

from __future__ import annotations

from typing import Any

from src.gradient_capture import CapturedGradientStep
from src.span_diagnostics import diagnose_two_q_projections


class GradientGateError(RuntimeError):
    """Raised when a Qwen3 diagnostic token representation is malformed."""


def diagnostic_thresholds(dtype: str) -> dict[str, float]:
    """Return the frozen, precision-specific structural diagnostic controls."""
    if dtype not in ("bfloat16", "float32"):
        raise GradientGateError(f"Unsupported dtype {dtype!r}.")
    return {
        "rank_atol": 1e-6,
        "rank_rtol": 1e-3,
        "delta_rtol": 1e-3,
        "identity_error_tol": 5e-3,
        "max_active_relative_residual": 3e-3 if dtype == "bfloat16" else 2e-4,
        "negative_control_factor": 10.0,
    }


def decode_token_texts(tokenizer: Any, token_ids: tuple[int, ...]) -> list[str]:
    """Decode immutable token ids for diagnostics without adding model inputs."""
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if not callable(convert):
        raise GradientGateError("Qwen3 tokenizer lacks convert_ids_to_tokens.")
    values = convert(list(token_ids))
    if not isinstance(values, list) or len(values) != len(token_ids) or any(
        not isinstance(value, str) for value in values
    ):
        raise GradientGateError("Qwen3 tokenizer returned invalid convert_ids_to_tokens values.")
    return values


def diagnose_captured_q_projections(
    *,
    captured: CapturedGradientStep,
    tokenizer: Any,
    token_ids: tuple[int, ...],
    eos_token_id: int,
    dtype: str,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Run the exact structural gate shared by none-only and calibration flows."""
    controls = diagnostic_thresholds(dtype)
    diagnostic = diagnose_two_q_projections(
        q_inputs=captured.q_inputs,
        q_output_gradients=captured.q_output_gradients,
        q_gradients=captured.q_gradients,
        q_parameter_names=captured.q_parameter_names,
        token_ids=token_ids,
        token_texts=decode_token_texts(tokenizer, token_ids),
        eos_token_id=eos_token_id,
        **controls,
    )
    return diagnostic, controls
