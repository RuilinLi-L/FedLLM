"""Canonical Qwen3 named-gradient capture and exact q_proj forward-input hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn


class GradientCaptureError(RuntimeError):
    """Raised when one-step Qwen3 gradient capture violates the required contract."""


@dataclass(frozen=True)
class QProjectionPair:
    """The two explicitly named Qwen3 q_proj modules and derived weight shape."""

    modules: tuple[nn.Module, nn.Module]
    parameter_names: tuple[str, str]
    expected_weight_shape: tuple[int, int]


@dataclass(frozen=True)
class CapturedGradientStep:
    """One completed forward/backward step and the corresponding true q inputs."""

    loss: float
    q_inputs: tuple[torch.Tensor, torch.Tensor]
    q_gradients: tuple[torch.Tensor, torch.Tensor]
    q_parameter_names: tuple[str, str]
    q_expected_weight_shape: tuple[int, int]
    gpu_peak_memory_bytes: int


def _config_int(model: nn.Module, name: str) -> int:
    config = getattr(model, "config", None)
    value = getattr(config, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GradientCaptureError(f"model.config.{name} must be one positive integer, got {value!r}.")
    return value


def _expected_q_weight_shape(model: nn.Module) -> tuple[int, int]:
    hidden_size = _config_int(model, "hidden_size")
    num_attention_heads = _config_int(model, "num_attention_heads")
    config = getattr(model, "config")
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        if hidden_size % num_attention_heads != 0:
            raise GradientCaptureError(
                "Cannot derive Qwen3 head_dim: hidden_size is not divisible by num_attention_heads."
            )
        head_dim = hidden_size // num_attention_heads
    if isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim <= 0:
        raise GradientCaptureError(f"model.config.head_dim must be positive when provided, got {head_dim!r}.")
    return (num_attention_heads * head_dim, hidden_size)


def resolve_first_two_q_projections(model: nn.Module) -> QProjectionPair:
    """Resolve Qwen3 q_proj modules structurally and assert their canonical names."""
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if not isinstance(backbone, nn.Module) or not isinstance(layers, nn.ModuleList) or len(layers) < 2:
        raise GradientCaptureError(
            "Expected Qwen3 sequence classifier structure model.model.layers with at least two layers."
        )
    parameter_lookup = dict(model.named_parameters())
    expected_shape = _expected_q_weight_shape(model)
    modules: list[nn.Module] = []
    parameter_names: list[str] = []
    for layer_index in (0, 1):
        layer = layers[layer_index]
        self_attn = getattr(layer, "self_attn", None)
        q_proj = getattr(self_attn, "q_proj", None)
        expected_name = f"model.layers.{layer_index}.self_attn.q_proj.weight"
        if not isinstance(q_proj, nn.Module) or not isinstance(getattr(q_proj, "weight", None), nn.Parameter):
            raise GradientCaptureError(
                f"Expected Qwen3 module path model.model.layers[{layer_index}].self_attn.q_proj."
            )
        named_weight = parameter_lookup.get(expected_name)
        if named_weight is None or named_weight is not q_proj.weight:
            raise GradientCaptureError(
                f"Canonical named parameter {expected_name!r} does not resolve to the structural q_proj weight."
            )
        actual_shape = tuple(int(value) for value in q_proj.weight.shape)
        if actual_shape != expected_shape:
            raise GradientCaptureError(
                f"{expected_name} has shape {actual_shape}; expected {expected_shape} derived from model.config."
            )
        modules.append(q_proj)
        parameter_names.append(expected_name)
    return QProjectionPair(
        modules=(modules[0], modules[1]),
        parameter_names=(parameter_names[0], parameter_names[1]),
        expected_weight_shape=expected_shape,
    )


def build_canonical_gradient_manifest(model: nn.Module) -> dict[str, Any]:
    """Return the current ``named_parameters()`` order and gradient availability."""
    entries: list[dict[str, Any]] = []
    gradient_tensor_count = 0
    gradient_numel = 0
    for canonical_index, (name, parameter) in enumerate(model.named_parameters()):
        grad_present = parameter.grad is not None
        if grad_present:
            gradient_tensor_count += 1
            gradient_numel += int(parameter.grad.numel())
        entries.append(
            {
                "name": name,
                "canonical_index": canonical_index,
                "shape": [int(value) for value in parameter.shape],
                "dtype": str(parameter.dtype),
                "requires_grad": bool(parameter.requires_grad),
                "grad_present": grad_present,
                "numel": int(parameter.numel()),
            }
        )
    return {
        "schema_version": 1,
        "canonical_order": "model.named_parameters() current deterministic traversal order",
        "entries": entries,
        "parameter_tensor_count": len(entries),
        "parameter_numel": sum(entry["numel"] for entry in entries),
        "gradient_tensor_count": gradient_tensor_count,
        "gradient_numel": gradient_numel,
    }


def _validate_batch(input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> None:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise GradientCaptureError(f"input_ids must have batch size 1, got shape {tuple(input_ids.shape)}.")
    if attention_mask.shape != input_ids.shape or not bool(torch.all(attention_mask == 1)):
        raise GradientCaptureError("attention_mask must match input_ids and contain only ones; padding is forbidden.")
    if labels.shape != (1,):
        raise GradientCaptureError(f"labels must have shape (1,), got {tuple(labels.shape)}.")


def capture_single_example_gradients(
    model: nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> CapturedGradientStep:
    """Run one BF16 batch-size-one forward/backward and capture q0/q1 inputs."""
    _validate_batch(input_ids, attention_mask, labels)
    if input_ids.device.type != "cuda":
        raise GradientCaptureError("Single-example Qwen3 diagnostics require CUDA inputs.")
    bf16_violations = [
        name
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point() and parameter.dtype != torch.bfloat16
    ]
    if bf16_violations:
        raise GradientCaptureError(
            f"BF16 forward/backward required, but these parameters are not BF16: {bf16_violations[:8]}"
        )
    q_pair = resolve_first_two_q_projections(model)
    captured_inputs: list[torch.Tensor | None] = [None, None]

    def make_hook(position: int):
        def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise GradientCaptureError(f"q{position} forward pre-hook received no tensor input.")
            if captured_inputs[position] is None:
                captured_inputs[position] = inputs[0].detach().clone()

        return hook

    handles = [module.register_forward_pre_hook(make_hook(position)) for position, module in enumerate(q_pair.modules)]
    model.zero_grad(set_to_none=True)
    try:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
        loss_tensor = getattr(outputs, "loss", None)
        if not isinstance(loss_tensor, torch.Tensor) or loss_tensor.ndim != 0:
            raise GradientCaptureError("Sequence-classification forward did not return one scalar loss.")
        loss_tensor.backward()
    finally:
        for handle in handles:
            handle.remove()
    if captured_inputs[0] is None or captured_inputs[1] is None:
        raise GradientCaptureError("Failed to capture the true inputs to q0 and q1.")
    q_inputs = (captured_inputs[0], captured_inputs[1])
    q_gradients: list[torch.Tensor] = []
    hidden_size = q_pair.expected_weight_shape[1]
    for position, (module, parameter_name, q_input) in enumerate(
        zip(q_pair.modules, q_pair.parameter_names, q_inputs)
    ):
        gradient = getattr(module, "weight").grad
        if gradient is None:
            raise GradientCaptureError(f"{parameter_name}.grad is absent after backward.")
        if tuple(int(value) for value in gradient.shape) != q_pair.expected_weight_shape:
            raise GradientCaptureError(
                f"{parameter_name}.grad shape {tuple(gradient.shape)} differs from config-derived "
                f"{q_pair.expected_weight_shape}."
            )
        if gradient.dtype != torch.bfloat16 or q_input.dtype != torch.bfloat16:
            raise GradientCaptureError(
                f"q{position} must use BF16 input and gradient, got input={q_input.dtype}, grad={gradient.dtype}."
            )
        if q_input.ndim != 3 or q_input.shape[0] != 1 or q_input.shape[-1] != hidden_size:
            raise GradientCaptureError(
                f"q{position} input has shape {tuple(q_input.shape)}; expected [1, sequence, {hidden_size}]."
            )
        q_gradients.append(gradient.detach().clone())
    torch.cuda.synchronize(input_ids.device)
    peak_memory = int(torch.cuda.max_memory_allocated(input_ids.device))
    return CapturedGradientStep(
        loss=float(loss_tensor.detach().float().cpu().item()),
        q_inputs=q_inputs,
        q_gradients=(q_gradients[0], q_gradients[1]),
        q_parameter_names=q_pair.parameter_names,
        q_expected_weight_shape=q_pair.expected_weight_shape,
        gpu_peak_memory_bytes=peak_memory,
    )
