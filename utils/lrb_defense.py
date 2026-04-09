"""
Layer-wise Recoverability Bottleneck (LRB) defense.

This defense is designed as a generic "make updates less recoverable" transform,
instead of a patch for a single attack algorithm. In the current DAGER framework
we implement a lightweight v1 that can be applied directly to the shared gradient
tuple before reconstruction:

1. Identify the most leakage-prone layers (embeddings + earliest transformer blocks)
2. Clip each layer to a sensitivity-aware norm budget
3. Project each tensor into a fixed low-resolution public subspace
4. Add noise in the discarded / residual directions

The projection is implemented with deterministic adaptive pooling + interpolation.
This acts like a shared low-frequency basis and is cheap enough to sweep inside the
existing attack-time defense framework.
"""
from __future__ import annotations

import re
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F


def _make_generator(seed: int, device: torch.device) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def _extract_layer_index(name: str) -> int | None:
    patterns = (
        r"\.h\.(\d+)\.",
        r"\.layer\.(\d+)\.",
        r"\.layers\.(\d+)\.",
        r"\.block\.(\d+)\.",
    )
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    return None


def _is_embedding_like(name: str) -> bool:
    return any(
        key in name
        for key in (
            "wte",
            "wpe",
            "embed_tokens",
            "embed_positions",
            "word_embeddings",
            "position_embeddings",
            "token_type_embeddings",
            "embeddings.",
        )
    )


def _is_attention_like(name: str) -> bool:
    return any(
        key in name
        for key in (
            "attn",
            "attention",
            "self_attn",
            "q_proj",
            "k_proj",
            "v_proj",
            "query",
            "key",
            "value",
            "c_attn",
        )
    )


def _layer_sensitivity(name: str, sensitive_n_layers: int) -> float:
    lower = name.lower()
    layer_idx = _extract_layer_index(lower)

    if _is_embedding_like(lower):
        return 1.0
    if layer_idx is not None and layer_idx < sensitive_n_layers and _is_attention_like(lower):
        return 1.0
    if layer_idx is not None and layer_idx < sensitive_n_layers:
        return 0.7
    if _is_attention_like(lower):
        return 0.45
    if any(key in lower for key in ("score.weight", "classifier.weight", "lm_head", "bias", "layernorm", "ln_")):
        return 0.15
    return 0.25


def _mix_by_sensitivity(other_value: float, sensitive_value: float, sensitivity: float) -> float:
    return other_value * (1.0 - sensitivity) + sensitive_value * sensitivity


def _clip_tensor(tensor: torch.Tensor, max_norm: float) -> torch.Tensor:
    if max_norm <= 0:
        return torch.zeros_like(tensor)
    norm = tensor.float().norm()
    norm_value = float(norm.item())
    if norm_value <= max_norm:
        return tensor
    scale = max_norm / (norm_value + 1e-12)
    return tensor * scale


def _project_low_resolution(tensor: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    if tensor.dim() == 0 or keep_ratio >= 0.999:
        return tensor.clone()

    keep_ratio = float(max(1e-4, min(1.0, keep_ratio)))
    x = tensor.float()

    if tensor.dim() == 1:
        n = x.shape[0]
        k = max(1, int(round(n * keep_ratio)))
        pooled = F.adaptive_avg_pool1d(x.view(1, 1, n), k)
        projected = F.interpolate(pooled, size=n, mode="linear", align_corners=False)
        return projected.view_as(x).to(dtype=tensor.dtype)

    if tensor.dim() == 2:
        m, n = x.shape
        km = max(1, int(round(m * keep_ratio)))
        kn = max(1, int(round(n * keep_ratio)))
        pooled = F.adaptive_avg_pool2d(x.view(1, 1, m, n), (km, kn))
        projected = F.interpolate(pooled, size=(m, n), mode="bilinear", align_corners=False)
        return projected.view_as(x).to(dtype=tensor.dtype)

    flat = x.view(-1)
    k = max(1, int(round(flat.numel() * keep_ratio)))
    pooled = F.adaptive_avg_pool1d(flat.view(1, 1, -1), k)
    projected = F.interpolate(pooled, size=flat.numel(), mode="linear", align_corners=False)
    return projected.view_as(x).to(dtype=tensor.dtype)


def _orthogonal_residual_noise(
    tensor: torch.Tensor,
    keep_ratio: float,
    noise_scale: float,
    seed: int,
    reference_norm: float,
) -> torch.Tensor:
    if noise_scale <= 0:
        return torch.zeros_like(tensor)

    noise = torch.randn(
        tensor.shape,
        device=tensor.device,
        dtype=torch.float32,
        generator=_make_generator(seed, tensor.device),
    )
    noise_proj = _project_low_resolution(noise, keep_ratio)
    residual_noise = noise - noise_proj.float()
    residual_norm = residual_noise.norm()
    if residual_norm <= 1e-12:
        return torch.zeros_like(tensor)

    target_norm = float(max(reference_norm, 1e-12)) * float(noise_scale)
    residual_noise = residual_noise * (target_norm / (residual_norm + 1e-12))
    return residual_noise.to(dtype=tensor.dtype)


def _median_layer_norm(grads: Sequence[torch.Tensor | None]) -> float:
    norms = []
    for g in grads:
        if g is None:
            continue
        norms.append(float(g.float().norm().item()))
    if not norms:
        return 1.0
    norms_tensor = torch.tensor(norms, dtype=torch.float32)
    return float(torch.median(norms_tensor).item())


def apply_lrb_defense(
    grads: Tuple[torch.Tensor, ...],
    args,
    layer_names: List[str] | None = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Apply Layer-wise Recoverability Bottleneck to a gradient tuple.

    Sensitive layers are compressed more aggressively and receive stronger
    residual-space noise. Less-sensitive layers keep a higher-resolution update.
    """
    if not grads:
        return grads

    sensitive_n_layers = int(getattr(args, "defense_lrb_sensitive_n_layers", 2))
    sensitive_keep = float(getattr(args, "defense_lrb_keep_ratio_sensitive", 0.2))
    other_keep = float(getattr(args, "defense_lrb_keep_ratio_other", 0.75))
    sensitive_clip = float(getattr(args, "defense_lrb_clip_scale_sensitive", 0.5))
    other_clip = float(getattr(args, "defense_lrb_clip_scale_other", 1.0))
    sensitive_noise = float(getattr(args, "defense_lrb_noise_sensitive", 0.03))
    other_noise = float(getattr(args, "defense_lrb_noise_other", 0.005))
    base_seed = int(getattr(args, "rng_seed", 0))

    median_norm = _median_layer_norm(grads)
    out = []

    for idx, grad in enumerate(grads):
        if grad is None:
            out.append(None)
            continue

        name = layer_names[idx] if layer_names is not None and idx < len(layer_names) else f"layer_{idx}"
        sensitivity = _layer_sensitivity(name, sensitive_n_layers)
        keep_ratio = _mix_by_sensitivity(other_keep, sensitive_keep, sensitivity)
        clip_scale = _mix_by_sensitivity(other_clip, sensitive_clip, sensitivity)
        noise_scale = _mix_by_sensitivity(other_noise, sensitive_noise, sensitivity)

        max_norm = median_norm * clip_scale
        clipped = _clip_tensor(grad, max_norm)
        projected = _project_low_resolution(clipped, keep_ratio)
        noise = _orthogonal_residual_noise(
            clipped,
            keep_ratio=keep_ratio,
            noise_scale=noise_scale,
            seed=base_seed + 1009 * (idx + 1),
            reference_norm=float(clipped.float().norm().item()) or max_norm,
        )
        out.append(projected + noise)

    return tuple(out)
