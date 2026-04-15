"""
Layer-wise Recoverability Bottleneck (LRB) defense.

LRB is intended as a generic "make shared updates less recoverable" transform,
instead of a patch for a single attack. The v2 implementation keeps the original
layer-wise clipping/compression/noise template, but upgrades two pieces:

1. Sensitivity is estimated by a hybrid of structural priors and cheap per-batch
   calibration on the current gradients.
2. The public subspace is no longer tied to the original coordinate axes; by
   default we use deterministic signed pooling to approximate a shared random
   low-resolution basis and then inject noise into the discarded residual space.
"""
from __future__ import annotations

import math
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


def _rademacher(shape, *, device: torch.device, generator: torch.Generator) -> torch.Tensor:
    out = torch.empty(shape, device=device, dtype=torch.float32)
    out.bernoulli_(0.5, generator=generator)
    out.mul_(2.0).sub_(1.0)
    return out


def _pool_interpolate_flat(flat: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    n = flat.shape[0]
    k = max(1, int(round(n * keep_ratio)))
    pooled = F.adaptive_avg_pool1d(flat.view(1, 1, n), k)
    projected = F.interpolate(pooled, size=n, mode="linear", align_corners=False)
    return projected.view_as(flat)


def _pool_interpolate_matrix(matrix: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    m, n = matrix.shape
    km = max(1, int(round(m * keep_ratio)))
    kn = max(1, int(round(n * keep_ratio)))
    pooled = F.adaptive_avg_pool2d(matrix.view(1, 1, m, n), (km, kn))
    projected = F.interpolate(pooled, size=(m, n), mode="bilinear", align_corners=False)
    return projected.view_as(matrix)


def _project_low_resolution(
    tensor: torch.Tensor,
    keep_ratio: float,
    *,
    seed: int = 0,
    mode: str = "signed_pool",
) -> torch.Tensor:
    if tensor.dim() == 0 or keep_ratio >= 0.999:
        return tensor.clone()

    keep_ratio = float(max(1e-4, min(1.0, keep_ratio)))
    x = tensor.float()

    if mode not in {"pool", "signed_pool"}:
        raise ValueError(f"Unsupported LRB projection mode: {mode}")

    if tensor.dim() == 1:
        if mode == "pool":
            projected = _pool_interpolate_flat(x.view(-1), keep_ratio)
        else:
            gen = _make_generator(seed, tensor.device)
            signs = _rademacher(x.shape, device=tensor.device, generator=gen)
            projected = _pool_interpolate_flat(x * signs, keep_ratio) * signs
        return projected.view_as(x).to(dtype=tensor.dtype)

    if tensor.dim() == 2:
        if mode == "pool":
            projected = _pool_interpolate_matrix(x, keep_ratio)
        else:
            gen = _make_generator(seed, tensor.device)
            row_signs = _rademacher((x.shape[0], 1), device=tensor.device, generator=gen)
            col_signs = _rademacher((1, x.shape[1]), device=tensor.device, generator=gen)
            projected = _pool_interpolate_matrix(x * row_signs * col_signs, keep_ratio)
            projected = projected * row_signs * col_signs
        return projected.view_as(x).to(dtype=tensor.dtype)

    flat = x.view(-1)
    if mode == "pool":
        projected = _pool_interpolate_flat(flat, keep_ratio)
    else:
        gen = _make_generator(seed, tensor.device)
        signs = _rademacher(flat.shape, device=tensor.device, generator=gen)
        projected = _pool_interpolate_flat(flat * signs, keep_ratio) * signs
    return projected.view_as(x).to(dtype=tensor.dtype)


def _orthogonal_residual_noise(
    tensor: torch.Tensor,
    keep_ratio: float,
    noise_scale: float,
    seed: int,
    reference_norm: float,
    projection_mode: str,
) -> torch.Tensor:
    if noise_scale <= 0:
        return torch.zeros_like(tensor)

    noise = torch.randn(
        tensor.shape,
        device=tensor.device,
        dtype=torch.float32,
        generator=_make_generator(seed, tensor.device),
    )
    noise_proj = _project_low_resolution(
        noise,
        keep_ratio,
        seed=seed,
        mode=projection_mode,
    )
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


def _normalize_metric(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    v_min = min(values)
    v_max = max(values)
    if abs(v_max - v_min) <= 1e-12:
        return [0.5 for _ in values]
    return [(value - v_min) / (v_max - v_min) for value in values]


def _sample_for_calibration(tensor: torch.Tensor, max_elements: int, seed: int) -> torch.Tensor:
    flat = tensor.detach().float().reshape(-1)
    if max_elements <= 0 or flat.numel() <= max_elements:
        return flat
    step = int(math.ceil(flat.numel() / max_elements))
    offset = seed % step
    sample = flat[offset::step]
    return sample[:max_elements]


def _tensor_spikiness(tensor: torch.Tensor) -> float:
    flat = tensor.detach().float().reshape(-1)
    if flat.numel() == 0:
        return 0.0
    rms = flat.norm().item() / math.sqrt(float(flat.numel()))
    return float(flat.abs().max().item() / (rms + 1e-12))


def _layer_projection_seed(base_seed: int, idx: int) -> int:
    return int(base_seed + 1009 * (idx + 1))


def _estimate_layer_sensitivities(
    grads: Sequence[torch.Tensor | None],
    *,
    layer_names: List[str] | None,
    sensitive_n_layers: int,
    empirical_weight: float,
    calibration_keep_ratio: float,
    calibration_samples: int,
    base_seed: int,
    projection_mode: str,
) -> list[float]:
    priors = []
    norm_values = []
    residual_values = []
    spikiness_values = []
    valid_indices = []

    for idx, grad in enumerate(grads):
        name = layer_names[idx] if layer_names is not None and idx < len(layer_names) else f"layer_{idx}"
        prior = _layer_sensitivity(name, sensitive_n_layers)
        priors.append(prior)

        if grad is None:
            continue

        valid_indices.append(idx)
        norm_values.append(math.log1p(float(grad.detach().float().norm().item())))
        sample = _sample_for_calibration(grad, calibration_samples, seed=_layer_projection_seed(base_seed, idx))
        sample_proj = _project_low_resolution(
            sample,
            calibration_keep_ratio,
            seed=_layer_projection_seed(base_seed, idx),
            mode=projection_mode,
        )
        sample_norm = float(sample.norm().item())
        residual_ratio = float((sample - sample_proj).norm().item() / (sample_norm + 1e-12))
        residual_values.append(residual_ratio)
        spikiness_values.append(math.log1p(_tensor_spikiness(grad)))

    norm_scores = _normalize_metric(norm_values)
    residual_scores = _normalize_metric(residual_values)
    spikiness_scores = _normalize_metric(spikiness_values)

    sensitivities = [0.0 for _ in grads]
    for pos, idx in enumerate(valid_indices):
        empirical = (
            0.45 * norm_scores[pos]
            + 0.40 * residual_scores[pos]
            + 0.15 * spikiness_scores[pos]
        )
        sensitivity = (1.0 - empirical_weight) * priors[idx] + empirical_weight * empirical
        sensitivities[idx] = float(min(1.0, max(0.0, sensitivity)))

    return sensitivities


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
    empirical_weight = float(getattr(args, "defense_lrb_empirical_weight", 0.6))
    calibration_samples = int(getattr(args, "defense_lrb_calibration_samples", 4096))
    projection_mode = str(getattr(args, "defense_lrb_projection", "signed_pool"))
    base_seed = int(getattr(args, "rng_seed", 0))

    median_norm = _median_layer_norm(grads)
    calibration_keep_ratio = 0.5 * (sensitive_keep + other_keep)
    sensitivities = _estimate_layer_sensitivities(
        grads,
        layer_names=layer_names,
        sensitive_n_layers=sensitive_n_layers,
        empirical_weight=empirical_weight,
        calibration_keep_ratio=calibration_keep_ratio,
        calibration_samples=calibration_samples,
        base_seed=base_seed,
        projection_mode=projection_mode,
    )
    out = []

    for idx, grad in enumerate(grads):
        if grad is None:
            out.append(None)
            continue

        sensitivity = sensitivities[idx]
        keep_ratio = _mix_by_sensitivity(other_keep, sensitive_keep, sensitivity)
        clip_scale = _mix_by_sensitivity(other_clip, sensitive_clip, sensitivity)
        noise_scale = _mix_by_sensitivity(other_noise, sensitive_noise, sensitivity)
        layer_seed = _layer_projection_seed(base_seed, idx)

        max_norm = median_norm * clip_scale
        clipped = _clip_tensor(grad, max_norm)
        projected = _project_low_resolution(
            clipped,
            keep_ratio,
            seed=layer_seed,
            mode=projection_mode,
        )
        noise = _orthogonal_residual_noise(
            clipped,
            keep_ratio=keep_ratio,
            noise_scale=noise_scale,
            seed=layer_seed,
            reference_norm=float(clipped.float().norm().item()) or max_norm,
            projection_mode=projection_mode,
        )
        out.append(projected + noise)

    return tuple(out)
