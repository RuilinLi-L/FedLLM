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

from collections import OrderedDict
import math
import re
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F


_CACHE_LIMIT = 512
_NO_CLIP_SENTINEL = 1_000_000.0
_LRB_UPDATE_SEED_STRIDE = 1_000_003
_BIN_BOUNDS_CACHE: OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
_SIGN_CACHE: OrderedDict[tuple, torch.Tensor] = OrderedDict()


def _cache_get(cache: OrderedDict, key: tuple):
    value = cache.get(key)
    if value is None:
        return None
    cache.move_to_end(key)
    return value


def _cache_put(cache: OrderedDict, key: tuple, value):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _CACHE_LIMIT:
        cache.popitem(last=False)


def _device_cache_key(device: torch.device) -> tuple[str, int | None]:
    normalized = torch.device(device)
    return normalized.type, normalized.index


def _make_generator(seed: int, device: torch.device) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def _adaptive_bin_bounds(input_size: int, output_size: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if input_size <= 0:
        raise ValueError("input_size must be positive")
    if output_size <= 0:
        raise ValueError("output_size must be positive")

    cache_key = (_device_cache_key(device), int(input_size), int(output_size))
    cached = _cache_get(_BIN_BOUNDS_CACHE, cache_key)
    if cached is not None:
        return cached

    positions = torch.arange(output_size, device=device, dtype=torch.long)
    starts = (positions * input_size) // output_size
    ends = ((positions + 1) * input_size + output_size - 1) // output_size
    starts = starts.clamp(min=0, max=input_size - 1)
    ends = torch.maximum(ends, starts + 1)
    ends = ends.clamp(max=input_size)
    _cache_put(_BIN_BOUNDS_CACHE, cache_key, (starts, ends))
    return starts, ends


def _adaptive_avg_pool_along_dim_manual(
    tensor: torch.Tensor,
    output_size: int,
    dim: int,
    *,
    max_chunk_elements: int = 1_048_576,
) -> torch.Tensor:
    dim = dim % tensor.dim()
    input_size = tensor.shape[dim]
    if output_size == input_size:
        return tensor.clone()

    starts, ends = _adaptive_bin_bounds(input_size, output_size, device=tensor.device)
    moved = tensor.movedim(dim, 0)
    trailing_shape = tuple(moved.shape[1:])
    trailing_numel = max(1, math.prod(trailing_shape))
    chunk_size = max(1, min(output_size, max_chunk_elements // trailing_numel))

    prefix = moved.new_zeros((input_size + 1,) + trailing_shape)
    prefix[1:] = moved.cumsum(dim=0)
    out = moved.new_empty((output_size,) + trailing_shape)

    for offset in range(0, output_size, chunk_size):
        next_offset = min(offset + chunk_size, output_size)
        chunk_starts = starts[offset:next_offset]
        chunk_ends = ends[offset:next_offset]
        sums = prefix[chunk_ends] - prefix[chunk_starts]
        count_shape = (next_offset - offset,) + (1,) * len(trailing_shape)
        counts = (chunk_ends - chunk_starts).to(dtype=moved.dtype).reshape(count_shape)
        out[offset:next_offset] = sums / counts

    return out.movedim(0, dim)


def _adaptive_avg_pool1d_manual(flat: torch.Tensor, out_size: int) -> torch.Tensor:
    return _adaptive_avg_pool_along_dim_manual(flat.reshape(-1), out_size, dim=0)


def _adaptive_avg_pool2d_manual(matrix: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    m, n = matrix.shape
    km, kn = output_size
    if km == m and kn == n:
        return matrix.clone()
    if km == m:
        return _adaptive_avg_pool_along_dim_manual(matrix, kn, dim=1)
    if kn == n:
        return _adaptive_avg_pool_along_dim_manual(matrix, km, dim=0)

    row_starts, row_ends = _adaptive_bin_bounds(m, km, device=matrix.device)
    row_prefix = matrix.new_zeros(m + 1, n)
    row_prefix[1:] = matrix.cumsum(dim=0)
    out = matrix.new_empty(km, kn)

    row_chunk_size = max(1, min(km, 1_048_576 // max(1, n)))
    for offset in range(0, km, row_chunk_size):
        next_offset = min(offset + row_chunk_size, km)
        chunk_starts = row_starts[offset:next_offset]
        chunk_ends = row_ends[offset:next_offset]
        row_sums = row_prefix[chunk_ends] - row_prefix[chunk_starts]
        row_counts = (chunk_ends - chunk_starts).to(dtype=matrix.dtype).reshape(-1, 1)
        row_means = row_sums / row_counts
        out[offset:next_offset] = _adaptive_avg_pool_along_dim_manual(row_means, kn, dim=1)

    return out


def _extract_layer_index(name: str) -> int | None:
    patterns = (
        (r"\.h\.(\d+)\.", 0),
        (r"\.layer\.(\d+)\.", 0),
        (r"\.layers\.(\d+)\.", 0),
        (r"\.block\.(\d+)\.", 0),
        (r"\.layer_(\d+)\.", 0),
        (r"(?:^|\.)encoder(\d+)(?:\.|$)", -1),
    )
    for pattern, offset in patterns:
        match = re.search(pattern, name)
        if match:
            return max(0, int(match.group(1)) + offset)
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
    return out.mul_(2.0).sub_(1.0)


def _cached_rademacher(shape, *, device: torch.device, seed: int, prior_shapes=()) -> torch.Tensor:
    normalized_shape = tuple(int(dim) for dim in shape)
    normalized_prior = tuple(tuple(int(dim) for dim in prior_shape) for prior_shape in prior_shapes)
    cache_key = (_device_cache_key(device), normalized_shape, int(seed), normalized_prior)
    cached = _cache_get(_SIGN_CACHE, cache_key)
    if cached is not None:
        return cached
    gen = _make_generator(seed, device)
    for prior_shape in normalized_prior:
        _rademacher(prior_shape, device=device, generator=gen)
    signs = _rademacher(normalized_shape, device=device, generator=gen)
    _cache_put(_SIGN_CACHE, cache_key, signs)
    return signs


def _pool_interpolate_flat(flat: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    n = flat.shape[0]
    k = max(1, int(round(n * keep_ratio)))
    pooled = _adaptive_avg_pool1d_manual(flat.reshape(-1), k).reshape(1, 1, k)
    projected = F.interpolate(pooled, size=n, mode="linear", align_corners=False)
    return projected.view_as(flat)


def _pool_interpolate_matrix(matrix: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    m, n = matrix.shape
    km = max(1, int(round(m * keep_ratio)))
    kn = max(1, int(round(n * keep_ratio)))
    pooled = _adaptive_avg_pool2d_manual(matrix.reshape(m, n), (km, kn)).reshape(1, 1, km, kn)
    projected = F.interpolate(pooled, size=(m, n), mode="bilinear", align_corners=False)
    return projected.view_as(matrix)


def _pool_nearest_flat(flat: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    n = flat.shape[0]
    k = max(1, int(round(n * keep_ratio)))
    pooled = _adaptive_avg_pool1d_manual(flat.reshape(-1), k).reshape(1, 1, k)
    return F.interpolate(pooled, size=n, mode="nearest").view_as(flat)


def _pool_nearest_matrix(matrix: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    m, n = matrix.shape
    km = max(1, int(round(m * keep_ratio)))
    kn = max(1, int(round(n * keep_ratio)))
    pooled = _adaptive_avg_pool2d_manual(matrix.reshape(m, n), (km, kn)).reshape(1, 1, km, kn)
    return F.interpolate(pooled, size=(m, n), mode="nearest").view_as(matrix)


def _stride_interpolate_flat(flat: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    n = flat.shape[0]
    k = max(1, int(round(n * keep_ratio)))
    indices = torch.linspace(0, n - 1, steps=k, device=flat.device).round().long()
    sampled = flat.index_select(0, indices).reshape(1, 1, k)
    return F.interpolate(sampled, size=n, mode="linear", align_corners=False).view_as(flat)


def _stride_interpolate_matrix(matrix: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    m, n = matrix.shape
    km = max(1, int(round(m * keep_ratio)))
    kn = max(1, int(round(n * keep_ratio)))
    row_indices = torch.linspace(0, m - 1, steps=km, device=matrix.device).round().long()
    col_indices = torch.linspace(0, n - 1, steps=kn, device=matrix.device).round().long()
    sampled = matrix.index_select(0, row_indices).index_select(1, col_indices).reshape(1, 1, km, kn)
    return F.interpolate(sampled, size=(m, n), mode="bilinear", align_corners=False).view_as(matrix)


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

    if mode not in {"pool", "signed_pool", "signed_pool_nearest", "signed_stride"}:
        raise ValueError(f"Unsupported LRB projection mode: {mode}")

    if tensor.dim() == 1:
        if mode == "pool":
            projected = _pool_interpolate_flat(x.reshape(-1), keep_ratio)
        elif mode == "signed_pool":
            signs = _cached_rademacher(x.shape, device=tensor.device, seed=seed)
            projected = _pool_interpolate_flat(x * signs, keep_ratio) * signs
        elif mode == "signed_pool_nearest":
            signs = _cached_rademacher(x.shape, device=tensor.device, seed=seed)
            projected = _pool_nearest_flat(x * signs, keep_ratio) * signs
        else:
            signs = _cached_rademacher(x.shape, device=tensor.device, seed=seed)
            projected = _stride_interpolate_flat(x * signs, keep_ratio) * signs
        return projected.view_as(x).to(dtype=tensor.dtype)

    if tensor.dim() == 2:
        if mode == "pool":
            projected = _pool_interpolate_matrix(x, keep_ratio)
        elif mode == "signed_pool":
            row_shape = (x.shape[0], 1)
            col_shape = (1, x.shape[1])
            row_signs = _cached_rademacher(row_shape, device=tensor.device, seed=seed)
            col_signs = _cached_rademacher(col_shape, device=tensor.device, seed=seed, prior_shapes=(row_shape,))
            projected = _pool_interpolate_matrix(x * row_signs * col_signs, keep_ratio)
            projected = projected * row_signs * col_signs
        else:
            row_shape = (x.shape[0], 1)
            col_shape = (1, x.shape[1])
            row_signs = _cached_rademacher(row_shape, device=tensor.device, seed=seed)
            col_signs = _cached_rademacher(col_shape, device=tensor.device, seed=seed, prior_shapes=(row_shape,))
            signed = x * row_signs * col_signs
            if mode == "signed_pool_nearest":
                projected = _pool_nearest_matrix(signed, keep_ratio)
            else:
                projected = _stride_interpolate_matrix(signed, keep_ratio)
            projected = projected * row_signs * col_signs
        return projected.view_as(x).to(dtype=tensor.dtype)

    flat = x.reshape(-1)
    if mode == "pool":
        projected = _pool_interpolate_flat(flat, keep_ratio)
    elif mode == "signed_pool":
        signs = _cached_rademacher(flat.shape, device=tensor.device, seed=seed)
        projected = _pool_interpolate_flat(flat * signs, keep_ratio) * signs
    elif mode == "signed_pool_nearest":
        signs = _cached_rademacher(flat.shape, device=tensor.device, seed=seed)
        projected = _pool_nearest_flat(flat * signs, keep_ratio) * signs
    else:
        signs = _cached_rademacher(flat.shape, device=tensor.device, seed=seed)
        projected = _stride_interpolate_flat(flat * signs, keep_ratio) * signs
    return projected.view_as(x).to(dtype=tensor.dtype)


def _effective_projection_base_seed(args) -> tuple[int, int | None]:
    configured_seed = getattr(args, "defense_lrb_seed", None)
    base_seed = int(getattr(args, "rng_seed", 0) if configured_seed is None else configured_seed)
    mode = str(getattr(args, "defense_lrb_seed_mode", "static"))
    if mode == "static":
        return base_seed, None
    if mode != "per_update":
        raise ValueError(f"Unsupported LRB seed mode: {mode}")

    step = getattr(args, "defense_rng_step", None)
    if step is None:
        step = int(getattr(args, "_lrb_defense_call_index", 0))
        setattr(args, "_lrb_defense_call_index", step + 1)
    return base_seed + _LRB_UPDATE_SEED_STRIDE * int(step), int(step)


def lrb_seed_summary_fields(args):
    if str(getattr(args, "defense", "none")) not in {"lrb", "lrbprojonly", "signed_bottleneck"}:
        return [
            ("defense_lrb_seed_mode", "n/a"),
            ("defense_lrb_seed_source", "n/a"),
            ("defense_lrb_seed", "n/a"),
        ]
    configured_seed = getattr(args, "defense_lrb_seed", None)
    resolved_seed = int(getattr(args, "rng_seed", 0) if configured_seed is None else configured_seed)
    return [
        ("defense_lrb_seed_mode", getattr(args, "defense_lrb_seed_mode", "static")),
        ("defense_lrb_seed_source", "rng_seed" if configured_seed is None else "explicit"),
        ("defense_lrb_seed", resolved_seed),
    ]


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
    base_seed, seed_step = _effective_projection_base_seed(args)

    clipping_disabled = sensitive_clip >= _NO_CLIP_SENTINEL and other_clip >= _NO_CLIP_SENTINEL
    noise_disabled = sensitive_noise <= 0.0 and other_noise <= 0.0
    median_norm = None if clipping_disabled and noise_disabled else _median_layer_norm(grads)
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
    layer_info = []

    for idx, grad in enumerate(grads):
        if grad is None:
            out.append(None)
            layer_info.append(
                {
                    "idx": idx,
                    "name": layer_names[idx] if layer_names is not None and idx < len(layer_names) else f"layer_{idx}",
                    "active": False,
                }
            )
            continue

        sensitivity = sensitivities[idx]
        keep_ratio = _mix_by_sensitivity(other_keep, sensitive_keep, sensitivity)
        clip_scale = _mix_by_sensitivity(other_clip, sensitive_clip, sensitivity)
        noise_scale = _mix_by_sensitivity(other_noise, sensitive_noise, sensitivity)
        layer_seed = _layer_projection_seed(base_seed, idx)

        max_norm = None if median_norm is None else median_norm * clip_scale
        clipped = grad if clipping_disabled else _clip_tensor(grad, float(max_norm))
        projected = _project_low_resolution(
            clipped,
            keep_ratio,
            seed=layer_seed,
            mode=projection_mode,
        )
        if noise_disabled or noise_scale <= 0.0:
            defended = projected
        else:
            reference_norm = float(clipped.float().norm().item())
            if reference_norm <= 0.0 and max_norm is not None:
                reference_norm = float(max_norm)
            defended = projected + _orthogonal_residual_noise(
                clipped,
                keep_ratio=keep_ratio,
                noise_scale=noise_scale,
                seed=layer_seed,
                reference_norm=reference_norm,
                projection_mode=projection_mode,
            )
        out.append(defended)
        layer_info.append(
            {
                "idx": idx,
                "name": layer_names[idx] if layer_names is not None and idx < len(layer_names) else f"layer_{idx}",
                "active": True,
                "sensitivity": float(sensitivity),
                "keep_ratio": float(keep_ratio),
                "clip_scale": float(clip_scale),
                "noise_scale": float(noise_scale),
                "projection_seed": int(layer_seed),
                "projection_base_seed": int(base_seed),
                "projection_seed_mode": str(getattr(args, "defense_lrb_seed_mode", "static")),
                "projection_seed_step": seed_step,
                "projection_mode": projection_mode,
                "shape": tuple(int(dim) for dim in grad.shape),
            }
        )

    setattr(args, "lrb_defense_layer_info", layer_info)
    return tuple(out)
