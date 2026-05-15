from __future__ import annotations

import torch
import torch.nn.functional as F


REP_BOTTLENECK_CHOICES = ("none", "mask", "dropout", "projection")


def rep_bottleneck_active(args) -> bool:
    return getattr(args, "defense_rep_bottleneck", "none") != "none"


def rep_bottleneck_summary_fields(args):
    return [
        ("rep_bottleneck_type", getattr(args, "defense_rep_bottleneck", "none")),
        ("rep_keep_ratio", getattr(args, "defense_rep_keep_ratio", None)),
        ("rep_dropout_p", getattr(args, "defense_rep_dropout_p", None)),
        ("rep_bottleneck_with_lrb", rep_bottleneck_active(args) and getattr(args, "defense", "none") in {"lrb", "lrbprojonly"}),
    ]


def validate_rep_bottleneck_args(args):
    mode = getattr(args, "defense_rep_bottleneck", "none")
    if mode not in REP_BOTTLENECK_CHOICES:
        raise ValueError(f"--defense_rep_bottleneck must be one of {REP_BOTTLENECK_CHOICES}; got {mode!r}.")
    keep_ratio = float(getattr(args, "defense_rep_keep_ratio", 0.5))
    dropout_p = float(getattr(args, "defense_rep_dropout_p", 0.1))
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError("--defense_rep_keep_ratio must be in (0, 1].")
    if not (0.0 <= dropout_p < 1.0):
        raise ValueError("--defense_rep_dropout_p must be in [0, 1).")
    return args


def _seed(args) -> int:
    base = int(getattr(args, "rng_seed", 0))
    step = int(getattr(args, "defense_rng_step", 0) or 0)
    return base + 1_000_003 * step + 9176


def _generator(args, device: torch.device) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(_seed(args))
    return gen


def _signed_pool_1d(x: torch.Tensor, keep_ratio: float, gen: torch.Generator) -> torch.Tensor:
    dim = x.shape[-1]
    if dim <= 1 or keep_ratio >= 1.0:
        return x
    target = max(1, int(round(dim * keep_ratio)))
    signs = torch.randint(0, 2, (dim,), device=x.device, generator=gen, dtype=torch.int64)
    signs = signs.to(dtype=x.dtype).mul_(2).sub_(1)
    signed = x * signs
    flat = signed.reshape(-1, dim).unsqueeze(1)
    pooled = F.adaptive_avg_pool1d(flat, target)
    restored = F.interpolate(pooled, size=dim, mode="linear", align_corners=False)
    return restored.squeeze(1).reshape_as(x) * signs


def apply_representation_bottleneck(representation: torch.Tensor, args) -> torch.Tensor:
    mode = getattr(args, "defense_rep_bottleneck", "none")
    if mode == "none":
        return representation

    keep_ratio = float(getattr(args, "defense_rep_keep_ratio", 0.5))
    gen = _generator(args, representation.device)

    if mode == "projection":
        return _signed_pool_1d(representation, keep_ratio, gen)

    if mode == "mask":
        if keep_ratio >= 1.0:
            return representation
        mask = torch.rand(
            representation.shape,
            device=representation.device,
            dtype=representation.dtype,
            generator=gen,
        ) < keep_ratio
        return representation * mask.to(dtype=representation.dtype) / max(keep_ratio, 1e-12)

    if mode == "dropout":
        dropout_p = float(getattr(args, "defense_rep_dropout_p", 0.1))
        if dropout_p <= 0.0:
            return representation
        keep = 1.0 - dropout_p
        mask = torch.rand(
            representation.shape,
            device=representation.device,
            dtype=representation.dtype,
            generator=gen,
        ) < keep
        return representation * mask.to(dtype=representation.dtype) / max(keep, 1e-12)

    raise ValueError(f"Unknown representation bottleneck: {mode!r}")
