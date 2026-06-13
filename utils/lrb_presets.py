from __future__ import annotations

from typing import Any


LRB_DEFENSE_NAMES = {"lrb", "lrbprojonly", "signed_bottleneck"}

LRB_PRESET_CHOICES = [
    "custom",
    "lrbprojonly",
    "identity_lrb",
    "clip_only",
    "proj_only",
    "proj_clip",
    "full_lrb",
    "pool_full",
    "rule_only",
    "empirical_only",
    "uniform_all_sensitive",
    "proj_rule_only",
    "proj_empirical_only",
    "proj_uniform",
    "signed_bottleneck",
    "proj_no_empirical",
]

NO_CLIP = 1_000_000.0


def _main_keep(args: Any) -> float:
    return float(getattr(args, "defense_lrb_keep_ratio_sensitive", 0.5))


def _set_lrb_args(
    args: Any,
    *,
    keep_sensitive: float,
    keep_other: float,
    clip_sensitive: float,
    clip_other: float,
    noise_sensitive: float,
    noise_other: float,
    empirical_weight: float,
    projection: str,
) -> Any:
    args.defense_lrb_keep_ratio_sensitive = float(keep_sensitive)
    args.defense_lrb_keep_ratio_other = float(keep_other)
    args.defense_lrb_clip_scale_sensitive = float(clip_sensitive)
    args.defense_lrb_clip_scale_other = float(clip_other)
    args.defense_lrb_noise_sensitive = float(noise_sensitive)
    args.defense_lrb_noise_other = float(noise_other)
    args.defense_lrb_empirical_weight = float(empirical_weight)
    args.defense_lrb_projection = projection
    return args


def apply_lrb_preset(args: Any) -> Any:
    """Normalize a named LRB preset into the existing low-level LRB arguments."""

    preset = getattr(args, "defense_lrb_preset", "custom")
    defense = getattr(args, "defense", "none")
    if preset not in LRB_PRESET_CHOICES:
        raise ValueError(f"Unsupported LRB preset: {preset}")

    if defense == "lrbprojonly":
        if preset not in {"custom", "lrbprojonly", "proj_only"}:
            raise ValueError(
                "--defense lrbprojonly cannot be combined with "
                f"--defense_lrb_preset {preset!r}; use --defense lrb for other LRB presets."
            )
        preset = "lrbprojonly"
        args.defense_lrb_preset = preset
    elif defense == "signed_bottleneck":
        if preset not in {"custom", "signed_bottleneck"}:
            raise ValueError(
                "--defense signed_bottleneck cannot be combined with "
                f"--defense_lrb_preset {preset!r}; use --defense lrb for other LRB presets."
            )
        preset = "signed_bottleneck"
        args.defense_lrb_preset = preset

    if defense not in LRB_DEFENSE_NAMES or preset == "custom":
        return args

    k = _main_keep(args)

    if preset == "identity_lrb":
        return _set_lrb_args(
            args,
            keep_sensitive=1.0,
            keep_other=1.0,
            clip_sensitive=NO_CLIP,
            clip_other=NO_CLIP,
            noise_sensitive=0.0,
            noise_other=0.0,
            empirical_weight=0.0,
            projection="signed_pool",
        )

    if preset == "clip_only":
        return _set_lrb_args(
            args,
            keep_sensitive=1.0,
            keep_other=1.0,
            clip_sensitive=0.5,
            clip_other=1.0,
            noise_sensitive=0.0,
            noise_other=0.0,
            empirical_weight=0.6,
            projection="signed_pool",
        )

    if preset in {"lrbprojonly", "proj_only", "proj_rule_only", "proj_empirical_only", "proj_no_empirical"}:
        empirical_weight = 0.6
        if preset in {"proj_rule_only", "proj_no_empirical"}:
            empirical_weight = 0.0
        elif preset == "proj_empirical_only":
            empirical_weight = 1.0
        return _set_lrb_args(
            args,
            keep_sensitive=k,
            keep_other=0.75,
            clip_sensitive=NO_CLIP,
            clip_other=NO_CLIP,
            noise_sensitive=0.0,
            noise_other=0.0,
            empirical_weight=empirical_weight,
            projection="signed_pool",
        )

    if preset in {"proj_uniform", "signed_bottleneck"}:
        return _set_lrb_args(
            args,
            keep_sensitive=k,
            keep_other=k,
            clip_sensitive=NO_CLIP,
            clip_other=NO_CLIP,
            noise_sensitive=0.0,
            noise_other=0.0,
            empirical_weight=0.0,
            projection="signed_pool",
        )

    if preset == "proj_clip":
        return _set_lrb_args(
            args,
            keep_sensitive=k,
            keep_other=0.75,
            clip_sensitive=0.5,
            clip_other=1.0,
            noise_sensitive=0.0,
            noise_other=0.0,
            empirical_weight=0.6,
            projection="signed_pool",
        )

    if preset in {"full_lrb", "pool_full", "rule_only", "empirical_only"}:
        empirical_weight = 0.6
        if preset == "rule_only":
            empirical_weight = 0.0
        elif preset == "empirical_only":
            empirical_weight = 1.0
        projection = "pool" if preset == "pool_full" else "signed_pool"
        return _set_lrb_args(
            args,
            keep_sensitive=k,
            keep_other=0.75,
            clip_sensitive=0.5,
            clip_other=1.0,
            noise_sensitive=0.03,
            noise_other=0.005,
            empirical_weight=empirical_weight,
            projection=projection,
        )

    if preset == "uniform_all_sensitive":
        return _set_lrb_args(
            args,
            keep_sensitive=k,
            keep_other=k,
            clip_sensitive=0.5,
            clip_other=0.5,
            noise_sensitive=0.03,
            noise_other=0.03,
            empirical_weight=0.0,
            projection="signed_pool",
        )

    raise ValueError(f"Unhandled LRB preset: {preset}")


def lrb_preset_param_value(args: Any) -> str | None:
    preset = getattr(args, "defense_lrb_preset", "custom")
    if getattr(args, "defense", "none") not in LRB_DEFENSE_NAMES or preset == "custom":
        return None
    k = getattr(args, "defense_lrb_keep_ratio_sensitive", None)
    if k is None:
        return preset
    return f"{preset}@k={float(k):.6g}"
