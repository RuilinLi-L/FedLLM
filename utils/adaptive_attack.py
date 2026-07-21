from __future__ import annotations

import math
from statistics import mean
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from utils.functional import check_if_in_span
from utils.lrb_defense import _layer_projection_seed


ADAPTIVE_ATTACK_CHOICES = ("none", "auto", "defense_aware")
ADAPTIVE_LRB_KNOWLEDGE_CHOICES = ("oracle", "ratio_hidden", "signs_hidden", "method_only")
ADAPTIVE_LRB_HYPOTHESIS_REDUCERS = ("min", "mean")
ADAPTIVE_LRB_SIGN_SOURCES = ("legacy_cpu", "defense_device")
ADAPTIVE_LRB_MAX_HYPOTHESES = 1024
_ATTACK_SEED_STRIDE = 1_000_033
_UPDATE_SEED_STRIDE = 1_000_003
LRB_PROJECTION_DEFENSES = {"lrb", "lrbprojonly", "signed_bottleneck"}
RANKED_SPAN_DEFENSES = {"topk", "compression", "lrb", "lrbprojonly", "signed_bottleneck"}


def normalize_adaptive_attack_args(args):
    mode = getattr(args, "adaptive_attack", "none")
    if mode is None:
        mode = "none"
    mode = str(mode).lower()
    if bool(getattr(args, "defense_adaptive_decoding", False)) and mode == "none":
        mode = "auto"
    if mode not in ADAPTIVE_ATTACK_CHOICES:
        raise ValueError(f"--adaptive_attack must be one of {ADAPTIVE_ATTACK_CHOICES}; got {mode!r}.")
    setattr(args, "adaptive_attack", mode)
    return args


def validate_adaptive_attack_args(args):
    normalize_adaptive_attack_args(args)
    multiplier = int(getattr(args, "adaptive_candidate_multiplier", 50))
    if multiplier <= 0:
        raise ValueError("--adaptive_candidate_multiplier must be positive.")
    cap = getattr(args, "adaptive_candidate_cap", None)
    if cap is not None and int(cap) <= 0:
        raise ValueError("--adaptive_candidate_cap must be positive when set.")
    knowledge = str(getattr(args, "adaptive_lrb_knowledge", "oracle"))
    if knowledge not in ADAPTIVE_LRB_KNOWLEDGE_CHOICES:
        raise ValueError(f"Unsupported --adaptive_lrb_knowledge: {knowledge}")
    reducer = str(getattr(args, "adaptive_lrb_hypothesis_reduce", "min"))
    if reducer not in ADAPTIVE_LRB_HYPOTHESIS_REDUCERS:
        raise ValueError(f"Unsupported --adaptive_lrb_hypothesis_reduce: {reducer}")
    sign_source = str(getattr(args, "adaptive_lrb_sign_source", "legacy_cpu"))
    if sign_source not in ADAPTIVE_LRB_SIGN_SOURCES:
        raise ValueError(f"Unsupported --adaptive_lrb_sign_source: {sign_source}")
    seed_samples = int(getattr(args, "adaptive_lrb_seed_samples", 16))
    if seed_samples <= 0:
        raise ValueError("--adaptive_lrb_seed_samples must be positive.")
    ratios = _adaptive_lrb_ratio_grid(args)
    hidden_signs = knowledge in {"signs_hidden", "method_only"}
    if hidden_signs and adaptive_defense_aware_active(args) and str(getattr(args, "defense", "none")) in LRB_PROJECTION_DEFENSES:
        attack_seed = getattr(args, "adaptive_lrb_attack_seed", None)
        if attack_seed is None:
            raise ValueError(f"--adaptive_lrb_attack_seed is required for {knowledge}.")
        configured_seed = getattr(args, "defense_lrb_seed", None)
        defense_seed = int(getattr(args, "rng_seed", 0) if configured_seed is None else configured_seed)
        if int(attack_seed) == defense_seed:
            raise ValueError("The hidden-sign attack seed must differ from the defense LRB seed.")
    ratio_count = len(ratios) if knowledge in {"ratio_hidden", "method_only"} else 1
    seed_count = seed_samples if hidden_signs else 1
    if ratio_count * seed_count > ADAPTIVE_LRB_MAX_HYPOTHESES:
        raise ValueError(
            f"Adaptive LRB hypotheses exceed {ADAPTIVE_LRB_MAX_HYPOTHESES}: "
            f"{ratio_count} ratios x {seed_count} seeds."
        )
    return args


def _adaptive_lrb_ratio_grid(args) -> list[float]:
    raw = str(getattr(args, "adaptive_lrb_ratio_grid", "auto") or "auto").strip().lower()
    if raw == "auto":
        sensitive = float(getattr(args, "defense_lrb_keep_ratio_sensitive", 0.2))
        other = float(getattr(args, "defense_lrb_keep_ratio_other", 0.75))
        low, high = sorted((sensitive, other))
        if math.isclose(low, high, rel_tol=0.0, abs_tol=1e-12):
            values = [low]
        else:
            values = [low + (high - low) * idx / 5.0 for idx in range(6)]
    else:
        try:
            values = [float(part.strip()) for part in raw.split(",") if part.strip()]
        except ValueError as exc:
            raise ValueError("--adaptive_lrb_ratio_grid must be 'auto' or comma-separated floats.") from exc
        if not values:
            raise ValueError("--adaptive_lrb_ratio_grid cannot be empty.")
    if any(not (0.0 < value <= 1.0) for value in values):
        raise ValueError("Every adaptive LRB keep ratio must be in (0, 1].")
    return list(dict.fromkeys(float(value) for value in values))


def _resolved_ratio_grid_text(args) -> str:
    return ",".join(f"{value:.6g}" for value in _adaptive_lrb_ratio_grid(args))


def _hidden_projection_seeds(args, idx: int) -> list[int]:
    attack_seed = int(getattr(args, "adaptive_lrb_attack_seed"))
    step = int(getattr(args, "defense_rng_step", 0) or 0)
    per_update = str(getattr(args, "defense_lrb_seed_mode", "static")) == "per_update"
    out = []
    for sample_idx in range(int(getattr(args, "adaptive_lrb_seed_samples", 16))):
        base_seed = attack_seed + _ATTACK_SEED_STRIDE * sample_idx
        if per_update:
            base_seed += _UPDATE_SEED_STRIDE * step
        out.append(_layer_projection_seed(base_seed, idx))
    return out


def adaptive_attack_active(args) -> bool:
    normalize_adaptive_attack_args(args)
    return getattr(args, "adaptive_attack", "none") != "none"


def adaptive_defense_aware_active(args) -> bool:
    normalize_adaptive_attack_args(args)
    return getattr(args, "adaptive_attack", "none") == "defense_aware"


def adaptive_attack_profile(args) -> str:
    if not adaptive_attack_active(args):
        return "none"

    mode = getattr(args, "adaptive_attack", "none")
    if mode == "auto":
        return "outlier_decode" if bool(getattr(args, "defense_adaptive_decoding", False)) else "legacy_auto"

    defense = str(getattr(args, "defense", "none"))
    profiles: list[str] = []
    if defense == "topk":
        profiles.append("topk_support")
    elif defense == "compression":
        profiles.append("quantization_robust")
    elif defense in LRB_PROJECTION_DEFENSES:
        profiles.append("projection_span")

    if getattr(args, "defense_pct_mask", None) is not None:
        profiles.append("random_mask_robust")
    if bool(getattr(args, "defense_adaptive_decoding", False)):
        profiles.append("outlier_decode")

    return "+".join(profiles) if profiles else "generic_ranked_span"


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def _selected_gradient_indices(args) -> list[int]:
    raw = getattr(args, "dager_selected_gradient_indices", None)
    if raw is None:
        return []
    if isinstance(raw, str):
        if raw in {"", "n/a", "none"}:
            return []
        return [int(part) for part in raw.split(";") if part]
    return [int(idx) for idx in raw]


def _selected_gradient_names(args, parameter_names: Sequence[str] | None) -> list[str]:
    raw = getattr(args, "dager_selected_gradient_names", None)
    if raw is not None:
        if isinstance(raw, str):
            return [part for part in raw.split(";") if part]
        return [str(name) for name in raw]
    names = list(parameter_names or [])
    out = []
    for idx in _selected_gradient_indices(args):
        out.append(names[idx] if idx < len(names) else f"param_{idx}")
    return out


def _oriented_grad_for_span(args, grad: torch.Tensor) -> torch.Tensor:
    if (
        getattr(args, "train_method", "full") == "full"
        and getattr(args, "model_path", None) in {"gpt2", "openai-community/gpt2-large"}
        and getattr(grad, "ndim", 0) >= 2
    ):
        return grad.T
    return grad


def _sample_unique_count(tensor: torch.Tensor, max_elements: int = 4096) -> int:
    flat = tensor.detach().reshape(-1)
    if flat.numel() == 0:
        return 0
    if flat.numel() > max_elements:
        step = int(math.ceil(flat.numel() / max_elements))
        flat = flat[::step][:max_elements]
    return int(torch.unique(flat.cpu()).numel())


def _support_mask_for_grad(grad: torch.Tensor) -> torch.Tensor | None:
    if getattr(grad, "ndim", 0) < 2:
        return None
    support = grad.detach().ne(0)
    if support.numel() == 0:
        return None
    support = support.reshape(-1, support.shape[-1]).any(dim=0)
    if not bool(support.any()):
        return None
    return support.to(dtype=torch.float32)


def _project_last_dim_signed_pool(
    values: torch.Tensor,
    keep_ratio: float,
    *,
    seed: int,
    mode: str,
    feature_signs: torch.Tensor | None = None,
) -> torch.Tensor:
    if values.shape[-1] <= 1 or keep_ratio >= 0.999:
        return values
    if mode not in {"pool", "signed_pool", "signed_pool_nearest", "signed_stride"}:
        raise ValueError(f"Unsupported adaptive projection mode: {mode}")

    keep_ratio = float(max(1e-4, min(1.0, keep_ratio)))
    hidden = int(values.shape[-1])
    target = max(1, int(round(hidden * keep_ratio)))
    if target >= hidden:
        return values

    dtype = values.dtype
    flat = values.float().reshape(-1, hidden)
    signs = None
    if mode in {"signed_pool", "signed_pool_nearest", "signed_stride"}:
        if feature_signs is not None:
            signs = feature_signs.to(device=values.device, dtype=torch.float32).reshape(-1)
        if signs is None or int(signs.numel()) != hidden:
            gen = torch.Generator(device=values.device)
            gen.manual_seed(int(seed))
            signs = torch.empty((hidden,), device=values.device, dtype=torch.float32)
            signs.bernoulli_(0.5, generator=gen)
            signs = signs.mul_(2.0).sub_(1.0)
        flat = flat * signs

    if mode == "signed_stride":
        indices = torch.linspace(0, hidden - 1, steps=target, device=values.device).round().long()
        pooled = flat.index_select(1, indices).unsqueeze(1)
    else:
        pooled = F.adaptive_avg_pool1d(flat.unsqueeze(1), target)
    if mode == "signed_pool_nearest":
        projected = F.interpolate(pooled, size=hidden, mode="nearest").squeeze(1)
    else:
        projected = F.interpolate(pooled, size=hidden, mode="linear", align_corners=False).squeeze(1)
    if signs is not None:
        projected = projected * signs
    return projected.reshape(values.shape).to(dtype=dtype)


def _lrb_projection_info_for_selected(args, selected_indices: Sequence[int]) -> dict[int, dict]:
    if not selected_indices:
        return {}
    layer_info = getattr(args, "lrb_defense_layer_info", None)
    if not layer_info:
        return {}
    out = {}
    for item in layer_info:
        if not item.get("active", False):
            continue
        idx = int(item.get("idx", -1))
        if idx in selected_indices:
            out[idx] = item
    return out


def _lrb_feature_axis(args, grad: torch.Tensor, original_shape: Sequence[int]) -> int | None:
    if len(original_shape) != 2 or getattr(grad, "ndim", 0) != 2:
        return None
    feature_dim = int(_oriented_grad_for_span(args, grad).shape[-1])
    if feature_dim == int(original_shape[-1]):
        return 1
    if feature_dim == int(original_shape[0]):
        return 0
    return None


def _lrb_feature_signs(
    seed: int,
    original_shape: Sequence[int],
    feature_axis: int | None,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor | None:
    if feature_axis not in {0, 1} or len(original_shape) != 2:
        return None
    rows, cols = (int(original_shape[0]), int(original_shape[1]))
    device = torch.device(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    row_signs = torch.empty((rows,), device=device, dtype=torch.float32)
    row_signs.bernoulli_(0.5, generator=gen)
    row_signs = row_signs.mul_(2.0).sub_(1.0)
    if feature_axis == 0:
        return row_signs
    col_signs = torch.empty((cols,), device=device, dtype=torch.float32)
    col_signs.bernoulli_(0.5, generator=gen)
    return col_signs.mul_(2.0).sub_(1.0)


def prepare_adaptive_attack(
    args,
    defended_grads: Sequence[torch.Tensor | None],
    *,
    parameter_names: Iterable[str] | None = None,
):
    """Prepare defense-aware span metadata after DAGER has selected gradient matrices."""

    normalize_adaptive_attack_args(args)
    selected_indices = _selected_gradient_indices(args)
    parameter_names = list(parameter_names or [])
    selected_names = _selected_gradient_names(args, parameter_names)
    profile = adaptive_attack_profile(args)
    active = adaptive_attack_active(args)
    defense = str(getattr(args, "defense", "none"))

    info = {
        "adaptive_attack": getattr(args, "adaptive_attack", "none"),
        "adaptive_attack_active": active,
        "adaptive_attack_profile": profile,
        "adaptive_candidate_multiplier": int(getattr(args, "adaptive_candidate_multiplier", 50)),
        "adaptive_candidate_cap": getattr(args, "adaptive_candidate_cap", None),
        "adaptive_span_transform": "none",
        "adaptive_support_density_mean": "n/a",
        "adaptive_quantization_levels_mean": "n/a",
        "adaptive_lrb_keep_ratio_mean": "n/a",
        "adaptive_lrb_knowledge": str(getattr(args, "adaptive_lrb_knowledge", "oracle")),
        "adaptive_lrb_sign_source": str(getattr(args, "adaptive_lrb_sign_source", "legacy_cpu")),
        "adaptive_lrb_ratio_grid": _resolved_ratio_grid_text(args),
        "adaptive_lrb_attack_seed": (
            "n/a"
            if getattr(args, "adaptive_lrb_attack_seed", None) is None
            else int(getattr(args, "adaptive_lrb_attack_seed"))
        ),
        "adaptive_lrb_seed_samples": int(getattr(args, "adaptive_lrb_seed_samples", 16)),
        "adaptive_lrb_hypothesis_reduce": str(getattr(args, "adaptive_lrb_hypothesis_reduce", "min")),
        "adaptive_lrb_hypothesis_count_mean": "n/a",
        "adaptive_lrb_ratio_knowledge": "n/a",
        "adaptive_lrb_sign_knowledge": "n/a",
        "adaptive_selected_gradients": ";".join(selected_names) if selected_names else "n/a",
    }
    state = {"layers": {}, "profile": profile}

    if not adaptive_defense_aware_active(args) or not selected_indices:
        setattr(args, "adaptive_attack_info", info)
        setattr(args, "adaptive_attack_state", state)
        return args

    knowledge = str(getattr(args, "adaptive_lrb_knowledge", "oracle"))
    sign_source = str(getattr(args, "adaptive_lrb_sign_source", "legacy_cpu"))
    lrb_projection_info = {}
    if defense in LRB_PROJECTION_DEFENSES and knowledge != "method_only":
        lrb_projection_info = _lrb_projection_info_for_selected(args, selected_indices)

    support_densities: list[float] = []
    quant_levels: list[float] = []
    lrb_keep_values: list[float] = []
    lrb_hypothesis_counts: list[float] = []
    transforms: list[str] = []
    projection_mode = str(getattr(args, "defense_lrb_projection", "signed_pool"))
    base_seed = int(getattr(args, "rng_seed", 0))

    for layer_pos, idx in enumerate(selected_indices):
        if idx >= len(defended_grads):
            continue
        grad = defended_grads[idx]
        if grad is None:
            continue
        oriented = _oriented_grad_for_span(args, grad)
        layer_state = {}

        if defense == "topk":
            support_mask = _support_mask_for_grad(oriented)
            if support_mask is not None:
                layer_state["support_mask"] = support_mask
                support_densities.append(float(support_mask.mean().item()))
                transforms.append("topk_support_mask")

        if defense == "compression":
            quant_levels.append(float(_sample_unique_count(oriented)))
            transforms.append("ranked_quantized_span")

        if defense in LRB_PROJECTION_DEFENSES:
            projection_info = lrb_projection_info.get(idx)
            if knowledge != "method_only" and projection_info is None:
                raise RuntimeError(
                    f"Adaptive LRB knowledge={knowledge} requires defense metadata for gradient index {idx}."
                )
            original_shape = tuple(
                int(dim)
                for dim in ((projection_info or {}).get("shape", tuple(grad.shape)))
            )
            feature_axis = _lrb_feature_axis(args, grad, original_shape)
            layer_projection_mode = str((projection_info or {}).get("projection_mode", projection_mode))
            if knowledge in {"oracle", "signs_hidden"}:
                keep_ratios = [float(projection_info["keep_ratio"])]
                lrb_keep_values.extend(keep_ratios)
            else:
                keep_ratios = _adaptive_lrb_ratio_grid(args)
            if knowledge in {"oracle", "ratio_hidden"}:
                projection_seeds = [
                    int(projection_info.get("projection_seed", _layer_projection_seed(base_seed, idx)))
                ]
            elif layer_projection_mode == "pool":
                projection_seeds = [0]
            else:
                projection_seeds = _hidden_projection_seeds(args, idx)

            hypotheses = []
            for keep_ratio in keep_ratios:
                for projection_seed in projection_seeds:
                    signs = None
                    if layer_projection_mode != "pool":
                        sign_device = "cpu"
                        if sign_source == "defense_device":
                            if knowledge in {"oracle", "ratio_hidden"}:
                                sign_device = (projection_info or {}).get("projection_device", grad.device)
                            else:
                                sign_device = grad.device
                        signs = _lrb_feature_signs(
                            projection_seed,
                            original_shape,
                            feature_axis,
                            device=sign_device,
                        )
                    hypotheses.append(
                        {
                            "keep_ratio": float(keep_ratio),
                            "projection_mode": layer_projection_mode,
                            "projection_seed": int(projection_seed),
                            "feature_signs": signs,
                        }
                    )
            if len(hypotheses) > ADAPTIVE_LRB_MAX_HYPOTHESES:
                raise RuntimeError(
                    f"Layer {idx} generated {len(hypotheses)} adaptive LRB hypotheses; "
                    f"maximum is {ADAPTIVE_LRB_MAX_HYPOTHESES}."
                )

            if knowledge == "oracle":
                keep_ratio = hypotheses[0]["keep_ratio"]
                projection_seed = hypotheses[0]["projection_seed"]
                layer_state["lrb_keep_ratio"] = keep_ratio
                layer_state["lrb_projection_mode"] = layer_projection_mode
                layer_state["lrb_projection_seed"] = projection_seed
                layer_state["lrb_feature_signs"] = hypotheses[0]["feature_signs"]
            else:
                layer_state["lrb_hypotheses"] = hypotheses
            layer_state["lrb_original_shape"] = original_shape
            layer_state["lrb_feature_axis"] = "n/a" if feature_axis is None else feature_axis
            lrb_hypothesis_counts.append(float(len(hypotheses)))
            transforms.append("lrb_public_projection" if knowledge == "oracle" else "lrb_hypothesis_projection")

        if layer_state:
            state["layers"][int(layer_pos)] = layer_state

    if support_densities:
        info["adaptive_support_density_mean"] = _fmt_float(mean(support_densities))
    if quant_levels:
        info["adaptive_quantization_levels_mean"] = _fmt_float(mean(quant_levels))
    if lrb_keep_values:
        info["adaptive_lrb_keep_ratio_mean"] = _fmt_float(mean(lrb_keep_values))
    if lrb_hypothesis_counts:
        info["adaptive_lrb_hypothesis_count_mean"] = _fmt_float(mean(lrb_hypothesis_counts))
        info["adaptive_lrb_ratio_knowledge"] = "exact" if knowledge in {"oracle", "signs_hidden"} else "hidden"
        if knowledge in {"signs_hidden", "method_only"}:
            info["adaptive_lrb_sign_knowledge"] = "hidden"
        elif sign_source == "defense_device":
            info["adaptive_lrb_sign_knowledge"] = "exact"
        else:
            info["adaptive_lrb_sign_knowledge"] = "legacy_seed_replay"
    if transforms:
        info["adaptive_span_transform"] = "+".join(sorted(set(transforms)))

    # A standalone state-inference experiment may install an estimated state
    # immediately before a normal DAGER decode.  Keep this opt-in hook private
    # and late in the function so every legacy caller retains the exact state
    # construction above.  The override is intentionally consumed only by the
    # new entrypoint; normal attack.py runs never set this attribute.
    override = getattr(args, "_state_inference_override", None)
    if override is not None:
        state = override
        info["adaptive_span_transform"] = "lrb_state_inference_projection"
        info["adaptive_lrb_knowledge"] = "state_inference_v1"
        info["adaptive_lrb_ratio_knowledge"] = "estimated"
        info["adaptive_lrb_sign_knowledge"] = "estimated"

    setattr(args, "adaptive_attack_info", info)
    setattr(args, "adaptive_attack_state", state)
    return args


def adaptive_attack_summary_fields(args):
    info = getattr(args, "adaptive_attack_info", None)
    if info is None:
        normalize_adaptive_attack_args(args)
        info = {
            "adaptive_attack": getattr(args, "adaptive_attack", "none"),
            "adaptive_attack_active": adaptive_attack_active(args),
            "adaptive_attack_profile": adaptive_attack_profile(args),
            "adaptive_candidate_multiplier": int(getattr(args, "adaptive_candidate_multiplier", 50)),
            "adaptive_candidate_cap": getattr(args, "adaptive_candidate_cap", None),
            "adaptive_span_transform": "none",
            "adaptive_support_density_mean": "n/a",
            "adaptive_quantization_levels_mean": "n/a",
            "adaptive_lrb_keep_ratio_mean": "n/a",
            "adaptive_lrb_knowledge": str(getattr(args, "adaptive_lrb_knowledge", "oracle")),
            "adaptive_lrb_sign_source": str(getattr(args, "adaptive_lrb_sign_source", "legacy_cpu")),
            "adaptive_lrb_ratio_grid": _resolved_ratio_grid_text(args),
            "adaptive_lrb_attack_seed": (
                "n/a"
                if getattr(args, "adaptive_lrb_attack_seed", None) is None
                else int(getattr(args, "adaptive_lrb_attack_seed"))
            ),
            "adaptive_lrb_seed_samples": int(getattr(args, "adaptive_lrb_seed_samples", 16)),
            "adaptive_lrb_hypothesis_reduce": str(getattr(args, "adaptive_lrb_hypothesis_reduce", "min")),
            "adaptive_lrb_hypothesis_count_mean": "n/a",
            "adaptive_lrb_ratio_knowledge": "n/a",
            "adaptive_lrb_sign_knowledge": "n/a",
            "adaptive_selected_gradients": "n/a",
        }
    return [
        ("adaptive_attack", info.get("adaptive_attack", "none")),
        ("adaptive_attack_active", info.get("adaptive_attack_active", False)),
        ("adaptive_attack_profile", info.get("adaptive_attack_profile", "none")),
        ("adaptive_candidate_multiplier", info.get("adaptive_candidate_multiplier", "n/a")),
        ("adaptive_candidate_cap", info.get("adaptive_candidate_cap", "n/a")),
        ("adaptive_span_transform", info.get("adaptive_span_transform", "none")),
        ("adaptive_support_density_mean", info.get("adaptive_support_density_mean", "n/a")),
        ("adaptive_quantization_levels_mean", info.get("adaptive_quantization_levels_mean", "n/a")),
        ("adaptive_lrb_keep_ratio_mean", info.get("adaptive_lrb_keep_ratio_mean", "n/a")),
        ("adaptive_lrb_knowledge", info.get("adaptive_lrb_knowledge", "oracle")),
        ("adaptive_lrb_sign_source", info.get("adaptive_lrb_sign_source", "legacy_cpu")),
        ("adaptive_lrb_ratio_grid", info.get("adaptive_lrb_ratio_grid", "n/a")),
        ("adaptive_lrb_attack_seed", info.get("adaptive_lrb_attack_seed", "n/a")),
        ("adaptive_lrb_seed_samples", info.get("adaptive_lrb_seed_samples", "n/a")),
        ("adaptive_lrb_hypothesis_reduce", info.get("adaptive_lrb_hypothesis_reduce", "min")),
        ("adaptive_lrb_hypothesis_count_mean", info.get("adaptive_lrb_hypothesis_count_mean", "n/a")),
        ("adaptive_lrb_ratio_knowledge", info.get("adaptive_lrb_ratio_knowledge", "n/a")),
        ("adaptive_lrb_sign_knowledge", info.get("adaptive_lrb_sign_knowledge", "n/a")),
        ("adaptive_selected_gradients", info.get("adaptive_selected_gradients", "n/a")),
    ]


def adaptive_transform_candidates(args, values: torch.Tensor, *, layer_position: int = 0) -> torch.Tensor:
    if not adaptive_defense_aware_active(args):
        return values
    state = getattr(args, "adaptive_attack_state", None)
    if not state:
        return values
    layer_state = state.get("layers", {}).get(int(layer_position), {})
    out = values

    support_mask = layer_state.get("support_mask")
    if support_mask is not None and int(support_mask.numel()) == int(out.shape[-1]):
        mask = support_mask.to(device=out.device, dtype=out.dtype).reshape(
            *((1,) * (out.dim() - 1)),
            out.shape[-1],
        )
        out = out * mask

    keep_ratio = layer_state.get("lrb_keep_ratio")
    if keep_ratio is not None:
        out = _project_last_dim_signed_pool(
            out,
            float(keep_ratio),
            seed=int(layer_state.get("lrb_projection_seed", 0)),
            mode=str(layer_state.get("lrb_projection_mode", "signed_pool")),
            feature_signs=layer_state.get("lrb_feature_signs"),
        )
    return out


def adaptive_check_if_in_span(args, R_K_norm, values, norm="l2", *, layer_position: int = 0):
    state = getattr(args, "adaptive_attack_state", None)
    layer_state = (state or {}).get("layers", {}).get(int(layer_position), {})
    hypotheses = layer_state.get("lrb_hypotheses", [])
    if adaptive_defense_aware_active(args) and hypotheses:
        scores = []
        for hypothesis in hypotheses:
            transformed = _project_last_dim_signed_pool(
                values,
                float(hypothesis["keep_ratio"]),
                seed=int(hypothesis["projection_seed"]),
                mode=str(hypothesis["projection_mode"]),
                feature_signs=hypothesis.get("feature_signs"),
            )
            scores.append(check_if_in_span(R_K_norm, transformed, norm))
        stacked = torch.stack(scores, dim=0)
        reducer = str(getattr(args, "adaptive_lrb_hypothesis_reduce", "min"))
        return stacked.amin(dim=0) if reducer == "min" else stacked.mean(dim=0)
    values = adaptive_transform_candidates(args, values, layer_position=layer_position)
    return check_if_in_span(R_K_norm, values, norm)


def _ranked_l1_enabled(args) -> bool:
    if not adaptive_defense_aware_active(args):
        return False
    defense = str(getattr(args, "defense", "none"))
    return defense in RANKED_SPAN_DEFENSES or adaptive_attack_profile(args) == "generic_ranked_span"


def _candidate_cap(args, requested_B: int, total: int) -> int:
    explicit = getattr(args, "adaptive_candidate_cap", None)
    if explicit is not None:
        return min(total, int(explicit))
    multiplier = int(getattr(args, "adaptive_candidate_multiplier", 50))
    return min(total, max(int(requested_B), int(requested_B) * multiplier))


def _unravel_indices(flat_indices: torch.Tensor, shape: Sequence[int]) -> tuple[torch.Tensor, ...]:
    coords = []
    remaining = flat_indices
    for dim in reversed(shape):
        coords.append(remaining % int(dim))
        remaining = torch.div(remaining, int(dim), rounding_mode="floor")
    return tuple(reversed(coords))


def adaptive_get_top_B_in_span(
    args,
    R_K_norm,
    values,
    B,
    thresh,
    norm,
    *,
    layer_position: int = 0,
):
    size = adaptive_check_if_in_span(args, R_K_norm, values, norm, layer_position=layer_position)
    if _ranked_l1_enabled(args):
        bools = size < thresh
        which = torch.where(bools)
        flat = size.reshape(-1)
        cap = _candidate_cap(args, int(B), int(flat.numel()))
        setattr(args, "_adaptive_l1_threshold_hit", bool(which[0].numel() > 0))
        setattr(args, "_adaptive_l1_ranked_fallback", bool(which[0].numel() == 0))
        setattr(args, "_adaptive_l1_stop_after_current_position", bool(which[0].numel() == 0))
        if which[0].numel() > 0:
            _, idx = torch.sort(size[which])
            idx = idx[:cap]
            return tuple(w[idx] for w in which)
        _, flat_idx = torch.topk(flat, k=cap, largest=False)
        return _unravel_indices(flat_idx, size.shape)

    setattr(args, "_adaptive_l1_threshold_hit", True)
    setattr(args, "_adaptive_l1_ranked_fallback", False)
    setattr(args, "_adaptive_l1_stop_after_current_position", False)
    bools = size < thresh
    which = torch.where(bools)
    _, idx = torch.sort(size[which])
    return tuple(w[idx] for w in which)


def adaptive_get_span_dists(args, model_wrapper, R_Qs, embeds, p=0, stage="token"):
    dists = []
    if stage == "token":
        dists.append(
            adaptive_check_if_in_span(args, R_Qs[0], embeds, args.dist_norm, layer_position=0).T
        )
        if p == 0:
            sentences = torch.arange(embeds.shape[1]).unsqueeze(1).to(model_wrapper.args.device)
            embs = model_wrapper.get_layer_inputs(sentences, layers=args.n_layers - 1)
    else:
        embs = [e.to(model_wrapper.args.device) for e in embeds]

    if p == 0:
        for i in range(model_wrapper.args.n_layers - 1):
            dists.append(
                adaptive_check_if_in_span(
                    args,
                    R_Qs[i + 1],
                    embs[i],
                    args.dist_norm,
                    layer_position=i + 1,
                )
            )

    joined = torch.cat(dists, axis=1).clamp(min=1e-12, max=1 - 1e-12)
    if bool(getattr(args, "verbose_attack_debug", False)):
        print("dists", joined.shape)
    return (torch.log(joined) - torch.log(1 - joined)).mean(axis=1).cpu().detach()
