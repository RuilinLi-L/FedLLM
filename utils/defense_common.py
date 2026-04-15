from __future__ import annotations

from argparse import ArgumentParser
from typing import Sequence

import torch


DEFENSE_CHOICES = [
    "none",
    "noise",
    "dpsgd",
    "topk",
    "compression",
    "soteria",
    "mixup",
    "dager",
    "lrb",
]


def add_shared_defense_args(parser: ArgumentParser, *, default_grad_mode: str = "eval") -> ArgumentParser:
    """Add the shared defense CLI used by attack.py and training-side utilities."""

    parser.add_argument("--rng_seed", type=int, default=101)
    parser.add_argument(
        "--defense",
        type=str,
        default="none",
        choices=DEFENSE_CHOICES,
        help="Defense applied to shared gradients or generated defended gradients.",
    )
    parser.add_argument(
        "--defense_noise",
        type=float,
        default=None,
        help="Gaussian noise scale for noise / dpsgd or legacy none+noise mode.",
    )
    parser.add_argument(
        "--defense_clip_norm",
        type=float,
        default=1.0,
        help="Per-example L2 clip norm C for dpsgd.",
    )
    parser.add_argument(
        "--defense_topk_ratio",
        type=float,
        default=0.1,
        help="Fraction of |gradient| elements to keep per tensor.",
    )
    parser.add_argument(
        "--defense_n_bits",
        type=int,
        default=8,
        help="Bits per tensor for uniform gradient quantization.",
    )
    parser.add_argument(
        "--defense_soteria_pruning_rate",
        type=float,
        default=60.0,
        help="Percent of classifier-input representation dimensions pruned by Soteria.",
    )
    parser.add_argument(
        "--defense_soteria_sample_dims",
        type=int,
        default=None,
        help="If set, score only this many random hidden dims.",
    )
    parser.add_argument(
        "--defense_mixup_alpha",
        type=float,
        default=1.0,
        help="Beta(alpha, alpha) for MixUp mixing coefficient.",
    )
    parser.add_argument(
        "--defense_pct_mask",
        type=float,
        default=None,
        help="Optional element-wise random mask applied after the main defense.",
    )
    parser.add_argument(
        "--defense_lrb_sensitive_n_layers",
        type=int,
        default=2,
        help="How many earliest transformer layers receive the strongest LRB protection.",
    )
    parser.add_argument(
        "--defense_lrb_keep_ratio_sensitive",
        type=float,
        default=0.2,
        help="Target keep ratio for sensitive layers under LRB.",
    )
    parser.add_argument(
        "--defense_lrb_keep_ratio_other",
        type=float,
        default=0.75,
        help="Target keep ratio for less-sensitive layers under LRB.",
    )
    parser.add_argument(
        "--defense_lrb_clip_scale_sensitive",
        type=float,
        default=0.5,
        help="Clip threshold multiplier for sensitive layers under LRB.",
    )
    parser.add_argument(
        "--defense_lrb_clip_scale_other",
        type=float,
        default=1.0,
        help="Clip threshold multiplier for less-sensitive layers under LRB.",
    )
    parser.add_argument(
        "--defense_lrb_noise_sensitive",
        type=float,
        default=0.03,
        help="Orthogonal noise multiplier for sensitive layers under LRB.",
    )
    parser.add_argument(
        "--defense_lrb_noise_other",
        type=float,
        default=0.005,
        help="Orthogonal noise multiplier for less-sensitive layers under LRB.",
    )
    parser.add_argument(
        "--defense_lrb_empirical_weight",
        type=float,
        default=0.6,
        help="Blend weight for on-the-fly gradient calibration in LRB.",
    )
    parser.add_argument(
        "--defense_lrb_calibration_samples",
        type=int,
        default=4096,
        help="Max elements per tensor used by LRB calibration sketches.",
    )
    parser.add_argument(
        "--defense_lrb_projection",
        type=str,
        default="signed_pool",
        choices=["signed_pool", "pool"],
        help="Public subspace projection used by LRB.",
    )
    parser.add_argument(
        "--defense_dager_basis_perturb",
        action="store_true",
        default=True,
        help="Enable dynamic basis perturbation for DAGER defense.",
    )
    parser.add_argument(
        "--no_defense_dager_basis_perturb",
        action="store_false",
        dest="defense_dager_basis_perturb",
        help="Disable dynamic basis perturbation for DAGER defense.",
    )
    parser.add_argument(
        "--defense_dager_basis_noise_scale",
        type=float,
        default=0.01,
        help="Noise scale for dynamic basis perturbation.",
    )
    parser.add_argument(
        "--defense_dager_offset_embedding",
        action="store_true",
        default=False,
        help="Enable stochastic offset embedding for DAGER defense.",
    )
    parser.add_argument(
        "--defense_dager_offset_scale",
        type=float,
        default=0.01,
        help="Offset scale for stochastic offset embedding.",
    )
    parser.add_argument(
        "--defense_dager_gradient_slicing",
        action="store_true",
        default=False,
        help="Enable gradient slicing for DAGER defense.",
    )
    parser.add_argument(
        "--defense_dager_slice_first_n",
        type=int,
        default=None,
        help="Send only the first n layers.",
    )
    parser.add_argument(
        "--defense_dager_slice_last_n",
        type=int,
        default=None,
        help="Send only the last n layers.",
    )
    parser.add_argument(
        "--defense_dager_random_slice",
        action="store_true",
        default=False,
        help="Randomly select layers to send.",
    )
    parser.add_argument(
        "--defense_dager_slice_prob",
        type=float,
        default=0.5,
        help="Probability of sending each layer when random slicing is enabled.",
    )
    parser.add_argument(
        "--defense_dager_rank_limit",
        action="store_true",
        default=False,
        help="Enable rank-limiting defense for DAGER.",
    )
    parser.add_argument(
        "--grad_mode",
        type=str,
        default=default_grad_mode,
        choices=["eval", "train"],
    )
    return parser


def normalize_legacy_training_defense_args(args):
    """Keep train.py compatible with historical --noise / --pct_mask usage."""

    if getattr(args, "noise", None) is not None and getattr(args, "defense_noise", None) is None:
        args.defense_noise = args.noise
    if getattr(args, "pct_mask", None) is not None and getattr(args, "defense_pct_mask", None) is None:
        args.defense_pct_mask = args.pct_mask
    return args


def defense_param_spec(args) -> tuple[str, object]:
    defense = getattr(args, "defense", "none")
    mapping = {
        "noise": ("defense_noise", getattr(args, "defense_noise", None)),
        "dpsgd": ("defense_noise", getattr(args, "defense_noise", None)),
        "topk": ("defense_topk_ratio", getattr(args, "defense_topk_ratio", None)),
        "compression": ("defense_n_bits", getattr(args, "defense_n_bits", None)),
        "soteria": (
            "defense_soteria_pruning_rate",
            getattr(args, "defense_soteria_pruning_rate", None),
        ),
        "mixup": ("defense_mixup_alpha", getattr(args, "defense_mixup_alpha", None)),
        "lrb": (
            "defense_lrb_keep_ratio_sensitive",
            getattr(args, "defense_lrb_keep_ratio_sensitive", None),
        ),
    }
    if defense == "none":
        return "n/a", "n/a"
    return mapping.get(defense, ("n/a", "n/a"))


def safe_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def fmt_summary_value(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def iter_trainable_parameters(model) -> list[torch.nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def capture_gradients(params: Sequence[torch.nn.Parameter]) -> tuple[torch.Tensor | None, ...]:
    return tuple(
        None if param.grad is None else param.grad.detach().clone()
        for param in params
    )


def overwrite_gradients(
    params: Sequence[torch.nn.Parameter],
    grads: Sequence[torch.Tensor | None],
) -> None:
    if len(params) != len(grads):
        raise ValueError("Parameter / gradient length mismatch.")

    for param, grad in zip(params, grads):
        if grad is None:
            param.grad = None
            continue
        grad = grad.detach().to(device=param.device, dtype=param.dtype)
        if param.grad is None:
            param.grad = grad.clone()
        else:
            param.grad.detach().copy_(grad)


def flatten_grads(grads: Sequence[torch.Tensor | None]) -> torch.Tensor:
    flat_parts = [
        grad.detach().float().reshape(-1)
        for grad in grads
        if grad is not None
    ]
    if not flat_parts:
        return torch.zeros(1, dtype=torch.float32)
    return torch.cat(flat_parts)


def grad_similarity_metrics(
    original_grads: Sequence[torch.Tensor | None],
    defended_grads: Sequence[torch.Tensor | None],
) -> tuple[float, float]:
    original = flatten_grads(original_grads)
    defended = flatten_grads(defended_grads)
    original_norm = float(original.norm().item())
    defended_norm = float(defended.norm().item())
    if original_norm <= 1e-12 and defended_norm <= 1e-12:
        return 1.0, 1.0
    cosine_denom = original_norm * defended_norm
    if cosine_denom <= 1e-12:
        cosine = 0.0
    else:
        cosine = float(torch.dot(original, defended).item() / cosine_denom)
    norm_retention = 1.0 if original_norm <= 1e-12 else defended_norm / (original_norm + 1e-12)
    return cosine, norm_retention
