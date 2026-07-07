from __future__ import annotations

import argparse
import datetime
import sys
import time
import torch

from attacks.peftleak_image.core import (
    PeftLeakRecoveryResult,
    build_public_patch_statistics,
    build_official_aligned_adapter_gradients,
    build_official_aligned_probe_statistics,
    build_peftleak_probe_statistics,
    build_peftleak_probe_statistics_from_patch_stats,
    build_shared_adapter_gradient_bundle,
    build_shared_adapter_gradients,
    cluster_and_reassemble,
    extract_image_patches,
    fold_image_patches,
    load_public_patch_statistics,
    move_peftleak_probe_statistics,
    mse_psnr,
    recover_patches_from_official_adapter_grads,
    recover_patches_from_shared_adapter_grads,
    recover_patch_from_adapter_grads,
    resolve_cluster_method,
    save_public_patch_statistics,
    simple_ssim,
)
from utils.defense_common import add_shared_defense_args, defense_param_spec, fmt_summary_value
from utils.defenses import dpsgd_defense, gradient_compression, noise_injection, topk_sparsification
from utils.gpu import resolve_cuda_device
from utils.lrb_defense import apply_lrb_defense
from utils.lrb_presets import apply_lrb_preset


SUMMARY_START = "===== RESULT SUMMARY START ====="
SUMMARY_END = "===== RESULT SUMMARY END ====="
PEFTLEAK_IMAGE_REPRODUCTION_LEVEL = "peftleak_style_shared_bins"
PEFTLEAK_IMAGE_OFFICIAL_REPRODUCTION_LEVEL = "peftleak_official_aligned_v1"
SYNTHETIC_RATIO_REPRODUCTION_LEVEL = "synthetic_ratio_debug"
PEFTLEAK_PROFILE_OFFICIAL_CIFAR32 = {
    "image_size": 32,
    "channels": 3,
    "n_classes": 100,
    "patch_size": 16,
    "peftleak_num_bins": 320,
    "adapter_hidden_dim": 64,
    "adapter_bottleneck_dim": 64,
    "peftleak_embed_scale": 0.5,
    "peftleak_gap": 0,
}
SUPPORTED_IMAGE_DEFENSES = {
    "none",
    "noise",
    "dpsgd",
    "topk",
    "compression",
    "soteria",
    "mixup",
    "lrb",
    "lrbprojonly",
    "signed_bottleneck",
}


class SyntheticFallbackError(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = str(reason)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PEFTLeak-style image-adapter shared-bin mechanism with synthetic ratio and torchvision ViT-adapter modes")
    parser.add_argument("--mode", choices=["vit_adapter", "official_vit_adapter", "synthetic_ratio"], default="vit_adapter")
    parser.add_argument(
        "--peftleak_profile",
        choices=["lightweight", "official_cifar32", "custom"],
        default="lightweight",
        help="Preset for PEFTLeak image geometry/probe settings. Explicit CLI values override preset defaults.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar10", "cifar100", "tinyimagenet", "imagenet", "synthetic"],
    )
    parser.add_argument("--data_root", type=str, default="./models_cache")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda", help="Runtime device. Bare 'cuda' auto-selects an idle visible GPU and falls back to CPU if CUDA is unavailable.")
    parser.add_argument("--model_path", type=str, default="torchvision_vit_small", help="Backbone selector such as torchvision_vit_small or vit_b_16.")
    parser.add_argument("--finetuned_path", type=str, default=None)
    parser.add_argument("--public_stats_path", type=str, default=None)
    parser.add_argument("--save_public_stats_path", type=str, default=None)
    parser.add_argument("--public_n_images", type=int, default=64)
    parser.add_argument("--n_images", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--n_classes", type=int, default=100)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--adapter_hidden_dim", type=int, default=8)
    parser.add_argument("--peftleak_num_bins", type=int, default=32)
    parser.add_argument("--peftleak_position_sigma", type=float, default=1.0)
    parser.add_argument("--peftleak_embed_scale", type=float, default=0.5)
    parser.add_argument("--peftleak_gap", type=int, default=0)
    parser.add_argument("--patch_recovery_mse_threshold", type=float, default=1e-6)
    parser.add_argument("--vit_config", type=str, default="cifar_small", choices=["cifar_small", "vit_b_16"])
    parser.add_argument("--adapter_layers", type=str, default="all", choices=["all", "msa", "mlp", "first_n", "last_n"])
    parser.add_argument("--attack_rounds", type=int, default=1)
    parser.add_argument("--adapter_bottleneck_dim", type=int, default=8)
    parser.add_argument("--public_split_size", type=int, default=None)
    parser.add_argument("--official_grouping", type=str, default="tag", choices=["tag", "cluster", "oracle_debug"])
    parser.add_argument("--metrics", type=str, default="mse,psnr,ssim,lpips,patch_recovery")
    parser.add_argument(
        "--fail_on_synthetic_fallback",
        action="store_true",
        help="Fail instead of falling back to synthetic data when a real image dataset is unavailable.",
    )
    add_shared_defense_args(parser, default_grad_mode="eval")
    return parser


def _explicit_cli_fields(argv) -> set[str]:
    raw = list(sys.argv[1:] if argv is None else argv)
    fields = set()
    for item in raw:
        if not isinstance(item, str) or not item.startswith("--"):
            continue
        option = item.split("=", 1)[0]
        key = option[2:].replace("-", "_")
        fields.add(key)
    return fields


def _apply_peftleak_profile(args, explicit_fields: set[str]):
    profile = str(getattr(args, "peftleak_profile", "lightweight"))
    profile_fields = set(PEFTLEAK_PROFILE_OFFICIAL_CIFAR32)
    override_fields = sorted(profile_fields.intersection(explicit_fields))
    if profile == "official_cifar32":
        for field, value in PEFTLEAK_PROFILE_OFFICIAL_CIFAR32.items():
            if field not in explicit_fields:
                setattr(args, field, value)
    official_like = all(
        getattr(args, field, None) == value
        for field, value in PEFTLEAK_PROFILE_OFFICIAL_CIFAR32.items()
    )
    if profile == "official_cifar32" and override_fields:
        warning = "profile_overrides:" + ",".join(override_fields)
    elif profile == "custom":
        warning = "custom_profile"
    else:
        warning = None
    args.profile_override_count = len(override_fields) if profile == "official_cifar32" else 0
    args.official_like_config = bool(official_like)
    args.config_warning = warning
    return args


def _synthetic_images(n_images: int, channels: int, image_size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    images = torch.rand(n_images, channels, image_size, image_size, generator=gen)
    labels = torch.arange(n_images, dtype=torch.long)
    return images, labels


def _load_images(args, *, public: bool = False) -> tuple[torch.Tensor, torch.Tensor, str, str | None]:
    public_count = args.public_split_size if getattr(args, "public_split_size", None) is not None else args.public_n_images
    n_images = int(public_count if public else args.n_images)
    seed = int(args.rng_seed) + (7919 if public else 0)
    fallback_reason = None
    if args.dataset in {"cifar10", "cifar100"}:
        try:
            from torchvision import datasets, transforms

            data_root = args.cache_dir or args.data_root
            dataset_cls = datasets.CIFAR10 if args.dataset == "cifar10" else datasets.CIFAR100
            ds = dataset_cls(
                root=data_root,
                train=not public,
                download=False,
                transform=transforms.Compose(
                    [
                        transforms.Resize((args.image_size, args.image_size)),
                        transforms.ToTensor(),
                    ]
                ),
            )
            max_count = max(0, min(n_images, len(ds)))
            imgs = []
            labels = []
            for idx in range(max_count):
                img, label = ds[idx]
                imgs.append(img)
                labels.append(int(label))
            if imgs:
                source = f"{args.dataset}_public" if public else f"{args.dataset}_train"
                return torch.stack(imgs, dim=0), torch.tensor(labels, dtype=torch.long), source, None
            fallback_reason = f"{args.dataset} split is empty at {data_root}."
        except Exception as exc:
            fallback_reason = str(exc)
    elif args.dataset in {"tinyimagenet", "imagenet"}:
        try:
            from torchvision import datasets, transforms

            data_root = args.cache_dir or args.data_root
            if args.dataset == "tinyimagenet":
                split = "val" if public else "train"
                candidates = [
                    f"{data_root}/tiny-imagenet-200/{split}",
                    f"{data_root}/tinyimagenet/{split}",
                ]
            else:
                split = "val" if public else "train"
                candidates = [
                    f"{data_root}/imagenet/{split}",
                    f"{data_root}/ImageNet/{split}",
                ]
            split_path = None
            for candidate in candidates:
                try:
                    from pathlib import Path

                    if Path(candidate).is_dir():
                        split_path = candidate
                        break
                except OSError:
                    continue
            if split_path is None:
                raise FileNotFoundError(f"No ImageFolder split found under {data_root}.")
            ds = datasets.ImageFolder(
                root=split_path,
                transform=transforms.Compose(
                    [
                        transforms.Resize((args.image_size, args.image_size)),
                        transforms.ToTensor(),
                    ]
                ),
            )
            max_count = max(0, min(n_images, len(ds)))
            imgs = []
            labels = []
            for local_idx in range(max_count):
                idx = local_idx
                img, label = ds[idx]
                imgs.append(img)
                labels.append(int(label))
            if imgs:
                source = f"{args.dataset}_public" if public else f"{args.dataset}_train"
                return torch.stack(imgs, dim=0), torch.tensor(labels, dtype=torch.long), source, None
            fallback_reason = f"{args.dataset} split is empty at {split_path}."
        except Exception as exc:
            fallback_reason = str(exc)

    if args.dataset != "synthetic":
        print(
            f"[peftleak-image] {args.dataset} unavailable ({fallback_reason}); falling back to synthetic images.",
            flush=True,
        )
        if getattr(args, "fail_on_synthetic_fallback", False):
            raise SyntheticFallbackError(fallback_reason or f"{args.dataset} unavailable.")

    images, labels = _synthetic_images(n_images, args.channels, args.image_size, seed)
    return images, labels % int(args.n_classes), "synthetic_public" if public else "synthetic_attack", fallback_reason


def _patch_to_grads(patch: torch.Tensor, hidden_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    bias_grad = torch.linspace(1.0, 2.0, hidden_dim, dtype=patch.dtype, device=patch.device)
    weight_grad = bias_grad.unsqueeze(1) * patch.unsqueeze(0)
    return weight_grad, bias_grad


def _make_raw_gradient_tuple(patches: torch.Tensor, hidden_dim: int):
    grads = []
    names = []
    for idx, patch in enumerate(patches.reshape(-1, patches.shape[-1])):
        w, b = _patch_to_grads(patch, hidden_dim)
        grads.extend([w, b])
        names.extend([f"vit.synthetic.patch_{idx}.weight", f"vit.synthetic.patch_{idx}.bias"])
    return tuple(grads), names


def _recover_from_gradient_tuple(grads):
    patches = []
    for idx in range(0, len(grads), 2):
        w = grads[idx]
        b = grads[idx + 1]
        if w is None or b is None:
            continue
        try:
            patches.append(recover_patch_from_adapter_grads(w, b))
        except ValueError:
            patches.append(torch.zeros(w.shape[-1], dtype=w.dtype, device=w.device))
    if not patches:
        raise ValueError("No patch gradients survived the defense.")
    return torch.stack(patches, dim=0)


def _soteria_like_patches(patches: torch.Tensor, pruning_rate: float) -> torch.Tensor:
    flat = patches.reshape(-1, patches.shape[-1]).clone()
    k = int(round(flat.shape[-1] * pruning_rate / 100.0))
    if k <= 0:
        return patches
    scores = flat.abs().mean(dim=0)
    top = torch.topk(scores, k=min(k, scores.numel()), largest=True).indices
    flat[:, top] = 0.0
    return flat.view_as(patches)


def _mixup_patches(patches: torch.Tensor, alpha: float) -> torch.Tensor:
    if patches.shape[0] < 2:
        return patches
    lam = float(torch.distributions.Beta(float(alpha), float(alpha)).sample().item())
    perm = torch.arange(patches.shape[0] - 1, -1, -1, device=patches.device)
    return lam * patches + (1.0 - lam) * patches[perm]


def _resolve_runtime_device(requested_device: str) -> torch.device:
    resolved = resolve_cuda_device(requested_device)
    if str(resolved).startswith("cuda") and not torch.cuda.is_available():
        print("[peftleak-image] CUDA unavailable; falling back to CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(resolved)


def _defend_gradient_tuple(args, raw_grads, names):
    if args.defense == "dpsgd":
        # For precomputed synthetic gradients, each tensor pair belongs to a single patch.
        return dpsgd_defense([tuple(raw_grads)], args.defense_clip_norm, args.defense_noise or 0.0, seed=args.rng_seed), names
    if args.defense == "none":
        if getattr(args, "defense_noise", None) is not None:
            return noise_injection(raw_grads, float(args.defense_noise), seed=args.rng_seed), names
        return raw_grads, names
    if args.defense == "noise":
        return noise_injection(raw_grads, float(args.defense_noise or 0.0), seed=args.rng_seed), names
    if args.defense == "topk":
        return topk_sparsification(raw_grads, float(args.defense_topk_ratio)), names
    if args.defense == "compression":
        return gradient_compression(raw_grads, int(args.defense_n_bits), seed=args.rng_seed), names
    if args.defense in {"lrb", "lrbprojonly", "signed_bottleneck"}:
        return apply_lrb_defense(raw_grads, args, layer_names=names), names
    if args.defense in {"soteria", "mixup"}:
        return raw_grads, names
    raise ValueError(f"Unsupported image defense: {args.defense!r}")


def _defended_images(args, images: torch.Tensor) -> torch.Tensor:
    patches = extract_image_patches(images, args.patch_size)
    if args.defense == "soteria":
        patches = _soteria_like_patches(patches, float(args.defense_soteria_pruning_rate))
    elif args.defense == "mixup":
        patches = _mixup_patches(patches, float(args.defense_mixup_alpha))
    else:
        return images
    grid = (images.shape[2] // args.patch_size, images.shape[3] // args.patch_size)
    return fold_image_patches(patches, channels=images.shape[1], grid_shape=grid, patch_size=args.patch_size).clamp(0.0, 1.0)


def _public_stats(args, public_images: torch.Tensor | None):
    if args.public_stats_path:
        return load_public_patch_statistics(args.public_stats_path), f"file:{args.public_stats_path}"
    if public_images is None:
        raise ValueError("public_images are required when --public_stats_path is not provided.")
    stats = build_public_patch_statistics(public_images, args.patch_size)
    if args.save_public_stats_path:
        save_public_patch_statistics(stats, args.save_public_stats_path)
    return stats, "public_split"


def _probe_stats(args, images: torch.Tensor, public_images: torch.Tensor | None):
    n_patches = extract_image_patches(images, args.patch_size).shape[1]
    if args.public_stats_path:
        patch_stats = load_public_patch_statistics(args.public_stats_path)
        probe_stats = build_peftleak_probe_statistics_from_patch_stats(
            patch_stats,
            n_patches=n_patches,
            num_bins=args.peftleak_num_bins,
            position_sigma=args.peftleak_position_sigma,
            embed_scale=args.peftleak_embed_scale,
            gap=args.peftleak_gap,
            seed=args.rng_seed,
            device=images.device,
            dtype=images.dtype,
        )
        return probe_stats, f"file:{args.public_stats_path}"
    if public_images is None:
        raise ValueError("public_images are required when --public_stats_path is not provided.")
    builder = build_official_aligned_probe_statistics if args.mode == "official_vit_adapter" else build_peftleak_probe_statistics
    probe_stats = builder(
        public_images,
        args.patch_size,
        num_bins=args.peftleak_num_bins,
        position_sigma=args.peftleak_position_sigma,
        embed_scale=args.peftleak_embed_scale,
        gap=args.peftleak_gap,
        seed=args.rng_seed,
    )
    if args.save_public_stats_path:
        save_public_patch_statistics(probe_stats.patch_stats, args.save_public_stats_path)
    return probe_stats, "public_split"


def _run_synthetic_ratio(args, images: torch.Tensor):
    patches = extract_image_patches(images, args.patch_size)
    if args.defense == "soteria":
        patches_for_grads = _soteria_like_patches(patches, float(args.defense_soteria_pruning_rate))
        raw_grads, names = _make_raw_gradient_tuple(patches_for_grads, args.adapter_hidden_dim)
    elif args.defense == "mixup":
        patches_for_grads = _mixup_patches(patches, float(args.defense_mixup_alpha))
        raw_grads, names = _make_raw_gradient_tuple(patches_for_grads, args.adapter_hidden_dim)
    else:
        raw_grads, names = _make_raw_gradient_tuple(patches, args.adapter_hidden_dim)
    defended_grads, _names = _defend_gradient_tuple(args, raw_grads, names)
    recovered_flat = _recover_from_gradient_tuple(defended_grads)
    return recovered_flat.view(patches.shape), patches, None, None, len(names)


def _run_vit_adapter(args, images: torch.Tensor, labels: torch.Tensor, probe_stats):
    reference_patches = extract_image_patches(images, args.patch_size)
    grad_images = _defended_images(args, images)
    probe_stats = move_peftleak_probe_statistics(probe_stats, device=images.device, dtype=images.dtype)
    if args.defense == "dpsgd":
        full_result, per_example_results = build_shared_adapter_gradient_bundle(
            grad_images,
            probe_stats,
            labels=labels,
            patch_size=args.patch_size,
            adapter_hidden_dim=args.adapter_hidden_dim,
            n_classes=args.n_classes,
            seed=args.rng_seed,
            model_path=args.model_path,
            finetuned_path=args.finetuned_path,
        )
        per_example = [tuple(sample_result.grads) for sample_result in per_example_results]
        defended_grads = dpsgd_defense(per_example, args.defense_clip_norm, args.defense_noise or 0.0, seed=args.rng_seed)
        names = full_result.names
        loss = full_result.loss
        logits = full_result.logits
        oracle_slot_indices = full_result.slot_indices
    else:
        result = build_shared_adapter_gradients(
            grad_images,
            probe_stats,
            labels=labels,
            patch_size=args.patch_size,
            adapter_hidden_dim=args.adapter_hidden_dim,
            n_classes=args.n_classes,
            seed=args.rng_seed,
            model_path=args.model_path,
            finetuned_path=args.finetuned_path,
        )
        defended_grads, names = _defend_gradient_tuple(args, result.grads, result.names)
        loss = result.loss
        logits = result.logits
        oracle_slot_indices = result.slot_indices
    recovery = recover_patches_from_shared_adapter_grads(
        defended_grads,
        names,
        probe_stats,
        batch_size=images.shape[0],
        n_patches=reference_patches.shape[1],
        slot_indices=oracle_slot_indices,
    )
    batch_top1_acc = float((logits.argmax(dim=-1) == labels.to(device=logits.device)).float().mean().item())
    return recovery.recovered_patches, reference_patches, loss, batch_top1_acc, len(names), recovery


def _run_official_vit_adapter(args, images: torch.Tensor, labels: torch.Tensor, probe_stats):
    reference_patches = extract_image_patches(images, args.patch_size)
    if args.defense in {"soteria", "mixup"}:
        grad_images = _defended_images(args, images)
    else:
        grad_images = images
    rounds = max(1, int(getattr(args, "attack_rounds", 1)))
    recoveries = []
    losses = []
    accs = []
    grad_count = 0
    result = None
    for round_idx in range(rounds):
        round_seed = int(args.rng_seed) + round_idx
        result = build_official_aligned_adapter_gradients(
            grad_images,
            probe_stats,
            labels=labels,
            patch_size=args.patch_size,
            n_classes=args.n_classes,
            seed=round_seed,
            model_path="vit_b_16" if args.vit_config == "vit_b_16" else args.model_path,
            finetuned_path=args.finetuned_path,
            vit_config=args.vit_config,
            adapter_layers=args.adapter_layers,
            adapter_bottleneck_dim=args.adapter_bottleneck_dim,
        )
        round_args = argparse.Namespace(**vars(args))
        round_args.rng_seed = round_seed
        defended_grads, names = _defend_gradient_tuple(round_args, result.grads, result.names)
        recovery = recover_patches_from_official_adapter_grads(
            defended_grads,
            names,
            probe_stats,
            batch_size=images.shape[0],
            n_patches=reference_patches.shape[1],
            grouping=args.official_grouping,
            slot_indices=result.slot_indices if args.official_grouping == "oracle_debug" else None,
        )
        recoveries.append(recovery)
        losses.append(float(result.loss))
        accs.append(float((result.logits.argmax(dim=-1) == labels.to(device=result.logits.device)).float().mean().item()))
        grad_count += len(names)
    if result is None:
        raise ValueError("official_vit_adapter produced no attack rounds.")
    recovery = _merge_same_batch_recoveries(
        recoveries,
        batch_size=int(images.shape[0]),
        n_patches=int(reference_patches.shape[1]),
        patch_dim=int(reference_patches.shape[2]),
    )
    batch_top1_acc = sum(accs) / max(1, len(accs))
    loss = sum(losses) / max(1, len(losses))
    return (
        recovery.recovered_patches,
        reference_patches,
        loss,
        batch_top1_acc,
        grad_count,
        recovery,
        {
            "adapter_layer_count": result.adapter_layer_count,
            "adapter_bottleneck_dim": result.adapter_bottleneck_dim,
        },
    )


def _requested_metrics(args) -> set[str]:
    metrics = {item.strip().lower() for item in str(getattr(args, "metrics", "")).split(",") if item.strip()}
    aliases = {"patch_recovery_rate": "patch_recovery", "patch-recovery": "patch_recovery"}
    return {aliases.get(metric, metric) for metric in metrics}


def _as_lpips_rgb(images: torch.Tensor) -> torch.Tensor:
    x = images.detach().float().clamp(0.0, 1.0)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    elif x.shape[1] > 3:
        x = x[:, :3]
    return x.mul(2.0).sub(1.0)


def _compute_optional_lpips(recovered: torch.Tensor | None, reference: torch.Tensor, requested: bool) -> tuple[float | None, str]:
    if not requested:
        return None, "not_requested"
    if recovered is None:
        return None, "unavailable"
    try:
        import lpips
    except Exception:
        return None, "unavailable"
    try:
        model = lpips.LPIPS(net="alex")
        model = model.to(device=reference.device)
        model.eval()
        with torch.no_grad():
            score = model(_as_lpips_rgb(recovered).to(reference.device), _as_lpips_rgb(reference))
        return float(score.detach().float().mean().item()), "ok"
    except Exception as exc:
        return None, f"failed:{type(exc).__name__}"


def _mean_or_none(values: list[float], weights: list[int]) -> float | None:
    if not values:
        return None
    total_weight = sum(weights[: len(values)])
    if total_weight <= 0:
        return sum(values) / len(values)
    return sum(value * weight for value, weight in zip(values, weights)) / total_weight


def _merge_metadata(metadata_items: list[dict]) -> dict:
    merged: dict[str, int] = {}
    first_value_keys = {
        "n_slots",
        "row_stride",
        "official_layer_count",
        "official_grouping_is_cluster",
        "official_grouping_is_oracle_debug",
    }
    for metadata in metadata_items:
        for key, value in metadata.items():
            if not isinstance(value, int):
                continue
            if key in first_value_keys:
                merged.setdefault(key, int(value))
            else:
                merged[key] = int(merged.get(key, 0)) + int(value)
    return merged


def _merge_same_batch_recoveries(
    recoveries: list[PeftLeakRecoveryResult],
    *,
    batch_size: int,
    n_patches: int,
    patch_dim: int,
) -> PeftLeakRecoveryResult:
    if not recoveries:
        raise ValueError("Cannot merge an empty official recovery list.")
    device = recoveries[0].recovered_patches.device
    dtype = recoveries[0].recovered_patches.dtype
    recovered_sum = torch.zeros(batch_size, n_patches, patch_dim, device=device, dtype=dtype)
    recovered_count = torch.zeros(batch_size, n_patches, device=device, dtype=torch.long)
    oracle_sum = torch.zeros(batch_size, n_patches, patch_dim, device=device, dtype=dtype)
    oracle_count = torch.zeros(batch_size, n_patches, device=device, dtype=torch.long)
    candidate_patches = []
    candidate_slots = []
    candidate_positions = []
    slot_counts = []
    slot_indices = None
    collision_values = []
    oracle_collision_values = []
    for recovery in recoveries:
        if recovery.recovered_patches.shape == recovered_sum.shape and recovery.recovery_mask.shape == recovered_count.shape:
            mask = recovery.recovery_mask
            recovered_sum[mask] += recovery.recovered_patches[mask]
            recovered_count += mask.to(dtype=torch.long)
        if recovery.oracle_recovered_patches is not None and recovery.oracle_recovery_mask is not None:
            if recovery.oracle_recovered_patches.shape == oracle_sum.shape and recovery.oracle_recovery_mask.shape == oracle_count.shape:
                oracle_mask = recovery.oracle_recovery_mask
                oracle_sum[oracle_mask] += recovery.oracle_recovered_patches[oracle_mask]
                oracle_count += oracle_mask.to(dtype=torch.long)
        if recovery.candidate_patches.numel() > 0:
            candidate_patches.append(recovery.candidate_patches)
            candidate_slots.append(recovery.candidate_slots)
            candidate_positions.append(recovery.candidate_position_indices)
        if recovery.slot_counts is not None:
            slot_counts.append(recovery.slot_counts)
        if slot_indices is None and recovery.slot_indices is not None:
            slot_indices = recovery.slot_indices
        if recovery.collision_patch_count is not None:
            collision_values.append(int(recovery.collision_patch_count))
        if recovery.oracle_collision_patch_count is not None:
            oracle_collision_values.append(int(recovery.oracle_collision_patch_count))

    mask = recovered_count > 0
    recovered = torch.zeros_like(recovered_sum)
    recovered[mask] = recovered_sum[mask] / recovered_count[mask].to(dtype=dtype).unsqueeze(-1)
    oracle_mask = oracle_count > 0
    has_oracle = bool(oracle_mask.any().item())
    oracle_recovered = None
    if has_oracle:
        oracle_recovered = torch.zeros_like(oracle_sum)
        oracle_recovered[oracle_mask] = oracle_sum[oracle_mask] / oracle_count[oracle_mask].to(dtype=dtype).unsqueeze(-1)

    empty_candidates = torch.empty(0, patch_dim, device=device, dtype=dtype)
    empty_indices = torch.empty(0, device=device, dtype=torch.long)
    return PeftLeakRecoveryResult(
        recovered_patches=recovered,
        recovery_mask=mask,
        candidate_patch_count=sum(item.candidate_patch_count for item in recoveries),
        recovered_patch_count=int(mask.sum().item()),
        collision_patch_count=sum(collision_values) if collision_values else None,
        unresolved_patch_count=int(batch_size * n_patches - int(mask.sum().item())),
        candidate_patches=torch.cat(candidate_patches, dim=0) if candidate_patches else empty_candidates,
        candidate_slots=torch.cat(candidate_slots, dim=0) if candidate_slots else empty_indices,
        candidate_position_indices=torch.cat(candidate_positions, dim=0) if candidate_positions else empty_indices,
        slot_indices=slot_indices,
        slot_counts=torch.stack(slot_counts, dim=0).sum(dim=0) if slot_counts else None,
        oracle_recovered_patches=oracle_recovered,
        oracle_recovery_mask=oracle_mask if has_oracle else None,
        oracle_recovered_patch_count=int(oracle_mask.sum().item()) if has_oracle else None,
        oracle_collision_patch_count=sum(oracle_collision_values) if oracle_collision_values else None,
        oracle_unresolved_patch_count=int(batch_size * n_patches - int(oracle_mask.sum().item())) if has_oracle else None,
        raw_candidate_metadata=_merge_metadata([item.raw_candidate_metadata for item in recoveries]),
    )


def _merge_disjoint_batch_recoveries(
    recoveries: list[tuple[PeftLeakRecoveryResult, int]],
    *,
    total_images: int,
    n_patches: int,
    patch_dim: int,
) -> PeftLeakRecoveryResult | None:
    if not recoveries:
        return None
    first = recoveries[0][0]
    device = first.recovered_patches.device
    dtype = first.recovered_patches.dtype
    recovered = torch.zeros(total_images, n_patches, patch_dim, device=device, dtype=dtype)
    mask = torch.zeros(total_images, n_patches, device=device, dtype=torch.bool)
    oracle_recovered = torch.zeros_like(recovered)
    oracle_mask = torch.zeros_like(mask)
    candidate_patches = []
    candidate_slots = []
    candidate_positions = []
    slot_counts = []
    slot_indices = []
    collision_values = []
    oracle_collision_values = []
    offset = 0
    for recovery, batch_size in recoveries:
        end = offset + int(batch_size)
        if recovery.recovered_patches.shape == (int(batch_size), n_patches, patch_dim):
            recovered[offset:end] = recovery.recovered_patches
            mask[offset:end] = recovery.recovery_mask
        if recovery.oracle_recovered_patches is not None and recovery.oracle_recovery_mask is not None:
            if recovery.oracle_recovered_patches.shape == (int(batch_size), n_patches, patch_dim):
                oracle_recovered[offset:end] = recovery.oracle_recovered_patches
                oracle_mask[offset:end] = recovery.oracle_recovery_mask
        if recovery.candidate_patches.numel() > 0:
            candidate_patches.append(recovery.candidate_patches)
            candidate_slots.append(recovery.candidate_slots)
            candidate_positions.append(recovery.candidate_position_indices)
        if recovery.slot_counts is not None:
            slot_counts.append(recovery.slot_counts)
        if recovery.slot_indices is not None and recovery.slot_indices.shape == (int(batch_size), n_patches):
            slot_indices.append(recovery.slot_indices)
        if recovery.collision_patch_count is not None:
            collision_values.append(int(recovery.collision_patch_count))
        if recovery.oracle_collision_patch_count is not None:
            oracle_collision_values.append(int(recovery.oracle_collision_patch_count))
        offset = end

    empty_candidates = torch.empty(0, patch_dim, device=device, dtype=dtype)
    empty_indices = torch.empty(0, device=device, dtype=torch.long)
    has_oracle = bool(oracle_mask.any().item())
    return PeftLeakRecoveryResult(
        recovered_patches=recovered,
        recovery_mask=mask,
        candidate_patch_count=sum(item.candidate_patch_count for item, _batch_size in recoveries),
        recovered_patch_count=int(mask.sum().item()),
        collision_patch_count=sum(collision_values) if collision_values else None,
        unresolved_patch_count=int(total_images * n_patches - int(mask.sum().item())),
        candidate_patches=torch.cat(candidate_patches, dim=0) if candidate_patches else empty_candidates,
        candidate_slots=torch.cat(candidate_slots, dim=0) if candidate_slots else empty_indices,
        candidate_position_indices=torch.cat(candidate_positions, dim=0) if candidate_positions else empty_indices,
        slot_indices=torch.cat(slot_indices, dim=0) if slot_indices else None,
        slot_counts=torch.stack(slot_counts, dim=0).sum(dim=0) if slot_counts else None,
        oracle_recovered_patches=oracle_recovered if has_oracle else None,
        oracle_recovery_mask=oracle_mask if has_oracle else None,
        oracle_recovered_patch_count=int(oracle_mask.sum().item()) if has_oracle else None,
        oracle_collision_patch_count=sum(oracle_collision_values) if oracle_collision_values else None,
        oracle_unresolved_patch_count=int(total_images * n_patches - int(oracle_mask.sum().item())) if has_oracle else None,
        raw_candidate_metadata=_merge_metadata([item.raw_candidate_metadata for item, _batch_size in recoveries]),
    )


def _candidate_batches_from_recovery(
    recovery: PeftLeakRecoveryResult,
    *,
    batch_size: int,
    n_patches: int,
    patch_dim: int,
) -> torch.Tensor | None:
    if recovery.candidate_patches.numel() == 0:
        return None
    device = recovery.candidate_patches.device
    dtype = recovery.candidate_patches.dtype
    batched = torch.zeros(batch_size, n_patches, patch_dim, device=device, dtype=dtype)
    for patch_idx in range(n_patches):
        position_mask = recovery.candidate_position_indices == int(patch_idx)
        if not bool(position_mask.any().item()):
            return None
        values = recovery.candidate_patches[position_mask]
        if batch_size == 1:
            batched[0, patch_idx] = values.mean(dim=0)
            continue
        if values.shape[0] < batch_size:
            return None
        feature = torch.stack((values.float().mean(dim=1), values.float().std(dim=1)), dim=1)
        ordering = sorted(range(values.shape[0]), key=lambda idx: (float(feature[idx, 0]), float(feature[idx, 1]), idx))
        ordered = values[torch.tensor(ordering, device=device)]
        chunks = torch.chunk(ordered, batch_size, dim=0)
        if len(chunks) != batch_size or any(chunk.shape[0] == 0 for chunk in chunks):
            return None
        for sample_idx, chunk in enumerate(chunks):
            batched[sample_idx, patch_idx] = chunk.mean(dim=0)
    return batched


def _run_attack_batches(args, images: torch.Tensor, labels: torch.Tensor, probe_stats):
    batch_size = max(1, int(args.batch_size))
    recovered_parts = []
    reference_parts = []
    recovery_parts: list[tuple[PeftLeakRecoveryResult, int]] = []
    losses = []
    accs = []
    weights = []
    grad_count = 0
    attack_variant = "vit_adapter_shared_bins"
    official_fields = {}
    for start_idx in range(0, int(images.shape[0]), batch_size):
        end_idx = min(int(images.shape[0]), start_idx + batch_size)
        batch_images = images[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        current_batch_size = int(batch_images.shape[0])
        weights.append(current_batch_size)
        if args.mode == "synthetic_ratio":
            recovered, reference_patches, vit_loss, batch_top1_acc, batch_grad_count = _run_synthetic_ratio(args, batch_images)
            attack_variant = "synthetic_ratio"
            recovered_parts.append(recovered)
        elif args.mode == "official_vit_adapter":
            recovered, reference_patches, vit_loss, batch_top1_acc, batch_grad_count, recovery, official_fields = _run_official_vit_adapter(
                args,
                batch_images,
                batch_labels,
                probe_stats,
            )
            attack_variant = "official_vit_adapter"
            recovery_parts.append((recovery, current_batch_size))
        else:
            recovered, reference_patches, vit_loss, batch_top1_acc, batch_grad_count, recovery = _run_vit_adapter(
                args,
                batch_images,
                batch_labels,
                probe_stats,
            )
            attack_variant = "vit_adapter_shared_bins"
            recovery_parts.append((recovery, current_batch_size))
        reference_parts.append(reference_patches)
        if vit_loss is not None:
            losses.append(float(vit_loss))
        if batch_top1_acc is not None:
            accs.append(float(batch_top1_acc))
        grad_count += int(batch_grad_count)

    reference_patches = torch.cat(reference_parts, dim=0)
    if recovery_parts:
        recovery = _merge_disjoint_batch_recoveries(
            recovery_parts,
            total_images=int(images.shape[0]),
            n_patches=int(reference_patches.shape[1]),
            patch_dim=int(reference_patches.shape[2]),
        )
        if recovery is None:
            raise ValueError("Expected a merged recovery result for adapter attack mode.")
        recovered = recovery.recovered_patches
    else:
        recovery = None
        recovered = torch.cat(recovered_parts, dim=0)
    return (
        recovered,
        reference_patches,
        _mean_or_none(losses, weights),
        _mean_or_none(accs, weights),
        grad_count,
        recovery,
        official_fields,
        attack_variant,
        len(weights),
        min(batch_size, int(images.shape[0])),
    )


def _emit_summary(args, fields: dict):
    defense_param_name, defense_param_value = defense_param_spec(args)
    variant = fields.get("attack_variant", args.mode)
    default_reproduction_level = (
        PEFTLEAK_IMAGE_REPRODUCTION_LEVEL
        if variant in {"vit_adapter", "vit_adapter_shared_bins"}
        else PEFTLEAK_IMAGE_OFFICIAL_REPRODUCTION_LEVEL
        if variant == "official_vit_adapter"
        else SYNTHETIC_RATIO_REPRODUCTION_LEVEL
    )
    default_oracle_scope = "debug_only" if variant in {"vit_adapter", "vit_adapter_shared_bins", "official_vit_adapter"} else "n/a"
    ordered = [
        ("summary_version", 2),
        ("result_status", fields.get("result_status", "ok")),
        ("attack", "peftleak_image_repro"),
        ("attack_variant", variant),
        ("reproduction_level", fields.get("reproduction_level", default_reproduction_level)),
        ("peftleak_profile", getattr(args, "peftleak_profile", None)),
        ("official_like_config", getattr(args, "official_like_config", None)),
        ("profile_override_count", getattr(args, "profile_override_count", None)),
        ("config_warning", fields.get("config_warning", getattr(args, "config_warning", None))),
        ("oracle_metric_scope", fields.get("oracle_metric_scope", default_oracle_scope)),
        ("non_oracle_primary_only", fields.get("non_oracle_primary_only", True)),
        ("dataset", args.dataset),
        ("rng_seed", getattr(args, "rng_seed", None)),
        ("device", fields.get("device")),
        ("data_source", fields.get("data_source")),
        ("public_stats_source", fields.get("public_stats_source")),
        ("synthetic_fallback", fields.get("synthetic_fallback")),
        ("fallback_reason", fields.get("fallback_reason")),
        ("model_path", args.model_path),
        ("finetuned_path", args.finetuned_path or "n/a"),
        ("vit_config", getattr(args, "vit_config", None)),
        ("official_alignment_version", fields.get("official_alignment_version")),
        ("adapter_layer_count", fields.get("adapter_layer_count")),
        ("adapter_bottleneck_dim", fields.get("adapter_bottleneck_dim")),
        ("adapter_layers", getattr(args, "adapter_layers", None)),
        ("attack_rounds", getattr(args, "attack_rounds", None)),
        ("non_oracle_grouping", fields.get("non_oracle_grouping")),
        ("batch_size", args.batch_size),
        ("effective_batch_size", fields.get("effective_batch_size")),
        ("attack_batch_count", fields.get("attack_batch_count")),
        ("n_images", args.n_images),
        ("patch_size", args.patch_size),
        ("adapter_hidden_dim", args.adapter_hidden_dim),
        ("peftleak_num_bins", getattr(args, "peftleak_num_bins", None)),
        ("peftleak_position_sigma", getattr(args, "peftleak_position_sigma", None)),
        ("peftleak_embed_scale", getattr(args, "peftleak_embed_scale", None)),
        ("peftleak_gap", getattr(args, "peftleak_gap", None)),
        ("adapter_gradient_count", fields.get("adapter_gradient_count")),
        ("vit_adapter_loss", fields.get("vit_adapter_loss")),
        ("defense", args.defense),
        ("defense_param_name", defense_param_name),
        ("defense_param_value", defense_param_value),
        ("mse", fields.get("mse")),
        ("psnr", fields.get("psnr")),
        ("ssim", fields.get("ssim")),
        ("lpips", fields.get("lpips")),
        ("lpips_status", fields.get("lpips_status")),
        ("primary_metric_source", fields.get("primary_metric_source")),
        ("cluster_method", fields.get("cluster_method")),
        ("direct_mse", fields.get("direct_mse")),
        ("direct_psnr", fields.get("direct_psnr")),
        ("clustered_mse", fields.get("clustered_mse")),
        ("clustered_psnr", fields.get("clustered_psnr")),
        ("oracle_direct_mse", fields.get("oracle_direct_mse")),
        ("oracle_direct_psnr", fields.get("oracle_direct_psnr")),
        ("candidate_patch_count", fields.get("candidate_patch_count")),
        ("nonzero_slot_count", fields.get("nonzero_slot_count")),
        ("ambiguous_position_count", fields.get("ambiguous_position_count")),
        ("empty_position_count", fields.get("empty_position_count")),
        ("recovered_patch_count", fields.get("recovered_patch_count")),
        ("collision_patch_count", fields.get("collision_patch_count")),
        ("unresolved_patch_count", fields.get("unresolved_patch_count")),
        ("patch_recovery_count", fields.get("patch_recovery_count")),
        ("patch_recovery_rate", fields.get("patch_recovery_rate")),
        ("oracle_patch_recovery_count", fields.get("oracle_patch_recovery_count")),
        ("oracle_patch_recovery_rate", fields.get("oracle_patch_recovery_rate")),
        ("oracle_recovered_patch_count", fields.get("oracle_recovered_patch_count")),
        ("oracle_collision_patch_count", fields.get("oracle_collision_patch_count")),
        ("oracle_unresolved_patch_count", fields.get("oracle_unresolved_patch_count")),
        ("patch_count", fields.get("patch_count")),
        ("batch_top1_acc", fields.get("batch_top1_acc")),
        ("public_patch_count", fields.get("public_patch_count")),
        ("attack_runtime", fields.get("attack_runtime")),
        ("runtime", fields.get("runtime")),
    ]
    if fields.get("error_type"):
        ordered.append(("error_type", fields["error_type"]))
    if fields.get("error_message"):
        ordered.append(("error_message", fields["error_message"]))
    print(SUMMARY_START, flush=True)
    for key, value in ordered:
        print(f"{key}={fmt_summary_value(value)}", flush=True)
    print(SUMMARY_END, flush=True)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    explicit_fields = _explicit_cli_fields(argv)
    _apply_peftleak_profile(args, explicit_fields)
    apply_lrb_preset(args)
    device = _resolve_runtime_device(args.device)
    args.device = str(device)
    start = time.time()
    if args.defense == "dager":
        _emit_summary(
            args,
            {
                "result_status": "unsupported",
                "device": args.device,
                "runtime": str(datetime.timedelta(seconds=time.time() - start)).split(".")[0],
                "error_type": "unsupported_defense",
                "error_message": "DAGER defense is DAGER-specific and is excluded from PEFTLeak image matrices.",
            },
        )
        return 0
    if args.defense not in SUPPORTED_IMAGE_DEFENSES:
        raise NotImplementedError(f"Unsupported image PEFTLeak defense: {args.defense!r}")

    try:
        fallback_reasons = []
        images, labels, data_source, data_fallback_reason = _load_images(args, public=False)
        if data_fallback_reason:
            fallback_reasons.append(f"attack:{data_fallback_reason}")
        images = images.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.long)
        if args.public_stats_path:
            public_data_source = "file"
            public_images = None
        else:
            public_images, _public_labels, public_data_source, public_fallback_reason = _load_images(args, public=True)
            if public_fallback_reason:
                fallback_reasons.append(f"public:{public_fallback_reason}")
            public_images = public_images.to(device=device, dtype=images.dtype)
        probe_stats, stats_source = _probe_stats(args, images, public_images)
        probe_stats = move_peftleak_probe_statistics(probe_stats, device=device, dtype=images.dtype)
        (
            recovered,
            reference_patches,
            vit_loss,
            batch_top1_acc,
            grad_count,
            recovery,
            official_fields,
            attack_variant,
            attack_batch_count,
            effective_batch_size,
        ) = _run_attack_batches(args, images, labels, probe_stats)
        requested_metrics = _requested_metrics(args)
        cluster_method = resolve_cluster_method("auto")
        grid_shape = (images.shape[2] // args.patch_size, images.shape[3] // args.patch_size)
        folded_reference = fold_image_patches(
            reference_patches,
            channels=images.shape[1],
            grid_shape=grid_shape,
            patch_size=args.patch_size,
        )
        patch_count = int(reference_patches.numel() // reference_patches.shape[-1])
        direct_mse = direct_psnr = None
        clustered_mse = clustered_psnr = None
        assembled_direct = assembled_clustered = None
        oracle_direct_mse = oracle_direct_psnr = None
        exact_count = None
        patch_recovery_rate = None
        oracle_exact_count = None
        oracle_patch_recovery_rate = None
        oracle_recovered_patch_count = None
        oracle_collision_patch_count = None
        oracle_unresolved_patch_count = None
        recovery_metadata = {}
        if recovery is not None:
            recovery_metadata = recovery.raw_candidate_metadata
            candidate_patch_count = recovery.candidate_patch_count
            recovered_patch_count = recovery.recovered_patch_count
            collision_patch_count = recovery.collision_patch_count
            unresolved_patch_count = recovery.unresolved_patch_count
            if recovered.shape == reference_patches.shape and bool(recovery.recovery_mask.all().item()):
                assembled_direct = fold_image_patches(
                    recovered,
                    channels=images.shape[1],
                    grid_shape=grid_shape,
                    patch_size=args.patch_size,
                )
                direct_mse, direct_psnr = mse_psnr(assembled_direct, folded_reference)
                patch_mse = (recovered - reference_patches).float().pow(2).mean(dim=-1)
                exact_mask = recovery.recovery_mask & (patch_mse < args.patch_recovery_mse_threshold)
                exact_count = int(exact_mask.sum().item())
                patch_recovery_rate = exact_count / max(1, patch_count)
            elif recovered.shape == reference_patches.shape and bool(recovery.recovery_mask.any().item()):
                patch_mse = (recovered - reference_patches).float().pow(2).mean(dim=-1)
                exact_mask = recovery.recovery_mask & (patch_mse < args.patch_recovery_mse_threshold)
                exact_count = int(exact_mask.sum().item())
                patch_recovery_rate = exact_count / max(1, patch_count)
            candidate_batches = None
            if recovery.candidate_patch_count >= patch_count and patch_count > 0:
                candidate_batches = _candidate_batches_from_recovery(
                    recovery,
                    batch_size=int(images.shape[0]),
                    n_patches=int(reference_patches.shape[1]),
                    patch_dim=int(reference_patches.shape[2]),
                )
            if candidate_batches is not None:
                assembled_clustered, _assignments = cluster_and_reassemble(
                    candidate_batches,
                    channels=images.shape[1],
                    grid_shape=grid_shape,
                    patch_size=args.patch_size,
                    seed=args.rng_seed,
                    method=cluster_method,
                )
                clustered_mse, clustered_psnr = mse_psnr(assembled_clustered, folded_reference)
            if recovery.oracle_recovered_patches is not None and recovery.oracle_recovery_mask is not None:
                oracle_recovered_patch_count = recovery.oracle_recovered_patch_count
                oracle_collision_patch_count = recovery.oracle_collision_patch_count
                oracle_unresolved_patch_count = recovery.oracle_unresolved_patch_count
                oracle_patch_mse = (recovery.oracle_recovered_patches - reference_patches).float().pow(2).mean(dim=-1)
                oracle_exact_mask = recovery.oracle_recovery_mask & (oracle_patch_mse < args.patch_recovery_mse_threshold)
                oracle_exact_count = int(oracle_exact_mask.sum().item())
                oracle_patch_recovery_rate = oracle_exact_count / max(1, patch_count)
                if bool(recovery.oracle_recovery_mask.all().item()):
                    assembled_oracle = fold_image_patches(
                        recovery.oracle_recovered_patches,
                        channels=images.shape[1],
                        grid_shape=grid_shape,
                        patch_size=args.patch_size,
                    )
                    oracle_direct_mse, oracle_direct_psnr = mse_psnr(assembled_oracle, folded_reference)
        else:
            assembled_direct = fold_image_patches(
                recovered,
                channels=images.shape[1],
                grid_shape=grid_shape,
                patch_size=args.patch_size,
            )
            assembled_clustered, _assignments = cluster_and_reassemble(
                recovered,
                channels=images.shape[1],
                grid_shape=grid_shape,
                patch_size=args.patch_size,
                seed=args.rng_seed,
                method=cluster_method,
            )
            direct_mse, direct_psnr = mse_psnr(assembled_direct, folded_reference)
            clustered_mse, clustered_psnr = mse_psnr(assembled_clustered, folded_reference)
            patch_mse = (recovered - reference_patches).float().pow(2).mean(dim=-1)
            exact_count = int((patch_mse < args.patch_recovery_mse_threshold).sum().item())
            patch_recovery_rate = exact_count / max(1, patch_count)
            candidate_patch_count = patch_count
            recovered_patch_count = patch_count
            collision_patch_count = 0
            unresolved_patch_count = 0
        report_mse = (direct_mse if direct_mse is not None else clustered_mse) if "mse" in requested_metrics else None
        report_psnr = (direct_psnr if direct_psnr is not None else clustered_psnr) if "psnr" in requested_metrics else None
        report_image = assembled_direct if direct_mse is not None else assembled_clustered
        report_ssim = simple_ssim(report_image, folded_reference) if report_image is not None and "ssim" in requested_metrics else None
        report_lpips, lpips_status = _compute_optional_lpips(report_image, folded_reference, "lpips" in requested_metrics)
        if direct_mse is not None:
            primary_metric_source = "direct"
        elif clustered_mse is not None:
            primary_metric_source = "clustered"
        else:
            primary_metric_source = "n/a"
        synthetic_fallback = int(
            args.dataset != "synthetic"
            and (str(data_source).startswith("synthetic") or str(public_data_source).startswith("synthetic"))
        )
        fallback_reason = "; ".join(fallback_reasons) if fallback_reasons else None
        _emit_summary(
            args,
            {
                "device": args.device,
                "attack_variant": attack_variant,
                "reproduction_level": (
                    PEFTLEAK_IMAGE_REPRODUCTION_LEVEL
                    if attack_variant == "vit_adapter_shared_bins"
                    else PEFTLEAK_IMAGE_OFFICIAL_REPRODUCTION_LEVEL
                    if attack_variant == "official_vit_adapter"
                    else SYNTHETIC_RATIO_REPRODUCTION_LEVEL
                ),
                "oracle_metric_scope": "debug_only" if attack_variant in {"vit_adapter_shared_bins", "official_vit_adapter"} else "n/a",
                "official_alignment_version": "v1" if attack_variant == "official_vit_adapter" else None,
                "adapter_layer_count": official_fields.get("adapter_layer_count"),
                "adapter_bottleneck_dim": official_fields.get("adapter_bottleneck_dim"),
                "non_oracle_grouping": args.official_grouping if attack_variant == "official_vit_adapter" else None,
                "mse": report_mse,
                "psnr": report_psnr,
                "ssim": report_ssim,
                "lpips": report_lpips,
                "lpips_status": lpips_status,
                "primary_metric_source": primary_metric_source,
                "cluster_method": cluster_method,
                "direct_mse": direct_mse,
                "direct_psnr": direct_psnr,
                "clustered_mse": clustered_mse,
                "clustered_psnr": clustered_psnr,
                "oracle_direct_mse": oracle_direct_mse,
                "oracle_direct_psnr": oracle_direct_psnr,
                "candidate_patch_count": candidate_patch_count,
                "nonzero_slot_count": recovery_metadata.get("nonzero_slot_count"),
                "ambiguous_position_count": recovery_metadata.get("ambiguous_position_count"),
                "empty_position_count": recovery_metadata.get("empty_position_count"),
                "recovered_patch_count": recovered_patch_count,
                "collision_patch_count": collision_patch_count,
                "unresolved_patch_count": unresolved_patch_count,
                "patch_recovery_count": exact_count if "patch_recovery" in requested_metrics else None,
                "patch_recovery_rate": patch_recovery_rate if "patch_recovery" in requested_metrics else None,
                "oracle_patch_recovery_count": oracle_exact_count,
                "oracle_patch_recovery_rate": oracle_patch_recovery_rate,
                "oracle_recovered_patch_count": oracle_recovered_patch_count,
                "oracle_collision_patch_count": oracle_collision_patch_count,
                "oracle_unresolved_patch_count": oracle_unresolved_patch_count,
                "patch_count": patch_count,
                "runtime": str(datetime.timedelta(seconds=time.time() - start)).split(".")[0],
                "public_patch_count": probe_stats.patch_stats.num_patches,
                "attack_runtime": str(datetime.timedelta(seconds=time.time() - start)).split(".")[0],
                "public_stats_source": stats_source if args.public_stats_path else f"{stats_source}:{public_data_source}",
                "data_source": data_source,
                "synthetic_fallback": synthetic_fallback,
                "fallback_reason": fallback_reason,
                "config_warning": getattr(args, "config_warning", None),
                "non_oracle_primary_only": True,
                "effective_batch_size": effective_batch_size,
                "attack_batch_count": attack_batch_count,
                "vit_adapter_loss": vit_loss,
                "batch_top1_acc": batch_top1_acc,
                "adapter_gradient_count": grad_count,
            },
        )
        return 0
    except Exception as exc:
        fallback_failed = isinstance(exc, SyntheticFallbackError)
        _emit_summary(
            args,
            {
                "result_status": "failed",
                "device": args.device,
                "synthetic_fallback": 1 if fallback_failed else None,
                "fallback_reason": exc.reason if fallback_failed else None,
                "runtime": str(datetime.timedelta(seconds=time.time() - start)).split(".")[0],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
