from __future__ import annotations

import argparse
import datetime
import time
import torch

from attacks.peftleak_image.core import (
    build_public_patch_statistics,
    build_vit_adapter_gradients,
    cluster_and_reassemble,
    extract_image_patches,
    fold_image_patches,
    load_public_patch_statistics,
    mse_psnr,
    recover_patch_from_adapter_grads,
    recover_patches_from_named_adapter_grads,
    save_public_patch_statistics,
)
from utils.defense_common import add_shared_defense_args, defense_param_spec, fmt_summary_value
from utils.defenses import dpsgd_defense, gradient_compression, noise_injection, topk_sparsification
from utils.lrb_defense import apply_lrb_defense
from utils.lrb_presets import apply_lrb_preset


SUMMARY_START = "===== RESULT SUMMARY START ====="
SUMMARY_END = "===== RESULT SUMMARY END ====="
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PEFTLeak image-adapter reproduction with synthetic ratio and torchvision ViT-adapter modes")
    parser.add_argument("--mode", choices=["vit_adapter", "synthetic_ratio"], default="vit_adapter")
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar100", "synthetic"])
    parser.add_argument("--data_root", type=str, default="./models_cache")
    parser.add_argument("--cache_dir", type=str, default=None)
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
    parser.add_argument("--patch_recovery_mse_threshold", type=float, default=1e-6)
    add_shared_defense_args(parser, default_grad_mode="eval")
    return parser


def _synthetic_images(n_images: int, channels: int, image_size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    images = torch.rand(n_images, channels, image_size, image_size, generator=gen)
    labels = torch.arange(n_images, dtype=torch.long)
    return images, labels


def _load_images(args, *, public: bool = False) -> tuple[torch.Tensor, torch.Tensor, str]:
    n_images = int(args.public_n_images if public else args.n_images)
    seed = int(args.rng_seed) + (7919 if public else 0)
    if args.dataset == "cifar100":
        try:
            from torchvision import datasets, transforms

            data_root = args.cache_dir or args.data_root
            ds = datasets.CIFAR100(
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
                source = "cifar100_public" if public else "cifar100_train"
                return torch.stack(imgs, dim=0), torch.tensor(labels, dtype=torch.long), source
        except Exception as exc:
            print(f"[peftleak-image] CIFAR100 unavailable ({exc}); falling back to synthetic images.", flush=True)

    images, labels = _synthetic_images(n_images, args.channels, args.image_size, seed)
    return images, labels % int(args.n_classes), "synthetic_public" if public else "synthetic_attack"


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
    perm = torch.arange(patches.shape[0] - 1, -1, -1)
    return lam * patches + (1.0 - lam) * patches[perm]


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
    return recovered_flat.view(patches.shape), patches, None, len(names)


def _run_vit_adapter(args, images: torch.Tensor, labels: torch.Tensor, stats):
    reference_patches = extract_image_patches(images, args.patch_size)
    grad_images = _defended_images(args, images)
    if args.defense == "dpsgd":
        full_result = build_vit_adapter_gradients(
            grad_images,
            stats,
            labels=labels,
            patch_size=args.patch_size,
            adapter_hidden_dim=args.adapter_hidden_dim,
            n_classes=args.n_classes,
            seed=args.rng_seed,
            model_path=args.model_path,
            finetuned_path=args.finetuned_path,
        )
        n_grads = len(full_result.grads)
        per_example = []
        for sample_idx in range(grad_images.shape[0]):
            sample_result = build_vit_adapter_gradients(
                grad_images[sample_idx : sample_idx + 1],
                stats,
                labels=labels[sample_idx : sample_idx + 1],
                patch_size=args.patch_size,
                adapter_hidden_dim=args.adapter_hidden_dim,
                n_classes=args.n_classes,
                seed=args.rng_seed + sample_idx * reference_patches.shape[1],
                model_path=args.model_path,
                finetuned_path=args.finetuned_path,
            )
            sample_grads = [None] * n_grads
            offset = sample_idx * reference_patches.shape[1] * 2
            for local_idx, grad in enumerate(sample_result.grads):
                sample_grads[offset + local_idx] = grad
            per_example.append(tuple(sample_grads))
        defended_grads = dpsgd_defense(per_example, args.defense_clip_norm, args.defense_noise or 0.0, seed=args.rng_seed)
        names = full_result.names
        loss = full_result.loss
    else:
        result = build_vit_adapter_gradients(
            grad_images,
            stats,
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
    recovered = recover_patches_from_named_adapter_grads(
        defended_grads,
        names,
        batch_size=images.shape[0],
        n_patches=reference_patches.shape[1],
        patch_mean=stats.mean,
        patch_std=stats.std,
    )
    return recovered, reference_patches, loss, len(names)


def _emit_summary(args, fields: dict):
    defense_param_name, defense_param_value = defense_param_spec(args)
    ordered = [
        ("summary_version", 2),
        ("result_status", fields.get("result_status", "ok")),
        ("attack", "peftleak_image_repro"),
        ("attack_variant", args.mode),
        ("dataset", args.dataset),
        ("data_source", fields.get("data_source")),
        ("public_stats_source", fields.get("public_stats_source")),
        ("model_path", args.model_path),
        ("finetuned_path", args.finetuned_path or "n/a"),
        ("batch_size", args.batch_size),
        ("effective_batch_size", fields.get("effective_batch_size")),
        ("n_images", args.n_images),
        ("patch_size", args.patch_size),
        ("adapter_hidden_dim", args.adapter_hidden_dim),
        ("adapter_gradient_count", fields.get("adapter_gradient_count")),
        ("vit_adapter_loss", fields.get("vit_adapter_loss")),
        ("defense", args.defense),
        ("defense_param_name", defense_param_name),
        ("defense_param_value", defense_param_value),
        ("mse", fields.get("mse")),
        ("psnr", fields.get("psnr")),
        ("patch_recovery_count", fields.get("patch_recovery_count")),
        ("patch_count", fields.get("patch_count")),
        ("public_patch_count", fields.get("public_patch_count")),
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
    apply_lrb_preset(args)
    start = time.time()
    if args.defense == "dager":
        _emit_summary(
            args,
            {
                "result_status": "unsupported",
                "runtime": str(datetime.timedelta(seconds=time.time() - start)).split(".")[0],
                "error_type": "unsupported_defense",
                "error_message": "DAGER defense is DAGER-specific and is excluded from PEFTLeak image matrices.",
            },
        )
        return 0
    if args.defense not in SUPPORTED_IMAGE_DEFENSES:
        raise NotImplementedError(f"Unsupported image PEFTLeak defense: {args.defense!r}")

    try:
        images, labels, data_source = _load_images(args, public=False)
        if args.public_stats_path:
            public_data_source = "file"
            stats, stats_source = _public_stats(args, None)
        else:
            public_images, _public_labels, public_data_source = _load_images(args, public=True)
            stats, stats_source = _public_stats(args, public_images)
        if args.mode == "synthetic_ratio":
            recovered, reference_patches, vit_loss, grad_count = _run_synthetic_ratio(args, images)
        else:
            recovered, reference_patches, vit_loss, grad_count = _run_vit_adapter(args, images, labels, stats)
        grid_shape = (images.shape[2] // args.patch_size, images.shape[3] // args.patch_size)
        assembled_clustered, _assignments = cluster_and_reassemble(
            recovered,
            channels=images.shape[1],
            grid_shape=grid_shape,
            patch_size=args.patch_size,
            seed=args.rng_seed,
        )
        folded_reference = fold_image_patches(
            reference_patches,
            channels=images.shape[1],
            grid_shape=grid_shape,
            patch_size=args.patch_size,
        )
        mse, psnr = mse_psnr(assembled_clustered, folded_reference)
        patch_mse = (recovered - reference_patches).float().pow(2).mean(dim=-1)
        recovered_count = int((patch_mse < args.patch_recovery_mse_threshold).sum().item())
        _emit_summary(
            args,
            {
                "mse": mse,
                "psnr": psnr,
                "patch_recovery_count": recovered_count,
                "patch_count": int(reference_patches.numel() // reference_patches.shape[-1]),
                "runtime": str(datetime.timedelta(seconds=time.time() - start)).split(".")[0],
                "public_patch_count": stats.num_patches,
                "public_stats_source": stats_source if args.public_stats_path else f"{stats_source}:{public_data_source}",
                "data_source": data_source,
                "effective_batch_size": int(images.shape[0]),
                "vit_adapter_loss": vit_loss,
                "adapter_gradient_count": grad_count,
            },
        )
        return 0
    except Exception as exc:
        _emit_summary(
            args,
            {
                "result_status": "failed",
                "runtime": str(datetime.timedelta(seconds=time.time() - start)).split(".")[0],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())