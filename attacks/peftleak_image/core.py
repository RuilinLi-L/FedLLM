from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PatchStatistics:
    mean: torch.Tensor
    std: torch.Tensor
    num_images: int
    num_patches: int
    patch_size: int


@dataclass(frozen=True)
class VitAdapterGradientResult:
    grads: tuple[torch.Tensor | None, ...]
    names: list[str]
    normalized_patches: torch.Tensor
    raw_patches: torch.Tensor
    logits: torch.Tensor
    loss: float


def extract_image_patches(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channels, height, width].")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    batch, channels, height, width = images.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("image height and width must be divisible by patch_size.")
    patches = F.unfold(images, kernel_size=patch_size, stride=patch_size)
    return patches.transpose(1, 2).contiguous().view(batch, -1, channels * patch_size * patch_size)


def fold_image_patches(
    patches: torch.Tensor,
    *,
    channels: int,
    grid_shape: tuple[int, int],
    patch_size: int,
) -> torch.Tensor:
    if patches.ndim != 3:
        raise ValueError("patches must have shape [batch, n_patches, patch_dim].")
    grid_h, grid_w = grid_shape
    expected = grid_h * grid_w
    if patches.shape[1] != expected:
        raise ValueError(f"Expected {expected} patches for grid_shape={grid_shape}; got {patches.shape[1]}.")
    patch_dim = channels * patch_size * patch_size
    if patches.shape[2] != patch_dim:
        raise ValueError(f"Expected patch_dim={patch_dim}; got {patches.shape[2]}.")
    folded = patches.view(patches.shape[0], expected, patch_dim).transpose(1, 2).contiguous()
    return F.fold(
        folded,
        output_size=(grid_h * patch_size, grid_w * patch_size),
        kernel_size=patch_size,
        stride=patch_size,
    )


def build_public_patch_statistics(images: torch.Tensor, patch_size: int) -> PatchStatistics:
    patches = extract_image_patches(images.detach().float(), patch_size)
    flat = patches.reshape(-1, patches.shape[-1])
    return PatchStatistics(
        mean=flat.mean(dim=0),
        std=flat.std(dim=0, unbiased=False).clamp_min(1e-12),
        num_images=int(images.shape[0]),
        num_patches=int(flat.shape[0]),
        patch_size=int(patch_size),
    )


def save_public_patch_statistics(stats: PatchStatistics, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mean": stats.mean.detach().cpu(),
            "std": stats.std.detach().cpu(),
            "num_images": int(stats.num_images),
            "num_patches": int(stats.num_patches),
            "patch_size": int(stats.patch_size),
        },
        target,
    )


def load_public_patch_statistics(path: str | Path) -> PatchStatistics:
    data = torch.load(Path(path), map_location=torch.device("cpu"))
    required = {"mean", "std", "num_images", "num_patches", "patch_size"}
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"Public patch statistics file is missing keys: {missing}.")
    return PatchStatistics(
        mean=data["mean"].detach().float(),
        std=data["std"].detach().float().clamp_min(1e-12),
        num_images=int(data["num_images"]),
        num_patches=int(data["num_patches"]),
        patch_size=int(data["patch_size"]),
    )


def move_patch_statistics(
    stats: PatchStatistics,
    *,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> PatchStatistics:
    target_dtype = dtype or stats.mean.dtype
    return PatchStatistics(
        mean=stats.mean.to(device=device, dtype=target_dtype),
        std=stats.std.to(device=device, dtype=target_dtype).clamp_min(1e-12),
        num_images=int(stats.num_images),
        num_patches=int(stats.num_patches),
        patch_size=int(stats.patch_size),
    )


def normalize_patches_with_public_stats(patches: torch.Tensor, stats: PatchStatistics) -> torch.Tensor:
    if int(stats.patch_size) <= 0:
        raise ValueError("stats.patch_size must be positive.")
    mean = stats.mean.to(device=patches.device, dtype=patches.dtype)
    std = stats.std.to(device=patches.device, dtype=patches.dtype).clamp_min(1e-12)
    if mean.numel() != patches.shape[-1] or std.numel() != patches.shape[-1]:
        raise ValueError(
            f"Public stats patch dimension mismatch: stats={mean.numel()}, patches={patches.shape[-1]}."
        )
    return (patches - mean.view(1, 1, -1)) / std.view(1, 1, -1)


def normalize_patches_with_stats(patches: torch.Tensor, stats: PatchStatistics) -> torch.Tensor:
    return normalize_patches_with_public_stats(patches, stats)


def denormalize_patches_with_public_stats(patches: torch.Tensor, stats: PatchStatistics) -> torch.Tensor:
    mean = stats.mean.to(device=patches.device, dtype=patches.dtype)
    std = stats.std.to(device=patches.device, dtype=patches.dtype).clamp_min(1e-12)
    if mean.numel() != patches.shape[-1] or std.numel() != patches.shape[-1]:
        raise ValueError(
            f"Public stats patch dimension mismatch: stats={mean.numel()}, patches={patches.shape[-1]}."
        )
    return patches * std.view(1, 1, -1) + mean.view(1, 1, -1)


def design_malicious_adapter_parameters(
    *,
    hidden_dim: int,
    patch_dim: int,
    seed: int = 0,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic adapter parameters for PEFTLeak-style probes."""

    if hidden_dim <= 0 or patch_dim <= 0:
        raise ValueError("hidden_dim and patch_dim must be positive.")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    weight = torch.randn(hidden_dim, patch_dim, generator=gen) * float(scale)
    signs = torch.randint(0, 2, (hidden_dim,), generator=gen, dtype=torch.int64).float()
    bias = signs.mul_(2.0).sub_(1.0) * float(scale)
    return weight, bias


class MaliciousPatchAdapter(nn.Module):
    """One adapter per sample/patch, so autograd gradients identify each patch."""

    def __init__(self, batch_size: int, n_patches: int, patch_dim: int, hidden_dim: int, seed: int = 0):
        super().__init__()
        if batch_size <= 0 or n_patches <= 0:
            raise ValueError("batch_size and n_patches must be positive.")
        self.batch_size = int(batch_size)
        self.n_patches = int(n_patches)
        self.patch_dim = int(patch_dim)
        self.hidden_dim = int(hidden_dim)
        self.adapters = nn.ModuleList()
        for sample_idx in range(self.batch_size):
            row = nn.ModuleList()
            for patch_idx in range(self.n_patches):
                adapter = nn.Linear(self.patch_dim, self.hidden_dim, bias=True)
                weight, bias = design_malicious_adapter_parameters(
                    hidden_dim=self.hidden_dim,
                    patch_dim=self.patch_dim,
                    seed=int(seed) + sample_idx * self.n_patches + patch_idx,
                    scale=0.01,
                )
                with torch.no_grad():
                    adapter.weight.copy_(weight.to(dtype=adapter.weight.dtype))
                    adapter.bias.copy_(bias.to(dtype=adapter.bias.dtype))
                row.append(adapter)
            self.adapters.append(row)

    def forward(self, normalized_patches: torch.Tensor) -> torch.Tensor:
        if normalized_patches.shape != (self.batch_size, self.n_patches, self.patch_dim):
            raise ValueError(
                "normalized_patches must have shape "
                f"{(self.batch_size, self.n_patches, self.patch_dim)}; got {tuple(normalized_patches.shape)}."
            )
        scores = []
        for sample_idx in range(self.batch_size):
            sample_scores = []
            for patch_idx in range(self.n_patches):
                hidden = self.adapters[sample_idx][patch_idx](normalized_patches[sample_idx, patch_idx])
                sample_scores.append(hidden.sum())
            scores.append(torch.stack(sample_scores, dim=0))
        return torch.stack(scores, dim=0)

    def parameter_names(self) -> list[str]:
        names: list[str] = []
        for sample_idx in range(self.batch_size):
            for patch_idx in range(self.n_patches):
                names.extend(
                    [
                        f"vit.adapter.sample_{sample_idx}.patch_{patch_idx}.weight",
                        f"vit.adapter.sample_{sample_idx}.patch_{patch_idx}.bias",
                    ]
                )
        return names

    def parameters_for_grad(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for sample_idx in range(self.batch_size):
            for patch_idx in range(self.n_patches):
                adapter = self.adapters[sample_idx][patch_idx]
                params.extend([adapter.weight, adapter.bias])
        return params


class TorchvisionVitWithMaliciousAdapter(nn.Module):
    """Torchvision ViT backbone with a malicious PEFT adapter branch."""

    def __init__(
        self,
        *,
        batch_size: int,
        channels: int,
        image_size: int,
        patch_size: int,
        adapter_hidden_dim: int,
        n_classes: int = 100,
        seed: int = 0,
        model_path: str = "torchvision_vit_small",
        finetuned_path: str | Path | None = None,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        patch_dim = channels * patch_size * patch_size
        n_patches = (image_size // patch_size) ** 2
        self.channels = int(channels)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.patch_dim = int(patch_dim)
        self.n_patches = int(n_patches)
        self.model_path = str(model_path)
        self.backbone = self._build_backbone(
            image_size=image_size,
            patch_size=patch_size,
            adapter_hidden_dim=adapter_hidden_dim,
            n_classes=n_classes,
            model_path=self.model_path,
        )
        if finetuned_path:
            self._load_backbone_checkpoint(finetuned_path)
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        self.adapter = MaliciousPatchAdapter(
            batch_size=batch_size,
            n_patches=n_patches,
            patch_dim=patch_dim,
            hidden_dim=adapter_hidden_dim,
            seed=seed,
        )

    @staticmethod
    def _build_backbone(
        *,
        image_size: int,
        patch_size: int,
        adapter_hidden_dim: int,
        n_classes: int,
        model_path: str,
    ) -> nn.Module:
        try:
            from torchvision.models import vit_b_16
            from torchvision.models.vision_transformer import VisionTransformer
        except Exception as exc:  # pragma: no cover - covered in server/conda env
            raise ImportError("vit_adapter mode requires torchvision; install the project conda environment.") from exc

        normalized_name = str(model_path or "").lower().replace("-", "_")
        if normalized_name in {"vit_b_16", "torchvision_vit_b_16"}:
            if image_size != 224 or patch_size != 16:
                raise ValueError(
                    "torchvision vit_b_16 requires --image_size 224 --patch_size 16; "
                    "use model_path=torchvision_vit_small for lightweight local runs."
                )
            return vit_b_16(weights=None, num_classes=int(n_classes))

        hidden_dim = max(32, int(adapter_hidden_dim) * 4)
        hidden_dim += (-hidden_dim) % 4
        return VisionTransformer(
            image_size=int(image_size),
            patch_size=int(patch_size),
            num_layers=1,
            num_heads=4,
            hidden_dim=hidden_dim,
            mlp_dim=hidden_dim * 2,
            dropout=0.0,
            attention_dropout=0.0,
            num_classes=int(n_classes),
        )

    @staticmethod
    def _resolve_checkpoint_file(path: str | Path) -> Path:
        checkpoint = Path(path)
        if checkpoint.is_file():
            return checkpoint
        if checkpoint.is_dir():
            for name in ("model.pth", "model.pt", "state_dict.pt", "pytorch_model.bin", "checkpoint.pt"):
                candidate = checkpoint / name
                if candidate.is_file():
                    return candidate
        raise FileNotFoundError(f"Could not find a ViT checkpoint file at {checkpoint}.")

    def _load_backbone_checkpoint(self, path: str | Path) -> None:
        checkpoint = self._resolve_checkpoint_file(path)
        state = torch.load(checkpoint, map_location=torch.device("cpu"))
        if isinstance(state, dict):
            for key in ("model_state_dict", "state_dict", "backbone", "backbone_state_dict"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported checkpoint format in {checkpoint}.")
        cleaned = {}
        for key, value in state.items():
            if not torch.is_tensor(value):
                continue
            clean_key = str(key)
            for prefix in ("module.", "model.", "backbone."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
            cleaned[clean_key] = value
        if not cleaned:
            raise ValueError(f"No tensor state dict entries found in {checkpoint}.")
        self.backbone.load_state_dict(cleaned, strict=False)

    def _backbone_input_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[1] == 3:
            return images
        if images.shape[1] == 1:
            return images.repeat(1, 3, 1, 1)
        raise ValueError("Torchvision ViT backbone expects images with 1 or 3 channels.")

    def forward(self, images: torch.Tensor, stats: PatchStatistics) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patches = extract_image_patches(images, self.patch_size)
        normalized = normalize_patches_with_public_stats(patches, stats)
        backbone_images = self._backbone_input_images(images)
        logits = self.backbone(backbone_images)
        adapter_scores = self.adapter(normalized)
        logits = logits.clone()
        logits[:, 0] = logits[:, 0] + adapter_scores.sum(dim=1)
        return logits, normalized, patches
    def adapter_parameter_names(self) -> list[str]:
        return self.adapter.parameter_names()

    def adapter_parameters_for_grad(self) -> list[nn.Parameter]:
        return self.adapter.parameters_for_grad()


TinyVitWithMaliciousAdapter = TorchvisionVitWithMaliciousAdapter


def build_vit_adapter_gradients(
    images: torch.Tensor,
    stats: PatchStatistics,
    *,
    labels: torch.Tensor | None = None,
    patch_size: int,
    adapter_hidden_dim: int,
    n_classes: int = 100,
    seed: int = 0,
    model_path: str = "torchvision_vit_small",
    finetuned_path: str | Path | None = None,
) -> VitAdapterGradientResult:
    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channels, height, width].")
    if int(stats.patch_size) != int(patch_size):
        raise ValueError(f"stats.patch_size={stats.patch_size} does not match patch_size={patch_size}.")
    batch_size, channels, height, width = images.shape
    if height != width:
        raise ValueError("vit_adapter mode expects square images.")
    model = TorchvisionVitWithMaliciousAdapter(
        batch_size=batch_size,
        channels=channels,
        image_size=height,
        patch_size=patch_size,
        adapter_hidden_dim=adapter_hidden_dim,
        n_classes=n_classes,
        seed=seed,
        model_path=model_path,
        finetuned_path=finetuned_path,
    ).to(device=images.device, dtype=images.dtype)
    model.eval()
    if labels is None:
        labels = torch.arange(batch_size, device=images.device) % int(n_classes)
    else:
        labels = labels.to(device=images.device, dtype=torch.long).view(-1)
    if labels.numel() != batch_size:
        raise ValueError(f"Expected {batch_size} labels, got {labels.numel()}.")
    logits, normalized, raw_patches = model(images, stats)
    loss = F.cross_entropy(logits, labels)
    grads = torch.autograd.grad(
        loss,
        model.adapter_parameters_for_grad(),
        allow_unused=True,
    )
    return VitAdapterGradientResult(
        grads=tuple(None if grad is None else grad.detach() for grad in grads),
        names=model.adapter_parameter_names(),
        normalized_patches=normalized.detach(),
        raw_patches=raw_patches.detach(),
        logits=logits.detach(),
        loss=float(loss.detach().item()),
    )


_NAMED_ADAPTER_RE = re.compile(r"sample_(\d+)\.patch_(\d+)\.(weight|bias)$")


def recover_patches_from_named_adapter_grads(
    grads: Sequence[torch.Tensor | None],
    names: Sequence[str],
    *,
    batch_size: int | None = None,
    n_patches: int | None = None,
    eps: float = 1e-12,
    aggregation: str = "median",
    patch_mean: torch.Tensor | float | None = None,
    patch_std: torch.Tensor | float | None = None,
) -> torch.Tensor:
    pairs: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    for grad, name in zip(grads, names):
        if grad is None:
            continue
        match = _NAMED_ADAPTER_RE.search(str(name))
        if match is None:
            continue
        key = (int(match.group(1)), int(match.group(2)))
        pairs.setdefault(key, {})[match.group(3)] = grad

    if not pairs:
        raise ValueError("No named adapter gradient pairs were available for recovery.")
    if batch_size is None:
        batch_size = max(sample for sample, _ in pairs) + 1
    if n_patches is None:
        n_patches = max(patch for _, patch in pairs) + 1

    recovered: list[torch.Tensor | None] = [None] * (int(batch_size) * int(n_patches))
    for (sample_idx, patch_idx), pair in pairs.items():
        if "weight" not in pair or "bias" not in pair:
            continue
        recovered[sample_idx * int(n_patches) + patch_idx] = recover_patch_from_adapter_grads(
            pair["weight"],
            pair["bias"],
            eps=eps,
            aggregation=aggregation,
            patch_mean=patch_mean,
            patch_std=patch_std,
        )
    if any(item is None for item in recovered):
        missing = [idx for idx, item in enumerate(recovered) if item is None]
        raise ValueError(f"Missing recovered patches for adapter pair indices: {missing[:8]}.")
    return torch.stack([item for item in recovered if item is not None], dim=0).view(int(batch_size), int(n_patches), -1)


def recover_patch_from_adapter_grads(
    weight_grad: torch.Tensor,
    bias_grad: torch.Tensor,
    *,
    eps: float = 1e-12,
    aggregation: str = "median",
    position_embedding: torch.Tensor | None = None,
    patch_mean: torch.Tensor | float | None = None,
    patch_std: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Recover one patch from adapter weight/bias gradients."""

    if weight_grad.ndim != 2:
        raise ValueError("weight_grad must have shape [hidden, patch_dim].")
    if bias_grad.ndim != 1:
        raise ValueError("bias_grad must have shape [hidden].")
    if weight_grad.shape[0] != bias_grad.shape[0]:
        raise ValueError("weight_grad and bias_grad hidden dimensions do not match.")

    mask = bias_grad.detach().abs() > eps
    if not bool(mask.any()):
        raise ValueError("No nonzero adapter bias gradients are available for ratio recovery.")

    candidates = weight_grad[mask].float() / bias_grad[mask].float().unsqueeze(1)
    if aggregation == "median":
        recovered = candidates.median(dim=0).values
    elif aggregation == "mean":
        recovered = candidates.mean(dim=0)
    else:
        raise ValueError("aggregation must be 'median' or 'mean'.")

    if position_embedding is not None:
        recovered = recovered - position_embedding.to(device=recovered.device, dtype=recovered.dtype)
    if patch_std is not None:
        recovered = recovered * torch.as_tensor(patch_std, device=recovered.device, dtype=recovered.dtype)
    if patch_mean is not None:
        recovered = recovered + torch.as_tensor(patch_mean, device=recovered.device, dtype=recovered.dtype)
    return recovered


def recover_patches_from_batch(
    weight_grads: Sequence[torch.Tensor],
    bias_grads: Sequence[torch.Tensor],
    *,
    eps: float = 1e-12,
    aggregation: str = "median",
    position_embeddings: Sequence[torch.Tensor] | None = None,
    patch_mean: torch.Tensor | float | None = None,
    patch_std: torch.Tensor | float | None = None,
) -> torch.Tensor:
    if len(weight_grads) != len(bias_grads):
        raise ValueError("weight_grads and bias_grads must have the same length.")
    patches = []
    for idx, (w_grad, b_grad) in enumerate(zip(weight_grads, bias_grads)):
        pos = None if position_embeddings is None else position_embeddings[idx]
        patches.append(
            recover_patch_from_adapter_grads(
                w_grad,
                b_grad,
                eps=eps,
                aggregation=aggregation,
                position_embedding=pos,
                patch_mean=patch_mean,
                patch_std=patch_std,
            )
        )
    return torch.stack(patches, dim=0)


def deterministic_kmeans(
    features: torch.Tensor,
    n_clusters: int,
    *,
    seed: int = 0,
    n_iter: int = 20,
) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError("features must have shape [n_items, dim].")
    if n_clusters <= 0 or n_clusters > features.shape[0]:
        raise ValueError("n_clusters must be in [1, n_items].")
    gen = torch.Generator(device=features.device)
    gen.manual_seed(int(seed))
    order = torch.randperm(features.shape[0], generator=gen, device=features.device)
    centers = features[order[:n_clusters]].clone().float()
    x = features.float()
    assignments = torch.zeros(features.shape[0], dtype=torch.long, device=features.device)
    for _ in range(max(1, int(n_iter))):
        distances = torch.cdist(x, centers, p=2)
        assignments = distances.argmin(dim=1)
        new_centers = []
        for cluster_idx in range(n_clusters):
            mask = assignments == cluster_idx
            if bool(mask.any()):
                new_centers.append(x[mask].mean(dim=0))
            else:
                new_centers.append(centers[cluster_idx])
        centers = torch.stack(new_centers, dim=0)
    return assignments.detach()


def cluster_and_reassemble(
    patches: torch.Tensor,
    *,
    channels: int,
    grid_shape: tuple[int, int],
    patch_size: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministically order recovered patches and fold them to images."""

    if patches.ndim == 2:
        patches = patches.unsqueeze(0)
    if patches.ndim != 3:
        raise ValueError("patches must have shape [batch, n_patches, patch_dim] or [n_patches, patch_dim].")
    n_patches = grid_shape[0] * grid_shape[1]
    if patches.shape[1] != n_patches:
        raise ValueError(f"grid_shape requires {n_patches} patches; got {patches.shape[1]}.")

    assembled_batches = []
    assignments = []
    for sample in patches:
        feature = torch.stack((sample.mean(dim=1), sample.std(dim=1)), dim=1)
        assignments.append(deterministic_kmeans(feature, n_clusters=max(1, grid_shape[0]), seed=seed))
        ordering = sorted(
            range(sample.shape[0]),
            key=lambda idx: (float(feature[idx, 0]), float(feature[idx, 1]), idx),
        )
        ordered = sample[torch.tensor(ordering, device=sample.device)]
        assembled_batches.append(
            fold_image_patches(
                ordered.unsqueeze(0),
                channels=channels,
                grid_shape=grid_shape,
                patch_size=patch_size,
            ).squeeze(0)
        )
    return torch.stack(assembled_batches, dim=0), torch.stack(assignments, dim=0)


def optimize_patch_baseline(
    target_weight_grad: torch.Tensor,
    target_bias_grad: torch.Tensor,
    *,
    steps: int = 200,
    lr: float = 0.05,
    tv_weight: float = 0.0,
    seed: int = 0,
) -> tuple[torch.Tensor, list[float]]:
    if target_weight_grad.ndim != 2 or target_bias_grad.ndim != 1:
        raise ValueError("target gradients must be [hidden, patch_dim] and [hidden].")
    gen = torch.Generator(device=target_weight_grad.device)
    gen.manual_seed(int(seed))
    dummy = torch.nn.Parameter(torch.randn(target_weight_grad.shape[1], device=target_weight_grad.device, generator=gen))
    opt = torch.optim.Adam([dummy], lr=float(lr))
    history: list[float] = []
    for _ in range(max(1, int(steps))):
        opt.zero_grad(set_to_none=True)
        pred_w = target_bias_grad.detach().unsqueeze(1) * dummy.unsqueeze(0)
        loss = F.mse_loss(pred_w, target_weight_grad.detach())
        if tv_weight > 0 and dummy.numel() > 1:
            loss = loss + float(tv_weight) * dummy[1:].sub(dummy[:-1]).abs().mean()
        loss.backward()
        opt.step()
        history.append(float(loss.detach().item()))
    return dummy.detach(), history


def mse_psnr(recovered: torch.Tensor, reference: torch.Tensor, *, data_range: float = 1.0) -> tuple[float, float]:
    mse = float(F.mse_loss(recovered.float(), reference.float()).item())
    if mse <= 0:
        return 0.0, float("inf")
    psnr = 20.0 * torch.log10(torch.tensor(float(data_range))).item() - 10.0 * torch.log10(torch.tensor(mse)).item()
    return mse, float(psnr)
