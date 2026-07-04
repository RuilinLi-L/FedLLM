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
class PeftLeakProbeStatistics:
    patch_stats: PatchStatistics
    position_embeddings: torch.Tensor
    projection_vectors: torch.Tensor
    bin_edges: torch.Tensor
    bin_counts: torch.Tensor
    num_bins: int
    embed_scale: float
    gap: int
    seed: int


@dataclass(frozen=True)
class PeftLeakRecoveryResult:
    recovered_patches: torch.Tensor
    recovery_mask: torch.Tensor
    candidate_patch_count: int
    recovered_patch_count: int
    collision_patch_count: int | None
    unresolved_patch_count: int
    candidate_patches: torch.Tensor
    candidate_slots: torch.Tensor
    candidate_position_indices: torch.Tensor
    slot_indices: torch.Tensor | None
    slot_counts: torch.Tensor | None
    oracle_recovered_patches: torch.Tensor | None
    oracle_recovery_mask: torch.Tensor | None
    oracle_recovered_patch_count: int | None
    oracle_collision_patch_count: int | None
    oracle_unresolved_patch_count: int | None
    raw_candidate_metadata: dict[str, int]


@dataclass(frozen=True)
class VitAdapterGradientResult:
    grads: tuple[torch.Tensor | None, ...]
    names: list[str]
    normalized_patches: torch.Tensor
    raw_patches: torch.Tensor
    logits: torch.Tensor
    loss: float
    exposed_patches: torch.Tensor | None = None
    slot_indices: torch.Tensor | None = None


@dataclass(frozen=True)
class OfficialAlignedGradientResult:
    grads: tuple[torch.Tensor | None, ...]
    names: list[str]
    raw_patches: torch.Tensor
    exposed_patches: torch.Tensor
    tagged_exposed_patches: torch.Tensor
    slot_indices: torch.Tensor
    logits: torch.Tensor
    loss: float
    adapter_layer_count: int
    adapter_bottleneck_dim: int
    sample_tags: torch.Tensor


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


def _make_generator(seed: int, device: torch.device | str) -> torch.Generator:
    gen = torch.Generator(device=torch.device(device))
    gen.manual_seed(int(seed))
    return gen


def _normalize_projection_vectors(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / vectors.float().norm(dim=-1, keepdim=True).clamp_min(1e-12).to(dtype=vectors.dtype)


def _bin_edges_from_scores(scores: torch.Tensor, num_bins: int) -> tuple[torch.Tensor, torch.Tensor]:
    if scores.ndim != 2:
        raise ValueError("scores must have shape [num_items, n_patches].")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")

    sorted_scores = scores.detach().float().sort(dim=0).values
    n_items, n_patches = sorted_scores.shape
    device = scores.device
    edges = torch.empty(n_patches, int(num_bins) + 1, device=device, dtype=torch.float32)
    edges[:, 0] = float("-inf")
    edges[:, -1] = float("inf")
    for edge_idx in range(1, int(num_bins)):
        source_idx = min(n_items - 1, max(0, int(round(edge_idx * (n_items - 1) / int(num_bins)))))
        edges[:, edge_idx] = sorted_scores[source_idx]

    thresholds = edges[:, 1:-1]
    counts = torch.zeros(n_patches, int(num_bins), device=device, dtype=torch.long)
    for patch_idx in range(n_patches):
        bins = torch.bucketize(scores[:, patch_idx].detach().float(), thresholds[patch_idx], right=False)
        counts[patch_idx].scatter_add_(0, bins, torch.ones_like(bins, dtype=torch.long))
    return edges, counts


def build_peftleak_probe_statistics(
    public_images: torch.Tensor,
    patch_size: int,
    num_bins: int = 32,
    position_sigma: float = 1.0,
    embed_scale: float = 0.5,
    gap: int = 0,
    seed: int = 0,
) -> PeftLeakProbeStatistics:
    if public_images.ndim != 4:
        raise ValueError("public_images must have shape [batch, channels, height, width].")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")
    if embed_scale == 0:
        raise ValueError("embed_scale must be nonzero.")
    if gap < 0:
        raise ValueError("gap must be non-negative.")

    patch_stats = build_public_patch_statistics(public_images, patch_size)
    patches = extract_image_patches(public_images.detach().float(), patch_size)
    normalized = normalize_patches_with_public_stats(patches, patch_stats)
    n_patches = normalized.shape[1]
    patch_dim = normalized.shape[2]
    gen = _make_generator(seed, public_images.device)
    position_embeddings = torch.randn(
        n_patches,
        patch_dim,
        generator=gen,
        device=public_images.device,
        dtype=torch.float32,
    ) * float(position_sigma)
    projection_vectors = _normalize_projection_vectors(
        torch.randn(
            n_patches,
            patch_dim,
            generator=gen,
            device=public_images.device,
            dtype=torch.float32,
        )
    )
    exposed = normalized.float() * float(embed_scale) + position_embeddings.view(1, n_patches, patch_dim)
    scores = (exposed * projection_vectors.view(1, n_patches, patch_dim)).sum(dim=-1)
    bin_edges, bin_counts = _bin_edges_from_scores(scores, int(num_bins))
    return PeftLeakProbeStatistics(
        patch_stats=patch_stats,
        position_embeddings=position_embeddings,
        projection_vectors=projection_vectors,
        bin_edges=bin_edges,
        bin_counts=bin_counts,
        num_bins=int(num_bins),
        embed_scale=float(embed_scale),
        gap=int(gap),
        seed=int(seed),
    )


def build_official_aligned_probe_statistics(
    public_images: torch.Tensor,
    patch_size: int,
    num_bins: int = 32,
    position_sigma: float = 1.0,
    embed_scale: float = 0.5,
    gap: int = 0,
    seed: int = 0,
) -> PeftLeakProbeStatistics:
    """Build PEFTLeak official-aligned public CDF probes.

    Unlike the lightweight shared-bin path, this uses deterministic projection
    vectors and public quantiles as a CDF-style interval construction. It is
    still a v1 alignment layer, not a byte-for-byte port of the official repo.
    """

    if public_images.ndim != 4:
        raise ValueError("public_images must have shape [batch, channels, height, width].")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")
    if embed_scale == 0:
        raise ValueError("embed_scale must be nonzero.")
    if gap < 0:
        raise ValueError("gap must be non-negative.")

    patch_stats = build_public_patch_statistics(public_images, patch_size)
    patches = extract_image_patches(public_images.detach().float(), patch_size)
    normalized = normalize_patches_with_public_stats(patches, patch_stats)
    n_patches = normalized.shape[1]
    patch_dim = normalized.shape[2]
    device = public_images.device

    base = torch.linspace(-1.0, 1.0, steps=patch_dim, device=device, dtype=torch.float32)
    positions = []
    projections = []
    for patch_idx in range(n_patches):
        offset = float(patch_idx + 1) / float(max(1, n_patches))
        pos = torch.sin(base * (patch_idx + 1)) + torch.cos(base * offset)
        positions.append(pos * float(position_sigma))
        rolled = torch.roll(base, shifts=patch_idx % max(1, patch_dim)).clone()
        if float(rolled.norm().item()) <= 1e-12:
            rolled = torch.ones_like(rolled)
        projections.append(rolled)

    position_embeddings = torch.stack(positions, dim=0)
    projection_vectors = _normalize_projection_vectors(torch.stack(projections, dim=0))
    exposed = normalized.float() * float(embed_scale) + position_embeddings.view(1, n_patches, patch_dim)
    scores = (exposed * projection_vectors.view(1, n_patches, patch_dim)).sum(dim=-1)
    bin_edges, bin_counts = _bin_edges_from_scores(scores, int(num_bins))
    return PeftLeakProbeStatistics(
        patch_stats=patch_stats,
        position_embeddings=position_embeddings,
        projection_vectors=projection_vectors,
        bin_edges=bin_edges,
        bin_counts=bin_counts,
        num_bins=int(num_bins),
        embed_scale=float(embed_scale),
        gap=int(gap),
        seed=int(seed),
    )


def build_peftleak_probe_statistics_from_patch_stats(
    patch_stats: PatchStatistics,
    *,
    n_patches: int,
    num_bins: int = 32,
    position_sigma: float = 1.0,
    embed_scale: float = 0.5,
    gap: int = 0,
    seed: int = 0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> PeftLeakProbeStatistics:
    if n_patches <= 0:
        raise ValueError("n_patches must be positive.")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")
    if embed_scale == 0:
        raise ValueError("embed_scale must be nonzero.")
    if gap < 0:
        raise ValueError("gap must be non-negative.")

    target_device = torch.device(device or patch_stats.mean.device)
    target_dtype = dtype or patch_stats.mean.dtype
    moved_stats = move_patch_statistics(patch_stats, device=target_device, dtype=target_dtype)
    patch_dim = int(moved_stats.mean.numel())
    gen = _make_generator(seed, target_device)
    position_embeddings = torch.randn(
        int(n_patches),
        patch_dim,
        generator=gen,
        device=target_device,
        dtype=torch.float32,
    ) * float(position_sigma)
    projection_vectors = _normalize_projection_vectors(
        torch.randn(
            int(n_patches),
            patch_dim,
            generator=gen,
            device=target_device,
            dtype=torch.float32,
        )
    )
    edges_1d = torch.linspace(-4.0, 4.0, steps=int(num_bins) + 1, device=target_device, dtype=torch.float32)
    edges_1d[0] = float("-inf")
    edges_1d[-1] = float("inf")
    bin_edges = edges_1d.view(1, -1).repeat(int(n_patches), 1)
    bin_counts = torch.zeros(int(n_patches), int(num_bins), device=target_device, dtype=torch.long)
    return PeftLeakProbeStatistics(
        patch_stats=moved_stats,
        position_embeddings=position_embeddings,
        projection_vectors=projection_vectors,
        bin_edges=bin_edges,
        bin_counts=bin_counts,
        num_bins=int(num_bins),
        embed_scale=float(embed_scale),
        gap=int(gap),
        seed=int(seed),
    )


def move_peftleak_probe_statistics(
    stats: PeftLeakProbeStatistics,
    *,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> PeftLeakProbeStatistics:
    target_dtype = dtype or stats.patch_stats.mean.dtype
    return PeftLeakProbeStatistics(
        patch_stats=move_patch_statistics(stats.patch_stats, device=device, dtype=target_dtype),
        position_embeddings=stats.position_embeddings.to(device=device, dtype=target_dtype),
        projection_vectors=_normalize_projection_vectors(stats.projection_vectors.to(device=device, dtype=target_dtype)),
        bin_edges=stats.bin_edges.to(device=device, dtype=torch.float32),
        bin_counts=stats.bin_counts.to(device=device),
        num_bins=int(stats.num_bins),
        embed_scale=float(stats.embed_scale),
        gap=int(stats.gap),
        seed=int(stats.seed),
    )


def peftleak_expose_patches(patches: torch.Tensor, stats: PeftLeakProbeStatistics) -> torch.Tensor:
    normalized = normalize_patches_with_public_stats(patches, stats.patch_stats)
    pos = stats.position_embeddings.to(device=patches.device, dtype=patches.dtype)
    return normalized * float(stats.embed_scale) + pos.view(1, pos.shape[0], pos.shape[1])


def assign_peftleak_patch_slots(exposed_patches: torch.Tensor, stats: PeftLeakProbeStatistics) -> torch.Tensor:
    if exposed_patches.ndim != 3:
        raise ValueError("exposed_patches must have shape [batch, n_patches, patch_dim].")
    n_patches = exposed_patches.shape[1]
    if n_patches != stats.position_embeddings.shape[0]:
        raise ValueError(f"Expected {stats.position_embeddings.shape[0]} patches; got {n_patches}.")
    projections = stats.projection_vectors.to(device=exposed_patches.device, dtype=exposed_patches.dtype)
    scores = (exposed_patches * projections.view(1, n_patches, -1)).sum(dim=-1).detach().float()
    edges = stats.bin_edges.to(device=exposed_patches.device, dtype=torch.float32)
    slots = torch.empty(scores.shape, device=exposed_patches.device, dtype=torch.long)
    for patch_idx in range(n_patches):
        bins = torch.bucketize(scores[:, patch_idx], edges[patch_idx, 1:-1], right=False)
        slots[:, patch_idx] = patch_idx * int(stats.num_bins) + bins
    return slots


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


class DebugOraclePatchAdapter(nn.Module):
    """Debug-only adapter with one parameter pair per sample/patch."""

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


class SharedBinPatchAdapter(nn.Module):
    """Shared PEFTLeak-style adapter where patch/bin slots share parameters."""

    def __init__(self, *, n_patches: int, patch_dim: int, num_bins: int, gap: int = 0):
        super().__init__()
        if n_patches <= 0 or patch_dim <= 0 or num_bins <= 0:
            raise ValueError("n_patches, patch_dim, and num_bins must be positive.")
        if gap < 0:
            raise ValueError("gap must be non-negative.")
        self.n_patches = int(n_patches)
        self.patch_dim = int(patch_dim)
        self.num_bins = int(num_bins)
        self.gap = int(gap)
        self.row_stride = int(gap) + 2
        self.n_slots = self.n_patches * self.num_bins
        self.weight = nn.Parameter(torch.zeros(self.n_slots * self.row_stride, self.patch_dim))
        self.bias = nn.Parameter(torch.zeros(self.n_slots * self.row_stride))

    def forward(self, exposed_patches: torch.Tensor, slot_indices: torch.Tensor) -> torch.Tensor:
        if exposed_patches.ndim != 3:
            raise ValueError("exposed_patches must have shape [batch, n_patches, patch_dim].")
        if exposed_patches.shape[1:] != (self.n_patches, self.patch_dim):
            raise ValueError(
                "exposed_patches must have shape "
                f"[batch, {self.n_patches}, {self.patch_dim}]; got {tuple(exposed_patches.shape)}."
            )
        if slot_indices.shape != exposed_patches.shape[:2]:
            raise ValueError("slot_indices must have shape [batch, n_patches].")
        if int(slot_indices.min().item()) < 0 or int(slot_indices.max().item()) >= self.n_slots:
            raise ValueError("slot_indices contain an out-of-range PEFTLeak slot.")

        flat_x = exposed_patches.reshape(-1, self.patch_dim)
        flat_slots = slot_indices.reshape(-1).to(device=flat_x.device, dtype=torch.long)
        row_a = flat_slots * self.row_stride
        row_b = row_a + self.gap + 1
        out_a = (flat_x * self.weight.index_select(0, row_a)).sum(dim=-1) + self.bias.index_select(0, row_a)
        out_b = (flat_x * self.weight.index_select(0, row_b)).sum(dim=-1) + self.bias.index_select(0, row_b)
        return (out_a - out_b).view(exposed_patches.shape[0], self.n_patches)

    def parameter_names(self) -> list[str]:
        return ["vit.adapter.shared.weight", "vit.adapter.shared.bias"]

    def parameters_for_grad(self) -> list[nn.Parameter]:
        return [self.weight, self.bias]


class OfficialAlignedPatchAdapter(nn.Module):
    """Multi-layer PEFTLeak-style adapter bank with observable sample tags."""

    def __init__(
        self,
        *,
        n_layers: int,
        n_patches: int,
        patch_dim: int,
        num_bins: int,
        bottleneck_dim: int,
        gap: int = 0,
    ):
        super().__init__()
        if n_layers <= 0:
            raise ValueError("n_layers must be positive.")
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive.")
        self.n_layers = int(n_layers)
        self.n_patches = int(n_patches)
        self.patch_dim = int(patch_dim)
        self.tagged_dim = int(patch_dim) + 1
        self.num_bins = int(num_bins)
        self.bottleneck_dim = int(bottleneck_dim)
        self.gap = int(gap)
        self.layers = nn.ModuleList(
            [
                SharedBinPatchAdapter(
                    n_patches=n_patches,
                    patch_dim=self.tagged_dim,
                    num_bins=num_bins,
                    gap=gap,
                )
                for _ in range(self.n_layers)
            ]
        )

    def forward(self, tagged_exposed_patches: torch.Tensor, slot_indices: torch.Tensor) -> torch.Tensor:
        scores = []
        for layer in self.layers:
            scores.append(layer(tagged_exposed_patches, slot_indices))
        stacked = torch.stack(scores, dim=0)
        return stacked.sum(dim=0) / float(max(1, self.n_layers))

    def parameter_names(self) -> list[str]:
        names: list[str] = []
        for layer_idx in range(self.n_layers):
            names.extend(
                [
                    f"vit.official_adapter.layer_{layer_idx}.shared.weight",
                    f"vit.official_adapter.layer_{layer_idx}.shared.bias",
                ]
            )
        return names

    def parameters_for_grad(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for layer in self.layers:
            params.extend(layer.parameters_for_grad())
        return params


MaliciousPatchAdapter = DebugOraclePatchAdapter


class TorchvisionVitWithMaliciousAdapter(nn.Module):
    """Torchvision ViT backbone with a shared PEFTLeak-style adapter branch."""

    def __init__(
        self,
        *,
        batch_size: int,
        channels: int,
        image_size: int,
        patch_size: int,
        adapter_hidden_dim: int,
        probe_stats: PeftLeakProbeStatistics,
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
        self.probe_stats = probe_stats
        if probe_stats.position_embeddings.shape != (self.n_patches, self.patch_dim):
            raise ValueError(
                "probe_stats position shape does not match this image geometry: "
                f"expected {(self.n_patches, self.patch_dim)}, got {tuple(probe_stats.position_embeddings.shape)}."
            )
        self.model_path = str(model_path)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
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
        self.adapter = SharedBinPatchAdapter(
            n_patches=n_patches,
            patch_dim=patch_dim,
            num_bins=probe_stats.num_bins,
            gap=probe_stats.gap,
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

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        patches = extract_image_patches(images, self.patch_size)
        normalized = normalize_patches_with_public_stats(patches, self.probe_stats.patch_stats)
        exposed = peftleak_expose_patches(patches, self.probe_stats)
        slot_indices = assign_peftleak_patch_slots(exposed, self.probe_stats)
        backbone_images = self._backbone_input_images(images)
        logits = self.backbone(backbone_images)
        adapter_scores = self.adapter(exposed, slot_indices)
        logits = logits.clone()
        logits[:, 0] = logits[:, 0] + adapter_scores.sum(dim=1)
        return logits, normalized, patches, exposed, slot_indices
    def adapter_parameter_names(self) -> list[str]:
        return self.adapter.parameter_names()

    def adapter_parameters_for_grad(self) -> list[nn.Parameter]:
        return self.adapter.parameters_for_grad()


TinyVitWithMaliciousAdapter = TorchvisionVitWithMaliciousAdapter


def _labels_for_shared_adapter_batch(
    labels: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    if labels is None:
        return torch.arange(batch_size, device=device) % int(n_classes)
    labels = labels.to(device=device, dtype=torch.long).view(-1)
    if labels.numel() != batch_size:
        raise ValueError(f"Expected {batch_size} labels, got {labels.numel()}.")
    return labels


def _build_shared_adapter_model(
    images: torch.Tensor,
    probe_stats: PeftLeakProbeStatistics,
    *,
    patch_size: int,
    adapter_hidden_dim: int,
    n_classes: int = 100,
    seed: int = 0,
    model_path: str = "torchvision_vit_small",
    finetuned_path: str | Path | None = None,
) -> TorchvisionVitWithMaliciousAdapter:
    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channels, height, width].")
    if int(probe_stats.patch_stats.patch_size) != int(patch_size):
        raise ValueError(f"stats.patch_size={probe_stats.patch_stats.patch_size} does not match patch_size={patch_size}.")
    batch_size, channels, height, width = images.shape
    if height != width:
        raise ValueError("vit_adapter mode expects square images.")
    probe_stats = move_peftleak_probe_statistics(probe_stats, device=images.device, dtype=images.dtype)
    model = TorchvisionVitWithMaliciousAdapter(
        batch_size=batch_size,
        channels=channels,
        image_size=height,
        patch_size=patch_size,
        adapter_hidden_dim=adapter_hidden_dim,
        probe_stats=probe_stats,
        n_classes=n_classes,
        seed=seed,
        model_path=model_path,
        finetuned_path=finetuned_path,
    ).to(device=images.device, dtype=images.dtype)
    model.eval()
    return model


def _compute_shared_adapter_gradients(
    model: TorchvisionVitWithMaliciousAdapter,
    images: torch.Tensor,
    *,
    labels: torch.Tensor | None = None,
    n_classes: int = 100,
) -> VitAdapterGradientResult:
    batch_size = int(images.shape[0])
    labels = _labels_for_shared_adapter_batch(
        labels,
        batch_size=batch_size,
        device=images.device,
        n_classes=n_classes,
    )
    logits, normalized, raw_patches, exposed, slot_indices = model(images)
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
        exposed_patches=exposed.detach(),
        slot_indices=slot_indices.detach(),
    )


def build_shared_adapter_gradients(
    images: torch.Tensor,
    probe_stats: PeftLeakProbeStatistics,
    *,
    labels: torch.Tensor | None = None,
    patch_size: int,
    adapter_hidden_dim: int,
    n_classes: int = 100,
    seed: int = 0,
    model_path: str = "torchvision_vit_small",
    finetuned_path: str | Path | None = None,
) -> VitAdapterGradientResult:
    model = _build_shared_adapter_model(
        images,
        probe_stats,
        patch_size=patch_size,
        adapter_hidden_dim=adapter_hidden_dim,
        n_classes=n_classes,
        seed=seed,
        model_path=model_path,
        finetuned_path=finetuned_path,
    )
    return _compute_shared_adapter_gradients(
        model,
        images,
        labels=labels,
        n_classes=n_classes,
    )


def build_shared_adapter_gradient_bundle(
    images: torch.Tensor,
    probe_stats: PeftLeakProbeStatistics,
    *,
    labels: torch.Tensor | None = None,
    patch_size: int,
    adapter_hidden_dim: int,
    n_classes: int = 100,
    seed: int = 0,
    model_path: str = "torchvision_vit_small",
    finetuned_path: str | Path | None = None,
) -> tuple[VitAdapterGradientResult, list[VitAdapterGradientResult]]:
    """Compute full-batch and per-example shared-adapter gradients on one model."""

    model = _build_shared_adapter_model(
        images,
        probe_stats,
        patch_size=patch_size,
        adapter_hidden_dim=adapter_hidden_dim,
        n_classes=n_classes,
        seed=seed,
        model_path=model_path,
        finetuned_path=finetuned_path,
    )
    full_result = _compute_shared_adapter_gradients(
        model,
        images,
        labels=labels,
        n_classes=n_classes,
    )
    labels = _labels_for_shared_adapter_batch(
        labels,
        batch_size=int(images.shape[0]),
        device=images.device,
        n_classes=n_classes,
    )
    per_example: list[VitAdapterGradientResult] = []
    for sample_idx in range(int(images.shape[0])):
        sample_result = _compute_shared_adapter_gradients(
            model,
            images[sample_idx : sample_idx + 1],
            labels=labels[sample_idx : sample_idx + 1],
            n_classes=n_classes,
        )
        if sample_result.names != full_result.names:
            raise ValueError("Per-example PEFTLeak adapter parameter names changed during DPSGD generation.")
        if len(sample_result.grads) != len(full_result.grads):
            raise ValueError("Per-example PEFTLeak gradient tuple length changed during DPSGD generation.")
        for full_grad, sample_grad in zip(full_result.grads, sample_result.grads):
            if full_grad is None or sample_grad is None:
                if full_grad is not sample_grad:
                    raise ValueError("Per-example PEFTLeak gradient optionality changed during DPSGD generation.")
                continue
            if full_grad.shape != sample_grad.shape:
                raise ValueError("Per-example PEFTLeak gradient shape changed during DPSGD generation.")
        per_example.append(sample_result)
    return full_result, per_example


def build_vit_adapter_gradients(
    images: torch.Tensor,
    stats: PeftLeakProbeStatistics,
    *,
    labels: torch.Tensor | None = None,
    patch_size: int,
    adapter_hidden_dim: int,
    n_classes: int = 100,
    seed: int = 0,
    model_path: str = "torchvision_vit_small",
    finetuned_path: str | Path | None = None,
) -> VitAdapterGradientResult:
    return build_shared_adapter_gradients(
        images,
        stats,
        labels=labels,
        patch_size=patch_size,
        adapter_hidden_dim=adapter_hidden_dim,
        n_classes=n_classes,
        seed=seed,
        model_path=model_path,
        finetuned_path=finetuned_path,
    )


def resolve_official_adapter_layer_count(adapter_layers: str, *, vit_config: str = "cifar_small") -> int:
    value = str(adapter_layers or "all").strip().lower()
    base_layers = 12 if str(vit_config).lower() == "vit_b_16" else 4
    if value == "all":
        return base_layers * 2
    if value == "msa" or value == "mlp":
        return base_layers
    if value == "first_n":
        return min(4, base_layers * 2)
    if value == "last_n":
        return min(4, base_layers * 2)
    raise ValueError(f"Unsupported official adapter layer selector: {adapter_layers!r}")


def _official_sample_tags(batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if batch_size <= 1:
        return torch.zeros(1, device=device, dtype=dtype)
    return torch.linspace(-1.0, 1.0, steps=int(batch_size), device=device, dtype=dtype)


def _tag_exposed_patches(exposed: torch.Tensor, sample_tags: torch.Tensor) -> torch.Tensor:
    if exposed.ndim != 3:
        raise ValueError("exposed patches must have shape [batch, n_patches, patch_dim].")
    tags = sample_tags.to(device=exposed.device, dtype=exposed.dtype).view(-1, 1, 1)
    if tags.shape[0] != exposed.shape[0]:
        raise ValueError("sample tag count must match batch size.")
    tag_plane = tags.expand(exposed.shape[0], exposed.shape[1], 1)
    return torch.cat((exposed, tag_plane), dim=-1)


class OfficialAlignedVitWithAdapters(nn.Module):
    """Frozen torchvision ViT backbone plus multi-layer official-aligned adapters."""

    _resolve_checkpoint_file = staticmethod(TorchvisionVitWithMaliciousAdapter._resolve_checkpoint_file)
    _load_backbone_checkpoint = TorchvisionVitWithMaliciousAdapter._load_backbone_checkpoint

    def __init__(
        self,
        *,
        channels: int,
        image_size: int,
        patch_size: int,
        probe_stats: PeftLeakProbeStatistics,
        n_classes: int,
        seed: int,
        model_path: str,
        finetuned_path: str | Path | None,
        adapter_layer_count: int,
        adapter_bottleneck_dim: int,
    ):
        super().__init__()
        patch_dim = channels * patch_size * patch_size
        n_patches = (image_size // patch_size) ** 2
        self.channels = int(channels)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.patch_dim = int(patch_dim)
        self.n_patches = int(n_patches)
        self.probe_stats = probe_stats
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.backbone = TorchvisionVitWithMaliciousAdapter._build_backbone(
                image_size=image_size,
                patch_size=patch_size,
                adapter_hidden_dim=max(1, adapter_bottleneck_dim),
                n_classes=n_classes,
                model_path=model_path,
        )
        if finetuned_path:
            self._load_backbone_checkpoint(finetuned_path)
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        self.adapter = OfficialAlignedPatchAdapter(
            n_layers=adapter_layer_count,
            n_patches=n_patches,
            patch_dim=patch_dim,
            num_bins=probe_stats.num_bins,
            bottleneck_dim=adapter_bottleneck_dim,
            gap=probe_stats.gap,
        )

    def _backbone_input_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[1] == 3:
            return images
        if images.shape[1] == 1:
            return images.repeat(1, 3, 1, 1)
        raise ValueError("Torchvision ViT backbone expects images with 1 or 3 channels.")

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        patches = extract_image_patches(images, self.patch_size)
        exposed = peftleak_expose_patches(patches, self.probe_stats)
        tags = _official_sample_tags(images.shape[0], device=images.device, dtype=images.dtype)
        tagged_exposed = _tag_exposed_patches(exposed, tags)
        slot_indices = assign_peftleak_patch_slots(exposed, self.probe_stats)
        logits = self.backbone(self._backbone_input_images(images))
        adapter_scores = self.adapter(tagged_exposed, slot_indices)
        logits = logits.clone()
        logits[:, 0] = logits[:, 0] + adapter_scores.sum(dim=1)
        return logits, patches, exposed, tagged_exposed, slot_indices

    def adapter_parameter_names(self) -> list[str]:
        return self.adapter.parameter_names()

    def adapter_parameters_for_grad(self) -> list[nn.Parameter]:
        return self.adapter.parameters_for_grad()


def build_official_aligned_adapter_gradients(
    images: torch.Tensor,
    probe_stats: PeftLeakProbeStatistics,
    *,
    labels: torch.Tensor | None = None,
    patch_size: int,
    n_classes: int = 100,
    seed: int = 0,
    model_path: str = "torchvision_vit_small",
    finetuned_path: str | Path | None = None,
    vit_config: str = "cifar_small",
    adapter_layers: str = "all",
    adapter_bottleneck_dim: int = 8,
) -> OfficialAlignedGradientResult:
    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channels, height, width].")
    batch_size, channels, height, width = images.shape
    if height != width:
        raise ValueError("official_vit_adapter mode expects square images.")
    layer_count = resolve_official_adapter_layer_count(adapter_layers, vit_config=vit_config)
    probe_stats = move_peftleak_probe_statistics(probe_stats, device=images.device, dtype=images.dtype)
    model = OfficialAlignedVitWithAdapters(
        channels=channels,
        image_size=height,
        patch_size=patch_size,
        probe_stats=probe_stats,
        n_classes=n_classes,
        seed=seed,
        model_path=model_path,
        finetuned_path=finetuned_path,
        adapter_layer_count=layer_count,
        adapter_bottleneck_dim=adapter_bottleneck_dim,
    ).to(device=images.device, dtype=images.dtype)
    model.eval()
    labels = _labels_for_shared_adapter_batch(
        labels,
        batch_size=int(batch_size),
        device=images.device,
        n_classes=n_classes,
    )
    logits, raw_patches, exposed, tagged_exposed, slot_indices = model(images)
    loss = F.cross_entropy(logits, labels)
    grads = torch.autograd.grad(
        loss,
        model.adapter_parameters_for_grad(),
        allow_unused=True,
    )
    return OfficialAlignedGradientResult(
        grads=tuple(None if grad is None else grad.detach() for grad in grads),
        names=model.adapter_parameter_names(),
        raw_patches=raw_patches.detach(),
        exposed_patches=exposed.detach(),
        tagged_exposed_patches=tagged_exposed.detach(),
        slot_indices=slot_indices.detach(),
        logits=logits.detach(),
        loss=float(loss.detach().item()),
        adapter_layer_count=int(layer_count),
        adapter_bottleneck_dim=int(adapter_bottleneck_dim),
        sample_tags=_official_sample_tags(int(batch_size), device=images.device, dtype=images.dtype).detach(),
    )


_NAMED_ADAPTER_RE = re.compile(r"sample_(\d+)\.patch_(\d+)\.(weight|bias)$")
_OFFICIAL_SHARED_RE = re.compile(r"official_adapter\.layer_(\d+)\.shared\.(weight|bias)$")


def _official_layer_weight_bias_pairs(
    grads: Sequence[torch.Tensor | None],
    names: Sequence[str],
) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
    by_layer: dict[int, dict[str, torch.Tensor]] = {}
    for grad, name in zip(grads, names):
        if grad is None:
            continue
        match = _OFFICIAL_SHARED_RE.search(str(name))
        if match is None:
            continue
        layer_idx = int(match.group(1))
        by_layer.setdefault(layer_idx, {})[match.group(2)] = grad
    pairs: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    for layer_idx in sorted(by_layer):
        parts = by_layer[layer_idx]
        if "weight" not in parts or "bias" not in parts:
            continue
        weight = parts["weight"]
        bias = parts["bias"]
        if weight.ndim != 2 or bias.ndim != 1 or weight.shape[0] != bias.shape[0]:
            raise ValueError(f"Official adapter layer {layer_idx} has invalid shared gradient shapes.")
        pairs.append((layer_idx, weight, bias))
    if not pairs:
        raise ValueError("No official-aligned adapter shared weight/bias gradient pairs were available.")
    return pairs


def recover_patches_from_official_adapter_grads(
    grads: Sequence[torch.Tensor | None],
    names: Sequence[str],
    probe_stats: PeftLeakProbeStatistics,
    *,
    batch_size: int,
    n_patches: int,
    grouping: str = "tag",
    slot_indices: torch.Tensor | None = None,
    tag_tolerance: float = 1e-3,
    eps: float = 1e-12,
) -> PeftLeakRecoveryResult:
    grouping = str(grouping or "tag").strip().lower()
    if grouping not in {"tag", "cluster", "oracle_debug"}:
        raise ValueError("official adapter grouping must be 'tag', 'cluster', or 'oracle_debug'.")
    pairs = _official_layer_weight_bias_pairs(grads, names)
    first_weight = pairs[0][1]
    device = first_weight.device
    dtype = first_weight.dtype
    stats = move_peftleak_probe_statistics(probe_stats, device=device, dtype=dtype)
    n_probe_patches = int(stats.position_embeddings.shape[0])
    raw_patch_dim = int(stats.position_embeddings.shape[1])
    tagged_dim = raw_patch_dim + 1
    if int(n_patches) != n_probe_patches:
        raise ValueError(f"Expected n_patches={n_probe_patches}; got {n_patches}.")

    row_stride = int(stats.gap) + 2
    n_slots = n_probe_patches * int(stats.num_bins)
    expected_rows = n_slots * row_stride
    tag_values: list[float] = []
    candidate_records: list[tuple[float, int, int, int, torch.Tensor]] = []
    nonzero_slot_count = 0

    for layer_idx, weight_grad, bias_grad in pairs:
        if weight_grad.shape != (expected_rows, tagged_dim) or bias_grad.shape[0] != expected_rows:
            raise ValueError(
                "Official adapter gradient shape mismatch: "
                f"expected weight={(expected_rows, tagged_dim)}, bias={(expected_rows,)}, "
                f"got weight={tuple(weight_grad.shape)}, bias={tuple(bias_grad.shape)}."
            )
        for slot in range(n_slots):
            row_a = slot * row_stride
            row_b = row_a + int(stats.gap) + 1
            denom = (bias_grad[row_a] - bias_grad[row_b]).float()
            if float(denom.detach().abs().item()) <= eps:
                continue
            nonzero_slot_count += 1
            tagged = (weight_grad[row_a].float() - weight_grad[row_b].float()) / denom
            tag = float(tagged[-1].detach().item())
            exposed = tagged[:-1]
            patch_idx = slot // int(stats.num_bins)
            normalized = (exposed - stats.position_embeddings[patch_idx].float()) / float(stats.embed_scale)
            raw_patch = denormalize_patches_with_public_stats(
                normalized.view(1, 1, -1).to(device=device, dtype=dtype),
                stats.patch_stats,
            ).view(-1)
            tag_values.append(tag)
            candidate_records.append((tag, int(patch_idx), int(layer_idx), int(slot), raw_patch.to(device=device, dtype=dtype)))

    if candidate_records:
        candidate_patches = torch.stack([record[4] for record in candidate_records], dim=0)
        candidate_slots = torch.tensor(
            [record[3] for record in candidate_records],
            device=device,
            dtype=torch.long,
        )
        candidate_position_indices = torch.tensor(
            [record[1] for record in candidate_records],
            device=device,
            dtype=torch.long,
        )
    else:
        candidate_patches = torch.empty(0, raw_patch_dim, device=device, dtype=dtype)
        candidate_slots = torch.empty(0, device=device, dtype=torch.long)
        candidate_position_indices = torch.empty(0, device=device, dtype=torch.long)

    sorted_tags: list[float] = []
    for tag in sorted(tag_values):
        if not sorted_tags or abs(tag - sorted_tags[-1]) > float(tag_tolerance):
            sorted_tags.append(tag)
    tag_to_row = {tag: idx for idx, tag in enumerate(sorted_tags[: int(batch_size)])}

    recovered = torch.zeros(int(batch_size), n_probe_patches, raw_patch_dim, device=device, dtype=dtype)
    mask = torch.zeros(int(batch_size), n_probe_patches, device=device, dtype=torch.bool)
    buckets: dict[tuple[int, int], list[torch.Tensor]] = {}
    if grouping in {"tag", "oracle_debug"}:
        for tag, patch_idx, _layer_idx, _slot, raw_patch in candidate_records:
            nearest = None
            nearest_dist = None
            for known_tag in tag_to_row:
                dist = abs(tag - known_tag)
                if nearest is None or dist < nearest_dist:
                    nearest = known_tag
                    nearest_dist = dist
            if nearest is None or nearest_dist is None or nearest_dist > max(float(tag_tolerance), 1e-6):
                continue
            row_idx = tag_to_row[nearest]
            buckets.setdefault((row_idx, patch_idx), []).append(raw_patch)

    collision_patch_count = 0
    for (row_idx, patch_idx), values in buckets.items():
        if len(values) > 1:
            collision_patch_count += len(values) - 1
        recovered[row_idx, patch_idx] = torch.stack(values, dim=0).mean(dim=0)
        mask[row_idx, patch_idx] = True

    recovered_patch_count = int(mask.sum().item())
    unresolved_patch_count = int(batch_size) * n_probe_patches - recovered_patch_count
    position_counts = torch.bincount(candidate_position_indices, minlength=n_probe_patches)
    slot_counts = None
    oracle_recovered = None
    oracle_mask = None
    oracle_recovered_patch_count = None
    oracle_collision_patch_count = None
    oracle_unresolved_patch_count = None
    if slot_indices is not None:
        slot_indices = slot_indices.to(device=device, dtype=torch.long)
        if slot_indices.shape != (int(batch_size), n_probe_patches):
            raise ValueError(
                f"slot_indices must have shape {(int(batch_size), n_probe_patches)}; "
                f"got {tuple(slot_indices.shape)}."
            )
        if int(slot_indices.min().item()) < 0 or int(slot_indices.max().item()) >= n_slots:
            raise ValueError("slot_indices contain an out-of-range official PEFTLeak slot.")
        flat_slots = slot_indices.reshape(-1)
        slot_counts = torch.bincount(flat_slots, minlength=n_slots)
        by_slot: dict[int, list[torch.Tensor]] = {}
        for _tag, _patch_idx, _layer_idx, slot, raw_patch in candidate_records:
            by_slot.setdefault(int(slot), []).append(raw_patch)
        oracle_recovered = torch.zeros(int(batch_size), n_probe_patches, raw_patch_dim, device=device, dtype=dtype)
        oracle_mask = torch.zeros(int(batch_size), n_probe_patches, device=device, dtype=torch.bool)
        oracle_recovered_patch_count = 0
        oracle_collision_patch_count = 0
        oracle_unresolved_patch_count = 0
        for sample_idx in range(int(batch_size)):
            for patch_idx in range(n_probe_patches):
                slot = int(slot_indices[sample_idx, patch_idx].item())
                if int(slot_counts[slot].item()) > 1:
                    oracle_collision_patch_count += 1
                    continue
                values = by_slot.get(slot)
                if not values:
                    oracle_unresolved_patch_count += 1
                    continue
                oracle_recovered[sample_idx, patch_idx] = torch.stack(values, dim=0).mean(dim=0)
                oracle_mask[sample_idx, patch_idx] = True
                oracle_recovered_patch_count += 1
    metadata = {
        "n_slots": int(n_slots),
        "row_stride": int(row_stride),
        "nonzero_slot_count": int(nonzero_slot_count),
        "candidate_slot_count": int(len(candidate_records)),
        "ambiguous_position_count": int((position_counts > int(batch_size)).sum().item()),
        "empty_position_count": int((position_counts == 0).sum().item()),
        "official_tag_count": int(len(sorted_tags)),
        "official_layer_count": int(len(pairs)),
        "official_grouping_is_cluster": int(grouping == "cluster"),
        "official_grouping_is_oracle_debug": int(grouping == "oracle_debug"),
    }
    return PeftLeakRecoveryResult(
        recovered_patches=recovered,
        recovery_mask=mask,
        candidate_patch_count=int(len(candidate_records)),
        recovered_patch_count=recovered_patch_count,
        collision_patch_count=collision_patch_count,
        unresolved_patch_count=max(0, unresolved_patch_count),
        candidate_patches=candidate_patches,
        candidate_slots=candidate_slots,
        candidate_position_indices=candidate_position_indices,
        slot_indices=slot_indices,
        slot_counts=slot_counts,
        oracle_recovered_patches=oracle_recovered,
        oracle_recovery_mask=oracle_mask,
        oracle_recovered_patch_count=oracle_recovered_patch_count,
        oracle_collision_patch_count=oracle_collision_patch_count,
        oracle_unresolved_patch_count=oracle_unresolved_patch_count,
        raw_candidate_metadata=metadata,
    )


def _shared_weight_bias_from_named_grads(
    grads: Sequence[torch.Tensor | None],
    names: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = None
    bias = None
    for grad, name in zip(grads, names):
        if grad is None:
            continue
        if str(name).endswith("shared.weight"):
            weight = grad
        elif str(name).endswith("shared.bias"):
            bias = grad
    if weight is None or bias is None:
        raise ValueError("Shared PEFTLeak adapter weight/bias gradients are required for recovery.")
    if weight.ndim != 2 or bias.ndim != 1:
        raise ValueError("Shared adapter gradients must have shapes [rows, patch_dim] and [rows].")
    if weight.shape[0] != bias.shape[0]:
        raise ValueError("Shared adapter weight/bias row counts do not match.")
    return weight, bias


def recover_patches_from_shared_adapter_grads(
    grads: Sequence[torch.Tensor | None],
    names: Sequence[str],
    probe_stats: PeftLeakProbeStatistics,
    *,
    batch_size: int | None = None,
    n_patches: int | None = None,
    slot_indices: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> PeftLeakRecoveryResult:
    weight_grad, bias_grad = _shared_weight_bias_from_named_grads(grads, names)
    device = weight_grad.device
    dtype = weight_grad.dtype
    stats = move_peftleak_probe_statistics(probe_stats, device=device, dtype=dtype)
    n_probe_patches = int(stats.position_embeddings.shape[0])
    patch_dim = int(stats.position_embeddings.shape[1])
    if n_patches is None:
        n_patches = n_probe_patches
    if int(n_patches) != n_probe_patches:
        raise ValueError(f"Expected n_patches={n_probe_patches}; got {n_patches}.")
    row_stride = int(stats.gap) + 2
    n_slots = n_probe_patches * int(stats.num_bins)
    expected_rows = n_slots * row_stride
    if weight_grad.shape != (expected_rows, patch_dim) or bias_grad.shape[0] != expected_rows:
        raise ValueError(
            "Shared adapter gradient shape mismatch: "
            f"expected weight={(expected_rows, patch_dim)}, bias={(expected_rows,)}, "
            f"got weight={tuple(weight_grad.shape)}, bias={tuple(bias_grad.shape)}."
        )

    candidates: dict[int, torch.Tensor] = {}
    candidate_slots_list: list[int] = []
    candidate_position_list: list[int] = []
    candidate_patch_list: list[torch.Tensor] = []
    nonzero_slot_count = 0
    for slot in range(n_slots):
        row_a = slot * row_stride
        row_b = row_a + int(stats.gap) + 1
        denom = (bias_grad[row_a] - bias_grad[row_b]).float()
        if float(denom.detach().abs().item()) <= eps:
            continue
        nonzero_slot_count += 1
        exposed = (weight_grad[row_a].float() - weight_grad[row_b].float()) / denom
        patch_idx = slot // int(stats.num_bins)
        normalized = (exposed - stats.position_embeddings[patch_idx].float()) / float(stats.embed_scale)
        raw_patch = denormalize_patches_with_public_stats(
            normalized.view(1, 1, -1).to(device=device, dtype=dtype),
            stats.patch_stats,
        ).view(-1)
        candidates[slot] = raw_patch
        candidate_slots_list.append(int(slot))
        candidate_position_list.append(int(patch_idx))
        candidate_patch_list.append(raw_patch.to(device=device, dtype=dtype))

    if candidate_patch_list:
        candidate_patches = torch.stack(candidate_patch_list, dim=0)
        candidate_slots = torch.tensor(candidate_slots_list, device=device, dtype=torch.long)
        candidate_position_indices = torch.tensor(candidate_position_list, device=device, dtype=torch.long)
    else:
        candidate_patches = torch.empty(0, patch_dim, device=device, dtype=dtype)
        candidate_slots = torch.empty(0, device=device, dtype=torch.long)
        candidate_position_indices = torch.empty(0, device=device, dtype=torch.long)

    if slot_indices is not None:
        slot_indices = slot_indices.to(device=device, dtype=torch.long)
        if slot_indices.ndim != 2:
            raise ValueError("slot_indices must have shape [batch, n_patches].")
        if batch_size is None:
            batch_size = int(slot_indices.shape[0])
        if slot_indices.shape != (int(batch_size), n_probe_patches):
            raise ValueError(
                f"slot_indices must have shape {(int(batch_size), n_probe_patches)}; "
                f"got {tuple(slot_indices.shape)}."
            )
        if int(slot_indices.min().item()) < 0 or int(slot_indices.max().item()) >= n_slots:
            raise ValueError("slot_indices contain an out-of-range PEFTLeak slot.")
    elif batch_size is None:
        batch_size = 1

    recovered = torch.zeros(1, n_probe_patches, patch_dim, device=device, dtype=dtype)
    mask = torch.zeros(1, n_probe_patches, device=device, dtype=torch.bool)
    slot_counts = None
    position_candidate_counts = torch.bincount(candidate_position_indices, minlength=n_probe_patches)
    ambiguous_position_count = int((position_candidate_counts > 1).sum().item())
    empty_position_count = int((position_candidate_counts == 0).sum().item())
    recovered_patch_count = 0
    if int(batch_size) == 1:
        for patch_idx in range(n_probe_patches):
            if int(position_candidate_counts[patch_idx].item()) != 1:
                continue
            candidate_offset = int((candidate_position_indices == patch_idx).nonzero(as_tuple=False)[0].item())
            recovered[0, patch_idx] = candidate_patches[candidate_offset]
            mask[0, patch_idx] = True
            recovered_patch_count += 1
    unresolved_patch_count = (
        n_probe_patches - recovered_patch_count
        if int(batch_size) == 1
        else int(batch_size) * n_probe_patches
    )

    oracle_recovered = None
    oracle_mask = None
    oracle_recovered_patch_count = None
    oracle_collision_patch_count = None
    oracle_unresolved_patch_count = None
    if slot_indices is not None:
        flat_slots = slot_indices.reshape(-1)
        slot_counts = torch.bincount(flat_slots, minlength=n_slots)
        oracle_recovered = torch.zeros(int(batch_size), n_probe_patches, patch_dim, device=device, dtype=dtype)
        oracle_mask = torch.zeros(int(batch_size), n_probe_patches, device=device, dtype=torch.bool)
        oracle_recovered_patch_count = 0
        oracle_collision_patch_count = 0
        oracle_unresolved_patch_count = 0
        for sample_idx in range(int(batch_size)):
            for patch_idx in range(n_probe_patches):
                slot = int(slot_indices[sample_idx, patch_idx].item())
                count = int(slot_counts[slot].item())
                if count > 1:
                    oracle_collision_patch_count += 1
                    continue
                candidate = candidates.get(slot)
                if candidate is None:
                    oracle_unresolved_patch_count += 1
                    continue
                oracle_recovered[sample_idx, patch_idx] = candidate.to(device=device, dtype=dtype)
                oracle_mask[sample_idx, patch_idx] = True
                oracle_recovered_patch_count += 1

    metadata = {
        "n_slots": int(n_slots),
        "row_stride": int(row_stride),
        "nonzero_slot_count": int(nonzero_slot_count),
        "candidate_slot_count": int(len(candidates)),
        "ambiguous_position_count": int(ambiguous_position_count),
        "empty_position_count": int(empty_position_count),
    }
    return PeftLeakRecoveryResult(
        recovered_patches=recovered,
        recovery_mask=mask,
        candidate_patch_count=int(len(candidates)),
        recovered_patch_count=int(recovered_patch_count),
        collision_patch_count=None,
        unresolved_patch_count=int(max(0, unresolved_patch_count)),
        candidate_patches=candidate_patches,
        candidate_slots=candidate_slots,
        candidate_position_indices=candidate_position_indices,
        slot_indices=slot_indices,
        slot_counts=slot_counts,
        oracle_recovered_patches=oracle_recovered,
        oracle_recovery_mask=oracle_mask,
        oracle_recovered_patch_count=oracle_recovered_patch_count,
        oracle_collision_patch_count=oracle_collision_patch_count,
        oracle_unresolved_patch_count=oracle_unresolved_patch_count,
        raw_candidate_metadata=metadata,
    )


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


def simple_ssim(recovered: torch.Tensor, reference: torch.Tensor, *, data_range: float = 1.0) -> float:
    """Small dependency-free global SSIM approximation for experiment summaries."""

    x = recovered.float().reshape(recovered.shape[0], -1)
    y = reference.float().reshape(reference.shape[0], -1)
    c1 = (0.01 * float(data_range)) ** 2
    c2 = (0.03 * float(data_range)) ** 2
    mu_x = x.mean(dim=1)
    mu_y = y.mean(dim=1)
    var_x = x.var(dim=1, unbiased=False)
    var_y = y.var(dim=1, unbiased=False)
    cov_xy = ((x - mu_x.view(-1, 1)) * (y - mu_y.view(-1, 1))).mean(dim=1)
    score = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + c1) * (var_x + var_y + c2)
    )
    return float(score.clamp(-1.0, 1.0).mean().item())
