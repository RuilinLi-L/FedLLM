"""
Defense baselines for gradient inversion experiments (FL-LLM.md).

Post-gradient: noise, topk, compression, optional random mask (defense_pct_mask).
Direct gradient generation: dpsgd, soteria, mixup.
DAGER-specific: dager defense methods targeting DAGER attack assumptions.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .dager_defense import apply_dager_defense
except ImportError:
    # Fallback for when dager_defense is not available
    def apply_dager_defense(grads, args, model_wrapper=None, batch=None, labels=None, layer_names=None):
        raise ImportError("DAGER defense module not available")

try:
    from .lrb_defense import apply_lrb_defense
except ImportError:
    def apply_lrb_defense(grads, args, layer_names=None):
        raise ImportError("LRB defense module not available")


SPECIAL_GRADIENT_DEFENSES = {"mixup", "dpsgd", "soteria"}


def requires_gradient_generation_defense(defense: str) -> bool:
    return defense in SPECIAL_GRADIENT_DEFENSES


def uses_noisy_gradient_decoding(args) -> bool:
    """Use outlier-based L1/L2 decoding paths only when actual Gaussian noise is present."""
    defense = getattr(args, "defense", "none")
    sigma = getattr(args, "defense_noise", None)
    if sigma is None:
        return False
    if float(sigma) <= 0:
        return False
    if defense in ("noise", "dpsgd"):
        return True
    if defense == "none":
        return True
    return False


def _make_generator(seed: int, device: torch.device) -> torch.Generator:
    """Create a seeded generator on the correct device."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def _apply_random_mask(grads, pct_mask: float, seed: int = 0):
    """Element-wise: keep with probability (1 - pct_mask), same semantics as train.py."""
    out = []
    for idx, g in enumerate(grads):
        if g is None:
            out.append(None)
            continue
        gen = _make_generator(seed + 104729 * (idx + 1), g.device)
        mask = (torch.rand(g.shape, device=g.device, dtype=g.dtype, generator=gen) > pct_mask).float()
        out.append(g * mask)
    return tuple(out)


def noise_injection(grads, sigma: float, seed: int = 0):
    out = []
    for idx, g in enumerate(grads):
        if g is None:
            out.append(None)
            continue
        gen = _make_generator(seed + 104729 * (idx + 1), g.device)
        noise = torch.randn(g.shape, device=g.device, dtype=g.dtype, generator=gen)
        out.append(g + noise * sigma)
    return tuple(out)


def _average_grad_list(grad_list):
    if not grad_list:
        return tuple()

    n_samples = len(grad_list)
    out = []
    for tensors in zip(*grad_list):
        ref = next((tensor for tensor in tensors if tensor is not None), None)
        if ref is None:
            out.append(None)
            continue
        acc = torch.zeros_like(ref)
        for tensor in tensors:
            if tensor is not None:
                acc = acc + tensor.to(device=ref.device, dtype=ref.dtype)
        out.append(acc / n_samples)
    return tuple(out)


def dpsgd_defense(per_example_grads, max_norm: float, sigma: float, seed: int = 0):
    """Faithful DP-SGD: per-example clip, mean aggregation, then Gaussian noise."""
    if not per_example_grads:
        return tuple()

    clipped = []
    for sample_grads in per_example_grads:
        flat_parts = [g.detach().float().flatten() for g in sample_grads if g is not None]
        if not flat_parts:
            clipped.append(sample_grads)
            continue
        total_norm = torch.cat(flat_parts).norm(2)
        clip_coef = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
        clipped.append(
            tuple(
                None if g is None else g * clip_coef.to(device=g.device, dtype=g.dtype)
                for g in sample_grads
            )
        )

    avg = _average_grad_list(clipped)
    noise_std = float(sigma) * float(max_norm) / float(len(per_example_grads))
    if noise_std > 0:
        avg = noise_injection(avg, noise_std, seed=seed)
    return avg


def topk_sparsification(grads, keep_ratio: float):
    """Keep top keep_ratio fraction of |gradient| elements per tensor; zero the rest."""
    if keep_ratio <= 0 or keep_ratio > 1:
        raise ValueError("defense_topk_ratio must be in (0, 1]")
    out = []
    for g in grads:
        if g is None:
            out.append(None)
            continue
        flat = g.flatten()
        k = max(1, int(flat.numel() * keep_ratio))
        _, idx = torch.topk(flat.abs(), k)
        mask = torch.zeros_like(flat)
        mask[idx] = 1.0
        out.append((flat * mask).view_as(g))
    return tuple(out)


def gradient_compression(grads, n_bits: int):
    """Per-tensor uniform quantization to n_bits (symmetric around zero)."""
    if n_bits < 1:
        raise ValueError("defense_n_bits must be >= 1")
    levels = 2**n_bits - 1
    out = []
    for g in grads:
        if g is None:
            out.append(None)
            continue
        g_f = g.float()
        max_abs = g_f.abs().max().clamp(min=1e-12)
        scaled = g_f / max_abs
        q = torch.round((scaled + 1.0) * 0.5 * levels) / levels * 2.0 - 1.0
        q = q * max_abs
        out.append(q.to(dtype=g.dtype))
    return tuple(out)


def soteria_defense(model_wrapper, batch, labels, args, create_graph=False):
    """
    Representation-side Soteria defense.

    For each sample, score classifier-input representation dimensions by
    |r_i| / (||dr_i / dX|| + eps), prune the highest-scoring fraction, recompute
    the sample loss with the masked representation, then average the resulting
    per-example gradients.
    """
    if getattr(args, "task", None) != "seq_class":
        raise NotImplementedError("Soteria baseline is only supported for task=seq_class.")
    if getattr(args, "train_method", "full") != "full":
        raise NotImplementedError("Soteria baseline is only supported for train_method=full.")

    pruning_rate = float(getattr(args, "defense_soteria_pruning_rate", 60.0))
    if pruning_rate < 0 or pruning_rate > 100:
        raise ValueError("defense_soteria_pruning_rate must be in [0, 100].")

    sample_dims = getattr(args, "defense_soteria_sample_dims", None)
    base_seed = int(getattr(args, "rng_seed", 0))
    eps = 1e-12

    def sample_grad_fn(sample_batch, sample_labels, sample_idx=0, create_graph=False):
        model_wrapper.model.zero_grad(set_to_none=True)
        # Preserve the parameter-to-embedding graph when it exists so proxy
        # metrics still see the full defended gradient structure.
        inputs_embeds = model_wrapper._seq_class_input_embeds(sample_batch)
        if not inputs_embeds.requires_grad:
            inputs_embeds = inputs_embeds.detach()
            inputs_embeds.requires_grad_(True)
        _, representation = model_wrapper._seq_class_logits_from_embeds(sample_batch, inputs_embeds)
        representation = representation.squeeze(0)

        dim = representation.shape[-1]
        dims_to_score = list(range(dim))
        if sample_dims is not None and 0 < sample_dims < dim:
            rng = np.random.RandomState(base_seed + sample_idx)
            dims_to_score = sorted(rng.choice(dim, size=sample_dims, replace=False).tolist())

        scores = torch.full((dim,), float("-inf"), device=representation.device, dtype=torch.float32)
        for feature_idx in dims_to_score:
            grad_emb, = torch.autograd.grad(
                representation[feature_idx],
                inputs_embeds,
                retain_graph=True,
                allow_unused=True,
                create_graph=False,
            )
            sensitivity = float(grad_emb.detach().float().norm().item()) if grad_emb is not None else 0.0
            scores[feature_idx] = representation[feature_idx].detach().abs().float() / (sensitivity + eps)

        prune_count = int(np.ceil(len(dims_to_score) * pruning_rate / 100.0))
        mask = torch.ones(dim, device=representation.device, dtype=representation.dtype)
        if prune_count > 0:
            dims_tensor = torch.tensor(dims_to_score, device=representation.device, dtype=torch.long)
            top_idx = torch.topk(scores[dims_tensor], k=prune_count, largest=True).indices
            mask[dims_tensor[top_idx]] = 0.0

        model_wrapper.model.zero_grad(set_to_none=True)
        logits, _ = model_wrapper._seq_class_logits_from_embeds(
            sample_batch,
            inputs_embeds,
            representation_mask=mask.unsqueeze(0),
        )
        loss = F.cross_entropy(logits, sample_labels.view(-1))
        return torch.autograd.grad(
            loss,
            model_wrapper.trainable_parameters(),
            create_graph=create_graph,
            allow_unused=True,
        )

    grad_list = model_wrapper.compute_per_example_grads(
        batch,
        labels,
        create_graph=create_graph,
        sample_grad_fn=sample_grad_fn,
    )
    return _average_grad_list(grad_list)


def apply_defense(grads, args, model_wrapper=None, batch=None, labels=None):
    """
    Apply selected defense to gradients (or compute defended gradients directly).

    Returns: gradient tuple (same structure as compute_grads).
    """
    defense = getattr(args, "defense", "none")
    seed = int(getattr(args, "rng_seed", 0))

    if defense in SPECIAL_GRADIENT_DEFENSES:
        if model_wrapper is None or batch is None or labels is None:
            raise ValueError(f"{defense} requires model_wrapper, batch, labels")

        if defense == "mixup":
            g = model_wrapper.compute_grads_mixup(batch, labels)
        elif defense == "dpsgd":
            sigma = getattr(args, "defense_noise", None)
            if sigma is None:
                raise ValueError("--defense dpsgd requires --defense_noise as noise multiplier")
            per_example_grads = model_wrapper.compute_per_example_grads(batch, labels)
            g = dpsgd_defense(
                per_example_grads,
                float(args.defense_clip_norm),
                float(sigma),
                seed=seed,
            )
        elif defense == "soteria":
            g = soteria_defense(model_wrapper, batch, labels, args)
        else:
            raise ValueError(f"Unknown direct-generation defense: {defense}")

        if getattr(args, "defense_pct_mask", None) is not None:
            g = _apply_random_mask(g, float(args.defense_pct_mask), seed=seed)
        return g

    if grads is None:
        raise ValueError("apply_defense: grads is None but defense does not generate gradients directly")

    if (
        defense == "none"
        and getattr(args, "defense_noise", None) is None
        and getattr(args, "defense_pct_mask", None) is None
    ):
        return grads

    g = grads
    if defense == "none":
        if getattr(args, "defense_noise", None) is not None:
            g = noise_injection(g, float(args.defense_noise), seed=seed)
    elif defense == "noise":
        sigma = args.defense_noise
        if sigma is None:
            raise ValueError("--defense noise requires --defense_noise sigma")
        g = noise_injection(g, float(sigma), seed=seed)
    elif defense == "topk":
        g = topk_sparsification(g, float(args.defense_topk_ratio))
    elif defense == "compression":
        g = gradient_compression(g, int(args.defense_n_bits))
    elif defense == "dager":
        if model_wrapper is None or batch is None or labels is None:
            raise ValueError("dager defense requires model_wrapper, batch, labels")
        layer_names = []
        for name, p in model_wrapper.model.named_parameters():
            if p.requires_grad:
                layer_names.append(name)
        g = apply_dager_defense(g, args, model_wrapper, batch, labels, layer_names)
    elif defense == "lrb":
        layer_names = None
        if model_wrapper is not None:
            layer_names = []
            for name, p in model_wrapper.model.named_parameters():
                if p.requires_grad:
                    layer_names.append(name)
        g = apply_lrb_defense(g, args, layer_names=layer_names)
    else:
        raise ValueError(f"Unknown defense: {defense}")

    if getattr(args, "defense_pct_mask", None) is not None:
        g = _apply_random_mask(g, float(args.defense_pct_mask), seed=seed)

    return g
