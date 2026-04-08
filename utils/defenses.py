"""
Defense baselines for gradient inversion experiments (FL-LLM.md).

Post-gradient: noise, dpsgd, topk, compression, optional random mask (defense_pct_mask).
Pre-gradient / special: soteria (prune classifier grads using representation sensitivity), mixup.
DAGER-specific: dager defense methods targeting DAGER attack assumptions.
"""
from __future__ import annotations

import numpy as np
import torch

try:
    from .dager_defense import apply_dager_defense
except ImportError:
    # Fallback for when dager_defense is not available
    def apply_dager_defense(grads, args, model_wrapper=None, batch=None, labels=None, layer_names=None):
        raise ImportError("DAGER defense module not available")


def uses_noisy_gradient_decoding(args) -> bool:
    """Use outlier-based L1/L2 decoding paths (same as legacy --defense_noise)."""
    defense = getattr(args, "defense", "none")
    if defense in ("noise", "dpsgd") and getattr(args, "defense_noise", None) is not None:
        return True
    if defense == "none" and getattr(args, "defense_noise", None) is not None:
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
    for g in grads:
        if g is None:
            out.append(None)
            continue
        gen = _make_generator(seed, g.device)
        mask = (torch.rand(g.shape, device=g.device, dtype=g.dtype, generator=gen) > pct_mask).float()
        out.append(g * mask)
    return tuple(out)


def noise_injection(grads, sigma: float, seed: int = 0):
    out = []
    for g in grads:
        if g is None:
            out.append(None)
            continue
        gen = _make_generator(seed, g.device)
        noise = torch.randn(g.shape, device=g.device, dtype=g.dtype, generator=gen)
        out.append(g + noise * sigma)
    return tuple(out)


def dpsgd_defense(grads, max_norm: float, sigma: float, seed: int = 0):
    """Batch-level L2 clip of stacked gradient norms + Gaussian noise (DP-SGD style)."""
    flat_parts = []
    for g in grads:
        if g is None:
            continue
        flat_parts.append(g.flatten())
    if not flat_parts:
        return grads
    stacked = torch.cat(flat_parts)
    total_norm = stacked.norm(2)
    clip_coef = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
    out = []
    for g in grads:
        if g is None:
            out.append(None)
            continue
        clipped = g * clip_coef
        gen = _make_generator(seed, g.device)
        noise = torch.randn(g.shape, device=g.device, dtype=g.dtype, generator=gen)
        out.append(clipped + noise * sigma)
    return tuple(out)


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


def _classifier_param_name(model_path: str) -> str:
    if "bert" in model_path.lower():
        return "classifier.weight"
    return "score.weight"


def _apply_soteria_hidden_mask(
    g: torch.Tensor | None,
    mask: torch.Tensor,
    hidden_dim: int,
) -> torch.Tensor | None:
    """
    Zero out gradient components along axes that align with hidden_dim, using
    a per-feature keep mask from Soteria (length hidden_dim).

    Matches last dim (*, H), first dim (H, *), or 1D (H,) biases.
    """
    if g is None:
        return None
    if g.dim() == 1 and g.shape[0] == hidden_dim:
        return g * mask
    if g.shape[-1] == hidden_dim:
        view_shape = (1,) * (g.dim() - 1) + (hidden_dim,)
        return g * mask.view(*view_shape)
    if g.shape[0] == hidden_dim:
        view_shape = (hidden_dim,) + (1,) * (g.dim() - 1)
        return g * mask.view(*view_shape)
    return g


def soteria_defense(
    grads,
    model_wrapper,
    batch,
    labels,
    args,
):
    """
    Transformer adaptation of Soteria: score hidden dimensions by ||dL/dEmb|| when L = sum_t h[t,f],
    prune features below the pruning_rate percentile of scores.

    The keep mask is applied to every trainable gradient tensor whose shape aligns with the
    model hidden size (same axes as classifier columns/rows), so transformer blocks used by
    DAGER are perturbed—not only the classifier head.
    """
    if getattr(args, "train_method", "full") == "lora":
        raise NotImplementedError("Soteria baseline is not supported with train_method=lora in this integration.")

    pruning_rate = float(getattr(args, "defense_soteria_pruning_rate", 60.0))
    sample_dims = getattr(args, "defense_soteria_sample_dims", None)
    model = model_wrapper.model
    device = next(model.parameters()).device
    param_name = _classifier_param_name(args.model_path)

    idx_map = {}
    i = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            idx_map[name] = i
            i += 1
    if param_name not in idx_map:
        raise RuntimeError(f"Soteria: could not find {param_name} in trainable parameters")
    cls_idx = idx_map[param_name]

    grad_list = list(grads)

    model.eval()
    b = {k: v.to(device) for k, v in batch.items()}
    lab = labels.to(device) if not isinstance(labels, torch.Tensor) else labels.to(device)
    if lab.dim() > 1:
        lab = lab.view(-1)

    emb = None
    h_pool = None

    if args.model_path in ["gpt2", "openai-community/gpt2-large"]:
        wte = model.transformer.wte
        wpe = model.transformer.wpe
        pos = torch.arange(b["input_ids"].size(1), device=device).unsqueeze(0).expand_as(b["input_ids"])
        emb = wte(b["input_ids"]) + wpe(pos)
        emb = emb.requires_grad_(True)
        tr_out = model.transformer(inputs_embeds=emb, attention_mask=b.get("attention_mask"))
        h_pool = tr_out.last_hidden_state.mean(dim=0)
    elif args.model_path in ["bert-base-uncased"]:
        bert = model.bert
        emb_layer = bert.embeddings
        input_ids = b["input_ids"]
        token_type_ids = b.get("token_type_ids", torch.zeros_like(input_ids))
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand_as(input_ids)
        word = emb_layer.word_embeddings(input_ids)
        pos_e = emb_layer.position_embeddings(position_ids)
        tok_e = emb_layer.token_type_embeddings(token_type_ids)
        emb = emb_layer.LayerNorm(word + pos_e + tok_e)
        emb = emb_layer.dropout(emb)
        emb = emb.requires_grad_(True)
        raw_mask = b.get("attention_mask")
        ext_mask = bert.get_extended_attention_mask(raw_mask, input_ids.shape) if raw_mask is not None else None
        enc_out = bert.encoder(emb, attention_mask=ext_mask)
        h_pool = enc_out.last_hidden_state.mean(dim=0)
    elif args.model_path in [
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-70b-hf",
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Meta-Llama-3.1-8B",
        "meta-llama/Meta-Llama-3-70B",
    ]:
        llama = model.model
        input_ids = b["input_ids"]
        emb = llama.embed_tokens(input_ids).requires_grad_(True)
        attn = b.get("attention_mask")
        pos = torch.arange(input_ids.size(1), device=device).unsqueeze(0).expand_as(input_ids)
        out = llama(inputs_embeds=emb, attention_mask=attn, position_ids=pos)
        h_pool = out.last_hidden_state.mean(dim=0)
    else:
        raise NotImplementedError(f"Soteria not implemented for model_path={args.model_path}")

    dim = h_pool.size(-1)
    dims_to_score = list(range(dim))
    if sample_dims is not None and sample_dims < dim:
        rng = np.random.RandomState(getattr(args, "rng_seed", 0))
        dims_to_score = sorted(rng.choice(dim, size=sample_dims, replace=False).tolist())

    scores = np.zeros(dim, dtype=np.float64)
    for f in dims_to_score:
        model.zero_grad(set_to_none=True)
        if emb.grad is not None:
            emb.grad.zero_()
        scalar = h_pool[:, f].sum()
        g_emb, = torch.autograd.grad(
            scalar, emb, retain_graph=True, allow_unused=True, create_graph=False
        )
        if g_emb is None:
            scores[f] = 0.0
        else:
            scores[f] = float(g_emb.detach().float().norm().cpu())

    sampled_vals = scores[dims_to_score]
    thresh = float(np.percentile(sampled_vals, pruning_rate))
    keep = np.ones(dim, dtype=bool)
    for f in dims_to_score:
        keep[f] = scores[f] >= thresh

    ref = grad_list[cls_idx]
    if ref is None:
        for g in grad_list:
            if g is not None:
                ref = g
                break
    if ref is None:
        return tuple(grad_list)

    mask = torch.tensor(keep, device=ref.device, dtype=ref.dtype)
    for j in range(len(grad_list)):
        grad_list[j] = _apply_soteria_hidden_mask(grad_list[j], mask, dim)
    return tuple(grad_list)


def apply_defense(grads, args, model_wrapper=None, batch=None, labels=None):
    """
    Apply selected defense to gradients (or compute mixup gradients).

    Returns: gradient tuple (same structure as compute_grads).
    """
    defense = getattr(args, "defense", "none")
    seed = int(getattr(args, "rng_seed", 0))

    if defense == "mixup":
        if model_wrapper is None or batch is None or labels is None:
            raise ValueError("mixup requires model_wrapper, batch, labels")
        g = model_wrapper.compute_grads_mixup(batch, labels)
        if getattr(args, "defense_pct_mask", None) is not None:
            g = _apply_random_mask(g, float(args.defense_pct_mask), seed=seed)
        return g

    if grads is None:
        raise ValueError("apply_defense: grads is None but defense is not mixup")

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
    elif defense == "dpsgd":
        sigma = args.defense_noise
        if sigma is None:
            raise ValueError("--defense dpsgd requires --defense_noise as noise multiplier")
        g = dpsgd_defense(g, float(args.defense_clip_norm), float(sigma), seed=seed)
    elif defense == "topk":
        g = topk_sparsification(g, float(args.defense_topk_ratio))
    elif defense == "compression":
        g = gradient_compression(g, int(args.defense_n_bits))
    elif defense == "soteria":
        if model_wrapper is None or batch is None or labels is None:
            raise ValueError("soteria requires model_wrapper, batch, labels")
        g = soteria_defense(g, model_wrapper, batch, labels, args)
    elif defense == "dager":
        if model_wrapper is None or batch is None or labels is None:
            raise ValueError("dager defense requires model_wrapper, batch, labels")
        # Get layer names for gradient slicing
        layer_names = []
        for name, p in model_wrapper.model.named_parameters():
            if p.requires_grad:
                layer_names.append(name)
        g = apply_dager_defense(g, args, model_wrapper, batch, labels, layer_names)
    else:
        raise ValueError(f"Unknown defense: {defense}")

    if getattr(args, "defense_pct_mask", None) is not None:
        g = _apply_random_mask(g, float(args.defense_pct_mask), seed=seed)

    return g
