from __future__ import annotations

import itertools
import re
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from attacks.peftleak_text import (
    build_dummy_embedding_prior,
    get_token_embedding_matrix,
    nearest_token_ids,
    token_recovery_ratio,
)


PARTIAL_TRANSFORMER_GRADIENTS_ATTACK = "partial_transformer_gradients"
PTG_GRADIENT_MATCHING_VARIANT = "ptg_gradient_matching"
PTG_DEFAULT_LAYER_SUBSET = "all"
PTG_DEFAULT_PARAM_FILTER = "all"
PTG_PARAM_FILTERS = frozenset(
    {
        "all",
        "qkv_only",
        "query_only",
        "key_only",
        "value_only",
        "attn_out_only",
        "attn_only",
        "ffn_in_only",
        "ffn_out_only",
        "ffn_only",
        "classifier_only",
    }
)
PTG_PACKED_QKV_MODELS = frozenset({"gpt2", "openai-community/gpt2-large"})
PTG_SOURCE_GRAD_TYPES = frozenset(
    {
        "all_layers",
        "encoder",
        "layer_encoder",
        "attn_qkv",
        "attn_query",
        "attn_key",
        "attn_value",
        "attn_output",
        "ffn_fc",
        "ffn_output",
        "word_emb",
    }
)
PTG_MATCH_LOSSES = frozenset({"cosine", "normalized_mse", "cos", "dlg", "tag"})
PTG_OPTIMIZERS = frozenset({"adam", "bfgs", "bert-adam"})
PTG_INIT_STRATEGIES = frozenset({"prior", "random", "lm"})


def _is_source_bert_parity(model_wrapper, parity_mode: str) -> bool:
    args = getattr(model_wrapper, "args", None)
    return str(parity_mode or "").strip().lower() == "source" and getattr(args, "model_path", None) == "bert-base-uncased"


def _word_input_embeds(model_wrapper, batch) -> torch.Tensor:
    embedding = getattr(model_wrapper.model, "get_input_embeddings", lambda: None)()
    if embedding is None:
        embedding = getattr(model_wrapper.base_model, "get_input_embeddings", lambda: None)()
    if embedding is None:
        raise ValueError("Could not resolve input word embeddings for source PTG parity.")
    return embedding(batch["input_ids"])


def _ptg_input_embeds_for_batch(model_wrapper, batch, *, source_bert_word_embeddings: bool = False) -> torch.Tensor:
    if source_bert_word_embeddings:
        return _word_input_embeds(model_wrapper, batch)
    return model_wrapper._seq_class_input_embeds(batch)


def parse_ptg_attack_layers(raw) -> list[int] | None:
    """Parse source-code style attack layer strings: all or comma-separated ids."""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return [int(item) for item in raw]
    text = str(raw).strip()
    if not text or text.lower() == "all":
        return None
    try:
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"attack_layer must be 'all' or comma-separated integers; got {raw!r}.") from exc


def format_ptg_attack_layers(layers: Sequence[int] | None) -> str:
    if layers is None:
        return "all"
    return ",".join(str(int(layer)) for layer in layers)


def ptg_effective_param_filter(param_filter: str, model_path: str | None = None) -> str:
    if param_filter in {"query_only", "key_only", "value_only"} and model_path in PTG_PACKED_QKV_MODELS:
        return "qkv_only"
    return param_filter


def validate_ptg_selector_args(args):
    layer_subset = getattr(args, "gradient_layer_subset", PTG_DEFAULT_LAYER_SUBSET)
    param_filter = getattr(args, "gradient_param_filter", PTG_DEFAULT_PARAM_FILTER)
    _parse_layer_subset(layer_subset)
    if param_filter not in PTG_PARAM_FILTERS:
        raise ValueError(
            "--gradient_param_filter for PTG must be one of "
            f"{sorted(PTG_PARAM_FILTERS)}; got {param_filter!r}."
        )
    source_grad_type = getattr(args, "grad_type", None)
    if source_grad_type is not None and source_grad_type not in PTG_SOURCE_GRAD_TYPES:
        raise ValueError(
            "--grad_type for source PTG parity must be one of "
            f"{sorted(PTG_SOURCE_GRAD_TYPES)}; got {source_grad_type!r}."
        )
    parse_ptg_attack_layers(getattr(args, "attack_layer", None))
    return args


def ptg_filter_active(args) -> bool:
    return (
        getattr(args, "gradient_layer_subset", PTG_DEFAULT_LAYER_SUBSET) != PTG_DEFAULT_LAYER_SUBSET
        or getattr(args, "gradient_param_filter", PTG_DEFAULT_PARAM_FILTER) != PTG_DEFAULT_PARAM_FILTER
    )


def filter_partial_transformer_gradients(
    grads: Sequence[torch.Tensor | None],
    *,
    parameter_names: Sequence[str],
    layer_subset: str = PTG_DEFAULT_LAYER_SUBSET,
    param_filter: str = PTG_DEFAULT_PARAM_FILTER,
    model_path: str | None = None,
    source_grad_type: str | None = None,
    source_attack_layers: Sequence[int] | str | None = None,
) -> tuple[tuple[torch.Tensor | None, ...], dict[str, object]]:
    _parse_layer_subset(layer_subset)
    if param_filter not in PTG_PARAM_FILTERS:
        raise ValueError(f"Unknown PTG gradient_param_filter: {param_filter!r}")
    if source_grad_type is not None and source_grad_type not in PTG_SOURCE_GRAD_TYPES:
        raise ValueError(f"Unknown source PTG grad_type: {source_grad_type!r}")

    names = list(parameter_names)
    grads = tuple(grads)
    if len(names) < len(grads):
        names.extend(f"param_{idx}" for idx in range(len(names), len(grads)))
    elif len(names) > len(grads):
        names = names[: len(grads)]

    attack_layers = parse_ptg_attack_layers(source_attack_layers)
    if source_grad_type is None:
        selector = _layer_selector(layer_subset, names)
        keep_fn = lambda name: selector(name) and _param_filter_matches(name, param_filter)
    else:
        keep_fn = lambda name: _source_grad_type_matches(name, source_grad_type, attack_layers)

    filtered: list[torch.Tensor | None] = []
    visible_names: list[str] = []
    matrix_count = 0
    for idx, grad in enumerate(grads):
        name = names[idx]
        keep = keep_fn(name)
        if keep:
            filtered.append(grad)
            if grad is not None:
                visible_names.append(name)
                if getattr(grad, "ndim", 0) >= 2:
                    matrix_count += 1
        else:
            filtered.append(None)

    info = {
        "gradient_layer_subset": layer_subset,
        "gradient_param_filter": param_filter,
        "effective_gradient_param_filter": ptg_effective_param_filter(param_filter, model_path),
        "partial_filter_active": layer_subset != PTG_DEFAULT_LAYER_SUBSET or param_filter != PTG_DEFAULT_PARAM_FILTER,
        "partial_attack_variant": PTG_GRADIENT_MATCHING_VARIANT,
        "visible_grad_count": len(visible_names),
        "visible_matrix_grad_count": matrix_count,
        "visible_param_names": _summarize_names(visible_names),
        "grad_type": source_grad_type or "n/a",
        "attack_layer": format_ptg_attack_layers(attack_layers) if source_grad_type is not None else "n/a",
        "unsupported_reason": "n/a",
    }
    return tuple(filtered), info


def ptg_selector_summary_fields(args):
    info = getattr(args, "ptg_gradient_info", None)
    if info is None:
        info = {
            "gradient_layer_subset": getattr(args, "gradient_layer_subset", PTG_DEFAULT_LAYER_SUBSET),
            "gradient_param_filter": getattr(args, "gradient_param_filter", PTG_DEFAULT_PARAM_FILTER),
            "effective_gradient_param_filter": ptg_effective_param_filter(
                getattr(args, "gradient_param_filter", PTG_DEFAULT_PARAM_FILTER),
                getattr(args, "model_path", None),
            ),
            "partial_filter_active": ptg_filter_active(args),
            "partial_attack_variant": PTG_GRADIENT_MATCHING_VARIANT,
            "visible_grad_count": "n/a",
            "visible_matrix_grad_count": "n/a",
            "visible_param_names": "n/a",
            "grad_type": getattr(args, "grad_type", None) or "n/a",
            "attack_layer": format_ptg_attack_layers(parse_ptg_attack_layers(getattr(args, "attack_layer", None)))
            if getattr(args, "grad_type", None) is not None
            else "n/a",
            "unsupported_reason": "n/a",
        }
    return [
        ("gradient_layer_subset", info.get("gradient_layer_subset", PTG_DEFAULT_LAYER_SUBSET)),
        ("gradient_param_filter", info.get("gradient_param_filter", PTG_DEFAULT_PARAM_FILTER)),
        ("effective_gradient_param_filter", info.get("effective_gradient_param_filter", PTG_DEFAULT_PARAM_FILTER)),
        ("partial_filter_active", info.get("partial_filter_active", False)),
        ("visible_grad_count", info.get("visible_grad_count", "n/a")),
        ("visible_matrix_grad_count", info.get("visible_matrix_grad_count", "n/a")),
        ("visible_param_names", info.get("visible_param_names", "n/a")),
        ("grad_type", info.get("grad_type", "n/a")),
        ("attack_layer", info.get("attack_layer", "n/a")),
        ("unsupported_reason", info.get("unsupported_reason", "n/a")),
    ]


def _parse_layer_subset(layer_subset: str):
    if layer_subset == PTG_DEFAULT_LAYER_SUBSET:
        return PTG_DEFAULT_LAYER_SUBSET, None
    match = re.fullmatch(r"(first|last|mid|middle)(\d+)", str(layer_subset))
    if not match:
        raise ValueError(
            "--gradient_layer_subset must be 'all' or a value like first2/last2/mid2; "
            f"got {layer_subset!r}."
        )
    n_layers = int(match.group(2))
    if n_layers <= 0:
        raise ValueError("--gradient_layer_subset layer count must be positive.")
    return match.group(1), n_layers


def _layer_selector(layer_subset: str, names: Sequence[str]):
    mode, n_layers = _parse_layer_subset(layer_subset)
    if mode == PTG_DEFAULT_LAYER_SUBSET:
        return lambda name: True
    layer_ids = sorted({layer_id for name in names for layer_id in [_extract_block_id(name)] if layer_id is not None})
    if mode == "first":
        selected = set(layer_ids[:n_layers])
    elif mode == "last":
        selected = set(layer_ids[-n_layers:])
    else:
        start = max(0, (len(layer_ids) - n_layers) // 2)
        selected = set(layer_ids[start:start + n_layers])
    return lambda name: _extract_block_id(name) in selected


def _extract_block_id(name: str) -> int | None:
    patterns = (
        r"(?:^|\.)transformer\.h\.(\d+)(?:\.|$)",
        r"(?:^|\.)encoder\.layer\.(\d+)(?:\.|$)",
        r"(?:^|\.)model\.layers\.(\d+)(?:\.|$)",
        r"(?:^|\.)layers\.(\d+)(?:\.|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    return None


def _source_layer_matches(name: str, attack_layers: Sequence[int] | None) -> bool:
    if attack_layers is None:
        return _extract_block_id(name) is not None
    layer_id = _extract_block_id(name)
    return layer_id in set(int(layer) for layer in attack_layers)


def _is_transformer_block_param(lower_name: str) -> bool:
    return any(
        part in lower_name
        for part in (
            "encoder.layer.",
            "transformer.h.",
            "model.layers.",
            ".layers.",
        )
    )


def _is_word_embedding_param(lower_name: str) -> bool:
    return any(
        part in lower_name
        for part in (
            "word_embeddings",
            "embedding",
            "wte",
            "embed_tokens",
        )
    )


def _is_token_embedding_param(lower_name: str) -> bool:
    if any(
        part in lower_name
        for part in (
            "position_embeddings",
            "position_embedding",
            "token_type_embeddings",
            "token_type_embedding",
            "layernorm",
            "layer_norm",
        )
    ):
        return False
    if "word_embeddings" in lower_name or "embed_tokens" in lower_name:
        return True
    parts = lower_name.replace("[", ".").replace("]", ".").split(".")
    if "wte" in parts:
        return True
    return lower_name.endswith("embedding.weight") or lower_name.endswith("embeddings.weight")


def is_ptg_word_embedding_param(name: str) -> bool:
    return _is_token_embedding_param(str(name).lower())


def _source_grad_type_matches(name: str, grad_type: str, attack_layers: Sequence[int] | None) -> bool:
    lower = name.lower()
    layer_match = _source_layer_matches(name, attack_layers)

    if grad_type == "all_layers":
        return True
    if grad_type == "encoder":
        return _is_transformer_block_param(lower)
    if grad_type == "layer_encoder":
        return layer_match
    if grad_type == "word_emb":
        return _is_word_embedding_param(lower)

    if not layer_match:
        return False
    if grad_type == "attn_query":
        return _is_qkv_param(lower, "query_only")
    if grad_type == "attn_key":
        return _is_qkv_param(lower, "key_only")
    if grad_type == "attn_value":
        return _is_qkv_param(lower, "value_only")
    if grad_type == "attn_qkv":
        return _is_qkv_param(lower, "qkv_only")
    if grad_type == "attn_output":
        return _is_attn_output_param(lower)
    if grad_type == "ffn_fc":
        return _is_ffn_input_param(lower)
    if grad_type == "ffn_output":
        return _is_ffn_output_param(lower)
    return False


def _param_filter_matches(name: str, param_filter: str) -> bool:
    lower = name.lower()
    if param_filter == PTG_DEFAULT_PARAM_FILTER:
        return True
    if param_filter == "classifier_only":
        return _is_classifier_param(lower)
    if param_filter in {"qkv_only", "query_only", "key_only", "value_only"}:
        return _is_qkv_param(lower, param_filter)
    if param_filter == "attn_out_only":
        return _is_attn_output_param(lower)
    if param_filter == "attn_only":
        return _is_qkv_param(lower, "qkv_only") or _is_attn_output_param(lower)
    if param_filter == "ffn_in_only":
        return _is_ffn_input_param(lower)
    if param_filter == "ffn_out_only":
        return _is_ffn_output_param(lower)
    if param_filter == "ffn_only":
        return _is_ffn_input_param(lower) or _is_ffn_output_param(lower)
    return False


def _is_classifier_param(lower_name: str) -> bool:
    return any(
        part in lower_name
        for part in (
            "classifier",
            "score.",
            "score.weight",
            "score.bias",
            "classification_head",
        )
    )


def _is_qkv_param(lower_name: str, param_filter: str) -> bool:
    if "c_attn" in lower_name:
        return True
    query_parts = (".query.", "query.weight", "query.bias", "q_proj")
    key_parts = (".key.", "key.weight", "key.bias", "k_proj")
    value_parts = (".value.", "value.weight", "value.bias", "v_proj")
    if param_filter == "query_only":
        return any(part in lower_name for part in query_parts)
    if param_filter == "key_only":
        return any(part in lower_name for part in key_parts)
    if param_filter == "value_only":
        return any(part in lower_name for part in value_parts)
    return any(part in lower_name for part in query_parts + key_parts + value_parts)


def _is_attn_output_param(lower_name: str) -> bool:
    return any(
        part in lower_name
        for part in (
            "attn.c_proj",
            "attention.output.dense",
            "self_attn.o_proj",
            "self_attn.out_proj",
            ".o_proj.",
            "o_proj.weight",
            "o_proj.bias",
        )
    )


def _is_ffn_input_param(lower_name: str) -> bool:
    return any(
        part in lower_name
        for part in (
            "mlp.c_fc",
            "intermediate.dense",
            "mlp.gate_proj",
            "mlp.up_proj",
            ".gate_proj.",
            ".up_proj.",
            "gate_proj.weight",
            "up_proj.weight",
        )
    )


def _is_ffn_output_param(lower_name: str) -> bool:
    return any(
        part in lower_name
        for part in (
            "mlp.c_proj",
            "output.dense",
            "mlp.down_proj",
            ".down_proj.",
            "down_proj.weight",
            "down_proj.bias",
        )
    ) and not _is_attn_output_param(lower_name)


def _summarize_names(names: Sequence[str], limit: int = 8) -> str:
    if not names:
        return "none"
    shown = [str(name) for name in names[:limit]]
    if len(names) > limit:
        shown.append(f"...(+{len(names) - limit})")
    return ";".join(shown)


def selected_partial_gradient_tensors(
    grads: Sequence[torch.Tensor | None],
    parameter_names: Sequence[str],
) -> tuple[list[int], list[str]]:
    indices: list[int] = []
    names: list[str] = []
    for idx, grad in enumerate(grads):
        if grad is None:
            continue
        indices.append(idx)
        if idx < len(parameter_names):
            names.append(parameter_names[idx])
        else:
            names.append(f"param_{idx}")
    return indices, names


def _reduce_gradient(grad: torch.Tensor, *, detach: bool) -> torch.Tensor:
    tensor = grad.detach() if detach else grad
    return tensor.float().reshape(-1)


def _iter_label_assignments(
    batch_size: int,
    candidates: Sequence[int],
    *,
    max_assignments: int = 64,
) -> list[torch.Tensor]:
    values = [int(value) for value in candidates]
    total = len(values) ** int(batch_size)
    if total > max_assignments:
        raise ValueError(
            "PTG label search space is too large; "
            f"got {total} assignments for batch_size={batch_size}."
        )
    return [torch.tensor(assignment, dtype=torch.long) for assignment in itertools.product(values, repeat=batch_size)]


def _gradient_match_loss(
    *,
    candidate_grads: Sequence[torch.Tensor | None],
    target_grads: Sequence[torch.Tensor | None],
    selected_indices: Sequence[int],
    match_loss: str,
    device: torch.device,
    tag_factor: float | None = None,
) -> torch.Tensor:
    match_loss = "cos" if match_loss == "cosine" else match_loss
    losses: list[torch.Tensor] = []
    for idx in selected_indices:
        if idx >= len(target_grads):
            continue
        target_tensor = target_grads[idx]
        if target_tensor is None:
            continue

        target = _reduce_gradient(target_tensor, detach=True).to(device=device)
        if idx >= len(candidate_grads) or candidate_grads[idx] is None:
            candidate = torch.zeros_like(target, device=device)
        else:
            candidate = _reduce_gradient(candidate_grads[idx], detach=False).to(device=device)

        if candidate.numel() != target.numel():
            raise ValueError(
                "PTG candidate/target gradient shape mismatch for selected parameter "
                f"index {idx}: {candidate.numel()} vs {target.numel()} flattened elements."
            )
        if match_loss == "cos":
            denom = candidate.float().norm(p=2) * target.float().norm(p=2)
            losses.append(1.0 - (candidate.float() * target.float()).sum() / denom.clamp_min(1e-12))
        elif match_loss == "normalized_mse":
            denom = target.detach().float().pow(2).mean().clamp_min(1e-12)
            losses.append(F.mse_loss(candidate, target) / denom.to(device=candidate.device, dtype=candidate.dtype))
        elif match_loss == "dlg":
            losses.append((candidate - target).float().square().sum())
        elif match_loss == "tag":
            factor = 1e-3 if tag_factor is None else float(tag_factor)
            diff = (candidate - target).float()
            losses.append(diff.square().sum() + factor * diff.abs().sum())
        else:
            raise ValueError(f"--ptg_match_loss must be one of: {sorted(PTG_MATCH_LOSSES)}.")

    if not losses:
        raise ValueError("No selected partial gradients were available for PTG matching.")
    stacked = torch.stack([loss.reshape(()) for loss in losses])
    if match_loss in {"cos", "normalized_mse"}:
        return stacked.mean()
    return stacked.sum()


def _label_candidates(model_wrapper, raw: Sequence[int] | None = None) -> list[int]:
    if raw is not None:
        return [int(value) for value in raw]
    return list(range(int(getattr(model_wrapper.model.config, "num_labels", 2))))


def _valid_token_ids(token_ids: Iterable[object] | None, *, vocab_size: int | None = None) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for token_id in token_ids or []:
        if token_id is None:
            continue
        try:
            value = int(token_id)
        except (TypeError, ValueError, OverflowError):
            continue
        if vocab_size is not None and not (0 <= value < vocab_size):
            continue
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _fixed_token_mask(
    batch,
    *,
    ignored_token_ids: Iterable[int] | None = None,
    vocab_size: int | None = None,
    fix_special_tokens: bool = True,
    know_padding: bool = True,
    source_bert_special_tokens: bool = False,
    cls_token_id: int | None = None,
    sep_token_id: int | None = None,
    pad_token_id: int | None = None,
) -> torch.Tensor:
    input_ids = batch["input_ids"]
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    if source_bert_special_tokens:
        if not fix_special_tokens:
            return mask
        mask[:, 0] = True
        pad_id = 0 if pad_token_id is None else int(pad_token_id)
        if know_padding:
            seq_len = int(input_ids.shape[1])
            for sample_idx in range(int(input_ids.shape[0])):
                pad_start = seq_len
                for pos in range(seq_len - 1, 0, -1):
                    if int(input_ids[sample_idx, pos].detach().item()) == pad_id:
                        pad_start = pos
                    else:
                        break
                if pad_start < seq_len:
                    mask[sample_idx, pad_start:] = True
                if pad_start > 0:
                    mask[sample_idx, pad_start - 1] = True
        elif int(input_ids.shape[0]) == 1:
            mask[:, -1] = True
        return mask

    attention_mask = batch.get("attention_mask")
    if know_padding and attention_mask is not None:
        mask = mask | (attention_mask == 0)
    if fix_special_tokens:
        special_ids = _valid_token_ids(ignored_token_ids, vocab_size=vocab_size)
        if special_ids:
            special = torch.tensor(special_ids, device=input_ids.device, dtype=input_ids.dtype)
            mask = mask | (input_ids[..., None] == special).any(dim=-1)
    return mask


def _source_bert_fixed_token_ids(
    batch,
    *,
    fix_special_tokens: bool = True,
    know_padding: bool = True,
    cls_token_id: int | None = None,
    sep_token_id: int | None = None,
    pad_token_id: int | None = None,
) -> torch.Tensor:
    fixed_ids = batch["input_ids"].detach().clone()
    if not fix_special_tokens:
        return fixed_ids
    cls_id = 101 if cls_token_id is None else int(cls_token_id)
    sep_id = 102 if sep_token_id is None else int(sep_token_id)
    pad_id = 0 if pad_token_id is None else int(pad_token_id)
    fixed_ids[:, 0] = cls_id
    if know_padding:
        seq_len = int(fixed_ids.shape[1])
        for sample_idx in range(int(fixed_ids.shape[0])):
            pad_start = seq_len
            for pos in range(seq_len - 1, 0, -1):
                if int(fixed_ids[sample_idx, pos].detach().item()) == pad_id:
                    pad_start = pos
                else:
                    break
            if pad_start < seq_len:
                fixed_ids[sample_idx, pad_start:] = pad_id
            if pad_start > 0:
                fixed_ids[sample_idx, pad_start - 1] = sep_id
    elif int(fixed_ids.shape[0]) == 1:
        fixed_ids[:, -1] = sep_id
    return fixed_ids


def _make_ptg_optimizer(name: str, params, lr: float):
    name = str(name or "adam").strip().lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=float(lr))
    if name == "bfgs":
        return torch.optim.LBFGS(params, lr=float(lr))
    if name == "bert-adam":
        return torch.optim.AdamW(params, lr=float(lr), betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)
    raise ValueError(f"--ptg_optimizer must be one of {sorted(PTG_OPTIMIZERS)}; got {name!r}.")


def _make_ptg_scheduler(optimizer, lr_decay_type: str | None, *, lr_decay: float, lr_max_it: int):
    decay_type = str(lr_decay_type or "none").strip().lower()
    if decay_type in {"none", "off", "false"}:
        return None
    if decay_type == "steplr":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=float(lr_decay))
    if decay_type == "lambdalr":
        max_it = max(1, int(lr_max_it))

        def lr_lambda(current_step: int):
            return max(0.0, float(max_it - current_step) / float(max_it))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    raise ValueError("--ptg_lr_decay_type must be one of none, StepLR, LambdaLR.")


def _clip_dummy_grad_(dummy: torch.nn.Parameter, grad_clip: float | None) -> None:
    if grad_clip is None or dummy.grad is None:
        return
    with torch.no_grad():
        grad_norm = dummy.grad.detach().norm()
        if grad_norm > float(grad_clip):
            dummy.grad.mul_(float(grad_clip) / (grad_norm + 1e-6))


def _project_ids_to_embeddings(
    ids: Sequence[Sequence[int]] | torch.Tensor,
    token_matrix: torch.Tensor,
    *,
    reference_dummy: torch.Tensor | None = None,
) -> torch.Tensor:
    if not torch.is_tensor(ids):
        ids = torch.tensor(ids, dtype=torch.long, device=token_matrix.device)
    ids = ids.to(device=token_matrix.device, dtype=torch.long)
    projected = token_matrix[ids].to(dtype=token_matrix.dtype)
    if reference_dummy is not None:
        ref = reference_dummy.detach().to(device=projected.device, dtype=projected.dtype)
        denom = projected.norm(dim=-1, p=2, keepdim=True).clamp_min(1e-12)
        projected = projected * (ref.norm(dim=-1, p=2, keepdim=True) / denom)
    return projected


def _lm_prior_loss_for_ids(
    ids: Sequence[Sequence[int]] | torch.Tensor,
    *,
    model_wrapper,
    lm_model=None,
    lm_tokenizer=None,
) -> torch.Tensor:
    if lm_model is None:
        device = ids.device if torch.is_tensor(ids) else get_token_embedding_matrix(model_wrapper).device
        return torch.tensor(0.0, device=device)

    target_tokenizer = getattr(model_wrapper, "tokenizer", None)
    lm_tokenizer = lm_tokenizer or target_tokenizer
    device = next(lm_model.parameters()).device
    if torch.is_tensor(ids):
        ids_tensor = ids.detach().to(device=device, dtype=torch.long)
    else:
        ids_tensor = torch.tensor(ids, device=device, dtype=torch.long)

    if lm_tokenizer is target_tokenizer or lm_tokenizer is None:
        input_ids = ids_tensor
        attention_mask = torch.ones_like(input_ids)
    else:
        texts = target_tokenizer.batch_decode(ids_tensor.detach().cpu().tolist(), skip_special_tokens=True)
        tokenized = lm_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized.get("attention_mask", torch.ones_like(input_ids)).to(device)

    labels = input_ids.clone()
    labels = labels.masked_fill(attention_mask == 0, -100)
    with torch.no_grad():
        out = lm_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return out.loss.detach().to(device=device)


def _lm_generated_initial_candidates(
    *,
    model_wrapper,
    lm_model,
    lm_tokenizer,
    batch,
    n_candidates: int,
    seed: int,
    prompt: str,
    source_bert_word_embeddings: bool = False,
) -> list[torch.Tensor]:
    if lm_model is None:
        raise ValueError("--ptg_init lm requires --ptg_lm_model_path or a supplied lm_model.")
    if not hasattr(lm_model, "generate"):
        raise ValueError("The supplied LM model does not implement generate(); cannot use --ptg_init lm.")

    tokenizer = getattr(model_wrapper, "tokenizer", None)
    if lm_tokenizer is None:
        lm_tokenizer = tokenizer
    device = batch["input_ids"].device
    seq_len = int(batch["input_ids"].shape[1])
    batch_size = int(batch["input_ids"].shape[0])
    total = max(1, int(n_candidates)) * batch_size

    prompt_ids = lm_tokenizer.encode(prompt or "the", return_tensors="pt").to(next(lm_model.parameters()).device)
    if prompt_ids.numel() == 0:
        fallback_id = getattr(lm_tokenizer, "bos_token_id", None) or getattr(lm_tokenizer, "eos_token_id", None) or 0
        prompt_ids = torch.tensor([[int(fallback_id)]], device=next(lm_model.parameters()).device)
    prompt_ids = prompt_ids.repeat(total, 1)
    gen = torch.Generator(device=prompt_ids.device)
    gen.manual_seed(int(seed))
    pad_token_id = getattr(lm_tokenizer, "pad_token_id", None) or getattr(lm_tokenizer, "eos_token_id", None)
    with torch.no_grad():
        generated = lm_model.generate(
            prompt_ids,
            do_sample=True,
            no_repeat_ngram_size=2,
            max_length=prompt_ids.shape[1] + seq_len + 4,
            pad_token_id=pad_token_id,
            generator=gen,
        )
    continuation = generated[:, prompt_ids.shape[1]:]
    texts = lm_tokenizer.batch_decode(continuation, skip_special_tokens=True)
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    ).to(device)
    embeds = _ptg_input_embeds_for_batch(
        model_wrapper,
        tokenized,
        source_bert_word_embeddings=source_bert_word_embeddings,
    ).detach()
    return [embeds[idx * batch_size:(idx + 1) * batch_size] for idx in range(max(1, int(n_candidates)))]


def _scale_init_embeddings(embeds: torch.Tensor, init_size: float | None) -> torch.Tensor:
    if init_size is None or float(init_size) < 0:
        return embeds
    return embeds / embeds.norm(dim=-1, keepdim=True).clamp_min(1e-12) * float(init_size)


def optimize_partial_text_embeddings(
    *,
    model_wrapper,
    batch,
    labels,
    target_grads: Sequence[torch.Tensor | None],
    parameter_names: Sequence[str],
    steps: int = 80,
    lr: float = 0.1,
    restarts: int = 1,
    match_loss: str = "cosine",
    label_mode: str = "known",
    label_candidates: Sequence[int] | None = None,
    decode_metric: str = "cos",
    tv_weight: float = 0.0,
    embed_norm_weight: float = 0.0,
    entropy_weight: float = 0.0,
    fix_special_tokens: bool = True,
    know_padding: bool = True,
    lm_prior_weight: float = 0.0,
    lm_model=None,
    lm_tokenizer=None,
    swap_steps: int = 0,
    use_swaps: bool = False,
    swap_burnin: float = 0.1,
    swap_every: int = 75,
    use_swaps_at_end: bool = False,
    init_strategy: str = "prior",
    init_candidates: int = 1,
    init_size: float | None = None,
    init_permutation_trials: int = 0,
    lm_init_prompt: str = "the",
    optimizer_name: str = "adam",
    lr_decay_type: str | None = "none",
    lr_decay: float = 0.9,
    lr_max_it: int | None = None,
    tag_factor: float | None = None,
    grad_clip: float | None = None,
    print_every: int = 0,
    parity_mode: str = "fedllm",
    ignored_token_ids: Iterable[int] | None = None,
    reference_mask: Sequence[Sequence[int]] | None = None,
) -> dict[str, object]:
    """Reconstruct text from selected partial Transformer gradients.

    This implements the FedLLM adaptation of the partial-gradient leakage core:
    optimize dummy input embeddings so their gradients on the visible
    layer/module subset match the shared partial gradients.
    """

    selected_indices, selected_names = selected_partial_gradient_tensors(target_grads, parameter_names)
    if not selected_indices:
        raise ValueError("No visible partial gradients after filtering/defense; cannot run PTG matching.")

    device = batch["input_ids"].device
    label_mode = str(label_mode or "known").strip().lower()
    match_loss = str(match_loss or "cosine").strip().lower()
    init_strategy = str(init_strategy or "prior").strip().lower()
    optimizer_name = str(optimizer_name or "adam").strip().lower()
    parity_mode = str(parity_mode or "fedllm").strip().lower()
    source_bert_mode = _is_source_bert_parity(model_wrapper, parity_mode)
    if label_mode not in {"known", "search"}:
        raise ValueError("--ptg_label_mode must be known or search.")
    if match_loss not in PTG_MATCH_LOSSES:
        raise ValueError(f"--ptg_match_loss must be one of {sorted(PTG_MATCH_LOSSES)}.")
    if init_strategy not in PTG_INIT_STRATEGIES:
        raise ValueError(f"--ptg_init must be one of {sorted(PTG_INIT_STRATEGIES)}.")
    if optimizer_name not in PTG_OPTIMIZERS:
        raise ValueError(f"--ptg_optimizer must be one of {sorted(PTG_OPTIMIZERS)}.")
    if init_strategy == "lm" and lm_model is None:
        raise ValueError("--ptg_init lm requires --ptg_lm_model_path or a supplied lm_model.")

    if label_mode == "known":
        candidate_assignments = [labels.view(-1).long().to(device)]
    else:
        candidates = _label_candidates(model_wrapper, label_candidates)
        candidate_assignments = [
            assignment.to(device)
            for assignment in _iter_label_assignments(batch["input_ids"].shape[0], candidates)
        ]

    best_objective = float("inf")
    best_loss = float("inf")
    best_ids: list[list[int]] | None = None
    best_scores: list[list[float]] | None = None
    best_label: list[int] | None = None
    best_lm_loss: float | None = None
    logits_shape = None
    loss_history: list[float] = []
    objective_history: list[float] = []
    lm_loss_history: list[float] = []
    token_matrix = get_token_embedding_matrix(model_wrapper).to(device=device)
    tokenizer = getattr(model_wrapper, "tokenizer", None)
    cls_token_id = getattr(tokenizer, "cls_token_id", None)
    sep_token_id = getattr(tokenizer, "sep_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    reference_embeds = _ptg_input_embeds_for_batch(
        model_wrapper,
        batch,
        source_bert_word_embeddings=source_bert_mode,
    ).detach().to(device=device)
    fixed_mask = _fixed_token_mask(
        batch,
        ignored_token_ids=ignored_token_ids,
        vocab_size=int(token_matrix.shape[0]),
        fix_special_tokens=fix_special_tokens,
        know_padding=know_padding,
        source_bert_special_tokens=source_bert_mode,
        cls_token_id=cls_token_id,
        sep_token_id=sep_token_id,
        pad_token_id=pad_token_id,
    ).to(device=device)
    fixed_reference_ids = None
    fixed_reference_embeds = reference_embeds
    if source_bert_mode:
        fixed_reference_ids = _source_bert_fixed_token_ids(
            batch,
            fix_special_tokens=fix_special_tokens,
            know_padding=know_padding,
            cls_token_id=cls_token_id,
            sep_token_id=sep_token_id,
            pad_token_id=pad_token_id,
        ).to(device=device)
        fixed_reference_embeds = token_matrix[fixed_reference_ids].to(device=device, dtype=reference_embeds.dtype)
    optimizable_mask = ~fixed_mask
    token_norm_mean = token_matrix.detach().float().norm(dim=-1).mean().to(device=device)
    base_seed = int(getattr(model_wrapper.args, "rng_seed", 0))
    n_steps = max(1, int(steps))
    lr_max_it = n_steps if lr_max_it is None else int(lr_max_it)
    n_init_candidates = max(1, int(init_candidates))
    n_perm_trials = max(0, int(init_permutation_trials))
    swap_trials = max(0, int(swap_steps))
    use_swaps = bool(use_swaps) or swap_trials > 0
    if use_swaps and swap_trials <= 0:
        swap_trials = 200

    best_initial_loss: float | None = None

    def apply_fixed_embeddings_(dummy: torch.nn.Parameter) -> None:
        if not bool(fixed_mask.any().item()):
            return
        with torch.no_grad():
            dummy.copy_(torch.where(fixed_mask.unsqueeze(-1), fixed_reference_embeds, dummy))

    def decode_embeddings(dummy: torch.Tensor) -> tuple[list[list[int]], list[list[float]]]:
        sample_ids = []
        sample_scores = []
        reference_ids = (
            fixed_reference_ids.detach().cpu().tolist()
            if fixed_reference_ids is not None
            else batch["input_ids"].detach().cpu().tolist()
        )
        fixed_mask_cpu = fixed_mask.detach().cpu().tolist()
        for sample_idx in range(dummy.shape[0]):
            token_ids, token_scores = nearest_token_ids(
                dummy.detach()[sample_idx],
                token_matrix,
                unused_token_ids=ignored_token_ids,
                metric=decode_metric,
            )
            ids = [int(tok) for tok in token_ids.detach().cpu().tolist()]
            scores = [float(score) for score in token_scores.detach().cpu().tolist()]
            if bool(fixed_mask.any().item()):
                for pos, is_fixed in enumerate(fixed_mask_cpu[sample_idx]):
                    if is_fixed and pos < len(ids):
                        ids[pos] = int(reference_ids[sample_idx][pos])
                        scores[pos] = 0.0
            sample_ids.append(ids)
            sample_scores.append(scores)
        return sample_ids, sample_scores

    def ids_tensor_from_embeddings(dummy: torch.Tensor) -> torch.Tensor:
        ids, _scores = decode_embeddings(dummy)
        return torch.tensor(ids, device=device, dtype=torch.long)

    def loss_for_labels(
        dummy: torch.Tensor,
        candidate_labels: torch.Tensor,
        *,
        create_graph: bool,
        include_lm: bool = True,
    ):
        if source_bert_mode:
            outputs = model_wrapper.model(
                inputs_embeds=dummy,
                labels=candidate_labels.view(-1).long(),
            )
            logits = outputs.logits
            task_loss = outputs.loss if getattr(outputs, "loss", None) is not None else F.cross_entropy(
                logits,
                candidate_labels.view(-1).long(),
            )
        else:
            logits, _representation = model_wrapper._seq_class_logits_from_embeds(batch, dummy)
            task_loss = F.cross_entropy(logits, candidate_labels.view(-1).long())
        candidate_grads = torch.autograd.grad(
            task_loss,
            model_wrapper.trainable_parameters(),
            create_graph=create_graph,
            allow_unused=True,
            retain_graph=create_graph,
        )
        grad_loss = _gradient_match_loss(
            candidate_grads=candidate_grads,
            target_grads=target_grads,
            selected_indices=selected_indices,
            match_loss=match_loss,
            device=device,
            tag_factor=tag_factor,
        )
        tv_loss = dummy[:, 1:, :].sub(dummy[:, :-1, :]).abs().mean() if dummy.shape[1] > 1 else dummy.new_tensor(0.0)
        entropy = dummy.new_tensor(0.0)
        if entropy_weight > 0:
            probs = logits.softmax(dim=-1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()
        if embed_norm_weight > 0 and bool(optimizable_mask.any().item()):
            if source_bert_mode:
                norm_target = (
                    float(init_size)
                    if init_size is not None and float(init_size) >= 0
                    else float(token_norm_mean.detach().cpu().item())
                )
                embed_norm_loss = dummy.float().norm(dim=-1).mean().sub(norm_target).pow(2)
            else:
                embed_norm_loss = dummy[optimizable_mask].float().norm(dim=-1).sub(token_norm_mean).pow(2).mean()
        else:
            embed_norm_loss = dummy.new_tensor(0.0)
        total = (
            grad_loss
            + float(tv_weight) * tv_loss
            + float(embed_norm_weight) * embed_norm_loss
            - float(entropy_weight) * entropy
        )
        lm_loss = dummy.new_tensor(0.0)
        if include_lm and float(lm_prior_weight) != 0.0 and lm_model is not None:
            lm_loss = _lm_prior_loss_for_ids(
                ids_tensor_from_embeddings(dummy),
                model_wrapper=model_wrapper,
                lm_model=lm_model,
                lm_tokenizer=lm_tokenizer,
            ).to(device=device, dtype=total.dtype)
            if not source_bert_mode:
                total = total + float(lm_prior_weight) * lm_loss
        return total, grad_loss, logits, embed_norm_loss, lm_loss

    def candidate_initial_embeddings(seed: int) -> list[torch.Tensor]:
        if init_strategy == "lm":
            return _lm_generated_initial_candidates(
                model_wrapper=model_wrapper,
                lm_model=lm_model,
                lm_tokenizer=lm_tokenizer,
                batch=batch,
                n_candidates=n_init_candidates,
                seed=seed,
                prompt=lm_init_prompt,
                source_bert_word_embeddings=source_bert_mode,
            )
        candidates = []
        for cand_idx in range(n_init_candidates):
            cand_seed = seed + cand_idx * 4099
            if init_strategy == "prior":
                initial = build_dummy_embedding_prior(model_wrapper, batch, seed=cand_seed).to(device=device)
            elif init_strategy == "random":
                gen = torch.Generator(device=device)
                gen.manual_seed(int(cand_seed))
                initial = torch.randn(
                    reference_embeds.shape,
                    device=device,
                    dtype=reference_embeds.dtype,
                    generator=gen,
                )
            else:
                raise ValueError(f"Unsupported PTG init strategy: {init_strategy!r}")
            candidates.append(initial.detach())
        return candidates

    def permute_candidate(candidate: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        out = candidate.detach().clone()
        for sample_idx in range(out.shape[0]):
            positions = torch.nonzero(~fixed_mask[sample_idx], as_tuple=False).view(-1)
            if positions.numel() < 2:
                continue
            perm = positions[torch.randperm(int(positions.numel()), device=device, generator=gen)]
            out[sample_idx, positions] = out[sample_idx, perm]
        return out

    def select_initial_embedding(candidate_labels: torch.Tensor, seed: int):
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed) + 1777)
        best_tensor = None
        best_init_objective = None
        best_init_grad_loss = None
        best_init_logits = None
        best_init_lm = None
        for initial in candidate_initial_embeddings(seed):
            dummy = torch.nn.Parameter(initial.detach().clone().to(device=device))
            apply_fixed_embeddings_(dummy)
            total, grad_loss, logits, _norm, lm_loss = loss_for_labels(
                dummy,
                candidate_labels,
                create_graph=False,
            )
            score_tensor = grad_loss if source_bert_mode else total
            current = float(score_tensor.detach().item())
            if best_init_objective is None or current < best_init_objective:
                best_init_objective = current
                best_init_grad_loss = float(grad_loss.detach().item())
                best_init_logits = logits.detach()
                best_init_lm = float(lm_loss.detach().item())
                best_tensor = dummy.detach().clone()

        if best_tensor is None:
            raise RuntimeError("PTG initialization did not produce any candidate embeddings.")

        for _trial in range(n_perm_trials):
            candidate = permute_candidate(best_tensor, gen)
            dummy = torch.nn.Parameter(candidate.detach().clone())
            apply_fixed_embeddings_(dummy)
            total, grad_loss, logits, _norm, lm_loss = loss_for_labels(
                dummy,
                candidate_labels,
                create_graph=False,
            )
            score_tensor = grad_loss if source_bert_mode else total
            current = float(score_tensor.detach().item())
            if best_init_objective is None or current < best_init_objective:
                best_init_objective = current
                best_init_grad_loss = float(grad_loss.detach().item())
                best_init_logits = logits.detach()
                best_init_lm = float(lm_loss.detach().item())
                best_tensor = dummy.detach().clone()

        best_tensor = _scale_init_embeddings(best_tensor, init_size)
        dummy = torch.nn.Parameter(best_tensor.detach().clone())
        apply_fixed_embeddings_(dummy)
        total, grad_loss, logits, _norm, lm_loss = loss_for_labels(
            dummy,
            candidate_labels,
            create_graph=False,
        )
        return (
            dummy,
            float(grad_loss.detach().item()),
            float(total.detach().item()),
            logits.detach(),
            float(lm_loss.detach().item()) if best_init_lm is not None else None,
        )

    def apply_best_state(
        *,
        current_objective: float,
        current_grad_loss: float,
        current_lm_loss: float,
        run_initial_loss: float,
        logits: torch.Tensor,
        candidate_labels: torch.Tensor,
        dummy: torch.Tensor,
    ) -> None:
        nonlocal best_objective, best_loss, best_initial_loss, logits_shape, best_label, best_ids, best_scores, best_lm_loss
        if current_objective < best_objective:
            best_objective = current_objective
            best_loss = current_grad_loss
            best_initial_loss = run_initial_loss
            best_lm_loss = current_lm_loss
            logits_shape = tuple(int(value) for value in logits.shape)
            best_label = candidate_labels.detach().cpu().tolist()
            best_ids, best_scores = decode_embeddings(dummy)

    def random_int(gen: torch.Generator, high: int) -> int:
        if high <= 0:
            return 0
        return int(torch.randint(high, (1,), device=device, generator=gen).item())

    def edit_positions(positions: torch.Tensor, sample_idx: int) -> torch.Tensor:
        if positions.numel() < 2:
            return positions
        pos_list = positions.detach().clone()
        move_type = sample_idx % 4
        n_pos = int(pos_list.numel())
        if move_type == 0:
            i = random_int(swap_generator, n_pos)
            j = random_int(swap_generator, n_pos)
            pos_list[i], pos_list[j] = pos_list[j].clone(), pos_list[i].clone()
        elif move_type == 1:
            i = random_int(swap_generator, n_pos)
            j = random_int(swap_generator, n_pos)
            values = pos_list.tolist()
            tok = values.pop(i)
            values.insert(j, tok)
            pos_list = torch.tensor(values, device=device, dtype=torch.long)
        elif move_type == 2 and n_pos >= 3:
            b = random_int(swap_generator, n_pos - 1)
            e = b + 1 + random_int(swap_generator, n_pos - b - 1)
            span_len = e - b
            p_choices = max(1, n_pos - span_len)
            p = random_int(swap_generator, p_choices)
            values = pos_list.tolist()
            span = values[b:e]
            rest = values[:b] + values[e:]
            rest[p:p] = span
            pos_list = torch.tensor(rest, device=device, dtype=torch.long)
        else:
            split = 1 + random_int(swap_generator, max(1, n_pos - 1))
            pos_list = torch.cat([pos_list[split:], pos_list[:split]])
        return pos_list

    swap_generator = torch.Generator(device=device)
    swap_generator.manual_seed(base_seed + 88_003)

    def attempt_token_swaps(dummy: torch.nn.Parameter, candidate_labels: torch.Tensor) -> tuple[float, float, torch.Tensor, float]:
        print("[ptg] Attempt swap", flush=True)
        current_ids = ids_tensor_from_embeddings(dummy)
        best_tensor = dummy.detach().clone()
        best_total, best_grad, best_logits, _norm, best_lm = loss_for_labels(
            best_tensor,
            candidate_labels,
            create_graph=False,
        )
        best_objective_local = float(best_total.detach().item())
        best_grad_local = float(best_grad.detach().item())
        best_lm_local = float(best_lm.detach().item())
        changed_type = None

        for sen_id in range(dummy.shape[0]):
            positions = torch.nonzero(~fixed_mask[sen_id], as_tuple=False).view(-1)
            if positions.numel() < 2:
                continue
            for trial_idx in range(max(1, swap_trials)):
                perm_positions = positions if trial_idx == 0 else edit_positions(positions, trial_idx)
                candidate = best_tensor.detach().clone()
                candidate[sen_id, positions] = best_tensor[sen_id, perm_positions]
                total, grad_loss, logits, _norm, lm_loss = loss_for_labels(
                    candidate,
                    candidate_labels,
                    create_graph=False,
                )
                current = float(total.detach().item())
                if current < best_objective_local:
                    best_objective_local = current
                    best_grad_local = float(grad_loss.detach().item())
                    best_logits = logits.detach()
                    best_lm_local = float(lm_loss.detach().item())
                    best_tensor = candidate.detach().clone()
                    changed_type = trial_idx % 4
                    current_ids = ids_tensor_from_embeddings(best_tensor)
        with torch.no_grad():
            dummy.copy_(best_tensor)
            apply_fixed_embeddings_(dummy)
        if changed_type is not None:
            change = ["Swapped tokens", "Moved token", "Moved sequence", "Put prefix at the end"][changed_type]
            print(f"[ptg] {change}", flush=True)
        return best_objective_local, best_grad_local, best_logits, best_lm_local

    for assignment_idx, candidate_labels in enumerate(candidate_assignments):
        for restart_idx in range(max(1, int(restarts))):
            init_seed = base_seed + assignment_idx * 9176 + restart_idx * 1009
            dummy, run_initial_loss, current_objective, initial_logits, initial_lm_loss = select_initial_embedding(
                candidate_labels,
                init_seed,
            )
            loss_history.append(run_initial_loss)
            objective_history.append(current_objective)
            if initial_lm_loss is not None:
                lm_loss_history.append(initial_lm_loss)
            apply_best_state(
                current_objective=current_objective,
                current_grad_loss=run_initial_loss,
                current_lm_loss=initial_lm_loss or 0.0,
                run_initial_loss=run_initial_loss,
                logits=initial_logits,
                candidate_labels=candidate_labels,
                dummy=dummy,
            )

            optimizer = _make_ptg_optimizer(optimizer_name, [dummy], float(lr))
            scheduler = _make_ptg_scheduler(
                optimizer,
                lr_decay_type,
                lr_decay=float(lr_decay),
                lr_max_it=lr_max_it,
            )

            for step_idx in range(n_steps):
                if optimizer_name == "bfgs":
                    def closure():
                        optimizer.zero_grad(set_to_none=True)
                        total, _grad_loss, _logits, _embed_norm_loss, _lm_loss = loss_for_labels(
                            dummy,
                            candidate_labels,
                            create_graph=True,
                        )
                        total.backward(retain_graph=True)
                        _clip_dummy_grad_(dummy, grad_clip)
                        return total

                    optimizer.step(closure)
                else:
                    optimizer.zero_grad(set_to_none=True)
                    total, _grad_loss, _logits, _embed_norm_loss, _lm_loss = loss_for_labels(
                        dummy,
                        candidate_labels,
                        create_graph=True,
                    )
                    total.backward(retain_graph=True)
                    _clip_dummy_grad_(dummy, grad_clip)
                    optimizer.step()
                apply_fixed_embeddings_(dummy)

                if scheduler is not None:
                    scheduler.step()

                if use_swaps and step_idx >= int(float(swap_burnin) * n_steps) and step_idx % max(1, int(swap_every)) == 1:
                    attempt_token_swaps(dummy, candidate_labels)

                eval_total, eval_grad_loss, eval_logits, _eval_norm, eval_lm_loss = loss_for_labels(
                    dummy,
                    candidate_labels,
                    create_graph=False,
                )
                current_objective = float(eval_total.detach().item())
                current_grad_loss = float(eval_grad_loss.detach().item())
                current_lm_loss = float(eval_lm_loss.detach().item())
                objective_history.append(current_objective)
                loss_history.append(current_grad_loss)
                lm_loss_history.append(current_lm_loss)
                apply_best_state(
                    current_objective=current_objective,
                    current_grad_loss=current_grad_loss,
                    current_lm_loss=current_lm_loss,
                    run_initial_loss=run_initial_loss,
                    logits=eval_logits,
                    candidate_labels=candidate_labels,
                    dummy=dummy,
                )

                steps_done = step_idx + 1
                if int(print_every) > 0 and steps_done % int(print_every) == 0:
                    decoded_ids = ids_tensor_from_embeddings(dummy)
                    projected = _project_ids_to_embeddings(
                        decoded_ids,
                        token_matrix,
                        reference_dummy=dummy,
                    ).to(device=device, dtype=dummy.dtype)
                    proj_total, proj_grad, _proj_logits, _proj_norm, proj_lm = loss_for_labels(
                        projected,
                        candidate_labels,
                        create_graph=False,
                    )
                    predictions = getattr(model_wrapper, "tokenizer").batch_decode(
                        decoded_ids.detach().cpu().tolist(),
                        skip_special_tokens=False,
                    )
                    print(
                        "[%4d/%4d] tot_loss=%.3f (perp=%.3f, rec=%.3f), tot_loss_proj:%.3f"
                        % (
                            steps_done,
                            n_steps,
                            current_objective,
                            current_lm_loss,
                            current_grad_loss,
                            float(proj_total.detach().item()),
                        ),
                        flush=True,
                    )
                    print(f"prediction: {predictions}", flush=True)

            if use_swaps_at_end and swap_trials > 0:
                swap_at_end_it = int((1.0 - float(swap_burnin)) * n_steps // max(1, int(swap_every)))
                print(f"[ptg] Trying {swap_at_end_it} swaps", flush=True)
                for _idx in range(max(0, swap_at_end_it)):
                    objective, grad_loss, logits, lm_loss = attempt_token_swaps(dummy, candidate_labels)
                    apply_best_state(
                        current_objective=objective,
                        current_grad_loss=grad_loss,
                        current_lm_loss=lm_loss,
                        run_initial_loss=run_initial_loss,
                        logits=logits,
                        candidate_labels=candidate_labels,
                        dummy=dummy,
                    )

    if best_ids is None or best_scores is None:
        raise RuntimeError("PTG matching did not produce decoded token candidates.")

    reference_ids = batch["input_ids"].detach().cpu().tolist()
    rec_token_mean = token_recovery_ratio(
        best_ids,
        reference_ids,
        ignored_token_ids=ignored_token_ids,
        reference_mask=reference_mask,
    )

    return {
        "loss": float(best_loss),
        "initial_loss": float(best_initial_loss) if best_initial_loss is not None else None,
        "loss_reduction": float(best_initial_loss - best_loss) if best_initial_loss is not None else None,
        "loss_history": loss_history,
        "objective_loss": float(best_objective),
        "objective_history": objective_history,
        "lm_loss": best_lm_loss,
        "lm_loss_history": lm_loss_history,
        "best_label": best_label,
        "logits_shape": logits_shape,
        "predicted_ids": best_ids,
        "predicted_scores": best_scores,
        "selected_gradient_indices": selected_indices,
        "selected_gradient_names": selected_names,
        "selected_gradient_count": len(selected_names),
        "sequence_length": int(batch["input_ids"].shape[1]),
        "label_mode": label_mode,
        "restarts": int(max(1, int(restarts))),
        "match_loss": match_loss,
        "decode_metric": decode_metric,
        "fixed_token_count": int(fixed_mask.sum().detach().item()),
        "fix_special_tokens": bool(fix_special_tokens),
        "know_padding": bool(know_padding),
        "embed_norm_weight": float(embed_norm_weight),
        "tv_weight": float(tv_weight),
        "entropy_weight": float(entropy_weight),
        "lm_prior_weight": float(lm_prior_weight),
        "swap_steps": int(swap_trials),
        "use_swaps": bool(use_swaps),
        "init_strategy": init_strategy,
        "init_candidates": int(n_init_candidates),
        "init_size": None if init_size is None else float(init_size),
        "init_permutation_trials": int(n_perm_trials),
        "optimizer": optimizer_name,
        "lr_decay_type": str(lr_decay_type or "none"),
        "lr_decay": float(lr_decay),
        "grad_clip": None if grad_clip is None else float(grad_clip),
        "parity_mode": parity_mode,
        "rec_token_mean": float(rec_token_mean),
    }
