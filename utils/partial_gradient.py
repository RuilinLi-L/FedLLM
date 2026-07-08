from __future__ import annotations

import re
from typing import Iterable, Sequence


DEFAULT_LAYER_SUBSET = "all"
DEFAULT_PARAM_FILTER = "all"
VALID_PARAM_FILTERS = frozenset(
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
        "lora_only",
    }
)
PARTIAL_ATTACK_FULL_VISIBLE = "full_gradient_visible"
PARTIAL_ATTACK_DAGER_PREFIX = "dager_prefix_visible"
PARTIAL_ATTACK_DAGER_NONPREFIX = "dager_nonprefix_visible"
PARTIAL_ATTACK_DAGER_QKV = "dager_qkv_visible"
PARTIAL_ATTACK_LORA_ADAPTER = "peft_adapter_visible"
PARTIAL_ATTACK_PEFT_ADAPTER = PARTIAL_ATTACK_LORA_ADAPTER
PARTIAL_ATTACK_UNSUPPORTED_NONPREFIX = "unsupported_nonprefix_dager"
PARTIAL_ATTACK_UNSUPPORTED_INSUFFICIENT = "unsupported_insufficient_visible_matrices"
PARTIAL_ATTACK_UNSUPPORTED_FEATURE_DIM = "unsupported_feature_dim_mismatch"
PARTIAL_ATTACK_UNSUPPORTED_PTG_ONLY = "unsupported_ptg_only_filter"
GPT2_NONPREFIX_DAGER_MODELS = frozenset({"gpt2", "openai-community/gpt2-large"})
DAGER_PARAM_FILTERS = frozenset({DEFAULT_PARAM_FILTER, "qkv_only", "lora_only"})


class UnsupportedPartialGradientExposureError(RuntimeError):
    def __init__(self, message: str, *, variant: str, reason: str):
        super().__init__(message)
        self.variant = variant
        self.reason = reason


def partial_gradient_active(args) -> bool:
    return (
        getattr(args, "gradient_layer_subset", DEFAULT_LAYER_SUBSET) != DEFAULT_LAYER_SUBSET
        or getattr(args, "gradient_param_filter", DEFAULT_PARAM_FILTER) != DEFAULT_PARAM_FILTER
    )


def validate_partial_gradient_args(args):
    layer_subset = getattr(args, "gradient_layer_subset", DEFAULT_LAYER_SUBSET)
    param_filter = getattr(args, "gradient_param_filter", DEFAULT_PARAM_FILTER)
    _parse_layer_subset(layer_subset)
    if param_filter not in VALID_PARAM_FILTERS:
        raise ValueError(
            "--gradient_param_filter must be one of "
            f"{sorted(VALID_PARAM_FILTERS)}; got {param_filter!r}."
        )
    return args


def infer_partial_attack_variant(args) -> str:
    layer_subset = getattr(args, "gradient_layer_subset", DEFAULT_LAYER_SUBSET)
    param_filter = getattr(args, "gradient_param_filter", DEFAULT_PARAM_FILTER)

    if not partial_gradient_active(args):
        return PARTIAL_ATTACK_FULL_VISIBLE
    mode, _ = _parse_layer_subset(layer_subset)
    if param_filter not in DAGER_PARAM_FILTERS:
        return PARTIAL_ATTACK_UNSUPPORTED_PTG_ONLY
    if mode in {"last", "mid", "middle"}:
        if supports_nonprefix_dager(args):
            return PARTIAL_ATTACK_DAGER_NONPREFIX
        return PARTIAL_ATTACK_UNSUPPORTED_NONPREFIX
    if param_filter == "lora_only":
        return PARTIAL_ATTACK_LORA_ADAPTER
    if param_filter == "qkv_only":
        return PARTIAL_ATTACK_DAGER_QKV
    if mode == "first":
        return PARTIAL_ATTACK_DAGER_PREFIX
    return PARTIAL_ATTACK_FULL_VISIBLE


def partial_gradient_unsupported_reason(args) -> str:
    if args is None:
        return "n/a"
    layer_subset = getattr(args, "gradient_layer_subset", DEFAULT_LAYER_SUBSET)
    param_filter = getattr(args, "gradient_param_filter", DEFAULT_PARAM_FILTER)
    mode, n_layers = _parse_layer_subset(layer_subset)
    if param_filter not in DAGER_PARAM_FILTERS:
        return "ptg_only_filter_requires_attack_partial_gradient"
    if mode in {"last", "mid", "middle"}:
        if n_layers is not None and n_layers < 2:
            return "nonprefix_dager_requires_at_least_two_visible_layers"
        if param_filter not in {DEFAULT_PARAM_FILTER, "qkv_only"}:
            return "nonprefix_dager_requires_all_or_qkv_visible_gradients"
        if getattr(args, "model_path", None) not in GPT2_NONPREFIX_DAGER_MODELS:
            return "nonprefix_layer_subset_requires_gpt2_full_decoder"
        if getattr(args, "train_method", "full") != "full":
            return "nonprefix_layer_subset_requires_gpt2_full_decoder"
    return "n/a"


def mark_partial_gradient_unsupported(args, *, variant: str, reason: str) -> None:
    info = getattr(args, "partial_gradient_info", _default_partial_gradient_info(args))
    info["partial_attack_variant"] = variant
    info["unsupported_reason"] = reason
    setattr(args, "partial_gradient_info", info)


def supports_nonprefix_dager(args) -> bool:
    layer_subset = getattr(args, "gradient_layer_subset", DEFAULT_LAYER_SUBSET)
    param_filter = getattr(args, "gradient_param_filter", DEFAULT_PARAM_FILTER)
    mode, n_layers = _parse_layer_subset(layer_subset)
    if mode not in {"last", "mid", "middle"}:
        return False
    if n_layers is not None and n_layers < 2:
        return False
    if param_filter not in {DEFAULT_PARAM_FILTER, "qkv_only"}:
        return False
    if getattr(args, "model_path", None) not in GPT2_NONPREFIX_DAGER_MODELS:
        return False
    train_method = getattr(args, "train_method", "full")
    return train_method == "full"


def nonprefix_candidate_cap(args) -> int:
    max_ids = getattr(args, "max_ids", -1)
    if max_ids is not None and max_ids > 0:
        return int(max_ids)
    return int(max(1, getattr(args, "partial_nonprefix_candidate_cap", 64)))


def nonprefix_layer_indices(args) -> list[int]:
    layer_indices = getattr(args, "partial_nonprefix_layer_indices", None)
    if layer_indices is None:
        layer_indices = getattr(args, "dager_selected_block_ids", None)
    if layer_indices is None:
        raise ValueError("Missing non-prefix layer indices for partial-gradient DAGER decoding.")
    layer_indices = [int(idx) for idx in layer_indices]
    if len(layer_indices) < 2:
        raise ValueError("Non-prefix partial-gradient DAGER requires at least two visible layers.")
    return layer_indices


def _default_partial_gradient_info(args=None):
    param_filter = getattr(args, "gradient_param_filter", DEFAULT_PARAM_FILTER)
    model_path = getattr(args, "model_path", None)
    return {
        "gradient_layer_subset": getattr(args, "gradient_layer_subset", DEFAULT_LAYER_SUBSET),
        "gradient_param_filter": param_filter,
        "effective_gradient_param_filter": effective_param_filter_for_model(param_filter, model_path),
        "partial_filter_active": partial_gradient_active(args) if args is not None else False,
        "partial_attack_variant": infer_partial_attack_variant(args) if args is not None else PARTIAL_ATTACK_FULL_VISIBLE,
        "visible_grad_count": "n/a",
        "visible_matrix_grad_count": "n/a",
        "visible_param_names": "n/a",
        "dager_visible_candidate_count": "n/a",
        "dager_visible_param_names": "n/a",
        "selected_block_ids": "n/a",
        "unsupported_reason": _default_unsupported_reason(args) if args is not None else "n/a",
    }


def apply_partial_gradient_filter(grads, args, parameter_names: Iterable[str] | None = None):
    layer_subset = getattr(args, "gradient_layer_subset", DEFAULT_LAYER_SUBSET)
    param_filter = getattr(args, "gradient_param_filter", DEFAULT_PARAM_FILTER)
    active = partial_gradient_active(args)

    info = _default_partial_gradient_info(args)
    info["gradient_layer_subset"] = layer_subset
    info["gradient_param_filter"] = param_filter
    info["effective_gradient_param_filter"] = effective_param_filter_for_model(
        param_filter,
        getattr(args, "model_path", None),
    )
    info["partial_filter_active"] = active

    if not active:
        setattr(args, "partial_gradient_info", info)
        return grads

    names = list(parameter_names or [])
    grads = tuple(grads)
    if len(names) < len(grads):
        names.extend(f"param_{idx}" for idx in range(len(names), len(grads)))
    elif len(names) > len(grads):
        names = names[: len(grads)]

    layer_selector = _layer_selector(layer_subset, names)
    filtered = []
    visible_names = []
    visible_matrix_count = 0
    for idx, grad in enumerate(grads):
        name = names[idx]
        keep = layer_selector(name) and _param_filter_matches(name, param_filter)
        if keep:
            filtered.append(grad)
            if grad is not None:
                visible_names.append(name)
                if getattr(grad, "ndim", 0) >= 2:
                    visible_matrix_count += 1
        else:
            filtered.append(None)

    info.update(
        {
            "visible_grad_count": len(visible_names),
            "visible_matrix_grad_count": visible_matrix_count,
            "visible_param_names": _summarize_names(visible_names),
        }
    )
    setattr(args, "partial_gradient_info", info)
    return tuple(filtered)


def partial_gradient_summary_fields(args):
    info = getattr(args, "partial_gradient_info", None)
    if info is None:
        info = _default_partial_gradient_info(args)
    return [
        ("gradient_layer_subset", info.get("gradient_layer_subset", DEFAULT_LAYER_SUBSET)),
        ("gradient_param_filter", info.get("gradient_param_filter", DEFAULT_PARAM_FILTER)),
        ("effective_gradient_param_filter", info.get("effective_gradient_param_filter", DEFAULT_PARAM_FILTER)),
        ("partial_filter_active", info.get("partial_filter_active", False)),
        ("partial_attack_variant", info.get("partial_attack_variant", PARTIAL_ATTACK_FULL_VISIBLE)),
        ("visible_grad_count", info.get("visible_grad_count", "n/a")),
        ("visible_matrix_grad_count", info.get("visible_matrix_grad_count", "n/a")),
        ("visible_param_names", info.get("visible_param_names", "n/a")),
        ("dager_visible_candidate_count", info.get("dager_visible_candidate_count", "n/a")),
        ("dager_visible_param_names", info.get("dager_visible_param_names", "n/a")),
        ("selected_block_ids", info.get("selected_block_ids", "n/a")),
        ("unsupported_reason", info.get("unsupported_reason", "n/a")),
    ]


def select_visible_matrix_candidates(
    grads: Sequence,
    candidate_indices: Sequence[int],
    candidate_names: Sequence[str],
    n_layers: int,
):
    selected_indices = []
    selected_names = []
    skipped = []
    for idx, name in zip(candidate_indices, candidate_names):
        if idx >= len(grads):
            skipped.append((idx, name, "index_out_of_range"))
            continue
        grad = grads[idx]
        if grad is None:
            skipped.append((idx, name, "not_visible"))
            continue
        if getattr(grad, "ndim", 0) < 2:
            skipped.append((idx, name, "not_matrix_like"))
            continue
        selected_indices.append(idx)
        selected_names.append(name)
        if len(selected_indices) >= n_layers:
            break
    return selected_indices, selected_names, skipped


def update_dager_candidate_summary(
    args,
    candidate_names: Sequence[str],
    *,
    variant: str | None = None,
    unsupported_reason: str | None = None,
) -> None:
    info = getattr(args, "partial_gradient_info", {})
    info["dager_visible_candidate_count"] = len(candidate_names)
    info["dager_visible_param_names"] = _summarize_names(candidate_names)
    info["selected_block_ids"] = _summarize_block_ids(candidate_names)
    if variant is not None:
        info["partial_attack_variant"] = variant
    if unsupported_reason is not None:
        info["unsupported_reason"] = unsupported_reason
    setattr(args, "partial_gradient_info", info)


def non_prefix_dager_block_ids(candidate_names: Sequence[str], n_layers: int) -> list[int] | None:
    block_ids = dager_block_ids(candidate_names, n_layers)
    if any(layer_id is None for layer_id in block_ids):
        return None
    expected = list(range(n_layers))
    if block_ids != expected:
        return block_ids
    return None


def dager_block_ids(candidate_names: Sequence[str], n_layers: int) -> list[int | None]:
    return [_extract_block_id(name) for name in candidate_names[:n_layers]]


def _parse_layer_subset(layer_subset: str):
    if layer_subset == DEFAULT_LAYER_SUBSET:
        return DEFAULT_LAYER_SUBSET, None
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
    if mode == DEFAULT_LAYER_SUBSET:
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


def _param_filter_matches(name: str, param_filter: str) -> bool:
    lower = name.lower()
    if param_filter == DEFAULT_PARAM_FILTER:
        return True
    if param_filter == "classifier_only":
        return _is_classifier_param(lower)
    if param_filter == "lora_only":
        if "modules_to_save" in lower:
            return False
        return (
            bool(re.search(r"(?:^|\.)lora_[ab](?:\.|$)", lower))
            or "ia3" in lower
            or "prompt_encoder" in lower
            or "prefix" in lower
            or "adapter_down" in lower
            or "adapter_up" in lower
            or ".adapters." in lower
            or ".adapter." in lower
            or "down_proj" in lower
            or "up_proj" in lower
        )
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
    raise ValueError(f"Unknown gradient_param_filter: {param_filter!r}")


def effective_param_filter_for_model(param_filter: str, model_path: str | None = None) -> str:
    """Return the v1 effective selector for packed-projection model families."""
    if param_filter in {"query_only", "key_only", "value_only"} and model_path in GPT2_NONPREFIX_DAGER_MODELS:
        return "qkv_only"
    return param_filter


def _is_classifier_param(lower_name: str) -> bool:
    return any(
        part in lower_name
        for part in (
            "classifier",
            "score.",
            "score.weight",
            "score.bias",
            "lm_head",
            "classification_head",
            "modules_to_save",
        )
    )


def _is_qkv_param(lower_name: str, param_filter: str) -> bool:
    if "c_attn" in lower_name:
        # GPT-2 packs q/k/v in one c_attn tensor, so single-projection selectors
        # intentionally expose the whole packed projection in v1.
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


def _summarize_block_ids(names: Sequence[str]) -> str:
    if not names:
        return "none"
    ids = []
    for name in names:
        layer_id = _extract_block_id(str(name))
        ids.append("unknown" if layer_id is None else str(layer_id))
    return ";".join(ids)


def _default_unsupported_reason(args) -> str:
    if args is None:
        return "n/a"
    return partial_gradient_unsupported_reason(args)
