from __future__ import annotations

import re
from typing import Iterable, Sequence


DEFAULT_LAYER_SUBSET = "all"
DEFAULT_PARAM_FILTER = "all"
VALID_PARAM_FILTERS = frozenset({"all", "qkv_only", "lora_only"})


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


def apply_partial_gradient_filter(grads, args, parameter_names: Iterable[str] | None = None):
    layer_subset = getattr(args, "gradient_layer_subset", DEFAULT_LAYER_SUBSET)
    param_filter = getattr(args, "gradient_param_filter", DEFAULT_PARAM_FILTER)
    active = partial_gradient_active(args)

    info = {
        "gradient_layer_subset": layer_subset,
        "gradient_param_filter": param_filter,
        "partial_filter_active": active,
        "visible_grad_count": "n/a",
        "visible_matrix_grad_count": "n/a",
        "visible_param_names": "n/a",
        "dager_visible_candidate_count": "n/a",
    }

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
        info = {
            "gradient_layer_subset": getattr(args, "gradient_layer_subset", DEFAULT_LAYER_SUBSET),
            "gradient_param_filter": getattr(args, "gradient_param_filter", DEFAULT_PARAM_FILTER),
            "partial_filter_active": partial_gradient_active(args),
            "visible_grad_count": "n/a",
            "visible_matrix_grad_count": "n/a",
            "visible_param_names": "n/a",
            "dager_visible_candidate_count": "n/a",
        }
    return [
        ("gradient_layer_subset", info.get("gradient_layer_subset", DEFAULT_LAYER_SUBSET)),
        ("gradient_param_filter", info.get("gradient_param_filter", DEFAULT_PARAM_FILTER)),
        ("partial_filter_active", info.get("partial_filter_active", False)),
        ("visible_grad_count", info.get("visible_grad_count", "n/a")),
        ("visible_matrix_grad_count", info.get("visible_matrix_grad_count", "n/a")),
        ("visible_param_names", info.get("visible_param_names", "n/a")),
        ("dager_visible_candidate_count", info.get("dager_visible_candidate_count", "n/a")),
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


def update_dager_candidate_summary(args, candidate_names: Sequence[str]) -> None:
    info = getattr(args, "partial_gradient_info", {})
    info["dager_visible_candidate_count"] = len(candidate_names)
    if candidate_names:
        info["dager_visible_param_names"] = _summarize_names(candidate_names)
    setattr(args, "partial_gradient_info", info)


def non_prefix_dager_block_ids(candidate_names: Sequence[str], n_layers: int) -> list[int] | None:
    block_ids = [_extract_block_id(name) for name in candidate_names[:n_layers]]
    if any(layer_id is None for layer_id in block_ids):
        return None
    expected = list(range(n_layers))
    if block_ids != expected:
        return block_ids
    return None


def _parse_layer_subset(layer_subset: str):
    if layer_subset == DEFAULT_LAYER_SUBSET:
        return DEFAULT_LAYER_SUBSET, None
    match = re.fullmatch(r"(first|last)(\d+)", str(layer_subset))
    if not match:
        raise ValueError(
            "--gradient_layer_subset must be 'all' or a value like first2/last2; "
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
    else:
        selected = set(layer_ids[-n_layers:])
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
    if param_filter == "lora_only":
        if "modules_to_save" in lower:
            return False
        return bool(re.search(r"(?:^|\.)lora_[ab](?:\.|$)", lower))
    if param_filter == "qkv_only":
        qkv_parts = (
            "c_attn",
            "q_proj",
            "k_proj",
            "v_proj",
            ".query.",
            ".key.",
            ".value.",
            "query.weight",
            "key.weight",
            "value.weight",
        )
        return any(part in lower for part in qkv_parts)
    raise ValueError(f"Unknown gradient_param_filter: {param_filter!r}")


def _summarize_names(names: Sequence[str], limit: int = 8) -> str:
    if not names:
        return "none"
    shown = [str(name) for name in names[:limit]]
    if len(names) > limit:
        shown.append(f"...(+{len(names) - limit})")
    return ";".join(shown)
