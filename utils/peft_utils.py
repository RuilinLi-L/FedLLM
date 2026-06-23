from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping
from pathlib import Path

import peft
import torch

try:
    import adapters as adapterlib
except Exception:  # pragma: no cover - optional runtime dependency
    adapterlib = None


GPT2_PEFT_MODELS = frozenset({"gpt2", "openai-community/gpt2-large"})
BERT_PEFT_MODELS = frozenset({"bert-base-uncased"})
LLAMA_PEFT_MODELS = frozenset(
    {
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-70b-hf",
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Meta-Llama-3.1-8B",
        "meta-llama/Meta-Llama-3-70B",
    }
)

SUPPORTED_PEFT_MODELS = GPT2_PEFT_MODELS | BERT_PEFT_MODELS | LLAMA_PEFT_MODELS
SUPPORTED_PEFT_METHODS = frozenset({"lora", "ia3", "prefix", "adapter"})
SUPPORTED_PEFT_EVAL_METHODS = frozenset({"lora", "ia3", "adapter"})
SUPPORTED_PEFT_DAGER_METHODS = SUPPORTED_PEFT_EVAL_METHODS
PEFT_TRAINING_ONLY_METHODS = frozenset({"prefix"})
PEFT_V2_PLANNED_METHODS = frozenset()
PEFT_EVAL_SCOPE_DAGER = "dager_eval"
PEFT_EVAL_SCOPE_TRAINING_ONLY = "training_only"
PEFT_EVAL_SCOPE_V2_PLANNED = "v2_planned"
PEFT_EVAL_SCOPE_NA = "n/a"
PEFT_EVAL_SCOPE_UNKNOWN = "unknown"
SUPPORTED_PEFT_DEFENSES = frozenset(
    {"none", "noise", "dpsgd", "topk", "compression", "soteria", "mixup", "lrb", "lrbprojonly", "signed_bottleneck"}
)
SUPPORTED_PEFT_TRAINING_POST_GRADIENT_DEFENSES = frozenset(
    {"none", "noise", "topk", "compression", "lrb", "lrbprojonly", "signed_bottleneck"}
)
SUPPORTED_PEFT_TRAINING_DIRECT_DEFENSES = frozenset({"dpsgd", "soteria", "mixup"})
SUPPORTED_PEFT_TRAINING_DAGER_DEFENSES = frozenset({"dager"})
SUPPORTED_PEFT_TRAINING_DEFENSES = (
    SUPPORTED_PEFT_TRAINING_POST_GRADIENT_DEFENSES
    | SUPPORTED_PEFT_TRAINING_DIRECT_DEFENSES
    | SUPPORTED_PEFT_TRAINING_DAGER_DEFENSES
)

LEGACY_LORA_STATE_SUFFIXES = frozenset({".pt", ".pth"})
PEFT_ADAPTER_CONFIG = "adapter_config.json"
PEFT_METADATA_FILE = "fedllm_peft_metadata.json"
PEFT_ADAPTER_HEAD_FILE = "adapter_head.bin"
PEFT_ADAPTER_WEIGHT_SUFFIXES = frozenset({".bin", ".safetensors"})
PEFT_ADAPTER_WEIGHT_FILES = frozenset(
    {
        "adapter_model.bin",
        "adapter_model.safetensors",
        "pytorch_adapter.bin",
        "pytorch_adapter.safetensors",
    }
)
ADAPTER_DEFAULT_NAME = "fedllm_adapter"
ADAPTER_DEFAULT_ARCHITECTURE = "double_seq_bn"
ADAPTER_DEFAULT_REDUCTION_FACTOR = 16

ADAPTER_ARCHITECTURE_ALIASES = {
    "adapter": "adapter",
    "seq_bn": "adapter",
    "double_seq_bn": "adapter",
    "par_bn": "adapter",
    "par_seq_bn": "adapter",
    "parallel_bn": "adapter",
    "seq_bn_inv": "adapter",
    "double_seq_bn_inv": "adapter",
    "par_bn_inv": "adapter",
    "par_seq_bn_inv": "adapter",
    "houlsby": "adapter",
    "pfeiffer": "adapter",
}

ADAPTER_DOWN_NAME_HINTS = (
    "adapter_down",
    ".adapter_down.",
    "down_proj",
    ".down.",
    "down.weight",
    ".down.weight",
    "bottleneck",
    "down_linear",
)
ADAPTER_UP_NAME_HINTS = (
    "adapter_up",
    ".adapter_up.",
    "up_proj",
    ".up.",
    "up.weight",
    ".up.weight",
    "up_linear",
)
ADAPTER_NAME_HINTS = ADAPTER_DOWN_NAME_HINTS + ADAPTER_UP_NAME_HINTS + (
    "adapter",
    "adapters",
)

LORA_TARGET_PRESETS = frozenset({"default", "c_attn", "q_proj", "qv", "qkvo", "bert-qv", "all-linear"})
IA3_TARGET_PRESETS = frozenset({"default", "gpt2-attn-mlp", "bert-qv-ffn", "llama-qv-ffn"})


@dataclass(frozen=True)
class PeftCheckpointMetadata:
    path: Path | None
    checkpoint_type: str
    peft_type: str | None = None
    adapter_r: int | None = None
    adapter_target_modules: tuple[str, ...] | None = None
    adapter_feedforward_modules: tuple[str, ...] | None = None
    adapter_num_virtual_tokens: int | None = None
    adapter_task_type: str | None = None
    adapter_base_model_name_or_path: str | None = None
    adapter_reduction_factor: int | None = None
    adapter_architecture: str | None = None
    adapter_name: str | None = None


@dataclass(frozen=True)
class PeftResolvedConfig:
    peft_method: str
    lora_r: int | None
    target_modules: tuple[str, ...] | None
    feedforward_modules: tuple[str, ...] | None
    num_virtual_tokens: int | None
    metadata: PeftCheckpointMetadata
    adapter_reduction_factor: int | None = None


# Backward-compatible names used by existing tests/docs.
LoraCheckpointMetadata = PeftCheckpointMetadata
LoraResolvedConfig = PeftResolvedConfig
SUPPORTED_LORA_MODELS = SUPPORTED_PEFT_MODELS
SUPPORTED_LORA_DEFENSES = SUPPORTED_PEFT_DEFENSES


def normalize_peft_method_name(value: str | None) -> str | None:
    if value is None:
        return None
    raw = str(getattr(value, "value", value)).strip().lower()
    raw = raw.replace("pefttype.", "").replace("-", "_")
    aliases = {
        "lora": "lora",
        "ia3": "ia3",
        "prefix": "prefix",
        "prefix_tuning": "prefix",
        "adapter": "adapter",
        **ADAPTER_ARCHITECTURE_ALIASES,
    }
    return aliases.get(raw, raw)


def peft_eval_scope(peft_method: str | None) -> str:
    method = normalize_peft_method_name(peft_method)
    if method is None:
        return PEFT_EVAL_SCOPE_NA
    if method in SUPPORTED_PEFT_EVAL_METHODS:
        return PEFT_EVAL_SCOPE_DAGER
    if method in PEFT_TRAINING_ONLY_METHODS:
        return PEFT_EVAL_SCOPE_TRAINING_ONLY
    if method in PEFT_V2_PLANNED_METHODS:
        return PEFT_EVAL_SCOPE_V2_PLANNED
    return PEFT_EVAL_SCOPE_UNKNOWN


def peft_eval_scope_message(peft_method: str | None) -> str:
    method = normalize_peft_method_name(peft_method)
    scope = peft_eval_scope(method)
    if scope == PEFT_EVAL_SCOPE_TRAINING_ONLY:
        return (
            "PEFT DAGER/partial eval supports LoRA/IA3/Houlsby-style adapter. "
            "Prefix tuning is training-only and excluded from DAGER/partial-gradient eval matrices."
        )
    if scope == PEFT_EVAL_SCOPE_V2_PLANNED:
        return (
            "This PEFT method is planned for a future evaluation matrix and is not enabled here."
        )
    if scope == PEFT_EVAL_SCOPE_DAGER:
        return "PEFT DAGER/partial eval v1 supports this method."
    return (
        "Unsupported PEFT method for PEFT eval. PEFT DAGER/partial eval supports LoRA/IA3/adapter; "
        f"got peft_method={method!r}."
    )


def _method_to_peft_type(method: str | None) -> str | None:
    normalized = normalize_peft_method_name(method)
    if normalized is None:
        return None
    return {
        "lora": "LORA",
        "ia3": "IA3",
        "prefix": "PREFIX_TUNING",
        "adapter": "ADAPTER",
    }.get(normalized, str(normalized).upper())


def normalize_peft_args(args):
    if getattr(args, "train_method", "full") == "lora":
        args.train_method = "peft"
        args.peft_method = "lora"
    elif getattr(args, "train_method", "full") == "peft":
        if getattr(args, "peft_method", None) is None:
            args.peft_method = "lora"
        else:
            args.peft_method = normalize_peft_method_name(args.peft_method)
    else:
        if not hasattr(args, "peft_method"):
            args.peft_method = None
    return args


def peft_active(args) -> bool:
    return getattr(args, "train_method", "full") in {"peft", "lora"}


def validate_peft_training_defense_args(args):
    normalize_peft_args(args)
    if not peft_active(args):
        return args

    defense = getattr(args, "defense", "none")
    peft_method = normalize_peft_method_name(getattr(args, "peft_method", None)) or "lora"
    if peft_method in {"lora", "ia3", "adapter"}:
        if defense not in SUPPORTED_PEFT_TRAINING_DEFENSES:
            raise NotImplementedError(
                "PEFT training currently supports these defenses for "
                f"peft_method={peft_method!r}: {sorted(SUPPORTED_PEFT_TRAINING_DEFENSES)}; "
                f"got defense={defense!r}."
            )
        return args

    if peft_method == "prefix":
        if defense not in SUPPORTED_PEFT_TRAINING_POST_GRADIENT_DEFENSES:
            raise NotImplementedError(
                "Prefix PEFT training currently supports only post-gradient defenses: "
                f"{sorted(SUPPORTED_PEFT_TRAINING_POST_GRADIENT_DEFENSES)}; got defense={defense!r}. "
                "Direct-generation and DAGER training defenses remain unsupported for prefix."
            )
        return args

    raise NotImplementedError(
        f"PEFT training defense validation does not support peft_method={peft_method!r}."
    )


def is_gpt2_lora_model(model_path: str) -> bool:
    return model_path in GPT2_PEFT_MODELS


def is_llama_lora_model(model_path: str) -> bool:
    return model_path in LLAMA_PEFT_MODELS


def is_supported_lora_model(model_path: str) -> bool:
    return model_path in SUPPORTED_PEFT_MODELS


def is_supported_peft_model(model_path: str) -> bool:
    return model_path in SUPPORTED_PEFT_MODELS


def _as_tuple(value) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped.replace("'", '"'))
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, (list, tuple)):
                return _as_tuple(parsed)
        if "," in stripped:
            out = tuple(part.strip() for part in stripped.split(",") if part.strip())
            return out or None
        return (stripped,) if stripped else None
    if isinstance(value, (list, tuple, set)):
        out = tuple(str(item).strip() for item in value if str(item).strip())
        return out or None
    return None


def _format_modules(value) -> str:
    if not value:
        return "n/a"
    return ",".join(value)


def _as_positive_int(value: Any, *, field_name: str) -> int | None:
    if value in (None, "", "n/a"):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer; got {value!r}.") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer; got {value!r}.")
    return parsed


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc


def _adapter_architecture_from_config(config: Mapping[str, Any], metadata: Mapping[str, Any] | None = None) -> str | None:
    metadata = metadata or {}
    candidates = (
        metadata.get("adapter_architecture"),
        metadata.get("architecture"),
        config.get("architecture"),
        config.get("adapter_architecture"),
        config.get("config"),
    )
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            nested = _adapter_architecture_from_config(candidate)
            if nested:
                return nested
            continue
        if candidate is None:
            continue
        value = str(candidate).strip().lower().replace("-", "_")
        if value:
            return value
    return None


def _adapter_reduction_factor_from_config(
    config: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> int | None:
    metadata = metadata or {}
    candidates = (
        metadata.get("adapter_reduction_factor"),
        metadata.get("reduction_factor"),
        config.get("adapter_reduction_factor"),
        config.get("reduction_factor"),
        config.get("reduction_factor_default"),
    )
    for candidate in candidates:
        parsed = _as_positive_int(candidate, field_name="adapter_reduction_factor")
        if parsed is not None:
            return parsed
    nested = config.get("config")
    if isinstance(nested, Mapping):
        return _adapter_reduction_factor_from_config(nested)
    return None


def _metadata_peft_type(config: Mapping[str, Any], metadata: Mapping[str, Any]) -> str | None:
    peft_type = metadata.get("peft_type") or metadata.get("peft_method") or config.get("peft_type")
    if peft_type is not None:
        return str(peft_type)
    architecture = _adapter_architecture_from_config(config, metadata)
    if architecture in ADAPTER_ARCHITECTURE_ALIASES:
        return "ADAPTER"
    return None


def parse_lora_target_modules(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    value = raw.strip()
    if not value or value == "default":
        return None
    if "," in value:
        modules = tuple(part.strip() for part in value.split(",") if part.strip())
        if not modules:
            raise ValueError("--lora_target_modules cannot be empty.")
        return modules
    if value == "all-linear":
        return ("all-linear",)
    if value == "qv":
        return ("q_proj", "v_proj")
    if value == "qkvo":
        return ("q_proj", "k_proj", "v_proj", "o_proj")
    if value == "bert-qv":
        return ("query", "value")
    if value in {"c_attn", "q_proj"}:
        return (value,)
    raise ValueError(
        "Unsupported --lora_target_modules value. Use one of: "
        f"{', '.join(sorted(LORA_TARGET_PRESETS))}, or a comma-separated module list."
    )


def parse_ia3_target_modules(raw: str | None, model_path: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    value = (raw or "default").strip()
    if not value or value == "default":
        if model_path in GPT2_PEFT_MODELS:
            return ("c_attn", "c_fc"), ("c_fc",)
        if model_path in BERT_PEFT_MODELS:
            return ("query", "value", "intermediate.dense"), ("intermediate.dense",)
        if model_path in LLAMA_PEFT_MODELS:
            return ("q_proj", "v_proj", "down_proj"), ("down_proj",)
    if value == "bert-qv-ffn":
        return ("query", "value", "intermediate.dense"), ("intermediate.dense",)
    if value == "gpt2-attn-mlp":
        return ("c_attn", "c_fc"), ("c_fc",)
    if value == "llama-qv-ffn":
        return ("q_proj", "v_proj", "down_proj"), ("down_proj",)
    modules = tuple(part.strip() for part in value.split(",") if part.strip())
    if not modules:
        raise ValueError("--lora_target_modules cannot be empty for IA3.")
    return modules, tuple(module for module in modules if any(key in module for key in ("fc", "dense", "down_proj")))


def _validate_lora_target_modules_for_model(model_path: str, modules: tuple[str, ...]) -> None:
    if modules == ("all-linear",):
        return
    if model_path in GPT2_PEFT_MODELS:
        unsupported = [module for module in modules if module != "c_attn"]
        if unsupported:
            raise ValueError(
                "GPT-2 LoRA target modules currently support c_attn or all-linear; "
                f"got {format_lora_target_modules(modules)!r}."
            )
        return
    if model_path in BERT_PEFT_MODELS:
        unsupported = [module for module in modules if module not in {"query", "key", "value"}]
        if unsupported:
            raise ValueError(
                "BERT LoRA target modules currently support query/key/value or all-linear; "
                f"got {format_lora_target_modules(modules)!r}."
            )
        return
    if model_path in LLAMA_PEFT_MODELS:
        unsupported = [module for module in modules if module not in {"q_proj", "k_proj", "v_proj", "o_proj"}]
        if unsupported:
            raise ValueError(
                "Llama LoRA target modules currently support q_proj, qv, qkvo, or all-linear; "
                f"got {format_lora_target_modules(modules)!r}."
            )
        return


def lora_target_modules(model_path: str, target_modules: str | None = None) -> list[str]:
    parsed = parse_lora_target_modules(target_modules)
    if parsed is not None:
        _validate_lora_target_modules_for_model(model_path, parsed)
        if parsed == ("all-linear",):
            return ["all-linear"]
        return list(parsed)
    if model_path in GPT2_PEFT_MODELS:
        return ["c_attn"]
    if model_path in BERT_PEFT_MODELS:
        return ["query", "value"]
    if model_path in LLAMA_PEFT_MODELS:
        return ["q_proj"]
    raise ValueError(
        "PEFT eval currently supports GPT-2, BERT, and Llama model families; "
        f"got model_path={model_path!r}."
    )


def format_lora_supported_models() -> str:
    return ", ".join(sorted(SUPPORTED_PEFT_MODELS))


def format_lora_supported_defenses() -> str:
    return ", ".join(sorted(SUPPORTED_PEFT_DEFENSES))


def format_lora_target_modules(target_modules: tuple[str, ...] | list[str] | None) -> str:
    return _format_modules(tuple(target_modules) if target_modules else None)


def peft_task_type(task: str | None):
    if task == "seq_class":
        return peft.TaskType.SEQ_CLS
    if task == "next_token_pred":
        return peft.TaskType.CAUSAL_LM
    return None


def lora_task_type(task: str | None):
    return peft_task_type(task)


def _task_type_name(task_type) -> str | None:
    if task_type is None:
        return None
    value = str(getattr(task_type, "value", task_type)).upper()
    value = value.replace("TASKTYPE.", "").replace("-", "_").replace(" ", "_")
    aliases = {
        "SEQ_CLS": "SEQ_CLS",
        "SEQCLS": "SEQ_CLS",
        "SEQ_CLASSIFICATION": "SEQ_CLS",
        "SEQUENCE_CLASSIFICATION": "SEQ_CLS",
        "CAUSAL_LM": "CAUSAL_LM",
        "CAUSALLM": "CAUSAL_LM",
    }
    return aliases.get(value, value)


def peft_modules_to_save(model, task: str | None) -> list[str] | None:
    if task != "seq_class":
        return None
    modules = []
    for name in ("score", "classifier"):
        if hasattr(model, name):
            modules.append(name)
    return modules or None


def lora_modules_to_save(model, task: str | None) -> list[str] | None:
    return peft_modules_to_save(model, task)


def _require_adapter_backend():
    if adapterlib is None:
        raise ImportError(
            "Houlsby-style adapter PEFT requires the AdapterHub `adapters` package. "
            "Install it with `pip install adapters` or update the project environment."
        )
    return adapterlib


def _adapterhub_active(model) -> bool:
    """Detect AdapterHub support, not the unrelated Transformers PEFT mixin."""

    return all(hasattr(model, attr) for attr in ("add_adapter", "train_adapter", "save_adapter"))


def _init_adapter_backend(model):
    backend = _require_adapter_backend()
    if not _adapterhub_active(model):
        init_fn = getattr(backend, "init", None)
        if not callable(init_fn):
            raise ImportError("The installed `adapters` package does not expose adapters.init(model).")
        initialized = init_fn(model)
        if initialized is not None:
            model = initialized
    if not _adapterhub_active(model):
        raise ImportError(
            "AdapterHub did not attach add_adapter/train_adapter/save_adapter to this model instance. "
            "If Transformers exposes PeftAdapterMixin.add_adapter, adapters.init(model) is still required."
        )
    return model


def _add_adapter_compat(model, adapter_name: str, config) -> None:
    add_adapter = getattr(model, "add_adapter", None)
    if not callable(add_adapter):
        raise ImportError("AdapterHub did not expose add_adapter() on this model instance.")
    try:
        add_adapter(adapter_name, config=config)
    except TypeError as exc:
        try:
            add_adapter(adapter_name, config)
        except TypeError:
            raise exc


def _build_houlsby_adapter_config(reduction_factor: int):
    backend = _require_adapter_backend()
    reduction_factor = _as_positive_int(
        reduction_factor,
        field_name="--adapter_reduction_factor",
    ) or ADAPTER_DEFAULT_REDUCTION_FACTOR
    config_cls = getattr(backend, "DoubleSeqBnConfig", None)
    if config_cls is not None:
        return config_cls(reduction_factor=reduction_factor)
    adapter_config = getattr(backend, "AdapterConfig", None)
    if adapter_config is not None and hasattr(adapter_config, "load"):
        try:
            return adapter_config.load(ADAPTER_DEFAULT_ARCHITECTURE, reduction_factor=reduction_factor)
        except TypeError:
            return adapter_config.load(ADAPTER_DEFAULT_ARCHITECTURE)
    raise ImportError("The installed `adapters` package does not expose DoubleSeqBnConfig or AdapterConfig.load().")


def _set_seq_class_heads_trainable(model, task: str | None) -> None:
    for module_name in peft_modules_to_save(model, task) or []:
        module = getattr(model, module_name, None)
        if module is None or not hasattr(module, "parameters"):
            continue
        for param in module.parameters():
            param.requires_grad = True


def _adapter_name_from_model(model) -> str:
    for attr in ("_fedllm_adapter_name", "active_adapter"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    active = getattr(model, "active_adapters", None)
    if isinstance(active, str) and active:
        return active
    if active is not None:
        try:
            flattened = list(active)
        except TypeError:
            flattened = []
        for value in flattened:
            if isinstance(value, str) and value:
                return value
    return ADAPTER_DEFAULT_NAME


def _activate_adapter_for_training(model, adapter_name: str, task: str | None) -> None:
    if hasattr(model, "set_active_adapters"):
        model.set_active_adapters(adapter_name)
    if hasattr(model, "train_adapter"):
        model.train_adapter(adapter_name)
    _set_seq_class_heads_trainable(model, task)
    setattr(model, "_fedllm_adapter_name", adapter_name)


def _save_seq_class_head(model, adapter_dir: Path) -> str:
    state = {}
    for name in ("score", "classifier"):
        module = getattr(model, name, None)
        if module is not None and hasattr(module, "state_dict"):
            state[name] = module.state_dict()
    if not state:
        return "n/a"
    head_path = adapter_dir / PEFT_ADAPTER_HEAD_FILE
    torch.save(state, str(head_path))
    return str(head_path)


def _load_seq_class_head(model, adapter_dir: Path) -> None:
    head_path = adapter_dir / PEFT_ADAPTER_HEAD_FILE
    if not head_path.is_file():
        return
    state = torch.load(str(head_path), map_location=torch.device("cpu"))
    if not isinstance(state, Mapping):
        return
    for name, module_state in state.items():
        module = getattr(model, str(name), None)
        if module is not None and hasattr(module, "load_state_dict"):
            module.load_state_dict(module_state, strict=False)


def is_peft_adapter_dir(path: str | Path) -> bool:
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_dir():
        return False
    if not (checkpoint_path / PEFT_ADAPTER_CONFIG).is_file():
        return False
    if any((checkpoint_path / weight_file).is_file() for weight_file in PEFT_ADAPTER_WEIGHT_FILES):
        return True
    for child in checkpoint_path.iterdir():
        if not child.is_file():
            continue
        if child.name in {PEFT_ADAPTER_CONFIG, PEFT_METADATA_FILE, PEFT_ADAPTER_HEAD_FILE}:
            continue
        if child.suffix.lower() in PEFT_ADAPTER_WEIGHT_SUFFIXES:
            return True
    return False


def read_peft_adapter_config(path: str | Path) -> dict[str, Any]:
    adapter_dir = Path(path).expanduser()
    config_path = adapter_dir / PEFT_ADAPTER_CONFIG
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"PEFT adapter config does not exist: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid PEFT adapter config JSON: {config_path}") from exc


def read_peft_metadata(path: str | Path) -> dict[str, Any]:
    adapter_dir = Path(path).expanduser()
    return _read_json_file(adapter_dir / PEFT_METADATA_FILE)


def resolve_peft_checkpoint_path(path: str) -> Path:
    checkpoint_path = Path(path).expanduser()
    if checkpoint_path.is_dir():
        if is_peft_adapter_dir(checkpoint_path):
            return checkpoint_path
        raise ValueError(
            "PEFT checkpoint directories must be adapter directories "
            f"containing {PEFT_ADAPTER_CONFIG!r} and adapter weights; got {path!r}."
        )
    if checkpoint_path.suffix.lower() not in LEGACY_LORA_STATE_SUFFIXES:
        raise ValueError(
            "PEFT eval supports PEFT adapter directories or legacy local LoRA .pt/.pth checkpoints; "
            f"got {path!r}."
        )
    if not checkpoint_path.is_file():
        raise ValueError(f"PEFT checkpoint file does not exist: {path!r}.")
    return checkpoint_path


def resolve_lora_checkpoint_path(path: str) -> Path:
    return resolve_peft_checkpoint_path(path)


def peft_checkpoint_metadata(path: str | None) -> PeftCheckpointMetadata:
    if path is None:
        return PeftCheckpointMetadata(path=None, checkpoint_type="new")
    checkpoint_path = resolve_peft_checkpoint_path(path)
    if not checkpoint_path.is_dir():
        return PeftCheckpointMetadata(
            path=checkpoint_path,
            checkpoint_type="legacy_state_dict",
            peft_type="LORA",
            adapter_base_model_name_or_path=None,
        )

    config = read_peft_adapter_config(checkpoint_path)
    fedllm_metadata = read_peft_metadata(checkpoint_path)
    r = config.get("r")
    num_virtual_tokens = config.get("num_virtual_tokens")
    target_modules = _as_tuple(config.get("target_modules"))
    feedforward_modules = _as_tuple(config.get("feedforward_modules"))
    if feedforward_modules is None:
        feedforward_modules = _as_tuple(config.get("feedforward_modules_"))
    architecture = _adapter_architecture_from_config(config, fedllm_metadata)
    reduction_factor = _adapter_reduction_factor_from_config(config, fedllm_metadata)
    peft_type = _metadata_peft_type(config, fedllm_metadata)
    return PeftCheckpointMetadata(
        path=checkpoint_path,
        checkpoint_type="adapter_dir",
        peft_type=peft_type,
        adapter_r=int(r) if r is not None else None,
        adapter_target_modules=target_modules,
        adapter_feedforward_modules=feedforward_modules,
        adapter_num_virtual_tokens=int(num_virtual_tokens) if num_virtual_tokens is not None else None,
        adapter_task_type=fedllm_metadata.get("task_type") or config.get("task_type"),
        adapter_base_model_name_or_path=(
            fedllm_metadata.get("base_model_name_or_path") or config.get("base_model_name_or_path")
        ),
        adapter_reduction_factor=reduction_factor,
        adapter_architecture=architecture,
        adapter_name=fedllm_metadata.get("adapter_name"),
    )


def lora_checkpoint_metadata(path: str | None) -> PeftCheckpointMetadata:
    return peft_checkpoint_metadata(path)


def load_lora_checkpoint(path: str):
    checkpoint_path = resolve_peft_checkpoint_path(path)
    if checkpoint_path.is_dir():
        raise ValueError(f"Use PeftModel.from_pretrained for PEFT adapter directories: {path!r}.")
    return torch.load(checkpoint_path, map_location=torch.device("cpu"))


def _resolve_method(peft_method: str | None, metadata: PeftCheckpointMetadata) -> str:
    peft_method = normalize_peft_method_name(peft_method)
    if metadata.peft_type:
        adapter_method = normalize_peft_method_name(metadata.peft_type)
        if peft_method is not None and peft_method != adapter_method:
            raise ValueError(
                "--peft_method does not match adapter_config.json peft_type: "
                f"cli={peft_method!r}, adapter={adapter_method!r}."
            )
        return adapter_method
    return peft_method or "lora"


def resolve_peft_config(
    *,
    model_path: str,
    peft_method: str | None = None,
    lora_r: int | None = None,
    checkpoint_path: str | None = None,
    task: str | None = None,
    target_modules: str | None = None,
    peft_num_virtual_tokens: int | None = None,
    adapter_reduction_factor: int | None = None,
    require_checkpoint: bool = False,
) -> PeftResolvedConfig:
    if not is_supported_peft_model(model_path):
        raise ValueError(
            "PEFT eval currently supports GPT-2, BERT, and Llama model families. "
            f"Supported model_path values: {format_lora_supported_models()}."
        )
    if require_checkpoint and checkpoint_path is None:
        raise ValueError("--train_method peft requires --finetuned_path PATH to an existing adapter.")

    metadata = peft_checkpoint_metadata(checkpoint_path)
    method = _resolve_method(peft_method, metadata)
    if method not in SUPPORTED_PEFT_METHODS:
        raise NotImplementedError(
            f"Unsupported --peft_method {method!r}; PEFT supports {sorted(SUPPORTED_PEFT_METHODS)}."
        )
    if metadata.checkpoint_type == "legacy_state_dict" and method != "lora":
        raise ValueError("Legacy .pt/.pth PEFT checkpoints are only supported for LoRA.")

    expected_task_type = _task_type_name(peft_task_type(task))
    adapter_task_type = _task_type_name(metadata.adapter_task_type)
    if metadata.checkpoint_type == "adapter_dir" and expected_task_type and adapter_task_type:
        if adapter_task_type != expected_task_type:
            raise ValueError(
                "--task does not match PEFT adapter config task_type: "
                f"cli={expected_task_type!r}, adapter={adapter_task_type!r}."
            )

    adapter_base_model = metadata.adapter_base_model_name_or_path
    if (
        metadata.checkpoint_type == "adapter_dir"
        and adapter_base_model in SUPPORTED_PEFT_MODELS
        and adapter_base_model != model_path
    ):
        raise ValueError(
            "--model_path does not match PEFT adapter config base_model_name_or_path: "
            f"cli={model_path!r}, adapter={adapter_base_model!r}."
        )

    if method == "lora":
        explicit_modules = parse_lora_target_modules(target_modules)
        if metadata.adapter_target_modules is not None:
            _validate_lora_target_modules_for_model(model_path, tuple(metadata.adapter_target_modules))
            if explicit_modules is not None and tuple(explicit_modules) != tuple(metadata.adapter_target_modules):
                raise ValueError(
                    "--lora_target_modules does not match PEFT adapter config: "
                    f"cli={format_lora_target_modules(explicit_modules)!r}, "
                    f"adapter={format_lora_target_modules(metadata.adapter_target_modules)!r}."
                )
            explicit_modules = tuple(metadata.adapter_target_modules)
        resolved_r = lora_r
        if metadata.adapter_r is not None:
            if resolved_r is not None and int(resolved_r) != int(metadata.adapter_r):
                raise ValueError(
                    "--lora_r does not match PEFT adapter config: "
                    f"cli={resolved_r!r}, adapter r={metadata.adapter_r!r}."
                )
            resolved_r = int(metadata.adapter_r)
        if metadata.checkpoint_type == "legacy_state_dict" and resolved_r is None:
            raise ValueError("--train_method peft/lora with a .pt/.pth checkpoint requires --lora_r R.")
        if resolved_r is None:
            raise ValueError("--peft_method lora requires --lora_r R.")
        if int(resolved_r) <= 0:
            raise ValueError(f"--lora_r must be a positive integer; got {resolved_r!r}.")
        modules = explicit_modules or tuple(lora_target_modules(model_path))
        _validate_lora_target_modules_for_model(model_path, tuple(modules))
        return PeftResolvedConfig(method, int(resolved_r), tuple(modules), None, None, metadata)

    if method == "ia3":
        target, feedforward = parse_ia3_target_modules(target_modules, model_path)
        if metadata.adapter_target_modules is not None:
            target = tuple(metadata.adapter_target_modules)
        if metadata.adapter_feedforward_modules is not None:
            feedforward = tuple(metadata.adapter_feedforward_modules)
        return PeftResolvedConfig(method, None, target, feedforward, None, metadata)

    if method == "prefix":
        n_tokens = metadata.adapter_num_virtual_tokens
        if peft_num_virtual_tokens is not None:
            if n_tokens is not None and int(peft_num_virtual_tokens) != int(n_tokens):
                raise ValueError(
                    "--peft_num_virtual_tokens does not match PEFT adapter config: "
                    f"cli={peft_num_virtual_tokens!r}, adapter={n_tokens!r}."
                )
            n_tokens = peft_num_virtual_tokens
        if n_tokens is None:
            n_tokens = 20
        if int(n_tokens) <= 0:
            raise ValueError("--peft_num_virtual_tokens must be positive.")
        if model_path in LLAMA_PEFT_MODELS:
            raise NotImplementedError("Prefix tuning for Llama is planned for v2; v1 supports BERT and GPT-2.")
        return PeftResolvedConfig(method, None, None, None, int(n_tokens), metadata)

    if method == "adapter":
        reduction = adapter_reduction_factor
        if metadata.adapter_reduction_factor is not None:
            if reduction is not None and int(reduction) != int(metadata.adapter_reduction_factor):
                raise ValueError(
                    "--adapter_reduction_factor does not match adapter metadata: "
                    f"cli={reduction!r}, adapter={metadata.adapter_reduction_factor!r}."
                )
            reduction = int(metadata.adapter_reduction_factor)
        if reduction is None:
            reduction = ADAPTER_DEFAULT_REDUCTION_FACTOR
        reduction = _as_positive_int(reduction, field_name="--adapter_reduction_factor")
        return PeftResolvedConfig(
            method,
            None,
            None,
            None,
            None,
            metadata,
            adapter_reduction_factor=reduction,
        )

    raise AssertionError(f"Unhandled PEFT method: {method}")


def resolve_lora_config(
    *,
    model_path: str,
    lora_r: int | None,
    checkpoint_path: str | None = None,
    task: str | None = None,
    target_modules: str | None = None,
    require_checkpoint: bool = False,
) -> PeftResolvedConfig:
    return resolve_peft_config(
        model_path=model_path,
        peft_method="lora",
        lora_r=lora_r,
        checkpoint_path=checkpoint_path,
        task=task,
        target_modules=target_modules,
        peft_num_virtual_tokens=None,
        require_checkpoint=require_checkpoint,
    )


def apply_peft_config_to_args(args, *, require_checkpoint: bool = False):
    normalize_peft_args(args)
    if not peft_active(args):
        return args
    resolved = resolve_peft_config(
        model_path=getattr(args, "model_path", ""),
        peft_method=getattr(args, "peft_method", None),
        lora_r=getattr(args, "lora_r", None),
        checkpoint_path=getattr(args, "finetuned_path", None),
        task=getattr(args, "task", None),
        target_modules=getattr(args, "lora_target_modules", None),
        peft_num_virtual_tokens=getattr(args, "peft_num_virtual_tokens", None),
        adapter_reduction_factor=getattr(args, "adapter_reduction_factor", None),
        require_checkpoint=require_checkpoint,
    )
    args.train_method = "peft"
    args.peft_method = resolved.peft_method
    base_model = resolved.metadata.adapter_base_model_name_or_path
    if base_model is None and resolved.metadata.checkpoint_type == "legacy_state_dict":
        base_model = getattr(args, "model_path", None)
    args.lora_r = resolved.lora_r
    args.lora_target_modules = format_lora_target_modules(resolved.target_modules)
    args.peft_target_modules = format_lora_target_modules(resolved.target_modules)
    args.peft_feedforward_modules = format_lora_target_modules(resolved.feedforward_modules)
    args.peft_num_virtual_tokens = resolved.num_virtual_tokens
    args.adapter_reduction_factor = resolved.adapter_reduction_factor
    args.peft_checkpoint_type = resolved.metadata.checkpoint_type
    args.peft_adapter_r = resolved.metadata.adapter_r
    args.peft_adapter_target_modules = format_lora_target_modules(resolved.metadata.adapter_target_modules)
    args.peft_adapter_feedforward_modules = format_lora_target_modules(resolved.metadata.adapter_feedforward_modules)
    args.peft_adapter_task_type = resolved.metadata.adapter_task_type
    args.peft_adapter_base_model = base_model
    args.peft_adapter_peft_type = resolved.metadata.peft_type
    args.peft_adapter_reduction_factor = resolved.metadata.adapter_reduction_factor
    args.peft_adapter_architecture = resolved.metadata.adapter_architecture
    args.peft_adapter_name = resolved.metadata.adapter_name
    args.peft_type = resolved.metadata.peft_type or _method_to_peft_type(resolved.peft_method)
    args.peft_eval_scope = peft_eval_scope(resolved.peft_method)

    # Backward-compatible LoRA summary fields.
    args.lora_checkpoint_type = args.peft_checkpoint_type
    args.lora_adapter_r = args.peft_adapter_r
    args.lora_adapter_target_modules = args.peft_adapter_target_modules
    args.lora_adapter_task_type = args.peft_adapter_task_type
    args.lora_adapter_base_model = args.peft_adapter_base_model
    args.lora_adapter_peft_type = args.peft_adapter_peft_type
    args.lora_adapter_feedforward_modules = args.peft_adapter_feedforward_modules
    return args


def apply_lora_config_to_args(args, *, require_checkpoint: bool = False):
    args.train_method = "peft" if getattr(args, "train_method", "full") == "lora" else getattr(args, "train_method", "full")
    args.peft_method = "lora"
    return apply_peft_config_to_args(args, require_checkpoint=require_checkpoint)


def build_peft_config(
    model,
    *,
    model_path: str,
    peft_method: str,
    lora_r: int | None = None,
    task: str | None = None,
    target_modules: str | None = None,
    peft_num_virtual_tokens: int | None = None,
    adapter_reduction_factor: int | None = None,
):
    peft_method = normalize_peft_method_name(peft_method) or "lora"
    task_type = peft_task_type(task)
    modules_to_save = peft_modules_to_save(model, task)

    if peft_method == "lora":
        kwargs = {"r": int(lora_r)}
        target = lora_target_modules(model_path, target_modules)
        kwargs["target_modules"] = "all-linear" if target == ["all-linear"] else target
        if task_type is not None:
            kwargs["task_type"] = task_type
        if modules_to_save is not None:
            kwargs["modules_to_save"] = modules_to_save
        if model_path in GPT2_PEFT_MODELS:
            kwargs["fan_in_fan_out"] = True
        return peft.LoraConfig(**kwargs)

    if peft_method == "ia3":
        target, feedforward = parse_ia3_target_modules(target_modules, model_path)
        kwargs = {
            "target_modules": list(target),
            "feedforward_modules": list(feedforward),
        }
        if task_type is not None:
            kwargs["task_type"] = task_type
        if modules_to_save is not None:
            kwargs["modules_to_save"] = modules_to_save
        return peft.IA3Config(**kwargs)

    if peft_method == "prefix":
        if model_path in LLAMA_PEFT_MODELS:
            raise NotImplementedError("Prefix tuning for Llama is planned for v2; v1 supports BERT and GPT-2.")
        n_tokens = 20 if peft_num_virtual_tokens is None else int(peft_num_virtual_tokens)
        if n_tokens <= 0:
            raise ValueError("--peft_num_virtual_tokens must be positive.")
        kwargs = {"num_virtual_tokens": n_tokens}
        if task_type is not None:
            kwargs["task_type"] = task_type
        return peft.PrefixTuningConfig(**kwargs)

    if peft_method == "adapter":
        reduction = ADAPTER_DEFAULT_REDUCTION_FACTOR if adapter_reduction_factor is None else int(adapter_reduction_factor)
        return _build_houlsby_adapter_config(reduction)
    raise NotImplementedError(f"Unsupported PEFT method: {peft_method!r}")


def build_lora_config(
    model,
    *,
    model_path: str,
    lora_r: int,
    task: str | None = None,
    target_modules: str | None = None,
):
    return build_peft_config(
        model,
        model_path=model_path,
        peft_method="lora",
        lora_r=lora_r,
        task=task,
        target_modules=target_modules,
    )


def unwrap_peft_model(model):
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    if hasattr(model, "model"):
        return model.model
    return model


def load_legacy_lora_state_dict(lora_model, checkpoint_path: str | Path):
    state_dict = load_lora_checkpoint(str(checkpoint_path))
    if hasattr(peft, "set_peft_model_state_dict"):
        load_result = peft.set_peft_model_state_dict(lora_model, state_dict)
    else:
        load_result = lora_model.load_state_dict(state_dict, strict=False)
    missing = getattr(load_result, "missing_keys", [])
    unexpected = getattr(load_result, "unexpected_keys", [])
    if missing:
        print(f"[dager] LoRA checkpoint loaded with missing keys: {len(missing)}", flush=True)
    if unexpected:
        print(f"[dager] LoRA checkpoint loaded with unexpected keys: {len(unexpected)}", flush=True)
    return lora_model


def apply_peft_adapter(
    model,
    *,
    model_path: str,
    peft_method: str | None = None,
    lora_r: int | None = None,
    checkpoint_path: str | None = None,
    unwrap_base_model: bool = False,
    task: str | None = None,
    target_modules: str | None = None,
    peft_num_virtual_tokens: int | None = None,
    adapter_reduction_factor: int | None = None,
):
    resolved = resolve_peft_config(
        model_path=model_path,
        peft_method=peft_method,
        lora_r=lora_r,
        checkpoint_path=checkpoint_path,
        task=task,
        target_modules=target_modules,
        peft_num_virtual_tokens=peft_num_virtual_tokens,
        adapter_reduction_factor=adapter_reduction_factor,
        require_checkpoint=False,
    )
    resolved_checkpoint = resolve_peft_checkpoint_path(checkpoint_path) if checkpoint_path is not None else None
    if resolved.peft_method == "adapter":
        adapter_model = _init_adapter_backend(model)
        adapter_name = resolved.metadata.adapter_name or ADAPTER_DEFAULT_NAME
        if resolved_checkpoint is not None:
            if not resolved_checkpoint.is_dir():
                raise ValueError("AdapterHub checkpoints must be directories; .pt/.pth is reserved for legacy LoRA.")
            try:
                loaded_name = adapter_model.load_adapter(
                    str(resolved_checkpoint),
                    load_as=adapter_name,
                    set_active=True,
                )
            except TypeError:
                try:
                    loaded_name = adapter_model.load_adapter(str(resolved_checkpoint), set_active=True)
                except TypeError:
                    loaded_name = adapter_model.load_adapter(str(resolved_checkpoint))
            if isinstance(loaded_name, str) and loaded_name:
                adapter_name = loaded_name
            _load_seq_class_head(adapter_model, resolved_checkpoint)
        else:
            cfg = build_peft_config(
                adapter_model,
                model_path=model_path,
                peft_method=resolved.peft_method,
                lora_r=None,
                task=task,
                target_modules=None,
                peft_num_virtual_tokens=None,
                adapter_reduction_factor=resolved.adapter_reduction_factor,
            )
            _add_adapter_compat(adapter_model, adapter_name, cfg)
        _activate_adapter_for_training(adapter_model, adapter_name, task)
        setattr(adapter_model, "_fedllm_adapter_reduction_factor", resolved.adapter_reduction_factor)
        setattr(adapter_model, "_fedllm_adapter_architecture", ADAPTER_DEFAULT_ARCHITECTURE)
        setattr(adapter_model, "_fedllm_adapter_task_type", _task_type_name(peft_task_type(task)))
        setattr(adapter_model, "_fedllm_base_model_name_or_path", model_path)
        return unwrap_peft_model(adapter_model) if unwrap_base_model else adapter_model

    if resolved_checkpoint is not None and resolved_checkpoint.is_dir():
        peft_model = peft.PeftModel.from_pretrained(model, str(resolved_checkpoint), is_trainable=True)
    else:
        cfg = build_peft_config(
            model,
            model_path=model_path,
            peft_method=resolved.peft_method,
            lora_r=resolved.lora_r,
            task=task,
            target_modules=format_lora_target_modules(resolved.target_modules),
            peft_num_virtual_tokens=resolved.num_virtual_tokens,
            adapter_reduction_factor=None,
        )
        peft_model = peft.get_peft_model(model, cfg)
        if resolved_checkpoint is not None:
            load_legacy_lora_state_dict(peft_model, resolved_checkpoint)
    return unwrap_peft_model(peft_model) if unwrap_base_model else peft_model


def apply_lora_adapter(
    model,
    *,
    model_path: str,
    lora_r: int,
    checkpoint_path: str | None = None,
    unwrap_base_model: bool = False,
    task: str | None = None,
    target_modules: str | None = None,
):
    return apply_peft_adapter(
        model,
        model_path=model_path,
        peft_method="lora",
        lora_r=lora_r,
        checkpoint_path=checkpoint_path,
        unwrap_base_model=unwrap_base_model,
        task=task,
        target_modules=target_modules,
    )


def peft_adapter_dir_for(save_path: str | Path) -> Path:
    path = Path(save_path)
    if path.suffix.lower() in LEGACY_LORA_STATE_SUFFIXES:
        return path.with_suffix("")
    return path


def lora_adapter_dir_for(save_path: str | Path) -> Path:
    return peft_adapter_dir_for(save_path)


def lora_legacy_state_path_for(save_path: str | Path) -> Path:
    path = Path(save_path)
    if path.suffix.lower() in LEGACY_LORA_STATE_SUFFIXES:
        return path
    return path.with_suffix(".pt")


def save_peft_checkpoint(model, tokenizer, save_path: str | Path) -> dict[str, str]:
    adapter_dir = peft_adapter_dir_for(save_path)
    legacy_state_path = lora_legacy_state_path_for(save_path)
    adapter_dir.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "save_adapter") and not isinstance(model, peft.PeftModel):
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_name = _adapter_name_from_model(model)
        model.save_adapter(str(adapter_dir), adapter_name)
        head_path = _save_seq_class_head(model, adapter_dir)
        metadata = {
            "peft_type": "ADAPTER",
            "peft_method": "adapter",
            "adapter_name": adapter_name,
            "adapter_architecture": getattr(model, "_fedllm_adapter_architecture", ADAPTER_DEFAULT_ARCHITECTURE),
            "adapter_reduction_factor": getattr(
                model,
                "_fedllm_adapter_reduction_factor",
                ADAPTER_DEFAULT_REDUCTION_FACTOR,
            ),
            "task_type": getattr(model, "_fedllm_adapter_task_type", None),
            "base_model_name_or_path": getattr(
                model,
                "_fedllm_base_model_name_or_path",
                getattr(getattr(model, "config", None), "_name_or_path", None),
            ),
        }
        (adapter_dir / PEFT_METADATA_FILE).write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        tokenizer_dir = adapter_dir.parent / f"{adapter_dir.name}_tokenizer"
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(tokenizer_dir))
        return {
            "primary_path": str(adapter_dir),
            "adapter_path": str(adapter_dir),
            "legacy_state_dict_path": "n/a",
            "tokenizer_path": str(tokenizer_dir),
            "adapter_head_path": head_path,
            "metadata_path": str(adapter_dir / PEFT_METADATA_FILE),
        }

    if not isinstance(model, peft.PeftModel):
        raise ValueError("PEFT native save requires a PeftModel or AdapterHub-enabled model; got a non-PEFT model.")

    model.save_pretrained(str(adapter_dir))
    legacy_state_path_value = "n/a"
    peft_type = getattr(model.peft_config.get("default"), "peft_type", "")
    if normalize_peft_method_name(peft_type) == "lora":
        legacy_state_path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = peft.get_peft_model_state_dict(model)
        torch.save(state_dict, str(legacy_state_path))
        legacy_state_path_value = str(legacy_state_path)
    tokenizer_dir = adapter_dir.parent / f"{adapter_dir.name}_tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(tokenizer_dir))

    return {
        "primary_path": str(adapter_dir),
        "adapter_path": str(adapter_dir),
        "legacy_state_dict_path": legacy_state_path_value,
        "tokenizer_path": str(tokenizer_dir),
    }


def save_lora_checkpoint(model, tokenizer, save_path: str | Path) -> dict[str, str]:
    return save_peft_checkpoint(model, tokenizer, save_path)


def validate_peft_eval_args(args):
    normalize_peft_args(args)
    if not peft_active(args):
        return args
    apply_peft_config_to_args(args, require_checkpoint=True)
    if peft_eval_scope(getattr(args, "peft_method", None)) != PEFT_EVAL_SCOPE_DAGER:
        raise NotImplementedError(peft_eval_scope_message(getattr(args, "peft_method", None)))
    defense = getattr(args, "defense", "none")
    if defense not in SUPPORTED_PEFT_DEFENSES:
        raise NotImplementedError(
            "PEFT eval currently supports only these defenses: "
            f"{format_lora_supported_defenses()}; got defense={defense!r}."
        )
    return args


def validate_lora_eval_args(args):
    if getattr(args, "train_method", "full") == "lora":
        args.peft_method = "lora"
    return validate_peft_eval_args(args)
