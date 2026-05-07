from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import peft
import torch


GPT2_LORA_MODELS = frozenset(
    {
        "gpt2",
        "openai-community/gpt2-large",
    }
)

LLAMA_LORA_MODELS = frozenset(
    {
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-70b-hf",
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Meta-Llama-3.1-8B",
        "meta-llama/Meta-Llama-3-70B",
    }
)

SUPPORTED_LORA_MODELS = GPT2_LORA_MODELS | LLAMA_LORA_MODELS
SUPPORTED_LORA_DEFENSES = frozenset({"none", "noise", "topk", "compression", "lrb"})
LEGACY_LORA_STATE_SUFFIXES = frozenset({".pt", ".pth"})
PEFT_ADAPTER_CONFIG = "adapter_config.json"
PEFT_ADAPTER_WEIGHT_FILES = frozenset(
    {
        "adapter_model.bin",
        "adapter_model.safetensors",
    }
)

LORA_TARGET_PRESETS = frozenset({"default", "c_attn", "q_proj", "qv", "qkvo", "all-linear"})


@dataclass(frozen=True)
class LoraCheckpointMetadata:
    path: Path | None
    checkpoint_type: str
    adapter_r: int | None = None
    adapter_target_modules: tuple[str, ...] | None = None
    adapter_task_type: str | None = None
    adapter_base_model_name_or_path: str | None = None
    adapter_peft_type: str | None = None


@dataclass(frozen=True)
class LoraResolvedConfig:
    lora_r: int | None
    target_modules: tuple[str, ...]
    metadata: LoraCheckpointMetadata


def is_gpt2_lora_model(model_path: str) -> bool:
    return model_path in GPT2_LORA_MODELS


def is_llama_lora_model(model_path: str) -> bool:
    return model_path in LLAMA_LORA_MODELS


def is_supported_lora_model(model_path: str) -> bool:
    return model_path in SUPPORTED_LORA_MODELS


def _as_tuple(value) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else None
    if isinstance(value, (list, tuple, set)):
        out = tuple(str(item).strip() for item in value if str(item).strip())
        return out or None
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
    if value in {"c_attn", "q_proj"}:
        return (value,)
    raise ValueError(
        "Unsupported --lora_target_modules value. Use one of: "
        f"{', '.join(sorted(LORA_TARGET_PRESETS))}, or a comma-separated module list."
    )


def _validate_lora_target_modules_for_model(model_path: str, modules: tuple[str, ...]) -> None:
    if modules == ("all-linear",):
        return
    if is_gpt2_lora_model(model_path):
        unsupported = [module for module in modules if module != "c_attn"]
        if unsupported:
            raise ValueError(
                "GPT-2 LoRA target modules currently support c_attn or all-linear; "
                f"got {format_lora_target_modules(modules)!r}."
            )
        return
    if is_llama_lora_model(model_path):
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
    if is_gpt2_lora_model(model_path):
        return ["c_attn"]
    if is_llama_lora_model(model_path):
        return ["q_proj"]
    raise ValueError(
        "LoRA/PEFT eval currently supports GPT-2 and Llama model families only; "
        f"got model_path={model_path!r}."
    )


def format_lora_supported_models() -> str:
    return ", ".join(sorted(SUPPORTED_LORA_MODELS))


def format_lora_supported_defenses() -> str:
    return ", ".join(sorted(SUPPORTED_LORA_DEFENSES))


def format_lora_target_modules(target_modules: tuple[str, ...] | list[str] | None) -> str:
    if not target_modules:
        return "n/a"
    return ",".join(target_modules)


def lora_task_type(task: str | None):
    if task == "seq_class":
        return peft.TaskType.SEQ_CLS
    if task == "next_token_pred":
        return peft.TaskType.CAUSAL_LM
    return None


def _task_type_name(task_type) -> str | None:
    if task_type is None:
        return None
    return str(getattr(task_type, "value", task_type)).upper()


def lora_modules_to_save(model, task: str | None) -> list[str] | None:
    if task != "seq_class":
        return None

    modules = []
    for name in ("score", "classifier"):
        if hasattr(model, name):
            modules.append(name)
    return modules or None


def is_peft_adapter_dir(path: str | Path) -> bool:
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_dir():
        return False
    if not (checkpoint_path / PEFT_ADAPTER_CONFIG).is_file():
        return False
    return any((checkpoint_path / weight_file).is_file() for weight_file in PEFT_ADAPTER_WEIGHT_FILES)


def read_peft_adapter_config(path: str | Path) -> dict[str, Any]:
    adapter_dir = Path(path).expanduser()
    config_path = adapter_dir / PEFT_ADAPTER_CONFIG
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"PEFT adapter config does not exist: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid PEFT adapter config JSON: {config_path}") from exc


def lora_checkpoint_metadata(path: str | None) -> LoraCheckpointMetadata:
    if path is None:
        return LoraCheckpointMetadata(path=None, checkpoint_type="new")
    checkpoint_path = resolve_lora_checkpoint_path(path)
    if not checkpoint_path.is_dir():
        return LoraCheckpointMetadata(path=checkpoint_path, checkpoint_type="legacy_state_dict")

    config = read_peft_adapter_config(checkpoint_path)
    target_modules = _as_tuple(config.get("target_modules"))
    peft_type = config.get("peft_type")
    if peft_type is not None and str(peft_type).upper() != "LORA":
        raise ValueError(
            "Only LoRA PEFT adapter directories are supported in this path; "
            f"got peft_type={peft_type!r}."
        )
    r = config.get("r")
    adapter_r = int(r) if r is not None else None
    return LoraCheckpointMetadata(
        path=checkpoint_path,
        checkpoint_type="adapter_dir",
        adapter_r=adapter_r,
        adapter_target_modules=target_modules,
        adapter_task_type=config.get("task_type"),
        adapter_base_model_name_or_path=config.get("base_model_name_or_path"),
        adapter_peft_type=peft_type,
    )


def resolve_lora_checkpoint_path(path: str) -> Path:
    checkpoint_path = Path(path).expanduser()
    if checkpoint_path.is_dir():
        if is_peft_adapter_dir(checkpoint_path):
            return checkpoint_path
        raise ValueError(
            "LoRA/PEFT checkpoint directories must be PEFT adapter directories "
            f"containing {PEFT_ADAPTER_CONFIG!r} and adapter weights; got {path!r}."
        )
    if checkpoint_path.suffix.lower() not in LEGACY_LORA_STATE_SUFFIXES:
        raise ValueError(
            "LoRA/PEFT eval supports PEFT adapter directories or local .pt/.pth state_dict checkpoints; "
            f"got {path!r}."
        )
    if not checkpoint_path.is_file():
        raise ValueError(f"LoRA checkpoint file does not exist: {path!r}.")
    return checkpoint_path


def load_lora_checkpoint(path: str):
    checkpoint_path = resolve_lora_checkpoint_path(path)
    if checkpoint_path.is_dir():
        raise ValueError(f"Use PeftModel.from_pretrained for PEFT adapter directories: {path!r}.")
    return torch.load(checkpoint_path, map_location=torch.device("cpu"))


def resolve_lora_config(
    *,
    model_path: str,
    lora_r: int | None,
    checkpoint_path: str | None = None,
    task: str | None = None,
    target_modules: str | None = None,
    require_checkpoint: bool = False,
) -> LoraResolvedConfig:
    if not is_supported_lora_model(model_path):
        raise ValueError(
            "LoRA/PEFT eval currently supports GPT-2 and Llama model families only. "
            f"Supported model_path values: {format_lora_supported_models()}."
        )

    if require_checkpoint and checkpoint_path is None:
        raise ValueError(
            "--train_method lora requires --finetuned_path PATH to an existing "
            "PEFT adapter directory or LoRA .pt/.pth checkpoint."
        )

    metadata = lora_checkpoint_metadata(checkpoint_path)
    adapter_modules = metadata.adapter_target_modules
    explicit_modules = parse_lora_target_modules(target_modules)

    expected_task_type = _task_type_name(lora_task_type(task))
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
        and adapter_base_model in SUPPORTED_LORA_MODELS
        and adapter_base_model != model_path
    ):
        raise ValueError(
            "--model_path does not match PEFT adapter config base_model_name_or_path: "
            f"cli={model_path!r}, adapter={adapter_base_model!r}."
        )

    resolved_r = lora_r
    if metadata.checkpoint_type == "adapter_dir":
        if metadata.adapter_r is not None:
            if resolved_r is not None and int(resolved_r) != int(metadata.adapter_r):
                raise ValueError(
                    "--lora_r does not match PEFT adapter config: "
                    f"cli={resolved_r!r}, adapter r={metadata.adapter_r!r}."
                )
            resolved_r = int(metadata.adapter_r)
        if adapter_modules is not None:
            _validate_lora_target_modules_for_model(model_path, tuple(adapter_modules))
            if explicit_modules is not None and tuple(explicit_modules) != tuple(adapter_modules):
                raise ValueError(
                    "--lora_target_modules does not match PEFT adapter config: "
                    f"cli={format_lora_target_modules(explicit_modules)!r}, "
                    f"adapter={format_lora_target_modules(adapter_modules)!r}."
                )
            explicit_modules = tuple(adapter_modules)
    elif metadata.checkpoint_type == "legacy_state_dict" and resolved_r is None:
        raise ValueError("--train_method lora with a .pt/.pth checkpoint requires --lora_r R.")

    if resolved_r is None:
        raise ValueError("--train_method lora requires --lora_r R.")
    if int(resolved_r) <= 0:
        raise ValueError(f"--lora_r must be a positive integer; got {resolved_r!r}.")

    modules = explicit_modules or tuple(lora_target_modules(model_path))
    _validate_lora_target_modules_for_model(model_path, tuple(modules))
    return LoraResolvedConfig(
        lora_r=int(resolved_r),
        target_modules=tuple(modules),
        metadata=metadata,
    )


def apply_lora_config_to_args(args, *, require_checkpoint: bool = False):
    if getattr(args, "train_method", "full") != "lora":
        return args
    resolved = resolve_lora_config(
        model_path=getattr(args, "model_path", ""),
        lora_r=getattr(args, "lora_r", None),
        checkpoint_path=getattr(args, "finetuned_path", None),
        task=getattr(args, "task", None),
        target_modules=getattr(args, "lora_target_modules", None),
        require_checkpoint=require_checkpoint,
    )
    args.lora_r = resolved.lora_r
    args.lora_target_modules = format_lora_target_modules(resolved.target_modules)
    args.lora_checkpoint_type = resolved.metadata.checkpoint_type
    args.lora_adapter_r = resolved.metadata.adapter_r
    args.lora_adapter_target_modules = format_lora_target_modules(resolved.metadata.adapter_target_modules)
    args.lora_adapter_task_type = resolved.metadata.adapter_task_type
    args.lora_adapter_base_model = resolved.metadata.adapter_base_model_name_or_path
    args.lora_adapter_peft_type = resolved.metadata.adapter_peft_type
    return args


def build_lora_config(
    model,
    *,
    model_path: str,
    lora_r: int,
    task: str | None = None,
    target_modules: str | None = None,
):
    kwargs = {
        "r": int(lora_r),
    }
    target = lora_target_modules(model_path, target_modules)
    kwargs["target_modules"] = "all-linear" if target == ["all-linear"] else target
    task_type = lora_task_type(task)
    if task_type is not None:
        kwargs["task_type"] = task_type
    modules_to_save = lora_modules_to_save(model, task)
    if modules_to_save is not None:
        kwargs["modules_to_save"] = modules_to_save
    if is_gpt2_lora_model(model_path):
        kwargs["fan_in_fan_out"] = True
    return peft.LoraConfig(**kwargs)


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
    resolved = resolve_lora_config(
        model_path=model_path,
        lora_r=lora_r,
        checkpoint_path=checkpoint_path,
        task=task,
        target_modules=target_modules,
        require_checkpoint=False,
    )
    resolved_checkpoint = resolve_lora_checkpoint_path(checkpoint_path) if checkpoint_path is not None else None
    if resolved_checkpoint is not None and resolved_checkpoint.is_dir():
        lora_model = peft.PeftModel.from_pretrained(
            model,
            str(resolved_checkpoint),
            is_trainable=True,
        )
    else:
        lora_cfg = build_lora_config(
            model,
            model_path=model_path,
            lora_r=resolved.lora_r,
            task=task,
            target_modules=format_lora_target_modules(resolved.target_modules),
        )
        lora_model = peft.get_peft_model(model, lora_cfg)
        if resolved_checkpoint is not None:
            load_legacy_lora_state_dict(lora_model, resolved_checkpoint)

    return unwrap_peft_model(lora_model) if unwrap_base_model else lora_model


def lora_adapter_dir_for(save_path: str | Path) -> Path:
    path = Path(save_path)
    if path.suffix.lower() in LEGACY_LORA_STATE_SUFFIXES:
        return path.with_suffix("")
    return path


def lora_legacy_state_path_for(save_path: str | Path) -> Path:
    path = Path(save_path)
    if path.suffix.lower() in LEGACY_LORA_STATE_SUFFIXES:
        return path
    return path.with_suffix(".pt")


def save_lora_checkpoint(model, tokenizer, save_path: str | Path) -> dict[str, str]:
    adapter_dir = lora_adapter_dir_for(save_path)
    legacy_state_path = lora_legacy_state_path_for(save_path)
    adapter_dir.parent.mkdir(parents=True, exist_ok=True)
    legacy_state_path.parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(model, peft.PeftModel):
        raise ValueError("LoRA native save requires a PEFT model; got a non-PEFT model.")

    model.save_pretrained(str(adapter_dir))
    state_dict = peft.get_peft_model_state_dict(model)
    torch.save(state_dict, str(legacy_state_path))

    tokenizer_dir = adapter_dir.parent / f"{adapter_dir.name}_tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(tokenizer_dir))

    return {
        "primary_path": str(adapter_dir),
        "adapter_path": str(adapter_dir),
        "legacy_state_dict_path": str(legacy_state_path),
        "tokenizer_path": str(tokenizer_dir),
    }


def validate_lora_eval_args(args):
    if getattr(args, "train_method", "full") != "lora":
        return args

    apply_lora_config_to_args(args, require_checkpoint=True)

    defense = getattr(args, "defense", "none")
    if defense not in SUPPORTED_LORA_DEFENSES:
        raise NotImplementedError(
            "LoRA/PEFT eval currently supports only these defenses: "
            f"{format_lora_supported_defenses()}; got defense={defense!r}."
        )
    return args
