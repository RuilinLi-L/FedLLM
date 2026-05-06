from __future__ import annotations

from pathlib import Path

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


def is_gpt2_lora_model(model_path: str) -> bool:
    return model_path in GPT2_LORA_MODELS


def is_llama_lora_model(model_path: str) -> bool:
    return model_path in LLAMA_LORA_MODELS


def is_supported_lora_model(model_path: str) -> bool:
    return model_path in SUPPORTED_LORA_MODELS


def lora_target_modules(model_path: str) -> list[str]:
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


def lora_task_type(task: str | None):
    if task == "seq_class":
        return peft.TaskType.SEQ_CLS
    if task == "next_token_pred":
        return peft.TaskType.CAUSAL_LM
    return None


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


def build_lora_config(model, *, model_path: str, lora_r: int, task: str | None = None):
    kwargs = {
        "r": int(lora_r),
        "target_modules": lora_target_modules(model_path),
    }
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
):
    if lora_r is None or int(lora_r) <= 0:
        raise ValueError(f"LoRA rank must be a positive integer; got {lora_r!r}.")

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
            lora_r=lora_r,
            task=task,
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

    if not is_supported_lora_model(getattr(args, "model_path", "")):
        raise ValueError(
            "LoRA/PEFT eval currently supports GPT-2 and Llama model families only. "
            f"Supported model_path values: {format_lora_supported_models()}."
        )

    if getattr(args, "finetuned_path", None) is None:
        raise ValueError(
            "--train_method lora requires --finetuned_path PATH to an existing "
            "PEFT adapter directory or LoRA .pt/.pth checkpoint."
        )
    resolve_lora_checkpoint_path(args.finetuned_path)

    if getattr(args, "lora_r", None) is None:
        raise ValueError("--train_method lora requires --lora_r R.")
    if int(args.lora_r) <= 0:
        raise ValueError(f"--lora_r must be a positive integer; got {args.lora_r!r}.")

    defense = getattr(args, "defense", "none")
    if defense not in SUPPORTED_LORA_DEFENSES:
        raise NotImplementedError(
            "LoRA/PEFT eval currently supports only these defenses: "
            f"{format_lora_supported_defenses()}; got defense={defense!r}."
        )
    return args
