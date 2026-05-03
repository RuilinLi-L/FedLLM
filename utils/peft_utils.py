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


def resolve_lora_checkpoint_path(path: str) -> Path:
    checkpoint_path = Path(path).expanduser()
    if checkpoint_path.is_dir():
        raise ValueError(
            "LoRA/PEFT eval only supports local .pt/.pth state_dict checkpoints; "
            f"adapter directories are not supported yet: {path!r}."
        )
    if checkpoint_path.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError(
            "LoRA/PEFT eval only supports local .pt/.pth state_dict checkpoints; "
            f"got {path!r}."
        )
    if not checkpoint_path.is_file():
        raise ValueError(f"LoRA checkpoint file does not exist: {path!r}.")
    return checkpoint_path


def load_lora_checkpoint(path: str):
    checkpoint_path = resolve_lora_checkpoint_path(path)
    return torch.load(checkpoint_path, map_location=torch.device("cpu"))


def apply_lora_adapter(
    model,
    *,
    model_path: str,
    lora_r: int,
    checkpoint_path: str | None = None,
    unwrap_base_model: bool = False,
):
    if lora_r is None or int(lora_r) <= 0:
        raise ValueError(f"LoRA rank must be a positive integer; got {lora_r!r}.")

    lora_cfg = peft.LoraConfig(
        r=int(lora_r),
        target_modules=lora_target_modules(model_path),
    )
    lora_model = peft.LoraModel(model, lora_cfg, "default")
    if checkpoint_path is not None:
        state_dict = load_lora_checkpoint(checkpoint_path)
        lora_model.load_state_dict(state_dict)
    return lora_model.model if unwrap_base_model else lora_model


def validate_lora_eval_args(args):
    if getattr(args, "train_method", "full") != "lora":
        return args

    if not is_supported_lora_model(getattr(args, "model_path", "")):
        raise ValueError(
            "LoRA/PEFT eval currently supports GPT-2 and Llama model families only. "
            f"Supported model_path values: {format_lora_supported_models()}."
        )

    if getattr(args, "finetuned_path", None) is None:
        raise ValueError("--train_method lora requires --finetuned_path PATH to an existing LoRA .pt/.pth checkpoint.")
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
