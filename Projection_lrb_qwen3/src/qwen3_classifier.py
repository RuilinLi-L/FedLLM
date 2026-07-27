"""Strict local Qwen3 sequence-classification loading and single-example tokenization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import torch
from torch import nn


class Qwen3ClassifierError(RuntimeError):
    """Raised when the required local Qwen3 classification setup is unavailable."""


ComputeDTypeName = Literal["bfloat16", "float32"]


@dataclass(frozen=True)
class Qwen3ClassifierBundle:
    """The explicitly initialized classifier and its local tokenizer."""

    model: nn.Module
    tokenizer: Any
    device: torch.device
    model_path: Path
    head_seed: int
    head_parameter_names: tuple[str, ...]
    compute_dtype: torch.dtype


@dataclass(frozen=True)
class SingleExampleBatch:
    """One unpadded, EOS-terminated classification example."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    token_ids: tuple[int, ...]


def _require_cuda_device(device: str | torch.device) -> torch.device:
    target = torch.device(device)
    if target.type != "cuda":
        raise Qwen3ClassifierError(
            f"Qwen3 gradient diagnostics require an explicit CUDA device, got {target}."
        )
    if not torch.cuda.is_available():
        raise Qwen3ClassifierError("CUDA is required for BF16 Qwen3 gradient diagnostics but is unavailable.")
    return target


def _require_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise Qwen3ClassifierError(f"head_seed must be an explicit integer, got {seed!r}.")


def resolve_compute_dtype(dtype: ComputeDTypeName) -> torch.dtype:
    """Map the explicit diagnostic dtype option without a fallback."""
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise Qwen3ClassifierError(f"Unsupported diagnostic dtype: {dtype!r}.")


def _initialize_head(model: nn.Module, *, head_seed: int, std: float = 1e-3) -> tuple[str, ...]:
    """Initialize the Qwen sequence-classification head from one explicit seed."""
    _require_seed(head_seed)
    if std <= 0.0:
        raise Qwen3ClassifierError(f"Classifier-head initialization std must be positive, got {std}.")
    head = getattr(model, "score", None)
    if not isinstance(head, nn.Module):
        raise Qwen3ClassifierError(
            "Expected the Qwen3 sequence-classification head at model.score; refusing to guess another head."
        )
    all_names = {id(parameter): name for name, parameter in model.named_parameters()}
    head_parameters = list(head.parameters(recurse=True))
    if not head_parameters:
        raise Qwen3ClassifierError("model.score has no trainable parameters.")
    missing_names = [parameter for parameter in head_parameters if id(parameter) not in all_names]
    if missing_names:
        raise Qwen3ClassifierError("Unable to derive canonical parameter names for model.score.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(head_seed)
    with torch.no_grad():
        for parameter in head_parameters:
            values = torch.empty(parameter.shape, dtype=torch.float32, device="cpu")
            values.normal_(mean=0.0, std=std, generator=generator)
            parameter.copy_(values.to(device=parameter.device, dtype=parameter.dtype))
    return tuple(all_names[id(parameter)] for parameter in head_parameters)


def load_local_qwen3_sequence_classifier(
    model_path: Path,
    *,
    head_seed: int,
    device: str | torch.device = "cuda",
    dtype: ComputeDTypeName = "bfloat16",
) -> Qwen3ClassifierBundle:
    """Load Qwen3 locally as a BF16, two-label sequence classifier.

    The loader never substitutes a causal-LM head or contacts a model hub.
    """
    target_device = _require_cuda_device(device)
    _require_seed(head_seed)
    compute_dtype = resolve_compute_dtype(dtype)
    resolved_model_path = model_path.resolve()
    if not resolved_model_path.is_dir():
        raise Qwen3ClassifierError(f"Configured local Qwen3 model directory does not exist: {resolved_model_path}")
    try:
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise Qwen3ClassifierError(
            "transformers is required to load the local Qwen3 sequence-classification model."
        ) from error
    try:
        model_config = AutoConfig.from_pretrained(str(resolved_model_path), local_files_only=True)
    except Exception as error:  # Transformers exposes several configuration exceptions.
        raise Qwen3ClassifierError(
            f"Unable to load local Qwen3 configuration from {resolved_model_path}: {error}"
        ) from error
    if getattr(model_config, "model_type", None) != "qwen3":
        raise Qwen3ClassifierError(
            f"Expected model_type='qwen3', got {getattr(model_config, 'model_type', None)!r}."
        )
    model_config.num_labels = 2
    model_config.use_cache = False
    eos_token_id = getattr(model_config, "eos_token_id", None)
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
        raise Qwen3ClassifierError("Qwen3 configuration must expose one integer eos_token_id.")
    model_config.pad_token_id = eos_token_id
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(resolved_model_path),
            local_files_only=True,
            trust_remote_code=False,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            str(resolved_model_path),
            config=model_config,
            local_files_only=True,
            torch_dtype=compute_dtype,
        )
    except Exception as error:  # Transformers loading errors must be surfaced with path context.
        raise Qwen3ClassifierError(
            f"Unable to load local Qwen3 sequence classifier from {resolved_model_path}: {error}"
        ) from error
    if getattr(model.config, "num_labels", None) != 2:
        raise Qwen3ClassifierError("Loaded classifier does not expose num_labels=2.")
    model.config.use_cache = False
    model.config.pad_token_id = eos_token_id
    head_parameter_names = _initialize_head(model, head_seed=head_seed)
    model.to(device=target_device, dtype=compute_dtype)
    model.train()
    wrong_dtype = [
        name
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point() and parameter.dtype != compute_dtype
    ]
    if wrong_dtype:
        raise Qwen3ClassifierError(
            f"{dtype} forward/backward was requested but floating model parameters have another dtype: "
            f"{wrong_dtype[:8]}"
        )
    return Qwen3ClassifierBundle(
        model=model,
        tokenizer=tokenizer,
        device=target_device,
        model_path=resolved_model_path,
        head_seed=head_seed,
        head_parameter_names=head_parameter_names,
        compute_dtype=compute_dtype,
    )


def tokenize_single_example(
    tokenizer: Any,
    *,
    sentence: str,
    label: int,
    max_length: int,
    device: torch.device,
) -> SingleExampleBatch:
    """Create a batch-size-one, unpadded, explicit-EOS classification batch."""
    if not isinstance(sentence, str) or not sentence.strip():
        raise Qwen3ClassifierError("sentence must be one non-empty string.")
    if isinstance(label, bool) or label not in (0, 1):
        raise Qwen3ClassifierError(f"label must be 0 or 1, got {label!r}.")
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 2:
        raise Qwen3ClassifierError("max_length must be an integer of at least 2 to reserve EOS.")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
        raise Qwen3ClassifierError("Tokenizer must expose one integer eos_token_id.")
    try:
        encoded: Mapping[str, Any] = tokenizer(
            sentence,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
    except Exception as error:
        raise Qwen3ClassifierError(f"Qwen3 tokenizer failed for the supplied sentence: {error}") from error
    raw_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
    if not isinstance(raw_ids, list) or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_ids):
        raise Qwen3ClassifierError("Tokenizer must return a one-dimensional list of integer input_ids.")
    token_ids = tuple([*raw_ids[: max_length - 1], eos_token_id])
    input_ids = torch.tensor([token_ids], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    labels = torch.tensor([label], device=device, dtype=torch.long)
    if input_ids.shape[0] != 1 or not bool(torch.all(attention_mask == 1)):
        raise Qwen3ClassifierError("Gradient diagnostic requires exactly one unpadded sample and all-one attention_mask.")
    return SingleExampleBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        token_ids=token_ids,
    )
