from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import peft
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_slug(model_path: str) -> str:
    base = Path(model_path.rstrip("/\\")).name or model_path
    return base.replace("/", "_").replace("\\", "_")


def dataset_sequence_key(dataset_name: str) -> str:
    return "text" if dataset_name == "rotten_tomatoes" else "sentence"


def _local_model_dir(model_path: str) -> Path | None:
    model_dir = Path(model_path).expanduser()
    return model_dir if model_dir.is_dir() else None


def _has_local_tokenizer_files(model_dir: Path) -> bool:
    if (model_dir / "tokenizer.json").is_file():
        return True
    if (model_dir / "vocab.txt").is_file():
        return True
    if (model_dir / "tokenizer.model").is_file():
        return True
    if (model_dir / "spiece.model").is_file():
        return True
    if (model_dir / "sentencepiece.bpe.model").is_file():
        return True
    return (model_dir / "vocab.json").is_file() and (model_dir / "merges.txt").is_file()


def _config_tokenizer_hint(model_dir: Path) -> str | None:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return None

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    for key in ("_name_or_path", "name_or_path"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _infer_tokenizer_fallback(model_path: str, model) -> str | None:
    config = getattr(model, "config", None)
    hinted_source = getattr(config, "_name_or_path", None)
    if isinstance(hinted_source, str) and hinted_source.strip() and hinted_source != model_path:
        return hinted_source.strip()

    model_type = getattr(config, "model_type", None)
    if model_type == "gpt2":
        return "gpt2"
    if model_type == "bert":
        return "bert-base-uncased"

    model_name = Path(model_path.rstrip("/\\")).name.lower()
    if model_name.startswith("gpt2"):
        return "gpt2"
    if model_name.startswith("bert"):
        return "bert-base-uncased"
    return None


def _candidate_tokenizer_sources(args, model) -> list[str]:
    model_path = args.model_path
    candidates: list[str] = []

    explicit_tokenizer = getattr(args, "tokenizer_path", None)
    if explicit_tokenizer:
        candidates.append(explicit_tokenizer)

    model_dir = _local_model_dir(model_path)
    if model_dir is None or _has_local_tokenizer_files(model_dir):
        candidates.append(model_path)

    if model_dir is not None:
        hinted = _config_tokenizer_hint(model_dir)
        if hinted and hinted not in candidates and hinted != model_path:
            candidates.append(hinted)

    inferred = _infer_tokenizer_fallback(model_path, model)
    if inferred and inferred not in candidates and inferred != model_path:
        candidates.append(inferred)

    if model_path not in candidates:
        candidates.append(model_path)

    return candidates


def _load_seq_class_tokenizer(args, model):
    attempted_sources: list[str] = []
    last_error: OSError | None = None

    for source in _candidate_tokenizer_sources(args, model):
        attempted_sources.append(source)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                source,
                use_fast=True,
                cache_dir=args.models_cache,
            )
            if source != args.model_path:
                warnings.warn(
                    (
                        f"Tokenizer files missing under {args.model_path!r}; "
                        f"loaded tokenizer from {source!r} instead."
                    ),
                    stacklevel=2,
                )
            return tokenizer
        except OSError as exc:
            last_error = exc

    attempted = ", ".join(repr(source) for source in attempted_sources)
    raise OSError(
        (
            f"Can't load tokenizer for {args.model_path!r}. "
            f"Tried tokenizer sources: {attempted}."
        )
    ) from last_error


def load_seq_class_model_and_tokenizer(args):
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=2,
        cache_dir=args.models_cache,
    )
    tokenizer = _load_seq_class_tokenizer(args, model)

    if tokenizer.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))
    else:
        model.config.pad_token_id = tokenizer.pad_token_id

    if getattr(args, "train_method", "full") == "lora":
        lora_cfg = peft.LoraConfig(r=args.lora_r, target_modules=["q_proj"])
        model = peft.LoraModel(model, lora_cfg, "default")

    tokenizer.model_max_length = 512
    return model, tokenizer


def load_seq_class_datasets(args, tokenizer):
    seq_key = dataset_sequence_key(args.dataset)

    def tokenize_function(examples):
        return tokenizer(examples[seq_key], truncation=True)

    if args.dataset in ["cola", "sst2", "rte"]:
        raw = load_dataset("glue", args.dataset)
    else:
        raw = load_dataset(args.dataset)

    tokenized = raw.map(tokenize_function, batched=True)
    if args.dataset in ["cola", "sst2"]:
        tokenized = tokenized.remove_columns(["idx", "sentence"])
    elif args.dataset == "rotten_tomatoes":
        tokenized = tokenized.remove_columns(["text"])
    elif args.dataset == "rte":
        tokenized = tokenized.remove_columns(["idx", "sentence1", "sentence2"])
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")
    return tokenized["train"], tokenized["validation"], DataCollatorWithPadding(tokenizer=tokenizer)


def classification_metrics(predictions, references, dataset_name: str) -> dict[str, float]:
    preds = np.asarray(predictions).reshape(-1)
    refs = np.asarray(references).reshape(-1)
    metrics = {
        "accuracy": float(accuracy_score(refs, preds)),
        "macro_f1": float(f1_score(refs, preds, average="macro", zero_division=0)),
    }
    if dataset_name == "cola":
        metrics["matthews_correlation"] = float(matthews_corrcoef(refs, preds))
    return metrics
