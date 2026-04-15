from __future__ import annotations

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


def load_seq_class_model_and_tokenizer(args):
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=2,
        cache_dir=args.models_cache,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        cache_dir=args.models_cache,
    )

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
