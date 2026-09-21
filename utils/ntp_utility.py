from __future__ import annotations

import math

import torch


def causal_lm_labels(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Build Hugging Face causal-LM labels without manually shifting targets."""

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return labels


def valid_shifted_target_count(labels: torch.Tensor) -> int:
    """Count targets scored by a causal LM after its internal one-token shift."""

    if labels.ndim != 2:
        raise ValueError(f"Expected rank-2 causal-LM labels, got shape={tuple(labels.shape)}")
    if labels.shape[1] < 2:
        return 0
    return int((labels[:, 1:] != -100).sum().item())


def safe_perplexity(nll: float) -> float:
    try:
        return math.exp(float(nll))
    except OverflowError:
        return math.inf


def evaluate_causal_lm(model, eval_loader, device) -> dict[str, float | int]:
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    for batch in eval_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = causal_lm_labels(input_ids, attention_mask)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        n_valid = valid_shifted_target_count(labels)
        if n_valid == 0:
            continue
        total_nll += float(outputs.loss.item()) * n_valid
        total_tokens += n_valid

    if total_tokens == 0:
        raise ValueError("Validation split contains no valid shifted causal-LM targets.")
    nll = total_nll / total_tokens
    return {
        "nll": nll,
        "perplexity": safe_perplexity(nll),
        "valid_tokens": total_tokens,
    }


def load_ntp_model_and_tokenizer(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        cache_dir=args.models_cache,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        cache_dir=args.models_cache,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.model_max_length = 512
    return model, tokenizer


def load_ntp_sst2_datasets(args, tokenizer):
    from datasets import load_dataset
    from transformers import DataCollatorWithPadding

    raw = load_dataset("glue", "sst2", cache_dir=args.models_cache)

    def tokenize_function(examples):
        return tokenizer(examples["sentence"], truncation=True)

    # Removing every source column guarantees that sentiment labels cannot enter
    # either the causal-LM training inputs or the validation computation.
    tokenized = raw.map(
        tokenize_function,
        batched=True,
        remove_columns=raw["train"].column_names,
    )
    train_dataset = tokenized["train"]
    eval_dataset = tokenized["validation"]
    max_train_samples = getattr(args, "max_train_samples", None)
    max_eval_samples = getattr(args, "max_eval_samples", None)
    if max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(max_train_samples, len(train_dataset))))
    if max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(max_eval_samples, len(eval_dataset))))
    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")
    return train_dataset, eval_dataset, DataCollatorWithPadding(tokenizer=tokenizer)
