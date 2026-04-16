#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.defense_common import (
    add_shared_defense_args,
    defense_param_spec,
    fmt_summary_value,
    grad_similarity_metrics,
    iter_trainable_parameters,
    normalize_legacy_training_defense_args,
)
from utils.defenses import apply_defense, requires_gradient_generation_defense
from utils.seq_class_utils import (
    load_seq_class_datasets,
    load_seq_class_model_and_tokenizer,
    set_random_seed,
)
from utils.training_defense_wrapper import TrainingDefenseModelWrapper

try:
    from torch.func import functional_call
except ImportError:  # pragma: no cover - older torch fallback
    from torch.nn.utils.stateless import functional_call


PROXY_SUMMARY_START = "===== PROXY UTILITY SUMMARY START ====="
PROXY_SUMMARY_END = "===== PROXY UTILITY SUMMARY END ====="


def install_terminal_logging(args) -> None:
    log_path = args.log_file
    if log_path is None and not os.environ.get("DAGER_NO_AUTO_LOG"):
        log_dir = os.environ.get("DAGER_LOG_DIR", os.path.join("log", "runs"))
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"proxy_utility_{args.dataset}_{stamp}.txt")
        args.log_file = log_path

    if log_path:
        from utils.terminal_log import install_terminal_log

        install_terminal_log(
            log_path,
            append=args.log_append,
            argv_for_banner=sys.argv,
        )
        print(f"[dager] Terminal log: {log_path}", flush=True)


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def init_tracker(args) -> dict:
    return {
        "summary_emitted": False,
        "summary_version": 1,
        "result_status": "ok",
        "steps_completed": 0,
        "base_val_metrics": {},
        "grad_cosine_values": [],
        "norm_retention_values": [],
        "delta_train_loss_values": [],
        "delta_val_loss_values": [],
        "delta_val_accuracy_values": [],
        "delta_val_macro_f1_values": [],
        "step_runtime_values": [],
        "error_type": None,
        "error_message": None,
    }


def emit_proxy_summary(args, tracker: dict) -> None:
    if tracker.get("summary_emitted"):
        return

    defense_param_name, defense_param_value = defense_param_spec(args)
    base_val = tracker.get("base_val_metrics", {})
    fields = [
        ("summary_version", tracker.get("summary_version", 1)),
        ("result_status", tracker.get("result_status", "ok")),
        ("dataset", args.dataset),
        ("task", args.task),
        ("model_path", args.model_path),
        ("batch_size", args.batch_size),
        ("defense", args.defense),
        ("defense_param_name", defense_param_name),
        ("defense_param_value", defense_param_value),
        ("n_train_batches", args.n_train_batches),
        ("val_size", args.val_size),
        ("steps_completed", tracker.get("steps_completed", 0)),
        ("base_val_accuracy", base_val.get("accuracy")),
        ("base_val_macro_f1", base_val.get("macro_f1")),
        ("base_val_loss", base_val.get("loss")),
        ("grad_cosine_mean", safe_mean(tracker.get("grad_cosine_values", []))),
        ("norm_retention_mean", safe_mean(tracker.get("norm_retention_values", []))),
        ("delta_train_loss_mean", safe_mean(tracker.get("delta_train_loss_values", []))),
        ("delta_val_loss_mean", safe_mean(tracker.get("delta_val_loss_values", []))),
        ("delta_val_accuracy_mean", safe_mean(tracker.get("delta_val_accuracy_values", []))),
        ("delta_val_macro_f1_mean", safe_mean(tracker.get("delta_val_macro_f1_values", []))),
        ("step_runtime_mean", safe_mean(tracker.get("step_runtime_values", []))),
    ]

    if tracker.get("error_type"):
        fields.append(("error_type", tracker["error_type"]))
    if tracker.get("error_message"):
        fields.append(("error_message", tracker["error_message"]))

    print(PROXY_SUMMARY_START, flush=True)
    for key, value in fields:
        print(f"{key}={fmt_summary_value(value)}", flush=True)
    print(PROXY_SUMMARY_END, flush=True)
    tracker["summary_emitted"] = True


def evaluate_metrics(model, loader, device, dataset_name: str, param_override=None) -> dict[str, float]:
    model.eval()
    losses = []
    predictions = []
    references = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]
        model_inputs = {k: v for k, v in batch.items() if k != "labels"}
        with torch.no_grad():
            if param_override is None:
                outputs = model(**model_inputs, labels=labels)
            else:
                outputs = functional_call(
                    model,
                    param_override,
                    (),
                    {**model_inputs, "labels": labels},
                )
        losses.append(float(outputs.loss.item()))
        predictions.extend(torch.argmax(outputs.logits, dim=-1).detach().cpu().tolist())
        references.extend(labels.detach().cpu().tolist())

    from utils.seq_class_utils import classification_metrics

    metrics = classification_metrics(predictions, references, dataset_name)
    metrics["loss"] = float(sum(losses) / max(1, len(losses)))
    return metrics


def build_updated_param_state(named_trainable_params, defended_grads, lr: float):
    updated = {}
    for (name, param), grad in zip(named_trainable_params, defended_grads):
        if grad is None:
            continue
        updated[name] = param.detach() - lr * grad.detach().to(device=param.device, dtype=param.dtype)
    return updated


def build_parser():
    parser = argparse.ArgumentParser(description="Proxy utility evaluation via one-step virtual updates.")
    parser.add_argument("--dataset", choices=["cola", "sst2", "rte", "rotten_tomatoes"], required=True)
    parser.add_argument("--task", choices=["seq_class"], default="seq_class")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Optional tokenizer source when model_path points to a checkpoint directory without tokenizer files.",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--models_cache", type=str, default="./models_cache")
    parser.add_argument("--train_method", type=str, default="full", choices=["full"])
    parser.add_argument("--n_train_batches", type=int, default=100)
    parser.add_argument("--val_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--noise", type=float, default=None)
    parser.add_argument("--pct_mask", type=float, default=None)
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Mirror stdout/stderr to this UTF-8 file (Python streams only).",
    )
    parser.add_argument("--log_append", action="store_true", help="Append to log_file instead of truncating.")
    add_shared_defense_args(parser, default_grad_mode="train")
    return parser


def main():
    global args

    parser = build_parser()
    args = parser.parse_args()
    normalize_legacy_training_defense_args(args)
    install_terminal_logging(args)
    tracker = init_tracker(args)

    try:
        set_random_seed(args.rng_seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer = load_seq_class_model_and_tokenizer(args)
        model = model.to(device)
        train_dataset, eval_dataset, data_collator = load_seq_class_datasets(args, tokenizer)

        val_size = min(args.val_size, len(eval_dataset))
        eval_dataset = eval_dataset.select(range(val_size))
        eval_loader = DataLoader(
            eval_dataset,
            shuffle=False,
            batch_size=args.eval_batch_size,
            collate_fn=data_collator,
        )

        generator = torch.Generator()
        generator.manual_seed(args.rng_seed)
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            generator=generator,
            batch_size=args.batch_size,
            collate_fn=data_collator,
        )

        trainable_params = iter_trainable_parameters(model)
        named_trainable_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
        wrapper = TrainingDefenseModelWrapper(model, args, trainable_params)
        tracker["base_val_metrics"] = evaluate_metrics(model, eval_loader, device, args.dataset)

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= args.n_train_batches:
                break

            batch_start = time.perf_counter()
            model.train()
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            model_inputs = {k: v for k, v in batch.items() if k != "labels"}

            model.zero_grad(set_to_none=True)
            outputs = model(**model_inputs, labels=labels)
            base_train_loss = float(outputs.loss.item())

            if requires_gradient_generation_defense(args.defense):
                outputs.loss.backward()
                raw_grads = tuple(
                    None if param.grad is None else param.grad.detach().clone()
                    for param in trainable_params
                )
                model.zero_grad(set_to_none=True)
                defended_grads = apply_defense(
                    None,
                    args,
                    model_wrapper=wrapper,
                    batch=batch,
                    labels=labels,
                )
            else:
                outputs.loss.backward()
                raw_grads = tuple(
                    None if param.grad is None else param.grad.detach().clone()
                    for param in trainable_params
                )
                defended_grads = apply_defense(
                    raw_grads,
                    args,
                    model_wrapper=wrapper,
                    batch=batch,
                    labels=labels,
                )

            grad_cosine, norm_retention = grad_similarity_metrics(raw_grads, defended_grads)
            updated_state = build_updated_param_state(named_trainable_params, defended_grads, args.learning_rate)
            updated_train = functional_call(
                model,
                updated_state,
                (),
                {**model_inputs, "labels": labels},
            )
            updated_val = evaluate_metrics(
                model,
                eval_loader,
                device,
                args.dataset,
                param_override=updated_state,
            )

            tracker["steps_completed"] = batch_idx + 1
            tracker["grad_cosine_values"].append(grad_cosine)
            tracker["norm_retention_values"].append(norm_retention)
            tracker["delta_train_loss_values"].append(float(updated_train.loss.item()) - base_train_loss)
            tracker["delta_val_loss_values"].append(updated_val["loss"] - tracker["base_val_metrics"]["loss"])
            tracker["delta_val_accuracy_values"].append(updated_val["accuracy"] - tracker["base_val_metrics"]["accuracy"])
            tracker["delta_val_macro_f1_values"].append(updated_val["macro_f1"] - tracker["base_val_metrics"]["macro_f1"])
            tracker["step_runtime_values"].append(time.perf_counter() - batch_start)

            model.zero_grad(set_to_none=True)

        emit_proxy_summary(args, tracker)
    except Exception as exc:
        tracker["result_status"] = "failed"
        tracker["error_type"] = type(exc).__name__
        tracker["error_message"] = str(exc)
        emit_proxy_summary(args, tracker)
        raise


if __name__ == "__main__":
    main()
