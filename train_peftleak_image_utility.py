#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import datetime
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from utils.defense_common import add_shared_defense_args, fmt_summary_value
from utils.peftleak_image_utility import (
    DEBUG_PROFILE,
    FORMAL_PROFILE,
    PROFILE_CHOICES,
    adapter_checkpoint_state,
    apply_shared_adapter_defense,
    build_image_datasets,
    build_image_loaders,
    build_utility_model,
    load_adapter_checkpoint_state,
    validate_parameter_scopes,
)


SUMMARY_START = "===== TRAIN RESULT SUMMARY START ====="
SUMMARY_END = "===== TRAIN RESULT SUMMARY END ====="


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train reportable CIFAR-100 ViT Adapters with shared-gradient defenses."
    )
    parser.add_argument("--dataset", choices=["cifar100"], default="cifar100")
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default=FORMAL_PROFILE)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--output_dir", default="outputs/peftleak_official_image/utility/single")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--adapter_bottleneck_dim", type=int, default=64)
    parser.add_argument(
        "--utility_control",
        choices=["standard", "head_only"],
        default="standard",
        help="head_only freezes Adapter updates to measure utility from the local classification head alone.",
    )
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--validation_size", type=int, default=5000)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr_adapter", type=float, default=1e-3)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug_train_size", type=int, default=32)
    parser.add_argument("--debug_validation_size", type=int, default=16)
    parser.add_argument("--debug_test_size", type=int, default=16)
    add_shared_defense_args(
        parser,
        default_grad_mode="train",
        extra_defense_choices=("proj_only", "full_lrb"),
    )
    return parser


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(raw_device: str) -> torch.device:
    device = torch.device(raw_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {raw_device}, but CUDA is unavailable.")
    return device


def validate_args(args) -> None:
    if args.profile == FORMAL_PROFILE and args.no_pretrained:
        raise ValueError("Formal image utility requires ImageNet-pretrained ViT-B/16 weights.")
    if args.num_epochs <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("Epoch and batch-size arguments must be positive.")
    if args.adapter_bottleneck_dim <= 0:
        raise ValueError("adapter_bottleneck_dim must be positive.")
    if args.warmup_epochs < 0 or args.warmup_epochs > args.num_epochs:
        raise ValueError("warmup_epochs must be in [0, num_epochs].")
    if args.defense not in {"none", "topk", "compression", "lrb", "proj_only", "full_lrb"}:
        raise NotImplementedError(
            "Image task utility supports none/topk/compression/lrb/proj_only/full_lrb only."
        )
    if args.defense == "lrb" and args.defense_lrb_preset == "custom":
        raise ValueError("Formal LRB utility runs must select an explicit defense_lrb_preset.")
    if args.utility_control == "head_only" and args.defense != "none":
        raise ValueError("head_only is a utility control and must use --defense none.")


def defense_summary(args) -> tuple[str, object]:
    if args.defense == "none":
        return "n/a", "n/a"
    if args.defense == "topk":
        return "defense_topk_ratio", float(args.defense_topk_ratio)
    if args.defense == "compression":
        return "defense_n_bits", int(args.defense_n_bits)
    return "defense_lrb_keep_ratio_sensitive", float(args.defense_lrb_keep_ratio_sensitive)


def make_scheduler(optimizer, *, total_steps: int, warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(model, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_examples = 0
    predictions: list[int] = []
    references: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * int(labels.shape[0])
            total_examples += int(labels.shape[0])
            predictions.extend(logits.argmax(dim=-1).detach().cpu().tolist())
            references.extend(labels.detach().cpu().tolist())
    if not references:
        raise RuntimeError("Evaluation loader produced no samples.")
    return {
        "accuracy": float(accuracy_score(references, predictions)),
        "macro_f1": float(f1_score(references, predictions, average="macro", zero_division=0)),
        "loss": total_loss / max(1, total_examples),
    }


def _save_checkpoint(path: Path, model, args, *, epoch: int, validation_metrics: dict[str, float]) -> None:
    state = adapter_checkpoint_state(model)
    state["metadata"] = {
        "profile": args.profile,
        "dataset": args.dataset,
        "pretrained_weights": "IMAGENET1K_V1" if args.profile == FORMAL_PROFILE else "none_debug",
        "adapter_bottleneck_dim": int(args.adapter_bottleneck_dim),
        "shared_scope": "adapter_only",
        "local_scope": "classification_head",
        "epoch": int(epoch),
        "validation_metrics": validation_metrics,
    }
    torch.save(state, path)


def _write_history(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "validation_accuracy",
        "validation_macro_f1",
        "validation_loss",
        "epoch_time_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def emit_summary(fields: dict) -> None:
    print(SUMMARY_START, flush=True)
    for key, value in fields.items():
        print(f"{key}={fmt_summary_value(value)}", flush=True)
    print(SUMMARY_END, flush=True)


def run(args) -> dict:
    validate_args(args)
    set_reproducible_seed(int(args.rng_seed))
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_image_datasets(
        profile=args.profile,
        data_root=args.data_root,
        split_seed=int(args.split_seed),
        validation_size=int(args.validation_size),
        download=bool(args.download),
        debug_train_size=int(args.debug_train_size),
        debug_validation_size=int(args.debug_validation_size),
        debug_test_size=int(args.debug_test_size),
    )
    train_loader, validation_loader, test_loader = build_image_loaders(
        bundle,
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.rng_seed),
        pin_memory=device.type == "cuda",
    )
    model = build_utility_model(
        profile=args.profile,
        num_classes=100,
        bottleneck_dim=int(args.adapter_bottleneck_dim),
        pretrained=not bool(args.no_pretrained),
    ).to(device)
    validate_parameter_scopes(model, formal=args.profile == FORMAL_PROFILE)

    adapter_parameters = model.shared_parameters()
    head_parameters = model.local_head_parameters()
    effective_adapter_lr = 0.0 if args.utility_control == "head_only" else float(args.lr_adapter)
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_parameters, "lr": effective_adapter_lr},
            {"params": head_parameters, "lr": float(args.lr_head)},
        ],
        weight_decay=float(args.weight_decay),
    )
    total_steps = max(1, int(args.num_epochs) * len(train_loader))
    warmup_steps = int(args.warmup_epochs) * len(train_loader)
    scheduler = make_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)
    criterion = nn.CrossEntropyLoss()
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_accuracy = -1.0
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    history: list[dict] = []
    global_step = 0
    train_start = time.perf_counter()
    step_time_total = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, int(args.num_epochs) + 1):
        model.train()
        epoch_start = time.perf_counter()
        loss_sum = 0.0
        example_count = 0
        for images, labels in train_loader:
            step_start = time.perf_counter()
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at step {global_step}.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            args.defense_rng_step = global_step
            apply_shared_adapter_defense(model, args)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_n = int(labels.shape[0])
            loss_sum += float(loss.item()) * batch_n
            example_count += batch_n
            global_step += 1
            step_time_total += time.perf_counter() - step_start

        validation = evaluate(model, validation_loader, device)
        train_loss = loss_sum / max(1, example_count)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_accuracy": validation["accuracy"],
                "validation_macro_f1": validation["macro_f1"],
                "validation_loss": validation["loss"],
                "epoch_time_seconds": time.perf_counter() - epoch_start,
            }
        )
        print(
            f"[peftleak-image-utility] epoch={epoch} train_loss={train_loss:.6f} "
            f"val_accuracy={validation['accuracy']:.6f} val_loss={validation['loss']:.6f}",
            flush=True,
        )
        improved = validation["accuracy"] > best_accuracy or (
            validation["accuracy"] == best_accuracy and validation["loss"] < best_loss
        )
        if improved:
            best_accuracy = validation["accuracy"]
            best_loss = validation["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(adapter_checkpoint_state(model))

    if best_state is None:
        raise RuntimeError("Training completed without a best checkpoint.")
    load_adapter_checkpoint_state(model, best_state)
    test_metrics = evaluate(model, test_loader, device)
    validation_metrics = evaluate(model, validation_loader, device)
    checkpoint_path = output_dir / "best_adapter_head.pt"
    _save_checkpoint(checkpoint_path, model, args, epoch=best_epoch, validation_metrics=validation_metrics)
    history_path = output_dir / "history.csv"
    _write_history(history_path, history)

    total_time = time.perf_counter() - train_start
    param_name, param_value = defense_summary(args)
    shared_parameter_count = sum(parameter.numel() for parameter in adapter_parameters)
    fields = {
        "result_status": "ok",
        "reportable": not bundle.synthetic,
        "utility_scope": "cifar100_downstream_adapter_training",
        "dataset": args.dataset,
        "profile": args.profile,
        "model_path": "torchvision_vit_b_16" if args.profile == FORMAL_PROFILE else "torchvision_vit_debug",
        "pretrained_weights": "IMAGENET1K_V1" if args.profile == FORMAL_PROFILE else "none_debug",
        "train_method": "image_adapter",
        "peft_method": "adapter",
        "shared_scope": "adapter_only",
        "local_scope": "classification_head",
        "utility_control": args.utility_control,
        "adapter_bottleneck_dim": int(args.adapter_bottleneck_dim),
        "shared_gradient_tensor_count": len(adapter_parameters),
        "shared_parameter_count": int(shared_parameter_count),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "validation_size": int(args.validation_size),
        "num_epochs": int(args.num_epochs),
        "seed": int(args.rng_seed),
        "split_seed": int(args.split_seed),
        "lr_adapter": effective_adapter_lr,
        "lr_head": float(args.lr_head),
        "weight_decay": float(args.weight_decay),
        "warmup_epochs": int(args.warmup_epochs),
        "amp": amp_enabled,
        "defense": args.defense,
        "defense_param_name": param_name,
        "defense_param_value": param_value,
        "defense_lrb_preset": (
            args.defense if args.defense in {"proj_only", "full_lrb"} else args.defense_lrb_preset
        ),
        "defense_rng_seed": int(args.rng_seed),
        "projection_seed_policy": "fixed_per_layer_across_steps",
        "best_epoch": int(best_epoch),
        "eval_accuracy": test_metrics["accuracy"],
        "eval_macro_f1": test_metrics["macro_f1"],
        "eval_loss": test_metrics["loss"],
        "validation_accuracy": validation_metrics["accuracy"],
        "validation_macro_f1": validation_metrics["macro_f1"],
        "validation_loss": validation_metrics["loss"],
        "final_train_loss": history[-1]["train_loss"],
        "steps_completed": int(global_step),
        "mean_step_time_seconds": step_time_total / max(1, global_step),
        "total_train_time": str(datetime.timedelta(seconds=int(total_time))),
        "total_train_time_seconds": total_time,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "final_model_path": str(checkpoint_path),
        "history_path": str(history_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(fields, indent=2, sort_keys=True), encoding="utf-8")
    fields["summary_path"] = str(summary_path)
    return fields


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        fields = run(args)
    except Exception as exc:  # noqa: BLE001 - CLI must emit machine-readable failure state
        param_name, param_value = defense_summary(args)
        fields = {
            "result_status": "failed",
            "reportable": False,
            "utility_scope": "cifar100_downstream_adapter_training",
            "dataset": args.dataset,
            "profile": args.profile,
            "utility_control": args.utility_control,
            "seed": int(args.rng_seed),
            "defense": args.defense,
            "defense_param_name": param_name,
            "defense_param_value": param_value,
            "error_type": type(exc).__name__,
            "error_message": str(exc).replace("\n", " "),
        }
        emit_summary(fields)
        return 1
    emit_summary(fields)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
