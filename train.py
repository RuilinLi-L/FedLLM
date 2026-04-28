from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AdamW, get_scheduler

from utils.defense_common import (
    add_shared_defense_args,
    capture_gradients,
    defense_param_spec,
    fmt_summary_value,
    iter_trainable_parameters,
    normalize_legacy_training_defense_args,
    overwrite_gradients,
)
from utils.defenses import apply_defense, requires_gradient_generation_defense
from utils.gpu import resolve_cuda_device
from utils.seq_class_utils import (
    classification_metrics,
    load_seq_class_datasets,
    load_seq_class_model_and_tokenizer,
    model_slug,
    set_random_seed,
)
from utils.training_defense_wrapper import TrainingDefenseModelWrapper


TRAIN_SUMMARY_START = "===== TRAIN RESULT SUMMARY START ====="
TRAIN_SUMMARY_END = "===== TRAIN RESULT SUMMARY END ====="


def resolve_default_output_dir(args) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    return Path("finetune") / f"{model_slug(args.model_path)}_{args.dataset}_{args.train_method}"


def install_terminal_logging(args) -> None:
    log_path = args.log_file
    if log_path is None and not os.environ.get("DAGER_NO_AUTO_LOG"):
        log_dir = os.environ.get("DAGER_LOG_DIR", os.path.join("log", "runs"))
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"train_{args.dataset}_{stamp}.txt")
        args.log_file = log_path

    if log_path:
        from utils.terminal_log import install_terminal_log

        install_terminal_log(
            log_path,
            append=args.log_append,
            argv_for_banner=sys.argv,
        )
        print(f"[dager] Terminal log: {log_path}", flush=True)


def save_model(model, tokenizer, save_path: Path, train_method: str) -> str:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if train_method != "lora":
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))
        return str(save_path)

    torch.save(model.state_dict(), str(save_path))
    tokenizer_dir = save_path.parent / f"{save_path.stem}_tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(tokenizer_dir))
    return str(save_path)


def init_result_tracker(args) -> dict:
    return {
        "summary_emitted": False,
        "summary_version": 1,
        "result_status": "ok",
        "dataset": args.dataset,
        "task": args.task,
        "model_path": args.model_path,
        "batch_size": args.batch_size,
        "train_method": args.train_method,
        "num_epochs": args.num_epochs,
        "seed": args.rng_seed,
        "output_dir": str(resolve_default_output_dir(args)),
        "final_train_loss": None,
        "eval_metrics": {},
        "steps_completed": 0,
        "total_train_time": None,
        "final_model_path": None,
        "error_type": None,
        "error_message": None,
    }


def emit_train_result_summary(args, tracker: dict) -> None:
    if tracker.get("summary_emitted"):
        return

    defense_param_name, defense_param_value = defense_param_spec(args)
    fields = [
        ("summary_version", tracker.get("summary_version", 1)),
        ("result_status", tracker.get("result_status", "ok")),
        ("dataset", tracker.get("dataset")),
        ("task", tracker.get("task")),
        ("model_path", tracker.get("model_path")),
        ("batch_size", tracker.get("batch_size")),
        ("train_method", tracker.get("train_method")),
        ("num_epochs", tracker.get("num_epochs")),
        ("seed", tracker.get("seed")),
        ("defense", getattr(args, "defense", "none")),
        ("defense_param_name", defense_param_name),
        ("defense_param_value", defense_param_value),
        ("output_dir", tracker.get("output_dir")),
        ("steps_completed", tracker.get("steps_completed")),
        ("final_train_loss", tracker.get("final_train_loss")),
        ("total_train_time", tracker.get("total_train_time")),
        ("final_model_path", tracker.get("final_model_path")),
    ]

    eval_metrics = tracker.get("eval_metrics", {})
    for key in sorted(eval_metrics):
        fields.append((f"eval_{key}", eval_metrics[key]))

    if tracker.get("error_type"):
        fields.append(("error_type", tracker["error_type"]))
    if tracker.get("error_message"):
        fields.append(("error_message", tracker["error_message"]))

    print(TRAIN_SUMMARY_START, flush=True)
    for key, value in fields:
        print(f"{key}={fmt_summary_value(value)}", flush=True)
    print(TRAIN_SUMMARY_END, flush=True)
    tracker["summary_emitted"] = True


def evaluate_model(model, eval_loader, device, dataset_name: str) -> dict[str, float]:
    model.eval()
    losses = []
    predictions = []
    references = []

    for batch in eval_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]
        model_inputs = {k: v for k, v in batch.items() if k != "labels"}
        with torch.no_grad():
            outputs = model(**model_inputs, labels=labels)
        losses.append(float(outputs.loss.item()))
        predictions.extend(torch.argmax(outputs.logits, dim=-1).detach().cpu().tolist())
        references.extend(labels.detach().cpu().tolist())

    metrics = classification_metrics(predictions, references, dataset_name)
    metrics["loss"] = float(sum(losses) / max(1, len(losses)))
    return metrics


def prepare_training_defense(model, args, trainable_params):
    if args.train_method == "full":
        return TrainingDefenseModelWrapper(model, args, trainable_params)
    if args.defense in {"dpsgd", "mixup", "soteria", "lrb", "dager"}:
        raise NotImplementedError(
            f"train_method={args.train_method} does not currently support --defense {args.defense!r}."
        )
    return None


def apply_training_defense(model, wrapper, trainable_params, batch, labels, loss, args):
    if requires_gradient_generation_defense(args.defense):
        defended_grads = apply_defense(
            None,
            args,
            model_wrapper=wrapper,
            batch=batch,
            labels=labels,
        )
    else:
        loss.backward()
        raw_grads = capture_gradients(trainable_params)
        defended_grads = apply_defense(
            raw_grads,
            args,
            model_wrapper=wrapper,
            batch=batch,
            labels=labels,
        )
    overwrite_gradients(trainable_params, defended_grads)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cola", "sst2", "rte", "rotten_tomatoes"], default="cola")
    parser.add_argument("--task", choices=["seq_class"], default="seq_class")
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--noise", type=float, default=None)
    parser.add_argument("--pct_mask", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--model_path", type=str, default="bert-base-uncased")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Optional tokenizer source when model_path points to a checkpoint directory without tokenizer files.",
    )
    parser.add_argument("--train_method", type=str, default="full", choices=["full", "lora"])
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--models_cache", type=str, default="./models_cache")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory used for checkpoints and the final saved model.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Mirror stdout/stderr to this UTF-8 file (Python streams only).",
    )
    parser.add_argument("--log_append", action="store_true", help="Append to log_file instead of truncating.")
    add_shared_defense_args(parser, default_grad_mode="train")
    return parser


def run_training(args, tracker: dict) -> None:
    set_random_seed(args.rng_seed)
    device = resolve_cuda_device("cuda") if torch.cuda.is_available() else "cpu"
    print(f"[dager] Using device: {device}", flush=True)
    output_dir = resolve_default_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_seq_class_model_and_tokenizer(args)
    model = model.to(device)
    train_dataset, eval_dataset, data_collator = load_seq_class_datasets(args, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=data_collator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=data_collator,
    )

    trainable_params = iter_trainable_parameters(model)
    opt = AdamW(trainable_params, lr=5e-5)
    num_training_steps = max(1, args.num_epochs * len(train_loader))
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=opt,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    progress_bar = tqdm(range(num_training_steps))
    wrapper = prepare_training_defense(model, args, trainable_params)
    train_start = time.time()
    n_steps = 0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        train_predictions = []
        train_references = []

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            model_inputs = {k: v for k, v in batch.items() if k != "labels"}

            opt.zero_grad(set_to_none=True)
            outputs = model(**model_inputs, labels=labels)
            logits = outputs.logits
            loss = outputs.loss

            epoch_loss += float(loss.item())
            train_predictions.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
            train_references.extend(labels.detach().cpu().tolist())

            apply_training_defense(
                model,
                wrapper,
                trainable_params,
                batch,
                labels,
                loss,
                args,
            )

            opt.step()
            lr_scheduler.step()
            progress_bar.update(1)

            n_steps += 1
            tracker["steps_completed"] = n_steps
            if args.save_every > 0 and n_steps % args.save_every == 0:
                ckpt_dir = output_dir / "checkpoints" / f"step_{n_steps}"
                tracker["last_checkpoint_path"] = save_model(model, tokenizer, ckpt_dir, args.train_method)

        train_metrics = classification_metrics(train_predictions, train_references, args.dataset)
        tracker["final_train_loss"] = epoch_loss / max(1, len(train_loader))
        print("metric train: ", train_metrics)
        print("loss train: ", f"{tracker['final_train_loss']:.6f}")

        eval_metrics = evaluate_model(model, eval_loader, device, args.dataset)
        tracker["eval_metrics"] = eval_metrics
        print("metric eval: ", eval_metrics)

    tracker["total_train_time"] = time.strftime("%H:%M:%S", time.gmtime(time.time() - train_start))
    final_path = output_dir / ("final" if args.train_method != "lora" else "final.pt")
    tracker["final_model_path"] = save_model(model, tokenizer, final_path, args.train_method)
    print("END")


def main():
    parser = build_parser()
    args = parser.parse_args()
    normalize_legacy_training_defense_args(args)
    install_terminal_logging(args)
    tracker = init_result_tracker(args)

    try:
        run_training(args, tracker)
    except Exception as exc:
        tracker["result_status"] = "failed"
        tracker["error_type"] = type(exc).__name__
        tracker["error_message"] = str(exc)
        emit_train_result_summary(args, tracker)
        raise

    emit_train_result_summary(args, tracker)


if __name__ == "__main__":
    main()
