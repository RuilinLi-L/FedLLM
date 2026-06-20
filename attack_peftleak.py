from __future__ import annotations

import argparse
import datetime
import time
from collections import Counter

import numpy as np
import torch

from attacks.peftleak_text import (
    optimize_text_embeddings,
    select_peft_gradient_tensors,
    summarize_token_predictions,
)
from utils.defense_common import add_shared_defense_args, defense_param_spec, fmt_summary_value, safe_mean
from utils.defenses import apply_defense, requires_gradient_generation_defense
from utils.gpu import resolve_cuda_device
from utils.lrb_presets import apply_lrb_preset
from utils.representation_bottleneck import rep_bottleneck_summary_fields, validate_rep_bottleneck_args


SUMMARY_START = "===== RESULT SUMMARY START ====="
SUMMARY_END = "===== RESULT SUMMARY END ====="
SUPPORTED_PEFTLEAK_TEXT_DEFENSES = {
    "none",
    "noise",
    "dpsgd",
    "topk",
    "compression",
    "soteria",
    "mixup",
    "lrb",
    "lrbprojonly",
    "signed_bottleneck",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FedLLM PEFT text attack for LoRA/IA3/adapter gradients")
    parser.add_argument("--dataset", choices=["cola", "sst2", "rte", "rotten_tomatoes", "stanfordnlp/imdb", "glnmario/ECHR"], required=True)
    parser.add_argument("--task", choices=["seq_class"], default="seq_class")
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--n_inputs", type=int, required=True)
    parser.add_argument("--start_input", type=int, default=0)
    parser.add_argument("--end_input", type=int, default=100000)
    parser.add_argument("--model_path", type=str, default="bert-base-uncased")
    parser.add_argument("--finetuned_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device_grad", type=str, default="cpu")
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["sdpa", "eager"])
    parser.add_argument("--precision", type=str, default="full", choices=["8bit", "half", "full", "double"])
    parser.add_argument("--pad", choices=["right", "left"], default="right")
    parser.add_argument("--grad_b", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--algo", type=str, default="sgd", choices=["sgd", "fedavg"])
    parser.add_argument("--avg_epochs", type=int, default=None)
    parser.add_argument("--b_mini", type=int, default=None)
    parser.add_argument("--avg_lr", type=float, default=None)
    parser.add_argument("--hidden_act", type=str, default=None)
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "mse"])

    parser.add_argument("--train_method", type=str, default="peft", choices=["peft", "lora"])
    parser.add_argument("--peft_method", type=str, default="lora", choices=["lora", "ia3", "prefix", "adapter"])
    parser.add_argument("--peft_num_virtual_tokens", type=int, default=None)
    parser.add_argument(
        "--adapter_reduction_factor",
        type=int,
        default=None,
        help="Adapter bottleneck reduction factor. Defaults to checkpoint metadata, or 16 for new adapters.",
    )
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_target_modules", type=str, default=None)

    parser.add_argument("--peftleak_steps", type=int, default=60)
    parser.add_argument("--peftleak_lr", type=float, default=0.1)
    parser.add_argument("--peftleak_tv_weight", type=float, default=0.0)
    parser.add_argument("--peftleak_entropy_weight", type=float, default=0.0)
    parser.add_argument("--peftleak_label_search", action="store_true", default=False)
    parser.add_argument(
        "--peftleak_label_candidates",
        type=str,
        default=None,
        help="Optional comma-separated classification labels for label search.",
    )
    add_shared_defense_args(parser, default_grad_mode="eval")
    return parser


def _init_tracker(args):
    requested = max(0, min(args.n_inputs, args.end_input) - args.start_input)
    return {
        "summary_emitted": False,
        "summary_version": 2,
        "result_status": "ok",
        "n_inputs_requested": requested,
        "n_inputs_completed": 0,
        "last_input_idx": None,
        "last_input_time": None,
        "last_total_time": None,
        "last_rec_status": None,
        "rec_token_values": [],
        "optimization_loss_values": [],
        "aggregate_metrics": {},
        "error_type": None,
        "error_message": None,
    }


def _rouge_ngram_f1(predictions, references, n: int) -> float:
    def ngrams(text):
        toks = text.split()
        if len(toks) < n:
            return []
        return [tuple(toks[i:i+n]) for i in range(len(toks) - n + 1)]

    scores = []
    for pred, ref in zip(predictions, references):
        p = Counter(ngrams(pred))
        r = Counter(ngrams(ref))
        if not p and not r:
            scores.append(1.0)
            continue
        overlap = sum((p & r).values())
        precision = overlap / max(1, sum(p.values()))
        recall = overlap / max(1, sum(r.values()))
        scores.append(0.0 if precision + recall <= 0 else (2 * precision * recall / (precision + recall)))
    return float(np.mean(scores)) * 100.0 if scores else 0.0


def _lcs_len(a, b) -> int:
    prev = [0] * (len(b) + 1)
    for tok_a in a:
        curr = [0]
        for j, tok_b in enumerate(b, start=1):
            curr.append(prev[j - 1] + 1 if tok_a == tok_b else max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def _rouge_l_f1(predictions, references) -> float:
    scores = []
    for pred, ref in zip(predictions, references):
        p = pred.split()
        r = ref.split()
        lcs = _lcs_len(p, r)
        precision = lcs / max(1, len(p))
        recall = lcs / max(1, len(r))
        scores.append(0.0 if precision + recall <= 0 else (2 * precision * recall / (precision + recall)))
    return float(np.mean(scores)) * 100.0 if scores else 0.0


def compute_text_metrics(predictions, references):
    try:
        from datasets import load_metric

        metric = load_metric("rouge")
        res = metric.compute(predictions=predictions, references=references)
        summary = {}
        for metric_name in ("rouge1", "rouge2", "rougeL", "rougeLsum"):
            curr = res[metric_name].mid
            summary[f"agg_{metric_name}_fm"] = curr.fmeasure * 100
            summary[f"agg_{metric_name}_p"] = curr.precision * 100
            summary[f"agg_{metric_name}_r"] = curr.recall * 100
        summary["agg_r1fm_r2fm"] = summary["agg_rouge1_fm"] + summary["agg_rouge2_fm"]
        summary["rouge_backend"] = "datasets"
        return summary
    except Exception:
        r1 = _rouge_ngram_f1(predictions, references, 1)
        r2 = _rouge_ngram_f1(predictions, references, 2)
        rl = _rouge_l_f1(predictions, references)
        return {
            "agg_rouge1_fm": r1,
            "agg_rouge1_p": "n/a",
            "agg_rouge1_r": "n/a",
            "agg_rouge2_fm": r2,
            "agg_rouge2_p": "n/a",
            "agg_rouge2_r": "n/a",
            "agg_rougeL_fm": rl,
            "agg_rougeL_p": "n/a",
            "agg_rougeL_r": "n/a",
            "agg_rougeLsum_fm": rl,
            "agg_rougeLsum_p": "n/a",
            "agg_rougeLsum_r": "n/a",
            "agg_r1fm_r2fm": r1 + r2,
            "rouge_backend": "simple_ngram_fallback",
        }


def _emit_result_summary(args, tracker):
    if tracker.get("summary_emitted"):
        return
    defense_param_name, defense_param_value = defense_param_spec(args)
    fields = [
        ("summary_version", tracker.get("summary_version", 2)),
        ("result_status", tracker.get("result_status", "ok")),
        ("attack", "fedllm_peft_text_opt"),
        ("attack_variant", "text_opt"),
        ("dataset", args.dataset),
        ("split", args.split),
        ("task", args.task),
        ("model_path", args.model_path),
        ("finetuned_path", args.finetuned_path),
        ("batch_size", args.batch_size),
        ("train_method", args.train_method),
        ("peft_method", getattr(args, "peft_method", None)),
        ("peft_type", getattr(args, "peft_type", None)),
        ("peft_eval_scope", getattr(args, "peft_eval_scope", "n/a")),
        ("peft_target_modules", getattr(args, "peft_target_modules", None)),
        ("peft_feedforward_modules", getattr(args, "peft_feedforward_modules", None)),
        ("peft_num_virtual_tokens", getattr(args, "peft_num_virtual_tokens", None)),
        ("adapter_reduction_factor", getattr(args, "adapter_reduction_factor", None)),
        ("peft_checkpoint_type", getattr(args, "peft_checkpoint_type", None)),
        ("peft_adapter_r", getattr(args, "peft_adapter_r", None)),
        ("peft_adapter_target_modules", getattr(args, "peft_adapter_target_modules", None)),
        ("peft_adapter_feedforward_modules", getattr(args, "peft_adapter_feedforward_modules", None)),
        ("peft_adapter_task_type", getattr(args, "peft_adapter_task_type", None)),
        ("peft_adapter_base_model", getattr(args, "peft_adapter_base_model", None)),
        ("peft_adapter_peft_type", getattr(args, "peft_adapter_peft_type", None)),
        ("peft_adapter_reduction_factor", getattr(args, "peft_adapter_reduction_factor", None)),
        ("peft_adapter_architecture", getattr(args, "peft_adapter_architecture", None)),
        ("peft_adapter_name", getattr(args, "peft_adapter_name", None)),
        ("lora_r", getattr(args, "lora_r", None)),
        ("lora_target_modules", getattr(args, "lora_target_modules", None)),
        ("defense", args.defense),
        ("defense_param_name", defense_param_name),
        ("defense_param_value", defense_param_value),
        *rep_bottleneck_summary_fields(args),
        ("peftleak_steps", args.peftleak_steps),
        ("peftleak_lr", args.peftleak_lr),
        ("peftleak_label_search", args.peftleak_label_search),
        ("peftleak_label_mode", "search" if args.peftleak_label_search else "known"),
        ("n_inputs_requested", tracker.get("n_inputs_requested")),
        ("n_inputs_completed", tracker.get("n_inputs_completed")),
        ("last_input_idx", tracker.get("last_input_idx")),
        ("last_input_time", tracker.get("last_input_time")),
        ("last_total_time", tracker.get("last_total_time")),
        ("last_rec_status", tracker.get("last_rec_status")),
        ("rec_token_mean", safe_mean(tracker.get("rec_token_values", []))),
        ("optimization_loss_mean", safe_mean(tracker.get("optimization_loss_values", []))),
        ("optimization_loss_first", tracker.get("optimization_loss_first")),
        ("optimization_loss_last", tracker.get("optimization_loss_last")),
        ("optimization_loss_reduction", tracker.get("optimization_loss_reduction")),
        ("selected_adapter_gradient_count", tracker.get("selected_adapter_gradient_count")),
        ("selected_adapter_gradient_names", tracker.get("selected_adapter_gradient_names")),
        ("sequence_length", tracker.get("sequence_length")),
    ]
    if tracker.get("error_type"):
        fields.append(("error_type", tracker["error_type"]))
    if tracker.get("error_message"):
        fields.append(("error_message", tracker["error_message"]))
    for key in sorted(tracker.get("aggregate_metrics", {})):
        fields.append((key, tracker["aggregate_metrics"][key]))

    print(SUMMARY_START, flush=True)
    for key, value in fields:
        print(f"{key}={fmt_summary_value(value)}", flush=True)
    print(SUMMARY_END, flush=True)
    tracker["summary_emitted"] = True


def _validate_args(args):
    from utils.peft_utils import apply_peft_config_to_args, normalize_peft_args

    normalize_peft_args(args)
    args.train_method = "peft"
    if args.peft_method not in {"lora", "ia3", "adapter"}:
        raise NotImplementedError("FedLLM PEFT text supports --peft_method lora|ia3|adapter; prefix is training-only.")
    if args.defense == "dager":
        return args
    if args.defense not in SUPPORTED_PEFTLEAK_TEXT_DEFENSES:
        raise NotImplementedError(
            f"FedLLM PEFT text supports defenses {sorted(SUPPORTED_PEFTLEAK_TEXT_DEFENSES)}; got {args.defense!r}."
        )
    apply_lrb_preset(args)
    validate_rep_bottleneck_args(args)
    apply_peft_config_to_args(args, require_checkpoint=True)
    apply_lrb_preset(args)
    return args


def _label_candidates(raw: str | None):
    if raw is None:
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def reconstruct(args, sample, model_wrapper):
    sequences, true_labels = sample
    tokenizer = model_wrapper.tokenizer
    orig_batch = tokenizer(
        sequences,
        padding=True,
        truncation=True,
        max_length=min(tokenizer.model_max_length, model_wrapper.emb_size - 20),
        return_tensors="pt",
    ).to(args.device)

    if requires_gradient_generation_defense(args.defense):
        true_grads = apply_defense(None, args, model_wrapper=model_wrapper, batch=orig_batch, labels=true_labels)
    else:
        true_grads = model_wrapper.compute_grads(orig_batch, true_labels)
        true_grads = apply_defense(true_grads, args, model_wrapper=model_wrapper, batch=orig_batch, labels=true_labels)

    indices, names = select_peft_gradient_tensors(
        true_grads,
        model_wrapper.trainable_parameter_names(),
        args.peft_method,
    )
    if not indices:
        raise ValueError("No visible LoRA/IA3/adapter gradients after defense; cannot run FedLLM PEFT text attack.")

    attention_mask = orig_batch.get("attention_mask")
    reference_mask = None if attention_mask is None else attention_mask.detach().cpu().tolist()
    ignored_token_ids = {
        getattr(model_wrapper, "pad_token", None),
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "bos_token_id", None),
        getattr(tokenizer, "cls_token_id", None),
        getattr(tokenizer, "sep_token_id", None),
    }
    attack_result = optimize_text_embeddings(
        model_wrapper=model_wrapper,
        batch=orig_batch,
        labels=true_labels,
        target_grads=true_grads,
        parameter_names=model_wrapper.trainable_parameter_names(),
        peft_method=args.peft_method,
        steps=args.peftleak_steps,
        lr=args.peftleak_lr,
        tv_weight=args.peftleak_tv_weight,
        entropy_weight=args.peftleak_entropy_weight,
        label_known=not args.peftleak_label_search,
        label_candidates=_label_candidates(args.peftleak_label_candidates),
        ignored_token_ids=ignored_token_ids,
        reference_mask=reference_mask,
    )
    predictions = summarize_token_predictions(attack_result["predicted_ids"], tokenizer)
    references = tokenizer.batch_decode(orig_batch["input_ids"].detach().cpu().tolist(), skip_special_tokens=True)
    return predictions, references, attack_result, names


def main(argv=None):
    from utils.data import TextDataset
    from utils.models import ModelWrapper

    parser = build_parser()
    args = parser.parse_args(argv)
    args.device = resolve_cuda_device(args.device)
    args.device_grad = args.device_grad or args.device
    _validate_args(args)
    np.random.seed(args.rng_seed)
    torch.manual_seed(args.rng_seed)
    print(f"[peftleak] Using device: {args.device}", flush=True)
    tracker = _init_tracker(args)
    start_time = time.time()

    if args.defense == "dager":
        tracker["result_status"] = "unsupported"
        tracker["last_rec_status"] = "unsupported"
        tracker["error_type"] = "unsupported_defense"
        tracker["error_message"] = "DAGER defense is DAGER-specific and is excluded from FedLLM PEFT text matrices."
        tracker["last_total_time"] = str(datetime.timedelta(seconds=time.time() - start_time)).split(".")[0]
        _emit_result_summary(args, tracker)
        return 0

    try:
        dataset = TextDataset(args.device, args.dataset, args.split, args.n_inputs, args.batch_size, args.cache_dir)
        model_wrapper = ModelWrapper(args)
        predictions = []
        references = []
        for input_idx in range(args.start_input, min(args.n_inputs, args.end_input)):
            input_start = time.time()
            args.defense_rng_step = tracker["n_inputs_completed"]
            sample = dataset[input_idx]
            print(f"[peftleak] Running input #{input_idx} of {args.n_inputs}.", flush=True)
            pred, ref, attack_result, selected_names = reconstruct(args, sample, model_wrapper)
            predictions.extend(pred)
            references.extend(ref)
            tracker["rec_token_values"].append(float(attack_result["rec_token_mean"]))
            tracker["optimization_loss_values"].append(float(attack_result["loss"]))
            tracker["optimization_loss_first"] = attack_result.get("initial_loss")
            tracker["optimization_loss_last"] = attack_result.get("loss")
            tracker["optimization_loss_reduction"] = attack_result.get("loss_reduction")
            tracker["selected_adapter_gradient_count"] = attack_result.get("selected_gradient_count", len(selected_names))
            tracker["selected_adapter_gradient_names"] = ";".join(selected_names)
            tracker["sequence_length"] = attack_result.get("sequence_length")
            tracker["last_rec_status"] = "ok"
            tracker["n_inputs_completed"] += 1
            tracker["last_input_idx"] = input_idx
            tracker["last_input_time"] = str(datetime.timedelta(seconds=time.time() - input_start)).split(".")[0]
            tracker["last_total_time"] = str(datetime.timedelta(seconds=time.time() - start_time)).split(".")[0]
            print(f"[peftleak] selected adapter gradients ({len(selected_names)}): {', '.join(selected_names[:6])}", flush=True)
            initial_loss = attack_result.get("initial_loss")
            loss_reduction = attack_result.get("loss_reduction")
            print(
                f"[peftleak] loss={attack_result['loss']:.6f} "
                f"initial_loss={initial_loss if initial_loss is not None else 'n/a'} "
                f"loss_reduction={loss_reduction if loss_reduction is not None else 'n/a'} "
                f"rec_token={attack_result['rec_token_mean']:.6f}",
                flush=True,
            )

        tracker["aggregate_metrics"] = compute_text_metrics(predictions, references)
        if tracker["last_total_time"] is None:
            tracker["last_total_time"] = str(datetime.timedelta(seconds=time.time() - start_time)).split(".")[0]
        _emit_result_summary(args, tracker)
        return 0
    except Exception as exc:
        tracker["result_status"] = "failed"
        tracker["last_rec_status"] = tracker.get("last_rec_status") or "failed"
        tracker["error_type"] = type(exc).__name__
        tracker["error_message"] = str(exc)
        if tracker["last_total_time"] is None:
            tracker["last_total_time"] = str(datetime.timedelta(seconds=time.time() - start_time)).split(".")[0]
        _emit_result_summary(args, tracker)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
