from __future__ import annotations

import argparse
import datetime
import time

import numpy as np
import torch

from attacks.partial_transformer_gradients import (
    PARTIAL_TRANSFORMER_GRADIENTS_ATTACK,
    PTG_GRADIENT_MATCHING_VARIANT,
    PTG_PARAM_FILTERS,
    filter_partial_transformer_gradients,
    optimize_partial_text_embeddings,
    ptg_selector_summary_fields,
    selected_partial_gradient_tensors,
    validate_ptg_selector_args,
)
from attacks.peftleak_text import summarize_token_predictions
from attack_peftleak import compute_text_metrics
from utils.defense_common import add_shared_defense_args, defense_param_spec, fmt_summary_value, safe_mean
from utils.defenses import apply_defense, requires_gradient_generation_defense
from utils.gpu import resolve_cuda_device, resolve_gradient_device
from utils.lrb_presets import apply_lrb_preset
from utils.representation_bottleneck import rep_bottleneck_summary_fields, validate_rep_bottleneck_args


SUMMARY_START = "===== RESULT SUMMARY START ====="
SUMMARY_END = "===== RESULT SUMMARY END ====="
SUPPORTED_PTG_DEFENSES = {
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
    parser = argparse.ArgumentParser(
        description="FedLLM reproduction of partial Transformer gradient leakage via gradient matching."
    )
    parser.add_argument(
        "--dataset",
        choices=["cola", "sst2", "rte", "rotten_tomatoes", "stanfordnlp/imdb", "glnmario/ECHR"],
        required=True,
    )
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
    parser.add_argument(
        "--device_grad",
        type=str,
        default="auto",
        help="Gradient computation device. 'auto' follows the resolved --device; use 'cpu' to reproduce the legacy path.",
    )
    parser.add_argument("--attn_implementation", type=str, default="eager", choices=["sdpa", "eager"])
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
    parser.add_argument("--train_method", type=str, default="full", choices=["full"])

    parser.add_argument(
        "--gradient_layer_subset",
        type=str,
        default="all",
        help="Gradient exposure by transformer layer: all, firstN, lastN, or midN.",
    )
    parser.add_argument(
        "--gradient_param_filter",
        type=str,
        default="all",
        choices=sorted(PTG_PARAM_FILTERS),
        help="Gradient exposure by module family for partial Transformer gradient matching.",
    )

    parser.add_argument("--ptg_steps", type=int, default=80)
    parser.add_argument("--ptg_lr", type=float, default=0.1)
    parser.add_argument("--ptg_restarts", type=int, default=1)
    parser.add_argument("--ptg_match_loss", type=str, default="cosine", choices=["cosine", "normalized_mse"])
    parser.add_argument("--ptg_label_mode", type=str, default="known", choices=["known", "search"])
    parser.add_argument("--ptg_label_candidates", type=str, default=None)
    parser.add_argument("--ptg_decode_metric", type=str, default="cos", choices=["cos", "l2"])
    parser.add_argument("--ptg_tv_weight", type=float, default=0.0)
    parser.add_argument("--ptg_embed_norm_weight", type=float, default=0.0)
    parser.add_argument("--ptg_entropy_weight", type=float, default=0.0)
    parser.add_argument("--ptg_fix_special_tokens", action="store_true", default=True)
    parser.add_argument("--no_ptg_fix_special_tokens", action="store_false", dest="ptg_fix_special_tokens")
    parser.add_argument("--ptg_lm_prior_weight", type=float, default=0.0)
    parser.add_argument("--ptg_swap_steps", type=int, default=0)

    add_shared_defense_args(parser, default_grad_mode="eval")
    return parser


def _init_tracker(args):
    requested = max(0, min(args.n_inputs, args.end_input) - args.start_input)
    return {
        "summary_emitted": False,
        "summary_version": 2,
        "result_status": "ok",
        "attack": PARTIAL_TRANSFORMER_GRADIENTS_ATTACK,
        "partial_attack_variant": PTG_GRADIENT_MATCHING_VARIANT,
        "n_inputs_requested": requested,
        "n_inputs_completed": 0,
        "last_input_idx": None,
        "last_input_time": None,
        "last_total_time": None,
        "last_rec_status": None,
        "rec_token_values": [],
        "ptg_final_loss_values": [],
        "ptg_initial_loss_values": [],
        "ptg_loss_reduction_values": [],
        "aggregate_metrics": {},
        "selected_gradient_count": None,
        "selected_gradient_names": None,
        "fixed_token_values": [],
        "sequence_length": None,
        "error_type": None,
        "error_message": None,
    }


def _ignored_token_ids(tokenizer, model_wrapper) -> set[int | None]:
    return {
        getattr(model_wrapper, "pad_token", None),
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "bos_token_id", None),
        getattr(tokenizer, "cls_token_id", None),
        getattr(tokenizer, "sep_token_id", None),
    }


def _label_candidates(raw: str | None):
    if raw is None:
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _mark_ptg_variant(args, selected_names=None):
    info = getattr(args, "ptg_gradient_info", {})
    info["partial_attack_variant"] = PTG_GRADIENT_MATCHING_VARIANT
    info["unsupported_reason"] = "n/a"
    if selected_names is not None:
        info["selected_gradient_count"] = len(selected_names)
        info["selected_gradient_names"] = ";".join(selected_names)
    setattr(args, "ptg_gradient_info", info)


def _emit_result_summary(args, tracker):
    if tracker.get("summary_emitted"):
        return
    defense_param_name, defense_param_value = defense_param_spec(args)
    fields = [
        ("summary_version", tracker.get("summary_version", 2)),
        ("result_status", tracker.get("result_status", "ok")),
        ("attack", tracker.get("attack")),
        ("partial_attack_variant", tracker.get("partial_attack_variant")),
        ("dataset", args.dataset),
        ("split", args.split),
        ("task", args.task),
        ("model_path", args.model_path),
        ("finetuned_path", args.finetuned_path),
        ("batch_size", args.batch_size),
        ("train_method", args.train_method),
        ("defense", args.defense),
        ("defense_param_name", defense_param_name),
        ("defense_param_value", defense_param_value),
        *rep_bottleneck_summary_fields(args),
        *ptg_selector_summary_fields(args),
        ("ptg_steps", args.ptg_steps),
        ("ptg_lr", args.ptg_lr),
        ("ptg_restarts", args.ptg_restarts),
        ("ptg_match_loss", args.ptg_match_loss),
        ("ptg_label_mode", args.ptg_label_mode),
        ("ptg_decode_metric", args.ptg_decode_metric),
        ("ptg_tv_weight", args.ptg_tv_weight),
        ("ptg_embed_norm_weight", args.ptg_embed_norm_weight),
        ("ptg_entropy_weight", args.ptg_entropy_weight),
        ("ptg_fix_special_tokens", args.ptg_fix_special_tokens),
        ("ptg_lm_prior_weight", args.ptg_lm_prior_weight),
        ("ptg_swap_steps", args.ptg_swap_steps),
        ("selected_gradient_count", tracker.get("selected_gradient_count")),
        ("selected_gradient_names", tracker.get("selected_gradient_names")),
        ("fixed_token_count", safe_mean(tracker.get("fixed_token_values", []))),
        ("sequence_length", tracker.get("sequence_length")),
        ("n_inputs_requested", tracker.get("n_inputs_requested")),
        ("n_inputs_completed", tracker.get("n_inputs_completed")),
        ("last_input_idx", tracker.get("last_input_idx")),
        ("last_input_time", tracker.get("last_input_time")),
        ("last_total_time", tracker.get("last_total_time")),
        ("last_rec_status", tracker.get("last_rec_status")),
        ("rec_token_mean", safe_mean(tracker.get("rec_token_values", []))),
        ("ptg_initial_loss", safe_mean(tracker.get("ptg_initial_loss_values", []))),
        ("ptg_final_loss", safe_mean(tracker.get("ptg_final_loss_values", []))),
        ("ptg_loss_reduction", safe_mean(tracker.get("ptg_loss_reduction_values", []))),
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
    args.train_method = "full"
    if args.defense == "dager":
        raise NotImplementedError("DAGER defense is DAGER-specific and is excluded from PTG reproduction runs.")
    if args.defense not in SUPPORTED_PTG_DEFENSES:
        raise NotImplementedError(f"PTG supports defenses {sorted(SUPPORTED_PTG_DEFENSES)}; got {args.defense!r}.")
    if getattr(args, "attn_implementation", None) == "sdpa":
        print(
            "[ptg] Switching --attn_implementation sdpa -> eager: PTG matching needs second-order gradients.",
            flush=True,
        )
        args.attn_implementation = "eager"
    apply_lrb_preset(args)
    validate_rep_bottleneck_args(args)
    validate_ptg_selector_args(args)
    return args


def _compute_defended_partial_grads(args, model_wrapper, batch, labels):
    if requires_gradient_generation_defense(args.defense):
        true_grads = apply_defense(None, args, model_wrapper=model_wrapper, batch=batch, labels=labels)
    else:
        true_grads = model_wrapper.compute_grads(batch, labels)
        true_grads = apply_defense(true_grads, args, model_wrapper=model_wrapper, batch=batch, labels=labels)
    parameter_names = model_wrapper.trainable_parameter_names()
    partial_grads, info = filter_partial_transformer_gradients(
        true_grads,
        parameter_names=parameter_names,
        layer_subset=args.gradient_layer_subset,
        param_filter=args.gradient_param_filter,
        model_path=args.model_path,
    )
    setattr(args, "ptg_gradient_info", info)
    selected_indices, selected_names = selected_partial_gradient_tensors(partial_grads, parameter_names)
    _mark_ptg_variant(args, selected_names)
    if not selected_indices:
        raise ValueError("No visible partial gradients after filtering/defense; cannot run PTG matching.")
    return partial_grads, parameter_names, selected_names


def reconstruct(args, sample, model_wrapper):
    sequences, true_labels = sample
    tokenizer = model_wrapper.tokenizer
    batch = tokenizer(
        sequences,
        padding=True,
        truncation=True,
        max_length=min(tokenizer.model_max_length, model_wrapper.emb_size - 20),
        return_tensors="pt",
    ).to(args.device)

    attention_mask = batch.get("attention_mask")
    reference_mask = None if attention_mask is None else attention_mask.detach().cpu().tolist()
    references = tokenizer.batch_decode(batch["input_ids"].detach().cpu().tolist(), skip_special_tokens=True)
    ignored_token_ids = _ignored_token_ids(tokenizer, model_wrapper)

    partial_grads, parameter_names, selected_names = _compute_defended_partial_grads(
        args,
        model_wrapper,
        batch,
        true_labels,
    )
    attack_result = optimize_partial_text_embeddings(
        model_wrapper=model_wrapper,
        batch=batch,
        labels=true_labels,
        target_grads=partial_grads,
        parameter_names=parameter_names,
        steps=args.ptg_steps,
        lr=args.ptg_lr,
        restarts=args.ptg_restarts,
        match_loss=args.ptg_match_loss,
        label_mode=args.ptg_label_mode,
        label_candidates=_label_candidates(args.ptg_label_candidates),
        decode_metric=args.ptg_decode_metric,
        tv_weight=args.ptg_tv_weight,
        embed_norm_weight=args.ptg_embed_norm_weight,
        entropy_weight=args.ptg_entropy_weight,
        fix_special_tokens=args.ptg_fix_special_tokens,
        lm_prior_weight=args.ptg_lm_prior_weight,
        swap_steps=args.ptg_swap_steps,
        ignored_token_ids=ignored_token_ids,
        reference_mask=reference_mask,
    )
    predictions = summarize_token_predictions(attack_result["predicted_ids"], tokenizer)
    return predictions, references, attack_result, selected_names


def main(argv=None):
    from utils.data import TextDataset
    from utils.models import ModelWrapper

    parser = build_parser()
    args = parser.parse_args(argv)
    args.device = resolve_cuda_device(args.device)
    args.device_grad = resolve_gradient_device(args.device_grad, args.device)
    _validate_args(args)
    np.random.seed(args.rng_seed)
    torch.manual_seed(args.rng_seed)
    print(f"[ptg] Using device: {args.device} | gradient device: {args.device_grad}", flush=True)
    tracker = _init_tracker(args)
    start_time = time.time()

    try:
        dataset = TextDataset(args.device, args.dataset, args.split, args.n_inputs, args.batch_size, args.cache_dir)
        model_wrapper = ModelWrapper(args)
        predictions = []
        references = []
        for input_idx in range(args.start_input, min(args.n_inputs, args.end_input)):
            input_start = time.time()
            args.defense_rng_step = tracker["n_inputs_completed"]
            print(f"[ptg] Running input #{input_idx} of {args.n_inputs}.", flush=True)
            pred, ref, attack_result, selected_names = reconstruct(args, dataset[input_idx], model_wrapper)
            predictions.extend(pred)
            references.extend(ref)
            tracker["rec_token_values"].append(float(attack_result["rec_token_mean"]))
            if attack_result.get("initial_loss") is not None:
                tracker["ptg_initial_loss_values"].append(float(attack_result["initial_loss"]))
            tracker["ptg_final_loss_values"].append(float(attack_result["loss"]))
            if attack_result.get("loss_reduction") is not None:
                tracker["ptg_loss_reduction_values"].append(float(attack_result["loss_reduction"]))
            tracker["fixed_token_values"].append(float(attack_result.get("fixed_token_count", 0)))
            tracker["selected_gradient_count"] = attack_result.get("selected_gradient_count", len(selected_names))
            tracker["selected_gradient_names"] = ";".join(selected_names)
            tracker["sequence_length"] = attack_result.get("sequence_length")
            tracker["last_rec_status"] = "ok"
            tracker["n_inputs_completed"] += 1
            tracker["last_input_idx"] = input_idx
            tracker["last_input_time"] = str(datetime.timedelta(seconds=time.time() - input_start)).split(".")[0]
            tracker["last_total_time"] = str(datetime.timedelta(seconds=time.time() - start_time)).split(".")[0]
            print(
                f"[ptg] selected={tracker['selected_gradient_count']} "
                f"initial_loss={attack_result.get('initial_loss')} "
                f"final_loss={attack_result.get('loss')} "
                f"rec_token={attack_result['rec_token_mean']:.6f}",
                flush=True,
            )

        tracker["aggregate_metrics"] = compute_text_metrics(predictions, references)
        if tracker["last_total_time"] is None:
            tracker["last_total_time"] = str(datetime.timedelta(seconds=time.time() - start_time)).split(".")[0]
        _emit_result_summary(args, tracker)
        return 0
    except ValueError as exc:
        if "No visible partial gradients" not in str(exc) and "No selected partial gradients" not in str(exc):
            tracker["result_status"] = "failed"
            tracker["last_rec_status"] = tracker.get("last_rec_status") or "failed"
            tracker["error_type"] = type(exc).__name__
            tracker["error_message"] = str(exc)
            if tracker["last_total_time"] is None:
                tracker["last_total_time"] = str(datetime.timedelta(seconds=time.time() - start_time)).split(".")[0]
            _emit_result_summary(args, tracker)
            raise
        tracker["result_status"] = "unsupported"
        tracker["last_rec_status"] = "unsupported"
        tracker["error_type"] = "unsupported_partial_exposure"
        tracker["error_message"] = str(exc)
        if tracker["last_total_time"] is None:
            tracker["last_total_time"] = str(datetime.timedelta(seconds=time.time() - start_time)).split(".")[0]
        _mark_ptg_variant(args, [])
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
