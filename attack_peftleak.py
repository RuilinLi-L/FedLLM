from __future__ import annotations

import argparse
import datetime
import time
from collections import Counter

import numpy as np
import torch

from attacks.peftleak_text import (
    get_token_embedding_matrix,
    optimize_text_embeddings,
    select_peft_gradient_tensors,
    summarize_token_predictions,
)
from attacks.peftleak_text_ratio import (
    build_text_ratio_gradients,
    build_text_token_statistics,
    decode_ratio_recovery,
)
from utils.defense_common import add_shared_defense_args, defense_param_spec, fmt_summary_value, safe_mean
from utils.data import ATTACK_SPLIT_CHOICES, dataset_summary_fields, record_dataset_protocol
from utils.defenses import (
    apply_defense,
    dpsgd_defense,
    gradient_compression,
    noise_injection,
    requires_gradient_generation_defense,
    topk_sparsification,
)
from utils.gpu import resolve_cuda_device, resolve_gradient_device
from utils.lrb_defense import apply_lrb_defense, lrb_seed_summary_fields
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
RATIO_DEFENSE_SEED_STRIDE = 1_000_003


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FedLLM PEFT text attack for LoRA/IA3/adapter gradients")
    parser.add_argument("--dataset", choices=["cola", "sst2", "rte", "rotten_tomatoes", "stanfordnlp/imdb", "glnmario/ECHR"], required=True)
    parser.add_argument("--task", choices=["seq_class"], default="seq_class")
    parser.add_argument("--split", choices=ATTACK_SPLIT_CHOICES, required=True)
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
    parser.add_argument("--peftleak_restarts", type=int, default=1)
    parser.add_argument("--peftleak_match_loss", type=str, default="normalized_mse", choices=["mse", "cosine", "normalized_mse"])
    parser.add_argument("--peftleak_attack_mode", type=str, default="opt", choices=["opt", "ratio", "both"])
    parser.add_argument("--peftleak_label_search", action="store_true", default=False)
    parser.add_argument(
        "--peftleak_label_candidates",
        type=str,
        default=None,
        help="Optional comma-separated classification labels for label search.",
    )
    parser.add_argument("--peftleak_ratio_bins", type=int, default=8)
    parser.add_argument("--peftleak_ratio_public_n_inputs", type=int, default=16)
    parser.add_argument("--peftleak_ratio_route", type=str, default="public_bins", choices=["public_bins", "oracle"])
    parser.add_argument("--peftleak_ratio_target", type=str, default="input_embedding", choices=["input_embedding"])
    add_shared_defense_args(parser, default_grad_mode="eval")
    return parser


def _init_tracker(args):
    requested = max(0, min(args.n_inputs, args.end_input) - args.start_input)
    attack_by_mode = {
        "opt": ("fedllm_peft_text_opt", "text_opt"),
        "ratio": ("fedllm_peft_text_ratio", "text_ratio"),
        "both": ("fedllm_peft_text_both", "text_ratio_plus_opt"),
    }
    attack, variant = attack_by_mode.get(getattr(args, "peftleak_attack_mode", "opt"), attack_by_mode["opt"])
    if getattr(args, "peftleak_attack_mode", "opt") in {"ratio", "both"} and getattr(args, "peft_method", None) != "adapter":
        attack, variant = "fedllm_peft_text_opt", "text_opt_ratio_unsupported_fallback"
    return {
        "summary_emitted": False,
        "summary_version": 2,
        "result_status": "ok",
        "attack": attack,
        "attack_variant": variant,
        "n_inputs_requested": requested,
        "n_inputs_completed": 0,
        "last_input_idx": None,
        "last_input_time": None,
        "last_total_time": None,
        "last_rec_status": None,
        "rec_token_values": [],
        "opt_rec_token_values": [],
        "ratio_rec_token_values": [],
        "optimization_loss_values": [],
        "ratio_loss_values": [],
        "aggregate_metrics": {},
        "ratio_status": "n/a",
        "ratio_collision_values": [],
        "ratio_recovered_hidden_values": [],
        "ratio_routed_token_values": [],
        "ratio_slot_count_values": [],
        "ratio_reportable": "n/a",
        "ratio_non_reportable_reason": None,
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


TEXT_METRIC_BACKENDS = {"auto", "datasets", "simple_ngram"}


def _normalize_text_metric_backend(backend: str) -> str:
    normalized = str(backend or "auto").strip().lower()
    if normalized not in TEXT_METRIC_BACKENDS:
        raise ValueError(
            f"Unsupported text metric backend {backend!r}; "
            f"expected one of {sorted(TEXT_METRIC_BACKENDS)}."
        )
    return normalized


def _load_datasets_rouge_metric():
    from datasets import load_metric

    return load_metric("rouge")


def _compute_datasets_text_metrics(predictions, references):
    try:
        metric = _load_datasets_rouge_metric()
        res = metric.compute(predictions=predictions, references=references)
    except Exception as exc:
        raise RuntimeError(
            "ROUGE backend 'datasets' failed. Install or cache the datasets rouge metric, "
            "or explicitly select the project-local 'simple_ngram' backend. "
            "No automatic metric fallback is used in strict mode."
        ) from exc

    summary = {}
    for metric_name in ("rouge1", "rouge2", "rougeL", "rougeLsum"):
        curr = res[metric_name].mid
        summary[f"agg_{metric_name}_fm"] = curr.fmeasure * 100
        summary[f"agg_{metric_name}_p"] = curr.precision * 100
        summary[f"agg_{metric_name}_r"] = curr.recall * 100
    summary["agg_r1fm_r2fm"] = summary["agg_rouge1_fm"] + summary["agg_rouge2_fm"]
    summary["rouge_backend"] = "datasets"
    return summary


def _compute_simple_ngram_text_metrics(predictions, references, *, fallback: bool):
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
        "rouge_backend": "simple_ngram_fallback" if fallback else "simple_ngram",
    }


def validate_text_metric_backend(backend: str) -> str:
    """Verify an explicitly selected backend before an expensive attack starts."""
    normalized = _normalize_text_metric_backend(backend)
    if normalized == "auto":
        raise ValueError("Metric preflight requires an explicit backend, not 'auto'.")
    if normalized == "datasets":
        _compute_datasets_text_metrics(["metric preflight"], ["metric preflight"])
    return normalized


def compute_text_metrics(predictions, references, *, backend: str = "auto"):
    """Compute text recovery metrics with an explicit or backward-compatible backend."""
    backend = _normalize_text_metric_backend(backend)
    if backend == "datasets":
        return _compute_datasets_text_metrics(predictions, references)
    if backend == "simple_ngram":
        return _compute_simple_ngram_text_metrics(predictions, references, fallback=False)

    try:
        return _compute_datasets_text_metrics(predictions, references)
    except Exception:
        return _compute_simple_ngram_text_metrics(predictions, references, fallback=True)


def _emit_result_summary(args, tracker):
    if tracker.get("summary_emitted"):
        return
    defense_param_name, defense_param_value = defense_param_spec(args)
    fields = [
        ("summary_version", tracker.get("summary_version", 2)),
        ("result_status", tracker.get("result_status", "ok")),
        ("attack", tracker.get("attack", "fedllm_peft_text_opt")),
        ("attack_variant", tracker.get("attack_variant", "text_opt")),
        ("dataset", args.dataset),
        ("split", args.split),
        *dataset_summary_fields(args),
        ("task", args.task),
        ("model_path", args.model_path),
        ("finetuned_path", args.finetuned_path),
        ("batch_size", args.batch_size),
        ("seed", getattr(args, "rng_seed", None)),
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
        *lrb_seed_summary_fields(args),
        *rep_bottleneck_summary_fields(args),
        ("peftleak_steps", args.peftleak_steps),
        ("peftleak_lr", args.peftleak_lr),
        ("peftleak_restarts", args.peftleak_restarts),
        ("peftleak_match_loss", args.peftleak_match_loss),
        ("peftleak_attack_mode", args.peftleak_attack_mode),
        ("peftleak_label_search", args.peftleak_label_search),
        ("peftleak_label_mode", "search" if args.peftleak_label_search else "known"),
        ("peftleak_ratio_bins", args.peftleak_ratio_bins),
        ("peftleak_ratio_public_n_inputs", args.peftleak_ratio_public_n_inputs),
        ("peftleak_ratio_route", args.peftleak_ratio_route),
        ("peftleak_ratio_target", args.peftleak_ratio_target),
        ("ratio_status", tracker.get("ratio_status")),
        ("public_stats_source", tracker.get("public_stats_source")),
        ("public_stats_n_inputs", tracker.get("public_stats_n_inputs")),
        ("n_inputs_requested", tracker.get("n_inputs_requested")),
        ("n_inputs_completed", tracker.get("n_inputs_completed")),
        ("last_input_idx", tracker.get("last_input_idx")),
        ("last_input_time", tracker.get("last_input_time")),
        ("last_total_time", tracker.get("last_total_time")),
        ("last_rec_status", tracker.get("last_rec_status")),
        ("rec_token_mean", safe_mean(tracker.get("rec_token_values", []))),
        ("opt_rec_token_mean", safe_mean(tracker.get("opt_rec_token_values", []))),
        ("ratio_rec_token_mean", safe_mean(tracker.get("ratio_rec_token_values", []))),
        ("ratio_collision_rate", safe_mean(tracker.get("ratio_collision_values", []))),
        ("ratio_recovered_hidden_count", safe_mean(tracker.get("ratio_recovered_hidden_values", []))),
        ("ratio_routed_token_count", safe_mean(tracker.get("ratio_routed_token_values", []))),
        ("ratio_slot_count", safe_mean(tracker.get("ratio_slot_count_values", []))),
        ("ratio_reportable", tracker.get("ratio_reportable", "n/a")),
        ("ratio_non_reportable_reason", tracker.get("ratio_non_reportable_reason")),
        ("optimization_loss_mean", safe_mean(tracker.get("optimization_loss_values", []))),
        ("ratio_loss_mean", safe_mean(tracker.get("ratio_loss_values", []))),
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
    if getattr(args, "train_method", "peft") == "lora":
        args.train_method = "peft"
        args.peft_method = "lora"
    else:
        args.train_method = "peft"
        args.peft_method = str(getattr(args, "peft_method", "lora") or "lora").strip().lower().replace("-", "_")

    if args.peft_method not in {"lora", "ia3", "adapter"}:
        raise NotImplementedError("FedLLM PEFT text supports --peft_method lora|ia3|adapter; prefix is training-only.")
    if (
        getattr(args, "peftleak_attack_mode", "opt") in {"opt", "both"}
        and getattr(args, "attn_implementation", None) == "sdpa"
    ):
        print(
            "[peftleak] Switching --attn_implementation sdpa -> eager for opt/both: "
            "the optimization attack needs second-order gradients, which PyTorch SDPA efficient attention does not support.",
            flush=True,
        )
        args.attn_implementation = "eager"
    if args.defense == "dager":
        return args
    if args.defense not in SUPPORTED_PEFTLEAK_TEXT_DEFENSES:
        raise NotImplementedError(
            f"FedLLM PEFT text supports defenses {sorted(SUPPORTED_PEFTLEAK_TEXT_DEFENSES)}; got {args.defense!r}."
        )
    apply_lrb_preset(args)
    validate_rep_bottleneck_args(args)
    try:
        from utils.peft_utils import apply_peft_config_to_args
    except ModuleNotFoundError as exc:
        if exc.name == "peft":
            raise ModuleNotFoundError(
                "FedLLM PEFT text attacks require the PEFT runtime. "
                "Create/activate the environment from environment-peftleak.yml before running reportable attacks."
            ) from exc
        raise
    apply_peft_config_to_args(args, require_checkpoint=True)
    apply_lrb_preset(args)
    return args


def _label_candidates(raw: str | None):
    if raw is None:
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _ignored_token_ids(tokenizer, model_wrapper) -> set[int | None]:
    return {
        getattr(model_wrapper, "pad_token", None),
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "bos_token_id", None),
        getattr(tokenizer, "cls_token_id", None),
        getattr(tokenizer, "sep_token_id", None),
    }


def _ratio_defense_seed(args, *, stochastic: bool) -> int:
    base_seed = int(getattr(args, "rng_seed", 0))
    if not stochastic:
        return base_seed
    step = getattr(args, "defense_rng_step", None)
    if step is None:
        return base_seed
    return base_seed + RATIO_DEFENSE_SEED_STRIDE * int(step)


def _apply_ratio_random_mask(grads, pct_mask: float, *, seed: int):
    out = []
    for idx, grad in enumerate(grads):
        if grad is None:
            out.append(None)
            continue
        gen = torch.Generator(device=grad.device)
        gen.manual_seed(int(seed) + 104729 * (idx + 1))
        mask = (torch.rand(grad.shape, device=grad.device, dtype=grad.dtype, generator=gen) > pct_mask).float()
        out.append(grad * mask)
    return tuple(out)


def _defend_ratio_gradient_tuple(args, raw_grads, names):
    stochastic_main = (
        args.defense in {"noise", "dpsgd", "compression"}
        or (args.defense == "none" and getattr(args, "defense_noise", None) is not None)
    )
    stochastic_mask = getattr(args, "defense_pct_mask", None) is not None
    fresh_seed = _ratio_defense_seed(args, stochastic=stochastic_main or stochastic_mask)
    main_seed = fresh_seed if stochastic_main else int(getattr(args, "rng_seed", 0))

    if args.defense == "dpsgd":
        defended = dpsgd_defense(
            [tuple(raw_grads)],
            float(args.defense_clip_norm),
            float(args.defense_noise or 0.0),
            seed=main_seed,
        )
    elif args.defense == "none":
        if getattr(args, "defense_noise", None) is not None:
            defended = noise_injection(raw_grads, float(args.defense_noise), seed=main_seed)
        else:
            defended = raw_grads
    elif args.defense == "noise":
        defended = noise_injection(raw_grads, float(args.defense_noise or 0.0), seed=main_seed)
    elif args.defense == "topk":
        defended = topk_sparsification(raw_grads, float(args.defense_topk_ratio))
    elif args.defense == "compression":
        defended = gradient_compression(raw_grads, int(args.defense_n_bits), seed=main_seed)
    elif args.defense in {"lrb", "lrbprojonly", "signed_bottleneck"}:
        defended = apply_lrb_defense(raw_grads, args, layer_names=names)
    elif args.defense in {"soteria", "mixup"}:
        defended = raw_grads
    else:
        raise ValueError(f"Unsupported FedLLM PEFT text ratio defense: {args.defense!r}")

    if getattr(args, "defense_pct_mask", None) is not None:
        defended = _apply_ratio_random_mask(defended, float(args.defense_pct_mask), seed=fresh_seed)
    return defended, names


def _run_opt_attack(args, orig_batch, true_labels, model_wrapper, ignored_token_ids, reference_mask):
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
        restarts=args.peftleak_restarts,
        match_loss=args.peftleak_match_loss,
        label_known=not args.peftleak_label_search,
        label_candidates=_label_candidates(args.peftleak_label_candidates),
        ignored_token_ids=ignored_token_ids,
        reference_mask=reference_mask,
    )
    predictions = summarize_token_predictions(attack_result["predicted_ids"], model_wrapper.tokenizer)
    return predictions, attack_result, names


def _run_ratio_attack(args, orig_batch, true_labels, model_wrapper, ratio_stats, ignored_token_ids, reference_mask):
    ratio_gradients = build_text_ratio_gradients(
        model_wrapper,
        orig_batch,
        true_labels,
        stats=ratio_stats,
        route=args.peftleak_ratio_route,
        target=args.peftleak_ratio_target,
        seed=int(args.rng_seed),
    )
    defended_grads, defended_names = _defend_ratio_gradient_tuple(args, ratio_gradients.grads, ratio_gradients.names)
    attack_result = decode_ratio_recovery(
        ratio_result=ratio_gradients,
        defended_grads=defended_grads,
        token_embedding_matrix=get_token_embedding_matrix(model_wrapper),
        batch=orig_batch,
        ignored_token_ids=ignored_token_ids,
        reference_mask=reference_mask,
    )
    if attack_result.get("gradient_names") is None:
        attack_result["gradient_names"] = defended_names
    predictions = summarize_token_predictions(attack_result["predicted_ids"], model_wrapper.tokenizer)
    return predictions, attack_result, defended_names


def reconstruct(args, sample, model_wrapper, ratio_stats=None):
    sequences, true_labels = sample
    tokenizer = model_wrapper.tokenizer
    orig_batch = tokenizer(
        sequences,
        padding=True,
        truncation=True,
        max_length=min(tokenizer.model_max_length, model_wrapper.emb_size - 20),
        return_tensors="pt",
    ).to(args.device)

    attention_mask = orig_batch.get("attention_mask")
    reference_mask = None if attention_mask is None else attention_mask.detach().cpu().tolist()
    ignored_token_ids = _ignored_token_ids(tokenizer, model_wrapper)
    references = tokenizer.batch_decode(orig_batch["input_ids"].detach().cpu().tolist(), skip_special_tokens=True)

    mode = args.peftleak_attack_mode
    opt_result = None
    opt_predictions = None
    opt_names: list[str] = []
    ratio_result = None
    ratio_predictions = None
    ratio_names: list[str] = []
    ratio_status = "not_requested"

    opt_requested = mode in {"opt", "both"} or (mode == "ratio" and args.peft_method != "adapter")
    ratio_requested = mode in {"ratio", "both"}

    if opt_requested:
        opt_predictions, opt_result, opt_names = _run_opt_attack(
            args,
            orig_batch,
            true_labels,
            model_wrapper,
            ignored_token_ids,
            reference_mask,
        )

    if ratio_requested:
        if args.peft_method != "adapter":
            ratio_status = f"unsupported_for_{args.peft_method}_fallback_opt"
        else:
            ratio_predictions, ratio_result, ratio_names = _run_ratio_attack(
                args,
                orig_batch,
                true_labels,
                model_wrapper,
                ratio_stats,
                ignored_token_ids,
                reference_mask,
            )
            ratio_status = "ok"

    primary = "ratio" if ratio_result is not None and mode in {"ratio", "both"} else "opt"
    if primary == "ratio":
        predictions = ratio_predictions or []
        selected_names = ratio_names
        rec_token_mean = ratio_result["rec_token_mean"]
    else:
        predictions = opt_predictions or []
        selected_names = opt_names
        rec_token_mean = 0.0 if opt_result is None else opt_result["rec_token_mean"]

    return {
        "predictions": predictions,
        "references": references,
        "selected_names": selected_names,
        "primary": primary,
        "rec_token_mean": rec_token_mean,
        "opt": opt_result,
        "ratio": ratio_result,
        "ratio_status": ratio_status,
    }


def _build_ratio_public_stats(args, model_wrapper, dataset_cls):
    if args.peftleak_attack_mode not in {"ratio", "both"}:
        return None, "not_requested", 0
    if args.peft_method != "adapter":
        return None, f"unsupported_for_{args.peft_method}", 0
    if args.peftleak_ratio_route == "oracle":
        return None, "oracle_debug_no_public_stats", 0

    public_n = int(args.peftleak_ratio_public_n_inputs)
    if public_n <= 0:
        raise ValueError("--peftleak_ratio_public_n_inputs must be positive for public_bins routing.")

    public_split = "test" if args.split != "test" else "val"
    np_state = np.random.get_state()
    try:
        public_dataset = dataset_cls(
            args.device,
            args.dataset,
            public_split,
            public_n,
            1,
            args.cache_dir,
        )
    finally:
        np.random.set_state(np_state)

    public_batches = []
    tokenizer = model_wrapper.tokenizer
    for idx in range(public_n):
        sequences, _labels = public_dataset[idx]
        batch = tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=min(tokenizer.model_max_length, model_wrapper.emb_size - 20),
            return_tensors="pt",
        ).to(args.device)
        public_batches.append(batch)

    stats = build_text_token_statistics(
        model_wrapper,
        public_batches,
        num_bins=args.peftleak_ratio_bins,
        target=args.peftleak_ratio_target,
    )
    return stats, f"{public_split}_public_split", public_n


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
    print(f"[peftleak] Using device: {args.device} | gradient device: {args.device_grad}", flush=True)
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
        record_dataset_protocol(args, dataset)
        model_wrapper = ModelWrapper(args)
        ratio_stats, public_stats_source, public_stats_n_inputs = _build_ratio_public_stats(args, model_wrapper, TextDataset)
        tracker["public_stats_source"] = public_stats_source
        tracker["public_stats_n_inputs"] = public_stats_n_inputs
        predictions = []
        references = []
        for input_idx in range(args.start_input, min(args.n_inputs, args.end_input)):
            input_start = time.time()
            args.defense_rng_step = tracker["n_inputs_completed"]
            sample = dataset[input_idx]
            print(f"[peftleak] Running input #{input_idx} of {args.n_inputs}.", flush=True)
            attack_result = reconstruct(args, sample, model_wrapper, ratio_stats=ratio_stats)
            pred = attack_result["predictions"]
            ref = attack_result["references"]
            selected_names = attack_result["selected_names"]
            predictions.extend(pred)
            references.extend(ref)
            tracker["rec_token_values"].append(float(attack_result["rec_token_mean"]))
            tracker["ratio_status"] = attack_result.get("ratio_status", tracker.get("ratio_status"))
            opt_result = attack_result.get("opt")
            ratio_result = attack_result.get("ratio")
            if opt_result is not None:
                tracker["opt_rec_token_values"].append(float(opt_result["rec_token_mean"]))
                tracker["optimization_loss_values"].append(float(opt_result["loss"]))
                tracker["optimization_loss_first"] = opt_result.get("initial_loss")
                tracker["optimization_loss_last"] = opt_result.get("loss")
                tracker["optimization_loss_reduction"] = opt_result.get("loss_reduction")
                tracker["sequence_length"] = opt_result.get("sequence_length")
            if ratio_result is not None:
                tracker["ratio_rec_token_values"].append(float(ratio_result["rec_token_mean"]))
                tracker["ratio_loss_values"].append(float(ratio_result["loss"]))
                tracker["ratio_collision_values"].append(float(ratio_result.get("collision_rate", 0.0)))
                tracker["ratio_recovered_hidden_values"].append(float(ratio_result.get("recovered_hidden_count", 0)))
                tracker["ratio_routed_token_values"].append(float(ratio_result.get("routed_token_count", 0)))
                tracker["ratio_slot_count_values"].append(float(ratio_result.get("slot_count", 0)))
                tracker["ratio_reportable"] = bool(ratio_result.get("reportable", False))
                if not tracker["ratio_reportable"]:
                    tracker["ratio_non_reportable_reason"] = f"{args.peftleak_ratio_route}_routing_debug_only"
                tracker["sequence_length"] = len(ratio_result["predicted_ids"][0]) if ratio_result.get("predicted_ids") else None
            if opt_result is not None:
                selected_count = opt_result.get("selected_gradient_count", len(selected_names))
            elif ratio_result is not None:
                selected_count = ratio_result.get("gradient_count", len(selected_names))
            else:
                selected_count = len(selected_names)
            tracker["selected_adapter_gradient_count"] = selected_count
            tracker["selected_adapter_gradient_names"] = ";".join(selected_names)
            tracker["last_rec_status"] = "ok"
            tracker["n_inputs_completed"] += 1
            tracker["last_input_idx"] = input_idx
            tracker["last_input_time"] = str(datetime.timedelta(seconds=time.time() - input_start)).split(".")[0]
            tracker["last_total_time"] = str(datetime.timedelta(seconds=time.time() - start_time)).split(".")[0]
            print(f"[peftleak] selected adapter gradients ({len(selected_names)}): {', '.join(selected_names[:6])}", flush=True)
            primary = attack_result.get("primary")
            initial_loss = None if opt_result is None else opt_result.get("initial_loss")
            loss_reduction = None if opt_result is None else opt_result.get("loss_reduction")
            print(
                f"[peftleak] primary={primary} "
                f"initial_loss={initial_loss if initial_loss is not None else 'n/a'} "
                f"loss_reduction={loss_reduction if loss_reduction is not None else 'n/a'} "
                f"rec_token={attack_result['rec_token_mean']:.6f} "
                f"ratio_status={tracker['ratio_status']}",
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
