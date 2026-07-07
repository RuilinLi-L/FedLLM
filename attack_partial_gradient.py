from __future__ import annotations

import argparse
import datetime
import itertools
import time
from collections import Counter

import numpy as np
import torch

from attacks.partial_transformer_gradients import (
    PARTIAL_TRANSFORMER_GRADIENTS_ATTACK,
    PTG_GRADIENT_MATCHING_VARIANT,
    PTG_INIT_STRATEGIES,
    PTG_MATCH_LOSSES,
    PTG_OPTIMIZERS,
    PTG_PARAM_FILTERS,
    filter_partial_transformer_gradients,
    optimize_partial_text_embeddings,
    parse_ptg_attack_layers,
    ptg_selector_summary_fields,
    selected_partial_gradient_tensors,
    validate_ptg_selector_args,
)
from attacks.peftleak_text import summarize_token_predictions
from attack_peftleak import compute_text_metrics
from utils.defense_common import add_shared_defense_args, defense_param_spec, fmt_summary_value, safe_mean
from utils.defenses import apply_defense, requires_gradient_generation_defense
from utils.dpsgd_opacus import (
    DPSGD_OPACUS_DEFENSE,
    capture_private_grads,
    create_source_dpsgd_dataloader,
    dpsgd_opacus_summary_fields,
    freeze_position_embeddings,
    is_empty_opacus_batch,
    make_private_with_opacus,
    normalize_dpsgd_opacus_args,
    record_dpsgd_opacus_summary,
    sync_private_model_to_public_model,
)
from utils.gpu import resolve_cuda_device, resolve_gradient_device
from utils.lrb_presets import apply_lrb_preset
from utils.representation_bottleneck import rep_bottleneck_summary_fields, validate_rep_bottleneck_args


SUMMARY_START = "===== RESULT SUMMARY START ====="
SUMMARY_END = "===== RESULT SUMMARY END ====="
SUPPORTED_PTG_DEFENSES = {
    "none",
    "noise",
    "dpsgd",
    DPSGD_OPACUS_DEFENSE,
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
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "mse", "cos", "dlg", "tag"])
    parser.add_argument("--train_method", type=str, default="full", choices=["full"])

    parser.add_argument("--baseline", action="store_true", default=False)
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--init_candidates", type=int, default=None)
    parser.add_argument("--init", type=str, default=None, choices=["lm", "random"])
    parser.add_argument("--use_swaps", type=bool, default=None)
    parser.add_argument("--no-use_swaps", dest="use_swaps", action="store_false")
    parser.add_argument("--use_swaps_at_end", action="store_true", default=False)
    parser.add_argument("--swap_burnin", type=float, default=None)
    parser.add_argument("--swap_every", type=int, default=None)
    parser.add_argument("--init_size", type=float, default=None)
    parser.add_argument("--opt_alg", type=str, default=None, choices=sorted(PTG_OPTIMIZERS))
    parser.add_argument("--coeff_perplexity", type=float, default=None)
    parser.add_argument("--coeff_reg", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr_decay_type", type=str, default=None, choices=["none", "StepLR", "LambdaLR"])
    parser.add_argument("--lr_decay", type=float, default=None)
    parser.add_argument("--tag_factor", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--lr_max_it", type=int, default=None)
    parser.add_argument("--print_every", type=int, default=None)

    parser.add_argument("--ptg_parity_mode", type=str, default="fedllm", choices=["fedllm", "source"])
    parser.add_argument(
        "--ptg_dpsgd_mode",
        type=str,
        default="dpsgd_style",
        choices=["dpsgd_style", "source_opacus"],
        help="dpsgd_style uses FedLLM per-example clipping/noise; source_opacus follows the official source DP-SGD loop.",
    )
    parser.add_argument("--noise_multiplier", type=float, default=None)
    parser.add_argument("--clipping_bound", type=float, default=None)

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
    parser.add_argument(
        "--grad_type",
        type=str,
        default=None,
        help="Official-source PTG selector alias: all_layers, encoder, layer_encoder, attn_qkv, attn_query, ...",
    )
    parser.add_argument(
        "--attack_layer",
        type=str,
        default=None,
        help="Official-source layer list, e.g. all or 0,1,2. Used with --grad_type.",
    )

    parser.add_argument("--ptg_steps", type=int, default=None)
    parser.add_argument("--ptg_lr", type=float, default=None)
    parser.add_argument("--ptg_restarts", type=int, default=1)
    parser.add_argument("--ptg_match_loss", type=str, default=None, choices=sorted(PTG_MATCH_LOSSES))
    parser.add_argument("--ptg_label_mode", type=str, default="known", choices=["known", "search"])
    parser.add_argument("--ptg_label_candidates", type=str, default=None)
    parser.add_argument("--ptg_decode_metric", type=str, default="cos", choices=["cos", "l2"])
    parser.add_argument("--ptg_tv_weight", type=float, default=0.0)
    parser.add_argument("--ptg_embed_norm_weight", type=float, default=None)
    parser.add_argument("--ptg_entropy_weight", type=float, default=0.0)
    parser.add_argument("--ptg_fix_special_tokens", action="store_true", default=True)
    parser.add_argument("--no_ptg_fix_special_tokens", action="store_false", dest="ptg_fix_special_tokens")
    parser.add_argument("--ptg_know_padding", action="store_true", default=True)
    parser.add_argument("--no_ptg_know_padding", action="store_false", dest="ptg_know_padding")
    parser.add_argument("--ptg_lm_prior_weight", type=float, default=None)
    parser.add_argument("--ptg_lm_model_path", type=str, default=None)
    parser.add_argument("--ptg_lm_tokenizer_path", type=str, default=None)
    parser.add_argument("--ptg_lm_init_prompt", type=str, default="the")
    parser.add_argument("--ptg_swap_steps", type=int, default=0)
    parser.add_argument("--ptg_use_swaps", action="store_true", default=None)
    parser.add_argument("--no_ptg_use_swaps", action="store_false", dest="ptg_use_swaps")
    parser.add_argument("--ptg_swap_burnin", type=float, default=0.1)
    parser.add_argument("--ptg_swap_every", type=int, default=75)
    parser.add_argument("--ptg_use_swaps_at_end", action="store_true", default=False)
    parser.add_argument("--ptg_init", type=str, default=None, choices=sorted(PTG_INIT_STRATEGIES))
    parser.add_argument("--ptg_init_candidates", type=int, default=None)
    parser.add_argument("--ptg_init_size", type=float, default=None)
    parser.add_argument("--ptg_init_permutation_trials", type=int, default=0)
    parser.add_argument("--ptg_optimizer", type=str, default="adam", choices=sorted(PTG_OPTIMIZERS))
    parser.add_argument("--ptg_lr_decay_type", type=str, default=None, choices=["none", "StepLR", "LambdaLR"])
    parser.add_argument("--ptg_lr_decay", type=float, default=0.9)
    parser.add_argument("--ptg_lr_max_it", type=int, default=None)
    parser.add_argument("--ptg_tag_factor", type=float, default=None)
    parser.add_argument("--ptg_grad_clip", type=float, default=None)
    parser.add_argument("--ptg_print_every", type=int, default=None)
    parser.add_argument("--ptg_batch_match", type=str, default="auto", choices=["auto", "rouge1", "none"])

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
        "ptg_lm_loss_values": [],
        "aggregate_metrics": {},
        "selected_gradient_count": None,
        "selected_gradient_names": None,
        "fixed_token_values": [],
        "sequence_length": None,
        "error_type": None,
        "error_message": None,
    }


def _ignored_token_ids(tokenizer, model_wrapper) -> set[int | None]:
    args = getattr(model_wrapper, "args", None)
    if getattr(args, "ptg_parity_mode", None) == "source" and getattr(args, "model_path", None) == "bert-base-uncased":
        vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
        upper = min(999, vocab_size)
        return set(range(1, min(100, vocab_size))) | set(range(104, upper))
    return {
        getattr(model_wrapper, "pad_token", None),
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "bos_token_id", None),
        getattr(tokenizer, "cls_token_id", None),
        getattr(tokenizer, "sep_token_id", None),
    }


def _source_bert_parity_active(args) -> bool:
    return args.ptg_parity_mode == "source" and args.model_path == "bert-base-uncased"


def _source_decode_bert_ids(tokenizer, ids) -> str:
    if torch.is_tensor(ids):
        ids = ids.detach().cpu().tolist()
    ids = [int(token_id) for token_id in ids]
    sep_id = getattr(tokenizer, "sep_token_id", None)
    if sep_id is not None:
        for idx in range(len(ids) - 1, -1, -1):
            if ids[idx] == int(sep_id):
                ids = ids[: idx + 1]
                break
    return tokenizer.decode(ids, skip_special_tokens=False)


def _decode_reference_ids(args, tokenizer, input_ids) -> list[str]:
    rows = input_ids.detach().cpu().tolist() if torch.is_tensor(input_ids) else input_ids
    if _source_bert_parity_active(args):
        return [_source_decode_bert_ids(tokenizer, row) for row in rows]
    return tokenizer.batch_decode(rows, skip_special_tokens=True)


def _decode_prediction_ids(args, tokenizer, predicted_ids) -> list[str]:
    if _source_bert_parity_active(args):
        return [_source_decode_bert_ids(tokenizer, row) for row in predicted_ids]
    return summarize_token_predictions(predicted_ids, tokenizer)


def _compute_source_bert_grads(args, model_wrapper, batch, labels):
    batch, labels, dev = model_wrapper._prepare_grad_context(batch, labels)
    try:
        prepared_labels = model_wrapper._prepare_task_labels(batch, labels)
        model_wrapper.model.zero_grad(set_to_none=True)
        embeddings = model_wrapper.model.get_input_embeddings()(batch["input_ids"])
        outputs = model_wrapper.model(inputs_embeds=embeddings, labels=prepared_labels.view(-1).long())
        return torch.autograd.grad(
            outputs.loss,
            model_wrapper.trainable_parameters(),
            create_graph=False,
            allow_unused=True,
        )
    finally:
        model_wrapper._restore_grad_context(batch, labels, dev)


def _rouge1_fmeasure(prediction: str, reference: str) -> float:
    pred = Counter(str(prediction).split())
    ref = Counter(str(reference).split())
    if not pred and not ref:
        return 1.0
    overlap = sum((pred & ref).values())
    precision = overlap / max(1, sum(pred.values()))
    recall = overlap / max(1, sum(ref.values()))
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _match_predictions_to_references(predictions: list[str], references: list[str], mode: str) -> list[str]:
    if mode == "none" or len(predictions) <= 1 or len(predictions) != len(references):
        return predictions
    n_items = len(predictions)
    cost = [
        [1.0 - _rouge1_fmeasure(predictions[i], references[j]) for j in range(n_items)]
        for i in range(n_items)
    ]
    try:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(np.array(cost))
        assignment = {int(row): int(col) for row, col in zip(row_ind, col_ind)}
        order = sorted(range(n_items), key=lambda row: assignment[row])
    except Exception:
        if n_items <= 8:
            best_perm = min(
                itertools.permutations(range(n_items)),
                key=lambda perm: sum(cost[row][perm[row]] for row in range(n_items)),
            )
            order = sorted(range(n_items), key=lambda row: best_perm[row])
        else:
            remaining = set(range(n_items))
            order = []
            for ref_idx in range(n_items):
                row = min(remaining, key=lambda pred_idx: cost[pred_idx][ref_idx])
                order.append(row)
                remaining.remove(row)
    return [predictions[idx] for idx in order]


def _label_candidates(raw: str | None):
    if raw is None:
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _normalize_ptg_source_args(args):
    raw_attack_layer = args.attack_layer
    args.attack_layer = parse_ptg_attack_layers(args.attack_layer)
    if args.loss in {"cos", "dlg", "tag"}:
        args.ptg_match_loss = args.loss
        args.loss = "ce"
    if args.n_steps is not None:
        args.ptg_steps = int(args.n_steps)
    if args.lr is not None:
        args.ptg_lr = float(args.lr)
    if args.init_candidates is not None:
        args.ptg_init_candidates = int(args.init_candidates)
    if args.init is not None:
        args.ptg_init = args.init
    if args.use_swaps is not None:
        args.ptg_use_swaps = bool(args.use_swaps)
    if args.use_swaps_at_end:
        args.ptg_use_swaps_at_end = True
    if args.swap_burnin is not None:
        args.ptg_swap_burnin = float(args.swap_burnin)
    if args.swap_every is not None:
        args.ptg_swap_every = int(args.swap_every)
    if args.init_size is not None:
        args.ptg_init_size = float(args.init_size)
    if args.opt_alg is not None:
        args.ptg_optimizer = args.opt_alg
    if args.coeff_perplexity is not None:
        args.ptg_lm_prior_weight = float(args.coeff_perplexity)
    if args.coeff_reg is not None:
        args.ptg_embed_norm_weight = float(args.coeff_reg)
    if args.lr_decay_type is not None:
        args.ptg_lr_decay_type = args.lr_decay_type
    if args.lr_decay is not None:
        args.ptg_lr_decay = float(args.lr_decay)
    if args.tag_factor is not None:
        args.ptg_tag_factor = float(args.tag_factor)
    if args.grad_clip is not None:
        args.ptg_grad_clip = float(args.grad_clip)
    if args.lr_max_it is not None:
        args.ptg_lr_max_it = int(args.lr_max_it)
    if args.print_every is not None:
        args.ptg_print_every = int(args.print_every)

    if args.ptg_parity_mode == "source":
        if args.grad_type is None:
            args.grad_type = "all_layers"
        if raw_attack_layer is None:
            args.attack_layer = [0]
        args.ptg_steps = 2000 if args.ptg_steps is None else int(args.ptg_steps)
        args.ptg_lr = 0.01 if args.ptg_lr is None else float(args.ptg_lr)
        args.ptg_match_loss = "cos" if args.ptg_match_loss is None else args.ptg_match_loss
        args.ptg_init = "random" if args.ptg_init is None else args.ptg_init
        args.ptg_init_candidates = 500 if args.ptg_init_candidates is None else int(args.ptg_init_candidates)
        args.ptg_init_size = 1.4 if args.ptg_init_size is None else float(args.ptg_init_size)
        args.ptg_lr_decay_type = "StepLR" if args.ptg_lr_decay_type is None else args.ptg_lr_decay_type
        args.ptg_embed_norm_weight = 0.1 if args.ptg_embed_norm_weight is None else float(args.ptg_embed_norm_weight)
        args.ptg_lm_prior_weight = 0.1 if args.ptg_lm_prior_weight is None else float(args.ptg_lm_prior_weight)
        args.ptg_print_every = 50 if args.ptg_print_every is None else int(args.ptg_print_every)
        if args.ptg_use_swaps is None:
            args.ptg_use_swaps = True
        if args.ptg_use_swaps_at_end:
            args.ptg_use_swaps = False
        if args.ptg_use_swaps and int(args.ptg_swap_steps) <= 0:
            args.ptg_swap_steps = 200
    else:
        args.ptg_steps = 80 if args.ptg_steps is None else int(args.ptg_steps)
        args.ptg_lr = 0.1 if args.ptg_lr is None else float(args.ptg_lr)
        args.ptg_match_loss = "cosine" if args.ptg_match_loss is None else args.ptg_match_loss
        args.ptg_init = "prior" if args.ptg_init is None else args.ptg_init
        args.ptg_init_candidates = 1 if args.ptg_init_candidates is None else int(args.ptg_init_candidates)
        args.ptg_lr_decay_type = "none" if args.ptg_lr_decay_type is None else args.ptg_lr_decay_type
        args.ptg_embed_norm_weight = 0.0 if args.ptg_embed_norm_weight is None else float(args.ptg_embed_norm_weight)
        args.ptg_lm_prior_weight = 0.0 if args.ptg_lm_prior_weight is None else float(args.ptg_lm_prior_weight)
        args.ptg_print_every = 0 if args.ptg_print_every is None else int(args.ptg_print_every)
        if args.ptg_use_swaps is None:
            args.ptg_use_swaps = False

    if args.baseline:
        args.ptg_init_candidates = 1
        args.ptg_use_swaps = False
        args.ptg_init_size = -1
        args.ptg_lm_prior_weight = 0.0
        args.ptg_embed_norm_weight = 0.0

    if args.defense == DPSGD_OPACUS_DEFENSE:
        args.ptg_dpsgd_mode = "source_opacus"
    if args.defense == "dpsgd" and args.ptg_dpsgd_mode == "source_opacus":
        normalize_dpsgd_opacus_args(args, active=True)
    elif args.defense == DPSGD_OPACUS_DEFENSE:
        normalize_dpsgd_opacus_args(args, active=True)
    elif args.defense == "dpsgd":
        if args.noise_multiplier is None and args.defense_noise is not None:
            args.noise_multiplier = float(args.defense_noise)
        if args.defense_noise is None and args.noise_multiplier is not None:
            args.defense_noise = float(args.noise_multiplier)
        if args.clipping_bound is None and args.defense_clip_norm is not None:
            args.clipping_bound = float(args.defense_clip_norm)
        if args.clipping_bound is not None:
            args.defense_clip_norm = float(args.clipping_bound)
    return args


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
        ("ptg_parity_mode", args.ptg_parity_mode),
        ("defense", args.defense),
        ("ptg_dpsgd_mode", args.ptg_dpsgd_mode),
        ("noise_multiplier", args.noise_multiplier),
        ("clipping_bound", args.clipping_bound),
        ("defense_param_name", defense_param_name),
        ("defense_param_value", defense_param_value),
        *dpsgd_opacus_summary_fields(args, tracker),
        *rep_bottleneck_summary_fields(args),
        *ptg_selector_summary_fields(args),
        ("ptg_steps", args.ptg_steps),
        ("ptg_lr", args.ptg_lr),
        ("ptg_restarts", args.ptg_restarts),
        ("ptg_match_loss", args.ptg_match_loss),
        ("ptg_optimizer", args.ptg_optimizer),
        ("ptg_lr_decay_type", args.ptg_lr_decay_type),
        ("ptg_lr_decay", args.ptg_lr_decay),
        ("ptg_lr_max_it", args.ptg_lr_max_it),
        ("ptg_tag_factor", args.ptg_tag_factor),
        ("ptg_grad_clip", args.ptg_grad_clip),
        ("ptg_label_mode", args.ptg_label_mode),
        ("ptg_decode_metric", args.ptg_decode_metric),
        ("ptg_tv_weight", args.ptg_tv_weight),
        ("ptg_embed_norm_weight", args.ptg_embed_norm_weight),
        ("ptg_entropy_weight", args.ptg_entropy_weight),
        ("ptg_fix_special_tokens", args.ptg_fix_special_tokens),
        ("ptg_know_padding", args.ptg_know_padding),
        ("ptg_init", args.ptg_init),
        ("ptg_init_candidates", args.ptg_init_candidates),
        ("ptg_init_size", args.ptg_init_size),
        ("ptg_init_permutation_trials", args.ptg_init_permutation_trials),
        ("ptg_lm_prior_weight", args.ptg_lm_prior_weight),
        ("ptg_lm_model_path", args.ptg_lm_model_path),
        ("ptg_swap_steps", args.ptg_swap_steps),
        ("ptg_use_swaps", args.ptg_use_swaps),
        ("ptg_swap_burnin", args.ptg_swap_burnin),
        ("ptg_swap_every", args.ptg_swap_every),
        ("ptg_use_swaps_at_end", args.ptg_use_swaps_at_end),
        ("ptg_batch_match", args.ptg_batch_match),
        ("ptg_print_every", args.ptg_print_every),
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
        ("ptg_lm_loss", safe_mean(tracker.get("ptg_lm_loss_values", []))),
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
    _normalize_ptg_source_args(args)
    if args.defense == "dager":
        raise NotImplementedError("DAGER defense is DAGER-specific and is excluded from PTG reproduction runs.")
    if args.defense not in SUPPORTED_PTG_DEFENSES:
        raise NotImplementedError(f"PTG supports defenses {sorted(SUPPORTED_PTG_DEFENSES)}; got {args.defense!r}.")
    if args.ptg_dpsgd_mode == "source_opacus" and args.defense not in {"dpsgd", DPSGD_OPACUS_DEFENSE}:
        raise ValueError("--ptg_dpsgd_mode source_opacus requires --defense dpsgd or --defense dpsgd_opacus.")
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
        if _source_bert_parity_active(args):
            true_grads = _compute_source_bert_grads(args, model_wrapper, batch, labels)
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
        source_grad_type=args.grad_type,
        source_attack_layers=args.attack_layer,
    )
    setattr(args, "ptg_gradient_info", info)
    selected_indices, selected_names = selected_partial_gradient_tensors(partial_grads, parameter_names)
    _mark_ptg_variant(args, selected_names)
    if not selected_indices:
        raise ValueError("No visible partial gradients after filtering/defense; cannot run PTG matching.")
    return partial_grads, parameter_names, selected_names


def _load_ptg_lm(args):
    if args.ptg_lm_model_path is None:
        return None, None
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_source = args.ptg_lm_tokenizer_path or args.ptg_lm_model_path
    tokenizer_kwargs = {"use_fast": True}
    model_kwargs = {}
    if args.cache_dir is not None:
        tokenizer_kwargs["cache_dir"] = args.cache_dir
        model_kwargs["cache_dir"] = args.cache_dir
    lm_tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    if getattr(lm_tokenizer, "pad_token_id", None) is None and getattr(lm_tokenizer, "eos_token", None) is not None:
        lm_tokenizer.pad_token = lm_tokenizer.eos_token
    lm_model = AutoModelForCausalLM.from_pretrained(args.ptg_lm_model_path, **model_kwargs)
    if getattr(lm_model.config, "pad_token_id", None) is None and getattr(lm_tokenizer, "pad_token_id", None) is not None:
        lm_model.config.pad_token_id = lm_tokenizer.pad_token_id
    lm_model.to(args.device)
    lm_model.eval()
    return lm_model, lm_tokenizer


def reconstruct(args, sample, model_wrapper, *, precomputed_partial=None, lm_model=None, lm_tokenizer=None):
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
    references = _decode_reference_ids(args, tokenizer, batch["input_ids"])
    ignored_token_ids = _ignored_token_ids(tokenizer, model_wrapper)

    if precomputed_partial is None:
        partial_grads, parameter_names, selected_names = _compute_defended_partial_grads(
            args,
            model_wrapper,
            batch,
            true_labels,
        )
    else:
        partial_grads, parameter_names, selected_names = precomputed_partial
        setattr(
            args,
            "ptg_gradient_info",
            getattr(args, "ptg_gradient_info", {}),
        )
        _mark_ptg_variant(args, selected_names)
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
        know_padding=args.ptg_know_padding,
        lm_prior_weight=args.ptg_lm_prior_weight,
        lm_model=lm_model,
        lm_tokenizer=lm_tokenizer,
        swap_steps=args.ptg_swap_steps,
        use_swaps=args.ptg_use_swaps,
        swap_burnin=args.ptg_swap_burnin,
        swap_every=args.ptg_swap_every,
        use_swaps_at_end=args.ptg_use_swaps_at_end,
        init_strategy=args.ptg_init,
        init_candidates=args.ptg_init_candidates,
        init_size=args.ptg_init_size,
        init_permutation_trials=args.ptg_init_permutation_trials,
        lm_init_prompt=args.ptg_lm_init_prompt,
        optimizer_name=args.ptg_optimizer,
        lr_decay_type=args.ptg_lr_decay_type,
        lr_decay=args.ptg_lr_decay,
        lr_max_it=args.ptg_lr_max_it,
        tag_factor=args.ptg_tag_factor,
        grad_clip=args.ptg_grad_clip,
        print_every=args.ptg_print_every,
        parity_mode=args.ptg_parity_mode,
        ignored_token_ids=ignored_token_ids,
        reference_mask=reference_mask,
    )
    predictions = _decode_prediction_ids(args, tokenizer, attack_result["predicted_ids"])
    match_mode = "rouge1" if args.ptg_batch_match == "auto" and len(predictions) > 1 else args.ptg_batch_match
    predictions = _match_predictions_to_references(predictions, references, match_mode)
    return predictions, references, attack_result, selected_names


def _record_attack_result(args, tracker, input_idx, input_start, start_time, attack_result, selected_names):
    tracker["rec_token_values"].append(float(attack_result["rec_token_mean"]))
    if attack_result.get("initial_loss") is not None:
        tracker["ptg_initial_loss_values"].append(float(attack_result["initial_loss"]))
    tracker["ptg_final_loss_values"].append(float(attack_result["loss"]))
    if attack_result.get("loss_reduction") is not None:
        tracker["ptg_loss_reduction_values"].append(float(attack_result["loss_reduction"]))
    if attack_result.get("lm_loss") is not None:
        tracker["ptg_lm_loss_values"].append(float(attack_result["lm_loss"]))
    tracker["fixed_token_values"].append(float(attack_result.get("fixed_token_count", 0)))
    tracker["selected_gradient_count"] = attack_result.get("selected_gradient_count", len(selected_names))
    tracker["selected_gradient_names"] = ";".join(selected_names)
    tracker["sequence_length"] = attack_result.get("sequence_length")
    tracker["last_rec_status"] = "ok"
    tracker["n_inputs_completed"] += 1
    tracker["last_input_idx"] = input_idx
    tracker["last_input_time"] = str(datetime.timedelta(seconds=time.time() - input_start)).split(".")[0]
    tracker["last_total_time"] = str(datetime.timedelta(seconds=time.time() - start_time)).split(".")[0]


def _create_source_dpsgd_dataloader(args, tokenizer):
    return create_source_dpsgd_dataloader(args, tokenizer, shuffle=False)


def _capture_source_opacus_grads(args, private_model):
    grads, names = capture_private_grads(private_model)
    partial_grads, info = filter_partial_transformer_gradients(
        tuple(grads),
        parameter_names=names,
        layer_subset=args.gradient_layer_subset,
        param_filter=args.gradient_param_filter,
        model_path=args.model_path,
        source_grad_type=args.grad_type,
        source_attack_layers=args.attack_layer,
    )
    setattr(args, "ptg_gradient_info", info)
    selected_indices, selected_names = selected_partial_gradient_tensors(partial_grads, names)
    _mark_ptg_variant(args, selected_names)
    if not selected_indices:
        raise ValueError("No visible partial gradients after source Opacus filtering; cannot run PTG matching.")
    return partial_grads, names, selected_names


def _sync_private_model_to_attack_model(private_model, attack_model):
    missing, unexpected = sync_private_model_to_public_model(private_model, attack_model)
    if missing or unexpected:
        print(
            f"[ptg:dpsgd] Loaded private state with missing={len(missing)} unexpected={len(unexpected)}.",
            flush=True,
        )


def _freeze_source_position_embeddings(model) -> int:
    return freeze_position_embeddings(model)


def _run_source_opacus_dpsgd(args, tracker, start_time, lm_model=None, lm_tokenizer=None):
    from utils.models import ModelWrapper

    normalize_dpsgd_opacus_args(args, active=True)
    attack_wrapper = ModelWrapper(args)
    train_wrapper = ModelWrapper(args)
    if _source_bert_parity_active(args):
        frozen_train = _freeze_source_position_embeddings(train_wrapper.model)
        frozen_attack = _freeze_source_position_embeddings(attack_wrapper.model)
        print(
            f"[ptg:dpsgd] Frozen BERT position embeddings before Opacus "
            f"(train={frozen_train}, attack={frozen_attack}).",
            flush=True,
        )
    train_wrapper.model.train()
    train_wrapper.set_model_device(args.device)
    train_loader = _create_source_dpsgd_dataloader(args, train_wrapper.tokenizer)
    optimizer = torch.optim.AdamW(
        (param for param in train_wrapper.model.parameters() if param.requires_grad),
        lr=5e-5,
    )
    privacy_engine, private_model, optimizer, train_loader = make_private_with_opacus(
        train_wrapper.model,
        optimizer,
        train_loader,
        args,
    )
    train_wrapper.model = private_model

    predictions = []
    references = []
    recover_idx = 0
    requested_stop = min(args.n_inputs, args.end_input)
    print("[ptg:dpsgd] Running source-style Opacus DP-SGD loop.", flush=True)
    for next_batch in train_loader:
        if recover_idx >= requested_stop:
            break
        if is_empty_opacus_batch(next_batch):
            continue
        batch = {key: value.to(args.device) for key, value in next_batch.items()}
        labels = batch.pop("labels").to(args.device)
        optimizer.zero_grad(set_to_none=True)
        outputs = private_model(**batch, labels=labels.view(-1).long())
        outputs.loss.backward()

        if recover_idx >= args.start_input:
            input_start = time.time()
            args.defense_rng_step = tracker["n_inputs_completed"]
            decoded = train_wrapper.tokenizer.batch_decode(
                batch["input_ids"].detach().cpu().tolist(),
                skip_special_tokens=True,
            )
            sample = (decoded, labels)
            print(f"[ptg:dpsgd] Running input #{recover_idx} of {args.n_inputs}.", flush=True)
            partial = _capture_source_opacus_grads(args, private_model)
            _sync_private_model_to_attack_model(private_model, attack_wrapper.model)
            attack_wrapper.model.eval()
            pred, ref, attack_result, selected_names = reconstruct(
                args,
                sample,
                attack_wrapper,
                precomputed_partial=partial,
                lm_model=lm_model,
                lm_tokenizer=lm_tokenizer,
            )
            predictions.extend(pred)
            references.extend(ref)
            _record_attack_result(args, tracker, recover_idx, input_start, start_time, attack_result, selected_names)
            print(
                f"[ptg:dpsgd] selected={tracker['selected_gradient_count']} "
                f"initial_loss={attack_result.get('initial_loss')} "
                f"final_loss={attack_result.get('loss')} "
                f"rec_token={attack_result['rec_token_mean']:.6f}",
                flush=True,
            )

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        recover_idx += 1

    tracker["aggregate_metrics"] = compute_text_metrics(predictions, references)
    if tracker["last_total_time"] is None:
        tracker["last_total_time"] = str(datetime.timedelta(seconds=time.time() - start_time)).split(".")[0]
    record_dpsgd_opacus_summary(args, tracker, privacy_engine)


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
        lm_model, lm_tokenizer = _load_ptg_lm(args)
        if args.defense == DPSGD_OPACUS_DEFENSE or (
            args.defense == "dpsgd" and args.ptg_dpsgd_mode == "source_opacus"
        ):
            _run_source_opacus_dpsgd(args, tracker, start_time, lm_model=lm_model, lm_tokenizer=lm_tokenizer)
            _emit_result_summary(args, tracker)
            return 0

        dataset = TextDataset(args.device, args.dataset, args.split, args.n_inputs, args.batch_size, args.cache_dir)
        model_wrapper = ModelWrapper(args)
        predictions = []
        references = []
        for input_idx in range(args.start_input, min(args.n_inputs, args.end_input)):
            input_start = time.time()
            args.defense_rng_step = tracker["n_inputs_completed"]
            print(f"[ptg] Running input #{input_idx} of {args.n_inputs}.", flush=True)
            pred, ref, attack_result, selected_names = reconstruct(
                args,
                dataset[input_idx],
                model_wrapper,
                lm_model=lm_model,
                lm_tokenizer=lm_tokenizer,
            )
            predictions.extend(pred)
            references.extend(ref)
            _record_attack_result(args, tracker, input_idx, input_start, start_time, attack_result, selected_names)
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
