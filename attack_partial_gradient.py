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
    is_ptg_word_embedding_param,
    optimize_partial_text_embeddings,
    parse_ptg_attack_layers,
    ptg_filter_active,
    ptg_selector_summary_fields,
    selected_partial_gradient_tensors,
    validate_ptg_selector_args,
)
from attacks.peftleak_text import get_token_embedding_matrix, summarize_token_predictions, token_recovery_ratio
from attack_peftleak import compute_text_metrics, validate_text_metric_backend
from utils.defense_common import add_shared_defense_args, defense_param_spec, fmt_summary_value, safe_mean
from utils.data import ATTACK_SPLIT_CHOICES, dataset_summary_fields, record_dataset_protocol
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
    use_effective_batch_size,
)
from utils.gpu import resolve_cuda_device, resolve_gradient_device
from utils.lrb_presets import apply_lrb_preset
from utils.lrb_defense import lrb_seed_summary_fields
from utils.representation_bottleneck import rep_bottleneck_summary_fields, validate_rep_bottleneck_args


SUMMARY_START = "===== RESULT SUMMARY START ====="
SUMMARY_END = "===== RESULT SUMMARY END ====="
SOURCE_PARITY_MODEL_PATH = "bert-base-uncased"
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
    parser.add_argument(
        "--use_embedding",
        "--ptg_use_embedding",
        dest="ptg_use_embedding",
        action="store_true",
        default=False,
        help="Infer unused tokens from visible word-embedding gradient rows, matching the official source option.",
    )
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
    parser.add_argument(
        "--ptg_rouge_backend",
        type=str,
        default="datasets",
        choices=["datasets", "simple_ngram"],
        help="ROUGE implementation for PTG summaries. PTG validates this backend before the attack and never falls back automatically.",
    )

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
        "ptg_use_embedding_status": None,
        "ptg_unused_token_count_values": [],
        "ptg_unused_token_count_last": None,
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


def _tokenizer_vocab_size(tokenizer) -> int | None:
    raw = getattr(tokenizer, "vocab_size", None)
    if raw is None:
        try:
            raw = len(tokenizer)
        except Exception:
            raw = None
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if value > 0 else None


def _token_embedding_row_count(tokenizer, model_wrapper) -> int | None:
    try:
        raw = int(get_token_embedding_matrix(model_wrapper).shape[0])
    except Exception:
        return _tokenizer_vocab_size(tokenizer)
    return raw if raw > 0 else _tokenizer_vocab_size(tokenizer)


def _same_parameter(left, right) -> bool:
    if left is right:
        return True
    try:
        return torch.is_tensor(left) and torch.is_tensor(right) and left.data_ptr() == right.data_ptr()
    except Exception:
        return False


def _input_embedding_parameter_names(model_wrapper) -> set[str]:
    names: set[str] = set()
    modules = []
    for attr in ("model", "base_model"):
        module = getattr(model_wrapper, attr, None)
        if module is not None and all(module is not existing for existing in modules):
            modules.append(module)
    for module in modules:
        get_embeddings = getattr(module, "get_input_embeddings", None)
        named_parameters = getattr(module, "named_parameters", None)
        if not callable(get_embeddings) or not callable(named_parameters):
            continue
        try:
            embedding = get_embeddings()
        except Exception:
            continue
        weight = getattr(embedding, "weight", None)
        if weight is None:
            continue
        try:
            for name, param in named_parameters():
                if _same_parameter(param, weight):
                    names.add(str(name))
        except Exception:
            continue
    return names


def _matches_input_embedding_name(name: str, input_embedding_names: set[str]) -> bool:
    if not input_embedding_names:
        return False
    name = str(name)
    for known in input_embedding_names:
        known = str(known)
        if name == known or name.endswith(f".{known}") or known.endswith(f".{name}"):
            return True
    return False


def _valid_token_id_count(token_ids) -> int:
    valid = set()
    for token_id in token_ids or []:
        if token_id is None:
            continue
        try:
            valid.add(int(token_id))
        except (TypeError, ValueError, OverflowError):
            continue
    return len(valid)


def _dynamic_unused_token_ids_from_embedding_gradient(tokenizer, model_wrapper, partial_grads, parameter_names):
    token_row_count = _token_embedding_row_count(tokenizer, model_wrapper)
    input_embedding_names = _input_embedding_parameter_names(model_wrapper)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    try:
        pad_token_id = None if pad_token_id is None else int(pad_token_id)
    except (TypeError, ValueError, OverflowError):
        pad_token_id = None

    for idx, grad in enumerate(partial_grads):
        if grad is None:
            continue
        name = parameter_names[idx] if idx < len(parameter_names) else f"param_{idx}"
        if not (_matches_input_embedding_name(name, input_embedding_names) or is_ptg_word_embedding_param(name)):
            continue
        if not torch.is_tensor(grad) or grad.ndim != 2:
            continue
        row_count = int(grad.shape[0])
        if token_row_count is not None:
            if row_count < token_row_count:
                continue
            row_count = int(token_row_count)
        if row_count <= 0:
            continue
        row_sums = grad.detach()[:row_count].float().abs().reshape(row_count, -1).sum(dim=1)
        zero_rows = torch.nonzero(row_sums < 1e-9, as_tuple=False).view(-1).detach().cpu().tolist()
        return {int(token_id) for token_id in zero_rows if pad_token_id is None or int(token_id) != pad_token_id}
    return None


def _record_ptg_use_embedding(args, status: str, token_ids) -> None:
    setattr(args, "ptg_use_embedding_status", status)
    setattr(args, "ptg_unused_token_count", _valid_token_id_count(token_ids))


def _resolve_ignored_token_ids(args, tokenizer, model_wrapper, partial_grads, parameter_names):
    if getattr(args, "ptg_use_embedding", False):
        dynamic_ids = _dynamic_unused_token_ids_from_embedding_gradient(
            tokenizer,
            model_wrapper,
            partial_grads,
            parameter_names,
        )
        if dynamic_ids is not None:
            _record_ptg_use_embedding(args, "dynamic", dynamic_ids)
            return dynamic_ids
        fallback_ids = _ignored_token_ids(tokenizer, model_wrapper)
        _record_ptg_use_embedding(args, "fallback_no_embedding_grad", fallback_ids)
        return fallback_ids

    fixed_ids = _ignored_token_ids(tokenizer, model_wrapper)
    _record_ptg_use_embedding(args, "fixed", fixed_ids)
    return fixed_ids


def _source_bert_parity_active(args) -> bool:
    return args.ptg_parity_mode == "source" and args.model_path == SOURCE_PARITY_MODEL_PATH


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


def _prediction_match_order(predictions: list[str], references: list[str], mode: str) -> list[int]:
    if mode == "none" or len(predictions) <= 1 or len(predictions) != len(references):
        return list(range(len(predictions)))
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
    return order


def _match_predictions_to_references(predictions: list[str], references: list[str], mode: str) -> list[str]:
    return [predictions[idx] for idx in _prediction_match_order(predictions, references, mode)]


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
            # Keep the PTG entry point partial by default. all_layers remains
            # available as an explicit full-gradient source control.
            args.grad_type = "layer_encoder"
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
    unused_count_values = tracker.get("ptg_unused_token_count_values", [])
    unused_count = (
        safe_mean(unused_count_values)
        if unused_count_values
        else getattr(args, "ptg_unused_token_count", None)
    )
    fields = [
        ("summary_version", tracker.get("summary_version", 2)),
        ("result_status", tracker.get("result_status", "ok")),
        ("attack", tracker.get("attack")),
        ("partial_attack_variant", tracker.get("partial_attack_variant")),
        ("dataset", args.dataset),
        ("split", args.split),
        *dataset_summary_fields(args),
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
        *lrb_seed_summary_fields(args),
        *dpsgd_opacus_summary_fields(args, tracker),
        *rep_bottleneck_summary_fields(args),
        *ptg_selector_summary_fields(args),
        ("ptg_exposure_scope", "partial" if ptg_filter_active(args) else "full_control"),
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
        ("ptg_use_embedding", getattr(args, "ptg_use_embedding", False)),
        (
            "ptg_use_embedding_status",
            tracker.get("ptg_use_embedding_status") or getattr(args, "ptg_use_embedding_status", "not_evaluated"),
        ),
        ("ptg_unused_token_count", unused_count),
        (
            "ptg_unused_token_count_last",
            tracker.get("ptg_unused_token_count_last", getattr(args, "ptg_unused_token_count", None)),
        ),
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
        ("ptg_rouge_backend_requested", args.ptg_rouge_backend),
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
    args.ptg_use_embedding_status = "not_evaluated"
    args.ptg_unused_token_count = None
    _normalize_ptg_source_args(args)
    if args.ptg_parity_mode == "source":
        if args.model_path != SOURCE_PARITY_MODEL_PATH:
            raise ValueError(
                "--ptg_parity_mode source requires --model_path bert-base-uncased; "
                "use --ptg_parity_mode fedllm for GPT-2, Llama, or other model families."
            )
        if float(args.ptg_lm_prior_weight) != 0.0 and args.ptg_lm_model_path is None:
            raise ValueError(
                "--ptg_parity_mode source with a nonzero LM prior requires --ptg_lm_model_path "
                "so source-style swap scoring has an LM objective."
            )
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
    ignored_token_ids = _resolve_ignored_token_ids(
        args,
        tokenizer,
        model_wrapper,
        partial_grads,
        parameter_names,
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
    match_order = _prediction_match_order(predictions, references, match_mode)
    if match_order != list(range(len(predictions))):
        attack_result["predicted_ids"] = [attack_result["predicted_ids"][idx] for idx in match_order]
        attack_result["predicted_scores"] = [attack_result["predicted_scores"][idx] for idx in match_order]
    attack_result["rec_token_mean"] = token_recovery_ratio(
        attack_result["predicted_ids"],
        batch["input_ids"].detach().cpu().tolist(),
        ignored_token_ids=ignored_token_ids,
        reference_mask=reference_mask,
    )
    attack_result["batch_match_mode"] = match_mode
    predictions = [predictions[idx] for idx in match_order]
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
    use_embedding_status = getattr(args, "ptg_use_embedding_status", None)
    if use_embedding_status is not None:
        previous = tracker.get("ptg_use_embedding_status")
        if previous in (None, use_embedding_status):
            tracker["ptg_use_embedding_status"] = use_embedding_status
        elif str(use_embedding_status) not in str(previous).split(";"):
            tracker["ptg_use_embedding_status"] = f"{previous};{use_embedding_status}"
    unused_token_count = getattr(args, "ptg_unused_token_count", None)
    if unused_token_count is not None:
        tracker["ptg_unused_token_count_values"].append(float(unused_token_count))
        tracker["ptg_unused_token_count_last"] = int(unused_token_count)
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
    frozen_train = _freeze_source_position_embeddings(train_wrapper.model)
    frozen_attack = _freeze_source_position_embeddings(attack_wrapper.model)
    if frozen_train or frozen_attack:
        print(
            f"[ptg:dpsgd] Frozen position embeddings before Opacus "
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

        should_reconstruct = recover_idx >= args.start_input
        if should_reconstruct:
            input_start = time.time()
            args.defense_rng_step = tracker["n_inputs_completed"]
            decoded = train_wrapper.tokenizer.batch_decode(
                batch["input_ids"].detach().cpu().tolist(),
                skip_special_tokens=True,
            )
            sample = (decoded, labels)
            print(f"[ptg:dpsgd] Running input #{recover_idx} of {args.n_inputs}.", flush=True)
            _sync_private_model_to_attack_model(private_model, attack_wrapper.model)

        optimizer.step()
        if should_reconstruct:
            partial = _capture_source_opacus_grads(args, private_model)
        optimizer.zero_grad(set_to_none=True)

        if should_reconstruct:
            attack_wrapper.model.eval()
            actual_batch_size = len(sample[0])
            print(f"[ptg:dpsgd] actual_batch_size={actual_batch_size}", flush=True)
            with use_effective_batch_size(args, actual_batch_size):
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

        recover_idx += 1

    tracker["aggregate_metrics"] = compute_text_metrics(
        predictions,
        references,
        backend=args.ptg_rouge_backend,
    )
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
        validate_text_metric_backend(args.ptg_rouge_backend)
        print(f"[ptg] ROUGE backend: {args.ptg_rouge_backend}", flush=True)
        lm_model, lm_tokenizer = _load_ptg_lm(args)
        if args.defense == DPSGD_OPACUS_DEFENSE or (
            args.defense == "dpsgd" and args.ptg_dpsgd_mode == "source_opacus"
        ):
            _run_source_opacus_dpsgd(args, tracker, start_time, lm_model=lm_model, lm_tokenizer=lm_tokenizer)
            _emit_result_summary(args, tracker)
            return 0

        dataset = TextDataset(args.device, args.dataset, args.split, args.n_inputs, args.batch_size, args.cache_dir)
        record_dataset_protocol(args, dataset)
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

        tracker["aggregate_metrics"] = compute_text_metrics(
            predictions,
            references,
            backend=args.ptg_rouge_backend,
        )
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
