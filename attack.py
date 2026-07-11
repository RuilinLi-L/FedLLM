import datetime
import numpy as np
import torch
from datasets import load_metric
from utils.models import ModelWrapper
from utils.data import TextDataset
from utils.filtering_encoder import filter_encoder
from utils.filtering_decoder import filter_decoder
from utils.functional import filter_outliers, remove_padding, resolve_dager_decomp_device
from utils.defenses import apply_defense, requires_gradient_generation_defense, uses_noisy_gradient_decoding
from utils.dpsgd_opacus import (
    DPSGD_OPACUS_DEFENSE,
    capture_private_grads,
    create_source_dpsgd_dataloader,
    decode_batch_texts,
    dpsgd_opacus_summary_fields,
    freeze_position_embeddings,
    is_empty_opacus_batch,
    make_private_with_opacus,
    normalize_dpsgd_opacus_args,
    record_dpsgd_opacus_summary,
    sync_private_model_to_public_model,
)
from utils.gpu import resolve_cuda_device, resolve_gradient_device
from utils.lrb_presets import lrb_preset_param_value
from utils.adaptive_attack import (
    adaptive_attack_summary_fields,
    adaptive_check_if_in_span,
    adaptive_get_span_dists,
    adaptive_get_top_B_in_span,
    prepare_adaptive_attack,
)
from utils.partial_gradient import (
    PARTIAL_ATTACK_DAGER_NONPREFIX,
    UnsupportedPartialGradientExposureError,
    apply_partial_gradient_filter,
    mark_partial_gradient_unsupported,
    nonprefix_candidate_cap,
    nonprefix_layer_indices,
    partial_gradient_summary_fields,
)
from utils.representation_bottleneck import rep_bottleneck_summary_fields
from utils.peft_utils import peft_active, peft_eval_scope
from args_factory import get_args
import time

from scipy.optimize import linear_sum_assignment

# old seed: 100
args = get_args()
args.device = resolve_cuda_device(args.device)
args.device_grad = resolve_gradient_device(args.device_grad, args.device)
args.dager_decomp_resolved_device = str(resolve_dager_decomp_device(args.dager_decomp_device, args.device))
print(
    f"[dager] Using device: {args.device} | gradient device: {args.device_grad} "
    f"| decomp device: {args.dager_decomp_resolved_device}",
    flush=True,
)
np.random.seed(args.rng_seed)
torch.manual_seed(args.rng_seed)

total_correct_tokens = 0
total_tokens = 0
total_correct_maxB_tokens = 0

SUMMARY_START = '===== RESULT SUMMARY START ====='
SUMMARY_END = '===== RESULT SUMMARY END ====='
SUMMARY_METRICS = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']


def _safe_mean(values):
    if not values:
        return None
    return float(np.mean(values))


def _safe_stat_values(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return arr


def _sequence_stats(values):
    arr = _safe_stat_values(values)
    if arr is None:
        return {
            'count': 0,
            'mean': None,
            'std': None,
            'var': None,
            'min': None,
            'p25': None,
            'median': None,
            'p75': None,
            'max': None,
        }
    return {
        'count': int(arr.size),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        'var': float(np.var(arr, ddof=1)) if arr.size > 1 else 0.0,
        'min': float(np.min(arr)),
        'p25': float(np.percentile(arr, 25)),
        'median': float(np.percentile(arr, 50)),
        'p75': float(np.percentile(arr, 75)),
        'max': float(np.max(arr)),
    }


def _fmt_summary_value(value):
    if value is None:
        return 'n/a'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, float):
        return f'{value:.6f}'
    return str(value)


def _init_result_tracker(args):
    requested = max(0, min(args.n_inputs, args.end_input) - args.start_input)
    return {
        'summary_emitted': False,
        'summary_version': 3,
        'result_status': 'ok',
        'n_inputs_requested': requested,
        'n_inputs_completed': 0,
        'last_input_idx': None,
        'last_input_time': None,
        'last_total_time': None,
        'last_rec_status': None,
        'rec_l1_mean_values': [],
        'rec_l1_maxb_mean_values': [],
        'rec_l2_mean_values': [],
        'rec_token_values': [],
        'rec_maxb_token_values': [],
        'input_metric_values': {},
        'aggregate_metrics': {},
        'error_type': None,
        'error_message': None,
    }


def _record_rec_metrics(args, rec_status, rec_l1, rec_l1_maxB, rec_l2, rec_maxb_token, rec_token):
    tracker = getattr(args, 'result_tracker', None)
    if tracker is None:
        return
    tracker['last_rec_status'] = rec_status
    tracker['rec_l1_mean_values'].append(float(np.mean(np.asarray(rec_l1, dtype=float))))
    tracker['rec_l1_maxb_mean_values'].append(float(np.mean(np.asarray(rec_l1_maxB, dtype=float))))
    tracker['rec_l2_mean_values'].append(float(np.mean(np.asarray(rec_l2, dtype=float))))
    tracker['rec_token_values'].append(float(rec_token))
    tracker['rec_maxb_token_values'].append(float(rec_maxb_token))


def _record_input_metric_summary(args, summary):
    tracker = getattr(args, 'result_tracker', None)
    if tracker is None:
        return
    values = tracker.setdefault('input_metric_values', {})
    for metric in SUMMARY_METRICS:
        for stat in ('fm', 'p', 'r'):
            source_key = f'agg_{metric}_{stat}'
            target_key = f'input_{metric}_{stat}'
            if source_key in summary:
                values.setdefault(target_key, []).append(float(summary[source_key]))
    if 'agg_r1fm_r2fm' in summary:
        values.setdefault('input_r1fm_r2fm', []).append(float(summary['agg_r1fm_r2fm']))


def _defense_param_spec(args):
    defense = getattr(args, 'defense', 'none')
    preset_value = lrb_preset_param_value(args)
    mapping = {
        'noise': ('defense_noise', getattr(args, 'defense_noise', None)),
        'dpsgd': ('defense_noise', getattr(args, 'defense_noise', None)),
        'dpsgd_opacus': ('defense_noise', getattr(args, 'defense_noise', None)),
        'topk': ('defense_topk_ratio', getattr(args, 'defense_topk_ratio', None)),
        'compression': ('defense_n_bits', getattr(args, 'defense_n_bits', None)),
        'soteria': (
            'defense_soteria_pruning_rate',
            getattr(args, 'defense_soteria_pruning_rate', None),
        ),
        'mixup': ('defense_mixup_alpha', getattr(args, 'defense_mixup_alpha', None)),
        'lrb': (
            'defense_lrb_preset' if preset_value is not None else 'defense_lrb_keep_ratio_sensitive',
            preset_value if preset_value is not None else getattr(args, 'defense_lrb_keep_ratio_sensitive', None),
        ),
        'lrbprojonly': (
            'defense_lrb_preset' if preset_value is not None else 'defense_lrb_keep_ratio_sensitive',
            preset_value if preset_value is not None else getattr(args, 'defense_lrb_keep_ratio_sensitive', None),
        ),
        'signed_bottleneck': (
            'defense_lrb_preset' if preset_value is not None else 'defense_lrb_keep_ratio_sensitive',
            preset_value if preset_value is not None else getattr(args, 'defense_lrb_keep_ratio_sensitive', None),
        ),
    }
    if defense == 'none':
        return 'n/a', 'n/a'
    return mapping.get(defense, ('n/a', 'n/a'))


def _metric_summary(res):
    summary = {}
    for metric in SUMMARY_METRICS:
        curr = res[metric].mid
        summary[f'agg_{metric}_fm'] = curr.fmeasure * 100
        summary[f'agg_{metric}_p'] = curr.precision * 100
        summary[f'agg_{metric}_r'] = curr.recall * 100
    summary['agg_r1fm_r2fm'] = (res['rouge1'].mid.fmeasure + res['rouge2'].mid.fmeasure) * 100
    return summary


def _all_sequence_stat_fields(args):
    tracker = getattr(args, 'result_tracker', None)
    if tracker is None:
        return []
    sequences = {
        'rec_l1': tracker.get('rec_l1_mean_values', []),
        'rec_l1_maxb': tracker.get('rec_l1_maxb_mean_values', []),
        'rec_l2': tracker.get('rec_l2_mean_values', []),
        'rec_token': tracker.get('rec_token_values', []),
        'rec_maxb_token': tracker.get('rec_maxb_token_values', []),
    }
    sequences.update(tracker.get('input_metric_values', {}))
    fields = []
    for name in sorted(sequences):
        stats = _sequence_stats(sequences[name])
        for stat_name in ('count', 'mean', 'std', 'var', 'min', 'p25', 'median', 'p75', 'max'):
            if name in ('rec_l1', 'rec_l1_maxb', 'rec_l2', 'rec_token', 'rec_maxb_token') and stat_name == 'mean':
                continue
            fields.append((f'{name}_{stat_name}', stats[stat_name]))
    return fields


def _print_sequence_stats_table(args):
    tracker = getattr(args, 'result_tracker', None)
    if tracker is None:
        return
    sequences = {
        'rec_l1': tracker.get('rec_l1_mean_values', []),
        'rec_l1_maxb': tracker.get('rec_l1_maxb_mean_values', []),
        'rec_l2': tracker.get('rec_l2_mean_values', []),
        'rec_token': tracker.get('rec_token_values', []),
        'rec_maxb_token': tracker.get('rec_maxb_token_values', []),
    }
    sequences.update(tracker.get('input_metric_values', {}))
    print('[Per-input metric statistics]:', flush=True)
    print(
        f'{"metric":24} | {"count":>5} | {"mean":>10} | {"std":>10} | {"var":>10} | '
        f'{"min":>10} | {"p25":>10} | {"median":>10} | {"p75":>10} | {"max":>10}',
        flush=True,
    )
    for name in sorted(sequences):
        stats = _sequence_stats(sequences[name])
        if stats['count'] == 0:
            print(f'{name:24} | {0:5d} | n/a', flush=True)
            continue
        print(
            f'{name:24} | {stats["count"]:5d} | '
            f'{stats["mean"]:10.6f} | {stats["std"]:10.6f} | {stats["var"]:10.6f} | '
            f'{stats["min"]:10.6f} | {stats["p25"]:10.6f} | {stats["median"]:10.6f} | '
            f'{stats["p75"]:10.6f} | {stats["max"]:10.6f}',
            flush=True,
        )
    print('', flush=True)


def _emit_result_summary(args):
    tracker = getattr(args, 'result_tracker', None)
    if tracker is None or tracker.get('summary_emitted'):
        return

    defense_param_name, defense_param_value = _defense_param_spec(args)
    fields = [
        ('summary_version', tracker.get('summary_version', 1)),
        ('result_status', tracker.get('result_status', 'ok')),
        ('dataset', args.dataset),
        ('split', args.split),
        ('task', args.task),
        ('model_path', args.model_path),
        ('finetuned_path', args.finetuned_path if args.finetuned_path is not None else 'n/a'),
        ('batch_size', args.batch_size),
        ('seed', getattr(args, 'rng_seed', None)),
        ('train_method', getattr(args, 'train_method', 'full')),
        ('peft_method', getattr(args, 'peft_method', None)),
        ('peft_type', getattr(args, 'peft_type', None)),
        ('peft_eval_scope', getattr(args, 'peft_eval_scope', peft_eval_scope(getattr(args, 'peft_method', None)))),
        ('peft_target_modules', getattr(args, 'peft_target_modules', None)),
        ('peft_feedforward_modules', getattr(args, 'peft_feedforward_modules', None)),
        ('peft_num_virtual_tokens', getattr(args, 'peft_num_virtual_tokens', None)),
        ('adapter_reduction_factor', getattr(args, 'adapter_reduction_factor', None)),
        ('peft_checkpoint_type', getattr(args, 'peft_checkpoint_type', None)),
        ('peft_adapter_r', getattr(args, 'peft_adapter_r', None)),
        ('peft_adapter_target_modules', getattr(args, 'peft_adapter_target_modules', None)),
        ('peft_adapter_feedforward_modules', getattr(args, 'peft_adapter_feedforward_modules', None)),
        ('peft_adapter_task_type', getattr(args, 'peft_adapter_task_type', None)),
        ('peft_adapter_base_model', getattr(args, 'peft_adapter_base_model', None)),
        ('peft_adapter_peft_type', getattr(args, 'peft_adapter_peft_type', None)),
        ('peft_adapter_reduction_factor', getattr(args, 'peft_adapter_reduction_factor', None)),
        ('peft_adapter_architecture', getattr(args, 'peft_adapter_architecture', None)),
        ('peft_adapter_name', getattr(args, 'peft_adapter_name', None)),
        ('lora_r', getattr(args, 'lora_r', None)),
        ('lora_target_modules', getattr(args, 'lora_target_modules', None)),
        ('lora_checkpoint_type', getattr(args, 'lora_checkpoint_type', None)),
        ('lora_adapter_r', getattr(args, 'lora_adapter_r', None)),
        ('lora_adapter_target_modules', getattr(args, 'lora_adapter_target_modules', None)),
        ('lora_adapter_feedforward_modules', getattr(args, 'lora_adapter_feedforward_modules', None)),
        ('lora_adapter_task_type', getattr(args, 'lora_adapter_task_type', None)),
        ('lora_adapter_base_model', getattr(args, 'lora_adapter_base_model', None)),
        ('lora_adapter_peft_type', getattr(args, 'lora_adapter_peft_type', None)),
        ('defense', getattr(args, 'defense', 'none')),
        ('defense_param_name', defense_param_name),
        ('defense_param_value', defense_param_value),
        *dpsgd_opacus_summary_fields(args, tracker),
        *rep_bottleneck_summary_fields(args),
        *partial_gradient_summary_fields(args),
        *adaptive_attack_summary_fields(args),
        ('n_inputs_requested', tracker.get('n_inputs_requested')),
        ('n_inputs_completed', tracker.get('n_inputs_completed')),
        ('last_input_idx', tracker.get('last_input_idx')),
        ('last_input_time', tracker.get('last_input_time')),
        ('last_total_time', tracker.get('last_total_time')),
        ('last_rec_status', tracker.get('last_rec_status')),
        ('rec_l1_mean', _safe_mean(tracker.get('rec_l1_mean_values', []))),
        ('rec_l1_maxb_mean', _safe_mean(tracker.get('rec_l1_maxb_mean_values', []))),
        ('rec_l2_mean', _safe_mean(tracker.get('rec_l2_mean_values', []))),
        ('rec_token_mean', _safe_mean(tracker.get('rec_token_values', []))),
        ('rec_maxb_token_mean', _safe_mean(tracker.get('rec_maxb_token_values', []))),
        *_all_sequence_stat_fields(args),
    ]

    if tracker.get('error_type'):
        fields.append(('error_type', tracker['error_type']))
    if tracker.get('error_message'):
        fields.append(('error_message', tracker['error_message']))

    aggregate_metrics = tracker.get('aggregate_metrics', {})
    for key in sorted(aggregate_metrics):
        fields.append((key, aggregate_metrics[key]))

    print(SUMMARY_START, flush=True)
    for key, value in fields:
        print(f'{key}={_fmt_summary_value(value)}', flush=True)
    print(SUMMARY_END, flush=True)
    tracker['summary_emitted'] = True

def emit_rec_metrics(
    args,
    orig_batch,
    rec_status,
    rec_l1=None,
    rec_l1_maxB=None,
    rec_l2=None,
    rec_maxb_token=0.0,
    rec_token=0.0,
):
    batch_size = orig_batch['input_ids'].shape[0]
    rec_l1 = rec_l1 if rec_l1 is not None else [False] * batch_size
    rec_l1_maxB = rec_l1_maxB if rec_l1_maxB is not None else [False] * batch_size
    rec_l2 = rec_l2 if rec_l2 is not None else [False] * batch_size

    print(f'Rec Status: {rec_status}')
    print(
        f'Rec L1: {rec_l1}, Rec L1 MaxB: {rec_l1_maxB}, '
        f'Rec MaxB Token: {rec_maxb_token}, Rec Token: {rec_token}, Rec L2: {rec_l2}'
    )
    _record_rec_metrics(args, rec_status, rec_l1, rec_l1_maxB, rec_l2, rec_maxb_token, rec_token)

    if args.neptune:
        args.neptune['logs/rec_l1'].log(np.array(rec_l1).sum())
        args.neptune['logs/rec_l1_max_b'].log(np.array(rec_l1_maxB).sum())
        args.neptune['logs/maxB token'].log(rec_maxb_token)
        args.neptune['logs/token'].log(rec_token)
        args.neptune['logs/rec_l2'].log(np.array(rec_l2).sum())


def _ranked_bert_l1_candidates(args, model_wrapper, R_Q, embeds):
    size = adaptive_check_if_in_span(
        args,
        R_Q,
        embeds,
        args.dist_norm,
        layer_position=0,
    )
    flat = size.reshape(-1)
    cap = min(
        int(flat.numel()),
        max(
            int(args.batch_size),
            max(50 * int(args.batch_size), int(0.05 * len(model_wrapper.tokenizer))),
        ),
    )
    _, flat_idx = torch.topk(flat, k=cap, largest=False)

    coords = []
    remaining = flat_idx
    for dim in reversed(size.shape):
        coords.append(remaining % int(dim))
        remaining = torch.div(remaining, int(dim), rounding_mode="floor")
    coords = tuple(reversed(coords))
    if len(coords) < 3:
        raise ValueError(f"Expected BERT token/type candidate dimensions, got shape={tuple(size.shape)}.")
    return coords[-2], coords[-1]


def filter_l1(args, model_wrapper, R_Qs):
    tokenizer = model_wrapper.tokenizer
    res_pos, res_ids, res_types = [], [], []
        
    sentence_ends = []
    p = 0
    n_tokens = 0

    while True:
        if args.verbose_attack_debug:
            print(f'L1 Position {p}')
        embeds = model_wrapper.get_embeddings(p)
        if model_wrapper.is_bert():
            if not uses_noisy_gradient_decoding(args):
                _, res_ids_new, res_types_new = adaptive_get_top_B_in_span(
                    args,
                    R_Qs[0],
                    embeds,
                    args.batch_size,
                    args.l1_span_thresh,
                    args.dist_norm,
                    layer_position=0,
                )
            else:
                res_ids_new, res_types_new = _ranked_bert_l1_candidates(
                    args,
                    model_wrapper,
                    R_Qs[0],
                    embeds,
                )
        else:
            if not uses_noisy_gradient_decoding(args):
                _, res_ids_new = adaptive_get_top_B_in_span(
                    args,
                    R_Qs[0],
                    embeds,
                    args.batch_size,
                    args.l1_span_thresh,
                    args.dist_norm,
                    layer_position=0,
                )
            else:
                std_thrs = args.p1_std_thrs if p==0 else None
                d = adaptive_get_span_dists(args, model_wrapper, R_Qs, embeds, p)
                res_ids_new = filter_outliers(
                    d,
                    std_thrs=std_thrs,
                    maxB=max(
                        50 * model_wrapper.args.batch_size,
                        int(0.05 * len(model_wrapper.tokenizer)),
                    ),
                    verbose=args.verbose_attack_debug,
                )
            res_types_new = torch.zeros_like(res_ids_new)
        res_pos_new = torch.ones_like( res_ids_new ) * p
        
        del embeds
        
        res_types += [res_types_new.tolist()]
        ids = res_ids_new.tolist()
        if len(ids) == 0 or p > tokenizer.model_max_length or p > args.max_len:
            break
        stop_after_current_position = bool(getattr(args, "_adaptive_l1_stop_after_current_position", False))
        setattr(args, "_adaptive_l1_stop_after_current_position", False)
        while model_wrapper.eos_token in ids:
            end_token_ind = ids.index(model_wrapper.eos_token)
            sentence_token_type = res_types[-1][ end_token_ind ]
            sentence_ends.append((p,sentence_token_type))
            ids = ids[:end_token_ind] + ids[end_token_ind+1:]
            res_types[-1] = res_types[-1][:end_token_ind] + res_types[-1][end_token_ind+1:]
        res_ids += [ids]
        res_pos += res_pos_new.tolist()
        n_tokens += len(ids)
        p += 1
        if model_wrapper.has_rope():
            break
        if stop_after_current_position:
            break
        
    return res_pos, res_ids, res_types, sentence_ends


def uses_nonprefix_dager(args):
    info = getattr(args, 'partial_gradient_info', {})
    return info.get('partial_attack_variant') == PARTIAL_ATTACK_DAGER_NONPREFIX


def _decode_gpt2_nonprefix(args, model_wrapper, R_Qs, orig_batch):
    tokenizer = model_wrapper.tokenizer
    layer_indices = nonprefix_layer_indices(args)
    cap = nonprefix_candidate_cap(args)
    max_len = min(int(args.max_len), orig_batch['input_ids'].shape[1])
    eos_id = model_wrapper.eos_token
    pad_id = model_wrapper.pad_token

    start_prefix = [model_wrapper.start_token] if model_wrapper.start_token is not None else []
    beams = [(start_prefix, 0.0)]
    completed = []
    res_ids = []
    res_pos = []
    res_types = []

    for pos in range(max_len):
        if args.verbose_attack_debug:
            print(f'Non-prefix partial position {pos}')
        proposals = []
        position_candidate_scores = {}
        for prefix, prefix_score in beams:
            if prefix and prefix[-1] == eos_id:
                completed.append((prefix, prefix_score))
                continue

            prefix_tensor = torch.tensor(prefix, dtype=torch.long, device=args.device).unsqueeze(0)
            prefix_layer = None
            if prefix_tensor.shape[1] > 0:
                attention_mask = torch.ones_like(prefix_tensor, device=args.device)
                prefix_layer = model_wrapper.get_layer_inputs(
                    prefix_tensor,
                    attention_mask=attention_mask,
                    layer_indices=[layer_indices[0]],
                )[0]
            chunk_scores = []
            chunk_ids = []
            chunk_size = max(1, int(getattr(args, 'parallel', 1000)))
            vocab_size = len(tokenizer)
            for start in range(0, vocab_size, chunk_size):
                ids_chunk = torch.arange(
                    start,
                    min(start + chunk_size, vocab_size),
                    dtype=torch.long,
                    device=args.device,
                )
                if prefix_tensor.shape[1] > 0:
                    candidate_batch = torch.cat(
                        (prefix_tensor.repeat(ids_chunk.shape[0], 1), ids_chunk.unsqueeze(1)),
                        dim=1,
                    )
                else:
                    candidate_batch = ids_chunk.unsqueeze(1)
                candidate_attention = torch.ones_like(candidate_batch, device=args.device)
                candidate_layer = model_wrapper.get_layer_inputs(
                    candidate_batch,
                    attention_mask=candidate_attention,
                    layer_indices=[layer_indices[0]],
                )[0][:, -1:, :]
                if prefix_layer is not None and prefix_layer.shape[1] > 0:
                    candidate_layer = torch.cat(
                        (prefix_layer.repeat(candidate_layer.shape[0], 1, 1), candidate_layer),
                        dim=1,
                    )
                token_scores = adaptive_check_if_in_span(
                    args,
                    R_Qs[0],
                    candidate_layer,
                    args.dist_norm,
                    layer_position=0,
                )[:, -1].detach()
                if pad_id is not None and start <= int(pad_id) < start + ids_chunk.shape[0]:
                    token_scores[int(pad_id) - start] = torch.inf
                keep = min(cap, token_scores.shape[0])
                scores_chunk, ids_order = torch.topk(token_scores, k=keep, largest=False)
                chunk_scores.append(scores_chunk)
                chunk_ids.append(ids_chunk[ids_order])
            top_pool_scores = torch.cat(chunk_scores)
            top_pool_ids = torch.cat(chunk_ids)
            keep = min(cap, top_pool_scores.shape[0])
            top_scores, order = torch.topk(top_pool_scores, k=keep, largest=False)
            top_ids = top_pool_ids[order]
            for tok_score, tok_id in zip(top_scores.detach().cpu().tolist(), top_ids.detach().cpu().tolist()):
                tok_id = int(tok_id)
                prev_score = position_candidate_scores.get(tok_id)
                if prev_score is None or tok_score < prev_score:
                    position_candidate_scores[tok_id] = float(tok_score)

            for tok_score, tok_id in zip(top_scores.detach().cpu().tolist(), top_ids.detach().cpu().tolist()):
                sentence = prefix + [int(tok_id)]
                sentence_tensor = torch.tensor(sentence, dtype=torch.long, device=args.device).unsqueeze(0)
                sentence_attention = torch.ones_like(sentence_tensor, device=args.device)
                layer_inputs = model_wrapper.get_layer_inputs(
                    sentence_tensor,
                    attention_mask=sentence_attention,
                    layer_indices=[layer_indices[1]],
                )[0]
                span_scores = adaptive_check_if_in_span(
                    args,
                    R_Qs[1],
                    layer_inputs,
                    args.dist_norm,
                    layer_position=1,
                )
                if sentence[0] == eos_id:
                    seq_score = float(span_scores[:, 1:].mean().item()) if span_scores.shape[1] > 1 else float(tok_score)
                else:
                    seq_score = float(span_scores.mean().item())
                proposals.append((sentence, prefix_score + seq_score, seq_score))

        if position_candidate_scores:
            ids = [
                tok
                for tok, _ in sorted(
                    position_candidate_scores.items(),
                    key=lambda item: item[1],
                )
            ]
            res_ids.append(ids)
            res_pos.extend([pos] * len(ids))
            res_types.append([0] * len(ids))

        if not proposals:
            break
        proposals.sort(key=lambda item: item[1])
        next_beams = []
        for sentence, score, _ in proposals:
            if sentence[-1] == eos_id:
                completed.append((sentence, score))
            else:
                next_beams.append((sentence, score))
            if len(next_beams) >= cap:
                break
        beams = next_beams[:cap]
        if not beams:
            break

    completed.extend(beams)
    completed.sort(key=lambda item: item[1])
    predicted_sentences = [sent for sent, _ in completed[:args.batch_size]]
    predicted_scores = []
    for sent in predicted_sentences:
        if not sent:
            predicted_scores.append(float('inf'))
            continue
        sentence_tensor = torch.tensor(sent, dtype=torch.long, device=args.device).unsqueeze(0)
        sentence_attention = torch.ones_like(sentence_tensor, device=args.device)
        layer_inputs = model_wrapper.get_layer_inputs(
            sentence_tensor,
            attention_mask=sentence_attention,
            layer_indices=[layer_indices[1]],
        )[0]
        span_scores = adaptive_check_if_in_span(
            args,
            R_Qs[1],
            layer_inputs,
            args.dist_norm,
            layer_position=1,
        )
        if sent and sent[0] == eos_id and span_scores.shape[1] > 1:
            predicted_scores.append(float(span_scores[:, 1:].mean().item()))
        else:
            predicted_scores.append(float(span_scores.mean().item()))
    if not predicted_sentences and completed:
        predicted_sentences = [completed[0][0]]
        predicted_scores = [float(completed[0][1])]
    return res_pos, res_ids, res_types, [], predicted_sentences, predicted_scores

def reconstruct(args, device, sample, metric, model_wrapper: ModelWrapper, precomputed_true_grads=None):
    global total_correct_tokens, total_tokens, total_correct_maxB_tokens

    tokenizer = model_wrapper.tokenizer
    
    sequences, true_labels = sample
    
    orig_batch = tokenizer(sequences,padding=True, truncation=True, max_length=min(tokenizer.model_max_length, model_wrapper.emb_size - 20),return_tensors='pt').to(args.device)
    args.defense_rng_step = int(args.result_tracker.get('n_inputs_completed', 0))
    
    if precomputed_true_grads is not None:
        true_grads = precomputed_true_grads
    elif requires_gradient_generation_defense(getattr(args, "defense", "none")):
        true_grads = apply_defense(
            None, args, model_wrapper=model_wrapper, batch=orig_batch, labels=true_labels
        )
    else:
        true_grads = model_wrapper.compute_grads(orig_batch, true_labels)
        true_grads = apply_defense(
            true_grads, args, model_wrapper=model_wrapper, batch=orig_batch, labels=true_labels
        )
    true_grads = apply_partial_gradient_filter(
        true_grads,
        args,
        parameter_names=model_wrapper.trainable_parameter_names(),
    )
    prediction, predicted_sentences, predicted_sentences_scores = [], [], []
    #import pdb;pdb.set_trace() 
    with torch.no_grad():
        B, R_Qs = model_wrapper.get_matrices_expansions(true_grads, B=None, tol=args.rank_tol)
        prepare_adaptive_attack(
            args,
            true_grads,
            parameter_names=model_wrapper.trainable_parameter_names(),
        )
        R_Q = R_Qs[0]
        R_Q2 = R_Qs[1]
        
        if B is None:
            reference = []
            for i in range(orig_batch['input_ids'].shape[0]):
                reference += [remove_padding(tokenizer, orig_batch['input_ids'][i], left=(args.pad=='left'))]
            emit_rec_metrics(args, orig_batch, rec_status='rank_decomp_failed')
            return ['' for _ in range(len(reference))], reference
        R_Q, R_Q2 = R_Q.to(args.device), R_Q2.to(args.device)
        total_true_token_count, total_true_token_count2 = 0, 0
        for i in range( orig_batch['input_ids'].shape[1] ):
            total_true_token_count2 += args.batch_size - ( orig_batch['input_ids'][:,i] == model_wrapper.pad_token).sum()
            uniques = torch.unique(orig_batch['input_ids'][:,i])
            total_true_token_count += uniques.numel()
            if model_wrapper.pad_token in uniques.tolist():
                total_true_token_count -= 1
   
        print(f"{B}/{total_true_token_count}/{total_true_token_count2}")
        if args.neptune:
            args.neptune['logs/max_rank'].log( B )
            args.neptune['logs/batch_tokens'].log( total_true_token_count2 ) 
            args.neptune['logs/batch_unique_tokens'].log( total_true_token_count )
         
        del true_grads 
       
        nonprefix_dager = uses_nonprefix_dager(args)
        if nonprefix_dager:
            (
                res_pos,
                res_ids,
                res_types,
                sentence_ends,
                predicted_sentences,
                predicted_sentences_scores,
            ) = _decode_gpt2_nonprefix(args, model_wrapper, R_Qs, orig_batch)
        else:
            res_pos, res_ids, res_types, sentence_ends = filter_l1(args, model_wrapper, R_Qs)
        
        if args.verbose_attack_debug:
            print(orig_batch)
            print(orig_batch['input_ids'].T)
        if len(res_ids) == 0:        
            reference = []
            for i in range(orig_batch['input_ids'].shape[0]):
                reference += [remove_padding(tokenizer, orig_batch['input_ids'][i], left=(args.pad=='left'))]
            emit_rec_metrics(args, orig_batch, rec_status='no_l1_candidates')
            return ['' for _ in reference], reference
        if args.verbose_attack_debug and len(res_ids[0]) < 100000:
            print(res_pos, res_ids, res_types)
        
        rec_l1, rec_l1_maxB, rec_l2 = [], [], []

        for s in range( orig_batch['input_ids'].shape[0] ):
            sentence_in = True
            sentence_in_max_B = True
            orig_sentence = orig_batch['input_ids'][s]
            last_idx = torch.where(orig_batch['input_ids'][s] != tokenizer.pad_token_id)[0][-1].item()
            for pos, token in enumerate( orig_sentence ):
                if not model_wrapper.is_bert() and pos == last_idx:
                    break
                if pos >= len(res_ids) and not model_wrapper.has_rope():
                    sentence_in = False
                    break
                if token == model_wrapper.pad_token and args.pad=='right':
                    pos-=1
                    break
                elif token == model_wrapper.pad_token and args.pad=='left':
                    continue
                if model_wrapper.has_rope():
                    total_correct_tokens += 1 if token in res_ids[0] else 0
                    total_correct_maxB_tokens += 1 if token in res_ids[0][:min(args.batch_size, len(res_ids[0]))] else 0
                    total_tokens += 1
                else:
                    total_correct_tokens += 1 if token in res_ids[pos] else 0
                    total_correct_maxB_tokens += 1 if token in res_ids[pos][:min(args.batch_size, len(res_ids[pos]))] else 0   
                    total_tokens += 1
                if token == model_wrapper.eos_token and args.pad=='right':
                    break
                
                if model_wrapper.has_rope():
                    if model_wrapper.has_bos() and token==model_wrapper.start_token:
                        continue
                    sentence_in = sentence_in and (token in res_ids[0])
                    sentence_in_max_B = sentence_in_max_B and (token in res_ids[0][:min(args.batch_size, len(res_ids[0]))])
                else:
                    sentence_in = sentence_in and (token in res_ids[pos]) 
                    sentence_in_max_B = sentence_in_max_B and (token in res_ids[pos][:min(args.batch_size, len(res_ids[pos]))]) 
            if model_wrapper.is_bert():
                sentence_in = sentence_in and (pos, orig_batch['token_type_ids'][s][pos]) in sentence_ends
                sentence_in_max_B = sentence_in and (pos, orig_batch['token_type_ids'][s][pos]) in sentence_ends

            rec_l1.append( sentence_in )
            rec_l1_maxB.append( sentence_in_max_B )
            if model_wrapper.has_rope():
                orig_sentence = (orig_sentence).reshape(1,-1)
            else:
                orig_sentence = (orig_sentence[:pos+1]).reshape(1,-1)
            if model_wrapper.is_bert():
                token_type_ids = (orig_batch['token_type_ids'][s][:orig_sentence.shape[1]]).reshape(1,-1)
                input_layer1 = model_wrapper.get_layer_inputs(orig_sentence, token_type_ids)[0]
            else:
                attention_mask = orig_batch['attention_mask'][s][:orig_sentence.shape[1]].reshape(1,-1)
                layer_indices = [nonprefix_layer_indices(args)[1]] if nonprefix_dager else None
                input_layer1 = model_wrapper.get_layer_inputs(
                    orig_sentence,
                    attention_mask=attention_mask,
                    layer_indices=layer_indices,
                )[0]

            sizesq2 = adaptive_check_if_in_span(
                args,
                R_Q2,
                input_layer1,
                args.dist_norm,
                layer_position=1,
            )
            boolsq2 = sizesq2 < args.l2_span_thresh
            if args.verbose_attack_debug:
                print(sizesq2)
        
            if args.task == 'next_token_pred':
                rec_l2.append( torch.all(boolsq2[:-1]).item() )
            elif model_wrapper.has_rope(): 
                rec_l2.append( torch.all(boolsq2[1:]).item() )
            else:
                rec_l2.append( torch.all(boolsq2).item() )

        rec_maxb_token = total_correct_maxB_tokens/total_tokens if total_tokens > 0 else 0.0
        rec_token = total_correct_tokens/total_tokens if total_tokens > 0 else 0.0
        emit_rec_metrics(
            args,
            orig_batch,
            rec_status='ok',
            rec_l1=rec_l1,
            rec_l1_maxB=rec_l1_maxB,
            rec_l2=rec_l2,
            rec_maxb_token=rec_maxb_token,
            rec_token=rec_token,
        )
             
        if model_wrapper.is_decoder() and not nonprefix_dager:
            max_ids = -1
            for i in range(len(res_ids)):
                if len(res_ids[i]) > args.max_ids:
                    max_ids = args.max_ids
            predicted_sentences, predicted_sentences_scores, top_B_incorrect_sentences, top_B_incorrect_scores  = filter_decoder(args, model_wrapper, R_Qs, res_ids, max_ids=max_ids)
            if len(predicted_sentences) < orig_batch['input_ids'].shape[0]:
                predicted_sentences += top_B_incorrect_sentences
                predicted_sentences_scores += top_B_incorrect_scores
        elif not model_wrapper.is_decoder():
            for l,token_type in sentence_ends:
                
                if args.l1_filter == 'maxB':
                    max_ids = args.batch_size
                elif args.l1_filter == 'all':
                    max_ids = -1
                else:
                    assert False

                if args.l2_filter == 'non-overlap':
                    correct_sentences = []
                    approx_sentences = []
                    approx_scores = []
                    for sent, sc in zip( predicted_sentences, predicted_sentences_scores ): 
                        if sc < args.l2_span_thresh:
                            correct_sentences.append( sent )
                        else:
                            approx_sentences.append( sent )
                            approx_scores.append( sc )

                    new_predicted_sentences, new_predicted_scores = filter_encoder(args, model_wrapper, R_Q2, l, token_type, res_ids, correct_sentences, approx_sentences, approx_scores, max_ids, args.batch_size)
                elif args.l2_filter == 'overlap':
                    new_predicted_sentences, new_predicted_scores = filter_encoder(args, model_wrapper, R_Q2, l, token_type, res_ids, [], [], [], max_ids, args.batch_size)
                else:
                    assert False

                predicted_sentences += new_predicted_sentences
                predicted_sentences_scores += new_predicted_scores

    correct_sentences = []
    approx_sentences = []
    approx_sentences_ext = []
    approx_sentences_lens = []
    approx_scores = []
    if len(predicted_sentences) == 0:
        reference = []
        for i in range(orig_batch['input_ids'].shape[0]):
            reference += [remove_padding(tokenizer, orig_batch['input_ids'][i, :tokenizer.model_max_length], left=(args.pad=='left'))]
        return ['' for _ in reference], reference
    max_len = max( [len(s) for s in predicted_sentences] )
    for sent, sc in zip( predicted_sentences, predicted_sentences_scores ): 
        if sc < args.l2_span_thresh:
            correct_sentences.append( sent )
        else:
            approx_sentences.append( sent )
            approx_sentences_ext.append( sent + [-1]*(max_len - len(sent)) )
            approx_sentences_lens.append( len(sent) )
            approx_scores.append( sc )
    approx_scores = torch.tensor(approx_scores)
    approx_sentences_lens = torch.tensor(approx_sentences_lens)

    if len(approx_sentences) > 0:
        for i in range(len(correct_sentences)):
            sent = correct_sentences[i]
            similar_sentences = (torch.tensor(sent) == torch.tensor(approx_sentences_ext)[:,:len(sent)]).sum(1) >= torch.min(approx_sentences_lens,torch.tensor(len(sent)))*args.distinct_thresh
            approx_scores[similar_sentences] = torch.inf
        
        predicted_sentences = correct_sentences.copy()
        for i in range(len(correct_sentences), args.batch_size):
            if approx_scores.numel() == 0 or torch.isinf(approx_scores).all():
                break
            idx = torch.argmin( approx_scores )
            predicted_sentences.append( approx_sentences[idx] )
            similar_sentences = (torch.tensor(approx_sentences_ext[idx]) == torch.tensor(approx_sentences_ext)).sum(1) >= max_len*args.distinct_thresh
            approx_scores[similar_sentences] = torch.inf

    for s in predicted_sentences:
        prediction.append( tokenizer.decode(s) )
    if args.neptune:
        args.neptune['logs/num_pred'].log( len(correct_sentences) ) 
    
    reference = []
    for i in range(orig_batch['input_ids'].shape[0]):
        reference += [remove_padding(tokenizer, orig_batch['input_ids'][i, :tokenizer.model_max_length], left=(args.pad=='left'))]
    if len(prediction) > len(reference):
        prediction = prediction[:len(reference)]

    if len(prediction) == 0:
        return ['' for _ in reference], reference

    if model_wrapper.is_decoder():
        new_prediction = []
        og_side = tokenizer.padding_side
        tokenizer.padding_side='right'
        for i in range(len(reference)):
            sequences = [reference[i]] + prediction
            batch = tokenizer(sequences,padding=True, truncation=True, return_tensors='pt')
            best_idx = (batch['input_ids'][1:] == batch['input_ids'][0]).sum(1).argmax()
            new_prediction.append(prediction[best_idx])
        tokenizer.padding_side=og_side
        prediction=new_prediction
    else:
        cost = np.zeros((len(prediction), len(prediction)))
        for i in range(len(prediction)):
            for j in range(len(prediction)):
                fm = metric.compute(predictions=[prediction[i]], references=[reference[j]])['rouge1'].mid.fmeasure
                cost[i, j] = 1.0 - fm
        row_ind, col_ind = linear_sum_assignment(cost)

        ids = list(range(len(prediction)))
        ids.sort(key=lambda i: col_ind[i])
        new_prediction = []
        for i in range(len(prediction)):
            new_prediction += [prediction[ids[i]]]
        prediction = new_prediction

    return prediction, reference


def print_metrics(args, res, suffix):
    #sys.stderr.write(str(res) + '\n')
    summary = _metric_summary(res)
    for metric in SUMMARY_METRICS:
        curr = res[metric].mid
        print(f'{metric:10} | fm: {curr.fmeasure*100:.3f} | p: {curr.precision*100:.3f} | r: {curr.recall*100:.3f}', flush=True)
        if args.neptune:
            args.neptune[f'logs/{metric}-fm_{suffix}'].log(curr.fmeasure*100)
            args.neptune[f'logs/{metric}-p_{suffix}'].log(curr.precision*100)
            args.neptune[f'logs/{metric}-r_{suffix}'].log(curr.recall*100)
    sum_12_fm = summary['agg_r1fm_r2fm']
    if args.neptune:
        args.neptune[f'logs/r1fm+r2fm_{suffix}'].log(sum_12_fm)
    print(f'r1fm+r2fm = {sum_12_fm:.3f}', flush=True)
    print()
    return summary


def _run_dpsgd_opacus_attack(args, device, metric, t_start):
    normalize_dpsgd_opacus_args(args, active=True)
    train_wrapper = ModelWrapper(args)
    attack_wrapper = ModelWrapper(args)
    frozen_train = freeze_position_embeddings(train_wrapper.model)
    frozen_attack = freeze_position_embeddings(attack_wrapper.model)
    if frozen_train or frozen_attack:
        print(
            f"[dager:dpsgd_opacus] Frozen BERT position embeddings "
            f"(train={frozen_train}, attack={frozen_attack}).",
            flush=True,
        )

    train_wrapper.set_model_device(args.device)
    attack_wrapper.set_model_device(args.device)
    train_loader = create_source_dpsgd_dataloader(args, train_wrapper.tokenizer, shuffle=False)
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

    print('\n\nAttacking with source-style Opacus DP-SGD..\n', flush=True)
    predictions, references = [], []
    recover_idx = 0
    requested_stop = min(args.n_inputs, args.end_input)
    for next_batch in train_loader:
        if recover_idx >= requested_stop:
            break
        if is_empty_opacus_batch(next_batch):
            continue

        batch = {key: value.to(args.device) for key, value in next_batch.items()}
        labels = batch["labels"]
        model_inputs = {key: value for key, value in batch.items() if key != "labels"}
        optimizer.zero_grad(set_to_none=True)
        outputs = private_model(**model_inputs, labels=labels.view(-1).long())
        outputs.loss.backward()

        if recover_idx >= args.start_input:
            t_input_start = time.time()
            args.defense_rng_step = int(args.result_tracker.get('n_inputs_completed', 0))
            sample = (decode_batch_texts(train_wrapper.tokenizer, batch), labels)
            print(f'Running input #{recover_idx} of {args.n_inputs}.', flush=True)
            if args.neptune:
                args.neptune['logs/curr_input'].log(recover_idx)

            print('reference: ', flush=True)
            for seq in sample[0]:
                print('========================', flush=True)
                print(seq, flush=True)
            print('========================', flush=True)

            true_grads, _ = capture_private_grads(private_model)
            missing, unexpected = sync_private_model_to_public_model(private_model, attack_wrapper.model)
            if missing or unexpected:
                print(
                    f"[dager:dpsgd_opacus] Loaded private state with "
                    f"missing={len(missing)} unexpected={len(unexpected)}.",
                    flush=True,
                )
            attack_wrapper.model.eval()
            prediction, reference = reconstruct(
                args,
                device,
                sample,
                metric,
                attack_wrapper,
                precomputed_true_grads=true_grads,
            )
            predictions += prediction
            references += reference

            print(f'Done with input #{recover_idx} of {args.n_inputs}.', flush=True)
            print('reference: ', flush=True)
            for seq in reference:
                print('========================', flush=True)
                print(seq, flush=True)
            print('========================', flush=True)

            print('predicted: ', flush=True)
            for seq in prediction:
                print('========================', flush=True)
                print(seq, flush=True)
            print('========================', flush=True)

            print('[Curr input metrics]:', flush=True)
            res = metric.compute(predictions=prediction, references=reference)
            curr_metric_summary = print_metrics(args, res, suffix='curr')
            _record_input_metric_summary(args, curr_metric_summary)

            print('[Aggregate metrics]:', flush=True)
            res = metric.compute(predictions=predictions, references=references)
            args.result_tracker['aggregate_metrics'] = print_metrics(args, res, suffix='agg')

            input_time = str(datetime.timedelta(seconds=time.time() - t_input_start)).split(".")[0]
            total_time = str(datetime.timedelta(seconds=time.time() - t_start)).split(".")[0]
            args.result_tracker['n_inputs_completed'] += 1
            args.result_tracker['last_input_idx'] = recover_idx
            args.result_tracker['last_input_time'] = input_time
            args.result_tracker['last_total_time'] = total_time
            print(f'input #{recover_idx} time: {input_time} | total time: {total_time}', flush=True)
            print()
            print()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        recover_idx += 1

    if args.result_tracker['last_total_time'] is None:
        args.result_tracker['last_total_time'] = str(datetime.timedelta(seconds=time.time() - t_start)).split(".")[0]
    record_dpsgd_opacus_summary(args, args.result_tracker, privacy_engine)
    print('Done with all.', flush=True)
    _print_sequence_stats_table(args)
    if args.neptune:
        args.neptune['logs/curr_input'].log(args.n_inputs)
    _emit_result_summary(args)


def main():
    args.result_tracker = _init_result_tracker(args)
    t_start = time.time()
    try:
        device = torch.device(args.device)
        metric = load_metric('rouge', cache_dir=args.cache_dir)
        if getattr(args, "defense", "none") == DPSGD_OPACUS_DEFENSE:
            _run_dpsgd_opacus_attack(args, device, metric, t_start)
            return

        dataset = TextDataset(args.device, args.dataset, args.split, args.n_inputs, args.batch_size, args.cache_dir)

        model_wrapper = ModelWrapper(args)

        print('\n\nAttacking..\n', flush=True)
        predictions, references = [], []
        
        for i in range(args.start_input, min(args.n_inputs, args.end_input)):
            t_input_start = time.time()
            sample = dataset[i] # (seqs, labels)

            print(f'Running input #{i} of {args.n_inputs}.', flush=True)
            if args.neptune:
                args.neptune['logs/curr_input'].log(i)

            print('reference: ', flush=True)
            for seq in sample[0]:
                print('========================', flush=True)
                print(seq, flush=True)

            print('========================', flush=True)
            
            prediction, reference = reconstruct(args, device, sample, metric, model_wrapper)
            predictions += prediction
            references += reference

            print(f'Done with input #{i} of {args.n_inputs}.', flush=True)
            print('reference: ', flush=True)
            for seq in reference:
                print('========================', flush=True)
                print(seq, flush=True)
            print('========================', flush=True)

            print('predicted: ', flush=True)
            for seq in prediction:
                print('========================', flush=True)
                print(seq, flush=True)
            print('========================', flush=True)

            print('[Curr input metrics]:', flush=True)
            res = metric.compute(predictions=prediction, references=reference)
            curr_metric_summary = print_metrics(args, res, suffix='curr')
            _record_input_metric_summary(args, curr_metric_summary)

            print('[Aggregate metrics]:', flush=True)
            res = metric.compute(predictions=predictions, references=references)
            args.result_tracker['aggregate_metrics'] = print_metrics(args, res, suffix='agg')

            input_time = str(datetime.timedelta(seconds=time.time() - t_input_start)).split(".")[0]
            total_time = str(datetime.timedelta(seconds=time.time() - t_start)).split(".")[0]
            args.result_tracker['n_inputs_completed'] += 1
            args.result_tracker['last_input_idx'] = i
            args.result_tracker['last_input_time'] = input_time
            args.result_tracker['last_total_time'] = total_time
            print(f'input #{i} time: {input_time} | total time: {total_time}', flush=True)
            print()
            print()

        if args.result_tracker['last_total_time'] is None:
            args.result_tracker['last_total_time'] = str(datetime.timedelta(seconds=time.time() - t_start)).split(".")[0]

        print('Done with all.', flush=True)
        _print_sequence_stats_table(args)
        if args.neptune:
            args.neptune['logs/curr_input'].log(args.n_inputs)
        _emit_result_summary(args)
    except Exception as exc:
        if isinstance(exc, UnsupportedPartialGradientExposureError):
            args.result_tracker['result_status'] = 'unsupported'
            args.result_tracker['error_type'] = type(exc).__name__
            args.result_tracker['error_message'] = str(exc)
            args.result_tracker['last_rec_status'] = 'unsupported'
            mark_partial_gradient_unsupported(args, variant=exc.variant, reason=exc.reason)
            if args.result_tracker['last_total_time'] is None:
                args.result_tracker['last_total_time'] = str(
                    datetime.timedelta(seconds=time.time() - t_start)
                ).split(".")[0]
            _emit_result_summary(args)
            return
        args.result_tracker['result_status'] = 'failed'
        args.result_tracker['error_type'] = type(exc).__name__
        args.result_tracker['error_message'] = str(exc)
        if args.result_tracker['last_total_time'] is None:
            args.result_tracker['last_total_time'] = str(
                datetime.timedelta(seconds=time.time() - t_start)
            ).split(".")[0]
        _emit_result_summary(args)
        raise

if __name__ == '__main__':
    main()
