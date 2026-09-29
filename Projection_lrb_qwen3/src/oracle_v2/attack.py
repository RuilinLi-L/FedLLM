"""Paired v2 attacks. Truth enters only diagnostics/metrics after candidate scans."""
from __future__ import annotations
from dataclasses import asdict
from time import perf_counter
from typing import Any
import torch

from src.dager_qwen3.gradient_decomposition import GradientSpan
from src.dager_qwen3.layer1_filter import scan_qwen3_vocab_layer1_distances, filter_qwen3_layer1_distance_scan
from src.dager_qwen3.layer2_decoder import decode_qwen3_rope_prefixes, Layer2DecoderConfig
from src.dager_qwen3.candidate_provider import RoPECandidateProvider
from src.dager_qwen3.metrics import compute_attack_metrics
from src.dager_qwen3.oracle_projection import QwenColumnTransform


def paired_spans(gradients, config):
    """One deterministic FP32 full SVD per observed tensor, shared across arms.

    Keep legacy absolute-rank/shared-B semantics; explicitly flag null-space
    completion instead of silently capping B to the realized feature dimension.
    """
    decomposed = [torch.linalg.svd(g.detach().float(), full_matrices=False) for g in gradients]
    ranks = [int((s > config['rank_atol']).sum()) for _, s, _ in decomposed]
    if max(ranks) == 0:
        raise ValueError('Both observed q gradients have zero absolute effective rank')
    requested = max(ranks)
    width = gradients[0].shape[1]
    cap = min(width - config['rank_cutoff'], *(min(g.shape) for g in gradients))
    b = min(requested, cap)
    spans = []
    for g, rank, (_, singular, vh) in zip(gradients, ranks, decomposed):
        spans.append(GradientSpan(basis=vh[:b].contiguous(), effective_rank=rank,
            absolute_tolerance=config['rank_atol'], requested_rank=requested, applied_rank=b,
            rank_cap=cap, rank_was_capped=b != requested,
            cap_reason='feature_or_matrix_dimension' if b != requested else None,
            feature_dim=width, gradient_shape=tuple(g.shape), rank_cutoff=config['rank_cutoff'],
            orientation='raw_qwen3_nn_linear_gradient_right_singular_vectors', decomposition_device=str(g.device)))
    return tuple(spans)


def capacity(spans, transforms):
    result = []
    for span, transform in zip(spans, transforms):
        q = span.feature_dim if transform is None else transform.q
        result.append({'B': span.applied_rank, 'q': q, 'B_over_q': span.applied_rank / q,
            'effective_gradient_rank': span.effective_rank,
            'shared_B_exceeds_effective_rank': span.applied_rank > span.effective_rank,
            'B_exceeds_feature_image_rank': span.applied_rank > q,
            'rank_atol': span.absolute_tolerance, 'rank_definition': 'absolute_matrix_rank_atol_rtol_zero',
            'requested_B': span.requested_rank, 'rank_was_capped': span.rank_was_capped,
            'cap_reason': span.cap_reason})
    return result


def token_diagnostics(scan, token_ids, eos, active_mask, grid):
    """Ranks use distance then token-id ordering; also report tie rank intervals."""
    distances = scan.distances
    order = torch.argsort(distances, stable=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(1, len(order) + 1)
    sorted_distances = distances[order]
    items = []
    for position, token in enumerate(token_ids):
        d = distances[token]
        items.append({'position': position, 'token_id': token, 'is_eos': token == eos,
            'active': bool(active_mask[position]), 'distance': float(d), 'rank': int(ranks[token]),
            'tie_rank_min': int(torch.searchsorted(sorted_distances, d, right=False)) + 1,
            'tie_rank_max': int(torch.searchsorted(sorted_distances, d, right=True))})
    false_mask = torch.ones(len(distances), dtype=torch.bool)
    false_mask[list(set(token_ids))] = False
    false = distances[false_mask]
    payload = [item for item in items if not item['is_eos']]
    unique = {item['token_id']: item for item in payload}
    summary = {'token_diagnostics': items,
        'false_q01_residual': float(torch.quantile(false, .01)),
        'true_mean_residual': sum(t['distance'] for t in payload) / len(payload),
        'true_token_rank_mean': sum(t['rank'] for t in payload) / len(payload),
        'candidate_counts': {str(tau): int((distances < tau).sum()) for tau in grid},
        'rank_tie_break': 'distance_then_token_id', 'aggregation': 'positions_excluding_eos'}
    for k in (1, 10, 100):
        summary[f'top{k}'] = sum(t['rank'] <= k for t in payload) / len(payload)
        summary[f'unique_top{k}'] = sum(t['rank'] <= k for t in unique.values()) / len(unique)
    return summary


def scan_arm(adapter, spans, transforms, variant, config):
    return scan_qwen3_vocab_layer1_distances(adapter=adapter, span=spans[0],
        vocab_chunk_size=config['vocab_chunk_size'],
        candidate_transform=transforms[0] if variant == 'oracle' else None)


def decode_observed_q_gradients(*, adapter, spans, transforms, scan, variant,
                               eos_token_id, config, tau1, tau2):
    """No labels, text, token ids or sample object in the production decoder API."""
    layer1 = filter_qwen3_layer1_distance_scan(scan, threshold=tau1)
    provider = RoPECandidateProvider.from_layer1_result(layer1, eos_token_id=eos_token_id, max_ids=-1)
    decoded = decode_qwen3_rope_prefixes(adapter=adapter, span=spans[1], candidate_provider=provider,
        config=Layer2DecoderConfig(max_sequence_length=config['max_length'], threshold=tau2,
            distance_norm='l2', search_budget=config['maxC'], decode_batch_size=config['decode_batch_size']),
        candidate_transform=transforms[1] if variant == 'oracle' else None)
    selected = min(decoded.survivor_prefixes, key=lambda p: (-len(p.token_ids), p.mean_span_distance, p.token_ids), default=None)
    status = 'search_budget_exhausted' if decoded.search_budget_exhausted else ('ok' if selected else 'no_candidate')
    return layer1, decoded, selected, status


def execute_oracle_dager_from_observed_q_gradients(*, adapter, spans, transforms, scan,
                                                  eos_token_id, config, tau1, tau2):
    return decode_observed_q_gradients(adapter=adapter, spans=spans, transforms=transforms,
        scan=scan, variant='oracle', eos_token_id=eos_token_id, config=config, tau1=tau1, tau2=tau2)


def report_decode(*, adapter, spans, transforms, scan, variant, sample, config, tau1, tau2, rouge_backend):
    start = perf_counter()
    layer1, decoded, selected, status = decode_observed_q_gradients(adapter=adapter,
        spans=spans, transforms=transforms, scan=scan, variant=variant,
        eos_token_id=sample.eos_token_id, config=config, tau1=tau1, tau2=tau2)
    # Ground truth is first consulted after search is finished.
    selected_ids = () if selected is None else selected.token_ids
    metrics = compute_attack_metrics(tokenizer=adapter.tokenizer,
        ground_truth_token_ids=sample.input_ids, reconstructed_token_ids=selected_ids,
        layer1_candidate_token_ids=layer1.token_ids, eos_token_id=sample.eos_token_id,
        rouge_metric=rouge_backend.metric)
    legacy = compute_attack_metrics(tokenizer=adapter.tokenizer,
        ground_truth_token_ids=sample.input_ids, reconstructed_token_ids=decoded.selected_token_ids,
        layer1_candidate_token_ids=layer1.token_ids, eos_token_id=sample.eos_token_id,
        rouge_metric=rouge_backend.metric)
    result = asdict(metrics)
    # The existing RoPE provider deliberately excludes EOS. Do not pretend an
    # EOS was generated or make exact recovery impossible by including it in
    # the ground-truth denominator. Keep the legacy metrics separately below.
    target_text_ids = tuple(sample.input_ids[:-1])
    predicted_text_ids = tuple(selected_ids[:-1] if selected_ids and selected_ids[-1] == sample.eos_token_id else selected_ids)
    result['exact_recovery'] = predicted_text_ids == target_text_ids
    result['token_recovery'] = sum(a == b for a, b in zip(predicted_text_ids, target_text_ids)) / len(target_text_ids)
    result['token_recovery_semantics'] = 'aligned_text_positions_excluding_explicit_terminal_eos'
    result['exact_recovery_semantics'] = 'text_token_sequence_exact_match_excluding_terminal_eos'
    result.update(status=status, tau1=tau1, tau2=tau2, reconstructed_token_ids=list(selected_ids),
        candidate_count=layer1.candidate_count, evaluated_prefix_count=decoded.evaluated_prefix_count,
        termination_reason=decoded.termination_reason, per_length_survivor_counts=decoded.per_length_survivor_counts,
        selection_rule=config['selection_rule'], legacy_first_prefix_metrics=asdict(legacy),
        attack_time_seconds=perf_counter() - start, rouge_backend=rouge_backend.json_metadata())
    if status == 'search_budget_exhausted':
        result['partial_metrics'] = {k: result[k] for k in ('token_recovery', 'exact_recovery', 'rouge_1', 'rouge_2')}
        for key in result['partial_metrics']:
            result[key] = None
    return result
