"""Pure, deterministic calibration rules; no final-stage inputs accepted."""
from __future__ import annotations
from statistics import median


def choose_operating_points(rows, targets):
    if not rows or any(r['stage'] != 'calibration' for r in rows):
        raise ValueError('Operating points require calibration rows exclusively')
    grouped = {}
    for row in rows:
        if row['preset'] == 'proj_uniform' and row['variant'] == 'oracle':
            if row['status'] != 'ok':
                raise ValueError('Cannot select from an incomplete or failed rho sweep')
            grouped.setdefault(row['rho'], []).append(row['capacity'][0]['B_over_q'])
    if not grouped:
        raise ValueError('No uniform calibration rows')
    counts = {len(v) for v in grouped.values()}
    if counts != {20}:
        raise ValueError('Every rho must have all 20 calibration samples')
    medians = {rho: median(values) for rho, values in grouped.items()}
    matches = []
    selected = []
    for target in targets:
        rho = min(medians, key=lambda r: (abs(medians[r] - target), -r))
        matches.append({'target': target, 'rho': rho, 'median_B_over_q': medians[rho],
            'absolute_gap': abs(medians[rho] - target),
            'target_bracketed': min(medians.values()) <= target <= max(medians.values())})
        if rho not in selected and rho < 1:
            selected.append(rho)
    return selected, matches


def choose_tau1(rows, grid):
    if len(rows) != 20 or any(r['stage'] != 'calibration' or r['status'] != 'ok' for r in rows):
        return {'status': 'calibration_failed', 'reason': 'missing_or_failed_samples'}
    for tau in sorted(grid):
        active = [x for r in rows for x in r['token_diagnostics'] if x['active'] and not x['is_eos']]
        recall = sum(t['distance'] < tau for t in active) / len(active) if active else 0.
        nonempty = sum(r['candidate_counts'][str(tau)] > 0 for r in rows)
        if recall >= .95 and nonempty == 20:
            return {'status': 'ok', 'tau1': tau, 'micro_active_recall': recall, 'nonempty_samples': nonempty}
    return {'status': 'calibration_failed', 'reason': 'no_tau1_satisfies_registered_rule'}


def choose_tau2(rows, grid):
    if any(r['stage'] != 'calibration' for r in rows):
        raise ValueError('tau2 selection may not read final outcomes')
    eligible = []
    for tau in grid:
        # A failed tau1 gate emits blocked records without a tau2 field.
        # They cannot qualify as completed tau2 observations.
        group = [r for r in rows if r.get('tau2') == tau]
        if len(group) != 20 or any(r['status'] not in ('ok', 'no_candidate') for r in group):
            continue
        eligible.append((tau, group))
    if not eligible:
        return {'status': 'calibration_failed', 'reason': 'no_complete_tau2_candidate'}
    tau, group = min(eligible, key=lambda item: (
        -sum(r['token_recovery'] for r in item[1]),
        -sum(r['exact_recovery'] for r in item[1]),
        -sum(r['rouge_1'] + r['rouge_2'] for r in item[1]),
        sum(r['evaluated_prefix_count'] for r in item[1]), f'{item[0]:.10f}'))
    return {'status': 'ok', 'tau2': tau, 'mean_token_recovery': sum(r['token_recovery'] for r in group) / 20}
