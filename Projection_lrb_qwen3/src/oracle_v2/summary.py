"""Report successful recovery separately from budget/timeout/error coverage."""
from __future__ import annotations
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from .protocol import read_json, write_json


def summarize(root):
    root = Path(root)
    output = root / 'summary'
    output.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    all_records = []
    for path in sorted((root / 'jobs').glob('final_*/records/*.json')):
        row = read_json(path)
        row['job'] = path.parents[1].name
        all_records.append(row)
        checkpoint_condition = row['job'].split('_', 2)[2]
        group = ('clean_mechanism' if checkpoint_condition == 'none' else 'matched_training',
                 checkpoint_condition, row['condition'], row['variant'])
        groups[group].append(row)
    with (output / 'per_sample.jsonl').open('w') as f:
        for row in all_records:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')
    fields = ['group', 'checkpoint_condition', 'condition', 'variant', 'n_total', 'n_completed',
              'ok', 'no_candidate', 'search_budget_exhausted', 'timeout', 'error', 'calibration_failed',
              'token_recovery', 'exact_recovery', 'rouge_1', 'rouge_2',
              'r1_r2_raw', 'r1_r2_pct']
    summary_rows = []
    for key, rows in sorted(groups.items()):
        statuses = Counter(r['status'] for r in rows)
        completed = [r for r in rows if r['status'] in ('ok', 'no_candidate')]
        result = dict(zip(fields[:4], key))
        result.update(n_total=len(rows), n_completed=len(completed))
        for status in fields[6:12]:
            result[status] = statuses[status]
        for metric in ('token_recovery', 'exact_recovery', 'rouge_1', 'rouge_2'):
            result[metric] = mean(float(r[metric]) for r in completed) if completed else None
        result['r1_r2_raw'] = result['rouge_1'] + result['rouge_2'] if completed else None
        result['r1_r2_pct'] = 100 * result['r1_r2_raw'] if completed else None
        summary_rows.append(result)
    with (output / 'privacy.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    checkpoints, utility_groups = [], defaultdict(list)
    for path in sorted((root / 'jobs').glob('formal_*/training/training_metadata.json')):
        m = read_json(path)
        checkpoints.append({'seed': m['seed'], 'condition': m['condition']['id'],
            'checkpoint_path': m['checkpoint_path'], 'checkpoint_files': m['checkpoint_files'],
            'accuracy': m['validation_accuracy'], 'macro_f1': m['validation_macro_f1'],
            'loss': m['validation_loss'], 'reload_verified': m['checkpoint_reload_verified']})
        utility_groups[m['condition']['id']].append(m)
    write_json(output / 'checkpoints.json', checkpoints, immutable=False)
    utility_rows = []
    for cid, rows in sorted(utility_groups.items()):
        utility_rows.append({'condition': cid, 'n_seeds': len(rows),
            'accuracy_mean': mean(r['validation_accuracy'] for r in rows),
            'accuracy_std': stdev(r['validation_accuracy'] for r in rows) if len(rows)>1 else None,
            'macro_f1_mean': mean(r['validation_macro_f1'] for r in rows),
            'loss_mean': mean(r['validation_loss'] for r in rows)})
    with (output / 'utility.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['condition', 'n_seeds', 'accuracy_mean', 'accuracy_std', 'macro_f1_mean', 'loss_mean'])
        writer.writeheader()
        writer.writerows(utility_rows)
    write_json(output / 'privacy_summary.json', summary_rows, immutable=False)
    utilities = {r['condition']: r for r in utility_rows}
    combined = [{**r, 'utility_accuracy_mean': utilities.get(r['checkpoint_condition'], {}).get('accuracy_mean'),
                 'utility_accuracy_std': utilities.get(r['checkpoint_condition'], {}).get('accuracy_std')}
                for r in summary_rows]
    with (output / 'privacy_utility.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields + ['utility_accuracy_mean', 'utility_accuracy_std'])
        writer.writeheader()
        writer.writerows(combined)
    scan_rows = []
    for path in sorted((root / 'jobs').glob('*/scans/*.json')):
        r = read_json(path)
        if 'true_token_rank_mean' in r:
            scan_rows.append({'job': path.parents[1].name, 'stage': r['stage'], 'condition': r['condition'],
                'variant': r['variant'], 'rho': r['rho'], 'sample_key': r['sample_key'],
                'B': r['capacity'][0]['B'], 'q': r['capacity'][0]['q'],
                'B_over_q': r['capacity'][0]['B_over_q'], 'rank_mean': r['true_token_rank_mean'],
                **{f'top{k}': r[f'top{k}'] for k in (1,10,100)}})
    with (output / 'rank_diagnostics.csv').open('w') as f:
        fieldnames = ['job','stage','condition','variant','rho','sample_key','B','q','B_over_q','rank_mean','top1','top10','top100']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scan_rows)
    if scan_rows:
        # Plotting dependencies are isolated from the shared training environment.
        import sys
        plot_runtime = Path(__file__).resolve().parents[3] / '.plot_runtime'
        if plot_runtime.is_dir():
            sys.path.insert(0, str(plot_runtime))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7,4))
        plot_job = 'pilot_clean_rho_sweep' if any(r['job'] == 'pilot_clean_rho_sweep' for r in scan_rows) else 'p0_random_head_scans'
        selected = [r for r in scan_rows if r['job'] == plot_job and r['B'] > 0]
        for variant in ('standard', 'oracle'):
            points = [r for r in selected if r['variant'] == variant]
            if points:
                ax.scatter([r['B_over_q'] for r in points], [r['rank_mean'] for r in points], s=14, alpha=.5, label=variant)
        ax.set(xlabel='Applied basis width B / feature image dimension q', ylabel='Mean true-token rank', yscale='log')
        ax.set_title('Calibration: trained clean pilot' if plot_job == 'pilot_clean_rho_sweep' else 'Smoke only: random classification head')
        if selected:
            ax.legend()
        fig.tight_layout()
        fig.savefig(output / 'b_over_q_rank.png', dpi=180)
        plt.close(fig)
    status = read_json(root / 'pipeline_status.json') if (root / 'pipeline_status.json').exists() else {}
    note = '# Qwen3 oracle v2 results\n\n'
    note += f'Pipeline state: `{status.get("phase", "unknown")}`.\n\n'
    note += 'Recovery averages include only completed searches (`ok` / `no_candidate`). Budget exhaustion, timeout, calibration failure and errors are reported separately; no failure is converted to zero recovery.\n\n'
    note += 'The rank scan is threshold-free. It does not establish formal privacy or architecture-independent behavior.\n'
    (output / 'README.md').write_text(note)
    return {'privacy_rows': len(all_records), 'utility_checkpoints': len(checkpoints), 'scan_rows': len(scan_rows)}
