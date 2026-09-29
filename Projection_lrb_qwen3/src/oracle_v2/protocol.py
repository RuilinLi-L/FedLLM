from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def directory_hashes(path):
    root = Path(path)
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob('*')) if p.is_file()}


def code_hashes(repository_root):
    root = Path(repository_root)
    files = []
    for directory in ('Projection_lrb_qwen3/src', 'Projection_lrb_qwen3/scripts', 'utils'):
        files.extend((root / directory).rglob('*.py'))
    files.append(root / 'constants.py')
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(files)}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value, *, immutable=True):
    path = Path(path)
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if path.exists() and immutable:
        if path.read_text() != content:
            raise ValueError(f'Refusing to overwrite different immutable artifact: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(content)
    os.replace(temp, path)


def load_config(path):
    config = read_json(path)
    if config['protocol'] != 'qwen3_oracle_v2' or config['max_length'] != 32:
        raise ValueError('Unsupported v2 protocol')
    return config


def condition_id(preset, rho=None):
    return 'none' if preset == 'none' else f'{preset}_rho{float(rho):g}'


def conditions(config, *, smoke=False):
    return [{'id': 'none', 'preset': 'none', 'rho': 1.0}] + [
        {'id': condition_id('proj_uniform', rho), 'preset': 'proj_uniform', 'rho': rho}
        for rho in config['smoke_rhos' if smoke else 'rho_grid']
    ] + [{'id': condition_id('proj_only', .5), 'preset': 'proj_only', 'rho': .5}]


def selected_conditions(config):
    return read_json(Path(config['output_root']) / 'controls' / 'operating_points.json')['conditions']


def fresh_final(eligible, old_keys, size=20):
    selected = sorted((s for s in eligible if s['sample_key'] not in old_keys), key=lambda s: s['sample_key'])[:size]
    if len(selected) != size or len({s['sample_key'] for s in selected}) != size:
        raise ValueError('Not enough disjoint final samples')
    return selected


def preregister(config, repository_root):
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    from src.preregister import prepare_eligible_samples

    project = Path(repository_root) / 'Projection_lrb_qwen3'
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'], local_files_only=True)
    data = load_from_disk(config['dataset_path'])
    eligible = prepare_eligible_samples(data['validation'], tokenizer,
        SimpleNamespace(max_length=32, min_effective_token_length=1))
    by_key = {s['sample_key']: s for s in eligible}
    old = {}
    for stage in ('calibration', 'smoke', 'final'):
        rows = [json.loads(s) for s in (project / 'manifests' / f'{stage}.jsonl').read_text().splitlines() if s]
        old[stage] = [r['sample'] for r in rows]
        for sample in old[stage]:
            if sample != by_key.get(sample['sample_key']):
                raise ValueError('Existing manifest disagrees with local tokenizer/dataset')
    old_keys = {s['sample_key'] for group in old.values() for s in group}
    if len(old_keys) != 45:
        raise ValueError('Expected 45 distinct historical samples')
    splits = {**old, 'final': fresh_final(eligible, old_keys)}
    output = Path(config['output_root']) / 'preregistration'
    provenance = {
        'protocol': config['protocol'], 'config_sha256': digest(config),
        'config': config, 'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository_root, text=True).strip(),
        'model_files': directory_hashes(config['model_path']),
        'dataset_files': directory_hashes(config['dataset_path']),
        'historical_sample_keys': sorted(old_keys),
        'stage_hashes': {stage: digest(samples) for stage, samples in splits.items()},
        'final_is_disjoint': not (set(s['sample_key'] for s in splits['final']) & old_keys),
    }
    for stage, samples in splits.items():
        write_json(output / f'{stage}.json', samples)
    provenance['identity_sha256'] = digest(provenance)
    write_json(output / 'manifest.json', provenance)
    return provenance


def load_samples(config, stage):
    root = Path(config['output_root']) / 'preregistration'
    manifest = read_json(root / 'manifest.json')
    check = {k: v for k, v in manifest.items() if k != 'identity_sha256'}
    if digest(check) != manifest['identity_sha256'] or manifest['config_sha256'] != digest(config):
        raise ValueError('Preregistration/config identity mismatch')
    samples = read_json(root / f'{stage}.json')
    if digest(samples) != manifest['stage_hashes'][stage]:
        raise ValueError('Sample manifest hash mismatch')
    return samples


def sample_namespace(sample):
    return SimpleNamespace(sample_key=sample['sample_key'], label=sample['label'],
        input_ids=tuple(sample['tokenization']['input_ids']), eos_token_id=sample['tokenization']['eos_token_id'])
