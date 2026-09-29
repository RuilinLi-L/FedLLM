from __future__ import annotations
import dataclasses
import json
import os
from pathlib import Path
import time
import traceback

import torch
from src.qwen3_classifier import load_local_qwen3_sequence_classifier
from src.gradient_capture import capture_single_example_gradients
from src.dager_qwen3.gradient_gate import diagnose_captured_q_projections
from src.dager_qwen3.model_adapter import Qwen3RoPEDagerAdapter
from src.dager_qwen3.oracle_projection import defend_canonical_gradients, QwenColumnTransform, tensor_sha256
from src.dager_qwen3.metrics import preflight_legacy_dager_rouge_backend
from .protocol import read_json, write_json, digest, file_hash, directory_hashes, code_hashes, load_samples, sample_namespace
from .attack import paired_spans, capacity, scan_arm, token_diagnostics, report_decode
from .training import train


def run_job(config, job):
    root = Path(job['output'])
    root.mkdir(parents=True, exist_ok=True)
    def progress(**fields):
        write_json(root / 'progress.json', {'timestamp': time.time(), 'pid': os.getpid(), **fields}, immutable=False)
        print(json.dumps(fields, ensure_ascii=False), flush=True)
    write_json(root / 'job.json', job)
    if (root / 'done.json').exists() or (root / 'failure.json').exists():
        raise ValueError('Job already has a terminal artifact; no implicit retries')
    progress(kind='starting')
    write_json(root / 'source.json', code_hashes(Path(__file__).resolve().parents[3]))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        if job['kind'] == 'train':
            result = train(config, seed=job['seed'], condition=job['condition'],
                output=root / 'training', smoke_steps=job.get('smoke_steps'), progress=progress)
            write_json(root / 'done.json', result)
            return
        if job['kind'] == 'checkpoint_smoke':
            checkpoint_smoke(config, job, progress)
            write_json(root / 'done.json', {'status': 'ok'})
            return
        execute_attacks(config, job, progress)
        write_json(root / 'done.json', {'status': 'ok'})
    except Exception as error:
        result = {'status': 'error', 'error_type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc()}
        write_json(root / 'failure.json', result)
        progress(kind='failed', **result)
        raise


def checkpoint_smoke(config, job, progress):
    root = Path(job['output'])
    samples = load_samples(config, 'smoke')
    checkpoint = Path(config['smoke_checkpoint'])
    bundle = load_local_qwen3_sequence_classifier(checkpoint, mode='trained_checkpoint', dtype='bfloat16')
    bundle.model.eval()
    head_hash = tensor_sha256(bundle.model.score.weight)
    inputs = torch.tensor([samples[0]['tokenization']['input_ids']], device='cuda:0')
    with torch.no_grad():
        expected = bundle.model(input_ids=inputs, attention_mask=torch.ones_like(inputs)).logits.detach().cpu()
    saved = root / 'roundtrip_checkpoint'
    bundle.model.save_pretrained(saved)
    bundle.tokenizer.save_pretrained(saved)
    del bundle
    torch.cuda.empty_cache()
    loaded = load_local_qwen3_sequence_classifier(saved, mode='trained_checkpoint', dtype='bfloat16')
    loaded.model.eval()
    with torch.no_grad():
        actual = loaded.model(input_ids=inputs, attention_mask=torch.ones_like(inputs)).logits.detach().cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if tensor_sha256(loaded.model.score.weight) != head_hash:
        raise ValueError('Checkpoint smoke: classification head was changed')
    write_json(root / 'checkpoint_verification.json', {'status': 'ok', 'head_sha256': head_hash,
        'logits': actual.float().tolist(), 'source_files': directory_hashes(checkpoint)})
    progress(kind='checkpoint_roundtrip_ok')


def execute_attacks(config, job, progress):
    root = Path(job['output'])
    stage = job['stage']
    samples = load_samples(config, stage)
    if job.get('sample_keys') is not None:
        wanted = set(job['sample_keys'])
        samples = [s for s in samples if s['sample_key'] in wanted]
        if len(samples) != len(wanted):
            raise ValueError('Unknown sample key')
    model_mode = job.get('model_mode', 'trained_checkpoint')
    model_path = Path(job.get('checkpoint', config['model_path']))
    model_files = directory_hashes(model_path)
    model_identity = digest(model_files)
    write_json(root / 'provenance.json', {'model_path': str(model_path), 'model_files': model_files,
        'model_sha256': model_identity, 'config_sha256': digest(config),
        'preregistration_sha256': read_json(Path(config['output_root']) / 'preregistration' / 'manifest.json')['identity_sha256'],
        'torch': torch.__version__, 'device': torch.cuda.get_device_name(0)})
    rouge = preflight_legacy_dager_rouge_backend() if job['kind'] == 'decode' else None
    bundle = load_local_qwen3_sequence_classifier(model_path,
        head_seed=job['seed'] if model_mode == 'random_head_diagnostic' else None,
        mode=model_mode, dtype='bfloat16')
    adapter = Qwen3RoPEDagerAdapter(bundle.model, bundle.tokenizer)
    names = tuple(name for name, _ in bundle.model.named_parameters())
    rank_verified = {}
    for row in samples:
        sample = sample_namespace(row)
        progress(kind='capture', sample_key=sample.sample_key)
        torch.manual_seed(job['seed'])
        torch.cuda.manual_seed_all(job['seed'])
        ids = torch.tensor([sample.input_ids], device='cuda:0')
        captured = capture_single_example_gradients(bundle.model, input_ids=ids,
            attention_mask=torch.ones_like(ids), labels=torch.tensor([sample.label], device='cuda:0'))
        diagnostic, gate = diagnose_captured_q_projections(captured=captured, tokenizer=bundle.tokenizer,
            token_ids=sample.input_ids, eos_token_id=sample.eos_token_id, dtype='bfloat16')
        write_json(root / 'diagnostics' / f'{sample.sample_key}.json', diagnostic)
        if diagnostic['passed'] is not True:
            raise ValueError(f'Raw gradient diagnostic failed for {sample.sample_key}')
        raw = tuple(p.grad for p in bundle.model.parameters())
        q_indices = tuple(names.index(name) for name in captured.q_parameter_names)
        active = captured.q_output_gradients[0].detach().float().norm(dim=-1)[0]
        active_mask = (active >= active.max() * 1e-3).cpu().tolist()
        none_scan = None
        for condition in job['conditions']:
            cid = condition['id']
            progress(kind='defense', sample_key=sample.sample_key, condition=cid)
            if condition['preset'] == 'none':
                observed = tuple(raw[i] for i in q_indices)
                transforms = (None, None)
                state_metadata = []
            else:
                update = defend_canonical_gradients(raw, names, preset=condition['preset'],
                    rho=condition['rho'], seed=config['defense_seed'])
                observed = tuple(update.gradients[i] for i in q_indices)
                transforms = tuple(QwenColumnTransform(update.state_for(name)) for name in captured.q_parameter_names)
                state_metadata = []
                for transform in transforms:
                    rank_key = (transform.width, transform.q)
                    if rank_key not in rank_verified:
                        rank_verified[rank_key] = transform.metadata(verify_rank=True)['operator_rank']
                    metadata = transform.metadata()
                    metadata['operator_rank'] = rank_verified[rank_key]
                    state_metadata.append(metadata)
                del update
            with torch.no_grad():
                spans = paired_spans(observed, config)
            cap = capacity(spans, transforms)
            observed_hashes = [tensor_sha256(g) for g in observed]
            variants = ['standard'] if condition['preset'] == 'none' else ['standard', 'oracle']
            pair_scans = {}
            for variant in variants:
                stem = f'{sample.sample_key}_{cid}_{variant}'
                base = {'stage': stage, 'sample_key': sample.sample_key, 'seed': job['seed'],
                    'checkpoint_sha256': model_identity, 'model_mode': model_mode,
                    'condition': cid, 'preset': condition['preset'], 'rho': condition['rho'],
                    'variant': variant, 'capacity': cap, 'transform_states': state_metadata,
                    'observed_gradient_sha256': observed_hashes, 'status': 'ok'}
                progress(kind='scan', record=stem, sample_key=sample.sample_key, condition=cid, variant=variant,
                    deadline=time.time()+config['attack_timeout_seconds'], failure_record=base)
                with torch.no_grad():
                    scan = scan_arm(adapter, spans, transforms, variant, config)
                pair_scans[variant] = scan
                diagnostics = token_diagnostics(scan, sample.input_ids, sample.eos_token_id, active_mask, config['tau1_grid'])
                record = {**base, **diagnostics}
                write_json(root / 'scans' / f'{stem}.json', record)
                if condition['preset'] == 'none':
                    none_scan = scan
                if job['kind'] == 'decode':
                    control = job['controls'][cid][variant]
                    if control['status'] != 'ok':
                        write_json(root / 'records' / f'{stem}_blocked.json',
                            {**base, 'status': 'calibration_failed', 'reason': control.get('reason')})
                        continue
                    for tau2 in control['tau2_grid']:
                        decode_stem = f'{stem}_tau2_{tau2:g}'
                        record_base = {**base, 'tau1': control['tau1'], 'tau2': tau2}
                        progress(kind='decode', record=decode_stem, sample_key=sample.sample_key,
                            deadline=time.time()+config['attack_timeout_seconds'], failure_record=record_base)
                        with torch.no_grad():
                            result = report_decode(adapter=adapter, spans=spans, transforms=transforms, scan=scan,
                                variant=variant, sample=sample, config=config, tau1=control['tau1'], tau2=tau2,
                                rouge_backend=rouge)
                        write_json(root / 'records' / f'{decode_stem}.json', {**base, **result})
                progress(kind='arm_complete', record=stem)
            if condition['preset'] == 'proj_uniform' and condition['rho'] == 1:
                if not all(torch.equal(g, raw[i]) for g, i in zip(observed, q_indices)):
                    raise ValueError('rho=1 gradient identity failed')
                torch.testing.assert_close(pair_scans['oracle'].distances, pair_scans['standard'].distances, atol=0, rtol=0)
                if none_scan is not None:
                    torch.testing.assert_close(pair_scans['oracle'].distances, none_scan.distances, atol=0, rtol=0)
                write_json(root / 'identity' / f'{sample.sample_key}.json', {'status': 'ok', 'rho': 1})
            del observed, spans, transforms
        bundle.model.zero_grad(set_to_none=True)
        del raw, captured
        torch.cuda.empty_cache()
