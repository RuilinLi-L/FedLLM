"""Dependency-gated experiment controller with external hard attack deadlines.

No automatic retries, batch changes, failed-run-to-zero conversions or selection
from final observations. A restart only skips checksum-verified completed jobs.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .protocol import (conditions, selected_conditions, condition_id, load_samples, read_json,
                       write_json, file_hash, digest, code_hashes)
from .calibration import choose_operating_points, choose_tau1, choose_tau2


def artifact_manifest(root):
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(Path(root).rglob('*.json'))
            if p.name not in ('progress.json', 'completion_receipt.json')}


class Pipeline:
    def __init__(self, config, config_path, repo, gpus):
        self.config, self.config_path, self.repo = config, str(config_path), Path(repo)
        self.root = Path(config['output_root'])
        self.gpus = gpus
        self.root.mkdir(parents=True, exist_ok=True)

    def status(self, phase, **kw):
        data = {'phase': phase, 'updated_at': time.time(), **kw}
        write_json(self.root / 'pipeline_status.json', data, immutable=False)
        print(json.dumps(data), flush=True)

    def job(self, name, **fields):
        return {'name': name, 'output': str(self.root / 'jobs' / name), **fields}

    def run_one(self, job, gpu):
        output = Path(job['output'])
        receipt_path = output / 'completion_receipt.json'
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt['job_sha256'] != digest(job) or receipt['artifacts'] != artifact_manifest(output):
                raise ValueError(f'Completed job was modified: {job["name"]}')
            return
        if (output / 'failure.json').exists() or (output / 'job.json').exists():
            raise ValueError(f'Failed/interrupted job requires an explicit new attempt: {job["name"]}')
        output.mkdir(parents=True, exist_ok=True)
        job_path = self.root / 'job_specs' / f'{job["name"]}.json'
        write_json(job_path, job)
        # A job owns a single physical GPU for its lifetime. Never evict other jobs.
        while True:
            memory = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip()
            smoke_metadata = list((self.root / 'jobs').glob('p0_train_*/training/training_metadata.json'))
            if job['kind'] == 'train' and len(smoke_metadata) == 3:
                peak = max(read_json(p)['peak_memory_bytes'] for p in smoke_metadata)
                needed = max(30000, int(peak / (1024**2) * 1.25) + 2048)
            else:
                needed = 65000 if job['kind'] == 'train' else 30000
            if int(memory) >= needed:
                break
            self.status('waiting_for_gpu', job=job['name'], gpu=gpu, free_mib=int(memory), required_mib=needed)
            time.sleep(30)
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'HF_HUB_OFFLINE': '1',
               'HF_DATASETS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '4'}
        command = [sys.executable, str(self.repo / 'Projection_lrb_qwen3/scripts/run_oracle_v2.py'),
                   '--config', self.config_path, '--job', str(job_path)]
        with (output / 'worker.log').open('w') as log:
            process = subprocess.Popen(command, cwd=self.repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                progress_path = output / 'progress.json'
                if progress_path.exists():
                    state = read_json(progress_path)
                    if state.get('deadline') and time.time() > state['deadline']:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        failure = {**state.get('failure_record', {}), 'status': 'timeout',
                            'error': 'External 1800-second hard attack deadline', 'record': state.get('record')}
                        write_json(output / 'records' / f'{state.get("record", "timeout")}_timeout.json', failure)
                        write_json(output / 'failure.json', failure)
                        raise RuntimeError(f'Attack timed out; no implicit retry: {job["name"]}')
                time.sleep(2)
        if process.returncode != 0 or not (output / 'done.json').exists():
            if not (output / 'failure.json').exists():
                write_json(output / 'failure.json', {'status': 'error', 'exit_code': process.returncode})
            raise RuntimeError(f'Job failed: {job["name"]}; see {output / "worker.log"}')
        write_json(receipt_path, {'job_sha256': digest(job), 'artifacts': artifact_manifest(output)})

    def run_group(self, phase, jobs):
        self.status(phase, jobs=[j['name'] for j in jobs])
        queues = [jobs[i::len(self.gpus)] for i in range(len(self.gpus))]
        def lane(gpu, items):
            for job in items:
                self.run_one(job, gpu)
        with ThreadPoolExecutor(max_workers=len(self.gpus)) as pool:
            results = [pool.submit(lane, gpu, items) for gpu, items in zip(self.gpus, queues)]
            errors = []
            for result in results:
                try:
                    result.result()
                except Exception as error:
                    errors.append(str(error))
        if errors:
            self.status('failed', failed_phase=phase, errors=errors)
            raise RuntimeError('; '.join(errors))

    def checkpoint(self, job):
        meta = read_json(Path(job['output']) / 'training' / 'training_metadata.json')
        if meta.get('smoke_only') or not meta.get('checkpoint_reload_verified'):
            raise ValueError('Only complete verified training checkpoints may enter formal attacks')
        for filename, checksum in meta['checkpoint_files'].items():
            if file_hash(Path(meta['checkpoint_path']) / filename) != checksum:
                raise ValueError('Checkpoint checksum mismatch')
        return meta['checkpoint_path']

    def rows(self, job, category):
        return [read_json(p) for p in sorted((Path(job['output']) / category).glob('*.json'))]

    def calibrate(self, tag, checkpoint, selected, scan_job=None):
        scan_job = scan_job or self.job(f'{tag}_calibration_scans', kind='scan', stage='calibration',
            seed=11, checkpoint=checkpoint, conditions=selected)
        if not (Path(scan_job['output']) / 'completion_receipt.json').exists():
            self.run_group(f'{tag}_calibration_scans', [scan_job])
        rows = self.rows(scan_job, 'scans')
        controls = {}
        none_tau1 = choose_tau1([r for r in rows if r['condition'] == 'none' and r['variant'] == 'standard'], self.config['tau1_grid'])
        for condition in selected:
            cid = condition['id']
            controls[cid] = {}
            for variant in (['standard'] if cid == 'none' else ['standard', 'oracle']):
                group = [r for r in rows if r['condition'] == cid and r['variant'] == variant]
                # Standard arms use the undefended control for this checkpoint
                # family. Only the oracle may calibrate transformed residuals.
                decision = none_tau1 if variant == 'standard' else choose_tau1(group, self.config['tau1_grid'])
                controls[cid][variant] = {**decision, 'tau2_grid': self.config['tau2_grid']}
        write_json(self.root / 'controls' / f'{tag}_tau1.json', {
            'controls': controls, 'input_receipt_sha256': file_hash(Path(scan_job['output']) / 'completion_receipt.json')})
        job = self.job(f'{tag}_calibration_decode', kind='decode', stage='calibration', seed=11,
                       checkpoint=checkpoint, conditions=selected, controls=controls)
        self.run_group(f'{tag}_calibration_decode', [job])
        records = self.rows(job, 'records')
        none_tau2 = choose_tau2([r for r in records if r['condition'] == 'none' and r['variant'] == 'standard'], self.config['tau2_grid'])
        for cid, variants in controls.items():
            for variant, control in variants.items():
                if control['status'] != 'ok':
                    continue
                group = [r for r in records if r['condition'] == cid and r['variant'] == variant]
                decision = none_tau2 if variant == 'standard' else choose_tau2(group, self.config['tau2_grid'])
                control.update(decision)
                if decision['status'] == 'ok':
                    control['tau2_grid'] = [decision['tau2']]
        write_json(self.root / 'controls' / f'{tag}_frozen.json', {
            'controls': controls, 'input_receipt_sha256': file_hash(Path(job['output']) / 'completion_receipt.json'),
            'calibration_checkpoint': checkpoint, 'config_sha256': digest(self.config)})
        return controls

    def run(self, stop_after='final'):
        cfg = self.config
        clean = {'id': 'none', 'preset': 'none', 'rho': 1.0}
        proj = {'id': condition_id('proj_only', .5), 'preset': 'proj_only', 'rho': .5}
        uniform = {'id': condition_id('proj_uniform', .01), 'preset': 'proj_uniform', 'rho': .01}
        p0 = [self.job('p0_checkpoint_roundtrip', kind='checkpoint_smoke'),
            self.job('p0_random_head_scans', kind='scan', stage='smoke', seed=22,
                     model_mode='random_head_diagnostic', conditions=conditions(cfg, smoke=True)),
            *[self.job(f'p0_train_{c["id"]}', kind='train', seed=22, condition=c, smoke_steps=3)
              for c in (clean, proj, uniform)]]
        self.run_group('P0_smoke', p0)
        if stop_after == 'smoke':
            self.status('smoke_complete')
            return
        pilot_clean = self.job('pilot_seed11_none', kind='train', seed=11, condition=clean)
        self.run_group('P1_clean_pilot', [pilot_clean])
        pilot_checkpoint = self.checkpoint(pilot_clean)
        sweep = self.job('pilot_clean_rho_sweep', kind='scan', stage='calibration', seed=11,
                        checkpoint=pilot_checkpoint, conditions=conditions(cfg))
        self.run_group('P1_rho_calibration', [sweep])
        selected_rhos, matches = choose_operating_points(self.rows(sweep, 'scans'), cfg['target_b_over_q'])
        selected = [clean, proj] + [{'id': condition_id('proj_uniform', r), 'preset': 'proj_uniform', 'rho': r}
                                   for r in selected_rhos]
        write_json(self.root / 'controls' / 'operating_points.json', {'conditions': selected, 'matches': matches,
            'input_receipt_sha256': file_hash(Path(sweep['output']) / 'completion_receipt.json')})
        # Same-clean-checkpoint mechanism calibration and matched defended pilots.
        clean_controls = self.calibrate('clean_mechanism', pilot_checkpoint, selected, scan_job=sweep)
        defended_pilots = [self.job(f'pilot_seed11_{c["id"]}', kind='train', seed=11, condition=c) for c in selected[1:]]
        self.run_group('P1_defended_pilots', defended_pilots)
        matched_controls = {'none': clean_controls['none']}
        for condition, job in zip(selected[1:], defended_pilots):
            ctl = self.calibrate(f'matched_{condition["id"]}', self.checkpoint(job), [clean, condition])
            matched_controls[condition['id']] = ctl
        write_json(self.root / 'controls' / 'frozen_protocol.json', {
            'conditions': selected, 'clean_controls': clean_controls, 'matched_controls': matched_controls,
            'config_sha256': digest(cfg), 'manifest_sha256': file_hash(self.root / 'preregistration/manifest.json'),
            'code_hashes': code_hashes(self.repo)})
        if stop_after == 'calibration':
            self.status('calibration_complete')
            return
        formal_jobs = [self.job(f'formal_seed{seed}_{c["id"]}', kind='train', seed=seed, condition=c)
                       for seed in cfg['final_seeds'] for c in selected]
        self.run_group('P2_formal_training', formal_jobs)
        # Verify pairing of initialization and data order within every formal seed.
        for seed in cfg['final_seeds']:
            metadata = [read_json(Path(j['output']) / 'training/training_metadata.json')
                        for j in formal_jobs if j['seed'] == seed]
            for field in ('initial_head_sha256', 'data_order_sha256', 'steps_completed'):
                if len({m[field] for m in metadata}) != 1:
                    raise ValueError(f'Training conditions were not paired: seed={seed}, field={field}')
        attacks = []
        for job in formal_jobs:
            seed, condition = job['seed'], job['condition']
            cid = condition['id']
            attacks.append(self.job(f'final_seed{seed}_{cid}', kind='decode', stage='final', seed=seed,
                checkpoint=self.checkpoint(job), conditions=selected if cid == 'none' else [clean, condition],
                controls=clean_controls if cid == 'none' else matched_controls[cid]))
        self.run_group('P3_formal_attacks', attacks)
        self.status('complete', training_jobs=[j['name'] for j in formal_jobs], attack_jobs=[j['name'] for j in attacks])
