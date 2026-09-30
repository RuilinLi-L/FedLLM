"""Offline full-parameter Qwen classifier utility with paired initial states."""
from __future__ import annotations
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import get_linear_schedule_with_warmup

from src.qwen3_classifier import load_local_qwen3_sequence_classifier
from src.dager_qwen3.oracle_projection import defend_canonical_gradients, tensor_sha256
from .protocol import digest, directory_hashes, write_json


def tokenize_rows(dataset, tokenizer, max_length):
    def encode(batch):
        ids = tokenizer(batch['sentence'], add_special_tokens=False, truncation=False,
                        return_attention_mask=False, return_token_type_ids=False)['input_ids']
        return {'input_ids': [tokens[:max_length-1] + [tokenizer.eos_token_id] for tokens in ids],
                'labels': batch['label']}
    return dataset.map(encode, batched=True, remove_columns=dataset.column_names,
                       load_from_cache_file=False, keep_in_memory=True)


def collate(rows, eos):
    width = max(len(row['input_ids']) for row in rows)
    return {'input_ids': torch.tensor([r['input_ids'] + [eos] * (width-len(r['input_ids'])) for r in rows]),
            'attention_mask': torch.tensor([[1]*len(r['input_ids']) + [0]*(width-len(r['input_ids'])) for r in rows]),
            'labels': torch.tensor([r['labels'] for r in rows])}


def train(config, *, seed, condition, output, smoke_steps=None, progress=lambda **kw: None):
    output = Path(output)
    if output.exists():
        raise ValueError(f'Training output already exists; refusing implicit retry: {output}')
    output.mkdir(parents=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    bundle = load_local_qwen3_sequence_classifier(Path(config['model_path']), head_seed=seed,
                                                  dtype='float32', device='cuda:0')
    model, tokenizer = bundle.model, bundle.tokenizer
    initial_head_hash = tensor_sha256(model.score.weight)
    data = load_from_disk(config['dataset_path'])
    train_data = tokenize_rows(data['train'], tokenizer, config['max_length'])
    # Smoke never evaluates utility or reads held-out validation outcomes.
    tconfig = config['training']
    if tconfig['microbatch_size'] != tconfig['batch_size']:
        raise ValueError('This registered runner requires microbatch == effective batch; accumulation is not implicit')
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(train_data), generator=generator).tolist()
    loader = DataLoader(train_data.select(permutation), batch_size=tconfig['batch_size'], shuffle=False,
                        collate_fn=lambda rows: collate(rows, tokenizer.eos_token_id))
    total_steps = len(loader) * tconfig['epochs']
    parameters = tuple(model.parameters())
    names = tuple(name for name, _ in model.named_parameters())
    opt = torch.optim.AdamW(parameters, lr=tconfig['learning_rate'], weight_decay=tconfig['weight_decay'], foreach=False)
    schedule = get_linear_schedule_with_warmup(opt, 0, total_steps)
    started = time.time()
    updates = 0
    defense_updates = 0
    losses = []
    torch.cuda.reset_peak_memory_stats()
    progress(kind='training', step=0, total_steps=total_steps)
    for epoch in range(tconfig['epochs']):
        model.train()
        for cpu_batch in loader:
            batch = {key: value.to('cuda:0') for key, value in cpu_batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(**batch, use_cache=False).loss
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            if condition['preset'] != 'none':
                update = defend_canonical_gradients(tuple(p.grad for p in parameters), names,
                    preset=condition['preset'], rho=condition['rho'], seed=config['defense_seed'])
                for parameter, gradient in zip(parameters, update.gradients):
                    parameter.grad = gradient
                del update
                defense_updates += 1
            opt.step()
            schedule.step()
            updates += 1
            losses.append(float(loss.detach()))
            if updates == 1 or updates % 25 == 0:
                progress(kind='training', step=updates, total_steps=total_steps,
                    loss=float(loss.detach()), elapsed_seconds=time.time()-started,
                    peak_memory_bytes=torch.cuda.max_memory_allocated())
            if smoke_steps is not None and updates >= smoke_steps:
                break
        if smoke_steps is not None and updates >= smoke_steps:
            break
    if condition['preset'] != 'none' and defense_updates != updates:
        raise ValueError('Every optimizer update must be defended')
    metadata = {'status': 'ok', 'seed': seed, 'condition': condition, 'config_sha256': digest(config),
        'initial_head_sha256': initial_head_hash, 'data_order_sha256': digest(permutation),
        'steps_completed': updates, 'expected_steps': total_steps, 'defense_updates': defense_updates,
        'train_loss': sum(losses)/len(losses), 'elapsed_seconds': time.time()-started,
        'peak_memory_bytes': torch.cuda.max_memory_allocated(), 'training': tconfig,
        'parameter_dtype': 'float32', 'compute_dtype': 'bfloat16_autocast',
        'optimizer_state_dtype': sorted({str(v.dtype) for state in opt.state.values() for v in state.values() if torch.is_tensor(v)}),
        'smoke_only': smoke_steps is not None}
    if smoke_steps is not None:
        write_json(output / 'training_metadata.json', metadata)
        return metadata
    if updates != total_steps:
        raise ValueError('Incomplete formal training')
    model.zero_grad(set_to_none=True)
    del opt, schedule
    torch.cuda.empty_cache()
    validation = tokenize_rows(data['validation'], tokenizer, config['max_length'])
    eval_loader = DataLoader(validation, batch_size=tconfig['batch_size'],
        collate_fn=lambda rows: collate(rows, tokenizer.eos_token_id))
    predictions, labels, total_loss = [], [], 0.
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for batch in eval_loader:
            batch = {k: v.to('cuda:0') for k, v in batch.items()}
            result = model(**batch, use_cache=False)
            predictions.extend(result.logits.argmax(-1).cpu().tolist())
            labels.extend(batch['labels'].cpu().tolist())
            total_loss += float(result.loss) * len(batch['labels'])
    metadata.update(validation_accuracy=accuracy_score(labels, predictions),
        validation_macro_f1=f1_score(labels, predictions, average='macro'),
        validation_loss=total_loss/len(labels), validation_count=len(labels))
    checkpoint = output / 'checkpoint'
    model.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    metadata['checkpoint_path'] = str(checkpoint)
    metadata['checkpoint_files'] = directory_hashes(checkpoint)
    metadata['final_head_sha256'] = tensor_sha256(model.score.weight)
    # Verify a saved FP32 checkpoint before reporting successful training.
    probe = next(iter(eval_loader))
    probe = {k: v[:2].to('cuda:0') for k, v in probe.items()}
    with torch.no_grad():
        expected_logits = model(**probe).logits.cpu()
    del bundle, model
    torch.cuda.empty_cache()
    loaded = load_local_qwen3_sequence_classifier(checkpoint, mode='trained_checkpoint', dtype='float32')
    loaded.model.eval()
    with torch.no_grad():
        reloaded_logits = loaded.model(**probe).logits.cpu()
    if tensor_sha256(loaded.model.score.weight) != metadata['final_head_sha256']:
        raise ValueError('Saved classification head changed on reload')
    torch.testing.assert_close(reloaded_logits, expected_logits, rtol=1e-6, atol=1e-6)
    metadata['checkpoint_reload_verified'] = True
    write_json(output / 'training_metadata.json', metadata)
    progress(kind='training_complete', step=updates, **{'validation_accuracy': metadata['validation_accuracy']})
    return metadata
