from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset


DPSGD_OPACUS_DEFENSE = "dpsgd_opacus"
DPSGD_OPACUS_DEFAULT_NOISE_MULTIPLIER = 0.01
DPSGD_OPACUS_DEFAULT_DELTA = 1e-5
DPSGD_OPACUS_ACCOUNTANT = "opacus_rdp"


@dataclass(frozen=True)
class DPSGDOpacusConfig:
    noise_multiplier: float
    max_grad_norm: float
    delta: float
    accountant: str = DPSGD_OPACUS_ACCOUNTANT


class _OpacusTensorTupleDataset(Dataset):
    def __init__(self, dataset, keys):
        self.dataset = dataset
        self.keys = tuple(keys)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        return tuple(item[key] for key in self.keys)


class _TupleToMappingCollator:
    def __init__(self, keys, collate_fn):
        self.keys = tuple(keys)
        self.collate_fn = collate_fn

    def __call__(self, features):
        mapped = [
            {key: value for key, value in zip(self.keys, feature)}
            for feature in features
        ]
        return self.collate_fn(mapped)


def dpsgd_opacus_active(args) -> bool:
    return getattr(args, "defense", "none") == DPSGD_OPACUS_DEFENSE


def normalize_dpsgd_opacus_args(args, *, active: bool | None = None):
    if active is None:
        active = dpsgd_opacus_active(args)

    if getattr(args, "defense_dp_delta", None) is None:
        args.defense_dp_delta = DPSGD_OPACUS_DEFAULT_DELTA

    if not active:
        return args

    source_noise = getattr(args, "noise_multiplier", None)
    if getattr(args, "defense_noise", None) is None:
        args.defense_noise = (
            float(source_noise)
            if source_noise is not None
            else DPSGD_OPACUS_DEFAULT_NOISE_MULTIPLIER
        )
    if hasattr(args, "noise_multiplier") and source_noise is None:
        args.noise_multiplier = float(args.defense_noise)

    source_clip = getattr(args, "clipping_bound", None)
    if getattr(args, "defense_clip_norm", None) is None:
        args.defense_clip_norm = float(source_clip) if source_clip is not None else 1.0
    if hasattr(args, "clipping_bound") and source_clip is None:
        args.clipping_bound = float(args.defense_clip_norm)

    return args


def dpsgd_opacus_config(args) -> DPSGDOpacusConfig:
    normalize_dpsgd_opacus_args(args, active=True)
    return DPSGDOpacusConfig(
        noise_multiplier=float(args.defense_noise),
        max_grad_norm=float(args.defense_clip_norm),
        delta=float(args.defense_dp_delta),
    )


def import_privacy_engine():
    try:
        from opacus import PrivacyEngine
    except ImportError as exc:
        raise ImportError(
            "--defense dpsgd_opacus requires opacus. Install opacus==1.5.4 "
            "to run source-style DP-SGD."
        ) from exc
    return PrivacyEngine


def _wrap_mapping_dataset_for_opacus_empty_batches(data_loader):
    dataset = getattr(data_loader, "dataset", None)
    if dataset is None:
        return data_loader
    try:
        if len(dataset) == 0:
            return data_loader
        sample = dataset[0]
    except Exception:
        return data_loader
    if not isinstance(sample, Mapping):
        return data_loader

    wrapped_dataset = _OpacusTensorTupleDataset(dataset, sample.keys())
    wrapped_collator = _TupleToMappingCollator(sample.keys(), data_loader.collate_fn)
    return DataLoader(
        wrapped_dataset,
        batch_size=data_loader.batch_size,
        shuffle=False,
        collate_fn=wrapped_collator,
        drop_last=data_loader.drop_last,
        num_workers=data_loader.num_workers,
        pin_memory=data_loader.pin_memory,
        timeout=data_loader.timeout,
        worker_init_fn=data_loader.worker_init_fn,
        multiprocessing_context=data_loader.multiprocessing_context,
        generator=data_loader.generator,
        prefetch_factor=data_loader.prefetch_factor,
        persistent_workers=data_loader.persistent_workers,
    )


def make_private_with_opacus(module, optimizer, data_loader, args):
    cfg = dpsgd_opacus_config(args)
    PrivacyEngine = import_privacy_engine()
    privacy_engine = PrivacyEngine(accountant="rdp")
    module.train()
    data_loader = _wrap_mapping_dataset_for_opacus_empty_batches(data_loader)
    private_module, private_optimizer, private_loader = privacy_engine.make_private(
        module=module,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=cfg.noise_multiplier,
        max_grad_norm=cfg.max_grad_norm,
    )
    return privacy_engine, private_module, private_optimizer, private_loader


def freeze_position_embeddings(model) -> int:
    frozen = 0
    for name, param in model.named_parameters():
        path_parts = name.split(".")
        if "position_embeddings" in path_parts or "wpe" in path_parts:
            param.requires_grad = False
            frozen += 1
    return frozen


def capture_opacus_dager_layer_names(model, layer_ids) -> tuple[str, ...]:
    parameter_names = [
        name
        for name, param in model.named_parameters()
        if getattr(param, "requires_grad", False)
    ]
    if len(parameter_names) != len(set(parameter_names)):
        raise ValueError("Cannot capture Opacus DAGER layers: trainable parameter names are not unique.")

    target_names = []
    for raw_idx in layer_ids:
        idx = int(raw_idx)
        if idx < 0 or idx >= len(parameter_names):
            raise ValueError(
                f"Cannot capture Opacus DAGER layer index {idx}: "
                f"trainable parameter count is {len(parameter_names)}."
            )
        target_names.append(parameter_names[idx])
    if len(target_names) != len(set(target_names)):
        raise ValueError("Cannot capture Opacus DAGER layers: target parameter names are not unique.")
    return tuple(target_names)


def remap_opacus_dager_layer_ids(target_names, gradients, gradient_names) -> list[int]:
    target_names = tuple(target_names)
    gradient_names = tuple(gradient_names)
    if len(gradients) != len(gradient_names):
        raise ValueError(
            "Cannot remap Opacus DAGER layers: gradient/name count mismatch "
            f"({len(gradients)} gradients, {len(gradient_names)} names)."
        )
    if len(gradient_names) != len(set(gradient_names)):
        raise ValueError("Cannot remap Opacus DAGER layers: gradient parameter names are not unique.")
    if len(target_names) != len(set(target_names)):
        raise ValueError("Cannot remap Opacus DAGER layers: target parameter names are not unique.")

    indices_by_name = {name: idx for idx, name in enumerate(gradient_names)}
    missing = [name for name in target_names if name not in indices_by_name]
    if missing:
        raise ValueError(
            "Cannot remap Opacus DAGER layers; captured gradients are missing target "
            f"parameter(s): {', '.join(missing)}."
        )
    return [indices_by_name[name] for name in target_names]


@contextmanager
def use_effective_batch_size(args, batch_size: int):
    nominal_batch_size = args.batch_size
    args.batch_size = int(batch_size)
    try:
        yield
    finally:
        args.batch_size = nominal_batch_size


def is_empty_opacus_batch(batch) -> bool:
    if isinstance(batch, Mapping):
        tensors = [value for value in batch.values() if torch.is_tensor(value)]
        return bool(tensors) and any(tensor.nelement() == 0 for tensor in tensors)
    if isinstance(batch, (list, tuple)):
        tensors = [value for value in batch if torch.is_tensor(value)]
        return bool(tensors) and all(tensor.nelement() == 0 for tensor in tensors)
    return False


def count_nonempty_batches(loader) -> int:
    count = 0
    for batch in loader:
        if is_empty_opacus_batch(batch):
            continue
        count += 1
    return max(1, count)


def opacus_public_model(model):
    return getattr(model, "_module", model)


def get_epsilon_or_none(privacy_engine, delta: float):
    if privacy_engine is None:
        return None
    try:
        return float(privacy_engine.get_epsilon(float(delta)))
    except Exception:
        return None


def record_dpsgd_opacus_summary(args, tracker: dict, privacy_engine=None) -> None:
    cfg = dpsgd_opacus_config(args)
    tracker["dpsgd_noise_multiplier"] = cfg.noise_multiplier
    tracker["dpsgd_max_grad_norm"] = cfg.max_grad_norm
    tracker["dpsgd_delta"] = cfg.delta
    tracker["dpsgd_accountant"] = cfg.accountant
    tracker["dpsgd_epsilon"] = get_epsilon_or_none(privacy_engine, cfg.delta)


def dpsgd_opacus_summary_fields(args, tracker: dict | None = None):
    active = dpsgd_opacus_active(args) or getattr(args, "ptg_dpsgd_mode", None) == "source_opacus"
    if not active and not (tracker and tracker.get("dpsgd_accountant")):
        return []
    normalize_dpsgd_opacus_args(args, active=True)
    tracker = tracker or {}
    return [
        ("dpsgd_noise_multiplier", tracker.get("dpsgd_noise_multiplier", getattr(args, "defense_noise", None))),
        ("dpsgd_max_grad_norm", tracker.get("dpsgd_max_grad_norm", getattr(args, "defense_clip_norm", None))),
        ("dpsgd_delta", tracker.get("dpsgd_delta", getattr(args, "defense_dp_delta", None))),
        ("dpsgd_epsilon", tracker.get("dpsgd_epsilon")),
        ("dpsgd_accountant", tracker.get("dpsgd_accountant", DPSGD_OPACUS_ACCOUNTANT)),
    ]


def create_source_dpsgd_dataloader(args, tokenizer, *, shuffle: bool = False):
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    cache_dir = getattr(args, "cache_dir", None)
    if args.dataset in ["cola", "sst2", "rte"]:
        datasets = load_dataset("glue", args.dataset, cache_dir=cache_dir)
    elif args.dataset == "rotten_tomatoes":
        datasets = load_dataset(args.dataset, cache_dir=cache_dir)
    else:
        raise NotImplementedError(
            "dpsgd_opacus currently supports source-style datasets: "
            "cola, sst2, rte, rotten_tomatoes."
        )

    if args.dataset == "rotten_tomatoes":
        text_columns = ("text",)
    elif args.dataset == "rte":
        text_columns = ("sentence1", "sentence2")
    else:
        text_columns = ("sentence",)

    def tokenize_function(examples):
        if len(text_columns) == 2:
            return tokenizer(examples[text_columns[0]], examples[text_columns[1]], truncation=True)
        return tokenizer(examples[text_columns[0]], truncation=True)

    tokenized = datasets.map(tokenize_function, batched=True)
    keep_columns = {"label", "input_ids", "attention_mask", "token_type_ids"}
    remove_columns = [
        column for column in tokenized["train"].column_names if column not in keep_columns
    ]
    if remove_columns:
        tokenized = tokenized.remove_columns(remove_columns)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return DataLoader(tokenized["train"], shuffle=shuffle, batch_size=args.batch_size, collate_fn=collator)


def decode_batch_texts(tokenizer, batch) -> list[str]:
    return tokenizer.batch_decode(
        batch["input_ids"].detach().cpu().tolist(),
        skip_special_tokens=True,
    )


def capture_private_grads(private_model):
    names = []
    grads = []
    for name, param in private_model.named_parameters():
        if not getattr(param, "requires_grad", False):
            continue
        names.append(name.replace("_module.", ""))
        grads.append(None if param.grad is None else param.grad.detach().clone())
    return tuple(grads), names


def sync_private_model_to_public_model(private_model, public_model):
    state = opacus_public_model(private_model).state_dict()
    missing, unexpected = public_model.load_state_dict(state, strict=False)
    return missing, unexpected
