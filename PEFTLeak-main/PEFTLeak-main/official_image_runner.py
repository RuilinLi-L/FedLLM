#!/usr/bin/env python3
"""CLI runner for the official PEFTLeak CIFAR100 image-adapter notebook path.

This script intentionally mirrors Adapter_attack.ipynb for the clean baseline.
Defense hooks are present but only ``none`` is enabled in this first phase.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import os
from pathlib import Path
import random
import time

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
import torchvision  # noqa: E402
from torchvision import transforms  # noqa: E402


SUMMARY_START = "===== RESULT SUMMARY START ====="
SUMMARY_END = "===== RESULT SUMMARY END ====="

IMG_SIZE = 32
PATCH_SIZE = 16
CHANNELS = 3
NUM_CLASSES = 100
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
PATCH_DIM = (PATCH_SIZE**2) * CHANNELS
EMBED_DIM = PATCH_DIM
NUM_HEADS = 12
HEAD_DIM = EMBED_DIM // NUM_HEADS
DROPOUT = 0
ADAPTER_R = 64
NUM_BINS = 320
POSITION_ZERO_DIM = 20
COEFF = 0.5
MULTIPLIER = 100
SCALE = 1
HIGH = 10000
GAP = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cifar100"], default="cifar100")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--img_list_path", default="img_list.npy")
    parser.add_argument("--public_split", choices=["test", "train"], default="test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument(
        "--defense",
        choices=["none", "topk", "compression", "proj_only", "full_lrb"],
        default="none",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="outputs/peftleak_official_image")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--patch_recovery_mse_threshold", type=float, default=1e-6)
    return parser


def fmt_summary_value(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "n/a"
        return f"{value:.6f}"
    return str(value)


def emit_summary(fields: dict):
    ordered_keys = [
        "result_status",
        "attack",
        "dataset",
        "model",
        "seed",
        "device",
        "data_root",
        "img_list_path",
        "public_split",
        "batch_size",
        "attack_index_count",
        "defense",
        "patch_count",
        "candidate_patch_count",
        "clustered_image_count",
        "patch_recovery_count",
        "patch_recovery_rate",
        "image_metric_scope",
        "mse",
        "ssim",
        "diagnostic_mse",
        "diagnostic_ssim",
        "diagnostic_image_metric_status",
        "lpips_status",
        "loss",
        "runtime",
        "error_type",
        "error_message",
    ]
    print(SUMMARY_START, flush=True)
    for key in ordered_keys:
        if key in fields:
            print(f"{key}={fmt_summary_value(fields.get(key))}", flush=True)
    print(SUMMARY_END, flush=True)


def resolve_existing_path(path_value: str | os.PathLike, *, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_file():
        return path
    if not path.is_absolute():
        candidate = base_dir / path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"File does not exist: {path_value}")


def set_reproducible_seed(seed: int, device: torch.device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_cifar100(args):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((mean[0], mean[1], mean[2]), (std[0], std[1], std[2])),
        ]
    )
    data_root = Path(args.data_root)
    trainset = torchvision.datasets.CIFAR100(
        root=str(data_root),
        train=True,
        download=bool(args.download),
        transform=transform,
    )
    public_train = args.public_split == "train"
    publicset = torchvision.datasets.CIFAR100(
        root=str(data_root),
        train=public_train,
        download=bool(args.download),
        transform=transform,
    )
    return trainset, publicset, torch.tensor(mean).reshape(3, 1, 1), torch.tensor(std).reshape(3, 1, 1)


def load_attack_images(trainset, img_list_path: Path, batch_size: int):
    indices = np.load(str(img_list_path), allow_pickle=False).astype(np.int64).tolist()
    if len(indices) < batch_size:
        raise ValueError(f"{img_list_path} contains {len(indices)} indices but batch_size={batch_size}.")
    images = []
    labels = []
    for idx in indices[:batch_size]:
        if idx < 0 or idx >= len(trainset):
            raise ValueError(f"Attack index {idx} is outside CIFAR100 train range [0, {len(trainset)}).")
        image, label = trainset[int(idx)]
        images.append(image)
        labels.append(int(label))
    return torch.stack(images), torch.tensor(labels, dtype=torch.long), indices[:batch_size]


def configure_position_embeddings(w_glob: dict[str, torch.Tensor]):
    w_glob["patch.position_embeddings"][:, :, 0:POSITION_ZERO_DIM] = torch.zeros(
        w_glob["patch.position_embeddings"][:, :, 0:POSITION_ZERO_DIM].size()
    )
    for idx in range(w_glob["patch.position_embeddings"].shape[1]):
        tail = w_glob["patch.position_embeddings"][0][idx][POSITION_ZERO_DIM:]
        mean1 = torch.mean(tail)
        std1 = torch.std(tail)
        w_glob["patch.position_embeddings"][0][idx][POSITION_ZERO_DIM:] = 10 * (tail - mean1) / std1


def build_public_bins(publicset, w_glob: dict[str, torch.Tensor], E: torch.Tensor):
    from processing_v2 import approx_distribution, get_bias_list

    biases_patch = [[]] * (NUM_PATCHES + 1)
    biases_patch[0] = torch.zeros(NUM_BINS)
    distn = approx_distribution(dataset=publicset)
    coordinates = [
        (0, 16, 0, 16),
        (0, 16, 16, 32),
        (16, 32, 0, 16),
        (16, 32, 16, 32),
    ]
    means = []
    sigmas = []
    for patch_id, (row1, row2, col1, col2) in enumerate(coordinates, start=1):
        mean, sigma = distn.get_mean_std3(
            SCALE,
            row1,
            row2,
            col1,
            col2,
            w_glob["patch.position_embeddings"][0][patch_id],
            E,
            MULTIPLIER,
        )
        means.append(mean)
        sigmas.append(sigma)
    for patch_id, (mean, sigma) in enumerate(zip(means, sigmas), start=1):
        get_bias = get_bias_list(mean, sigma)
        biases_patch[patch_id], _bin_width = get_bias.create_bins(NUM_BINS)
    return biases_patch


def apply_malicious_design(model: ViT, publicset) -> dict[str, torch.Tensor]:
    from Design_Model_Adapter import Design

    w_glob = model.state_dict()
    E = COEFF * torch.eye(PATCH_DIM)
    configure_position_embeddings(w_glob)
    biases_patch = build_public_bins(publicset, w_glob, E)

    W1 = torch.zeros(ADAPTER_R, EMBED_DIM)
    W1[:, :ADAPTER_R] = torch.eye(ADAPTER_R)
    W2 = torch.zeros(EMBED_DIM, ADAPTER_R)
    W2[0:ADAPTER_R] = W1[:, :ADAPTER_R].T

    malicious_model = Design(
        ADAPTER_R,
        GAP,
        POSITION_ZERO_DIM,
        NUM_BINS,
        EMBED_DIM,
        PATCH_DIM,
        NUM_HEADS,
        NUM_PATCHES,
    )
    w_glob["patch.fc1.weight"] = malicious_model.linear_embed(w_glob["patch.fc1.weight"], COEFF)
    w_glob["encoder1.attn.QKV.weight"] = malicious_model.first_encoder(w_glob["encoder1.attn.QKV.weight"])
    w_glob["encoder1.attn.adapt1.weight"], w_glob["encoder1.attn.adapt2.weight"] = malicious_model.first_adapter(
        w_glob["encoder1.attn.adapt1.weight"],
        w_glob["encoder1.attn.adapt2.weight"],
        W1,
        W2,
    )
    w_glob["encoder1.attn.adapt1.bias"] = HIGH * torch.ones(w_glob["encoder1.attn.adapt1.bias"].size())
    w_glob["encoder1.attn.adapt2.bias"] = -W2 @ w_glob["encoder1.attn.adapt1.bias"]
    target = 1
    w_glob["encoder1.attn.QKV.bias"][0:HEAD_DIM] = (
        10**6 * w_glob["patch.position_embeddings"][:, target, 0:HEAD_DIM].reshape(HEAD_DIM)
    )
    w_glob["encoder1.attn.QKV.bias"][HEAD_DIM:] = torch.zeros(w_glob["encoder1.attn.QKV.bias"][HEAD_DIM:].size())
    w_glob["encoder1.attn.msa.weight"] = 4 * torch.eye(EMBED_DIM)

    start = 0
    r2 = start
    patch_id = 1
    count = 0
    for encoder_idx in range(1, 12):
        key1 = f"encoder{encoder_idx}.mlp.fc1.weight"
        w_glob[key1] = malicious_model.mlp_identity(w_glob[key1])

        key2 = f"encoder{encoder_idx}.mlp.fc2.weight"
        w_glob[key2] = malicious_model.mlp_identity(w_glob[key2])

        key3 = f"encoder{encoder_idx}.mlp.fc1.bias"
        w_glob[key3] = torch.zeros(w_glob[key3].size())
        w_glob[key3][0:EMBED_DIM] = HIGH * torch.ones(w_glob[key3][0:EMBED_DIM].size())

        key4 = f"encoder{encoder_idx}.mlp.fc2.bias"
        w_glob[key4] = -HIGH * torch.ones(w_glob[key4].size())

        e_pos_std = torch.std(w_glob["patch.position_embeddings"][:, 2])
        key5 = f"encoder{encoder_idx}.attn.LN1.weight"
        w_glob[key5] = e_pos_std * torch.ones(w_glob[key5].size())

        key6 = f"encoder{encoder_idx}.mlp.LN2.weight"
        w_glob[key6] = e_pos_std * torch.ones(w_glob[key6].size())

        key7 = f"encoder{encoder_idx}.mlp.adapt1.weight"
        w_glob[key7] = malicious_model.adapter(
            w_glob[key7],
            w_glob["patch.position_embeddings"][0],
            patch_id,
            MULTIPLIER,
        )
        w_glob[key7][:, 0:64] = 0

        key8 = f"encoder{encoder_idx}.mlp.adapt2.weight"
        w_glob[key8] = torch.zeros(w_glob[key8].size())
        w_glob[key8][-1] = 10 ** (-6) / w_glob[key8].shape[1]

        key9 = f"encoder{encoder_idx}.mlp.adapt2.bias"
        w_glob[key9] = torch.zeros(w_glob[key9].size())

        if encoder_idx > 1:
            key10 = f"encoder{encoder_idx}.attn.adapt1.weight"
            w_glob[key10] = malicious_model.adapter(
                w_glob[key10],
                w_glob["patch.position_embeddings"][0],
                patch_id,
                MULTIPLIER,
            )
            w_glob[key10][:, 0:64] = 0

            key11 = f"encoder{encoder_idx}.attn.adapt2.weight"
            w_glob[key11] = torch.zeros(w_glob[key11].size())
            w_glob[key11][-1] = 10 ** (-6) / w_glob[key11].shape[1]

            key12 = f"encoder{encoder_idx}.attn.adapt2.bias"
            w_glob[key12] = torch.zeros(w_glob[key12].size())

            key13 = f"encoder{encoder_idx}.attn.QKV.weight"
            w_glob[key13] = malicious_model.attention(w_glob[key13])

            key14 = f"encoder{encoder_idx}.attn.QKV.bias"
            w_glob[key14] = torch.zeros(w_glob[key14].size())

            key15 = f"encoder{encoder_idx}.attn.msa.weight"
            w_glob[key15] = torch.eye(EMBED_DIM)

            r1 = r2
            r2 += ADAPTER_R - GAP
            key18 = f"encoder{encoder_idx}.attn.adapt1.weight"
            key19 = f"encoder{encoder_idx}.attn.adapt1.bias"
            w_glob[key19][GAP:ADAPTER_R] = (
                -w_glob[key18][GAP].T @ (w_glob["patch.position_embeddings"][0][patch_id])
                - biases_patch[patch_id][r1:r2]
            )
            count += 1
            if count == 5:
                if patch_id < 4:
                    patch_id += 1
                r2 = start
                count = 0

        r1 = r2
        r2 += ADAPTER_R - GAP
        key16 = f"encoder{encoder_idx}.mlp.adapt1.weight"
        key17 = f"encoder{encoder_idx}.mlp.adapt1.bias"
        w_glob[key17][GAP:ADAPTER_R] = (
            -w_glob[key16][GAP].T @ (w_glob["patch.position_embeddings"][0][patch_id])
            - biases_patch[patch_id][r1:r2]
        )

        count += 1
        if count == 5:
            if patch_id < 4:
                patch_id += 1
            r2 = start
            count = 0
    return w_glob


def set_adapter_grads_only(model: ViT):
    for _name, param in model.named_parameters():
        param.requires_grad = "adapt1" in _name or "adapt2" in _name


def collect_adapter_gradients(model: ViT):
    weight_names = ["encoder1.mlp.adapt1.weight"]
    bias_names = ["encoder1.mlp.adapt1.bias"]
    for idx in range(2, 12):
        weight_names.extend([f"encoder{idx}.attn.adapt1.weight", f"encoder{idx}.mlp.adapt1.weight"])
        bias_names.extend([f"encoder{idx}.attn.adapt1.bias", f"encoder{idx}.mlp.adapt1.bias"])
    weight_name_set = set(weight_names)
    bias_name_set = set(bias_names)

    weight_grad = []
    bias_grad = []
    for name, param in model.named_parameters():
        if name in weight_name_set:
            weight_grad.append(param.grad.detach().cpu() if param.grad is not None else None)
        if name in bias_name_set:
            bias_grad.append(param.grad.detach().cpu() if param.grad is not None else None)
    if any(grad is None for grad in weight_grad + bias_grad):
        missing = [
            name
            for name, grad in zip(weight_names + bias_names, weight_grad + bias_grad)
            if grad is None
        ]
        raise RuntimeError(f"Missing official adapter gradients: {missing[:5]}")
    return weight_grad, bias_grad, weight_names, bias_names


def apply_official_image_defense(weight_grad, bias_grad, args):
    if args.defense != "none":
        raise NotImplementedError(
            "official_image_runner phase 1 only supports --defense none. "
            "Add topk/compression/proj_only/full_lrb after the official clean baseline passes."
        )
    return weight_grad, bias_grad


def reconstruct_patch_candidates(weight_grad, bias_grad, patch_id: int, w_pos: torch.Tensor, scale: float):
    """Notebook-equivalent adapter recovery without the fixed 4x8 plotting grid."""
    candidates = []
    with torch.no_grad():
        for block_idx in range(len(weight_grad)):
            for row_idx in range(GAP, len(weight_grad[block_idx]) - 1):
                denom = bias_grad[block_idx][row_idx] - bias_grad[block_idx][row_idx + 1]
                if float(denom.detach().cpu().item()) == 0.0:
                    continue
                rec = (weight_grad[block_idx][row_idx] - weight_grad[block_idx][row_idx + 1]) / denom
                rec = rec - w_pos[patch_id]
                rec = torch.div(rec, scale)
                candidates.append(rec.reshape(3, PATCH_SIZE, PATCH_SIZE).clone().detach())
    return candidates


def recover_candidate_patches(weight_grad, bias_grad, w_glob, mean_inv, std_inv):
    del mean_inv, std_inv
    w_pos = w_glob["patch.position_embeddings"][0]
    return [
        reconstruct_patch_candidates(weight_grad[0:5], bias_grad[0:5], 1, w_pos, COEFF),
        reconstruct_patch_candidates(weight_grad[5:10], bias_grad[5:10], 2, w_pos, COEFF),
        reconstruct_patch_candidates(weight_grad[10:15], bias_grad[10:15], 3, w_pos, COEFF),
        reconstruct_patch_candidates(weight_grad[15:20], bias_grad[15:20], 4, w_pos, COEFF),
    ]


def reference_patches(images: torch.Tensor) -> torch.Tensor:
    patches = []
    coords = [
        (0, 16, 0, 16),
        (0, 16, 16, 32),
        (16, 32, 0, 16),
        (16, 32, 16, 32),
    ]
    for row1, row2, col1, col2 in coords:
        patches.append(images[:, :, row1:row2, col1:col2])
    return torch.stack(patches, dim=1)


def count_patch_recovery(candidate_by_position, refs: torch.Tensor, threshold: float) -> int:
    exact = 0
    for position_idx, candidates in enumerate(candidate_by_position):
        if not candidates:
            continue
        used_refs = set()
        ref = refs[:, position_idx].float()
        for candidate in candidates:
            cand = candidate.detach().cpu().float().view(1, 3, PATCH_SIZE, PATCH_SIZE)
            mse = (ref - cand).pow(2).mean(dim=(1, 2, 3))
            order = torch.argsort(mse).tolist()
            for ref_idx in order:
                if ref_idx in used_refs:
                    continue
                if float(mse[ref_idx].item()) < threshold:
                    used_refs.add(ref_idx)
                    exact += 1
                break
    return int(exact)


def fold_candidate_image(patches: list[torch.Tensor]) -> torch.Tensor:
    image = torch.zeros(3, IMG_SIZE, IMG_SIZE, dtype=patches[0].dtype)
    image[:, 0:16, 0:16] = patches[0]
    image[:, 0:16, 16:32] = patches[1]
    image[:, 16:32, 0:16] = patches[2]
    image[:, 16:32, 16:32] = patches[3]
    return image


def simple_ssim(image: torch.Tensor, reference: torch.Tensor) -> float:
    x = image.float()
    y = reference.float()
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = x.mean()
    mu_y = y.mean()
    var_x = ((x - mu_x) ** 2).mean()
    var_y = ((y - mu_y) ** 2).mean()
    cov = ((x - mu_x) * (y - mu_y)).mean()
    return float(((2 * mu_x * mu_y + c1) * (2 * cov + c2) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))).item())


def cluster_and_score_images(candidate_by_position, refs: torch.Tensor, mean_inv: torch.Tensor, std_inv: torch.Tensor):
    try:
        from Clustering import match_cluster
    except Exception as exc:  # noqa: BLE001 - image-level scoring is diagnostic only.
        return None, None, 0, f"unavailable:{type(exc).__name__}:{exc}"

    cluster_comp = []
    cluster_positions = []
    for position_idx, candidates in enumerate(candidate_by_position):
        for candidate in candidates:
            cluster_comp.append(candidate.reshape(PATCH_DIM)[0:POSITION_ZERO_DIM])
            cluster_positions.append(position_idx)
    if not cluster_comp:
        return None, None, 0, "unavailable:no_candidates"

    try:
        labels = match_cluster(torch.stack(cluster_comp), refs.shape[0], NUM_PATCHES)
    except Exception as exc:  # noqa: BLE001 - keep patch recovery reportable.
        return None, None, 0, f"unavailable:{type(exc).__name__}:{exc}"

    cluster_patches: dict[int, dict[int, torch.Tensor]] = {}
    for label, position_idx, candidate in zip(labels.tolist(), cluster_positions, [c for group in candidate_by_position for c in group]):
        cluster_patches.setdefault(int(label), {})
        if position_idx not in cluster_patches[int(label)]:
            cluster_patches[int(label)][position_idx] = candidate.detach().cpu()

    reconstructed = []
    for patch_map in cluster_patches.values():
        if all(idx in patch_map for idx in range(NUM_PATCHES)):
            reconstructed.append(fold_candidate_image([patch_map[idx] for idx in range(NUM_PATCHES)]))
    if not reconstructed:
        return None, None, 0, "unavailable:no_complete_cluster"

    rec = torch.stack(reconstructed).float()
    ref_images = torch.zeros(refs.shape[0], 3, IMG_SIZE, IMG_SIZE)
    for idx in range(refs.shape[0]):
        ref_images[idx] = fold_candidate_image([refs[idx, pos] for pos in range(NUM_PATCHES)])

    mean = mean_inv.float()
    std = std_inv.float()
    rec_denorm = (rec * std + mean).clamp(0, 1)
    ref_denorm = (ref_images.float() * std + mean).clamp(0, 1)

    used = set()
    mses = []
    ssims = []
    for image in rec_denorm:
        mse_by_ref = (ref_denorm - image.unsqueeze(0)).pow(2).mean(dim=(1, 2, 3))
        order = torch.argsort(mse_by_ref).tolist()
        chosen = next((idx for idx in order if idx not in used), order[0])
        used.add(chosen)
        mses.append(float(mse_by_ref[chosen].item()))
        ssims.append(simple_ssim(image, ref_denorm[chosen]))
    return float(np.mean(mses)), float(np.mean(ssims)), int(len(reconstructed)), "diagnostic_clustered_first_candidate"


def run(args) -> dict:
    from Transformer_Model_neuron import ViT

    script_dir = Path(__file__).resolve().parent
    img_list_path = resolve_existing_path(args.img_list_path, base_dir=script_dir)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is not available.")
    set_reproducible_seed(int(args.seed), device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    trainset, publicset, mean_inv, std_inv = load_cifar100(args)
    images, labels, attack_indices = load_attack_images(trainset, img_list_path, int(args.batch_size))

    model = ViT(
        ADAPTER_R,
        EMBED_DIM,
        PATCH_DIM,
        PATCH_SIZE,
        NUM_PATCHES,
        NUM_HEADS,
        HEAD_DIM,
        DROPOUT,
        CHANNELS,
        NUM_CLASSES,
    )
    w_glob = apply_malicious_design(model, publicset)
    model.load_state_dict(w_glob)
    model = model.to(device)
    set_adapter_grads_only(model)

    data = images.reshape(int(args.batch_size), 1, 3, IMG_SIZE, IMG_SIZE).to(device)
    y = labels.to(device)
    criterion = nn.CrossEntropyLoss()
    model.train()
    model.zero_grad()
    output = model(data)
    loss = criterion(output, y)
    loss.backward()

    weight_grad, bias_grad, _weight_names, _bias_names = collect_adapter_gradients(model)
    weight_grad, bias_grad = apply_official_image_defense(weight_grad, bias_grad, args)
    candidate_by_position = recover_candidate_patches(weight_grad, bias_grad, w_glob, mean_inv, std_inv)
    refs = reference_patches(images)
    candidate_patch_count = sum(len(candidates) for candidates in candidate_by_position)
    patch_count = int(args.batch_size) * NUM_PATCHES
    patch_recovery_count = count_patch_recovery(candidate_by_position, refs, float(args.patch_recovery_mse_threshold))
    diagnostic_mse, diagnostic_ssim, clustered_image_count, image_metric_status = cluster_and_score_images(
        candidate_by_position,
        refs,
        mean_inv,
        std_inv,
    )

    return {
        "result_status": "ok",
        "attack": "peftleak_official_image",
        "dataset": args.dataset,
        "model": "official_custom_vit",
        "seed": int(args.seed),
        "device": str(device),
        "data_root": args.data_root,
        "img_list_path": str(img_list_path),
        "public_split": args.public_split,
        "batch_size": int(args.batch_size),
        "attack_index_count": len(attack_indices),
        "defense": args.defense,
        "patch_count": patch_count,
        "candidate_patch_count": candidate_patch_count,
        "clustered_image_count": clustered_image_count,
        "patch_recovery_count": patch_recovery_count,
        "patch_recovery_rate": patch_recovery_count / max(1, patch_count),
        "image_metric_scope": "patch_recovery_primary",
        "mse": None,
        "ssim": None,
        "diagnostic_mse": diagnostic_mse,
        "diagnostic_ssim": diagnostic_ssim,
        "diagnostic_image_metric_status": image_metric_status,
        "lpips_status": "unavailable",
        "loss": float(loss.detach().cpu().item()),
    }


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    start = time.time()
    try:
        fields = run(args)
        fields["runtime"] = str(_datetime.timedelta(seconds=time.time() - start)).split(".")[0]
        emit_summary(fields)
        return 0
    except Exception as exc:  # noqa: BLE001 - emit CLI-friendly failed summary
        emit_summary(
            {
                "result_status": "failed",
                "attack": "peftleak_official_image",
                "dataset": getattr(args, "dataset", None),
                "model": "official_custom_vit",
                "seed": getattr(args, "seed", None),
                "device": getattr(args, "device", None),
                "data_root": getattr(args, "data_root", None),
                "img_list_path": getattr(args, "img_list_path", None),
                "public_split": getattr(args, "public_split", None),
                "batch_size": getattr(args, "batch_size", None),
                "defense": getattr(args, "defense", None),
                "runtime": str(_datetime.timedelta(seconds=time.time() - start)).split(".")[0],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
