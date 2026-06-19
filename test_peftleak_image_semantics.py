#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attacks.peftleak_image.core import (
    build_public_patch_statistics,
    build_vit_adapter_gradients,
    cluster_and_reassemble,
    extract_image_patches,
    fold_image_patches,
    load_public_patch_statistics,
    move_patch_statistics,
    optimize_patch_baseline,
    recover_patch_from_adapter_grads,
    recover_patches_from_batch,
    recover_patches_from_named_adapter_grads,
    save_public_patch_statistics,
)
from attack_peftleak_image import (
    _defend_gradient_tuple,
    _make_raw_gradient_tuple,
    _recover_from_gradient_tuple,
    _run_synthetic_ratio,
    _run_vit_adapter,
    build_parser,
)


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def runtime_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_gradient_ratio_formula_recovers_known_patch():
    patch = torch.tensor([0.2, -0.5, 1.3, 0.7], dtype=torch.float32)
    bias_grad = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32)
    weight_grad = bias_grad.unsqueeze(1) * patch.unsqueeze(0)

    recovered = recover_patch_from_adapter_grads(weight_grad, bias_grad)

    assert_true(torch.allclose(recovered, patch, atol=1e-6), "adapter gradient-ratio formula should recover the patch")


def test_batch_recovery_applies_position_and_public_stats():
    patch = torch.tensor([0.1, 0.3, -0.2], dtype=torch.float32)
    pos = torch.tensor([0.5, -0.1, 0.2], dtype=torch.float32)
    mean = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    std = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32)
    exposed = (patch - mean) / std + pos
    bias_grad = torch.tensor([1.0, 2.0], dtype=torch.float32)
    weight_grad = bias_grad.unsqueeze(1) * exposed.unsqueeze(0)

    recovered = recover_patches_from_batch(
        [weight_grad],
        [bias_grad],
        position_embeddings=[pos],
        patch_mean=mean,
        patch_std=std,
    )

    assert_true(torch.allclose(recovered[0], patch, atol=1e-6), "recovery should invert position and public normalization")


def test_patch_extract_fold_roundtrip_and_public_stats():
    images = torch.arange(2 * 1 * 4 * 4, dtype=torch.float32).view(2, 1, 4, 4) / 32.0
    patches = extract_image_patches(images, patch_size=2)
    folded = fold_image_patches(patches, channels=1, grid_shape=(2, 2), patch_size=2)
    stats = build_public_patch_statistics(images, patch_size=2)

    assert_true(torch.allclose(folded, images), "patch extraction/folding should round-trip")
    assert_true(stats.num_images == 2, "public stats should record image count")
    assert_true(stats.num_patches == 8, "public stats should record patch count")


def test_public_stats_file_roundtrip_uses_explicit_file():
    images = torch.linspace(0, 1, steps=2 * 1 * 4 * 4, dtype=torch.float32).view(2, 1, 4, 4)
    stats = build_public_patch_statistics(images, patch_size=2)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "public_stats.pt")
        save_public_patch_statistics(stats, path)
        loaded = load_public_patch_statistics(path)
    assert_true(torch.allclose(loaded.mean, stats.mean), "public stats mean should round-trip through file")
    assert_true(torch.allclose(loaded.std, stats.std), "public stats std should round-trip through file")
    assert_true(loaded.num_patches == stats.num_patches, "public stats metadata should round-trip")


def test_public_stats_move_preserves_metadata_and_device():
    device = runtime_device()
    images = torch.rand(2, 1, 4, 4)
    stats = build_public_patch_statistics(images, patch_size=2)

    moved = move_patch_statistics(stats, device=device, dtype=torch.float32)

    assert_true(moved.mean.device == device, "public stats mean should move to runtime device")
    assert_true(moved.std.device == device, "public stats std should move to runtime device")
    assert_true(moved.num_images == stats.num_images, "public stats image count should be preserved")
    assert_true(moved.num_patches == stats.num_patches, "public stats patch count should be preserved")


def test_vit_adapter_autograd_gradients_recover_patches():
    device = runtime_device()
    attack_images = torch.rand(2, 1, 4, 4, device=device)
    public_images = torch.rand(3, 1, 4, 4, device=device)
    stats = build_public_patch_statistics(public_images, patch_size=2)
    result = build_vit_adapter_gradients(
        attack_images,
        stats,
        patch_size=2,
        adapter_hidden_dim=3,
        n_classes=4,
        seed=11,
    )
    recovered = recover_patches_from_named_adapter_grads(
        result.grads,
        result.names,
        batch_size=2,
        n_patches=4,
        patch_mean=stats.mean,
        patch_std=stats.std,
    )
    reference = extract_image_patches(attack_images, patch_size=2)
    assert_true(len(result.names) == 2 * 4 * 2, "ViT adapter path should expose weight/bias gradients per patch")
    assert_true(
        all(grad is None or grad.device == device for grad in result.grads),
        "ViT adapter gradients should stay on the runtime device",
    )
    assert_true(result.normalized_patches.device == device, "normalized patches should stay on the runtime device")
    assert_true(result.raw_patches.device == device, "raw patches should stay on the runtime device")
    assert_true(recovered.device == device, "recovered patches should stay on the runtime device")
    assert_true(torch.allclose(recovered, reference, atol=1e-5), "autograd adapter gradients should recover patches")


def test_vit_adapter_run_reports_loss_acc_and_device_local_outputs():
    device = runtime_device()
    parser = build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "vit_adapter",
            "--dataset",
            "synthetic",
            "--n_images",
            "2",
            "--image_size",
            "4",
            "--channels",
            "1",
            "--n_classes",
            "4",
            "--patch_size",
            "2",
            "--adapter_hidden_dim",
            "3",
            "--device",
            str(device),
        ]
    )
    images = torch.rand(2, 1, 4, 4, device=device)
    labels = torch.tensor([0, 1], dtype=torch.long, device=device)
    stats = move_patch_statistics(build_public_patch_statistics(torch.rand(3, 1, 4, 4, device=device), 2), device=device)

    recovered, reference, loss, top1, grad_count = _run_vit_adapter(args, images, labels, stats)

    assert_true(recovered.device == device, "run_vit_adapter should return recovered patches on runtime device")
    assert_true(reference.device == device, "run_vit_adapter should return reference patches on runtime device")
    assert_true(loss is not None and loss >= 0.0, "run_vit_adapter should report a finite loss")
    assert_true(top1 is not None and 0.0 <= top1 <= 1.0, "run_vit_adapter should report batch top-1 accuracy")
    assert_true(grad_count == 2 * 4 * 2, "run_vit_adapter should report weight/bias gradients per patch")


def test_clustering_reassembly_is_deterministic_for_fixed_seed():
    patches = torch.tensor(
        [
            [[0.1, 0.2, 0.3, 0.4], [0.9, 0.8, 0.7, 0.6], [0.4, 0.5, 0.6, 0.7], [0.2, 0.1, 0.0, -0.1]],
        ],
        dtype=torch.float32,
    )

    img_a, assign_a = cluster_and_reassemble(patches, channels=1, grid_shape=(2, 2), patch_size=2, seed=13)
    img_b, assign_b = cluster_and_reassemble(patches, channels=1, grid_shape=(2, 2), patch_size=2, seed=13)

    assert_true(torch.equal(assign_a, assign_b), "fixed seed should produce deterministic cluster assignments")
    assert_true(torch.allclose(img_a, img_b), "fixed seed should produce deterministic reassembly")


def test_clustered_reassembly_changes_patch_order_for_unsorted_inputs():
    patches = torch.tensor(
        [
            [[10.0, 10.0, 10.0, 10.0], [1.0, 1.0, 1.0, 1.0], [7.0, 7.0, 7.0, 7.0], [3.0, 3.0, 3.0, 3.0]],
        ],
        dtype=torch.float32,
    )
    clustered, assignments = cluster_and_reassemble(patches, channels=1, grid_shape=(2, 2), patch_size=2, seed=0)
    raw_fold = fold_image_patches(patches, channels=1, grid_shape=(2, 2), patch_size=2)

    assert_true(clustered.shape == raw_fold.shape, "clustered image should preserve shape")
    assert_true(not torch.allclose(clustered, raw_fold), "clustered output should differ from raw patch order for unsorted inputs")
    assert_true(assignments.shape == (1, 4), "cluster assignments should cover all patches")


def test_clustering_reassembly_preserves_device():
    device = runtime_device()
    patches = torch.rand(1, 4, 4, device=device)

    clustered, assignments = cluster_and_reassemble(patches, channels=1, grid_shape=(2, 2), patch_size=2, seed=3)

    assert_true(clustered.device == device, "clustered image should stay on runtime device")
    assert_true(assignments.device == device, "cluster assignments should stay on runtime device")


def test_synthetic_ratio_is_exact_without_defense_and_device_local():
    device = runtime_device()
    parser = build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "synthetic_ratio",
            "--dataset",
            "synthetic",
            "--n_images",
            "2",
            "--image_size",
            "4",
            "--channels",
            "1",
            "--patch_size",
            "2",
            "--adapter_hidden_dim",
            "3",
            "--device",
            str(device),
        ]
    )
    images = torch.rand(2, 1, 4, 4, device=device)

    recovered, reference, loss, top1, grad_count = _run_synthetic_ratio(args, images)

    assert_true(loss is None and top1 is None, "synthetic_ratio should not report ViT utility metrics")
    assert_true(grad_count == 2 * 4 * 2, "synthetic_ratio should expose weight/bias gradients per patch")
    assert_true(recovered.device == device and reference.device == device, "synthetic_ratio outputs should stay on runtime device")
    assert_true(torch.allclose(recovered, reference, atol=1e-6), "synthetic_ratio should exactly recover patches without defense")


def test_image_defense_branches_return_device_local_gradients():
    device = runtime_device()
    parser = build_parser()
    defense_args = {
        "none": [],
        "noise": ["--defense_noise", "1e-5"],
        "dpsgd": ["--defense_noise", "1e-5"],
        "topk": ["--defense_topk_ratio", "0.5"],
        "compression": ["--defense_n_bits", "8"],
        "soteria": [],
        "mixup": ["--defense_mixup_alpha", "1.0"],
        "lrb": ["--defense_lrb_keep_ratio_sensitive", "0.8"],
        "lrbprojonly": ["--defense_lrb_keep_ratio_sensitive", "0.8"],
        "signed_bottleneck": ["--defense_lrb_preset", "signed_bottleneck", "--defense_lrb_keep_ratio_sensitive", "0.99"],
    }
    patches = torch.rand(2, 4, 4, device=device)
    raw_grads, names = _make_raw_gradient_tuple(patches, hidden_dim=3)

    for defense, extra in defense_args.items():
        args = parser.parse_args(["--dataset", "synthetic", "--defense", defense, *extra])
        defended, defended_names = _defend_gradient_tuple(args, raw_grads, names)
        assert_true(defended_names == names, f"{defense} should preserve gradient names")
        assert_true(
            all(grad is None or grad.device == device for grad in defended),
            f"{defense} should keep defended gradients on runtime device",
        )
        if defense in {"none", "noise", "dpsgd", "topk", "compression"}:
            recovered = _recover_from_gradient_tuple(defended)
            assert_true(recovered.device == device, f"{defense} recovered patches should stay on runtime device")


def test_optimization_patch_baseline_loss_decreases():
    patch = torch.tensor([0.3, -0.7, 0.4, 1.2], dtype=torch.float32)
    bias_grad = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    weight_grad = bias_grad.unsqueeze(1) * patch.unsqueeze(0)

    recovered, history = optimize_patch_baseline(weight_grad, bias_grad, steps=40, lr=0.2, seed=5)

    assert_true(history[-1] < history[0], "optimization baseline should reduce gradient matching loss")
    assert_true(torch.isfinite(recovered).all(), "optimized patch should stay finite")


def main():
    tests = [
        test_gradient_ratio_formula_recovers_known_patch,
        test_batch_recovery_applies_position_and_public_stats,
        test_patch_extract_fold_roundtrip_and_public_stats,
        test_public_stats_file_roundtrip_uses_explicit_file,
        test_public_stats_move_preserves_metadata_and_device,
        test_vit_adapter_autograd_gradients_recover_patches,
        test_vit_adapter_run_reports_loss_acc_and_device_local_outputs,
        test_clustering_reassembly_is_deterministic_for_fixed_seed,
        test_clustered_reassembly_changes_patch_order_for_unsorted_inputs,
        test_clustering_reassembly_preserves_device,
        test_synthetic_ratio_is_exact_without_defense_and_device_local,
        test_image_defense_branches_return_device_local_gradients,
        test_optimization_patch_baseline_loss_decreases,
    ]
    for test in tests:
        print(f"Running {test.__name__}...")
        test()
    print("All PEFTLeak image semantic tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

