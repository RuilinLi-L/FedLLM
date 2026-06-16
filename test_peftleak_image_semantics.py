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
    optimize_patch_baseline,
    recover_patch_from_adapter_grads,
    recover_patches_from_batch,
    recover_patches_from_named_adapter_grads,
    save_public_patch_statistics,
)


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


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


def test_vit_adapter_autograd_gradients_recover_patches():
    attack_images = torch.rand(2, 1, 4, 4)
    public_images = torch.rand(3, 1, 4, 4)
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
    assert_true(torch.allclose(recovered, reference, atol=1e-5), "autograd adapter gradients should recover patches")


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
        test_vit_adapter_autograd_gradients_recover_patches,
        test_clustering_reassembly_is_deterministic_for_fixed_seed,
        test_clustered_reassembly_changes_patch_order_for_unsorted_inputs,
        test_optimization_patch_baseline_loss_decreases,
    ]
    for test in tests:
        print(f"Running {test.__name__}...")
        test()
    print("All PEFTLeak image semantic tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

