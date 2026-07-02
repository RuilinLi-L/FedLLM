#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attacks.peftleak_image.core import (
    PatchStatistics,
    PeftLeakProbeStatistics,
    build_peftleak_probe_statistics,
    build_public_patch_statistics,
    build_shared_adapter_gradient_bundle,
    build_shared_adapter_gradients,
    cluster_and_reassemble,
    extract_image_patches,
    fold_image_patches,
    load_public_patch_statistics,
    move_patch_statistics,
    optimize_patch_baseline,
    recover_patch_from_adapter_grads,
    recover_patches_from_batch,
    recover_patches_from_shared_adapter_grads,
    save_public_patch_statistics,
)
from attack_peftleak_image import (
    SUMMARY_END,
    SUMMARY_START,
    SyntheticFallbackError,
    _defend_gradient_tuple,
    _make_raw_gradient_tuple,
    _recover_from_gradient_tuple,
    _run_synthetic_ratio,
    _run_vit_adapter,
    build_parser,
    main as image_main,
)


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def runtime_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_result_summary(output: str) -> dict[str, str]:
    lines = output.splitlines()
    try:
        start = lines.index(SUMMARY_START)
        end = lines.index(SUMMARY_END)
    except ValueError as exc:
        raise AssertionError(f"Missing result summary in output:\n{output}") from exc
    fields = {}
    for line in lines[start + 1 : end]:
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value
    return fields


def run_image_main(argv, *, expect_error: bool = False):
    buffer = io.StringIO()
    caught = None
    code = None
    with contextlib.redirect_stdout(buffer):
        try:
            code = image_main(argv)
        except Exception as exc:  # noqa: BLE001 - test helper returns the captured failure
            caught = exc
    if caught is not None and not expect_error:
        raise caught
    if caught is None and expect_error:
        raise AssertionError("Expected image_main to raise, but it returned normally.")
    return code, caught, parse_result_summary(buffer.getvalue()), buffer.getvalue()


def tiny_cli_args(batch_size: int) -> list[str]:
    return [
        "--mode",
        "vit_adapter",
        "--dataset",
        "synthetic",
        "--n_images",
        str(batch_size),
        "--batch_size",
        str(batch_size),
        "--public_n_images",
        "4",
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
        "--peftleak_num_bins",
        "16",
        "--device",
        "cpu",
    ]


def manual_probe(mean, std, position, *, num_bins=1, embed_scale=0.5, gap=0, device=None):
    device = device or position.device
    position = position.to(device=device, dtype=torch.float32)
    patch_dim = int(position.numel())
    patch_stats = PatchStatistics(
        mean=mean.to(device=device, dtype=torch.float32),
        std=std.to(device=device, dtype=torch.float32),
        num_images=1,
        num_patches=1,
        patch_size=1,
    )
    projection = torch.ones(1, patch_dim, device=device, dtype=torch.float32)
    projection = projection / projection.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    edges = torch.full((1, int(num_bins) + 1), float("inf"), device=device, dtype=torch.float32)
    edges[:, 0] = float("-inf")
    if num_bins > 1:
        edges[:, 1:-1] = torch.linspace(-1.0, 1.0, steps=int(num_bins) - 1, device=device)
    return PeftLeakProbeStatistics(
        patch_stats=patch_stats,
        position_embeddings=position.view(1, patch_dim),
        projection_vectors=projection,
        bin_edges=edges,
        bin_counts=torch.zeros(1, int(num_bins), device=device, dtype=torch.long),
        num_bins=int(num_bins),
        embed_scale=float(embed_scale),
        gap=int(gap),
        seed=0,
    )


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


def test_shared_adjacent_difference_recovery_inverts_probe_transform():
    patch = torch.tensor([0.1, 0.3, -0.2], dtype=torch.float32)
    pos = torch.tensor([0.5, -0.1, 0.2], dtype=torch.float32)
    mean = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    std = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32)
    scale = 0.25
    exposed = ((patch - mean) / std) * scale + pos
    weight_grad = torch.stack((3.0 * exposed, -3.0 * exposed), dim=0)
    bias_grad = torch.tensor([3.0, -3.0], dtype=torch.float32)
    probe = manual_probe(mean, std, pos, embed_scale=scale)

    result = recover_patches_from_shared_adapter_grads(
        (weight_grad, bias_grad),
        ["vit.adapter.shared.weight", "vit.adapter.shared.bias"],
        probe,
        batch_size=1,
        n_patches=1,
    )

    assert_true(result.recovered_patch_count == 1, "unique shared slot should recover one patch")
    assert_true(torch.equal(result.recovery_mask, torch.ones(1, 1, dtype=torch.bool)), "unique slot should be marked recovered")
    assert_true(torch.allclose(result.recovered_patches[0, 0], patch, atol=1e-6), "adjacent-difference recovery should invert probe transform")
    assert_true(torch.equal(result.candidate_slots, torch.zeros(1, dtype=torch.long)), "observable candidate should record its shared slot")
    assert_true(result.collision_patch_count is None, "observable shared gradients cannot infer private bin collisions")

    oracle = recover_patches_from_shared_adapter_grads(
        (weight_grad, bias_grad),
        ["vit.adapter.shared.weight", "vit.adapter.shared.bias"],
        probe,
        batch_size=1,
        n_patches=1,
        slot_indices=torch.zeros(1, 1, dtype=torch.long),
    )
    assert_true(oracle.oracle_recovered_patch_count == 1, "oracle debug path should recover the known unique assignment")
    assert_true(
        torch.allclose(oracle.oracle_recovered_patches[0, 0], patch, atol=1e-6),
        "oracle debug recovery should use slot_indices only in oracle fields",
    )


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


def test_shared_vit_adapter_autograd_gradients_recover_patches():
    device = runtime_device()
    attack_images = torch.rand(1, 1, 4, 4, device=device)
    public_images = torch.rand(4, 1, 4, 4, device=device)
    probe = build_peftleak_probe_statistics(
        public_images,
        patch_size=2,
        num_bins=16,
        position_sigma=0.5,
        embed_scale=0.5,
        seed=11,
    )
    result = build_shared_adapter_gradients(
        attack_images,
        probe,
        labels=torch.tensor([1], dtype=torch.long, device=device),
        patch_size=2,
        adapter_hidden_dim=3,
        n_classes=4,
        seed=11,
    )
    recovery = recover_patches_from_shared_adapter_grads(
        result.grads,
        result.names,
        probe,
        batch_size=1,
        n_patches=4,
        slot_indices=result.slot_indices,
    )
    reference = extract_image_patches(attack_images, patch_size=2)
    assert_true(len(result.names) == 2, "ViT adapter path should expose one shared weight and one shared bias gradient")
    assert_true(not any("sample_" in name or "patch_" in name for name in result.names), "shared adapter names must not expose sample/patch oracle ids")
    assert_true(
        all(grad is None or grad.device == device for grad in result.grads),
        "ViT adapter gradients should stay on the runtime device",
    )
    assert_true(result.normalized_patches.device == device, "normalized patches should stay on the runtime device")
    assert_true(result.raw_patches.device == device, "raw patches should stay on the runtime device")
    assert_true(recovery.recovered_patches.device == device, "recovered patches should stay on the runtime device")
    assert_true(recovery.collision_patch_count is None, "observable shared gradients should not claim private collision counts")
    assert_true(recovery.oracle_collision_patch_count == 0, "single-image oracle debug slots should avoid patch collisions")
    assert_true(torch.allclose(recovery.recovered_patches, reference, atol=1e-5), "shared autograd adapter gradients should recover patches")


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
    probe = build_peftleak_probe_statistics(torch.rand(4, 1, 4, 4, device=device), 2, num_bins=16, seed=7)

    recovered, reference, loss, top1, grad_count, recovery = _run_vit_adapter(args, images, labels, probe)

    assert_true(recovered.device == device, "run_vit_adapter should return recovered patches on runtime device")
    assert_true(reference.device == device, "run_vit_adapter should return reference patches on runtime device")
    assert_true(loss is not None and loss >= 0.0, "run_vit_adapter should report a finite loss")
    assert_true(top1 is not None and 0.0 <= top1 <= 1.0, "run_vit_adapter should report batch top-1 accuracy")
    assert_true(grad_count == 2, "run_vit_adapter should report shared weight/bias gradients")
    assert_true(recovery.recovered_patches.shape == recovered.shape, "run_vit_adapter should return structured recovery metadata")
    assert_true(recovered.shape[0] == 1, "multi-sample vit_adapter should not fabricate per-sample direct recovery")
    assert_true(recovery.oracle_recovered_patches.shape == reference.shape, "oracle debug recovery should stay separate from reportable output")


def test_cli_batch1_reports_direct_primary_metric_and_shared_metadata():
    code, _exc, summary, _output = run_image_main(tiny_cli_args(1))

    assert_true(code == 0, "batch=1 CLI smoke should succeed")
    assert_true(summary["result_status"] == "ok", "batch=1 summary should be ok")
    assert_true(summary["attack_variant"] == "vit_adapter_shared_bins", "vit_adapter should use shared-bin variant")
    assert_true(
        summary["reproduction_level"] == "peftleak_style_shared_bins",
        "summary should label the current reproduction level precisely",
    )
    assert_true(summary["primary_metric_source"] == "direct", "batch=1 should use non-oracle direct metrics")
    assert_true(summary["direct_mse"] != "n/a", "batch=1 direct metric should be available")
    assert_true(summary["oracle_metric_scope"] == "debug_only", "oracle metrics should be explicitly debug-only")
    assert_true(summary["nonzero_slot_count"] != "n/a", "shared-bin nonzero slot metadata should be reported")
    assert_true(summary["ambiguous_position_count"] != "n/a", "shared-bin ambiguity metadata should be reported")
    assert_true(summary["empty_position_count"] != "n/a", "shared-bin empty-position metadata should be reported")


def test_cli_batch2_keeps_direct_metrics_non_oracle():
    code, _exc, summary, _output = run_image_main(tiny_cli_args(2))

    assert_true(code == 0, "batch=2 CLI smoke should succeed")
    assert_true(summary["primary_metric_source"] in {"clustered", "n/a"}, "batch=2 should not use oracle direct metrics")
    assert_true(summary["direct_mse"] == "n/a", "batch=2 direct metric should remain unavailable without oracle assignment")
    assert_true(summary["patch_recovery_rate"] == "n/a", "batch=2 exact recovery should not use oracle slot assignments")
    assert_true(summary["oracle_patch_recovery_rate"] != "n/a", "oracle debug recovery can still be reported separately")
    assert_true(summary["oracle_metric_scope"] == "debug_only", "oracle fields should be labeled debug-only")


def test_cli_cifar100_fallback_is_marked_by_default():
    with tempfile.TemporaryDirectory() as tmpdir:
        code, _exc, summary, _output = run_image_main(
            [
                "--mode",
                "synthetic_ratio",
                "--dataset",
                "cifar100",
                "--data_root",
                tmpdir,
                "--cache_dir",
                tmpdir,
                "--n_images",
                "1",
                "--public_n_images",
                "1",
                "--image_size",
                "4",
                "--channels",
                "1",
                "--patch_size",
                "2",
                "--adapter_hidden_dim",
                "3",
                "--device",
                "cpu",
            ]
        )

    assert_true(code == 0, "default CIFAR100 fallback should keep smoke runs compatible")
    assert_true(summary["result_status"] == "ok", "fallback smoke summary should be ok")
    assert_true(summary["synthetic_fallback"] == "1", "CIFAR100 synthetic fallback should be marked")
    assert_true(summary["fallback_reason"] != "n/a", "fallback reason should be reported")


def test_cli_cifar100_fallback_can_fail_explicitly():
    with tempfile.TemporaryDirectory() as tmpdir:
        _code, exc, summary, _output = run_image_main(
            [
                "--mode",
                "synthetic_ratio",
                "--dataset",
                "cifar100",
                "--data_root",
                tmpdir,
                "--cache_dir",
                tmpdir,
                "--n_images",
                "1",
                "--public_n_images",
                "1",
                "--image_size",
                "4",
                "--channels",
                "1",
                "--patch_size",
                "2",
                "--adapter_hidden_dim",
                "3",
                "--device",
                "cpu",
                "--fail_on_synthetic_fallback",
            ],
            expect_error=True,
        )

    assert_true(isinstance(exc, SyntheticFallbackError), "explicit fallback guard should raise SyntheticFallbackError")
    assert_true(summary["result_status"] == "failed", "guarded fallback should emit a failed summary")
    assert_true(summary["synthetic_fallback"] == "1", "failed fallback summary should mark synthetic fallback")
    assert_true(summary["fallback_reason"] != "n/a", "failed fallback summary should include the reason")


def test_shared_adapter_gradient_shape_is_stable_across_batch_size():
    device = runtime_device()
    public_images = torch.rand(4, 1, 4, 4, device=device)
    probe = build_peftleak_probe_statistics(public_images, 2, num_bins=8, seed=19)
    one = build_shared_adapter_gradients(
        torch.rand(1, 1, 4, 4, device=device),
        probe,
        labels=torch.tensor([1], dtype=torch.long, device=device),
        patch_size=2,
        adapter_hidden_dim=3,
        n_classes=4,
        seed=19,
    )
    two = build_shared_adapter_gradients(
        torch.rand(2, 1, 4, 4, device=device),
        probe,
        labels=torch.tensor([1, 2], dtype=torch.long, device=device),
        patch_size=2,
        adapter_hidden_dim=3,
        n_classes=4,
        seed=19,
    )

    assert_true(one.names == two.names == ["vit.adapter.shared.weight", "vit.adapter.shared.bias"], "shared adapter names should be stable")
    assert_true(len(one.grads) == len(two.grads) == 2, "shared adapter should expose two gradient tensors")
    assert_true(one.grads[0].shape == two.grads[0].shape, "shared weight gradient shape should not scale with batch size")
    assert_true(one.grads[1].shape == two.grads[1].shape, "shared bias gradient shape should not scale with batch size")


def test_dpsgd_per_example_shared_gradient_shapes_match_full_batch():
    device = runtime_device()
    public_images = torch.rand(4, 1, 4, 4, device=device)
    probe = build_peftleak_probe_statistics(public_images, 2, num_bins=8, seed=23)
    images = torch.rand(2, 1, 4, 4, device=device)
    labels = torch.tensor([1, 2], dtype=torch.long, device=device)
    full, per_example = build_shared_adapter_gradient_bundle(
        images,
        probe,
        labels=labels,
        patch_size=2,
        adapter_hidden_dim=3,
        n_classes=4,
        seed=23,
    )

    assert_true(len(per_example) == 2, "DPSGD helper should return one shared-gradient tuple per example")
    for sample in per_example:
        assert_true(sample.names == full.names, "full and per-example shared gradients should use the same model parameters")
        assert_true(len(full.grads) == len(sample.grads) == 2, "full and per-example shared gradients should have the same tuple length")
        assert_true(full.grads[0].shape == sample.grads[0].shape, "DPSGD per-example shared weight shape should match full-batch shape")
        assert_true(full.grads[1].shape == sample.grads[1].shape, "DPSGD per-example shared bias shape should match full-batch shape")


def test_shared_recovery_counts_collisions_as_unresolved():
    patch_a = torch.tensor([0.1, 0.2], dtype=torch.float32)
    patch_b = torch.tensor([0.3, 0.4], dtype=torch.float32)
    mean = torch.zeros(2, dtype=torch.float32)
    std = torch.ones(2, dtype=torch.float32)
    pos = torch.zeros(2, dtype=torch.float32)
    probe = manual_probe(mean, std, pos, num_bins=1, embed_scale=1.0)
    exposed_mean = 0.5 * (patch_a + patch_b)
    weight_grad = torch.stack((2.0 * exposed_mean, -2.0 * exposed_mean), dim=0)
    bias_grad = torch.tensor([2.0, -2.0], dtype=torch.float32)

    result = recover_patches_from_shared_adapter_grads(
        (weight_grad, bias_grad),
        ["vit.adapter.shared.weight", "vit.adapter.shared.bias"],
        probe,
        batch_size=2,
        n_patches=1,
        slot_indices=torch.zeros(2, 1, dtype=torch.long),
    )

    assert_true(result.candidate_patch_count == 1, "collision slot should still produce one gradient candidate")
    assert_true(result.collision_patch_count is None, "observable shared gradients should not infer private collision counts")
    assert_true(result.recovered_patch_count == 0, "multi-sample shared candidates should not be counted as direct recovered patches")
    assert_true(not bool(result.recovery_mask.any()), "multi-sample reportable output should remain unresolved without oracle assignment")
    assert_true(result.oracle_collision_patch_count == 2, "oracle debug fields should count both patch instances in a shared slot")
    assert_true(result.oracle_recovered_patch_count == 0, "oracle collided patches should not be counted as exact recovered patches")
    assert_true(not bool(result.oracle_recovery_mask.any()), "oracle collided patches should remain unresolved")


def test_shared_recovery_without_oracle_keeps_batch_candidates_unordered():
    patch_a = torch.tensor([0.1, 0.2], dtype=torch.float32)
    patch_b = torch.tensor([0.8, 0.9], dtype=torch.float32)
    mean = torch.zeros(2, dtype=torch.float32)
    std = torch.ones(2, dtype=torch.float32)
    pos = torch.zeros(2, dtype=torch.float32)
    probe = manual_probe(mean, std, pos, num_bins=2, embed_scale=1.0)
    weight_grad = torch.zeros(4, 2, dtype=torch.float32)
    bias_grad = torch.zeros(4, dtype=torch.float32)
    weight_grad[0] = 2.0 * patch_a
    weight_grad[1] = -2.0 * patch_a
    bias_grad[0] = 2.0
    bias_grad[1] = -2.0
    weight_grad[2] = 3.0 * patch_b
    weight_grad[3] = -3.0 * patch_b
    bias_grad[2] = 3.0
    bias_grad[3] = -3.0

    result = recover_patches_from_shared_adapter_grads(
        (weight_grad, bias_grad),
        ["vit.adapter.shared.weight", "vit.adapter.shared.bias"],
        probe,
        batch_size=2,
        n_patches=1,
    )

    assert_true(result.candidate_patch_count == 2, "two observable slots should produce two unordered candidates")
    assert_true(result.candidate_patches.shape == (2, 2), "observable candidates should be a flat candidate set")
    assert_true(result.recovered_patches.shape == (1, 1, 2), "reportable direct output should not fabricate batch rows")
    assert_true(result.recovered_patch_count == 0, "batch-level candidates should not be counted as direct recovered patches")
    assert_true(result.oracle_recovered_patches is None, "oracle fields should remain absent without private slot_indices")


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
        test_shared_adjacent_difference_recovery_inverts_probe_transform,
        test_patch_extract_fold_roundtrip_and_public_stats,
        test_public_stats_file_roundtrip_uses_explicit_file,
        test_public_stats_move_preserves_metadata_and_device,
        test_shared_vit_adapter_autograd_gradients_recover_patches,
        test_vit_adapter_run_reports_loss_acc_and_device_local_outputs,
        test_cli_batch1_reports_direct_primary_metric_and_shared_metadata,
        test_cli_batch2_keeps_direct_metrics_non_oracle,
        test_cli_cifar100_fallback_is_marked_by_default,
        test_cli_cifar100_fallback_can_fail_explicitly,
        test_shared_adapter_gradient_shape_is_stable_across_batch_size,
        test_dpsgd_per_example_shared_gradient_shapes_match_full_batch,
        test_shared_recovery_counts_collisions_as_unresolved,
        test_shared_recovery_without_oracle_keeps_batch_candidates_unordered,
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

