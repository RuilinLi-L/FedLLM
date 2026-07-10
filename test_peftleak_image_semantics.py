#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OFFICIAL_IMAGE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "PEFTLeak-main",
    "PEFTLeak-main",
)
sys.path.insert(0, OFFICIAL_IMAGE_DIR)

from attacks.peftleak_image.core import (
    PatchStatistics,
    PeftLeakProbeStatistics,
    build_official_aligned_adapter_gradients,
    build_official_aligned_probe_statistics,
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
    recover_patches_from_official_adapter_grads,
    recover_patches_from_batch,
    recover_patches_from_shared_adapter_grads,
    resolve_cluster_method,
    resolve_official_adapter_layer_count,
    save_public_patch_statistics,
    simple_ssim,
)
from attack_peftleak_image import (
    SUMMARY_END,
    SUMMARY_START,
    SampleSelectionError,
    SyntheticFallbackError,
    _defend_gradient_tuple,
    _indices_hash,
    _load_indices_file,
    _make_raw_gradient_tuple,
    _recover_from_gradient_tuple,
    _run_synthetic_ratio,
    _run_vit_adapter,
    _select_dataset_indices,
    build_parser,
    main as image_main,
)
from official_image_runner import (
    ADAPTER_R,
    CHANNELS,
    DROPOUT,
    EMBED_DIM,
    HEAD_DIM,
    NUM_CLASSES,
    NUM_HEADS,
    NUM_PATCHES,
    PATCH_DIM,
    PATCH_SIZE,
    build_denorm_recovery_metrics,
    build_matched_patch_metrics,
    collect_adapter_gradients,
    official_adapter_gradient_names,
    select_official_attack_gradients,
    set_adapter_grads_only,
)
from Transformer_Model_neuron import ViT as OfficialImageViT
from utils.lrb_defense import _extract_layer_index, _layer_sensitivity, apply_lrb_defense


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


def sampling_args(strategy: str, *, split_seed: int = 101, attack_indices_path=None, public_indices_path=None):
    return argparse.Namespace(
        sample_strategy=strategy,
        split_seed=split_seed,
        rng_seed=split_seed,
        attack_indices_path=attack_indices_path,
        public_indices_path=public_indices_path,
    )


def test_sampling_first_n_and_hash_are_stable():
    args = sampling_args("first_n")

    selected = _select_dataset_indices(args, public=False, dataset_len=10, requested_count=4)

    assert_true(selected == [0, 1, 2, 3], "first_n should preserve legacy leading-index behavior")
    assert_true(_indices_hash(selected) == _indices_hash([0, 1, 2, 3]), "index hash should be deterministic")


def test_seeded_shuffle_sampling_uses_split_seed():
    args_101 = sampling_args("seeded_shuffle", split_seed=101)
    args_202 = sampling_args("seeded_shuffle", split_seed=202)

    selected_101 = _select_dataset_indices(args_101, public=False, dataset_len=50, requested_count=8)
    selected_101_repeat = _select_dataset_indices(args_101, public=False, dataset_len=50, requested_count=8)
    selected_202 = _select_dataset_indices(args_202, public=False, dataset_len=50, requested_count=8)
    public_101 = _select_dataset_indices(args_101, public=True, dataset_len=50, requested_count=8)

    assert_true(selected_101 == selected_101_repeat, "seeded_shuffle should be reproducible for one split_seed")
    assert_true(selected_101 != selected_202, "different split_seed values should produce different victim indices")
    assert_true(selected_101 != public_101, "public selection should use an offset split seed")


def test_indices_file_sampling_reads_text_and_npy():
    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        raise AssertionError("numpy is required for .npy index-file sampling tests") from exc

    with tempfile.TemporaryDirectory() as tmpdir:
        text_path = os.path.join(tmpdir, "public_indices.txt")
        npy_path = os.path.join(tmpdir, "attack_indices.npy")
        with open(text_path, "w", encoding="utf-8") as handle:
            handle.write("3, 1\n0\n")
        np.save(npy_path, np.array([7, 5, 2, 1], dtype=np.int64))
        args = sampling_args("indices_file", attack_indices_path=npy_path, public_indices_path=text_path)

        assert_true(_load_indices_file(text_path) == [3, 1, 0], "text index files should support commas and whitespace")
        attack_selected = _select_dataset_indices(args, public=False, dataset_len=10, requested_count=3)
        public_selected = _select_dataset_indices(args, public=True, dataset_len=10, requested_count=2)

    assert_true(attack_selected == [7, 5, 2], "indices_file should preserve .npy ordering")
    assert_true(public_selected == [3, 1], "indices_file should preserve text-file ordering")


def test_indices_file_sampling_reports_clear_errors():
    with tempfile.TemporaryDirectory() as tmpdir:
        short_path = os.path.join(tmpdir, "short.txt")
        empty_path = os.path.join(tmpdir, "empty.txt")
        bad_path = os.path.join(tmpdir, "bad.txt")
        with open(short_path, "w", encoding="utf-8") as handle:
            handle.write("2 4\n")
        open(empty_path, "w", encoding="utf-8").close()
        with open(bad_path, "w", encoding="utf-8") as handle:
            handle.write("99\n")

        missing_args = sampling_args("indices_file")
        short_args = sampling_args("indices_file", attack_indices_path=short_path)
        empty_args = sampling_args("indices_file", attack_indices_path=empty_path)
        bad_args = sampling_args("indices_file", attack_indices_path=bad_path)

        for args, requested_count, expected in [
            (missing_args, 3, "--attack_indices_path is required"),
            (short_args, 3, "contains 2 indices"),
            (empty_args, 3, "Index file is empty"),
            (bad_args, 1, "outside dataset range"),
        ]:
            try:
                _select_dataset_indices(args, public=False, dataset_len=10, requested_count=requested_count)
            except SampleSelectionError as exc:
                assert_true(expected in str(exc), f"sampling error should mention {expected!r}")
            else:
                raise AssertionError(f"Expected SampleSelectionError containing {expected!r}")


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


def test_official_public_cdf_probe_is_deterministic():
    images = torch.linspace(0, 1, steps=4 * 1 * 4 * 4, dtype=torch.float32).view(4, 1, 4, 4)

    first = build_official_aligned_probe_statistics(images, patch_size=2, num_bins=8, seed=17)
    second = build_official_aligned_probe_statistics(images, patch_size=2, num_bins=8, seed=17)

    assert_true(torch.allclose(first.bin_edges, second.bin_edges), "official CDF bins should be deterministic")
    assert_true(torch.allclose(first.position_embeddings, second.position_embeddings), "official position probes should be deterministic")
    assert_true(first.bin_counts.sum().item() == first.patch_stats.num_patches, "bin counts should cover the public patch set")
    assert_true(resolve_official_adapter_layer_count("all", vit_config="cifar_small") > 0, "official layer resolver should produce layers")


def test_official_gradient_ratio_recovers_batch1_without_oracle():
    device = runtime_device()
    images = torch.rand(1, 1, 4, 4, device=device)
    public = torch.rand(8, 1, 4, 4, device=device)
    probe = build_official_aligned_probe_statistics(public, patch_size=2, num_bins=32, seed=31)
    result = build_official_aligned_adapter_gradients(
        images,
        probe,
        labels=torch.tensor([1], dtype=torch.long, device=device),
        patch_size=2,
        n_classes=4,
        seed=31,
        adapter_layers="first_n",
        adapter_bottleneck_dim=3,
    )
    recovery = recover_patches_from_official_adapter_grads(
        result.grads,
        result.names,
        probe,
        batch_size=1,
        n_patches=4,
    )
    reference = extract_image_patches(images, patch_size=2)

    assert_true(recovery.oracle_recovered_patches is None, "official primary recovery should not use oracle slot assignments")
    assert_true(recovery.recovered_patches.shape == reference.shape, "official recovery should return batch-shaped patches")
    assert_true(bool(recovery.recovery_mask.all().item()), "official batch=1 recovery should recover all patches non-oracle")
    assert_true(torch.allclose(recovery.recovered_patches, reference, atol=1e-5), "official batch=1 recovery should match reference patches")
    assert_true(torch.isfinite(recovery.recovered_patches).all(), "official recovered patches should stay finite")


def test_official_source_gradient_inventory_and_lrb_layer_indices():
    weight_names, bias_names = official_adapter_gradient_names()
    assert_true(len(weight_names) == len(bias_names) == 21, "official source inventory should preserve its trailing unused Adapter pair")
    assert_true(_extract_layer_index("encoder1.attn.adapt1.weight") == 0, "encoder1 should map to zero-based layer 0")
    assert_true(_extract_layer_index("encoder2.mlp.adapt1.bias") == 1, "encoder2 should map to zero-based layer 1")
    assert_true(_extract_layer_index("vit.official_adapter.layer_3.shared.weight") == 3, "proxy layer_ names should remain parseable")
    assert_true(_layer_sensitivity("encoder1.attn.adapt1.weight", 2) == 1.0, "first attention Adapter should be sensitive")
    assert_true(_layer_sensitivity("encoder3.attn.adapt1.weight", 2) == 0.45, "later attention Adapter should use the non-sensitive prior")

    all_names = [*weight_names, *bias_names, "encoder1.attn.adapt2.weight"]
    all_grads = [torch.full((1,), float(idx)) for idx in range(len(all_names))]
    selected_w, selected_b, selected_w_names, selected_b_names = select_official_attack_gradients(all_grads, all_names)
    assert_true(selected_w_names == weight_names[:20] and selected_b_names == bias_names[:20], "attack gradient selection should preserve the 20 consumed source pairs")
    assert_true(len(selected_w) == len(selected_b) == 20, "attack selection should use 20 defended weight/bias pairs")

    args = argparse.Namespace(
        defense_lrb_sensitive_n_layers=2,
        defense_lrb_keep_ratio_sensitive=0.5,
        defense_lrb_keep_ratio_other=0.9,
        defense_lrb_clip_scale_sensitive=1_000_000.0,
        defense_lrb_clip_scale_other=1_000_000.0,
        defense_lrb_noise_sensitive=0.0,
        defense_lrb_noise_other=0.0,
        defense_lrb_empirical_weight=0.0,
        defense_lrb_calibration_samples=64,
        defense_lrb_projection="signed_pool",
        rng_seed=17,
    )
    apply_lrb_defense(
        (torch.rand(4, 4), torch.rand(4, 4)),
        args,
        layer_names=["encoder1.attn.adapt1.weight", "encoder3.attn.adapt1.weight"],
    )
    layer_info = args.lrb_defense_layer_info
    assert_true(layer_info[0]["sensitivity"] == 1.0, "LRB should mark encoder1 attention as sensitive")
    assert_true(layer_info[1]["sensitivity"] == 0.45, "LRB should preserve the later attention prior")
    assert_true(layer_info[0]["keep_ratio"] < layer_info[1]["keep_ratio"], "sensitive official layers should receive stronger projection")


def test_official_matched_patch_metrics_use_one_to_one_ground_truth_matches():
    gen = torch.Generator(device="cpu")
    gen.manual_seed(123)
    refs = torch.rand(2, 4, 3, 16, 16, generator=gen)
    candidates = []
    for position_idx in range(4):
        candidates.append(
            [
                refs[1, position_idx].clone(),
                refs[0, position_idx].clone(),
                torch.zeros_like(refs[0, position_idx]),
            ]
        )
    mean = torch.zeros(3, 1, 1)
    std = torch.ones(3, 1, 1)
    recovery = build_denorm_recovery_metrics(
        candidates,
        refs,
        mean,
        std,
        [(1e-8, "1e-08")],
    )
    matches = recovery["matches_by_label"]["1e-08"]
    metrics = build_matched_patch_metrics(
        candidates,
        refs,
        mean,
        std,
        matches,
        metrics={"mse", "ssim"},
    )
    assert_true(len(matches) == 8, "one-to-one matching should recover every exact patch once")
    assert_true(metrics["matched_patch_count"] == 8, "matched metric count should equal the target patch count")
    assert_true(abs(metrics["matched_patch_rate"] - 1.0) < 1e-12, "exact candidates should have full recovery rate")
    assert_true(metrics["matched_patch_mse_mean"] < 1e-12, "exact matched patches should have zero MSE")
    assert_true(
        metrics["matched_patch_ssim_status"] in {"ok", "unavailable"},
        "standard SSIM should either run or report its optional dependency clearly",
    )
    if metrics["matched_patch_ssim_status"] == "ok":
        assert_true(abs(metrics["matched_patch_ssim_mean"] - 1.0) < 1e-6, "exact matched patches should have SSIM one")


def test_official_source_collects_complete_adapter_update_before_attack_selection():
    model = OfficialImageViT(
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
    set_adapter_grads_only(model)
    images = torch.rand(1, 1, CHANNELS, 32, 32)
    model(images).sum().backward()

    adapter_grads, adapter_names = collect_adapter_gradients(model)
    selected_w, selected_b, selected_w_names, selected_b_names = select_official_attack_gradients(
        adapter_grads,
        adapter_names,
    )
    assert_true(len(adapter_grads) == len(adapter_names) == 96, "12-layer ViT should share the complete 96-tensor Adapter update")
    assert_true(all("adapt1" in name or "adapt2" in name for name in adapter_names), "complete update should contain Adapter tensors only")
    assert_true(len(selected_w) == len(selected_b) == 20, "closed-form attack should consume exactly 20 defended gradient pairs")
    assert_true(len(selected_w_names) == len(selected_b_names) == 20, "selected attack names should match consumed pairs")


def test_simple_ssim_reports_perfect_match():
    images = torch.rand(2, 1, 4, 4)

    score = simple_ssim(images, images)

    assert_true(abs(score - 1.0) < 1e-6, "SSIM should be one for identical tensors")


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


def test_cli_summary_reports_sampling_protocol_fields():
    code, _exc, summary, _output = run_image_main(tiny_cli_args(1))

    assert_true(code == 0, "sampling summary CLI smoke should succeed")
    assert_true(summary["sample_strategy"] == "first_n", "default sample strategy should preserve legacy first_n behavior")
    assert_true(summary["peftleak_protocol"] == "legacy_first_n", "first_n runs should be labeled as legacy protocol")
    assert_true(summary["split_seed"] == summary["rng_seed"], "split_seed should default to rng_seed")
    assert_true(summary["attack_index_count"] == "1", "attack index count should reflect synthetic n_images")
    assert_true(summary["public_index_count"] == "4", "public index count should reflect public_n_images")
    assert_true(summary["attack_indices_hash"] != "n/a", "attack index hash should be reported")
    assert_true(summary["public_indices_hash"] != "n/a", "public index hash should be reported")


def test_cli_batch2_keeps_direct_metrics_non_oracle():
    code, _exc, summary, _output = run_image_main(tiny_cli_args(2))

    assert_true(code == 0, "batch=2 CLI smoke should succeed")
    assert_true(summary["primary_metric_source"] in {"clustered", "n/a"}, "batch=2 should not use oracle direct metrics")
    assert_true(summary["direct_mse"] == "n/a", "batch=2 direct metric should remain unavailable without oracle assignment")
    assert_true(summary["patch_recovery_rate"] == "n/a", "batch=2 exact recovery should not use oracle slot assignments")
    assert_true(summary["oracle_patch_recovery_rate"] != "n/a", "oracle debug recovery can still be reported separately")
    assert_true(summary["oracle_metric_scope"] == "debug_only", "oracle fields should be labeled debug-only")


def test_cli_official_batch2_reports_non_oracle_primary_metrics():
    argv = tiny_cli_args(2)
    mode_idx = argv.index("--mode") + 1
    argv[mode_idx] = "official_vit_adapter"
    argv.extend(
        [
            "--vit_config",
            "cifar_small",
            "--adapter_layers",
            "first_n",
            "--adapter_bottleneck_dim",
            "3",
            "--official_grouping",
            "tag",
            "--peftleak_num_bins",
            "32",
        ]
    )

    code, _exc, summary, _output = run_image_main(argv)

    assert_true(code == 0, "official batch=2 CLI smoke should succeed")
    assert_true(summary["attack_variant"] == "official_vit_adapter", "official mode should report its attack variant")
    assert_true(
        summary["reproduction_level"] == "peftleak_official_aligned_v1",
        "official mode should use the official-aligned reproduction level",
    )
    assert_true(summary["non_oracle_grouping"] == "tag", "official mode should report tag grouping")
    assert_true(summary["oracle_patch_recovery_rate"] == "n/a", "official primary metrics should not use oracle recovery")
    assert_true(
        summary["primary_metric_source"] in {"direct", "clustered"},
        "official tag grouping should provide a non-oracle primary metric even when collisions require clustering",
    )
    assert_true(summary["ssim"] != "n/a", "official mode should report SSIM")


def test_cli_official_cluster_grouping_does_not_report_direct_metrics():
    argv = tiny_cli_args(2)
    mode_idx = argv.index("--mode") + 1
    argv[mode_idx] = "official_vit_adapter"
    argv.extend(
        [
            "--vit_config",
            "cifar_small",
            "--adapter_layers",
            "first_n",
            "--attack_rounds",
            "2",
            "--official_grouping",
            "cluster",
            "--peftleak_num_bins",
            "32",
        ]
    )

    code, _exc, summary, _output = run_image_main(argv)

    assert_true(code == 0, "official cluster grouping CLI smoke should succeed")
    assert_true(summary["non_oracle_grouping"] == "cluster", "summary should report requested cluster grouping")
    assert_true(summary["attack_rounds"] == "2", "summary should report requested attack rounds")
    assert_true(summary["adapter_gradient_count"] == "16", "two first_n official rounds should expose two rounds of adapter gradients")
    assert_true(summary["direct_mse"] == "n/a", "cluster grouping should not silently reuse tag-direct metrics")
    assert_true(summary["primary_metric_source"] in {"clustered", "n/a"}, "cluster grouping should use clustered or unavailable primary metrics")


def test_cli_batch_size_drives_attack_batches():
    argv = tiny_cli_args(4)
    batch_idx = argv.index("--batch_size") + 1
    argv[batch_idx] = "2"

    code, _exc, summary, _output = run_image_main(argv)

    assert_true(code == 0, "batched CLI smoke should succeed")
    assert_true(summary["effective_batch_size"] == "2", "effective batch size should follow --batch_size")
    assert_true(summary["attack_batch_count"] == "2", "n_images=4,batch_size=2 should run two attack batches")


def test_cli_metrics_controls_primary_report_fields():
    argv = tiny_cli_args(1)
    argv.extend(["--metrics", "psnr"])

    code, _exc, summary, _output = run_image_main(argv)

    assert_true(code == 0, "metrics-filtered CLI smoke should succeed")
    assert_true(summary["mse"] == "n/a", "mse should be omitted from primary report when not requested")
    assert_true(summary["psnr"] != "n/a", "requested psnr should still be reported")
    assert_true(summary["patch_recovery_rate"] == "n/a", "patch recovery should be omitted when not requested")


def test_official_cifar32_profile_sets_official_like_defaults():
    code, _exc, summary, _output = run_image_main(
        [
            "--mode",
            "synthetic_ratio",
            "--peftleak_profile",
            "official_cifar32",
            "--dataset",
            "synthetic",
            "--n_images",
            "1",
            "--batch_size",
            "1",
            "--public_n_images",
            "1",
            "--device",
            "cpu",
            "--metrics",
            "psnr",
        ]
    )

    assert_true(code == 0, "official_cifar32 profile smoke should succeed")
    assert_true(summary["peftleak_profile"] == "official_cifar32", "summary should report requested profile")
    assert_true(summary["official_like_config"] == "true", "official profile should use official-like defaults")
    assert_true(summary["profile_override_count"] == "0", "official profile should have no overrides here")
    assert_true(summary["patch_size"] == "16", "official profile should set patch_size=16")
    assert_true(summary["adapter_hidden_dim"] == "64", "official profile should set adapter hidden dim")
    assert_true(summary["peftleak_num_bins"] == "320", "official profile should set official-style bin count")
    assert_true(summary["non_oracle_primary_only"] == "true", "summary should mark primary metrics as non-oracle")


def test_official_cifar32_profile_respects_explicit_overrides():
    code, _exc, summary, _output = run_image_main(
        [
            "--mode",
            "synthetic_ratio",
            "--peftleak_profile",
            "official_cifar32",
            "--dataset",
            "synthetic",
            "--n_images",
            "1",
            "--batch_size",
            "1",
            "--public_n_images",
            "1",
            "--patch_size",
            "8",
            "--device",
            "cpu",
            "--metrics",
            "psnr",
        ]
    )

    assert_true(code == 0, "official profile override smoke should succeed")
    assert_true(summary["patch_size"] == "8", "explicit CLI patch_size should override profile")
    assert_true(summary["official_like_config"] == "false", "overriding official geometry should clear official-like flag")
    assert_true(summary["profile_override_count"] == "1", "one profile field should be counted as overridden")
    assert_true(summary["config_warning"] == "profile_overrides:patch_size", "summary should name overridden profile field")


def test_cli_official_cifar32_profile_batch1_smoke():
    code, _exc, summary, _output = run_image_main(
        [
            "--mode",
            "official_vit_adapter",
            "--peftleak_profile",
            "official_cifar32",
            "--dataset",
            "synthetic",
            "--n_images",
            "1",
            "--batch_size",
            "1",
            "--public_split_size",
            "4",
            "--adapter_layers",
            "first_n",
            "--device",
            "cpu",
            "--metrics",
            "mse,psnr,ssim,lpips,patch_recovery",
        ]
    )

    assert_true(code == 0, "official profile official_vit_adapter smoke should succeed")
    assert_true(summary["attack_variant"] == "official_vit_adapter", "official profile smoke should use official variant")
    assert_true(summary["reproduction_level"] == "peftleak_official_aligned_v1", "official profile smoke should remain v1 aligned")
    assert_true(summary["official_like_config"] == "true", "official profile smoke should be marked official-like")
    assert_true(summary["non_oracle_primary_only"] == "true", "main metrics should be explicitly non-oracle")
    assert_true(summary["lpips_status"] in {"ok", "unavailable"} or summary["lpips_status"].startswith("failed:"), "LPIPS status should be explicit")
    assert_true(summary["cluster_method"] in {"deterministic", "constrained_kmeans"}, "cluster method should be explicit")


def test_cli_lpips_unavailable_is_nonfatal():
    code, _exc, summary, _output = run_image_main([*tiny_cli_args(1), "--metrics", "lpips"])

    assert_true(code == 0, "missing LPIPS dependency should not fail the attack")
    assert_true(summary["lpips"] == "n/a" or summary["lpips"] != "", "LPIPS field should be present")
    assert_true(summary["lpips_status"] in {"ok", "unavailable"} or summary["lpips_status"].startswith("failed:"), "LPIPS status should explain availability")


def test_cli_lpips_fake_module_reports_value():
    previous = sys.modules.get("lpips")

    class FakeLpipsModule:
        class LPIPS:
            def __init__(self, net="alex"):
                self.net = net

            def to(self, device=None):
                return self

            def eval(self):
                return self

            def __call__(self, recovered, reference):
                return torch.ones(recovered.shape[0], 1, 1, 1, device=recovered.device) * 0.25

    sys.modules["lpips"] = FakeLpipsModule()
    try:
        code, _exc, summary, _output = run_image_main([*tiny_cli_args(1), "--metrics", "lpips"])
    finally:
        if previous is None:
            sys.modules.pop("lpips", None)
        else:
            sys.modules["lpips"] = previous

    assert_true(code == 0, "fake LPIPS dependency smoke should succeed")
    assert_true(summary["lpips_status"] == "ok", "fake LPIPS dependency should report ok")
    assert_true(summary["lpips"] == "0.250000", "fake LPIPS value should be summarized")


def test_cli_synthetic_ratio_defense_matrix_smoke():
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
    base_args = [
        "--mode",
        "synthetic_ratio",
        "--dataset",
        "synthetic",
        "--n_images",
        "1",
        "--batch_size",
        "1",
        "--public_n_images",
        "1",
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
        "8",
        "--device",
        "cpu",
        "--metrics",
        "mse,psnr,patch_recovery",
    ]

    for defense, extra in defense_args.items():
        code, _exc, summary, _output = run_image_main([*base_args, "--defense", defense, *extra])
        assert_true(code == 0, f"{defense} CLI defense smoke should succeed")
        assert_true(summary["result_status"] == "ok", f"{defense} summary status should be ok")
        assert_true(summary["defense"] == defense, f"{defense} summary should report the selected defense")
        assert_true(summary["reproduction_level"] == "synthetic_ratio_debug", f"{defense} smoke should use synthetic ratio mode")
        assert_true(summary["non_oracle_primary_only"] == "true", f"{defense} summary should mark non-oracle primary metrics")


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


def test_cluster_method_resolver_is_explicit():
    method = resolve_cluster_method("auto")

    assert_true(method in {"deterministic", "constrained_kmeans"}, "auto cluster method should resolve to an explicit backend")
    assert_true(resolve_cluster_method("deterministic") == "deterministic", "deterministic cluster method should be stable")


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
        test_sampling_first_n_and_hash_are_stable,
        test_seeded_shuffle_sampling_uses_split_seed,
        test_indices_file_sampling_reads_text_and_npy,
        test_indices_file_sampling_reports_clear_errors,
        test_gradient_ratio_formula_recovers_known_patch,
        test_batch_recovery_applies_position_and_public_stats,
        test_shared_adjacent_difference_recovery_inverts_probe_transform,
        test_patch_extract_fold_roundtrip_and_public_stats,
        test_public_stats_file_roundtrip_uses_explicit_file,
        test_public_stats_move_preserves_metadata_and_device,
        test_official_public_cdf_probe_is_deterministic,
        test_official_gradient_ratio_recovers_batch1_without_oracle,
        test_official_source_gradient_inventory_and_lrb_layer_indices,
        test_official_matched_patch_metrics_use_one_to_one_ground_truth_matches,
        test_official_source_collects_complete_adapter_update_before_attack_selection,
        test_simple_ssim_reports_perfect_match,
        test_shared_vit_adapter_autograd_gradients_recover_patches,
        test_vit_adapter_run_reports_loss_acc_and_device_local_outputs,
        test_cli_batch1_reports_direct_primary_metric_and_shared_metadata,
        test_cli_summary_reports_sampling_protocol_fields,
        test_cli_batch2_keeps_direct_metrics_non_oracle,
        test_cli_official_batch2_reports_non_oracle_primary_metrics,
        test_cli_official_cluster_grouping_does_not_report_direct_metrics,
        test_cli_batch_size_drives_attack_batches,
        test_cli_metrics_controls_primary_report_fields,
        test_official_cifar32_profile_sets_official_like_defaults,
        test_official_cifar32_profile_respects_explicit_overrides,
        test_cli_official_cifar32_profile_batch1_smoke,
        test_cli_lpips_unavailable_is_nonfatal,
        test_cli_lpips_fake_module_reports_value,
        test_cli_synthetic_ratio_defense_matrix_smoke,
        test_cli_cifar100_fallback_is_marked_by_default,
        test_cli_cifar100_fallback_can_fail_explicitly,
        test_shared_adapter_gradient_shape_is_stable_across_batch_size,
        test_dpsgd_per_example_shared_gradient_shapes_match_full_batch,
        test_shared_recovery_counts_collisions_as_unresolved,
        test_shared_recovery_without_oracle_keeps_batch_candidates_unordered,
        test_clustering_reassembly_is_deterministic_for_fixed_seed,
        test_clustered_reassembly_changes_patch_order_for_unsorted_inputs,
        test_clustering_reassembly_preserves_device,
        test_cluster_method_resolver_is_explicit,
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

