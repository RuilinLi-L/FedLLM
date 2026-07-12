#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile

import torch
from torch import nn
from torchvision.models.vision_transformer import VisionTransformer

from scripts import peftleak_image_results as results
from train_peftleak_image_utility import main as utility_main
from utils.peftleak_image_utility import (
    AdapterizedVisionTransformer,
    BottleneckAdapter,
    DEBUG_PROFILE,
    apply_shared_adapter_defense,
    build_utility_model,
    validate_parameter_scopes,
)


ROOT = Path(__file__).resolve().parent


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def lrb_args(defense: str = "proj_only"):
    return SimpleNamespace(
        defense=defense,
        rng_seed=17,
        defense_rng_step=0,
        defense_pct_mask=None,
        defense_noise=None,
        defense_topk_ratio=0.1,
        defense_n_bits=8,
        defense_lrb_preset="custom",
        defense_lrb_sensitive_n_layers=2,
        defense_lrb_keep_ratio_sensitive=0.5,
        defense_lrb_keep_ratio_other=0.75,
        defense_lrb_clip_scale_sensitive=0.5,
        defense_lrb_clip_scale_other=1.0,
        defense_lrb_noise_sensitive=0.03,
        defense_lrb_noise_other=0.005,
        defense_lrb_empirical_weight=0.6,
        defense_lrb_calibration_samples=4096,
        defense_lrb_projection="signed_pool",
    )


def tiny_model(num_layers: int = 2) -> AdapterizedVisionTransformer:
    backbone = VisionTransformer(
        image_size=32,
        patch_size=8,
        num_layers=num_layers,
        num_heads=4,
        hidden_dim=64,
        mlp_dim=128,
        num_classes=100,
    )
    return AdapterizedVisionTransformer(backbone, bottleneck_dim=8, num_classes=100)


def test_bottleneck_adapter_is_identity_at_initialization():
    adapter = BottleneckAdapter(hidden_dim=16, bottleneck_dim=4)
    inputs = torch.randn(3, 5, 16)
    assert_true(torch.equal(adapter(inputs), inputs), "zero-initialized Adapter must start as exact identity")


def test_twelve_layer_model_exposes_exact_official_adapter_tensor_count():
    model = tiny_model(num_layers=12)
    validate_parameter_scopes(model, formal=True)
    assert_true(len(model.shared_parameters()) == 96, "12-layer dual-Adapter ViT must expose 96 tensors")
    assert_true(len(model.local_head_parameters()) == 2, "classification head must remain a local scope")


def test_defense_changes_only_shared_adapter_gradients():
    model = build_utility_model(profile=DEBUG_PROFILE, bottleneck_dim=8, pretrained=False)
    images = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([1, 2, 3, 4])
    loss = nn.CrossEntropyLoss()(model(images), labels)
    loss.backward()
    raw_shared = [parameter.grad.detach().clone() for parameter in model.shared_parameters()]
    raw_head = [parameter.grad.detach().clone() for parameter in model.local_head_parameters()]
    apply_shared_adapter_defense(model, lrb_args())
    defended_shared = [parameter.grad.detach().clone() for parameter in model.shared_parameters()]
    defended_head = [parameter.grad.detach().clone() for parameter in model.local_head_parameters()]
    assert_true(
        any(not torch.equal(before, after) for before, after in zip(raw_shared, defended_shared)),
        "Projection-LRB should transform at least one shared Adapter gradient",
    )
    assert_true(
        all(torch.equal(before, after) for before, after in zip(raw_head, defended_head)),
        "local classification-head gradients must remain untouched",
    )


def test_none_defense_preserves_shared_gradients_exactly():
    model = tiny_model()
    loss = nn.CrossEntropyLoss()(model(torch.randn(2, 3, 32, 32)), torch.tensor([2, 3]))
    loss.backward()
    before = [parameter.grad.detach().clone() for parameter in model.shared_parameters()]
    apply_shared_adapter_defense(model, lrb_args(defense="none"))
    after = [parameter.grad.detach().clone() for parameter in model.shared_parameters()]
    assert_true(all(torch.equal(a, b) for a, b in zip(before, after)), "none must be gradient identity")


def test_optimizer_step_updates_adapters_and_head_but_not_backbone():
    model = tiny_model()
    shared_before = [parameter.detach().clone() for parameter in model.shared_parameters()]
    head_before = [parameter.detach().clone() for parameter in model.local_head_parameters()]
    frozen_name, frozen_parameter = model.frozen_parameter_items()[0]
    frozen_before = frozen_parameter.detach().clone()
    optimizer = torch.optim.SGD(
        [
            {"params": model.shared_parameters()},
            {"params": model.local_head_parameters()},
        ],
        lr=0.1,
    )
    optimizer.zero_grad(set_to_none=True)
    loss = nn.CrossEntropyLoss()(model(torch.randn(4, 3, 32, 32)), torch.tensor([1, 2, 3, 4]))
    loss.backward()
    apply_shared_adapter_defense(model, lrb_args(defense="none"))
    optimizer.step()
    assert_true(
        any(not torch.equal(before, after) for before, after in zip(shared_before, model.shared_parameters())),
        "at least one shared Adapter parameter must update",
    )
    assert_true(
        any(not torch.equal(before, after) for before, after in zip(head_before, model.local_head_parameters())),
        "local classification head must update",
    )
    assert_true(
        torch.equal(frozen_before, frozen_parameter),
        f"frozen pretrained parameter changed: {frozen_name}",
    )


def test_head_only_control_updates_head_without_adapter_update():
    model = tiny_model()
    shared_before = [parameter.detach().clone() for parameter in model.shared_parameters()]
    head_before = [parameter.detach().clone() for parameter in model.local_head_parameters()]
    optimizer = torch.optim.SGD(
        [
            {"params": model.shared_parameters(), "lr": 0.0},
            {"params": model.local_head_parameters(), "lr": 0.1},
        ]
    )
    loss = nn.CrossEntropyLoss()(model(torch.randn(4, 3, 32, 32)), torch.tensor([1, 2, 3, 4]))
    loss.backward()
    apply_shared_adapter_defense(model, lrb_args(defense="none"))
    optimizer.step()
    assert_true(
        all(torch.equal(before, after) for before, after in zip(shared_before, model.shared_parameters())),
        "head_only must freeze Adapter updates",
    )
    assert_true(
        any(not torch.equal(before, after) for before, after in zip(head_before, model.local_head_parameters())),
        "head_only must still train the local classification head",
    )


def test_debug_cli_writes_nonreportable_checkpoint_and_summary():
    runtime_root = ROOT / ".runtime"
    runtime_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=runtime_root) as tmp:
        output_dir = Path(tmp) / "run"
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = utility_main(
                [
                    "--profile",
                    "debug_tiny",
                    "--device",
                    "cpu",
                    "--num_epochs",
                    "1",
                    "--batch_size",
                    "4",
                    "--eval_batch_size",
                    "4",
                    "--num_workers",
                    "0",
                    "--adapter_bottleneck_dim",
                    "8",
                    "--debug_train_size",
                    "8",
                    "--debug_validation_size",
                    "4",
                    "--debug_test_size",
                    "4",
                    "--no_pretrained",
                    "--no-amp",
                    "--defense",
                    "none",
                    "--output_dir",
                    str(output_dir),
                ]
            )
        output = buffer.getvalue()
        assert_true(code == 0, f"debug utility CLI failed:\n{output}")
        assert_true("reportable=false" in output, "synthetic debug utility must not be reportable")
        assert_true((output_dir / "best_adapter_head.pt").is_file(), "debug utility must save checkpoint")
        assert_true((output_dir / "summary.json").is_file(), "debug utility must save JSON summary")


def test_results_deduplicate_seed_and_exclude_debug_utility():
    privacy_row = {
        "result_status": "ok",
        "attack": "peftleak_official_image",
        "defense": "proj_only",
        "defense_param_name": "defense_lrb_keep_ratio_sensitive",
        "defense_param_value": "0.5",
        "seed": "42",
        "patch_recovery_rate": "0.0",
        "mse": "0.3",
        "ssim": "0.01",
        "lpips": "0.7",
    }
    privacy = results.aggregate_privacy([privacy_row, dict(privacy_row)])
    assert_true(privacy[0]["privacy_n_runs"] == 1, "duplicate seed logs must not become pseudoreplicates")

    utility_rows = [
        {
            "result_status": "ok",
            "reportable": "true",
            "defense": "none",
            "defense_param_name": "n/a",
            "defense_param_value": "n/a",
            "seed": "101",
            "eval_accuracy": "0.8",
        },
        {
            "result_status": "ok",
            "reportable": "true",
            "defense": "proj_only",
            "defense_param_name": "defense_lrb_keep_ratio_sensitive",
            "defense_param_value": "0.5",
            "seed": "101",
            "eval_accuracy": "0.79",
        },
        {
            "result_status": "ok",
            "reportable": "false",
            "defense": "proj_only",
            "defense_param_name": "defense_lrb_keep_ratio_sensitive",
            "defense_param_value": "0.5",
            "seed": "999",
            "eval_accuracy": "1.0",
        },
    ]
    utility = results.aggregate_utility(utility_rows)
    proj = next(row for row in utility if row["defense"] == "proj_only")
    assert_true(proj["utility_n_runs"] == 1, "debug utility must be excluded from formal aggregation")
    assert_true(abs(proj["utility_drop"] - 0.01) < 1e-9, "utility drop must be paired by seed")


def test_results_keep_protocols_separate_and_reject_conflicts():
    privacy_base = {
        "result_status": "ok",
        "attack": "peftleak_official_image",
        "dataset": "cifar100",
        "model": "official_custom_vit",
        "attack_index_count": "1",
        "img_list_path": "img_list.npy",
        "public_split": "test",
        "defense": "none",
        "defense_param_name": "n/a",
        "defense_param_value": "n/a",
        "seed": "42",
        "patch_recovery_rate": "0.5",
    }
    privacy_batch1 = {**privacy_base, "batch_size": "1", "log_path": "smoke.log"}
    privacy_batch32 = {
        **privacy_base,
        "batch_size": "32",
        "attack_index_count": "32",
        "patch_recovery_rate": "0.1",
        "log_path": "formal.log",
    }
    assert_true(
        len(results.aggregate_privacy([privacy_batch1, privacy_batch32])) == 2,
        "smoke and formal privacy protocols must remain separate",
    )
    conflicting = {**privacy_batch32, "patch_recovery_rate": "0.2", "log_path": "conflict.log"}
    try:
        results.aggregate_privacy([privacy_batch32, conflicting])
    except ValueError as exc:
        assert_true("Conflicting duplicate" in str(exc), "conflict error should name duplicate runs")
    else:
        raise AssertionError("conflicting duplicate privacy runs must fail aggregation")

    utility_base = {
        "result_status": "ok",
        "reportable": "true",
        "dataset": "cifar100",
        "profile": "formal_vit_b16",
        "model_path": "torchvision_vit_b_16",
        "pretrained_weights": "IMAGENET1K_V1",
        "shared_scope": "adapter_only",
        "local_scope": "classification_head",
        "utility_control": "standard",
        "batch_size": "128",
        "eval_batch_size": "256",
        "validation_size": "5000",
        "num_epochs": "20",
        "split_seed": "42",
        "lr_adapter": "0.001",
        "lr_head": "0.001",
        "weight_decay": "0.01",
        "warmup_epochs": "1",
        "amp": "true",
        "defense": "none",
        "defense_param_name": "n/a",
        "defense_param_value": "n/a",
        "seed": "101",
        "eval_accuracy": "0.8",
    }
    utility_r64 = {**utility_base, "adapter_bottleneck_dim": "64"}
    utility_r32 = {**utility_base, "adapter_bottleneck_dim": "32"}
    assert_true(
        len(results.aggregate_utility([utility_r64, utility_r32])) == 2,
        "different Adapter bottlenecks must remain separate",
    )


def test_cross_protocol_comparison_requires_matching_seeds_and_excludes_head_only():
    privacy = [
        {
            "defense": "none",
            "defense_param_name": "n/a",
            "defense_param_value": "n/a",
            "privacy_seeds": "101 202 303",
            **{f"privacy_{field}": "privacy" for field in results.PRIVACY_PROTOCOL_FIELDS},
        }
    ]
    standard = {
        "defense": "none",
        "defense_param_name": "n/a",
        "defense_param_value": "n/a",
        "utility_seeds": "101 202 303",
        **{f"utility_{field}": "utility" for field in results.UTILITY_PROTOCOL_FIELDS},
    }
    standard["utility_utility_control"] = "standard"
    head_only = dict(standard)
    head_only["utility_utility_control"] = "head_only"
    comparison = results.build_cross_protocol_comparison(privacy, [standard, head_only])
    assert_true(len(comparison) == 1, "head-only control must not enter cross-protocol comparison")
    mismatched = dict(standard)
    mismatched["utility_seeds"] = "101 202"
    try:
        results.build_cross_protocol_comparison(privacy, [mismatched])
    except ValueError as exc:
        assert_true("seed mismatch" in str(exc), "seed mismatch should fail explicitly")
    else:
        raise AssertionError("cross-protocol comparison must reject mismatched seeds")


def main() -> int:
    tests = [
        test_bottleneck_adapter_is_identity_at_initialization,
        test_twelve_layer_model_exposes_exact_official_adapter_tensor_count,
        test_defense_changes_only_shared_adapter_gradients,
        test_none_defense_preserves_shared_gradients_exactly,
        test_optimizer_step_updates_adapters_and_head_but_not_backbone,
        test_head_only_control_updates_head_without_adapter_update,
        test_debug_cli_writes_nonreportable_checkpoint_and_summary,
        test_results_deduplicate_seed_and_exclude_debug_utility,
        test_results_keep_protocols_separate_and_reject_conflicts,
        test_cross_protocol_comparison_requires_matching_seeds_and_excludes_head_only,
    ]
    for test in tests:
        print(f"Running {test.__name__}...")
        test()
    print("All PEFTLeak image utility semantic tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
