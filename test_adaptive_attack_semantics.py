#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:
    torch = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if torch is not None:
    from utils.adaptive_attack import (
        adaptive_attack_summary_fields,
        adaptive_check_if_in_span,
        adaptive_defense_aware_active,
        adaptive_get_top_B_in_span,
        adaptive_transform_candidates,
        prepare_adaptive_attack,
        validate_adaptive_attack_args,
    )


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def _base_args(defense="topk"):
    return SimpleNamespace(
        adaptive_attack="defense_aware",
        defense_adaptive_decoding=False,
        adaptive_candidate_multiplier=4,
        adaptive_candidate_cap=None,
        defense=defense,
        defense_lrb_projection="signed_pool",
        rng_seed=7,
        train_method="full",
        model_path="bert-base-uncased",
        dist_norm="l2",
        dager_selected_gradient_indices=[0, 1],
        dager_selected_gradient_names=["encoder.layer.0.attention.self.query.weight", "encoder.layer.1.attention.self.query.weight"],
    )


def test_legacy_adaptive_decoding_normalizes_to_auto_profile():
    args = SimpleNamespace(
        adaptive_attack="none",
        defense_adaptive_decoding=True,
        adaptive_candidate_multiplier=50,
        adaptive_candidate_cap=None,
        defense="lrb",
        defense_pct_mask=None,
    )
    validate_adaptive_attack_args(args)
    fields = dict(adaptive_attack_summary_fields(args))
    assert_true(args.adaptive_attack == "auto", "legacy adaptive decoding should enable auto adaptive profile")
    assert_true(fields["adaptive_attack_profile"] == "outlier_decode", "auto profile should preserve legacy outlier decoding")
    assert_true(not adaptive_defense_aware_active(args), "auto mode should not enable defense-aware span transforms")


def test_topk_support_mask_is_applied_to_candidate_vectors():
    args = _base_args("topk")
    grads = (
        torch.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 3.0, 0.0], [0.0, 4.0, 0.0]]),
    )
    prepare_adaptive_attack(args, grads, parameter_names=args.dager_selected_gradient_names)

    values = torch.ones(2, 3)
    transformed = adaptive_transform_candidates(args, values, layer_position=0)

    assert_true(torch.allclose(transformed, torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])), "top-k support mask should zero unsupported hidden dims")
    fields = dict(adaptive_attack_summary_fields(args))
    assert_true(fields["adaptive_span_transform"] == "topk_support_mask", "summary should record top-k support transform")
    assert_true(fields["adaptive_support_density_mean"] != "n/a", "summary should record support density")


def test_auto_mode_does_not_apply_topk_support_mask():
    args = _base_args("topk")
    args.adaptive_attack = "auto"
    args.defense_adaptive_decoding = True
    grads = (
        torch.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 3.0, 0.0], [0.0, 4.0, 0.0]]),
    )
    prepare_adaptive_attack(args, grads, parameter_names=args.dager_selected_gradient_names)

    values = torch.ones(2, 3)
    transformed = adaptive_transform_candidates(args, values, layer_position=0)

    assert_true(torch.allclose(transformed, values), "auto mode should leave candidate vectors untouched")


def test_ranked_adaptive_l1_ignores_too_strict_threshold():
    args = _base_args("topk")
    args.dager_selected_gradient_indices = []
    prepare_adaptive_attack(args, (), parameter_names=[])

    values = torch.eye(4).unsqueeze(0)
    rq = torch.eye(4)[:1]
    which = adaptive_get_top_B_in_span(
        args,
        rq,
        values,
        B=1,
        thresh=0.0,
        norm="l2",
        layer_position=0,
    )

    assert_true(len(which[0]) == 4, "ranked adaptive L1 should keep B * multiplier candidates even below threshold")
    assert_true(args._adaptive_l1_ranked_fallback is True, "ranked fallback should be marked when threshold has no hits")
    assert_true(args._adaptive_l1_stop_after_current_position is True, "ranked fallback should request L1 stop after current position")


def test_ranked_adaptive_l1_uses_threshold_hits_when_available():
    args = _base_args("topk")
    args.dager_selected_gradient_indices = []
    prepare_adaptive_attack(args, (), parameter_names=[])

    values = torch.eye(4).unsqueeze(0)
    rq = torch.eye(4)
    which = adaptive_get_top_B_in_span(
        args,
        rq,
        values,
        B=1,
        thresh=1e-4,
        norm="l2",
        layer_position=0,
    )

    assert_true(len(which[0]) == 4, "threshold hits should be returned when available")
    assert_true(args._adaptive_l1_ranked_fallback is False, "threshold hits should not be marked as ranked fallback")
    assert_true(args._adaptive_l1_stop_after_current_position is False, "threshold hits should not request L1 stop")


def test_lrb_projection_metadata_reuses_defense_layer_info():
    args = _base_args("lrb")
    args.lrb_defense_layer_info = [
        {
            "idx": 0,
            "active": True,
            "keep_ratio": 0.5,
            "projection_seed": 11,
            "projection_mode": "pool",
        },
        {
            "idx": 1,
            "active": True,
            "keep_ratio": 1.0,
            "projection_seed": 12,
            "projection_mode": "pool",
        },
    ]
    grads = (
        torch.ones(4, 4),
        torch.ones(4, 4),
    )
    prepare_adaptive_attack(args, grads, parameter_names=args.dager_selected_gradient_names)

    fields = dict(adaptive_attack_summary_fields(args))
    assert_true(fields["adaptive_span_transform"] == "lrb_public_projection", "summary should record LRB projection transform")
    assert_true(fields["adaptive_lrb_keep_ratio_mean"] == "0.750000", "summary should average selected LRB keep ratios")


def test_lrb_signed_projection_uses_feature_axis_signs_for_transposed_gpt2_grad():
    args = _base_args("lrb")
    args.model_path = "gpt2"
    args.lrb_defense_layer_info = [
        {
            "idx": 0,
            "active": True,
            "keep_ratio": 0.5,
            "projection_seed": 5,
            "projection_mode": "signed_pool",
            "shape": (2, 4),
        }
    ]
    args.dager_selected_gradient_indices = [0]
    args.dager_selected_gradient_names = ["transformer.h.0.attn.c_attn.weight"]
    prepare_adaptive_attack(args, (torch.ones(2, 4),), parameter_names=args.dager_selected_gradient_names)

    layer_state = args.adaptive_attack_state["layers"][0]
    signs = layer_state["lrb_feature_signs"]

    assert_true(layer_state["lrb_feature_axis"] == 0, "GPT-2 span transpose should map features to original row signs")
    assert_true(signs.numel() == 2, "feature signs should follow the oriented hidden dimension")


def test_adaptive_span_check_preserves_plain_shape():
    args = _base_args("compression")
    args.dager_selected_gradient_indices = []
    prepare_adaptive_attack(args, (), parameter_names=[])

    values = torch.eye(3).unsqueeze(0)
    rq = torch.eye(3)[:2]
    sizes = adaptive_check_if_in_span(args, rq, values, "l2", layer_position=0)

    assert_true(tuple(sizes.shape) == (1, 3), "adaptive span check should preserve check_if_in_span output shape")


def main():
    if torch is None:
        print("Skipping adaptive-attack semantic tests: torch is not installed in this Python environment.")
        return 0

    tests = [
        test_legacy_adaptive_decoding_normalizes_to_auto_profile,
        test_topk_support_mask_is_applied_to_candidate_vectors,
        test_auto_mode_does_not_apply_topk_support_mask,
        test_ranked_adaptive_l1_ignores_too_strict_threshold,
        test_ranked_adaptive_l1_uses_threshold_hits_when_available,
        test_lrb_projection_metadata_reuses_defense_layer_info,
        test_lrb_signed_projection_uses_feature_axis_signs_for_transposed_gpt2_grad,
        test_adaptive_span_check_preserves_plain_shape,
    ]
    for test in tests:
        print(f"Running {test.__name__}...")
        test()
    print("All adaptive-attack semantic tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
