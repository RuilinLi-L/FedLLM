#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import os
import sys
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:
    torch = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if torch is not None:
    import utils.adaptive_attack as adaptive_module
    from utils.adaptive_attack import (
        adaptive_attack_summary_fields,
        adaptive_check_if_in_span,
        adaptive_defense_aware_active,
        adaptive_get_span_dists,
        adaptive_get_top_B_in_span,
        adaptive_transform_candidates,
        prepare_adaptive_attack,
        validate_adaptive_attack_args,
    )
    from utils.functional import filter_outliers
    from utils.lrb_defense import _cached_rademacher


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
        defense_lrb_keep_ratio_sensitive=0.5,
        defense_lrb_keep_ratio_other=0.75,
        defense_lrb_seed=None,
        defense_lrb_seed_mode="static",
        defense_rng_step=0,
        adaptive_lrb_knowledge="oracle",
        adaptive_lrb_sign_source="legacy_cpu",
        adaptive_lrb_ratio_grid="auto",
        adaptive_lrb_attack_seed=None,
        adaptive_lrb_seed_samples=16,
        adaptive_lrb_hypothesis_reduce="min",
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
    assert_true(
        torch.equal(signs, adaptive_module._lrb_feature_signs(5, (2, 4), 0)),
        "legacy oracle must preserve the original CPU seed replay tensor",
    )


def test_defense_device_sign_source_matches_defense_rng_on_available_devices():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for device in devices:
        args = _base_args("lrb")
        args.adaptive_lrb_sign_source = "defense_device"
        args.dager_selected_gradient_indices = [0]
        args.dager_selected_gradient_names = ["layer.weight"]
        args.lrb_defense_layer_info = [{
            "idx": 0,
            "active": True,
            "keep_ratio": 0.5,
            "projection_seed": 12345,
            "projection_mode": "signed_pool",
            "projection_device": str(device),
            "shape": (4, 6),
        }]
        grad = torch.ones(4, 6, device=device)
        prepare_adaptive_attack(args, (grad,), parameter_names=args.dager_selected_gradient_names)

        actual = args.adaptive_attack_state["layers"][0]["lrb_feature_signs"]
        expected = _cached_rademacher(
            (1, 6),
            device=device,
            seed=12345,
            prior_shapes=((4, 1),),
        ).reshape(-1)
        fields = dict(adaptive_attack_summary_fields(args))
        assert_true(torch.equal(actual, expected), f"oracle signs must match defense RNG on {device}")
        assert_true(fields["adaptive_lrb_sign_knowledge"] == "exact", "defense-device replay must be audited as exact")


def test_legacy_sign_source_preserves_cpu_seed_replay_label():
    args = _base_args("lrb")
    args.dager_selected_gradient_indices = [0]
    args.dager_selected_gradient_names = ["layer.weight"]
    args.lrb_defense_layer_info = [{
        "idx": 0,
        "active": True,
        "keep_ratio": 0.5,
        "projection_seed": 7,
        "projection_mode": "signed_pool",
        "projection_device": "cuda",
        "shape": (4, 6),
    }]
    prepare_adaptive_attack(args, (torch.ones(4, 6),), parameter_names=args.dager_selected_gradient_names)

    fields = dict(adaptive_attack_summary_fields(args))
    assert_true(fields["adaptive_lrb_sign_source"] == "legacy_cpu", "old commands must retain legacy CPU replay")
    assert_true(
        fields["adaptive_lrb_sign_knowledge"] == "legacy_seed_replay",
        "legacy replay must not be mislabeled as exact",
    )


def _hypothesis_signature(args):
    hypotheses = args.adaptive_attack_state["layers"][0]["lrb_hypotheses"]
    return [
        (
            item["keep_ratio"],
            item["projection_seed"],
            None if item["feature_signs"] is None else item["feature_signs"].clone(),
        )
        for item in hypotheses
    ]


def test_method_only_does_not_read_oracle_lrb_metadata():
    args_a = _base_args("lrb")
    args_a.adaptive_lrb_knowledge = "method_only"
    args_a.adaptive_lrb_sign_source = "defense_device"
    args_a.adaptive_lrb_ratio_grid = "0.4,0.6"
    args_a.adaptive_lrb_attack_seed = 9001
    args_a.adaptive_lrb_seed_samples = 2
    args_a.dager_selected_gradient_indices = [0]
    args_a.dager_selected_gradient_names = ["transformer.h.0.attn.c_attn.weight"]
    args_a.lrb_defense_layer_info = [
        {"idx": 0, "active": True, "keep_ratio": 0.1, "projection_seed": 111, "projection_device": "cuda:99", "shape": (4, 4)}
    ]
    grad = torch.ones(4, 4)
    prepare_adaptive_attack(args_a, (grad,), parameter_names=args_a.dager_selected_gradient_names)
    first = _hypothesis_signature(args_a)

    args_b = _base_args("lrb")
    args_b.adaptive_lrb_knowledge = "method_only"
    args_b.adaptive_lrb_sign_source = "defense_device"
    args_b.adaptive_lrb_ratio_grid = "0.4,0.6"
    args_b.adaptive_lrb_attack_seed = 9001
    args_b.adaptive_lrb_seed_samples = 2
    args_b.dager_selected_gradient_indices = [0]
    args_b.dager_selected_gradient_names = list(args_a.dager_selected_gradient_names)
    args_b.lrb_defense_layer_info = [
        {"idx": 0, "active": True, "keep_ratio": 0.9, "projection_seed": 999, "projection_device": "cpu", "shape": (4, 4)}
    ]
    prepare_adaptive_attack(args_b, (grad,), parameter_names=args_b.dager_selected_gradient_names)
    second = _hypothesis_signature(args_b)

    assert_true(len(first) == len(second) == 4, "method-only should build ratio x seed hypotheses")
    for left, right in zip(first, second):
        assert_true(left[0] == right[0], "method-only ratios must ignore oracle metadata")
        assert_true(left[1] == right[1], "method-only seeds must ignore oracle metadata")
        assert_true(torch.equal(left[2], right[2]), "method-only signs must ignore oracle metadata")


def test_hidden_knowledge_profiles_isolate_ratio_and_sign_metadata():
    grad = torch.ones(4, 4)
    metadata = [{
        "idx": 0,
        "active": True,
        "keep_ratio": 0.55,
        "projection_seed": 123,
        "projection_mode": "signed_pool",
        "projection_device": "cpu",
        "shape": (4, 4),
    }]

    ratio_args = _base_args("lrb")
    ratio_args.adaptive_lrb_knowledge = "ratio_hidden"
    ratio_args.adaptive_lrb_sign_source = "defense_device"
    ratio_args.adaptive_lrb_ratio_grid = "0.25,0.75"
    ratio_args.dager_selected_gradient_indices = [0]
    ratio_args.dager_selected_gradient_names = ["layer.weight"]
    ratio_args.lrb_defense_layer_info = metadata
    prepare_adaptive_attack(ratio_args, (grad,), parameter_names=ratio_args.dager_selected_gradient_names)
    ratio_hypotheses = ratio_args.adaptive_attack_state["layers"][0]["lrb_hypotheses"]
    assert_true([item["keep_ratio"] for item in ratio_hypotheses] == [0.25, 0.75], "ratio-hidden must use the grid")
    assert_true(all(item["projection_seed"] == 123 for item in ratio_hypotheses), "ratio-hidden must retain exact signs")
    assert_true(
        dict(adaptive_attack_summary_fields(ratio_args))["adaptive_lrb_sign_knowledge"] == "exact",
        "ratio-hidden defense-device replay must be audited as exact",
    )

    sign_args = _base_args("lrb")
    sign_args.adaptive_lrb_knowledge = "signs_hidden"
    sign_args.adaptive_lrb_attack_seed = 9001
    sign_args.adaptive_lrb_seed_samples = 3
    sign_args.dager_selected_gradient_indices = [0]
    sign_args.dager_selected_gradient_names = ["layer.weight"]
    sign_args.lrb_defense_layer_info = metadata
    prepare_adaptive_attack(sign_args, (grad,), parameter_names=sign_args.dager_selected_gradient_names)
    sign_hypotheses = sign_args.adaptive_attack_state["layers"][0]["lrb_hypotheses"]
    assert_true(len(sign_hypotheses) == 3, "sign-hidden must build the requested seed ensemble")
    assert_true(all(item["keep_ratio"] == 0.55 for item in sign_hypotheses), "sign-hidden must retain the exact ratio")
    assert_true(all(item["projection_seed"] != 123 for item in sign_hypotheses), "sign-hidden must not reuse the true seed")
    fields = dict(adaptive_attack_summary_fields(sign_args))
    assert_true(fields["adaptive_lrb_attack_seed"] == 9001, "summary must record the hidden-sign attack seed")


def test_lrb_hypothesis_reducers_match_min_and_mean_distances():
    args = _base_args("lrb")
    hypotheses = [
        {"keep_ratio": 0.5, "projection_mode": "pool", "projection_seed": 0, "feature_signs": None},
        {"keep_ratio": 0.75, "projection_mode": "pool", "projection_seed": 0, "feature_signs": None},
    ]
    args.adaptive_attack_state = {"layers": {0: {"lrb_hypotheses": hypotheses}}}
    values = torch.tensor([[[1.0, 4.0, 2.0, 8.0], [3.0, 1.0, 7.0, 2.0]]])
    span = torch.eye(4)[:2]

    individual = []
    for hypothesis in hypotheses:
        transformed = adaptive_module._project_last_dim_signed_pool(
            values.clone(),
            hypothesis["keep_ratio"],
            seed=hypothesis["projection_seed"],
            mode=hypothesis["projection_mode"],
        )
        individual.append(adaptive_module.check_if_in_span(span, transformed, "l2"))
    stacked = torch.stack(individual)

    args.adaptive_lrb_hypothesis_reduce = "min"
    actual_min = adaptive_check_if_in_span(args, span, values.clone(), "l2", layer_position=0)
    args.adaptive_lrb_hypothesis_reduce = "mean"
    actual_mean = adaptive_check_if_in_span(args, span, values.clone(), "l2", layer_position=0)

    assert_true(torch.allclose(actual_min, stacked.amin(dim=0)), "min reducer must be attacker-friendly per candidate")
    assert_true(torch.allclose(actual_mean, stacked.mean(dim=0)), "mean reducer must average hypothesis distances")


def test_adaptive_span_check_preserves_plain_shape():
    args = _base_args("compression")
    args.dager_selected_gradient_indices = []
    prepare_adaptive_attack(args, (), parameter_names=[])

    values = torch.eye(3).unsqueeze(0)
    rq = torch.eye(3)[:2]
    sizes = adaptive_check_if_in_span(args, rq, values, "l2", layer_position=0)

    assert_true(tuple(sizes.shape) == (1, 3), "adaptive span check should preserve check_if_in_span output shape")


class _SpanDistanceModelWrapper:
    def __init__(self, layer_inputs=None):
        self.args = SimpleNamespace(device="cpu", n_layers=2)
        self.layer_inputs = layer_inputs
        self.layer_input_calls = 0

    def get_layer_inputs(self, sentences, layers):
        self.layer_input_calls += 1
        if self.layer_inputs is None:
            raise AssertionError("p > 0 must not compute unused layer inputs")
        return [value.clone() for value in self.layer_inputs]


def _span_distance_args(verbose=False):
    return SimpleNamespace(
        adaptive_attack="none",
        defense_adaptive_decoding=False,
        defense="noise",
        dist_norm="l2",
        n_layers=2,
        verbose_attack_debug=verbose,
    )


def _manual_span_logit_mean(args, matrices, values):
    distances = [
        adaptive_check_if_in_span(
            args,
            matrix,
            value.clone(),
            args.dist_norm,
            layer_position=layer_position,
        ).T
        if layer_position == 0
        else adaptive_check_if_in_span(
            args,
            matrix,
            value.clone(),
            args.dist_norm,
            layer_position=layer_position,
        )
        for layer_position, (matrix, value) in enumerate(zip(matrices, values))
    ]
    joined = torch.cat(distances, dim=1).clamp(min=1e-12, max=1 - 1e-12)
    return (torch.log(joined) - torch.log(1 - joined)).mean(dim=1)


def test_noisy_span_distances_skip_unused_layer_forward_after_position_zero():
    args = _span_distance_args()
    wrapper = _SpanDistanceModelWrapper()
    matrix = torch.tensor([[1.0, 0.0, 0.0]])
    embeds = torch.eye(3).unsqueeze(0)
    expected = _manual_span_logit_mean(args, [matrix], [embeds])

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        actual = adaptive_get_span_dists(args, wrapper, [matrix], embeds.clone(), p=3)

    assert_true(wrapper.layer_input_calls == 0, "p > 0 should skip the unused vocabulary layer forward")
    assert_true(torch.allclose(actual, expected), "p > 0 optimized distances should match the original formula")
    assert_true(output.getvalue() == "", "span-distance debug output should be disabled by default")


def test_noisy_span_distances_keep_position_zero_multilayer_semantics():
    args = _span_distance_args()
    matrices = [
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 1.0, 0.0]]),
    ]
    embeds = torch.eye(3).unsqueeze(0)
    layer_inputs = torch.eye(3).unsqueeze(1)
    wrapper = _SpanDistanceModelWrapper([layer_inputs])
    expected = _manual_span_logit_mean(args, matrices, [embeds, layer_inputs])

    actual = adaptive_get_span_dists(args, wrapper, matrices, embeds.clone(), p=0)

    assert_true(wrapper.layer_input_calls == 1, "p == 0 should retain the required layer forward")
    assert_true(torch.allclose(actual, expected), "p == 0 multilayer distances should preserve baseline semantics")


def test_outlier_debug_output_is_opt_in_without_changing_candidates():
    distances = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    quiet_output = io.StringIO()
    with contextlib.redirect_stdout(quiet_output):
        quiet_ids = filter_outliers(distances, std_thrs=0.5, maxB=2)

    verbose_output = io.StringIO()
    with contextlib.redirect_stdout(verbose_output):
        verbose_ids = filter_outliers(distances, std_thrs=0.5, maxB=2, verbose=True)

    assert_true(torch.equal(quiet_ids, verbose_ids), "verbosity must not change outlier candidates")
    assert_true(quiet_output.getvalue() == "", "outlier diagnostics should be quiet by default")
    assert_true("Wrong dists:" in verbose_output.getvalue(), "verbose mode should retain outlier diagnostics")


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
        test_defense_device_sign_source_matches_defense_rng_on_available_devices,
        test_legacy_sign_source_preserves_cpu_seed_replay_label,
        test_method_only_does_not_read_oracle_lrb_metadata,
        test_hidden_knowledge_profiles_isolate_ratio_and_sign_metadata,
        test_lrb_hypothesis_reducers_match_min_and_mean_distances,
        test_adaptive_span_check_preserves_plain_shape,
        test_noisy_span_distances_skip_unused_layer_forward_after_position_zero,
        test_noisy_span_distances_keep_position_zero_multilayer_semantics,
        test_outlier_debug_output_is_opt_in_without_changing_candidates,
    ]
    for test in tests:
        print(f"Running {test.__name__}...")
        test()
    print("All adaptive-attack semantic tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
