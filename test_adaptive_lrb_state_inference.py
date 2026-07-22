"""Semantic checks for the standalone static-state inference primitives."""
from __future__ import annotations

import math

import torch

from utils.adaptive_attack import _lrb_feature_signs, _project_last_dim_signed_pool
from utils.lrb_defense import _layer_projection_seed, _project_low_resolution
from utils.adaptive_lrb_state_inference import (
    StaticStateEstimator,
    StateInferenceObservation,
    continuous_signed_pool,
    fit_state,
    hard_sign_ste,
    sign_agreement_mod_global_flip,
    stage_grads_for_decode,
    stage_selected_grads_cpu,
)


def test_integer_q_matches_existing_signed_pool():
    values = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    signs = torch.tensor([1, -1, 1, -1, 1, 1, -1, -1], dtype=torch.float32)
    logits = signs * 10.0
    actual = continuous_signed_pool(values, 4.0, logits, temperature=0.1)
    expected = _project_last_dim_signed_pool(values, 4 / 8, seed=0, mode="signed_pool", feature_signs=signs)
    assert torch.allclose(actual, expected, atol=1e-6), "integer q must preserve existing candidate operator"


def test_defense_device_feature_operator_parity():
    # A one-row matrix removes the defense-side row pooling while retaining
    # the exact seed stream used to draw its column signs.  This directly
    # checks the feature operator used by the DAGER candidate-side surrogate.
    values = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    layer_seed = _layer_projection_seed(700001, 4)
    signs = _lrb_feature_signs(layer_seed, values.shape, 1, device="cpu")
    defended = _project_low_resolution(values, 0.5, seed=layer_seed, mode="signed_pool")
    candidate_side = _project_last_dim_signed_pool(
        values, 0.5, seed=layer_seed, mode="signed_pool", feature_signs=signs
    )
    assert torch.equal(defended, candidate_side), "defense_device signs must match the candidate-side operator"


def test_hard_sign_ste_has_gradient():
    logits = torch.tensor([-0.2, 0.3, 1.1], requires_grad=True)
    output = hard_sign_ste(logits).sum()
    output.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert set(hard_sign_ste(logits.detach()).tolist()) <= {-1.0, 1.0}


def test_global_flip_is_equivalent():
    truth = torch.tensor([1.0, -1.0, 1.0, -1.0])
    assert math.isclose(sign_agreement_mod_global_flip(-truth, truth), 1.0)
    assert math.isclose(sign_agreement_mod_global_flip(torch.tensor([1.0, 1.0, -1.0, -1.0]), truth), 0.5)


def test_fit_state_accepts_observed_only_object():
    basis = torch.eye(4)[:2]
    candidates = torch.eye(4).unsqueeze(0)
    observation = StateInferenceObservation(span_bases=(basis,), candidate_values=(candidates,))
    assert not hasattr(observation, "labels") and not hasattr(observation, "reference")
    estimator = StaticStateEstimator((4,), 1, min_ratio=0.5, max_ratio=0.5 + 1e-3)
    trace = fit_state(estimator, [observation], steps=1, learning_rate=0.01)
    assert len(trace) == 1 and math.isfinite(trace[0])


def test_captured_updates_keep_only_selected_cpu_gradients():
    grads = (torch.ones(2, 2), torch.ones(2, 2) * 2, torch.ones(2, 2) * 3)
    staged = stage_selected_grads_cpu(grads, (0, 2))
    assert staged[0] is not None and staged[1] is None and staged[2] is not None
    assert staged[0].device.type == "cpu" and staged[2].device.type == "cpu"
    assert torch.equal(staged[0], grads[0]) and torch.equal(staged[2], grads[2])
    restored = stage_grads_for_decode(staged, torch.device("cpu"))
    assert restored[1] is None
    assert torch.equal(restored[0], grads[0]) and torch.equal(restored[2], grads[2])


def main():
    for test in (
        test_integer_q_matches_existing_signed_pool,
        test_defense_device_feature_operator_parity,
        test_hard_sign_ste_has_gradient,
        test_global_flip_is_equivalent,
        test_fit_state_accepts_observed_only_object,
        test_captured_updates_keep_only_selected_cpu_gradients,
    ):
        test()
    print("state-inference semantic tests passed")


if __name__ == "__main__":
    main()
