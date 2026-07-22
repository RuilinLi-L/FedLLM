"""Semantic checks for the standalone static-state inference primitives."""
from __future__ import annotations

import ast
import math
import tempfile
from pathlib import Path

import torch

from scripts.collect_state_inference_results import validate as validate_collected_results
from utils.adaptive_attack import _lrb_feature_signs, _project_last_dim_signed_pool
from utils.lrb_defense import _layer_projection_seed, _project_low_resolution
from utils.adaptive_lrb_state_inference import (
    StaticStateEstimator,
    StateInferenceObservation,
    continuous_signed_pool,
    fit_state,
    hard_sign_ste,
    oracle_gate_passes,
    sign_agreement_mod_global_flip,
    stage_grads_for_decode,
    stage_selected_grads_cpu,
    span_distance,
)


def _collector_row(variant: str, *, status: str = "ok", oracle_score: str = "100.000000") -> dict[str, str]:
    row = {
        "protocol": "state_inference_v1",
        "result_status": status,
        "state_lifecycle": "static",
        "state_attack_variant": variant,
        "state_fit_updates": "64" if variant == "state_estimator" else "n/a",
        "state_budget": "400" if variant == "state_estimator" else "n/a",
        "n_inputs_requested": "20",
        "n_inputs_completed": "20",
        "agg_r1fm_r2fm": oracle_score,
        "rng_seed": "1101",
        "state_selected_gradients": "4;16",
        "defense_lrb_seed_mode": "static",
        "defense_lrb_seed": "700001",
        "adaptive_lrb_sign_source": "defense_device",
        "calibration_batches": "512",
        "state_decode_gradient_storage": "held_out_selected_cpu",
        "state_dager_expansions_source": "captured_precomputed",
        "state_public_calibration_decomposition": "skipped",
        "state_truth_used_for_fit": "false",
        "state_truth_used_for_decode": "true" if variant == "oracle" else "false",
        "calibration_hash": "a" * 64,
        "target_index_hash": "b" * 64,
    }
    return row


def _collector_errors(rows: list[dict[str, str]], *, oracle_threshold: float | None) -> list[str]:
    manifest = {
        "protocol": "state_inference_v1",
        "eval_count": "20",
        "m_values": "64",
        "budgets": "400",
        "seeds": "1101",
        "calibration_batches": "512",
        "selected_gradients": "4;16",
        "defense_lrb_seed_mode": "static",
        "defense_lrb_seed": "700001",
        "adaptive_lrb_sign_source": "defense_device",
        "decode_gradient_storage": "held_out_selected_cpu",
        "dager_expansions_source": "captured_precomputed",
        "dager_decomp_device": "cuda",
        "public_calibration_decomposition": "skipped",
    }
    if oracle_threshold is not None:
        manifest["oracle_min_r1r2"] = str(oracle_threshold)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "run_manifest.txt").write_text(
            "".join(f"{key}={value}\n" for key, value in manifest.items()), encoding="utf-8"
        )
        (root / "legacy_inputs_before.sha256").write_text("same\n", encoding="utf-8")
        (root / "legacy_inputs_after.sha256").write_text("same\n", encoding="utf-8")
        (root / "exit_codes.csv").write_text("seed,exit_code\n1101,0\n", encoding="utf-8")
        return validate_collected_results(root, rows)


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


def test_global_flip_accepts_cross_device_audit_tensors():
    if not torch.cuda.is_available():
        return
    truth = torch.tensor([1.0, -1.0, 1.0, -1.0])
    assert math.isclose(sign_agreement_mod_global_flip((-truth).to("cuda"), truth), 1.0)


def test_oracle_gate_is_disabled_by_default_and_inclusive_at_threshold():
    assert oracle_gate_passes(0.0, None)
    assert oracle_gate_passes(80.0, 80.0)
    assert not oracle_gate_passes(79.999999, 80.0)


def test_collector_accepts_legacy_unconditional_coverage():
    rows = [
        _collector_row("method_only"),
        _collector_row("oracle"),
        _collector_row("state_estimator"),
    ]
    assert _collector_errors(rows, oracle_threshold=None) == []


def test_collector_accepts_oracle_gate_pass_and_estimator():
    oracle = _collector_row("oracle", oracle_score="80.000000")
    oracle.update(
        {
            "state_oracle_gate_enabled": "true",
            "state_oracle_min_r1r2": "80.000000",
            "state_oracle_gate_passed": "true",
            "state_oracle_gate_reason": "oracle_r1r2_meets_threshold",
        }
    )
    rows = [_collector_row("method_only"), oracle, _collector_row("state_estimator")]
    assert _collector_errors(rows, oracle_threshold=80.0) == []


def test_collector_accepts_complete_oracle_gate_stop_without_estimator():
    oracle = _collector_row("oracle", status="oracle_gate_stopped", oracle_score="79.999999")
    oracle.update(
        {
            "state_oracle_gate_enabled": "true",
            "state_oracle_min_r1r2": "80.000000",
            "state_oracle_gate_passed": "false",
            "state_oracle_gate_reason": "oracle_r1r2_below_threshold",
        }
    )
    rows = [_collector_row("method_only"), oracle]
    assert _collector_errors(rows, oracle_threshold=80.0) == []


def test_collector_rejects_estimator_after_oracle_gate_stop():
    oracle = _collector_row("oracle", status="oracle_gate_stopped", oracle_score="10.000000")
    oracle.update(
        {
            "state_oracle_gate_enabled": "true",
            "state_oracle_min_r1r2": "80.000000",
            "state_oracle_gate_passed": "false",
            "state_oracle_gate_reason": "oracle_r1r2_below_threshold",
        }
    )
    errors = _collector_errors(
        [_collector_row("method_only"), oracle, _collector_row("state_estimator")],
        oracle_threshold=80.0,
    )
    assert any("ran estimator conditions after the oracle gate stopped" in error for error in errors)


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


def test_observation_microbatch_matches_full_mean_update():
    observations = (
        StateInferenceObservation(span_bases=(torch.eye(4)[:2],), candidate_values=(torch.eye(4).unsqueeze(0),)),
        StateInferenceObservation(span_bases=(torch.eye(4)[1:3],), candidate_values=((torch.eye(4) * 0.75).unsqueeze(0),)),
    )
    expected = StaticStateEstimator((4,), 2)
    actual = StaticStateEstimator((4,), 2)
    actual.load_state_dict(expected.state_dict())

    optimizer = torch.optim.Adam(expected.parameters(), lr=0.01)
    losses = []
    for update_index, observation in enumerate(observations):
        logits = expected.sign_logits[0]
        q_value = expected.q_values(update_index)[0]
        transformed = continuous_signed_pool(observation.candidate_values[0], q_value, logits, temperature=2.0)
        distances = span_distance(observation.span_bases[0], transformed)
        losses.append(-0.05 * torch.logsumexp(-distances.reshape(-1) / 0.05, dim=0))
    expected_loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    expected_loss.backward()
    optimizer.step()

    progress = []
    trace = fit_state(
        actual,
        observations,
        steps=1,
        learning_rate=0.01,
        progress_callback=lambda completed, total, loss: progress.append((completed, total, loss)),
    )
    assert math.isclose(trace[0], float(expected_loss.detach()), rel_tol=1e-6, abs_tol=1e-6)
    assert progress == [(1, 1, trace[0])]
    for expected_parameter, actual_parameter in zip(expected.parameters(), actual.parameters()):
        assert torch.allclose(expected_parameter, actual_parameter, atol=1e-6, rtol=1e-6)


def test_reconstruct_precomputed_expansions_is_opt_in():
    tree = ast.parse(Path("attack.py").read_text(encoding="utf-8"))
    reconstruct = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "reconstruct")
    argument_names = [argument.arg for argument in reconstruct.args.args]
    assert argument_names[-1] == "precomputed_matrices_expansions"
    assert isinstance(reconstruct.args.defaults[-1], ast.Constant)
    assert reconstruct.args.defaults[-1].value is None


def test_runner_uses_server_legacy_roots_and_requires_them():
    runner = Path("scripts/run_adaptive_lrb_state_inference.sh").read_text(encoding="utf-8")
    assert "log/runs/adaptive_lrb_matrix_sst2_official_validation_20260718_114719" in runner
    assert "log/runs/adaptive_lrb_matrix_sst2_official_validation_20260718_114758" in runner
    assert "required legacy result root is missing" in runner
    assert 'if [ ! -e "$RUN_DIR/exit_codes.csv" ]' in runner
    assert 'MODE="gate"' in runner
    assert 'elif [ "$MODE" = "gate" ]' in runner
    assert 'elif [ "$MODE" = "formal_pilot" ]' in runner
    assert "CALIBRATION_BATCHES=64" in runner
    assert "CANDIDATES=512" in runner
    assert "DECODER_CANDIDATE_MULTIPLIER=50" in runner
    assert "ORACLE_MIN_R1R2=80" in runner
    assert "M_VALUES=64" in runner
    assert "BUDGETS=400" in runner
    assert 'command+=(--state-oracle-min-r1r2 "$ORACLE_MIN_R1R2")' in runner
    assert "--dager_decomp_device cuda" in runner
    assert "EXPECTED_CONDITIONS=" in runner
    assert 'completed_conditions=$(grep -c' in runner


def test_oracle_gate_returns_before_state_fitting_loop():
    source = Path("attack_adaptive_lrb_state_inference.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--state-oracle-min-r1r2", type=float, default=None)' in source
    gate_stop = source.index("if oracle_gate_stopped:")
    estimator_loop = source.index("for m_value in state_args.m_values:", gate_stop)
    assert gate_stop < estimator_loop
    assert 'fields["result_status"] = "oracle_gate_stopped"' in source


def test_public_calibration_skips_dager_decomposition():
    source = Path("attack_adaptive_lrb_state_inference.py").read_text(encoding="utf-8")
    start = source.index("def _capture_public_calibration_record")
    end = source.index("def _hash_dataset_indices", start)
    assert "get_matrices_expansions" not in source[start:end]
    assert '"state_public_calibration_decomposition": "skipped"' in source


def test_dataset_exposes_read_only_sample_provenance():
    source = Path("utils/data.py").read_text(encoding="utf-8")
    assert "self.selected_indices = tuple" in source


def main():
    for test in (
        test_integer_q_matches_existing_signed_pool,
        test_defense_device_feature_operator_parity,
        test_hard_sign_ste_has_gradient,
        test_global_flip_is_equivalent,
        test_global_flip_accepts_cross_device_audit_tensors,
        test_oracle_gate_is_disabled_by_default_and_inclusive_at_threshold,
        test_collector_accepts_legacy_unconditional_coverage,
        test_collector_accepts_oracle_gate_pass_and_estimator,
        test_collector_accepts_complete_oracle_gate_stop_without_estimator,
        test_collector_rejects_estimator_after_oracle_gate_stop,
        test_fit_state_accepts_observed_only_object,
        test_captured_updates_keep_only_selected_cpu_gradients,
        test_observation_microbatch_matches_full_mean_update,
        test_reconstruct_precomputed_expansions_is_opt_in,
        test_runner_uses_server_legacy_roots_and_requires_them,
        test_oracle_gate_returns_before_state_fitting_loop,
        test_public_calibration_skips_dager_decomposition,
        test_dataset_exposes_read_only_sample_provenance,
    ):
        test()
    print("state-inference semantic tests passed")


if __name__ == "__main__":
    main()
