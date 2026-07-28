"""Failure-path tests for the Qwen3 DAGER gradient-diagnostic sidecar."""

from __future__ import annotations

from contextlib import redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch


SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_none_attack.py"
SPEC = importlib.util.spec_from_file_location("qwen3_none_attack_sidecar_test", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load attack script for isolated test: {SCRIPT_PATH}")
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


class _Tokenizer:
    eos_token_id = 2

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [f"token-{token_id}" for token_id in token_ids]


def _failure_diagnostic() -> dict[str, object]:
    def layer(*, passed: bool, active: float, inactive: float) -> dict[str, object]:
        return {
            "passed": passed,
            "identity": {
                "gradient_relative_error": 0.01,
                "gradient_t_relative_error": 1.4,
            },
            "rank": {
                "relative_threshold_rank": 4,
                "theoretical_rank_cap": 3,
                "spectral_gap_suggestion": {
                    "suggested_rank": 3,
                    "largest_gap_ratio": 20.0,
                    "note": "reporting_only_not_used_to_change_any_configuration",
                },
            },
            "row_space_residual": {"active_tokens": {"max": active}},
            "gradient_t_negative_control": {
                "gradient_t_active_token_residual": {"max": 0.99},
            },
            "per_token": [
                {"active_by_delta": True, "relative_row_space_residual": active},
                {"active_by_delta": False, "relative_row_space_residual": inactive},
            ],
            "checks": {
                "gradient_identity": passed,
                "relative_rank_within_theoretical_cap": passed,
                "active_token_residual": passed,
            },
        }

    return {
        "passed": False,
        "layers": {
            "q0": layer(passed=False, active=0.8, inactive=0.9),
            "q1": layer(passed=True, active=1e-5, inactive=0.7),
        },
    }


class GradientFailureSidecarTest(unittest.TestCase):
    def test_precision_specific_diagnostic_thresholds(self) -> None:
        self.assertEqual(
            RUNNER._diagnostic_thresholds("float32")["max_active_relative_residual"],
            2e-4,
        )
        self.assertEqual(
            RUNNER._diagnostic_thresholds("bfloat16")["max_active_relative_residual"],
            3e-3,
        )

    def test_failure_writes_summary_before_any_attack_search(self) -> None:
        q_parameter_names = (
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.1.self_attn.q_proj.weight",
        )
        captured = SimpleNamespace(
            q_inputs=(torch.zeros((1, 2, 2)), torch.zeros((1, 2, 2))),
            q_output_gradients=(torch.zeros((1, 2, 2)), torch.zeros((1, 2, 2))),
            q_gradients=(torch.zeros((2, 2)), torch.zeros((2, 2))),
            q_parameter_names=q_parameter_names,
        )
        sample = SimpleNamespace(input_ids=(11, 2), eos_token_id=2, label=1)
        bundle = SimpleNamespace(model=object(), tokenizer=_Tokenizer(), device=torch.device("cpu"))
        diagnostic = _failure_diagnostic()

        outputs_root = PROJECT_ROOT / "outputs"
        outputs_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=outputs_root) as temporary_directory:
            output_path = Path(temporary_directory) / "run1.jsonl"
            output_argument = output_path.relative_to(REPOSITORY_ROOT).as_posix()
            args = SimpleNamespace(
                defense="none",
                config="Projection_lrb_qwen3/configs/experiment.json",
                stage="smoke",
                sample_key="a" * 64,
                head_seed=404,
                device="cpu",
                dtype="float32",
                output=output_argument,
            )
            with mock.patch.object(RUNNER, "preflight_legacy_dager_rouge_backend", return_value=SimpleNamespace()), mock.patch.object(
                RUNNER, "load_experiment_config", return_value=SimpleNamespace(model_path="unused")
            ), mock.patch.object(RUNNER, "registered_head_seed"), mock.patch.object(
                RUNNER, "load_registered_sample", return_value=sample
            ), mock.patch.object(RUNNER, "load_none_attack_controls"), mock.patch.object(
                RUNNER, "load_local_qwen3_sequence_classifier", return_value=bundle
            ), mock.patch.object(RUNNER, "capture_single_example_gradients", return_value=captured), mock.patch.object(
                RUNNER,
                "build_canonical_gradient_manifest",
                return_value={
                    "entries": [
                        {"name": q_parameter_names[0], "canonical_index": 0},
                        {"name": q_parameter_names[1], "canonical_index": 1},
                    ]
                },
            ), mock.patch.object(RUNNER, "diagnose_two_q_projections", return_value=diagnostic), mock.patch.object(
                RUNNER, "shared_dager_rank_for_qwen3_qproj_gradients", side_effect=AssertionError("must not decompose")
            ) as shared_rank, mock.patch.object(
                RUNNER, "filter_qwen3_vocab_layer1", side_effect=AssertionError("must not scan vocabulary")
            ) as layer1_filter, mock.patch.object(
                RUNNER, "decode_qwen3_rope_prefixes", side_effect=AssertionError("must not search prefixes")
            ) as layer2_decoder:
                with mock.patch.object(RUNNER, "parse_args", return_value=args), redirect_stderr(io.StringIO()) as error_stream:
                    exit_code = RUNNER.main()

            sidecar_path = output_path.with_suffix(".gradient_diagnostic.json")
            self.assertEqual(exit_code, 2)
            self.assertIn("details were written", error_stream.getvalue())
            self.assertTrue(sidecar_path.is_file())
            document = json.loads(sidecar_path.read_text(encoding="utf-8"))
            self.assertEqual(document["status"], "failed_gradient_diagnostic")
            self.assertEqual(document["gradient_diagnostic"], diagnostic)
            summary = document["diagnostic_summary"]
            self.assertFalse(summary["passed"])
            self.assertIn("gradient_identity", summary["failed_checks"]["q0"])
            self.assertIn("q0.gradient_identity", summary["failure_reasons"])
            self.assertEqual(summary["q0"]["relative_effective_rank"], 4)
            self.assertEqual(summary["q0"]["theoretical_rank_cap"], 3)
            self.assertEqual(summary["q0"]["max_inactive_relative_residual"], 0.9)
            self.assertEqual(summary["q0"]["negative_control_identity_error"], 1.4)
            self.assertEqual(
                summary["diagnostic_thresholds"]["max_active_relative_residual"],
                2e-4,
            )
            shared_rank.assert_not_called()
            layer1_filter.assert_not_called()
            layer2_decoder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
