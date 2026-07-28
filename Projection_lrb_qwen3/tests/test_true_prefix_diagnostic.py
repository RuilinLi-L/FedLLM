"""Contracts for the smoke-only, read-only Qwen3 true-prefix diagnostic."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
for root in (REPOSITORY_ROOT, PROJECT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from src.dager_qwen3.true_prefix_diagnostic import diagnose_true_prefix


def _load_none_runner():
    spec = importlib.util.spec_from_file_location(
        "qwen3_none_attack_true_prefix_diagnostic_test",
        PROJECT_ROOT / "scripts" / "run_none_attack.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load standalone none runner for true-prefix diagnostic tests.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_none_runner()


class _Adapter:
    device = torch.device("cpu")
    metadata = SimpleNamespace(hidden_size=2)

    def __init__(self, q1_inputs: torch.Tensor) -> None:
        self._q1_inputs = q1_inputs

    def layer1_qproj_inputs_from_prefixes(
        self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        self.last_input_ids = input_ids.detach().clone()
        self.last_attention_mask = attention_mask.detach().clone()
        return self._q1_inputs.detach().clone()


class TruePrefixDiagnosticTest(unittest.TestCase):
    def test_reports_fixed_layer1_membership_and_strict_layer2_predicate(self) -> None:
        sample = SimpleNamespace(input_ids=(4, 7, 9))
        adapter = _Adapter(torch.zeros((1, 3, 2), dtype=torch.float32))
        layer1 = SimpleNamespace(token_ids=(4, 7))
        provider = SimpleNamespace(token_ids=(4,))
        span = SimpleNamespace(basis=torch.eye(2))
        with mock.patch(
            "src.dager_qwen3.true_prefix_diagnostic._span_distances",
            return_value=torch.tensor((0.0005, 0.001, 0.002), dtype=torch.float32),
        ):
            result = diagnose_true_prefix(
                adapter=adapter,
                sample=sample,
                layer1=layer1,
                candidate_provider=provider,
                layer2_span=span,
                threshold=0.001,
                distance_norm="l2",
            )

        self.assertFalse(result["affects_candidate_search"])
        self.assertFalse(result["affects_attack_result"])
        self.assertTrue(result["layer_1"]["first_token_in_threshold_candidate_set"])
        self.assertTrue(result["layer_1"]["first_token_in_decoder_candidate_set"])
        self.assertEqual(result["layer_1"]["missing_input_token_ids_from_threshold_candidate_set"], [9])
        self.assertEqual(result["layer_1"]["missing_input_token_ids_from_decoder_candidate_set"], [7, 9])
        self.assertEqual(
            [item["passes_threshold"] for item in result["layer_2"]["position_results"]],
            [True, False, False],
        )
        self.assertFalse(result["layer_2"]["all_positions_pass_threshold"])
        self.assertTrue(torch.equal(adapter.last_input_ids, torch.tensor(((4, 7, 9),))))
        self.assertTrue(torch.equal(adapter.last_attention_mask, torch.ones((1, 3), dtype=torch.long)))

    def test_empty_layer1_candidates_remain_a_reportable_diagnostic_outcome(self) -> None:
        sample = SimpleNamespace(input_ids=(3,))
        adapter = _Adapter(torch.zeros((1, 1, 2), dtype=torch.float32))
        span = SimpleNamespace(basis=torch.eye(2))
        with mock.patch(
            "src.dager_qwen3.true_prefix_diagnostic._span_distances",
            return_value=torch.tensor((0.0001,), dtype=torch.float32),
        ):
            result = diagnose_true_prefix(
                adapter=adapter,
                sample=sample,
                layer1=SimpleNamespace(token_ids=()),
                candidate_provider=SimpleNamespace(token_ids=()),
                layer2_span=span,
                threshold=0.001,
                distance_norm="l2",
            )
        self.assertFalse(result["layer_1"]["first_token_in_threshold_candidate_set"])
        self.assertFalse(result["layer_1"]["first_token_in_decoder_candidate_set"])
        self.assertTrue(result["layer_2"]["all_positions_pass_threshold"])

    def test_runner_rejects_true_prefix_diagnostic_outside_smoke_before_control_loading(self) -> None:
        args = SimpleNamespace(
            defense="none",
            stage="final",
            true_prefix_diagnostic=True,
            tau1_control="Projection_lrb_qwen3/frozen_controls/qwen3_none_tau1_calibration.json",
        )
        with mock.patch.object(RUNNER, "verify_frozen_tau1_control") as control:
            with self.assertRaisesRegex(RUNNER.NoneAttackScriptError, "permitted only for preregistered smoke"):
                RUNNER.run_attack(args)
        control.assert_not_called()

    def test_shared_core_places_diagnostic_after_fixed_candidate_provider(self) -> None:
        source = (PROJECT_ROOT / "src" / "dager_qwen3" / "none_attack_core.py").read_text(encoding="utf-8")
        provider_index = source.index("candidate_provider = RoPECandidateProvider.from_layer1_result(")
        diagnostic_index = source.index("if true_prefix_diagnostic:")
        decoder_index = source.index("layer2 = decode_qwen3_rope_prefixes(")
        self.assertLess(provider_index, diagnostic_index)
        self.assertLess(diagnostic_index, decoder_index)
        self.assertIn("affects_attack_result", (PROJECT_ROOT / "src" / "dager_qwen3" / "true_prefix_diagnostic.py").read_text(encoding="utf-8"))

    def test_diagnostic_source_has_no_search_or_lrb_import(self) -> None:
        source = (PROJECT_ROOT / "src" / "dager_qwen3" / "true_prefix_diagnostic.py").read_text(encoding="utf-8").lower()
        self.assertNotIn("decode_qwen3_rope_prefixes", source)
        self.assertNotIn("lrb", source)


if __name__ == "__main__":
    unittest.main()
