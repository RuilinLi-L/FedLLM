"""Eligibility, scoring, tie-break, and shared-core tests for Stage 5."""

from __future__ import annotations

import ast
import importlib.util
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

from src.calibration import CalibrationManifest, CalibrationRunContext, CalibrationSample, _default_executor
from src.config import ExperimentConfig
from src.config_selection import select_calibration_configuration
from src.hashing import sha256_json


SAMPLES = ("a" * 64, "b" * 64)
RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_none_attack.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("qwen3_none_attack_equivalence", RUNNER_PATH)
if RUNNER_SPEC is None or RUNNER_SPEC.loader is None:
    raise RuntimeError(f"Unable to load standalone none runner: {RUNNER_PATH}")
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = RUNNER
RUNNER_SPEC.loader.exec_module(RUNNER)


def _rows(parameters: dict[str, object], *, token: float, exact: bool, rouge: float, cost: float, failed: bool = False) -> list[dict[str, object]]:
    identifier = sha256_json(parameters)
    return [{"candidate_id": identifier, "parameters": parameters, "sample_key": key, "result_status": "error" if failed else "ok", "token_recovery": token, "exact_recovery": exact, "rouge_1": rouge / 2, "rouge_2": rouge / 2, "evaluated_prefix_cost": cost} for key in SAMPLES]


class SelectionRuleTest(unittest.TestCase):
    def test_failed_candidate_is_not_eligible_and_all_failed_selects_none(self) -> None:
        rows = _rows({"tau2": .001}, token=1, exact=True, rouge=2, cost=1, failed=True)
        result = select_calibration_configuration(rows, expected_sample_keys=SAMPLES)
        self.assertIsNone(result.selected)
        self.assertFalse(result.candidates[0].eligible)

    def test_fixed_scoring_and_candidate_id_tie_break(self) -> None:
        best = {"tau2": .002}
        lower = {"tau2": .001}
        rows = _rows(best, token=.8, exact=True, rouge=1.4, cost=20)
        rows += _rows(lower, token=.8, exact=True, rouge=1.4, cost=20)
        selected = select_calibration_configuration(rows, expected_sample_keys=SAMPLES).selected
        self.assertIsNotNone(selected)
        self.assertEqual(selected.candidate_id, min(sha256_json(best), sha256_json(lower)))

    def test_calibration_default_executor_and_standalone_runner_use_same_core_fields(self) -> None:
        sample = CalibrationSample("a" * 64, 0, "x", 1, (1, 7), 7)
        manifest = CalibrationManifest(Path("calibration.jsonl"), "b" * 64, "c" * 64, "d" * 64, (sample,))
        config = ExperimentConfig(Path("c"), Path("repo"), Path("repo/Projection_lrb_qwen3"), Path("model"), Path("data"), Path("out"), 32, 1, 11, 22, (101,202,303), 1, {}, {}, {}, "e" * 64)
        parameters = {"tau1": .002, "tau2": .001, "numerical_rank_threshold": .001, "rank_cutoff": 20, "candidate_budget": {"max_ids": -1}, "search_budget": {"maxC": 10, "parallel": 1}}
        context = CalibrationRunContext(config, manifest, sample, parameters, "f" * 64, "g" * 64, "h" * 64, "cpu", "float32")
        core = {"status": "ok", "tau1": .002, "tau2": .001, "layer_1_candidate_count": 3, "reconstructed_token_ids": [1,7], "token_recovery": 1.0, "exact_recovery": True, "rouge_1": 1.0, "rouge_2": 1.0, "termination_reason": "completed_prefix_found"}
        with mock.patch("src.calibration.execute_none_only_dager", return_value=core) as calibration_core:
            from_calibration = _default_executor(context, object())
        calibration_core.assert_called_once()
        core_keys = ("tau1", "tau2", "layer_1_candidate_count", "reconstructed_token_ids", "token_recovery", "exact_recovery", "rouge_1", "rouge_2", "termination_reason")
        self.assertEqual({key: from_calibration[key] for key in core_keys}, {key: core[key] for key in core_keys})

        standalone_core = {**core, "thresholds": {"l1_span_thresh": .002, "l2_span_thresh": .001}, "search_budget": {"evaluated_prefix_count": 7}}
        frozen = {
            "selected_tau1": .002, "frozen_control_identity_sha256": "a" * 64,
            "aggregation_sha256": "b" * 64, "bfloat16_gate": .0075,
            "bf16_gate_amendment_identity": "c" * 64, "preregistration_sha256": "d" * 64,
            "calibration_sample_list_sha256": "e" * 64,
        }
        sample_for_runner = SimpleNamespace(
            sample_key="a" * 64, original_index=0, stage="smoke", preregistration_sha256="d" * 64
        )
        controls = SimpleNamespace(
            l1_span_threshold=.002, l2_span_threshold=.001, rank_tolerance=.001, rank_cutoff=20,
            max_search_candidates=10, max_candidate_ids=-1, decode_batch_size=1, max_sequence_length=32,
        )
        with TemporaryDirectory(dir=PROJECT_ROOT / "outputs") as temporary:
            output = Path(temporary) / "result.jsonl"
            arguments = SimpleNamespace(
                defense="none", config="Projection_lrb_qwen3/configs/experiment.json", stage="smoke",
                sample_key="a" * 64, head_seed=22,
                tau1_control="Projection_lrb_qwen3/frozen_controls/qwen3_none_tau1_calibration.json",
                device="cpu", dtype="float32", output=output.relative_to(REPOSITORY_ROOT).as_posix(),
            )
            with mock.patch.object(RUNNER, "verify_frozen_tau1_control", return_value=frozen), mock.patch.object(
                RUNNER, "load_experiment_config", return_value=SimpleNamespace(model_path=Path("model"))
            ), mock.patch.object(RUNNER, "registered_head_seed"), mock.patch.object(
                RUNNER, "load_registered_sample", return_value=sample_for_runner
            ), mock.patch.object(RUNNER, "load_none_attack_controls", return_value=controls), mock.patch.object(
                RUNNER, "preflight_legacy_dager_rouge_backend", return_value=object()
            ), mock.patch.object(RUNNER, "execute_none_only_dager", return_value=standalone_core), mock.patch.object(
                RUNNER, "sha256_file", return_value="f" * 64
            ):
                from_standalone = RUNNER.run_attack(arguments)
        self.assertEqual(
            {key: from_standalone[key] for key in core_keys},
            {key: from_calibration[key] for key in core_keys},
        )
        source = (PROJECT_ROOT / "scripts" / "run_none_attack.py").read_text(encoding="utf-8")
        self.assertIn("execute_none_only_dager(", source)
        self.assertNotIn("filter_qwen3_vocab_layer1", source)
        imported = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.append(node.module)
        self.assertFalse(any("lrb" in name.lower() for name in imported))


if __name__ == "__main__":
    unittest.main()
