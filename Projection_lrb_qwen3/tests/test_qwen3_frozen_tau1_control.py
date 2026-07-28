"""Tamper-resistant tests for frozen Qwen3 none-only tau1 controls."""

from __future__ import annotations

import ast
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig
from src.dager_qwen3.diagnostics import load_none_attack_controls
from src.dager_qwen3.frozen_tau1_control import (
    EXPECTED_CALIBRATION_HEAD_SEED,
    EXPECTED_CALIBRATION_SAMPLE_COUNT,
    EXPECTED_SELECTED_TAU1,
    FIXED_TAU1_GRID,
    FrozenTau1ControlError,
    verify_frozen_tau1_control,
    write_or_verify_frozen_tau1_control,
)
from src.hashing import hash_sample_list, sha256_file, sha256_json


SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_none_attack.py"
try:
    SPEC = importlib.util.spec_from_file_location("qwen3_none_attack_frozen_control_test", SCRIPT_PATH)
    if SPEC is None or SPEC.loader is None:
        raise RuntimeError(f"Unable to load attack script for isolated test: {SCRIPT_PATH}")
    RUNNER = importlib.util.module_from_spec(SPEC)
    sys.modules[SPEC.name] = RUNNER
    SPEC.loader.exec_module(RUNNER)
    RUNNER_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as error:
    RUNNER = None
    RUNNER_IMPORT_ERROR = error


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


class FrozenTau1ControlTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.repository_root = Path(self.temporary_directory.name) / "repo"
        self.project_root = self.repository_root / "Projection_lrb_qwen3"
        self.project_root.mkdir(parents=True)
        self.preregistration_sha256 = "b" * 64
        self.config_sha256 = "c" * 64
        self.samples = [
            {
                "sample_key": f"{index:064x}",
                "original_index": index,
                "sentence": f"calibration sample {index}",
                "label": index % 2,
                "tokenization": {"input_ids": [index + 1, 7], "eos_token_id": 7},
            }
            for index in range(EXPECTED_CALIBRATION_SAMPLE_COUNT)
        ]
        self.calibration_hash = hash_sample_list(self.samples)
        self.amendment_document = {
            "amendment_identity_sha256": "a" * 64,
            "selected_bfloat16_gate": 7.5e-3,
            "fixed_candidate_grid": [3e-3, 5e-3, 7.5e-3, 1e-2],
        }
        self.amendment_path = (
            self.project_root
            / "prereg_amendments"
            / "bf16_gradient_gate_pre_attack_075"
            / "amendment.json"
        )
        _write_json(self.amendment_path, self.amendment_document)
        self.amendment_summary = {
            "path": "Projection_lrb_qwen3/prereg_amendments/bf16_gradient_gate_pre_attack_075/amendment.json",
            "sha256": sha256_file(self.amendment_path),
            "amendment_identity_sha256": self.amendment_document["amendment_identity_sha256"],
            "selected_bfloat16_gate": 7.5e-3,
            "fixed_candidate_grid": self.amendment_document["fixed_candidate_grid"],
        }
        self._write_active_protocol()
        self.aggregation_path = (
            self.project_root
            / "outputs"
            / "calibration"
            / "layer1_tau1_calibration_bf16_gate075_full20"
            / "aggregation.json"
        )
        self._write_samples_and_aggregation()
        self.control_path = self.project_root / "frozen_controls" / "qwen3_none_tau1_calibration.json"
        self._patcher = mock.patch(
            "src.dager_qwen3.frozen_tau1_control.verify_bf16_gate_profile_amendment",
            return_value=self.amendment_document,
        )
        self._patcher.start()
        self.addCleanup(self._patcher.stop)
        self._git_patcher = mock.patch(
            "src.dager_qwen3.frozen_tau1_control._git_commit", return_value="g" * 40
        )
        self._git_patcher.start()
        self.addCleanup(self._git_patcher.stop)

    def _write_active_protocol(self) -> None:
        preregistration = {
            "preregistration_sha256": self.preregistration_sha256,
            "config_sha256": self.config_sha256,
            "config": {"calibration_head_seed": EXPECTED_CALIBRATION_HEAD_SEED},
            "sample_lists": {"calibration": self.samples},
            "sample_list_sha256": {"calibration": self.calibration_hash},
        }
        _write_json(self.project_root / "manifests" / "preregistration.json", preregistration)
        records = [
            {
                "record_type": "preregistered_sst2_validation_sample",
                "stage": "calibration",
                "preregistration_sha256": self.preregistration_sha256,
                "stage_sample_list_sha256": self.calibration_hash,
                "sample": sample,
            }
            for sample in self.samples
        ]
        manifest = self.project_root / "manifests" / "calibration.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("\n".join(json.dumps(row, sort_keys=True) for row in records) + "\n", encoding="utf-8")

    def _sample_record(self, sample: dict[str, object]) -> dict[str, object]:
        return {
            "record_type": "qwen3_layer1_tau1_calibration_sample",
            "status": "ok",
            "stage": "calibration",
            "layer2_invoked": False,
            "sample_key": sample["sample_key"],
            "head_seed": EXPECTED_CALIBRATION_HEAD_SEED,
            "dtype": "bfloat16",
            "identity": {"preregistration_sha256": self.preregistration_sha256},
            "gradient_diagnostic_summary": {"passed": True},
            "gradient_diagnostic_controls": {"max_active_relative_residual": 7.5e-3},
            "bf16_gate_profile_amendment": self.amendment_summary,
            "vocabulary_scan": {"vocab_size": 8, "scanned_token_count": 8},
        }

    def _per_tau(self) -> list[dict[str, object]]:
        return [
            {
                "tau": tau,
                "micro_active_position_recall": 0.96 if tau >= EXPECTED_SELECTED_TAU1 else 0.94,
                "nonempty_sample_count": EXPECTED_CALIBRATION_SAMPLE_COUNT,
            }
            for tau in FIXED_TAU1_GRID
        ]

    def _write_samples_and_aggregation(self) -> None:
        sample_output_files: list[dict[str, str]] = []
        for sample in self.samples:
            path = self.aggregation_path.parent / "samples" / f"{sample['sample_key']}.json"
            _write_json(path, self._sample_record(sample))
            sample_output_files.append(
                {
                    "sample_key": str(sample["sample_key"]),
                    "path": path.relative_to(self.repository_root).as_posix(),
                    "sha256": sha256_file(path),
                }
            )
        aggregation_identity = sha256_json(
            {
                "protocol": "qwen3_layer1_tau1_calibration_v1",
                "preregistration_sha256": self.preregistration_sha256,
                "config_sha256": self.config_sha256,
                "dtype": "bfloat16",
                "head_seed": EXPECTED_CALIBRATION_HEAD_SEED,
                "tau_grid": list(FIXED_TAU1_GRID),
                "sample_output_files": sample_output_files,
                "bf16_gate_profile_amendment": self.amendment_summary,
            }
        )
        aggregation = {
            "record_type": "qwen3_layer1_tau1_calibration_aggregation",
            "status": "ok",
            "selection_rule_passed": True,
            "layer2_invoked": False,
            "calibration_sample_count": EXPECTED_CALIBRATION_SAMPLE_COUNT,
            "fixed_tau1_grid": list(FIXED_TAU1_GRID),
            "selected_tau1": EXPECTED_SELECTED_TAU1,
            "selection_rule": {
                "minimum_micro_active_position_recall": 0.95,
                "required_nonempty_sample_count": EXPECTED_CALIBRATION_SAMPLE_COUNT,
            },
            "per_tau": self._per_tau(),
            "sample_output_files": sample_output_files,
            "bf16_gate_profile_amendment": self.amendment_summary,
            "calibration_aggregation_identity_sha256": aggregation_identity,
        }
        _write_json(self.aggregation_path, aggregation)

    def _write_control(self) -> tuple[Path, dict[str, object], bool]:
        path, document, written = write_or_verify_frozen_tau1_control(
            project_root=self.project_root,
            aggregation_path=self.aggregation_path,
            output_path=self.control_path,
        )
        return path, dict(document), written

    def _aggregation(self) -> dict[str, object]:
        return json.loads(self.aggregation_path.read_text(encoding="utf-8"))

    def test_valid_aggregation_creates_and_reverifies_frozen_control(self) -> None:
        path, document, written = self._write_control()
        self.assertTrue(written)
        self.assertEqual(path, self.control_path)
        self.assertEqual(document["record_type"], "qwen3_frozen_tau1_control")
        self.assertEqual(document["selected_tau1"], EXPECTED_SELECTED_TAU1)
        self.assertEqual(len(document["sample_output_files"]), EXPECTED_CALIBRATION_SAMPLE_COUNT)
        self.assertEqual(verify_frozen_tau1_control(project_root=self.project_root, control_path=path), document)
        _, second, second_written = self._write_control()
        self.assertFalse(second_written)
        self.assertEqual(second, document)

    def test_existing_conflicting_control_is_not_overwritten(self) -> None:
        self._write_control()
        conflicting = json.loads(self.control_path.read_text(encoding="utf-8"))
        conflicting["selected_tau1"] = 3e-3
        _write_json(self.control_path, conflicting)
        with self.assertRaises(FrozenTau1ControlError):
            self._write_control()

    def test_rejects_failed_or_unselected_or_tampered_tau_aggregation(self) -> None:
        cases = (
            ("status", "failed"),
            ("selection_rule_passed", False),
            ("selected_tau1", 4e-3),
            ("selected_tau1", 3e-3),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                aggregation = self._aggregation()
                aggregation[field] = value
                _write_json(self.aggregation_path, aggregation)
                with self.assertRaises(FrozenTau1ControlError):
                    self._write_control()
                self._write_samples_and_aggregation()

    def test_rejects_tampered_sample_hash_and_wrong_sample_count(self) -> None:
        aggregation = self._aggregation()
        first = aggregation["sample_output_files"][0]
        path = self.repository_root / first["path"]
        path.write_text("{\"tampered\":true}", encoding="utf-8")
        with self.assertRaises(FrozenTau1ControlError):
            self._write_control()
        self._write_samples_and_aggregation()
        aggregation = self._aggregation()
        aggregation["sample_output_files"] = aggregation["sample_output_files"][:-1]
        _write_json(self.aggregation_path, aggregation)
        with self.assertRaises(FrozenTau1ControlError):
            self._write_control()

    def test_rejects_preregistration_sample_list_and_bf16_amendment_mismatches(self) -> None:
        preregistration_path = self.project_root / "manifests" / "preregistration.json"
        preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
        preregistration["preregistration_sha256"] = "d" * 64
        _write_json(preregistration_path, preregistration)
        with self.assertRaises(FrozenTau1ControlError):
            self._write_control()
        self._write_active_protocol()

        preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
        preregistration["sample_lists"]["calibration"][0]["sentence"] = "tampered list"
        _write_json(preregistration_path, preregistration)
        with self.assertRaises(FrozenTau1ControlError):
            self._write_control()
        self._write_active_protocol()

        different_amendment = dict(self.amendment_document)
        different_amendment["amendment_identity_sha256"] = "e" * 64
        with mock.patch(
            "src.dager_qwen3.frozen_tau1_control.verify_bf16_gate_profile_amendment",
            return_value=different_amendment,
        ):
            with self.assertRaises(FrozenTau1ControlError):
                self._write_control()


@unittest.skipIf(RUNNER is None, f"run_none_attack import dependency unavailable: {RUNNER_IMPORT_ERROR}")
class FrozenTau1RunnerContractTest(unittest.TestCase):
    def test_cli_requires_control_and_exposes_no_tau_override(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            [
                "run_none_attack.py",
                "--stage",
                "smoke",
                "--sample-key",
                "0" * 64,
                "--head-seed",
                "22",
                "--output",
                "Projection_lrb_qwen3/outputs/smoke/test.jsonl",
            ],
        ), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                RUNNER.parse_args()
        self.assertEqual(error.exception.code, 2)
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertNotIn('add_argument("--tau1"', source)
        self.assertNotIn('add_argument("--l1-span-thresh"', source)

    def test_invalid_control_prevents_rouge_and_model_loading(self) -> None:
        args = type(
            "Args",
            (),
            {
                "defense": "none",
                "tau1_control": "Projection_lrb_qwen3/frozen_controls/missing.json",
                "config": "Projection_lrb_qwen3/configs/experiment.json",
                "stage": "smoke",
                "sample_key": "0" * 64,
                "head_seed": 22,
                "output": "Projection_lrb_qwen3/outputs/smoke/test.jsonl",
                "device": "cpu",
                "dtype": "float32",
            },
        )()
        with mock.patch.object(
            RUNNER,
            "verify_frozen_tau1_control",
            side_effect=FrozenTau1ControlError("tampered control"),
        ), mock.patch.object(RUNNER, "preflight_legacy_dager_rouge_backend") as rouge, mock.patch.object(
            RUNNER, "load_local_qwen3_sequence_classifier"
        ) as model_loader:
            with self.assertRaises(FrozenTau1ControlError):
                RUNNER.run_attack(args)
        rouge.assert_not_called()
        model_loader.assert_not_called()

    def test_valid_control_is_the_only_tau1_input_and_other_controls_stay_locked(self) -> None:
        configuration = ExperimentConfig(
            config_path=Path("config.json"),
            repository_root=Path("repo"),
            project_root=Path("repo/Projection_lrb_qwen3"),
            model_path=Path("repo/models/Qwen3"),
            dataset_path=Path("repo/data/glue_sst2"),
            output_root=Path("repo/Projection_lrb_qwen3/outputs"),
            max_length=32,
            min_effective_token_length=1,
            calibration_head_seed=11,
            smoke_head_seed=22,
            final_head_seeds=(101, 202, 303),
            defense_base_seed=700001,
            calibration_parameter_grid={
                "l1_span_thresh": [1e-5],
                "l2_span_thresh": [1e-3],
                "rank_tol": [1e-3],
                "rank_cutoff": [20],
            },
            attack_budget={"parallel": 1000, "max_ids": -1, "maxC": 10_000_000},
            raw={},
            config_sha256="c" * 64,
        )
        controls = load_none_attack_controls(configuration, frozen_tau1=EXPECTED_SELECTED_TAU1)
        self.assertEqual(controls.l1_span_threshold, EXPECTED_SELECTED_TAU1)
        self.assertEqual(controls.l2_span_threshold, 1e-3)
        self.assertEqual(controls.rank_tolerance, 1e-3)
        self.assertEqual(controls.rank_cutoff, 20)
        self.assertEqual(controls.max_search_candidates, 10_000_000)
        self.assertEqual(controls.max_candidate_ids, -1)
        self.assertEqual(controls.vocab_chunk_size, 1000)

    def test_runner_passes_the_verified_tau1_to_layer1_attack_controls(self) -> None:
        args = type(
            "Args",
            (),
            {
                "defense": "none",
                "tau1_control": "Projection_lrb_qwen3/frozen_controls/control.json",
                "config": "Projection_lrb_qwen3/configs/experiment.json",
                "stage": "smoke",
                "sample_key": "0" * 64,
                "head_seed": 22,
                "output": "Projection_lrb_qwen3/outputs/smoke/test.jsonl",
                "device": "cpu",
                "dtype": "float32",
            },
        )()
        observed: dict[str, float] = {}

        def stop_after_controls(_config: object, *, frozen_tau1: float | None = None) -> object:
            observed["tau1"] = float(frozen_tau1) if frozen_tau1 is not None else -1.0
            raise RuntimeError("stop after verifying Layer-1 controls")

        with mock.patch.object(
            RUNNER,
            "verify_frozen_tau1_control",
            return_value={"selected_tau1": EXPECTED_SELECTED_TAU1},
        ), mock.patch.object(RUNNER, "preflight_legacy_dager_rouge_backend", return_value=object()), mock.patch.object(
            RUNNER, "load_experiment_config", return_value=object()
        ), mock.patch.object(RUNNER, "registered_head_seed"), mock.patch.object(
            RUNNER, "load_registered_sample", return_value=object()
        ), mock.patch.object(
            RUNNER, "load_none_attack_controls", side_effect=stop_after_controls
        ), mock.patch.object(RUNNER, "load_local_qwen3_sequence_classifier") as model_loader:
            with self.assertRaisesRegex(RuntimeError, "stop after verifying"):
                RUNNER.run_attack(args)
        self.assertEqual(observed["tau1"], EXPECTED_SELECTED_TAU1)
        model_loader.assert_not_called()

    def test_frozen_control_and_runner_imports_have_no_lrb_dependency(self) -> None:
        for path in (
            PROJECT_ROOT / "src" / "dager_qwen3" / "frozen_tau1_control.py",
            PROJECT_ROOT / "scripts" / "run_none_attack.py",
        ):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module is not None:
                    imported.append(node.module)
            self.assertFalse(any("lrb" in name.lower() for name in imported), path)


if __name__ == "__main__":
    unittest.main()
