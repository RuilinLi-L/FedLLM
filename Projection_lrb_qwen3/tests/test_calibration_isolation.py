"""Isolation and recovery tests for the none-only Qwen3 calibration runner."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import CalibrationError, candidate_grid_from_experiment, load_calibration_manifest, run_calibration
from src.config import ExperimentConfig, PreregistrationConfigError, load_experiment_config
from src.hashing import hash_sample_list


def _sample(index: int) -> dict[str, object]:
    token_ids = [index + 10, 151643]
    return {
        "sample_key": f"{index + 1:064x}",
        "original_index": index,
        "sentence": f"calibration only sentence {index}",
        "label": index % 2,
        "tokenization": {"input_ids": token_ids, "eos_token_id": 151643},
    }


class CalibrationIsolationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.repository_root = Path(self.temporary_directory.name) / "repo"
        self.project_root = self.repository_root / "Projection_lrb_qwen3"
        self.manifests = self.project_root / "manifests"
        self.manifests.mkdir(parents=True)
        self.model_path = self.repository_root / "models" / "Qwen3-1.7B-Base"
        self.model_path.mkdir(parents=True)
        (self.model_path / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
        self.dataset_path = self.repository_root / "data" / "glue_sst2"
        self.dataset_path.mkdir(parents=True)
        self.samples = [_sample(0), _sample(1)]
        self.stage_hash = hash_sample_list(self.samples)
        self.manifest_path = self.manifests / "calibration.jsonl"
        self.manifest_path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "record_type": "preregistered_sst2_validation_sample",
                        "stage": "calibration",
                        "preregistration_sha256": "a" * 64,
                        "stage_sample_list_sha256": self.stage_hash,
                        "sample": sample,
                    },
                    sort_keys=True,
                )
                for sample in self.samples
            )
            + "\n",
            encoding="utf-8",
        )
        # These are deliberately malformed decoys.  A calibration-only runner
        # must never attempt to inspect either one.
        (self.manifests / "smoke.jsonl").write_text("NOT JSON", encoding="utf-8")
        (self.manifests / "final.jsonl").write_text("NOT JSON", encoding="utf-8")
        self.config = ExperimentConfig(
            config_path=self.project_root / "configs" / "experiment.json",
            repository_root=self.repository_root,
            project_root=self.project_root,
            model_path=self.model_path,
            dataset_path=self.dataset_path,
            output_root=self.project_root / "outputs",
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
            attack_budget={"parallel": 1000, "max_ids": -1, "maxC": 10_000},
            raw={},
            config_sha256="b" * 64,
        )

    def test_runner_uses_only_calibration_manifest_and_retains_failures(self) -> None:
        observed_keys: list[str] = []

        def executor(context: object) -> dict[str, object]:
            sample_key = getattr(getattr(context, "sample"), "sample_key")
            observed_keys.append(sample_key)
            if sample_key == self.samples[1]["sample_key"]:
                raise RuntimeError("deliberate per-sample failure")
            return {
                "result_status": "ok",
                "token_recovery": 0.5,
                "exact_recovery": False,
                "empty_reconstruction": False,
                "attack_time_seconds": 1.25,
            }

        with mock.patch("src.calibration._git_commit", return_value="c" * 40):
            result = run_calibration(
                config=self.config,
                manifest_path=self.manifest_path,
                output_root=self.project_root / "outputs" / "calibration",
                device="cpu",
                dtype="float32",
                executor=executor,
            )
        self.assertEqual(observed_keys, [sample["sample_key"] for sample in self.samples])
        all_rows = [
            json.loads(line)
            for line in Path(result["all_results_path"]).read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(all_rows), 2)
        self.assertEqual([row["sample_key"] for row in all_rows], observed_keys)
        self.assertEqual(all_rows[1]["result_status"], "error")
        self.assertIn("deliberate per-sample failure", all_rows[1]["error"])
        summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
        self.assertEqual(summary["candidate_summaries"][0]["failed_row_count"], 1)
        self.assertEqual(summary["candidate_summaries"][0]["sample_count"], 2)
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["frozen_attack_config_path"])
        self.assertFalse((self.manifests / "frozen_attack_config.json").exists())
        self.assertFalse(summary["freeze_eligibility"]["eligible"])

    def test_non_calibration_manifest_is_rejected_before_reading_decoys(self) -> None:
        with self.assertRaises(CalibrationError):
            load_calibration_manifest(self.manifests / "smoke.jsonl", expected_path=self.manifest_path)
        with self.assertRaises(CalibrationError):
            load_calibration_manifest(self.manifests / "final.jsonl", expected_path=self.manifest_path)

    def test_complete_calibration_freezes_all_required_provenance(self) -> None:
        def executor(_context: object) -> dict[str, object]:
            return {
                "result_status": "ok",
                "token_recovery": 0.5,
                "exact_recovery": False,
                "empty_reconstruction": False,
                "attack_time_seconds": 1.25,
            }

        with mock.patch("src.calibration._git_commit", return_value="c" * 40):
            result = run_calibration(
                config=self.config,
                manifest_path=self.manifest_path,
                output_root=self.project_root / "outputs" / "calibration",
                device="cpu",
                dtype="float32",
                executor=executor,
            )
        self.assertEqual(result["status"], "ok")
        frozen = json.loads(Path(result["frozen_attack_config_path"]).read_text(encoding="utf-8"))
        self.assertEqual(frozen["selected_parameters"], result["selected_parameters"])
        self.assertEqual(frozen["head_seed"], 11)
        self.assertEqual(frozen["code_commit"], "c" * 40)
        for key in (
            "selection_rule",
            "calibration_manifest_sha256",
            "model_sha256",
            "candidate_grid_sha256",
            "all_results_sha256",
        ):
            self.assertIn(key, frozen)

    def test_every_candidate_runs_the_identical_calibration_sample_set(self) -> None:
        multi_candidate_config = replace(
            self.config,
            calibration_parameter_grid={
                "l1_span_thresh": [1e-5, 2e-5],
                "l2_span_thresh": [1e-3],
                "rank_tol": [1e-3],
                "rank_cutoff": [20],
            },
        )
        observed: dict[str, list[str]] = {}

        def executor(context: object) -> dict[str, object]:
            parameters = getattr(getattr(context, "parameters"), "candidate_id")
            sample_key = getattr(getattr(context, "sample"), "sample_key")
            observed.setdefault(parameters, []).append(sample_key)
            return {
                "result_status": "ok",
                "token_recovery": 0.0,
                "exact_recovery": False,
                "empty_reconstruction": True,
                "attack_time_seconds": 1.0,
            }

        with mock.patch("src.calibration._git_commit", return_value="c" * 40):
            run_calibration(
                config=multi_candidate_config,
                manifest_path=self.manifest_path,
                output_root=self.project_root / "outputs" / "calibration",
                device="cpu",
                dtype="float32",
                executor=executor,
            )
        self.assertEqual(len(observed), 2)
        expected = [sample["sample_key"] for sample in self.samples]
        self.assertTrue(all(keys == expected for keys in observed.values()))

    def test_grid_keeps_configured_budget_values_without_code_side_expansion(self) -> None:
        candidates = candidate_grid_from_experiment(self.config)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].as_json()["candidate_budget"], {"max_ids": -1})
        self.assertEqual(candidates[0].as_json()["search_budget"], {"maxC": 10_000, "parallel": 1000})

    def test_calibration_config_loading_can_skip_the_unread_dataset_directory(self) -> None:
        config_path = self.project_root / "configs" / "experiment.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(
            json.dumps(
                {
                    "model_path": "models/Qwen3-1.7B-Base",
                    "dataset_path": "data/not_read_by_calibration",
                    "max_length": 32,
                    "min_effective_token_length": 1,
                    "calibration_head_seed": 11,
                    "smoke_head_seed": 22,
                    "final_head_seeds": [101, 202, 303],
                    "defense_base_seed": 700001,
                    "calibration_parameter_grid": self.config.calibration_parameter_grid,
                    "attack_budget": self.config.attack_budget,
                    "output_root": "Projection_lrb_qwen3/outputs",
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaises(PreregistrationConfigError):
            load_experiment_config(config_path)
        loaded = load_experiment_config(config_path, require_dataset_path=False)
        self.assertEqual(loaded.dataset_path, self.repository_root / "data" / "not_read_by_calibration")

    def test_cli_help_is_available_without_threshold_or_budget_overrides(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "run_calibration.py"), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("--config", completed.stdout)
        self.assertNotIn("--tau1", completed.stdout)
        self.assertNotIn("--tau2", completed.stdout)
        self.assertNotIn("--maxC", completed.stdout)


if __name__ == "__main__":
    unittest.main()
