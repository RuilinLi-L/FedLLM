"""Protocol-boundary tests for Stage-5 none-only calibration."""

from __future__ import annotations

from dataclasses import replace
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

from src.calibration import CalibrationError, CalibrationManifest, CalibrationRunContext, CalibrationSample, run_calibration
from src.config import ExperimentConfig, load_experiment_config
from src.dager_qwen3.calibration_grid_control import candidate_parameters_from_grid, verify_calibration_grid_control
from src.hashing import hash_sample_list, sha256_file, sha256_lf_normalized_text_file


TAU = PROJECT_ROOT / "frozen_controls" / "qwen3_none_tau1_calibration.json"
GRID = PROJECT_ROOT / "frozen_controls" / "qwen3_none_attack_calibration_grid.json"
MANIFEST = PROJECT_ROOT / "manifests" / "calibration.jsonl"


class Stage5ControlTest(unittest.TestCase):
    def test_plan_only_does_not_preflight_rouge_or_load_model(self) -> None:
        config = load_experiment_config(PROJECT_ROOT / "configs" / "experiment.json", require_dataset_path=False)
        with mock.patch("src.dager_qwen3.metrics.preflight_legacy_dager_rouge_backend") as rouge, mock.patch(
            "src.calibration._default_executor"
        ) as executor:
            result = run_calibration(
                config=config, manifest_path=MANIFEST, tau1_control_path=TAU,
                calibration_grid_control_path=GRID, output_root=PROJECT_ROOT / "outputs" / "calibration",
                device="cuda", dtype="bfloat16", plan_only=True,
            )
        self.assertEqual(result["status"], "planned")
        self.assertEqual(result["candidate_count"], 3)
        self.assertEqual(result["sample_count"], 20)
        self.assertEqual(result["total_runs"], 60)
        rouge.assert_not_called()
        executor.assert_not_called()

    def test_tau1_and_all_non_tau2_axes_come_from_controls_not_experiment_grid(self) -> None:
        config = load_experiment_config(PROJECT_ROOT / "configs" / "experiment.json", require_dataset_path=False)
        grid = verify_calibration_grid_control(project_root=PROJECT_ROOT, control_path=GRID)
        candidates = candidate_parameters_from_grid(grid)
        self.assertEqual({candidate["tau1"] for candidate in candidates}, {0.002})
        self.assertNotIn(1e-5, {candidate["tau1"] for candidate in candidates})
        self.assertEqual({candidate["numerical_rank_threshold"] for candidate in candidates}, {0.001})
        self.assertEqual({candidate["rank_cutoff"] for candidate in candidates}, {20})
        self.assertEqual({candidate["candidate_budget"]["max_ids"] for candidate in candidates}, {-1})
        self.assertEqual({candidate["search_budget"]["maxC"] for candidate in candidates}, {10_000_000})
        self.assertEqual({candidate["search_budget"]["parallel"] for candidate in candidates}, {1000})
        self.assertEqual(len({candidate["tau2"] for candidate in candidates}), grid["candidate_count"])
        self.assertEqual(config.calibration_parameter_grid["l1_span_thresh"], [1e-5])

    def test_candidate_count_matches_the_declared_grid_and_only_tau2_varies(self) -> None:
        grid = verify_calibration_grid_control(project_root=PROJECT_ROOT, control_path=GRID)
        candidates = candidate_parameters_from_grid(grid)
        self.assertEqual(len(candidates), grid["candidate_count"])
        varying_axes = {
            key
            for key in candidates[0]
            if len({json.dumps(candidate[key], sort_keys=True) for candidate in candidates}) > 1
        }
        self.assertEqual(varying_axes, {"tau2"})

    def test_tau1_control_file_hash_is_invariant_to_checkout_line_endings(self) -> None:
        grid = json.loads(GRID.read_text(encoding="utf-8"))
        lf = TAU.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        crlf = lf.replace(b"\n", b"\r\n")
        with TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            lf_path = Path(temporary) / "tau_lf.json"
            crlf_path = Path(temporary) / "tau_crlf.json"
            lf_path.write_bytes(lf)
            crlf_path.write_bytes(crlf)
            self.assertEqual(sha256_lf_normalized_text_file(lf_path), sha256_lf_normalized_text_file(crlf_path))
            self.assertEqual(grid["frozen_tau1_control_file_sha256"], sha256_lf_normalized_text_file(lf_path))

    def test_tampered_grid_is_rejected_and_manifest_inputs_remain_unchanged(self) -> None:
        before = {path: sha256_file(path) for path in (PROJECT_ROOT / "configs" / "experiment.json", MANIFEST, PROJECT_ROOT / "manifests" / "smoke.jsonl", PROJECT_ROOT / "manifests" / "final.jsonl")}
        with TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            copied = Path(temporary) / "grid.json"
            document = json.loads(GRID.read_text(encoding="utf-8"))
            document["tau2_candidate_grid"] = [1e-3]
            copied.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(Exception):
                verify_calibration_grid_control(project_root=PROJECT_ROOT, control_path=copied)
        self.assertEqual(before, {path: sha256_file(path) for path in before})

    def test_calibration_source_does_not_reference_other_stages_or_lrb(self) -> None:
        source = (PROJECT_ROOT / "src" / "calibration.py").read_text(encoding="utf-8").lower()
        self.assertNotIn("smoke.jsonl", source)
        self.assertNotIn("final.jsonl", source)
        self.assertNotIn("lrb", source)


class Stage5RunnerUnitTest(unittest.TestCase):
    def _fixture(self) -> tuple[TemporaryDirectory[str], ExperimentConfig, Path, dict[str, object], dict[str, object]]:
        temporary = TemporaryDirectory()
        root = Path(temporary.name) / "repo"
        project = root / "Projection_lrb_qwen3"
        (project / "manifests").mkdir(parents=True)
        model = root / "models" / "qwen3"; model.mkdir(parents=True); (model / "config.json").write_text("{}", encoding="utf-8")
        samples = [{"sample_key": f"{index:064x}", "original_index": index, "sentence": f"s{index}", "label": index % 2, "tokenization": {"input_ids": [index + 1, 7], "eos_token_id": 7}} for index in range(20)]
        sample_hash = hash_sample_list(samples)
        rows = [{"record_type": "preregistered_sst2_validation_sample", "stage": "calibration", "preregistration_sha256": "a" * 64, "stage_sample_list_sha256": sample_hash, "sample": sample} for sample in samples]
        manifest = project / "manifests" / "calibration.jsonl"; manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        config = ExperimentConfig(project / "configs" / "experiment.json", root, project, model, root / "data", project / "outputs", 32, 1, 11, 22, (101, 202, 303), 700001, {}, {}, {}, "b" * 64)
        tau = {"frozen_control_identity_sha256": "c" * 64, "preregistration_sha256": "a" * 64, "calibration_sample_list_sha256": sample_hash}
        grid = {"identity_sha256": "d" * 64, "frozen_tau1_control_identity_sha256": "c" * 64, "frozen_tau1_control_file_sha256": "e" * 64, "preregistration_sha256": "a" * 64, "calibration_sample_list_sha256": sample_hash, "candidate_count": 2}
        return temporary, config, manifest, tau, grid

    def test_single_candidate_fails_fast_before_executor(self) -> None:
        temporary, config, manifest, tau, grid = self._fixture(); self.addCleanup(temporary.cleanup)
        with mock.patch("src.calibration.verify_tau1_reference", return_value=tau), mock.patch("src.calibration.verify_calibration_grid_control", return_value=grid), mock.patch("src.calibration.sha256_lf_normalized_text_file", return_value="e" * 64), mock.patch("src.calibration.candidate_parameters_from_grid", return_value=({"tau1": .002},)), mock.patch("src.calibration._default_executor") as executor:
            with self.assertRaises(CalibrationError):
                run_calibration(config=config, manifest_path=manifest, tau1_control_path=config.project_root / "tau.json", calibration_grid_control_path=config.project_root / "grid.json", output_root=config.project_root / "outputs" / "calibration", device="cpu", dtype="float32", plan_only=True)
        executor.assert_not_called()

    def test_all_failed_candidates_are_ineligible_and_do_not_freeze(self) -> None:
        temporary, config, manifest, tau, grid = self._fixture(); self.addCleanup(temporary.cleanup)
        candidates = (
            {"tau1": .002, "tau2": .001, "numerical_rank_threshold": .001, "rank_cutoff": 20, "candidate_budget": {"max_ids": -1}, "search_budget": {"maxC": 10, "parallel": 1}},
            {"tau1": .002, "tau2": .002, "numerical_rank_threshold": .001, "rank_cutoff": 20, "candidate_budget": {"max_ids": -1}, "search_budget": {"maxC": 10, "parallel": 1}},
        )
        with mock.patch("src.calibration.verify_tau1_reference", return_value=tau), mock.patch("src.calibration.verify_calibration_grid_control", return_value=grid), mock.patch("src.calibration.sha256_lf_normalized_text_file", return_value="e" * 64), mock.patch("src.calibration.candidate_parameters_from_grid", return_value=candidates), mock.patch("src.calibration._git_commit", return_value="f" * 40):
            result = run_calibration(config=config, manifest_path=manifest, tau1_control_path=config.project_root / "tau.json", calibration_grid_control_path=config.project_root / "grid.json", output_root=config.project_root / "outputs" / "calibration", device="cpu", dtype="float32", executor=lambda _context, _rouge: (_ for _ in ()).throw(RuntimeError("failed")))
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["frozen_attack_config_path"])
        self.assertFalse((config.project_root / "manifests" / "frozen_attack_config.json").exists())

    def test_repeated_run_verifies_and_skips_existing_candidate_sample_records(self) -> None:
        temporary, config, manifest, tau, grid = self._fixture(); self.addCleanup(temporary.cleanup)
        candidates = (
            {"tau1": .002, "tau2": .001, "numerical_rank_threshold": .001, "rank_cutoff": 20, "candidate_budget": {"max_ids": -1}, "search_budget": {"maxC": 10, "parallel": 1}},
            {"tau1": .002, "tau2": .002, "numerical_rank_threshold": .001, "rank_cutoff": 20, "candidate_budget": {"max_ids": -1}, "search_budget": {"maxC": 10, "parallel": 1}},
        )
        calls: list[tuple[str, str]] = []

        def successful(context: CalibrationRunContext, _rouge: object) -> dict[str, object]:
            calls.append((context.candidate_id, context.sample.sample_key))
            return {
                "status": "ok", "tau1": .002, "tau2": context.parameters["tau2"],
                "token_recovery": .5, "exact_recovery": False, "rouge_1": .1, "rouge_2": .2,
                "attack_time_seconds": .01, "search_budget": {"evaluated_prefix_count": 3},
            }

        patches = (
            mock.patch("src.calibration.verify_tau1_reference", return_value=tau),
            mock.patch("src.calibration.verify_calibration_grid_control", return_value=grid),
            mock.patch("src.calibration.sha256_lf_normalized_text_file", return_value="e" * 64),
            mock.patch("src.calibration.candidate_parameters_from_grid", return_value=candidates),
            mock.patch("src.calibration._git_commit", return_value="f" * 40),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            kwargs = dict(config=config, manifest_path=manifest, tau1_control_path=config.project_root / "tau.json", calibration_grid_control_path=config.project_root / "grid.json", output_root=config.project_root / "outputs" / "calibration", device="cpu", dtype="float32", executor=successful)
            first = run_calibration(**kwargs)
            second = run_calibration(**kwargs)
        self.assertEqual(first["status"], "ok")
        self.assertEqual(second["status"], "ok")
        self.assertEqual(len(calls), 40)
        self.assertTrue((config.project_root / "manifests" / "frozen_attack_config.json").is_file())


if __name__ == "__main__":
    unittest.main()
