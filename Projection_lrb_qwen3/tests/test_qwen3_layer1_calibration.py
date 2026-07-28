"""CPU regression tests for the isolated Qwen3 Layer-1 tau1 calibration path."""

from __future__ import annotations

import importlib.util
import inspect
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

from src.config import ExperimentConfig
from src.dager_qwen3.diagnostics import RegisteredAttackSample
from src.dager_qwen3.gradient_decomposition import GradientSpan, SharedDagerRank
from src.dager_qwen3.gradient_decomposition import (
    decompose_qwen3_qproj_gradient,
    shared_dager_rank_for_qwen3_qproj_gradients,
)
from src.dager_qwen3.layer1_calibration import (
    EXPECTED_CALIBRATION_SAMPLE_COUNT,
    TAU1_CALIBRATION_GRID,
    Layer1CalibrationError,
    aggregate_calibration_records,
    build_sample_record,
    evaluate_tau_grid,
)
from src.dager_qwen3.layer1_calibration_amendment import (
    AMENDMENT_RELATIVE_PATH,
    Layer1CalibrationAmendmentError,
    verify_amendment,
    write_or_verify_amendment,
)
from src.dager_qwen3.layer1_filter import (
    Layer1DistanceScanResult,
    VocabularyDistanceChunkDiagnostic,
    filter_qwen3_layer1_distance_scan,
    filter_qwen3_vocab_layer1,
    scan_qwen3_vocab_layer1_distances,
)


SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_layer1_calibration.py"
SPEC = importlib.util.spec_from_file_location("qwen3_layer1_calibration_test", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load isolated calibration script: {SCRIPT_PATH}")
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


class _Tokenizer:
    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [f"token-{token_id}" for token_id in token_ids]


class _Adapter:
    def __init__(self) -> None:
        self.metadata = SimpleNamespace(hidden_size=2, vocab_size=5)
        self.device = torch.device("cpu")
        self._embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.02],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ],
            dtype=torch.float32,
        )

    def layer0_qproj_inputs_for_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self._embeddings[token_ids]


def _span() -> GradientSpan:
    return GradientSpan(
        basis=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        effective_rank=1,
        absolute_tolerance=1e-3,
        requested_rank=1,
        applied_rank=1,
        rank_cap=2,
        rank_was_capped=False,
        cap_reason=None,
        feature_dim=2,
        gradient_shape=(2, 2),
        rank_cutoff=0,
        orientation="raw_qwen3_nn_linear_gradient_right_singular_vectors",
        decomposition_device="cpu",
    )


def _scan() -> Layer1DistanceScanResult:
    return Layer1DistanceScanResult(
        token_ids=torch.arange(5, dtype=torch.long),
        distances=torch.tensor([0.009, 0.00001, 0.0002, 0.002, 0.007], dtype=torch.float32),
        distance_norm="l2",
        chunk_size=3,
        chunk_diagnostics=(
            VocabularyDistanceChunkDiagnostic(0, 3, 0.01, 3),
            VocabularyDistanceChunkDiagnostic(3, 5, 0.01, 2),
        ),
    )


def _diagnostic() -> dict[str, object]:
    layer = {
        "passed": True,
        "checks": {"gradient_identity": True},
        "identity": {"gradient_relative_error": 0.0},
        "rank": {"relative_threshold_rank": 1, "relative_threshold": 0.1},
        "delta_activity": {"active_token_count": 2},
        "row_space_residual": {"active_tokens": {"max": 1e-4}},
    }
    return {
        "passed": True,
        "layers": {
            "q0": {
                **layer,
                "per_token": [
                    {"position": 0, "token_id": 1, "active_by_delta": True},
                    {"position": 1, "token_id": 2, "active_by_delta": True},
                ],
            },
            "q1": layer,
        },
    }


def _sample_record(*, key: str, per_tau: list[dict[str, object]]) -> dict[str, object]:
    return {
        "status": "ok",
        "layer2_invoked": False,
        "sample_key": key,
        "per_tau": per_tau,
    }


def _per_tau(*, position_recall: float, candidate_count: int) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for tau in TAU1_CALIBRATION_GRID:
        values.append(
            {
                "tau": tau,
                "candidate_count": candidate_count,
                "active_position_hits": int(round(position_recall * 20)),
                "active_position_total": 20,
                "active_unique_token_hits": int(round(position_recall * 10)),
                "active_unique_token_total": 10,
            }
        )
    return values


class Layer1CalibrationTest(unittest.TestCase):
    def test_shared_distance_scan_and_threshold_filter_choose_identical_candidates(self) -> None:
        scan = scan_qwen3_vocab_layer1_distances(
            adapter=_Adapter(),
            span=_span(),
            vocab_chunk_size=3,
            distance_norm="l2",
        )
        tau = 0.001
        direct_result = filter_qwen3_layer1_distance_scan(scan, threshold=tau)
        result = filter_qwen3_vocab_layer1(
            adapter=_Adapter(),
            span=_span(),
            threshold=tau,
            vocab_chunk_size=3,
            distance_norm="l2",
        )
        expected = [
            (int(token_id), float(distance))
            for token_id, distance in zip(scan.token_ids.tolist(), scan.distances.tolist())
            if distance < tau
        ]
        expected = sorted(expected, key=lambda item: (item[1], item[0]))
        self.assertEqual(list(result.token_ids), [token_id for token_id, _ in expected])
        self.assertEqual(list(result.distances), [distance for _, distance in expected])
        self.assertEqual(result.token_ids, direct_result.token_ids)
        self.assertEqual(result.distances, direct_result.distances)
        self.assertEqual(sum(item.passing_candidate_count for item in result.chunk_diagnostics), len(expected))

    def test_scan_helper_accepts_no_ground_truth_inputs(self) -> None:
        prohibited = {"ground_truth", "token_ids", "sample", "labels", "targets"}
        for helper in (
            shared_dager_rank_for_qwen3_qproj_gradients,
            decompose_qwen3_qproj_gradient,
            scan_qwen3_vocab_layer1_distances,
        ):
            parameter_names = set(inspect.signature(helper).parameters)
            self.assertFalse(
                prohibited & parameter_names,
                msg=f"{helper.__name__} must not accept ground-truth controls.",
            )

    def test_tau_candidate_count_is_monotone(self) -> None:
        active = [
            {"position": 0, "token_id": 1},
            {"position": 1, "token_id": 2},
        ]
        metrics = evaluate_tau_grid(scan=_scan(), active_positions=active)
        self.assertEqual([item.candidate_count for item in metrics], sorted(item.candidate_count for item in metrics))

    def test_distance_equal_to_tau_is_rejected_by_calibration_and_attack_filter(self) -> None:
        scan = Layer1DistanceScanResult(
            token_ids=torch.arange(3, dtype=torch.long),
            distances=torch.tensor([0.0009, 0.0010, 0.0011], dtype=torch.float32),
            distance_norm="l2",
            chunk_size=3,
            chunk_diagnostics=(VocabularyDistanceChunkDiagnostic(0, 3, 0.01, 3),),
        )
        filtered = filter_qwen3_layer1_distance_scan(scan, threshold=0.001)
        self.assertEqual(filtered.token_ids, (0,))
        metrics = evaluate_tau_grid(
            scan=scan,
            active_positions=[{"position": 0, "token_id": 0}, {"position": 1, "token_id": 1}],
        )
        metric = next(item for item in metrics if item.tau == 0.001)
        self.assertEqual(metric.candidate_count, 1)
        self.assertEqual(metric.active_position_hits, 1)

    def test_selection_uses_smallest_passing_tau(self) -> None:
        records: list[dict[str, object]] = []
        for index in range(EXPECTED_CALIBRATION_SAMPLE_COUNT):
            per_tau = _per_tau(position_recall=1.0, candidate_count=1)
            for item in per_tau[:4]:
                item["active_position_hits"] = 18
                item["active_unique_token_hits"] = 8
            records.append(_sample_record(key=f"{index:064x}", per_tau=per_tau))
        aggregate = aggregate_calibration_records(records, sample_output_files=[])
        self.assertTrue(aggregate["selection_rule_passed"])
        self.assertEqual(aggregate["selected_tau1"], TAU1_CALIBRATION_GRID[4])

    def test_selection_fails_closed_when_no_tau_passes(self) -> None:
        records = [
            _sample_record(key=f"{index:064x}", per_tau=_per_tau(position_recall=0.90, candidate_count=1))
            for index in range(EXPECTED_CALIBRATION_SAMPLE_COUNT)
        ]
        aggregate = aggregate_calibration_records(records, sample_output_files=[])
        self.assertFalse(aggregate["selection_rule_passed"])
        self.assertIsNone(aggregate["selected_tau1"])
        self.assertEqual(aggregate["status"], "failed")
        self.assertIn("no_fixed_tau1_grid_value", str(aggregate["failure_reason"]))

    def test_aggregation_requires_and_records_one_verified_bf16_gate_amendment(self) -> None:
        amendment = {
            "path": "Projection_lrb_qwen3/prereg_amendments/bf16_gradient_gate_pre_attack_075/amendment.json",
            "sha256": "a" * 64,
            "amendment_identity_sha256": "b" * 64,
            "selected_bfloat16_gate": 7.5e-3,
            "fixed_candidate_grid": [3e-3, 5e-3, 7.5e-3, 1e-2],
        }
        records = []
        for index in range(EXPECTED_CALIBRATION_SAMPLE_COUNT):
            record = _sample_record(key=f"{index:064x}", per_tau=_per_tau(position_recall=1.0, candidate_count=1))
            record["bf16_gate_profile_amendment"] = amendment
            records.append(record)
        aggregate = aggregate_calibration_records(
            records,
            sample_output_files=[],
            bf16_gate_profile_amendment=amendment,
        )
        self.assertEqual(aggregate["bf16_gate_profile_amendment"], amendment)

    def test_sample_record_explicitly_marks_layer2_not_invoked(self) -> None:
        sample = RegisteredAttackSample(
            stage="calibration",
            preregistration_sha256="p" * 64,
            sample_key="s" * 64,
            original_index=7,
            sentence="one sentence",
            label=1,
            input_ids=(1, 2),
            eos_token_id=2,
        )
        config = ExperimentConfig(
            config_path=Path("config.json"),
            repository_root=REPOSITORY_ROOT,
            project_root=PROJECT_ROOT,
            model_path=PROJECT_ROOT / "models" / "fake",
            dataset_path=PROJECT_ROOT / "data" / "fake",
            output_root=PROJECT_ROOT / "outputs",
            max_length=32,
            min_effective_token_length=1,
            calibration_head_seed=11,
            smoke_head_seed=22,
            final_head_seeds=(101, 202, 303),
            defense_base_seed=700001,
            calibration_parameter_grid={},
            attack_budget={},
            raw={},
            config_sha256="c" * 64,
        )
        shared = SharedDagerRank(
            rank_definition="absolute_matrix_rank_atol_rtol_zero",
            rank_atol=1e-3,
            q0_effective_rank=1,
            q1_effective_rank=1,
            requested_shared_rank=1,
            applied_shared_rank=1,
            rank_cap=2,
            rank_was_capped=False,
            cap_reason=None,
        )
        record = build_sample_record(
            config=config,
            preregistration={
                "model_key_file_sha256": {"model.safetensors": "a" * 64},
                "tokenizer_key_file_sha256": {"tokenizer.json": "b" * 64},
                "tokenizer_sha256": "d" * 64,
            },
            sample=sample,
            head_seed=11,
            dtype="float32",
            original_l1_span_threshold=1e-5,
            shared_rank=shared,
            q0_span=_span(),
            scan=_scan(),
            tokenizer=_Tokenizer(),
            gradient_diagnostic=_diagnostic(),
            gradient_capture_seconds=1.0,
            scan_seconds=2.0,
            loss=0.5,
            gpu_peak_memory_bytes=1024,
            distance_sidecar={"path": "sidecar.npy", "shape": [5], "dtype": "float32", "sha256": "e" * 64},
        )
        self.assertIs(record["layer2_invoked"], False)
        self.assertEqual(record["vocabulary_scan"]["scanned_token_count"], 5)
        self.assertEqual(len(record["top_100_lowest_distance_tokens"]), 5)

    def test_runner_never_calls_layer2_decoder(self) -> None:
        layer2_module = __import__("src.dager_qwen3.layer2_decoder", fromlist=["decode_qwen3_rope_prefixes"])
        sample = SimpleNamespace(sample_key="a" * 64)
        record = {"sample_key": sample.sample_key}
        outputs_root = PROJECT_ROOT / "outputs"
        outputs_root.mkdir(parents=True, exist_ok=True)
        with (
            TemporaryDirectory(dir=outputs_root) as temporary_directory,
            mock.patch.object(RUNNER, "load_experiment_config", return_value=SimpleNamespace()),
            mock.patch.object(RUNNER, "registered_head_seed"),
            mock.patch.object(RUNNER, "verify_amendment"),
            mock.patch.object(
                RUNNER,
                "verify_bf16_gate_profile_amendment",
                return_value={
                    "amendment_identity_sha256": "a" * 64,
                    "selected_bfloat16_gate": 7.5e-3,
                    "fixed_candidate_grid": [3e-3, 5e-3, 7.5e-3, 1e-2],
                },
            ) as verify_bf16_profile,
            mock.patch.object(RUNNER, "sha256_file", return_value="b" * 64),
            mock.patch.object(
                RUNNER, "_load_preregistration", return_value={"preregistration_sha256": "p" * 64}
            ),
            mock.patch.object(RUNNER, "_select_samples", return_value=(sample,)),
            mock.patch.object(
                RUNNER,
                "_run_one_sample",
                return_value=(record, Path(temporary_directory) / "sample.json"),
            ) as run_one_sample,
            mock.patch.object(layer2_module, "decode_qwen3_rope_prefixes") as layer2,
        ):
            args = SimpleNamespace(
                stage="calibration",
                config="Projection_lrb_qwen3/configs/experiment.json",
                head_seed=11,
                sample_key=sample.sample_key,
                dtype="bfloat16",
                output_root="Projection_lrb_qwen3/outputs",
            )
            result = RUNNER.run_calibration(args)
        self.assertEqual(result["sample_count"], 1)
        self.assertIs(result["layer2_invoked"], False)
        verify_bf16_profile.assert_called_once_with(project_root=PROJECT_ROOT)
        self.assertEqual(
            run_one_sample.call_args.kwargs["bf16_gate_profile_amendment"][
                "selected_bfloat16_gate"
            ],
            7.5e-3,
        )
        layer2.assert_not_called()

    def test_amendment_hashes_the_actual_run6_failure_log(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            run6 = project_root / "outputs" / "calibration" / "stage4_diagnostic" / "run6.log"
            run6.parent.mkdir(parents=True)
            run6.write_text(
                '{"error":"Layer-1 DAGER filtering produced no candidate tokens."}\n',
                encoding="utf-8",
            )
            written = write_or_verify_amendment(project_root=project_root, run6_log=run6)
            self.assertEqual(written, project_root / AMENDMENT_RELATIVE_PATH)
            document = verify_amendment(project_root=project_root)
            self.assertEqual(document["fixed_tau1_grid"], list(TAU1_CALIBRATION_GRID))
            self.assertFalse(document["attack_configuration_changed"])
            self.assertIs(document["layer2_invoked"], False)
            run6.write_text("different content\n", encoding="utf-8")
            with self.assertRaises(Layer1CalibrationAmendmentError):
                verify_amendment(project_root=project_root)


if __name__ == "__main__":
    unittest.main()
