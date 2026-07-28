"""Contracts for the isolated Qwen3 Layer-1 baseline smoke and its reader."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
for import_root in (REPOSITORY_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import torch


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {filename}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_module("qwen3_layer1_baselines_smoke_runner", "run_layer1_baselines_smoke.py")
SUMMARY = _load_module("qwen3_layer1_baselines_smoke_summary", "summarize_layer1_baselines_smoke.py")
Q0 = "model.layers.0.self_attn.q_proj.weight"
Q1 = "model.layers.1.self_attn.q_proj.weight"
SAMPLE_KEYS = tuple(f"{index:064x}" for index in range(1, 6))


class Layer1BaselineSmokeTest(unittest.TestCase):
    def _canonical_fixture(self):
        names = ("model.embed_tokens.weight", Q0, "model.score.weight", Q1)
        gradients = (
            torch.tensor([1.0, 2.0]),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            None,
            torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        )
        return names, gradients, {Q0: 1, Q1: 3}

    def test_fixed_arms_exclude_none_and_projection_only(self) -> None:
        self.assertEqual({arm.defense for arm in RUNNER.BASELINE_ARMS}, {"topk", "compression", "noise"})
        source = (PROJECT_ROOT / "scripts" / "run_layer1_baselines_smoke.py").read_text(encoding="utf-8")
        self.assertNotIn('defense="none"', source)
        self.assertNotIn("lrbprojonly", source)
        self.assertNotIn("minimal_projonly_pair_l1metrics_v1", source)

    def test_all_shared_helpers_receive_full_tuple_and_preserve_holes(self) -> None:
        names, gradients, _indices = self._canonical_fixture()
        received: list[tuple] = []

        def helper(grads, *args, **kwargs):
            received.append(tuple(grads))
            return tuple(grads)

        with mock.patch.object(RUNNER, "topk_sparsification", side_effect=helper), mock.patch.object(RUNNER, "gradient_compression", side_effect=helper), mock.patch.object(RUNNER, "noise_injection", side_effect=helper):
            outputs = [RUNNER.apply_baseline_to_canonical_tuple(arm=arm, canonical_gradients=gradients, canonical_parameter_names=names) for arm in RUNNER.BASELINE_ARMS]
        self.assertEqual(len(received), 3)
        self.assertTrue(all(len(value) == len(gradients) and value[2] is None for value in received))
        self.assertTrue(all(len(value) == len(gradients) and value[2] is None for value in outputs))
        self.assertTrue(all(value[1] is not gradients[1] for value in received))

    def test_compression_and_noise_are_reproducible_at_fixed_seed(self) -> None:
        names, gradients, _indices = self._canonical_fixture()
        for defense in ("compression", "noise"):
            arm = next(arm for arm in RUNNER.BASELINE_ARMS if arm.defense == defense)
            first = RUNNER.apply_baseline_to_canonical_tuple(arm=arm, canonical_gradients=gradients, canonical_parameter_names=names)
            second = RUNNER.apply_baseline_to_canonical_tuple(arm=arm, canonical_gradients=gradients, canonical_parameter_names=names)
            self.assertTrue(all(a is None if b is None else torch.equal(a, b) for a, b in zip(first, second)))

    def test_each_arm_extracts_q_observations_only_after_full_tuple_transform(self) -> None:
        names, gradients, indices = self._canonical_fixture()
        decoded = RUNNER._decode_baseline_arms(
            canonical_gradients=gradients,
            canonical_parameter_names=names,
            q_parameter_names=(Q0, Q1),
            canonical_q_indices=indices,
            decode_arm=lambda arm, observations: {"arm": arm.defense, "q_shapes": [tuple(item.shape) for item in observations]},
        )
        self.assertEqual([arm.defense for arm, _core in decoded], ["topk", "compression", "noise"])
        self.assertTrue(all(core["q_shapes"] == [(2, 2), (2, 2)] for _arm, core in decoded))

    def test_no_layer1_candidates_is_a_normal_per_arm_result(self) -> None:
        names, gradients, indices = self._canonical_fixture()
        decoded = RUNNER._decode_baseline_arms(
            canonical_gradients=gradients,
            canonical_parameter_names=names,
            q_parameter_names=(Q0, Q1),
            canonical_q_indices=indices,
            decode_arm=lambda arm, _observations: {"status": "ok", "termination_reason": "no_layer1_candidates", "arm": arm.defense},
        )
        self.assertEqual(len(decoded), 3)
        self.assertTrue(all(core["status"] == "ok" and core["termination_reason"] == "no_layer1_candidates" for _arm, core in decoded))

    def test_fixed_protocol_and_output_root_are_isolated(self) -> None:
        source = (PROJECT_ROOT / "scripts" / "run_layer1_baselines_smoke.py").read_text(encoding="utf-8")
        self.assertIn('SMOKE_HEAD_SEED = 22', source)
        self.assertIn('SMOKE_DTYPE = "bfloat16"', source)
        self.assertIn('OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "smoke" / "layer1_baselines_v1"', source)
        self.assertIn("verify_frozen_tau1_control", source)
        self.assertIn("load_none_attack_controls", source)
        self.assertEqual(source.count("controls = _standard_controls"), 1)
        self.assertIn("controls=controls", source)
        self.assertIn("--all-smoke", source)

    def test_all_smoke_contract_covers_five_manifest_samples_times_three_arms(self) -> None:
        config = type("Config", (), {"project_root": PROJECT_ROOT})()
        loaded_keys: list[str] = []
        with mock.patch.object(
            RUNNER,
            "load_registered_sample",
            side_effect=lambda *, config, stage, sample_key: loaded_keys.append(sample_key) or type("Sample", (), {"sample_key": sample_key})(),
        ):
            samples = RUNNER._all_registered_smoke_samples(config)
        self.assertEqual(len(samples), 5)
        self.assertEqual(len(set(loaded_keys)), 5)
        self.assertEqual(len(samples) * len(RUNNER.BASELINE_ARMS), 15)

    def _record(self, *, defense: str, sample_key: str, status: str = "ok", tau1: float = 0.002) -> dict:
        return {
            "record_type": "qwen3_dager_layer1_baseline_smoke",
            "result_status": status,
            "sample_key": sample_key,
            "defense": defense,
            "task": "sst2",
            "batch_size": 1,
            "gradient_steps": 1,
            "dtype": "bfloat16",
            "head_seed": 22,
            "tau1": tau1,
            "tau2": 0.001,
            "frozen_tau1_control_identity_sha256": "a" * 64,
            "canonical_q_proj_indices": {Q0: 1, Q1: 3},
            "defense_awareness": "defense_unaware_observed_q_proj_only",
            "candidate_count": 0 if defense == "noise" else 2,
            "legacy_l1_token_membership": 0.5,
            "layer_2_survivor_count": 0,
            "token_recovery": 0.0,
            "termination_reason": "no_layer1_candidates" if defense == "noise" else "completed_prefix_found",
        }

    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8")

    def test_summary_merges_five_arms_with_cross_run_pairing(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy_path, baseline_path = root / "legacy.jsonl", root / "baselines.jsonl"
            self._write_jsonl(legacy_path, [self._record(defense=arm, sample_key=key) for key in SAMPLE_KEYS for arm in ("none", "lrbprojonly")])
            self._write_jsonl(baseline_path, [self._record(defense=arm, sample_key=key) for key in SAMPLE_KEYS for arm in ("topk", "compression", "noise")])
            summary = SUMMARY.build_layer1_summary(legacy_path=legacy_path, baseline_path=baseline_path)
        self.assertEqual(summary["comparison_pairing"], "protocol_matched_cross_run")
        self.assertEqual(set(summary["defenses"]), {"none", "lrbprojonly", "topk", "compression", "noise"})
        self.assertEqual(summary["defenses"]["noise"]["zero_candidate_samples"], 5)

    def test_summary_rejects_mismatch_missing_duplicate_failure_and_wrong_arm(self) -> None:
        cases = (
            ("mismatch", lambda legacy, baseline: baseline.__setitem__(0, {**baseline[0], "tau1": 0.1})),
            ("missing", lambda legacy, baseline: baseline.pop()),
            ("duplicate", lambda legacy, baseline: baseline.append(dict(baseline[0]))),
            ("failed", lambda legacy, baseline: baseline.__setitem__(0, {**baseline[0], "result_status": "error"})),
            ("wrong_arm", lambda legacy, baseline: baseline.__setitem__(0, {**baseline[0], "defense": "none"})),
        )
        for name, mutate in cases:
            with self.subTest(name=name), TemporaryDirectory() as temporary:
                root = Path(temporary)
                legacy = [self._record(defense=arm, sample_key=key) for key in SAMPLE_KEYS for arm in ("none", "lrbprojonly")]
                baseline = [self._record(defense=arm, sample_key=key) for key in SAMPLE_KEYS for arm in ("topk", "compression", "noise")]
                mutate(legacy, baseline)
                legacy_path, baseline_path = root / "legacy.jsonl", root / "baselines.jsonl"
                self._write_jsonl(legacy_path, legacy)
                self._write_jsonl(baseline_path, baseline)
                with self.assertRaises(SUMMARY.Layer1SummaryError):
                    SUMMARY.build_layer1_summary(legacy_path=legacy_path, baseline_path=baseline_path)


if __name__ == "__main__":
    unittest.main()
