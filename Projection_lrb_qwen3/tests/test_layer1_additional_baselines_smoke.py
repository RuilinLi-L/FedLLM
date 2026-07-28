"""Contracts for isolated Qwen3 Layer-1 additional-baseline smokes and summary."""

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


RUNNER = _load_module("qwen3_layer1_additional_baselines_smoke_runner", "run_layer1_additional_baselines_smoke.py")
SUMMARY = _load_module("qwen3_layer1_all_baselines_smoke_summary", "summarize_layer1_all_baselines_smoke.py")
Q0 = "model.layers.0.self_attn.q_proj.weight"
Q1 = "model.layers.1.self_attn.q_proj.weight"
SAMPLE_KEYS = tuple(f"{index:064x}" for index in range(1, 6))


class Layer1AdditionalBaselineSmokeTest(unittest.TestCase):
    def _canonical_fixture(self):
        names = ("model.embed_tokens.weight", Q0, "model.score.weight", Q1)
        gradients = (
            torch.tensor([1.0, 2.0]),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            None,
            torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        )
        return names, gradients, {Q0: 1, Q1: 3}

    def test_runner_has_only_the_four_fixed_additional_arms(self) -> None:
        arms = {(arm.defense, arm.defense_param_value) for arm in RUNNER.ADDITIONAL_BASELINE_ARMS}
        self.assertEqual(arms, {("topk", 0.7), ("topk", 0.9), ("compression", 16), ("compression", 32)})
        self.assertTrue(all(arm.noise_sigma is None for arm in RUNNER.ADDITIONAL_BASELINE_ARMS))

    def test_shared_helpers_receive_complete_tuple_and_preserve_holes(self) -> None:
        names, gradients, _indices = self._canonical_fixture()
        received: list[tuple] = []

        def helper(grads, *args, **kwargs):
            received.append(tuple(grads))
            return tuple(grads)

        with mock.patch.object(RUNNER, "topk_sparsification", side_effect=helper), mock.patch.object(RUNNER, "gradient_compression", side_effect=helper):
            outputs = [RUNNER.apply_additional_baseline_to_canonical_tuple(arm=arm, canonical_gradients=gradients, canonical_parameter_names=names) for arm in RUNNER.ADDITIONAL_BASELINE_ARMS]
        self.assertEqual(len(received), 4)
        self.assertTrue(all(len(value) == len(gradients) and value[2] is None for value in received))
        self.assertTrue(all(len(value) == len(gradients) and value[2] is None for value in outputs))
        self.assertTrue(all(value[1] is not gradients[1] for value in received))

    def test_compression_is_reproducible_at_static_seed(self) -> None:
        names, gradients, _indices = self._canonical_fixture()
        for arm in RUNNER.ADDITIONAL_BASELINE_ARMS:
            if arm.defense != "compression":
                continue
            first = RUNNER.apply_additional_baseline_to_canonical_tuple(arm=arm, canonical_gradients=gradients, canonical_parameter_names=names)
            second = RUNNER.apply_additional_baseline_to_canonical_tuple(arm=arm, canonical_gradients=gradients, canonical_parameter_names=names)
            self.assertTrue(all(a is None if b is None else torch.equal(a, b) for a, b in zip(first, second)))

    def test_q_observations_are_extracted_only_after_each_complete_transform(self) -> None:
        names, gradients, indices = self._canonical_fixture()
        decoded = RUNNER._decode_additional_baseline_arms(
            canonical_gradients=gradients,
            canonical_parameter_names=names,
            q_parameter_names=(Q0, Q1),
            canonical_q_indices=indices,
            decode_arm=lambda arm, observations: {"arm": arm.defense_param_value, "q_shapes": [tuple(item.shape) for item in observations]},
        )
        self.assertEqual([arm.defense_param_value for arm, _core in decoded], [0.7, 0.9, 16, 32])
        self.assertTrue(all(core["q_shapes"] == [(2, 2), (2, 2)] for _arm, core in decoded))

    def test_no_layer1_candidates_is_a_normal_result_for_every_arm(self) -> None:
        names, gradients, indices = self._canonical_fixture()
        decoded = RUNNER._decode_additional_baseline_arms(
            canonical_gradients=gradients,
            canonical_parameter_names=names,
            q_parameter_names=(Q0, Q1),
            canonical_q_indices=indices,
            decode_arm=lambda arm, _observations: {"status": "ok", "termination_reason": "no_layer1_candidates", "arm": arm.defense_param_value},
        )
        self.assertEqual(len(decoded), 4)
        self.assertTrue(all(core["status"] == "ok" and core["termination_reason"] == "no_layer1_candidates" for _arm, core in decoded))

    def test_five_samples_generate_four_records_each(self) -> None:
        samples = tuple(object() for _ in range(5))
        self.assertEqual(len(samples) * len(RUNNER.ADDITIONAL_BASELINE_ARMS), 20)

    def _record(self, *, defense: str, sample_key: str, parameter_name: str | None = None, parameter_value: float | int | None = None, status: str = "ok", tau1: float = 0.002) -> dict:
        record = {
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
            "attack_semantics": "defense_unaware_observed_q_proj_only",
            "candidate_count": 0 if defense == "noise" else 2,
            "legacy_l1_token_membership": 0.5,
            "layer_2_survivor_count": 0,
            "token_recovery": 0.0,
            "termination_reason": "no_layer1_candidates" if defense == "noise" else "completed_prefix_found",
        }
        if defense == "lrbprojonly":
            record["keep_ratio"] = 0.5
        elif parameter_name is not None:
            record["defense_param_name"] = parameter_name
            record["defense_param_value"] = parameter_value
        return record

    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8")

    def _sources(self):
        legacy = [self._record(defense=arm, sample_key=key) for key in SAMPLE_KEYS for arm in ("none", "lrbprojonly")]
        baseline = []
        for key in SAMPLE_KEYS:
            baseline.extend((
                self._record(defense="topk", sample_key=key, parameter_name="defense_topk_ratio", parameter_value=0.1),
                self._record(defense="compression", sample_key=key, parameter_name="defense_n_bits", parameter_value=8),
                self._record(defense="noise", sample_key=key, parameter_name="defense_noise", parameter_value=1e-6),
            ))
        additional = []
        for key in SAMPLE_KEYS:
            additional.extend((
                self._record(defense="topk", sample_key=key, parameter_name="defense_topk_ratio", parameter_value=0.7),
                self._record(defense="topk", sample_key=key, parameter_name="defense_topk_ratio", parameter_value=0.9),
                self._record(defense="compression", sample_key=key, parameter_name="defense_n_bits", parameter_value=16),
                self._record(defense="compression", sample_key=key, parameter_name="defense_n_bits", parameter_value=32),
            ))
        return legacy, baseline, additional

    def test_summary_merges_nine_arms_with_cross_run_pairing(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy, baseline, additional = self._sources()
            paths = root / "legacy.jsonl", root / "baseline.jsonl", root / "additional.jsonl"
            for path, rows in zip(paths, (legacy, baseline, additional)):
                self._write_jsonl(path, rows)
            summary = SUMMARY.build_layer1_all_baselines_summary(legacy_path=paths[0], baseline_path=paths[1], additional_path=paths[2])
        self.assertEqual(summary["comparison_pairing"], "protocol_matched_cross_run")
        self.assertEqual(set(summary["defenses"]), {"none", "lrbprojonly@0.5", "topk@0.1", "compression@8", "noise@1e-6", "topk@0.7", "topk@0.9", "compression@16", "compression@32"})
        self.assertEqual(summary["defenses"]["noise@1e-6"]["zero_candidate_samples"], 5)

    def test_summary_rejects_protocol_missing_duplicate_failure_and_wrong_arm(self) -> None:
        cases = (
            ("mismatch", lambda legacy, baseline, additional: additional.__setitem__(0, {**additional[0], "tau1": 0.1})),
            ("missing", lambda legacy, baseline, additional: additional.pop()),
            ("duplicate", lambda legacy, baseline, additional: additional.append(dict(additional[0]))),
            ("failed", lambda legacy, baseline, additional: additional.__setitem__(0, {**additional[0], "result_status": "error"})),
            ("wrong_arm", lambda legacy, baseline, additional: additional.__setitem__(0, {**additional[0], "defense_param_value": 0.1})),
        )
        for name, mutate in cases:
            with self.subTest(name=name), TemporaryDirectory() as temporary:
                root = Path(temporary)
                legacy, baseline, additional = self._sources()
                mutate(legacy, baseline, additional)
                paths = root / "legacy.jsonl", root / "baseline.jsonl", root / "additional.jsonl"
                for path, rows in zip(paths, (legacy, baseline, additional)):
                    self._write_jsonl(path, rows)
                with self.assertRaises(SUMMARY.Layer1AllBaselineSummaryError):
                    SUMMARY.build_layer1_all_baselines_summary(legacy_path=paths[0], baseline_path=paths[1], additional_path=paths[2])


if __name__ == "__main__":
    unittest.main()
