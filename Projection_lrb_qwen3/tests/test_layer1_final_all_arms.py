"""Contracts for the isolated Qwen3 Layer-1 final all-arm mechanism path."""

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


RUNNER = _load_module("qwen3_layer1_final_all_arms_runner", "run_layer1_final_all_arms.py")
SUMMARY = _load_module("qwen3_layer1_final_all_arms_summary", "summarize_layer1_final_all_arms.py")
Q0 = "model.layers.0.self_attn.q_proj.weight"
Q1 = "model.layers.1.self_attn.q_proj.weight"
SAMPLE_KEYS = tuple(f"{index:064x}" for index in range(1, 21))


class Layer1FinalAllArmsTest(unittest.TestCase):
    def _canonical_fixture(self):
        names = ("model.embed_tokens.weight", Q0, "model.score.weight", Q1)
        gradients = (
            torch.tensor([1.0, 2.0]),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            None,
            torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        )
        return names, gradients, {Q0: 1, Q1: 3}

    def test_final_manifest_seeds_and_fixed_arms(self) -> None:
        manifest_rows = [json.loads(line) for line in (PROJECT_ROOT / "manifests" / "final.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(manifest_rows), 20)
        self.assertEqual(len({row["sample"]["sample_key"] for row in manifest_rows}), 20)
        self.assertEqual(RUNNER.FINAL_HEAD_SEEDS, (101, 202, 303))
        self.assertEqual(
            [arm.label for arm in RUNNER.FINAL_ARMS],
            ["none", "lrbprojonly@0.2", "lrbprojonly@0.5", "lrbprojonly@0.65", "topk@0.1", "topk@0.7", "topk@0.9", "compression@8", "compression@16", "compression@32", "noise@1e-6"],
        )

    def test_each_arm_uses_an_independent_complete_tuple_and_preserves_holes(self) -> None:
        names, gradients, _indices = self._canonical_fixture()
        received: list[tuple] = []

        def helper(grads, *args, **kwargs):
            received.append(tuple(grads))
            return tuple(grads)

        with mock.patch.object(RUNNER, "topk_sparsification", side_effect=helper), mock.patch.object(RUNNER, "gradient_compression", side_effect=helper), mock.patch.object(RUNNER, "noise_injection", side_effect=helper), mock.patch.object(RUNNER, "apply_lrb_defense", side_effect=helper):
            outputs = [RUNNER.apply_final_arm_to_canonical_tuple(arm=arm, canonical_gradients=gradients, canonical_parameter_names=names) for arm in RUNNER.FINAL_ARMS]
        self.assertEqual(len(received), 10)
        self.assertTrue(all(len(value) == len(gradients) and value[2] is None for value in received))
        self.assertTrue(all(len(value) == len(gradients) and value[2] is None for value in outputs))
        self.assertTrue(all(value[1] is not gradients[1] for value in received))
        self.assertIsNot(outputs[0][1], gradients[1])

    def test_compression_noise_and_lrb_are_reproducible_at_fixed_static_seed(self) -> None:
        names, gradients, _indices = self._canonical_fixture()
        for label in ("compression@8", "noise@1e-6", "lrbprojonly@0.5"):
            arm = next(item for item in RUNNER.FINAL_ARMS if item.label == label)
            first = RUNNER.apply_final_arm_to_canonical_tuple(arm=arm, canonical_gradients=gradients, canonical_parameter_names=names)
            second = RUNNER.apply_final_arm_to_canonical_tuple(arm=arm, canonical_gradients=gradients, canonical_parameter_names=names)
            self.assertTrue(all(a is None if b is None else torch.equal(a, b) for a, b in zip(first, second)))

    def test_all_arms_transform_before_q_projection_extraction(self) -> None:
        names, gradients, indices = self._canonical_fixture()
        transformed_labels: list[str] = []

        def transform(*, arm, canonical_gradients, canonical_parameter_names):
            transformed_labels.append(arm.label)
            return tuple(None if gradient is None else gradient + len(transformed_labels) for gradient in canonical_gradients)

        observations: list[tuple] = []
        def extract(**kwargs):
            q = tuple(kwargs["canonical_gradients"][kwargs["q_canonical_indices"][name]] for name in kwargs["q_parameter_names"])
            observations.append(q)
            return q

        with mock.patch.object(RUNNER, "apply_final_arm_to_canonical_tuple", side_effect=transform), mock.patch.object(RUNNER, "q_projection_observations_from_canonical_tuple", side_effect=extract):
            decoded = RUNNER._decode_final_arms(canonical_gradients=gradients, canonical_parameter_names=names, q_parameter_names=(Q0, Q1), canonical_q_indices=indices, decode_arm=lambda arm, q: {"label": arm.label, "q": q})
        self.assertEqual(transformed_labels, [arm.label for arm in RUNNER.FINAL_ARMS])
        self.assertEqual(len(decoded), 11)
        self.assertEqual(len(observations), 11)
        self.assertTrue(all(tuple(item.shape for item in q) == (torch.Size([2, 2]), torch.Size([2, 2])) for q in observations))

    def test_capture_site_is_unique_and_no_layer1_candidates_continues_all_arms(self) -> None:
        source = (PROJECT_ROOT / "scripts" / "run_layer1_final_all_arms.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("capture_single_example_gradients("), 1)
        names, gradients, indices = self._canonical_fixture()
        decoded = RUNNER._decode_final_arms(
            canonical_gradients=gradients,
            canonical_parameter_names=names,
            q_parameter_names=(Q0, Q1),
            canonical_q_indices=indices,
            decode_arm=lambda arm, _q: {"status": "ok", "termination_reason": "no_layer1_candidates", "label": arm.label},
        )
        self.assertEqual(len(decoded), 11)
        self.assertTrue(all(core["status"] == "ok" and core["termination_reason"] == "no_layer1_candidates" for _arm, core in decoded))

    def _record(self, *, head_seed: int, sample_key: str, arm, status: str = "ok", tau1: float = 0.002) -> dict:
        return {
            "record_type": "qwen3_dager_layer1_final_all_arms",
            "result_status": status,
            "stage": "final",
            "sample_key": sample_key,
            "head_seed": head_seed,
            "defense": arm.defense,
            "preset": "proj_only" if arm.defense == "lrbprojonly" else arm.defense,
            "defense_param_name": arm.parameter_name,
            "defense_param_value": arm.parameter_value,
            "task": "sst2",
            "batch_size": 1,
            "gradient_steps": 1,
            "dtype": "bfloat16",
            "tau1": tau1,
            "tau2": 0.001,
            "frozen_tau1_control_identity_sha256": "a" * 64,
            "canonical_q_proj_indices": {Q0: 1, Q1: 3},
            "attack_semantics": "defense_unaware_observed_q_proj_only",
            "candidate_count": 0 if arm.label == "noise@1e-6" else 2,
            "legacy_l1_token_membership": 0.5,
            "layer_2_survivor_count": 0,
            "token_recovery": 0.0,
        }

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")

    def _sources(self, root: Path):
        manifest = root / "final.jsonl"
        self._write_jsonl(manifest, [{"record_type": "preregistered_sst2_validation_sample", "stage": "final", "sample": {"sample_key": key}} for key in SAMPLE_KEYS])
        paths = {}
        for seed in SUMMARY.FINAL_HEAD_SEEDS:
            rows = [self._record(head_seed=seed, sample_key=key, arm=arm) for key in SAMPLE_KEYS for arm in SUMMARY.FINAL_ARMS]
            path = root / f"seed{seed}.jsonl"
            self._write_jsonl(path, rows)
            paths[seed] = path
        return manifest, paths

    def test_summary_validates_complete_three_seed_lattice_and_reports_pairing(self) -> None:
        with TemporaryDirectory() as temporary:
            manifest, paths = self._sources(Path(temporary))
            summary = SUMMARY.build_layer1_final_all_arms_summary(input_paths=paths, manifest_path=manifest)
        self.assertEqual(summary["comparison_pairing"], "same_capture_within_head_seed")
        self.assertEqual(summary["three_preregistered_fixed_head_seeds"], [101, 202, 303])
        self.assertEqual(set(summary["defenses"]), {arm.label for arm in SUMMARY.FINAL_ARMS})
        self.assertTrue(all(summary["defenses"][arm.label]["n"] == 60 for arm in SUMMARY.FINAL_ARMS))
        self.assertEqual(summary["defenses"]["noise@1e-6"]["zero_candidate_samples"], 60)
        self.assertTrue(all(set(summary["defenses"][arm.label]["by_head_seed"]) == {"101", "202", "303"} for arm in SUMMARY.FINAL_ARMS))

    def test_summary_rejects_missing_duplicate_failure_wrong_arm_illegal_seed_and_protocol_mismatch(self) -> None:
        cases = (
            ("missing_sample", lambda rows: rows.__delitem__(slice(0, 11))),
            ("duplicate", lambda rows: rows.append(dict(rows[0]))),
            ("failed", lambda rows: rows.__setitem__(0, {**rows[0], "result_status": "error"})),
            ("wrong_arm", lambda rows: rows.__setitem__(0, {**rows[0], "defense": "unknown"})),
            ("illegal_head_seed", lambda rows: rows.__setitem__(0, {**rows[0], "head_seed": 999})),
            ("protocol_mismatch", lambda rows: rows.__setitem__(1, {**rows[1], "tau1": 0.1})),
        )
        for name, mutate in cases:
            with self.subTest(name=name), TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest, paths = self._sources(root)
                target = paths[101]
                rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
                mutate(rows)
                self._write_jsonl(target, rows)
                with self.assertRaises(SUMMARY.Layer1FinalAllArmsSummaryError):
                    SUMMARY.build_layer1_final_all_arms_summary(input_paths=paths, manifest_path=manifest)


if __name__ == "__main__":
    unittest.main()
