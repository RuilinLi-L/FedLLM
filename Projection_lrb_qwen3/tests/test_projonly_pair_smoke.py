"""Unit contracts for the isolated Qwen3 Projection-LRB paired smoke."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
for import_root in (REPOSITORY_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import torch

from src.dager_qwen3.none_attack_core import (
    NoneAttackCoreControls,
    execute_dager_from_observed_q_gradients,
    q_projection_observations_from_canonical_tuple,
)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "qwen3_projonly_pair_smoke_runner",
        PROJECT_ROOT / "scripts" / "run_projonly_pair_smoke.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load paired smoke runner module for unit tests.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()
Q0 = "model.layers.0.self_attn.q_proj.weight"
Q1 = "model.layers.1.self_attn.q_proj.weight"


def _dager_dependency_stubs() -> dict[str, ModuleType]:
    candidate_provider = ModuleType("src.dager_qwen3.candidate_provider")

    class Provider:
        @staticmethod
        def from_layer1_result(_layer1, *, eos_token_id, max_ids):
            return SimpleNamespace(token_ids=(eos_token_id,), max_ids=max_ids)

    candidate_provider.RoPECandidateProvider = Provider

    decomposition = ModuleType("src.dager_qwen3.gradient_decomposition")
    decomposition.shared_dager_rank_for_qwen3_qproj_gradients = lambda *_args, **_kwargs: SimpleNamespace(
        applied_shared_rank=2,
        q0_effective_rank=2,
        q1_effective_rank=2,
        rank_definition="relative_svd_threshold",
        rank_rtol=1e-3,
        q0_relative_threshold=0.1,
        q1_relative_threshold=0.1,
        requested_shared_rank=2,
        rank_was_capped=False,
        rank_cap=2,
        cap_reason=None,
    )
    decomposition.decompose_qwen3_qproj_gradient = lambda *_args, **_kwargs: SimpleNamespace(truncated_rank=2)

    layer1_filter = ModuleType("src.dager_qwen3.layer1_filter")
    layer1_filter.filter_qwen3_vocab_layer1 = lambda **_kwargs: SimpleNamespace(candidate_count=1)

    layer2_decoder = ModuleType("src.dager_qwen3.layer2_decoder")
    layer2_decoder.Layer2DecoderConfig = lambda **kwargs: kwargs
    layer2_decoder.decode_qwen3_rope_prefixes = lambda **_kwargs: SimpleNamespace(
        search_budget_exhausted=False,
        selected_token_ids=(9,),
        evaluated_prefix_count=1,
        per_length_survivor_counts=((1, 1),),
        completed_prefixes=(SimpleNamespace(token_ids=(9,)),),
        selected_mean_span_distance=0.1,
    )
    layer2_decoder.layer2_audit_json_fields = lambda _layer2: {
        "termination_reason": "completed_prefix_found"
    }

    metrics = ModuleType("src.dager_qwen3.metrics")
    metrics.compute_attack_metrics = lambda **_kwargs: SimpleNamespace(
        ground_truth_token_text=("ground",),
        ground_truth_text="ground",
        reconstructed_token_text=("reconstructed",),
        reconstructed_text="reconstructed",
        token_recovery=1.0,
        exact_recovery=True,
        rouge_1=1.0,
        rouge_2=1.0,
        empty_reconstruction=False,
    )
    return {
        candidate_provider.__name__: candidate_provider,
        decomposition.__name__: decomposition,
        layer1_filter.__name__: layer1_filter,
        layer2_decoder.__name__: layer2_decoder,
        metrics.__name__: metrics,
    }


class ProjectionLrbPairSmokeTest(unittest.TestCase):
    def _canonical_fixture(self):
        names = ("model.embed_tokens.weight", Q0, "model.score.weight", Q1)
        gradients = (
            torch.tensor([1.0, 2.0]),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            None,
            torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        )
        indices = {Q0: 1, Q1: 3}
        return names, gradients, indices

    def test_identity_lrb_preserves_q_observations_and_shared_core_result(self) -> None:
        names, raw_gradients, indices = self._canonical_fixture()
        identity_gradients = RUNNER.apply_lrb_to_canonical_tuple(
            canonical_gradients=raw_gradients,
            canonical_parameter_names=names,
            defense="lrb",
            preset="identity_lrb",
            keep_ratio=1.0,
            seed=RUNNER.LRB_SEED,
        )
        raw_observations = q_projection_observations_from_canonical_tuple(
            canonical_gradients=raw_gradients,
            canonical_parameter_names=names,
            q_parameter_names=(Q0, Q1),
            q_canonical_indices=indices,
        )
        identity_observations = q_projection_observations_from_canonical_tuple(
            canonical_gradients=identity_gradients,
            canonical_parameter_names=names,
            q_parameter_names=(Q0, Q1),
            q_canonical_indices=indices,
        )
        for raw, identity in zip(raw_observations, identity_observations):
            self.assertTrue(torch.equal(raw, identity))

        adapter = SimpleNamespace(
            device=torch.device("cpu"),
            metadata=SimpleNamespace(hidden_size=2, execution_path="test", vocab_size=10),
        )
        sample = SimpleNamespace(input_ids=(9,), eos_token_id=9)
        controls = NoneAttackCoreControls(
            tau1=1e-3, tau2=1e-3, rank_tolerance=1e-3, rank_cutoff=0,
            max_search_candidates=10, max_candidate_ids=-1, parallel=1, max_sequence_length=2,
        )
        rouge_backend = SimpleNamespace(metric=object(), json_metadata=lambda: {"backend": "test"})
        with mock.patch.dict(sys.modules, _dager_dependency_stubs()):
            none_result = execute_dager_from_observed_q_gradients(
                adapter=adapter, tokenizer=object(), sample=sample, observed_q_gradients=raw_observations,
                q_parameter_names=(Q0, Q1), q_canonical_indices=indices, controls=controls, rouge_backend=rouge_backend,
            )
            identity_result = execute_dager_from_observed_q_gradients(
                adapter=adapter, tokenizer=object(), sample=sample, observed_q_gradients=identity_observations,
                q_parameter_names=(Q0, Q1), q_canonical_indices=indices, controls=controls, rouge_backend=rouge_backend,
            )
        none_result.pop("attack_time_seconds")
        identity_result.pop("attack_time_seconds")
        self.assertEqual(none_result, identity_result)

    def test_projonly_receives_full_canonical_tuple_then_reads_q_gradients_by_index(self) -> None:
        names, raw_gradients, indices = self._canonical_fixture()
        observed_call: dict[str, object] = {}

        def fake_lrb(gradients, _args, *, layer_names):
            observed_call["gradients"] = gradients
            observed_call["layer_names"] = layer_names
            return tuple(None if gradient is None else gradient + 10.0 for gradient in gradients)

        with mock.patch.object(RUNNER, "apply_lrb_defense", side_effect=fake_lrb) as defense:
            transformed = RUNNER.apply_lrb_to_canonical_tuple(
                canonical_gradients=raw_gradients,
                canonical_parameter_names=names,
            )
        defense.assert_called_once()
        self.assertIs(observed_call["gradients"], raw_gradients)
        self.assertEqual(observed_call["layer_names"], list(names))
        self.assertIsNone(transformed[2])
        q0, q1 = q_projection_observations_from_canonical_tuple(
            canonical_gradients=transformed,
            canonical_parameter_names=names,
            q_parameter_names=(Q0, Q1),
            q_canonical_indices=indices,
        )
        self.assertTrue(torch.equal(q0, raw_gradients[1] + 10.0))
        self.assertTrue(torch.equal(q1, raw_gradients[3] + 10.0))

    def test_runner_is_fixed_smoke_only_and_has_no_other_attack_branch(self) -> None:
        source = (PROJECT_ROOT / "scripts" / "run_projonly_pair_smoke.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.append(node.module)
        self.assertIn('SMOKE_STAGE = "smoke"', source)
        self.assertIn("SMOKE_HEAD_SEED = 22", source)
        self.assertIn('SMOKE_DTYPE = "bfloat16"', source)
        self.assertIn("apply_lrb_defense(canonical_gradients", source)
        self.assertNotIn("final.jsonl", source)
        self.assertNotIn("run_calibration", source)
        self.assertNotIn("run_none_attack", source)
        self.assertFalse(any(name == "src.calibration" or ".calibration" in name for name in imported))
        self.assertFalse(any("adaptive" in name.lower() or "peft" in name.lower() for name in imported))


if __name__ == "__main__":
    unittest.main()
