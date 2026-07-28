"""Unit contracts for the isolated Qwen3 Projection-LRB paired smoke."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
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


_LAYER1_FILTER_MODULE = "src.dager_qwen3.layer1_filter"
_saved_layer1_filter = sys.modules.get(_LAYER1_FILTER_MODULE)
_layer1_filter_stub = ModuleType(_LAYER1_FILTER_MODULE)
_layer1_filter_stub.Layer1FilterResult = object
sys.modules[_LAYER1_FILTER_MODULE] = _layer1_filter_stub
try:
    from src.dager_qwen3.candidate_provider import RoPECandidateProvider
finally:
    if _saved_layer1_filter is None:
        del sys.modules[_LAYER1_FILTER_MODULE]
    else:
        sys.modules[_LAYER1_FILTER_MODULE] = _saved_layer1_filter


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
        rank_definition="absolute_matrix_rank_atol_rtol_zero",
        rank_atol=1e-3,
        requested_shared_rank=2,
        rank_was_capped=False,
        rank_cap=2,
        cap_reason=None,
    )
    decomposition.decompose_qwen3_qproj_gradient = lambda *_args, **_kwargs: SimpleNamespace(truncated_rank=2)

    layer1_filter = ModuleType("src.dager_qwen3.layer1_filter")
    layer1_filter.filter_qwen3_vocab_layer1 = lambda **_kwargs: SimpleNamespace(
        candidate_count=1, token_ids=(9,), distances=(0.1,)
    )

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
        legacy_l1_token_membership=1.0,
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

    def test_empty_layer1_vocabulary_is_a_normal_terminal_provider(self) -> None:
        provider = RoPECandidateProvider.from_layer1_result(
            SimpleNamespace(token_ids=(), distances=()),
            eos_token_id=9,
            max_ids=-1,
        )
        self.assertEqual(provider.token_ids, ())
        self.assertEqual(provider.distances, ())
        self.assertEqual(provider.candidates_for_position(0), ())

    def test_shared_core_reports_no_l1_candidates_without_a_decode_error(self) -> None:
        adapter = SimpleNamespace(
            device=torch.device("cpu"),
            metadata=SimpleNamespace(hidden_size=2, execution_path="test", vocab_size=10),
        )
        sample = SimpleNamespace(input_ids=(9,), eos_token_id=9)
        controls = NoneAttackCoreControls(
            tau1=.002, tau2=.001, rank_tolerance=.001, rank_cutoff=0,
            max_search_candidates=10, max_candidate_ids=-1, parallel=1, max_sequence_length=2,
        )
        rouge_backend = SimpleNamespace(metric=object(), json_metadata=lambda: {"backend": "test"})
        stubs = _dager_dependency_stubs()
        layer1_filter = stubs["src.dager_qwen3.layer1_filter"]
        layer1_filter.filter_qwen3_vocab_layer1 = lambda **_kwargs: SimpleNamespace(
            token_ids=(), distances=(), candidate_count=0
        )
        provider_module = stubs["src.dager_qwen3.candidate_provider"]
        provider_module.RoPECandidateProvider.from_layer1_result = staticmethod(
            lambda result, **_kwargs: SimpleNamespace(
                token_ids=tuple(result.token_ids), distances=tuple(result.distances), eos_token_id=9
            )
        )
        layer2_decoder = stubs["src.dager_qwen3.layer2_decoder"]
        seen: dict[str, object] = {}

        def decode(*, candidate_provider, **_kwargs):
            seen["token_ids"] = candidate_provider.token_ids
            return SimpleNamespace(
                search_budget_exhausted=False,
                selected_token_ids=(),
                evaluated_prefix_count=0,
                per_length_survivor_counts=((1, 0),),
                completed_prefixes=(),
                selected_mean_span_distance=None,
                termination_reason="no_l1_candidates",
            )

        layer2_decoder.decode_qwen3_rope_prefixes = decode
        layer2_decoder.layer2_audit_json_fields = lambda layer2: {
            "termination_reason": layer2.termination_reason,
            "layer_2_distance_audit": [],
        }
        metrics = stubs["src.dager_qwen3.metrics"]
        metrics.compute_attack_metrics = lambda **_kwargs: SimpleNamespace(
            ground_truth_token_text=("ground",),
            ground_truth_text="ground",
            reconstructed_token_text=(),
            reconstructed_text="",
            token_recovery=0.0,
            legacy_l1_token_membership=0.0,
            exact_recovery=False,
            rouge_1=0.0,
            rouge_2=0.0,
            empty_reconstruction=True,
        )
        with mock.patch.dict(sys.modules, stubs):
            result = execute_dager_from_observed_q_gradients(
                adapter=adapter,
                tokenizer=object(),
                sample=sample,
                observed_q_gradients=(torch.eye(2), torch.eye(2)),
                q_parameter_names=(Q0, Q1),
                q_canonical_indices={Q0: 0, Q1: 1},
                controls=controls,
                rouge_backend=rouge_backend,
            )
        self.assertEqual(seen["token_ids"], ())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["termination_reason"], "no_l1_candidates")
        self.assertEqual(result["layer_1_candidate_count"], 0)
        self.assertEqual(result["search_budget"]["evaluated_prefix_count"], 0)

    def test_no_l1_terminal_result_does_not_abort_the_other_pair_arm(self) -> None:
        names = (Q0, Q1)
        raw_gradients = (
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        )
        canonical_manifest = {
            "parameter_tensor_count": 2,
            "gradient_tensor_count": 2,
            "entries": [
                {"name": Q0, "grad_present": True},
                {"name": Q1, "grad_present": True},
            ],
        }
        captured = SimpleNamespace(
            q_inputs=(object(), object()),
            q_output_gradients=(object(), object()),
            q_gradients=raw_gradients,
            q_parameter_names=(Q0, Q1),
        )
        sample = SimpleNamespace(
            eos_token_id=9,
            input_ids=(3, 9),
            label=1,
            sample_key="sample-key",
            preregistration_sha256="a" * 64,
        )
        bundle = SimpleNamespace(
            model=object(),
            tokenizer=SimpleNamespace(eos_token_id=9),
            device=torch.device("cpu"),
        )
        control_values = SimpleNamespace(
            l1_span_threshold=.002,
            l2_span_threshold=.001,
            rank_tolerance=.001,
            rank_cutoff=20,
            max_search_candidates=10,
            max_candidate_ids=-1,
            decode_batch_size=1,
            max_sequence_length=8,
        )

        def core(*, status: str, termination_reason: str, candidate_count: int) -> dict[str, object]:
            return {
                "status": status,
                "tau1": .002,
                "tau2": .001,
                "layer_1_rank": 2,
                "layer_2_rank": 2,
                "requested_shared_rank": 2,
                "applied_shared_rank": 2,
                "layer_1_candidate_count": candidate_count,
                "layer_1_decoder_candidate_count": candidate_count,
                "reconstructed_token_ids": [],
                "token_recovery": 0.0,
                "exact_recovery": False,
                "rouge_1": 0.0,
                "rouge_2": 0.0,
                "termination_reason": termination_reason,
                "attack_time_seconds": .01,
            }

        gradient_gate = ModuleType("src.dager_qwen3.gradient_gate")
        gradient_gate.decode_token_texts = lambda _tokenizer, token_ids: tuple(str(value) for value in token_ids)
        gradient_gate.diagnostic_thresholds = lambda _dtype: {"gate": .1}
        model_adapter = ModuleType("src.dager_qwen3.model_adapter")
        model_adapter.Qwen3RoPEDagerAdapter = lambda *_args: SimpleNamespace()
        gradient_capture = ModuleType("src.gradient_capture")
        gradient_capture.capture_single_example_gradients = lambda *_args, **_kwargs: captured
        classifier = ModuleType("src.qwen3_classifier")
        classifier.load_local_qwen3_sequence_classifier = lambda *_args, **_kwargs: bundle
        span_diagnostics = ModuleType("src.span_diagnostics")
        span_diagnostics.diagnose_two_q_projections = lambda **_kwargs: {"passed": True}
        modules = {
            gradient_gate.__name__: gradient_gate,
            model_adapter.__name__: model_adapter,
            gradient_capture.__name__: gradient_capture,
            classifier.__name__: classifier,
            span_diagnostics.__name__: span_diagnostics,
        }
        frozen_tau1 = {"selected_tau1": .002, "frozen_control_identity_sha256": "b" * 64}
        config = SimpleNamespace(model_path=Path("model"), config_sha256="c" * 64)
        no_l1 = core(status="ok", termination_reason="no_l1_candidates", candidate_count=0)
        recovered = core(status="ok", termination_reason="completed_prefix_found", candidate_count=1)

        with TemporaryDirectory() as temporary, mock.patch.dict(sys.modules, modules), mock.patch.object(
            RUNNER, "load_experiment_config", return_value=config
        ), mock.patch.object(RUNNER, "registered_head_seed"), mock.patch.object(
            RUNNER, "load_registered_sample", return_value=sample
        ), mock.patch.object(RUNNER, "verify_frozen_tau1_control", return_value=frozen_tau1), mock.patch.object(
            RUNNER, "load_none_attack_controls", return_value=control_values
        ), mock.patch.object(RUNNER, "preflight_legacy_dager_rouge_backend", return_value=object()), mock.patch.object(
            RUNNER, "_canonical_gradient_tuple", return_value=(names, raw_gradients, canonical_manifest)
        ), mock.patch.object(RUNNER, "q_canonical_indices", return_value={Q0: 0, Q1: 1}), mock.patch.object(
            RUNNER, "apply_lrb_to_canonical_tuple", side_effect=lambda **kwargs: kwargs["canonical_gradients"]
        ), mock.patch.object(RUNNER, "_decode_observed_arm", side_effect=(no_l1, recovered)) as decode:
            output = Path(temporary) / "paired_smoke.jsonl"
            with mock.patch.object(RUNNER, "OUTPUT_PATH", output):
                records = RUNNER.run_pair_smoke(SimpleNamespace(device="cpu"))
            persisted_lines = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(decode.call_count, 2)
        self.assertEqual(len(persisted_lines), 2)
        self.assertEqual([record["defense"] for record in records], ["none", "lrbprojonly"])
        self.assertEqual(records[0]["termination_reason"], "no_l1_candidates")
        self.assertEqual(records[0]["result_status"], "ok")
        self.assertEqual(records[0]["candidate_count"], 0)
        self.assertEqual(records[1]["termination_reason"], "completed_prefix_found")

    def test_all_smoke_loader_uses_each_manifest_key_without_manual_selection(self) -> None:
        first_key = "a" * 64
        second_key = "b" * 64
        with TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            manifest = project_root / "manifests" / "smoke.jsonl"
            manifest.parent.mkdir()
            manifest.write_text(
                "\n".join(
                    json.dumps({"sample": {"sample_key": key}})
                    for key in (first_key, second_key)
                ) + "\n",
                encoding="utf-8",
            )
            config = SimpleNamespace(project_root=project_root)
            with mock.patch.object(
                RUNNER,
                "load_registered_sample",
                side_effect=lambda **kwargs: SimpleNamespace(sample_key=kwargs["sample_key"]),
            ) as load_sample:
                samples = RUNNER._all_registered_smoke_samples(config)
        self.assertEqual(tuple(sample.sample_key for sample in samples), (first_key, second_key))
        self.assertEqual(
            [call.kwargs for call in load_sample.call_args_list],
            [
                {"config": config, "stage": "smoke", "sample_key": first_key},
                {"config": config, "stage": "smoke", "sample_key": second_key},
            ],
        )

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
        self.assertIn("--all-smoke", source)
        self.assertIn("ALL_SMOKE_OUTPUT_PATH", source)
        self.assertNotIn("final.jsonl", source)
        self.assertNotIn("run_calibration", source)
        self.assertNotIn("run_none_attack", source)
        self.assertFalse(any(name == "src.calibration" or ".calibration" in name for name in imported))
        self.assertFalse(any("adaptive" in name.lower() or "peft" in name.lower() for name in imported))


if __name__ == "__main__":
    unittest.main()
