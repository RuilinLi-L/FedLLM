"""Read-only audit tests for none-only Qwen3 Layer-2 DAGER decoding."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
import sys
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

_LAYER1_FILTER_MODULE = "src.dager_qwen3.layer1_filter"
_saved_layer1_filter = sys.modules.get(_LAYER1_FILTER_MODULE)
_layer1_filter_stub = ModuleType(_LAYER1_FILTER_MODULE)


class _StubLayer1FilterError(RuntimeError):
    pass


def _unexpected_span_distances(**_kwargs: object) -> torch.Tensor:
    raise AssertionError("decoder-control tests must not invoke the real span-distance computation")


_layer1_filter_stub.DistanceNorm = str
_layer1_filter_stub.Layer1FilterError = _StubLayer1FilterError
_layer1_filter_stub.span_distances = _unexpected_span_distances
sys.modules[_LAYER1_FILTER_MODULE] = _layer1_filter_stub
try:
    from src.dager_qwen3.layer2_decoder import (
        Layer2DecoderConfig,
        decode_qwen3_rope_prefixes,
        layer2_audit_json_fields,
    )
finally:
    if _saved_layer1_filter is None:
        del sys.modules[_LAYER1_FILTER_MODULE]
    else:
        sys.modules[_LAYER1_FILTER_MODULE] = _saved_layer1_filter



class Layer2DecoderAuditTest(unittest.TestCase):
    def _decode(
        self,
        *,
        responses: list[tuple[list[bool], list[float]]],
        max_sequence_length: int = 2,
        search_budget: int = 100,
        token_ids: tuple[int, ...] = (1, 2),
    ):
        adapter = SimpleNamespace(metadata=SimpleNamespace(hidden_size=3))
        span = SimpleNamespace(feature_dim=3)
        provider = SimpleNamespace(
            token_ids=token_ids,
            distances=tuple(0.1 for _ in token_ids),
            eos_token_id=9,
            candidates_for_position=lambda _position: token_ids,
        )
        iterator = iter(responses)

        def evaluate(**_kwargs):
            return next(iterator)

        with mock.patch("src.dager_qwen3.layer2_decoder._evaluate_prefix_batch", side_effect=evaluate):
            return decode_qwen3_rope_prefixes(
                adapter=adapter,
                span=span,
                candidate_provider=provider,
                config=Layer2DecoderConfig(
                    max_sequence_length=max_sequence_length,
                    threshold=0.5,
                    distance_norm="l2",
                    search_budget=search_budget,
                    decode_batch_size=64,
                ),
            )

    def test_all_finite_rejections_report_length_one_termination(self) -> None:
        token_ids = tuple(range(1, 20))
        distances = [0.6 + 0.01 * index for index in range(19)]
        result = self._decode(
            responses=[([False] * 19, distances)],
            token_ids=token_ids,
        )
        audit = result.per_length_distance_audit
        self.assertEqual(result.per_length_survivor_counts, ((1, 0), (2, 0)))
        self.assertEqual(result.termination_reason, "no_layer2_survivor_at_length_1")
        self.assertEqual(audit[0].evaluated_count, 19)
        self.assertEqual(audit[0].finite_distance_count, 19)
        self.assertEqual(audit[0].nonfinite_distance_count, 0)
        self.assertEqual(audit[0].passing_count, 0)
        self.assertEqual(audit[0].rejected_count, 19)
        self.assertEqual((audit[0].distance_min, audit[0].distance_max), (0.6, 0.78))
        self.assertAlmostEqual(float(audit[0].distance_median), 0.69)

    def test_equality_verdict_is_audit_only_and_completed_prefix_order_is_preserved(self) -> None:
        # The mocked evaluator supplies the pre-existing predicate verdict.  The
        # audit must not recompute or alter it merely because a reported mean
        # distance is exactly the threshold.
        result = self._decode(
            responses=[([True, False], [0.5, 0.7])],
            max_sequence_length=1,
            token_ids=(9, 2),
        )
        self.assertEqual(result.selected_token_ids, (9,))
        self.assertEqual([item.token_ids for item in result.completed_prefixes], [(9,)])
        self.assertEqual(result.termination_reason, "completed_prefix_found")
        self.assertEqual(result.per_length_distance_audit[0].passing_count, 1)
        self.assertEqual(result.per_length_distance_audit[0].rejected_count, 1)

    def test_nonfinite_means_are_counted_and_not_summarized(self) -> None:
        result = self._decode(responses=[([False, False], [float("nan"), float("inf")])])
        audit = result.per_length_distance_audit[0]
        self.assertEqual(audit.finite_distance_count, 0)
        self.assertEqual(audit.nonfinite_distance_count, 2)
        self.assertEqual(audit.passing_count, 0)
        self.assertEqual(audit.rejected_count, 2)
        self.assertIsNone(audit.distance_min)
        self.assertIsNone(audit.distance_median)
        self.assertIsNone(audit.distance_max)

    def test_raw_nonfinite_distances_cannot_become_survivors(self) -> None:
        from src.dager_qwen3 import layer2_decoder

        adapter = SimpleNamespace(
            device=torch.device("cpu"),
            layer1_qproj_inputs_from_prefixes=lambda *, input_ids, attention_mask: torch.zeros(
                (input_ids.shape[0], input_ids.shape[1], 3), dtype=torch.float32
            ),
        )
        span = SimpleNamespace(basis=torch.eye(3, dtype=torch.float32))
        with mock.patch.object(
            layer2_decoder,
            "span_distances",
            return_value=torch.tensor([float("nan"), float("inf")], dtype=torch.float32),
        ):
            passes, means = layer2_decoder._evaluate_prefix_batch(
                adapter=adapter,
                span=span,
                prefixes=[(1,), (2,)],
                threshold=0.5,
                distance_norm="l2",
            )
        self.assertEqual(passes, [False, False])
        self.assertTrue(math.isnan(means[0]))
        self.assertTrue(math.isinf(means[1]))

    def test_empty_layer1_candidate_provider_reports_its_distinct_termination(self) -> None:
        result = self._decode(responses=[], token_ids=())
        self.assertEqual(result.evaluated_prefix_count, 0)
        self.assertEqual(result.termination_reason, "no_layer1_candidates")
        self.assertEqual(result.per_length_distance_audit[0].evaluated_count, 0)

    def test_survivors_keep_existing_prefix_order_and_counts(self) -> None:
        result = self._decode(
            responses=[([True, True], [0.1, 0.2]), ([False, True], [0.8, 0.3])],
            max_sequence_length=2,
            token_ids=(1, 9),
        )
        self.assertEqual(result.per_length_survivor_counts, ((1, 1), (2, 0)))
        self.assertEqual([item.token_ids for item in result.completed_prefixes], [(9,), (1, 9)])
        self.assertEqual(result.selected_token_ids, (9,))
        self.assertEqual(result.per_length_distance_audit[0].passing_count, 2)
        self.assertEqual(result.per_length_distance_audit[1].passing_count, 1)
        self.assertEqual(result.termination_reason, "completed_prefix_found")

    def test_budget_exhaustion_is_reported_without_changing_existing_counts(self) -> None:
        result = self._decode(
            responses=[([True], [0.1])],
            max_sequence_length=2,
            search_budget=1,
            token_ids=(1, 2),
        )
        self.assertTrue(result.search_budget_exhausted)
        self.assertEqual(result.evaluated_prefix_count, 1)
        self.assertEqual(result.per_length_survivor_counts, ((1, 1),))
        self.assertEqual(result.termination_reason, "search_budget_exhausted")


class Layer2RunnerSerializationTest(unittest.TestCase):
    def test_json_fields_include_layer2_audit_and_termination_reason(self) -> None:
        layer2 = SimpleNamespace(
            per_length_distance_audit=(
                SimpleNamespace(
                    prefix_length=1,
                    evaluated_count=19,
                    finite_distance_count=19,
                    nonfinite_distance_count=0,
                    passing_count=0,
                    rejected_count=19,
                    threshold=1e-3,
                    distance_min=0.2,
                    distance_median=0.4,
                    distance_max=0.9,
                ),
            ),
            termination_reason="no_layer2_survivor_at_length_1",
        )
        fields = layer2_audit_json_fields(layer2)
        self.assertEqual(fields["termination_reason"], "no_layer2_survivor_at_length_1")
        self.assertEqual(fields["layer_2_distance_audit"][0]["evaluated_count"], 19)
        self.assertEqual(fields["layer_2_distance_audit"][0]["passing_count"], 0)
        self.assertEqual(fields["layer_2_distance_audit"][0]["rejected_count"], 19)
        self.assertTrue(math.isfinite(fields["layer_2_distance_audit"][0]["distance_median"]))
        self.assertIn("layer_2_distance_audit", json.dumps(fields, sort_keys=True))
        runner_source = (PROJECT_ROOT / "scripts" / "run_none_attack.py").read_text(encoding="utf-8")
        self.assertIn("**layer2_audit_json_fields(layer2)", runner_source)


if __name__ == "__main__":
    unittest.main()
