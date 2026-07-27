"""Isolated tests for the mandatory cached legacy ROUGE preflight."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dager_qwen3.metrics import AttackMetricsError, preflight_legacy_dager_rouge_backend


class _FakeMid:
    fmeasure = 1.0


class _FakeScore:
    mid = _FakeMid()


class _FakeRougeMetric:
    observed_offline_values: tuple[str | None, str | None] | None = None

    def compute(self, *, predictions: list[str], references: list[str]) -> dict[str, _FakeScore]:
        _FakeRougeMetric.observed_offline_values = (
            os.environ.get("HF_DATASETS_OFFLINE"),
            os.environ.get("HF_HUB_OFFLINE"),
        )
        if predictions != references:
            raise AssertionError("The preflight must use a fixed exact-match pair.")
        return {"rouge1": _FakeScore(), "rouge2": _FakeScore()}


def _fake_datasets_module(*, expose_load_metric: bool = True) -> types.ModuleType:
    module = types.ModuleType("datasets")
    module.__version__ = "test-datasets-1.0"
    if expose_load_metric:
        module.load_metric = lambda name: _FakeRougeMetric() if name == "rouge" else None  # type: ignore[attr-defined]
    return module


class LegacyRougePreflightTest(unittest.TestCase):
    def test_preflight_forces_offline_legacy_backend_and_records_provenance(self) -> None:
        _FakeRougeMetric.observed_offline_values = None
        with mock.patch.dict(sys.modules, {"datasets": _fake_datasets_module()}), mock.patch.dict(
            os.environ,
            {"HF_DATASETS_OFFLINE": "0", "HF_HUB_OFFLINE": "0"},
        ):
            backend = preflight_legacy_dager_rouge_backend()
            metadata = backend.json_metadata()

        self.assertEqual(_FakeRougeMetric.observed_offline_values, ("1", "1"))
        self.assertEqual(backend.backend, "datasets.load_metric('rouge')")
        self.assertEqual(backend.datasets_version, "test-datasets-1.0")
        self.assertEqual(len(backend.metric_script_sha256), 64)
        self.assertEqual(metadata["hf_datasets_offline"], "1")
        self.assertEqual(metadata["hf_hub_offline"], "1")
        self.assertEqual(metadata["self_test"]["kind"], "fixed_exact_match")
        self.assertEqual(metadata["self_test"]["rouge_1"], 1.0)
        self.assertEqual(metadata["self_test"]["rouge_2"], 1.0)

    def test_preflight_rejects_datasets_without_load_metric(self) -> None:
        with mock.patch.dict(sys.modules, {"datasets": _fake_datasets_module(expose_load_metric=False)}):
            with self.assertRaisesRegex(AttackMetricsError, "does not expose datasets.load_metric"):
                preflight_legacy_dager_rouge_backend()


if __name__ == "__main__":
    unittest.main()
