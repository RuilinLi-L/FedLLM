"""Tests for immutable, canonical JSON and JSONL preregistration artifacts."""

from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.result_schema import ResultSchemaError, write_or_verify_json, write_or_verify_jsonl


class ResultSchemaTests(unittest.TestCase):
    def test_json_is_immutable_by_identity(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "preregistration.json"
            document = {"preregistration_sha256": "a" * 64, "value": [1, 2]}
            self.assertTrue(write_or_verify_json(path, document, identity_key="preregistration_sha256"))
            self.assertFalse(write_or_verify_json(path, document, identity_key="preregistration_sha256"))
            with self.assertRaises(ResultSchemaError):
                write_or_verify_json(
                    path,
                    {"preregistration_sha256": "a" * 64, "value": [3, 4]},
                    identity_key="preregistration_sha256",
                )
            with self.assertRaises(ResultSchemaError):
                write_or_verify_json(
                    path,
                    {"preregistration_sha256": "b" * 64, "value": [1, 2]},
                    identity_key="preregistration_sha256",
                )

    def test_jsonl_is_canonical_and_conflict_detecting(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "calibration.jsonl"
            records = [{"b": 2, "a": 1}, {"stage": "calibration", "index": 0}]
            self.assertTrue(write_or_verify_jsonl(path, records))
            self.assertFalse(write_or_verify_jsonl(path, records))
            self.assertEqual(path.read_text(encoding="utf-8"), '{"a":1,"b":2}\n{"index":0,"stage":"calibration"}\n')
            with self.assertRaises(ResultSchemaError):
                write_or_verify_jsonl(path, [{"a": 2}])


if __name__ == "__main__":
    unittest.main()
