"""Failure-path ownership tests for the shared Qwen3 DAGER core."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class GradientFailureOwnershipTest(unittest.TestCase):
    def test_shared_core_checks_orientation_before_layer1_or_layer2_search(self) -> None:
        source = (PROJECT_ROOT / "src" / "dager_qwen3" / "none_attack_core.py").read_text(encoding="utf-8")
        self.assertIn("diagnostic_thresholds(dtype)", source)
        self.assertIn("raise GradientDiagnosticFailure(diagnostic, diagnostic_controls)", source)
        self.assertLess(source.index("raise GradientDiagnosticFailure"), source.index("filter_qwen3_vocab_layer1("))
        self.assertLess(source.index("raise GradientDiagnosticFailure"), source.index("decode_qwen3_rope_prefixes("))

    def test_standalone_runner_delegates_to_shared_core_without_local_search(self) -> None:
        source = (PROJECT_ROOT / "scripts" / "run_none_attack.py").read_text(encoding="utf-8")
        self.assertIn("execute_none_only_dager(", source)
        self.assertNotIn("diagnose_two_q_projections", source)
        self.assertNotIn("filter_qwen3_vocab_layer1", source)
        self.assertNotIn("decode_qwen3_rope_prefixes", source)


if __name__ == "__main__":
    unittest.main()
