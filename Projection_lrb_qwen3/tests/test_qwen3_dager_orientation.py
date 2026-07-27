"""Regression checks for Qwen3 nn.Linear DAGER gradient orientation."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.dager_qwen3.gradient_decomposition import (
    GradientDecompositionError,
    decompose_qwen3_qproj_gradient,
)
from src.dager_qwen3.layer1_filter import span_distances


class Qwen3DagerOrientationTest(unittest.TestCase):
    def test_raw_linear_gradient_row_space_is_correct_and_transpose_is_rejected(self) -> None:
        torch.manual_seed(7)
        hidden = torch.randn(1, 3, 3)
        delta = torch.randn(1, 3, 4)
        gradient = delta.reshape(-1, 4).transpose(0, 1) @ hidden.reshape(-1, 3)
        span = decompose_qwen3_qproj_gradient(
            gradient,
            feature_dim=3,
            rank_tolerance=1e-6,
            rank_cutoff=0,
            decomposition_device=torch.device("cpu"),
        )
        correct = span_distances(
            basis=span.basis,
            representations=hidden.reshape(-1, 3),
            norm="l2",
        )
        self.assertLess(float(correct.max()), 1e-4)
        with self.assertRaises(GradientDecompositionError):
            decompose_qwen3_qproj_gradient(
                gradient.transpose(0, 1),
                feature_dim=3,
                rank_tolerance=1e-6,
                rank_cutoff=0,
                decomposition_device=torch.device("cpu"),
            )

    def test_none_attack_cli_has_no_free_text_argument(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_none_attack.py"),
                "--stage",
                "smoke",
                "--sample-key",
                "0" * 64,
                "--head-seed",
                "1",
                "--output",
                "Projection_lrb_qwen3/outputs/smoke/test.jsonl",
                "--sentence",
                "forbidden",
            ],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("unrecognized arguments: --sentence forbidden", completed.stderr)


if __name__ == "__main__":
    unittest.main()
