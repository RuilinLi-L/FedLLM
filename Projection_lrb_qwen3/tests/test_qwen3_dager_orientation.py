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
    shared_dager_rank_for_qwen3_qproj_gradients,
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

    def test_relative_rank_ignores_bf16_noise_without_using_absolute_rank(self) -> None:
        dimension = 32
        values = torch.tensor([10.0, 8.0, 5.0, *([1e-3] * (dimension - 3))], dtype=torch.float32)
        q0_gradient = torch.diag(values).to(dtype=torch.bfloat16)
        q1_gradient = torch.diag(values * 0.5).to(dtype=torch.bfloat16)
        absolute_diagnostic_rank = int(
            torch.linalg.matrix_rank(q0_gradient.float(), atol=1e-6, rtol=0.0).item()
        )
        self.assertGreater(absolute_diagnostic_rank, 3)
        selection = shared_dager_rank_for_qwen3_qproj_gradients(
            (q0_gradient, q1_gradient),
            feature_dim=dimension,
            rank_tolerance=0.1,
            rank_cutoff=0,
            decomposition_device=torch.device("cpu"),
        )
        self.assertEqual(selection.rank_definition, "relative_svd_threshold")
        self.assertEqual(selection.q0_effective_rank, 3)
        self.assertEqual(selection.q1_effective_rank, 3)
        self.assertEqual(selection.requested_shared_rank, 3)
        self.assertEqual(selection.applied_shared_rank, 3)
        self.assertFalse(selection.rank_was_capped)
        self.assertEqual(selection.rank_cap, dimension)
        span = decompose_qwen3_qproj_gradient(
            q0_gradient,
            feature_dim=dimension,
            rank_tolerance=0.1,
            rank_cutoff=0,
            decomposition_device=torch.device("cpu"),
            shared_truncated_rank=selection.applied_shared_rank,
        )
        self.assertEqual(span.effective_rank, 3)
        self.assertEqual(span.applied_rank, 3)
        self.assertGreater(span.relative_threshold, 0.0)


if __name__ == "__main__":
    unittest.main()
