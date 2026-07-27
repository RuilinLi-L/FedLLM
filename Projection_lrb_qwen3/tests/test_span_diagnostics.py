"""FP32 raw-gradient row-space direction and residual tests."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.span_diagnostics import SpanDiagnosticsError, diagnose_raw_gradient_row_space


class SpanDiagnosticsTests(unittest.TestCase):
    def test_input_in_raw_gradient_row_space_passes_in_fp32(self) -> None:
        gradient = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.bfloat16)
        q_input = torch.tensor([[[0.5, 0.25, 0.0], [1.0, -2.0, 0.0]]], dtype=torch.bfloat16)
        result = diagnose_raw_gradient_row_space(
            q_input=q_input,
            raw_gradient=gradient,
            rank_tol=1e-6,
            max_relative_residual=1e-5,
            gradient_orientation="gradient",
        )
        self.assertEqual(result["working_dtype"], "torch.float32")
        self.assertEqual(result["numerical_rank"], 2)
        self.assertTrue(result["passes_relative_residual"])
        self.assertLess(result["relative_residual"]["max"], 1e-6)
        self.assertEqual(result["svd_basis_direction"], "right_singular_vectors_of_gradient")

    def test_out_of_row_space_input_fails_without_transpose_fallback(self) -> None:
        gradient = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.bfloat16)
        q_input = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.bfloat16)
        result = diagnose_raw_gradient_row_space(
            q_input=q_input,
            raw_gradient=gradient,
            rank_tol=1e-6,
            max_relative_residual=1e-4,
            gradient_orientation="gradient",
        )
        self.assertFalse(result["passes_relative_residual"])
        self.assertAlmostEqual(result["relative_residual"]["max"], 1.0, places=6)
        self.assertEqual(result["gradient_orientation"], "gradient")

    def test_explicit_transpose_dimension_mismatch_is_an_error(self) -> None:
        gradient = torch.ones((2, 3), dtype=torch.bfloat16)
        q_input = torch.ones((1, 1, 3), dtype=torch.bfloat16)
        with self.assertRaisesRegex(SpanDiagnosticsError, "gradient.T"):
            diagnose_raw_gradient_row_space(
                q_input=q_input,
                raw_gradient=gradient,
                rank_tol=1e-6,
                max_relative_residual=1e-4,
                gradient_orientation="gradient.T",
            )


if __name__ == "__main__":
    unittest.main()
