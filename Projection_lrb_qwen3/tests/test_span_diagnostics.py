"""Model-free FP32 identity, rank, and active-token diagnostic tests."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.span_diagnostics import diagnose_q_projection_layer, linear_gradient_identity


def reconstructed_gradient(h: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    return delta.reshape(-1, delta.shape[-1]).transpose(0, 1) @ h.reshape(-1, h.shape[-1])


def diagnose(h: torch.Tensor, delta: torch.Tensor, gradient: torch.Tensor) -> dict[str, object]:
    sequence_length = h.shape[1]
    return diagnose_q_projection_layer(
        q_input=h,
        delta=delta,
        raw_gradient=gradient,
        token_ids=list(range(10, 10 + sequence_length)),
        token_texts=[f"t{position}" for position in range(sequence_length)],
        eos_token_id=10 + sequence_length - 1,
        rank_atol=1e-6,
        rank_rtol=1e-3,
        delta_rtol=1e-3,
        identity_error_tol=5e-3,
        max_active_relative_residual=1e-4,
        negative_control_factor=10.0,
    )


class SpanDiagnosticsTests(unittest.TestCase):
    def test_linear_gradient_identity_reconstructs_delta_transpose_h(self) -> None:
        h = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]], dtype=torch.float32)
        delta = torch.tensor([[[1.0, 0.0], [2.0, 1.0]]], dtype=torch.float32)
        gradient = reconstructed_gradient(h, delta)
        identity = linear_gradient_identity(q_input=h, delta=delta, raw_gradient=gradient)
        self.assertLess(identity["gradient_relative_error"], 1e-7)
        self.assertGreater(identity["gradient_t_relative_error"], 0.1)
        self.assertTrue(identity["gradient_t_comparable"])

    def test_bf16_relative_rank_respects_theoretical_cap(self) -> None:
        h = torch.tensor(
            [[[1.0, 0.5, -0.25], [2.0, 1.0, -0.5], [3.0, 1.5, -0.75],
              [4.0, 2.0, -1.0], [5.0, 2.5, -1.25], [6.0, 3.0, -1.5]]],
            dtype=torch.bfloat16,
        )
        delta = torch.tensor(
            [[[1.0, -2.0], [2.0, -4.0], [3.0, -6.0],
              [4.0, -8.0], [5.0, -10.0], [6.0, -12.0]]],
            dtype=torch.bfloat16,
        )
        gradient = reconstructed_gradient(h.float(), delta.float()).to(torch.bfloat16)
        result = diagnose(h, delta, gradient)
        rank = result["rank"]
        self.assertLessEqual(rank["relative_threshold_rank"], rank["theoretical_rank_cap"])
        self.assertTrue(rank["relative_rank_exceeds_theoretical_cap"] is False)

    def test_inactive_delta_token_does_not_fail_active_residual_gate(self) -> None:
        h = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float32)
        delta = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]], dtype=torch.float32)
        gradient = reconstructed_gradient(h, delta)
        result = diagnose(h, delta, gradient)
        per_token = result["per_token"]
        self.assertTrue(per_token[0]["active_by_delta"])
        self.assertFalse(per_token[1]["active_by_delta"])
        self.assertEqual(result["delta_activity"]["inactive_token_positions"], [1])
        self.assertLess(result["row_space_residual"]["active_tokens"]["max"], 1e-6)
        self.assertGreater(result["row_space_residual"]["all_tokens"]["max"], 0.9)
        self.assertTrue(result["checks"]["active_token_residual"])

    def test_gradient_transpose_is_a_worse_negative_control(self) -> None:
        h = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]], dtype=torch.float32)
        delta = torch.tensor([[[1.0, 0.0], [2.0, 1.0]]], dtype=torch.float32)
        gradient = reconstructed_gradient(h, delta)
        result = diagnose(h, delta, gradient)
        identity = result["identity"]
        self.assertGreater(identity["gradient_t_relative_error"], identity["gradient_relative_error"] * 10.0)
        self.assertTrue(result["gradient_t_negative_control"]["is_obviously_worse"])

    def test_per_token_records_match_sequence_length(self) -> None:
        h = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]], dtype=torch.float32)
        delta = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]], dtype=torch.float32)
        result = diagnose(h, delta, reconstructed_gradient(h, delta))
        per_token = result["per_token"]
        self.assertEqual(len(per_token), h.shape[1])
        self.assertEqual([item["position"] for item in per_token], [0, 1, 2])
        self.assertEqual(per_token[-1]["token_id"], 12)
        self.assertTrue(per_token[-1]["is_eos"])


if __name__ == "__main__":
    unittest.main()
