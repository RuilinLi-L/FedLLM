"""Native Qwen3/RoPE prefix adapter equality test without external model files."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

import torch

from src.dager_qwen3.model_adapter import Qwen3RoPEDagerAdapter
from test_qwen3_layer1_true_token import TinyQwenClassifier, TinyTokenizer


class Qwen3PrefixForwardTest(unittest.TestCase):
    def test_adapter_prefix_forward_equals_manually_captured_layer1_qproj_input(self) -> None:
        model = TinyQwenClassifier()
        adapter = Qwen3RoPEDagerAdapter(model, TinyTokenizer())
        input_ids = torch.tensor([[3, 4, 7]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        captured = adapter.capture_layer1_qproj_input_from_model_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        generated = adapter.layer1_qproj_inputs_from_prefixes(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        self.assertTrue(torch.allclose(captured, generated, rtol=0.0, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
