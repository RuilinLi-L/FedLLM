"""Model-free tests for canonical Qwen3 q_proj location and manifest ordering."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn

from src.gradient_capture import build_canonical_gradient_manifest, resolve_first_two_q_projections
from src.qwen3_classifier import _initialize_head


class DummyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)


class DummyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = DummyAttention()


class DummyQwenClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([DummyLayer(), DummyLayer()])
        self.score = nn.Linear(4, 2, bias=False)
        self.config = SimpleNamespace(hidden_size=4, num_attention_heads=2, head_dim=2)


class ParameterManifestTests(unittest.TestCase):
    def test_structural_q_projection_resolution_and_config_shape(self) -> None:
        model = DummyQwenClassifier()
        pair = resolve_first_two_q_projections(model)
        self.assertEqual(pair.parameter_names, (
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.1.self_attn.q_proj.weight",
        ))
        self.assertEqual(pair.expected_weight_shape, (4, 4))
        self.assertIs(pair.modules[0].weight, dict(model.named_parameters())[pair.parameter_names[0]])
        self.assertIs(pair.modules[1].weight, dict(model.named_parameters())[pair.parameter_names[1]])

    def test_manifest_uses_current_named_parameter_order(self) -> None:
        model = DummyQwenClassifier()
        named = list(model.named_parameters())
        named[0][1].grad = torch.ones_like(named[0][1])
        manifest = build_canonical_gradient_manifest(model)
        self.assertEqual([entry["name"] for entry in manifest["entries"]], [name for name, _ in named])
        self.assertEqual([entry["canonical_index"] for entry in manifest["entries"]], list(range(len(named))))
        self.assertTrue(manifest["entries"][0]["grad_present"])
        self.assertEqual(manifest["gradient_tensor_count"], 1)
        self.assertEqual(manifest["gradient_numel"], named[0][1].numel())

    def test_explicit_head_seed_initializes_only_score_deterministically(self) -> None:
        first = DummyQwenClassifier()
        second = DummyQwenClassifier()
        first_backbone = first.model.layers[0].self_attn.q_proj.weight.detach().clone()
        second_backbone = second.model.layers[0].self_attn.q_proj.weight.detach().clone()
        names_a = _initialize_head(first, head_seed=404)
        names_b = _initialize_head(second, head_seed=404)
        self.assertEqual(names_a, names_b)
        self.assertEqual(names_a, ("score.weight",))
        self.assertTrue(torch.equal(first.score.weight, second.score.weight))
        self.assertTrue(torch.equal(first.model.layers[0].self_attn.q_proj.weight, first_backbone))
        self.assertTrue(torch.equal(second.model.layers[0].self_attn.q_proj.weight, second_backbone))


if __name__ == "__main__":
    unittest.main()
