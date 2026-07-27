"""Model-independent true-token first-layer candidate checks for Qwen3 DAGER."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn

from src.dager_qwen3.gradient_decomposition import decompose_qwen3_qproj_gradient
from src.dager_qwen3.layer1_filter import filter_qwen3_vocab_layer1, span_distances
from src.dager_qwen3.model_adapter import Qwen3RoPEDagerAdapter


class TinyRMSNorm(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states / hidden_states.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()


class TinyRotary(nn.Module):
    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        del position_ids
        return torch.ones_like(hidden_states), torch.zeros_like(hidden_states)


class TinyAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(hidden_size))


class TinyLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.input_layernorm = TinyRMSNorm()
        self.self_attn = TinyAttention(hidden_size)

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor]:
        del attention_mask, position_ids, cache_position, position_embeddings, output_attentions
        q_input = self.input_layernorm(hidden_states)
        return (hidden_states + 0.125 * self.self_attn.q_proj(q_input),)


class TinyBackbone(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([TinyLayer(hidden_size), TinyLayer(hidden_size)])
        self.rotary_emb = TinyRotary()

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: object | None,
        output_attentions: bool,
    ) -> torch.Tensor:
        del input_tensor, cache_position, past_key_values, output_attentions
        return attention_mask

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        del use_cache
        hidden = self.embed_tokens(input_ids)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
        cache = torch.arange(input_ids.shape[1], device=input_ids.device)
        rope = self.rotary_emb(hidden, positions)
        causal = self._update_causal_mask(attention_mask, hidden, cache, None, False)
        for layer in self.layers:
            hidden = layer(
                hidden_states=hidden,
                attention_mask=causal,
                position_ids=positions,
                cache_position=cache,
                position_embeddings=rope,
                output_attentions=False,
            )[0]
        return hidden


class TinyQwenClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hidden_size = 3
        self.config = SimpleNamespace(hidden_size=hidden_size, num_attention_heads=1, head_dim=hidden_size)
        self.model = TinyBackbone(hidden_size=hidden_size, vocab_size=8)
        self.score = nn.Linear(hidden_size, 2, bias=False)
        with torch.no_grad():
            self.model.embed_tokens.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0],
                        [1.0, 0.0, 1.0],
                        [-1.0, 1.0, 0.5],
                        [0.25, -0.5, 1.0],
                    ]
                )
            )

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        return self.score(self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache))


class TinyTokenizer:
    eos_token_id = 7


class Qwen3Layer1TrueTokenTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = TinyQwenClassifier()
        self.adapter = Qwen3RoPEDagerAdapter(self.model, TinyTokenizer())

    def test_true_qproj_input_is_closer_than_most_random_tokens_and_enters_wide_filter(self) -> None:
        true_id = 3
        true_representation = self.adapter.layer0_qproj_inputs_for_token_ids(torch.tensor([true_id]))
        delta = torch.tensor([[[1.0, -2.0, 0.5]]])
        gradient = delta.reshape(-1, 3).transpose(0, 1) @ true_representation
        span = decompose_qwen3_qproj_gradient(
            gradient,
            feature_dim=3,
            rank_tolerance=1e-6,
            rank_cutoff=0,
            decomposition_device=torch.device("cpu"),
        )
        all_representations = self.adapter.layer0_qproj_inputs_for_token_ids(torch.arange(8, dtype=torch.long))
        distances = span_distances(basis=span.basis, representations=all_representations, norm="l2")
        self.assertLess(float(distances[true_id]), float(torch.quantile(distances, 0.75)))
        result = filter_qwen3_vocab_layer1(
            adapter=self.adapter,
            span=span,
            threshold=1e-3,
            vocab_chunk_size=3,
            distance_norm="l2",
        )
        self.assertIn(true_id, result.token_ids)
        self.assertEqual(sum(chunk.end_token_id_exclusive - chunk.start_token_id for chunk in result.chunk_diagnostics), 8)


if __name__ == "__main__":
    unittest.main()
