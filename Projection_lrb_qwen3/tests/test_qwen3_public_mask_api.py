from __future__ import annotations

import unittest

import torch
try:
    from transformers import Qwen3Config, Qwen3ForSequenceClassification
except ImportError:
    Qwen3Config = None
    Qwen3ForSequenceClassification = None

from Projection_lrb_qwen3.src.dager_qwen3.model_adapter import (
    Qwen3RoPEDagerAdapter,
)


class _Tokenizer:
    eos_token_id = 31


class Qwen3PublicMaskApiTest(unittest.TestCase):
    @unittest.skipUnless(Qwen3Config is not None, "transformers with Qwen3 support is required")
    def test_layer1_input_matches_native_qwen3_forward(self) -> None:
        torch.manual_seed(20260728)

        config = Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=32,
            use_cache=False,
            layer_types=[
                "full_attention",
                "full_attention",
            ],
            pad_token_id=0,
            eos_token_id=31,
            num_labels=2,
        )
        config._attn_implementation = "eager"

        model = Qwen3ForSequenceClassification(config)
        model.eval()

        self.assertFalse(
            callable(
                getattr(model.model, "_update_causal_mask", None)
            )
        )

        adapter = Qwen3RoPEDagerAdapter(
            model=model,
            tokenizer=_Tokenizer(),
        )

        input_ids = torch.tensor(
            [
                [3, 4, 5, 31],
                [6, 7, 8, 31],
            ],
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)

        captured: list[torch.Tensor] = []

        def capture_q1_input(
            _module: torch.nn.Module,
            args: tuple[torch.Tensor, ...],
        ) -> None:
            captured.append(args[0].detach().clone())

        handle = (
            model.model.layers[1]
            .self_attn.q_proj
            .register_forward_pre_hook(capture_q1_input)
        )
        try:
            with torch.no_grad():
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
        finally:
            handle.remove()

        self.assertEqual(len(captured), 1)

        with torch.no_grad():
            actual = adapter.layer1_qproj_inputs_from_prefixes(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        expected = captured[0]

        self.assertEqual(actual.shape, expected.shape)
        torch.testing.assert_close(
            actual,
            expected,
            rtol=1e-5,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
