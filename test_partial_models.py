#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

import torch
from transformers import GPT2Config, GPT2Model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.partial_models import add_partial_forward_gpt2


class CachelessGPT2Block(torch.nn.Module):
    def __init__(self, original_block):
        super().__init__()
        self.ln_1 = original_block.ln_1

    def forward(self, hidden_states, **kwargs):
        return (hidden_states,)


def test_gpt2_partial_hidden_states_do_not_require_cache_outputs():
    config = GPT2Config(
        vocab_size=16,
        n_positions=8,
        n_ctx=8,
        n_embd=8,
        n_layer=2,
        n_head=1,
        use_cache=True,
    )
    model = GPT2Model(config)
    model.h[0] = CachelessGPT2Block(model.h[0])
    add_partial_forward_gpt2(model)

    input_ids = torch.tensor([[1, 2, 3]])
    hidden_states = model.get_hidden_states(input_ids=input_ids, n_layers=1)

    assert len(hidden_states) == 1
    assert tuple(hidden_states[0].shape) == (1, 3, 8)


if __name__ == "__main__":
    test_gpt2_partial_hidden_states_do_not_require_cache_outputs()
    print("[PASS] partial GPT-2 hidden-state cache compatibility")
