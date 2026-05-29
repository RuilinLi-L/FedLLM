#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.filtering_decoder import _decoder_candidate_chunk_size


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def test_noisy_decoder_chunk_respects_parallel_when_candidate_set_is_large():
    args = SimpleNamespace(
        parallel=1000,
        defense="dpsgd",
        defense_noise=1e-6,
        defense_adaptive_decoding=False,
    )

    chunk_size = _decoder_candidate_chunk_size(args, n_ends=10048)

    assert_true(
        chunk_size == 1000,
        "DPSGD/noisy decoder chunks should not expand to the full candidate-token count.",
    )


def test_clean_decoder_keeps_complete_prefix_groups():
    args = SimpleNamespace(
        parallel=1000,
        defense="none",
        defense_noise=None,
        defense_adaptive_decoding=False,
    )

    chunk_size = _decoder_candidate_chunk_size(args, n_ends=64)

    assert_true(
        chunk_size == 960,
        "Clean decoder chunks should remain whole candidate-token groups for prefix completion checks.",
    )
