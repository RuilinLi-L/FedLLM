#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.partial_gradient import (
    apply_partial_gradient_filter,
    non_prefix_dager_block_ids,
    partial_gradient_active,
    select_visible_matrix_candidates,
    validate_partial_gradient_args,
)


class FakeTensor:
    def __init__(self, ndim):
        self.ndim = ndim


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def _args(layer_subset="all", param_filter="all"):
    return SimpleNamespace(
        gradient_layer_subset=layer_subset,
        gradient_param_filter=param_filter,
    )


def _grads(n):
    return tuple(FakeTensor(2) for _ in range(n))


def test_default_filter_is_inactive_and_preserves_tuple():
    grads = _grads(2)
    args = _args()
    filtered = apply_partial_gradient_filter(
        grads,
        args,
        ["transformer.h.0.attn.c_attn.weight", "transformer.h.1.attn.c_attn.weight"],
    )
    assert_true(not partial_gradient_active(args), "default all/all should be inactive")
    assert_true(filtered is grads, "inactive filter should preserve the original gradient tuple")
    assert_true(args.partial_gradient_info["partial_filter_active"] is False, "summary should mark inactive")


def test_default_filter_preserves_list_identity():
    grads = list(_grads(2))
    args = _args()
    filtered = apply_partial_gradient_filter(
        grads,
        args,
        ["transformer.h.0.attn.c_attn.weight", "transformer.h.1.attn.c_attn.weight"],
    )
    assert_true(filtered is grads, "inactive filter should preserve non-tuple gradient containers too")


def test_first2_keeps_first_two_transformer_blocks():
    names = [
        "transformer.h.0.attn.c_attn.weight",
        "transformer.h.0.mlp.c_fc.weight",
        "transformer.h.1.attn.c_attn.weight",
        "transformer.h.2.attn.c_attn.weight",
        "score.weight",
    ]
    filtered = apply_partial_gradient_filter(_grads(len(names)), _args("first2"), names)
    kept = [idx for idx, grad in enumerate(filtered) if grad is not None]
    assert_true(kept == [0, 1, 2], f"first2 kept wrong indices: {kept}")


def test_last2_keeps_last_two_transformer_blocks():
    names = [
        "transformer.h.0.attn.c_attn.weight",
        "transformer.h.1.attn.c_attn.weight",
        "transformer.h.2.attn.c_attn.weight",
        "transformer.h.3.attn.c_attn.weight",
        "score.weight",
    ]
    filtered = apply_partial_gradient_filter(_grads(len(names)), _args("last2"), names)
    kept = [idx for idx, grad in enumerate(filtered) if grad is not None]
    assert_true(kept == [2, 3], f"last2 kept wrong indices: {kept}")


def test_qkv_only_matches_supported_model_families():
    names = [
        "transformer.h.0.attn.c_attn.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "bert.encoder.layer.0.attention.self.query.weight",
        "bert.encoder.layer.0.attention.self.key.weight",
        "bert.encoder.layer.0.attention.self.value.weight",
        "transformer.h.0.mlp.c_fc.weight",
    ]
    filtered = apply_partial_gradient_filter(_grads(len(names)), _args(param_filter="qkv_only"), names)
    kept = [idx for idx, grad in enumerate(filtered) if grad is not None]
    assert_true(kept == list(range(7)), f"qkv_only kept wrong indices: {kept}")


def test_lora_only_excludes_modules_to_save_heads():
    names = [
        "base_model.model.transformer.h.0.attn.c_attn.lora_A.default.weight",
        "base_model.model.transformer.h.0.attn.c_attn.lora_B.default.weight",
        "base_model.model.transformer.h.0.attn.c_attn.lora_embedding_A.default.weight",
        "base_model.model.score.modules_to_save.default.weight",
        "base_model.model.transformer.h.0.mlp.c_fc.weight",
    ]
    filtered = apply_partial_gradient_filter(_grads(len(names)), _args(param_filter="lora_only"), names)
    kept = [idx for idx, grad in enumerate(filtered) if grad is not None]
    assert_true(kept == [0, 1], f"lora_only kept wrong indices: {kept}")


def test_visible_matrix_candidates_skip_hidden_and_vector_tensors():
    grads = (
        None,
        FakeTensor(1),
        FakeTensor(2),
        FakeTensor(2),
    )
    indices, names, skipped = select_visible_matrix_candidates(
        grads,
        [0, 1, 2, 3],
        ["hidden", "vector", "matrix_a", "matrix_b"],
        2,
    )
    assert_true(indices == [2, 3], f"selected wrong matrix candidates: {indices}")
    assert_true(names == ["matrix_a", "matrix_b"], f"selected wrong names: {names}")
    assert_true(len(skipped) == 2, "hidden/vector candidates should be recorded as skipped")


def test_non_prefix_dager_candidates_are_detected():
    first_two = [
        "transformer.h.0.attn.c_attn.weight",
        "transformer.h.1.attn.c_attn.weight",
    ]
    last_two = [
        "transformer.h.10.attn.c_attn.weight",
        "transformer.h.11.attn.c_attn.weight",
    ]
    unknown = [
        "score.weight",
        "classifier.bias",
    ]
    assert_true(non_prefix_dager_block_ids(first_two, 2) is None, "first two blocks should be accepted")
    assert_true(non_prefix_dager_block_ids(last_two, 2) == [10, 11], "last blocks should be flagged")
    assert_true(non_prefix_dager_block_ids(unknown, 2) is None, "unknown names should not trigger layer mismatch")


def test_invalid_layer_subset_fails_fast():
    try:
        validate_partial_gradient_args(_args("middle2"))
    except ValueError as exc:
        assert_true("first2/last2" in str(exc), "invalid layer subset error should be explicit")
    else:
        raise AssertionError("invalid layer subset should fail validation")


def main():
    tests = [
        test_default_filter_is_inactive_and_preserves_tuple,
        test_default_filter_preserves_list_identity,
        test_first2_keeps_first_two_transformer_blocks,
        test_last2_keeps_last_two_transformer_blocks,
        test_qkv_only_matches_supported_model_families,
        test_lora_only_excludes_modules_to_save_heads,
        test_visible_matrix_candidates_skip_hidden_and_vector_tensors,
        test_non_prefix_dager_candidates_are_detected,
        test_invalid_layer_subset_fails_fast,
    ]
    for test in tests:
        print(f"Running {test.__name__}...")
        test()
    print("All partial-gradient semantic tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
