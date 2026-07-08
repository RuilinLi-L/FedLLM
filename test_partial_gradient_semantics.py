#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.partial_gradient import (
    PARTIAL_ATTACK_DAGER_NONPREFIX,
    PARTIAL_ATTACK_DAGER_PREFIX,
    PARTIAL_ATTACK_DAGER_QKV,
    PARTIAL_ATTACK_LORA_ADAPTER,
    PARTIAL_ATTACK_UNSUPPORTED_NONPREFIX,
    PARTIAL_ATTACK_UNSUPPORTED_PTG_ONLY,
    apply_partial_gradient_filter,
    infer_partial_attack_variant,
    nonprefix_candidate_cap,
    nonprefix_layer_indices,
    non_prefix_dager_block_ids,
    partial_gradient_unsupported_reason,
    partial_gradient_active,
    partial_gradient_summary_fields,
    select_visible_matrix_candidates,
    supports_nonprefix_dager,
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
        model_path="gpt2",
        train_method="full",
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
    assert_true(
        infer_partial_attack_variant(_args("first2")) == PARTIAL_ATTACK_DAGER_PREFIX,
        "first2 should be classified as prefix-visible DAGER",
    )


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
    assert_true(
        infer_partial_attack_variant(_args("last2")) == PARTIAL_ATTACK_DAGER_NONPREFIX,
        "GPT-2 full last2 should be classified as non-prefix DAGER",
    )


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
    assert_true(
        infer_partial_attack_variant(_args(param_filter="qkv_only")) == PARTIAL_ATTACK_DAGER_QKV,
        "qkv_only should be classified as QKV-visible DAGER",
    )


def test_split_projection_filters_match_bert_and_llama_modules():
    names = [
        "bert.encoder.layer.0.attention.self.query.weight",
        "bert.encoder.layer.0.attention.self.key.weight",
        "bert.encoder.layer.0.attention.self.value.weight",
        "bert.encoder.layer.0.attention.output.dense.weight",
        "bert.encoder.layer.0.intermediate.dense.weight",
        "bert.encoder.layer.0.output.dense.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "classifier.weight",
    ]

    query_kept = [
        idx for idx, grad in enumerate(apply_partial_gradient_filter(_grads(len(names)), _args(param_filter="query_only"), names))
        if grad is not None
    ]
    key_kept = [
        idx for idx, grad in enumerate(apply_partial_gradient_filter(_grads(len(names)), _args(param_filter="key_only"), names))
        if grad is not None
    ]
    value_kept = [
        idx for idx, grad in enumerate(apply_partial_gradient_filter(_grads(len(names)), _args(param_filter="value_only"), names))
        if grad is not None
    ]
    attn_out_kept = [
        idx for idx, grad in enumerate(apply_partial_gradient_filter(_grads(len(names)), _args(param_filter="attn_out_only"), names))
        if grad is not None
    ]
    ffn_in_kept = [
        idx for idx, grad in enumerate(apply_partial_gradient_filter(_grads(len(names)), _args(param_filter="ffn_in_only"), names))
        if grad is not None
    ]
    ffn_out_kept = [
        idx for idx, grad in enumerate(apply_partial_gradient_filter(_grads(len(names)), _args(param_filter="ffn_out_only"), names))
        if grad is not None
    ]
    classifier_kept = [
        idx for idx, grad in enumerate(apply_partial_gradient_filter(_grads(len(names)), _args(param_filter="classifier_only"), names))
        if grad is not None
    ]

    assert_true(query_kept == [0, 6], f"query_only kept wrong indices: {query_kept}")
    assert_true(key_kept == [1, 7], f"key_only kept wrong indices: {key_kept}")
    assert_true(value_kept == [2, 8], f"value_only kept wrong indices: {value_kept}")
    assert_true(attn_out_kept == [3, 9], f"attn_out_only kept wrong indices: {attn_out_kept}")
    assert_true(ffn_in_kept == [4, 10], f"ffn_in_only kept wrong indices: {ffn_in_kept}")
    assert_true(ffn_out_kept == [5, 11], f"ffn_out_only kept wrong indices: {ffn_out_kept}")
    assert_true(classifier_kept == [12], f"classifier_only kept wrong indices: {classifier_kept}")


def test_gpt2_split_projection_filters_expose_packed_c_attn():
    names = [
        "transformer.h.0.attn.c_attn.weight",
        "transformer.h.0.attn.c_proj.weight",
        "transformer.h.0.mlp.c_fc.weight",
        "transformer.h.0.mlp.c_proj.weight",
    ]
    for param_filter in ("query_only", "key_only", "value_only"):
        args = _args(param_filter=param_filter)
        kept = [idx for idx, grad in enumerate(apply_partial_gradient_filter(_grads(len(names)), args, names)) if grad is not None]
        assert_true(kept == [0], f"{param_filter} should expose GPT-2 packed c_attn, got {kept}")
        fields = dict(partial_gradient_summary_fields(args))
        assert_true(
            fields["effective_gradient_param_filter"] == "qkv_only",
            f"{param_filter} should be summarized as effective qkv_only for GPT-2",
        )
        assert_true(
            fields["partial_attack_variant"] == PARTIAL_ATTACK_UNSUPPORTED_PTG_ONLY,
            f"{param_filter} should be explicit unsupported in the DAGER entrypoint",
        )


def test_ptg_only_filters_are_not_reported_as_full_gradient_dager():
    for param_filter in (
        "query_only",
        "key_only",
        "value_only",
        "attn_out_only",
        "ffn_in_only",
        "ffn_out_only",
        "ffn_only",
    ):
        args = _args(param_filter=param_filter)
        assert_true(
            infer_partial_attack_variant(args) == PARTIAL_ATTACK_UNSUPPORTED_PTG_ONLY,
            f"{param_filter} should be marked PTG-only for the DAGER entrypoint",
        )
        assert_true(
            partial_gradient_unsupported_reason(args) == "ptg_only_filter_requires_attack_partial_gradient",
            f"{param_filter} should point users to attack_partial_gradient.py",
        )


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
    assert_true(
        infer_partial_attack_variant(_args(param_filter="lora_only")) == PARTIAL_ATTACK_LORA_ADAPTER,
        "lora_only should be classified as LoRA adapter-visible",
    )


def test_lora_only_filter_includes_peft_adapter_families():
    names = [
        "base_model.model.bert.encoder.layer.0.attention.self.query.ia3_l.default",
        "prompt_encoder.default.embedding.weight",
        "base_model.model.bert.encoder.layer.0.output.adapters.default.adapter_down.0.weight",
        "base_model.model.classifier.modules_to_save.default.weight",
        "base_model.model.bert.encoder.layer.0.attention.self.query.weight",
    ]
    filtered = apply_partial_gradient_filter(_grads(len(names)), _args(param_filter="lora_only"), names)
    kept = [idx for idx, grad in enumerate(filtered) if grad is not None]
    assert_true(kept == [0, 1, 2], f"lora_only should now include IA3/Prefix/adapter tensors: {kept}")


def test_first2_qkv_is_and_filter():
    names = [
        "transformer.h.0.attn.c_attn.weight",
        "transformer.h.0.mlp.c_fc.weight",
        "transformer.h.1.attn.c_attn.weight",
        "transformer.h.2.attn.c_attn.weight",
        "score.weight",
    ]
    args = _args("first2", "qkv_only")
    filtered = apply_partial_gradient_filter(_grads(len(names)), args, names)
    kept = [idx for idx, grad in enumerate(filtered) if grad is not None]
    assert_true(kept == [0, 2], f"first2 + qkv_only should keep only first-two-layer QKV: {kept}")
    assert_true(
        args.partial_gradient_info["partial_attack_variant"] == PARTIAL_ATTACK_DAGER_QKV,
        "qkv_only should classify the combined exposure as QKV-visible",
    )


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


def test_summary_fields_include_partial_variant_and_dager_names():
    args = _args("first2")
    apply_partial_gradient_filter(
        _grads(2),
        args,
        ["transformer.h.0.attn.c_attn.weight", "transformer.h.1.attn.c_attn.weight"],
    )
    args.partial_gradient_info["dager_visible_param_names"] = "transformer.h.0.attn.c_attn.weight"
    fields = dict(partial_gradient_summary_fields(args))
    assert_true(fields["partial_attack_variant"] == PARTIAL_ATTACK_DAGER_PREFIX, "summary should expose variant")
    assert_true(
        fields["dager_visible_param_names"] == "transformer.h.0.attn.c_attn.weight",
        "summary should expose DAGER-visible parameter names",
    )


def test_invalid_layer_subset_fails_fast():
    try:
        validate_partial_gradient_args(_args("bogus2"))
    except ValueError as exc:
        assert_true("first2/last2/mid2" in str(exc), "invalid layer subset error should be explicit")
    else:
        raise AssertionError("invalid layer subset should fail validation")


def test_middle_layer_subset_is_supported_for_gpt2_nonprefix():
    args = _args("middle2")
    validate_partial_gradient_args(args)
    assert_true(
        infer_partial_attack_variant(args) == PARTIAL_ATTACK_DAGER_NONPREFIX,
        "GPT-2 full middle2 should parse as non-prefix DAGER",
    )


def test_nonprefix_support_is_gpt2_full_only():
    gpt2_args = _args("last2")
    gpt2_large_args = _args("last2")
    gpt2_large_args.model_path = "openai-community/gpt2-large"
    last1_args = _args("last1")
    bert_args = _args("last2")
    bert_args.model_path = "bert-base-uncased"
    peft_args = _args("last2")
    peft_args.train_method = "peft"

    assert_true(supports_nonprefix_dager(gpt2_args), "GPT-2 full last2 should support non-prefix DAGER")
    assert_true(supports_nonprefix_dager(gpt2_large_args), "GPT-2 large full last2 should support non-prefix DAGER")
    assert_true(not supports_nonprefix_dager(last1_args), "non-prefix DAGER should require two visible layers")
    assert_true(
        dict(partial_gradient_summary_fields(last1_args))["unsupported_reason"]
        == "nonprefix_dager_requires_at_least_two_visible_layers",
        "last1 should explain that non-prefix DAGER needs two visible layers",
    )
    assert_true(not supports_nonprefix_dager(bert_args), "BERT last2 should stay unsupported in v1")
    assert_true(not supports_nonprefix_dager(peft_args), "PEFT last2 should stay unsupported in v1")
    assert_true(
        infer_partial_attack_variant(bert_args) == PARTIAL_ATTACK_UNSUPPORTED_NONPREFIX,
        "BERT last2 should be explicitly unsupported",
    )
    assert_true(
        dict(partial_gradient_summary_fields(bert_args))["unsupported_reason"]
        == "nonprefix_layer_subset_requires_gpt2_full_decoder",
        "BERT last2 summary should explain the GPT-2 full-only non-prefix path",
    )
    assert_true(
        infer_partial_attack_variant(peft_args) == PARTIAL_ATTACK_UNSUPPORTED_NONPREFIX,
        "PEFT last2 should be explicitly unsupported",
    )
    assert_true(
        partial_gradient_unsupported_reason(peft_args) == "nonprefix_layer_subset_requires_gpt2_full_decoder",
        "PEFT last2 should explain that non-prefix is GPT-2 full-only",
    )


def test_nonprefix_support_rejects_llama_and_single_layer():
    llama_args = _args("mid2")
    llama_args.model_path = "meta-llama/Meta-Llama-3.1-8B"
    last1_args = _args("last1")

    assert_true(not supports_nonprefix_dager(llama_args), "Llama mid2 should stay unsupported in v1")
    assert_true(
        infer_partial_attack_variant(llama_args) == PARTIAL_ATTACK_UNSUPPORTED_NONPREFIX,
        "Llama mid2 should be explicitly unsupported",
    )
    assert_true(
        partial_gradient_unsupported_reason(llama_args) == "nonprefix_layer_subset_requires_gpt2_full_decoder",
        "Llama mid2 should explain the GPT-2 full-only non-prefix path",
    )
    assert_true(
        partial_gradient_unsupported_reason(last1_args) == "nonprefix_dager_requires_at_least_two_visible_layers",
        "last1 should fail fast as a single-layer non-prefix exposure",
    )


def test_peft_adapter_only_variant_is_for_supported_peft_methods():
    for peft_method in ("lora", "ia3", "adapter"):
        args = _args(param_filter="lora_only")
        args.train_method = "peft"
        args.peft_method = peft_method
        assert_true(
            infer_partial_attack_variant(args) == PARTIAL_ATTACK_LORA_ADAPTER,
            f"{peft_method} adapter-only exposure should use the PEFT adapter-visible variant",
        )


def test_nonprefix_attack_helpers_read_cap_and_layers():
    args = SimpleNamespace(
        max_ids=-1,
        partial_nonprefix_candidate_cap=16,
        partial_nonprefix_layer_indices=[10, 11],
    )
    assert_true(nonprefix_candidate_cap(args) == 16, "default non-prefix cap should come from the flag")
    args.max_ids = 4
    assert_true(nonprefix_candidate_cap(args) == 4, "max_ids should override the non-prefix cap")
    assert_true(nonprefix_layer_indices(args) == [10, 11], "layer indices should be read from partial metadata")

    args = SimpleNamespace(dager_selected_block_ids=[8, 9])
    assert_true(nonprefix_layer_indices(args) == [8, 9], "selected block ids should be the fallback")


def main():
    tests = [
        test_default_filter_is_inactive_and_preserves_tuple,
        test_default_filter_preserves_list_identity,
        test_first2_keeps_first_two_transformer_blocks,
        test_last2_keeps_last_two_transformer_blocks,
        test_qkv_only_matches_supported_model_families,
        test_split_projection_filters_match_bert_and_llama_modules,
        test_gpt2_split_projection_filters_expose_packed_c_attn,
        test_ptg_only_filters_are_not_reported_as_full_gradient_dager,
        test_lora_only_excludes_modules_to_save_heads,
        test_lora_only_filter_includes_peft_adapter_families,
        test_first2_qkv_is_and_filter,
        test_visible_matrix_candidates_skip_hidden_and_vector_tensors,
        test_non_prefix_dager_candidates_are_detected,
        test_summary_fields_include_partial_variant_and_dager_names,
        test_invalid_layer_subset_fails_fast,
        test_middle_layer_subset_is_supported_for_gpt2_nonprefix,
        test_nonprefix_support_is_gpt2_full_only,
        test_nonprefix_support_rejects_llama_and_single_layer,
        test_peft_adapter_only_variant_is_for_supported_peft_methods,
        test_nonprefix_attack_helpers_read_cap_and_layers,
    ]
    for test in tests:
        print(f"Running {test.__name__}...")
        test()
    print("All partial-gradient semantic tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
