#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from unittest.mock import patch

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attacks.partial_transformer_gradients import (
    PTG_GRADIENT_MATCHING_VARIANT,
    _fixed_token_mask,
    _gradient_match_loss,
    filter_partial_transformer_gradients,
    optimize_partial_text_embeddings,
    parse_ptg_attack_layers,
    ptg_filter_active,
    ptg_selector_summary_fields,
    ptg_swap_objective,
    selected_partial_gradient_tensors,
)
from attack_partial_gradient import (
    _capture_source_opacus_grads,
    _prediction_match_order,
    _resolve_ignored_token_ids,
    _validate_args,
    build_parser,
)
from attack_peftleak import compute_text_metrics, validate_text_metric_backend


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


class TinyTokenizer:
    vocab_size = 8
    pad_token_id = 0
    eos_token_id = None
    bos_token_id = None
    cls_token_id = None
    sep_token_id = None

    def decode(self, ids, skip_special_tokens=True):
        toks = [str(int(tok)) for tok in ids if not skip_special_tokens or int(tok) != 0]
        return " ".join(toks)


class TinyPTGModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"num_labels": 2})()
        self.embedding = torch.nn.Embedding(8, 4)
        self.query = torch.nn.Linear(4, 4, bias=False)
        self.ffn = torch.nn.Linear(4, 4, bias=False)
        self.classifier = torch.nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.2, -0.1, 0.3],
                        [0.5, -0.4, 0.8, 0.1],
                        [-0.3, 0.7, 0.2, -0.6],
                        [0.9, -0.2, -0.5, 0.4],
                        [-0.8, 0.1, 0.3, 0.9],
                        [0.2, 0.4, 0.6, -0.7],
                        [-0.6, 0.3, -0.2, 0.5],
                    ],
                    dtype=torch.float32,
                )
            )
            self.query.weight.copy_(torch.eye(4))
            self.ffn.weight.copy_(torch.tensor([[0.7, -0.2, 0.1, 0.3], [0.2, 0.6, -0.4, 0.1], [0.0, 0.5, 0.8, -0.3], [-0.2, 0.1, 0.4, 0.9]]))
            self.classifier.weight.copy_(torch.tensor([[0.4, -0.1, 0.3, 0.2], [-0.2, 0.5, -0.3, 0.6]]))

    def get_input_embeddings(self):
        return self.embedding


class TinyPTGWrapper:
    def __init__(self):
        self.model = TinyPTGModel()
        self.base_model = self.model
        self.args = type("Args", (), {"rng_seed": 13})()
        self.tokenizer = TinyTokenizer()
        self.pad_token = 0

    def _seq_class_input_embeds(self, batch):
        return self.model.embedding(batch["input_ids"])

    def _seq_class_logits_from_embeds(self, batch, inputs_embeds, representation_mask=None):
        hidden = torch.tanh(self.model.query(inputs_embeds))
        hidden = torch.tanh(self.model.ffn(hidden))
        representation = hidden.mean(dim=1)
        if representation_mask is not None:
            representation = representation * representation_mask
        return self.model.classifier(representation), representation

    def trainable_parameters(self):
        return (self.model.query.weight, self.model.ffn.weight, self.model.classifier.weight)

    def trainable_parameter_names(self):
        return [
            "bert.encoder.layer.0.attention.self.query.weight",
            "bert.encoder.layer.0.intermediate.dense.weight",
            "classifier.weight",
        ]


class TinySourceBertModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"num_labels": 2})()
        self.embedding = torch.nn.Embedding(8, 4)
        self.query = torch.nn.Linear(4, 4, bias=False)
        self.classifier = torch.nn.Linear(4, 2, bias=False)
        self.inputs_embeds_calls = 0
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.2, -0.1, 0.3],
                        [0.5, -0.4, 0.8, 0.1],
                        [-0.3, 0.7, 0.2, -0.6],
                        [0.9, -0.2, -0.5, 0.4],
                        [-0.8, 0.1, 0.3, 0.9],
                        [0.2, 0.4, 0.6, -0.7],
                        [-0.6, 0.3, -0.2, 0.5],
                    ],
                    dtype=torch.float32,
                )
            )
            self.query.weight.copy_(torch.eye(4))
            self.classifier.weight.copy_(torch.tensor([[0.4, -0.1, 0.3, 0.2], [-0.2, 0.5, -0.3, 0.6]]))

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, inputs_embeds=None, labels=None, **_kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        else:
            self.inputs_embeds_calls += 1
        hidden = torch.tanh(self.query(inputs_embeds))
        logits = self.classifier(hidden.mean(dim=1))
        loss = F.cross_entropy(logits, labels.view(-1).long()) if labels is not None else None
        return type("TinySourceBertOut", (), {"loss": loss, "logits": logits})()


class TinySourceBertWrapper:
    def __init__(self):
        self.model = TinySourceBertModel()
        self.base_model = self.model
        self.args = type("Args", (), {"rng_seed": 17, "model_path": "bert-base-uncased"})()
        self.tokenizer = TinyTokenizer()
        self.pad_token = 0
        self.seq_embed_calls = 0

    def _seq_class_input_embeds(self, batch):
        self.seq_embed_calls += 1
        return self.model.embedding(batch["input_ids"]) + 100.0

    def _seq_class_logits_from_embeds(self, batch, inputs_embeds, representation_mask=None):
        hidden = torch.tanh(self.model.query(inputs_embeds + 100.0))
        representation = hidden.mean(dim=1)
        return self.model.classifier(representation), representation

    def trainable_parameters(self):
        return (self.model.query.weight, self.model.classifier.weight)

    def trainable_parameter_names(self):
        return [
            "bert.encoder.layer.0.attention.self.query.weight",
            "classifier.weight",
        ]


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        value = input_ids.float().mean() * 0.0 + self.anchor * 0.0 + torch.tensor(0.5, device=input_ids.device)
        return type("TinyLMOut", (), {"loss": value})()


def target_grads(wrapper, batch, labels):
    embeds = wrapper._seq_class_input_embeds(batch)
    logits, _ = wrapper._seq_class_logits_from_embeds(batch, embeds)
    loss = F.cross_entropy(logits, labels)
    return torch.autograd.grad(loss, wrapper.trainable_parameters(), allow_unused=True)


def test_selected_partial_gradient_tensors_ignores_hidden_gradients():
    grads = (torch.ones(2, 2), None, torch.ones(1))
    indices, names = selected_partial_gradient_tensors(grads, ["a", "b", "c"])
    assert_true(indices == [0, 2], f"selected wrong indices: {indices}")
    assert_true(names == ["a", "c"], f"selected wrong names: {names}")


def test_ptg_selector_does_not_emit_dager_variant_or_reason():
    grads = (torch.ones(2, 2), torch.ones(2, 2), torch.ones(2, 2))
    names = [
        "bert.encoder.layer.0.attention.self.query.weight",
        "bert.encoder.layer.1.intermediate.dense.weight",
        "classifier.weight",
    ]
    partial, info = filter_partial_transformer_gradients(
        grads,
        parameter_names=names,
        layer_subset="last1",
        param_filter="ffn_in_only",
        model_path="bert-base-uncased",
    )

    assert_true([idx for idx, grad in enumerate(partial) if grad is not None] == [1], "PTG selector should keep BERT FFN layer")
    fields = dict(ptg_selector_summary_fields(type("Args", (), {"ptg_gradient_info": info})()))
    assert_true(fields["unsupported_reason"] == "n/a", "PTG selector should not inherit DAGER unsupported reasons")
    assert_true("dager" not in str(info.get("partial_attack_variant", "")), "PTG selector should not emit DAGER variants")


def test_source_grad_type_selectors_match_official_names():
    grads = tuple(torch.ones(2, 2) * idx for idx in range(16))
    names = [
        "bert.embeddings.word_embeddings.weight",
        "bert.encoder.layer.0.attention.self.query.weight",
        "bert.encoder.layer.0.attention.self.query.bias",
        "bert.encoder.layer.0.attention.self.key.weight",
        "bert.encoder.layer.0.attention.self.key.bias",
        "bert.encoder.layer.0.attention.self.value.weight",
        "bert.encoder.layer.0.attention.self.value.bias",
        "bert.encoder.layer.0.attention.output.dense.weight",
        "bert.encoder.layer.0.attention.output.dense.bias",
        "bert.encoder.layer.0.intermediate.dense.weight",
        "bert.encoder.layer.0.intermediate.dense.bias",
        "bert.encoder.layer.0.output.dense.weight",
        "bert.encoder.layer.0.output.dense.bias",
        "bert.encoder.layer.1.attention.self.query.weight",
        "classifier.weight",
        "bert.pooler.dense.weight",
    ]
    cases = {
        "word_emb": [0],
        "attn_query": [1],
        "attn_key": [3],
        "attn_value": [5],
        "attn_qkv": [1, 3, 5],
        "attn_output": [7],
        "ffn_fc": [9],
        "ffn_output": [11],
        "layer_encoder": list(range(1, 13)),
    }
    for grad_type, expected in cases.items():
        partial, info = filter_partial_transformer_gradients(
            grads,
            parameter_names=names,
            source_grad_type=grad_type,
            source_attack_layers=[0],
        )
        kept = [idx for idx, grad in enumerate(partial) if grad is not None]
        assert_true(kept == expected, f"{grad_type} selected {kept}, expected {expected}")
        assert_true(info["grad_type"] == grad_type, "source grad_type should be summarized")
        assert_true(info["attack_layer"] == "0", "source attack_layer should be summarized")


def test_ptg_use_embedding_dynamic_unused_tokens_from_visible_embedding_grad():
    wrapper = TinyPTGWrapper()
    args = type("Args", (), {"ptg_use_embedding": True})()
    embedding_grad = torch.zeros(8, 4)
    embedding_grad[1].fill_(1.0)
    embedding_grad[3].fill_(0.5)
    grads = (embedding_grad, torch.ones(2, 2))
    names = [
        "bert.embeddings.word_embeddings.weight",
        "bert.encoder.layer.0.attention.self.query.weight",
    ]

    ignored = _resolve_ignored_token_ids(args, wrapper.tokenizer, wrapper, grads, names)

    assert_true(ignored == {2, 4, 5, 6, 7}, f"dynamic embedding unused tokens mismatch: {ignored}")
    assert_true(0 not in ignored, "dynamic embedding inference should keep PAD out of unused tokens")
    assert_true(args.ptg_use_embedding_status == "dynamic", "dynamic embedding status should be recorded")
    assert_true(args.ptg_unused_token_count == 5, "dynamic unused token count should be recorded")


def test_ptg_use_embedding_ignores_non_token_embedding_tables():
    wrapper = TinyPTGWrapper()
    args = type("Args", (), {"ptg_use_embedding": True})()
    position_grad = torch.zeros(8, 4)
    position_grad[1].fill_(1.0)
    grads = (position_grad,)
    names = ["bert.embeddings.position_embeddings.weight"]

    ignored = _resolve_ignored_token_ids(args, wrapper.tokenizer, wrapper, grads, names)

    assert_true(0 in ignored, "non-token embedding tables should fall back to fixed ignored tokens")
    assert_true(args.ptg_use_embedding_status == "fallback_no_embedding_grad", "position embeddings should not trigger dynamic status")


def test_ptg_use_embedding_scans_resized_embedding_rows():
    wrapper = TinyPTGWrapper()
    wrapper.model.embedding = torch.nn.Embedding(10, 4)
    args = type("Args", (), {"ptg_use_embedding": True})()
    embedding_grad = torch.zeros(10, 4)
    embedding_grad[1].fill_(1.0)
    embedding_grad[3].fill_(0.5)
    embedding_grad[9].fill_(0.25)
    grads = (embedding_grad,)
    names = ["bert.embeddings.word_embeddings.weight"]

    ignored = _resolve_ignored_token_ids(args, wrapper.tokenizer, wrapper, grads, names)

    assert_true(8 in ignored, "dynamic inference should use the model embedding row count, not tokenizer.vocab_size")
    assert_true(9 not in ignored, "nonzero resized embedding rows should remain decodable")
    assert_true(args.ptg_unused_token_count == 6, "resized dynamic unused token count should include row 8 and exclude PAD")


def test_ptg_use_embedding_falls_back_without_visible_embedding_grad():
    wrapper = TinyPTGWrapper()
    args = type("Args", (), {"ptg_use_embedding": True})()
    grads = (torch.ones(2, 2),)
    names = ["bert.encoder.layer.0.attention.self.query.weight"]

    ignored = _resolve_ignored_token_ids(args, wrapper.tokenizer, wrapper, grads, names)

    assert_true(0 in ignored, "fallback should use the fixed ignored-token set")
    assert_true(args.ptg_use_embedding_status == "fallback_no_embedding_grad", "fallback status should be recorded")
    assert_true(args.ptg_unused_token_count == 1, "fallback count should ignore non-token None values")


def test_ptg_use_embedding_cli_aliases_set_shared_flag():
    for flag in ("--use_embedding", "--ptg_use_embedding"):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--dataset",
                "sst2",
                "--split",
                "test",
                "--n_inputs",
                "2",
                "--finetuned_path",
                "dummy",
                flag,
            ]
        )
        _validate_args(args)
        assert_true(args.ptg_use_embedding is True, f"{flag} should enable ptg_use_embedding")
        assert_true(args.ptg_use_embedding_status == "not_evaluated", "status should be initialized before attack")


def test_strict_source_selector_rejects_packed_gpt2_attention():
    grads = (torch.ones(2, 2), torch.ones(2, 2))
    names = [
        "transformer.h.0.attn.c_attn.weight",
        "transformer.h.0.mlp.c_fc.weight",
    ]
    for grad_type in ("attn_query", "attn_key", "attn_value", "attn_qkv"):
        partial, _info = filter_partial_transformer_gradients(
            grads,
            parameter_names=names,
            model_path="gpt2",
            source_grad_type=grad_type,
            source_attack_layers=[0],
        )
        kept = [idx for idx, grad in enumerate(partial) if grad is not None]
        assert_true(kept == [], f"Strict source selectors should reject packed GPT-2 tensors, got {kept}")


def test_source_mode_defaults_to_partial_bert_and_requires_lm():
    parser = build_parser()
    common = [
        "--dataset",
        "sst2",
        "--split",
        "test",
        "--n_inputs",
        "2",
        "--finetuned_path",
        "dummy",
        "--ptg_parity_mode",
        "source",
    ]
    try:
        _validate_args(parser.parse_args(common))
    except ValueError as exc:
        assert_true("--ptg_lm_model_path" in str(exc), "source mode should require an LM for its nonzero default prior")
    else:
        raise AssertionError("source mode should reject a nonzero LM prior without an LM path")

    args = parser.parse_args(common + ["--ptg_lm_model_path", "dummy-lm"])
    _validate_args(args)
    assert_true(args.grad_type == "layer_encoder", "source PTG should default to a partial layer selector")
    assert_true(args.attack_layer == [0], "source PTG should default to layer 0")
    assert_true(ptg_filter_active(args), "source PTG default should be marked as partial")

    non_bert = parser.parse_args(common + ["--model_path", "gpt2", "--ptg_lm_model_path", "dummy-lm"])
    try:
        _validate_args(non_bert)
    except ValueError as exc:
        assert_true("bert-base-uncased" in str(exc), "source mode should reject non-BERT model families")
    else:
        raise AssertionError("source mode should reject GPT-2")


def test_ptg_swap_objective_uses_lm_only_for_source_mode():
    total = torch.tensor(7.0)
    grad = torch.tensor(2.0)
    lm = torch.tensor(3.0)
    source = ptg_swap_objective(
        total=total,
        grad_loss=grad,
        lm_loss=lm,
        source_bert_mode=True,
        lm_prior_weight=0.5,
    )
    fedllm = ptg_swap_objective(
        total=total,
        grad_loss=grad,
        lm_loss=lm,
        source_bert_mode=False,
        lm_prior_weight=0.5,
    )
    assert_true(torch.equal(source, torch.tensor(3.5)), "source swaps should use reconstruction plus weighted LM loss")
    assert_true(torch.equal(fedllm, total), "FedLLM swaps should preserve the optimization objective")


def test_prediction_match_order_aligns_token_and_text_metrics():
    order = _prediction_match_order(["second sample", "first sample"], ["first sample", "second sample"], "rouge1")
    assert_true(order == [1, 0], f"batch matching should align predictions to references, got {order}")


def test_ptg_source_loss_names_match_formulas():
    candidate = (torch.tensor([1.0, 2.0, 3.0]),)
    target = (torch.tensor([1.0, 0.0, 4.0]),)
    cos = _gradient_match_loss(
        candidate_grads=candidate,
        target_grads=target,
        selected_indices=[0],
        match_loss="cos",
        device=torch.device("cpu"),
    )
    expected_cos = 1.0 - (candidate[0] * target[0]).sum() / (candidate[0].norm() * target[0].norm())
    assert_true(torch.allclose(cos, expected_cos), "cos loss should match official source formula")

    dlg = _gradient_match_loss(
        candidate_grads=candidate,
        target_grads=target,
        selected_indices=[0],
        match_loss="dlg",
        device=torch.device("cpu"),
    )
    diff = candidate[0] - target[0]
    assert_true(torch.allclose(dlg, diff.square().sum()), "dlg loss should be squared L2 sum")

    tag = _gradient_match_loss(
        candidate_grads=candidate,
        target_grads=target,
        selected_indices=[0],
        match_loss="tag",
        tag_factor=0.25,
        device=torch.device("cpu"),
    )
    assert_true(torch.allclose(tag, diff.square().sum() + 0.25 * diff.abs().sum()), "tag loss should be L2 plus scaled L1")


def test_source_bert_parity_uses_raw_word_embeddings():
    torch.manual_seed(19)
    wrapper = TinySourceBertWrapper()
    batch = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}
    labels = torch.tensor([1], dtype=torch.long)
    raw_embeds = wrapper.model.get_input_embeddings()(batch["input_ids"])
    outputs = wrapper.model(inputs_embeds=raw_embeds, labels=labels)
    grads = torch.autograd.grad(outputs.loss, wrapper.trainable_parameters(), allow_unused=True)

    result = optimize_partial_text_embeddings(
        model_wrapper=wrapper,
        batch=batch,
        labels=labels,
        target_grads=grads,
        parameter_names=wrapper.trainable_parameter_names(),
        steps=1,
        lr=0.05,
        restarts=1,
        match_loss="cos",
        init_strategy="random",
        init_candidates=1,
        fix_special_tokens=False,
        parity_mode="source",
    )

    assert_true(wrapper.model.inputs_embeds_calls > 1, "source BERT parity should call model(inputs_embeds=...)")
    assert_true(wrapper.seq_embed_calls == 0, "source BERT parity must not use wrapper preprocessed embeddings")
    assert_true(result["parity_mode"] == "source", "source parity mode should be recorded")


def test_ptg_gradient_matching_loss_decreases_on_toy_model():
    torch.manual_seed(5)
    wrapper = TinyPTGWrapper()
    batch = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}
    labels = torch.tensor([1], dtype=torch.long)
    grads = target_grads(wrapper, batch, labels)
    partial, _info = filter_partial_transformer_gradients(
        grads,
        parameter_names=wrapper.trainable_parameter_names(),
        layer_subset="all",
        param_filter="query_only",
        model_path="bert-base-uncased",
    )

    result = optimize_partial_text_embeddings(
        model_wrapper=wrapper,
        batch=batch,
        labels=labels,
        target_grads=partial,
        parameter_names=wrapper.trainable_parameter_names(),
        steps=30,
        lr=0.12,
        restarts=1,
        match_loss="normalized_mse",
    )

    assert_true(result["loss_history"][-1] < result["loss_history"][0], "PTG matching loss should decrease")
    assert_true(abs(result["loss_reduction"] - (result["initial_loss"] - result["loss"])) < 1e-6, "loss reduction should match the best run")
    assert_true(result["selected_gradient_count"] == 1, "query_only should select one toy gradient")
    assert_true(len(result["predicted_ids"]) == 1, "PTG should decode one sequence")
    assert_true(all(0 <= tok < 8 for tok in result["predicted_ids"][0]), "decoded token ids should be valid")


def test_ptg_source_init_lm_prior_and_swaps_are_executable():
    torch.manual_seed(11)
    wrapper = TinyPTGWrapper()
    batch = {
        "input_ids": torch.tensor([[1, 2, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.long),
    }
    labels = torch.tensor([1], dtype=torch.long)
    grads = target_grads(wrapper, batch, labels)
    partial, _info = filter_partial_transformer_gradients(
        grads,
        parameter_names=wrapper.trainable_parameter_names(),
        source_grad_type="attn_query",
        source_attack_layers=[0],
    )

    result = optimize_partial_text_embeddings(
        model_wrapper=wrapper,
        batch=batch,
        labels=labels,
        target_grads=partial,
        parameter_names=wrapper.trainable_parameter_names(),
        steps=2,
        lr=0.05,
        restarts=1,
        match_loss="cos",
        init_strategy="random",
        init_candidates=3,
        init_size=1.4,
        init_permutation_trials=2,
        lm_prior_weight=0.1,
        lm_model=TinyLM(),
        swap_steps=2,
        use_swaps=True,
        swap_every=1,
        ignored_token_ids={0},
        fix_special_tokens=True,
        know_padding=True,
        parity_mode="source",
    )
    assert_true(result["init_strategy"] == "random", "source init strategy should be recorded")
    assert_true(result["init_candidates"] == 3, "source init candidate count should be recorded")
    assert_true(result["swap_steps"] == 2, "swap steps should be recorded")
    assert_true(result["lm_loss"] is not None, "LM prior should produce a logged LM loss")
    assert_true(result["predicted_ids"][0][2] == 0, "known padding should keep pad token fixed")


def test_ptg_label_search_reports_search_mode_and_decodes():
    torch.manual_seed(7)
    wrapper = TinyPTGWrapper()
    batch = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}
    labels = torch.tensor([1], dtype=torch.long)
    grads = target_grads(wrapper, batch, labels)
    partial, _info = filter_partial_transformer_gradients(
        grads,
        parameter_names=wrapper.trainable_parameter_names(),
        layer_subset="all",
        param_filter="ffn_in_only",
        model_path="bert-base-uncased",
    )

    result = optimize_partial_text_embeddings(
        model_wrapper=wrapper,
        batch=batch,
        labels=labels,
        target_grads=partial,
        parameter_names=wrapper.trainable_parameter_names(),
        steps=8,
        lr=0.1,
        label_mode="search",
        label_candidates=[0, 1],
        match_loss="cosine",
    )

    assert_true(result["label_mode"] == "search", f"expected label search, got {result['label_mode']}")
    assert_true(result["best_label"] is not None, "label search should report the best labels")
    assert_true(len(result["predicted_ids"][0]) == 3, "decoded sequence length should match the input")


def test_ptg_fixed_special_tokens_are_restored_in_decoding():
    torch.manual_seed(9)
    wrapper = TinyPTGWrapper()
    batch = {
        "input_ids": torch.tensor([[1, 2, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.long),
    }
    labels = torch.tensor([1], dtype=torch.long)
    grads = target_grads(wrapper, batch, labels)
    partial, _info = filter_partial_transformer_gradients(
        grads,
        parameter_names=wrapper.trainable_parameter_names(),
        layer_subset="all",
        param_filter="query_only",
        model_path="bert-base-uncased",
    )

    result = optimize_partial_text_embeddings(
        model_wrapper=wrapper,
        batch=batch,
        labels=labels,
        target_grads=partial,
        parameter_names=wrapper.trainable_parameter_names(),
        steps=3,
        lr=0.05,
        ignored_token_ids={0},
        reference_mask=[[1, 1, 0]],
        fix_special_tokens=True,
        embed_norm_weight=0.01,
    )

    assert_true(result["predicted_ids"][0][2] == 0, "fixed pad position should decode back to the reference id")
    assert_true(result["fixed_token_count"] == 1, "one pad position should be fixed")


def test_ptg_know_padding_false_does_not_fix_pad_position():
    wrapper = TinyPTGWrapper()
    batch = {
        "input_ids": torch.tensor([[1, 2, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.long),
    }
    mask = torch.zeros_like(batch["input_ids"], dtype=torch.bool)
    from attacks.partial_transformer_gradients import _fixed_token_mask

    fixed = _fixed_token_mask(
        batch,
        ignored_token_ids={0},
        vocab_size=8,
        fix_special_tokens=False,
        know_padding=False,
    )
    assert_true(torch.equal(fixed, mask), "unknown padding mode should not fix pad positions when special-token fixing is disabled")


def test_source_bert_padding_mask_matches_official_fix_special_tokens():
    batch = {
        "input_ids": torch.tensor([[101, 2000, 102, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
    }
    known = _fixed_token_mask(
        batch,
        ignored_token_ids={0, 101, 102},
        vocab_size=30522,
        fix_special_tokens=True,
        know_padding=True,
        source_bert_special_tokens=True,
        cls_token_id=101,
        sep_token_id=102,
        pad_token_id=0,
    )
    assert_true(
        torch.equal(known, torch.tensor([[True, False, True, True]])),
        f"known padding should fix CLS, SEP before pads, and pads; got {known}",
    )

    unknown = _fixed_token_mask(
        batch,
        ignored_token_ids={0, 101, 102},
        vocab_size=30522,
        fix_special_tokens=True,
        know_padding=False,
        source_bert_special_tokens=True,
        cls_token_id=101,
        sep_token_id=102,
        pad_token_id=0,
    )
    assert_true(
        torch.equal(unknown, torch.tensor([[True, False, False, True]])),
        f"unknown single-sample padding should fix only CLS and final SEP slot; got {unknown}",
    )


def test_source_attack_layer_parser():
    assert_true(parse_ptg_attack_layers("all") is None, "all should parse to None")
    assert_true(parse_ptg_attack_layers("0,2,5") == [0, 2, 5], "comma layer list should parse")


def test_source_opacus_capture_uses_parameter_grads():
    class FakePrivate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.query = torch.nn.Parameter(torch.ones(2, 2))
            self.ffn = torch.nn.Parameter(torch.ones(2, 2))
            self.query.grad = torch.full((2, 2), 7.0)
            self.ffn.grad = torch.full((2, 2), 3.0)

        def named_parameters(self, prefix="", recurse=True):
            return iter(
                [
                    ("_module.bert.encoder.layer.0.attention.self.query.weight", self.query),
                    ("_module.bert.encoder.layer.0.intermediate.dense.weight", self.ffn),
                ]
            )

    args = type(
        "Args",
        (),
        {
            "gradient_layer_subset": "all",
            "gradient_param_filter": "all",
            "model_path": "bert-base-uncased",
            "grad_type": "attn_query",
            "attack_layer": [0],
            "ptg_gradient_info": {},
        },
    )()
    partial, names, selected = _capture_source_opacus_grads(args, FakePrivate())
    assert_true(selected == ["bert.encoder.layer.0.attention.self.query.weight"], f"unexpected selected names: {selected}")
    assert_true(torch.equal(partial[0], torch.full((2, 2), 7.0)), "source Opacus capture should use the live .grad tensor")


def test_ptg_variant_constant_names_summary_variant():
    assert_true(PTG_GRADIENT_MATCHING_VARIANT == "ptg_gradient_matching", "PTG summary variant should stay stable")


def test_dpsgd_opacus_defense_selects_source_opacus_mode():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--dataset",
            "sst2",
            "--split",
            "test",
            "--n_inputs",
            "2",
            "--finetuned_path",
            "dummy",
            "--defense",
            "dpsgd_opacus",
        ]
    )
    _validate_args(args)
    assert_true(args.ptg_dpsgd_mode == "source_opacus", "dpsgd_opacus should select the source Opacus loop")
    assert_true(args.defense_noise == 0.01, "dpsgd_opacus should use the source default noise multiplier")
    assert_true(args.noise_multiplier == 0.01, "source alias should mirror defense_noise")
    assert_true(args.defense_dp_delta == 1e-5, "dpsgd_opacus should default delta to 1e-5")


def test_ptg_rouge_backend_defaults_to_strict_datasets():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--dataset",
            "sst2",
            "--split",
            "test",
            "--n_inputs",
            "2",
            "--finetuned_path",
            "dummy",
        ]
    )
    _validate_args(args)
    assert_true(args.ptg_rouge_backend == "datasets", "PTG should default to strict datasets ROUGE")


def test_explicit_simple_ngram_backend_never_loads_datasets_metric():
    with patch(
        "attack_peftleak._load_datasets_rouge_metric",
        side_effect=AssertionError("simple_ngram must not load datasets ROUGE"),
    ):
        validate_text_metric_backend("simple_ngram")
        summary = compute_text_metrics(["one two"], ["one two"], backend="simple_ngram")
    assert_true(summary["rouge_backend"] == "simple_ngram", "explicit simple backend should be recorded")
    assert_true(summary["agg_r1fm_r2fm"] == 200.0, "identical text should have perfect simple ROUGE-1/2")


def test_strict_datasets_backend_does_not_fallback_on_failure():
    with patch("attack_peftleak._load_datasets_rouge_metric", side_effect=RuntimeError("metric cache missing")):
        try:
            validate_text_metric_backend("datasets")
        except RuntimeError as exc:
            assert_true("No automatic metric fallback" in str(exc), "strict failure should explain the no-fallback policy")
        else:
            raise AssertionError("strict datasets preflight should fail instead of silently falling back")


def main():
    tests = [
        test_selected_partial_gradient_tensors_ignores_hidden_gradients,
        test_ptg_selector_does_not_emit_dager_variant_or_reason,
        test_source_grad_type_selectors_match_official_names,
        test_ptg_use_embedding_dynamic_unused_tokens_from_visible_embedding_grad,
        test_ptg_use_embedding_ignores_non_token_embedding_tables,
        test_ptg_use_embedding_scans_resized_embedding_rows,
        test_ptg_use_embedding_falls_back_without_visible_embedding_grad,
        test_ptg_use_embedding_cli_aliases_set_shared_flag,
        test_strict_source_selector_rejects_packed_gpt2_attention,
        test_source_mode_defaults_to_partial_bert_and_requires_lm,
        test_ptg_swap_objective_uses_lm_only_for_source_mode,
        test_prediction_match_order_aligns_token_and_text_metrics,
        test_ptg_source_loss_names_match_formulas,
        test_source_bert_parity_uses_raw_word_embeddings,
        test_ptg_gradient_matching_loss_decreases_on_toy_model,
        test_ptg_source_init_lm_prior_and_swaps_are_executable,
        test_ptg_label_search_reports_search_mode_and_decodes,
        test_ptg_fixed_special_tokens_are_restored_in_decoding,
        test_ptg_know_padding_false_does_not_fix_pad_position,
        test_source_bert_padding_mask_matches_official_fix_special_tokens,
        test_source_attack_layer_parser,
        test_source_opacus_capture_uses_parameter_grads,
        test_ptg_variant_constant_names_summary_variant,
        test_dpsgd_opacus_defense_selects_source_opacus_mode,
        test_ptg_rouge_backend_defaults_to_strict_datasets,
        test_explicit_simple_ngram_backend_never_loads_datasets_metric,
        test_strict_datasets_backend_does_not_fallback_on_failure,
    ]
    for test in tests:
        print(f"Running {test.__name__}...")
        test()
    print("All partial Transformer gradient semantic tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
