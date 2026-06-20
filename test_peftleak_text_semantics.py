#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attacks.peftleak_text import (
    nearest_token_ids,
    optimize_text_embeddings,
    select_peft_gradient_tensors,
    token_recovery_ratio,
)


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


class DummyTokenizer:
    def decode(self, ids):
        return " ".join(str(int(tok)) for tok in ids)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_labels=2)
        self.embedding = torch.nn.Embedding(6, 4)
        self.adapter = torch.nn.Linear(4, 2, bias=False)
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
                    ],
                    dtype=torch.float32,
                )
            )
            self.adapter.weight.copy_(torch.tensor([[0.7, -0.4, 0.2, 0.3], [-0.1, 0.6, -0.5, 0.4]]))

    def get_input_embeddings(self):
        return self.embedding


class DummyWrapper:
    def __init__(self):
        self.model = DummyModel()
        self.args = SimpleNamespace(rng_seed=7)
        self.tokenizer = DummyTokenizer()
        self.pad_token = 0

    def _seq_class_input_embeds(self, batch):
        return self.model.embedding(batch["input_ids"])

    def _seq_class_logits_from_embeds(self, batch, inputs_embeds, representation_mask=None):
        representation = inputs_embeds.mean(dim=1)
        if representation_mask is not None:
            representation = representation * representation_mask
        return self.model.adapter(representation), representation

    def trainable_parameters(self):
        return (self.model.adapter.weight,)

    def trainable_parameter_names(self):
        return ["base_model.model.encoder.layer.0.attention.self.query.lora_A.default.weight"]


def _target_grads(wrapper, batch, labels):
    embeds = wrapper._seq_class_input_embeds(batch)
    logits, _ = wrapper._seq_class_logits_from_embeds(batch, embeds)
    loss = F.cross_entropy(logits, labels)
    return torch.autograd.grad(loss, wrapper.trainable_parameters(), allow_unused=True)


def test_peft_gradient_inventory_selects_lora_ia3_and_excludes_saved_heads():
    grads = (torch.ones(2, 2), torch.ones(2), torch.ones(2, 2), None, torch.ones(2, 4))
    names = [
        "base_model.model.layer.0.lora_A.default.weight",
        "base_model.model.score.modules_to_save.default.weight",
        "base_model.model.layer.0.ia3_l.default",
        "base_model.model.layer.1.lora_A.default.weight",
        "base_model.model.bert.encoder.layer.0.output.adapters.default.adapter_down.0.weight",
    ]
    lora_idx, lora_names = select_peft_gradient_tensors(grads, names, "lora")
    ia3_idx, ia3_names = select_peft_gradient_tensors(grads, names, "ia3")
    adapter_idx, adapter_names = select_peft_gradient_tensors(grads, names, "adapter")

    assert_true(lora_idx == [0], f"LoRA selector should keep only adapter tensors with gradients: {lora_idx}")
    assert_true("modules_to_save" not in ";".join(lora_names), "selector should exclude modules_to_save")
    assert_true(ia3_idx == [2], f"IA3 selector should keep IA3 tensors: {ia3_idx}")
    assert_true("ia3" in ia3_names[0].lower(), "IA3 selector should preserve IA3 name")
    assert_true(adapter_idx == [4], f"Adapter selector should keep AdapterHub bottleneck tensors: {adapter_idx}")
    assert_true("adapter_down" in adapter_names[0].lower(), "Adapter selector should preserve adapter down-projection name")


def test_nearest_token_decoding_returns_valid_ids_and_ignores_pad():
    table = torch.eye(4, dtype=torch.float32)
    embeddings = torch.stack([table[1] + 0.01, table[3] - 0.01], dim=0)
    token_ids, scores = nearest_token_ids(embeddings, table, unused_token_ids={0}, metric="cos")

    assert_true(token_ids.tolist() == [1, 3], f"nearest token ids should be valid and pad-free: {token_ids.tolist()}")
    assert_true(torch.isfinite(scores).all(), "nearest token scores should be finite")


def test_token_recovery_is_positionwise():
    recovered = token_recovery_ratio([[1, 2, 3], [4, 5]], [[1, 9, 3], [4, 0]])
    assert_true(abs(recovered - 0.6) < 1e-8, f"positionwise token recovery should be 3/5, got {recovered}")


def test_token_recovery_ignores_padding_and_special_tokens():
    recovered = token_recovery_ratio(
        [[101, 7, 8, 0], [101, 4, 9, 0]],
        [[101, 7, 3, 0], [101, 5, 9, 0]],
        ignored_token_ids={0, 101},
        reference_mask=[[1, 1, 1, 0], [1, 1, 1, 0]],
    )
    assert_true(abs(recovered - 0.5) < 1e-8, f"content-token recovery should ignore pads/specials, got {recovered}")


def test_gradient_matching_loss_decreases_on_tiny_peft_model():
    torch.manual_seed(3)
    wrapper = DummyWrapper()
    batch = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}
    labels = torch.tensor([1], dtype=torch.long)
    target = _target_grads(wrapper, batch, labels)

    result = optimize_text_embeddings(
        model_wrapper=wrapper,
        batch=batch,
        labels=labels,
        target_grads=target,
        parameter_names=wrapper.trainable_parameter_names(),
        peft_method="lora",
        steps=25,
        lr=0.15,
    )

    history = result["loss_history"]
    assert_true(history[-1] < history[0], f"gradient matching loss should decrease: {history[0]} -> {history[-1]}")
    assert_true(len(result["predicted_ids"]) == 1, "attack should return one decoded sequence")
    assert_true(all(0 <= tok < 6 for tok in result["predicted_ids"][0]), "decoded ids should be valid tokenizer ids")


def test_gradient_matching_label_search_reports_search_mode():
    torch.manual_seed(4)
    wrapper = DummyWrapper()
    batch = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}
    labels = torch.tensor([1], dtype=torch.long)
    target = _target_grads(wrapper, batch, labels)

    result = optimize_text_embeddings(
        model_wrapper=wrapper,
        batch=batch,
        labels=labels,
        target_grads=target,
        parameter_names=wrapper.trainable_parameter_names(),
        peft_method="lora",
        steps=6,
        lr=0.1,
        label_known=False,
        label_candidates=[0, 1],
    )

    assert_true(result["label_mode"] == "search", f"label search should report search mode, got {result['label_mode']}")
    assert_true(result["best_label"] is not None, "label search should record the best candidate labels")
    assert_true(result["selected_gradient_count"] == 1, "adapter gradient count should be reported")
    assert_true(result["sequence_length"] == 3, "sequence length should be reported")


def test_attack_entrypoint_policy_rejects_prefix_and_keeps_dager_unsupported():
    from attack_peftleak import _validate_args, build_parser

    parser = build_parser()
    prefix_args = parser.parse_args(
        [
            "--dataset", "sst2",
            "--split", "val",
            "--n_inputs", "1",
            "--finetuned_path", "dummy",
            "--peft_method", "prefix",
        ]
    )
    try:
        _validate_args(prefix_args)
    except NotImplementedError as exc:
        assert_true("lora|ia3|adapter" in str(exc), "prefix rejection should mention supported FedLLM PEFT text methods")
    else:
        raise AssertionError("prefix FedLLM PEFT text eval should be rejected")

    adapter_args = parser.parse_args(
        [
            "--dataset", "sst2",
            "--split", "val",
            "--n_inputs", "1",
            "--finetuned_path", "dummy",
            "--peft_method", "adapter",
        ]
    )
    assert_true(adapter_args.peft_method == "adapter", "adapter should be accepted as a PEFTLeak CLI choice")

    dager_args = parser.parse_args(
        [
            "--dataset", "sst2",
            "--split", "val",
            "--n_inputs", "1",
            "--finetuned_path", "dummy",
            "--peft_method", "lora",
            "--defense", "dager",
        ]
    )
    validated = _validate_args(dager_args)
    assert_true(validated.defense == "dager", "DAGER defense should parse so the entrypoint can emit unsupported summary")


def main():
    tests = [
        test_peft_gradient_inventory_selects_lora_ia3_and_excludes_saved_heads,
        test_nearest_token_decoding_returns_valid_ids_and_ignores_pad,
        test_token_recovery_is_positionwise,
        test_token_recovery_ignores_padding_and_special_tokens,
        test_gradient_matching_loss_decreases_on_tiny_peft_model,
        test_gradient_matching_label_search_reports_search_mode,
        test_attack_entrypoint_policy_rejects_prefix_and_keeps_dager_unsupported,
    ]
    for test in tests:
        print(f"Running {test.__name__}...")
        test()
    print("All FedLLM PEFT text semantic tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

