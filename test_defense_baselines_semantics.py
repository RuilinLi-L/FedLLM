#!/usr/bin/env python3
"""
Targeted regression tests for defense baseline semantics.

These tests focus on the paper-aligned changes:
- RNG independence + reproducibility
- DP-SGD per-example clipping semantics
- Soteria representation-side masking
- BERT seq-class path keeps pooler/dropout/classifier semantics
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.defense_common import grad_similarity_metrics
from utils.defenses import _apply_random_mask, dpsgd_defense, noise_injection, soteria_defense
from utils.lrb_defense import _estimate_layer_sensitivities, _project_low_resolution


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def test_noise_rng_behavior():
    grads = (torch.zeros(2, 3), torch.zeros(2, 3))
    out_a = noise_injection(grads, sigma=1.0, seed=7)
    out_b = noise_injection(grads, sigma=1.0, seed=7)
    assert_true(torch.allclose(out_a[0], out_b[0]), "noise should be reproducible for a fixed seed")
    assert_true(torch.allclose(out_a[1], out_b[1]), "noise should be reproducible for a fixed seed")
    assert_true(not torch.allclose(out_a[0], out_a[1]), "same-shape tensors should not share identical noise")

    masked_a = _apply_random_mask((torch.ones(6), torch.ones(6)), pct_mask=0.5, seed=11)
    masked_b = _apply_random_mask((torch.ones(6), torch.ones(6)), pct_mask=0.5, seed=11)
    assert_true(torch.equal(masked_a[0], masked_b[0]), "masking should be reproducible for a fixed seed")
    assert_true(torch.equal(masked_a[1], masked_b[1]), "masking should be reproducible for a fixed seed")
    assert_true(not torch.equal(masked_a[0], masked_a[1]), "same-shape tensors should not share identical masks")


def test_dpsgd_matches_manual_formula():
    per_example_grads = [
        (torch.tensor([3.0, 4.0]),),
        (torch.tensor([0.0, 5.0]),),
    ]
    max_norm = 2.0
    sigma = 0.0
    defended = dpsgd_defense(per_example_grads, max_norm=max_norm, sigma=sigma, seed=0)

    g1 = per_example_grads[0][0]
    g2 = per_example_grads[1][0]
    c1 = min(1.0, max_norm / (g1.norm().item() + 1e-6))
    c2 = min(1.0, max_norm / (g2.norm().item() + 1e-6))
    expected = (g1 * c1 + g2 * c2) / 2.0
    aggregated_clip = (g1 + g2)
    aggregated_clip = aggregated_clip * min(1.0, max_norm / (aggregated_clip.norm().item() + 1e-6))

    assert_true(torch.allclose(defended[0], expected, atol=1e-5), "dpsgd must clip each example before averaging")
    assert_true(not torch.allclose(defended[0], aggregated_clip, atol=1e-5), "dpsgd must differ from aggregated clipping")


class DummySoteriaModel:
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32))

    def zero_grad(self, set_to_none=True):
        if self.weight.grad is not None:
            if set_to_none:
                self.weight.grad = None
            else:
                self.weight.grad.zero_()

    def parameters(self):
        yield self.weight


class DummySoteriaWrapper:
    def __init__(self):
        self.model = DummySoteriaModel()

    def trainable_parameters(self):
        return (self.model.weight,)

    def _seq_class_input_embeds(self, sample_batch):
        return sample_batch["inputs"].clone()

    def _seq_class_logits_from_embeds(self, sample_batch, inputs_embeds, representation_mask=None):
        representation = inputs_embeds.squeeze(0)
        if representation_mask is not None:
            representation = representation * representation_mask.squeeze(0)
        logits = representation.unsqueeze(0) @ self.model.weight
        return logits, representation.unsqueeze(0)

    def compute_per_example_grads(self, batch, labels, create_graph=False, sample_grad_fn=None):
        grad_list = []
        flat_labels = labels.view(-1).long()
        for idx in range(batch["inputs"].shape[0]):
            sample_batch = {"inputs": batch["inputs"][idx:idx+1]}
            sample_labels = flat_labels[idx:idx+1]
            grad_list.append(
                sample_grad_fn(
                    sample_batch,
                    sample_labels,
                    sample_idx=idx,
                    create_graph=create_graph,
                )
            )
        return grad_list


class DummySoteriaEmbeddingModel:
    def __init__(self):
        self.embedding = torch.nn.Embedding(6, 4)
        self.classifier = torch.nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.5, -0.5, 0.25],
                        [0.2, 1.5, 0.3, -0.7],
                        [-0.4, 0.1, 1.0, 0.8],
                        [0.9, -0.2, 0.4, 1.1],
                        [0.3, 0.7, -1.2, 0.6],
                    ],
                    dtype=torch.float32,
                )
            )
            self.classifier.weight.copy_(
                torch.tensor(
                    [
                        [1.0, -0.5, 0.2, 0.8],
                        [-0.3, 0.6, 1.1, -0.4],
                    ],
                    dtype=torch.float32,
                )
            )

    def zero_grad(self, set_to_none=True):
        for param in self.parameters():
            if param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    param.grad.zero_()

    def parameters(self):
        yield self.embedding.weight
        yield self.classifier.weight


class DummySoteriaEmbeddingWrapper:
    def __init__(self):
        self.model = DummySoteriaEmbeddingModel()

    def trainable_parameters(self):
        return (self.model.embedding.weight, self.model.classifier.weight)

    def _seq_class_input_embeds(self, sample_batch):
        return self.model.embedding(sample_batch["input_ids"])

    def _seq_class_logits_from_embeds(self, sample_batch, inputs_embeds, representation_mask=None):
        representation = inputs_embeds.mean(dim=1)
        if representation_mask is not None:
            representation = representation * representation_mask
        logits = self.model.classifier(representation)
        return logits, representation

    def compute_per_example_grads(self, batch, labels, create_graph=False, sample_grad_fn=None):
        grad_list = []
        flat_labels = labels.view(-1).long()
        for idx in range(batch["input_ids"].shape[0]):
            sample_batch = {"input_ids": batch["input_ids"][idx:idx+1]}
            sample_labels = flat_labels[idx:idx+1]
            grad_list.append(
                sample_grad_fn(
                    sample_batch,
                    sample_labels,
                    sample_idx=idx,
                    create_graph=create_graph,
                )
            )
        return grad_list


def test_soteria_masks_representation_during_gradient_generation():
    wrapper = DummySoteriaWrapper()
    args = SimpleNamespace(
        task="seq_class",
        train_method="full",
        defense_soteria_pruning_rate=50.0,
        defense_soteria_sample_dims=None,
        rng_seed=0,
    )
    batch = {"inputs": torch.tensor([[2.0, 1.0]], dtype=torch.float32)}
    labels = torch.tensor([0], dtype=torch.long)

    defended = soteria_defense(wrapper, batch, labels, args)
    undefended_logits = batch["inputs"] @ wrapper.model.weight
    undefended_loss = torch.nn.functional.cross_entropy(undefended_logits, labels)
    undefended_grad = torch.autograd.grad(undefended_loss, wrapper.trainable_parameters(), allow_unused=True)[0]

    assert_true(defended[0].shape == undefended_grad.shape, "soteria should preserve gradient shape")
    assert_true(not torch.allclose(defended[0], undefended_grad), "soteria should change gradients via representation masking")


def test_soteria_keeps_embedding_gradients_for_proxy_metrics():
    wrapper = DummySoteriaEmbeddingWrapper()
    args = SimpleNamespace(
        task="seq_class",
        train_method="full",
        defense_soteria_pruning_rate=50.0,
        defense_soteria_sample_dims=None,
        rng_seed=0,
    )
    batch = {"input_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.long)}
    labels = torch.tensor([0, 1], dtype=torch.long)

    defended = soteria_defense(wrapper, batch, labels, args)
    inputs_embeds = wrapper._seq_class_input_embeds(batch)
    logits, _ = wrapper._seq_class_logits_from_embeds(batch, inputs_embeds)
    raw_loss = torch.nn.functional.cross_entropy(logits, labels)
    raw_grads = torch.autograd.grad(raw_loss, wrapper.trainable_parameters(), allow_unused=True)

    assert_true(defended[0] is not None, "soteria should preserve embedding gradients for proxy comparisons")
    assert_true(defended[0].shape == wrapper.model.embedding.weight.shape, "embedding gradient shape should be preserved")
    cosine, norm_retention = grad_similarity_metrics(raw_grads, defended)
    assert_true(isinstance(cosine, float), "proxy cosine should be computable after soteria")
    assert_true(isinstance(norm_retention, float), "proxy norm retention should be computable after soteria")


def test_bert_seq_class_structure():
    content = Path(__file__).with_name("utils").joinpath("models.py").read_text(encoding="utf8")
    assert_true("representation = bert.pooler(sequence_output)" in content, "BERT path should use pooler_output")
    assert_true("logits = model.classifier(model.dropout(representation))" in content, "BERT path should apply dropout before classifier")
    assert_true("pooled = enc_out.last_hidden_state[:, 0]" not in content, "BERT path should not classify from raw CLS hidden state")


def test_lrb_signed_projection_is_reproducible_and_not_plain_pooling():
    tensor = torch.arange(1, 17, dtype=torch.float32)
    signed_a = _project_low_resolution(tensor, 0.5, seed=13, mode="signed_pool")
    signed_b = _project_low_resolution(tensor, 0.5, seed=13, mode="signed_pool")
    plain = _project_low_resolution(tensor, 0.5, seed=13, mode="pool")

    assert_true(torch.allclose(signed_a, signed_b), "signed-pool projection should be reproducible for a fixed seed")
    assert_true(not torch.allclose(signed_a, plain), "signed-pool projection should differ from plain coordinate pooling")


def test_lrb_hybrid_sensitivity_uses_empirical_calibration():
    grads = (
        torch.tensor([10.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
    )
    layer_names = [
        "transformer.h.4.attn.c_attn.weight",
        "transformer.h.5.attn.c_attn.weight",
    ]
    scores = _estimate_layer_sensitivities(
        grads,
        layer_names=layer_names,
        sensitive_n_layers=2,
        empirical_weight=1.0,
        calibration_keep_ratio=0.5,
        calibration_samples=16,
        base_seed=0,
        projection_mode="signed_pool",
    )

    assert_true(scores[0] > scores[1], "empirical calibration should score the more concentrated layer as more sensitive")


def main():
    tests = [
        test_noise_rng_behavior,
        test_dpsgd_matches_manual_formula,
        test_soteria_masks_representation_during_gradient_generation,
        test_soteria_keeps_embedding_gradients_for_proxy_metrics,
        test_bert_seq_class_structure,
        test_lrb_signed_projection_is_reproducible_and_not_plain_pooling,
        test_lrb_hybrid_sensitivity_uses_empirical_calibration,
    ]
    for test in tests:
        print(f"Running {test.__name__}...")
        test()
    print("All defense baseline semantic tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
