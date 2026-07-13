#!/usr/bin/env python3
from __future__ import annotations

import inspect
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from attacks.peftleak_text_new import (
    ProbedEmbedding,
    apply_probe_defense,
    build_public_probe_statistics,
    compute_probe_gradient_observation,
    install_fixed_embedding_probe,
    observe_lora_gradients,
    recover_tokens_from_probe_gradients,
    select_shared_lora_parameters,
    token_recovery_accuracy,
)
from utils.lrb_presets import apply_lrb_preset


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


class TinyTransformer(nn.Module):
    def __init__(self, table: torch.Tensor):
        super().__init__()
        self.wte = nn.Embedding.from_pretrained(table.clone(), freeze=True)


class TinySequenceClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        table = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [-0.5, 0.5, -0.5, 0.5],
            ],
            dtype=torch.float32,
        )
        self.transformer = TinyTransformer(table)
        self.score = nn.Linear(4, 2, bias=False)
        self.config = SimpleNamespace(num_labels=2)
        with torch.no_grad():
            self.score.weight.copy_(torch.tensor([[0.7, -0.2, 0.5, 0.1], [-0.4, 0.8, -0.1, 0.6]]))

    def forward(self, input_ids, attention_mask=None, labels=None):
        vectors = self.transformer.wte(input_ids)
        if attention_mask is None:
            representation = vectors.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(vectors.dtype)
            representation = (vectors * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        logits = self.score(representation)
        loss = None if labels is None else F.cross_entropy(logits, labels.view(-1))
        return SimpleNamespace(logits=logits, loss=loss)


class TinyAdapterWrapper:
    def __init__(self):
        self.model = TinySequenceClassifier()
        self.base_model = self.model
        self.args = SimpleNamespace(model_path="gpt2")
        self.pad_token = 0


def _public_batches():
    return [
        {
            "input_ids": torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.long),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([[4, 3, 2]], dtype=torch.long),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        },
    ]


def _defense_args(defense="none"):
    args = SimpleNamespace(
        defense=defense,
        rng_seed=17,
        defense_rng_step=0,
        defense_noise=0.0,
        defense_clip_norm=1.0,
        defense_topk_ratio=0.5,
        defense_n_bits=8,
        defense_lrb_preset="proj_only",
        defense_lrb_sensitive_n_layers=2,
        defense_lrb_keep_ratio_sensitive=0.5,
        defense_lrb_keep_ratio_other=0.75,
        defense_lrb_clip_scale_sensitive=1_000_000.0,
        defense_lrb_clip_scale_other=1_000_000.0,
        defense_lrb_noise_sensitive=0.0,
        defense_lrb_noise_other=0.0,
        defense_lrb_empirical_weight=0.0,
        defense_lrb_calibration_samples=4096,
        defense_lrb_projection="signed_pool",
    )
    return apply_lrb_preset(args)


def _install_tiny_probe():
    wrapper = TinyAdapterWrapper()
    statistics = build_public_probe_statistics(
        wrapper.model.transformer.wte,
        _public_batches(),
        max_positions=4,
        num_bins=2,
    )
    clean_batch = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    clean_logits = wrapper.model(**clean_batch).logits.detach().clone()
    installed = install_fixed_embedding_probe(wrapper, statistics, rows_per_bin=4, seed=11)
    return wrapper, installed, clean_batch, clean_logits


def test_probe_inventory_is_fixed_and_zero_initialization_preserves_logits():
    wrapper, installed, batch, clean_logits = _install_tiny_probe()
    assert_true(isinstance(wrapper.model.transformer.wte, ProbedEmbedding), "probe must be registered in the model")
    probed_logits = wrapper.model(**batch).logits.detach()
    assert_true(torch.equal(probed_logits, clean_logits), "zero-initialized probe must preserve exact clean logits")

    names_before = installed.parameter_names
    count_before = sum(parameter.numel() for parameter in installed.parameters)
    wrapper.model(**{"input_ids": torch.tensor([[4, 1]], dtype=torch.long), "attention_mask": torch.ones(1, 2)})
    names_after = tuple(f"{installed.embedding_path}.peftleak_probe.{name}" for name, _ in installed.probe.named_parameters())
    count_after = sum(parameter.numel() for parameter in installed.parameters)
    assert_true(names_before == names_after, "private tokens must not change the probe parameter names")
    assert_true(count_before == count_after, "private tokens must not change the probe parameter count")


def test_real_classification_loss_gradients_recover_clean_batch_one_tokens():
    wrapper, installed, batch, _ = _install_tiny_probe()
    labels = torch.tensor([1], dtype=torch.long)
    observation = compute_probe_gradient_observation(wrapper, batch, labels, installed, _defense_args("none"))
    assert_true(observation.loss > 0, "probe gradients must come from a real classification loss")
    assert_true(any(torch.count_nonzero(gradient) for gradient in observation.raw_gradients), "real loss must reach probe")

    decoded = recover_tokens_from_probe_gradients(
        observation.observed_gradients,
        observation.parameter_names,
        wrapper.model.transformer.wte.weight.detach(),
        max_positions=4,
        num_bins=2,
        fallback_token_id=0,
    )
    recovered = token_recovery_accuracy(
        decoded["predicted_ids"],
        batch["input_ids"].tolist(),
        ignored_token_ids={0},
        reference_mask=batch["attention_mask"].tolist(),
    )
    assert_true(abs(recovered - 1.0) < 1e-8, f"clean batch-one ratio recovery should be exact, got {recovered}")
    assert_true(decoded["decoder_private_routing"] is False, "decoder must not use private routing")


def test_private_tokens_change_gradients_without_changing_inventory():
    wrapper, installed, batch_a, _ = _install_tiny_probe()
    labels = torch.tensor([1], dtype=torch.long)
    obs_a = compute_probe_gradient_observation(wrapper, batch_a, labels, installed, _defense_args("none"))
    batch_b = {
        "input_ids": torch.tensor([[4, 3, 1]], dtype=torch.long),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    obs_b = compute_probe_gradient_observation(wrapper, batch_b, labels, installed, _defense_args("none"))
    assert_true(obs_a.parameter_names == obs_b.parameter_names, "gradient inventory must be private-batch independent")
    changed = any(not torch.equal(left, right) for left, right in zip(obs_a.raw_gradients, obs_b.raw_gradients))
    assert_true(changed, "changing private tokens should change observed probe gradients")


def test_decoder_api_has_no_private_batch_or_routing_argument():
    parameters = inspect.signature(recover_tokens_from_probe_gradients).parameters
    forbidden = {"batch", "slot_keys", "routing", "private_embeddings", "reference_ids"}
    assert_true(not forbidden.intersection(parameters), f"decoder exposes forbidden private inputs: {parameters}")


def test_strict_entrypoint_rejects_batch_collisions_and_unsupported_defenses():
    from attack_peftleak_new import _validate_args, build_parser

    parser = build_parser()
    previous = sys.modules.get("utils.peft_utils")
    fake_peft_utils = ModuleType("utils.peft_utils")
    fake_peft_utils.apply_peft_config_to_args = lambda args, require_checkpoint=True: args
    sys.modules["utils.peft_utils"] = fake_peft_utils
    try:
        collision_args = parser.parse_args(
            [
                "--dataset", "sst2", "--split", "val", "--n_inputs", "1",
                "--batch_size", "2", "--finetuned_path", "dummy", "--peft_method", "adapter",
                "--text_metric_backend", "simple_ngram",
            ]
        )
        try:
            _validate_args(collision_args)
        except ValueError as exc:
            assert_true("batch_size 1" in str(exc), "collision rejection should explain the formal batch limit")
        else:
            raise AssertionError("Adapter ratio v2 must reject batch_size > 1")

        for defense in ("soteria", "mixup"):
            unsupported = parser.parse_args(
                [
                    "--dataset", "sst2", "--split", "val", "--n_inputs", "1",
                    "--finetuned_path", "dummy", "--peft_method", "adapter", "--defense", defense,
                    "--text_metric_backend", "simple_ngram",
                ]
            )
            try:
                _validate_args(unsupported)
            except NotImplementedError:
                pass
            else:
                raise AssertionError(f"{defense} must fail explicitly instead of becoming a no-op")
    finally:
        if previous is None:
            sys.modules.pop("utils.peft_utils", None)
        else:
            sys.modules["utils.peft_utils"] = previous


class TinyLoRAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.Parameter(torch.ones(2, 3))
        self.modules_to_save = nn.Linear(3, 2, bias=False)


def test_lora_inventory_excludes_modules_to_save_classifier():
    wrapper = SimpleNamespace(model=TinyLoRAModel())
    parameters, names = select_shared_lora_parameters(wrapper)
    assert_true(len(parameters) == 1, f"only the LoRA tensor should be shared, got {names}")
    assert_true(names == ("lora_A",), f"modules_to_save must be excluded, got {names}")


def _assert_observation_optimization_decreases(defense: str):
    args = _defense_args(defense)
    names = ["base_model.transformer.wte.peftleak_probe.weight"]
    target_raw = (torch.tensor([[2.0, -1.0, 0.5, -0.25]], dtype=torch.float32),)
    target = tuple(value.detach() for value in observe_lora_gradients(target_raw, names, args, differentiable=False))
    candidate = nn.Parameter(target_raw[0] + 0.2)
    optimizer = torch.optim.SGD([candidate], lr=0.2)
    history = []
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        observed = observe_lora_gradients((candidate,), names, args, differentiable=True)[0]
        loss = F.mse_loss(observed, target[0])
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().item()))
    assert_true(history[-1] < history[0], f"{defense} defense-aware loss should decrease: {history[0]} -> {history[-1]}")


def test_defense_aware_topk_candidate_loss_decreases():
    _assert_observation_optimization_decreases("topk")


def test_defense_aware_projection_candidate_loss_decreases():
    _assert_observation_optimization_decreases("lrb")


def test_probe_defense_rejects_soteria_and_mixup_instead_of_noop():
    gradients = (torch.ones(2, 2),)
    names = ("base_model.transformer.wte.peftleak_probe.positions.0.weight",)
    for defense in ("soteria", "mixup"):
        try:
            apply_probe_defense(gradients, names, _defense_args(defense))
        except NotImplementedError:
            pass
        else:
            raise AssertionError(f"{defense} must not silently pass probe gradients through")


def test_entrypoint_installs_probe_before_private_dataset_and_has_strict_metrics():
    root = Path(os.path.dirname(os.path.abspath(__file__)))
    source = (root / "attack_peftleak_new.py").read_text(encoding="utf-8")
    install_index = source.index("installed = install_fixed_embedding_probe")
    private_index = source.index("private_dataset = TextDataset")
    assert_true(install_index < private_index, "probe must be installed before private data is instantiated")

    from attack_peftleak_new import build_parser

    metric_action = next(action for action in build_parser()._actions if action.dest == "text_metric_backend")
    assert_true("auto" not in metric_action.choices, "v2 must not silently fall back between ROUGE backends")


def test_v2_privacy_runner_uses_the_strict_entrypoint_and_gated_matrix():
    root = Path(os.path.dirname(os.path.abspath(__file__)))
    runner = (root / "scripts" / "peftleak_text_v2_privacy.sh").read_text(encoding="utf-8")
    runbook = (root / "docs" / "PEFTLEAK_TEXT_V2_PRIVACY_RUNBOOK.md").read_text(encoding="utf-8")

    assert_true("attack_peftleak_new.py" in runner, "v2 runner must call the strict attack entrypoint")
    assert_true("attack_peftleak.py" not in runner, "v2 runner must not fall back to the legacy attack")
    assert_true("--peftleak_ratio_route" not in runner, "v2 runner must not pass a legacy routing argument")
    assert_true('SEEDS:-101 202 303' in runner, "formal privacy seeds must default to 101/202/303")
    assert_true('FORMAL_N:-100' in runner, "formal privacy must default to n_inputs=100")
    assert_true(
        "MATRIX_LABELS=(none topk_0.1 compression_6 noise_1e-3 "
        "proj_only_0.5 proj_only_0.65 proj_only_0.75 proj_only_0.9)" in runner,
        "v2 runner must contain the eight utility-matched privacy configurations",
    )
    for expected in (
        "--defense topk --defense_topk_ratio 0.1",
        "--defense compression --defense_n_bits 6",
        "--defense noise --defense_noise 1e-3",
        "for keep_ratio in 0.5 0.65 0.75 0.9",
    ):
        assert_true(expected in runner, f"missing utility-matched matrix command: {expected}")
    assert_true("validate_stage smoke" in runner, "pilot must be gated by smoke validation")
    assert_true("validate_stage pilot" in runner, "formal runs must be gated by pilot validation")
    assert_true("shared_gradient_names_sha256" in runner, "runner must enforce fixed shared-gradient inventory")
    assert_true("refusing to overwrite incomplete log" in runner, "runner must protect incomplete logs")
    assert_true("cross-protocol supplementary privacy-utility comparison" in runbook, "runbook must state the evidence boundary")


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} PEFTLeak text v2 semantic tests passed.")


if __name__ == "__main__":
    main()
