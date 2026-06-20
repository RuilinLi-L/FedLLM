#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import args_factory
import train
import utils.peft_utils as peft_utils_module
from train import prepare_training_defense
from utils.models import select_lora_gradient_indices, select_peft_gradient_indices
from utils.peft_utils import (
    apply_peft_config_to_args,
    apply_peft_adapter,
    apply_lora_config_to_args,
    PEFT_EVAL_SCOPE_DAGER,
    PEFT_EVAL_SCOPE_NA,
    PEFT_EVAL_SCOPE_TRAINING_ONLY,
    peft_eval_scope,
    peft_eval_scope_message,
    parse_lora_target_modules,
    is_peft_adapter_dir,
    lora_modules_to_save,
    lora_target_modules,
    resolve_peft_config,
    resolve_lora_checkpoint_path,
    save_peft_checkpoint,
    validate_peft_eval_args,
    validate_lora_eval_args,
)


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def _temp_checkpoint() -> str:
    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    torch.save({"dummy": torch.tensor([1.0])}, path)
    return path


def _temp_adapter_dir() -> str:
    return _temp_peft_adapter_dir("LORA")


def _temp_peft_adapter_dir(peft_type: str, model_path: str = "gpt2", num_virtual_tokens: int = 20) -> str:
    path = Path(tempfile.mkdtemp())
    config = {"task_type": "SEQ_CLS", "base_model_name_or_path": model_path}
    if peft_type != "ADAPTER":
        config["peft_type"] = peft_type
    if peft_type == "LORA":
        config.update({"r": 16, "target_modules": ["query", "value"] if model_path == "bert-base-uncased" else ["c_attn"]})
    elif peft_type == "IA3":
        config.update({"target_modules": ["query", "value", "intermediate.dense"], "feedforward_modules": ["intermediate.dense"]})
    elif peft_type == "PREFIX_TUNING":
        config.update({"num_virtual_tokens": num_virtual_tokens})
    elif peft_type == "ADAPTER":
        config.update({"architecture": "double_seq_bn", "reduction_factor": 16})
    (path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    torch.save({"dummy": torch.tensor([1.0])}, path / "adapter_model.bin")
    return str(path)


def test_lora_target_modules_match_supported_families():
    assert_true(lora_target_modules("gpt2") == ["c_attn"], "GPT-2 LoRA target module should be c_attn")
    assert_true(lora_target_modules("bert-base-uncased") == ["query", "value"], "BERT LoRA target modules should be query,value")
    assert_true(
        lora_target_modules("meta-llama/Meta-Llama-3.1-8B") == ["q_proj"],
        "Llama LoRA target module should be q_proj",
    )


def test_parse_lora_target_module_presets():
    assert_true(parse_lora_target_modules("qv") == ("q_proj", "v_proj"), "qv preset should expand")
    assert_true(
        parse_lora_target_modules("qkvo") == ("q_proj", "k_proj", "v_proj", "o_proj"),
        "qkvo preset should expand",
    )
    assert_true(parse_lora_target_modules("all-linear") == ("all-linear",), "all-linear should pass through")
    assert_true(
        parse_lora_target_modules("q_proj,v_proj") == ("q_proj", "v_proj"),
        "comma-separated target modules should parse",
    )


def test_seq_class_modules_to_save_include_classifier_heads():
    gpt2_like = SimpleNamespace(score=object())
    bert_like = SimpleNamespace(classifier=object())
    assert_true(
        lora_modules_to_save(gpt2_like, "seq_class") == ["score"],
        "GPT/Llama seq-class LoRA should save the score head",
    )
    assert_true(
        lora_modules_to_save(bert_like, "seq_class") == ["classifier"],
        "BERT seq-class LoRA should save the classifier head when supported",
    )
    assert_true(
        lora_modules_to_save(gpt2_like, "next_token_pred") is None,
        "Causal LM LoRA should not force seq-class modules_to_save",
    )


def test_resolve_lora_checkpoint_accepts_peft_adapter_dir():
    adapter_dir = Path(_temp_adapter_dir())
    try:
        assert_true(is_peft_adapter_dir(adapter_dir), "temp adapter directory should be recognized")
        resolved = resolve_lora_checkpoint_path(str(adapter_dir))
        assert_true(resolved == adapter_dir, "adapter directory should resolve directly")
    finally:
        for child in adapter_dir.iterdir():
            child.unlink()
        adapter_dir.rmdir()


def test_validate_lora_eval_args_accepts_supported_gpt2_setup():
    checkpoint = _temp_checkpoint()
    try:
        args = SimpleNamespace(
            train_method="lora",
            peft_method=None,
            model_path="gpt2",
            finetuned_path=checkpoint,
            lora_r=16,
            defense="lrb",
            task="seq_class",
            lora_target_modules=None,
            peft_num_virtual_tokens=20,
        )
        validated = validate_lora_eval_args(args)
        assert_true(validated is args, "validation should round-trip supported LoRA args")
    finally:
        Path(checkpoint).unlink(missing_ok=True)


def test_validate_lora_eval_args_accepts_direct_generation_baselines():
    checkpoint = _temp_checkpoint()
    try:
        for defense in ("dpsgd", "soteria", "mixup"):
            args = SimpleNamespace(
                train_method="lora",
                peft_method=None,
                model_path="gpt2",
                finetuned_path=checkpoint,
                lora_r=16,
                defense=defense,
                task="seq_class",
                lora_target_modules=None,
                peft_num_virtual_tokens=20,
            )
            validated = validate_lora_eval_args(args)
            assert_true(validated is args, f"{defense} should be accepted for LoRA eval")
    finally:
        Path(checkpoint).unlink(missing_ok=True)


def test_validate_lora_eval_args_accepts_peft_adapter_dir():
    adapter_dir = Path(_temp_adapter_dir())
    try:
        args = SimpleNamespace(
            train_method="lora",
            peft_method=None,
            model_path="gpt2",
            finetuned_path=str(adapter_dir),
            lora_r=None,
            defense="lrb",
            task="seq_class",
            lora_target_modules=None,
            peft_num_virtual_tokens=20,
        )
        validated = validate_lora_eval_args(args)
        assert_true(validated is args, "validation should accept PEFT adapter directories")
        assert_true(args.lora_r == 16, "adapter config should fill lora_r")
        assert_true(args.lora_target_modules == "c_attn", "adapter config should fill target modules")
    finally:
        for child in adapter_dir.iterdir():
            child.unlink()
        adapter_dir.rmdir()


def test_validate_lora_eval_args_rejects_adapter_rank_mismatch():
    adapter_dir = Path(_temp_adapter_dir())
    try:
        args = SimpleNamespace(
            train_method="lora",
            peft_method=None,
            model_path="gpt2",
            finetuned_path=str(adapter_dir),
            lora_r=8,
            defense="lrb",
            task="seq_class",
            lora_target_modules=None,
            peft_num_virtual_tokens=20,
        )
        try:
            validate_lora_eval_args(args)
        except ValueError as exc:
            assert_true("lora_r does not match" in str(exc), "rank mismatch error should be explicit")
        else:
            raise AssertionError("adapter rank mismatch should fail validation")
    finally:
        for child in adapter_dir.iterdir():
            child.unlink()
        adapter_dir.rmdir()


def test_validate_lora_eval_args_rejects_adapter_target_mismatch():
    adapter_dir = Path(_temp_adapter_dir())
    try:
        args = SimpleNamespace(
            train_method="lora",
            peft_method=None,
            model_path="gpt2",
            finetuned_path=str(adapter_dir),
            lora_r=16,
            defense="lrb",
            task="seq_class",
            lora_target_modules="all-linear",
            peft_num_virtual_tokens=20,
        )
        try:
            validate_lora_eval_args(args)
        except ValueError as exc:
            assert_true("lora_target_modules does not match" in str(exc), "target mismatch error should be explicit")
        else:
            raise AssertionError("adapter target module mismatch should fail validation")
    finally:
        for child in adapter_dir.iterdir():
            child.unlink()
        adapter_dir.rmdir()


def test_legacy_lora_checkpoint_requires_rank():
    checkpoint = _temp_checkpoint()
    try:
        args = SimpleNamespace(
            train_method="lora",
            peft_method=None,
            model_path="gpt2",
            finetuned_path=checkpoint,
            lora_r=None,
            defense="lrb",
            task="seq_class",
            lora_target_modules=None,
            peft_num_virtual_tokens=20,
        )
        try:
            validate_lora_eval_args(args)
        except ValueError as exc:
            assert_true("requires --lora_r" in str(exc), "legacy checkpoint should require lora_r")
        else:
            raise AssertionError("legacy LoRA state dict without rank should fail validation")
    finally:
        Path(checkpoint).unlink(missing_ok=True)


def test_apply_lora_config_to_args_validates_adapter_task_type():
    adapter_dir = Path(_temp_adapter_dir())
    try:
        args = SimpleNamespace(
            train_method="lora",
            peft_method=None,
            model_path="gpt2",
            finetuned_path=str(adapter_dir),
            lora_r=None,
            task="next_token_pred",
            lora_target_modules=None,
            peft_num_virtual_tokens=20,
        )
        try:
            apply_lora_config_to_args(args, require_checkpoint=True)
        except ValueError as exc:
            assert_true("task_type" in str(exc), "task mismatch should mention adapter task_type")
        else:
            raise AssertionError("adapter task_type mismatch should fail validation")
    finally:
        for child in adapter_dir.iterdir():
            child.unlink()
        adapter_dir.rmdir()


def test_lora_gradient_inventory_prefers_adapter_a_and_excludes_saved_heads():
    names = [
        "base_model.model.transformer.h.0.attn.c_attn.lora_B.default.weight",
        "base_model.model.transformer.h.0.attn.c_attn.lora_A.default.weight",
        "base_model.model.score.modules_to_save.default.weight",
        "base_model.model.transformer.h.0.mlp.c_fc.lora_A.default.weight",
        "base_model.model.transformer.h.1.attn.c_attn.lora_A.default.weight",
    ]
    selected = select_lora_gradient_indices(
        names,
        target_modules="all-linear",
        preferred_modules=["c_attn"],
    )
    assert_true(
        selected == [
            (1, "base_model.model.transformer.h.0.attn.c_attn.lora_A.default.weight"),
            (4, "base_model.model.transformer.h.1.attn.c_attn.lora_A.default.weight"),
        ],
        "LoRA gradient inventory should use c_attn lora_A tensors and exclude modules_to_save",
    )


def test_validate_lora_eval_args_rejects_unsupported_defense():
    checkpoint = _temp_checkpoint()
    try:
        args = SimpleNamespace(
            train_method="lora",
            peft_method=None,
            model_path="gpt2",
            finetuned_path=checkpoint,
            lora_r=16,
            defense="dager",
            task="seq_class",
            lora_target_modules=None,
            peft_num_virtual_tokens=20,
        )
        try:
            validate_lora_eval_args(args)
        except NotImplementedError as exc:
            assert_true("supports only these defenses" in str(exc), "unsupported defense error should be explicit")
        else:
            raise AssertionError("unsupported LoRA defense should fail validation")
    finally:
        Path(checkpoint).unlink(missing_ok=True)


def test_validate_lora_eval_args_rejects_unsupported_model_family():
    checkpoint = _temp_checkpoint()
    try:
        args = SimpleNamespace(
            train_method="lora",
            peft_method=None,
            model_path="roberta-base",
            finetuned_path=checkpoint,
            lora_r=16,
            defense="lrb",
            task="seq_class",
            lora_target_modules=None,
            peft_num_virtual_tokens=20,
        )
        try:
            validate_lora_eval_args(args)
        except ValueError as exc:
            assert_true("GPT-2, BERT, and Llama" in str(exc), "unsupported model family error should mention supported families")
        else:
            raise AssertionError("unsupported LoRA model should fail validation")
    finally:
        Path(checkpoint).unlink(missing_ok=True)


def test_bert_lora_eval_args_are_supported():
    checkpoint = _temp_checkpoint()
    try:
        args = SimpleNamespace(
            train_method="peft",
            peft_method="lora",
            model_path="bert-base-uncased",
            finetuned_path=checkpoint,
            lora_r=8,
            defense="lrbprojonly",
            task="seq_class",
            lora_target_modules=None,
            peft_num_virtual_tokens=20,
        )
        validated = validate_peft_eval_args(args)
        assert_true(validated is args, "BERT LoRA PEFT eval args should validate")
        assert_true(args.train_method == "peft", "PEFT validation should normalize train_method")
        assert_true(args.peft_method == "lora", "PEFT method should stay lora")
        assert_true(args.lora_target_modules == "query,value", "BERT LoRA should default to query,value")
    finally:
        Path(checkpoint).unlink(missing_ok=True)


def test_ia3_and_prefix_adapter_metadata_normalize():
    ia3_dir = Path(_temp_peft_adapter_dir("IA3", model_path="bert-base-uncased"))
    prefix_dir = Path(_temp_peft_adapter_dir("PREFIX_TUNING", model_path="bert-base-uncased", num_virtual_tokens=8))
    try:
        ia3_args = SimpleNamespace(
            train_method="peft",
            peft_method="ia3",
            model_path="bert-base-uncased",
            finetuned_path=str(ia3_dir),
            lora_r=None,
            defense="lrb",
            task="seq_class",
            lora_target_modules=None,
            peft_num_virtual_tokens=20,
        )
        apply_peft_config_to_args(ia3_args, require_checkpoint=True)
        assert_true(ia3_args.peft_method == "ia3", "IA3 adapter should resolve to peft_method=ia3")
        assert_true(ia3_args.peft_type == "IA3", "IA3 peft_type should come from adapter metadata")
        assert_true(
            ia3_args.peft_feedforward_modules == "intermediate.dense",
            "IA3 feedforward modules should be recorded",
        )

        prefix_args = SimpleNamespace(
            train_method="peft",
            peft_method="prefix",
            model_path="bert-base-uncased",
            finetuned_path=str(prefix_dir),
            lora_r=None,
            defense="lrb",
            task="seq_class",
            lora_target_modules=None,
            peft_num_virtual_tokens=None,
        )
        apply_peft_config_to_args(prefix_args, require_checkpoint=True)
        assert_true(prefix_args.peft_method == "prefix", "PREFIX_TUNING metadata should normalize to peft_method=prefix")
        assert_true(prefix_args.peft_type == "PREFIX_TUNING", "prefix peft_type should preserve adapter metadata")
        assert_true(prefix_args.peft_num_virtual_tokens == 8, "prefix virtual tokens should be read from adapter config")
    finally:
        for directory in (ia3_dir, prefix_dir):
            for child in directory.iterdir():
                child.unlink()
            directory.rmdir()


def test_adapter_method_resolves_default_reduction_factor():
    resolved = resolve_peft_config(
        model_path="bert-base-uncased",
        peft_method="adapter",
        task="seq_class",
    )
    assert_true(resolved.peft_method == "adapter", "Houlsby-style adapter should resolve as a supported PEFT method")
    assert_true(resolved.adapter_reduction_factor == 16, "adapter should default to reduction factor 16")


def test_adapter_metadata_normalizes_from_adapterhub_architecture():
    adapter_dir = Path(_temp_peft_adapter_dir("ADAPTER", model_path="bert-base-uncased"))
    try:
        args = SimpleNamespace(
            train_method="peft",
            peft_method=None,
            model_path="bert-base-uncased",
            finetuned_path=str(adapter_dir),
            lora_r=None,
            defense="lrb",
            task="seq_class",
            lora_target_modules=None,
            peft_num_virtual_tokens=None,
            adapter_reduction_factor=None,
        )
        apply_peft_config_to_args(args, require_checkpoint=True)
        assert_true(args.peft_method == "adapter", "AdapterHub architecture should normalize to peft_method=adapter")
        assert_true(args.peft_type == "ADAPTER", "adapter peft_type should be recorded as ADAPTER")
        assert_true(args.adapter_reduction_factor == 16, "adapter reduction factor should be read from metadata")
        assert_true(args.peft_adapter_architecture == "double_seq_bn", "adapter architecture should be retained")
    finally:
        for child in adapter_dir.iterdir():
            child.unlink()
        adapter_dir.rmdir()


def test_adapter_backend_smoke_save_and_reload_with_fake_backend():
    class FakeDoubleSeqBnConfig:
        def __init__(self, reduction_factor):
            self.reduction_factor = reduction_factor

    class FakeBackend:
        DoubleSeqBnConfig = FakeDoubleSeqBnConfig

        @staticmethod
        def init(model):
            model.adapter_events = []

            def add_adapter(name, config):
                model.adapter_events.append(("add", name, config.reduction_factor))
                model.adapter_down = torch.nn.Linear(2, 1, bias=False)

            def train_adapter(name):
                model.adapter_events.append(("train", name))
                for param in model.parameters():
                    param.requires_grad = False
                for param in model.adapter_down.parameters():
                    param.requires_grad = True

            def save_adapter(path, name):
                model.adapter_events.append(("save", name))
                out = Path(path)
                out.mkdir(parents=True, exist_ok=True)
                (out / "adapter_config.json").write_text(
                    json.dumps({"architecture": "double_seq_bn", "reduction_factor": 16}),
                    encoding="utf-8",
                )
                torch.save(model.adapter_down.state_dict(), out / "adapter_model.bin")

            def load_adapter(path, load_as=None, set_active=False):
                name = load_as or "loaded_adapter"
                model.adapter_events.append(("load", name, bool(set_active)))
                model.adapter_down = torch.nn.Linear(2, 1, bias=False)
                state = torch.load(Path(path) / "adapter_model.bin", map_location="cpu")
                model.adapter_down.load_state_dict(state)
                return name

            def set_active_adapters(name):
                model.adapter_events.append(("active", name))

            model.add_adapter = add_adapter
            model.train_adapter = train_adapter
            model.save_adapter = save_adapter
            model.load_adapter = load_adapter
            model.set_active_adapters = set_active_adapters

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(_name_or_path="bert-base-uncased")
            self.backbone = torch.nn.Linear(2, 2)
            self.classifier = torch.nn.Linear(2, 2)

        def forward(self, x):
            hidden = self.backbone(x)
            if hasattr(self, "adapter_down"):
                hidden = hidden + self.adapter_down(hidden).repeat(1, 2)
            return self.classifier(hidden)

    class FakeTokenizer:
        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    old_backend = peft_utils_module.adapterlib
    peft_utils_module.adapterlib = FakeBackend
    save_dir = Path(tempfile.mkdtemp()) / "adapter"
    try:
        model = apply_peft_adapter(
            FakeModel(),
            model_path="bert-base-uncased",
            peft_method="adapter",
            task="seq_class",
            adapter_reduction_factor=16,
        )
        output = model(torch.ones(1, 2))
        assert_true(tuple(output.shape) == (1, 2), "fake adapter model should run a forward pass")
        assert_true(any(event[0] == "add" for event in model.adapter_events), "adapter should be added")
        assert_true(any(event[0] == "train" for event in model.adapter_events), "adapter should be marked trainable")
        assert_true(model.classifier.weight.requires_grad, "seq-class classifier head should stay trainable")

        saved = save_peft_checkpoint(model, FakeTokenizer(), save_dir)
        assert_true(Path(saved["adapter_path"]).is_dir(), "adapter save should create a directory")
        assert_true((Path(saved["adapter_path"]) / "fedllm_peft_metadata.json").is_file(), "adapter metadata should be saved")

        reloaded = apply_peft_adapter(
            FakeModel(),
            model_path="bert-base-uncased",
            peft_method="adapter",
            checkpoint_path=saved["adapter_path"],
            task="seq_class",
        )
        reloaded(torch.ones(1, 2))
        assert_true(any(event[0] == "load" for event in reloaded.adapter_events), "adapter should reload from directory")
        assert_true(reloaded.classifier.weight.requires_grad, "reloaded classifier head should stay trainable")
    finally:
        peft_utils_module.adapterlib = old_backend
        if save_dir.parent.exists():
            shutil.rmtree(save_dir.parent)


def test_prefix_eval_is_rejected_for_dager_span():
    prefix_dir = Path(_temp_peft_adapter_dir("PREFIX_TUNING", model_path="bert-base-uncased", num_virtual_tokens=8))
    try:
        args = SimpleNamespace(
            train_method="peft",
            peft_method="prefix",
            model_path="bert-base-uncased",
            finetuned_path=str(prefix_dir),
            lora_r=None,
            defense="lrb",
            task="seq_class",
            lora_target_modules=None,
            peft_num_virtual_tokens=None,
        )
        try:
            validate_peft_eval_args(args)
        except NotImplementedError as exc:
            msg = str(exc)
            assert_true("training-only" in msg, "Prefix DAGER rejection should say prefix is training-only")
            assert_true(
                "excluded from DAGER/partial-gradient eval matrices" in msg,
                "Prefix DAGER rejection should mention the v1 eval matrix exclusion",
            )
        else:
            raise AssertionError("Prefix PEFT eval should be rejected until DAGER span support exists")
    finally:
        for child in prefix_dir.iterdir():
            child.unlink()
        prefix_dir.rmdir()


def test_prefix_virtual_token_override_validates_positive():
    try:
        resolve_peft_config(
            model_path="bert-base-uncased",
            peft_method="prefix",
            task="seq_class",
            peft_num_virtual_tokens=0,
        )
    except ValueError as exc:
        assert_true("positive" in str(exc), "zero prefix virtual tokens should fail validation")
    else:
        raise AssertionError("zero prefix virtual tokens should not silently fall back to default")


def test_prefix_virtual_token_override_rejects_adapter_mismatch():
    prefix_dir = Path(_temp_peft_adapter_dir("PREFIX_TUNING", model_path="bert-base-uncased", num_virtual_tokens=8))
    try:
        try:
            resolve_peft_config(
                model_path="bert-base-uncased",
                peft_method="prefix",
                checkpoint_path=str(prefix_dir),
                task="seq_class",
                peft_num_virtual_tokens=20,
            )
        except ValueError as exc:
            assert_true("peft_num_virtual_tokens" in str(exc), "prefix token mismatch should mention the flag")
        else:
            raise AssertionError("prefix token mismatch should fail validation")
    finally:
        for child in prefix_dir.iterdir():
            child.unlink()
        prefix_dir.rmdir()


def test_adapter_is_exposed_as_cli_choice():
    adapter_dir = Path(_temp_peft_adapter_dir("ADAPTER", model_path="bert-base-uncased"))
    try:
        parsed = args_factory.get_args([
            "--dataset", "sst2",
            "--task", "seq_class",
            "--split", "val",
            "--n_inputs", "1",
            "--l1_filter", "all",
            "--l2_filter", "non-overlap",
            "--train_method", "peft",
            "--peft_method", "adapter",
            "--finetuned_path", str(adapter_dir),
        ])
        assert_true(parsed.peft_method == "adapter", "attack CLI should accept adapter")
    finally:
        for child in adapter_dir.iterdir():
            child.unlink()
        adapter_dir.rmdir()

    parsed = train.build_parser().parse_args([
        "--train_method", "peft",
        "--peft_method", "adapter",
    ])
    assert_true(parsed.peft_method == "adapter", "train CLI should accept adapter")


def test_peft_eval_scope_helper_classifies_v1_policy():
    assert_true(peft_eval_scope("lora") == PEFT_EVAL_SCOPE_DAGER, "LoRA should be in PEFT DAGER eval scope")
    assert_true(peft_eval_scope("ia3") == PEFT_EVAL_SCOPE_DAGER, "IA3 should be in PEFT DAGER eval scope")
    assert_true(peft_eval_scope(None) == PEFT_EVAL_SCOPE_NA, "Non-PEFT runs should report n/a eval scope")
    assert_true(
        peft_eval_scope("prefix") == PEFT_EVAL_SCOPE_TRAINING_ONLY,
        "Prefix should be training-only in v1",
    )
    assert_true(
        peft_eval_scope("adapter") == PEFT_EVAL_SCOPE_DAGER,
        "Houlsby-style adapter should be in PEFT DAGER eval scope",
    )
    assert_true(
        "training-only" in peft_eval_scope_message("prefix"),
        "Prefix scope message should be explicit",
    )
    assert_true(
        "supports this method" in peft_eval_scope_message("adapter"),
        "Adapter scope message should be explicit",
    )


def test_privacy_matrix_includes_adapter():
    in_scope = [method for method in ("lora", "ia3", "prefix", "adapter") if peft_eval_scope(method) == PEFT_EVAL_SCOPE_DAGER]
    assert_true(in_scope == ["lora", "ia3", "adapter"], f"PEFT privacy matrix should include LoRA/IA3/adapter: {in_scope}")
    assert_true(
        peft_eval_scope("prefix") == PEFT_EVAL_SCOPE_TRAINING_ONLY,
        "Prefix should remain training/smoke only",
    )


def test_peft_gradient_inventory_deduplicates_transformer_layers():
    names = [
        "base_model.model.bert.encoder.layer.0.attention.self.query.ia3_l.default",
        "base_model.model.bert.encoder.layer.0.attention.self.value.ia3_l.default",
        "base_model.model.bert.encoder.layer.1.attention.self.query.ia3_l.default",
        "base_model.model.bert.encoder.layer.1.attention.self.value.ia3_l.default",
        "base_model.model.classifier.modules_to_save.default.weight",
    ]
    ia3_selected = select_peft_gradient_indices(
        names,
        peft_method="ia3",
        target_modules="query,value",
        preferred_modules=["query"],
    )
    assert_true(
        ia3_selected == [(0, names[0]), (2, names[2])],
        f"IA3 inventory should select one preferred tensor per transformer layer: {ia3_selected}",
    )

    lora_names = [
        "base_model.model.bert.encoder.layer.0.attention.self.query.lora_A.default.weight",
        "base_model.model.bert.encoder.layer.0.attention.self.value.lora_A.default.weight",
        "base_model.model.bert.encoder.layer.1.attention.self.query.lora_A.default.weight",
        "base_model.model.bert.encoder.layer.1.attention.self.value.lora_A.default.weight",
    ]
    lora_selected = select_lora_gradient_indices(
        lora_names,
        target_modules="query,value",
        preferred_modules=["query"],
    )
    assert_true(
        lora_selected == [(0, lora_names[0]), (2, lora_names[2])],
        f"LoRA inventory should select one preferred tensor per transformer layer: {lora_selected}",
    )

    adapter_names = [
        "bert.encoder.layer.0.output.adapters.default.adapter_up.weight",
        "bert.encoder.layer.0.output.adapters.default.adapter_down.0.weight",
        "bert.encoder.layer.1.output.adapters.default.adapter_down.0.weight",
        "bert.encoder.layer.1.output.adapters.default.adapter_up.weight",
        "classifier.modules_to_save.default.weight",
    ]
    adapter_selected = select_peft_gradient_indices(adapter_names, peft_method="adapter")
    assert_true(
        adapter_selected == [(1, adapter_names[1]), (2, adapter_names[2])],
        f"Adapter inventory should prefer one down-projection tensor per layer: {adapter_selected}",
    )


def test_lora_training_allows_post_gradient_lrb():
    class DummyConfig:
        model_type = "gpt2"

    dummy_model = SimpleNamespace(config=DummyConfig(), transformer=object(), score=object())
    args = SimpleNamespace(train_method="peft", peft_method="lora", defense="lrb")
    wrapper = prepare_training_defense(model=dummy_model, args=args, trainable_params=[])
    assert_true(wrapper is not None, "LoRA training should allow post-gradient LRB")


def test_lora_ia3_training_allow_direct_and_dager_defenses():
    for peft_method in ("lora", "ia3", "adapter"):
        for defense in ("dpsgd", "soteria", "mixup", "dager"):
            args = SimpleNamespace(train_method="peft", peft_method=peft_method, defense=defense)
            wrapper = prepare_training_defense(model=object(), args=args, trainable_params=[])
            assert_true(
                wrapper is not None,
                f"{peft_method} training should allow {defense} defense",
            )


def test_prefix_training_rejects_direct_and_dager_defenses():
    for defense in ("dpsgd", "soteria", "mixup", "dager"):
        args = SimpleNamespace(train_method="peft", peft_method="prefix", defense=defense)
        try:
            prepare_training_defense(model=object(), args=args, trainable_params=[])
        except NotImplementedError as exc:
            assert_true(
                "Prefix PEFT training currently supports only post-gradient defenses" in str(exc),
                "Prefix direct/DAGER defense rejection should be explicit",
            )
        else:
            raise AssertionError(f"Prefix training should reject {defense}")


def main():
    tests = [
        test_lora_target_modules_match_supported_families,
        test_parse_lora_target_module_presets,
        test_seq_class_modules_to_save_include_classifier_heads,
        test_resolve_lora_checkpoint_accepts_peft_adapter_dir,
        test_validate_lora_eval_args_accepts_supported_gpt2_setup,
        test_validate_lora_eval_args_accepts_direct_generation_baselines,
        test_validate_lora_eval_args_accepts_peft_adapter_dir,
        test_validate_lora_eval_args_rejects_adapter_rank_mismatch,
        test_validate_lora_eval_args_rejects_adapter_target_mismatch,
        test_legacy_lora_checkpoint_requires_rank,
        test_apply_lora_config_to_args_validates_adapter_task_type,
        test_lora_gradient_inventory_prefers_adapter_a_and_excludes_saved_heads,
        test_validate_lora_eval_args_rejects_unsupported_defense,
        test_validate_lora_eval_args_rejects_unsupported_model_family,
        test_bert_lora_eval_args_are_supported,
        test_ia3_and_prefix_adapter_metadata_normalize,
        test_adapter_method_resolves_default_reduction_factor,
        test_adapter_metadata_normalizes_from_adapterhub_architecture,
        test_adapter_backend_smoke_save_and_reload_with_fake_backend,
        test_prefix_eval_is_rejected_for_dager_span,
        test_prefix_virtual_token_override_validates_positive,
        test_prefix_virtual_token_override_rejects_adapter_mismatch,
        test_adapter_is_exposed_as_cli_choice,
        test_peft_eval_scope_helper_classifies_v1_policy,
        test_privacy_matrix_includes_adapter,
        test_peft_gradient_inventory_deduplicates_transformer_layers,
        test_lora_training_allows_post_gradient_lrb,
        test_lora_ia3_training_allow_direct_and_dager_defenses,
        test_prefix_training_rejects_direct_and_dager_defenses,
    ]
    for test in tests:
        print(f"Running {test.__name__}...")
        test()
    print("All PEFT eval semantic tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
