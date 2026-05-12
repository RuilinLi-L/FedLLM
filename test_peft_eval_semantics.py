#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import prepare_training_defense
from utils.models import select_lora_gradient_indices
from utils.peft_utils import (
    apply_lora_config_to_args,
    parse_lora_target_modules,
    is_peft_adapter_dir,
    lora_modules_to_save,
    lora_target_modules,
    resolve_lora_checkpoint_path,
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
    path = Path(tempfile.mkdtemp())
    config = {
        "peft_type": "LORA",
        "r": 16,
        "target_modules": ["c_attn"],
        "task_type": "SEQ_CLS",
        "base_model_name_or_path": "gpt2",
    }
    (path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    torch.save({"dummy": torch.tensor([1.0])}, path / "adapter_model.bin")
    return str(path)


def test_lora_target_modules_match_supported_families():
    assert_true(lora_target_modules("gpt2") == ["c_attn"], "GPT-2 LoRA target module should be c_attn")
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
            model_path="gpt2",
            finetuned_path=checkpoint,
            lora_r=16,
            defense="lrb",
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
                model_path="gpt2",
                finetuned_path=checkpoint,
                lora_r=16,
                defense=defense,
                task="seq_class",
                lora_target_modules=None,
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
            model_path="gpt2",
            finetuned_path=str(adapter_dir),
            lora_r=None,
            defense="lrb",
            task="seq_class",
            lora_target_modules=None,
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
            model_path="gpt2",
            finetuned_path=str(adapter_dir),
            lora_r=8,
            defense="lrb",
            task="seq_class",
            lora_target_modules=None,
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
            model_path="gpt2",
            finetuned_path=str(adapter_dir),
            lora_r=16,
            defense="lrb",
            task="seq_class",
            lora_target_modules="all-linear",
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
            model_path="gpt2",
            finetuned_path=checkpoint,
            lora_r=None,
            defense="lrb",
            task="seq_class",
            lora_target_modules=None,
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
            model_path="gpt2",
            finetuned_path=str(adapter_dir),
            lora_r=None,
            task="next_token_pred",
            lora_target_modules=None,
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
            model_path="gpt2",
            finetuned_path=checkpoint,
            lora_r=16,
            defense="dager",
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
            model_path="bert-base-uncased",
            finetuned_path=checkpoint,
            lora_r=16,
            defense="lrb",
        )
        try:
            validate_lora_eval_args(args)
        except ValueError as exc:
            assert_true("GPT-2 and Llama" in str(exc), "unsupported model family error should mention supported families")
        else:
            raise AssertionError("unsupported LoRA model should fail validation")
    finally:
        Path(checkpoint).unlink(missing_ok=True)


def test_lora_training_allows_post_gradient_lrb():
    args = SimpleNamespace(train_method="lora", defense="lrb")
    wrapper = prepare_training_defense(model=object(), args=args, trainable_params=[])
    assert_true(wrapper is not None, "LoRA training should allow post-gradient LRB")


def test_lora_training_rejects_direct_generation_defenses():
    args = SimpleNamespace(train_method="lora", defense="dpsgd")
    try:
        prepare_training_defense(model=object(), args=args, trainable_params=[])
    except NotImplementedError as exc:
        assert_true("does not currently support" in str(exc), "LoRA direct defense rejection should be explicit")
    else:
        raise AssertionError("LoRA training should reject DP-SGD-style direct defenses until the per-example path is supported")


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
        test_lora_training_allows_post_gradient_lrb,
        test_lora_training_rejects_direct_generation_defenses,
    ]
    for test in tests:
        print(f"Running {test.__name__}...")
        test()
    print("All PEFT eval semantic tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
