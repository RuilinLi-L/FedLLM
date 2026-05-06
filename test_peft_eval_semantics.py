#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import prepare_training_defense
from utils.peft_utils import (
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
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    torch.save({"dummy": torch.tensor([1.0])}, path / "adapter_model.bin")
    return str(path)


def test_lora_target_modules_match_supported_families():
    assert_true(lora_target_modules("gpt2") == ["c_attn"], "GPT-2 LoRA target module should be c_attn")
    assert_true(
        lora_target_modules("meta-llama/Meta-Llama-3.1-8B") == ["q_proj"],
        "Llama LoRA target module should be q_proj",
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


def test_validate_lora_eval_args_accepts_peft_adapter_dir():
    adapter_dir = Path(_temp_adapter_dir())
    try:
        args = SimpleNamespace(
            train_method="lora",
            model_path="gpt2",
            finetuned_path=str(adapter_dir),
            lora_r=16,
            defense="lrb",
        )
        validated = validate_lora_eval_args(args)
        assert_true(validated is args, "validation should accept PEFT adapter directories")
    finally:
        for child in adapter_dir.iterdir():
            child.unlink()
        adapter_dir.rmdir()


def test_validate_lora_eval_args_rejects_unsupported_defense():
    checkpoint = _temp_checkpoint()
    try:
        args = SimpleNamespace(
            train_method="lora",
            model_path="gpt2",
            finetuned_path=checkpoint,
            lora_r=16,
            defense="soteria",
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
        raise AssertionError("LoRA training should reject DP-SGD until per-example path is supported")


def main():
    tests = [
        test_lora_target_modules_match_supported_families,
        test_seq_class_modules_to_save_include_classifier_heads,
        test_resolve_lora_checkpoint_accepts_peft_adapter_dir,
        test_validate_lora_eval_args_accepts_supported_gpt2_setup,
        test_validate_lora_eval_args_accepts_peft_adapter_dir,
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
