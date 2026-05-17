#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.collect_experiment_logs import build_privacy_utility_tradeoff, build_utility_results


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def test_prefix_utility_rows_are_training_only_not_failed_privacy():
    rows = [
        {
            "log_kind": "train",
            "dataset": "sst2",
            "batch_size": "2",
            "train_method": "peft",
            "peft_method": "prefix",
            "peft_eval_scope": "training_only",
            "lora_r": "n/a",
            "lora_target_modules": "n/a",
            "rep_bottleneck_type": "none",
            "rep_keep_ratio": "n/a",
            "rep_dropout_p": "n/a",
            "defense": "none",
            "defense_param_name": "n/a",
            "defense_param_value": "n/a",
            "result_status": "ok",
            "eval_accuracy": "0.900000",
        }
    ]

    utility = build_utility_results(rows)
    assert_true(len(utility) == 1, "one prefix utility summary should be built")
    assert_true(utility[0]["peft_eval_scope"] == "training_only", "prefix utility should retain training-only scope")

    tradeoff = build_privacy_utility_tradeoff(rows)
    assert_true(len(tradeoff) == 1, "one prefix tradeoff row should be built")
    assert_true(tradeoff[0]["peft_eval_scope"] == "training_only", "prefix tradeoff should retain training-only scope")
    assert_true(
        tradeoff[0]["privacy_eval_status"] == "training_only",
        "prefix utility should not be treated as a failed missing privacy run",
    )
    assert_true(
        tradeoff[0]["privacy_eval_note"] == "not_in_v1_dager_partial_eval_scope",
        "prefix utility should explain why no privacy row is joined",
    )
    assert_true(tradeoff[0]["privacy_score"] == "", "prefix training-only row should not invent a privacy score")


def test_prefix_scope_is_inferred_without_explicit_scope_field():
    rows = [
        {
            "log_kind": "train",
            "dataset": "sst2",
            "batch_size": "2",
            "train_method": "peft",
            "peft_method": "prefix",
            "lora_r": "n/a",
            "lora_target_modules": "n/a",
            "rep_bottleneck_type": "none",
            "rep_keep_ratio": "n/a",
            "rep_dropout_p": "n/a",
            "defense": "none",
            "defense_param_name": "n/a",
            "defense_param_value": "n/a",
            "result_status": "ok",
            "eval_accuracy": "0.900000",
        }
    ]

    tradeoff = build_privacy_utility_tradeoff(rows)
    assert_true(tradeoff[0]["peft_eval_scope"] == "training_only", "prefix method should infer training-only scope")
    assert_true(tradeoff[0]["privacy_eval_status"] == "training_only", "inferred prefix scope should mark privacy status")


def test_prefix_scope_is_inferred_from_adapter_metadata():
    rows = [
        {
            "log_kind": "train",
            "dataset": "sst2",
            "batch_size": "2",
            "train_method": "peft",
            "peft_adapter_peft_type": "PREFIX_TUNING",
            "lora_r": "n/a",
            "lora_target_modules": "n/a",
            "rep_bottleneck_type": "none",
            "rep_keep_ratio": "n/a",
            "rep_dropout_p": "n/a",
            "defense": "none",
            "defense_param_name": "n/a",
            "defense_param_value": "n/a",
            "result_status": "ok",
            "eval_accuracy": "0.900000",
        }
    ]

    utility = build_utility_results(rows)
    assert_true(utility[0]["peft_method"] == "prefix", "adapter metadata should infer peft_method=prefix")
    assert_true(utility[0]["peft_eval_scope"] == "training_only", "adapter metadata should infer training-only scope")

    tradeoff = build_privacy_utility_tradeoff(rows)
    assert_true(tradeoff[0]["peft_eval_scope"] == "training_only", "metadata-derived prefix should stay training-only")
    assert_true(tradeoff[0]["privacy_eval_status"] == "training_only", "metadata-derived prefix should mark privacy status")


def test_prefix_scope_ignores_na_placeholders_when_metadata_is_available():
    rows = [
        {
            "log_kind": "train",
            "dataset": "sst2",
            "batch_size": "2",
            "train_method": "peft",
            "peft_method": "n/a",
            "peft_eval_scope": "n/a",
            "peft_adapter_peft_type": "PREFIX_TUNING",
            "lora_r": "n/a",
            "lora_target_modules": "n/a",
            "rep_bottleneck_type": "none",
            "rep_keep_ratio": "n/a",
            "rep_dropout_p": "n/a",
            "defense": "none",
            "defense_param_name": "n/a",
            "defense_param_value": "n/a",
            "result_status": "ok",
            "eval_accuracy": "0.900000",
        }
    ]

    tradeoff = build_privacy_utility_tradeoff(rows)
    assert_true(tradeoff[0]["peft_method"] == "prefix", "n/a method placeholder should not hide prefix metadata")
    assert_true(tradeoff[0]["peft_eval_scope"] == "training_only", "n/a scope placeholder should not hide prefix metadata")


def main():
    tests = [
        test_prefix_utility_rows_are_training_only_not_failed_privacy,
        test_prefix_scope_is_inferred_without_explicit_scope_field,
        test_prefix_scope_is_inferred_from_adapter_metadata,
        test_prefix_scope_ignores_na_placeholders_when_metadata_is_available,
    ]
    for test in tests:
        print(f"Running {test.__name__}...")
        test()
    print("All collect_experiment_logs semantic tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
