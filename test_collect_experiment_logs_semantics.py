#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.collect_experiment_logs import build_privacy_utility_tradeoff, build_utility_results
from scripts.collect_experiment_logs import classify_and_parse, build_attack_anchor_results


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


def test_adapter_scope_is_supported_and_keeps_reduction_factor():
    rows = [
        {
            "log_kind": "train",
            "dataset": "sst2",
            "batch_size": "2",
            "train_method": "peft",
            "peft_method": "adapter",
            "peft_eval_scope": "dager_eval",
            "adapter_reduction_factor": "16",
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
    assert_true(utility[0]["peft_eval_scope"] == "dager_eval", "adapter utility should be in DAGER eval scope")
    assert_true(utility[0]["adapter_reduction_factor"] == "16", "adapter reduction factor should be retained")

    tradeoff = build_privacy_utility_tradeoff(rows)
    assert_true(tradeoff[0]["peft_eval_scope"] == "dager_eval", "adapter should keep DAGER eval scope")
    assert_true(tradeoff[0].get("privacy_eval_status", "") == "", "missing adapter privacy row should remain empty, not v2 planned")


def test_partial_attack_surfaces_do_not_mix_in_privacy_aggregation():
    rows = [
        {
            "log_kind": "attack_dager",
            "dataset": "sst2",
            "batch_size": "2",
            "train_method": "full",
            "defense": "none",
            "defense_param_value": "n/a",
            "result_status": "ok",
            "n_inputs_requested": "1",
            "n_inputs_completed": "1",
            "rec_token_mean": "0.8",
            "gradient_layer_subset": "all",
            "gradient_param_filter": "all",
            "partial_attack_variant": "full_gradient_visible",
        },
        {
            "log_kind": "attack_dager",
            "dataset": "sst2",
            "batch_size": "2",
            "train_method": "full",
            "defense": "none",
            "defense_param_value": "n/a",
            "result_status": "ok",
            "n_inputs_requested": "1",
            "n_inputs_completed": "1",
            "rec_token_mean": "0.2",
            "gradient_layer_subset": "all",
            "gradient_param_filter": "qkv_only",
            "partial_attack_variant": "dager_qkv_visible",
        },
    ]

    anchors = build_attack_anchor_results(rows)
    assert_true(len(anchors) == 2, f"full and qkv partial attacks should not aggregate together: {anchors}")
    surfaces = sorted((row["attack_surface"], row["partial_attack_variant"]) for row in anchors)
    assert_true(
        surfaces == [("full_gradient", "full_gradient_visible"), ("partial_gradient", "dager_qkv_visible")],
        f"unexpected attack surface grouping: {surfaces}",
    )


def test_unsupported_partial_attack_status_is_preserved():
    rows = [
        {
            "log_kind": "attack_dager",
            "dataset": "sst2",
            "batch_size": "2",
            "train_method": "full",
            "defense": "none",
            "defense_param_value": "n/a",
            "result_status": "unsupported",
            "n_inputs_requested": "1",
            "n_inputs_completed": "0",
            "gradient_layer_subset": "last1",
            "gradient_param_filter": "all",
            "partial_attack_variant": "unsupported_nonprefix_dager",
            "unsupported_reason": "nonprefix_dager_requires_at_least_two_visible_layers",
        }
    ]

    anchors = build_attack_anchor_results(rows)
    assert_true(anchors[0]["result_status"] == "unsupported", "unsupported partial exposure should not become failed")
    assert_true(
        anchors[0]["partial_attack_variant"] == "unsupported_nonprefix_dager",
        "unsupported partial variant should remain visible in the aggregate row",
    )


def test_adaptive_fallback_summary_stays_in_adaptive_group():
    text = """
===== VARIANT START defense=lrbprojonly param=lrbprojonly@k=0.5 dataset=sst2 batch=2 model=gpt2 start=now =====
===== RESULT SUMMARY START =====
summary_version=2
result_status=failed
dataset=sst2
batch_size=2
train_method=full
defense=lrbprojonly
defense_param_value=0.500000
n_inputs_requested=3
n_inputs_completed=0
error_type=runner_error
script_variant=lrbprojonly_0_5_adaptive
script_exit_code=137
===== RESULT SUMMARY END =====
===== VARIANT END end=later exit_code=137 =====
""".strip()

    rows = classify_and_parse(Path("lrbprojonly_0_5_adaptive.txt"), text)
    anchors = build_attack_anchor_results(rows)

    assert_true(rows[0]["adaptive_attack"] == "defense_aware", "adaptive fallback summary should infer adaptive attack")
    assert_true(anchors[0]["adaptive_attack"] == "defense_aware", "failed adaptive run should aggregate under adaptive group")
    assert_true(anchors[0]["result_status"] == "failed", "exit 137 fallback should not be successful")
    assert_true(anchors[0]["failed_or_incomplete_privacy_runs"] == "1", "failed adaptive run should be counted")


def main():
    tests = [
        test_prefix_utility_rows_are_training_only_not_failed_privacy,
        test_prefix_scope_is_inferred_without_explicit_scope_field,
        test_prefix_scope_is_inferred_from_adapter_metadata,
        test_prefix_scope_ignores_na_placeholders_when_metadata_is_available,
        test_adapter_scope_is_supported_and_keeps_reduction_factor,
        test_partial_attack_surfaces_do_not_mix_in_privacy_aggregation,
        test_unsupported_partial_attack_status_is_preserved,
        test_adaptive_fallback_summary_stays_in_adaptive_group,
    ]
    for test in tests:
        print(f"Running {test.__name__}...")
        test()
    print("All collect_experiment_logs semantic tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
