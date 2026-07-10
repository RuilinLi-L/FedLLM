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


def test_rouge_backends_do_not_mix_in_privacy_aggregation():
    base = {
        "log_kind": "attack_dager",
        "attack": "partial_transformer_gradients",
        "dataset": "sst2",
        "batch_size": "1",
        "train_method": "full",
        "defense": "none",
        "defense_param_name": "n/a",
        "defense_param_value": "n/a",
        "result_status": "ok",
        "n_inputs_requested": "1",
        "n_inputs_completed": "1",
        "gradient_layer_subset": "first2",
        "gradient_param_filter": "all",
        "partial_attack_variant": "ptg_gradient_matching",
        "rec_token_mean": "0.1",
    }
    rows = [
        {
            **base,
            "rouge_backend": "datasets",
            "ptg_rouge_backend_requested": "datasets",
            "agg_r1fm_r2fm": "15",
        },
        {
            **base,
            "rouge_backend": "simple_ngram",
            "ptg_rouge_backend_requested": "simple_ngram",
            "agg_r1fm_r2fm": "30",
        },
    ]

    anchors = build_attack_anchor_results(rows)
    assert_true(len(anchors) == 2, f"ROUGE backends must not aggregate together: {anchors}")
    by_backend = {row["rouge_backend"]: row for row in anchors}
    assert_true(by_backend["datasets"]["agg_r1fm_r2fm"] == "15.000000", "datasets ROUGE should remain isolated")
    assert_true(
        by_backend["simple_ngram"]["agg_r1fm_r2fm"] == "30.000000",
        "simple-ngram ROUGE should remain isolated",
    )


def test_rouge_backends_do_not_overwrite_each_other_in_tradeoff_results():
    utility = {
        "log_kind": "train",
        "dataset": "sst2",
        "batch_size": "1",
        "train_method": "full",
        "defense": "none",
        "defense_param_name": "n/a",
        "defense_param_value": "n/a",
        "result_status": "ok",
        "eval_accuracy": "0.9",
    }
    attack = {
        "log_kind": "attack_dager",
        "attack": "partial_transformer_gradients",
        "dataset": "sst2",
        "batch_size": "1",
        "train_method": "full",
        "defense": "none",
        "defense_param_name": "n/a",
        "defense_param_value": "n/a",
        "result_status": "ok",
        "n_inputs_requested": "1",
        "n_inputs_completed": "1",
        "gradient_layer_subset": "first2",
        "gradient_param_filter": "all",
        "partial_attack_variant": "ptg_gradient_matching",
        "rec_token_mean": "0.1",
    }
    rows = [
        utility,
        {**attack, "rouge_backend": "datasets", "agg_r1fm_r2fm": "15"},
        {**attack, "rouge_backend": "simple_ngram", "agg_r1fm_r2fm": "30"},
    ]

    tradeoff = build_privacy_utility_tradeoff(rows)
    assert_true(len(tradeoff) == 2, f"ROUGE backends must not overwrite each other: {tradeoff}")
    by_backend = {row["rouge_backend"]: row for row in tradeoff}
    assert_true(by_backend["datasets"]["agg_r1fm_r2fm"] == "15.000000", "datasets tradeoff should remain isolated")
    assert_true(
        by_backend["simple_ngram"]["agg_r1fm_r2fm"] == "30.000000",
        "simple-ngram tradeoff should remain isolated",
    )


def test_ptg_attack_surface_and_losses_are_preserved():
    rows = [
        {
            "log_kind": "attack_dager",
            "attack": "partial_transformer_gradients",
            "dataset": "sst2",
            "batch_size": "1",
            "train_method": "full",
            "defense": "none",
            "defense_param_value": "n/a",
            "result_status": "ok",
            "n_inputs_requested": "1",
            "n_inputs_completed": "1",
            "gradient_layer_subset": "all",
            "gradient_param_filter": "query_only",
            "partial_attack_variant": "ptg_gradient_matching",
            "rec_token_mean": "0.25",
            "ptg_initial_loss": "0.9",
            "ptg_final_loss": "0.3",
            "ptg_loss_reduction": "0.6",
            "ptg_embed_norm_weight": "0.01",
            "ptg_fix_special_tokens": "true",
            "ptg_parity_mode": "source",
            "ptg_dpsgd_mode": "source_opacus",
            "grad_type": "attn_qkv",
            "attack_layer": "0",
            "ptg_optimizer": "bert-adam",
            "ptg_init": "random",
            "ptg_init_candidates": "500",
            "ptg_know_padding": "true",
            "ptg_lm_loss": "1.2",
            "fixed_token_count": "1",
            "selected_gradient_count": "2",
            "selected_gradient_names": "encoder.layer.0.attention.self.query.weight;encoder.layer.0.attention.self.query.bias",
        }
    ]

    anchors = build_attack_anchor_results(rows)
    assert_true(len(anchors) == 1, f"one PTG anchor should be built: {anchors}")
    anchor = anchors[0]
    assert_true(anchor["attack_surface"] == "partial_gradient", "PTG should be grouped as a partial-gradient surface")
    assert_true(
        anchor["partial_attack_variant"] == "ptg_gradient_matching",
        "PTG variant should remain distinct from DAGER visible-gradient variants",
    )
    assert_true(anchor["ptg_final_loss"] == "0.300000", "PTG final loss should be aggregated")
    assert_true(anchor["ptg_loss_reduction"] == "0.600000", "PTG loss reduction should be aggregated")
    assert_true(anchor["ptg_lm_loss"] == "1.200000", "PTG LM prior loss should be aggregated")
    assert_true(anchor["fixed_token_count"] == "1.000000", "fixed token count should be aggregated")
    assert_true(anchor["ptg_embed_norm_weight"] == "0.01", "PTG embed norm weight should be retained")
    assert_true(anchor["ptg_parity_mode"] == "source", "PTG source parity mode should be retained")
    assert_true(anchor["ptg_dpsgd_mode"] == "source_opacus", "source DP-SGD mode should be retained")
    assert_true(anchor["grad_type"] == "attn_qkv", "source grad_type should be retained")
    assert_true(anchor["attack_layer"] == "0", "source attack_layer should be retained")
    assert_true(anchor["ptg_optimizer"] == "bert-adam", "PTG optimizer should be retained")
    assert_true(anchor["ptg_init"] == "random", "PTG init strategy should be retained")
    assert_true(anchor["ptg_init_candidates"] == "500", "PTG init candidates should be retained")
    assert_true(anchor["ptg_know_padding"] == "true", "PTG padding-knowledge flag should be retained")
    assert_true(anchor["selected_gradient_count"] == "2", "selected gradient count should be retained")


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


def test_full_variant_log_with_per_input_stats_parses_once():
    text = """
===== VARIANT START defense=compression param=8 dataset=rotten_tomatoes batch=2 model=gpt2 start=now =====
===== RAW ATTACK OUTPUT START =====
Running input #0 of 2.
Done with input #0 of 2.
reference:
========================
good movie
========================
predicted:
========================
good movie
========================
[Curr input metrics]:
rouge1     | fm: 100.000 | p: 100.000 | r: 100.000
rouge2     | fm: 100.000 | p: 100.000 | r: 100.000
rougeL     | fm: 100.000 | p: 100.000 | r: 100.000
rougeLsum  | fm: 100.000 | p: 100.000 | r: 100.000
r1fm+r2fm = 200.000

[Aggregate metrics]:
rouge1     | fm: 100.000 | p: 100.000 | r: 100.000
rouge2     | fm: 100.000 | p: 100.000 | r: 100.000
rougeL     | fm: 100.000 | p: 100.000 | r: 100.000
rougeLsum  | fm: 100.000 | p: 100.000 | r: 100.000
r1fm+r2fm = 200.000

input #0 time: 0:00:01 | total time: 0:00:02
Done with all.
[Per-input metric statistics]:
metric                   | count |       mean |        std |        var |        min |        p25 |     median |        p75 |        max
input_rouge1_fm          |     2 |  75.000000 |  35.355339 | 1250.000000 |  50.000000 |  62.500000 |  75.000000 |  87.500000 | 100.000000
rec_token                |     2 |   0.750000 |   0.353553 |   0.125000 |   0.500000 |   0.625000 |   0.750000 |   0.875000 |   1.000000

===== RESULT SUMMARY START =====
summary_version=3
result_status=ok
dataset=rotten_tomatoes
split=val
task=seq_class
model_path=gpt2
batch_size=2
train_method=full
defense=compression
defense_param_value=8
n_inputs_requested=2
n_inputs_completed=2
last_rec_status=ok
rec_token_mean=0.750000
rec_token_count=2
rec_token_std=0.353553
rec_token_var=0.125000
input_rouge1_fm_mean=75.000000
input_rouge1_fm_std=35.355339
input_rouge1_fm_var=1250.000000
input_r1fm_r2fm_mean=150.000000
input_r1fm_r2fm_std=70.710678
input_r1fm_r2fm_var=5000.000000
agg_rouge1_fm=100.000000
agg_rouge2_fm=100.000000
agg_r1fm_r2fm=200.000000
===== RESULT SUMMARY END =====
===== RAW ATTACK OUTPUT END =====
===== VARIANT END end=later exit_code=0 =====
""".strip()

    rows = classify_and_parse(Path("compression_8.txt"), text)

    assert_true(len(rows) == 1, f"full variant log should parse exactly once: {rows}")
    assert_true(rows[0]["log_kind"] == "attack_dager", "variant log should remain an attack row")
    assert_true(rows[0]["summary_version"] == "3", "new attack summary version should be preserved")
    assert_true(rows[0]["rec_token_var"] == "0.125000", "rec token variance should be parsed")
    assert_true(rows[0]["input_rouge1_fm_var"] == "1250.000000", "per-input ROUGE variance should be parsed")
    assert_true(rows[0]["input_r1fm_r2fm_std"] == "70.710678", "per-input r1+r2 std should be parsed")


def main():
    tests = [
        test_prefix_utility_rows_are_training_only_not_failed_privacy,
        test_prefix_scope_is_inferred_without_explicit_scope_field,
        test_prefix_scope_is_inferred_from_adapter_metadata,
        test_prefix_scope_ignores_na_placeholders_when_metadata_is_available,
        test_adapter_scope_is_supported_and_keeps_reduction_factor,
        test_partial_attack_surfaces_do_not_mix_in_privacy_aggregation,
        test_rouge_backends_do_not_mix_in_privacy_aggregation,
        test_rouge_backends_do_not_overwrite_each_other_in_tradeoff_results,
        test_ptg_attack_surface_and_losses_are_preserved,
        test_unsupported_partial_attack_status_is_preserved,
        test_adaptive_fallback_summary_stays_in_adaptive_group,
        test_full_variant_log_with_per_input_stats_parses_once,
    ]
    for test in tests:
        print(f"Running {test.__name__}...")
        test()
    print("All collect_experiment_logs semantic tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
