#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:
    torch = None

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import collect_experiment_logs as cel  # noqa: E402
if torch is not None:
    from utils.defense_common import defense_param_spec, grad_similarity_metrics  # noqa: E402
    from utils.dpsgd_opacus import dpsgd_opacus_summary_fields, record_dpsgd_opacus_summary  # noqa: E402


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def test_defense_param_spec_tracks_shared_cli_mapping():
    args = SimpleNamespace(
        defense="lrb",
        defense_lrb_keep_ratio_sensitive=0.2,
    )
    name, value = defense_param_spec(args)
    assert_true(name == "defense_lrb_keep_ratio_sensitive", "lrb should report its keep-ratio parameter")
    assert_true(float(value) == 0.2, "lrb parameter value should round-trip")

    args = SimpleNamespace(defense="dpsgd", defense_noise=5e-4)
    name, value = defense_param_spec(args)
    assert_true(name == "defense_noise", "dpsgd should report defense_noise")
    assert_true(float(value) == 5e-4, "dpsgd noise should round-trip")

    args = SimpleNamespace(defense="dpsgd_opacus", defense_noise=0.01)
    name, value = defense_param_spec(args)
    assert_true(name == "defense_noise", "dpsgd_opacus should report defense_noise")
    assert_true(float(value) == 0.01, "dpsgd_opacus noise multiplier should round-trip")


def test_dpsgd_opacus_summary_fields_include_accounting_terms():
    class FakePrivacyEngine:
        def get_epsilon(self, delta):
            return 3.25 + float(delta)

    args = SimpleNamespace(
        defense="dpsgd_opacus",
        defense_noise=0.01,
        defense_clip_norm=1.0,
        defense_dp_delta=1e-5,
    )
    tracker = {}
    record_dpsgd_opacus_summary(args, tracker, FakePrivacyEngine())
    fields = dict(dpsgd_opacus_summary_fields(args, tracker))

    assert_true(fields["dpsgd_noise_multiplier"] == 0.01, "summary should include Opacus noise multiplier")
    assert_true(fields["dpsgd_max_grad_norm"] == 1.0, "summary should include Opacus clipping bound")
    assert_true(fields["dpsgd_delta"] == 1e-5, "summary should include DP delta")
    assert_true(fields["dpsgd_accountant"] == "opacus_rdp", "summary should name the accountant")
    assert_true(abs(fields["dpsgd_epsilon"] - 3.25001) < 1e-9, "summary should include epsilon from privacy engine")


def test_grad_similarity_metrics_returns_cosine_and_norm_retention():
    base = (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))
    defended = (torch.tensor([0.5, 0.0]), torch.tensor([0.0, 0.5]))
    cosine, norm_retention = grad_similarity_metrics(base, defended)
    assert_true(abs(cosine - 1.0) < 1e-6, "scaled gradients should keep cosine 1")
    assert_true(abs(norm_retention - 0.5) < 1e-6, "norm retention should track overall scale")


def test_collect_parser_splits_multi_variant_attack_summaries():
    text = """
===== RESULT SUMMARY START =====
dataset=sst2
batch_size=2
train_method=lora
lora_r=16
defense=lrb
defense_param_name=defense_lrb_keep_ratio_sensitive
defense_param_value=0.200000
rec_token_mean=0.000000
agg_rouge1_fm=0.000000
agg_rouge2_fm=0.000000
===== RESULT SUMMARY END =====
===== RESULT SUMMARY START =====
dataset=sst2
batch_size=2
train_method=lora
lora_r=16
defense=lrb
defense_param_name=defense_lrb_keep_ratio_sensitive
defense_param_value=0.350000
rec_token_mean=0.010000
agg_rouge1_fm=1.000000
agg_rouge2_fm=0.500000
===== RESULT SUMMARY END =====
""".strip()
    rows = cel.classify_and_parse(Path("summary_lrb.txt"), text)
    assert_true(len(rows) == 2, "collector should emit one row per summary block")
    assert_true(rows[0]["defense_param_value"] == "0.200000", "first variant should be preserved")
    assert_true(rows[1]["defense_param_value"] == "0.350000", "second variant should be preserved")
    assert_true(rows[0]["train_method"] == "lora", "attack summary should preserve train_method")
    assert_true(rows[0]["lora_r"] == "16", "attack summary should preserve lora_r")


def test_collect_parser_reads_train_and_proxy_summaries():
    train_text = """
===== TRAIN RESULT SUMMARY START =====
dataset=sst2
batch_size=2
train_method=lora
lora_r=16
defense=lrb
defense_param_name=defense_lrb_keep_ratio_sensitive
defense_param_value=0.200000
seed=101
result_status=ok
eval_accuracy=0.850000
eval_macro_f1=0.849000
final_train_loss=0.410000
total_train_time=00:01:30
===== TRAIN RESULT SUMMARY END =====
metric eval: {'accuracy': 0.85, 'macro_f1': 0.849}
""".strip()
    proxy_text = """
===== PROXY UTILITY SUMMARY START =====
dataset=sst2
batch_size=2
defense=lrb
defense_param_name=defense_lrb_keep_ratio_sensitive
defense_param_value=0.200000
proxy_defense_semantics=native
grad_cosine_mean=0.750000
delta_val_accuracy_mean=-0.020000
===== PROXY UTILITY SUMMARY END =====
""".strip()
    train_rows = cel.classify_and_parse(Path("train_lrb.txt"), train_text)
    proxy_rows = cel.classify_and_parse(Path("proxy_lrb.txt"), proxy_text)
    assert_true(train_rows[0]["log_kind"] == "train", "train summary should classify as train")
    assert_true(proxy_rows[0]["log_kind"] == "proxy_utility", "proxy summary should classify as proxy_utility")
    assert_true(train_rows[0]["eval_accuracy"] == "0.850000", "train eval accuracy should come from summary block")
    assert_true(proxy_rows[0]["grad_cosine_mean"] == "0.750000", "proxy metrics should come from summary block")
    assert_true(proxy_rows[0]["proxy_defense_semantics"] == "native", "proxy semantics should be preserved")
    assert_true(train_rows[0]["train_method"] == "lora", "train summary should preserve train_method")
    assert_true(train_rows[0]["lora_r"] == "16", "train summary should preserve lora_r")


def test_tradeoff_join_uses_none_as_utility_anchor():
    rows = [
        {
            "log_kind": "train",
            "dataset": "sst2",
            "batch_size": "2",
            "train_method": "full",
            "lora_r": "",
            "defense": "none",
            "defense_param_name": "n/a",
            "defense_param_value": "n/a",
            "seed": "101",
            "result_status": "ok",
            "eval_accuracy": "0.900000",
            "eval_macro_f1": "0.900000",
            "final_train_loss": "0.400000",
            "total_train_time": "00:01:00",
        },
        {
            "log_kind": "train",
            "dataset": "sst2",
            "batch_size": "2",
            "train_method": "lora",
            "lora_r": "16",
            "defense": "none",
            "defense_param_name": "n/a",
            "defense_param_value": "n/a",
            "seed": "101",
            "result_status": "ok",
            "eval_accuracy": "0.900000",
            "eval_macro_f1": "0.900000",
            "final_train_loss": "0.400000",
            "total_train_time": "00:01:00",
        },
        {
            "log_kind": "train",
            "dataset": "sst2",
            "batch_size": "2",
            "train_method": "lora",
            "lora_r": "16",
            "defense": "lrb",
            "defense_param_name": "defense_lrb_keep_ratio_sensitive",
            "defense_param_value": "0.200000",
            "seed": "101",
            "result_status": "ok",
            "eval_accuracy": "0.880000",
            "eval_macro_f1": "0.878000",
            "final_train_loss": "0.420000",
            "total_train_time": "00:01:10",
        },
        {
            "log_kind": "attack_dager",
            "dataset": "sst2",
            "batch_size": "2",
            "train_method": "full",
            "lora_r": "",
            "defense": "none",
            "defense_param_name": "n/a",
            "defense_param_value": "n/a",
            "rec_token_mean": "0.900000",
            "agg_rouge1_fm": "80.000000",
            "agg_rouge2_fm": "70.000000",
        },
        {
            "log_kind": "attack_dager",
            "dataset": "sst2",
            "batch_size": "2",
            "train_method": "lora",
            "lora_r": "16",
            "defense": "none",
            "defense_param_name": "n/a",
            "defense_param_value": "n/a",
            "rec_token_mean": "0.900000",
            "agg_rouge1_fm": "80.000000",
            "agg_rouge2_fm": "70.000000",
        },
        {
            "log_kind": "attack_dager",
            "dataset": "sst2",
            "batch_size": "2",
            "train_method": "lora",
            "lora_r": "16",
            "defense": "lrb",
            "defense_param_name": "defense_lrb_keep_ratio_sensitive",
            "defense_param_value": "0.200000",
            "rec_token_mean": "0.000000",
            "agg_rouge1_fm": "0.000000",
            "agg_rouge2_fm": "0.000000",
        },
    ]
    tradeoff = cel.build_privacy_utility_tradeoff(rows)
    lrb_row = next(row for row in tradeoff if row.get("defense") == "lrb")
    assert_true(lrb_row["utility_drop"] == "0.020000", "utility drop should be none_accuracy - method_accuracy")
    assert_true(lrb_row["privacy_score"] == "1.000000", "privacy score should be 1 - rec_token_mean")
    assert_true(lrb_row["train_method"] == "lora", "tradeoff rows should keep train_method as a primary key")
    assert_true(lrb_row["lora_r"] == "16", "tradeoff rows should keep lora_r as a distinguishing field")


def test_tradeoff_join_distinguishes_peft_method_and_rep_bottleneck():
    rows = [
        {
            "log_kind": "train",
            "dataset": "sst2",
            "batch_size": "2",
            "train_method": "peft",
            "peft_method": "ia3",
            "lora_r": "",
            "lora_target_modules": "query,value,intermediate.dense",
            "rep_bottleneck_type": "projection",
            "rep_keep_ratio": "0.5",
            "rep_dropout_p": "0.1",
            "defense": "none",
            "defense_param_value": "n/a",
            "seed": "101",
            "result_status": "ok",
            "eval_accuracy": "0.800000",
            "eval_macro_f1": "0.800000",
            "total_train_time": "00:01:00",
        },
        {
            "log_kind": "train",
            "dataset": "sst2",
            "batch_size": "2",
            "train_method": "peft",
            "peft_method": "ia3",
            "lora_r": "",
            "lora_target_modules": "query,value,intermediate.dense",
            "rep_bottleneck_type": "projection",
            "rep_keep_ratio": "0.5",
            "rep_dropout_p": "0.1",
            "defense": "lrbprojonly",
            "defense_param_value": "0.500000",
            "seed": "101",
            "result_status": "ok",
            "eval_accuracy": "0.780000",
            "eval_macro_f1": "0.780000",
            "total_train_time": "00:01:05",
        },
        {
            "log_kind": "attack_dager",
            "dataset": "sst2",
            "batch_size": "2",
            "train_method": "peft",
            "peft_method": "ia3",
            "lora_r": "",
            "lora_target_modules": "query,value,intermediate.dense",
            "rep_bottleneck_type": "projection",
            "rep_keep_ratio": "0.5",
            "rep_dropout_p": "0.1",
            "defense": "lrbprojonly",
            "defense_param_value": "0.500000",
            "rec_token_mean": "0.250000",
        },
    ]
    tradeoff = cel.build_privacy_utility_tradeoff(rows)
    row = next(item for item in tradeoff if item.get("defense") == "lrbprojonly")
    assert_true(row["peft_method"] == "ia3", "tradeoff rows should keep PEFT method")
    assert_true(row["rep_bottleneck_type"] == "projection", "tradeoff rows should keep representation bottleneck")
    assert_true(row["utility_drop"] == "0.020000", "none anchor should match the same PEFT/rep setting")
    assert_true(row["privacy_score"] == "0.750000", "attack row should join through PEFT/rep keys")


def test_attack_anchor_keeps_adaptive_attack_separate():
    rows = [
        {
            "log_kind": "attack_dager",
            "dataset": "sst2",
            "batch_size": "2",
            "train_method": "full",
            "defense": "lrbprojonly",
            "defense_param_value": "0.500000",
            "adaptive_attack": "none",
            "adaptive_attack_profile": "none",
            "rec_token_mean": "0.000000",
        },
        {
            "log_kind": "attack_dager",
            "dataset": "sst2",
            "batch_size": "2",
            "train_method": "full",
            "defense": "lrbprojonly",
            "defense_param_value": "0.500000",
            "adaptive_attack": "defense_aware",
            "adaptive_attack_profile": "projection_span",
            "rec_token_mean": "0.250000",
        },
    ]
    anchors = cel.build_attack_anchor_results(rows)
    assert_true(len(anchors) == 2, "adaptive and non-adaptive attack rows should not be averaged together")
    adaptive = next(row for row in anchors if row.get("adaptive_attack") == "defense_aware")
    assert_true(adaptive["adaptive_attack_profile"] == "projection_span", "adaptive profile should survive aggregation")
    assert_true(adaptive["rec_token_mean"] == "0.250000", "adaptive row should keep its own leakage metric")


def test_tradeoff_join_emits_adaptive_and_nonadaptive_rows():
    rows = [
        {
            "log_kind": "train",
            "dataset": "sst2",
            "batch_size": "2",
            "train_method": "full",
            "lora_r": "",
            "defense": "none",
            "defense_param_value": "n/a",
            "result_status": "ok",
            "steps_completed": "10",
            "eval_accuracy": "0.900000",
        },
        {
            "log_kind": "train",
            "dataset": "sst2",
            "batch_size": "2",
            "train_method": "full",
            "lora_r": "",
            "defense": "lrbprojonly",
            "defense_param_value": "0.500000",
            "result_status": "ok",
            "steps_completed": "10",
            "eval_accuracy": "0.880000",
        },
        {
            "log_kind": "attack_dager",
            "dataset": "sst2",
            "batch_size": "2",
            "train_method": "full",
            "defense": "lrbprojonly",
            "defense_param_value": "0.500000",
            "adaptive_attack": "none",
            "adaptive_attack_profile": "none",
            "result_status": "ok",
            "n_inputs_requested": "3",
            "n_inputs_completed": "3",
            "rec_token_mean": "0.000000",
        },
        {
            "log_kind": "attack_dager",
            "dataset": "sst2",
            "batch_size": "2",
            "train_method": "full",
            "defense": "lrbprojonly",
            "defense_param_value": "0.500000",
            "adaptive_attack": "defense_aware",
            "adaptive_attack_profile": "projection_span",
            "result_status": "ok",
            "n_inputs_requested": "3",
            "n_inputs_completed": "3",
            "rec_token_mean": "0.250000",
        },
    ]
    tradeoff = cel.build_privacy_utility_tradeoff(rows)
    method_rows = [row for row in tradeoff if row.get("defense") == "lrbprojonly"]

    assert_true(len(method_rows) == 2, "tradeoff should include both adaptive and non-adaptive privacy rows")
    assert_true(
        {row.get("adaptive_attack") for row in method_rows} == {"none", "defense_aware"},
        "adaptive attack should distinguish tradeoff rows",
    )


def test_utility_aggregation_filters_failed_runs():
    rows = [
        {
            "log_kind": "train",
            "dataset": "sst2",
            "batch_size": "2",
            "train_method": "full",
            "defense": "lrbprojonly",
            "defense_param_value": "0.500000",
            "result_status": "ok",
            "steps_completed": "10",
            "eval_accuracy": "0.880000",
        },
        {
            "log_kind": "train",
            "dataset": "sst2",
            "batch_size": "2",
            "train_method": "full",
            "defense": "lrbprojonly",
            "defense_param_value": "0.500000",
            "result_status": "failed",
            "steps_completed": "0",
            "eval_accuracy": "0.100000",
        },
    ]
    utility = cel.build_utility_results(rows)
    row = utility[0]

    assert_true(row["result_status"] == "mixed", "mixed successful/failed utility runs should be marked mixed")
    assert_true(row["failed_runs"] == "1", "failed utility runs should be counted")
    assert_true(row["n_valid_runs"] == "1", "valid utility runs should be counted separately")
    assert_true(row["eval_accuracy"] == "0.880000", "failed utility metrics should not enter the mean")


def main():
    tests = []
    if torch is not None:
        tests.extend(
            [
                test_defense_param_spec_tracks_shared_cli_mapping,
                test_dpsgd_opacus_summary_fields_include_accounting_terms,
                test_grad_similarity_metrics_returns_cosine_and_norm_retention,
            ]
        )
    else:
        print("Skipping torch-backed utility tests: torch is not installed in this Python environment.")
    tests.extend([
        test_collect_parser_splits_multi_variant_attack_summaries,
        test_collect_parser_reads_train_and_proxy_summaries,
        test_tradeoff_join_uses_none_as_utility_anchor,
        test_tradeoff_join_distinguishes_peft_method_and_rep_bottleneck,
        test_attack_anchor_keeps_adaptive_attack_separate,
        test_tradeoff_join_emits_adaptive_and_nonadaptive_rows,
        test_utility_aggregation_filters_failed_runs,
    ])
    for test in tests:
        print(f"Running {test.__name__}...")
        test()
    print("All utility pipeline tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
