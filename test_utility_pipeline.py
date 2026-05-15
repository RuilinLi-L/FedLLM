#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import collect_experiment_logs as cel  # noqa: E402
from utils.defense_common import defense_param_spec, grad_similarity_metrics  # noqa: E402


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


def main():
    tests = [
        test_defense_param_spec_tracks_shared_cli_mapping,
        test_grad_similarity_metrics_returns_cosine_and_norm_retention,
        test_collect_parser_splits_multi_variant_attack_summaries,
        test_collect_parser_reads_train_and_proxy_summaries,
        test_tradeoff_join_uses_none_as_utility_anchor,
        test_tradeoff_join_distinguishes_peft_method_and_rep_bottleneck,
    ]
    for test in tests:
        print(f"Running {test.__name__}...")
        test()
    print("All utility pipeline tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
