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


def test_collect_parser_reads_train_and_proxy_summaries():
    train_text = """
===== TRAIN RESULT SUMMARY START =====
dataset=sst2
batch_size=2
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


def test_tradeoff_join_uses_none_as_utility_anchor():
    rows = [
        {
            "log_kind": "train",
            "dataset": "sst2",
            "batch_size": "2",
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


def main():
    tests = [
        test_defense_param_spec_tracks_shared_cli_mapping,
        test_grad_similarity_metrics_returns_cosine_and_norm_retention,
        test_collect_parser_splits_multi_variant_attack_summaries,
        test_collect_parser_reads_train_and_proxy_summaries,
        test_tradeoff_join_uses_none_as_utility_anchor,
    ]
    for test in tests:
        print(f"Running {test.__name__}...")
        test()
    print("All utility pipeline tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
