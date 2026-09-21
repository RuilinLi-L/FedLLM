from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts import collect_experiment_logs as cel
from utils.defense_common import defense_param_spec
from utils.lrb_presets import apply_lrb_preset
from utils.ntp_utility import causal_lm_labels, evaluate_causal_lm, safe_perplexity


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def test_ntp_label_semantics_ignore_sst2_labels():
    input_ids = torch.tensor(
        [
            [10, 11, 12, 50256],
            [20, 21, 50256, 50256],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ]
    )
    expected = torch.tensor(
        [
            [10, 11, 12, -100],
            [20, 21, -100, -100],
        ]
    )

    labels_with_class_a = causal_lm_labels(input_ids, attention_mask)
    _sst2_labels_a = torch.tensor([0, 1])
    labels_with_class_b = causal_lm_labels(input_ids, attention_mask)
    _sst2_labels_b = torch.tensor([1, 0])

    assert_true(torch.equal(labels_with_class_a, expected), "causal-LM padding labels are incorrect")
    assert_true(
        torch.equal(labels_with_class_a, labels_with_class_b),
        "changing SST-2 class labels must not affect next-token labels",
    )


def test_token_weighted_nll_uses_shifted_token_counts():
    class DummyCausalLM:
        def eval(self):
            return self

        def __call__(self, input_ids, attention_mask, labels):
            assert_true(torch.equal(labels, causal_lm_labels(input_ids, attention_mask)), "wrong eval labels")
            return SimpleNamespace(loss=input_ids[0, 0].float())

    batches = [
        {
            "input_ids": torch.tensor([[1, 10, 11, 50256]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
        },
        {
            "input_ids": torch.tensor([[3, 20, 21, 22]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        },
    ]
    metrics = evaluate_causal_lm(DummyCausalLM(), batches, "cpu")
    assert_true(math.isclose(metrics["nll"], 2.2), "validation evaluator must weight 1*2 and 3*3")
    assert_true(not math.isclose(metrics["nll"], 2.0), "NLL must not be the unweighted batch-loss mean")
    assert_true(metrics["valid_tokens"] == 5, "validation evaluator counted shifted targets incorrectly")


def test_safe_perplexity():
    nll = 1.25
    assert_true(math.isclose(safe_perplexity(nll), math.exp(nll)), "PPL must equal exp(NLL)")
    assert_true(math.isinf(safe_perplexity(1000.0)), "PPL overflow must return infinity")


def test_seq_class_evaluation_backward_compatibility():
    current_source = Path("train.py").read_text(encoding="utf-8")
    parser_tree = ast.parse(current_source)
    evaluate_node = next(
        item for item in parser_tree.body if isinstance(item, ast.FunctionDef) and item.name == "evaluate_model"
    )
    namespace = {
        "torch": torch,
        "classification_metrics": lambda predictions, references, _dataset: {
            "accuracy": float(predictions == references),
            "macro_f1": float(predictions == references),
        },
    }
    exec(compile(ast.Module(body=[evaluate_node], type_ignores=[]), "train.py", "exec"), namespace)

    class DummyClassificationModel:
        def eval(self):
            return self

        def __call__(self, input_ids, attention_mask, labels):
            return SimpleNamespace(
                loss=torch.tensor(0.25),
                logits=torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            )

    batches = [
        {
            "input_ids": torch.tensor([[10], [20]]),
            "attention_mask": torch.ones(2, 1, dtype=torch.long),
            "labels": torch.tensor([0, 1]),
        }
    ]
    metrics = namespace["evaluate_model"](DummyClassificationModel(), batches, "cpu", "sst2")
    assert_true(metrics["accuracy"] == 1.0, "seq_class accuracy behavior changed")
    assert_true(metrics["macro_f1"] == 1.0, "seq_class macro-F1 behavior changed")
    assert_true(metrics["loss"] == 0.25, "seq_class loss behavior changed")

    build_parser = next(
        item for item in parser_tree.body if isinstance(item, ast.FunctionDef) and item.name == "build_parser"
    )
    task_calls = [
        node
        for node in ast.walk(build_parser)
        if isinstance(node, ast.Call)
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--task"
    ]
    assert_true(len(task_calls) == 1, "expected one --task parser declaration")
    defaults = {
        keyword.arg: keyword.value.value
        for keyword in task_calls[0].keywords
        if keyword.arg == "default" and isinstance(keyword.value, ast.Constant)
    }
    assert_true(defaults.get("default") == "seq_class", "seq_class must remain the default task")


def test_utility_aggregation_is_task_aware_and_uses_sample_std():
    rows = []
    for task, values in (("seq_class", (0.2, 0.4)), ("next_token_pred", (2.0, 4.0))):
        for seed, value in zip((101, 202), values):
            rows.append(
                {
                    "log_kind": "train",
                    "dataset": "sst2",
                    "task": task,
                    "batch_size": "1",
                    "train_method": "full",
                    "defense": "none",
                    "defense_param_name": "n/a",
                    "defense_param_value": "n/a",
                    "seed": str(seed),
                    "result_status": "ok",
                    "steps_completed": "2",
                    "val_nll": str(value),
                    "val_perplexity": str(math.exp(value)),
                }
            )

    aggregated = cel.build_utility_results(rows)
    assert_true(len(aggregated) == 2, "seq_class and next_token_pred rows were merged")
    ntp = next(row for row in aggregated if row["task"] == "next_token_pred")
    assert_true(ntp["val_nll"] == "3.000000", "NTP seed mean is incorrect")
    assert_true(
        math.isclose(float(ntp["val_nll_std"]), math.sqrt(2.0), rel_tol=1e-6),
        "NTP standard deviation must use ddof=1",
    )


def test_individual_runner_names_all_formal_logs():
    script = Path("scripts/run_ntp_sst2_utility.sh").read_text(encoding="utf-8")
    assert_true('LOG_DIR="log/runs/ntp_sst2_utility"' in script, "formal log directory changed")
    assert_true('LOG_FILE="${LOG_DIR}/${RUN_NAME}.txt"' in script, "runner must create one named TXT log")
    assert_true('RUN_NAME="none_seed${SEED}"' in script, "None log naming changed")
    assert_true('RUN_NAME="slr_rho05_seed${SEED}"' in script, "SLR log naming changed")


def test_slr_transform_precedes_every_optimizer_step():
    source = Path("train.py").read_text(encoding="utf-8")
    start = source.index("def run_next_token_pred_training")
    end = source.index("\ndef run_training(", start)
    ntp_loop = source[start:end]
    assert_true(
        ntp_loop.index("apply_training_defense(") < ntp_loop.index("opt.step()"),
        "SLR transformation must precede optimizer.step",
    )
    assert_true(
        "defense_updates != n_steps" in ntp_loop,
        "SLR path must enforce transformation coverage for every optimizer update",
    )


def test_slr_rho05_resolves_to_layerwise_existing_preset():
    args = SimpleNamespace(
        defense="lrbprojonly",
        defense_lrb_preset="custom",
        defense_lrb_keep_ratio_sensitive=0.5,
        defense_lrb_seed_mode="static",
    )
    apply_lrb_preset(args)
    param_name, param_value = defense_param_spec(args)
    assert_true(param_name == "defense_lrb_preset", "SLR should report the resolved preset")
    assert_true(param_value == "lrbprojonly@k=0.5", "SLR rho*=0.5 summary metadata changed")
    assert_true(args.defense_lrb_keep_ratio_sensitive == 0.5, "SLR sensitive endpoint changed")
    assert_true(args.defense_lrb_keep_ratio_other == 0.75, "SLR must not become uniform projection")
    assert_true(args.defense_lrb_seed_mode == "static", "SLR seed mode changed")


def main():
    tests = [
        test_ntp_label_semantics_ignore_sst2_labels,
        test_token_weighted_nll_uses_shifted_token_counts,
        test_safe_perplexity,
        test_seq_class_evaluation_backward_compatibility,
        test_utility_aggregation_is_task_aware_and_uses_sample_std,
        test_individual_runner_names_all_formal_logs,
        test_slr_transform_precedes_every_optimizer_step,
        test_slr_rho05_resolves_to_layerwise_existing_preset,
    ]
    for test in tests:
        print(f"Running {test.__name__}...")
        test()
    print("All NTP utility semantics tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
