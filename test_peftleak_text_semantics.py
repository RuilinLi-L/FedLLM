#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import os
import sys
from types import ModuleType, SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attacks.peftleak_text import (
    nearest_token_ids,
    optimize_text_embeddings,
    select_peft_gradient_tensors,
    token_recovery_ratio,
)
from attacks.peftleak_text_ratio import (
    MaliciousTextTokenAdapter,
    TextRatioGradientResult,
    TextRatioRoutingInfo,
    TextTokenStatistics,
    decode_ratio_recovery,
    recover_hidden_slots_from_ratio_grads,
    route_text_tokens,
)


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


class DummyTokenizer:
    def decode(self, ids):
        return " ".join(str(int(tok)) for tok in ids)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_labels=2)
        self.embedding = torch.nn.Embedding(6, 4)
        self.adapter = torch.nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.2, -0.1, 0.3],
                        [0.5, -0.4, 0.8, 0.1],
                        [-0.3, 0.7, 0.2, -0.6],
                        [0.9, -0.2, -0.5, 0.4],
                        [-0.8, 0.1, 0.3, 0.9],
                    ],
                    dtype=torch.float32,
                )
            )
            self.adapter.weight.copy_(torch.tensor([[0.7, -0.4, 0.2, 0.3], [-0.1, 0.6, -0.5, 0.4]]))

    def get_input_embeddings(self):
        return self.embedding


class DummyWrapper:
    def __init__(self):
        self.model = DummyModel()
        self.args = SimpleNamespace(rng_seed=7)
        self.tokenizer = DummyTokenizer()
        self.pad_token = 0

    def _seq_class_input_embeds(self, batch):
        return self.model.embedding(batch["input_ids"])

    def _seq_class_logits_from_embeds(self, batch, inputs_embeds, representation_mask=None):
        representation = inputs_embeds.mean(dim=1)
        if representation_mask is not None:
            representation = representation * representation_mask
        return self.model.adapter(representation), representation

    def trainable_parameters(self):
        return (self.model.adapter.weight,)

    def trainable_parameter_names(self):
        return ["base_model.model.encoder.layer.0.attention.self.query.lora_A.default.weight"]


def _target_grads(wrapper, batch, labels):
    embeds = wrapper._seq_class_input_embeds(batch)
    logits, _ = wrapper._seq_class_logits_from_embeds(batch, embeds)
    loss = F.cross_entropy(logits, labels)
    return torch.autograd.grad(loss, wrapper.trainable_parameters(), allow_unused=True)


def test_peft_gradient_inventory_selects_lora_ia3_and_excludes_saved_heads():
    grads = (torch.ones(2, 2), torch.ones(2), torch.ones(2, 2), None, torch.ones(2, 4))
    names = [
        "base_model.model.layer.0.lora_A.default.weight",
        "base_model.model.score.modules_to_save.default.weight",
        "base_model.model.layer.0.ia3_l.default",
        "base_model.model.layer.1.lora_A.default.weight",
        "base_model.model.bert.encoder.layer.0.output.adapters.default.adapter_down.0.weight",
    ]
    lora_idx, lora_names = select_peft_gradient_tensors(grads, names, "lora")
    ia3_idx, ia3_names = select_peft_gradient_tensors(grads, names, "ia3")
    adapter_idx, adapter_names = select_peft_gradient_tensors(grads, names, "adapter")

    assert_true(lora_idx == [0], f"LoRA selector should keep only adapter tensors with gradients: {lora_idx}")
    assert_true("modules_to_save" not in ";".join(lora_names), "selector should exclude modules_to_save")
    assert_true(ia3_idx == [2], f"IA3 selector should keep IA3 tensors: {ia3_idx}")
    assert_true("ia3" in ia3_names[0].lower(), "IA3 selector should preserve IA3 name")
    assert_true(adapter_idx == [4], f"Adapter selector should keep AdapterHub bottleneck tensors: {adapter_idx}")
    assert_true("adapter_down" in adapter_names[0].lower(), "Adapter selector should preserve adapter down-projection name")


def test_nearest_token_decoding_returns_valid_ids_and_ignores_pad():
    table = torch.eye(4, dtype=torch.float32)
    embeddings = torch.stack([table[1] + 0.01, table[3] - 0.01], dim=0)
    token_ids, scores = nearest_token_ids(embeddings, table, unused_token_ids={0}, metric="cos")

    assert_true(token_ids.tolist() == [1, 3], f"nearest token ids should be valid and pad-free: {token_ids.tolist()}")
    assert_true(torch.isfinite(scores).all(), "nearest token scores should be finite")


def test_token_recovery_is_positionwise():
    recovered = token_recovery_ratio([[1, 2, 3], [4, 5]], [[1, 9, 3], [4, 0]])
    assert_true(abs(recovered - 0.6) < 1e-8, f"positionwise token recovery should be 3/5, got {recovered}")


def test_token_recovery_ignores_padding_and_special_tokens():
    recovered = token_recovery_ratio(
        [[101, 7, 8, 0], [101, 4, 9, 0]],
        [[101, 7, 3, 0], [101, 5, 9, 0]],
        ignored_token_ids={0, 101},
        reference_mask=[[1, 1, 1, 0], [1, 1, 1, 0]],
    )
    assert_true(abs(recovered - 0.5) < 1e-8, f"content-token recovery should ignore pads/specials, got {recovered}")


def test_gradient_matching_loss_decreases_on_tiny_peft_model():
    torch.manual_seed(3)
    wrapper = DummyWrapper()
    batch = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}
    labels = torch.tensor([1], dtype=torch.long)
    target = _target_grads(wrapper, batch, labels)

    result = optimize_text_embeddings(
        model_wrapper=wrapper,
        batch=batch,
        labels=labels,
        target_grads=target,
        parameter_names=wrapper.trainable_parameter_names(),
        peft_method="lora",
        steps=25,
        lr=0.15,
        match_loss="mse",
    )

    history = result["loss_history"]
    assert_true(history[-1] < history[0], f"gradient matching loss should decrease: {history[0]} -> {history[-1]}")
    assert_true(len(result["predicted_ids"]) == 1, "attack should return one decoded sequence")
    assert_true(all(0 <= tok < 6 for tok in result["predicted_ids"][0]), "decoded ids should be valid tokenizer ids")


def test_gradient_matching_label_search_reports_search_mode():
    torch.manual_seed(4)
    wrapper = DummyWrapper()
    batch = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}
    labels = torch.tensor([1], dtype=torch.long)
    target = _target_grads(wrapper, batch, labels)

    result = optimize_text_embeddings(
        model_wrapper=wrapper,
        batch=batch,
        labels=labels,
        target_grads=target,
        parameter_names=wrapper.trainable_parameter_names(),
        peft_method="lora",
        steps=6,
        lr=0.1,
        restarts=2,
        match_loss="normalized_mse",
        label_known=False,
        label_candidates=[0, 1],
    )

    assert_true(result["label_mode"] == "search", f"label search should report search mode, got {result['label_mode']}")
    assert_true(result["best_label"] is not None, "label search should record the best candidate labels")
    assert_true(result["selected_gradient_count"] == 1, "adapter gradient count should be reported")
    assert_true(result["sequence_length"] == 3, "sequence length should be reported")
    assert_true(result["restarts"] == 2, "gradient matching should report restart count")
    assert_true(result["match_loss"] == "normalized_mse", "gradient matching should report match loss")


def test_ratio_adapter_gradient_differences_recover_hidden_vector():
    hidden = torch.tensor([[[0.25, -0.5, 1.25, 0.75]]], dtype=torch.float32)
    adapter = MaliciousTextTokenAdapter(hidden_dim=4, slots=["s0_p0"], rows_per_slot=4, seed=11)
    logits = adapter(hidden, [["s0_p0"]], num_labels=2)
    loss = F.cross_entropy(logits, torch.tensor([1], dtype=torch.long))
    grads = torch.autograd.grad(loss, adapter.parameters_for_grad(), allow_unused=True)
    recovered = recover_hidden_slots_from_ratio_grads(grads, adapter.parameter_names())

    assert_true("s0_p0" in recovered, "ratio recovery should produce the routed slot")
    assert_true(
        torch.allclose(recovered["s0_p0"], hidden[0, 0], atol=1e-5, rtol=1e-5),
        f"ratio recovery should recover hidden vector, got {recovered['s0_p0']}",
    )


def test_ratio_decode_recovers_nearest_token_after_subtracting_position():
    table = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    non_token = torch.tensor([[[0.2, -0.1, 0.3, 0.0]]], dtype=torch.float32)
    hidden = table[1].view(1, 1, -1) + non_token
    adapter = MaliciousTextTokenAdapter(hidden_dim=4, slots=["s0_p0"], rows_per_slot=4, seed=12)
    logits = adapter(hidden, [["s0_p0"]], num_labels=2)
    loss = F.cross_entropy(logits, torch.tensor([1], dtype=torch.long))
    grads = torch.autograd.grad(loss, adapter.parameters_for_grad(), allow_unused=True)
    routing = TextRatioRoutingInfo(
        route="oracle",
        slot_keys=[["s0_p0"]],
        slot_counts={"s0_p0": 1},
        routed_token_count=1,
        colliding_token_count=0,
        collision_rate=0.0,
        reportable=False,
    )
    ratio_result = TextRatioGradientResult(
        grads=tuple(None if grad is None else grad.detach() for grad in grads),
        names=adapter.parameter_names(),
        input_vectors=hidden.detach(),
        non_token_vectors=non_token,
        routing=routing,
        logits=logits.detach(),
        loss=float(loss.detach().item()),
    )
    decoded = decode_ratio_recovery(
        ratio_result=ratio_result,
        defended_grads=ratio_result.grads,
        token_embedding_matrix=table,
        batch={"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])},
        ignored_token_ids={0},
        reference_mask=[[1]],
    )

    assert_true(decoded["predicted_ids"] == [[1]], f"decoded token should match reference: {decoded['predicted_ids']}")
    assert_true(abs(decoded["rec_token_mean"] - 1.0) < 1e-8, "ratio decode should recover the token")
    assert_true(decoded["reportable"] is False, "oracle ratio routing should be marked non-reportable")


def test_ratio_decode_skips_degenerate_gradient_slots_without_aborting():
    table = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    grads = (
        torch.zeros(4, 4),
        torch.ones(4),
    )
    names = [
        "text_ratio_adapter.slot_p0_b0.weight",
        "text_ratio_adapter.slot_p0_b0.bias",
    ]
    recovered = recover_hidden_slots_from_ratio_grads(grads, names)
    assert_true(recovered == {}, "degenerate ratio slots should be skipped instead of aborting recovery")

    routing = TextRatioRoutingInfo(
        route="public_bins",
        slot_keys=[["p0_b0"]],
        slot_counts={"p0_b0": 1},
        routed_token_count=1,
        colliding_token_count=0,
        collision_rate=0.0,
        reportable=True,
    )
    ratio_result = TextRatioGradientResult(
        grads=grads,
        names=names,
        input_vectors=table[1].view(1, 1, -1),
        non_token_vectors=torch.zeros(1, 1, 4),
        routing=routing,
        logits=torch.zeros(1, 2),
        loss=0.0,
    )
    decoded = decode_ratio_recovery(
        ratio_result=ratio_result,
        defended_grads=ratio_result.grads,
        token_embedding_matrix=table,
        batch={"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])},
        ignored_token_ids={0},
        reference_mask=[[1]],
    )

    assert_true(decoded["predicted_ids"] == [[0]], "unrecovered slots should use the pad fallback token")
    assert_true(decoded["rec_token_mean"] == 0.0, "unrecovered routed tokens should count as failed recovery")
    assert_true(decoded["recovered_hidden_count"] == 0, "degenerate slots should not count as recovered hidden states")
    assert_true(decoded["recovered_slot_count"] == 0, "degenerate slots should not count as recovered slots")


def test_public_bin_routing_reports_collisions_without_oracle():
    vectors = torch.zeros(2, 2, 4)
    batch = {"input_ids": torch.tensor([[1, 2], [3, 4]]), "attention_mask": torch.ones(2, 2, dtype=torch.long)}
    stats = TextTokenStatistics(
        mean=torch.zeros(4),
        std=torch.ones(4),
        bin_edges=torch.zeros(2, 1),
        global_bin_edges=torch.zeros(1),
        num_sequences=4,
        num_tokens=8,
        num_bins=2,
        max_seq_len=2,
    )
    routing = route_text_tokens(vectors, batch, stats=stats, route="public_bins")

    assert_true(routing.reportable, "public_bins routing should be reportable")
    assert_true(routing.collision_rate > 0.0, "duplicate public bins should report collisions")
    assert_true(all(not key.startswith("s0_") for key in routing.slot_counts), "public bins should not expose sample ids")


def test_attack_entrypoint_policy_rejects_prefix_and_keeps_dager_unsupported():
    from attack_peftleak import _init_tracker, _validate_args, build_parser

    parser = build_parser()
    default_args = parser.parse_args(
        [
            "--dataset", "sst2",
            "--split", "val",
            "--n_inputs", "1",
            "--finetuned_path", "dummy",
        ]
    )
    assert_true(
        default_args.attn_implementation == "eager",
        "PEFTLeak opt default should use eager attention for second-order gradient compatibility",
    )

    prefix_args = parser.parse_args(
        [
            "--dataset", "sst2",
            "--split", "val",
            "--n_inputs", "1",
            "--finetuned_path", "dummy",
            "--peft_method", "prefix",
        ]
    )
    try:
        _validate_args(prefix_args)
    except NotImplementedError as exc:
        assert_true("lora|ia3|adapter" in str(exc), "prefix rejection should mention supported FedLLM PEFT text methods")
    else:
        raise AssertionError("prefix FedLLM PEFT text eval should be rejected")

    adapter_args = parser.parse_args(
        [
            "--dataset", "sst2",
            "--split", "val",
            "--n_inputs", "1",
            "--finetuned_path", "dummy",
            "--peft_method", "adapter",
            "--peftleak_attack_mode", "both",
            "--peftleak_restarts", "2",
            "--peftleak_match_loss", "cosine",
        ]
    )
    assert_true(adapter_args.peft_method == "adapter", "adapter should be accepted as a PEFTLeak CLI choice")
    assert_true(adapter_args.peftleak_attack_mode == "both", "attack mode should parse")
    assert_true(adapter_args.peftleak_restarts == 2, "restart count should parse")
    assert_true(adapter_args.peftleak_match_loss == "cosine", "match loss should parse")

    previous_peft_utils = sys.modules.get("utils.peft_utils")
    fake_peft_utils = ModuleType("utils.peft_utils")
    fake_peft_utils.apply_peft_config_to_args = lambda args, require_checkpoint=True: args
    sys.modules["utils.peft_utils"] = fake_peft_utils
    try:
        opt_sdpa_args = parser.parse_args(
            [
                "--dataset", "sst2",
                "--split", "val",
                "--n_inputs", "1",
                "--finetuned_path", "dummy",
                "--peft_method", "lora",
                "--peftleak_attack_mode", "opt",
                "--attn_implementation", "sdpa",
            ]
        )
        _validate_args(opt_sdpa_args)
        assert_true(
            opt_sdpa_args.attn_implementation == "eager",
            "opt attack should force eager attention when sdpa is requested",
        )

        both_sdpa_args = parser.parse_args(
            [
                "--dataset", "sst2",
                "--split", "val",
                "--n_inputs", "1",
                "--finetuned_path", "dummy",
                "--peft_method", "adapter",
                "--peftleak_attack_mode", "both",
                "--attn_implementation", "sdpa",
            ]
        )
        _validate_args(both_sdpa_args)
        assert_true(
            both_sdpa_args.attn_implementation == "eager",
            "both attack should force eager attention because it includes optimization matching",
        )

        ratio_sdpa_args = parser.parse_args(
            [
                "--dataset", "sst2",
                "--split", "val",
                "--n_inputs", "1",
                "--finetuned_path", "dummy",
                "--peft_method", "adapter",
                "--peftleak_attack_mode", "ratio",
                "--attn_implementation", "sdpa",
            ]
        )
        _validate_args(ratio_sdpa_args)
        assert_true(
            ratio_sdpa_args.attn_implementation == "sdpa",
            "ratio-only attack should preserve explicitly requested sdpa attention",
        )
    finally:
        if previous_peft_utils is None:
            sys.modules.pop("utils.peft_utils", None)
        else:
            sys.modules["utils.peft_utils"] = previous_peft_utils

    lora_ratio_args = parser.parse_args(
        [
            "--dataset", "sst2",
            "--split", "val",
            "--n_inputs", "1",
            "--finetuned_path", "dummy",
            "--peft_method", "lora",
            "--peftleak_attack_mode", "both",
        ]
    )
    lora_ratio_tracker = _init_tracker(lora_ratio_args)
    assert_true(
        lora_ratio_tracker["attack_variant"] == "text_opt_ratio_unsupported_fallback",
        "LoRA/IA3 ratio requests should be clearly marked as optimization fallback",
    )

    dager_args = parser.parse_args(
        [
            "--dataset", "sst2",
            "--split", "val",
            "--n_inputs", "1",
            "--finetuned_path", "dummy",
            "--peft_method", "lora",
            "--defense", "dager",
        ]
    )
    validated = _validate_args(dager_args)
    assert_true(validated.defense == "dager", "DAGER defense should parse so the entrypoint can emit unsupported summary")


def test_attack_result_summary_emits_seed():
    from attack_peftleak import _emit_result_summary, _init_tracker, build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "--dataset", "sst2",
            "--split", "val",
            "--n_inputs", "1",
            "--finetuned_path", "dummy",
            "--peft_method", "adapter",
            "--peftleak_attack_mode", "ratio",
            "--rng_seed", "202",
        ]
    )
    tracker = _init_tracker(args)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _emit_result_summary(args, tracker)
    text = buf.getvalue()

    assert_true("\nseed=202\n" in text, f"PEFTLeak text summary should emit the rng seed, got:\n{text}")


def test_peftleak_eval_script_multi_seed_logging_contract():
    root = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(root, "scripts", "peftleak_eval.sh")
    with open(script_path, "r", encoding="utf-8") as handle:
        script = handle.read()

    assert_true("PEFTLEAK_SEEDS:-101 202 303" in script, "default PEFTLeak text seeds should be 101/202/303")
    assert_true("PEFTLEAK_LOG_DIR:-log/peftleak_text_${DATASET}/privacy" in script, "privacy log dir should be overridable")
    assert_true("_seed${seed}.txt" in script, "log filenames should include the seed suffix")
    assert_true('tee "$log_file"' in script, "wrapper should tee terminal output into each seed log")
    assert_true('if has_flag "--rng_seed"' in script, "explicit --rng_seed should switch the wrapper to single-seed mode")
    assert_true('RUN_EXTRA+=( "$arg" )' in script, "wrapper should remove user seed args before appending per-run seed")

    for expected in (
        "--defense_topk_ratio",
        "--defense_n_bits",
        "--defense_noise",
        "--defense_lrb_keep_ratio_sensitive",
    ):
        assert_true(expected in script, f"filename parameter labels should cover {expected}")


def main():
    tests = [
        test_peft_gradient_inventory_selects_lora_ia3_and_excludes_saved_heads,
        test_nearest_token_decoding_returns_valid_ids_and_ignores_pad,
        test_token_recovery_is_positionwise,
        test_token_recovery_ignores_padding_and_special_tokens,
        test_gradient_matching_loss_decreases_on_tiny_peft_model,
        test_gradient_matching_label_search_reports_search_mode,
        test_ratio_adapter_gradient_differences_recover_hidden_vector,
        test_ratio_decode_recovers_nearest_token_after_subtracting_position,
        test_ratio_decode_skips_degenerate_gradient_slots_without_aborting,
        test_public_bin_routing_reports_collisions_without_oracle,
        test_attack_entrypoint_policy_rejects_prefix_and_keeps_dager_unsupported,
        test_attack_result_summary_emits_seed,
        test_peftleak_eval_script_multi_seed_logging_contract,
    ]
    for test in tests:
        print(f"Running {test.__name__}...")
        test()
    print("All FedLLM PEFT text semantic tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

