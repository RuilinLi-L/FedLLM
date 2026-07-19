#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import hashlib
import time

import numpy as np
import torch

from attack_peftleak import compute_text_metrics, validate_text_metric_backend
from attacks.peftleak_text import get_token_embedding_matrix
from attacks.peftleak_text_new import (
    ADAPTER_RATIO_DEFENSES,
    LORA_OPT_DEFENSES,
    build_public_probe_statistics,
    compute_probe_gradient_observation,
    install_fixed_embedding_probe,
    optimize_lora_embeddings_defense_aware,
    recover_tokens_from_probe_gradients,
    resolve_base_word_embedding,
    token_recovery_accuracy,
)
from utils.defense_common import add_shared_defense_args, defense_param_spec, fmt_summary_value, safe_mean
from utils.data import ATTACK_SPLIT_CHOICES, dataset_summary_fields, record_dataset_protocol
from utils.gpu import resolve_cuda_device, resolve_gradient_device
from utils.lrb_presets import apply_lrb_preset
from utils.lrb_defense import lrb_seed_summary_fields
from utils.representation_bottleneck import validate_rep_bottleneck_args


SUMMARY_START = "===== RESULT SUMMARY START ====="
SUMMARY_END = "===== RESULT SUMMARY END ====="


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strict FedLLM PEFT leakage v2: a PEFTLeak-style registered Adapter probe "
            "or defense-aware optimization-based LoRA inversion."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["cola", "sst2", "rte", "rotten_tomatoes", "stanfordnlp/imdb", "glnmario/ECHR"],
        required=True,
    )
    parser.add_argument("--task", choices=["seq_class"], default="seq_class")
    parser.add_argument("--split", choices=ATTACK_SPLIT_CHOICES, required=True)
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--n_inputs", type=int, required=True)
    parser.add_argument("--start_input", type=int, default=0)
    parser.add_argument("--end_input", type=int, default=100000)
    parser.add_argument("--model_path", choices=["gpt2", "openai-community/gpt2-large", "bert-base-uncased"], default="gpt2")
    parser.add_argument("--finetuned_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device_grad", type=str, default="auto")
    parser.add_argument("--attn_implementation", choices=["sdpa", "eager"], default="eager")
    parser.add_argument("--precision", choices=["8bit", "half", "full", "double"], default="full")
    parser.add_argument("--pad", choices=["right", "left"], default="right")
    parser.add_argument("--grad_b", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--algo", choices=["sgd", "fedavg"], default="sgd")
    parser.add_argument("--avg_epochs", type=int, default=None)
    parser.add_argument("--b_mini", type=int, default=None)
    parser.add_argument("--avg_lr", type=float, default=None)
    parser.add_argument("--hidden_act", type=str, default=None)
    parser.add_argument("--loss", choices=["ce"], default="ce")

    parser.add_argument("--train_method", choices=["peft", "lora"], default="peft")
    parser.add_argument("--peft_method", choices=["adapter", "lora"], required=True)
    parser.add_argument("--peft_num_virtual_tokens", type=int, default=None)
    parser.add_argument("--adapter_reduction_factor", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_target_modules", type=str, default=None)

    parser.add_argument("--peftleak_attack_mode", choices=["auto", "ratio", "opt"], default="auto")
    parser.add_argument("--peftleak_max_length", type=int, default=128)
    parser.add_argument("--peftleak_ratio_bins", type=int, default=8)
    parser.add_argument("--peftleak_ratio_rows_per_bin", type=int, default=4)
    parser.add_argument("--peftleak_ratio_public_n_inputs", type=int, default=32)
    parser.add_argument("--peftleak_steps", type=int, default=60)
    parser.add_argument("--peftleak_lr", type=float, default=0.1)
    parser.add_argument("--peftleak_restarts", type=int, default=1)
    parser.add_argument(
        "--peftleak_match_loss",
        choices=["mse", "normalized_mse", "cosine"],
        default="normalized_mse",
    )
    parser.add_argument(
        "--peftleak_known_labels",
        action="store_true",
        default=False,
        help="Use private labels as declared side information. The reportable default searches labels.",
    )
    parser.add_argument("--peftleak_label_candidates", type=str, default=None)
    parser.add_argument(
        "--text_metric_backend",
        choices=["datasets", "simple_ngram"],
        default="datasets",
        help="Explicit ROUGE backend. There is no automatic fallback in this entrypoint.",
    )
    add_shared_defense_args(parser, default_grad_mode="eval")
    return parser


def _label_candidates(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    return values or None


def _resolved_attack_mode(args) -> str:
    requested = str(args.peftleak_attack_mode)
    if requested == "auto":
        return "ratio" if args.peft_method == "adapter" else "opt"
    return requested


def _validate_args(args):
    if args.train_method == "lora":
        args.train_method = "peft"
        args.peft_method = "lora"
    else:
        args.train_method = "peft"

    try:
        from utils.peft_utils import apply_peft_config_to_args
    except ModuleNotFoundError as exc:
        if exc.name == "peft":
            raise ModuleNotFoundError(
                "attack_peftleak_new.py requires the PEFT runtime; activate environment-peftleak.yml."
            ) from exc
        raise
    apply_peft_config_to_args(args, require_checkpoint=True)
    args.peft_method = str(args.peft_method).strip().lower().replace("-", "_")
    if args.peft_method not in {"adapter", "lora"}:
        raise NotImplementedError("Strict PEFT leakage v2 supports Adapter and LoRA checkpoints only.")
    if args.precision == "8bit":
        raise NotImplementedError("Strict PEFT leakage v2 does not support 8-bit model loading.")

    args.peftleak_attack_mode = _resolved_attack_mode(args)
    if args.peft_method == "adapter" and args.peftleak_attack_mode != "ratio":
        raise ValueError("Adapter is reportable only with --peftleak_attack_mode ratio in v2.")
    if args.peft_method == "lora" and args.peftleak_attack_mode != "opt":
        raise ValueError("LoRA is reportable only with --peftleak_attack_mode opt in v2.")
    if args.peftleak_attack_mode == "ratio" and int(args.batch_size) != 1:
        raise ValueError("Reportable Adapter ratio v2 requires --batch_size 1; collision recovery is not implemented.")
    if args.peftleak_attack_mode == "opt" and args.attn_implementation != "eager":
        print("[peftleak:v2] Switching attention implementation to eager for second-order gradients.", flush=True)
        args.attn_implementation = "eager"

    if int(args.peftleak_max_length) < 1:
        raise ValueError("--peftleak_max_length must be positive.")
    if int(args.peftleak_ratio_bins) < 1:
        raise ValueError("--peftleak_ratio_bins must be positive.")
    if int(args.peftleak_ratio_rows_per_bin) < 2:
        raise ValueError("--peftleak_ratio_rows_per_bin must be at least two.")
    if args.peftleak_attack_mode == "ratio" and int(args.peftleak_ratio_public_n_inputs) < 1:
        raise ValueError("Adapter ratio v2 requires a non-empty disjoint public split.")
    if int(args.peftleak_steps) < 1 or int(args.peftleak_restarts) < 1:
        raise ValueError("Optimization steps and restarts must be positive.")
    if getattr(args, "defense_pct_mask", None) is not None:
        raise NotImplementedError("Random post-defense masking is not implemented in strict PEFT leakage v2.")

    validate_rep_bottleneck_args(args)
    if getattr(args, "defense_rep_bottleneck", "none") != "none":
        raise NotImplementedError("Representation bottlenecks are not part of the strict v2 observation matrix.")
    supported = ADAPTER_RATIO_DEFENSES if args.peftleak_attack_mode == "ratio" else LORA_OPT_DEFENSES
    if args.defense not in supported:
        raise NotImplementedError(
            f"{args.peftleak_attack_mode} v2 supports defenses {sorted(supported)}; got {args.defense!r}."
        )
    apply_lrb_preset(args)
    args.peft_eval_scope = (
        "peftleak_style_text_adapter_ratio_v2"
        if args.peftleak_attack_mode == "ratio"
        else "optimization_peft_gradient_inversion_v2"
    )
    return args


def _ignored_token_ids(tokenizer, model_wrapper) -> set[int]:
    values = {
        getattr(model_wrapper, "pad_token", None),
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "bos_token_id", None),
        getattr(tokenizer, "cls_token_id", None),
        getattr(tokenizer, "sep_token_id", None),
    }
    return {int(value) for value in values if value is not None}


def _tokenize(tokenizer, sequences, args):
    return tokenizer(
        sequences,
        padding=True,
        truncation=True,
        max_length=int(args.peftleak_max_length),
        return_tensors="pt",
    ).to(args.device)


def _inventory_hash(names) -> str:
    return hashlib.sha256("\n".join(str(name) for name in names).encode("utf-8")).hexdigest()


def _duration(seconds: float) -> str:
    return str(datetime.timedelta(seconds=seconds)).split(".")[0]


def _initial_tracker(args) -> dict[str, object]:
    ratio = args.peftleak_attack_mode == "ratio"
    requested = max(0, min(int(args.n_inputs), int(args.end_input)) - int(args.start_input))
    return {
        "summary_version": 3,
        "result_status": "ok",
        "attack": (
            "fedllm_peftleak_style_text_adapter_ratio_v2"
            if ratio
            else "fedllm_peft_gradient_inversion_opt_v2"
        ),
        "attack_variant": (
            "fixed_public_bins_registered_embedding_probe"
            if ratio
            else "defense_aware_lora_gradient_matching"
        ),
        "reproduction_level": (
            "peftleak_style_text_adaptation"
            if ratio
            else "optimization_based_peft_gradient_inversion_not_original_peftleak"
        ),
        "n_inputs_requested": requested,
        "n_inputs_completed": 0,
        "last_input_idx": None,
        "last_input_time": None,
        "last_total_time": None,
        "last_rec_status": None,
        "rec_token_values": [],
        "attack_loss_values": [],
        "optimization_loss_values": [],
        "recovered_position_values": [],
        "predictions": [],
        "references": [],
        "aggregate_metrics": {},
        "public_stats_source": "n/a",
        "public_stats_n_inputs": 0,
        "public_stats_num_tokens": 0,
        "probe_installed_before_private_data": False,
        "probe_inventory_count": 0,
        "probe_inventory_sha256": None,
        "shared_gradient_count": 0,
        "shared_gradient_names_sha256": None,
        "label_mode": "n/a",
        "optimization_loss_first": None,
        "optimization_loss_last": None,
        "optimization_loss_reduction": None,
        "error_type": None,
        "error_message": None,
        "summary_emitted": False,
    }


def _emit_summary(args, tracker):
    if tracker.get("summary_emitted"):
        return
    defense_name, defense_value = defense_param_spec(args)
    fields = [
        ("summary_version", tracker["summary_version"]),
        ("result_status", tracker["result_status"]),
        ("attack", tracker["attack"]),
        ("attack_variant", tracker["attack_variant"]),
        ("reproduction_level", tracker["reproduction_level"]),
        ("dataset", args.dataset),
        ("split", args.split),
        *dataset_summary_fields(args),
        ("task", args.task),
        ("model_path", args.model_path),
        ("finetuned_path", args.finetuned_path),
        ("batch_size", args.batch_size),
        ("seed", args.rng_seed),
        ("train_method", args.train_method),
        ("peft_method", args.peft_method),
        ("peft_type", getattr(args, "peft_type", None)),
        ("peft_eval_scope", args.peft_eval_scope),
        ("attack_mode", args.peftleak_attack_mode),
        ("defense", args.defense),
        ("defense_param_name", defense_name),
        ("defense_param_value", defense_value),
        *lrb_seed_summary_fields(args),
        ("defense_aware", args.peftleak_attack_mode == "opt"),
        ("classification_head_shared", False),
        ("declared_side_information", "sequence_length,attention_mask"),
        ("decoder_private_routing", False),
        ("probe_inventory_fixed", args.peftleak_attack_mode == "ratio"),
        ("probe_installed_before_private_data", tracker["probe_installed_before_private_data"]),
        ("probe_inventory_count", tracker["probe_inventory_count"]),
        ("probe_inventory_sha256", tracker["probe_inventory_sha256"]),
        ("public_stats_source", tracker["public_stats_source"]),
        ("public_stats_n_inputs", tracker["public_stats_n_inputs"]),
        ("public_stats_num_tokens", tracker["public_stats_num_tokens"]),
        ("peftleak_max_length", args.peftleak_max_length),
        ("peftleak_ratio_bins", args.peftleak_ratio_bins),
        ("peftleak_ratio_rows_per_bin", args.peftleak_ratio_rows_per_bin),
        ("peftleak_steps", args.peftleak_steps),
        ("peftleak_lr", args.peftleak_lr),
        ("peftleak_restarts", args.peftleak_restarts),
        ("peftleak_match_loss", args.peftleak_match_loss),
        ("label_mode", tracker["label_mode"]),
        ("text_metric_backend_requested", args.text_metric_backend),
        ("n_inputs_requested", tracker["n_inputs_requested"]),
        ("n_inputs_completed", tracker["n_inputs_completed"]),
        ("last_input_idx", tracker["last_input_idx"]),
        ("last_input_time", tracker["last_input_time"]),
        ("last_total_time", tracker["last_total_time"]),
        ("last_rec_status", tracker["last_rec_status"]),
        ("rec_token_mean", safe_mean(tracker["rec_token_values"])),
        ("attack_loss_mean", safe_mean(tracker["attack_loss_values"])),
        ("optimization_loss_mean", safe_mean(tracker["optimization_loss_values"])),
        ("optimization_loss_first", tracker["optimization_loss_first"]),
        ("optimization_loss_last", tracker["optimization_loss_last"]),
        ("optimization_loss_reduction", tracker["optimization_loss_reduction"]),
        ("recovered_position_count_mean", safe_mean(tracker["recovered_position_values"])),
        ("shared_gradient_count", tracker["shared_gradient_count"]),
        ("shared_gradient_names_sha256", tracker["shared_gradient_names_sha256"]),
    ]
    if tracker.get("error_type"):
        fields.append(("error_type", tracker["error_type"]))
    if tracker.get("error_message"):
        fields.append(("error_message", tracker["error_message"]))
    for key in sorted(tracker["aggregate_metrics"]):
        fields.append((key, tracker["aggregate_metrics"][key]))

    print(SUMMARY_START, flush=True)
    for key, value in fields:
        print(f"{key}={fmt_summary_value(value)}", flush=True)
    print(SUMMARY_END, flush=True)
    tracker["summary_emitted"] = True


def _build_public_batches(args, model_wrapper, dataset_cls):
    public_split = "test" if args.split == "val" else "val"
    numpy_state = np.random.get_state()
    try:
        public_dataset = dataset_cls(
            args.device,
            args.dataset,
            public_split,
            int(args.peftleak_ratio_public_n_inputs),
            1,
            args.cache_dir,
        )
        batches = [
            _tokenize(model_wrapper.tokenizer, public_dataset[index][0], args)
            for index in range(int(args.peftleak_ratio_public_n_inputs))
        ]
    finally:
        # The private split is constructed from the identical permutation and the disjoint partition.
        np.random.set_state(numpy_state)
    return batches, public_split


def _run_adapter_ratio(args, model_wrapper, installed, sample, ignored_token_ids):
    sequences, labels = sample
    batch = _tokenize(model_wrapper.tokenizer, sequences, args)
    observation = compute_probe_gradient_observation(model_wrapper, batch, labels, installed, args)
    decoded = recover_tokens_from_probe_gradients(
        observation.observed_gradients,
        observation.parameter_names,
        get_token_embedding_matrix(model_wrapper),
        max_positions=int(args.peftleak_max_length),
        num_bins=int(args.peftleak_ratio_bins),
        ignored_token_ids=ignored_token_ids,
        fallback_token_id=int(getattr(model_wrapper, "pad_token", 0) or 0),
    )
    attention_mask = batch.get("attention_mask")
    reference_mask = None if attention_mask is None else attention_mask.detach().cpu().tolist()
    reference_ids = batch["input_ids"].detach().cpu().tolist()
    rec_token = token_recovery_accuracy(
        decoded["predicted_ids"],
        reference_ids,
        ignored_token_ids=ignored_token_ids,
        reference_mask=reference_mask,
    )
    sequence_length = int(batch["input_ids"].shape[1])
    predicted_for_text = [row[:sequence_length] for row in decoded["predicted_ids"]]
    predictions = model_wrapper.tokenizer.batch_decode(predicted_for_text, skip_special_tokens=True)
    references = model_wrapper.tokenizer.batch_decode(reference_ids, skip_special_tokens=True)
    decoded.update(
        {
            "predictions": predictions,
            "references": references,
            "rec_token_mean": float(rec_token),
            "loss": observation.loss,
            "sequence_length": sequence_length,
        }
    )
    return decoded


def _run_lora_opt(args, model_wrapper, sample, ignored_token_ids):
    sequences, labels = sample
    batch = _tokenize(model_wrapper.tokenizer, sequences, args)
    result = optimize_lora_embeddings_defense_aware(
        model_wrapper=model_wrapper,
        batch=batch,
        labels=labels,
        args=args,
        steps=int(args.peftleak_steps),
        lr=float(args.peftleak_lr),
        restarts=int(args.peftleak_restarts),
        match_loss=args.peftleak_match_loss,
        known_labels=bool(args.peftleak_known_labels),
        label_candidates=_label_candidates(args.peftleak_label_candidates),
        ignored_token_ids=ignored_token_ids,
    )
    reference_ids = batch["input_ids"].detach().cpu().tolist()
    result["predictions"] = model_wrapper.tokenizer.batch_decode(result["predicted_ids"], skip_special_tokens=True)
    result["references"] = model_wrapper.tokenizer.batch_decode(reference_ids, skip_special_tokens=True)
    return result


def main(argv=None):
    args = build_parser().parse_args(argv)
    from utils.data import TextDataset
    from utils.models import ModelWrapper

    args.device = resolve_cuda_device(args.device)
    args.device_grad = resolve_gradient_device(args.device_grad, args.device)
    if args.device_grad != args.device:
        raise ValueError("Strict PEFT leakage v2 requires --device_grad to match --device.")
    _validate_args(args)
    validate_text_metric_backend(args.text_metric_backend)
    np.random.seed(int(args.rng_seed))
    torch.manual_seed(int(args.rng_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.rng_seed))

    tracker = _initial_tracker(args)
    start_time = time.time()
    print(
        f"[peftleak:v2] device={args.device} gradient_device={args.device_grad} "
        f"method={args.peft_method} mode={args.peftleak_attack_mode}",
        flush=True,
    )

    try:
        model_wrapper = ModelWrapper(args)
        installed = None
        if args.peftleak_attack_mode == "ratio":
            public_batches, public_split = _build_public_batches(args, model_wrapper, TextDataset)
            _, _, base_embedding = resolve_base_word_embedding(model_wrapper)
            statistics = build_public_probe_statistics(
                base_embedding,
                public_batches,
                max_positions=int(args.peftleak_max_length),
                num_bins=int(args.peftleak_ratio_bins),
            )
            installed = install_fixed_embedding_probe(
                model_wrapper,
                statistics,
                rows_per_bin=int(args.peftleak_ratio_rows_per_bin),
                seed=int(args.rng_seed),
            )
            tracker["public_stats_source"] = f"{public_split}_disjoint_partition"
            tracker["public_stats_n_inputs"] = statistics.num_sequences
            tracker["public_stats_num_tokens"] = statistics.num_tokens
            tracker["probe_installed_before_private_data"] = True
            tracker["probe_inventory_count"] = len(installed.parameter_names)
            tracker["probe_inventory_sha256"] = _inventory_hash(installed.parameter_names)
            tracker["shared_gradient_count"] = len(installed.parameter_names)
            tracker["shared_gradient_names_sha256"] = tracker["probe_inventory_sha256"]

        # This must stay after public calibration and probe installation for the Adapter threat model.
        private_dataset = TextDataset(
            args.device,
            args.dataset,
            args.split,
            args.n_inputs,
            args.batch_size,
            args.cache_dir,
        )
        record_dataset_protocol(args, private_dataset)
        ignored_token_ids = _ignored_token_ids(model_wrapper.tokenizer, model_wrapper)

        for input_index in range(int(args.start_input), min(int(args.n_inputs), int(args.end_input))):
            input_start = time.time()
            args.defense_rng_step = int(input_index)
            print(f"[peftleak:v2] Running input #{input_index} of {args.n_inputs}.", flush=True)
            if args.peftleak_attack_mode == "ratio":
                result = _run_adapter_ratio(args, model_wrapper, installed, private_dataset[input_index], ignored_token_ids)
                tracker["attack_loss_values"].append(float(result["loss"]))
                tracker["recovered_position_values"].append(float(result["recovered_position_count"]))
                tracker["label_mode"] = "observed_through_task_gradient"
            else:
                result = _run_lora_opt(args, model_wrapper, private_dataset[input_index], ignored_token_ids)
                tracker["optimization_loss_values"].append(float(result["loss"]))
                tracker["optimization_loss_first"] = result.get("initial_loss")
                tracker["optimization_loss_last"] = result.get("loss")
                tracker["optimization_loss_reduction"] = result.get("loss_reduction")
                tracker["label_mode"] = result.get("label_mode", "n/a")
                names = result.get("selected_gradient_names", [])
                tracker["shared_gradient_count"] = len(names)
                tracker["shared_gradient_names_sha256"] = _inventory_hash(names)

            tracker["predictions"].extend(result["predictions"])
            tracker["references"].extend(result["references"])
            tracker["rec_token_values"].append(float(result["rec_token_mean"]))
            tracker["n_inputs_completed"] += 1
            tracker["last_input_idx"] = input_index
            tracker["last_input_time"] = _duration(time.time() - input_start)
            tracker["last_total_time"] = _duration(time.time() - start_time)
            tracker["last_rec_status"] = "ok"
            print(
                f"[peftleak:v2] rec_token={float(result['rec_token_mean']):.6f} "
                f"shared_gradients={tracker['shared_gradient_count']}",
                flush=True,
            )

        tracker["aggregate_metrics"] = compute_text_metrics(
            tracker["predictions"],
            tracker["references"],
            backend=args.text_metric_backend,
        )
        tracker["last_total_time"] = tracker["last_total_time"] or _duration(time.time() - start_time)
        _emit_summary(args, tracker)
        return 0
    except Exception as exc:
        tracker["result_status"] = "failed"
        tracker["last_rec_status"] = tracker["last_rec_status"] or "failed"
        tracker["error_type"] = type(exc).__name__
        tracker["error_message"] = str(exc)
        tracker["last_total_time"] = tracker["last_total_time"] or _duration(time.time() - start_time)
        _emit_summary(args, tracker)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
