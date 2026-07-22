"""Standalone passive static-state inference experiment for Projection-LRB.

The entrypoint is intentionally separate from ``attack.py``.  It creates new
target cohorts, captures their defended gradients once, estimates static signs
from observed DAGER spans, and only then invokes the unchanged DAGER decoder
through the private opt-in state override hook.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import sys
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from datasets import load_metric

from utils.adaptive_attack import _lrb_feature_axis, _lrb_feature_signs, _oriented_grad_for_span
from utils.adaptive_lrb_state_inference import (
    STATE_INFERENCE_PROTOCOL,
    PublicCalibrationInitializer,
    PublicCalibrationRecord,
    StateInferenceObservation,
    StaticStateEstimator,
    fit_state,
    q_audit,
    sign_agreement_mod_global_flip,
    stage_grads_for_decode,
    stage_selected_grads_cpu,
    state_override,
)
from utils.data import TextDataset
from utils.defenses import apply_defense
from utils.models import ModelWrapper


EXPECTED_DAGER_INDICES = (4, 16)
# `attack.py` parses `sys.argv` at import time. Keep its import delayed
# until the state-specific flags have been stripped in `main`.
dagger_attack = None


@dataclass
class CapturedUpdate:
    sample: tuple
    defended_grads: tuple | None
    rank_B: int
    observation: StateInferenceObservation
    feature_sketches: tuple[torch.Tensor, ...]
    feature_widths: tuple[int, ...]
    audit_q: tuple[int, ...]
    audit_signs: tuple[torch.Tensor, ...]
    audit_projection_modes: tuple[str, ...]


def _custom_args(argv: Sequence[str]):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--state-calibration-batches", type=int, default=512)
    parser.add_argument("--state-calibration-seed", type=int, default=424242)
    parser.add_argument("--state-target-inputs", type=int, default=100)
    parser.add_argument("--state-fit-end", type=int, default=64)
    parser.add_argument("--state-eval-start", type=int, default=80)
    parser.add_argument("--state-eval-count", type=int, default=20)
    parser.add_argument("--state-m-values", default="1,4,16,64")
    parser.add_argument("--state-budgets", default="0,100,400")
    parser.add_argument("--state-candidate-count", type=int, default=2048)
    parser.add_argument("--state-candidate-seed", type=int, default=20260722)
    parser.add_argument("--state-learning-rate", type=float, default=0.05)
    parser.add_argument("--state-min-ratio", type=float, default=0.2)
    parser.add_argument("--state-max-ratio", type=float, default=0.9)
    parser.add_argument("--state-progress-every", type=int, default=8)
    custom, base_argv = parser.parse_known_args(argv)
    custom.m_values = tuple(int(part) for part in custom.state_m_values.split(",") if part.strip())
    custom.budgets = tuple(int(part) for part in custom.state_budgets.split(",") if part.strip())
    if not custom.m_values or not custom.budgets:
        raise ValueError("--state-m-values and --state-budgets must be non-empty CSV lists.")
    return custom, base_argv


def _load_dagger_after_state_parse(base_argv: Sequence[str]):
    """Import DAGER only after hiding this entrypoint's private CLI flags."""
    global dagger_attack
    original_sys_argv = sys.argv
    try:
        sys.argv = [original_sys_argv[0], *base_argv]
        dagger_attack = importlib.import_module("attack")
        return dagger_attack
    finally:
        sys.argv = original_sys_argv


def _validate_protocol(args, state_args) -> None:
    required = {
        "dataset": "sst2",
        "split": "official_validation",
        "model_path": "gpt2",
        "batch_size": 2,
        "defense": "lrb",
        "defense_lrb_preset": "proj_only",
        "defense_lrb_seed_mode": "static",
        "defense_lrb_seed": 700001,
        "adaptive_lrb_sign_source": "defense_device",
        "adaptive_lrb_knowledge": "method_only",
        "defense_lrb_projection": "signed_pool",
    }
    for name, expected in required.items():
        actual = getattr(args, name)
        if actual != expected:
            raise ValueError(f"{STATE_INFERENCE_PROTOCOL} requires --{name}={expected!r}; got {actual!r}")
    if abs(float(args.defense_lrb_keep_ratio_sensitive) - 0.5) > 1e-9:
        raise ValueError("state_inference_v1 requires --defense_lrb_keep_ratio_sensitive 0.5")
    if getattr(args, "adaptive_lrb_attack_seed", None) is None:
        raise ValueError("state_inference_v1 requires an independent --adaptive_lrb_attack_seed.")
    if args.n_layers != 2 or args.task != "seq_class" or args.algo != "sgd":
        raise ValueError("state_inference_v1 requires the full-gradient GPT-2 DAGER seq_class/sgd two-layer protocol.")
    if state_args.state_target_inputs < state_args.state_eval_start + state_args.state_eval_count:
        raise ValueError("Target cohort does not include the requested held-out evaluation range.")
    if max(state_args.m_values) > state_args.state_fit_end:
        raise ValueError("Every M must fit inside --state-fit-end.")
    if state_args.state_fit_end > state_args.state_eval_start:
        raise ValueError("State-fitting updates must not overlap the held-out evaluation range.")
    if state_args.state_progress_every <= 0:
        raise ValueError("--state-progress-every must be positive.")


def _batch_for_sample(args, wrapper: ModelWrapper, sample):
    sequences, labels = sample
    batch = wrapper.tokenizer(
        sequences,
        padding=True,
        truncation=True,
        max_length=min(wrapper.tokenizer.model_max_length, wrapper.emb_size - 20),
        return_tensors="pt",
    ).to(args.device)
    return batch, labels


def _candidate_bank(args, wrapper: ModelWrapper, count: int, seed: int) -> tuple[torch.Tensor, ...]:
    embeddings = wrapper.get_embeddings(0).detach()
    vocab = int(embeddings.shape[1])
    generator = torch.Generator(device=embeddings.device)
    generator.manual_seed(int(seed))
    ids = torch.randperm(vocab, generator=generator, device=embeddings.device)[: min(vocab, int(count))]
    bank = embeddings.index_select(1, ids).detach()
    # Both selected DAGER layers use public candidate vectors for the fitting
    # surrogate.  Full sequence assembly remains the unchanged DAGER decoder.
    return tuple(bank for _ in range(args.n_layers))


def _feature_sketch(args, grads, indices: Sequence[int]) -> tuple[torch.Tensor, ...]:
    sketches = []
    for index in indices:
        oriented = _oriented_grad_for_span(args, grads[index]).detach().float()
        sketches.append(oriented.reshape(-1, oriented.shape[-1]).mean(dim=0).cpu())
    return tuple(sketches)


def _audit_state(
    args, grads, indices: Sequence[int]
) -> tuple[tuple[int, ...], tuple[torch.Tensor, ...], tuple[str, ...]]:
    info_by_index = {int(item["idx"]): item for item in getattr(args, "lrb_defense_layer_info", []) if item.get("active")}
    q_values, signs, modes = [], [], []
    for index in indices:
        item = info_by_index.get(int(index))
        if item is None:
            raise RuntimeError(f"Missing LRB audit metadata for selected DAGER gradient {index}.")
        grad = grads[index]
        oriented = _oriented_grad_for_span(args, grad)
        width = int(oriented.shape[-1])
        q_values.append(max(1, int(round(float(item["keep_ratio"]) * width))))
        modes.append(str(item["projection_mode"]))
        axis = _lrb_feature_axis(args, grad, tuple(item["shape"]))
        vector = _lrb_feature_signs(
            int(item["projection_seed"]), tuple(item["shape"]), axis, device="cpu"
        )
        if vector is None or int(vector.numel()) != width:
            raise RuntimeError("Could not derive feature-axis audit signs for the selected DAGER matrix.")
        signs.append(vector.cpu())
    return tuple(q_values), tuple(signs), tuple(modes)


def _capture_update(
    args,
    wrapper: ModelWrapper,
    sample,
    update_index: int,
    candidate_values,
    *,
    retain_decode_grads: bool = False,
) -> CapturedUpdate:
    batch, labels = _batch_for_sample(args, wrapper, sample)
    args.defense_rng_step = int(update_index)
    raw_grads = wrapper.compute_grads(batch, labels)
    defended = apply_defense(raw_grads, args, model_wrapper=wrapper, batch=batch, labels=labels)
    rank_B, span_bases = wrapper.get_matrices_expansions(defended, B=None, tol=args.rank_tol)
    if rank_B is None:
        raise RuntimeError("State inference requires a valid captured DAGER rank.")
    selected = tuple(int(index) for index in args.dager_selected_gradient_indices)
    if selected != EXPECTED_DAGER_INDICES:
        raise RuntimeError(f"Expected selected DAGER indices {EXPECTED_DAGER_INDICES}; got {selected}.")
    widths = tuple(int(_oriented_grad_for_span(args, defended[index]).shape[-1]) for index in selected)
    if any(values.shape[-1] != width for values, width in zip(candidate_values, widths)):
        raise RuntimeError("Public candidate bank width no longer matches selected DAGER gradients.")
    q_values, signs, modes = _audit_state(args, defended, selected)
    staged_grads = stage_selected_grads_cpu(defended, selected) if retain_decode_grads else None
    return CapturedUpdate(
        sample=sample,
        defended_grads=staged_grads,
        rank_B=int(rank_B),
        observation=StateInferenceObservation(
            span_bases=tuple(base.detach().to(device="cpu", copy=True) for base in span_bases),
            candidate_values=tuple(value.detach() for value in candidate_values),
        ),
        feature_sketches=_feature_sketch(args, defended, selected),
        feature_widths=widths,
        audit_q=q_values,
        audit_signs=signs,
        audit_projection_modes=modes,
    )


def _capture_public_calibration_record(args, wrapper: ModelWrapper, sample, update_index: int):
    """Capture only the public initializer inputs, without a DAGER SVD.

    Public calibration consumes feature sketches and attacker-generated state
    labels.  Computing span bases here was both unused and the dominant cost of
    the original 512-batch calibration phase.
    """
    batch, labels = _batch_for_sample(args, wrapper, sample)
    args.defense_rng_step = int(update_index)
    raw_grads = wrapper.compute_grads(batch, labels)
    defended = apply_defense(raw_grads, args, model_wrapper=wrapper, batch=batch, labels=labels)
    selected = tuple(int(index) for index in args.dager_selected_gradient_indices)
    if selected != EXPECTED_DAGER_INDICES:
        raise RuntimeError(f"Expected selected DAGER indices {EXPECTED_DAGER_INDICES}; got {selected}.")
    widths = tuple(int(_oriented_grad_for_span(args, defended[index]).shape[-1]) for index in selected)
    q_values, signs, _ = _audit_state(args, defended, selected)
    record = PublicCalibrationRecord(
        feature_sketches=_feature_sketch(args, defended, selected),
        ratios=tuple(q / width for q, width in zip(q_values, widths)),
        signs=signs,
    )
    return record, widths


def _hash_dataset_indices(dataset: TextDataset) -> str:
    digest = hashlib.sha256()
    for index in dataset.selected_indices:
        digest.update(int(index).to_bytes(8, byteorder="little", signed=False))
    return digest.hexdigest()


def _progress(phase: str, completed: int, total: int, every: int, **fields) -> None:
    if completed == 1 or completed == total or completed % every == 0:
        suffix = " ".join(f"{key}={value}" for key, value in fields.items())
        print(
            f"[state-inference] phase={phase} progress={completed}/{total}"
            + (f" {suffix}" if suffix else ""),
            flush=True,
        )


def _release_device_cache(args) -> None:
    gc.collect()
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _public_calibration(args, wrapper: ModelWrapper, state_args, feature_widths):
    # split=val still draws only the SST-2 train partition; it is kept separate
    # from the official-validation target cohort and never exposes target text.
    np.random.seed(int(state_args.state_calibration_seed))
    dataset = TextDataset(
        args.device, "sst2", "val", int(state_args.state_calibration_batches), args.batch_size, args.cache_dir
    )
    records = []
    original_seed = args.defense_lrb_seed
    original_step = getattr(args, "defense_rng_step", None)
    try:
        for index in range(len(dataset.seqs)):
            args.defense_lrb_seed = int(state_args.state_calibration_seed) + 1_000_003 * index
            record, widths = _capture_public_calibration_record(args, wrapper, dataset[index], 0)
            if widths != tuple(feature_widths):
                raise RuntimeError("Public calibration changed the selected DAGER feature widths.")
            records.append(record)
            _progress(
                "public_calibration",
                index + 1,
                len(dataset.seqs),
                int(state_args.state_progress_every),
            )
    finally:
        args.defense_lrb_seed = original_seed
        args.defense_rng_step = original_step
    initializer = PublicCalibrationInitializer(feature_widths)
    initializer.fit(records)
    return initializer, _hash_dataset_indices(dataset)


def _state_summary(**fields) -> None:
    print("===== STATE INFERENCE SUMMARY START =====", flush=True)
    for key, value in fields.items():
        print(f"{key}={value}", flush=True)
    print("===== STATE INFERENCE SUMMARY END =====", flush=True)


def _oracle_override(update: CapturedUpdate) -> dict:
    """Create the explicit diagnostic state only at oracle decode time.

    This helper deliberately consumes the audited state after fitting.  It is
    never passed to the estimator or its public initializer.
    """
    layers = {}
    for position, (q_value, width, signs, mode) in enumerate(
        zip(update.audit_q, update.feature_widths, update.audit_signs, update.audit_projection_modes)
    ):
        layers[position] = {
            "lrb_keep_ratio": float(q_value) / float(width),
            "lrb_projection_mode": mode,
            "lrb_projection_seed": -1,
            "lrb_feature_signs": signs,
            "lrb_feature_axis": "oracle_audit",
        }
    return {"layers": layers, "profile": "state_inference_v1_oracle"}


def _evaluate_condition(
    args,
    wrapper,
    metric,
    captured,
    estimator,
    state_args,
    *,
    attack_variant: str,
    m_value: int | str,
    budget: int | str,
):
    if attack_variant not in {"method_only", "state_estimator", "oracle"}:
        raise ValueError(f"Unsupported state-inference attack variant: {attack_variant}")
    if attack_variant == "state_estimator" and estimator is None:
        raise ValueError("state_estimator evaluation requires a fitted estimator.")
    if attack_variant != "state_estimator" and estimator is not None:
        raise ValueError("Only the estimator variant may receive estimator parameters.")
    start = int(state_args.state_eval_start)
    end = start + int(state_args.state_eval_count)
    dagger_attack.total_correct_tokens = 0
    dagger_attack.total_tokens = 0
    dagger_attack.total_correct_maxB_tokens = 0
    predictions, references = [], []
    l1_recovery, l1_topb_recovery, l2_recovery = [], [], []
    t0 = time.time()
    for update_index in range(start, end):
        update = captured[update_index]
        if update.defended_grads is None:
            raise RuntimeError(f"Held-out update {update_index} is missing its CPU-staged decode gradients.")
        if attack_variant == "state_estimator":
            args._state_inference_override = state_override(estimator, update_index)
        elif attack_variant == "oracle":
            args._state_inference_override = _oracle_override(update)
        else:
            args._state_inference_override = None
        args.result_tracker = dagger_attack._init_result_tracker(args)
        decode_grads = stage_grads_for_decode(update.defended_grads, torch.device(args.device))
        decode_expansions = (
            update.rank_B,
            tuple(base.to(args.device, non_blocking=False) for base in update.observation.span_bases),
        )
        try:
            predicted, reference = dagger_attack.reconstruct(
                args,
                torch.device(args.device),
                update.sample,
                metric,
                wrapper,
                precomputed_true_grads=decode_grads,
                defense_rng_step=update_index,
                precomputed_matrices_expansions=decode_expansions,
            )
        finally:
            del decode_grads, decode_expansions
        predictions.extend(predicted)
        references.extend(reference)
        l1_recovery.extend(args.result_tracker.get("rec_l1_mean_values", []))
        l1_topb_recovery.extend(args.result_tracker.get("rec_l1_maxb_mean_values", []))
        l2_recovery.extend(args.result_tracker.get("rec_l2_mean_values", []))
        _progress(
            f"decode_{attack_variant}",
            update_index - start + 1,
            end - start,
            int(state_args.state_progress_every),
        )
    args._state_inference_override = None
    metrics = metric.compute(predictions=predictions, references=references)
    if attack_variant == "state_estimator":
        estimated_q_rows = [
            [value.item() for value in estimator.q_values(update_index)]
            for update_index in range(start, end)
        ]
        q_rows = [
            q_audit(estimated_q, captured[update_index].audit_q)
            for estimated_q, update_index in zip(estimated_q_rows, range(start, end))
        ]
        ratio_errors = [
            sum(
                abs(float(estimated) - float(truth)) / float(width)
                for estimated, truth, width in zip(
                    estimated_q,
                    captured[update_index].audit_q,
                    captured[update_index].feature_widths,
                )
            )
            / len(captured[update_index].feature_widths)
            for estimated_q, update_index in zip(estimated_q_rows, range(start, end))
        ]
        sign_scores = [
            sign_agreement_mod_global_flip(estimate, truth)
            for estimate, truth in zip(estimator.hard_signs(), captured[start].audit_signs)
        ]
        q_exact, q_error, ratio_error, sign_agreement = (
            f"{sum(row['q_exact_match'] for row in q_rows) / len(q_rows):.6f}",
            f"{sum(row['q_abs_error_mean'] for row in q_rows) / len(q_rows):.6f}",
            f"{sum(ratio_errors) / len(ratio_errors):.6f}",
            f"{sum(sign_scores) / len(sign_scores):.6f}",
        )
    elif attack_variant == "oracle":
        q_exact, q_error, ratio_error, sign_agreement = "1.000000", "0.000000", "0.000000", "1.000000"
    else:
        q_exact = q_error = ratio_error = sign_agreement = "n/a"
    token_total = max(int(dagger_attack.total_tokens), 1)
    return {
        "protocol": STATE_INFERENCE_PROTOCOL,
        "result_status": "ok",
        "state_lifecycle": "static",
        "state_attack_variant": attack_variant,
        "state_fit_updates": m_value,
        "state_budget": budget,
        "n_inputs_requested": int(state_args.state_eval_count),
        "n_inputs_completed": int(state_args.state_eval_count),
        "rec_token_mean": f"{dagger_attack.total_correct_tokens / token_total:.6f}",
        "rec_maxb_token_mean": f"{dagger_attack.total_correct_maxB_tokens / token_total:.6f}",
        "agg_r1fm_r2fm": f"{(metrics['rouge1'].mid.fmeasure + metrics['rouge2'].mid.fmeasure) * 100:.6f}",
        "dagger_candidate_recovery_mean": f"{sum(l1_recovery) / max(len(l1_recovery), 1):.6f}",
        "dagger_topb_candidate_recovery_mean": f"{sum(l1_topb_recovery) / max(len(l1_topb_recovery), 1):.6f}",
        "dagger_l2_recovery_mean": f"{sum(l2_recovery) / max(len(l2_recovery), 1):.6f}",
        "state_q_exact_match": q_exact,
        "state_q_abs_error_mean": q_error,
        "state_ratio_abs_error_mean": ratio_error,
        "state_sign_agreement_mod_global_flip": sign_agreement,
        "attack_time_seconds": f"{time.time() - t0:.3f}",
        "state_truth_used_for_fit": "false",
        "state_truth_used_for_decode": str(attack_variant == "oracle").lower(),
        "state_target_fit_visibility": "observed_dager_spans_only",
        "state_selected_gradients": ";".join(str(index) for index in EXPECTED_DAGER_INDICES),
        "state_decode_gradient_storage": "held_out_selected_cpu",
        "state_dager_expansions_source": "captured_precomputed",
        "state_public_calibration_decomposition": "skipped",
    }


def main(argv: Sequence[str] | None = None) -> int:
    state_args, base_argv = _custom_args(sys.argv[1:] if argv is None else argv)
    dagger_module = _load_dagger_after_state_parse(base_argv)
    args = dagger_module.args
    _validate_protocol(args, state_args)
    args.result_tracker = dagger_attack._init_result_tracker(args)
    try:
        np.random.seed(int(args.rng_seed))
        torch.manual_seed(int(args.rng_seed))
        metric = load_metric("rouge", cache_dir=args.cache_dir)
        target_dataset = TextDataset(
            args.device, args.dataset, args.split, int(state_args.state_target_inputs), args.batch_size, args.cache_dir
        )
        target_index_hash = _hash_dataset_indices(target_dataset)
        wrapper = ModelWrapper(args)
        candidates = _candidate_bank(args, wrapper, state_args.state_candidate_count, state_args.state_candidate_seed)
        eval_start = int(state_args.state_eval_start)
        eval_end = eval_start + int(state_args.state_eval_count)
        captured = []
        for index in range(len(target_dataset.seqs)):
            captured.append(
                _capture_update(
                args,
                wrapper,
                target_dataset[index],
                index,
                candidates,
                retain_decode_grads=eval_start <= index < eval_end,
            )
            )
            _progress(
                "target_capture",
                index + 1,
                len(target_dataset.seqs),
                int(state_args.state_progress_every),
            )
        retained_updates = sum(update.defended_grads is not None for update in captured)
        if retained_updates != int(state_args.state_eval_count):
            raise RuntimeError(
                f"Expected CPU decode gradients for {state_args.state_eval_count} held-out updates; "
                f"retained {retained_updates}."
            )
        widths = captured[0].feature_widths
        if any(update.feature_widths != widths for update in captured):
            raise RuntimeError("Selected DAGER widths changed within the target cohort.")
        _release_device_cache(args)
        initializer, calibration_hash = _public_calibration(args, wrapper, state_args, widths)
        _release_device_cache(args)
        target_sketches = [update.feature_sketches for update in captured]

        # These two diagnostics decode the same fixed blind updates as every
        # estimator condition.  The former is the unchanged finite-hypothesis
        # DAGER setting; the latter is explicitly marked as an oracle and is
        # never used while fitting state parameters.
        for attack_variant in ("method_only", "oracle"):
            fields = _evaluate_condition(
                args,
                wrapper,
                metric,
                captured,
                None,
                state_args,
                attack_variant=attack_variant,
                m_value="n/a",
                budget="n/a",
            )
            fields.update(
                {
                    "dataset": args.dataset,
                    "split": args.split,
                    "checkpoint": args.finetuned_path,
                    "rng_seed": args.rng_seed,
                    "defense": args.defense,
                    "defense_lrb_preset": args.defense_lrb_preset,
                    "defense_lrb_seed": args.defense_lrb_seed,
                    "defense_lrb_seed_mode": args.defense_lrb_seed_mode,
                    "adaptive_lrb_sign_source": args.adaptive_lrb_sign_source,
                    "calibration_partition": "sst2_train_legacy_val",
                    "calibration_batches": state_args.state_calibration_batches,
                    "calibration_hash": calibration_hash,
                    "target_index_hash": target_index_hash,
                    "state_fit_loss_last": "n/a",
                }
            )
            _state_summary(**fields)
            _release_device_cache(args)

        for m_value in state_args.m_values:
            fit_indices = tuple(range(int(m_value)))
            fit_observations = [captured[index].observation for index in fit_indices]
            init_signs, _ = initializer.predict([target_sketches[index] for index in fit_indices])
            _, all_ratios = initializer.predict(target_sketches)
            for budget in state_args.budgets:
                estimator = StaticStateEstimator(
                    widths,
                    len(captured),
                    min_ratio=state_args.state_min_ratio,
                    max_ratio=state_args.state_max_ratio,
                ).to(args.device)
                estimator.install_initializer(init_signs, all_ratios)
                trace = fit_state(
                    estimator,
                    fit_observations,
                    update_indices=fit_indices,
                    steps=int(budget),
                    learning_rate=state_args.state_learning_rate,
                    progress_callback=lambda completed, total, loss, m=m_value, b=budget: _progress(
                        "state_fit",
                        completed,
                        total,
                        int(state_args.state_progress_every),
                        m=m,
                        budget=b,
                        loss=f"{loss:.8f}",
                    ),
                )
                fields = _evaluate_condition(
                    args,
                    wrapper,
                    metric,
                    captured,
                    estimator,
                    state_args,
                    attack_variant="state_estimator",
                    m_value=int(m_value),
                    budget=int(budget),
                )
                fields.update(
                    {
                        "dataset": args.dataset,
                        "split": args.split,
                        "checkpoint": args.finetuned_path,
                        "rng_seed": args.rng_seed,
                        "defense": args.defense,
                        "defense_lrb_preset": args.defense_lrb_preset,
                        "defense_lrb_seed": args.defense_lrb_seed,
                        "defense_lrb_seed_mode": args.defense_lrb_seed_mode,
                        "adaptive_lrb_sign_source": args.adaptive_lrb_sign_source,
                        "calibration_partition": "sst2_train_legacy_val",
                        "calibration_batches": state_args.state_calibration_batches,
                        "calibration_hash": calibration_hash,
                        "target_index_hash": target_index_hash,
                        "state_fit_loss_last": "n/a" if not trace else f"{trace[-1]:.8f}",
                    }
                )
                _state_summary(**fields)
                del estimator
                _release_device_cache(args)
        return 0
    except Exception as exc:
        _state_summary(protocol=STATE_INFERENCE_PROTOCOL, result_status="failed", error_type=type(exc).__name__, error_message=str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
