#!/usr/bin/env python3
"""Run one preregistered none-only Qwen3/RoPE DAGER attack; no LRB is available."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping


QWEN_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = QWEN_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(QWEN_ROOT) not in sys.path:
    sys.path.insert(0, str(QWEN_ROOT))

import torch

from src.config import ExperimentConfig, load_experiment_config
from src.dager_qwen3 import ATTACK_NAME
from src.dager_qwen3.candidate_provider import RoPECandidateProvider
from src.dager_qwen3.diagnostics import (
    AttackProtocolError,
    load_none_attack_controls,
    load_registered_sample,
    none_only_attack_metadata,
    registered_head_seed,
)
from src.dager_qwen3.gradient_decomposition import (
    decompose_qwen3_qproj_gradient,
    shared_dager_rank_for_qwen3_qproj_gradients,
)
from src.dager_qwen3.layer1_filter import filter_qwen3_vocab_layer1
from src.dager_qwen3.layer2_decoder import Layer2DecoderConfig, decode_qwen3_rope_prefixes
from src.dager_qwen3.metrics import compute_attack_metrics, load_existing_dager_rouge_metric
from src.dager_qwen3.model_adapter import Qwen3RoPEDagerAdapter
from src.gradient_capture import build_canonical_gradient_manifest, capture_single_example_gradients
from src.qwen3_classifier import load_local_qwen3_sequence_classifier
from src.result_schema import ResultSchemaError, write_or_verify_jsonl
from src.span_diagnostics import diagnose_two_q_projections


class NoneAttackScriptError(RuntimeError):
    """Raised when this fixed-protocol attack request is not executable."""


def _resolve_repository_path(value: str, *, description: str) -> Path:
    candidate = Path(value).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (REPOSITORY_ROOT / candidate).resolve()
    try:
        resolved.relative_to(QWEN_ROOT)
    except ValueError as error:
        raise NoneAttackScriptError(f"{description} must remain under {QWEN_ROOT}, got {resolved}.") from error
    return resolved


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value if value else None


def _decode_token_texts(tokenizer: Any, token_ids: tuple[int, ...]) -> list[str]:
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if not callable(convert):
        raise NoneAttackScriptError("Qwen3 tokenizer lacks convert_ids_to_tokens.")
    values = convert(list(token_ids))
    if not isinstance(values, list) or len(values) != len(token_ids) or any(not isinstance(value, str) for value in values):
        raise NoneAttackScriptError("Qwen3 tokenizer returned invalid convert_ids_to_tokens values.")
    return values


def _diagnostic_thresholds(dtype: str) -> dict[str, float]:
    if dtype not in ("bfloat16", "float32"):
        raise NoneAttackScriptError(f"Unsupported dtype {dtype!r}.")
    return {
        "rank_atol": 1e-6,
        "rank_rtol": 1e-3,
        "delta_rtol": 1e-3,
        "identity_error_tol": 5e-3,
        "max_active_relative_residual": 5e-4 if dtype == "bfloat16" else 1e-4,
        "negative_control_factor": 10.0,
    }


def _q_canonical_indices(manifest: Mapping[str, Any], names: tuple[str, str]) -> dict[str, int]:
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise NoneAttackScriptError("Canonical gradient manifest lacks entries.")
    result: dict[str, int] = {}
    for entry in entries:
        if isinstance(entry, Mapping) and entry.get("name") in names:
            index = entry.get("canonical_index")
            if isinstance(index, int) and not isinstance(index, bool):
                result[str(entry["name"])] = index
    if set(result) != set(names):
        raise NoneAttackScriptError("Canonical gradient manifest does not contain both structural q_proj parameter names.")
    return result


def parse_args() -> argparse.Namespace:
    """Parse one immutable-manifest attack request; free-text inputs are intentionally absent."""
    parser = argparse.ArgumentParser(
        description=(
            "Run defense-unaware DAGER on one preregistered Qwen3 SST-2 sample. "
            "Only defense=none is implemented; no LRB code is loaded."
        )
    )
    parser.add_argument(
        "--config",
        default="Projection_lrb_qwen3/configs/experiment.json",
        help="Repository-relative immutable Qwen3 experiment.json.",
    )
    parser.add_argument("--stage", choices=("calibration", "smoke", "final"), required=True)
    parser.add_argument("--sample-key", required=True, help="SHA256 sample key from the selected immutable stage manifest.")
    parser.add_argument("--head-seed", required=True, type=int, help="Registered random classifier-head seed for this stage.")
    parser.add_argument("--defense", choices=("none",), default="none", help="Only none is supported by this entrypoint.")
    parser.add_argument("--device", default="cuda", help="Explicit CUDA device, e.g. cuda or cuda:0.")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--output",
        required=True,
        help="Repository-relative JSONL output path under Projection_lrb_qwen3/outputs/.",
    )
    return parser.parse_args()


def _result_identity(
    *, preregistration_sha256: str, stage: str, sample_key: str, head_seed: int, dtype: str
) -> str:
    # A deterministic identifier enables write-or-verify recovery without
    # allowing one sample/seed to overwrite a distinct attack configuration.
    import hashlib

    payload = f"{ATTACK_NAME}|none|{preregistration_sha256}|{stage}|{sample_key}|{head_seed}|{dtype}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_attack(args: argparse.Namespace) -> dict[str, Any]:
    """Execute capture, raw-gradient DAGER decomposition, filtering, and decoding."""
    if args.defense != "none":
        raise NoneAttackScriptError("This entrypoint only permits defense=none.")
    config_path = _resolve_repository_path(args.config, description="config path")
    config: ExperimentConfig = load_experiment_config(config_path)
    registered_head_seed(config, stage=args.stage, requested_seed=args.head_seed)
    sample = load_registered_sample(config=config, stage=args.stage, sample_key=args.sample_key)
    controls = load_none_attack_controls(config)
    rouge_metric = load_existing_dager_rouge_metric()
    output_path = _resolve_repository_path(args.output, description="output path")
    outputs_root = (QWEN_ROOT / "outputs").resolve()
    try:
        output_path.relative_to(outputs_root)
    except ValueError as error:
        raise NoneAttackScriptError(f"Attack output must remain under {outputs_root}, got {output_path}.") from error
    if output_path.suffix != ".jsonl":
        raise NoneAttackScriptError("Attack output must use a .jsonl filename.")

    torch.manual_seed(args.head_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.head_seed)
    bundle = load_local_qwen3_sequence_classifier(
        config.model_path,
        head_seed=args.head_seed,
        device=args.device,
        dtype=args.dtype,
    )
    if getattr(bundle.tokenizer, "eos_token_id", None) != sample.eos_token_id:
        raise NoneAttackScriptError(
            f"Manifest EOS id {sample.eos_token_id} differs from loaded Qwen3 tokenizer EOS "
            f"{getattr(bundle.tokenizer, 'eos_token_id', None)!r}."
        )
    input_ids = torch.tensor([sample.input_ids], dtype=torch.long, device=bundle.device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=bundle.device)
    labels = torch.tensor([sample.label], dtype=torch.long, device=bundle.device)
    capture_started = perf_counter()
    captured = capture_single_example_gradients(
        bundle.model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    capture_seconds = perf_counter() - capture_started
    canonical_manifest = build_canonical_gradient_manifest(bundle.model)
    canonical_indices = _q_canonical_indices(canonical_manifest, captured.q_parameter_names)
    diagnostic_controls = _diagnostic_thresholds(args.dtype)
    token_texts = _decode_token_texts(bundle.tokenizer, sample.input_ids)
    diagnostic = diagnose_two_q_projections(
        q_inputs=captured.q_inputs,
        q_output_gradients=captured.q_output_gradients,
        q_gradients=captured.q_gradients,
        q_parameter_names=captured.q_parameter_names,
        token_ids=sample.input_ids,
        token_texts=token_texts,
        eos_token_id=sample.eos_token_id,
        **diagnostic_controls,
    )
    if not diagnostic["passed"]:
        raise NoneAttackScriptError("Qwen3 gradient diagnostic failed; refusing to run DAGER with an unverified orientation.")

    attack_started = perf_counter()
    adapter = Qwen3RoPEDagerAdapter(bundle.model, bundle.tokenizer)
    shared_rank, shared_raw_ranks = shared_dager_rank_for_qwen3_qproj_gradients(
        captured.q_gradients,
        feature_dim=adapter.metadata.hidden_size,
        rank_tolerance=controls.rank_tolerance,
        rank_cutoff=controls.rank_cutoff,
        decomposition_device=bundle.device,
    )
    q0_span = decompose_qwen3_qproj_gradient(
        captured.q_gradients[0],
        feature_dim=adapter.metadata.hidden_size,
        rank_tolerance=controls.rank_tolerance,
        rank_cutoff=controls.rank_cutoff,
        decomposition_device=bundle.device,
        shared_truncated_rank=shared_rank,
    )
    q1_span = decompose_qwen3_qproj_gradient(
        captured.q_gradients[1],
        feature_dim=adapter.metadata.hidden_size,
        rank_tolerance=controls.rank_tolerance,
        rank_cutoff=controls.rank_cutoff,
        decomposition_device=bundle.device,
        shared_truncated_rank=shared_rank,
    )
    layer1 = filter_qwen3_vocab_layer1(
        adapter=adapter,
        span=q0_span,
        threshold=controls.l1_span_threshold,
        vocab_chunk_size=controls.vocab_chunk_size,
        distance_norm="l2",
    )
    candidate_provider = RoPECandidateProvider.from_layer1_result(
        layer1,
        eos_token_id=sample.eos_token_id,
        max_ids=controls.max_candidate_ids,
    )
    layer2 = decode_qwen3_rope_prefixes(
        adapter=adapter,
        span=q1_span,
        candidate_provider=candidate_provider,
        config=Layer2DecoderConfig(
            max_sequence_length=controls.max_sequence_length,
            threshold=controls.l2_span_threshold,
            distance_norm="l2",
            search_budget=controls.max_search_candidates,
            decode_batch_size=controls.decode_batch_size,
        ),
    )
    attack_seconds = perf_counter() - attack_started
    metrics = compute_attack_metrics(
        tokenizer=bundle.tokenizer,
        ground_truth_token_ids=sample.input_ids,
        reconstructed_token_ids=layer2.selected_token_ids,
        eos_token_id=sample.eos_token_id,
        rouge_metric=rouge_metric,
    )
    identity = _result_identity(
        preregistration_sha256=sample.preregistration_sha256,
        stage=sample.stage,
        sample_key=sample.sample_key,
        head_seed=args.head_seed,
        dtype=args.dtype,
    )
    record: dict[str, Any] = {
        "schema_version": 1,
        "record_type": "qwen3_dager_attack_result",
        "result_identity_sha256": identity,
        **none_only_attack_metadata(),
        "status": "ok" if not layer2.search_budget_exhausted else "search_budget_exhausted",
        "sample_id": sample.sample_key,
        "sample_key": sample.sample_key,
        "original_index": sample.original_index,
        "stage": sample.stage,
        "preregistration_sha256": sample.preregistration_sha256,
        "head_seed": args.head_seed,
        "dtype": args.dtype,
        "ground_truth_token_ids": list(sample.input_ids),
        "ground_truth_token_text": list(metrics.ground_truth_token_text),
        "ground_truth_text": metrics.ground_truth_text,
        "reconstructed_token_ids": list(layer2.selected_token_ids),
        "reconstructed_token_text": list(metrics.reconstructed_token_text),
        "reconstructed_text": metrics.reconstructed_text,
        "token_recovery": metrics.token_recovery,
        "exact_recovery": metrics.exact_recovery,
        "rouge_1": metrics.rouge_1,
        "rouge_2": metrics.rouge_2,
        "empty_reconstruction": metrics.empty_reconstruction,
        "layer_1_candidate_count": layer1.candidate_count,
        "layer_1_decoder_candidate_count": len(candidate_provider.token_ids),
        "layer_1_rank": q0_span.truncated_rank,
        "layer_1_raw_rank": shared_raw_ranks[0],
        "layer_2_rank": q1_span.truncated_rank,
        "layer_2_raw_rank": shared_raw_ranks[1],
        "shared_dager_rank": shared_rank,
        "attack_time_seconds": attack_seconds,
        "gradient_capture_time_seconds": capture_seconds,
        "loss": captured.loss,
        "gpu_peak_memory_bytes": captured.gpu_peak_memory_bytes,
        "thresholds": {
            "l1_span_thresh": controls.l1_span_threshold,
            "l2_span_thresh": controls.l2_span_threshold,
            "rank_tol": controls.rank_tolerance,
            "rank_cutoff": controls.rank_cutoff,
            "distance_norm": "l2",
        },
        "search_budget": {
            "maxC": controls.max_search_candidates,
            "parallel": controls.decode_batch_size,
            "vocab_chunk_size": controls.vocab_chunk_size,
            "max_ids": controls.max_candidate_ids,
            "max_length": controls.max_sequence_length,
            "evaluated_prefix_count": layer2.evaluated_prefix_count,
            "search_budget_exhausted": layer2.search_budget_exhausted,
            "per_length_survivor_counts": [list(value) for value in layer2.per_length_survivor_counts],
        },
        "layer_1_chunk_diagnostics": [
            {
                "start_token_id": item.start_token_id,
                "end_token_id_exclusive": item.end_token_id_exclusive,
                "elapsed_seconds": item.elapsed_seconds,
                "passing_candidate_count": item.passing_candidate_count,
            }
            for item in layer1.chunk_diagnostics
        ],
        "layer_2_completed_prefix_count": len(layer2.completed_prefixes),
        "selected_layer_2_mean_span_distance": layer2.selected_mean_span_distance,
        "q_proj": {
            "parameter_names": list(captured.q_parameter_names),
            "canonical_indices": canonical_indices,
            "gradient_shapes": [list(gradient.shape) for gradient in captured.q_gradients],
            "orientation": "raw_qwen3_nn_linear_gradient_right_singular_vectors",
        },
        "canonical_gradient_summary": {
            "gradient_tensor_count": canonical_manifest["gradient_tensor_count"],
            "gradient_numel": canonical_manifest["gradient_numel"],
        },
        "diagnostic_status": "ok",
        "gradient_diagnostic": diagnostic,
        "adapter": {
            "execution_path": adapter.metadata.execution_path,
            "hidden_size": adapter.metadata.hidden_size,
            "vocab_size": adapter.metadata.vocab_size,
        },
        "git_commit": _git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    try:
        # The timestamp is operational metadata rather than immutable identity;
        # one completed run is retained when all scientific fields match.
        if output_path.exists():
            existing = output_path.read_text(encoding="utf-8").splitlines()
            if len(existing) == 1:
                existing_record = json.loads(existing[0])
                if isinstance(existing_record, dict):
                    existing_record.pop("created_at", None)
                    comparable = dict(record)
                    comparable.pop("created_at", None)
                    if existing_record == comparable:
                        return record
        write_or_verify_jsonl(output_path, [record])
    except (OSError, json.JSONDecodeError, ResultSchemaError) as error:
        raise NoneAttackScriptError(f"Unable to recoverably write attack JSONL {output_path}: {error}") from error
    return record


def main() -> int:
    args = parse_args()
    try:
        record = run_attack(args)
    except Exception as error:
        print(
            json.dumps(
                {
                    "record_type": "qwen3_dager_attack_error",
                    "attack_name": ATTACK_NAME,
                    "defense": "none",
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(json.dumps(record, sort_keys=True), flush=True)
    return 0 if record["status"] == "ok" else 3


if __name__ == "__main__":
    raise SystemExit(main())
