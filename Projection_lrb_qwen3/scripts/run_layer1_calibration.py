#!/usr/bin/env python3
"""Observe Qwen3 DAGER Layer-1 distances without invoking recovery or Layer-2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence


QWEN_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = QWEN_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(QWEN_ROOT) not in sys.path:
    sys.path.insert(0, str(QWEN_ROOT))

import torch

from src.config import ExperimentConfig, load_experiment_config
from src.dager_qwen3.diagnostics import load_none_attack_controls, registered_head_seed
from src.dager_qwen3.gradient_decomposition import (
    decompose_qwen3_qproj_gradient,
    shared_dager_rank_for_qwen3_qproj_gradients,
)
from src.dager_qwen3.gradient_gate import diagnose_captured_q_projections
from src.dager_qwen3.bf16_gate_profile_amendment import (
    AMENDMENT_RELATIVE_PATH as BF16_GATE_PROFILE_AMENDMENT_RELATIVE_PATH,
    verify_amendment as verify_bf16_gate_profile_amendment,
)
from src.dager_qwen3.layer1_calibration import (
    Layer1CalibrationError,
    aggregate_calibration_records,
    build_sample_record,
    calibration_aggregation_identity,
    ordered_calibration_samples,
    write_or_verify_aggregation,
    write_or_verify_distance_sidecar,
    write_or_verify_sample_record,
)
from src.dager_qwen3.layer1_calibration_amendment import verify_amendment
from src.dager_qwen3.layer1_filter import scan_qwen3_vocab_layer1_distances
from src.dager_qwen3.model_adapter import Qwen3RoPEDagerAdapter
from src.gradient_capture import capture_single_example_gradients
from src.hashing import sha256_file
from src.qwen3_classifier import load_local_qwen3_sequence_classifier


class Layer1CalibrationScriptError(RuntimeError):
    """Raised when the isolated Layer-1 calibration execution cannot proceed."""


def _resolve_repository_path(value: str, *, description: str) -> Path:
    candidate = Path(value).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (REPOSITORY_ROOT / candidate).resolve()
    try:
        resolved.relative_to(QWEN_ROOT)
    except ValueError as error:
        raise Layer1CalibrationScriptError(
            f"{description} must remain under {QWEN_ROOT}, got {resolved}."
        ) from error
    return resolved


def _load_preregistration(config: ExperimentConfig) -> Mapping[str, Any]:
    path = config.project_root / "manifests" / "preregistration.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Layer1CalibrationScriptError(f"Unable to read preregistration identity {path}: {error}") from error
    if not isinstance(document, Mapping):
        raise Layer1CalibrationScriptError("preregistration.json must contain one object.")
    if document.get("config_sha256") != config.config_sha256:
        raise Layer1CalibrationScriptError(
            "Current experiment.json differs from preregistration.json; calibration is not allowed."
        )
    if not isinstance(document.get("preregistration_sha256"), str):
        raise Layer1CalibrationScriptError("preregistration.json lacks preregistration_sha256.")
    return document


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


def _relative_output_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError as error:
        raise Layer1CalibrationScriptError(f"Calibration artifact is outside repository root: {path}") from error


def _select_samples(config: ExperimentConfig, sample_key: str | None):
    samples = ordered_calibration_samples(config)
    if sample_key is None:
        return samples
    selected = tuple(sample for sample in samples if sample.sample_key == sample_key)
    if len(selected) != 1:
        raise Layer1CalibrationScriptError(
            "--sample-key must name exactly one immutable calibration-manifest sample."
        )
    return selected


def _run_one_sample(
    *,
    config: ExperimentConfig,
    preregistration: Mapping[str, Any],
    sample: Any,
    head_seed: int,
    dtype: str,
    output_root: Path,
    bf16_gate_profile_amendment: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], Path]:
    """Capture one immutable update and stop after the actual Layer-1 distance scan."""
    torch.manual_seed(head_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(head_seed)
    bundle = load_local_qwen3_sequence_classifier(
        config.model_path,
        head_seed=head_seed,
        device="cuda",
        dtype=dtype,
    )
    try:
        if getattr(bundle.tokenizer, "eos_token_id", None) != sample.eos_token_id:
            raise Layer1CalibrationScriptError(
                f"Manifest EOS id {sample.eos_token_id} differs from the loaded tokenizer EOS "
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
        try:
            diagnostic, diagnostic_controls = diagnose_captured_q_projections(
                captured=captured,
                tokenizer=bundle.tokenizer,
                token_ids=sample.input_ids,
                eos_token_id=sample.eos_token_id,
                dtype=dtype,
            )
        except Exception as error:
            raise Layer1CalibrationScriptError(f"Qwen3 gradient diagnostic could not run: {error}") from error
        if diagnostic.get("passed") is not True:
            raise Layer1CalibrationScriptError(
                "Qwen3 gradient diagnostic failed; refusing to construct a Layer-1 calibration span."
            )

        controls = load_none_attack_controls(config)
        adapter = Qwen3RoPEDagerAdapter(bundle.model, bundle.tokenizer)
        shared_rank = shared_dager_rank_for_qwen3_qproj_gradients(
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
            shared_truncated_rank=shared_rank.applied_shared_rank,
        )
        scan_started = perf_counter()
        scan = scan_qwen3_vocab_layer1_distances(
            adapter=adapter,
            span=q0_span,
            vocab_chunk_size=controls.vocab_chunk_size,
            distance_norm="l2",
        )
        scan_seconds = perf_counter() - scan_started

        sample_directory = output_root / "samples"
        distance_path = sample_directory / f"{sample.sample_key}.distances.npy"
        distance_sidecar = write_or_verify_distance_sidecar(
            distance_path,
            scan.distances,
        )
        distance_sidecar["path"] = _relative_output_path(distance_path)
        record = build_sample_record(
            config=config,
            preregistration=preregistration,
            sample=sample,
            head_seed=head_seed,
            dtype=dtype,
            original_l1_span_threshold=controls.l1_span_threshold,
            shared_rank=shared_rank,
            q0_span=q0_span,
            scan=scan,
            tokenizer=bundle.tokenizer,
            gradient_diagnostic=diagnostic,
            gradient_diagnostic_controls=diagnostic_controls,
            bf16_gate_profile_amendment=bf16_gate_profile_amendment,
            gradient_capture_seconds=capture_seconds,
            scan_seconds=scan_seconds,
            loss=captured.loss,
            gpu_peak_memory_bytes=captured.gpu_peak_memory_bytes,
            distance_sidecar=distance_sidecar,
        )
        record["git_commit"] = _git_commit()
        record_path = sample_directory / f"{sample.sample_key}.json"
        write_or_verify_sample_record(record_path, record)
        return record, record_path
    finally:
        del bundle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    """Run one or all immutable calibration samples without importing Layer-2."""
    if args.stage != "calibration":
        raise Layer1CalibrationScriptError("Layer-1 tau1 calibration is restricted to stage=calibration.")
    config_path = _resolve_repository_path(args.config, description="config path")
    config = load_experiment_config(config_path)
    registered_head_seed(config, stage="calibration", requested_seed=args.head_seed)
    verify_amendment(project_root=QWEN_ROOT)
    bf16_gate_profile_amendment: dict[str, Any] | None = None
    if args.dtype == "bfloat16":
        amendment = verify_bf16_gate_profile_amendment(project_root=QWEN_ROOT)
        amendment_path = QWEN_ROOT / BF16_GATE_PROFILE_AMENDMENT_RELATIVE_PATH
        bf16_gate_profile_amendment = {
            "path": _relative_output_path(amendment_path),
            "sha256": sha256_file(amendment_path),
            "amendment_identity_sha256": amendment["amendment_identity_sha256"],
            "selected_bfloat16_gate": amendment["selected_bfloat16_gate"],
            "fixed_candidate_grid": amendment["fixed_candidate_grid"],
        }
    preregistration = _load_preregistration(config)
    output_root = _resolve_repository_path(args.output_root, description="output root")
    expected_outputs_root = (QWEN_ROOT / "outputs").resolve()
    try:
        output_root.relative_to(expected_outputs_root)
    except ValueError as error:
        raise Layer1CalibrationScriptError(
            f"Calibration output root must remain under {expected_outputs_root}, got {output_root}."
        ) from error
    samples = _select_samples(config, args.sample_key)
    records_and_paths = [
        _run_one_sample(
            config=config,
            preregistration=preregistration,
            sample=sample,
            head_seed=args.head_seed,
            dtype=args.dtype,
            output_root=output_root,
            bf16_gate_profile_amendment=bf16_gate_profile_amendment,
        )
        for sample in samples
    ]
    records = [record for record, _path in records_and_paths]
    result: dict[str, Any] = {
        "status": "ok",
        "stage": "calibration",
        "layer2_invoked": False,
        "sample_count": len(records),
        "sample_keys": [record["sample_key"] for record in records],
        "sample_record_paths": [_relative_output_path(path) for _record, path in records_and_paths],
    }
    if args.sample_key is not None:
        return result

    sample_output_files = [
        {
            "sample_key": str(record["sample_key"]),
            "path": _relative_output_path(path),
            "sha256": sha256_file(path),
        }
        for record, path in records_and_paths
    ]
    aggregation = aggregate_calibration_records(
        records,
        sample_output_files=sample_output_files,
        bf16_gate_profile_amendment=bf16_gate_profile_amendment,
    )
    aggregation["calibration_aggregation_identity_sha256"] = calibration_aggregation_identity(
        preregistration_sha256=str(preregistration["preregistration_sha256"]),
        config_sha256=config.config_sha256,
        dtype=args.dtype,
        head_seed=args.head_seed,
        sample_output_files=sample_output_files,
        bf16_gate_profile_amendment=bf16_gate_profile_amendment,
    )
    aggregation["git_commit"] = _git_commit()
    aggregation_path = output_root / "aggregation.json"
    write_or_verify_aggregation(aggregation_path, aggregation)
    result["aggregation_path"] = _relative_output_path(aggregation_path)
    result["aggregation_status"] = aggregation["status"]
    result["selected_tau1"] = aggregation["selected_tau1"]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed Qwen3 Layer-1 tau1 calibration scan. "
            "It does not load a Layer-2 decoder, ROUGE, or an LRB observation."
        )
    )
    parser.add_argument(
        "--config",
        default="Projection_lrb_qwen3/configs/experiment.json",
        help="Repository-relative immutable Qwen3 experiment config.",
    )
    parser.add_argument("--stage", default="calibration", help="Must be exactly calibration.")
    parser.add_argument("--head-seed", required=True, type=int, help="Must equal the registered calibration head seed.")
    parser.add_argument("--sample-key", help="Optional one immutable calibration-manifest sample key.")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--output-root",
        default="Projection_lrb_qwen3/outputs/calibration/layer1_tau1_calibration",
        help="Repository-relative root for Layer-1-only calibration artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_calibration(args)
    except Exception as error:
        print(
            json.dumps(
                {
                    "record_type": "qwen3_layer1_tau1_calibration_error",
                    "status": "error",
                    "layer2_invoked": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "ok" else 3


if __name__ == "__main__":
    raise SystemExit(main())
