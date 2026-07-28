"""Layer-1-only Qwen3 DAGER distance calibration utilities.

The functions in this module deliberately stop after the shared Layer-1
vocabulary-distance scan.  They never import a candidate provider, Layer-2
decoder, ROUGE metric, or any recovery scorer.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from src.config import ExperimentConfig
from src.hashing import sha256_file, sha256_json
from src.result_schema import ResultSchemaError, write_or_verify_json

from .diagnostics import AttackProtocolError, RegisteredAttackSample, load_registered_sample
from .gradient_decomposition import GradientSpan, SharedDagerRank
from .layer1_filter import Layer1DistanceScanResult


TAU1_CALIBRATION_GRID: tuple[float, ...] = (
    1e-5,
    3e-5,
    1e-4,
    2e-4,
    3e-4,
    5e-4,
    7.5e-4,
    1e-3,
    1.5e-3,
    2e-3,
    3e-3,
    5e-3,
    1e-2,
)
EXPECTED_CALIBRATION_SAMPLE_COUNT = 20
DECOMPOSITION_BACKEND = "utils.functional.get_layer_decomp(torch.svd_lowrank,q=shared_rank,niter=10)"


class Layer1CalibrationError(RuntimeError):
    """Raised when the Layer-1-only calibration protocol is violated."""


@dataclass(frozen=True)
class TauSampleMetrics:
    """Thresholded Layer-1 observations for one immutable calibration sample."""

    tau: float
    candidate_count: int
    active_position_hits: int
    active_position_total: int
    active_unique_token_hits: int
    active_unique_token_total: int

    @property
    def active_position_recall(self) -> float:
        return self.active_position_hits / self.active_position_total

    @property
    def active_unique_token_recall(self) -> float:
        return self.active_unique_token_hits / self.active_unique_token_total


def _require_tau_grid(tau_grid: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in tau_grid)
    if values != TAU1_CALIBRATION_GRID:
        raise Layer1CalibrationError(
            "Layer-1 calibration must use the fixed preregistered tau1 grid exactly."
        )
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise Layer1CalibrationError("Layer-1 calibration thresholds must be finite and positive.")
    if tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise Layer1CalibrationError("Layer-1 calibration thresholds must be strictly increasing and unique.")
    return values


def _quantile(values: torch.Tensor, q: float) -> float:
    if values.ndim != 1 or values.numel() == 0:
        raise Layer1CalibrationError("Distance quantiles require one non-empty vector.")
    return float(torch.quantile(values, q).item())


def _distribution_summary(values: Sequence[int | float]) -> dict[str, float | int]:
    if not values:
        raise Layer1CalibrationError("Cannot summarize an empty calibration distribution.")
    tensor = torch.tensor(values, dtype=torch.float64)
    ordered = torch.sort(tensor).values
    p90_index = max(0, math.ceil(0.90 * len(values)) - 1)
    return {
        "min": int(ordered[0].item()) if all(float(value).is_integer() for value in values) else float(ordered[0].item()),
        "median": float(torch.median(ordered).item()),
        "mean": float(torch.mean(ordered).item()),
        "p90": int(ordered[p90_index].item())
        if all(float(value).is_integer() for value in values)
        else float(ordered[p90_index].item()),
        "max": int(ordered[-1].item()) if all(float(value).is_integer() for value in values) else float(ordered[-1].item()),
        "quantile_method": "nearest_rank",
    }


def _token_strings(tokenizer: Any, token_ids: Sequence[int]) -> list[str]:
    converter = getattr(tokenizer, "convert_ids_to_tokens", None)
    if not callable(converter):
        raise Layer1CalibrationError("Qwen3 tokenizer lacks convert_ids_to_tokens for calibration output.")
    values = converter([int(token_id) for token_id in token_ids])
    if not isinstance(values, list) or len(values) != len(token_ids) or any(
        not isinstance(value, str) for value in values
    ):
        raise Layer1CalibrationError("Qwen3 tokenizer returned invalid token strings for calibration output.")
    return values


def _active_ground_truth_positions(
    *,
    sample: RegisteredAttackSample,
    gradient_diagnostic: Mapping[str, Any],
    scan: Layer1DistanceScanResult,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    layers = gradient_diagnostic.get("layers")
    if not isinstance(layers, Mapping):
        raise Layer1CalibrationError("Gradient diagnostic lacks layer records.")
    q0 = layers.get("q0")
    if not isinstance(q0, Mapping):
        raise Layer1CalibrationError("Gradient diagnostic lacks q0 record.")
    per_token = q0.get("per_token")
    if not isinstance(per_token, list) or len(per_token) != len(sample.input_ids):
        raise Layer1CalibrationError("q0 diagnostic token records do not match the immutable sample tokenization.")

    active: list[dict[str, Any]] = []
    for item in per_token:
        if not isinstance(item, Mapping):
            raise Layer1CalibrationError("q0 diagnostic contains a non-mapping token record.")
        if item.get("active_by_delta") is not True:
            continue
        position = item.get("position")
        token_id = item.get("token_id")
        if (
            isinstance(position, bool)
            or not isinstance(position, int)
            or isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or position < 0
            or position >= len(sample.input_ids)
            or token_id != sample.input_ids[position]
            or token_id < 0
            or token_id >= scan.scanned_token_count
        ):
            raise Layer1CalibrationError("q0 diagnostic active token identity is incompatible with the immutable scan.")
        active.append(
            {
                "position": position,
                "token_id": token_id,
                "decoded_token": "",  # Filled only after all ids are validated.
                "distance": float(scan.distances[token_id].item()),
            }
        )
    if not active:
        raise Layer1CalibrationError(
            "q0 gradient diagnostic has no active-by-delta ground-truth positions; calibration fails closed."
        )
    decoded = _token_strings(tokenizer, [item["token_id"] for item in active])
    for item, token in zip(active, decoded):
        item["decoded_token"] = token
    return active


def evaluate_tau_grid(
    *,
    scan: Layer1DistanceScanResult,
    active_positions: Sequence[Mapping[str, Any]],
    tau_grid: Sequence[float] = TAU1_CALIBRATION_GRID,
) -> tuple[TauSampleMetrics, ...]:
    """Evaluate fixed thresholds after, never during, distance construction."""
    values = _require_tau_grid(tau_grid)
    active_token_ids = [item.get("token_id") for item in active_positions]
    if not active_token_ids or any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in active_token_ids):
        raise Layer1CalibrationError("Calibration recall requires one or more validated active token ids.")
    active_unique_ids = tuple(sorted(set(int(token_id) for token_id in active_token_ids)))
    active_positions_total = len(active_token_ids)
    active_unique_total = len(active_unique_ids)
    result: list[TauSampleMetrics] = []
    for tau in values:
        passing = scan.distances <= tau
        candidate_count = int(torch.count_nonzero(passing).item())
        active_position_hits = sum(bool(passing[int(token_id)].item()) for token_id in active_token_ids)
        active_unique_hits = sum(bool(passing[token_id].item()) for token_id in active_unique_ids)
        result.append(
            TauSampleMetrics(
                tau=tau,
                candidate_count=candidate_count,
                active_position_hits=active_position_hits,
                active_position_total=active_positions_total,
                active_unique_token_hits=active_unique_hits,
                active_unique_token_total=active_unique_total,
            )
        )
    counts = [item.candidate_count for item in result]
    if counts != sorted(counts):
        raise Layer1CalibrationError("Layer-1 candidate counts must be monotone non-decreasing over tau1.")
    return tuple(result)


def _diagnostic_summary(diagnostic: Mapping[str, Any]) -> dict[str, Any]:
    layers = diagnostic.get("layers")
    if not isinstance(layers, Mapping):
        raise Layer1CalibrationError("Gradient diagnostic lacks layers for calibration output.")
    output: dict[str, Any] = {"passed": bool(diagnostic.get("passed", False)), "layers": {}}
    for layer_name in ("q0", "q1"):
        layer = layers.get(layer_name)
        if not isinstance(layer, Mapping):
            raise Layer1CalibrationError(f"Gradient diagnostic lacks {layer_name} for calibration output.")
        identity = layer.get("identity")
        rank = layer.get("rank")
        residual = layer.get("row_space_residual")
        active_residual = residual.get("active_tokens") if isinstance(residual, Mapping) else None
        output["layers"][layer_name] = {
            "passed": bool(layer.get("passed", False)),
            "checks": layer.get("checks"),
            "identity_relative_error": identity.get("gradient_relative_error") if isinstance(identity, Mapping) else None,
            "relative_rank": rank.get("relative_threshold_rank") if isinstance(rank, Mapping) else None,
            "relative_threshold": rank.get("relative_threshold") if isinstance(rank, Mapping) else None,
            "active_token_count": layer.get("delta_activity", {}).get("active_token_count")
            if isinstance(layer.get("delta_activity"), Mapping)
            else None,
            "max_active_relative_residual": active_residual.get("max") if isinstance(active_residual, Mapping) else None,
        }
    return output


def _distance_summary(scan: Layer1DistanceScanResult) -> dict[str, float | int]:
    distances = scan.distances
    return {
        "min": float(distances.min().item()),
        "max": float(distances.max().item()),
        "mean": float(distances.mean().item()),
        "median": _quantile(distances, 0.50),
        "p90": _quantile(distances, 0.90),
        "p95": _quantile(distances, 0.95),
        "p99": _quantile(distances, 0.99),
        "quantile_interpolation": "torch_default_linear",
    }


def _top_distance_tokens(scan: Layer1DistanceScanResult, *, tokenizer: Any, limit: int = 100) -> list[dict[str, Any]]:
    if limit <= 0:
        raise Layer1CalibrationError("Top-token limit must be positive.")
    ordered = sorted(
        zip(scan.token_ids.tolist(), scan.distances.tolist()),
        key=lambda item: (item[1], item[0]),
    )[: min(limit, scan.scanned_token_count)]
    decoded = _token_strings(tokenizer, [token_id for token_id, _ in ordered])
    return [
        {"token_id": int(token_id), "decoded_token": token, "distance": float(distance)}
        for (token_id, distance), token in zip(ordered, decoded)
    ]


def _serialize_tau_metrics(metrics: Iterable[TauSampleMetrics]) -> list[dict[str, float | int]]:
    return [
        {
            "tau": item.tau,
            "candidate_count": item.candidate_count,
            "active_position_hits": item.active_position_hits,
            "active_position_total": item.active_position_total,
            "active_position_recall": item.active_position_recall,
            "active_unique_token_hits": item.active_unique_token_hits,
            "active_unique_token_total": item.active_unique_token_total,
            "active_unique_token_recall": item.active_unique_token_recall,
        }
        for item in metrics
    ]


def calibration_sample_identity(
    *,
    config_sha256: str,
    preregistration_sha256: str,
    sample_key: str,
    head_seed: int,
    dtype: str,
) -> str:
    return sha256_json(
        {
            "protocol": "qwen3_layer1_tau1_calibration_v1",
            "config_sha256": config_sha256,
            "preregistration_sha256": preregistration_sha256,
            "sample_key": sample_key,
            "head_seed": head_seed,
            "dtype": dtype,
            "tau_grid": list(TAU1_CALIBRATION_GRID),
        }
    )


def build_sample_record(
    *,
    config: ExperimentConfig,
    preregistration: Mapping[str, Any],
    sample: RegisteredAttackSample,
    head_seed: int,
    dtype: str,
    original_l1_span_threshold: float,
    shared_rank: SharedDagerRank,
    q0_span: GradientSpan,
    scan: Layer1DistanceScanResult,
    tokenizer: Any,
    gradient_diagnostic: Mapping[str, Any],
    gradient_diagnostic_controls: Mapping[str, float] | None = None,
    bf16_gate_profile_amendment: Mapping[str, Any] | None = None,
    gradient_capture_seconds: float,
    scan_seconds: float,
    loss: float,
    gpu_peak_memory_bytes: int,
    distance_sidecar: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one complete Layer-1-only output after distance construction."""
    active_positions = _active_ground_truth_positions(
        sample=sample,
        gradient_diagnostic=gradient_diagnostic,
        scan=scan,
        tokenizer=tokenizer,
    )
    tau_metrics = evaluate_tau_grid(scan=scan, active_positions=active_positions)
    model_hashes = preregistration.get("model_key_file_sha256")
    tokenizer_hashes = preregistration.get("tokenizer_key_file_sha256")
    if not isinstance(model_hashes, Mapping) or not isinstance(tokenizer_hashes, Mapping):
        raise Layer1CalibrationError("Preregistration lacks required model/tokenizer identity hashes.")
    record: dict[str, Any] = {
        "schema_version": 1,
        "record_type": "qwen3_layer1_tau1_calibration_sample",
        "status": "ok",
        "calibration_sample_identity_sha256": calibration_sample_identity(
            config_sha256=config.config_sha256,
            preregistration_sha256=sample.preregistration_sha256,
            sample_key=sample.sample_key,
            head_seed=head_seed,
            dtype=dtype,
        ),
        "protocol": "qwen3_layer1_tau1_calibration_v1",
        "layer2_invoked": False,
        "stage": "calibration",
        "sample_key": sample.sample_key,
        "original_index": sample.original_index,
        "sentence_character_length": len(sample.sentence),
        "input_token_count": len(sample.input_ids),
        "head_seed": head_seed,
        "dtype": dtype,
        "identity": {
            "experiment_config_sha256": config.config_sha256,
            "preregistration_sha256": sample.preregistration_sha256,
            "model_key_file_sha256": dict(model_hashes),
            "tokenizer_key_file_sha256": dict(tokenizer_hashes),
            "tokenizer_sha256": preregistration.get("tokenizer_sha256"),
        },
        "raw_relative_ranks": {
            "q0": shared_rank.q0_effective_rank,
            "q1": shared_rank.q1_effective_rank,
            "rank_definition": shared_rank.rank_definition,
            "rank_rtol": shared_rank.rank_rtol,
        },
        "shared_rank": {
            "requested": shared_rank.requested_shared_rank,
            "applied": shared_rank.applied_shared_rank,
            "rank_cap": shared_rank.rank_cap,
            "rank_was_capped": shared_rank.rank_was_capped,
            "cap_reason": shared_rank.cap_reason,
            "rank_cutoff": q0_span.rank_cutoff,
        },
        "q0_basis": {
            "shape": list(q0_span.basis.shape),
            "dtype": str(q0_span.basis.dtype),
            "orientation": q0_span.orientation,
            "decomposition_backend": DECOMPOSITION_BACKEND,
        },
        "vocabulary_scan": {
            "vocab_size": scan.scanned_token_count,
            "scanned_token_count": scan.scanned_token_count,
            "chunk_size": scan.chunk_size,
            "distance_norm": scan.distance_norm,
            "special_token_or_exclusion_rule": "none; every native vocabulary id is scanned",
            "chunks": [
                {
                    "start_token_id": item.start_token_id,
                    "end_token_id_exclusive": item.end_token_id_exclusive,
                    "scanned_token_count": item.scanned_token_count,
                    "elapsed_seconds": item.elapsed_seconds,
                }
                for item in scan.chunk_diagnostics
            ],
        },
        "original_experiment_l1_span_thresh": original_l1_span_threshold,
        "fixed_tau1_grid": list(TAU1_CALIBRATION_GRID),
        "per_tau": _serialize_tau_metrics(tau_metrics),
        "vocab_distance_summary": _distance_summary(scan),
        "top_100_lowest_distance_tokens": _top_distance_tokens(scan, tokenizer=tokenizer, limit=100),
        "active_ground_truth_positions": active_positions,
        "gradient_diagnostic_summary": _diagnostic_summary(gradient_diagnostic),
        "distance_sidecar": dict(distance_sidecar),
        "timing": {
            "gradient_capture_seconds": gradient_capture_seconds,
            "layer1_distance_scan_seconds": scan_seconds,
        },
        "loss": loss,
        "gpu_peak_memory_bytes": gpu_peak_memory_bytes,
    }
    if gradient_diagnostic_controls is not None:
        record["gradient_diagnostic_controls"] = dict(gradient_diagnostic_controls)
    if bf16_gate_profile_amendment is not None:
        record["bf16_gate_profile_amendment"] = dict(bf16_gate_profile_amendment)
    return record


def write_or_verify_distance_sidecar(path: Path, distances: torch.Tensor) -> dict[str, Any]:
    """Persist exact CPU FP32 full-vocabulary distances without silent replacement."""
    if distances.ndim != 1 or distances.dtype != torch.float32 or distances.device.type != "cpu":
        raise Layer1CalibrationError("Distance sidecar requires one CPU FP32 vector.")
    expected = distances.numpy()
    if path.exists():
        try:
            existing = np.load(path, allow_pickle=False)
        except Exception as error:
            raise Layer1CalibrationError(f"Unable to read existing distance sidecar {path}: {error}") from error
        if existing.dtype != np.float32 or existing.shape != expected.shape or not np.array_equal(existing, expected):
            raise Layer1CalibrationError(f"Existing distance sidecar conflicts with this immutable calibration run: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
            ) as handle:
                temporary_path = Path(handle.name)
                np.save(handle, expected, allow_pickle=False)
            temporary_path.replace(path)
        except OSError as error:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise Layer1CalibrationError(f"Unable to write distance sidecar {path}: {error}") from error
    return {
        "path": path.as_posix(),
        "shape": [int(value) for value in expected.shape],
        "dtype": str(expected.dtype),
        "sha256": sha256_file(path),
    }


def ordered_calibration_samples(config: ExperimentConfig) -> tuple[RegisteredAttackSample, ...]:
    """Load all calibration samples in their immutable JSONL order."""
    manifest_path = config.project_root / "manifests" / "calibration.jsonl"
    try:
        rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise Layer1CalibrationError(f"Unable to read calibration manifest {manifest_path}: {error}") from error
    if len(rows) != EXPECTED_CALIBRATION_SAMPLE_COUNT:
        raise Layer1CalibrationError(
            f"Layer-1 calibration requires exactly {EXPECTED_CALIBRATION_SAMPLE_COUNT} preregistered samples, "
            f"found {len(rows)}."
        )
    keys: list[str] = []
    for row in rows:
        sample = row.get("sample") if isinstance(row, Mapping) else None
        key = sample.get("sample_key") if isinstance(sample, Mapping) else None
        if not isinstance(key, str) or len(key) != 64:
            raise Layer1CalibrationError("Calibration manifest contains an invalid sample key.")
        keys.append(key)
    if len(set(keys)) != len(keys):
        raise Layer1CalibrationError("Calibration manifest contains a duplicate sample key.")
    try:
        return tuple(load_registered_sample(config=config, stage="calibration", sample_key=key) for key in keys)
    except AttackProtocolError as error:
        raise Layer1CalibrationError(str(error)) from error


def aggregate_calibration_records(
    records: Sequence[Mapping[str, Any]],
    *,
    sample_output_files: Sequence[Mapping[str, str]],
    bf16_gate_profile_amendment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate all 20 Layer-1-only records and apply the fixed tau rule."""
    if len(records) != EXPECTED_CALIBRATION_SAMPLE_COUNT:
        raise Layer1CalibrationError(
            f"Aggregation requires exactly {EXPECTED_CALIBRATION_SAMPLE_COUNT} successful calibration records."
        )
    keys = [record.get("sample_key") for record in records]
    if len(set(keys)) != len(keys) or any(not isinstance(key, str) for key in keys):
        raise Layer1CalibrationError("Calibration aggregation requires unique sample keys.")
    if any(record.get("status") != "ok" or record.get("layer2_invoked") is not False for record in records):
        raise Layer1CalibrationError("Calibration aggregation refuses records that are not successful Layer-1-only observations.")
    if bf16_gate_profile_amendment is not None and any(
        record.get("bf16_gate_profile_amendment") != dict(bf16_gate_profile_amendment)
        for record in records
    ):
        raise Layer1CalibrationError(
            "Calibration aggregation requires every BF16 sample record to carry the same verified gate amendment."
        )

    per_tau: list[dict[str, Any]] = []
    selected_tau: float | None = None
    for tau_index, tau in enumerate(TAU1_CALIBRATION_GRID):
        sample_metrics: list[Mapping[str, Any]] = []
        for record in records:
            metrics = record.get("per_tau")
            if not isinstance(metrics, list) or len(metrics) != len(TAU1_CALIBRATION_GRID):
                raise Layer1CalibrationError("Calibration record has an incompatible tau metric list.")
            item = metrics[tau_index]
            if not isinstance(item, Mapping) or float(item.get("tau", float("nan"))) != tau:
                raise Layer1CalibrationError("Calibration record tau ordering differs from the fixed grid.")
            sample_metrics.append(item)
        position_hits = sum(int(item["active_position_hits"]) for item in sample_metrics)
        position_total = sum(int(item["active_position_total"]) for item in sample_metrics)
        unique_hits = sum(int(item["active_unique_token_hits"]) for item in sample_metrics)
        unique_total = sum(int(item["active_unique_token_total"]) for item in sample_metrics)
        if position_total <= 0 or unique_total <= 0:
            raise Layer1CalibrationError("Calibration aggregation encountered an empty active-token denominator.")
        counts = [int(item["candidate_count"]) for item in sample_metrics]
        row = {
            "tau": tau,
            "micro_active_position_recall": position_hits / position_total,
            "micro_active_position_hits": position_hits,
            "micro_active_position_total": position_total,
            "micro_active_unique_token_recall": unique_hits / unique_total,
            "micro_active_unique_token_hits": unique_hits,
            "micro_active_unique_token_total": unique_total,
            "nonempty_sample_count": sum(count > 0 for count in counts),
            "candidate_count_distribution": _distribution_summary(counts),
        }
        per_tau.append(row)
        if (
            selected_tau is None
            and row["micro_active_position_recall"] >= 0.95
            and row["nonempty_sample_count"] == EXPECTED_CALIBRATION_SAMPLE_COUNT
        ):
            selected_tau = tau

    selection_rule_passed = selected_tau is not None
    aggregation: dict[str, Any] = {
        "schema_version": 1,
        "record_type": "qwen3_layer1_tau1_calibration_aggregation",
        "protocol": "qwen3_layer1_tau1_calibration_v1",
        "layer2_invoked": False,
        "calibration_sample_count": len(records),
        "fixed_tau1_grid": list(TAU1_CALIBRATION_GRID),
        "per_tau": per_tau,
        "selected_tau1": selected_tau,
        "selection_rule_passed": selection_rule_passed,
        "selection_rule": {
            "minimum_micro_active_position_recall": 0.95,
            "required_nonempty_sample_count": EXPECTED_CALIBRATION_SAMPLE_COUNT,
            "selection": "smallest_tau_satisfying_both_conditions",
        },
        "status": "ok" if selection_rule_passed else "failed",
        "failure_reason": None
        if selection_rule_passed
        else "no_fixed_tau1_grid_value_satisfies_micro_active_position_recall>=0.95_and_all_samples_nonempty",
        "sample_output_files": [dict(item) for item in sample_output_files],
    }
    if bf16_gate_profile_amendment is not None:
        aggregation["bf16_gate_profile_amendment"] = dict(bf16_gate_profile_amendment)
    return aggregation


def write_or_verify_sample_record(path: Path, record: Mapping[str, Any]) -> bool:
    """Write one immutable sample observation keyed by its calibration identity."""
    try:
        return write_or_verify_json(
            path,
            record,
            identity_key="calibration_sample_identity_sha256",
        )
    except ResultSchemaError as error:
        raise Layer1CalibrationError(str(error)) from error


def calibration_aggregation_identity(
    *,
    preregistration_sha256: str,
    config_sha256: str,
    dtype: str,
    head_seed: int,
    sample_output_files: Sequence[Mapping[str, str]],
    bf16_gate_profile_amendment: Mapping[str, Any] | None = None,
) -> str:
    return sha256_json(
        {
            "protocol": "qwen3_layer1_tau1_calibration_v1",
            "preregistration_sha256": preregistration_sha256,
            "config_sha256": config_sha256,
            "dtype": dtype,
            "head_seed": head_seed,
            "tau_grid": list(TAU1_CALIBRATION_GRID),
            "sample_output_files": [dict(item) for item in sample_output_files],
            "bf16_gate_profile_amendment": (
                None if bf16_gate_profile_amendment is None else dict(bf16_gate_profile_amendment)
            ),
        }
    )


def write_or_verify_aggregation(path: Path, record: Mapping[str, Any]) -> bool:
    """Write one immutable aggregate keyed by its complete sample-output list."""
    try:
        return write_or_verify_json(
            path,
            record,
            identity_key="calibration_aggregation_identity_sha256",
        )
    except ResultSchemaError as error:
        raise Layer1CalibrationError(str(error)) from error
