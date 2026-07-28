"""Deterministic selection and freezing for Qwen3 DAGER calibration.

This module is deliberately model-free.  It turns already-recorded calibration
rows into one deterministic choice without reading any smoke or final artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .hashing import canonical_json_bytes, sha256_json
from .result_schema import ResultSchemaError, write_or_verify_json


class ConfigurationSelectionError(RuntimeError):
    """Raised when calibration rows cannot be selected without ambiguity."""


SELECTION_RULE: dict[str, Any] = {
    "priority_order": [
        "maximize_mean_token_recovery",
        "maximize_exact_recovery_rate",
        "minimize_empty_reconstruction_rate",
        "minimize_mean_attack_time_seconds",
    ],
    "tie_breaker": "lexicographically_smallest_canonical_parameter_dictionary",
    "failed_row_scoring": {
        "token_recovery": 0.0,
        "exact_recovery": False,
        "empty_reconstruction": True,
        "attack_time_seconds": "recorded_elapsed_seconds",
    },
}


@dataclass(frozen=True)
class CandidateSummary:
    """Selection statistics for one complete candidate/sample rectangle."""

    candidate_id: str
    parameters: Mapping[str, Any]
    parameter_serialization: str
    sample_count: int
    successful_row_count: int
    failed_row_count: int
    mean_token_recovery: float
    exact_recovery_rate: float
    empty_reconstruction_rate: float
    mean_attack_time_seconds: float

    def as_json(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "parameters": dict(self.parameters),
            "parameter_serialization": self.parameter_serialization,
            "sample_count": self.sample_count,
            "successful_row_count": self.successful_row_count,
            "failed_row_count": self.failed_row_count,
            "mean_token_recovery": self.mean_token_recovery,
            "exact_recovery_rate": self.exact_recovery_rate,
            "empty_reconstruction_rate": self.empty_reconstruction_rate,
            "mean_attack_time_seconds": self.mean_attack_time_seconds,
        }


@dataclass(frozen=True)
class SelectionResult:
    """The selected candidate plus every candidate's auditable summary."""

    selected: CandidateSummary
    candidates: tuple[CandidateSummary, ...]


def canonical_parameter_serialization(parameters: Mapping[str, Any]) -> str:
    """Return the exact serialization used for the specified final tie-break."""
    if not isinstance(parameters, Mapping):
        raise ConfigurationSelectionError("Candidate parameters must be one mapping.")
    try:
        return canonical_json_bytes(dict(parameters)).decode("utf-8")
    except Exception as error:
        raise ConfigurationSelectionError(
            f"Candidate parameters are not canonical-JSON serializable: {error}"
        ) from error


def _finite_probability(row: Mapping[str, Any], key: str, *, fallback: float) -> float:
    value = row.get(key, fallback)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationSelectionError(f"Calibration row {key} must be numeric when present.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise ConfigurationSelectionError(f"Calibration row {key} must be finite in [0, 1].")
    return result


def _attack_time(row: Mapping[str, Any]) -> float:
    value = row.get("attack_time_seconds")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationSelectionError("Every calibration row must record attack_time_seconds.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ConfigurationSelectionError("attack_time_seconds must be finite and non-negative.")
    return result


def _row_is_completed(row: Mapping[str, Any]) -> bool:
    return row.get("result_status") in {"ok", "search_budget_exhausted"}


def _summarize_candidate(
    *,
    candidate_id: str,
    rows: Sequence[Mapping[str, Any]],
    expected_sample_keys: Sequence[str],
) -> CandidateSummary:
    if not rows:
        raise ConfigurationSelectionError(f"Candidate {candidate_id} has no calibration rows.")
    parameters = rows[0].get("parameters")
    if not isinstance(parameters, Mapping):
        raise ConfigurationSelectionError(f"Candidate {candidate_id} lacks parameters.")
    parameter_serialization = canonical_parameter_serialization(parameters)
    seen_keys: set[str] = set()
    token_recoveries: list[float] = []
    exact: list[float] = []
    empty: list[float] = []
    attack_times: list[float] = []
    successful = 0
    for row in rows:
        if row.get("candidate_id") != candidate_id:
            raise ConfigurationSelectionError("Candidate grouping contains a mismatched candidate_id.")
        row_parameters = row.get("parameters")
        if not isinstance(row_parameters, Mapping) or canonical_parameter_serialization(row_parameters) != parameter_serialization:
            raise ConfigurationSelectionError(
                f"Candidate {candidate_id} has inconsistent parameter dictionaries."
            )
        sample_key = row.get("sample_key")
        if not isinstance(sample_key, str) or sample_key in seen_keys:
            raise ConfigurationSelectionError(
                f"Candidate {candidate_id} must contain each calibration sample exactly once."
            )
        seen_keys.add(sample_key)
        completed = _row_is_completed(row)
        if completed:
            successful += 1
            token_recoveries.append(_finite_probability(row, "token_recovery", fallback=0.0))
            exact_value = row.get("exact_recovery")
            empty_value = row.get("empty_reconstruction")
            if not isinstance(exact_value, bool) or not isinstance(empty_value, bool):
                raise ConfigurationSelectionError(
                    f"Completed candidate {candidate_id} row must record boolean exact/empty metrics."
                )
            exact.append(float(exact_value))
            empty.append(float(empty_value))
        else:
            # Failed rows are retained and receive the declared worst recovery
            # score.  They are never filtered out before selection.
            token_recoveries.append(0.0)
            exact.append(0.0)
            empty.append(1.0)
        attack_times.append(_attack_time(row))
    if seen_keys != set(expected_sample_keys):
        raise ConfigurationSelectionError(
            f"Candidate {candidate_id} was not run on exactly the calibration manifest sample set."
        )
    if len(seen_keys) != len(expected_sample_keys):
        raise ConfigurationSelectionError(
            f"Candidate {candidate_id} has {len(seen_keys)} rows, expected {len(expected_sample_keys)}."
        )
    count = len(rows)
    return CandidateSummary(
        candidate_id=candidate_id,
        parameters=dict(parameters),
        parameter_serialization=parameter_serialization,
        sample_count=count,
        successful_row_count=successful,
        failed_row_count=count - successful,
        mean_token_recovery=sum(token_recoveries) / count,
        exact_recovery_rate=sum(exact) / count,
        empty_reconstruction_rate=sum(empty) / count,
        mean_attack_time_seconds=sum(attack_times) / count,
    )


def select_calibration_configuration(
    rows: Iterable[Mapping[str, Any]], *, expected_sample_keys: Sequence[str]
) -> SelectionResult:
    """Apply the user-specified four-level ranking without hidden filters."""
    keys = list(expected_sample_keys)
    if not keys or any(not isinstance(key, str) for key in keys) or len(set(keys)) != len(keys):
        raise ConfigurationSelectionError("The calibration manifest must provide unique non-empty sample keys.")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ConfigurationSelectionError("Calibration results must be JSON objects.")
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or len(candidate_id) != 64:
            raise ConfigurationSelectionError("Every calibration row needs one SHA256 candidate_id.")
        grouped.setdefault(candidate_id, []).append(row)
    if not grouped:
        raise ConfigurationSelectionError("Cannot select from zero calibration candidates.")
    summaries = tuple(
        _summarize_candidate(
            candidate_id=candidate_id,
            rows=candidate_rows,
            expected_sample_keys=keys,
        )
        for candidate_id, candidate_rows in grouped.items()
    )
    selected = min(
        summaries,
        key=lambda item: (
            -item.mean_token_recovery,
            -item.exact_recovery_rate,
            item.empty_reconstruction_rate,
            item.mean_attack_time_seconds,
            item.parameter_serialization,
        ),
    )
    return SelectionResult(
        selected=selected,
        candidates=tuple(sorted(summaries, key=lambda item: item.parameter_serialization)),
    )


def frozen_attack_config_document(
    *,
    selection: SelectionResult,
    calibration_manifest_sha256: str,
    model_sha256: str,
    head_seed: int,
    candidate_grid_sha256: str,
    code_commit: str,
    all_results_sha256: str,
) -> dict[str, Any]:
    """Build the immutable output consumed by subsequent smoke/final runners."""
    values = {
        "selected_parameters": dict(selection.selected.parameters),
        "selection_rule": SELECTION_RULE,
        "calibration_manifest_sha256": calibration_manifest_sha256,
        "model_sha256": model_sha256,
        "head_seed": head_seed,
        "candidate_grid_sha256": candidate_grid_sha256,
        "code_commit": code_commit,
        "all_results_sha256": all_results_sha256,
    }
    if isinstance(head_seed, bool) or not isinstance(head_seed, int):
        raise ConfigurationSelectionError("head_seed must be an integer.")
    if any(not isinstance(values[key], str) or len(values[key]) != 64 for key in (
        "calibration_manifest_sha256",
        "model_sha256",
        "candidate_grid_sha256",
        "all_results_sha256",
    )):
        raise ConfigurationSelectionError("Frozen configuration SHA256 fields must be 64-character strings.")
    if not isinstance(code_commit, str) or not code_commit:
        raise ConfigurationSelectionError("Frozen configuration requires a non-empty code commit.")
    identity = sha256_json(values)
    return {
        "schema_version": 1,
        "record_type": "qwen3_frozen_attack_config",
        "frozen_attack_config_identity_sha256": identity,
        **values,
    }


def write_or_verify_frozen_attack_config(path: Path, document: Mapping[str, Any]) -> bool:
    """Atomically create the immutable frozen config, or verify an exact replay."""
    try:
        return write_or_verify_json(
            path,
            document,
            identity_key="frozen_attack_config_identity_sha256",
        )
    except ResultSchemaError as error:
        raise ConfigurationSelectionError(f"Unable to write frozen attack config {path}: {error}") from error
