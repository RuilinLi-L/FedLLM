"""Deterministic eligible-only selection for Stage-5 none-only calibration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .hashing import sha256_json
from .result_schema import ResultSchemaError, write_or_verify_json


class ConfigurationSelectionError(RuntimeError):
    """Raised when Stage-5 calibration rows are inconsistent."""


SELECTION_RULE: dict[str, Any] = {
    "eligible_candidates": "failed_sample_count == 0",
    "priority_order": [
        "maximize_mean_token_recovery",
        "maximize_exact_recovery_rate",
        "maximize_mean_rouge_1_plus_rouge_2",
        "minimize_mean_evaluated_prefix_cost",
    ],
}
TIE_BREAK_RULE = "canonical_candidate_id_lexicographic_ascending"


@dataclass(frozen=True)
class CandidateSummary:
    candidate_id: str
    parameters: Mapping[str, Any]
    sample_count: int
    failed_sample_count: int
    mean_token_recovery: float | None
    exact_recovery_rate: float | None
    mean_rouge_1_plus_rouge_2: float | None
    mean_evaluated_prefix_cost: float | None

    @property
    def eligible(self) -> bool:
        return self.failed_sample_count == 0

    def as_json(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "parameters": dict(self.parameters),
            "sample_count": self.sample_count,
            "failed_sample_count": self.failed_sample_count,
            "eligible": self.eligible,
            "mean_token_recovery": self.mean_token_recovery,
            "exact_recovery_rate": self.exact_recovery_rate,
            "mean_rouge_1_plus_rouge_2": self.mean_rouge_1_plus_rouge_2,
            "mean_evaluated_prefix_cost": self.mean_evaluated_prefix_cost,
        }


@dataclass(frozen=True)
class SelectionResult:
    selected: CandidateSummary | None
    candidates: tuple[CandidateSummary, ...]


def _finite(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ConfigurationSelectionError(f"Successful calibration row has invalid {key}.")
    return float(value)


def _summary(rows: Sequence[Mapping[str, Any]], expected_sample_keys: Sequence[str]) -> CandidateSummary:
    candidate_id = rows[0].get("candidate_id")
    parameters = rows[0].get("parameters")
    if not isinstance(candidate_id, str) or len(candidate_id) != 64 or not isinstance(parameters, Mapping):
        raise ConfigurationSelectionError("Calibration row lacks canonical candidate identity/parameters.")
    seen: set[str] = set()
    successful: list[Mapping[str, Any]] = []
    for row in rows:
        if row.get("candidate_id") != candidate_id or row.get("parameters") != parameters:
            raise ConfigurationSelectionError("Candidate rows have inconsistent identifiers or parameters.")
        sample_key = row.get("sample_key")
        if not isinstance(sample_key, str) or sample_key in seen:
            raise ConfigurationSelectionError("Candidate rows must contain each calibration sample exactly once.")
        seen.add(sample_key)
        if row.get("result_status") in {"ok", "search_budget_exhausted"}:
            successful.append(row)
    if seen != set(expected_sample_keys) or len(rows) != len(expected_sample_keys):
        raise ConfigurationSelectionError("Every candidate must cover exactly the calibration manifest sample set.")
    failed = len(rows) - len(successful)
    if failed:
        return CandidateSummary(candidate_id, dict(parameters), len(rows), failed, None, None, None, None)
    token = [_finite(row, "token_recovery") for row in successful]
    exact_values: list[float] = []
    for row in successful:
        exact = row.get("exact_recovery")
        if not isinstance(exact, bool):
            raise ConfigurationSelectionError("Successful calibration row has non-boolean exact_recovery.")
        exact_values.append(float(exact))
    rouge = [_finite(row, "rouge_1") + _finite(row, "rouge_2") for row in successful]
    prefix = [_finite(row, "evaluated_prefix_cost") for row in successful]
    return CandidateSummary(
        candidate_id, dict(parameters), len(rows), 0,
        sum(token) / len(token), sum(exact_values) / len(exact_values),
        sum(rouge) / len(rouge), sum(prefix) / len(prefix),
    )


def select_calibration_configuration(
    rows: Iterable[Mapping[str, Any]], *, expected_sample_keys: Sequence[str]
) -> SelectionResult:
    """Select only fully successful candidates under the fixed five-level order."""
    if not expected_sample_keys or len(set(expected_sample_keys)) != len(expected_sample_keys):
        raise ConfigurationSelectionError("Expected calibration samples must be non-empty and unique.")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ConfigurationSelectionError("Calibration results must be JSON objects.")
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str):
            raise ConfigurationSelectionError("Calibration result lacks candidate_id.")
        grouped.setdefault(candidate_id, []).append(row)
    if not grouped:
        raise ConfigurationSelectionError("Cannot select from zero candidates.")
    candidates = tuple(sorted((_summary(group, expected_sample_keys) for group in grouped.values()), key=lambda item: item.candidate_id))
    eligible = [item for item in candidates if item.eligible]
    selected = None if not eligible else min(
        eligible,
        key=lambda item: (
            -float(item.mean_token_recovery),
            -float(item.exact_recovery_rate),
            -float(item.mean_rouge_1_plus_rouge_2),
            float(item.mean_evaluated_prefix_cost),
            item.candidate_id,
        ),
    )
    return SelectionResult(selected=selected, candidates=candidates)


def frozen_attack_config_document(
    *,
    selected: CandidateSummary,
    tau1_control_identity_sha256: str,
    calibration_grid_control_identity_sha256: str,
    selected_result_files: Sequence[Mapping[str, str]],
    all_results_sha256: str,
    preregistration_sha256: str,
    calibration_sample_list_sha256: str,
    model_tokenizer_identity: Mapping[str, Any],
    head_seed: int,
    git_commit: str,
) -> dict[str, Any]:
    if not selected.eligible:
        raise ConfigurationSelectionError("Cannot freeze a failed calibration candidate.")
    if len(selected_result_files) != 20:
        raise ConfigurationSelectionError("Frozen Stage-5 config requires exactly 20 selected result files.")
    values: dict[str, Any] = {
        "selected_candidate": selected.as_json(),
        "fixed_tau1_control_identity_sha256": tau1_control_identity_sha256,
        "calibration_grid_control_identity_sha256": calibration_grid_control_identity_sha256,
        "selection_rule": SELECTION_RULE,
        "tie_break_rule": TIE_BREAK_RULE,
        "selected_result_files": [dict(item) for item in selected_result_files],
        "all_results_sha256": all_results_sha256,
        "preregistration_sha256": preregistration_sha256,
        "calibration_sample_list_sha256": calibration_sample_list_sha256,
        "model_tokenizer_identity": dict(model_tokenizer_identity),
        "head_seed": head_seed,
        "git_commit": git_commit,
    }
    document = {"schema_version": 1, "record_type": "qwen3_frozen_attack_config", **values}
    document["frozen_attack_config_identity_sha256"] = sha256_json(document)
    return document


def write_or_verify_frozen_attack_config(path: Path, document: Mapping[str, Any]) -> bool:
    try:
        return write_or_verify_json(path, document, identity_key="frozen_attack_config_identity_sha256")
    except ResultSchemaError as error:
        raise ConfigurationSelectionError(f"Unable to write frozen attack config {path}: {error}") from error
