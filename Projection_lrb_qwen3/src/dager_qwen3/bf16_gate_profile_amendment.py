"""Immutable BF16 structural-gate profile for Qwen3 pre-attack calibration.

This module only reads the 20 output documents produced by
``check_qwen3_gradient.py``.  It neither imports nor invokes the Layer-1
vocabulary scanner, Layer-2 decoder, ROUGE, reconstruction metrics, or LRB.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.hashing import sha256_file, sha256_json
from src.result_schema import ResultSchemaError, write_or_verify_json


AMENDMENT_RELATIVE_PATH = Path(
    "prereg_amendments/bf16_gradient_gate_pre_attack_075/amendment.json"
)
DEFAULT_DIAGNOSTIC_ROOT_RELATIVE_PATH = Path("outputs/calibration/bf16_gate_profile")
EXPECTED_CALIBRATION_SAMPLE_COUNT = 20
OLD_BFLOAT16_GATE = 3e-3
FIXED_BFLOAT16_GATE_GRID: tuple[float, ...] = (3e-3, 5e-3, 7.5e-3, 1e-2)
SELECTED_BFLOAT16_GATE = 7.5e-3
REQUIRED_NON_RESIDUAL_CHECKS: tuple[str, ...] = (
    "gradient_identity",
    "relative_rank_within_theoretical_cap",
    "active_tokens_present",
    "all_numeric_values_finite",
    "orientation_is_fixed_gradient",
    "gradient_t_negative_control_worse",
)


class Bfloat16GateProfileAmendmentError(RuntimeError):
    """Raised when the fixed BF16 structural profile is absent or inconsistent."""


def _relative_to_project(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as error:
        raise Bfloat16GateProfileAmendmentError(
            f"Required artifact must remain under {project_root}: {path}"
        ) from error


def _load_json(path: Path, *, description: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Bfloat16GateProfileAmendmentError(
            f"Unable to read {description} {path}: {error}"
        ) from error
    if not isinstance(value, Mapping):
        raise Bfloat16GateProfileAmendmentError(f"{description} must contain one JSON object: {path}")
    return value


def _calibration_sample_keys(project_root: Path) -> tuple[str, ...]:
    manifest_path = project_root / "manifests" / "calibration.jsonl"
    try:
        rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise Bfloat16GateProfileAmendmentError(
            f"Unable to read calibration manifest {manifest_path}: {error}"
        ) from error
    if len(rows) != EXPECTED_CALIBRATION_SAMPLE_COUNT:
        raise Bfloat16GateProfileAmendmentError(
            f"Expected exactly {EXPECTED_CALIBRATION_SAMPLE_COUNT} calibration samples, found {len(rows)}."
        )
    keys: list[str] = []
    for ordinal, row in enumerate(rows):
        sample = row.get("sample") if isinstance(row, Mapping) else None
        key = sample.get("sample_key") if isinstance(sample, Mapping) else None
        if not isinstance(key, str) or len(key) != 64:
            raise Bfloat16GateProfileAmendmentError(
                f"Calibration manifest row {ordinal} lacks a valid sample_key."
            )
        keys.append(key)
    if len(set(keys)) != len(keys):
        raise Bfloat16GateProfileAmendmentError("Calibration manifest sample keys must be unique.")
    return tuple(keys)


def _calibration_head_seed(project_root: Path) -> int:
    config = _load_json(project_root / "configs" / "experiment.json", description="experiment config")
    value = config.get("calibration_head_seed")
    if isinstance(value, bool) or not isinstance(value, int):
        raise Bfloat16GateProfileAmendmentError(
            "experiment.json must contain an explicit integer calibration_head_seed."
        )
    return value


def _finite_float(value: Any, *, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Bfloat16GateProfileAmendmentError(f"{description} must be a finite numeric value.")
    result = float(value)
    if not math.isfinite(result):
        raise Bfloat16GateProfileAmendmentError(f"{description} must be finite.")
    return result


def _layer_profile(document: Mapping[str, Any], *, layer_name: str, path: Path) -> dict[str, Any]:
    span = document.get("span_diagnostics")
    layers = span.get("layers") if isinstance(span, Mapping) else None
    layer = layers.get(layer_name) if isinstance(layers, Mapping) else None
    if not isinstance(layer, Mapping):
        raise Bfloat16GateProfileAmendmentError(
            f"Diagnostic {path} lacks span_diagnostics.layers.{layer_name}."
        )
    checks = layer.get("checks")
    active_residual_check = checks.get("active_token_residual") if isinstance(checks, Mapping) else None
    if active_residual_check is not True and active_residual_check is not False:
        raise Bfloat16GateProfileAmendmentError(
            f"Diagnostic {path} lacks a boolean {layer_name} active-token residual check."
        )
    observed_non_residual = {str(check_name) for check_name in checks if check_name != "active_token_residual"}
    required_or_observed = observed_non_residual | set(REQUIRED_NON_RESIDUAL_CHECKS)
    non_residual_failed_checks = sorted(
        check_name for check_name in required_or_observed if checks.get(check_name) is not True
    )
    residual_container = layer.get("row_space_residual")
    active_container = (
        residual_container.get("active_tokens") if isinstance(residual_container, Mapping) else None
    )
    residual = active_container.get("max") if isinstance(active_container, Mapping) else None
    return {
        "max_active_relative_residual": _finite_float(
            residual,
            description=f"{path} {layer_name} active residual",
        ),
        "active_token_residual_check": active_residual_check,
        "non_residual_failed_checks": non_residual_failed_checks,
        "identity_check_passed": checks.get("gradient_identity") is True,
        "relative_rank_check_passed": checks.get("relative_rank_within_theoretical_cap") is True,
        "transpose_negative_control_passed": checks.get("gradient_t_negative_control_worse") is True,
    }


def _select_smallest_covering_gate(maximum_residual: float) -> float:
    for gate in FIXED_BFLOAT16_GATE_GRID:
        if maximum_residual <= gate:
            return gate
    raise Bfloat16GateProfileAmendmentError(
        "No fixed BF16 candidate gate covers the observed maximum active-token residual; "
        "the grid must not be expanded automatically."
    )


def _profile_document(
    *,
    project_root: Path,
    diagnostic_root: Path,
    sample_keys: Sequence[str],
    calibration_head_seed: int,
) -> tuple[list[dict[str, Any]], float, int]:
    records: list[dict[str, Any]] = []
    samples_exceeding_old_gate = 0
    maximum_residual = -math.inf
    for ordinal, sample_key in enumerate(sample_keys):
        path = diagnostic_root / f"{sample_key}.json"
        if not path.is_file():
            raise Bfloat16GateProfileAmendmentError(
                f"Missing BF16 diagnostic for immutable calibration sample {sample_key}: {path}"
            )
        document = _load_json(path, description="BF16 gradient diagnostic")
        if document.get("record_type") != "qwen3_single_sample_gradient_diagnostic":
            raise Bfloat16GateProfileAmendmentError(
                f"Diagnostic {path} is not a qwen3_single_sample_gradient_diagnostic."
            )
        if document.get("compute_dtype") != "torch.bfloat16":
            raise Bfloat16GateProfileAmendmentError(f"Diagnostic {path} is not BF16.")
        if document.get("head_seed") != calibration_head_seed:
            raise Bfloat16GateProfileAmendmentError(
                f"Diagnostic {path} head_seed differs from the registered calibration seed."
            )
        diagnostic_sha256 = document.get("diagnostic_sha256")
        if not isinstance(diagnostic_sha256, str) or len(diagnostic_sha256) != 64:
            raise Bfloat16GateProfileAmendmentError(f"Diagnostic {path} lacks diagnostic_sha256.")
        layers = {
            name: _layer_profile(document, layer_name=name, path=path)
            for name in ("q0", "q1")
        }
        record_maximum = max(
            layers["q0"]["max_active_relative_residual"],
            layers["q1"]["max_active_relative_residual"],
        )
        maximum_residual = max(maximum_residual, record_maximum)
        samples_exceeding_old_gate += int(record_maximum > OLD_BFLOAT16_GATE)
        records.append(
            {
                "ordinal": ordinal,
                "sample_key": sample_key,
                "path": _relative_to_project(project_root, path),
                "sha256": sha256_file(path),
                "diagnostic_sha256": diagnostic_sha256,
                "status": document.get("status"),
                "layers": layers,
            }
        )
    if len(records) != EXPECTED_CALIBRATION_SAMPLE_COUNT:
        raise Bfloat16GateProfileAmendmentError("BF16 profile must contain exactly 20 diagnostics.")
    return records, maximum_residual, samples_exceeding_old_gate


def build_amendment_document(*, project_root: Path, diagnostic_root: Path) -> dict[str, Any]:
    """Build a machine-verifiable amendment from the immutable profile files."""
    _relative_to_project(project_root, diagnostic_root)
    sample_keys = _calibration_sample_keys(project_root)
    head_seed = _calibration_head_seed(project_root)
    records, maximum_residual, exceeding_old_gate = _profile_document(
        project_root=project_root,
        diagnostic_root=diagnostic_root,
        sample_keys=sample_keys,
        calibration_head_seed=head_seed,
    )
    selected_gate = _select_smallest_covering_gate(maximum_residual)
    all_layers = [record["layers"][name] for record in records for name in ("q0", "q1")]
    all_non_residual_checks_pass = all(
        not layer["non_residual_failed_checks"] for layer in all_layers
    )
    if not all_non_residual_checks_pass:
        raise Bfloat16GateProfileAmendmentError(
            "At least one BF16 profile diagnostic failed a non-residual structural check."
        )
    if selected_gate != SELECTED_BFLOAT16_GATE:
        raise Bfloat16GateProfileAmendmentError(
            f"The fixed profile selects {selected_gate:g}, not the required {SELECTED_BFLOAT16_GATE:g}."
        )
    immutable: dict[str, Any] = {
        "schema_version": 1,
        "event": "qwen3_bfloat16_gradient_gate_profile_pre_attack",
        "amendment_type": "pre_attack_numeric_diagnostic_gate_revision",
        "timing": "before_any_successful_dager_reconstruction_result",
        "scope": (
            "Read only 20 single-sample BF16 structural diagnostic JSON files; "
            "do not invoke Layer-1 candidate scanning, Layer-2 decoding, ROUGE, reconstruction, or LRB."
        ),
        "attack_configuration_changed": False,
        "experiment_config_changed": False,
        "preregistration_regenerated": False,
        "sample_lists_changed": False,
        "head_seeds_changed": False,
        "old_bfloat16_gate": OLD_BFLOAT16_GATE,
        "selected_bfloat16_gate": selected_gate,
        "fixed_candidate_grid": list(FIXED_BFLOAT16_GATE_GRID),
        "diagnostic_sample_count": len(records),
        "calibration_head_seed": head_seed,
        "diagnostic_root": _relative_to_project(project_root, diagnostic_root),
        "samples_exceeding_old_gate": exceeding_old_gate,
        "maximum_observed_active_token_residual": maximum_residual,
        "all_non_residual_checks_passed": all_non_residual_checks_pass,
        "all_identity_checks_passed": all(layer["identity_check_passed"] for layer in all_layers),
        "all_relative_rank_checks_passed": all(
            layer["relative_rank_check_passed"] for layer in all_layers
        ),
        "all_transpose_negative_controls_passed": all(
            layer["transpose_negative_control_passed"] for layer in all_layers
        ),
        "layer1_attack_results_observed": False,
        "layer2_attack_results_observed": False,
        "dager_reconstruction_results_observed": False,
        "lrb_used": False,
        "diagnostic_json_files": records,
        "selection_rule": {
            "rule": "smallest_fixed_candidate_grid_value_greater_than_or_equal_to_maximum_observed_active_token_residual",
            "automatic_grid_expansion": False,
            "manual_selection": False,
        },
    }
    immutable["amendment_identity_sha256"] = sha256_json(immutable)
    return immutable


def write_or_verify_amendment(*, project_root: Path, diagnostic_root: Path) -> Path:
    """Write the profile amendment once, or retain only an exact recomputation."""
    document = build_amendment_document(project_root=project_root, diagnostic_root=diagnostic_root)
    path = project_root / AMENDMENT_RELATIVE_PATH
    try:
        write_or_verify_json(path, document, identity_key="amendment_identity_sha256")
    except ResultSchemaError as error:
        raise Bfloat16GateProfileAmendmentError(str(error)) from error
    return path


def verify_amendment(*, project_root: Path) -> Mapping[str, Any]:
    """Recompute and require the 20-sample BF16 profile before Layer-1 scanning."""
    path = project_root / AMENDMENT_RELATIVE_PATH
    document = _load_json(path, description="BF16 gate profile amendment")
    diagnostic_root_value = document.get("diagnostic_root")
    if not isinstance(diagnostic_root_value, str):
        raise Bfloat16GateProfileAmendmentError("BF16 gate amendment lacks diagnostic_root.")
    diagnostic_root = (project_root / diagnostic_root_value).resolve()
    _relative_to_project(project_root, diagnostic_root)
    expected = build_amendment_document(project_root=project_root, diagnostic_root=diagnostic_root)
    if document != expected:
        raise Bfloat16GateProfileAmendmentError(
            "BF16 gate profile amendment does not match a fresh hash-verified recomputation."
        )
    return document
