"""Server-local creation and validation of the Layer-1-only calibration amendment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from src.hashing import sha256_file, sha256_json
from src.result_schema import ResultSchemaError, write_or_verify_json

from .layer1_calibration import TAU1_CALIBRATION_GRID


AMENDMENT_RELATIVE_PATH = Path("prereg_amendments/layer1_tau1_calibration_pre_attack/amendment.json")
RUN6_LOG_RELATIVE_PATH = Path("outputs/calibration/stage4_diagnostic/run6.log")


class Layer1CalibrationAmendmentError(RuntimeError):
    """Raised when the required pre-calibration amendment is absent or inconsistent."""


def _relative_to_project(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as error:
        raise Layer1CalibrationAmendmentError(
            f"Required artifact must remain under {project_root}: {path}"
        ) from error


def build_amendment_document(*, project_root: Path, run6_log: Path) -> dict[str, Any]:
    """Build the fixed pre-calibration record from the actual server run6 log."""
    if not run6_log.is_file():
        raise Layer1CalibrationAmendmentError(f"run6 failure log is missing: {run6_log}")
    try:
        run6_text = run6_log.read_text(encoding="utf-8")
    except OSError as error:
        raise Layer1CalibrationAmendmentError(f"Unable to read run6 failure log {run6_log}: {error}") from error
    if "Layer-1 DAGER filtering produced no candidate tokens." not in run6_text:
        raise Layer1CalibrationAmendmentError(
            "run6 log does not establish the required no-Layer-1-candidate calibration trigger."
        )
    if "qwen3_dager_attack_result" in run6_text:
        raise Layer1CalibrationAmendmentError(
            "run6 log contains a successful DAGER result; refusing to create a pre-reconstruction amendment."
        )
    run6_relative = _relative_to_project(project_root, run6_log)
    immutable = {
        "schema_version": 1,
        "event": "qwen3_layer1_tau1_calibration_pre_attack",
        "timing": "before_any_successful_dager_reconstruction_result",
        "run6_failure_log": {
            "path": run6_relative,
            "sha256": sha256_file(run6_log),
            "required_error": "Layer-1 DAGER filtering produced no candidate tokens.",
        },
        "attack_configuration_changed": False,
        "experiment_config_changed": False,
        "preregistration_regenerated": False,
        "sample_lists_changed": False,
        "head_seeds_changed": False,
        "observation_scope": (
            "Observe only the existing shared-rank Layer-1 vocabulary distances; "
            "do not construct a candidate provider, invoke Layer-2, load ROUGE, or score reconstruction."
        ),
        "layer2_invoked": False,
        "fixed_tau1_grid": list(TAU1_CALIBRATION_GRID),
        "selection_rule": {
            "calibration_sample_count": 20,
            "ground_truth_use": "evaluation_only_after_basis_and_distance_construction",
            "active_position_definition": "gradient_diagnostic.q0.per_token.active_by_delta=true",
            "per_tau_metrics": [
                "micro_active_position_recall",
                "micro_active_unique_token_recall",
                "nonempty_sample_count",
                "candidate_count_min_median_mean_p90_max",
            ],
            "selected_tau": "smallest_tau_with_micro_active_position_recall>=0.95_and_20_of_20_nonempty_samples",
            "no_match_behavior": "failed_no_grid_expansion_and_no_experiment_config_change",
            "manual_selection": "forbidden",
        },
    }
    immutable["amendment_identity_sha256"] = sha256_json(immutable)
    return immutable


def write_or_verify_amendment(*, project_root: Path, run6_log: Path) -> Path:
    """Write the immutable server-local amendment before any calibration scan."""
    document = build_amendment_document(project_root=project_root, run6_log=run6_log)
    path = project_root / AMENDMENT_RELATIVE_PATH
    try:
        write_or_verify_json(path, document, identity_key="amendment_identity_sha256")
    except ResultSchemaError as error:
        raise Layer1CalibrationAmendmentError(str(error)) from error
    return path


def verify_amendment(*, project_root: Path) -> Mapping[str, Any]:
    """Require a complete, hash-valid pre-calibration amendment before scanning."""
    path = project_root / AMENDMENT_RELATIVE_PATH
    if not path.is_file():
        raise Layer1CalibrationAmendmentError(
            f"Required pre-calibration amendment is missing: {path}. "
            "Create it from the actual run6 log before scanning any calibration sample."
        )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Layer1CalibrationAmendmentError(f"Unable to read amendment {path}: {error}") from error
    if not isinstance(document, Mapping):
        raise Layer1CalibrationAmendmentError("Layer-1 calibration amendment must contain one object.")
    required = {
        "event": "qwen3_layer1_tau1_calibration_pre_attack",
        "timing": "before_any_successful_dager_reconstruction_result",
        "attack_configuration_changed": False,
        "experiment_config_changed": False,
        "preregistration_regenerated": False,
        "sample_lists_changed": False,
        "head_seeds_changed": False,
        "layer2_invoked": False,
    }
    for key, value in required.items():
        if document.get(key) != value:
            raise Layer1CalibrationAmendmentError(f"Layer-1 calibration amendment field {key!r} is invalid.")
    if document.get("fixed_tau1_grid") != list(TAU1_CALIBRATION_GRID):
        raise Layer1CalibrationAmendmentError("Layer-1 calibration amendment tau1 grid differs from the fixed protocol.")
    run6 = document.get("run6_failure_log")
    if not isinstance(run6, Mapping) or not isinstance(run6.get("path"), str) or not isinstance(run6.get("sha256"), str):
        raise Layer1CalibrationAmendmentError("Layer-1 calibration amendment lacks a run6 path/hash.")
    run6_path = (project_root / run6["path"]).resolve()
    _relative_to_project(project_root, run6_path)
    if sha256_file(run6_path) != run6["sha256"]:
        raise Layer1CalibrationAmendmentError("run6 failure log SHA256 no longer matches the pre-calibration amendment.")
    observed = dict(document)
    observed_identity = observed.pop("amendment_identity_sha256", None)
    if observed_identity != sha256_json(observed):
        raise Layer1CalibrationAmendmentError("Layer-1 calibration amendment identity hash is invalid.")
    return document
