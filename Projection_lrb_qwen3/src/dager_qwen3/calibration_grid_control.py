"""Immutable none-only calibration-grid control for Qwen3 Stage 5.

The control is intentionally independent from ``experiment.json``.  In
particular, Layer-1 tau1 is never inferred from its historical grid.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from src.hashing import canonical_json_bytes, hash_sample_list, sha256_file, sha256_json
from src.result_schema import ResultSchemaError, write_or_verify_json


RECORD_TYPE = "qwen3_none_attack_calibration_grid"
FIXED_TAU1 = 0.002
FIXED_BF16_GATE = 0.0075
FIXED_RANK_RTOL = 0.001
FIXED_RANK_CUTOFF = 20
FIXED_MAX_C = 10_000_000
FIXED_MAX_IDS = -1
FIXED_PARALLEL = 1000
TAU2_CANDIDATE_GRID: tuple[float, ...] = (5e-4, 1e-3, 2e-3)
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


class CalibrationGridControlError(RuntimeError):
    """Raised when a Stage-5 grid control is malformed or inconsistent."""


def _read_json(path: Path, *, description: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationGridControlError(f"Unable to read {description} {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise CalibrationGridControlError(f"{description} must be one JSON object.")
    return value


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise CalibrationGridControlError(f"Unable to read calibration manifest {path}: {error}") from error
    if not lines:
        raise CalibrationGridControlError("calibration.jsonl must not be empty.")
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise CalibrationGridControlError(
                f"Invalid calibration JSONL at {path}:{line_number}: {error}"
            ) from error
        if not isinstance(row, Mapping):
            raise CalibrationGridControlError("Calibration JSONL rows must be objects.")
        rows.append(row)
    return rows


def _require_sha256(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise CalibrationGridControlError(f"{description} must be a 64-character SHA256 string.")
    try:
        int(value, 16)
    except ValueError as error:
        raise CalibrationGridControlError(f"{description} must be hexadecimal SHA256.") from error
    return value


def _repository_root(project_root: Path) -> Path:
    return project_root.resolve().parent


def _relative(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(_repository_root(project_root)).as_posix()
    except ValueError as error:
        raise CalibrationGridControlError(f"Artifact path is outside the repository: {path}") from error


def _resolve_project_relative(project_root: Path, value: Any, *, description: str) -> Path:
    if not isinstance(value, str) or not value.strip() or Path(value).is_absolute():
        raise CalibrationGridControlError(f"{description} must be a non-empty repository-relative path.")
    resolved = (_repository_root(project_root) / value).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as error:
        raise CalibrationGridControlError(f"{description} must remain under {project_root}.") from error
    return resolved


def _git_commit(repository_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository_root, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise CalibrationGridControlError("Unable to resolve git HEAD for calibration-grid control.") from error
    value = completed.stdout.strip()
    if not value:
        raise CalibrationGridControlError("git rev-parse HEAD returned an empty commit.")
    return value


def verify_tau1_reference(*, project_root: Path, control_path: Path) -> Mapping[str, Any]:
    """Verify the frozen tau1 document identity and active calibration bindings.

    This intentionally does not require historical Layer-1 output files: those
    are already committed into the frozen control by their hashes, whereas the
    Stage-5 boundary requires identity/file-hash and active-manifest checks.
    """
    document = _read_json(control_path, description="frozen tau1 control")
    identity = _require_sha256(document.get("frozen_control_identity_sha256"), description="tau1 control identity")
    identity_input = dict(document)
    identity_input.pop("frozen_control_identity_sha256", None)
    if sha256_json(identity_input) != identity:
        raise CalibrationGridControlError("Frozen tau1 control canonical identity does not match its content.")
    if document.get("record_type") != "qwen3_frozen_tau1_control" or document.get("defense") != "none":
        raise CalibrationGridControlError("Frozen tau1 control is not the expected none-only control.")
    if float(document.get("selected_tau1", -1.0)) != FIXED_TAU1:
        raise CalibrationGridControlError(f"Frozen tau1 must equal {FIXED_TAU1:g}.")
    if float(document.get("bfloat16_gate", -1.0)) != FIXED_BF16_GATE:
        raise CalibrationGridControlError(f"Frozen BF16 gate must equal {FIXED_BF16_GATE:g}.")
    preregistration = _read_json(project_root / "manifests" / "preregistration.json", description="preregistration")
    preregistration_sha = _require_sha256(preregistration.get("preregistration_sha256"), description="preregistration SHA256")
    sample_hashes = preregistration.get("sample_list_sha256")
    if not isinstance(sample_hashes, Mapping):
        raise CalibrationGridControlError("preregistration lacks sample_list_sha256.")
    calibration_hash = _require_sha256(sample_hashes.get("calibration"), description="calibration sample-list SHA256")
    rows = _read_jsonl(project_root / "manifests" / "calibration.jsonl")
    samples: list[Mapping[str, Any]] = []
    for row in rows:
        if row.get("stage") != "calibration" or row.get("preregistration_sha256") != preregistration_sha:
            raise CalibrationGridControlError("Calibration manifest does not match active preregistration.")
        if row.get("stage_sample_list_sha256") != calibration_hash or not isinstance(row.get("sample"), Mapping):
            raise CalibrationGridControlError("Calibration manifest has an invalid sample-list binding.")
        samples.append(row["sample"])
    if len(samples) != 20 or hash_sample_list(samples) != calibration_hash:
        raise CalibrationGridControlError("Calibration manifest must contain the active 20-sample list exactly.")
    if document.get("preregistration_sha256") != preregistration_sha:
        raise CalibrationGridControlError("Frozen tau1 control preregistration SHA256 does not match active protocol.")
    if document.get("calibration_sample_list_sha256") != calibration_hash:
        raise CalibrationGridControlError("Frozen tau1 control calibration sample-list SHA256 does not match active protocol.")
    return document


def _build_document(
    *, project_root: Path, tau1_control_path: Path, created_at_utc: str, git_commit: str
) -> dict[str, Any]:
    tau1 = verify_tau1_reference(project_root=project_root, control_path=tau1_control_path)
    if not isinstance(created_at_utc, str) or not created_at_utc:
        raise CalibrationGridControlError("created_at_utc must be a non-empty string.")
    if not isinstance(git_commit, str) or not git_commit:
        raise CalibrationGridControlError("git_commit must be non-empty.")
    document: dict[str, Any] = {
        "schema_version": 1,
        "record_type": RECORD_TYPE,
        "defense": "none",
        "stage": "calibration",
        "preregistration_sha256": tau1["preregistration_sha256"],
        "calibration_sample_list_sha256": tau1["calibration_sample_list_sha256"],
        "frozen_tau1_control_path": _relative(project_root, tau1_control_path),
        "frozen_tau1_control_file_sha256": sha256_file(tau1_control_path),
        "frozen_tau1_control_identity_sha256": tau1["frozen_control_identity_sha256"],
        "fixed_tau1": FIXED_TAU1,
        "bf16_gate": FIXED_BF16_GATE,
        "fixed_rank_rtol": FIXED_RANK_RTOL,
        "fixed_rank_cutoff": FIXED_RANK_CUTOFF,
        "fixed_maxC": FIXED_MAX_C,
        "fixed_max_ids": FIXED_MAX_IDS,
        "fixed_parallel": FIXED_PARALLEL,
        "tau2_candidate_grid": list(TAU2_CANDIDATE_GRID),
        "candidate_count": len(TAU2_CANDIDATE_GRID),
        "selection_rule": SELECTION_RULE,
        "tie_break_rule": TIE_BREAK_RULE,
        "created_at_utc": created_at_utc,
        "git_commit": git_commit,
    }
    document["identity_sha256"] = sha256_json(document)
    return document


def write_or_verify_calibration_grid_control(
    *, project_root: Path, tau1_control_path: Path, output_path: Path
) -> tuple[Path, Mapping[str, Any], bool]:
    """Create the control once or verify it without changing its creation metadata."""
    existing = _read_json(output_path, description="existing calibration grid control") if output_path.exists() else None
    document = _build_document(
        project_root=project_root,
        tau1_control_path=tau1_control_path,
        created_at_utc=(
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            if existing is None
            else str(existing.get("created_at_utc", ""))
        ),
        git_commit=_git_commit(_repository_root(project_root)) if existing is None else str(existing.get("git_commit", "")),
    )
    try:
        written = write_or_verify_json(output_path, document, identity_key="identity_sha256")
    except ResultSchemaError as error:
        raise CalibrationGridControlError(str(error)) from error
    return output_path, document, written


def verify_calibration_grid_control(*, project_root: Path, control_path: Path) -> Mapping[str, Any]:
    """Verify immutable grid content and every input allowed before attack work."""
    document = _read_json(control_path, description="calibration grid control")
    stored_identity = _require_sha256(document.get("identity_sha256"), description="grid control identity")
    identity_input = dict(document)
    identity_input.pop("identity_sha256", None)
    if sha256_json(identity_input) != stored_identity:
        raise CalibrationGridControlError("Calibration grid control canonical identity does not match its content.")
    if document.get("record_type") != RECORD_TYPE or document.get("defense") != "none" or document.get("stage") != "calibration":
        raise CalibrationGridControlError("Calibration grid control has an invalid protocol boundary.")
    tau1_path = _resolve_project_relative(
        project_root, document.get("frozen_tau1_control_path"), description="frozen_tau1_control_path"
    )
    tau1 = verify_tau1_reference(project_root=project_root, control_path=tau1_path)
    required = {
        "preregistration_sha256": tau1["preregistration_sha256"],
        "calibration_sample_list_sha256": tau1["calibration_sample_list_sha256"],
        "frozen_tau1_control_file_sha256": sha256_file(tau1_path),
        "frozen_tau1_control_identity_sha256": tau1["frozen_control_identity_sha256"],
        "fixed_tau1": FIXED_TAU1,
        "bf16_gate": FIXED_BF16_GATE,
        "fixed_rank_rtol": FIXED_RANK_RTOL,
        "fixed_rank_cutoff": FIXED_RANK_CUTOFF,
        "fixed_maxC": FIXED_MAX_C,
        "fixed_max_ids": FIXED_MAX_IDS,
        "fixed_parallel": FIXED_PARALLEL,
        "candidate_count": len(TAU2_CANDIDATE_GRID),
        "selection_rule": SELECTION_RULE,
        "tie_break_rule": TIE_BREAK_RULE,
    }
    for key, expected in required.items():
        if document.get(key) != expected:
            raise CalibrationGridControlError(f"Calibration grid control field {key} differs from its fixed value.")
    if document.get("tau2_candidate_grid") != list(TAU2_CANDIDATE_GRID):
        raise CalibrationGridControlError("Calibration grid control tau2 candidate grid is invalid.")
    if len(TAU2_CANDIDATE_GRID) < 2:
        raise CalibrationGridControlError("Formal calibration requires at least two tau2 candidates.")
    return document


def candidate_parameters_from_grid(document: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Materialize the grid's only declared varying axis: tau2."""
    tau2_values = document.get("tau2_candidate_grid")
    if not isinstance(tau2_values, list) or len(tau2_values) != document.get("candidate_count"):
        raise CalibrationGridControlError("Grid candidate_count does not match tau2_candidate_grid.")
    candidates: list[dict[str, Any]] = []
    for tau2 in tau2_values:
        if isinstance(tau2, bool) or not isinstance(tau2, (int, float)) or float(tau2) <= 0.0:
            raise CalibrationGridControlError("tau2 candidates must be finite positive numbers.")
        candidates.append(
            {
                "tau1": document["fixed_tau1"],
                "tau2": float(tau2),
                "numerical_rank_threshold": document["fixed_rank_rtol"],
                "rank_cutoff": document["fixed_rank_cutoff"],
                "candidate_budget": {"max_ids": document["fixed_max_ids"]},
                "search_budget": {"maxC": document["fixed_maxC"], "parallel": document["fixed_parallel"]},
            }
        )
    candidate_ids = [sha256_json(candidate) for candidate in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise CalibrationGridControlError("Grid generates duplicate canonical candidate IDs.")
    return tuple(candidates)
