"""Immutable consumption control for the audited Qwen3 Layer-1 tau1 calibration.

This module turns the completed Layer-1-only calibration aggregation into the
only accepted source of ``tau1`` for the none-only Qwen3 DAGER entrypoint.  It
does not import a model, allocate CUDA memory, invoke ROUGE, run Layer-2, or
load any LRB code.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from src.hashing import hash_sample_list, sha256_file, sha256_json
from src.result_schema import ResultSchemaError, write_or_verify_json

from .bf16_gate_profile_amendment import (
    AMENDMENT_RELATIVE_PATH as BF16_GATE_AMENDMENT_RELATIVE_PATH,
    SELECTED_BFLOAT16_GATE,
    verify_amendment as verify_bf16_gate_profile_amendment,
)


FROZEN_CONTROL_RECORD_TYPE = "qwen3_frozen_tau1_control"
EXPECTED_CALIBRATION_SAMPLE_COUNT = 20
EXPECTED_CALIBRATION_HEAD_SEED = 11
EXPECTED_SELECTED_TAU1 = 2e-3
FIXED_TAU1_GRID: tuple[float, ...] = (
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


class FrozenTau1ControlError(RuntimeError):
    """Raised when a frozen tau1 control or any of its sources is inconsistent."""


def _repository_root(project_root: Path) -> Path:
    return project_root.resolve().parent


def _read_json(path: Path, *, description: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FrozenTau1ControlError(f"Unable to read {description} {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise FrozenTau1ControlError(f"{description} must contain one JSON object: {path}")
    return value


def _read_jsonl(path: Path, *, description: str) -> list[Mapping[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise FrozenTau1ControlError(f"Unable to read {description} {path}: {error}") from error
    if not lines:
        raise FrozenTau1ControlError(f"{description} must not be empty: {path}")
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise FrozenTau1ControlError(
                f"Invalid JSON in {description} at {path}:{line_number}: {error}"
            ) from error
        if not isinstance(value, Mapping):
            raise FrozenTau1ControlError(f"{description} row {line_number} must contain one object.")
        records.append(value)
    return records


def _require_sha256(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise FrozenTau1ControlError(f"{description} must be one 64-character SHA256 string.")
    try:
        int(value, 16)
    except ValueError as error:
        raise FrozenTau1ControlError(f"{description} must be hexadecimal SHA256.") from error
    return value


def _require_exact_int(value: Any, *, expected: int, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise FrozenTau1ControlError(f"{description} must equal {expected}, got {value!r}.")
    return value


def _require_exact_float(value: Any, *, expected: float, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FrozenTau1ControlError(f"{description} must be numeric, got {value!r}.")
    result = float(value)
    if not math.isfinite(result) or result != expected:
        raise FrozenTau1ControlError(f"{description} must equal {expected:g}, got {value!r}.")
    return result


def _relative_to_repository(project_root: Path, path: Path, *, description: str) -> str:
    try:
        return path.resolve().relative_to(_repository_root(project_root)).as_posix()
    except ValueError as error:
        raise FrozenTau1ControlError(f"{description} must remain under the repository: {path}") from error


def _resolve_repository_relative(
    project_root: Path,
    value: Any,
    *,
    description: str,
    require_project_subtree: bool = True,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise FrozenTau1ControlError(f"{description} must be a non-empty repository-relative path.")
    candidate = Path(value)
    if candidate.is_absolute():
        raise FrozenTau1ControlError(f"{description} must be repository-relative, not absolute: {value!r}.")
    resolved = (_repository_root(project_root) / candidate).resolve()
    try:
        if require_project_subtree:
            resolved.relative_to(project_root.resolve())
        else:
            resolved.relative_to(_repository_root(project_root))
    except ValueError as error:
        boundary = project_root if require_project_subtree else _repository_root(project_root)
        raise FrozenTau1ControlError(f"{description} must remain under {boundary}: {value!r}.") from error
    return resolved


def _require_under(path: Path, parent: Path, *, description: str) -> None:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError as error:
        raise FrozenTau1ControlError(f"{description} must remain under {parent}, got {path}.") from error


def _git_commit(repository_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value if value else None


def _active_calibration_protocol(project_root: Path) -> tuple[Mapping[str, Any], str, list[Mapping[str, Any]]]:
    manifests_root = project_root / "manifests"
    preregistration = _read_json(manifests_root / "preregistration.json", description="active preregistration")
    preregistration_sha256 = _require_sha256(
        preregistration.get("preregistration_sha256"),
        description="active preregistration_sha256",
    )
    sample_lists = preregistration.get("sample_lists")
    sample_hashes = preregistration.get("sample_list_sha256")
    if not isinstance(sample_lists, Mapping) or not isinstance(sample_hashes, Mapping):
        raise FrozenTau1ControlError("Active preregistration lacks sample_lists or sample_list_sha256.")
    calibration_samples = sample_lists.get("calibration")
    calibration_hash = _require_sha256(
        sample_hashes.get("calibration"),
        description="active calibration sample-list SHA256",
    )
    if not isinstance(calibration_samples, list) or len(calibration_samples) != EXPECTED_CALIBRATION_SAMPLE_COUNT:
        raise FrozenTau1ControlError("Active preregistration must contain exactly 20 calibration samples.")
    if hash_sample_list(calibration_samples) != calibration_hash:
        raise FrozenTau1ControlError("Active preregistration calibration sample list no longer matches its SHA256.")

    config = preregistration.get("config")
    if not isinstance(config, Mapping):
        raise FrozenTau1ControlError("Active preregistration lacks its immutable config object.")
    _require_exact_int(
        config.get("calibration_head_seed"),
        expected=EXPECTED_CALIBRATION_HEAD_SEED,
        description="active preregistration calibration_head_seed",
    )

    calibration_manifest = _read_jsonl(
        manifests_root / "calibration.jsonl",
        description="active calibration manifest",
    )
    if len(calibration_manifest) != EXPECTED_CALIBRATION_SAMPLE_COUNT:
        raise FrozenTau1ControlError("Active calibration manifest must contain exactly 20 rows.")
    manifest_samples: list[Mapping[str, Any]] = []
    for ordinal, row in enumerate(calibration_manifest):
        if row.get("record_type") != "preregistered_sst2_validation_sample":
            raise FrozenTau1ControlError(f"Active calibration manifest row {ordinal} has an unexpected record_type.")
        if row.get("stage") != "calibration" or row.get("preregistration_sha256") != preregistration_sha256:
            raise FrozenTau1ControlError(f"Active calibration manifest row {ordinal} does not match active protocol identity.")
        if row.get("stage_sample_list_sha256") != calibration_hash:
            raise FrozenTau1ControlError(f"Active calibration manifest row {ordinal} has a mismatched sample-list hash.")
        sample = row.get("sample")
        if not isinstance(sample, Mapping):
            raise FrozenTau1ControlError(f"Active calibration manifest row {ordinal} lacks its sample object.")
        manifest_samples.append(sample)
    if hash_sample_list(manifest_samples) != calibration_hash or manifest_samples != calibration_samples:
        raise FrozenTau1ControlError("Active calibration JSONL does not exactly match preregistration sample-list identity.")
    return preregistration, calibration_hash, calibration_samples


def _expected_bf16_gate_summary(project_root: Path) -> dict[str, Any]:
    amendment = verify_bf16_gate_profile_amendment(project_root=project_root)
    amendment_path = project_root / BF16_GATE_AMENDMENT_RELATIVE_PATH
    if not amendment_path.is_file():
        raise FrozenTau1ControlError(f"Active BF16 gate amendment is missing: {amendment_path}")
    _require_exact_float(
        amendment.get("selected_bfloat16_gate"),
        expected=SELECTED_BFLOAT16_GATE,
        description="active BF16 gate amendment selected_bfloat16_gate",
    )
    return {
        "path": _relative_to_repository(project_root, amendment_path, description="BF16 gate amendment"),
        "sha256": sha256_file(amendment_path),
        "amendment_identity_sha256": _require_sha256(
            amendment.get("amendment_identity_sha256"),
            description="active BF16 gate amendment identity",
        ),
        "selected_bfloat16_gate": SELECTED_BFLOAT16_GATE,
        "fixed_candidate_grid": amendment.get("fixed_candidate_grid"),
    }


def _validate_sample_record(
    *,
    record: Mapping[str, Any],
    expected_sample: Mapping[str, Any],
    preregistration_sha256: str,
    expected_bf16_gate: Mapping[str, Any],
    path: Path,
) -> None:
    if record.get("record_type") != "qwen3_layer1_tau1_calibration_sample":
        raise FrozenTau1ControlError(f"Calibration sample output has wrong record_type: {path}")
    if record.get("status") != "ok" or record.get("stage") != "calibration" or record.get("layer2_invoked") is not False:
        raise FrozenTau1ControlError(f"Calibration sample output is not a successful Layer-1-only observation: {path}")
    if record.get("sample_key") != expected_sample.get("sample_key"):
        raise FrozenTau1ControlError(f"Calibration sample output does not match its immutable sample key: {path}")
    _require_exact_int(
        record.get("head_seed"),
        expected=EXPECTED_CALIBRATION_HEAD_SEED,
        description=f"Calibration sample output head_seed in {path}",
    )
    if record.get("dtype") != "bfloat16":
        raise FrozenTau1ControlError(f"Calibration sample output must be BF16: {path}")
    identity = record.get("identity")
    if not isinstance(identity, Mapping) or identity.get("preregistration_sha256") != preregistration_sha256:
        raise FrozenTau1ControlError(f"Calibration sample output has a mismatched preregistration identity: {path}")
    diagnostic = record.get("gradient_diagnostic_summary")
    if not isinstance(diagnostic, Mapping) or diagnostic.get("passed") is not True:
        raise FrozenTau1ControlError(f"Calibration sample output lacks a passed gradient diagnostic: {path}")
    controls = record.get("gradient_diagnostic_controls")
    if not isinstance(controls, Mapping):
        raise FrozenTau1ControlError(f"Calibration sample output lacks diagnostic controls: {path}")
    _require_exact_float(
        controls.get("max_active_relative_residual"),
        expected=SELECTED_BFLOAT16_GATE,
        description=f"Calibration sample BF16 gate in {path}",
    )
    if record.get("bf16_gate_profile_amendment") != dict(expected_bf16_gate):
        raise FrozenTau1ControlError(f"Calibration sample output has an inconsistent BF16 gate amendment: {path}")
    scan = record.get("vocabulary_scan")
    if not isinstance(scan, Mapping):
        raise FrozenTau1ControlError(f"Calibration sample output lacks vocabulary_scan: {path}")
    vocab_size = scan.get("vocab_size")
    scanned = scan.get("scanned_token_count")
    if (
        isinstance(vocab_size, bool)
        or isinstance(scanned, bool)
        or not isinstance(vocab_size, int)
        or not isinstance(scanned, int)
        or vocab_size <= 0
        or scanned != vocab_size
    ):
        raise FrozenTau1ControlError(f"Calibration sample output did not scan its full vocabulary exactly once: {path}")


def _validate_aggregation(
    *, project_root: Path, aggregation_path: Path
) -> tuple[Mapping[str, Any], list[dict[str, str]], str, str, Mapping[str, Any]]:
    _require_under(
        aggregation_path,
        project_root / "outputs" / "calibration",
        description="Frozen-control aggregation",
    )
    aggregation = _read_json(aggregation_path, description="Layer-1 calibration aggregation")
    if aggregation.get("record_type") != "qwen3_layer1_tau1_calibration_aggregation":
        raise FrozenTau1ControlError("Aggregation record_type is not qwen3_layer1_tau1_calibration_aggregation.")
    if aggregation.get("status") != "ok" or aggregation.get("selection_rule_passed") is not True:
        raise FrozenTau1ControlError("Aggregation must have status=ok and selection_rule_passed=true.")
    if aggregation.get("layer2_invoked") is not False:
        raise FrozenTau1ControlError("Aggregation must certify layer2_invoked=false.")
    _require_exact_int(
        aggregation.get("calibration_sample_count"),
        expected=EXPECTED_CALIBRATION_SAMPLE_COUNT,
        description="aggregation calibration_sample_count",
    )
    grid = aggregation.get("fixed_tau1_grid")
    if not isinstance(grid, list) or tuple(grid) != FIXED_TAU1_GRID:
        raise FrozenTau1ControlError("Aggregation fixed_tau1_grid differs from the immutable 13-point calibration grid.")
    _require_exact_float(
        aggregation.get("selected_tau1"),
        expected=EXPECTED_SELECTED_TAU1,
        description="aggregation selected_tau1",
    )
    selection_rule = aggregation.get("selection_rule")
    if not isinstance(selection_rule, Mapping):
        raise FrozenTau1ControlError("Aggregation lacks a selection_rule object.")
    _require_exact_float(
        selection_rule.get("minimum_micro_active_position_recall"),
        expected=0.95,
        description="aggregation selection-rule recall threshold",
    )
    _require_exact_int(
        selection_rule.get("required_nonempty_sample_count"),
        expected=EXPECTED_CALIBRATION_SAMPLE_COUNT,
        description="aggregation selection-rule nonempty sample count",
    )
    per_tau = aggregation.get("per_tau")
    if not isinstance(per_tau, list) or len(per_tau) != len(FIXED_TAU1_GRID):
        raise FrozenTau1ControlError("Aggregation per_tau must cover the immutable 13-point tau1 grid exactly once.")
    recomputed_selected: float | None = None
    for expected_tau, row in zip(FIXED_TAU1_GRID, per_tau):
        if not isinstance(row, Mapping):
            raise FrozenTau1ControlError("Aggregation per_tau rows must be objects.")
        _require_exact_float(row.get("tau"), expected=expected_tau, description="aggregation per_tau tau")
        recall = row.get("micro_active_position_recall")
        nonempty = row.get("nonempty_sample_count")
        if isinstance(recall, bool) or not isinstance(recall, (int, float)) or not math.isfinite(float(recall)):
            raise FrozenTau1ControlError("Aggregation per_tau recall must be finite numeric.")
        if isinstance(nonempty, bool) or not isinstance(nonempty, int):
            raise FrozenTau1ControlError("Aggregation per_tau nonempty_sample_count must be an integer.")
        if recomputed_selected is None and float(recall) >= 0.95 and nonempty == EXPECTED_CALIBRATION_SAMPLE_COUNT:
            recomputed_selected = expected_tau
    if recomputed_selected != EXPECTED_SELECTED_TAU1:
        raise FrozenTau1ControlError(
            "Aggregation selected_tau1 does not equal the smallest tau satisfying the fixed selection rule."
        )

    preregistration, calibration_hash, calibration_samples = _active_calibration_protocol(project_root)
    preregistration_sha256 = _require_sha256(
        preregistration.get("preregistration_sha256"), description="active preregistration_sha256"
    )
    expected_bf16_gate = _expected_bf16_gate_summary(project_root)
    if aggregation.get("bf16_gate_profile_amendment") != expected_bf16_gate:
        raise FrozenTau1ControlError("Aggregation BF16 gate amendment does not match the active immutable amendment.")

    sample_output_files = aggregation.get("sample_output_files")
    if not isinstance(sample_output_files, list) or len(sample_output_files) != EXPECTED_CALIBRATION_SAMPLE_COUNT:
        raise FrozenTau1ControlError("Aggregation must name exactly 20 calibration sample output files.")
    expected_keys = [sample.get("sample_key") for sample in calibration_samples]
    if any(not isinstance(key, str) or len(key) != 64 for key in expected_keys):
        raise FrozenTau1ControlError("Active calibration sample list contains an invalid sample_key.")
    normalized_files: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    seen_keys: set[str] = set()
    for ordinal, (expected_key, item) in enumerate(zip(expected_keys, sample_output_files)):
        if not isinstance(item, Mapping):
            raise FrozenTau1ControlError(f"Aggregation sample_output_files[{ordinal}] must be an object.")
        sample_key = item.get("sample_key")
        if sample_key != expected_key:
            raise FrozenTau1ControlError("Aggregation sample outputs must cover the active calibration manifest in order.")
        relative_path = item.get("path")
        sample_path = _resolve_repository_relative(
            project_root, relative_path, description=f"aggregation sample output {ordinal}"
        )
        _require_under(
            sample_path,
            project_root / "outputs" / "calibration",
            description="Frozen-control calibration sample output",
        )
        relative_path = _relative_to_repository(project_root, sample_path, description="calibration sample output")
        expected_sha = _require_sha256(item.get("sha256"), description=f"aggregation sample output SHA256 {ordinal}")
        if relative_path in seen_paths or sample_key in seen_keys:
            raise FrozenTau1ControlError("Aggregation sample output paths and sample keys must each be unique.")
        seen_paths.add(relative_path)
        seen_keys.add(sample_key)
        if not sample_path.is_file() or sha256_file(sample_path) != expected_sha:
            raise FrozenTau1ControlError(f"Aggregation sample output SHA256 mismatch: {sample_path}")
        sample_record = _read_json(sample_path, description="calibration sample output")
        _validate_sample_record(
            record=sample_record,
            expected_sample=calibration_samples[ordinal],
            preregistration_sha256=preregistration_sha256,
            expected_bf16_gate=expected_bf16_gate,
            path=sample_path,
        )
        normalized_files.append({"sample_key": sample_key, "path": relative_path, "sha256": expected_sha})

    if len(seen_paths) != EXPECTED_CALIBRATION_SAMPLE_COUNT or len(seen_keys) != EXPECTED_CALIBRATION_SAMPLE_COUNT:
        raise FrozenTau1ControlError("Aggregation does not provide exactly 20 unique calibration sample outputs.")
    expected_aggregation_identity = sha256_json(
        {
            "protocol": "qwen3_layer1_tau1_calibration_v1",
            "preregistration_sha256": preregistration_sha256,
            "config_sha256": preregistration.get("config_sha256"),
            "dtype": "bfloat16",
            "head_seed": EXPECTED_CALIBRATION_HEAD_SEED,
            "tau_grid": list(FIXED_TAU1_GRID),
            "sample_output_files": normalized_files,
            "bf16_gate_profile_amendment": expected_bf16_gate,
        }
    )
    if aggregation.get("calibration_aggregation_identity_sha256") != expected_aggregation_identity:
        raise FrozenTau1ControlError("Aggregation identity does not match active protocol and hash-verified sample outputs.")
    return aggregation, normalized_files, preregistration_sha256, calibration_hash, expected_bf16_gate


def _build_control_document(
    *,
    project_root: Path,
    aggregation_path: Path,
    created_at_utc: str | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    aggregation, sample_files, preregistration_sha256, calibration_hash, bf16_gate = _validate_aggregation(
        project_root=project_root,
        aggregation_path=aggregation_path,
    )
    if created_at_utc is None:
        created_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if not isinstance(created_at_utc, str) or not created_at_utc:
        raise FrozenTau1ControlError("frozen control created_at_utc must be a non-empty string.")
    document: dict[str, Any] = {
        "schema_version": 1,
        "record_type": FROZEN_CONTROL_RECORD_TYPE,
        "defense": "none",
        "stage_source": "calibration",
        "selected_tau1": EXPECTED_SELECTED_TAU1,
        "fixed_tau1_grid": list(FIXED_TAU1_GRID),
        "selection_rule": dict(aggregation["selection_rule"]),
        "selection_rule_passed": True,
        "calibration_sample_count": EXPECTED_CALIBRATION_SAMPLE_COUNT,
        "calibration_head_seed": EXPECTED_CALIBRATION_HEAD_SEED,
        "aggregation_path": _relative_to_repository(project_root, aggregation_path, description="aggregation"),
        "aggregation_sha256": sha256_file(aggregation_path),
        "sample_output_files": sample_files,
        "preregistration_sha256": preregistration_sha256,
        "calibration_sample_list_sha256": calibration_hash,
        "bfloat16_gate": SELECTED_BFLOAT16_GATE,
        "bf16_gate_amendment_path": bf16_gate["path"],
        "bf16_gate_amendment_identity": bf16_gate["amendment_identity_sha256"],
        "bf16_gate_amendment_artifact_sha256": bf16_gate["sha256"],
        "git_commit": _git_commit(_repository_root(project_root)) if git_commit is None else git_commit,
        "created_at_utc": created_at_utc,
    }
    document["frozen_control_identity_sha256"] = sha256_json(document)
    return document


def write_or_verify_frozen_tau1_control(
    *, project_root: Path, aggregation_path: Path, output_path: Path
) -> tuple[Path, Mapping[str, Any], bool]:
    """Create a frozen tau1 control once, or retain only a byte-equivalent identity.

    If the control already exists, its historical creation time and git commit
    are retained while every scientific source is freshly revalidated.  Thus a
    subsequent code commit cannot rewrite a calibration decision, while any
    changed aggregation, sample output, preregistration, or amendment conflicts.
    """
    _require_under(output_path, project_root, description="Frozen tau1 control output")
    existing: Mapping[str, Any] | None = None
    if output_path.exists():
        existing = _read_json(output_path, description="existing frozen tau1 control")
    document = _build_control_document(
        project_root=project_root,
        aggregation_path=aggregation_path,
        created_at_utc=None if existing is None else existing.get("created_at_utc"),
        git_commit=None if existing is None else existing.get("git_commit"),
    )
    try:
        written = write_or_verify_json(
            output_path,
            document,
            identity_key="frozen_control_identity_sha256",
        )
    except ResultSchemaError as error:
        raise FrozenTau1ControlError(str(error)) from error
    return output_path, document, written


def verify_frozen_tau1_control(*, project_root: Path, control_path: Path) -> Mapping[str, Any]:
    """Verify one frozen control and every active source before formal attack work."""
    _require_under(control_path, project_root, description="Frozen tau1 control")
    document = _read_json(control_path, description="frozen tau1 control")
    if document.get("record_type") != FROZEN_CONTROL_RECORD_TYPE:
        raise FrozenTau1ControlError("Frozen control has an unexpected record_type.")
    stored_identity = _require_sha256(
        document.get("frozen_control_identity_sha256"),
        description="frozen control identity",
    )
    identity_input = dict(document)
    identity_input.pop("frozen_control_identity_sha256", None)
    if sha256_json(identity_input) != stored_identity:
        raise FrozenTau1ControlError("Frozen control identity SHA256 does not match canonical JSON content.")
    if document.get("defense") != "none" or document.get("stage_source") != "calibration":
        raise FrozenTau1ControlError("Frozen control is not a none-only control derived from calibration.")
    aggregation_path = _resolve_repository_relative(
        project_root,
        document.get("aggregation_path"),
        description="frozen control aggregation_path",
    )
    expected = _build_control_document(
        project_root=project_root,
        aggregation_path=aggregation_path,
        created_at_utc=document.get("created_at_utc"),
        git_commit=document.get("git_commit"),
    )
    if document != expected:
        raise FrozenTau1ControlError(
            "Frozen control does not match a fresh verification of its aggregation, calibration outputs, "
            "active preregistration, and BF16 gate amendment."
        )
    return document
