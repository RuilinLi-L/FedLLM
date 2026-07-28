"""Strict configuration loading for the Qwen3 preregistration-only scaffold."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .hashing import sha256_json


class PreregistrationConfigError(ValueError):
    """Raised when experiment.json is incomplete or violates protocol scope."""


REQUIRED_FIELDS = (
    "model_path",
    "dataset_path",
    "max_length",
    "calibration_head_seed",
    "smoke_head_seed",
    "final_head_seeds",
    "defense_base_seed",
    "calibration_parameter_grid",
    "attack_budget",
    "output_root",
)
ALLOWED_FINAL_HEAD_SEEDS = (101, 202, 303)


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated, repository-relative preregistration configuration."""

    config_path: Path
    repository_root: Path
    project_root: Path
    model_path: Path
    dataset_path: Path
    output_root: Path
    max_length: int
    min_effective_token_length: int
    calibration_head_seed: int
    smoke_head_seed: int
    final_head_seeds: tuple[int, int, int]
    defense_base_seed: int
    calibration_parameter_grid: Mapping[str, Any]
    attack_budget: Mapping[str, Any]
    raw: Mapping[str, Any]
    config_sha256: str

    def manifest_config(self) -> dict[str, Any]:
        """Return the original explicit configuration, without machine paths."""
        return dict(self.raw)


def _require_int(raw: Mapping[str, Any], key: str) -> int:
    value = raw.get(key)
    if value is None:
        raise PreregistrationConfigError(
            f"{key} must be explicitly configured; automatic seed generation is forbidden."
        )
    if isinstance(value, bool) or not isinstance(value, int):
        raise PreregistrationConfigError(f"{key} must be an integer, got {value!r}.")
    return value


def _resolve_from_repository(value: Any, *, repository_root: Path, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PreregistrationConfigError(f"{field_name} must be a non-empty path string.")
    candidate = Path(value).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (repository_root / candidate).resolve()
    return resolved


def _must_be_within(path: Path, parent: Path, *, field_name: str) -> None:
    try:
        path.relative_to(parent)
    except ValueError as error:
        raise PreregistrationConfigError(
            f"{field_name} must remain inside {parent}; got {path}."
        ) from error


def load_experiment_config(
    config_path: Path, *, require_dataset_path: bool = True
) -> ExperimentConfig:
    """Load and validate ``Projection_lrb_qwen3/configs/experiment.json``.

    Preregistration requires the saved DatasetDict to exist.  A later
    calibration run consumes only its immutable ``calibration.jsonl`` rows and
    therefore explicitly disables that unrelated filesystem check.
    """
    resolved_config = config_path.resolve()
    if not resolved_config.is_file():
        raise PreregistrationConfigError(f"Configuration file does not exist: {resolved_config}")
    try:
        raw_value = json.loads(resolved_config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreregistrationConfigError(
            f"Unable to read JSON configuration {resolved_config}: {error}"
        ) from error
    if not isinstance(raw_value, dict):
        raise PreregistrationConfigError("experiment.json must contain one JSON object.")
    raw: dict[str, Any] = raw_value

    missing = [field for field in REQUIRED_FIELDS if field not in raw]
    if missing:
        raise PreregistrationConfigError(f"experiment.json is missing required fields: {missing}")

    project_root = resolved_config.parent.parent
    repository_root = project_root.parent
    model_path = _resolve_from_repository(raw["model_path"], repository_root=repository_root, field_name="model_path")
    _must_be_within(model_path, repository_root, field_name="model_path")
    dataset_path = _resolve_from_repository(
        raw["dataset_path"], repository_root=repository_root, field_name="dataset_path"
    )
    _must_be_within(dataset_path, repository_root, field_name="dataset_path")
    if require_dataset_path and not dataset_path.is_dir():
        raise PreregistrationConfigError(
            f"dataset_path must exist and be a directory containing a saved DatasetDict: {dataset_path}"
        )
    output_root = _resolve_from_repository(raw["output_root"], repository_root=repository_root, field_name="output_root")
    expected_output_root = (project_root / "outputs").resolve()
    if output_root != expected_output_root:
        raise PreregistrationConfigError(
            "output_root must resolve exactly to Projection_lrb_qwen3/outputs; "
            f"got {output_root}."
        )

    max_length = _require_int(raw, "max_length")
    if max_length != 32:
        raise PreregistrationConfigError(
            f"This preregistration protocol requires max_length=32, got {max_length}."
        )
    min_effective_token_length = _require_int(raw, "min_effective_token_length")
    if min_effective_token_length < 1 or min_effective_token_length >= max_length:
        raise PreregistrationConfigError(
            "min_effective_token_length must be in [1, max_length - 1]."
        )

    calibration_head_seed = _require_int(raw, "calibration_head_seed")
    smoke_head_seed = _require_int(raw, "smoke_head_seed")
    final_raw = raw["final_head_seeds"]
    if not isinstance(final_raw, list) or any(isinstance(value, bool) or not isinstance(value, int) for value in final_raw):
        raise PreregistrationConfigError("final_head_seeds must be a JSON list of integers.")
    final_head_seeds = tuple(final_raw)
    if final_head_seeds != ALLOWED_FINAL_HEAD_SEEDS:
        raise PreregistrationConfigError(
            "final_head_seeds must be exactly [101, 202, 303] for this protocol."
        )
    registered_seeds = (calibration_head_seed, smoke_head_seed, *final_head_seeds)
    if len(set(registered_seeds)) != len(registered_seeds):
        raise PreregistrationConfigError(
            "calibration_head_seed and smoke_head_seed must be distinct from each other "
            "and from final_head_seeds."
        )

    defense_base_seed = _require_int(raw, "defense_base_seed")
    grid = raw["calibration_parameter_grid"]
    budget = raw["attack_budget"]
    if not isinstance(grid, dict) or not grid:
        raise PreregistrationConfigError("calibration_parameter_grid must be a non-empty JSON object.")
    if not isinstance(budget, dict) or not budget:
        raise PreregistrationConfigError("attack_budget must be a non-empty JSON object.")

    return ExperimentConfig(
        config_path=resolved_config,
        repository_root=repository_root,
        project_root=project_root,
        model_path=model_path,
        dataset_path=dataset_path,
        output_root=output_root,
        max_length=max_length,
        min_effective_token_length=min_effective_token_length,
        calibration_head_seed=calibration_head_seed,
        smoke_head_seed=smoke_head_seed,
        final_head_seeds=(101, 202, 303),
        defense_base_seed=defense_base_seed,
        calibration_parameter_grid=grid,
        attack_budget=budget,
        raw=raw,
        config_sha256=sha256_json(raw),
    )
