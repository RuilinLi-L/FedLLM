"""Strict preregistration and none-only diagnostics for Qwen3 DAGER runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.config import ExperimentConfig
from src.hashing import hash_sample_list

from . import ATTACK_NAME


class AttackProtocolError(RuntimeError):
    """Raised when an attack request departs from the preregistered protocol."""


@dataclass(frozen=True)
class RegisteredAttackSample:
    """One immutable manifest sample selected by sample key rather than free text."""

    stage: str
    preregistration_sha256: str
    sample_key: str
    original_index: int
    sentence: str
    label: int
    input_ids: tuple[int, ...]
    eos_token_id: int


@dataclass(frozen=True)
class NoneAttackControls:
    """One fixed calibration point and bounded standard DAGER search budget."""

    l1_span_threshold: float
    l2_span_threshold: float
    rank_tolerance: float | None
    rank_cutoff: int
    vocab_chunk_size: int
    decode_batch_size: int
    max_search_candidates: int
    max_candidate_ids: int
    max_sequence_length: int


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AttackProtocolError(f"Unable to read JSON artifact {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise AttackProtocolError(f"JSON artifact {path} must contain one object.")
    return value


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise AttackProtocolError(f"Unable to read JSONL manifest {path}: {error}") from error
    if not lines:
        raise AttackProtocolError(f"Manifest JSONL is empty: {path}")
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise AttackProtocolError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
        if not isinstance(record, Mapping):
            raise AttackProtocolError(f"JSONL record at {path}:{line_number} must be one object.")
        records.append(record)
    return records


def _only_grid_value(grid: Mapping[str, Any], key: str, *, allow_none: bool = False) -> Any:
    values = grid.get(key)
    if not isinstance(values, list) or len(values) != 1:
        raise AttackProtocolError(
            f"Calibration is not locked for {key}: expected exactly one configured value, got {values!r}."
        )
    value = values[0]
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AttackProtocolError(f"Calibration value {key} must be numeric, got {value!r}.")
    return value


def load_none_attack_controls(config: ExperimentConfig) -> NoneAttackControls:
    """Read only predeclared calibration/budget fields; no CLI tuning is accepted."""
    grid = config.calibration_parameter_grid
    budget = config.attack_budget
    l1 = float(_only_grid_value(grid, "l1_span_thresh"))
    l2 = float(_only_grid_value(grid, "l2_span_thresh"))
    rank_value = _only_grid_value(grid, "rank_tol", allow_none=True)
    rank_tolerance = None if rank_value is None else float(rank_value)
    rank_cutoff_value = _only_grid_value(grid, "rank_cutoff")
    if isinstance(rank_cutoff_value, bool) or not isinstance(rank_cutoff_value, int) or rank_cutoff_value < 0:
        raise AttackProtocolError("calibration rank_cutoff must be one non-negative integer.")
    parallel = budget.get("parallel")
    max_ids = budget.get("max_ids")
    max_search = budget.get("maxC")
    for name, value in (("attack_budget.parallel", parallel), ("attack_budget.maxC", max_search)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise AttackProtocolError(f"{name} must be a positive integer, got {value!r}.")
    if isinstance(max_ids, bool) or not isinstance(max_ids, int) or max_ids == 0 or max_ids < -1:
        raise AttackProtocolError("attack_budget.max_ids must be -1 or a positive integer.")
    if not (l1 > 0.0 and l2 > 0.0):
        raise AttackProtocolError("Locked l1_span_thresh and l2_span_thresh must both be positive.")
    # The existing DAGER configuration calls this bounded work unit ``parallel``.
    # It is used as both the chunked vocabulary scan and decoder forward batch,
    # so the scan never materializes the complete vocabulary in FP32.
    return NoneAttackControls(
        l1_span_threshold=l1,
        l2_span_threshold=l2,
        rank_tolerance=rank_tolerance,
        rank_cutoff=rank_cutoff_value,
        vocab_chunk_size=parallel,
        decode_batch_size=parallel,
        max_search_candidates=max_search,
        max_candidate_ids=max_ids,
        max_sequence_length=config.max_length,
    )


def registered_head_seed(config: ExperimentConfig, *, stage: str, requested_seed: int) -> None:
    if stage == "calibration":
        permitted = (config.calibration_head_seed,)
    elif stage == "smoke":
        permitted = (config.smoke_head_seed,)
    elif stage == "final":
        permitted = config.final_head_seeds
    else:
        raise AttackProtocolError(f"Unknown preregistration stage {stage!r}.")
    if requested_seed not in permitted:
        raise AttackProtocolError(
            f"head_seed={requested_seed} is not registered for stage={stage}; permitted values are {list(permitted)}."
        )


def load_registered_sample(
    *, config: ExperimentConfig, stage: str, sample_key: str
) -> RegisteredAttackSample:
    """Select exactly one sample from the immutable expected stage manifest."""
    if stage not in ("calibration", "smoke", "final"):
        raise AttackProtocolError(f"stage must be calibration, smoke, or final; got {stage!r}.")
    if not isinstance(sample_key, str) or len(sample_key) != 64:
        raise AttackProtocolError("sample_key must be one 64-character preregistered SHA256 string.")
    manifests = config.project_root / "manifests"
    preregistration = _read_json(manifests / "preregistration.json")
    if preregistration.get("config_sha256") != config.config_sha256:
        raise AttackProtocolError(
            "Current experiment.json SHA256 differs from preregistration.json; attack configuration is not locked."
        )
    preregistration_sha256 = preregistration.get("preregistration_sha256")
    stage_hashes = preregistration.get("sample_list_sha256")
    if not isinstance(preregistration_sha256, str) or not isinstance(stage_hashes, Mapping):
        raise AttackProtocolError("Malformed preregistration manifest identity fields.")
    expected_stage_hash = stage_hashes.get(stage)
    if not isinstance(expected_stage_hash, str):
        raise AttackProtocolError(f"preregistration.json lacks stage hash for {stage}.")
    records = _read_jsonl(manifests / f"{stage}.jsonl")
    samples: list[Mapping[str, Any]] = []
    selected: Mapping[str, Any] | None = None
    for record in records:
        if record.get("record_type") != "preregistered_sst2_validation_sample":
            raise AttackProtocolError("Stage JSONL contains an unexpected record type.")
        if record.get("stage") != stage or record.get("preregistration_sha256") != preregistration_sha256:
            raise AttackProtocolError("Stage JSONL does not match preregistration identity/stage.")
        if record.get("stage_sample_list_sha256") != expected_stage_hash:
            raise AttackProtocolError("Stage JSONL does not match the preregistered sample-list hash.")
        sample = record.get("sample")
        if not isinstance(sample, Mapping):
            raise AttackProtocolError("Stage JSONL record lacks its sample object.")
        samples.append(sample)
        if sample.get("sample_key") == sample_key:
            if selected is not None:
                raise AttackProtocolError("Stage JSONL contains the requested sample_key more than once.")
            selected = sample
    if hash_sample_list(samples) != expected_stage_hash:
        raise AttackProtocolError("Stage JSONL samples no longer hash to the preregistered sample list.")
    if selected is None:
        raise AttackProtocolError(f"sample_key={sample_key} is not registered in the {stage} manifest.")
    tokenization = selected.get("tokenization")
    if not isinstance(tokenization, Mapping):
        raise AttackProtocolError("Selected sample lacks immutable tokenization metadata.")
    values = {
        "original_index": selected.get("original_index"),
        "sentence": selected.get("sentence"),
        "label": selected.get("label"),
        "input_ids": tokenization.get("input_ids"),
        "eos_token_id": tokenization.get("eos_token_id"),
    }
    if isinstance(values["original_index"], bool) or not isinstance(values["original_index"], int):
        raise AttackProtocolError("Selected sample has invalid original_index.")
    if not isinstance(values["sentence"], str) or not values["sentence"].strip():
        raise AttackProtocolError("Selected sample has invalid sentence.")
    if isinstance(values["label"], bool) or values["label"] not in (0, 1):
        raise AttackProtocolError("Selected sample has invalid binary label.")
    input_ids = values["input_ids"]
    if not isinstance(input_ids, list) or not input_ids or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in input_ids
    ):
        raise AttackProtocolError("Selected sample has invalid tokenization.input_ids.")
    eos_token_id = values["eos_token_id"]
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int) or input_ids[-1] != eos_token_id:
        raise AttackProtocolError("Selected sample must retain its explicit EOS as the final token.")
    return RegisteredAttackSample(
        stage=stage,
        preregistration_sha256=preregistration_sha256,
        sample_key=sample_key,
        original_index=values["original_index"],
        sentence=values["sentence"],
        label=values["label"],
        input_ids=tuple(input_ids),
        eos_token_id=eos_token_id,
    )


def none_only_attack_metadata() -> dict[str, str]:
    """Return the fixed attack/defense declaration for every JSONL row."""
    return {
        "attack_name": ATTACK_NAME,
        "defense": "none",
        "defense_awareness": "defense_unaware",
    }
