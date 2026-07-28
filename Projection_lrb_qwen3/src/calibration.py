"""Stage-5 Qwen3 none-only calibration protocol runner implementation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

from .config import ExperimentConfig
from .config_selection import (
    SELECTION_RULE, TIE_BREAK_RULE, frozen_attack_config_document,
    select_calibration_configuration, write_or_verify_frozen_attack_config,
)
from .dager_qwen3.calibration_grid_control import (
    CalibrationGridControlError, candidate_parameters_from_grid,
    verify_calibration_grid_control, verify_tau1_reference,
)
from .dager_qwen3.none_attack_core import NoneAttackCoreControls, execute_none_only_dager
from .hashing import hash_directory_contents, hash_file_map, hash_sample_list, sha256_file, sha256_json
from .result_schema import ResultSchemaError, write_or_verify_json, write_or_verify_jsonl


class CalibrationError(RuntimeError):
    """Raised when Stage-5 calibration violates its frozen protocol boundary."""


@dataclass(frozen=True)
class CalibrationSample:
    sample_key: str
    original_index: int
    sentence: str
    label: int
    input_ids: tuple[int, ...]
    eos_token_id: int


@dataclass(frozen=True)
class CalibrationManifest:
    path: Path
    sha256: str
    preregistration_sha256: str
    sample_list_sha256: str
    samples: tuple[CalibrationSample, ...]


@dataclass(frozen=True)
class CalibrationRunContext:
    config: ExperimentConfig
    manifest: CalibrationManifest
    sample: CalibrationSample
    parameters: Mapping[str, Any]
    candidate_id: str
    tau1_control_identity_sha256: str
    grid_control_identity_sha256: str
    device: str
    dtype: str


CalibrationExecutor = Callable[[CalibrationRunContext, Any], Mapping[str, Any]]


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise CalibrationError(f"Unable to read calibration manifest {path}: {error}") from error
    rows: list[Mapping[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise CalibrationError(f"Invalid JSONL at {path}:{number}: {error}") from error
        if not isinstance(row, Mapping):
            raise CalibrationError("Calibration manifest rows must be objects.")
        rows.append(row)
    if not rows:
        raise CalibrationError("Calibration manifest is empty.")
    return rows


def load_calibration_manifest(path: Path, *, expected_path: Path, expected_preregistration_sha256: str, expected_sample_list_sha256: str) -> CalibrationManifest:
    if path.resolve() != expected_path.resolve() or path.name != "calibration.jsonl":
        raise CalibrationError("Stage-5 runner accepts only the registered calibration.jsonl.")
    rows = _read_jsonl(path)
    samples: list[CalibrationSample] = []
    stored: list[Mapping[str, Any]] = []
    for row in rows:
        if row.get("record_type") != "preregistered_sst2_validation_sample" or row.get("stage") != "calibration":
            raise CalibrationError("Calibration manifest has an invalid record type/stage.")
        if row.get("preregistration_sha256") != expected_preregistration_sha256 or row.get("stage_sample_list_sha256") != expected_sample_list_sha256:
            raise CalibrationError("Calibration manifest does not match frozen control protocol hashes.")
        sample = row.get("sample")
        tokenization = sample.get("tokenization") if isinstance(sample, Mapping) else None
        if not isinstance(sample, Mapping) or not isinstance(tokenization, Mapping):
            raise CalibrationError("Calibration sample lacks immutable tokenization.")
        values = (sample.get("sample_key"), sample.get("original_index"), sample.get("sentence"), sample.get("label"), tokenization.get("input_ids"), tokenization.get("eos_token_id"))
        if (not isinstance(values[0], str) or not isinstance(values[1], int) or not isinstance(values[2], str) or values[3] not in (0, 1) or not isinstance(values[4], list) or not isinstance(values[5], int) or not values[4] or values[4][-1] != values[5]):
            raise CalibrationError("Calibration sample has invalid immutable fields.")
        samples.append(CalibrationSample(values[0], values[1], values[2], values[3], tuple(values[4]), values[5]))
        stored.append(sample)
    if len(samples) != 20 or len({sample.sample_key for sample in samples}) != 20:
        raise CalibrationError("Formal Stage-5 calibration requires exactly 20 unique samples.")
    if hash_sample_list(stored) != expected_sample_list_sha256:
        raise CalibrationError("Calibration manifest sample-list hash is invalid.")
    return CalibrationManifest(path.resolve(), sha256_file(path), expected_preregistration_sha256, expected_sample_list_sha256, tuple(samples))


def _git_commit(repository_root: Path) -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository_root, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise CalibrationError("Unable to resolve git commit.") from error
    return result.stdout.strip()


def _record_identity(context: CalibrationRunContext) -> str:
    return sha256_json({"protocol": "qwen3_stage5_none_calibration_v1", "sample_key": context.sample.sample_key, "candidate_id": context.candidate_id, "head_seed": context.config.calibration_head_seed, "dtype": context.dtype, "tau1_control": context.tau1_control_identity_sha256, "grid_control": context.grid_control_identity_sha256})


def _controls(parameters: Mapping[str, Any], max_length: int) -> NoneAttackCoreControls:
    try:
        return NoneAttackCoreControls(
            tau1=float(parameters["tau1"]), tau2=float(parameters["tau2"]),
            rank_tolerance=float(parameters["numerical_rank_threshold"]), rank_cutoff=int(parameters["rank_cutoff"]),
            max_search_candidates=int(parameters["search_budget"]["maxC"]), max_candidate_ids=int(parameters["candidate_budget"]["max_ids"]),
            parallel=int(parameters["search_budget"]["parallel"]), max_sequence_length=max_length,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CalibrationError("Grid candidate parameters cannot form complete DAGER controls.") from error


def _default_executor(context: CalibrationRunContext, rouge_backend: Any) -> Mapping[str, Any]:
    return execute_none_only_dager(
        model_path=context.config.model_path, sample=context.sample,
        controls=_controls(context.parameters, context.config.max_length), head_seed=context.config.calibration_head_seed,
        device=context.device, dtype=context.dtype, rouge_backend=rouge_backend,
    )


def _path_for(output_root: Path, context: CalibrationRunContext) -> Path:
    return output_root / "runs" / context.candidate_id / f"{context.sample.sample_key}.jsonl"


def _execute_or_verify(*, context: CalibrationRunContext, output_root: Path, executor: CalibrationExecutor, rouge_backend: Any) -> Mapping[str, Any]:
    path = _path_for(output_root, context)
    if path.exists():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            row = json.loads(lines[0]) if len(lines) == 1 else None
        except (OSError, json.JSONDecodeError) as error:
            raise CalibrationError(f"Unable to verify existing calibration result {path}: {error}") from error
        if not isinstance(row, Mapping) or row.get("result_identity_sha256") != _record_identity(context):
            raise CalibrationError("Existing calibration result conflicts with this immutable candidate/sample request.")
        return row
    started = perf_counter()
    base: dict[str, Any] = {
        "schema_version": 1, "record_type": "qwen3_stage5_calibration_result", "result_identity_sha256": _record_identity(context),
        "stage": "calibration", "defense": "none", "defense_awareness": "defense_unaware",
        "preregistration_sha256": context.manifest.preregistration_sha256, "calibration_sample_list_sha256": context.manifest.sample_list_sha256,
        "sample_key": context.sample.sample_key, "original_index": context.sample.original_index,
        "head_seed": context.config.calibration_head_seed, "dtype": context.dtype,
        "candidate_id": context.candidate_id, "parameters": dict(context.parameters),
        "fixed_tau1_control_identity_sha256": context.tau1_control_identity_sha256,
        "calibration_grid_control_identity_sha256": context.grid_control_identity_sha256,
    }
    try:
        core = dict(executor(context, rouge_backend))
        status = core.get("status")
        if status not in {"ok", "search_budget_exhausted"}:
            raise CalibrationError("Shared attack core returned an invalid status.")
        base.update(core)
        base["result_status"] = status
        base["evaluated_prefix_cost"] = core["search_budget"]["evaluated_prefix_count"]
    except Exception as error:
        base.update({"result_status": "error", "error_type": type(error).__name__, "error": str(error), "evaluated_prefix_cost": 0})
    base["attack_time_seconds"] = float(base.get("attack_time_seconds", perf_counter() - started))
    try:
        write_or_verify_jsonl(path, [base])
    except ResultSchemaError as error:
        raise CalibrationError(f"Unable to write immutable calibration result {path}: {error}") from error
    return base


def _model_tokenizer_identity(config: ExperimentConfig) -> dict[str, Any]:
    model_sha, _ = hash_directory_contents(config.model_path)
    tokenizer_files = [path for path in config.model_path.rglob("*") if path.is_file() and path.name in {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}]
    return {"model_path": config.model_path.relative_to(config.repository_root).as_posix(), "model_directory_sha256": model_sha, "tokenizer_files_sha256": {} if not tokenizer_files else hash_file_map(tokenizer_files, base_dir=config.model_path)}


def run_calibration(*, config: ExperimentConfig, manifest_path: Path, tau1_control_path: Path, calibration_grid_control_path: Path, output_root: Path, device: str, dtype: str, plan_only: bool = False, executor: CalibrationExecutor | None = None) -> dict[str, Any]:
    """Validate controls before ROUGE/model/CUDA, then run the grid or print its plan."""
    try:
        tau1 = verify_tau1_reference(project_root=config.project_root, control_path=tau1_control_path)
        grid = verify_calibration_grid_control(project_root=config.project_root, control_path=calibration_grid_control_path)
    except CalibrationGridControlError as error:
        raise CalibrationError(str(error)) from error
    if grid["frozen_tau1_control_identity_sha256"] != tau1["frozen_control_identity_sha256"] or grid["frozen_tau1_control_file_sha256"] != sha256_file(tau1_control_path):
        raise CalibrationError("tau1 control identity/file hash does not match calibration-grid control.")
    manifest = load_calibration_manifest(manifest_path, expected_path=config.project_root / "manifests" / "calibration.jsonl", expected_preregistration_sha256=grid["preregistration_sha256"], expected_sample_list_sha256=grid["calibration_sample_list_sha256"])
    candidates = candidate_parameters_from_grid(grid)
    if len(candidates) < 2:
        raise CalibrationError("Formal calibration requires grid candidate_count >= 2.")
    candidate_ids = [sha256_json(candidate) for candidate in candidates]
    plan = {"status": "planned", "defense": "none", "candidate_count": len(candidates), "sample_count": len(manifest.samples), "total_runs": len(candidates) * len(manifest.samples), "candidates": [{"candidate_id": identifier, "parameters": candidate} for identifier, candidate in zip(candidate_ids, candidates)]}
    if plan_only:
        return plan
    expected_output = (config.project_root / "outputs" / "calibration").resolve()
    if output_root.resolve() != expected_output:
        raise CalibrationError("Stage-5 calibration outputs must be exactly outputs/calibration.")
    if executor is None:
        from .dager_qwen3.metrics import preflight_legacy_dager_rouge_backend
        rouge_backend = preflight_legacy_dager_rouge_backend()
        executor = _default_executor
    else:
        rouge_backend = None
    rows: list[Mapping[str, Any]] = []
    for candidate, identifier in zip(candidates, candidate_ids):
        for sample in manifest.samples:
            context = CalibrationRunContext(config, manifest, sample, candidate, identifier, tau1["frozen_control_identity_sha256"], grid["identity_sha256"], device, dtype)
            rows.append(_execute_or_verify(context=context, output_root=expected_output, executor=executor, rouge_backend=rouge_backend))
    all_results = expected_output / "all_results.jsonl"
    write_or_verify_jsonl(all_results, rows)
    selection = select_calibration_configuration(rows, expected_sample_keys=[sample.sample_key for sample in manifest.samples])
    summary: dict[str, Any] = {**plan, "status": "ok" if selection.selected is not None else "failed", "selection_rule": SELECTION_RULE, "tie_break_rule": TIE_BREAK_RULE, "candidate_summaries": [item.as_json() for item in selection.candidates], "selected_candidate_id": None if selection.selected is None else selection.selected.candidate_id, "all_results_sha256": sha256_file(all_results)}
    summary["summary_identity_sha256"] = sha256_json(summary)
    write_or_verify_json(expected_output / "summary.json", summary, identity_key="summary_identity_sha256")
    frozen_path = config.project_root / "manifests" / "frozen_attack_config.json"
    if selection.selected is None:
        return {**summary, "frozen_attack_config_path": None}
    selected_files = []
    for sample in manifest.samples:
        path = _path_for(expected_output, CalibrationRunContext(config, manifest, sample, selection.selected.parameters, selection.selected.candidate_id, tau1["frozen_control_identity_sha256"], grid["identity_sha256"], device, dtype))
        selected_files.append({"sample_key": sample.sample_key, "path": path.relative_to(config.repository_root).as_posix(), "sha256": sha256_file(path)})
    document = frozen_attack_config_document(selected=selection.selected, tau1_control_identity_sha256=tau1["frozen_control_identity_sha256"], calibration_grid_control_identity_sha256=grid["identity_sha256"], selected_result_files=selected_files, all_results_sha256=summary["all_results_sha256"], preregistration_sha256=manifest.preregistration_sha256, calibration_sample_list_sha256=manifest.sample_list_sha256, model_tokenizer_identity=_model_tokenizer_identity(config), head_seed=config.calibration_head_seed, git_commit=_git_commit(config.repository_root))
    write_or_verify_frozen_attack_config(frozen_path, document)
    return {**summary, "frozen_attack_config_path": frozen_path}
