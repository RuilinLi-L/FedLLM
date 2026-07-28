"""None-only, manifest-isolated Qwen3 DAGER calibration execution.

The calibration runner is the only component that enumerates candidate attack
controls.  It reads the calibration JSONL and ``experiment.json`` only; in
particular it does not inspect smoke/final manifests, text, labels, or results.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
import math
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

from .config import ExperimentConfig
from .config_selection import (
    SELECTION_RULE,
    frozen_attack_config_document,
    select_calibration_configuration,
    write_or_verify_frozen_attack_config,
)
from .hashing import (
    canonical_json_bytes,
    hash_directory_contents,
    hash_sample_list,
    sha256_file,
    sha256_json,
)
from .result_schema import ResultSchemaError, write_or_verify_json, write_or_verify_jsonl


class CalibrationError(RuntimeError):
    """Raised when isolated none-only calibration cannot be executed safely."""


@dataclass(frozen=True)
class CalibrationSample:
    """One complete immutable calibration row, read solely from calibration.jsonl."""

    sample_key: str
    original_index: int
    sentence: str
    label: int
    input_ids: tuple[int, ...]
    eos_token_id: int
    preregistration_sha256: str


@dataclass(frozen=True)
class CalibrationManifest:
    """Validated content and byte hash of the sole permissible stage manifest."""

    path: Path
    sha256: str
    samples: tuple[CalibrationSample, ...]
    preregistration_sha256: str
    stage_sample_list_sha256: str


@dataclass(frozen=True)
class CandidateParameters:
    """One cartesian-product point copied exactly from experiment.json."""

    tau1: float
    tau2: float
    numerical_rank_threshold: float
    rank_cutoff: int
    candidate_budget: int
    search_budget: int
    parallel: int

    def as_json(self) -> dict[str, Any]:
        return {
            "candidate_budget": {"max_ids": self.candidate_budget},
            "numerical_rank_threshold": self.numerical_rank_threshold,
            "rank_cutoff": self.rank_cutoff,
            "search_budget": {"maxC": self.search_budget, "parallel": self.parallel},
            "tau1": self.tau1,
            "tau2": self.tau2,
        }

    @property
    def candidate_id(self) -> str:
        return sha256_json(self.as_json())


@dataclass(frozen=True)
class CalibrationRunContext:
    """Inputs passed to the model-backed executor for one candidate/sample pair."""

    config: ExperimentConfig
    manifest: CalibrationManifest
    sample: CalibrationSample
    parameters: CandidateParameters
    head_seed: int
    device: str
    dtype: str


CalibrationExecutor = Callable[[CalibrationRunContext], Mapping[str, Any]]


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise CalibrationError(f"Unable to read calibration manifest {path}: {error}") from error
    if not lines:
        raise CalibrationError("calibration.jsonl must contain at least one preregistered sample.")
    records: list[Mapping[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise CalibrationError(f"Invalid calibration JSONL at {path}:{number}: {error}") from error
        if not isinstance(record, Mapping):
            raise CalibrationError(f"Calibration JSONL row {number} must be one object.")
        records.append(record)
    return records


def _require_sha256(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise CalibrationError(f"{description} must be one 64-character SHA256 string.")
    try:
        int(value, 16)
    except ValueError as error:
        raise CalibrationError(f"{description} must be hexadecimal SHA256.") from error
    return value


def _sample_from_record(record: Mapping[str, Any], *, position: int) -> CalibrationSample:
    if record.get("record_type") != "preregistered_sst2_validation_sample":
        raise CalibrationError(f"Calibration row {position} has an unexpected record_type.")
    if record.get("stage") != "calibration":
        raise CalibrationError("Calibration runner refuses every manifest row whose stage is not calibration.")
    preregistration_sha256 = _require_sha256(
        record.get("preregistration_sha256"), description=f"calibration row {position} preregistration_sha256"
    )
    sample = record.get("sample")
    if not isinstance(sample, Mapping):
        raise CalibrationError(f"Calibration row {position} lacks its immutable sample object.")
    tokenization = sample.get("tokenization")
    if not isinstance(tokenization, Mapping):
        raise CalibrationError(f"Calibration row {position} lacks immutable tokenization.")
    sample_key = sample.get("sample_key")
    original_index = sample.get("original_index")
    sentence = sample.get("sentence")
    label = sample.get("label")
    input_ids = tokenization.get("input_ids")
    eos_token_id = tokenization.get("eos_token_id")
    if not isinstance(sample_key, str) or len(sample_key) != 64:
        raise CalibrationError(f"Calibration row {position} has an invalid sample_key.")
    if isinstance(original_index, bool) or not isinstance(original_index, int):
        raise CalibrationError(f"Calibration row {position} has an invalid original_index.")
    if not isinstance(sentence, str) or not sentence.strip():
        raise CalibrationError(f"Calibration row {position} has an invalid sentence.")
    if isinstance(label, bool) or label not in (0, 1):
        raise CalibrationError(f"Calibration row {position} has an invalid SST-2 label.")
    if not isinstance(input_ids, list) or not input_ids or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in input_ids
    ):
        raise CalibrationError(f"Calibration row {position} has invalid tokenization.input_ids.")
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int) or input_ids[-1] != eos_token_id:
        raise CalibrationError(f"Calibration row {position} must end its immutable input_ids in EOS.")
    return CalibrationSample(
        sample_key=sample_key,
        original_index=original_index,
        sentence=sentence,
        label=label,
        input_ids=tuple(input_ids),
        eos_token_id=eos_token_id,
        preregistration_sha256=preregistration_sha256,
    )


def load_calibration_manifest(path: Path, *, expected_path: Path | None = None) -> CalibrationManifest:
    """Read and validate precisely ``calibration.jsonl``, with no other stage access."""
    resolved = path.resolve()
    if expected_path is not None and resolved != expected_path.resolve():
        raise CalibrationError(
            f"Calibration runner accepts only {expected_path.resolve().name}, not {resolved.name}."
        )
    if resolved.name != "calibration.jsonl":
        raise CalibrationError("Calibration runner accepts only a file named calibration.jsonl.")
    records = _read_jsonl(resolved)
    samples = tuple(_sample_from_record(record, position=index) for index, record in enumerate(records, start=1))
    if len({sample.sample_key for sample in samples}) != len(samples):
        raise CalibrationError("calibration.jsonl must not contain duplicate sample keys.")
    preregistration_ids = {sample.preregistration_sha256 for sample in samples}
    if len(preregistration_ids) != 1:
        raise CalibrationError("calibration.jsonl must have one common preregistration_sha256.")
    stage_hashes = {_require_sha256(row.get("stage_sample_list_sha256"), description="stage_sample_list_sha256") for row in records}
    if len(stage_hashes) != 1:
        raise CalibrationError("calibration.jsonl must have one common stage sample-list hash.")
    stored_samples = [record["sample"] for record in records]
    if hash_sample_list(stored_samples) != next(iter(stage_hashes)):
        raise CalibrationError("calibration.jsonl samples do not match their declared stage sample-list hash.")
    return CalibrationManifest(
        path=resolved,
        sha256=sha256_file(resolved),
        samples=samples,
        preregistration_sha256=next(iter(preregistration_ids)),
        stage_sample_list_sha256=next(iter(stage_hashes)),
    )


def _positive_float_grid(grid: Mapping[str, Any], key: str) -> tuple[float, ...]:
    values = grid.get(key)
    if not isinstance(values, list) or not values:
        raise CalibrationError(f"experiment.json calibration_parameter_grid.{key} must be one non-empty list.")
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CalibrationError(f"Grid {key} contains a non-numeric value: {value!r}.")
        converted = float(value)
        if not math.isfinite(converted) or converted <= 0.0:
            raise CalibrationError(f"Grid {key} values must be finite and positive.")
        result.append(converted)
    if len(set(result)) != len(result):
        raise CalibrationError(f"Grid {key} must not contain duplicate values.")
    return tuple(result)


def _nonnegative_int_grid(grid: Mapping[str, Any], key: str) -> tuple[int, ...]:
    values = grid.get(key)
    if not isinstance(values, list) or not values:
        raise CalibrationError(f"experiment.json calibration_parameter_grid.{key} must be one non-empty list.")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CalibrationError(f"Grid {key} values must be non-negative integers.")
        result.append(value)
    if len(set(result)) != len(result):
        raise CalibrationError(f"Grid {key} must not contain duplicate values.")
    return tuple(result)


def _budget_grid(budget: Mapping[str, Any], key: str, *, allow_minus_one: bool = False) -> tuple[int, ...]:
    configured = budget.get(key)
    values = configured if isinstance(configured, list) else [configured]
    if not values:
        raise CalibrationError(f"experiment.json attack_budget.{key} must have at least one value.")
    result: list[int] = []
    for value in values:
        valid = isinstance(value, int) and not isinstance(value, bool) and (
            value > 0 or (allow_minus_one and value == -1)
        )
        if not valid:
            expectation = "-1 or a positive integer" if allow_minus_one else "a positive integer"
            raise CalibrationError(f"attack_budget.{key} must contain {expectation}, got {value!r}.")
        result.append(value)
    if len(set(result)) != len(result):
        raise CalibrationError(f"attack_budget.{key} must not contain duplicate values.")
    return tuple(result)


def candidate_grid_from_experiment(config: ExperimentConfig) -> tuple[CandidateParameters, ...]:
    """Construct exactly the declared grid; no code-side values are appended."""
    grid = config.calibration_parameter_grid
    budget = config.attack_budget
    if not isinstance(grid, Mapping) or not isinstance(budget, Mapping):
        raise CalibrationError("experiment.json must provide mapping calibration_parameter_grid and attack_budget.")
    tau1 = _positive_float_grid(grid, "l1_span_thresh")
    tau2 = _positive_float_grid(grid, "l2_span_thresh")
    rank_threshold = _positive_float_grid(grid, "rank_tol")
    rank_cutoff = _nonnegative_int_grid(grid, "rank_cutoff")
    max_ids = _budget_grid(budget, "max_ids", allow_minus_one=True)
    max_search = _budget_grid(budget, "maxC")
    parallel = _budget_grid(budget, "parallel")
    candidates = tuple(
        CandidateParameters(
            tau1=tau1_value,
            tau2=tau2_value,
            numerical_rank_threshold=rank_value,
            rank_cutoff=rank_cutoff_value,
            candidate_budget=max_ids_value,
            search_budget=max_search_value,
            parallel=parallel_value,
        )
        for (
            tau1_value,
            tau2_value,
            rank_value,
            rank_cutoff_value,
            max_ids_value,
            max_search_value,
            parallel_value,
        ) in product(tau1, tau2, rank_threshold, rank_cutoff, max_ids, max_search, parallel)
    )
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise CalibrationError("experiment.json produces duplicate candidate parameter combinations.")
    return tuple(sorted(candidates, key=lambda candidate: canonical_json_bytes(candidate.as_json())))


def candidate_grid_sha256(candidates: Sequence[CandidateParameters]) -> str:
    """Hash the full ordered grid that was actually evaluated."""
    if not candidates:
        raise CalibrationError("Cannot hash an empty calibration parameter grid.")
    return sha256_json([candidate.as_json() for candidate in candidates])


def _git_commit(repository_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise CalibrationError("Unable to resolve the code commit required for frozen_attack_config.json.") from error
    commit = completed.stdout.strip()
    if not commit:
        raise CalibrationError("git rev-parse HEAD returned an empty commit.")
    return commit


def _record_identity(context: CalibrationRunContext) -> str:
    return sha256_json(
        {
            "protocol": "qwen3_dager_calibration_v1",
            "defense": "none",
            "calibration_manifest_sha256": context.manifest.sha256,
            "sample_key": context.sample.sample_key,
            "candidate_id": context.parameters.candidate_id,
            "head_seed": context.head_seed,
            "dtype": context.dtype,
        }
    )


def _base_record(context: CalibrationRunContext, *, attack_time_seconds: float) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "record_type": "qwen3_dager_calibration_result",
        "result_identity_sha256": _record_identity(context),
        "stage": "calibration",
        "attack_name": "DAGER",
        "defense": "none",
        "defense_awareness": "defense_unaware",
        "calibration_manifest_sha256": context.manifest.sha256,
        "preregistration_sha256": context.manifest.preregistration_sha256,
        "sample_key": context.sample.sample_key,
        "original_index": context.sample.original_index,
        "head_seed": context.head_seed,
        "dtype": context.dtype,
        "candidate_id": context.parameters.candidate_id,
        "parameters": context.parameters.as_json(),
        "attack_time_seconds": attack_time_seconds,
    }


def _execute_qwen3_dager(context: CalibrationRunContext) -> Mapping[str, Any]:
    """Run exactly one standard, defense-unaware none-only DAGER candidate."""
    try:
        import torch

        from .dager_qwen3.candidate_provider import RoPECandidateProvider
        from .dager_qwen3.gradient_decomposition import (
            decompose_qwen3_qproj_gradient,
            shared_dager_rank_for_qwen3_qproj_gradients,
        )
        from .dager_qwen3.gradient_gate import decode_token_texts, diagnostic_thresholds
        from .dager_qwen3.layer1_filter import filter_qwen3_vocab_layer1
        from .dager_qwen3.layer2_decoder import (
            Layer2DecoderConfig,
            decode_qwen3_rope_prefixes,
            layer2_audit_json_fields,
        )
        from .dager_qwen3.metrics import compute_attack_metrics, preflight_legacy_dager_rouge_backend
        from .dager_qwen3.model_adapter import Qwen3RoPEDagerAdapter
        from .gradient_capture import build_canonical_gradient_manifest, capture_single_example_gradients
        from .qwen3_classifier import load_local_qwen3_sequence_classifier
        from .span_diagnostics import diagnose_two_q_projections
    except Exception as error:
        raise CalibrationError(f"Qwen3 calibration dependencies are unavailable: {error}") from error

    # The legacy ROUGE definition is preflighted before expensive model work;
    # there is intentionally no replacement metric or online fallback.
    rouge_backend = preflight_legacy_dager_rouge_backend()
    torch.manual_seed(context.head_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(context.head_seed)
    bundle: Any | None = None
    try:
        bundle = load_local_qwen3_sequence_classifier(
            context.config.model_path,
            head_seed=context.head_seed,
            device=context.device,
            dtype=context.dtype,
        )
        if getattr(bundle.tokenizer, "eos_token_id", None) != context.sample.eos_token_id:
            raise CalibrationError("Calibration manifest EOS id differs from the loaded Qwen3 tokenizer EOS id.")
        input_ids = torch.tensor([context.sample.input_ids], dtype=torch.long, device=bundle.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=bundle.device)
        labels = torch.tensor([context.sample.label], dtype=torch.long, device=bundle.device)
        capture_started = perf_counter()
        captured = capture_single_example_gradients(
            bundle.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        capture_seconds = perf_counter() - capture_started
        token_texts = decode_token_texts(bundle.tokenizer, context.sample.input_ids)
        diagnostic = diagnose_two_q_projections(
            q_inputs=captured.q_inputs,
            q_output_gradients=captured.q_output_gradients,
            q_gradients=captured.q_gradients,
            q_parameter_names=captured.q_parameter_names,
            token_ids=context.sample.input_ids,
            token_texts=token_texts,
            eos_token_id=context.sample.eos_token_id,
            **diagnostic_thresholds(context.dtype),
        )
        if diagnostic.get("passed") is not True:
            raise CalibrationError("Qwen3 gradient orientation diagnostic failed for this calibration sample.")
        attack_started = perf_counter()
        adapter = Qwen3RoPEDagerAdapter(bundle.model, bundle.tokenizer)
        shared_rank = shared_dager_rank_for_qwen3_qproj_gradients(
            captured.q_gradients,
            feature_dim=adapter.metadata.hidden_size,
            rank_tolerance=context.parameters.numerical_rank_threshold,
            rank_cutoff=context.parameters.rank_cutoff,
            decomposition_device=bundle.device,
        )
        q0_span = decompose_qwen3_qproj_gradient(
            captured.q_gradients[0],
            feature_dim=adapter.metadata.hidden_size,
            rank_tolerance=context.parameters.numerical_rank_threshold,
            rank_cutoff=context.parameters.rank_cutoff,
            decomposition_device=bundle.device,
            shared_truncated_rank=shared_rank.applied_shared_rank,
        )
        q1_span = decompose_qwen3_qproj_gradient(
            captured.q_gradients[1],
            feature_dim=adapter.metadata.hidden_size,
            rank_tolerance=context.parameters.numerical_rank_threshold,
            rank_cutoff=context.parameters.rank_cutoff,
            decomposition_device=bundle.device,
            shared_truncated_rank=shared_rank.applied_shared_rank,
        )
        layer1 = filter_qwen3_vocab_layer1(
            adapter=adapter,
            span=q0_span,
            threshold=context.parameters.tau1,
            vocab_chunk_size=context.parameters.parallel,
            distance_norm="l2",
        )
        candidates = RoPECandidateProvider.from_layer1_result(
            layer1,
            eos_token_id=context.sample.eos_token_id,
            max_ids=context.parameters.candidate_budget,
        )
        layer2 = decode_qwen3_rope_prefixes(
            adapter=adapter,
            span=q1_span,
            candidate_provider=candidates,
            config=Layer2DecoderConfig(
                max_sequence_length=context.config.max_length,
                threshold=context.parameters.tau2,
                distance_norm="l2",
                search_budget=context.parameters.search_budget,
                decode_batch_size=context.parameters.parallel,
            ),
        )
        attack_seconds = perf_counter() - attack_started
        metrics = compute_attack_metrics(
            tokenizer=bundle.tokenizer,
            ground_truth_token_ids=context.sample.input_ids,
            reconstructed_token_ids=layer2.selected_token_ids,
            eos_token_id=context.sample.eos_token_id,
            rouge_metric=rouge_backend.metric,
        )
        canonical = build_canonical_gradient_manifest(bundle.model)
        return {
            "result_status": "search_budget_exhausted" if layer2.search_budget_exhausted else "ok",
            "token_recovery": metrics.token_recovery,
            "exact_recovery": metrics.exact_recovery,
            "empty_reconstruction": metrics.empty_reconstruction,
            "rouge_1": metrics.rouge_1,
            "rouge_2": metrics.rouge_2,
            "reconstructed_token_ids": list(layer2.selected_token_ids),
            "reconstructed_text": metrics.reconstructed_text,
            "attack_time_seconds": attack_seconds,
            "gradient_capture_time_seconds": capture_seconds,
            "loss": captured.loss,
            "gpu_peak_memory_bytes": captured.gpu_peak_memory_bytes,
            "rank": {
                "definition": shared_rank.rank_definition,
                "q0_effective_rank": shared_rank.q0_effective_rank,
                "q1_effective_rank": shared_rank.q1_effective_rank,
                "requested_shared_rank": shared_rank.requested_shared_rank,
                "applied_shared_rank": shared_rank.applied_shared_rank,
            },
            "search": {
                "layer1_candidate_count": layer1.candidate_count,
                "layer1_decoder_candidate_count": len(candidates.token_ids),
                "evaluated_prefix_count": layer2.evaluated_prefix_count,
                "termination_reason": layer2.termination_reason,
                **layer2_audit_json_fields(layer2),
            },
            "canonical_gradient_summary": {
                "gradient_tensor_count": canonical["gradient_tensor_count"],
                "gradient_numel": canonical["gradient_numel"],
            },
            "gradient_diagnostic": diagnostic,
            "legacy_rouge_backend": rouge_backend.json_metadata(),
        }
    finally:
        if bundle is not None:
            del bundle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _candidate_sample_path(output_root: Path, context: CalibrationRunContext) -> Path:
    return output_root / "runs" / context.parameters.candidate_id / f"{context.sample.sample_key}.jsonl"


def _existing_or_execute(
    *,
    context: CalibrationRunContext,
    output_root: Path,
    executor: CalibrationExecutor,
) -> Mapping[str, Any]:
    path = _candidate_sample_path(output_root, context)
    if path.exists():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            if len(lines) != 1:
                raise CalibrationError("Recoverable calibration sample JSONL must contain exactly one record.")
            existing = json.loads(lines[0])
        except (OSError, json.JSONDecodeError) as error:
            raise CalibrationError(f"Unable to reuse prior calibration record {path}: {error}") from error
        if not isinstance(existing, Mapping) or existing.get("result_identity_sha256") != _record_identity(context):
            raise CalibrationError(f"Existing calibration record conflicts with the requested candidate/sample: {path}")
        return existing
    started = perf_counter()
    try:
        details = dict(executor(context))
        attack_seconds = details.pop("attack_time_seconds", perf_counter() - started)
        if isinstance(attack_seconds, bool) or not isinstance(attack_seconds, (int, float)):
            raise CalibrationError("Calibration executor returned an invalid attack_time_seconds.")
        record = _base_record(context, attack_time_seconds=float(attack_seconds))
        record.update(details)
        if record.get("result_status") not in {"ok", "search_budget_exhausted"}:
            raise CalibrationError("Calibration executor must return result_status=ok or search_budget_exhausted.")
    except Exception as error:
        record = _base_record(context, attack_time_seconds=perf_counter() - started)
        record.update(
            {
                "result_status": "error",
                "token_recovery": 0.0,
                "exact_recovery": False,
                "empty_reconstruction": True,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
    try:
        write_or_verify_jsonl(path, [record])
    except ResultSchemaError as error:
        raise CalibrationError(f"Unable to recoverably write calibration row {path}: {error}") from error
    return record


def run_calibration(
    *,
    config: ExperimentConfig,
    manifest_path: Path,
    output_root: Path,
    device: str,
    dtype: str,
    executor: CalibrationExecutor | None = None,
) -> dict[str, Any]:
    """Execute every configured candidate on every immutable calibration sample."""
    expected_manifest = config.project_root / "manifests" / "calibration.jsonl"
    manifest = load_calibration_manifest(manifest_path, expected_path=expected_manifest)
    candidates = candidate_grid_from_experiment(config)
    resolved_output = output_root.resolve()
    expected_output = (config.project_root / "outputs" / "calibration").resolve()
    if resolved_output != expected_output:
        raise CalibrationError(f"Calibration outputs must be exactly {expected_output}, got {resolved_output}.")
    chosen_executor = _execute_qwen3_dager if executor is None else executor
    rows: list[Mapping[str, Any]] = []
    for candidate in candidates:
        for sample in manifest.samples:
            context = CalibrationRunContext(
                config=config,
                manifest=manifest,
                sample=sample,
                parameters=candidate,
                head_seed=config.calibration_head_seed,
                device=device,
                dtype=dtype,
            )
            rows.append(_existing_or_execute(context=context, output_root=resolved_output, executor=chosen_executor))
    all_results_path = resolved_output / "all_results.jsonl"
    try:
        write_or_verify_jsonl(all_results_path, rows)
    except ResultSchemaError as error:
        raise CalibrationError(f"Unable to write all_results.jsonl: {error}") from error
    selection = select_calibration_configuration(
        rows,
        expected_sample_keys=[sample.sample_key for sample in manifest.samples],
    )
    grid_hash = candidate_grid_sha256(candidates)
    commit = _git_commit(config.repository_root)
    all_results_hash = sha256_file(all_results_path)
    selected_is_complete = selection.selected.failed_row_count == 0
    summary = {
        "schema_version": 1,
        "record_type": "qwen3_dager_calibration_summary",
        "status": "ok" if selected_is_complete else "incomplete",
        "defense": "none",
        "calibration_manifest": {
            "path": manifest.path.relative_to(config.repository_root).as_posix(),
            "sha256": manifest.sha256,
            "sample_count": len(manifest.samples),
            "stage_sample_list_sha256": manifest.stage_sample_list_sha256,
        },
        "head_seed": config.calibration_head_seed,
        "candidate_grid_sha256": grid_hash,
        "candidate_count": len(candidates),
        "selection_rule": SELECTION_RULE,
        "selected_candidate_id": selection.selected.candidate_id,
        "selected_parameters": dict(selection.selected.parameters),
        "freeze_eligibility": {
            "eligible": selected_is_complete,
            "reason": (
                "selected_candidate_has_complete_calibration_rows"
                if selected_is_complete
                else "selected_candidate_contains_failed_calibration_rows"
            ),
        },
        "candidate_summaries": [candidate.as_json() for candidate in selection.candidates],
        "all_results": {
            "path": all_results_path.relative_to(config.repository_root).as_posix(),
            "sha256": all_results_hash,
            "row_count": len(rows),
        },
        "code_commit": commit,
    }
    summary["summary_identity_sha256"] = sha256_json(
        {key: value for key, value in summary.items() if key != "summary_identity_sha256"}
    )
    try:
        write_or_verify_json(
            resolved_output / "summary.json",
            summary,
            identity_key="summary_identity_sha256",
        )
    except ResultSchemaError as error:
        raise CalibrationError(f"Unable to write calibration summary: {error}") from error
    frozen_path = config.project_root / "manifests" / "frozen_attack_config.json"
    if selected_is_complete:
        model_sha256, _model_files = hash_directory_contents(config.model_path)
        frozen = frozen_attack_config_document(
            selection=selection,
            calibration_manifest_sha256=manifest.sha256,
            model_sha256=model_sha256,
            head_seed=config.calibration_head_seed,
            candidate_grid_sha256=grid_hash,
            code_commit=commit,
            all_results_sha256=all_results_hash,
        )
        try:
            write_or_verify_frozen_attack_config(frozen_path, frozen)
        except Exception as error:
            raise CalibrationError(f"Unable to freeze selected calibration configuration: {error}") from error
    return {
        "status": "ok" if selected_is_complete else "incomplete",
        "defense": "none",
        "all_results_path": all_results_path,
        "summary_path": resolved_output / "summary.json",
        "frozen_attack_config_path": frozen_path if selected_is_complete else None,
        "selected_candidate_id": selection.selected.candidate_id,
        "selected_parameters": dict(selection.selected.parameters),
    }
