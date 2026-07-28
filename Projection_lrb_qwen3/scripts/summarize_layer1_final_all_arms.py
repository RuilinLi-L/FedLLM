#!/usr/bin/env python3
"""Read and strictly validate the Qwen3 Layer-1 final all-arm lattice."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
for import_root in (REPOSITORY_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from src.hashing import sha256_file, sha256_json
from src.result_schema import ResultSchemaError, write_or_verify_json


FINAL_HEAD_SEEDS = (101, 202, 303)
FINAL_SAMPLE_COUNT = 20
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "final" / "layer1_all_arms_v1"
SUMMARY_OUTPUT_PATH = OUTPUT_ROOT / "layer1_final_all_arms_summary.json"
FINAL_MANIFEST_PATH = PROJECT_ROOT / "manifests" / "final.jsonl"
INPUT_PATHS = {seed: OUTPUT_ROOT / f"seed{seed}" / "paired_final.jsonl" for seed in FINAL_HEAD_SEEDS}
PROTOCOL_FIELDS = (
    "task",
    "batch_size",
    "gradient_steps",
    "dtype",
    "tau1",
    "tau2",
    "frozen_tau1_control_identity_sha256",
    "canonical_q_proj_indices",
    "attack_semantics",
)


class Layer1FinalAllArmsSummaryError(RuntimeError):
    """Raised when the immutable final lattice is absent or inconsistent."""


@dataclass(frozen=True)
class ArmSpec:
    label: str
    defense: str
    parameter_name: str
    parameter_value: float | int | None

    def matches(self, record: Mapping[str, Any]) -> bool:
        return (
            record.get("defense") == self.defense
            and record.get("defense_param_name") == self.parameter_name
            and record.get("defense_param_value") == self.parameter_value
        )


FINAL_ARMS = (
    ArmSpec("none", "none", "none", None),
    ArmSpec("lrbprojonly@0.2", "lrbprojonly", "keep_ratio", 0.2),
    ArmSpec("lrbprojonly@0.5", "lrbprojonly", "keep_ratio", 0.5),
    ArmSpec("lrbprojonly@0.65", "lrbprojonly", "keep_ratio", 0.65),
    ArmSpec("topk@0.1", "topk", "defense_topk_ratio", 0.1),
    ArmSpec("topk@0.7", "topk", "defense_topk_ratio", 0.7),
    ArmSpec("topk@0.9", "topk", "defense_topk_ratio", 0.9),
    ArmSpec("compression@8", "compression", "defense_n_bits", 8),
    ArmSpec("compression@16", "compression", "defense_n_bits", 16),
    ArmSpec("compression@32", "compression", "defense_n_bits", 32),
    ArmSpec("noise@1e-6", "noise", "defense_noise", 1e-6),
)
ARM_LABELS = frozenset(arm.label for arm in FINAL_ARMS)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise Layer1FinalAllArmsSummaryError(f"Unable to read required JSONL {path}: {error}") from error
    if not lines:
        raise Layer1FinalAllArmsSummaryError(f"Required JSONL is empty: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise Layer1FinalAllArmsSummaryError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
        if not isinstance(record, dict):
            raise Layer1FinalAllArmsSummaryError(f"JSONL record at {path}:{line_number} must be an object.")
        records.append(record)
    return records


def _final_manifest_keys(manifest_path: Path) -> frozenset[str]:
    keys: list[str] = []
    for record in _read_jsonl(manifest_path):
        if record.get("record_type") != "preregistered_sst2_validation_sample" or record.get("stage") != "final":
            raise Layer1FinalAllArmsSummaryError("Final manifest has an unexpected record type or stage.")
        sample = record.get("sample")
        key = sample.get("sample_key") if isinstance(sample, Mapping) else None
        if not isinstance(key, str) or len(key) != 64:
            raise Layer1FinalAllArmsSummaryError("Final manifest contains an invalid sample_key.")
        keys.append(key)
    if len(keys) != FINAL_SAMPLE_COUNT or len(set(keys)) != FINAL_SAMPLE_COUNT:
        raise Layer1FinalAllArmsSummaryError("Final manifest must contain exactly 20 unique preregistered samples.")
    return frozenset(keys)


def _record_label(record: Mapping[str, Any], *, source_name: str) -> str:
    labels = [arm.label for arm in FINAL_ARMS if arm.matches(record)]
    if len(labels) != 1:
        raise Layer1FinalAllArmsSummaryError(
            f"{source_name} contains an unexpected defense arm: defense={record.get('defense')!r}, parameter={record.get('defense_param_value')!r}."
        )
    return labels[0]


def _validate_record(record: Mapping[str, Any], *, expected_seed: int, expected_keys: frozenset[str], source_name: str) -> tuple[str, str]:
    if record.get("record_type") != "qwen3_dager_layer1_final_all_arms":
        raise Layer1FinalAllArmsSummaryError(f"{source_name} has an unexpected record_type.")
    if record.get("result_status") != "ok":
        raise Layer1FinalAllArmsSummaryError(f"{source_name} has non-ok result_status={record.get('result_status')!r}.")
    if record.get("stage") != "final":
        raise Layer1FinalAllArmsSummaryError(f"{source_name} has stage={record.get('stage')!r}, expected 'final'.")
    if record.get("head_seed") != expected_seed or expected_seed not in FINAL_HEAD_SEEDS:
        raise Layer1FinalAllArmsSummaryError(f"{source_name} has invalid head_seed={record.get('head_seed')!r}.")
    sample_key = record.get("sample_key")
    if sample_key not in expected_keys:
        raise Layer1FinalAllArmsSummaryError(f"{source_name} has an unregistered final sample_key.")
    if record.get("attack_semantics") != "defense_unaware_observed_q_proj_only":
        raise Layer1FinalAllArmsSummaryError(f"{source_name} has incompatible attack semantics.")
    for field in PROTOCOL_FIELDS:
        if field not in record:
            raise Layer1FinalAllArmsSummaryError(f"{source_name} lacks protocol field {field!r}.")
    return sample_key, _record_label(record, source_name=source_name)


def _protocol_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    return {field: record[field] for field in PROTOCOL_FIELDS}


def _index_seed_records(
    records: Iterable[dict[str, Any]], *, expected_seed: int, expected_keys: frozenset[str], source_name: str
) -> dict[str, dict[str, dict[str, Any]]]:
    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        sample_key, label = _validate_record(record, expected_seed=expected_seed, expected_keys=expected_keys, source_name=source_name)
        by_arm = indexed.setdefault(sample_key, {})
        if label in by_arm:
            raise Layer1FinalAllArmsSummaryError(f"{source_name} repeats arm={label!r} for sample_key={sample_key}.")
        by_arm[label] = record
    if set(indexed) != set(expected_keys):
        raise Layer1FinalAllArmsSummaryError(f"{source_name} does not cover exactly the registered final samples.")
    if len(indexed) != FINAL_SAMPLE_COUNT:
        raise Layer1FinalAllArmsSummaryError(f"{source_name} must cover exactly 20 final samples.")
    for sample_key, by_arm in indexed.items():
        if set(by_arm) != ARM_LABELS:
            raise Layer1FinalAllArmsSummaryError(f"{source_name} sample_key={sample_key} does not have exactly the 11 fixed arms.")
        first = _protocol_identity(next(iter(by_arm.values())))
        for record in by_arm.values():
            if _protocol_identity(record) != first:
                differing = sorted(key for key in first if first.get(key) != _protocol_identity(record).get(key))
                raise Layer1FinalAllArmsSummaryError(f"{source_name} protocol mismatch for sample_key={sample_key}: {differing}.")
    if sum(len(by_arm) for by_arm in indexed.values()) != FINAL_SAMPLE_COUNT * len(FINAL_ARMS):
        raise Layer1FinalAllArmsSummaryError(f"{source_name} does not contain exactly 220 final records.")
    return indexed


def _number(record: Mapping[str, Any], field: str) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Layer1FinalAllArmsSummaryError(f"Record arm={record.get('defense')!r} lacks numeric {field!r}.")
    return float(value)


def _summarize_arm(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise Layer1FinalAllArmsSummaryError("Cannot summarize an empty fixed arm.")
    return {
        "n": len(records),
        "mean_candidate_count": sum(_number(record, "candidate_count") for record in records) / len(records),
        "mean_l1_membership": sum(_number(record, "legacy_l1_token_membership") for record in records) / len(records),
        "zero_candidate_samples": sum(_number(record, "candidate_count") == 0 for record in records),
        "layer2_survivor_samples": sum(_number(record, "layer_2_survivor_count") > 0 for record in records),
        "mean_token_recovery": sum(_number(record, "token_recovery") for record in records) / len(records),
    }


def build_layer1_final_all_arms_summary(
    *, input_paths: Mapping[int, Path] = INPUT_PATHS, manifest_path: Path = FINAL_MANIFEST_PATH
) -> dict[str, Any]:
    """Read only the three final JSONL inputs and validate their full lattice."""
    if set(input_paths) != set(FINAL_HEAD_SEEDS):
        raise Layer1FinalAllArmsSummaryError("Final summary requires exactly the registered seed101, seed202, and seed303 JSONL inputs.")
    expected_keys = _final_manifest_keys(manifest_path)
    indexed_by_seed = {
        seed: _index_seed_records(_read_jsonl(Path(input_paths[seed])), expected_seed=seed, expected_keys=expected_keys, source_name=f"seed{seed} final JSONL")
        for seed in FINAL_HEAD_SEEDS
    }
    grouped: dict[str, list[Mapping[str, Any]]] = {label: [] for label in ARM_LABELS}
    by_head_seed: dict[str, dict[str, dict[str, Any]]] = {label: {} for label in ARM_LABELS}
    for seed, indexed in indexed_by_seed.items():
        for label in ARM_LABELS:
            records = [indexed[sample_key][label] for sample_key in sorted(expected_keys)]
            grouped[label].extend(records)
            by_head_seed[label][str(seed)] = _summarize_arm(records)
    summary = {
        "schema_version": 1,
        "record_type": "qwen3_dager_layer1_final_all_arms_summary",
        "comparison_pairing": "same_capture_within_head_seed",
        "three_preregistered_fixed_head_seeds": list(FINAL_HEAD_SEEDS),
        "comparison_scope": "qwen3_1_7b_base_sst2_batch1_single_example_gradient_layer1_token_set_mechanism_final",
        "attack_semantics": "defense_unaware_observed_q_proj_only",
        "primary_metric_semantics": "legacy_l1_token_membership is the fraction of non-EOS real tokens covered by the Layer-1 candidate set; it is not ordered text recovery.",
        "diagnostic_fields": "layer_2_survivor_count and token_recovery are diagnostics only.",
        "sample_count_per_head_seed": FINAL_SAMPLE_COUNT,
        "source_files": {str(seed): {"path": str(input_paths[seed]), "sha256": sha256_file(Path(input_paths[seed]))} for seed in FINAL_HEAD_SEEDS},
        "final_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "defenses": {
            label: {**_summarize_arm(grouped[label]), "by_head_seed": by_head_seed[label]}
            for label in sorted(ARM_LABELS)
        },
    }
    summary["summary_identity_sha256"] = sha256_json(summary)
    return summary


def write_layer1_final_all_arms_summary(
    *, input_paths: Mapping[int, Path] = INPUT_PATHS, manifest_path: Path = FINAL_MANIFEST_PATH, output_path: Path = SUMMARY_OUTPUT_PATH
) -> dict[str, Any]:
    summary = build_layer1_final_all_arms_summary(input_paths=input_paths, manifest_path=manifest_path)
    try:
        write_or_verify_json(output_path, summary, identity_key="summary_identity_sha256")
    except ResultSchemaError as error:
        raise Layer1FinalAllArmsSummaryError(str(error)) from error
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only Qwen3 Layer-1 final all-arm summary; reads only final seed JSONL inputs and never reruns DAGER.")
    parser.parse_args()
    return argparse.Namespace()


def main() -> int:
    parse_args()
    try:
        print(json.dumps(write_layer1_final_all_arms_summary(), sort_keys=True))
    except Exception as error:
        print(json.dumps({"record_type": "qwen3_dager_layer1_final_all_arms_summary_error", "result_status": "error", "error_type": type(error).__name__, "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
