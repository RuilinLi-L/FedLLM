#!/usr/bin/env python3
"""Read-only Layer-1 token-set leakage summary for protocol-matched Qwen3 smokes."""

from __future__ import annotations

import argparse
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


LEGACY_INPUT_PATH = PROJECT_ROOT / "outputs" / "smoke" / "minimal_projonly_pair_l1metrics_v1" / "paired_smoke_all.jsonl"
BASELINE_INPUT_PATH = PROJECT_ROOT / "outputs" / "smoke" / "layer1_baselines_v1" / "paired_smoke_all.jsonl"
SUMMARY_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "smoke" / "layer1_baselines_v1" / "layer1_summary.json"
LEGACY_ARMS = frozenset(("none", "lrbprojonly"))
BASELINE_ARMS = frozenset(("topk", "compression", "noise"))
ALL_ARMS = LEGACY_ARMS | BASELINE_ARMS
PROTOCOL_FIELDS = (
    "task",
    "batch_size",
    "gradient_steps",
    "dtype",
    "head_seed",
    "tau1",
    "tau2",
    "frozen_tau1_control_identity_sha256",
    "canonical_q_proj_indices",
)


class Layer1SummaryError(RuntimeError):
    """Raised when protocol-matched cross-run comparison is not justified."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise Layer1SummaryError(f"Unable to read required input JSONL {path}: {error}") from error
    if not lines:
        raise Layer1SummaryError(f"Required input JSONL is empty: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise Layer1SummaryError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
        if not isinstance(record, dict):
            raise Layer1SummaryError(f"JSONL record at {path}:{line_number} must be an object.")
        records.append(record)
    return records


def _attack_semantics(record: Mapping[str, Any]) -> str:
    value = record.get("attack_semantics", record.get("defense_awareness"))
    if value != "defense_unaware_observed_q_proj_only":
        raise Layer1SummaryError(f"Record has incompatible attack semantics: {value!r}.")
    return value


def _validate_record(record: Mapping[str, Any], *, expected_arms: frozenset[str], source_name: str) -> tuple[str, str]:
    defense = record.get("defense")
    sample_key = record.get("sample_key")
    if defense not in expected_arms:
        raise Layer1SummaryError(f"{source_name} contains unexpected defense arm {defense!r}.")
    if not isinstance(sample_key, str) or len(sample_key) != 64:
        raise Layer1SummaryError(f"{source_name} contains an invalid sample_key.")
    if record.get("result_status") != "ok":
        raise Layer1SummaryError(f"{source_name} {defense}@{sample_key} has result_status={record.get('result_status')!r}.")
    for field in PROTOCOL_FIELDS:
        if field not in record:
            raise Layer1SummaryError(f"{source_name} {defense}@{sample_key} lacks protocol field {field!r}.")
    _attack_semantics(record)
    return sample_key, defense


def _index_records(records: Iterable[dict[str, Any]], *, expected_arms: frozenset[str], source_name: str) -> dict[str, dict[str, dict[str, Any]]]:
    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        sample_key, defense = _validate_record(record, expected_arms=expected_arms, source_name=source_name)
        arms = indexed.setdefault(sample_key, {})
        if defense in arms:
            raise Layer1SummaryError(f"{source_name} repeats defense={defense!r} for sample_key={sample_key}.")
        arms[defense] = record
    for sample_key, arms in indexed.items():
        if set(arms) != set(expected_arms):
            raise Layer1SummaryError(f"{source_name} sample_key={sample_key} has arms {sorted(arms)}, expected {sorted(expected_arms)}.")
    return indexed


def _protocol_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    identity = {field: record[field] for field in PROTOCOL_FIELDS}
    identity["attack_semantics"] = _attack_semantics(record)
    return identity


def _require_protocol_match(records: Iterable[Mapping[str, Any]], *, sample_key: str) -> None:
    records = tuple(records)
    expected = _protocol_identity(records[0])
    for record in records[1:]:
        actual = _protocol_identity(record)
        if actual != expected:
            differing = sorted(field for field in expected if expected.get(field) != actual.get(field))
            raise Layer1SummaryError(f"Protocol mismatch for sample_key={sample_key}: {differing}.")


def _number(record: Mapping[str, Any], field: str) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Layer1SummaryError(f"Record defense={record.get('defense')!r} lacks numeric {field!r}.")
    return float(value)


def _summarize_arm(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(records),
        "mean_candidate_count": sum(_number(record, "candidate_count") for record in records) / len(records),
        "mean_l1_membership": sum(_number(record, "legacy_l1_token_membership") for record in records) / len(records),
        "zero_candidate_samples": sum(_number(record, "candidate_count") == 0 for record in records),
        "layer2_survivor_samples": sum(_number(record, "layer_2_survivor_count") > 0 for record in records),
        "mean_token_recovery": sum(_number(record, "token_recovery") for record in records) / len(records),
    }


def build_layer1_summary(*, legacy_path: Path = LEGACY_INPUT_PATH, baseline_path: Path = BASELINE_INPUT_PATH) -> dict[str, Any]:
    """Validate and merge exactly five protocol-matched arms without running DAGER."""
    legacy = _index_records(_read_jsonl(legacy_path), expected_arms=LEGACY_ARMS, source_name="legacy JSONL")
    baseline = _index_records(_read_jsonl(baseline_path), expected_arms=BASELINE_ARMS, source_name="baseline JSONL")
    if set(legacy) != set(baseline):
        missing_legacy = sorted(set(baseline) - set(legacy))
        missing_baseline = sorted(set(legacy) - set(baseline))
        raise Layer1SummaryError(f"Sample coverage differs between inputs; missing legacy={missing_legacy}, missing baseline={missing_baseline}.")
    if len(legacy) != 5:
        raise Layer1SummaryError(f"Layer-1 baseline summary requires exactly five smoke samples, got {len(legacy)}.")
    grouped: dict[str, list[Mapping[str, Any]]] = {arm: [] for arm in sorted(ALL_ARMS)}
    for sample_key in sorted(legacy):
        merged = {**legacy[sample_key], **baseline[sample_key]}
        if set(merged) != ALL_ARMS:
            raise Layer1SummaryError(f"sample_key={sample_key} does not have exactly five expected arms.")
        _require_protocol_match(merged.values(), sample_key=sample_key)
        for arm, record in merged.items():
            grouped[arm].append(record)
    summary = {
        "schema_version": 1,
        "record_type": "qwen3_dager_layer1_baseline_smoke_summary",
        "comparison_pairing": "protocol_matched_cross_run",
        "comparison_scope": "qwen3_base_fixed_random_sst2_head_five_preregistered_smoke_samples",
        "attack_semantics": "defense_unaware_observed_q_proj_only",
        "primary_metric_semantics": "legacy_l1_token_membership is the fraction of real non-EOS tokens covered by the DAGER Layer-1 candidate set; it is token-set leakage, not ordered text recovery.",
        "diagnostic_fields": ["layer_2_survivor_count", "token_recovery", "termination_reason"],
        "sample_count": len(legacy),
        "source_files": {"legacy": str(legacy_path), "legacy_sha256": sha256_file(legacy_path), "baselines": str(baseline_path), "baselines_sha256": sha256_file(baseline_path)},
        "defenses": {arm: _summarize_arm(records) for arm, records in grouped.items()},
    }
    summary["summary_identity_sha256"] = sha256_json(summary)
    return summary


def write_layer1_summary(*, legacy_path: Path = LEGACY_INPUT_PATH, baseline_path: Path = BASELINE_INPUT_PATH, output_path: Path = SUMMARY_OUTPUT_PATH) -> dict[str, Any]:
    summary = build_layer1_summary(legacy_path=legacy_path, baseline_path=baseline_path)
    try:
        write_or_verify_json(output_path, summary, identity_key="summary_identity_sha256")
    except ResultSchemaError as error:
        raise Layer1SummaryError(str(error)) from error
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only fixed Qwen3 Layer-1 baseline smoke summary; does not rerun DAGER or change input JSONL files.")
    parser.parse_args()
    return argparse.Namespace()


def main() -> int:
    parse_args()
    try:
        print(json.dumps(write_layer1_summary(), sort_keys=True))
    except Exception as error:
        print(json.dumps({"record_type": "qwen3_dager_layer1_baseline_smoke_summary_error", "result_status": "error", "error_type": type(error).__name__, "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
