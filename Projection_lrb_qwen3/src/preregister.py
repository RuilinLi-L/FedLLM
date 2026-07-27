"""Deterministic SST-2 preregistration; intentionally no DAGER implementation."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from .config import ExperimentConfig
from .hashing import HashingError, hash_file_map, hash_sample_list, sample_key, sha256_json
from .result_schema import write_or_verify_json, write_or_verify_jsonl


class PreregistrationError(RuntimeError):
    """Raised for an invalid model, dataset, tokenizer, or protocol artifact."""


class TokenizerProtocol(Protocol):
    """Small tokenizer surface required for explicit EOS preregistration."""

    eos_token_id: int | None

    def __call__(self, text: str, **kwargs: Any) -> Mapping[str, Any]:
        """Tokenize one text sample."""


STAGE_SIZES: Mapping[str, int] = {"calibration": 20, "smoke": 5, "final": 20}
PROTOCOL_NAME = "qwen3_sst2_preregistration_v1"


def _require_model_dir(config: ExperimentConfig) -> None:
    if not config.model_path.is_dir():
        raise PreregistrationError(f"Configured Qwen3 model directory does not exist: {config.model_path}")
    config_file = config.model_path / "config.json"
    if not config_file.is_file():
        raise PreregistrationError(f"Qwen3 model is missing config.json: {config.model_path}")
    try:
        model_config = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreregistrationError(f"Unable to parse model config {config_file}: {error}") from error
    if model_config.get("model_type") != "qwen3":
        raise PreregistrationError(
            f"Expected a Qwen3 model (model_type='qwen3'), got {model_config.get('model_type')!r}."
        )


def load_qwen3_tokenizer(config: ExperimentConfig) -> TokenizerProtocol:
    """Load only the configured local Qwen3 tokenizer; never substitute another."""
    _require_model_dir(config)
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise PreregistrationError("transformers is required to load the local Qwen3 tokenizer.") from error
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(config.model_path),
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as error:  # Third-party loader errors have heterogeneous types.
        raise PreregistrationError(
            f"Unable to load the configured local Qwen3 tokenizer from {config.model_path}: {error}"
        ) from error
    if getattr(tokenizer, "eos_token_id", None) is None:
        raise PreregistrationError("Qwen3 tokenizer has no eos_token_id; explicit EOS is required.")
    return tokenizer


def load_official_sst2_validation(
    *,
    loader: Callable[..., Iterable[Mapping[str, Any]]] | None = None,
) -> list[Mapping[str, Any]]:
    """Load exactly the official GLUE SST-2 validation split."""
    if loader is None:
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise PreregistrationError("datasets is required to load GLUE SST-2 validation.") from error
        loader = load_dataset
    try:
        dataset = loader("glue", "sst2", split="validation")
    except Exception as error:  # Dataset backends may raise several exception classes.
        raise PreregistrationError(f"Unable to load official GLUE SST-2 validation split: {error}") from error
    return list(dataset)


def _extract_input_ids(tokenizer_output: Mapping[str, Any]) -> list[int]:
    if "input_ids" not in tokenizer_output:
        raise PreregistrationError("Tokenizer output lacks input_ids.")
    value = tokenizer_output["input_ids"]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PreregistrationError("Tokenizer input_ids must be a one-dimensional sequence of integers.")
    ids: list[int] = []
    for token in value:
        if isinstance(token, bool) or not isinstance(token, int):
            raise PreregistrationError(f"Tokenizer returned a non-integer token id: {token!r}")
        ids.append(token)
    return ids


def prepare_eligible_samples(
    rows: Iterable[Mapping[str, Any]],
    tokenizer: TokenizerProtocol,
    config: ExperimentConfig,
) -> list[dict[str, Any]]:
    """Tokenize, filter, key, and sort the complete official validation split."""
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
        raise PreregistrationError("Tokenizer eos_token_id must be one integer.")
    eligible: list[dict[str, Any]] = []
    for original_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise PreregistrationError(f"Dataset row {original_index} is not a mapping.")
        sentence = row.get("sentence")
        label = row.get("label")
        if not isinstance(sentence, str):
            raise PreregistrationError(f"Dataset row {original_index} has a non-string sentence.")
        if isinstance(label, bool) or not isinstance(label, int):
            raise PreregistrationError(f"Dataset row {original_index} has a non-integer label.")
        if not sentence.strip():
            continue
        try:
            output = tokenizer(
                sentence,
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
        except Exception as error:
            raise PreregistrationError(f"Tokenizer failed for SST-2 row {original_index}: {error}") from error
        if not isinstance(output, Mapping):
            raise PreregistrationError(f"Tokenizer returned a non-mapping result for row {original_index}.")
        untruncated_text_token_ids = _extract_input_ids(output)
        was_truncated = len(untruncated_text_token_ids) > config.max_length - 1
        text_token_ids = untruncated_text_token_ids[: config.max_length - 1]
        if len(text_token_ids) < config.min_effective_token_length:
            continue
        input_ids = [*text_token_ids, eos_token_id]
        if len(input_ids) > config.max_length:
            raise PreregistrationError(
                f"Internal truncation failure for row {original_index}: {len(input_ids)} > {config.max_length}."
            )
        eligible.append(
            {
                "original_index": original_index,
                "sentence": sentence,
                "label": label,
                "sample_key": sample_key(
                    original_index=original_index,
                    sentence=sentence,
                    label=label,
                ),
                "tokenization": {
                    "add_special_tokens": False,
                    "eos_token_id": eos_token_id,
                    "text_token_ids": text_token_ids,
                    "input_ids": input_ids,
                    "effective_token_length": len(text_token_ids),
                    "total_token_length": len(input_ids),
                    "was_truncated": was_truncated,
                },
            }
        )
    eligible.sort(key=lambda sample: sample["sample_key"])
    return eligible


def allocate_stages(eligible_samples: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Allocate the fixed, ordered 20/5/20 protocol slices without randomness."""
    required = sum(STAGE_SIZES.values())
    if len(eligible_samples) < required:
        raise PreregistrationError(
            f"Only {len(eligible_samples)} eligible SST-2 examples; protocol requires {required}."
        )
    stages: dict[str, list[dict[str, Any]]] = {}
    offset = 0
    for stage, count in STAGE_SIZES.items():
        stages[stage] = [dict(sample) for sample in eligible_samples[offset : offset + count]]
        offset += count
    keys = {stage: {str(sample["sample_key"]) for sample in samples} for stage, samples in stages.items()}
    overlaps = {
        "calibration_smoke": sorted(keys["calibration"] & keys["smoke"]),
        "calibration_final": sorted(keys["calibration"] & keys["final"]),
        "smoke_final": sorted(keys["smoke"] & keys["final"]),
    }
    if any(overlaps.values()):
        raise PreregistrationError(f"Internal stage overlap detected: {overlaps}")
    return stages


def _model_key_files(model_dir: Path) -> list[Path]:
    required = [model_dir / "config.json"]
    optional = [model_dir / "generation_config.json", model_dir / "model.safetensors.index.json"]
    files = [path for path in required if path.is_file()]
    if len(files) != len(required):
        raise PreregistrationError(f"Missing required Qwen3 model key file: {model_dir / 'config.json'}")
    files.extend(path for path in optional if path.is_file())
    weight_files = sorted(model_dir.glob("*.safetensors"), key=lambda path: path.name)
    if not weight_files:
        raise PreregistrationError(
            f"No safetensors weights found in configured Qwen3 directory: {model_dir}"
        )
    files.extend(weight_files)
    return files


def _tokenizer_key_files(model_dir: Path) -> list[Path]:
    required = [model_dir / "tokenizer.json", model_dir / "tokenizer_config.json"]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise PreregistrationError(f"Qwen3 tokenizer is missing required key files: {missing}")
    optional_names = ("special_tokens_map.json", "added_tokens.json", "merges.txt", "vocab.json")
    return [*required, *(model_dir / name for name in optional_names if (model_dir / name).is_file())]


def collect_model_and_tokenizer_hashes(config: ExperimentConfig) -> tuple[dict[str, str], str, dict[str, str]]:
    """Hash model and tokenizer key files, preserving exact file identities."""
    _require_model_dir(config)
    try:
        model_hashes = hash_file_map(_model_key_files(config.model_path), base_dir=config.model_path)
        tokenizer_hashes = hash_file_map(_tokenizer_key_files(config.model_path), base_dir=config.model_path)
    except HashingError as error:
        raise PreregistrationError(str(error)) from error
    return model_hashes, sha256_json(tokenizer_hashes), tokenizer_hashes


def runtime_versions() -> dict[str, str]:
    """Collect required runtime versions; missing packages are explicit errors."""
    versions = {"python": __import__("platform").python_version()}
    for package in ("torch", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise PreregistrationError(
                f"Required package {package!r} is not installed; version metadata is mandatory."
            ) from error
    return versions


def git_commit(repository_root: Path) -> str:
    """Return the exact current commit; a missing Git checkout is an error."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise PreregistrationError(
            f"Unable to resolve Git commit at {repository_root}: {error}"
        ) from error
    commit = completed.stdout.strip()
    if len(commit) != 40:
        raise PreregistrationError(f"Git returned an invalid commit identifier: {commit!r}")
    return commit


def intersection_check(stages: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Return explicit empty intersections for the three preregistered stages."""
    expected = set(STAGE_SIZES)
    if set(stages) != expected:
        raise PreregistrationError(f"Stage names must be exactly {sorted(expected)}, got {sorted(stages)}.")
    keys = {stage: {str(sample["sample_key"]) for sample in samples} for stage, samples in stages.items()}
    check = {
        "calibration_smoke": sorted(keys["calibration"] & keys["smoke"]),
        "calibration_final": sorted(keys["calibration"] & keys["final"]),
        "smoke_final": sorted(keys["smoke"] & keys["final"]),
    }
    check["all_disjoint"] = not any(check.values())
    return check


def build_preregistration_document(
    *,
    config: ExperimentConfig,
    stages: Mapping[str, Sequence[Mapping[str, Any]]],
    model_key_file_sha256: Mapping[str, str],
    tokenizer_sha256: str,
    tokenizer_key_file_sha256: Mapping[str, str],
    created_at: str,
    commit: str,
    versions: Mapping[str, str],
) -> dict[str, Any]:
    """Build the full manifest and its timestamp-independent protocol identity."""
    for stage, size in STAGE_SIZES.items():
        if len(stages.get(stage, ())) != size:
            raise PreregistrationError(f"Stage {stage!r} must contain exactly {size} samples.")
    stage_lists = {stage: [dict(sample) for sample in stages[stage]] for stage in STAGE_SIZES}
    stage_hashes = {stage: hash_sample_list(samples) for stage, samples in stage_lists.items()}
    intersections = intersection_check(stage_lists)
    if not intersections["all_disjoint"]:
        raise PreregistrationError(f"Stage intersection check failed: {intersections}")
    identity_input = {
        "protocol": PROTOCOL_NAME,
        "dataset": {"builder": "glue", "configuration": "sst2", "split": "validation"},
        "config_sha256": config.config_sha256,
        "model_key_file_sha256": dict(model_key_file_sha256),
        "tokenizer_sha256": tokenizer_sha256,
        "sample_list_sha256": stage_hashes,
        "intersection_check": intersections,
    }
    return {
        "schema_version": 1,
        "protocol": PROTOCOL_NAME,
        "preregistration_sha256": sha256_json(identity_input),
        "dataset": {"builder": "glue", "configuration": "sst2", "split": "validation"},
        "tokenization_protocol": {
            "add_special_tokens": False,
            "explicit_eos_append": True,
            "max_length": config.max_length,
            "min_effective_token_length": config.min_effective_token_length,
        },
        "config": config.manifest_config(),
        "config_sha256": config.config_sha256,
        "model_path": str(config.model_path),
        "model_key_file_sha256": dict(model_key_file_sha256),
        "tokenizer_sha256": tokenizer_sha256,
        "tokenizer_key_file_sha256": dict(tokenizer_key_file_sha256),
        "sample_lists": stage_lists,
        "sample_list_sha256": stage_hashes,
        "intersection_check": intersections,
        "created_at": created_at,
        "git_commit": commit,
        "runtime_versions": dict(versions),
    }


def stage_jsonl_records(document: Mapping[str, Any], stage: str) -> list[dict[str, Any]]:
    """Create deterministic JSONL records for one preregistered stage."""
    sample_lists = document.get("sample_lists")
    sample_hashes = document.get("sample_list_sha256")
    if not isinstance(sample_lists, Mapping) or not isinstance(sample_hashes, Mapping):
        raise PreregistrationError("Malformed preregistration document: missing sample lists or hashes.")
    samples = sample_lists.get(stage)
    sample_list_sha256 = sample_hashes.get(stage)
    if not isinstance(samples, list) or not isinstance(sample_list_sha256, str):
        raise PreregistrationError(f"Malformed preregistration document for stage {stage!r}.")
    return [
        {
            "schema_version": 1,
            "record_type": "preregistered_sst2_validation_sample",
            "protocol": PROTOCOL_NAME,
            "preregistration_sha256": document["preregistration_sha256"],
            "stage": stage,
            "stage_sample_list_sha256": sample_list_sha256,
            "sample": sample,
        }
        for sample in samples
    ]


def preregister_experiment(config_path: Path) -> dict[str, Any]:
    """Execute the complete deterministic preregistration and recover safely."""
    from .config import load_experiment_config

    config = load_experiment_config(config_path)
    tokenizer = load_qwen3_tokenizer(config)
    rows = load_official_sst2_validation()
    eligible = prepare_eligible_samples(rows, tokenizer, config)
    stages = allocate_stages(eligible)
    model_hashes, tokenizer_hash, tokenizer_hashes = collect_model_and_tokenizer_hashes(config)
    document = build_preregistration_document(
        config=config,
        stages=stages,
        model_key_file_sha256=model_hashes,
        tokenizer_sha256=tokenizer_hash,
        tokenizer_key_file_sha256=tokenizer_hashes,
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        commit=git_commit(config.repository_root),
        versions=runtime_versions(),
    )

    config.output_root.mkdir(parents=True, exist_ok=True)
    for stage in (*STAGE_SIZES, "utility"):
        (config.output_root / stage).mkdir(parents=True, exist_ok=True)
    manifests = config.project_root / "manifests"
    manifest_written = write_or_verify_json(
        manifests / "preregistration.json",
        document,
        identity_key="preregistration_sha256",
        ignored_existing_keys=("created_at",),
    )
    jsonl_written = {
        stage: write_or_verify_jsonl(manifests / f"{stage}.jsonl", stage_jsonl_records(document, stage))
        for stage in STAGE_SIZES
    }
    return {
        "status": "created" if manifest_written or any(jsonl_written.values()) else "already_preregistered",
        "preregistration_sha256": document["preregistration_sha256"],
        "sample_list_sha256": document["sample_list_sha256"],
        "eligible_sample_count": len(eligible),
        "stages": {stage: len(samples) for stage, samples in stages.items()},
        "manifest_written": manifest_written,
        "jsonl_written": jsonl_written,
    }
