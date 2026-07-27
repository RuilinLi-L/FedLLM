"""Token recovery and offline ROUGE-N metrics for one Qwen3 DAGER record."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import inspect
import math
import os
from pathlib import Path
from typing import Any, Protocol, Sequence


class AttackMetricsError(RuntimeError):
    """Raised when one attack metric cannot be computed without ambiguity."""


class RougeMetricProtocol(Protocol):
    """The legacy ``datasets.load_metric('rouge')`` interface used by DAGER."""

    def compute(self, *, predictions: list[str], references: list[str]) -> Any:
        """Compute the existing repository ROUGE result object."""


@dataclass(frozen=True)
class LegacyRougeBackend:
    """Verified provenance for the legacy cached ``datasets`` ROUGE backend."""

    metric: RougeMetricProtocol
    backend: str
    datasets_version: str
    metric_script_sha256: str
    self_test_rouge_1: float
    self_test_rouge_2: float

    def json_metadata(self) -> dict[str, Any]:
        """Return only JSON-serializable ROUGE provenance for an attack row."""
        return {
            "backend": self.backend,
            "datasets_version": self.datasets_version,
            "metric_script_sha256": self.metric_script_sha256,
            "hf_datasets_offline": "1",
            "hf_hub_offline": "1",
            "self_test": {
                "kind": "fixed_exact_match",
                "rouge_1": self.self_test_rouge_1,
                "rouge_2": self.self_test_rouge_2,
            },
        }


@dataclass(frozen=True)
class AttackMetrics:
    """Ground-truth reporting metrics; none of these values influence decoding."""

    token_recovery: float
    exact_recovery: bool
    rouge_1: float
    rouge_2: float
    empty_reconstruction: bool
    ground_truth_text: str
    reconstructed_text: str
    ground_truth_token_text: tuple[str, ...]
    reconstructed_token_text: tuple[str, ...]


def _require_token_ids(name: str, token_ids: Sequence[int]) -> tuple[int, ...]:
    result: list[int] = []
    for token_id in token_ids:
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise AttackMetricsError(f"{name} contains invalid token id {token_id!r}.")
        result.append(token_id)
    return tuple(result)


def _token_texts(tokenizer: Any, token_ids: tuple[int, ...]) -> tuple[str, ...]:
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if not callable(convert):
        raise AttackMetricsError("Qwen3 tokenizer lacks convert_ids_to_tokens for structured attack output.")
    try:
        values = convert(list(token_ids))
    except Exception as error:
        raise AttackMetricsError(f"Qwen3 tokenizer failed to render token ids: {error}") from error
    if not isinstance(values, list) or len(values) != len(token_ids) or any(not isinstance(value, str) for value in values):
        raise AttackMetricsError("Qwen3 tokenizer returned invalid token-text output.")
    return tuple(values)


def _decoded_text(tokenizer: Any, token_ids: tuple[int, ...], eos_token_id: int) -> str:
    if any(token_id == eos_token_id for token_id in token_ids[:-1]):
        raise AttackMetricsError("EOS may terminate a reconstruction but cannot appear before its final position.")
    payload = token_ids[:-1] if token_ids and token_ids[-1] == eos_token_id else token_ids
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise AttackMetricsError("Qwen3 tokenizer lacks decode for ROUGE reporting.")
    try:
        text = decode(list(payload), skip_special_tokens=True, clean_up_tokenization_spaces=False)
    except Exception as error:
        raise AttackMetricsError(f"Qwen3 tokenizer failed to decode attack tokens: {error}") from error
    if not isinstance(text, str):
        raise AttackMetricsError(f"Qwen3 tokenizer returned non-string decoded text {type(text).__name__}.")
    return text


def _legacy_rouge_f1_from_result(result: Any, key: str) -> float:
    """Extract one legacy ``datasets.load_metric`` F1 score with validation."""
    try:
        value = float(result[key].mid.fmeasure)
    except Exception as error:
        raise AttackMetricsError(f"Existing DAGER ROUGE computation failed for {key}: {error}") from error
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise AttackMetricsError(f"Existing DAGER ROUGE {key} is not finite in [0, 1]: {value}.")
    return value


def _metric_script_sha256(metric: RougeMetricProtocol) -> str:
    """Hash the exact cached legacy metric script rather than its package name."""
    try:
        source_file = inspect.getfile(type(metric))
    except (OSError, TypeError) as error:
        raise AttackMetricsError("Unable to identify the loaded legacy ROUGE metric script.") from error
    metric_script = Path(source_file).resolve()
    if not metric_script.is_file():
        raise AttackMetricsError(f"Legacy ROUGE metric script is not a readable file: {metric_script}.")
    try:
        return hashlib.sha256(metric_script.read_bytes()).hexdigest()
    except OSError as error:
        raise AttackMetricsError(f"Unable to hash legacy ROUGE metric script {metric_script}: {error}") from error


def preflight_legacy_dager_rouge_backend() -> LegacyRougeBackend:
    """Verify the cached legacy ROUGE backend before a model is loaded.

    This is intentionally the only ROUGE backend accepted by the Qwen3 attack:
    ``datasets.load_metric('rouge')``.  Offline variables are force-set before
    importing ``datasets`` and remain enabled for the process, so a cache miss
    cannot turn into a download.  There is no ``evaluate`` or custom fallback.
    """
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        import datasets
    except Exception as error:
        raise AttackMetricsError(
            "datasets.load_metric('rouge') is required for the existing DAGER ROUGE definition."
        ) from error
    load_metric = getattr(datasets, "load_metric", None)
    if not callable(load_metric):
        raise AttackMetricsError(
            "Installed datasets does not expose datasets.load_metric; the legacy DAGER ROUGE backend is unavailable."
        )
    try:
        metric = load_metric("rouge")
    except Exception as error:
        raise AttackMetricsError(
            f"Unable to load the existing offline DAGER ROUGE metric; no replacement metric is permitted: {error}"
        ) from error
    if not callable(getattr(metric, "compute", None)):
        raise AttackMetricsError("datasets.load_metric('rouge') returned an object without compute().")
    datasets_version = getattr(datasets, "__version__", None)
    if not isinstance(datasets_version, str) or not datasets_version:
        try:
            datasets_version = importlib.metadata.version("datasets")
        except importlib.metadata.PackageNotFoundError as error:
            raise AttackMetricsError("Unable to determine the datasets version for ROUGE provenance.") from error
    fixed_exact_match = "qwen3 dager legacy rouge preflight"
    try:
        self_test_result = metric.compute(
            predictions=[fixed_exact_match],
            references=[fixed_exact_match],
        )
        rouge_1 = _legacy_rouge_f1_from_result(self_test_result, "rouge1")
        rouge_2 = _legacy_rouge_f1_from_result(self_test_result, "rouge2")
    except AttackMetricsError:
        raise
    except Exception as error:
        raise AttackMetricsError(f"Legacy ROUGE exact-match self-test failed: {error}") from error
    if not math.isclose(rouge_1, 1.0, rel_tol=0.0, abs_tol=1e-12) or not math.isclose(
        rouge_2, 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise AttackMetricsError(
            "Legacy ROUGE exact-match self-test did not return ROUGE-1=ROUGE-2=1.0; refusing to run attack."
        )
    return LegacyRougeBackend(
        metric=metric,
        backend="datasets.load_metric('rouge')",
        datasets_version=datasets_version,
        metric_script_sha256=_metric_script_sha256(metric),
        self_test_rouge_1=rouge_1,
        self_test_rouge_2=rouge_2,
    )


def _legacy_rouge_f1(metric: RougeMetricProtocol, prediction: str, reference: str, key: str) -> float:
    try:
        result = metric.compute(predictions=[prediction], references=[reference])
    except Exception as error:
        raise AttackMetricsError(f"Existing DAGER ROUGE computation failed for {key}: {error}") from error
    return _legacy_rouge_f1_from_result(result, key)


def compute_attack_metrics(
    *,
    tokenizer: Any,
    ground_truth_token_ids: Sequence[int],
    reconstructed_token_ids: Sequence[int],
    eos_token_id: int,
    rouge_metric: RougeMetricProtocol,
) -> AttackMetrics:
    """Report metrics after decoding; ground truth is never passed to DAGER search."""
    ground_truth = _require_token_ids("ground_truth_token_ids", ground_truth_token_ids)
    reconstructed = _require_token_ids("reconstructed_token_ids", reconstructed_token_ids)
    if not ground_truth:
        raise AttackMetricsError("Ground-truth preregistered token sequence must not be empty.")
    if ground_truth[-1] != eos_token_id:
        raise AttackMetricsError("Ground-truth preregistered sequence must end with its explicit EOS token.")
    ground_truth_text = _decoded_text(tokenizer, ground_truth, eos_token_id)
    reconstructed_text = _decoded_text(tokenizer, reconstructed, eos_token_id)
    aligned_matches = sum(
        int(predicted == expected)
        for predicted, expected in zip(reconstructed, ground_truth)
    )
    return AttackMetrics(
        token_recovery=aligned_matches / len(ground_truth),
        exact_recovery=reconstructed == ground_truth,
        rouge_1=_legacy_rouge_f1(rouge_metric, reconstructed_text, ground_truth_text, "rouge1"),
        rouge_2=_legacy_rouge_f1(rouge_metric, reconstructed_text, ground_truth_text, "rouge2"),
        empty_reconstruction=not reconstructed,
        ground_truth_text=ground_truth_text,
        reconstructed_text=reconstructed_text,
        ground_truth_token_text=_token_texts(tokenizer, ground_truth),
        reconstructed_token_text=_token_texts(tokenizer, reconstructed),
    )
