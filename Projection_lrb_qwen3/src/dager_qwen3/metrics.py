"""Token recovery and offline ROUGE-N metrics for one Qwen3 DAGER record."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


class AttackMetricsError(RuntimeError):
    """Raised when one attack metric cannot be computed without ambiguity."""


class RougeMetricProtocol(Protocol):
    """The legacy ``datasets.load_metric('rouge')`` interface used by DAGER."""

    def compute(self, *, predictions: list[str], references: list[str]) -> Any:
        """Compute the existing repository ROUGE result object."""


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


def load_existing_dager_rouge_metric() -> RougeMetricProtocol:
    """Load the exact offline ROUGE metric interface used by existing DAGER scripts.

    The repository's GPT-2 DAGER entrypoints use ``datasets.load_metric('rouge')``.
    This function intentionally has no network call or simplified metric fallback:
    if the local metric cache/package is unavailable, the attack fails explicitly.
    """
    try:
        from datasets import load_metric
    except ImportError as error:
        raise AttackMetricsError(
            "datasets.load_metric('rouge') is required for the existing DAGER ROUGE definition."
        ) from error
    try:
        metric = load_metric("rouge")
    except Exception as error:
        raise AttackMetricsError(
            f"Unable to load the existing offline DAGER ROUGE metric; no replacement metric is permitted: {error}"
        ) from error
    if not callable(getattr(metric, "compute", None)):
        raise AttackMetricsError("datasets.load_metric('rouge') returned an object without compute().")
    return metric


def _legacy_rouge_f1(metric: RougeMetricProtocol, prediction: str, reference: str, key: str) -> float:
    try:
        result = metric.compute(predictions=[prediction], references=[reference])
        score = result[key].mid.fmeasure
        value = float(score)
    except Exception as error:
        raise AttackMetricsError(f"Existing DAGER ROUGE computation failed for {key}: {error}") from error
    if value < 0.0 or value > 1.0:
        raise AttackMetricsError(f"Existing DAGER ROUGE {key} is outside [0, 1]: {value}.")
    return value


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
