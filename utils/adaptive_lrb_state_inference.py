"""Differentiable, passive state inference primitives for static Projection-LRB.

This module is deliberately independent from ``attack.py``.  It receives only
observed DAGER span bases and public candidate embeddings while fitting a shared
static sign state.  Ground-truth defense metadata is accepted exclusively by
the audit helpers, never by :func:`fit_state`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


STATE_INFERENCE_PROTOCOL = "state_inference_v1"


def hard_sign_ste(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Hard Rademacher signs with a tanh straight-through backward surrogate."""
    soft = torch.tanh(logits / max(float(temperature), 1e-6))
    hard = torch.where(soft >= 0, torch.ones_like(soft), -torch.ones_like(soft))
    return hard.detach() - soft.detach() + soft


def _project_with_q(values: torch.Tensor, q: int, signs: torch.Tensor, mode: str) -> torch.Tensor:
    """Match ``_project_last_dim_signed_pool`` for an integer output width."""
    hidden = int(values.shape[-1])
    q = max(1, min(int(q), hidden))
    if q >= hidden:
        return values
    flat = values.float().reshape(-1, hidden)
    signed = flat * signs.to(device=values.device, dtype=torch.float32).reshape(1, hidden)
    if mode == "signed_stride":
        indices = torch.linspace(0, hidden - 1, steps=q, device=values.device).round().long()
        pooled = signed.index_select(1, indices).unsqueeze(1)
    else:
        pooled = F.adaptive_avg_pool1d(signed.unsqueeze(1), q)
    interpolate_mode = "nearest" if mode == "signed_pool_nearest" else "linear"
    projected = F.interpolate(pooled, size=hidden, mode=interpolate_mode).squeeze(1)
    return (projected * signs.to(device=values.device, dtype=torch.float32).reshape(1, hidden)).reshape_as(values).to(values.dtype)


def continuous_signed_pool(
    values: torch.Tensor,
    q_continuous: torch.Tensor | float,
    sign_logits: torch.Tensor,
    *,
    mode: str = "signed_pool",
    temperature: float = 1.0,
) -> torch.Tensor:
    """Continuous pooled-width surrogate with exact integer endpoints.

    The forward transform interpolates adjacent integer-width pooling operators.
    At an integer q it is byte-for-byte equivalent in value to the existing
    candidate-side signed-pool operator, subject only to dtype conversion.
    """
    if mode not in {"signed_pool", "signed_pool_nearest", "signed_stride"}:
        raise ValueError(f"Unsupported state-inference projection mode: {mode}")
    hidden = int(values.shape[-1])
    if int(sign_logits.numel()) != hidden:
        raise ValueError(f"sign logits length {sign_logits.numel()} does not match feature width {hidden}")
    q_value = torch.as_tensor(q_continuous, dtype=torch.float32, device=values.device).clamp(1.0, float(hidden))
    low = int(torch.floor(q_value.detach()).item())
    high = min(hidden, low + 1)
    alpha = (q_value - float(low)).to(dtype=values.dtype)
    signs = hard_sign_ste(sign_logits.to(values.device), temperature=temperature)
    low_out = _project_with_q(values, low, signs, mode)
    if high == low:
        return low_out
    high_out = _project_with_q(values, high, signs, mode)
    return low_out + alpha * (high_out - low_out)


def span_distance(span_basis: torch.Tensor, values: torch.Tensor, *, norm: str = "l2") -> torch.Tensor:
    """Differentiable non-mutating equivalent of ``check_if_in_span``."""
    normalized = values / values.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-12)
    projected = torch.einsum("ik,ij,...j->...k", span_basis, span_basis, normalized)
    residual = projected - normalized
    if norm == "l2":
        return residual.pow(2).sum(-1).sqrt()
    if norm == "l1":
        return residual.abs().mean(-1)
    raise ValueError(f"Unsupported span norm: {norm}")


def sign_agreement_mod_global_flip(estimated: torch.Tensor, truth: torch.Tensor) -> float:
    """Agreement modulo the global sign symmetry of S P(Sx)."""
    if estimated.numel() != truth.numel():
        raise ValueError("Estimated and true sign vectors must have equal length.")
    # This is a post-hoc audit only. Estimator signs live on the fitting device
    # while captured truth is deliberately retained on CPU, so compare detached
    # CPU signs rather than moving private audit tensors into the GPU workflow.
    estimated_signs = estimated.detach().reshape(-1).sign().to(device="cpu")
    truth_signs = truth.detach().reshape(-1).sign().to(device="cpu")
    equal = (estimated_signs == truth_signs).float().mean()
    return float(torch.maximum(equal, 1.0 - equal).item())


@dataclass(frozen=True)
class StateInferenceObservation:
    """Observed-only fitting inputs for one private update.

    ``span_bases`` and ``candidate_values`` are computed from the uploaded
    defended update and public model/tokenizer.  This object intentionally has
    no sample text, label, raw gradient, or defense metadata field.
    """
    span_bases: tuple[torch.Tensor, ...]
    candidate_values: tuple[torch.Tensor, ...]


def stage_selected_grads_cpu(grads, indices: Sequence[int]) -> tuple[torch.Tensor | None, ...]:
    """Persist only selected defended matrices in host memory.

    The state-inference protocol's span fitting and decoder use only the
    selected DAGER matrices. Retaining all full GPT-2 gradients for 100 target
    updates exhausts device memory; retaining exact CPU copies of the selected
    matrices does not change the fit or decode input values.
    """
    selected = {int(index) for index in indices}
    staged: list[torch.Tensor | None] = []
    for index, grad in enumerate(grads):
        if index in selected:
            if grad is None:
                raise RuntimeError(f"Selected DAGER gradient {index} is missing.")
            staged.append(grad.detach().to(device="cpu", copy=True))
        else:
            staged.append(None)
    return tuple(staged)


def stage_grads_for_decode(grads, device: torch.device) -> tuple[torch.Tensor | None, ...]:
    """Materialize one captured CPU update on the attack device for decoding."""
    return tuple(
        None if grad is None else grad.to(device=device, non_blocking=False)
        for grad in grads
    )


class StaticStateEstimator(nn.Module):
    """Shared signs plus per-update continuous pooled widths."""

    def __init__(self, feature_widths: Sequence[int], n_updates: int, *, min_ratio: float = 0.2, max_ratio: float = 0.9):
        super().__init__()
        if not feature_widths or n_updates <= 0:
            raise ValueError("State estimator needs feature widths and at least one observed update.")
        self.feature_widths = tuple(int(width) for width in feature_widths)
        self.min_ratio = float(min_ratio)
        self.max_ratio = float(max_ratio)
        self.sign_logits = nn.ParameterList([nn.Parameter(torch.zeros(width)) for width in self.feature_widths])
        self.ratio_logits = nn.Parameter(torch.zeros(n_updates, len(self.feature_widths)))

    def q_values(self, update_index: int) -> list[torch.Tensor]:
        ratios = self.min_ratio + (self.max_ratio - self.min_ratio) * torch.sigmoid(self.ratio_logits[update_index])
        return [ratio * width for ratio, width in zip(ratios, self.feature_widths)]

    def hard_signs(self) -> list[torch.Tensor]:
        return [torch.where(logits.detach() >= 0, torch.ones_like(logits), -torch.ones_like(logits)) for logits in self.sign_logits]

    def install_initializer(self, sign_logits: Sequence[torch.Tensor] | None = None, ratios: torch.Tensor | None = None) -> None:
        """Install public-calibration predictions without accessing target metadata."""
        with torch.no_grad():
            if sign_logits is not None:
                if len(sign_logits) != len(self.sign_logits):
                    raise ValueError("Initializer layer count does not match estimator.")
                for target, source in zip(self.sign_logits, sign_logits):
                    if target.numel() != source.numel():
                        raise ValueError("Initializer sign width does not match estimator.")
                    target.copy_(source.to(target.device, target.dtype))
            if ratios is not None:
                if tuple(ratios.shape) != tuple(self.ratio_logits.shape):
                    raise ValueError("Initializer ratio shape does not match estimator.")
                clipped = ratios.to(self.ratio_logits.device).clamp(self.min_ratio + 1e-4, self.max_ratio - 1e-4)
                scaled = (clipped - self.min_ratio) / (self.max_ratio - self.min_ratio)
                self.ratio_logits.copy_(torch.logit(scaled))


def fit_state(
    estimator: StaticStateEstimator,
    observations: Sequence[StateInferenceObservation],
    *,
    update_indices: Sequence[int] | None = None,
    steps: int,
    learning_rate: float = 0.05,
    temperature_start: float = 2.0,
    temperature_end: float = 0.25,
    softmin_temperature: float = 0.05,
    progress_callback=None,
) -> list[float]:
    """Fit from observed span consistency only; returns the unlabelled loss trace."""
    if update_indices is None:
        update_indices = tuple(range(len(observations)))
    if len(observations) != len(update_indices):
        raise ValueError("Each fitting observation needs one update index.")
    if any(index < 0 or index >= estimator.ratio_logits.shape[0] for index in update_indices):
        raise ValueError("Fitting update index is outside the estimator ratio table.")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if steps == 0:
        return []
    optimizer = torch.optim.Adam(estimator.parameters(), lr=float(learning_rate))
    trace: list[float] = []
    loss_terms_per_step = sum(len(observation.span_bases) for observation in observations)
    if loss_terms_per_step <= 0:
        raise ValueError("State fitting needs at least one observed layer basis.")
    for step in range(int(steps)):
        progress = step / max(steps - 1, 1)
        temperature = temperature_start + progress * (temperature_end - temperature_start)
        optimizer.zero_grad(set_to_none=True)
        detached_step_loss = 0.0
        for update_index, observation in zip(update_indices, observations):
            if len(observation.span_bases) != len(estimator.sign_logits):
                raise ValueError("Observation layer count does not match estimator.")
            observation_losses = []
            for layer_index, (basis, candidates, logits, q_value) in enumerate(
                zip(observation.span_bases, observation.candidate_values, estimator.sign_logits, estimator.q_values(update_index))
            ):
                device = logits.device
                transformed = continuous_signed_pool(candidates.to(device), q_value, logits, temperature=temperature)
                distances = span_distance(basis.to(device), transformed)
                flat = distances.reshape(-1)
                observation_losses.append(-softmin_temperature * torch.logsumexp(-flat / softmin_temperature, dim=0))
            observation_loss = torch.stack(observation_losses).sum() / float(loss_terms_per_step)
            observation_loss.backward()
            detached_step_loss += float(observation_loss.detach().cpu())
        optimizer.step()
        trace.append(detached_step_loss)
        if progress_callback is not None:
            progress_callback(step + 1, int(steps), detached_step_loss)
    return trace


def state_override(estimator: StaticStateEstimator, update_index: int, *, mode: str = "signed_pool") -> dict:
    """Return the private override shape consumed by ``prepare_adaptive_attack``."""
    layers = {}
    for pos, (q, signs, width) in enumerate(zip(estimator.q_values(update_index), estimator.hard_signs(), estimator.feature_widths)):
        layers[pos] = {
            "lrb_keep_ratio": float((q.detach() / float(width)).item()),
            "lrb_projection_mode": mode,
            "lrb_projection_seed": -1,
            "lrb_feature_signs": signs.detach(),
            "lrb_feature_axis": "estimated",
        }
    return {"layers": layers, "profile": STATE_INFERENCE_PROTOCOL}


def q_audit(estimated_q: Iterable[float], true_q: Iterable[int]) -> dict[str, float]:
    estimated = [int(round(float(value))) for value in estimated_q]
    truth = [int(value) for value in true_q]
    if len(estimated) != len(truth) or not truth:
        raise ValueError("q audit needs equally sized non-empty estimates and truth values.")
    errors = [abs(left - right) for left, right in zip(estimated, truth)]
    return {"q_exact_match": sum(error == 0 for error in errors) / len(errors), "q_abs_error_mean": sum(errors) / len(errors)}


@dataclass(frozen=True)
class PublicCalibrationRecord:
    """A simulated public update with labels generated under an attacker-sampled state."""
    feature_sketches: tuple[torch.Tensor, ...]
    ratios: tuple[float, ...]
    signs: tuple[torch.Tensor, ...]


class PublicCalibrationInitializer:
    """Small public-only initializer for ratios and static feature-sign logits.

    It is intentionally a low-capacity correlation model rather than a target
    reconstruction network: calibration examples are generated from public
    batches under attacker-sampled states, and the target state is still fitted
    solely by observed-span consistency in :func:`fit_state`.
    """

    def __init__(self, feature_widths: Sequence[int]):
        self.feature_widths = tuple(int(width) for width in feature_widths)
        self._sign_weights: list[torch.Tensor] | None = None
        self._ratio_scale: torch.Tensor | None = None
        self._ratio_bias: torch.Tensor | None = None

    def fit(self, records: Sequence[PublicCalibrationRecord]) -> None:
        if not records:
            raise ValueError("Public calibration needs at least one record.")
        if any(len(record.feature_sketches) != len(self.feature_widths) for record in records):
            raise ValueError("Calibration record layer count does not match feature widths.")
        weights = []
        ratio_features = []
        ratio_targets = []
        for layer, width in enumerate(self.feature_widths):
            sketches = torch.stack([record.feature_sketches[layer].reshape(-1) for record in records]).float()
            signs = torch.stack([record.signs[layer].reshape(-1) for record in records]).float()
            if sketches.shape[1] != width or signs.shape[1] != width:
                raise ValueError("Calibration sketch/sign width mismatch.")
            normalized = sketches / sketches.abs().mean(dim=1, keepdim=True).clamp_min(1e-6)
            weights.append((normalized * signs).mean(dim=0))
            ratio_features.append(normalized.abs().mean(dim=1))
            ratio_targets.append(torch.tensor([record.ratios[layer] for record in records], dtype=torch.float32))
        features = torch.stack(ratio_features, dim=1)
        targets = torch.stack(ratio_targets, dim=1)
        design = torch.cat((features, torch.ones(features.shape[0], 1)), dim=1)
        solution = torch.linalg.lstsq(design, targets).solution
        self._sign_weights = weights
        self._ratio_scale = solution[:-1]
        self._ratio_bias = solution[-1]

    def predict(self, target_sketches: Sequence[Sequence[torch.Tensor]]) -> tuple[list[torch.Tensor], torch.Tensor]:
        if self._sign_weights is None or self._ratio_scale is None or self._ratio_bias is None:
            raise RuntimeError("Fit the public calibration initializer before prediction.")
        if not target_sketches:
            raise ValueError("Need at least one target sketch.")
        n_updates = len(target_sketches)
        sign_logits = []
        ratios = torch.empty(n_updates, len(self.feature_widths))
        for layer, weight in enumerate(self._sign_weights):
            vectors = []
            for sketches in target_sketches:
                vector = sketches[layer].reshape(-1).float()
                vector = vector / vector.abs().mean().clamp_min(1e-6)
                vectors.append(vector)
            stacked = torch.stack(vectors)
            sign_logits.append((stacked.mean(dim=0) * weight).to(dtype=torch.float32))
            features = stacked.abs().mean(dim=1)
            ratios[:, layer] = features * self._ratio_scale[layer, layer] + self._ratio_bias[layer]
        return sign_logits, ratios
