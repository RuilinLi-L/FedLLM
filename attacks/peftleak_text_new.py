from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import re
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from attacks.peftleak_text import get_token_embedding_matrix, nearest_token_ids
from utils.defenses import (
    dpsgd_defense,
    gradient_compression,
    noise_injection,
    topk_sparsification,
)
from utils.lrb_defense import apply_lrb_defense


ADAPTER_RATIO_DEFENSES = {
    "none",
    "noise",
    "dpsgd",
    "topk",
    "compression",
    "lrb",
    "lrbprojonly",
    "signed_bottleneck",
}
LORA_OPT_DEFENSES = {
    "none",
    "topk",
    "compression",
    "lrb",
    "lrbprojonly",
    "signed_bottleneck",
}


@dataclass(frozen=True)
class PublicProbeStatistics:
    mean: torch.Tensor
    std: torch.Tensor
    bin_edges: torch.Tensor
    global_bin_edges: torch.Tensor
    num_sequences: int
    num_tokens: int
    num_bins: int
    max_positions: int


@dataclass(frozen=True)
class InstalledEmbeddingProbe:
    probe: "FixedBinnedEmbeddingProbe"
    wrapped_embedding: "ProbedEmbedding"
    embedding_path: str
    parameter_names: tuple[str, ...]
    parameters: tuple[nn.Parameter, ...]


@dataclass(frozen=True)
class ProbeGradientObservation:
    raw_gradients: tuple[torch.Tensor, ...]
    observed_gradients: tuple[torch.Tensor, ...]
    parameter_names: tuple[str, ...]
    logits: torch.Tensor
    loss: float


def _valid_mask(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        return torch.ones_like(input_ids, dtype=torch.bool)
    return attention_mask.to(device=input_ids.device).bool()


def build_public_probe_statistics(
    base_embedding: nn.Module,
    public_batches: Sequence[Mapping[str, torch.Tensor]],
    *,
    max_positions: int,
    num_bins: int,
) -> PublicProbeStatistics:
    """Build fixed routing thresholds from a public, disjoint text split."""

    if max_positions < 1:
        raise ValueError("max_positions must be positive.")
    if num_bins < 1:
        raise ValueError("num_bins must be positive.")

    vectors_by_position: dict[int, list[torch.Tensor]] = {}
    all_vectors: list[torch.Tensor] = []
    num_sequences = 0
    with torch.no_grad():
        for batch in public_batches:
            vectors = base_embedding(batch["input_ids"])
            if vectors.ndim != 3:
                raise ValueError("The token embedding must return [batch, sequence, hidden] tensors.")
            mask = _valid_mask(batch)
            num_sequences += int(vectors.shape[0])
            limit = min(int(vectors.shape[1]), int(max_positions))
            for position in range(limit):
                selected = vectors[:, position][mask[:, position]]
                if selected.numel() == 0:
                    continue
                selected_cpu = selected.detach().float().cpu()
                vectors_by_position.setdefault(position, []).append(selected_cpu)
                all_vectors.append(selected_cpu)

    if not all_vectors:
        raise ValueError("Cannot build PEFTLeak public statistics from an empty public split.")

    flat = torch.cat(all_vectors, dim=0)
    mean = flat.mean(dim=0)
    std = flat.std(dim=0, unbiased=False).clamp_min(1e-6)
    edge_count = max(0, int(num_bins) - 1)
    quantiles = torch.linspace(0, 1, int(num_bins) + 1, dtype=torch.float32)[1:-1]
    global_scores = ((flat - mean.unsqueeze(0)) / std.unsqueeze(0)).mean(dim=-1)
    global_edges = torch.quantile(global_scores, quantiles) if edge_count else torch.empty(0)
    bin_edges = torch.empty(int(max_positions), edge_count, dtype=torch.float32)

    for position in range(int(max_positions)):
        parts = vectors_by_position.get(position)
        if not parts:
            bin_edges[position] = global_edges
            continue
        position_vectors = torch.cat(parts, dim=0)
        scores = ((position_vectors - mean.unsqueeze(0)) / std.unsqueeze(0)).mean(dim=-1)
        bin_edges[position] = torch.quantile(scores, quantiles) if edge_count else torch.empty(0)

    return PublicProbeStatistics(
        mean=mean,
        std=std,
        bin_edges=bin_edges,
        global_bin_edges=global_edges,
        num_sequences=num_sequences,
        num_tokens=int(flat.shape[0]),
        num_bins=int(num_bins),
        max_positions=int(max_positions),
    )


class PositionBinProbe(nn.Module):
    """Zero-output scalar probes whose weight/bias gradient ratio reveals one input vector."""

    def __init__(self, hidden_dim: int, num_bins: int, rows_per_bin: int, *, seed: int):
        super().__init__()
        if hidden_dim < 1 or num_bins < 1 or rows_per_bin < 2:
            raise ValueError("Probe dimensions must be positive and rows_per_bin must be at least two.")
        self.weight = nn.Parameter(torch.zeros(num_bins, rows_per_bin, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(num_bins, rows_per_bin))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        directions = torch.randn(rows_per_bin, hidden_dim, generator=generator)
        directions = F.normalize(directions, dim=-1)
        self.register_buffer("directions", directions)

    def forward(self, vectors: torch.Tensor, bin_indices: torch.Tensor) -> torch.Tensor:
        weights = self.weight[bin_indices]
        biases = self.bias[bin_indices]
        scalar_outputs = (weights * vectors.unsqueeze(1)).sum(dim=-1) + biases
        directions = self.directions.to(device=vectors.device, dtype=vectors.dtype)
        return scalar_outputs.matmul(directions) / math.sqrt(float(directions.shape[0]))


class FixedBinnedEmbeddingProbe(nn.Module):
    """Fixed position-by-public-bin probe inventory allocated before private data is loaded."""

    def __init__(
        self,
        hidden_dim: int,
        statistics: PublicProbeStatistics,
        *,
        rows_per_bin: int,
        seed: int,
    ):
        super().__init__()
        if hidden_dim != int(statistics.mean.numel()):
            raise ValueError("Public statistics hidden dimension does not match the embedding dimension.")
        self.hidden_dim = int(hidden_dim)
        self.num_bins = int(statistics.num_bins)
        self.max_positions = int(statistics.max_positions)
        self.rows_per_bin = int(rows_per_bin)
        self.register_buffer("public_mean", statistics.mean.detach().clone())
        self.register_buffer("public_std", statistics.std.detach().clone())
        self.register_buffer("public_bin_edges", statistics.bin_edges.detach().clone())
        self.positions = nn.ModuleList(
            PositionBinProbe(
                self.hidden_dim,
                self.num_bins,
                self.rows_per_bin,
                seed=int(seed) + 104729 * (position + 1),
            )
            for position in range(self.max_positions)
        )

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        if vectors.ndim != 3:
            raise ValueError("Probe input must have shape [batch, sequence, hidden].")
        if vectors.shape[1] > self.max_positions:
            raise ValueError(
                f"Private sequence length {vectors.shape[1]} exceeds the fixed probe inventory "
                f"({self.max_positions})."
            )
        normalized = (
            vectors - self.public_mean.to(device=vectors.device, dtype=vectors.dtype).view(1, 1, -1)
        ) / self.public_std.to(device=vectors.device, dtype=vectors.dtype).view(1, 1, -1)
        scores = normalized.float().mean(dim=-1)
        residuals: list[torch.Tensor] = []
        for position in range(int(vectors.shape[1])):
            edges = self.public_bin_edges[position].to(device=vectors.device).float()
            bins = torch.bucketize(scores[:, position].contiguous(), edges, right=False)
            residuals.append(self.positions[position](vectors[:, position], bins))
        return torch.stack(residuals, dim=1)


class ProbedEmbedding(nn.Module):
    """Embedding wrapper that keeps clean outputs unchanged at zero probe initialization."""

    def __init__(self, base_embedding: nn.Module, probe: FixedBinnedEmbeddingProbe):
        super().__init__()
        self.base_embedding = base_embedding
        self.peftleak_probe = probe

    @property
    def weight(self):
        return self.base_embedding.weight

    @property
    def padding_idx(self):
        return getattr(self.base_embedding, "padding_idx", None)

    def forward(self, input_ids: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        vectors = self.base_embedding(input_ids, *args, **kwargs)
        return vectors + self.peftleak_probe(vectors)


def resolve_base_word_embedding(model_wrapper) -> tuple[nn.Module, str, nn.Module]:
    model_path = str(getattr(model_wrapper.args, "model_path", ""))
    if model_path in {"gpt2", "openai-community/gpt2-large"}:
        parent = model_wrapper.base_model.transformer
        return parent, "wte", parent.wte
    if model_path == "bert-base-uncased":
        parent = model_wrapper.base_model.bert.embeddings
        return parent, "word_embeddings", parent.word_embeddings
    raise NotImplementedError("Strict PEFTLeak text v2 supports GPT-2 and BERT sequence classifiers only.")


def install_fixed_embedding_probe(
    model_wrapper,
    statistics: PublicProbeStatistics,
    *,
    rows_per_bin: int,
    seed: int,
) -> InstalledEmbeddingProbe:
    parent, attribute, base_embedding = resolve_base_word_embedding(model_wrapper)
    if isinstance(base_embedding, ProbedEmbedding):
        raise ValueError("A PEFTLeak embedding probe is already installed.")
    hidden_dim = int(base_embedding.weight.shape[-1])
    probe = FixedBinnedEmbeddingProbe(
        hidden_dim,
        statistics,
        rows_per_bin=rows_per_bin,
        seed=seed,
    ).to(device=base_embedding.weight.device, dtype=base_embedding.weight.dtype)
    wrapped = ProbedEmbedding(base_embedding, probe)
    setattr(parent, attribute, wrapped)

    embedding_path = None
    for name, module in model_wrapper.model.named_modules():
        if module is wrapped:
            embedding_path = name
            break
    if embedding_path is None:
        raise RuntimeError("Installed PEFTLeak probe is not registered in the actual model module tree.")

    names = tuple(f"{embedding_path}.peftleak_probe.{name}" for name, _ in probe.named_parameters())
    parameters = tuple(parameter for _, parameter in probe.named_parameters())
    if len(names) != len(parameters) or not names:
        raise RuntimeError("Installed PEFTLeak probe has an invalid parameter inventory.")
    return InstalledEmbeddingProbe(
        probe=probe,
        wrapped_embedding=wrapped,
        embedding_path=embedding_path,
        parameter_names=names,
        parameters=parameters,
    )


def apply_probe_defense(
    gradients: Sequence[torch.Tensor],
    parameter_names: Sequence[str],
    args,
) -> tuple[torch.Tensor, ...]:
    defense = str(getattr(args, "defense", "none"))
    if defense not in ADAPTER_RATIO_DEFENSES:
        raise NotImplementedError(
            f"Adapter ratio v2 supports defenses {sorted(ADAPTER_RATIO_DEFENSES)}; got {defense!r}."
        )
    seed = int(getattr(args, "rng_seed", 0)) + 1_000_003 * int(getattr(args, "defense_rng_step", 0) or 0)
    raw = tuple(gradients)
    if defense == "none":
        return raw
    if defense == "noise":
        sigma = getattr(args, "defense_noise", None)
        if sigma is None:
            raise ValueError("--defense noise requires --defense_noise.")
        return tuple(noise_injection(raw, float(sigma), seed=seed))
    if defense == "dpsgd":
        sigma = getattr(args, "defense_noise", None)
        if sigma is None:
            raise ValueError("--defense dpsgd requires --defense_noise.")
        return tuple(
            dpsgd_defense(
                [raw],
                float(getattr(args, "defense_clip_norm", 1.0)),
                float(sigma),
                seed=seed,
            )
        )
    if defense == "topk":
        return tuple(topk_sparsification(raw, float(getattr(args, "defense_topk_ratio", 0.1))))
    if defense == "compression":
        return tuple(gradient_compression(raw, int(getattr(args, "defense_n_bits", 8)), seed=seed))
    return tuple(apply_lrb_defense(raw, args, layer_names=list(parameter_names)))


def compute_probe_gradient_observation(
    model_wrapper,
    batch: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    installed: InstalledEmbeddingProbe,
    args,
) -> ProbeGradientObservation:
    model_wrapper.model.eval()
    model_wrapper.model.zero_grad(set_to_none=True)
    outputs = model_wrapper.model(**batch, labels=labels.view(-1).long())
    loss = outputs.loss
    gradients = torch.autograd.grad(loss, installed.parameters, allow_unused=True)
    raw = tuple(
        torch.zeros_like(parameter) if gradient is None else gradient.detach()
        for parameter, gradient in zip(installed.parameters, gradients)
    )
    observed = apply_probe_defense(raw, installed.parameter_names, args)
    return ProbeGradientObservation(
        raw_gradients=raw,
        observed_gradients=tuple(gradient.detach() for gradient in observed),
        parameter_names=installed.parameter_names,
        logits=outputs.logits.detach(),
        loss=float(loss.detach().item()),
    )


_PROBE_PARAMETER_RE = re.compile(r"\.positions\.(\d+)\.(weight|bias)$")


def _recover_bin_vector(
    weight_gradient: torch.Tensor,
    bias_gradient: torch.Tensor,
    bin_index: int,
    *,
    eps: float,
) -> tuple[torch.Tensor | None, float]:
    weight_rows = weight_gradient[bin_index].float()
    bias_rows = bias_gradient[bin_index].float()
    candidates: list[torch.Tensor] = []
    for row in range(int(bias_rows.shape[0])):
        denominator = bias_rows[row]
        if bool(denominator.detach().abs() > eps):
            candidates.append(weight_rows[row] / denominator)
    for row in range(max(0, int(bias_rows.shape[0]) - 1)):
        denominator = bias_rows[row] - bias_rows[row + 1]
        if bool(denominator.detach().abs() > eps):
            candidates.append((weight_rows[row] - weight_rows[row + 1]) / denominator)
    signal = float(bias_rows.detach().abs().max().item())
    if not candidates:
        return None, signal
    return torch.stack(candidates, dim=0).median(dim=0).values, signal


def recover_tokens_from_probe_gradients(
    gradients: Sequence[torch.Tensor],
    parameter_names: Sequence[str],
    token_embedding_matrix: torch.Tensor,
    *,
    max_positions: int,
    num_bins: int,
    ignored_token_ids: Iterable[int] | None = None,
    fallback_token_id: int = 0,
    eps: float = 1e-12,
) -> dict[str, object]:
    """Decode solely from observed gradients, fixed model metadata, and declared side information."""

    pairs: dict[int, dict[str, torch.Tensor]] = {}
    for gradient, name in zip(gradients, parameter_names):
        match = _PROBE_PARAMETER_RE.search(str(name))
        if match is None:
            continue
        pairs.setdefault(int(match.group(1)), {})[match.group(2)] = gradient

    predicted_ids = [int(fallback_token_id)] * int(max_positions)
    predicted_scores = [float("inf")] * int(max_positions)
    observed_bins: dict[int, int] = {}
    recovered_positions = 0
    for position in range(int(max_positions)):
        pair = pairs.get(position, {})
        if "weight" not in pair or "bias" not in pair:
            continue
        candidates: list[tuple[float, int, torch.Tensor]] = []
        for bin_index in range(int(num_bins)):
            vector, signal = _recover_bin_vector(pair["weight"], pair["bias"], bin_index, eps=eps)
            if vector is not None:
                candidates.append((signal, bin_index, vector))
        if not candidates:
            continue
        _, selected_bin, recovered = max(candidates, key=lambda item: item[0])
        token_ids, scores = nearest_token_ids(
            recovered.unsqueeze(0),
            token_embedding_matrix,
            unused_token_ids=ignored_token_ids,
            metric="cos",
        )
        predicted_ids[position] = int(token_ids.item())
        predicted_scores[position] = float(scores.item())
        observed_bins[position] = int(selected_bin)
        recovered_positions += 1

    return {
        "predicted_ids": [predicted_ids],
        "predicted_scores": [predicted_scores],
        "observed_bins": observed_bins,
        "recovered_position_count": int(recovered_positions),
        "gradient_count": int(len(parameter_names)),
        "gradient_names": list(parameter_names),
        "decoder_private_routing": False,
        "reportable": True,
    }


def token_recovery_accuracy(
    predicted_ids: Sequence[Sequence[int]],
    reference_ids: Sequence[Sequence[int]],
    *,
    ignored_token_ids: Iterable[int] | None = None,
    reference_mask: Sequence[Sequence[int]] | None = None,
) -> float:
    ignored = {int(value) for value in (ignored_token_ids or []) if value is not None}
    recovered = 0
    total = 0
    for sample_index, reference in enumerate(reference_ids):
        prediction = predicted_ids[sample_index] if sample_index < len(predicted_ids) else []
        mask = None if reference_mask is None else reference_mask[sample_index]
        for position, token in enumerate(reference):
            if mask is not None and (position >= len(mask) or not int(mask[position])):
                continue
            if int(token) in ignored:
                continue
            total += 1
            recovered += int(position < len(prediction) and int(prediction[position]) == int(token))
    return float(recovered / total) if total else 0.0


def select_shared_lora_parameters(model_wrapper) -> tuple[tuple[nn.Parameter, ...], tuple[str, ...]]:
    selected_parameters: list[nn.Parameter] = []
    selected_names: list[str] = []
    for name, parameter in model_wrapper.model.named_parameters():
        lower = name.lower()
        if not parameter.requires_grad or "lora_" not in lower or "modules_to_save" in lower:
            continue
        selected_parameters.append(parameter)
        selected_names.append(name)
    if not selected_parameters:
        raise ValueError("No shared LoRA parameters were found after excluding modules_to_save heads.")
    return tuple(selected_parameters), tuple(selected_names)


def _compression_with_straight_through(
    gradients: Sequence[torch.Tensor],
    *,
    n_bits: int,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    quantized = gradient_compression(tuple(gradients), n_bits, seed=seed)
    return tuple(
        gradient + (compressed.to(device=gradient.device, dtype=gradient.dtype) - gradient).detach()
        for gradient, compressed in zip(gradients, quantized)
    )


def observe_lora_gradients(
    gradients: Sequence[torch.Tensor],
    parameter_names: Sequence[str],
    args,
    *,
    differentiable: bool,
) -> tuple[torch.Tensor, ...]:
    defense = str(getattr(args, "defense", "none"))
    if defense not in LORA_OPT_DEFENSES:
        raise NotImplementedError(
            f"Defense-aware LoRA optimization v2 supports {sorted(LORA_OPT_DEFENSES)}; got {defense!r}."
        )
    raw = tuple(gradients)
    if defense == "none":
        return raw
    if defense == "topk":
        return tuple(topk_sparsification(raw, float(getattr(args, "defense_topk_ratio", 0.1))))
    if defense == "compression":
        n_bits = int(getattr(args, "defense_n_bits", 8))
        seed = int(getattr(args, "rng_seed", 0))
        if differentiable:
            return _compression_with_straight_through(raw, n_bits=n_bits, seed=seed)
        return tuple(gradient_compression(raw, n_bits, seed=seed))
    return tuple(apply_lrb_defense(raw, args, layer_names=list(parameter_names)))


def _label_assignments(batch_size: int, candidates: Sequence[int], *, maximum: int = 64):
    total = len(candidates) ** int(batch_size)
    if total > maximum:
        raise ValueError(f"Label search requires {total} assignments, exceeding the v2 limit {maximum}.")
    return [torch.tensor(values, dtype=torch.long) for values in itertools.product(candidates, repeat=batch_size)]


def _gradient_match_loss(
    candidate: Sequence[torch.Tensor],
    target: Sequence[torch.Tensor],
    *,
    mode: str,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for candidate_tensor, target_tensor in zip(candidate, target):
        candidate_flat = candidate_tensor.float().reshape(-1)
        target_flat = target_tensor.detach().to(device=candidate_flat.device).float().reshape(-1)
        if mode == "mse":
            losses.append(F.mse_loss(candidate_flat, target_flat))
        elif mode == "normalized_mse":
            denominator = target_flat.pow(2).mean().clamp_min(1e-12)
            losses.append(F.mse_loss(candidate_flat, target_flat) / denominator)
        elif mode == "cosine":
            losses.append(1.0 - F.cosine_similarity(candidate_flat, target_flat, dim=0))
        else:
            raise ValueError("match_loss must be mse, normalized_mse, or cosine.")
    if not losses:
        raise ValueError("No LoRA gradients were available for matching.")
    return torch.stack([loss.reshape(()) for loss in losses]).mean()


def optimize_lora_embeddings_defense_aware(
    *,
    model_wrapper,
    batch: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    args,
    steps: int,
    lr: float,
    restarts: int,
    match_loss: str,
    known_labels: bool,
    label_candidates: Sequence[int] | None,
    ignored_token_ids: Iterable[int] | None,
) -> dict[str, object]:
    """Optimization-based PEFT inversion; this is not the original PEFTLeak ratio attack."""

    parameters, parameter_names = select_shared_lora_parameters(model_wrapper)
    model_wrapper.model.eval()
    model_wrapper.model.zero_grad(set_to_none=True)
    outputs = model_wrapper.model(**batch, labels=labels.view(-1).long())
    target_raw = torch.autograd.grad(outputs.loss, parameters, allow_unused=False)
    target_observed = tuple(
        gradient.detach()
        for gradient in observe_lora_gradients(target_raw, parameter_names, args, differentiable=False)
    )

    device = batch["input_ids"].device
    token_table = get_token_embedding_matrix(model_wrapper).detach().to(device=device).float()
    batch_size, sequence_length = (int(value) for value in batch["input_ids"].shape)
    hidden_dim = int(token_table.shape[-1])
    table_mean = token_table.mean(dim=0).view(1, 1, hidden_dim)
    table_std = token_table.std(dim=0, unbiased=False).clamp_min(1e-6).view(1, 1, hidden_dim)

    if known_labels:
        assignments = [labels.view(-1).long().detach().cpu()]
    else:
        candidates = list(label_candidates or range(int(model_wrapper.model.config.num_labels)))
        assignments = _label_assignments(batch_size, candidates)

    best_loss = float("inf")
    best_embeddings = None
    best_labels = None
    loss_history: list[float] = []
    initial_loss = None
    base_seed = int(getattr(args, "rng_seed", 0))

    for assignment_index, assignment in enumerate(assignments):
        candidate_labels = assignment.to(device=device)
        for restart in range(max(1, int(restarts))):
            generator = torch.Generator(device=device)
            generator.manual_seed(base_seed + 9176 * assignment_index + 1009 * restart)
            noise = torch.randn(
                batch_size,
                sequence_length,
                hidden_dim,
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            dummy = nn.Parameter(table_mean + noise * table_std)
            optimizer = torch.optim.Adam([dummy], lr=float(lr))

            for _ in range(max(1, int(steps))):
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model_wrapper._seq_class_logits_from_embeds(batch, dummy)
                task_loss = F.cross_entropy(logits, candidate_labels)
                candidate_raw = torch.autograd.grad(
                    task_loss,
                    parameters,
                    create_graph=True,
                    allow_unused=False,
                )
                candidate_observed = observe_lora_gradients(
                    candidate_raw,
                    parameter_names,
                    args,
                    differentiable=True,
                )
                total = _gradient_match_loss(candidate_observed, target_observed, mode=match_loss)
                gradient_dummy, = torch.autograd.grad(total, dummy, allow_unused=False)
                dummy.grad = gradient_dummy
                current = float(total.detach().item())
                if initial_loss is None:
                    initial_loss = current
                loss_history.append(current)
                if current < best_loss:
                    best_loss = current
                    best_embeddings = dummy.detach().clone()
                    best_labels = candidate_labels.detach().cpu().tolist()
                optimizer.step()

    if best_embeddings is None:
        raise RuntimeError("Defense-aware LoRA inversion did not produce a candidate sequence.")

    predicted_ids: list[list[int]] = []
    predicted_scores: list[list[float]] = []
    for sample_index in range(batch_size):
        ids, scores = nearest_token_ids(
            best_embeddings[sample_index],
            token_table,
            unused_token_ids=ignored_token_ids,
            metric="cos",
        )
        predicted_ids.append([int(value) for value in ids.tolist()])
        predicted_scores.append([float(value) for value in scores.tolist()])

    reference_mask = batch.get("attention_mask")
    rec_token = token_recovery_accuracy(
        predicted_ids,
        batch["input_ids"].detach().cpu().tolist(),
        ignored_token_ids=ignored_token_ids,
        reference_mask=None if reference_mask is None else reference_mask.detach().cpu().tolist(),
    )
    return {
        "predicted_ids": predicted_ids,
        "predicted_scores": predicted_scores,
        "rec_token_mean": float(rec_token),
        "loss": float(best_loss),
        "initial_loss": initial_loss,
        "loss_reduction": None if initial_loss is None else float(initial_loss - best_loss),
        "loss_history": loss_history,
        "best_label": best_labels,
        "label_mode": "known" if known_labels else "search",
        "selected_gradient_count": len(parameter_names),
        "selected_gradient_names": list(parameter_names),
        "sequence_length": sequence_length,
        "defense_aware": True,
        "attack_family": "optimization_based_peft_gradient_inversion",
    }
