from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from attacks.peftleak_text import nearest_token_ids, token_recovery_ratio


@dataclass(frozen=True)
class TextTokenStatistics:
    mean: torch.Tensor
    std: torch.Tensor
    bin_edges: torch.Tensor
    global_bin_edges: torch.Tensor
    num_sequences: int
    num_tokens: int
    num_bins: int
    max_seq_len: int
    target: str = "input_embedding"


@dataclass(frozen=True)
class TextRatioRoutingInfo:
    route: str
    slot_keys: list[list[str | None]]
    slot_counts: dict[str, int]
    routed_token_count: int
    colliding_token_count: int
    collision_rate: float
    reportable: bool


@dataclass(frozen=True)
class TextRatioGradientResult:
    grads: tuple[torch.Tensor | None, ...]
    names: list[str]
    input_vectors: torch.Tensor
    non_token_vectors: torch.Tensor
    routing: TextRatioRoutingInfo
    logits: torch.Tensor
    loss: float


class RatioRecoveryDegenerateError(ValueError):
    """Raised when a ratio slot has no usable bias-gradient denominator."""


_SLOT_RE = re.compile(r"text_ratio_adapter\.slot_(.+)\.(weight|bias)$")


def _bert_backbone(model_wrapper):
    base = getattr(model_wrapper, "base_model", None)
    if base is not None and hasattr(base, "bert"):
        return base.bert
    model = getattr(model_wrapper, "model", None)
    if model is not None and hasattr(model, "bert"):
        return model.bert
    raise NotImplementedError("PEFTLeak text ratio supports BERT and GPT-2 seq_class wrappers in v1.")


def _gpt2_backbone(model_wrapper):
    base = getattr(model_wrapper, "base_model", None)
    if base is not None and hasattr(base, "transformer"):
        return base.transformer
    model = getattr(model_wrapper, "model", None)
    if model is not None and hasattr(model, "transformer"):
        return model.transformer
    raise NotImplementedError("PEFTLeak text ratio supports BERT and GPT-2 seq_class wrappers in v1.")


def text_input_vectors(model_wrapper, batch, *, target: str = "input_embedding") -> tuple[torch.Tensor, torch.Tensor]:
    """Return PEFTLeak-style input vectors and the non-token component to subtract before decoding."""

    if target != "input_embedding":
        raise NotImplementedError("v1 text ratio attack supports --peftleak_ratio_target input_embedding only.")

    model_path = str(getattr(model_wrapper.args, "model_path", ""))
    input_ids = batch["input_ids"]
    if model_path in {"bert-base-uncased"}:
        bert = _bert_backbone(model_wrapper)
        embeddings = bert.embeddings
        token_type_ids = batch.get("token_type_ids", torch.zeros_like(input_ids))
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        word = embeddings.word_embeddings(input_ids)
        pos = embeddings.position_embeddings(position_ids)
        tok = embeddings.token_type_embeddings(token_type_ids)
        return word + pos + tok, pos + tok

    if model_path in {"gpt2", "openai-community/gpt2-large"}:
        transformer = _gpt2_backbone(model_wrapper)
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        word = transformer.wte(input_ids)
        pos = transformer.wpe(position_ids)
        return word + pos, pos

    raise NotImplementedError("PEFTLeak text ratio supports bert-base-uncased and GPT-2 model families in v1.")


def _valid_mask(batch) -> torch.Tensor:
    input_ids = batch["input_ids"]
    mask = batch.get("attention_mask")
    if mask is None:
        return torch.ones_like(input_ids, dtype=torch.bool)
    return mask.to(device=input_ids.device).bool()


def _score_vectors(vectors: torch.Tensor, stats: TextTokenStatistics | None = None) -> torch.Tensor:
    if stats is None:
        return vectors.float().mean(dim=-1)
    mean = stats.mean.to(device=vectors.device, dtype=vectors.dtype).view(1, 1, -1)
    std = stats.std.to(device=vectors.device, dtype=vectors.dtype).clamp_min(1e-12).view(1, 1, -1)
    return ((vectors - mean) / std).float().mean(dim=-1)


def build_text_token_statistics(
    model_wrapper,
    public_batches: Sequence[Mapping[str, torch.Tensor]],
    *,
    num_bins: int,
    target: str = "input_embedding",
) -> TextTokenStatistics:
    if int(num_bins) < 1:
        raise ValueError("num_bins must be positive.")
    vectors_by_position: dict[int, list[torch.Tensor]] = {}
    all_vectors: list[torch.Tensor] = []
    num_sequences = 0
    for batch in public_batches:
        vectors, _ = text_input_vectors(model_wrapper, batch, target=target)
        mask = _valid_mask(batch)
        num_sequences += int(vectors.shape[0])
        for pos in range(vectors.shape[1]):
            selected = vectors[:, pos][mask[:, pos]]
            if selected.numel() == 0:
                continue
            vectors_by_position.setdefault(pos, []).append(selected.detach().float().cpu())
            all_vectors.append(selected.detach().float().cpu())
    if not all_vectors:
        raise ValueError("Cannot build text token public statistics from an empty public set.")

    flat = torch.cat(all_vectors, dim=0)
    mean = flat.mean(dim=0)
    std = flat.std(dim=0, unbiased=False).clamp_min(1e-12)

    max_seq_len = max(vectors_by_position) + 1
    edge_count = max(0, int(num_bins) - 1)
    bin_edges = torch.empty(max_seq_len, edge_count, dtype=torch.float32)
    global_scores = ((flat - mean.view(1, -1)) / std.view(1, -1)).mean(dim=-1)
    quantiles = torch.linspace(0, 1, int(num_bins) + 1, dtype=torch.float32)[1:-1]
    global_edges = torch.quantile(global_scores, quantiles) if edge_count else torch.empty(0)

    for pos in range(max_seq_len):
        parts = vectors_by_position.get(pos)
        if not parts:
            bin_edges[pos] = global_edges
            continue
        pos_vectors = torch.cat(parts, dim=0)
        scores = ((pos_vectors - mean.view(1, -1)) / std.view(1, -1)).mean(dim=-1)
        bin_edges[pos] = torch.quantile(scores, quantiles) if edge_count else torch.empty(0)

    return TextTokenStatistics(
        mean=mean,
        std=std,
        bin_edges=bin_edges,
        global_bin_edges=global_edges,
        num_sequences=num_sequences,
        num_tokens=int(flat.shape[0]),
        num_bins=int(num_bins),
        max_seq_len=max_seq_len,
        target=target,
    )


def _bucket_for_score(score: torch.Tensor, edges: torch.Tensor) -> int:
    if edges.numel() == 0:
        return 0
    return int(torch.bucketize(score.detach().float().cpu(), edges.detach().float().cpu(), right=False).item())


def route_text_tokens(
    vectors: torch.Tensor,
    batch,
    *,
    stats: TextTokenStatistics | None,
    route: str,
) -> TextRatioRoutingInfo:
    if route not in {"oracle", "public_bins"}:
        raise ValueError("route must be 'oracle' or 'public_bins'.")

    mask = _valid_mask(batch)
    scores = _score_vectors(vectors, stats)
    slot_keys: list[list[str | None]] = []
    counts: dict[str, int] = {}
    routed = 0
    for sample_idx in range(vectors.shape[0]):
        row: list[str | None] = []
        for pos in range(vectors.shape[1]):
            if not bool(mask[sample_idx, pos]):
                row.append(None)
                continue
            if route == "oracle":
                key = f"s{sample_idx}_p{pos}"
            else:
                if stats is None:
                    raise ValueError("public_bins routing requires TextTokenStatistics.")
                edges = stats.bin_edges[pos] if pos < stats.bin_edges.shape[0] else stats.global_bin_edges
                bucket = _bucket_for_score(scores[sample_idx, pos], edges)
                key = f"p{pos}_b{bucket}"
            counts[key] = counts.get(key, 0) + 1
            routed += 1
            row.append(key)
        slot_keys.append(row)

    colliding = sum(count for count in counts.values() if count > 1)
    return TextRatioRoutingInfo(
        route=route,
        slot_keys=slot_keys,
        slot_counts=counts,
        routed_token_count=routed,
        colliding_token_count=colliding,
        collision_rate=float(colliding / max(1, routed)),
        reportable=(route != "oracle"),
    )


def _safe_slot_name(slot: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", slot)


class MaliciousTextTokenAdapter(nn.Module):
    """PEFTLeak-style malicious adapter probes keyed by text token routes."""

    def __init__(self, hidden_dim: int, slots: Iterable[str], *, rows_per_slot: int = 4, seed: int = 0):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if rows_per_slot < 2:
            raise ValueError("rows_per_slot must be at least 2 for ratio recovery.")
        self.hidden_dim = int(hidden_dim)
        self.rows_per_slot = int(rows_per_slot)
        self.slot_to_module: dict[str, str] = {}
        self.modules_by_slot = nn.ModuleDict()
        for index, slot in enumerate(sorted(set(slots))):
            module_name = f"{index}_{_safe_slot_name(slot)}"
            layer = nn.Linear(self.hidden_dim, self.rows_per_slot, bias=True)
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed) + 104729 * (index + 1))
            with torch.no_grad():
                layer.weight.copy_(torch.randn(layer.weight.shape, generator=gen) * 1e-3)
                coeff = torch.linspace(-0.5, 0.5, self.rows_per_slot)
                coeff[coeff == 0] = 0.25
                layer.bias.copy_(coeff)
            self.slot_to_module[slot] = module_name
            self.modules_by_slot[module_name] = layer
        coeff = torch.arange(1, self.rows_per_slot + 1, dtype=torch.float32)
        self.register_buffer("coefficients", coeff)

    def forward(self, vectors: torch.Tensor, slot_keys: Sequence[Sequence[str | None]], *, num_labels: int) -> torch.Tensor:
        coeff = self.coefficients.to(device=vectors.device, dtype=vectors.dtype)
        sample_scores: list[torch.Tensor] = []
        for sample_idx, row in enumerate(slot_keys):
            score = vectors.new_tensor(0.0)
            for pos, slot in enumerate(row):
                if slot is None:
                    continue
                module = self.modules_by_slot[self.slot_to_module[slot]]
                score = score + (module(vectors[sample_idx, pos]) * coeff).sum()
            sample_scores.append(score)
        scores = torch.stack(sample_scores, dim=0) if sample_scores else vectors.new_zeros((0,))
        if int(num_labels) <= 1:
            return scores.unsqueeze(1)
        tail = torch.zeros(vectors.shape[0], int(num_labels) - 1, device=vectors.device, dtype=vectors.dtype)
        return torch.cat([scores.unsqueeze(1), tail], dim=1)

    def parameter_names(self) -> list[str]:
        out: list[str] = []
        for slot in sorted(self.slot_to_module):
            safe = _safe_slot_name(slot)
            out.extend([f"text_ratio_adapter.slot_{safe}.weight", f"text_ratio_adapter.slot_{safe}.bias"])
        return out

    def parameters_for_grad(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for slot in sorted(self.slot_to_module):
            module = self.modules_by_slot[self.slot_to_module[slot]]
            params.extend([module.weight, module.bias])
        return params


def build_text_ratio_gradients(
    model_wrapper,
    batch,
    labels,
    *,
    stats: TextTokenStatistics | None,
    route: str,
    target: str = "input_embedding",
    rows_per_slot: int = 4,
    seed: int = 0,
) -> TextRatioGradientResult:
    vectors, non_token = text_input_vectors(model_wrapper, batch, target=target)
    vectors = vectors.detach()
    non_token = non_token.detach()
    routing = route_text_tokens(vectors, batch, stats=stats, route=route)
    slots = sorted(routing.slot_counts)
    if not slots:
        raise ValueError("No text tokens were routed to malicious adapter probes.")
    adapter = MaliciousTextTokenAdapter(
        vectors.shape[-1],
        slots,
        rows_per_slot=rows_per_slot,
        seed=seed,
    ).to(device=vectors.device, dtype=vectors.dtype)
    adapter.eval()
    num_labels = int(getattr(getattr(model_wrapper, "model", None).config, "num_labels", 2))
    logits = adapter(vectors, routing.slot_keys, num_labels=num_labels)
    loss = F.cross_entropy(logits, labels.view(-1).long().to(device=vectors.device))
    grads = torch.autograd.grad(loss, adapter.parameters_for_grad(), allow_unused=True)
    return TextRatioGradientResult(
        grads=tuple(None if grad is None else grad.detach() for grad in grads),
        names=adapter.parameter_names(),
        input_vectors=vectors,
        non_token_vectors=non_token,
        routing=routing,
        logits=logits.detach(),
        loss=float(loss.detach().item()),
    )


def recover_hidden_from_ratio_pair(
    weight_grad: torch.Tensor,
    bias_grad: torch.Tensor,
    *,
    eps: float = 1e-12,
    aggregation: str = "median",
) -> torch.Tensor:
    if weight_grad.ndim != 2:
        raise ValueError("weight_grad must have shape [rows, hidden_dim].")
    if bias_grad.ndim != 1:
        raise ValueError("bias_grad must have shape [rows].")
    if weight_grad.shape[0] != bias_grad.shape[0]:
        raise ValueError("weight_grad and bias_grad rows do not match.")
    candidates = []
    for row in range(weight_grad.shape[0] - 1):
        denom = bias_grad[row] - bias_grad[row + 1]
        if bool(denom.detach().abs() > eps):
            candidates.append((weight_grad[row] - weight_grad[row + 1]).float() / denom.float())
    if not candidates:
        raise RatioRecoveryDegenerateError(
            "No nonzero adjacent bias-gradient differences were available for ratio recovery."
        )
    stacked = torch.stack(candidates, dim=0)
    if aggregation == "median":
        return stacked.median(dim=0).values
    if aggregation == "mean":
        return stacked.mean(dim=0)
    raise ValueError("aggregation must be 'median' or 'mean'.")


def recover_hidden_slots_from_ratio_grads(
    grads: Sequence[torch.Tensor | None],
    names: Sequence[str],
    *,
    eps: float = 1e-12,
    aggregation: str = "median",
) -> dict[str, torch.Tensor]:
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for grad, name in zip(grads, names):
        if grad is None:
            continue
        match = _SLOT_RE.search(str(name))
        if match is None:
            continue
        slot = match.group(1)
        pairs.setdefault(slot, {})[match.group(2)] = grad
    recovered: dict[str, torch.Tensor] = {}
    for slot, pair in pairs.items():
        if "weight" not in pair or "bias" not in pair:
            continue
        try:
            recovered[slot] = recover_hidden_from_ratio_pair(
                pair["weight"],
                pair["bias"],
                eps=eps,
                aggregation=aggregation,
            )
        except RatioRecoveryDegenerateError:
            continue
    return recovered


def decode_ratio_recovery(
    *,
    ratio_result: TextRatioGradientResult,
    defended_grads: Sequence[torch.Tensor | None],
    token_embedding_matrix: torch.Tensor,
    batch,
    ignored_token_ids: Iterable[int] | None = None,
    reference_mask: Sequence[Sequence[int]] | None = None,
) -> dict[str, object]:
    recovered_slots = recover_hidden_slots_from_ratio_grads(defended_grads, ratio_result.names)
    input_ids = batch["input_ids"].detach().cpu().tolist()
    predicted_ids: list[list[int]] = []
    predicted_scores: list[list[float]] = []
    recovered_hidden_count = 0
    pad_fallback = 0
    if ignored_token_ids:
        for candidate in ignored_token_ids:
            if candidate is not None:
                pad_fallback = int(candidate)
                break

    for sample_idx, row in enumerate(ratio_result.routing.slot_keys):
        sample_ids: list[int] = []
        sample_scores: list[float] = []
        for pos, slot in enumerate(row):
            if slot is None:
                sample_ids.append(int(input_ids[sample_idx][pos]))
                sample_scores.append(0.0)
                continue
            safe = _safe_slot_name(slot)
            recovered = recovered_slots.get(safe)
            if recovered is None:
                sample_ids.append(pad_fallback)
                sample_scores.append(float("inf"))
                continue
            recovered_hidden_count += 1
            token_vector = recovered.to(
                device=ratio_result.non_token_vectors.device,
                dtype=ratio_result.non_token_vectors.dtype,
            ) - ratio_result.non_token_vectors[sample_idx, pos]
            ids, scores = nearest_token_ids(
                token_vector.unsqueeze(0),
                token_embedding_matrix,
                unused_token_ids=ignored_token_ids,
                metric="cos",
            )
            sample_ids.append(int(ids.item()))
            sample_scores.append(float(scores.item()))
        predicted_ids.append(sample_ids)
        predicted_scores.append(sample_scores)

    rec_token_mean = token_recovery_ratio(
        predicted_ids,
        input_ids,
        ignored_token_ids=ignored_token_ids,
        reference_mask=reference_mask,
    )
    return {
        "predicted_ids": predicted_ids,
        "predicted_scores": predicted_scores,
        "rec_token_mean": float(rec_token_mean),
        "recovered_hidden_count": int(recovered_hidden_count),
        "recovered_slot_count": int(len(recovered_slots)),
        "routed_token_count": int(ratio_result.routing.routed_token_count),
        "collision_rate": float(ratio_result.routing.collision_rate),
        "colliding_token_count": int(ratio_result.routing.colliding_token_count),
        "slot_count": int(len(ratio_result.routing.slot_counts)),
        "route": ratio_result.routing.route,
        "reportable": bool(ratio_result.routing.reportable),
        "loss": float(ratio_result.loss),
        "gradient_count": int(len(ratio_result.names)),
        "gradient_names": ratio_result.names,
    }
