from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GradientMatchResult:
    loss: float
    token_ids: list[list[int]]
    token_scores: list[list[float]]
    rec_token_mean: float
    rouge_like: float | None = None


def _is_adapter_name(name: str, peft_method: str) -> bool:
    lower = name.lower()
    if "modules_to_save" in lower:
        return False
    peft_method = str(peft_method or "lora").strip().lower().replace("-", "_")
    if peft_method == "lora":
        return "lora_" in lower
    if peft_method == "ia3":
        return "ia3" in lower
    if peft_method in {"adapter", "double_seq_bn", "seq_bn", "par_bn", "par_seq_bn", "houlsby", "pfeiffer"}:
        return any(
            part in lower
            for part in (
                "adapter_down",
                "adapter_up",
                ".adapters.",
                ".adapter.",
                "down_proj",
                "up_proj",
            )
        )
    raise ValueError(f"Unsupported PEFT method for adapter filtering: {peft_method!r}")


def select_peft_gradient_tensors(
    grads: Sequence[torch.Tensor | None],
    parameter_names: Sequence[str],
    peft_method: str,
) -> tuple[list[int], list[str]]:
    indices: list[int] = []
    names: list[str] = []
    for idx, name in enumerate(parameter_names):
        if idx >= len(grads):
            break
        if grads[idx] is None:
            continue
        if not _is_adapter_name(name, peft_method):
            continue
        indices.append(idx)
        names.append(name)
    return indices, names


def flatten_selected_grads(
    grads: Sequence[torch.Tensor | None],
    parameter_names: Sequence[str],
    peft_method: str,
) -> tuple[torch.Tensor, list[int], list[str]]:
    indices, names = select_peft_gradient_tensors(grads, parameter_names, peft_method)
    parts: list[torch.Tensor] = []
    for idx in indices:
        grad = grads[idx]
        if grad is None:
            continue
        parts.append(grad.detach().float().reshape(-1))
    if not parts:
        raise ValueError("No PEFT adapter gradients were selected for matching.")
    return torch.cat(parts), indices, names


def get_token_embedding_matrix(model_wrapper) -> torch.Tensor:
    candidates = (
        getattr(model_wrapper, "model", None),
        getattr(model_wrapper, "base_model", None),
    )
    for module in candidates:
        if module is None:
            continue
        get_embeddings = getattr(module, "get_input_embeddings", None)
        if callable(get_embeddings):
            emb = get_embeddings()
            if emb is not None and hasattr(emb, "weight"):
                return emb.weight.detach()

    # Family-specific fallbacks for wrapped PEFT models.
    if hasattr(model_wrapper, "base_model") and hasattr(model_wrapper.base_model, "transformer"):
        return model_wrapper.base_model.transformer.wte.weight.detach()
    if hasattr(model_wrapper, "base_model") and hasattr(model_wrapper.base_model, "model"):
        return model_wrapper.base_model.model.embed_tokens.weight.detach()
    if hasattr(model_wrapper, "base_model") and hasattr(model_wrapper.base_model, "bert"):
        return model_wrapper.base_model.bert.embeddings.word_embeddings.weight.detach()
    raise ValueError("Could not resolve token embedding matrix from model_wrapper.")


def build_dummy_embedding_prior(model_wrapper, batch, *, seed: int = 0) -> torch.Tensor:
    table = get_token_embedding_matrix(model_wrapper).detach().float()
    mean = table.mean(dim=0, keepdim=True)
    std = table.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    shape = model_wrapper._seq_class_input_embeds(batch).shape
    gen = torch.Generator(device=table.device)
    gen.manual_seed(int(seed))
    noise = torch.randn(shape, device=table.device, dtype=table.dtype, generator=gen)
    return mean.view(1, 1, -1).expand(shape[0], shape[1], -1) + noise * std.view(1, 1, -1)


def _normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def _valid_token_ids(token_ids: Iterable[object] | None, *, vocab_size: int | None = None) -> list[int]:
    valid: list[int] = []
    seen: set[int] = set()
    for token_id in token_ids or []:
        if token_id is None:
            continue
        try:
            value = int(token_id)
        except (TypeError, ValueError, OverflowError):
            continue
        if vocab_size is not None and not (0 <= value < vocab_size):
            continue
        if value in seen:
            continue
        seen.add(value)
        valid.append(value)
    return valid

def _reduce_target_like(grad: torch.Tensor, *, detach: bool = True) -> torch.Tensor:
    g = grad.detach() if detach else grad
    g = g.float()
    if g.ndim == 0:
        return g.reshape(1)
    if g.ndim == 1:
        return g
    if g.ndim == 2:
        return g
    return g.reshape(g.shape[0], -1)


def build_target_gradient_vector(
    grads: Sequence[torch.Tensor | None],
    parameter_names: Sequence[str],
    peft_method: str,
    *,
    detach: bool = True,
) -> tuple[torch.Tensor, list[int], list[str]]:
    indices, names = select_peft_gradient_tensors(grads, parameter_names, peft_method)
    parts: list[torch.Tensor] = []
    for idx in indices:
        grad = grads[idx]
        if grad is None:
            continue
        parts.append(_reduce_target_like(grad, detach=detach).reshape(-1))
    if not parts:
        raise ValueError("No PEFT adapter gradients were selected for matching.")
    return torch.cat(parts), indices, names


def nearest_token_ids(
    embeddings: torch.Tensor,
    token_embedding_matrix: torch.Tensor,
    *,
    unused_token_ids: Iterable[int] | None = None,
    metric: str = "cos",
) -> tuple[torch.Tensor, torch.Tensor]:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [seq, hidden].")
    if token_embedding_matrix.ndim != 2:
        raise ValueError("token_embedding_matrix must have shape [vocab, hidden].")

    emb = embeddings.detach().float()
    table = token_embedding_matrix.detach().to(device=emb.device).float()
    if metric == "cos":
        emb_n = _normalize_rows(emb)
        table_n = _normalize_rows(table)
        scores = -(emb_n @ table_n.t())
    elif metric == "l2":
        scores = torch.cdist(emb, table, p=2)
    else:
        raise ValueError(f"Unsupported metric: {metric!r}")

    valid_unused = _valid_token_ids(unused_token_ids, vocab_size=table.shape[0])
    if valid_unused:
        unused = torch.tensor(valid_unused, device=scores.device, dtype=torch.long)
        scores[:, unused] = float("inf")
    token_ids = scores.argmin(dim=-1)
    token_scores = scores.gather(1, token_ids.unsqueeze(1)).squeeze(1)
    return token_ids, token_scores


def token_recovery_ratio(
    predicted: Sequence[Sequence[int]],
    reference: Sequence[Sequence[int]],
    *,
    ignored_token_ids: Iterable[int] | None = None,
    reference_mask: Sequence[Sequence[int]] | None = None,
) -> float:
    ignored = set(_valid_token_ids(ignored_token_ids))
    total = 0
    recovered = 0
    for sample_idx, (pred, ref) in enumerate(zip(predicted, reference)):
        mask = None if reference_mask is None else reference_mask[sample_idx]
        n = min(len(pred), len(ref))
        for pos in range(n):
            if mask is not None and (pos >= len(mask) or int(mask[pos]) == 0):
                continue
            ref_tok = int(ref[pos])
            if ref_tok in ignored:
                continue
            total += 1
            recovered += int(int(pred[pos]) == ref_tok)
    if total <= 0:
        return 0.0
    return recovered / total


def _sequence_logits_from_embeddings(model_wrapper, batch, embeddings):
    logits, representation = model_wrapper._seq_class_logits_from_embeds(batch, embeddings)
    return logits, representation


def _iter_label_assignments(
    batch_size: int,
    candidates: Sequence[int],
    *,
    max_assignments: int = 64,
):
    if batch_size <= 0:
        return []
    values = [int(v) for v in candidates]
    if not values:
        return []
    total = len(values) ** batch_size
    if total > max_assignments:
        raise ValueError(
            "Label search space is too large for this lightweight FedLLM PEFT text implementation; "
            f"got {total} assignments for batch_size={batch_size}."
        )
    return [torch.tensor(assignment, dtype=torch.long) for assignment in itertools.product(values, repeat=batch_size)]


def optimize_text_embeddings(
    *,
    model_wrapper,
    batch,
    labels,
    target_grads: Sequence[torch.Tensor | None],
    parameter_names: Sequence[str],
    peft_method: str,
    steps: int = 60,
    lr: float = 0.1,
    tv_weight: float = 0.0,
    entropy_weight: float = 0.0,
    restarts: int = 1,
    match_loss: str = "normalized_mse",
    label_known: bool = True,
    label_candidates: Sequence[int] | None = None,
    ignored_token_ids: Iterable[int] | None = None,
    reference_mask: Sequence[Sequence[int]] | None = None,
) -> dict[str, object]:
    """Optimization-based gradient matching on PEFT adapter gradients.

    This is intentionally small and deterministic enough for semantic tests.
    """

    device = batch["input_ids"].device
    target_vec, selected_indices, selected_names = build_target_gradient_vector(
        target_grads,
        parameter_names,
        peft_method,
    )
    target_vec = target_vec.to(device=device)

    logits_shape = None
    best_loss = float("inf")
    best_ids = None
    best_scores = None
    best_label = None
    loss_history: list[float] = []

    def _gradient_match_loss(grad: Sequence[torch.Tensor | None]) -> torch.Tensor:
        if match_loss == "mse":
            grad_vec, _, _ = build_target_gradient_vector(grad, parameter_names, peft_method, detach=False)
            return F.mse_loss(grad_vec, target_vec)

        losses: list[torch.Tensor] = []
        for idx in selected_indices:
            if idx >= len(grad) or idx >= len(target_grads):
                continue
            grad_tensor = grad[idx]
            target_tensor = target_grads[idx]
            if grad_tensor is None or target_tensor is None:
                continue
            candidate = _reduce_target_like(grad_tensor, detach=False).reshape(-1)
            target = _reduce_target_like(target_tensor, detach=True).reshape(-1).to(device=device)
            if match_loss == "normalized_mse":
                denom = target.detach().float().pow(2).mean().clamp_min(1e-12)
                losses.append(F.mse_loss(candidate, target) / denom.to(device=candidate.device, dtype=candidate.dtype))
            elif match_loss == "cosine":
                losses.append(1.0 - F.cosine_similarity(candidate.float(), target.float(), dim=0))
            else:
                raise ValueError("match_loss must be one of: mse, normalized_mse, cosine.")
        if not losses:
            raise ValueError("No PEFT adapter gradients were selected for matching.")
        return torch.stack([loss.reshape(()) for loss in losses]).mean()

    def _loss_for_labels(dummy: torch.nn.Parameter, candidate_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, representation = _sequence_logits_from_embeddings(model_wrapper, batch, dummy)
        loss = F.cross_entropy(logits, candidate_labels.view(-1).long())
        grad = torch.autograd.grad(
            loss,
            model_wrapper.trainable_parameters(),
            create_graph=True,
            allow_unused=True,
        )
        grad_loss = _gradient_match_loss(grad)
        tv_loss = dummy[:, 1:, :].sub(dummy[:, :-1, :]).abs().mean() if dummy.shape[1] > 1 else dummy.new_tensor(0.0)
        entropy = torch.tensor(0.0, device=device)
        if entropy_weight > 0:
            probs = logits.softmax(dim=-1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()
        total = grad_loss + tv_weight * tv_loss - entropy_weight * entropy
        return total, logits, representation

    candidate_label_values: list[int]
    if label_known:
        candidate_label_values = [int(x) for x in labels.view(-1).tolist()]
    else:
        if label_candidates is not None:
            candidate_label_values = [int(x) for x in label_candidates]
        else:
            candidate_label_values = list(range(int(getattr(model_wrapper.model.config, "num_labels", 2))))

    if label_known:
        candidate_assignments = [torch.tensor(candidate_label_values, device=device, dtype=torch.long)]
    else:
        candidate_assignments = [assignment.to(device) for assignment in _iter_label_assignments(batch["input_ids"].shape[0], candidate_label_values)]

    base_seed = int(getattr(model_wrapper.args, "rng_seed", 0))
    for assignment_idx, candidate_labels in enumerate(candidate_assignments):
        for restart_idx in range(max(1, int(restarts))):
            inputs_embeds = build_dummy_embedding_prior(
                model_wrapper,
                batch,
                seed=base_seed + assignment_idx * 9176 + restart_idx * 1009,
            ).to(device=device)
            dummy = torch.nn.Parameter(inputs_embeds.clone())
            optimizer = torch.optim.Adam([dummy], lr=lr)
            for _ in range(max(1, int(steps))):
                optimizer.zero_grad(set_to_none=True)
                total, logits, _ = _loss_for_labels(dummy, candidate_labels)
                total.backward()
                optimizer.step()
                current_loss = float(total.detach().item())
                loss_history.append(current_loss)
                if current_loss < best_loss:
                    best_loss = current_loss
                    logits_shape = logits.shape
                    best_label = candidate_labels.detach().cpu().tolist()
                    best_logits = logits.detach()
                    best_representation = _sequence_logits_from_embeddings(model_wrapper, batch, dummy)[1].detach()
                    token_matrix = get_token_embedding_matrix(model_wrapper)
                    sample_ids = []
                    sample_scores = []
                    for sample_idx in range(dummy.shape[0]):
                        token_ids, token_scores = nearest_token_ids(
                            dummy.detach()[sample_idx],
                            token_matrix,
                            unused_token_ids=ignored_token_ids,
                            metric="cos",
                        )
                        sample_ids.append(token_ids.tolist())
                        sample_scores.append(token_scores.tolist())
                    best_ids = sample_ids
                    best_scores = sample_scores

    if best_ids is None or best_scores is None:
        raise RuntimeError("Gradient matching failed to produce candidate tokens.")

    reference_ids = batch["input_ids"].detach().cpu().tolist()
    predicted_ids = [[int(tok) for tok in sample] for sample in best_ids]
    rec_token_mean = token_recovery_ratio(
        predicted_ids,
        reference_ids,
        ignored_token_ids=ignored_token_ids,
        reference_mask=reference_mask,
    )

    return {
        "loss": float(best_loss),
        "initial_loss": float(loss_history[0]) if loss_history else None,
        "loss_reduction": float(loss_history[0] - best_loss) if loss_history else None,
        "loss_history": loss_history,
        "best_label": best_label,
        "logits_shape": tuple(int(x) for x in logits_shape) if logits_shape is not None else None,
        "predicted_ids": predicted_ids,
        "predicted_scores": best_scores,
        "selected_gradient_indices": selected_indices,
        "selected_gradient_names": selected_names,
        "selected_gradient_count": len(selected_names),
        "sequence_length": int(batch["input_ids"].shape[1]),
        "label_mode": "known" if label_known else "search",
        "restarts": int(max(1, int(restarts))),
        "match_loss": match_loss,
        "rec_token_mean": rec_token_mean,
        "best_logits": best_logits.detach().cpu() if 'best_logits' in locals() else None,
        "best_representation": best_representation.detach().cpu() if 'best_representation' in locals() else None,
    }


def summarize_token_predictions(predicted_ids: Sequence[Sequence[int]], tokenizer) -> list[str]:
    outputs = []
    for ids in predicted_ids:
        try:
            outputs.append(tokenizer.decode(ids, skip_special_tokens=True))
        except TypeError:
            outputs.append(tokenizer.decode(ids))
    return outputs
