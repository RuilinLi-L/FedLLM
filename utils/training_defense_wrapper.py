from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


class TrainingDefenseModelWrapper:
    """Thin wrapper that lets training-side code reuse utils.defenses.apply_defense."""

    def __init__(self, model, args, trainable_params: Sequence[torch.nn.Parameter]):
        self.model = model
        self.args = args
        self._trainable_params = tuple(trainable_params)

    def trainable_parameters(self):
        return self._trainable_params

    def _model_family(self) -> str:
        model_type = getattr(self.model.config, "model_type", "")
        if model_type == "gpt2" or (hasattr(self.model, "transformer") and hasattr(self.model, "score")):
            return "gpt2"
        if model_type == "bert" or hasattr(self.model, "bert"):
            return "bert"
        if model_type == "llama" or (
            hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens")
        ):
            return "llama"
        raise NotImplementedError(f"Training defense wrapper does not support model_type={model_type!r}.")

    def _seq_class_input_embeds(self, batch):
        family = self._model_family()
        if family == "gpt2":
            return self.model.transformer.wte(batch["input_ids"])
        if family == "bert":
            bert = self.model.bert
            emb = bert.embeddings
            input_ids = batch["input_ids"]
            token_type_ids = batch.get("token_type_ids", torch.zeros_like(input_ids))
            seq_length = input_ids.size(1)
            position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
            word = emb.word_embeddings(input_ids)
            pos_e = emb.position_embeddings(position_ids)
            tok_e = emb.token_type_embeddings(token_type_ids)
            hidden = emb.LayerNorm(word + pos_e + tok_e)
            return emb.dropout(hidden)
        if family == "llama":
            return self.model.model.embed_tokens(batch["input_ids"])
        raise AssertionError("Unreachable model family")

    def _seq_class_logits_from_embeds(self, batch, inputs_embeds, representation_mask=None):
        family = self._model_family()
        attn = batch.get("attention_mask")

        if family == "gpt2":
            position_ids = torch.arange(
                batch["input_ids"].size(1),
                device=batch["input_ids"].device,
            ).unsqueeze(0).expand_as(batch["input_ids"])
            tr_out = self.model.transformer(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                position_ids=position_ids,
            )
            hidden = tr_out.last_hidden_state
            if attn is not None:
                idx = attn.long().sum(dim=1) - 1
                idx = idx.clamp(min=0)
            else:
                idx = torch.full(
                    (hidden.size(0),),
                    hidden.size(1) - 1,
                    dtype=torch.long,
                    device=hidden.device,
                )
            representation = hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
            if representation_mask is not None:
                representation = representation * representation_mask
            logits = self.model.score(representation)
            return logits, representation

        if family == "bert":
            bert = self.model.bert
            input_ids = batch["input_ids"]
            raw_mask = batch.get("attention_mask")
            ext_mask = (
                bert.get_extended_attention_mask(raw_mask, input_ids.shape)
                if raw_mask is not None
                else None
            )
            enc_out = bert.encoder(inputs_embeds, attention_mask=ext_mask)
            sequence_output = enc_out.last_hidden_state
            if bert.pooler is not None:
                representation = bert.pooler(sequence_output)
            else:
                representation = sequence_output[:, 0]
            if representation_mask is not None:
                representation = representation * representation_mask
            logits = self.model.classifier(self.model.dropout(representation))
            return logits, representation

        if family == "llama":
            position_ids = torch.arange(
                batch["input_ids"].size(1),
                device=batch["input_ids"].device,
            ).unsqueeze(0).expand_as(batch["input_ids"])
            out = self.model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                position_ids=position_ids,
            )
            hidden = out.last_hidden_state
            if attn is not None:
                idx = attn.long().sum(dim=1) - 1
                idx = idx.clamp(min=0)
            else:
                idx = torch.full(
                    (hidden.size(0),),
                    hidden.size(1) - 1,
                    dtype=torch.long,
                    device=hidden.device,
                )
            representation = hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
            if representation_mask is not None:
                representation = representation * representation_mask
            logits = self.model.score(representation)
            return logits, representation

        raise AssertionError("Unreachable model family")

    def compute_per_example_grads(self, batch, labels, create_graph=False, sample_grad_fn=None):
        flat_labels = labels.view(-1).long()
        grad_list = []
        for idx in range(batch["input_ids"].shape[0]):
            sample_batch = {k: v[idx:idx + 1] for k, v in batch.items() if k != "labels"}
            sample_labels = flat_labels[idx:idx + 1]
            if sample_grad_fn is None:
                self.model.zero_grad(set_to_none=True)
                outputs = self.model(**sample_batch, labels=sample_labels)
                grad = torch.autograd.grad(
                    outputs.loss,
                    self.trainable_parameters(),
                    create_graph=create_graph,
                    allow_unused=True,
                )
            else:
                grad = sample_grad_fn(
                    sample_batch,
                    sample_labels,
                    sample_idx=idx,
                    create_graph=create_graph,
                )
            grad_list.append(
                tuple(g if (g is None or create_graph) else g.detach().clone() for g in grad)
            )
        return grad_list

    def compute_grads_mixup(self, batch, labels, create_graph=False):
        if batch["input_ids"].shape[0] < 2:
            self.model.zero_grad(set_to_none=True)
            outputs = self.model(**{k: v for k, v in batch.items() if k != "labels"}, labels=labels.view(-1).long())
            return torch.autograd.grad(
                outputs.loss,
                self.trainable_parameters(),
                create_graph=create_graph,
                allow_unused=True,
            )

        alpha = float(getattr(self.args, "defense_mixup_alpha", 1.0))
        lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
        labels = labels.view(-1).long()
        perm = torch.randperm(batch["input_ids"].shape[0], device=batch["input_ids"].device)
        emb = self._seq_class_input_embeds(batch)
        emb_mixed = lam * emb + (1.0 - lam) * emb[perm]
        logits, _ = self._seq_class_logits_from_embeds(batch, emb_mixed)
        loss = lam * F.cross_entropy(logits, labels) + (1.0 - lam) * F.cross_entropy(logits, labels[perm])
        self.model.zero_grad(set_to_none=True)
        return torch.autograd.grad(
            loss,
            self.trainable_parameters(),
            create_graph=create_graph,
            allow_unused=True,
        )
