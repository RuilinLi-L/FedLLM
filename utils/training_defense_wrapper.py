from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence

import torch
import torch.nn.functional as F

from utils.representation_bottleneck import apply_representation_bottleneck, rep_bottleneck_active


class TrainingDefenseModelWrapper:
    """Thin wrapper that lets training-side code reuse utils.defenses.apply_defense."""

    def __init__(self, model, args, trainable_params: Sequence[torch.nn.Parameter]):
        self.model = model
        self.base_model = self._unwrap_model(model)
        self.args = args
        self._trainable_params = tuple(trainable_params)

    def trainable_parameters(self):
        return self._trainable_params

    def _unwrap_model(self, model):
        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            return model.base_model.model
        if hasattr(model, "model") and not hasattr(model, "transformer"):
            return model.model
        return model

    def _model_family(self) -> str:
        model_type = getattr(self.base_model.config, "model_type", "")
        if model_type == "gpt2" or (hasattr(self.base_model, "transformer") and hasattr(self.base_model, "score")):
            return "gpt2"
        if model_type == "bert" or hasattr(self.base_model, "bert"):
            return "bert"
        if model_type == "llama" or (
            hasattr(self.base_model, "model") and hasattr(self.base_model.model, "embed_tokens")
        ):
            return "llama"
        raise NotImplementedError(f"Training defense wrapper does not support model_type={model_type!r}.")

    def _seq_class_input_embeds(self, batch):
        family = self._model_family()
        if family == "gpt2":
            return self.base_model.transformer.wte(batch["input_ids"])
        if family == "bert":
            bert = self.base_model.bert
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
            return self.base_model.model.embed_tokens(batch["input_ids"])
        raise AssertionError("Unreachable model family")

    def _seq_class_representation_from_embeds(self, batch, inputs_embeds, representation_mask=None):
        family = self._model_family()
        attn = batch.get("attention_mask")

        if family == "gpt2":
            position_ids = torch.arange(
                batch["input_ids"].size(1),
                device=batch["input_ids"].device,
            ).unsqueeze(0).expand_as(batch["input_ids"])
            tr_out = self.base_model.transformer(
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
            representation = apply_representation_bottleneck(representation, self.args)
            return representation

        if family == "bert":
            bert = self.base_model.bert
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
            representation = apply_representation_bottleneck(representation, self.args)
            return representation

        if family == "llama":
            position_ids = torch.arange(
                batch["input_ids"].size(1),
                device=batch["input_ids"].device,
            ).unsqueeze(0).expand_as(batch["input_ids"])
            out = self.base_model.model(
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
            representation = apply_representation_bottleneck(representation, self.args)
            return representation

        raise AssertionError("Unreachable model family")

    def _seq_class_logits_from_representation(self, representation):
        family = self._model_family()

        if family == "gpt2":
            return self._classifier_model().score(representation)
        if family == "bert":
            classifier_model = self._classifier_model()
            return classifier_model.classifier(classifier_model.dropout(representation))
        if family == "llama":
            return self._classifier_model().score(representation)
        raise AssertionError("Unreachable model family")

    def _classifier_model(self):
        if hasattr(self.model, "score") or hasattr(self.model, "classifier"):
            return self.model
        return self.base_model

    def _classifier_head(self):
        family = self._model_family()
        classifier_model = self._classifier_model()
        if family in {"gpt2", "llama"}:
            return getattr(classifier_model, "score", None)
        if family == "bert":
            return getattr(classifier_model, "classifier", None)
        raise AssertionError("Unreachable model family")

    @contextmanager
    def _representation_bottleneck_hook(self):
        if not rep_bottleneck_active(self.args):
            yield {"representation": None}
            return

        captured = {"representation": None}

        def hook(_module, inputs):
            if not inputs:
                return None
            representation = apply_representation_bottleneck(inputs[0], self.args)
            captured["representation"] = representation
            return (representation, *inputs[1:])

        classifier_head = self._classifier_head()
        if classifier_head is None:
            yield captured
            return
        handle = classifier_head.register_forward_pre_hook(hook)
        try:
            yield captured
        finally:
            handle.remove()

    def _seq_class_logits_from_embeds(self, batch, inputs_embeds, representation_mask=None):
        representation = self._seq_class_representation_from_embeds(
            batch,
            inputs_embeds,
            representation_mask=representation_mask,
        )
        logits = self._seq_class_logits_from_representation(representation)
        return logits, representation

    def seq_class_loss_logits(self, batch, labels, representation_mask=None):
        if representation_mask is None:
            model_inputs = {k: v for k, v in batch.items() if k != "labels"}
            with self._representation_bottleneck_hook() as captured:
                outputs = self.model(**model_inputs, labels=labels.view(-1).long())
            return outputs.loss, outputs.logits, captured.get("representation")

        inputs_embeds = self._seq_class_input_embeds(batch)
        logits, representation = self._seq_class_logits_from_embeds(
            batch,
            inputs_embeds,
            representation_mask=representation_mask,
        )
        loss = F.cross_entropy(logits, labels.view(-1).long())
        return loss, logits, representation

    def _compute_standard_grads(self, batch, labels, create_graph=False):
        self.model.zero_grad(set_to_none=True)
        if getattr(self.args, "task", "seq_class") == "seq_class":
            loss, _, _ = self.seq_class_loss_logits(batch, labels)
            return torch.autograd.grad(
                loss,
                self.trainable_parameters(),
                create_graph=create_graph,
                allow_unused=True,
            )
        outputs = self.model(
            **{k: v for k, v in batch.items() if k != "labels"},
            labels=labels.view(-1).long(),
        )
        return torch.autograd.grad(
            outputs.loss,
            self.trainable_parameters(),
            create_graph=create_graph,
            allow_unused=True,
        )

    def compute_grads(self, batch, labels, create_graph=False):
        return self._compute_standard_grads(batch, labels, create_graph=create_graph)

    def compute_per_example_grads(self, batch, labels, create_graph=False, sample_grad_fn=None):
        flat_labels = labels.view(-1).long()
        grad_list = []
        for idx in range(batch["input_ids"].shape[0]):
            sample_batch = {k: v[idx:idx + 1] for k, v in batch.items() if k != "labels"}
            sample_labels = flat_labels[idx:idx + 1]
            if sample_grad_fn is None:
                grad = self._compute_standard_grads(
                    sample_batch,
                    sample_labels,
                    create_graph=create_graph,
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
        """Representation-level manifold MixUp-style gradient baseline."""
        if getattr(self.args, "task", "seq_class") != "seq_class":
            return self._compute_standard_grads(batch, labels, create_graph=create_graph)
        if batch["input_ids"].shape[0] < 2:
            return self._compute_standard_grads(batch, labels, create_graph=create_graph)

        alpha = float(getattr(self.args, "defense_mixup_alpha", 1.0))
        lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
        labels = labels.view(-1).long()
        perm = torch.randperm(batch["input_ids"].shape[0], device=batch["input_ids"].device)
        emb = self._seq_class_input_embeds(batch)
        representation = self._seq_class_representation_from_embeds(batch, emb)
        representation_mixed = lam * representation + (1.0 - lam) * representation[perm]
        logits = self._seq_class_logits_from_representation(representation_mixed)
        loss = lam * F.cross_entropy(logits, labels) + (1.0 - lam) * F.cross_entropy(logits, labels[perm])
        self.model.zero_grad(set_to_none=True)
        return torch.autograd.grad(
            loss,
            self.trainable_parameters(),
            create_graph=create_graph,
            allow_unused=True,
        )
