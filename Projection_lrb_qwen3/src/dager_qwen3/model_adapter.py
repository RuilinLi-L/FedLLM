"""Strict structural adaptation from Qwen3/RoPE modules to DAGER inputs."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

try:
    from transformers.masking_utils import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )
except ImportError:
    # Layer-0-only structural checks remain useful without Transformers.  The
    # native Layer-1 path below fails explicitly if these APIs are needed.
    create_causal_mask = None
    create_sliding_window_causal_mask = None

from src.gradient_capture import GradientCaptureError, QProjectionPair, resolve_first_two_q_projections


class ModelAdapterError(RuntimeError):
    """Raised when a loaded model cannot satisfy the explicit Qwen3 adapter contract."""


def _require_tensor(name: str, value: Any, *, ndim: int | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ModelAdapterError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    if ndim is not None and value.ndim != ndim:
        raise ModelAdapterError(f"{name} must have rank {ndim}, got shape {tuple(value.shape)}.")
    return value


def _call_with_supported_keywords(callable_object: Any, values: Mapping[str, Any], *, context: str) -> Any:
    """Call one known Transformers interface, rejecting unknown required parameters."""
    try:
        inspected = callable_object.forward if isinstance(callable_object, nn.Module) else callable_object
        signature = inspect.signature(inspected)
    except (TypeError, ValueError) as error:
        raise ModelAdapterError(f"Unable to inspect native Qwen3 {context} signature: {error}") from error
    parameters = signature.parameters
    # Qwen3 decoder layers often expose ``**kwargs`` solely to forward a
    # version-specific cache field to self-attention.  Passing every known name
    # through that catch-all (for example both past_key_value spellings) can
    # create a silent semantic mismatch downstream.  Send only explicitly
    # declared arguments and fail if this installed version needs another one.
    keyword_values = {name: value for name, value in values.items() if name in parameters}
    missing = [
        name
        for name, parameter in parameters.items()
        if name not in ("self", "args", "kwargs")
        and parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name not in keyword_values
    ]
    if missing:
        raise ModelAdapterError(
            f"Installed Transformers Qwen3 {context} requires unsupported argument(s) {missing}; "
            "refusing to approximate its forward path."
        )
    try:
        return callable_object(**keyword_values)
    except Exception as error:  # Native implementation errors require version-aware context.
        raise ModelAdapterError(f"Native Qwen3 {context} failed: {type(error).__name__}: {error}") from error


@dataclass(frozen=True)
class Qwen3RoPEAdapterMetadata:
    """Resolved architecture facts written into attack diagnostics."""

    q_parameter_names: tuple[str, str]
    q_weight_shape: tuple[int, int]
    vocab_size: int
    hidden_size: int
    execution_path: str


class Qwen3RoPEDagerAdapter:
    """Expose actual Qwen3 q_proj inputs without GPT-2 positional assumptions.

    Layer 0 candidates use ``embed_tokens`` followed by the native first
    ``input_layernorm``.  Layer 1 candidates invoke the native first decoder
    layer with its native causal-mask and RoPE inputs, then apply the native
    second ``input_layernorm``.  No RoPE approximation is implemented here.
    """

    def __init__(self, model: nn.Module, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer
        try:
            self.q_pair: QProjectionPair = resolve_first_two_q_projections(model)
        except GradientCaptureError as error:
            raise ModelAdapterError(str(error)) from error
        if any(not isinstance(module, nn.Linear) for module in self.q_pair.modules):
            actual = [type(module).__name__ for module in self.q_pair.modules]
            raise ModelAdapterError(
                "Qwen3 DAGER requires nn.Linear q_proj modules; got " f"{actual}."
            )
        backbone = getattr(model, "model", None)
        layers = getattr(backbone, "layers", None)
        embed_tokens = getattr(backbone, "embed_tokens", None)
        rotary_emb = getattr(backbone, "rotary_emb", None)
        if not isinstance(backbone, nn.Module) or not isinstance(layers, nn.ModuleList) or len(layers) < 2:
            raise ModelAdapterError("Expected model.model.layers with at least two native Qwen3 decoder layers.")
        if not isinstance(embed_tokens, nn.Module):
            raise ModelAdapterError("Expected Qwen3 token embedding module at model.model.embed_tokens.")
        if not isinstance(rotary_emb, nn.Module):
            raise ModelAdapterError("Expected native Qwen3 rotary embedding module at model.model.rotary_emb.")
        if not isinstance(getattr(layers[0], "input_layernorm", None), nn.Module):
            raise ModelAdapterError("Expected Qwen3 RMSNorm at model.model.layers[0].input_layernorm.")
        if not isinstance(getattr(layers[1], "input_layernorm", None), nn.Module):
            raise ModelAdapterError("Expected Qwen3 RMSNorm at model.model.layers[1].input_layernorm.")
        self.backbone = backbone
        self.layers = layers
        self.embed_tokens = embed_tokens
        self.rotary_emb = rotary_emb
        embedding_weight = getattr(embed_tokens, "weight", None)
        if not isinstance(embedding_weight, nn.Parameter) or embedding_weight.ndim != 2:
            raise ModelAdapterError("model.model.embed_tokens.weight must be a rank-2 parameter.")
        config = getattr(model, "config", None)
        hidden_size = getattr(config, "hidden_size", None)
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ModelAdapterError(f"model.config.hidden_size must be positive, got {hidden_size!r}.")
        if int(embedding_weight.shape[1]) != hidden_size:
            raise ModelAdapterError(
                f"Embedding hidden dimension {embedding_weight.shape[1]} does not match config.hidden_size={hidden_size}."
            )
        self.metadata = Qwen3RoPEAdapterMetadata(
            q_parameter_names=self.q_pair.parameter_names,
            q_weight_shape=self.q_pair.expected_weight_shape,
            vocab_size=int(embedding_weight.shape[0]),
            hidden_size=hidden_size,
            execution_path="native_qwen3_layer0_with_native_rope_and_causal_mask",
        )

    @property
    def device(self) -> torch.device:
        return self.q_pair.modules[0].weight.device

    @property
    def compute_dtype(self) -> torch.dtype:
        return self.q_pair.modules[0].weight.dtype

    def _validate_prefix_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        _require_tensor("input_ids", input_ids, ndim=2)
        _require_tensor("attention_mask", attention_mask, ndim=2)
        if input_ids.shape != attention_mask.shape or input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
            raise ModelAdapterError(
                "Prefix input_ids and attention_mask must have equal non-empty [batch, sequence] shapes."
            )
        if input_ids.dtype != torch.long:
            raise ModelAdapterError(f"input_ids must be torch.long, got {input_ids.dtype}.")
        if input_ids.device != self.device or attention_mask.device != self.device:
            raise ModelAdapterError("Prefix tensors must reside on the model q_proj device.")
        if not bool(torch.all(attention_mask == 1)):
            raise ModelAdapterError("Qwen3 DAGER prefix decoding forbids padding; attention_mask must be all ones.")
        if int(input_ids.min().item()) < 0 or int(input_ids.max().item()) >= self.metadata.vocab_size:
            raise ModelAdapterError("Prefix token ids are outside the configured Qwen3 vocabulary.")

    def layer0_qproj_inputs_for_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return the exact layer-0 q_proj input representation for each token id."""
        _require_tensor("token_ids", token_ids, ndim=1)
        if token_ids.dtype != torch.long:
            raise ModelAdapterError(f"token_ids must be torch.long, got {token_ids.dtype}.")
        if token_ids.device != self.device:
            raise ModelAdapterError("Vocabulary token ids must be placed on the q_proj device.")
        if token_ids.numel() == 0:
            raise ModelAdapterError("Vocabulary scan chunk must contain at least one token id.")
        if int(token_ids.min().item()) < 0 or int(token_ids.max().item()) >= self.metadata.vocab_size:
            raise ModelAdapterError("Vocabulary scan token id is outside the Qwen3 vocabulary.")
        with torch.no_grad():
            embeddings = self.embed_tokens(token_ids)
            representations = self.layers[0].input_layernorm(embeddings)
        if representations.ndim != 2 or representations.shape != (token_ids.shape[0], self.metadata.hidden_size):
            raise ModelAdapterError(
                "Native layer-0 RMSNorm returned an unexpected candidate shape "
                f"{tuple(representations.shape)}."
            )
        return representations

    def _native_causal_mask(
        self,
        *,
        attention_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_position: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        """Construct the native version-specific Qwen3 mask for decoder layer 0."""

        legacy_update = getattr(self.backbone, "_update_causal_mask", None)
        if callable(legacy_update):
            result = _call_with_supported_keywords(
                legacy_update,
                {
                    "attention_mask": attention_mask,
                    "input_tensor": hidden_states,
                    "cache_position": cache_position,
                    "past_key_values": None,
                    "output_attentions": False,
                    "use_cache": False,
                },
                context="_update_causal_mask",
            )
        else:
            if create_causal_mask is None:
                raise ModelAdapterError(
                    "transformers.masking_utils is required for the native Qwen3 Layer-1 forward path."
                )
            config = getattr(
                self.backbone,
                "config",
                getattr(self.model, "config", None),
            )
            if config is None:
                raise ModelAdapterError(
                    "Native Qwen3 causal-mask construction requires a model config."
                )

            mask_kwargs = {
                "config": config,
                "input_embeds": hidden_states,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": None,
                "position_ids": position_ids,
            }

            causal_mask_mapping: dict[str, torch.Tensor | None] = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }

            if bool(getattr(self.backbone, "has_sliding_layers", False)):
                if create_sliding_window_causal_mask is None:
                    raise ModelAdapterError(
                        "transformers.masking_utils lacks create_sliding_window_causal_mask for this Qwen3 model."
                    )
                causal_mask_mapping["sliding_attention"] = (
                    create_sliding_window_causal_mask(**mask_kwargs)
                )

            attention_type = getattr(
                self.layers[0],
                "attention_type",
                None,
            )
            if attention_type not in causal_mask_mapping:
                raise ModelAdapterError(
                    "Unsupported native Qwen3 layer-0 attention type "
                    f"{attention_type!r}; available masks are "
                    f"{sorted(causal_mask_mapping)}."
                )

            result = causal_mask_mapping[attention_type]

        if result is not None and not isinstance(result, torch.Tensor):
            raise ModelAdapterError(
                "Native Qwen3 causal-mask construction returned "
                f"{type(result).__name__}, expected Tensor or None."
            )
        return result

    def _native_layer0_forward(
        self,
        *,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        position_embeddings: Any,
    ) -> torch.Tensor:
        output = _call_with_supported_keywords(
            self.layers[0],
            {
                "hidden_states": hidden_states,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": None,
                "past_key_value": None,
                "cache_position": cache_position,
                "position_embeddings": position_embeddings,
                "use_cache": False,
                "output_attentions": False,
            },
            context="decoder layer 0 forward",
        )
        if isinstance(output, tuple):
            if not output:
                raise ModelAdapterError("Native Qwen3 decoder layer 0 returned an empty tuple.")
            hidden = output[0]
        else:
            hidden = output
        _require_tensor("native decoder-layer output", hidden, ndim=3)
        return hidden

    def layer1_qproj_inputs_from_prefixes(
        self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Run Qwen3's native first layer and return the true layer-1 q_proj input.

        This is intentionally a native layer forward: the adapter derives causal
        masks and RoPE position embeddings from the installed model rather than
        reproducing RoPE mathematics.
        """
        self._validate_prefix_batch(input_ids, attention_mask)
        with torch.no_grad():
            hidden_states = self.embed_tokens(input_ids)
            sequence_length = int(input_ids.shape[1])
            cache_position = torch.arange(sequence_length, device=self.device, dtype=torch.long)
            position_ids = cache_position.unsqueeze(0).expand(input_ids.shape[0], -1)
            try:
                position_embeddings = self.rotary_emb(hidden_states, position_ids)
            except Exception as error:
                raise ModelAdapterError(
                    f"Native Qwen3 rotary_emb(hidden_states, position_ids) failed: {type(error).__name__}: {error}"
                ) from error
            causal_mask = self._native_causal_mask(
                attention_mask=attention_mask,
                hidden_states=hidden_states,
                cache_position=cache_position,
                position_ids=position_ids,
            )
            layer0_output = self._native_layer0_forward(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            q1_input = self.layers[1].input_layernorm(layer0_output)
        if q1_input.shape != (input_ids.shape[0], input_ids.shape[1], self.metadata.hidden_size):
            raise ModelAdapterError(
                f"Native prefix forward returned q1 input shape {tuple(q1_input.shape)}, expected "
                f"[{input_ids.shape[0]}, {input_ids.shape[1]}, {self.metadata.hidden_size}]."
            )
        return q1_input

    def capture_layer1_qproj_input_from_model_forward(
        self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Capture layer-1 q_proj input from a full native model forward for tests."""
        self._validate_prefix_batch(input_ids, attention_mask)
        captured: list[torch.Tensor] = []

        def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise ModelAdapterError("Layer-1 q_proj forward hook received no tensor input.")
            captured.append(inputs[0].detach().clone())

        handle = self.q_pair.modules[1].register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                _call_with_supported_keywords(
                    self.model,
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "use_cache": False,
                        "output_attentions": False,
                        "output_hidden_states": False,
                        "return_dict": True,
                    },
                    context="full sequence-classification forward",
                )
        finally:
            handle.remove()
        if len(captured) != 1:
            raise ModelAdapterError(f"Expected exactly one layer-1 q_proj hook capture, got {len(captured)}.")
        return captured[0]
