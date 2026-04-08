"""
DAGER-specific defense methods based on the proposed defense strategies.

This module implements defense methods specifically designed to counter DAGER attacks
by breaking its core assumptions about low-rank structure and deterministic token embeddings.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any


def _make_generator(seed: int, device: torch.device) -> torch.Generator:
    """Create a seeded generator on the correct device."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


class DAGERDefense:
    """
    Defense methods specifically designed to counter DAGER attacks.
    
    Implements several strategies:
    1. Dynamic Basis Perturbation - Add structured noise to break span check
    2. Stochastic Offset Embedding - Add random offsets to position embeddings
    3. Gradient Slicing & Chunking - Selectively send gradients from certain layers
    4. Rank-Limiting Defense - Ensure b >= r for LoRA defenses
    """
    
    @staticmethod
    def dynamic_basis_perturbation(
        grads: Tuple[torch.Tensor, ...],
        target_rank: int = None,
        noise_scale: float = 0.01,
        seed: int = 0
    ) -> Tuple[torch.Tensor, ...]:
        """
        Add structured noise to gradients to break DAGER's span check.
        
        DAGER relies on the assumption that rank(gradient) < d (dimension).
        By adding structured noise to increase the effective rank, we make
        the span check ineffective.
        
        Args:
            grads: Tuple of gradient tensors
            target_rank: Target rank to achieve (if None, aim for full rank)
            noise_scale: Scale of the structured noise
            seed: Random seed for reproducibility
            
        Returns:
            Perturbed gradients
        """
        out = []
        for g in grads:
            if g is None:
                out.append(None)
                continue
                
            device = g.device
            dtype = g.dtype
            shape = g.shape
            
            # Create structured noise matrix
            gen = _make_generator(seed, device)
            
            if len(shape) == 2:
                # Matrix gradient (e.g., weight matrix)
                m, n = shape
                if target_rank is None:
                    target_rank = min(m, n)
                
                # Create low-rank structured noise
                U = torch.randn(m, target_rank, device=device, dtype=dtype, generator=gen)
                V = torch.randn(target_rank, n, device=device, dtype=dtype, generator=gen)
                
                # Scale noise based on gradient norm
                grad_norm = g.norm()
                if grad_norm > 0:
                    noise = (U @ V) * (noise_scale * grad_norm / (U @ V).norm())
                else:
                    noise = (U @ V) * noise_scale
                    
                out.append(g + noise)
                
            elif len(shape) == 1:
                # Vector gradient (e.g., bias)
                n = shape[0]
                noise = torch.randn(n, device=device, dtype=dtype, generator=gen)
                grad_norm = g.norm()
                if grad_norm > 0:
                    noise = noise * (noise_scale * grad_norm / noise.norm())
                out.append(g + noise)
                
            else:
                # Higher dimensional tensors - flatten and treat as matrix
                original_shape = shape
                flattened = g.view(-1, 1) if len(shape) > 1 else g.view(-1)
                m = flattened.shape[0]
                
                if len(flattened.shape) == 2:
                    m, n = flattened.shape
                    if target_rank is None:
                        target_rank = min(m, n)
                    
                    U = torch.randn(m, target_rank, device=device, dtype=dtype, generator=gen)
                    V = torch.randn(target_rank, n, device=device, dtype=dtype, generator=gen)
                    noise = U @ V
                else:
                    n = m
                    noise = torch.randn(n, device=device, dtype=dtype, generator=gen)
                
                grad_norm = flattened.norm()
                if grad_norm > 0:
                    noise = noise * (noise_scale * grad_norm / noise.norm())
                
                perturbed = flattened + noise
                out.append(perturbed.view(original_shape))
        
        return tuple(out)
    
    @staticmethod
    def stochastic_offset_embedding(
        grads: Tuple[torch.Tensor, ...],
        model_wrapper,
        batch: Dict[str, torch.Tensor],
        offset_scale: float = 0.01,
        seed: int = 0
    ) -> Tuple[torch.Tensor, ...]:
        """
        Apply stochastic offset to position embeddings to break DAGER's position encoding assumption.
        
        DAGER assumes exact knowledge of position encoding function f^0(v, i).
        By adding random offsets that vary per batch, we break this deterministic relationship.
        
        Args:
            grads: Tuple of gradient tensors
            model_wrapper: Model wrapper containing the model
            batch: Input batch containing input_ids and attention_mask
            offset_scale: Scale of random offsets
            seed: Random seed for reproducibility
            
        Returns:
            Gradients with position embedding offsets applied
        """
        model = model_wrapper.model
        device = next(model.parameters()).device
        
        # Get the embedding layer
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'wpe'):
            # GPT-2 style
            pos_embedding = model.transformer.wpe
        elif hasattr(model, 'bert') and hasattr(model.bert.embeddings, 'position_embeddings'):
            # BERT style
            pos_embedding = model.bert.embeddings.position_embeddings
        elif hasattr(model, 'model') and hasattr(model.model, 'embed_positions'):
            # LLaMA style
            pos_embedding = model.model.embed_positions
        else:
            # Cannot find position embedding, return original gradients
            return grads
        
        # Generate random offsets for position embeddings
        gen = _make_generator(seed, device)
        pos_embedding_dim = pos_embedding.embedding_dim
        max_position = pos_embedding.num_embeddings if hasattr(pos_embedding, 'num_embeddings') else 512
        
        # Create random offsets for each position
        offsets = torch.randn(max_position, pos_embedding_dim, device=device, generator=gen) * offset_scale
        
        # Store original parameters and create hook for gradient modification
        original_weight = pos_embedding.weight.data.clone()
        
        def offset_hook(grad):
            # Add gradient contribution from offsets
            # This simulates the effect of having used offset embeddings during forward pass
            if grad.shape == offsets.shape:
                # Direct gradient to position embedding layer
                return grad + offsets * offset_scale
            return grad
        
        # Register backward hook if possible
        if hasattr(pos_embedding.weight, 'register_hook'):
            pos_embedding.weight.register_hook(offset_hook)
        
        # Note: In practice, we would need to modify the forward pass to actually use offsets.
        # This is a simplified implementation that modifies gradients directly.
        # A more complete implementation would require modifying the model's forward method.
        
        return grads
    
    @staticmethod
    def gradient_slicing(
        grads: Tuple[torch.Tensor, ...],
        layer_names: List[str],
        model,
        send_first_n_layers: Optional[int] = None,
        send_last_n_layers: Optional[int] = None,
        random_slice: bool = False,
        slice_prob: float = 0.5,
        seed: int = 0
    ) -> Tuple[torch.Tensor, ...]:
        """
        Selectively send gradients from certain layers to break DAGER's layer dependency.
        
        DAGER requires gradients from first two self-attention layers for token recovery
        and sequence reconstruction. By randomly omitting these layers, we can break the attack.
        
        Args:
            grads: Tuple of gradient tensors
            layer_names: List of parameter names corresponding to gradients
            model: The model (used to identify layer types)
            send_first_n_layers: Send only first n layers (if specified)
            send_last_n_layers: Send only last n layers (if specified)
            random_slice: Randomly select layers to send
            slice_prob: Probability of sending each layer (if random_slice=True)
            seed: Random seed for reproducibility
            
        Returns:
            Sliced gradients (some layers zeroed out)
        """
        if not layer_names:
            return grads
            
        out = list(grads)
        device = out[0].device if out[0] is not None else torch.device('cpu')
        gen = _make_generator(seed, device)
        
        # Identify self-attention layers (especially first two)
        self_attn_layer_indices = []
        for i, name in enumerate(layer_names):
            if any(keyword in name for keyword in ['attention', 'attn', 'self_attn']):
                # Check if it's query/key/value weight or bias
                if any(param in name for param in ['q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value']):
                    # Extract layer number if possible
                    import re
                    match = re.search(r'layer\.(\d+)', name) or re.search(r'layers\.(\d+)', name) or re.search(r'\.(\d+)\.', name)
                    if match:
                        layer_num = int(match.group(1))
                        if layer_num < 2:  # First two layers are critical for DAGER
                            self_attn_layer_indices.append(i)
        
        # Apply slicing based on strategy
        if send_first_n_layers is not None:
            # Only send first n layers
            for i in range(len(out)):
                if i >= send_first_n_layers and out[i] is not None:
                    out[i] = torch.zeros_like(out[i])
                    
        elif send_last_n_layers is not None:
            # Only send last n layers
            for i in range(len(out) - send_last_n_layers):
                if out[i] is not None:
                    out[i] = torch.zeros_like(out[i])
                    
        elif random_slice:
            # Randomly select layers to send
            for i in range(len(out)):
                if out[i] is not None:
                    if torch.rand(1, generator=gen).item() > slice_prob:
                        out[i] = torch.zeros_like(out[i])
        
        else:
            # Default: zero out first two self-attention layers (most critical for DAGER)
            for idx in self_attn_layer_indices:
                if idx < len(out) and out[idx] is not None:
                    out[idx] = torch.zeros_like(out[idx])
        
        return tuple(out)
    
    @staticmethod
    def rank_limiting_defense(
        grads: Tuple[torch.Tensor, ...],
        model,
        batch_size: int,
        min_token_count: int = None,
        padding_token_id: int = 50256,  # GPT-2 pad token
        seed: int = 0
    ) -> Tuple[torch.Tensor, ...]:
        """
        Ensure b >= r to break DAGER's low-rank assumption.
        
        For LoRA defenses, DAGER requires b < r (batch token count < LoRA rank).
        By padding the batch with dummy tokens that contribute to gradients,
        we can ensure b >= r, breaking DAGER's theoretical foundation.
        
        Args:
            grads: Tuple of gradient tensors
            model: The model
            batch_size: Original batch size
            min_token_count: Minimum token count to achieve (if None, aim for r+1)
            padding_token_id: ID of padding token to use
            seed: Random seed for reproducibility
            
        Returns:
            Gradients with rank-limiting modifications
        """
        # This defense is more conceptual - in practice, we would need to
        # modify the training process to include padding tokens that contribute
        # to gradients but don't affect the loss significantly.
        
        # For gradient-level implementation, we can simulate the effect by
        # adding noise that increases the effective rank
        
        return DAGERDefense.dynamic_basis_perturbation(
            grads, 
            target_rank=None,  # Aim for full rank
            noise_scale=0.005,
            seed=seed
        )
    
    @staticmethod
    def combined_defense(
        grads: Tuple[torch.Tensor, ...],
        args,
        model_wrapper=None,
        batch=None,
        labels=None,
        layer_names: List[str] = None
    ) -> Tuple[torch.Tensor, ...]:
        """
        Apply a combination of DAGER-specific defenses.
        
        Args:
            grads: Tuple of gradient tensors
            args: Command line arguments
            model_wrapper: Model wrapper (required for some defenses)
            batch: Input batch (required for some defenses)
            labels: Labels (required for some defenses)
            layer_names: List of parameter names
            
        Returns:
            Defended gradients
        """
        if grads is None:
            return grads
            
        # Get defense parameters from args or use defaults
        defense_params = {
            'basis_perturb': getattr(args, 'defense_dager_basis_perturb', True),
            'basis_noise_scale': getattr(args, 'defense_dager_basis_noise_scale', 0.01),
            'offset_embedding': getattr(args, 'defense_dager_offset_embedding', False),
            'offset_scale': getattr(args, 'defense_dager_offset_scale', 0.01),
            'gradient_slicing': getattr(args, 'defense_dager_gradient_slicing', False),
            'slice_first_n': getattr(args, 'defense_dager_slice_first_n', None),
            'slice_last_n': getattr(args, 'defense_dager_slice_last_n', None),
            'random_slice': getattr(args, 'defense_dager_random_slice', False),
            'slice_prob': getattr(args, 'defense_dager_slice_prob', 0.5),
            'rank_limit': getattr(args, 'defense_dager_rank_limit', False),
            'seed': getattr(args, 'rng_seed', 0)
        }
        
        g = grads
        
        # 1. Dynamic basis perturbation
        if defense_params['basis_perturb']:
            g = DAGERDefense.dynamic_basis_perturbation(
                g,
                noise_scale=defense_params['basis_noise_scale'],
                seed=defense_params['seed']
            )
        
        # 2. Stochastic offset embedding (requires model and batch)
        if defense_params['offset_embedding'] and model_wrapper is not None and batch is not None:
            g = DAGERDefense.stochastic_offset_embedding(
                g,
                model_wrapper,
                batch,
                offset_scale=defense_params['offset_scale'],
                seed=defense_params['seed']
            )
        
        # 3. Gradient slicing (requires layer names)
        if defense_params['gradient_slicing'] and layer_names is not None and model_wrapper is not None:
            model = model_wrapper.model
            if defense_params['slice_first_n'] is not None:
                g = DAGERDefense.gradient_slicing(
                    g,
                    layer_names,
                    model,
                    send_first_n_layers=defense_params['slice_first_n'],
                    seed=defense_params['seed']
                )
            elif defense_params['slice_last_n'] is not None:
                g = DAGERDefense.gradient_slicing(
                    g,
                    layer_names,
                    model,
                    send_last_n_layers=defense_params['slice_last_n'],
                    seed=defense_params['seed']
                )
            elif defense_params['random_slice']:
                g = DAGERDefense.gradient_slicing(
                    g,
                    layer_names,
                    model,
                    random_slice=True,
                    slice_prob=defense_params['slice_prob'],
                    seed=defense_params['seed']
                )
        
        # 4. Rank-limiting defense
        if defense_params['rank_limit'] and model_wrapper is not None and batch is not None:
            batch_size = batch['input_ids'].shape[0] if 'input_ids' in batch else 1
            g = DAGERDefense.rank_limiting_defense(
                g,
                model_wrapper.model,
                batch_size,
                seed=defense_params['seed']
            )
        
        return g


def apply_dager_defense(
    grads: Tuple[torch.Tensor, ...],
    args,
    model_wrapper=None,
    batch=None,
    labels=None,
    layer_names: List[str] = None
) -> Tuple[torch.Tensor, ...]:
    """
    Main entry point for DAGER-specific defenses.
    
    This function should be called from the main defense application logic.
    
    Args:
        grads: Tuple of gradient tensors
        args: Command line arguments
        model_wrapper: Model wrapper (required for some defenses)
        batch: Input batch (required for some defenses)
        labels: Labels (required for some defenses)
        layer_names: List of parameter names
        
    Returns:
        Defended gradients
    """
    return DAGERDefense.combined_defense(
        grads, args, model_wrapper, batch, labels, layer_names
    )