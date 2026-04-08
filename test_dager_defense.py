#!/usr/bin/env python3
"""
Test script for DAGER defense methods.
This script tests the basic functionality of the DAGER defense implementation.
"""

import torch
import numpy as np
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.dager_defense import DAGERDefense, apply_dager_defense

def test_dynamic_basis_perturbation():
    """Test dynamic basis perturbation defense."""
    print("Testing dynamic basis perturbation...")
    
    # Create dummy gradients
    grads = (
        torch.randn(10, 20),  # 2D tensor (weight matrix)
        torch.randn(30),      # 1D tensor (bias)
        torch.randn(2, 3, 4), # 3D tensor
        None,                 # None gradient
    )
    
    original_norms = [g.norm().item() if g is not None else 0 for g in grads]
    
    # Apply defense
    perturbed_grads = DAGERDefense.dynamic_basis_perturbation(
        grads, 
        target_rank=5,
        noise_scale=0.01,
        seed=42
    )
    
    # Check results
    assert len(perturbed_grads) == len(grads), "Output length should match input"
    assert perturbed_grads[3] is None, "None gradients should remain None"
    
    for i, (orig, pert) in enumerate(zip(grads, perturbed_grads)):
        if orig is not None:
            assert pert.shape == orig.shape, "Shape mismatch at index {}".format(i)
            # Check that noise was added (gradients should be different)
            if i < 3:  # Skip None
                assert not torch.allclose(orig, pert), "Gradients unchanged at index {}".format(i)
    
    print("[PASS] Dynamic basis perturbation test passed")
    return True

def test_gradient_slicing():
    """Test gradient slicing defense."""
    print("Testing gradient slicing...")
    
    # Create dummy gradients
    grads = (
        torch.randn(10, 20),
        torch.randn(30, 40),
        torch.randn(50),
    )
    
    # Create dummy layer names
    layer_names = [
        "transformer.h.0.attn.q_proj.weight",  # First layer attention
        "transformer.h.1.attn.k_proj.weight",  # Second layer attention  
        "transformer.h.2.mlp.dense.weight",    # Third layer MLP
    ]
    
    # Test sending only first n layers
    sliced_grads = DAGERDefense.gradient_slicing(
        grads,
        layer_names,
        model=None,  # Model not needed for this test
        send_first_n_layers=1,
        seed=42
    )
    
    # First gradient should be zeroed, others should remain
    assert torch.allclose(sliced_grads[0], torch.zeros_like(grads[0])), "First layer should be zeroed"
    assert torch.allclose(sliced_grads[1], grads[1]), "Second layer should remain"
    assert torch.allclose(sliced_grads[2], grads[2]), "Third layer should remain"
    
    # Test random slicing
    sliced_grads = DAGERDefense.gradient_slicing(
        grads,
        layer_names,
        model=None,
        random_slice=True,
        slice_prob=0.0,  # Zero probability means all should be zeroed
        seed=42
    )
    
    for i in range(len(grads)):
        assert torch.allclose(sliced_grads[i], torch.zeros_like(grads[i])), "All layers should be zeroed with slice_prob=0"
    
    print("[PASS] Gradient slicing test passed")
    return True

def test_combined_defense():
    """Test combined defense with mock arguments."""
    print("Testing combined defense...")
    
    # Create dummy gradients
    grads = (
        torch.randn(5, 10),
        torch.randn(15),
    )
    
    # Create mock args object
    class MockArgs:
        defense_dager_basis_perturb = True
        defense_dager_basis_noise_scale = 0.01
        defense_dager_offset_embedding = False
        defense_dager_offset_scale = 0.01
        defense_dager_gradient_slicing = False
        defense_dager_slice_first_n = None
        defense_dager_slice_last_n = None
        defense_dager_random_slice = False
        defense_dager_slice_prob = 0.5
        defense_dager_rank_limit = False
        rng_seed = 42
    
    args = MockArgs()
    
    # Apply combined defense
    defended_grads = DAGERDefense.combined_defense(
        grads,
        args,
        model_wrapper=None,
        batch=None,
        labels=None,
        layer_names=None
    )
    
    # Check results
    assert len(defended_grads) == len(grads), "Output length should match input"
    
    for orig, defended in zip(grads, defended_grads):
        assert defended.shape == orig.shape, "Shape should be preserved"
        assert not torch.allclose(orig, defended), "Gradients should be modified"
    
    print("[PASS] Combined defense test passed")
    return True

def test_apply_dager_defense():
    """Test the main apply_dager_defense function."""
    print("Testing apply_dager_defense...")
    
    # Create dummy gradients
    grads = (
        torch.randn(3, 4),
        torch.randn(5),
    )
    
    # Create mock args object
    class MockArgs:
        defense_dager_basis_perturb = True
        defense_dager_basis_noise_scale = 0.01
        defense_dager_offset_embedding = False
        defense_dager_offset_scale = 0.01
        defense_dager_gradient_slicing = False
        defense_dager_slice_first_n = None
        defense_dager_slice_last_n = None
        defense_dager_random_slice = False
        defense_dager_slice_prob = 0.5
        defense_dager_rank_limit = False
        rng_seed = 42
    
    args = MockArgs()
    
    # Apply defense
    defended_grads = apply_dager_defense(
        grads,
        args,
        model_wrapper=None,
        batch=None,
        labels=None,
        layer_names=None
    )
    
    # Check results
    assert len(defended_grads) == len(grads), "Output length should match input"
    
    for orig, defended in zip(grads, defended_grads):
        assert defended.shape == orig.shape, "Shape should be preserved"
        assert not torch.allclose(orig, defended), "Gradients should be modified"
    
    print("[PASS] apply_dager_defense test passed")
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing DAGER Defense Implementation")
    print("=" * 60)
    
    try:
        test_dynamic_basis_perturbation()
        test_gradient_slicing()
        test_combined_defense()
        test_apply_dager_defense()
        
        print("\n" + "=" * 60)
        print("All tests passed! [SUCCESS]")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print("\nTest failed with error: {}".format(e))
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())