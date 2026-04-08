#!/usr/bin/env python3
"""
Simple test script for DAGER defense methods - tests code structure without torch.
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that the modules can be imported."""
    print("Testing imports...")
    
    try:
        # Try to import the module
        from utils.dager_defense import DAGERDefense, apply_dager_defense
        print("[PASS] DAGER defense module imports successfully")
        
        # Check that the class has expected methods
        expected_methods = [
            'dynamic_basis_perturbation',
            'stochastic_offset_embedding', 
            'gradient_slicing',
            'rank_limiting_defense',
            'combined_defense'
        ]
        
        for method in expected_methods:
            assert hasattr(DAGERDefense, method), f"DAGERDefense missing method: {method}"
            assert callable(getattr(DAGERDefense, method)), f"DAGERDefense.{method} is not callable"
        
        print("[PASS] DAGERDefense class has all expected methods")
        
        # Check that apply_dager_defense is callable
        assert callable(apply_dager_defense), "apply_dager_defense is not callable"
        print("[PASS] apply_dager_defense is callable")
        
        return True
        
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Other error: {e}")
        return False

def test_code_structure():
    """Test the code structure by reading the file."""
    print("\nTesting code structure...")
    
    try:
        file_path = os.path.join(os.path.dirname(__file__), 'utils', 'dager_defense.py')
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for key components
        checks = [
            ("class DAGERDefense:", "DAGERDefense class definition"),
            ("def dynamic_basis_perturbation", "dynamic_basis_perturbation method"),
            ("def stochastic_offset_embedding", "stochastic_offset_embedding method"),
            ("def gradient_slicing", "gradient_slicing method"),
            ("def rank_limiting_defense", "rank_limiting_defense method"),
            ("def combined_defense", "combined_defense method"),
            ("def apply_dager_defense", "apply_dager_defense function"),
        ]
        
        all_passed = True
        for pattern, description in checks:
            if pattern in content:
                print(f"[PASS] Found {description}")
            else:
                print(f"[FAIL] Missing {description}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"[FAIL] Error reading file: {e}")
        return False

def test_args_integration():
    """Test that args_factory has been updated."""
    print("\nTesting args_factory integration...")
    
    try:
        file_path = os.path.join(os.path.dirname(__file__), 'args_factory.py')
        with open(file_path, 'r') as f:
            content = f.read()
        
        checks = [
            ("'dager'", "dager defense option in choices"),
            ("defense_dager_basis_perturb", "DAGER basis perturbation parameter"),
            ("defense_dager_basis_noise_scale", "DAGER noise scale parameter"),
            ("defense_dager_offset_embedding", "DAGER offset embedding parameter"),
            ("defense_dager_gradient_slicing", "DAGER gradient slicing parameter"),
        ]
        
        all_passed = True
        for pattern, description in checks:
            if pattern in content:
                print(f"[PASS] Found {description}")
            else:
                print(f"[FAIL] Missing {description}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"[FAIL] Error reading file: {e}")
        return False

def test_defenses_integration():
    """Test that defenses.py has been updated."""
    print("\nTesting defenses.py integration...")
    
    try:
        file_path = os.path.join(os.path.dirname(__file__), 'utils', 'defenses.py')
        with open(file_path, 'r') as f:
            content = f.read()
        
        checks = [
            ("from .dager_defense import apply_dager_defense", "DAGER defense import"),
            ("elif defense == \"dager\":", "DAGER defense case"),
            ("dager defense requires model_wrapper, batch, labels", "DAGER defense error message"),
        ]
        
        all_passed = True
        for pattern, description in checks:
            if pattern in content:
                print(f"[PASS] Found {description}")
            else:
                print(f"[FAIL] Missing {description}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"[FAIL] Error reading file: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing DAGER Defense Implementation - Structure Check")
    print("=" * 60)
    
    try:
        test1 = test_imports()
        test2 = test_code_structure()
        test3 = test_args_integration()
        test4 = test_defenses_integration()
        
        all_passed = test1 and test2 and test3 and test4
        
        print("\n" + "=" * 60)
        if all_passed:
            print("All structure tests passed! [SUCCESS]")
            print("\nSummary:")
            print("- Created dager_defense.py with DAGER-specific defense methods")
            print("- Updated args_factory.py with DAGER defense parameters")
            print("- Updated defenses.py to integrate DAGER defense")
            print("- Implemented 4 defense strategies:")
            print("  1. Dynamic Basis Perturbation")
            print("  2. Stochastic Offset Embedding")
            print("  3. Gradient Slicing & Chunking")
            print("  4. Rank-Limiting Defense")
        else:
            print("Some tests failed. [FAILED]")
        print("=" * 60)
        
        return 0 if all_passed else 1
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())