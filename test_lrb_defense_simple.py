#!/usr/bin/env python3
"""
Simple structure test for the LRB defense integration.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    print("Testing LRB imports...")
    try:
        from utils.lrb_defense import apply_lrb_defense

        assert callable(apply_lrb_defense), "apply_lrb_defense is not callable"
        print("[PASS] LRB defense module imports successfully")
        return True
    except Exception as exc:
        print(f"[FAIL] Import error: {exc}")
        return False


def test_code_structure():
    print("\nTesting LRB code structure...")
    try:
        file_path = os.path.join(os.path.dirname(__file__), "utils", "lrb_defense.py")
        with open(file_path, "r", encoding="utf8") as handle:
            content = handle.read()

        checks = [
            ("def _layer_sensitivity", "layer sensitivity scorer"),
            ("def _project_low_resolution", "low-resolution projection"),
            ("def _orthogonal_residual_noise", "residual-space noise"),
            ("def apply_lrb_defense", "public entry point"),
        ]

        all_passed = True
        for pattern, description in checks:
            if pattern in content:
                print(f"[PASS] Found {description}")
            else:
                print(f"[FAIL] Missing {description}")
                all_passed = False
        return all_passed
    except Exception as exc:
        print(f"[FAIL] Error reading LRB file: {exc}")
        return False


def test_args_integration():
    print("\nTesting args_factory integration...")
    try:
        file_path = os.path.join(os.path.dirname(__file__), "args_factory.py")
        with open(file_path, "r", encoding="utf8") as handle:
            content = handle.read()

        checks = [
            ("'lrb'", "lrb defense option"),
            ("defense_lrb_sensitive_n_layers", "sensitive layer count arg"),
            ("defense_lrb_keep_ratio_sensitive", "sensitive keep ratio arg"),
            ("defense_lrb_noise_sensitive", "sensitive noise arg"),
        ]

        all_passed = True
        for pattern, description in checks:
            if pattern in content:
                print(f"[PASS] Found {description}")
            else:
                print(f"[FAIL] Missing {description}")
                all_passed = False
        return all_passed
    except Exception as exc:
        print(f"[FAIL] Error reading args_factory.py: {exc}")
        return False


def test_defense_router():
    print("\nTesting defenses.py integration...")
    try:
        file_path = os.path.join(os.path.dirname(__file__), "utils", "defenses.py")
        with open(file_path, "r", encoding="utf8") as handle:
            content = handle.read()

        checks = [
            ("from .lrb_defense import apply_lrb_defense", "LRB import"),
            ("elif defense == \"lrb\":", "LRB routing branch"),
        ]

        all_passed = True
        for pattern, description in checks:
            if pattern in content:
                print(f"[PASS] Found {description}")
            else:
                print(f"[FAIL] Missing {description}")
                all_passed = False
        return all_passed
    except Exception as exc:
        print(f"[FAIL] Error reading defenses.py: {exc}")
        return False


def main():
    print("=" * 60)
    print("Testing LRB Defense Integration - Structure Check")
    print("=" * 60)

    results = [
        test_imports(),
        test_code_structure(),
        test_args_integration(),
        test_defense_router(),
    ]
    all_passed = all(results)

    print("\n" + "=" * 60)
    if all_passed:
        print("All LRB structure tests passed! [SUCCESS]")
    else:
        print("Some LRB structure tests failed. [FAILED]")
    print("=" * 60)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
