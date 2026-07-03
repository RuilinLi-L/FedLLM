#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from types import ModuleType

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    sys.modules["torch"] = ModuleType("torch")

from utils.gpu import resolve_gradient_device


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def test_auto_gradient_device_follows_resolved_cuda_device():
    assert_true(
        resolve_gradient_device("auto", "cuda:0") == "cuda:0",
        "auto gradient device should follow resolved CUDA device",
    )
    assert_true(
        resolve_gradient_device(None, "cuda:2") == "cuda:2",
        "None gradient device should follow resolved CUDA device",
    )
    assert_true(
        resolve_gradient_device("", "cpu") == "cpu",
        "empty gradient device should follow resolved CPU device",
    )


def test_explicit_gradient_device_overrides_are_preserved():
    assert_true(
        resolve_gradient_device("cpu", "cuda:0") == "cpu",
        "explicit CPU gradient device should preserve the legacy path",
    )
    assert_true(
        resolve_gradient_device("cuda:3", "cuda:0") == "cuda:3",
        "explicit indexed CUDA gradient device should not be rewritten",
    )


def test_bare_cuda_gradient_device_uses_resolved_cuda_index():
    assert_true(
        resolve_gradient_device("cuda", "cuda:1") == "cuda:1",
        "bare CUDA gradient device should use the already selected CUDA index",
    )


def main():
    tests = [
        test_auto_gradient_device_follows_resolved_cuda_device,
        test_explicit_gradient_device_overrides_are_preserved,
        test_bare_cuda_gradient_device_uses_resolved_cuda_index,
    ]
    for test in tests:
        print(f"Running {test.__name__}...")
        test()
    print("All device gradient semantic tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
