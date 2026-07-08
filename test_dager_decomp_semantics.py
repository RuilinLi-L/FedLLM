#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
except ModuleNotFoundError:
    print("[SKIP] torch is unavailable; DAGER decomposition tests require torch.")
    raise SystemExit(0)

from utils.functional import get_layer_decomp, resolve_dager_decomp_device, torch_matrix_rank


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def low_rank_matrix(dtype=torch.float32, device="cpu"):
    matrix = torch.zeros((6, 5), dtype=dtype, device=device)
    matrix[0, 0] = 4.0
    matrix[1, 1] = 3.0
    matrix[2, 2] = 2.0
    return matrix


def projection_from_basis(R):
    eye = torch.eye(R.shape[1], dtype=R.dtype, device=R.device)
    return eye @ R.T @ R


def test_cpu_decomposition_rank_shape_device_and_dtype():
    matrix = low_rank_matrix()
    rank = torch_matrix_rank(matrix, tol=1e-5, device="cpu")
    B, R = get_layer_decomp(matrix, B=None, tol=1e-5, device="cpu")

    assert_true(rank == 3, f"expected rank 3, got {rank}")
    assert_true(B == 3, f"expected returned B 3, got {B}")
    assert_true(tuple(R.shape) == (3, 5), f"unexpected R shape: {tuple(R.shape)}")
    assert_true(R.device.type == "cpu", f"expected CPU R, got {R.device}")
    assert_true(R.dtype == torch.float32, f"expected float32 R, got {R.dtype}")


def test_double_decomposition_preserves_dtype():
    matrix = low_rank_matrix(dtype=torch.float64)
    _, R = get_layer_decomp(matrix, B=3, tol=1e-8, device="cpu")
    assert_true(R.dtype == torch.float64, f"expected float64 R, got {R.dtype}")


def test_auto_decomposition_device_resolution():
    device = resolve_dager_decomp_device("auto", "cuda:0")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert_true(device.type == expected, f"expected {expected} auto device, got {device}")


def test_cpu_and_gpu_projection_are_close_when_cuda_is_available():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA is unavailable; skipping CPU/GPU projection comparison.")
        return

    matrix_cpu = low_rank_matrix()
    matrix_gpu = matrix_cpu.to("cuda")
    _, R_cpu = get_layer_decomp(matrix_cpu, B=3, tol=1e-5, device="cpu")
    _, R_gpu = get_layer_decomp(matrix_gpu, B=3, tol=1e-5, device="cuda")

    assert_true(R_gpu.device.type == "cuda", f"expected CUDA R, got {R_gpu.device}")
    proj_cpu = projection_from_basis(R_cpu)
    proj_gpu = projection_from_basis(R_gpu).cpu()
    assert_true(
        torch.allclose(proj_cpu, proj_gpu, atol=1e-3, rtol=1e-3),
        "CPU and GPU span projections should be approximately equal",
    )


def main():
    tests = [
        test_cpu_decomposition_rank_shape_device_and_dtype,
        test_double_decomposition_preserves_dtype,
        test_auto_decomposition_device_resolution,
        test_cpu_and_gpu_projection_are_close_when_cuda_is_available,
    ]
    for test in tests:
        print(f"Running {test.__name__}...")
        test()
    print("All DAGER decomposition semantic tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
