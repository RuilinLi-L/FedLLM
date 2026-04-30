from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

import torch


_MIB = 1024 * 1024


@dataclass(frozen=True)
class _GpuSnapshot:
    visible_index: int
    free_mb: int
    total_mb: int
    utilization: int | None


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _parse_int(value: str) -> int | None:
    value = value.strip()
    if value.upper() == "N/A":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _query_nvidia_smi() -> dict[str, tuple[int | None, int | None, int | None]]:
    """Return physical GPU index -> (utilization %, free MiB, total MiB)."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return {}

    if proc.returncode != 0:
        return {}

    out: dict[str, tuple[int | None, int | None, int | None]] = {}
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        gpu_index = parts[0]
        out[gpu_index] = (_parse_int(parts[1]), _parse_int(parts[2]), _parse_int(parts[3]))
    return out


def _visible_physical_indices(device_count: int) -> list[str] | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible.strip() == "":
        return [str(idx) for idx in range(device_count)]

    entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
    if len(entries) < device_count:
        return None
    if all(entry.isdigit() for entry in entries):
        return entries[:device_count]
    return None


def _torch_memory_mb(visible_index: int) -> tuple[int, int]:
    with torch.cuda.device(visible_index):
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    return int(free_bytes // _MIB), int(total_bytes // _MIB)


def _gpu_snapshots() -> list[_GpuSnapshot]:
    device_count = torch.cuda.device_count()
    smi = _query_nvidia_smi()
    physical_indices = _visible_physical_indices(device_count)

    snapshots: list[_GpuSnapshot] = []
    for visible_index in range(device_count):
        free_mb, total_mb = _torch_memory_mb(visible_index)
        utilization = None

        if physical_indices is not None:
            smi_info = smi.get(physical_indices[visible_index])
            if smi_info is not None:
                utilization, smi_free_mb, smi_total_mb = smi_info
                if smi_free_mb is not None:
                    free_mb = smi_free_mb
                if smi_total_mb is not None:
                    total_mb = smi_total_mb

        snapshots.append(
            _GpuSnapshot(
                visible_index=visible_index,
                free_mb=free_mb,
                total_mb=total_mb,
                utilization=utilization,
            )
        )

    return snapshots


def _select_gpu(snapshots: list[_GpuSnapshot]) -> _GpuSnapshot:
    min_free_mb = _env_int("DAGER_GPU_MIN_FREE_MB", 8000)
    max_util = _env_int("DAGER_GPU_MAX_UTIL", 10)
    has_utilization = any(snapshot.utilization is not None for snapshot in snapshots)

    def util_or_busy(snapshot: _GpuSnapshot) -> int:
        if snapshot.utilization is None:
            return 101
        return snapshot.utilization

    if has_utilization:
        idle = [
            snapshot
            for snapshot in snapshots
            if snapshot.free_mb >= min_free_mb and util_or_busy(snapshot) <= max_util
        ]
        if idle:
            return max(idle, key=lambda snapshot: (snapshot.free_mb, -snapshot.visible_index))

        enough_memory = [snapshot for snapshot in snapshots if snapshot.free_mb >= min_free_mb]
        if enough_memory:
            return min(enough_memory, key=lambda snapshot: (util_or_busy(snapshot), -snapshot.free_mb, snapshot.visible_index))

    return max(snapshots, key=lambda snapshot: (snapshot.free_mb, -util_or_busy(snapshot), -snapshot.visible_index))


def resolve_cuda_device(requested_device: str = "cuda") -> str:
    """Resolve bare 'cuda' to an idle visible GPU.

    Selection prefers GPUs with utilization <= DAGER_GPU_MAX_UTIL (default 10)
    and free memory >= DAGER_GPU_MIN_FREE_MB (default 8000). If utilization is
    unavailable, it falls back to the visible GPU with the most free memory.
    """
    if requested_device != "cuda":
        return requested_device
    if not torch.cuda.is_available():
        return requested_device

    snapshots = _gpu_snapshots()
    if not snapshots:
        return requested_device

    best = _select_gpu(snapshots)
    return f"cuda:{best.visible_index}"
