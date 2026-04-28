import torch


def resolve_cuda_device(requested_device: str = "cuda") -> str:
    """Resolve bare 'cuda' to the visible GPU with the most free memory."""
    if requested_device != "cuda" or not torch.cuda.is_available():
        return requested_device

    best_idx = 0
    best_free = -1
    for idx in range(torch.cuda.device_count()):
        with torch.cuda.device(idx):
            free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes > best_free:
            best_free = free_bytes
            best_idx = idx

    return f"cuda:{best_idx}"
