"""Device, dtype and memory helpers."""
import torch


def resolve_device(device=None):
    """Return the requested torch device, defaulting to CUDA when available."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_dtype(device, dtype):
    """Reject float64 on Apple MPS, which does not support it."""
    if device.type == "mps" and dtype == torch.float64:
        raise ValueError("MPS does not support float64; pass dtype=torch.float32.")


def auto_chunk_size(n, k, dtype, device, copies, cpu_budget=2e9):
    """Models per batch so that about `copies` (n, k) tensors per model fit in memory."""
    per_model = n * max(k, 1) * (torch.finfo(dtype).bits // 8) * copies
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        budget = 0.5 * free
    else:
        budget = cpu_budget
    return max(1, int(budget // per_model))
