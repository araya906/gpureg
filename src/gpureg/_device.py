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


def auto_chunk_size(n, k, dtype, device, copies, cpu_budget=256e6, cpu_cap=64):
    """Models per batch so that about `copies` (n, k) tensors per model fit in memory.

    On CUDA the budget is half the free memory. Elsewhere (CPU, MPS) a small budget and at most `cpu_cap`
    models keep chunks cache-friendly (validation/results/bench_chunks.md: 16-64 models were
    fastest; 1-2 GB chunks were 1.3-2.2x slower).
    """
    per_model = n * max(k, 1) * (torch.finfo(dtype).bits // 8) * copies
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        return max(1, int(0.5 * free // per_model))
    return max(1, min(cpu_cap, int(cpu_budget // per_model)))
