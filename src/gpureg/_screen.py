"""Chunked one-model-per-feature screening shared by GLM and Cox."""
import numpy as np
import torch

from ._device import auto_chunk_size
from ._results import term_summary


def _chunk_design(Fc, Ct, intercept):
    """(b, n, k) designs [1?, covariates, feature] and (b, n) weights masking missing features."""
    f = torch.as_tensor(np.ascontiguousarray(Fc.T), device=Ct.device, dtype=Ct.dtype)
    w = (~torch.isnan(f)).to(Ct.dtype)
    f = torch.nan_to_num(f, nan=0.0)
    b, n = f.shape
    cols = [Ct.unsqueeze(0).expand(b, n, Ct.shape[1]), f.unsqueeze(-1)]
    if intercept:
        cols.insert(0, torch.ones_like(f).unsqueeze(-1))
    return torch.cat(cols, dim=2), w


def screen_features(F, C, fit_chunk, intercept, chunk_size, device, dtype, copies):
    """Fit outcome ~ [1] + C + F[:, j] for every column j; summarise the feature term."""
    n, m = F.shape
    if m == 0:
        raise ValueError("features has no columns.")
    k = C.shape[1] + 1 + int(intercept)
    size = chunk_size or auto_chunk_size(n, k, dtype, device, copies)
    Ct = torch.as_tensor(C, device=device, dtype=dtype)
    parts = []
    for start in range(0, m, size):
        X, w = _chunk_design(F[:, start:start + size], Ct, intercept)
        parts.append(term_summary(fit_chunk(X, w), -1))
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
