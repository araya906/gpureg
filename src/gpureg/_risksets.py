"""Tied-time group structure for Cox partial likelihoods (rows sorted by descending time)."""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RiskSets:
    gid: torch.Tensor           # (n,) group index of each row
    last: torch.Tensor          # (G,) row index of the last member of each group
    event_groups: torch.Tensor  # (Ge,) groups with at least one event
    max_ties: int               # largest number of events sharing one time


def build_risk_sets(time_desc, event, device):
    """Group rows (already sorted by descending time) into tied-time groups."""
    t = torch.as_tensor(np.ascontiguousarray(time_desc), device=device)
    _, counts = torch.unique_consecutive(t, return_counts=True)
    gid = torch.repeat_interleave(torch.arange(counts.numel(), device=device), counts)
    ev = torch.as_tensor(np.asarray(event) > 0, device=device).long()
    deaths = torch.zeros(counts.numel(), dtype=torch.long, device=device).index_add_(0, gid, ev)
    groups = torch.nonzero(deaths > 0).squeeze(1)
    max_ties = int(deaths.max().item()) if deaths.numel() else 0
    return RiskSets(gid, torch.cumsum(counts, 0) - 1, groups, max_ties)


def group_sum(values, gid, n_groups):
    """Sum (b, n, ...) values over rows into (b, G, ...) by group index."""
    out = values.new_zeros((values.shape[0], n_groups) + tuple(values.shape[2:]))
    return out.index_add_(1, gid, values)


def reverse_cumsum(x, dim):
    """Cumulative sum from the end of `dim` backwards."""
    return torch.flip(torch.cumsum(torch.flip(x, (dim,)), dim), (dim,))
