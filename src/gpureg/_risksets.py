"""Tied-time group structure for Cox partial likelihoods.

Rows are sorted by stratum, then by descending time within stratum; risk sets never cross strata.
"""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RiskSets:
    gid: torch.Tensor           # (n,) group index of each row
    last: torch.Tensor          # (G,) row index of the last member of each group
    event_groups: torch.Tensor  # (Ge,) groups with at least one event
    max_ties: int               # largest number of events sharing one time
    start: torch.Tensor         # (G,) first row of each group's stratum
    stop: torch.Tensor          # (G,) last group of each group's stratum


def build_risk_sets(time_desc, event, device, strata=None):
    """Group sorted rows into tied-time groups; `strata` (sorted codes) splits risk sets."""
    t = np.asarray(time_desc)
    s = np.zeros(len(t)) if strata is None else np.asarray(strata)
    new_stratum = np.r_[True, s[1:] != s[:-1]]
    new_group = new_stratum | np.r_[True, t[1:] != t[:-1]]
    gid = np.cumsum(new_group) - 1
    first = np.flatnonzero(new_group)
    last = np.r_[first[1:] - 1, len(t) - 1]
    sid = (np.cumsum(new_stratum) - 1)[first]
    s_first = np.flatnonzero(new_stratum)
    stop = gid[np.r_[s_first[1:] - 1, len(t) - 1]][sid]
    deaths = np.bincount(gid, weights=np.asarray(event) > 0, minlength=len(first))
    max_ties = int(deaths.max()) if deaths.size else 0
    idx = [torch.as_tensor(np.asarray(a, dtype=np.int64), device=device)
           for a in (gid, last, np.flatnonzero(deaths > 0), s_first[sid], stop)]
    return RiskSets(*idx[:3], max_ties, *idx[3:])


def group_sum(values, gid, n_groups):
    """Sum (b, n, ...) values over rows into (b, G, ...) by group index."""
    out = values.new_zeros((values.shape[0], n_groups) + tuple(values.shape[2:]))
    return out.index_add_(1, gid, values)


def risk_set_sum(values, rs, groups):
    """Sum (b, n, ...) rows at risk at `groups` (same stratum, row <= group end): (b, len, ...)."""
    cs = torch.cumsum(values, 1)
    cs = torch.cat([torch.zeros_like(cs[:, :1]), cs], 1)
    return cs[:, rs.last[groups] + 1] - cs[:, rs.start[groups]]


def later_group_sum(x, rs):
    """For (b, G) x, sum over groups g..(last group of g's stratum) for every group g."""
    rc = reverse_cumsum(x, 1)
    return rc - torch.cat([rc, torch.zeros_like(rc[:, :1])], 1)[:, rs.stop + 1]


def reverse_cumsum(x, dim):
    """Cumulative sum from the end of `dim` backwards."""
    return torch.flip(torch.cumsum(torch.flip(x, (dim,)), dim), (dim,))
