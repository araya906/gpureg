"""Tied-time group structure for Cox partial likelihoods.

Rows are sorted by stratum, then by descending time within stratum; risk sets never cross strata.
With entry times (left truncation / counting-process data) a row is at risk at t when entry < t <= time.
"""
from dataclasses import dataclass
from typing import Optional

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
    entry: Optional["EntrySets"] = None


@dataclass
class EntrySets:
    order: torch.Tensor  # (n,) rows sorted by stratum, then descending entry time
    last: torch.Tensor   # (G,) last position in `order` (same stratum) with entry >= group time
    gid: torch.Tensor    # (n,) first group of the row's stratum with time <= entry; G if none


def _entry_sets(t, e, s, gid, first, stop, device):
    """Positions of rows not yet at risk (entry >= t), via exact integer ranks of the times."""
    vals = np.unique(np.r_[t, e])
    big = len(vals)
    rank_t, rank_e = np.searchsorted(vals, t), np.searchsorted(vals, e)
    order = np.lexsort((-e, s))
    key_rows = s[order] * big + (big - 1 - rank_e[order])
    key_groups = s[first] * big + (big - 1 - rank_t[first])
    last = np.searchsorted(key_rows, key_groups, side="right") - 1
    egid = np.searchsorted(key_groups, s * big + (big - 1 - rank_e), side="left")
    egid = np.where(egid > stop[gid], len(first), egid)
    return EntrySets(*(torch.as_tensor(np.asarray(a, dtype=np.int64), device=device)
                       for a in (order, last, egid)))


def build_risk_sets(time_desc, event, device, strata=None, entry=None):
    """Group sorted rows into tied-time groups; `strata` (sorted codes) splits risk sets and
    `entry` (row-aligned entry times) removes rows from risk sets at times <= their entry."""
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
    ent = None if entry is None else _entry_sets(t, np.asarray(entry), s, gid, first, stop, device)
    return RiskSets(*idx[:3], max_ties, *idx[3:], ent)


def group_sum(values, gid, n_groups):
    """Sum (b, n, ...) values over rows into (b, G, ...) by group index."""
    out = values.new_zeros((values.shape[0], n_groups) + tuple(values.shape[2:]))
    return out.index_add_(1, gid, values)


def _padded_cumsum(values):
    cs = torch.cumsum(values, 1)
    return torch.cat([torch.zeros_like(cs[:, :1]), cs], 1)


def risk_set_sum(values, rs, groups):
    """Sum (b, n, ...) rows at risk at `groups` (same stratum, row <= group end, entered before
    the group time): (b, len, ...). A stratum starts at the same position in both row orders."""
    cs = _padded_cumsum(values)
    out = cs[:, rs.last[groups] + 1] - cs[:, rs.start[groups]]
    if rs.entry is not None:
        ce = _padded_cumsum(values[:, rs.entry.order])
        out = out - (ce[:, rs.entry.last[groups] + 1] - ce[:, rs.start[groups]])
    return out


def later_group_sum(x, rs):
    """For (b, G) x and every row i, sum x over the groups of i's stratum at which i is at risk
    (groups gid(i)..stratum end, minus those at or before its entry): (b, n)."""
    rc = reverse_cumsum(x, 1)
    after = torch.cat([rc, torch.zeros_like(rc[:, :1])], 1)
    out = (rc - after[:, rs.stop + 1])[:, rs.gid]
    if rs.entry is not None:
        egid = rs.entry.gid.clamp(max=rc.shape[1] - 1)
        out = out - (after[:, rs.entry.gid] - after[:, rs.stop[egid] + 1])
    return out


def reverse_cumsum(x, dim):
    """Cumulative sum from the end of `dim` backwards."""
    return torch.flip(torch.cumsum(torch.flip(x, (dim,)), dim), (dim,))
