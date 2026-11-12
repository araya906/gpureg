"""Batched Cox proportional hazards models (Efron or Breslow ties) by Newton-Raphson."""
import numpy as np
import torch

from ._device import check_dtype, resolve_device
from ._inputs import as_matrix, as_vector, check_rows, covariate_matrix, require_complete
from ._linalg import spd_inverse
from ._newton import newton_maximize
from ._results import BatchFit, coefficient_frame, feature_frame
from ._risksets import build_risk_sets, group_sum, reverse_cumsum
from ._screen import screen_features


def _check_ties(ties):
    if ties not in ("efron", "breslow"):
        raise ValueError(f"ties must be 'efron' or 'breslow', got {ties!r}.")


def _cox_setup(time, event, device, dtype):
    """Validate the outcome; return descending-time order, sorted event tensor, risk sets."""
    tv, ev = as_vector(time, "time"), as_vector(event, "event")
    if len(tv) != len(ev):
        raise ValueError("time and event have different lengths.")
    if not np.isin(ev, (0.0, 1.0)).all():
        raise ValueError("event must be coded 0/1.")
    order = np.argsort(-tv, kind="stable")
    rs = build_risk_sets(tv[order], ev[order], device)
    return order, torch.as_tensor(ev[order], device=device, dtype=dtype), rs


def _cox_group_sums(r, X, event, rs):
    """Risk-set sums (s0, s1) and tied-event sums (e0, e1) at event-time groups."""
    n_groups = rs.last.numel()
    at = rs.last[rs.event_groups]
    rx = r.unsqueeze(-1) * X
    s0 = torch.cumsum(r, 1)[:, at]
    s1 = torch.cumsum(rx, 1)[:, at]
    e0 = group_sum(r * event, rs.gid, n_groups)[:, rs.event_groups]
    e1 = group_sum(rx * event[None, :, None], rs.gid, n_groups)[:, rs.event_groups]
    return s0, s1, e0, e1


def _cox_information(r, X, event, rs, valid, frac, den, a):
    """Observed information without materialising per-group (k, k) sums."""
    n_groups = rs.last.numel()
    h_all = r.new_zeros(r.shape[0], n_groups)
    e_all = r.new_zeros(r.shape[0], n_groups)
    h_all[:, rs.event_groups] = (valid / den).sum(2)
    e_all[:, rs.event_groups] = (valid * frac / den).sum(2)
    c = reverse_cumsum(h_all, 1)[:, rs.gid] - event * e_all[:, rs.gid]
    first = (X * (r * c).unsqueeze(-1)).transpose(1, 2) @ X
    flat = a.reshape(a.shape[0], -1, a.shape[-1])
    return first - flat.transpose(1, 2) @ flat


def _cox_objective(X, event, w, rs, efron):
    """fun(beta) -> (partial log-lik, score, information) for a batch of Cox models."""
    dw = w * event
    deaths = group_sum(dw, rs.gid, rs.last.numel())[:, rs.event_groups]
    ell = torch.arange(max(rs.max_ties, 1), device=X.device, dtype=X.dtype)
    valid = ell < deaths.unsqueeze(-1)
    frac = valid * (ell / deaths.clamp(min=1).unsqueeze(-1))
    if not efron:
        frac = torch.zeros_like(frac)
    score0 = torch.einsum("bn,bnk->bk", dw, X)

    def fun(beta):
        eta = torch.einsum("bnk,bk->bn", X, beta)
        shift = torch.nan_to_num(eta.masked_fill(w <= 0, float("-inf")).amax(1), neginf=0.0)
        r = torch.where(w > 0, torch.exp(eta - shift[:, None]), torch.zeros_like(eta))
        s0, s1, e0, e1 = _cox_group_sums(r, X, event, rs)
        den = torch.where(valid, s0.unsqueeze(-1) - frac * e0.unsqueeze(-1), torch.ones_like(frac))
        ll = (dw * eta).sum(1) - (valid * (torch.log(den) + shift[:, None, None])).sum((1, 2))
        a = (s1.unsqueeze(2) - frac.unsqueeze(-1) * e1.unsqueeze(2)) / den.unsqueeze(-1)
        a = a * valid.unsqueeze(-1)
        info = _cox_information(r, X, event, rs, valid, frac, den, a)
        return ll, score0 - a.sum((1, 2)), info
    return fun


def _fit_cox_batch(X, event, w, rs, efron, max_iter, tol):
    fun = _cox_objective(X, event, w, rs, efron)
    beta0 = X.new_zeros(X.shape[0], X.shape[2])
    beta, ll, info, conv, n_iter = newton_maximize(fun, beta0, max_iter, tol)
    cov, singular = spd_inverse(info)
    se = torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2).clamp(min=0))
    return BatchFit(beta, se, ll, conv, n_iter, singular, w.sum(1))


def cox_screen(features, time, event, covariates=None, ties="efron", device=None,
               dtype=torch.float64, chunk_size=None, max_iter=30, tol=None):
    """Fit Surv(time, event) ~ covariates + feature_j for every feature column j.

    Rows where feature j is missing (NaN) are dropped for that model only.
    """
    _check_ties(ties)
    dev = resolve_device(device)
    check_dtype(dev, dtype)
    order, ev, rs = _cox_setup(time, event, dev, dtype)
    Fa, names = as_matrix(features, "feature")
    check_rows(Fa, "features", len(order))
    Ca = covariate_matrix(covariates, len(order))

    def fit_chunk(X, w):
        return _fit_cox_batch(X, ev, w, rs, ties == "efron", max_iter, tol)

    rows = screen_features(Fa[order], Ca[order], fit_chunk, False, chunk_size, dev, dtype,
                           copies=8 + 2 * rs.max_ties)
    return feature_frame(names, rows)


def cox_fit(X, time, event, ties="efron", device=None, dtype=torch.float64,
            max_iter=30, tol=None):
    """Fit a single Cox model and return its coefficient table (diagnostics in .attrs)."""
    _check_ties(ties)
    dev = resolve_device(device)
    check_dtype(dev, dtype)
    order, ev, rs = _cox_setup(time, event, dev, dtype)
    Xa, names = as_matrix(X, "x")
    require_complete(Xa, "X", len(order))
    Xt = torch.as_tensor(Xa[order], device=dev, dtype=dtype).unsqueeze(0)
    w = torch.ones(1, len(order), device=dev, dtype=dtype)
    return coefficient_frame(names, _fit_cox_batch(Xt, ev, w, rs, ties == "efron", max_iter, tol))
