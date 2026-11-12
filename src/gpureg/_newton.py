"""Batched Newton-Raphson maximisation with per-model step halving."""
import torch

from ._linalg import spd_solve


def _take_step(beta, step, scale):
    """beta + scale * step; scale == 0 leaves beta untouched even if step is not finite."""
    moved = scale[:, None] * step
    return beta + torch.where(scale[:, None] > 0, moved, torch.zeros_like(moved))


def _rounding_floor(dtype, base):
    """base, raised to a small multiple of the dtype's machine epsilon (float32: ~1e-6)."""
    return max(base, 10 * torch.finfo(dtype).eps)


def _halve_until_better(fun, beta, step, ll_old, max_halving):
    """Shrink each model's step until its log-likelihood does not decrease.

    The no-decrease slack is relative to |ll| and never below the dtype's rounding error.
    Models that never improve are returned unchanged and flagged as failed.
    """
    slack = _rounding_floor(ll_old.dtype, 1e-10)
    scale = torch.ones_like(ll_old)
    for _ in range(max_halving):
        cand = _take_step(beta, step, scale)
        ll, score, info = fun(cand)
        worse = ~(ll >= ll_old - slack * ll_old.abs())
        if not bool(worse.any()):
            return cand, ll, score, info, worse
        scale = torch.where(worse, scale / 2, scale)
    scale = torch.where(worse, torch.zeros_like(scale), scale)
    cand = _take_step(beta, step, scale)
    ll, score, info = fun(cand)
    return cand, ll, score, info, worse


def newton_maximize(fun, beta, max_iter=25, tol=None, max_halving=10):
    """Maximise a batch of concave log-likelihoods; fun(beta) -> (ll, score, information).

    Convergence uses the relative change in log-likelihood, as in R's coxph.
    tol (default 1e-9) is raised to the dtype's rounding floor (float32: ~1e-6).
    Returns (beta, ll, information at beta, converged mask, iterations).
    """
    tol = _rounding_floor(beta.dtype, 1e-9 if tol is None else tol)
    ll, score, info = fun(beta)
    done = torch.zeros_like(ll, dtype=torch.bool)
    stuck = torch.zeros_like(done)
    n_iter = torch.zeros_like(ll, dtype=torch.long)
    for _ in range(max_iter):
        active = ~(done | stuck)
        if not bool(active.any()):
            break
        step = torch.where(active[:, None], spd_solve(info, score), torch.zeros_like(score))
        beta, ll_new, score, info, failed = _halve_until_better(fun, beta, step, ll, max_halving)
        stuck |= active & failed
        small = (ll_new - ll).abs() <= tol * (ll.abs() + tol)
        done |= active & ~failed & small
        n_iter += active.long()
        ll = ll_new
    return beta, ll, info, done, n_iter
