"""Lasso / elastic-net paths with K-fold CV, fitted by batched FISTA (folds = one GPU batch).

Objective per model: -loglik / n_w + lambda * sum_j pf_j * (alpha |b_j| + (1 - alpha) / 2 b_j^2),
on standardized columns (population SD), matching glmnet / scikit-learn scaling.
"""
import math
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ._device import check_dtype, resolve_device
from ._inputs import as_matrix, as_vector, require_complete
from ._risksets import build_risk_sets, group_sum, reverse_cumsum

LASSO_FAMILIES = ("gaussian", "binomial", "cox")
_NULL_LAMBDA = 1e12  # keeps every penalized coefficient at exactly zero


@dataclass
class _LassoProblem:
    family: str
    X: torch.Tensor      # (n, k) standardized design, intercept column first if present
    y: torch.Tensor      # outcome; event indicator for cox (rows sorted by descending time)
    W: torch.Tensor      # (1 + n_folds, n) training weights; row 0 = full data
    pf: torch.Tensor     # (k,) penalty factors, 0 = unpenalized
    mean: torch.Tensor
    scale: torch.Tensor
    intercept: bool
    rs: object
    names: list


@dataclass
class LassoResult:
    feature_names: list
    lambdas: np.ndarray
    coef: np.ndarray                  # (n_lambda, p) on the original scale
    intercept: Optional[np.ndarray]   # (n_lambda,), None for cox
    cv_mean: Optional[np.ndarray]     # mean held-out deviance per lambda
    cv_se: Optional[np.ndarray]
    penalized: np.ndarray             # (p,) bool
    family: str

    def _require_cv(self):
        if self.cv_mean is None:
            raise ValueError("fit with n_folds >= 2 to use cross-validated lambdas.")

    @property
    def lambda_min(self):
        self._require_cv()
        return float(self.lambdas[np.argmin(self.cv_mean)])

    @property
    def lambda_1se(self):
        self._require_cv()
        i = np.argmin(self.cv_mean)
        return float(self.lambdas[self.cv_mean <= self.cv_mean[i] + self.cv_se[i]].max())

    def _lambda_index(self, lam):
        if lam == "min":
            lam = self.lambda_min
        elif lam == "1se":
            lam = self.lambda_1se
        return int(np.argmin(np.abs(np.log(self.lambdas) - math.log(lam))))

    def coef_at(self, lam="1se"):
        """Coefficients at 'min', '1se' or the path lambda nearest to a value."""
        return pd.Series(self.coef[self._lambda_index(lam)], index=self.feature_names)

    def selected(self, lam="1se"):
        """Penalized features with non-zero coefficients."""
        c = self.coef_at(lam)
        return list(c.index[(c.to_numpy() != 0) & self.penalized])

    def path_frame(self):
        return pd.DataFrame({"lambda": self.lambdas,
                             "n_nonzero": ((self.coef != 0) & self.penalized).sum(1),
                             "cv_mean": self.cv_mean, "cv_se": self.cv_se})


def _lasso_inputs(X, y, family, time, event):
    """Arrays for the problem; cox rows are sorted by descending time."""
    if family not in LASSO_FAMILIES:
        raise ValueError(f"family must be one of {LASSO_FAMILIES}, got {family!r}.")
    Xa, names = as_matrix(X, "x")
    tv = None
    if family == "cox":
        tv, yv = as_vector(time, "time"), as_vector(event, "event")
        order = np.argsort(-tv, kind="stable")
        tv, yv, Xa = tv[order], yv[order], Xa[order]
    else:
        yv = as_vector(y, "y")
    if family != "gaussian" and not np.isin(yv, (0.0, 1.0)).all():
        raise ValueError("binomial outcome / cox event must be coded 0/1.")
    require_complete(Xa, "X", len(yv))
    return Xa, names, yv, tv


def _lasso_penalty(names, unpenalized, intercept):
    pf = np.ones(len(names))
    for item in unpenalized or []:
        pf[int(item) if isinstance(item, (int, np.integer)) else names.index(str(item))] = 0.0
    return np.concatenate([[0.0], pf]) if intercept else pf


def _lasso_standardize(Xt, intercept):
    mean = Xt.mean(0)
    scale = Xt.std(0, correction=0)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    Xs = (Xt - mean) / scale
    if intercept:
        Xs = torch.cat([torch.ones_like(Xs[:, :1]), Xs], 1)
        mean, scale = torch.cat([mean.new_zeros(1), mean]), torch.cat([scale.new_ones(1), scale])
    return Xs, mean, scale


def _lasso_folds(n, n_folds, seed, device, dtype):
    rows = [np.ones(n)]
    if n_folds:
        if n_folds < 2:
            raise ValueError("n_folds must be 0 (no CV) or >= 2.")
        folds = np.random.default_rng(seed).permutation(n) % n_folds
        rows += [(folds != k).astype(float) for k in range(n_folds)]
    return torch.as_tensor(np.stack(rows), device=device, dtype=dtype)


def _lasso_problem(X, y, family, time, event, unpenalized, n_folds, seed, device, dtype):
    Xa, names, yv, tv = _lasso_inputs(X, y, family, time, event)
    intercept = family != "cox"
    Xs, mean, scale = _lasso_standardize(torch.as_tensor(Xa, device=device, dtype=dtype), intercept)
    pf = torch.as_tensor(_lasso_penalty(names, unpenalized, intercept), device=device, dtype=dtype)
    rs = build_risk_sets(tv, yv, device) if family == "cox" else None
    W = _lasso_folds(len(yv), n_folds, seed, device, dtype)
    yt = torch.as_tensor(yv, device=device, dtype=dtype)
    return _LassoProblem(family, Xs, yt, W, pf, mean, scale, intercept, rs, names)


def _lasso_breslow(eta, event, w, rs):
    """Breslow partial log-likelihood and residual r with d loglik / d beta = X^T r."""
    dw = w * event
    shift = torch.nan_to_num(eta.masked_fill(w <= 0, float("-inf")).amax(1), neginf=0.0)
    r = torch.where(w > 0, torch.exp(eta - shift[:, None]), torch.zeros_like(eta))
    d = group_sum(dw, rs.gid, rs.last.numel())
    has = d > 0
    s0 = torch.where(has, torch.cumsum(r, 1)[:, rs.last], torch.ones_like(d))
    ll = (dw * eta).sum(1) - (d * (torch.log(s0) + shift[:, None])).sum(1)
    h = torch.where(has, d / s0, torch.zeros_like(d))
    return ll, dw - r * reverse_cumsum(h, 1)[:, rs.gid]


def _lasso_loglik(family, eta, y, w, rs):
    """Weighted log-likelihood per model and residual r with d loglik / d beta = X^T r."""
    if family == "gaussian":
        res = y - eta
        return -0.5 * (w * res ** 2).sum(1), w * res
    if family == "binomial":
        return (w * (y * eta - F.softplus(eta))).sum(1), w * (y - torch.sigmoid(eta))
    return _lasso_breslow(eta, y, w, rs)


def _lasso_objective(prob, w):
    """obj(beta) -> (-loglik / n_w, gradient) for a (b, k) batch of coefficients."""
    n_w = w.sum(1)

    def obj(beta, grad=True):
        ll, resid = _lasso_loglik(prob.family, beta @ prob.X.T, prob.y, w, prob.rs)
        g = -(resid @ prob.X) / n_w[:, None] if grad else None
        return -ll / n_w, g
    return obj


def _soft_prox(u, step, lam, pf, alpha):
    thr = step[:, None] * lam * alpha * pf
    shrink = 1 + step[:, None] * lam * (1 - alpha) * pf
    return torch.sign(u) * torch.clamp(u.abs() - thr, min=0) / shrink


def _backtrack(obj, z, fz, gz, L, lam, pf, alpha, max_doublings=50):
    """Proximal step with per-model backtracking on the Lipschitz estimate L.

    The acceptance slack is relative to |f| and scaled to the dtype's rounding error,
    so float32 objectives do not keep doubling L on rounding noise.
    """
    slack = 10 * torch.finfo(fz.dtype).eps
    for _ in range(max_doublings):
        step = 1.0 / L
        cand = _soft_prox(z - step[:, None] * gz, step, lam, pf, alpha)
        diff = cand - z
        fc, _ = obj(cand, grad=False)
        bound = fz + (gz * diff).sum(1) + 0.5 * L * (diff * diff).sum(1)
        ok = fc <= bound + slack * fz.abs()
        if bool(ok.all()):
            break
        L = torch.where(ok, L, 2 * L)
    return cand, L


def _fista(obj, beta, lam, pf, alpha, L, max_iter, tol):
    """Accelerated proximal gradient with adaptive restart, batched over models."""
    z, t = beta.clone(), torch.ones_like(L)
    it = 0
    for it in range(1, max_iter + 1):
        fz, gz = obj(z)
        new, L = _backtrack(obj, z, fz, gz, L, lam, pf, alpha)
        restart = ((z - new) * (new - beta)).sum(1) > 0
        t_next = (1 + torch.sqrt(1 + 4 * t * t)) / 2
        mom = torch.where(restart, torch.zeros_like(t), (t - 1) / t_next)
        z = new + mom[:, None] * (new - beta)
        t = torch.where(restart, torch.ones_like(t), t_next)
        delta = (new - beta).abs().amax(1)
        beta = new
        if bool((delta < tol).all()):
            break
    return beta, L, it


def _lasso_lipschitz(prob, n_iter=30):
    """Starting Lipschitz estimate from the top eigenvalue of X^T X / n (power iteration)."""
    X = prob.X
    v = X.new_ones(X.shape[1]) / math.sqrt(X.shape[1])
    for _ in range(n_iter):
        v = X.T @ (X @ v)
        v = v / v.norm()
    top = float(v @ (X.T @ (X @ v))) / X.shape[0]
    factor = {"gaussian": 1.0, "binomial": 0.25}.get(prob.family)
    return max(top * (factor if factor is not None else float(prob.y.mean())), 1e-8)


def _lasso_lambdas(prob, obj, alpha, n_lambda, ratio, L, max_iter, tol):
    """Fit the unpenalized part, then a log-spaced path from lambda_max (full data)."""
    beta = prob.X.new_zeros(prob.W.shape[0], prob.X.shape[1])
    beta, L, _ = _fista(obj, beta, beta.new_tensor(_NULL_LAMBDA), prob.pf, alpha, L, max_iter, tol)
    _, g = obj(beta)
    pen = prob.pf > 0
    if not bool(pen.any()):
        raise ValueError("every column is unpenalized; nothing to select.")
    lam_max = float((g[0, pen].abs() / prob.pf[pen]).max()) / max(alpha, 1e-3)
    if ratio is None:
        ratio = 1e-4 if prob.X.shape[0] > prob.X.shape[1] else 1e-2
    return np.geomspace(lam_max, lam_max * ratio, n_lambda), beta, L


def _lasso_cv_deviance(prob, beta):
    """(n_folds,) held-out deviance per observation; Verweij-van Houwelingen for cox."""
    train, eta = prob.W[1:], beta[1:] @ prob.X.T
    held = 1.0 - train
    n_held = held.sum(1)
    if prob.family == "cox":
        ll_full, _ = _lasso_loglik("cox", eta, prob.y, torch.ones_like(held), prob.rs)
        ll_train, _ = _lasso_loglik("cox", eta, prob.y, train, prob.rs)
        return -2 * (ll_full - ll_train) / n_held
    ll, _ = _lasso_loglik(prob.family, eta, prob.y, held, prob.rs)
    return -2 * ll / n_held


def _lasso_run(prob, obj, lambdas, beta, L, alpha, max_iter, tol):
    coefs, devs, iters = [], [], []
    for lam in lambdas:
        beta, L, it = _fista(obj, beta, beta.new_tensor(float(lam)), prob.pf, alpha, L, max_iter, tol)
        coefs.append(beta[0].clone())
        iters.append(it)
        if prob.W.shape[0] > 1:
            devs.append(_lasso_cv_deviance(prob, beta))
    return torch.stack(coefs), (torch.stack(devs) if devs else None), iters


def _lasso_result(prob, lambdas, coefs, devs):
    beta = coefs / prob.scale
    if prob.intercept:
        beta[:, 0] -= (beta[:, 1:] * prob.mean[1:]).sum(1)
    beta = beta.cpu().numpy()
    pf = prob.pf.cpu().numpy()
    cv_mean = cv_se = None
    if devs is not None:
        d = devs.cpu().numpy()
        cv_mean, cv_se = d.mean(1), d.std(1, ddof=1) / math.sqrt(d.shape[1])
    if prob.intercept:
        return LassoResult(prob.names, lambdas, beta[:, 1:], beta[:, 0], cv_mean, cv_se,
                           pf[1:] > 0, prob.family)
    return LassoResult(prob.names, lambdas, beta, None, cv_mean, cv_se, pf > 0, prob.family)


def lasso_cv(X, y=None, *, family="gaussian", time=None, event=None, unpenalized=None,
             alpha=1.0, n_lambda=100, lambda_min_ratio=None, n_folds=10, seed=0,
             device=None, dtype=torch.float64, max_iter=5000, tol=1e-7):
    """Lasso (alpha=1) or elastic-net path with K-fold CV; all folds fitted as one batch.

    family: 'gaussian' or 'binomial' (pass y), or 'cox' (pass time and event; Breslow ties).
    unpenalized: column names or indices kept in every model (e.g. confounders).
    n_folds=0 fits the path on the full data only.
    """
    dev = resolve_device(device)
    check_dtype(dev, dtype)
    prob = _lasso_problem(X, y, family, time, event, unpenalized, n_folds, seed, dev, dtype)
    obj = _lasso_objective(prob, prob.W)
    L = torch.full((prob.W.shape[0],), _lasso_lipschitz(prob), device=dev, dtype=dtype)
    lambdas, beta, L = _lasso_lambdas(prob, obj, alpha, n_lambda, lambda_min_ratio, L, max_iter, tol)
    coefs, devs, iters = _lasso_run(prob, obj, lambdas, beta, L, alpha, max_iter, tol)
    if max(iters) >= max_iter:
        warnings.warn("FISTA reached max_iter for some lambdas; increase max_iter or tol.")
    return _lasso_result(prob, lambdas, coefs, devs)
