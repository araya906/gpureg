"""Batched generalized linear models (canonical links) fitted by Newton-Raphson."""
import numpy as np
import torch
import torch.nn.functional as F

from ._device import check_dtype, resolve_device
from ._inputs import as_matrix, as_vector, check_rows, covariate_matrix, require_complete
from ._linalg import spd_inverse
from ._newton import newton_maximize
from ._results import BatchFit, coefficient_frame, feature_frame
from ._screen import screen_features

GLM_FAMILIES = ("gaussian", "binomial", "poisson")


def _check_family(family, y):
    if family not in GLM_FAMILIES:
        raise ValueError(f"family must be one of {GLM_FAMILIES}, got {family!r}.")
    if family == "binomial" and not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("binomial outcome must be coded 0/1.")
    if family == "poisson" and (y < 0).any():
        raise ValueError("poisson outcome must be non-negative.")


def _glm_terms(family, eta, y):
    """Mean, variance function and per-row log-likelihood (up to constants)."""
    if family == "gaussian":
        return eta, torch.ones_like(eta), -0.5 * (y - eta) ** 2
    if family == "binomial":
        mu = torch.sigmoid(eta)
        return mu, mu * (1 - mu), y * eta - F.softplus(eta)
    mu = torch.exp(eta)
    return mu, mu, y * eta - mu


def _glm_objective(X, y, w, family):
    """fun(beta) -> (log-lik, score, information) for a batch of GLMs."""
    def fun(beta):
        eta = torch.einsum("bnk,bk->bn", X, beta)
        mu, var, ll_row = _glm_terms(family, eta, y)
        ll = (w * ll_row).sum(1)
        score = torch.einsum("bnk,bn->bk", X, w * (y - mu))
        info = (X * (w * var).unsqueeze(-1)).transpose(1, 2) @ X
        return ll, score, info
    return fun


def _glm_dispersion(family, X, y, w, beta, n_obs):
    """Pearson dispersion for gaussian (as statsmodels/R); 1 otherwise."""
    if family != "gaussian":
        return torch.ones_like(n_obs)
    resid = y - torch.einsum("bnk,bk->bn", X, beta)
    return (w * resid ** 2).sum(1) / (n_obs - X.shape[2]).clamp(min=1)


def _fit_glm_batch(X, y, w, family, max_iter, tol):
    fun = _glm_objective(X, y, w, family)
    beta0 = X.new_zeros(X.shape[0], X.shape[2])
    beta, ll, info, conv, n_iter = newton_maximize(fun, beta0, max_iter, tol)
    cov, singular = spd_inverse(info)
    n_obs = w.sum(1)
    phi = _glm_dispersion(family, X, y, w, beta, n_obs)
    se = torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2).clamp(min=0) * phi[:, None])
    return BatchFit(beta, se, ll, conv, n_iter, singular, n_obs)


def glm_screen(features, y, covariates=None, family="binomial", device=None,
               dtype=torch.float64, chunk_size=None, max_iter=25, tol=1e-9):
    """Fit y ~ 1 + covariates + feature_j for every feature column j.

    Rows where feature j is missing (NaN) are dropped for that model only.
    Returns one row per feature with the feature's beta, se, z, p and diagnostics.
    """
    Fa, names = as_matrix(features, "feature")
    yv = as_vector(y, "y")
    check_rows(Fa, "features", len(yv))
    Ca = covariate_matrix(covariates, len(yv))
    _check_family(family, yv)
    dev = resolve_device(device)
    check_dtype(dev, dtype)
    yt = torch.as_tensor(yv, device=dev, dtype=dtype)

    def fit_chunk(X, w):
        return _fit_glm_batch(X, yt, w, family, max_iter, tol)

    rows = screen_features(Fa, Ca, fit_chunk, True, chunk_size, dev, dtype, copies=6)
    return feature_frame(names, rows)


def glm_fit(X, y, family="gaussian", intercept=True, device=None, dtype=torch.float64,
            max_iter=25, tol=1e-9):
    """Fit a single GLM and return its coefficient table (diagnostics in .attrs)."""
    Xa, names = as_matrix(X, "x")
    yv = as_vector(y, "y")
    require_complete(Xa, "X", len(yv))
    _check_family(family, yv)
    dev = resolve_device(device)
    check_dtype(dev, dtype)
    if intercept:
        Xa, names = np.column_stack([np.ones(len(yv)), Xa]), ["(Intercept)"] + names
    Xt = torch.as_tensor(Xa, device=dev, dtype=dtype).unsqueeze(0)
    yt = torch.as_tensor(yv, device=dev, dtype=dtype)
    w = torch.ones(1, len(yv), device=dev, dtype=dtype)
    return coefficient_frame(names, _fit_glm_batch(Xt, yt, w, family, max_iter, tol))
