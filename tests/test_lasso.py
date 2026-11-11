import numpy as np
import pytest
import torch
from sklearn.linear_model import Lasso

from gpureg import lasso_cv


def _standardized(n, p, seed):
    X = np.random.default_rng(seed).normal(size=(n, p))
    return (X - X.mean(0)) / X.std(0)


def test_gaussian_path_matches_sklearn():
    X = _standardized(300, 20, 0)
    y = X[:, :3] @ np.array([1.0, -0.8, 0.5]) + np.random.default_rng(1).normal(size=300)
    res = lasso_cv(X, y, family="gaussian", n_folds=0, n_lambda=20, device="cpu",
                   tol=1e-10, max_iter=20000)
    for i in (2, 10, 19):
        ref = Lasso(alpha=res.lambdas[i], tol=1e-12, max_iter=100000).fit(X, y)
        np.testing.assert_allclose(res.coef[i], ref.coef_, atol=1e-5)
        assert res.intercept[i] == pytest.approx(ref.intercept_, abs=1e-5)


def test_binomial_cv_selects_signals():
    X = _standardized(800, 30, 2)
    eta = X[:, :3] @ np.array([1.2, -1.0, 0.8])
    y = np.random.default_rng(3).binomial(1, 1 / (1 + np.exp(-eta))).astype(float)
    res = lasso_cv(X, y, family="binomial", n_folds=5, n_lambda=30, device="cpu")
    assert {"x0", "x1", "x2"} <= set(res.selected("min"))


def test_cox_cv_selects_signals_and_keeps_unpenalized():
    X = _standardized(800, 30, 4)
    rng = np.random.default_rng(5)
    t_event = rng.exponential(1 / np.exp(X[:, :3] @ np.array([0.9, -0.7, 0.6])))
    t_cens = rng.exponential(2.0, size=800)
    time, event = np.minimum(t_event, t_cens), (t_event <= t_cens).astype(float)
    res = lasso_cv(X, family="cox", time=time, event=event, unpenalized=["x29"],
                   n_folds=5, n_lambda=30, device="cpu")
    assert {"x0", "x1", "x2"} <= set(res.selected("min"))
    assert "x29" not in res.selected("min")
    assert np.all(res.coef[:, 29] != 0)


def test_float32_path_matches_float64():
    X = _standardized(300, 20, 6)
    y = X[:, :3] @ np.array([1.0, -0.8, 0.5]) + np.random.default_rng(7).normal(size=300)
    fits = [lasso_cv(X, y, family="gaussian", n_folds=0, n_lambda=10, device="cpu", dtype=dt,
                     tol=1e-10, max_iter=20000) for dt in (torch.float32, torch.float64)]
    np.testing.assert_allclose(fits[0].coef.astype(np.float64), fits[1].coef, atol=1e-4)
