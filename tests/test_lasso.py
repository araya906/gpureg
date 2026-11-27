import warnings

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


def test_float32_tol_floored_at_rounding_error():
    X = _standardized(300, 20, 6)
    y = X[:, :3] @ np.array([1.0, -0.8, 0.5]) + np.random.default_rng(7).normal(size=300)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lasso_cv(X, y, family="gaussian", n_folds=0, n_lambda=10, device="cpu",
                 dtype=torch.float32, tol=1e-10, max_iter=2000)


@pytest.mark.parametrize("family", ["gaussian", "cox"])
def test_strong_rules_give_same_path(family):
    X = _standardized(400, 60, 8)
    rng = np.random.default_rng(9)
    eta = X[:, :3] @ np.array([0.9, -0.7, 0.6])
    if family == "gaussian":
        kw = {"y": eta + rng.normal(size=400)}
    else:
        t_event, t_cens = rng.exponential(1 / np.exp(eta)), rng.exponential(2.0, size=400)
        kw = {"time": np.minimum(t_event, t_cens), "event": (t_event <= t_cens).astype(float)}
    fits = [lasso_cv(X, family=family, n_folds=3, n_lambda=20, device="cpu", unpenalized=["x59"],
                     strong_rules=sr, **kw) for sr in (True, False)]
    np.testing.assert_allclose(fits[0].coef, fits[1].coef, atol=1e-10)
    np.testing.assert_allclose(fits[0].cv_mean, fits[1].cv_mean, atol=1e-10)
    assert fits[0].n_fitted[0] < 60 and fits[0].n_fitted.max() <= 60


def test_strong_rules_kkt_check_recovers_dropped_features(monkeypatch):
    # A strong set holding only the unpenalized column forces the KKT check to add the signals.
    from gpureg import lasso as lasso_mod
    monkeypatch.setattr(lasso_mod, "_strong_set", lambda prob, *a: prob.pf == 0)
    X = _standardized(300, 20, 10)
    y = X[:, :3] @ np.array([1.0, -0.8, 0.5]) + np.random.default_rng(11).normal(size=300)
    fits = [lasso_cv(X, y, family="gaussian", n_folds=0, n_lambda=10, device="cpu",
                     strong_rules=sr) for sr in (True, False)]
    np.testing.assert_allclose(fits[0].coef, fits[1].coef, atol=1e-6)  # refits restart FISTA; tol 1e-7
    assert fits[0].n_fitted[-1] > 3


def _stratified_cox(n, p, seed):
    X = _standardized(n, p, seed)
    rng = np.random.default_rng(seed + 1)
    strata = rng.integers(0, 3, size=n)
    eta = X[:, :2] @ np.array([0.8, -0.6]) + 0.7 * strata
    t_event = np.ceil(rng.exponential(1 / np.exp(eta)) * 15)
    t_cens = np.ceil(rng.exponential(2.0, size=n) * 15)
    return X, np.minimum(t_event, t_cens), (t_event <= t_cens).astype(float), strata


def test_cox_strata_matches_statsmodels_regularized():
    import statsmodels.api as sm
    X, time, event, strata = _stratified_cox(400, 8, 20)
    res = lasso_cv(X, family="cox", time=time, event=event, strata=strata, n_folds=0,
                   n_lambda=8, device="cpu", tol=1e-10, max_iter=50000)
    for i in (2, 7):
        ref = sm.PHReg(time, X, status=event, ties="breslow", strata=strata).fit_regularized(
            alpha=res.lambdas[i], L1_wt=1.0)
        np.testing.assert_allclose(res.coef[i], ref.params, atol=1e-6)


def test_cox_strata_gradient_matches_autograd():
    from gpureg import lasso as lasso_mod
    X, time, event, strata = _stratified_cox(200, 4, 22)
    prob = lasso_mod._lasso_problem(X, None, "cox", time, event, strata, None, 3, 0, "cpu",
                                    torch.float64)
    beta = torch.randn(prob.W.shape[0], X.shape[1], dtype=torch.float64, requires_grad=True)
    eta = beta @ prob.X.T
    ll, r = lasso_mod._lasso_loglik("cox", eta, prob.y, prob.W, prob.rs)
    (grad,) = torch.autograd.grad(ll.sum(), beta)
    torch.testing.assert_close(grad, r.detach() @ prob.X)


def test_cox_single_stratum_equals_unstratified():
    X, time, event, _ = _stratified_cox(300, 6, 24)
    kw = dict(family="cox", time=time, event=event, n_folds=3, n_lambda=10, device="cpu")
    a = lasso_cv(X, strata=np.full(300, "s"), **kw)
    b = lasso_cv(X, **kw)
    np.testing.assert_allclose(a.coef, b.coef, atol=1e-12)
    np.testing.assert_allclose(a.cv_mean, b.cv_mean, atol=1e-12)


def test_cox_strata_cv_runs_and_rejects_other_families():
    X, time, event, strata = _stratified_cox(300, 6, 26)
    res = lasso_cv(X, family="cox", time=time, event=event, strata=strata, n_folds=3,
                   n_lambda=10, device="cpu")
    assert np.isfinite(res.cv_mean).all() and "x0" in res.selected("min")
    with pytest.raises(ValueError, match="strata"):
        lasso_cv(X, event, family="binomial", strata=strata, device="cpu")
