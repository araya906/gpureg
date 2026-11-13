import numpy as np
import pytest
import statsmodels.api as sm
import torch

from gpureg import glm_fit, glm_screen

SM_FAMILIES = {"gaussian": sm.families.Gaussian(), "binomial": sm.families.Binomial(),
               "poisson": sm.families.Poisson()}


def _simulate_glm(family, n=600, m=4, seed=0):
    rng = np.random.default_rng(seed)
    C, F = rng.normal(size=(n, 2)), rng.normal(size=(n, m))
    eta = 0.3 + C @ np.array([0.5, -0.4]) + 0.6 * F[:, 0]
    if family == "gaussian":
        y = eta + rng.normal(size=n)
    elif family == "binomial":
        y = rng.binomial(1, 1 / (1 + np.exp(-eta))).astype(float)
    else:
        y = rng.poisson(np.exp(0.5 * eta)).astype(float)
    return C, F, y


@pytest.mark.parametrize("family", list(SM_FAMILIES))
def test_screen_matches_statsmodels(family):
    C, F, y = _simulate_glm(family)
    res = glm_screen(F, y, C, family=family, device="cpu")
    for j in range(F.shape[1]):
        X = sm.add_constant(np.column_stack([C, F[:, j]]))
        ref = sm.GLM(y, X, family=SM_FAMILIES[family]).fit()
        assert res["beta"][j] == pytest.approx(ref.params[-1], abs=1e-6)
        assert res["se"][j] == pytest.approx(ref.bse[-1], rel=1e-5)
        assert res["p"][j] == pytest.approx(ref.pvalues[-1], rel=1e-4, abs=1e-12)
        assert res["converged"][j]


def test_missing_feature_rows_are_dropped():
    C, F, y = _simulate_glm("binomial")
    F[::7, 1] = np.nan
    res = glm_screen(F, y, C, family="binomial", device="cpu", chunk_size=1)
    keep = ~np.isnan(F[:, 1])
    X = sm.add_constant(np.column_stack([C[keep], F[keep, 1]]))
    ref = sm.GLM(y[keep], X, family=SM_FAMILIES["binomial"]).fit()
    assert res["n_obs"][1] == keep.sum()
    assert res["beta"][1] == pytest.approx(ref.params[-1], abs=1e-6)
    assert res["se"][1] == pytest.approx(ref.bse[-1], rel=1e-5)


def test_single_fit_matches_statsmodels():
    C, F, y = _simulate_glm("binomial")
    X = np.column_stack([C, F])
    res = glm_fit(X, y, family="binomial", device="cpu")
    ref = sm.GLM(y, sm.add_constant(X), family=SM_FAMILIES["binomial"]).fit()
    np.testing.assert_allclose(res["beta"], ref.params, atol=1e-6)
    np.testing.assert_allclose(res["se"], ref.bse, rtol=1e-5)


def test_float32_fit_returns_float64_pvalues():
    rng = np.random.default_rng(3)
    x = rng.normal(size=5000)
    y = 0.5 * x + rng.normal(size=5000)  # z ~ 35, p ~ 1e-270: below float32 tiny
    res = glm_fit(x, y, device="cpu", dtype=torch.float32)
    assert res["p"].dtype == np.float64 and res["beta"].dtype == np.float32
    assert 0 < res["p"].iloc[1] < np.finfo(np.float32).tiny
