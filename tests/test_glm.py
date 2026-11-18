import warnings

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


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_constant_and_collinear_features_are_singular(dtype):
    # A constant feature used to crash the whole screen (cholesky_inverse raised on singular H).
    C, F, y = _simulate_glm("binomial")
    F = np.column_stack([F[:, 0], np.ones(len(y)), 2.0 * C[:, 0] + 1.0])
    res = glm_screen(F, y, C, family="binomial", dtype=dtype)
    assert res["singular"].tolist() == [False, True, True]
    assert np.isfinite(res["p"][0]) and res[["se", "p"]].iloc[1:].isna().all().all()


def test_complete_separation_is_not_reported_as_converged():
    C, F, y = _simulate_glm("binomial")
    with pytest.warns(UserWarning, match="infinite"):
        res = glm_screen(y[:, None], y, C, family="binomial")
    assert not res["converged"][0] and res["p"][0] > 0.5 and res["infinite"][0]


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_quasi_separation_is_flagged_infinite(dtype):
    # Quasi-separation converges with a huge se, as in R; only the remaining Newton step shows it.
    C, F, y = _simulate_glm("binomial")
    quasi = ((y == 1) & (np.arange(len(y)) % 2 == 0)).astype(float)
    with pytest.warns(UserWarning, match="1 screened models"):
        res = glm_screen(np.column_stack([F[:, 0], quasi]), y, C, family="binomial", dtype=dtype)
    assert res["infinite"].tolist() == [False, True] and res["converged"].all()


def test_glm_fit_flags_infinite_term_only():
    C, F, y = _simulate_glm("poisson")
    zero_group = ((y == 0) & (np.arange(len(y)) % 2 == 0)).astype(float)
    with pytest.warns(UserWarning, match="1 coefficients may be infinite"):
        res = glm_fit(np.column_stack([C, zero_group]), y, family="poisson", device="cpu")
    assert res["infinite"].tolist() == [False, False, False, True] and res.attrs["converged"]


@pytest.mark.parametrize("family", ["gaussian", "binomial", "poisson"])
def test_regular_screen_has_no_infinite_flags(family):
    C, F, y = _simulate_glm(family)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = glm_screen(F, y, C, family=family)
    assert not res["infinite"].any()


def _simulate_rates(n=800, m=3, seed=5):
    rng = np.random.default_rng(seed)
    C, F = rng.normal(size=(n, 2)), rng.normal(size=(n, m))
    offset = np.log(rng.uniform(0.5, 20.0, size=n))  # log person-time
    y = rng.poisson(np.exp(-2.0 + 0.3 * C[:, 0] + 0.4 * F[:, 0] + offset)).astype(float)
    return C, F, y, offset


def test_poisson_offset_matches_statsmodels():
    C, F, y, offset = _simulate_rates()
    F[::9, 2] = np.nan  # offset must follow the rows kept for each feature
    res = glm_screen(F, y, C, family="poisson", device="cpu", offset=offset)
    for j in range(F.shape[1]):
        keep = ~np.isnan(F[:, j])
        X = sm.add_constant(np.column_stack([C[keep], F[keep, j]]))
        ref = sm.GLM(y[keep], X, family=SM_FAMILIES["poisson"], offset=offset[keep]).fit()
        assert res["beta"][j] == pytest.approx(ref.params[-1], abs=1e-6)
        assert res["se"][j] == pytest.approx(ref.bse[-1], rel=1e-5)
        assert res["converged"][j]


@pytest.mark.parametrize("family", list(SM_FAMILIES))
def test_glm_fit_offset_matches_statsmodels(family):
    C, F, y, offset = _simulate_rates()
    if family != "poisson":
        C, F, y = _simulate_glm(family, n=800)
        offset = 0.2 * C[:, 1]
    X = np.column_stack([C, F])
    res = glm_fit(X, y, family=family, device="cpu", offset=offset)
    ref = sm.GLM(y, sm.add_constant(X), family=SM_FAMILIES[family], offset=offset).fit()
    np.testing.assert_allclose(res["beta"], ref.params, atol=1e-6)
    np.testing.assert_allclose(res["se"], ref.bse, rtol=1e-5)


def test_offset_is_validated():
    C, F, y, offset = _simulate_rates(n=50)
    with pytest.raises(ValueError, match="rows"):
        glm_screen(F, y, C, family="poisson", offset=offset[:-1])
    with pytest.raises(ValueError, match="finite"):
        glm_fit(F, y, family="poisson", offset=np.full(50, -np.inf))
