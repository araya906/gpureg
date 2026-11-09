import numpy as np
import pytest
import statsmodels.api as sm

from gpureg import cox_fit, cox_screen


def _simulate_cox(n=500, m=3, seed=0):
    """Coarse integer times so that tied event times are common."""
    rng = np.random.default_rng(seed)
    C, F = rng.normal(size=(n, 2)), rng.normal(size=(n, m))
    hazard = np.exp(C @ np.array([0.4, -0.3]) + 0.5 * F[:, 0])
    t_event = np.ceil(rng.exponential(1 / hazard) * 20)
    t_cens = np.ceil(rng.exponential(2.0, size=n) * 20)
    return C, F, np.minimum(t_event, t_cens), (t_event <= t_cens).astype(float)


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_screen_matches_statsmodels(ties):
    C, F, time, event = _simulate_cox()
    res = cox_screen(F, time, event, C, ties=ties, device="cpu")
    for j in range(F.shape[1]):
        X = np.column_stack([C, F[:, j]])
        ref = sm.PHReg(time, X, status=event, ties=ties).fit()
        assert res["beta"][j] == pytest.approx(ref.params[-1], abs=1e-6)
        assert res["se"][j] == pytest.approx(ref.bse[-1], rel=1e-5)
        assert res["converged"][j]


def test_missing_feature_rows_are_dropped():
    C, F, time, event = _simulate_cox()
    F[::5, 2] = np.nan
    res = cox_screen(F, time, event, C, device="cpu", chunk_size=2)
    keep = ~np.isnan(F[:, 2])
    X = np.column_stack([C[keep], F[keep, 2]])
    ref = sm.PHReg(time[keep], X, status=event[keep], ties="efron").fit()
    assert res["n_obs"][2] == keep.sum()
    assert res["beta"][2] == pytest.approx(ref.params[-1], abs=1e-6)


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_single_fit_matches_statsmodels(ties):
    C, F, time, event = _simulate_cox()
    X = np.column_stack([C, F])
    res = cox_fit(X, time, event, ties=ties, device="cpu")
    ref = sm.PHReg(time, X, status=event, ties=ties).fit()
    np.testing.assert_allclose(res["beta"], ref.params, atol=1e-6)
    np.testing.assert_allclose(res["se"], ref.bse, rtol=1e-5)
