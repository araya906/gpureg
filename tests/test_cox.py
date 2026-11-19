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


def test_constant_feature_is_singular():
    # Without an intercept a constant column has only rounding-noise information.
    C, F, time, event = _simulate_cox()
    half = np.where(np.arange(len(time)) % 2 == 0, np.nan, 3.0)
    res = cox_screen(np.column_stack([F[:, 0], np.ones(len(time)), half]), time, event, C)
    assert res["singular"].tolist() == [False, True, True]
    assert res[["se", "p"]].iloc[1:].isna().all().all()


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_monotone_likelihood_is_flagged_infinite(ties):
    # A group with no events: the partial likelihood keeps rising as its beta -> -inf.
    C, F, time, event = _simulate_cox()
    no_events = ((event == 0) & (np.arange(len(time)) % 2 == 0)).astype(float)
    with pytest.warns(UserWarning, match="1 screened models"):
        res = cox_screen(np.column_stack([F[:, 0], no_events]), time, event, C, ties=ties)
    assert res["infinite"].tolist() == [False, True] and res["p"][1] > 0.5


def test_cox_fit_regular_data_has_no_infinite_terms():
    C, F, time, event = _simulate_cox()
    res = cox_fit(np.column_stack([C, F]), time, event, device="cpu")
    assert not res["infinite"].any()


def _strata(n, seed=1):
    return np.random.default_rng(seed).integers(0, 4, n)


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_stratified_screen_and_fit_match_statsmodels(ties):
    C, F, time, event = _simulate_cox()
    s = _strata(len(time))
    F[::7, 1] = np.nan
    res = cox_screen(F, time, event, C, ties=ties, strata=s, device="cpu", chunk_size=2)
    for j in range(F.shape[1]):
        keep = ~np.isnan(F[:, j])
        X = np.column_stack([C[keep], F[keep, j]])
        ref = sm.PHReg(time[keep], X, status=event[keep], ties=ties, strata=s[keep]).fit()
        assert res["beta"][j] == pytest.approx(ref.params[-1], abs=1e-6)
        assert res["se"][j] == pytest.approx(ref.bse[-1], rel=1e-5)
    X = np.column_stack([C, F[:, [0, 2]]])
    tab = cox_fit(X, time, event, ties=ties, strata=s, device="cpu")
    ref = sm.PHReg(time, X, status=event, ties=ties, strata=s).fit()
    np.testing.assert_allclose(tab["beta"], ref.params, atol=1e-6)
    np.testing.assert_allclose(tab["se"], ref.bse, rtol=1e-5)


def test_single_stratum_equals_unstratified_and_labels_can_be_strings():
    C, F, time, event = _simulate_cox()
    plain = cox_screen(F, time, event, C, device="cpu")
    one = cox_screen(F, time, event, C, strata=np.repeat("a", len(time)), device="cpu")
    np.testing.assert_allclose(one["beta"], plain["beta"], rtol=1e-12)
    labels = np.array(["x", "y", "z", "w"])[_strata(len(time))]
    codes = cox_screen(F, time, event, C, strata=_strata(len(time)), device="cpu")
    named = cox_screen(F, time, event, C, strata=labels, device="cpu")
    np.testing.assert_allclose(named["beta"], codes["beta"], rtol=1e-12)


def test_feature_constant_within_strata_is_singular():
    C, F, time, event = _simulate_cox()
    s = _strata(len(time))
    res = cox_screen(np.column_stack([F[:, 0], s.astype(float)]), time, event, C, strata=s)
    assert res["singular"].tolist() == [False, True] and np.isnan(res["se"][1])


def test_bad_strata_are_rejected():
    C, F, time, event = _simulate_cox()
    s = _strata(len(time)).astype(float)
    with pytest.raises(ValueError, match="rows"):
        cox_screen(F, time, event, C, strata=s[:-1])
    s[3] = np.nan
    with pytest.raises(ValueError, match="missing"):
        cox_fit(F, time, event, strata=s)
