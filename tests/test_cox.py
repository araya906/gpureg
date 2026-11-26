import numpy as np
import pytest
import statsmodels.api as sm
import torch

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


def _entry(time, seed=2):
    """Non-integer delayed entry on 60% of rows (never equal to an integer event time)."""
    rng = np.random.default_rng(seed)
    return np.where(rng.random(len(time)) < 0.6, rng.uniform(0.0, 0.9, len(time)) * time, 0.0)


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_entry_screen_and_fit_match_statsmodels(ties):
    C, F, time, event = _simulate_cox()
    entry, s = _entry(time), _strata(len(time))
    F[::7, 1] = np.nan
    res = cox_screen(F, time, event, C, ties=ties, strata=s, entry=entry, chunk_size=2)
    for j in range(F.shape[1]):
        keep = ~np.isnan(F[:, j])
        X = np.column_stack([C[keep], F[keep, j]])
        ref = sm.PHReg(time[keep], X, status=event[keep], ties=ties, strata=s[keep],
                       entry=entry[keep]).fit()
        assert res["beta"][j] == pytest.approx(ref.params[-1], abs=1e-6)
        assert res["se"][j] == pytest.approx(ref.bse[-1], rel=1e-5)
    X = np.column_stack([C, F[:, [0, 2]]])
    tab = cox_fit(X, time, event, ties=ties, entry=entry)
    ref = sm.PHReg(time, X, status=event, ties=ties, entry=entry).fit()
    np.testing.assert_allclose(tab["beta"], ref.params, atol=1e-6)
    np.testing.assert_allclose(tab["se"], ref.bse, rtol=1e-5)


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_episode_splitting_leaves_the_fit_unchanged(ties):
    """Counting-process rows (0, c], (c, time] for one subject: at risk on (start, stop], so the
    split at a tied event time c must not count the subject twice (R survival convention)."""
    C, F, time, event = _simulate_cox()
    X = np.column_stack([C, F[:, 0]])
    whole = cox_fit(X, time, event, ties=ties)
    cut = np.floor(time / 2)
    split = cut > 0
    idx = np.r_[np.arange(len(time)), np.flatnonzero(split)]
    entry = np.r_[np.where(split, cut, 0.0), np.zeros(split.sum())]
    stop = np.r_[time, cut[split]]
    ev = np.r_[event, np.zeros(split.sum())]
    parts = cox_fit(X[idx], stop, ev, ties=ties, entry=entry)
    np.testing.assert_allclose(parts["beta"], whole["beta"], rtol=1e-9)
    np.testing.assert_allclose(parts["se"], whole["se"], rtol=1e-9)


def test_zero_entry_equals_no_entry_and_bad_entry_is_rejected():
    C, F, time, event = _simulate_cox()
    plain = cox_screen(F, time, event, C)
    zero = cox_screen(F, time, event, C, entry=np.zeros(len(time)))
    np.testing.assert_allclose(zero["beta"], plain["beta"], rtol=1e-12)
    with pytest.raises(ValueError, match="rows"):
        cox_screen(F, time, event, C, entry=np.zeros(len(time) - 1))
    with pytest.raises(ValueError, match="smaller than time"):
        cox_fit(F, time, event, entry=time)
    with pytest.raises(ValueError, match="missing"):
        cox_fit(F, time, event, entry=np.r_[np.nan, np.zeros(len(time) - 1)])


def _efron_score_residuals_loop(X, time, event, beta):
    """Reference loop over death times (R's coxscore, Efron): tied deaths share each l/d step,
    and their own risk weight at step l is (1 - l/d)."""
    r, U = np.exp(X @ beta), np.zeros_like(X)
    for t in np.unique(time[event == 1]):
        dead, risk = (time == t) & (event == 1), time >= t
        d, s0, s1 = dead.sum(), r[risk].sum(), r[risk] @ X[risk]
        e0, e1 = r[dead].sum(), r[dead] @ X[dead]
        for ell in range(d):
            f = ell / d
            den, xbar = s0 - f * e0, (s1 - f * e1) / (s0 - f * e0)
            U[dead] += (X[dead] - xbar) / d
            wt = np.where(dead, 1 - f, 1.0)[risk] * r[risk] / den
            U[risk] -= wt[:, None] * (X[risk] - xbar)
    return U


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_score_residuals_sum_to_score_and_match_loop(ties):
    from gpureg._risksets import build_risk_sets
    from gpureg.cox import _cox_score_residuals
    C, F, time, event = _simulate_cox(n=300)
    X = np.column_stack([C, F[:, 0]])
    beta = np.array([0.3, -0.2, 0.4])
    order = np.argsort(-time, kind="stable")
    rs = build_risk_sets(time[order], event[order], "cpu")
    U = _cox_score_residuals(torch.tensor(beta)[None], torch.tensor(X[order])[None],
                             torch.tensor(event[order]), torch.ones(1, len(time),
                             dtype=torch.float64), rs, ties == "efron")[0].numpy()
    ref = sm.PHReg(time, X, status=event, ties=ties)
    np.testing.assert_allclose(U.sum(0), ref.score(beta), rtol=1e-10, atol=1e-10)
    if ties == "efron":
        np.testing.assert_allclose(U, _efron_score_residuals_loop(X, time, event, beta)[order],
                                   rtol=1e-10, atol=1e-12)


def test_robust_efron_fit_matches_loop_sandwich_with_ties():
    C, F, time, event = _simulate_cox(seed=3)
    X = np.column_stack([C, F[:, :2]])
    tab = cox_fit(X, time, event, ties="efron", robust=True)
    fit = sm.PHReg(time, X, status=event, ties="efron").fit()
    U = _efron_score_residuals_loop(X, time, event, fit.params)
    cov = fit.cov_params()
    np.testing.assert_allclose(tab["beta"], fit.params, atol=1e-8)
    np.testing.assert_allclose(tab["se"], np.sqrt(np.diag(cov @ U.T @ U @ cov)), rtol=1e-6)


def test_robust_keeps_estimates_flags_and_float32():
    C, F, time, event = _simulate_cox()
    F[:, 2] = 1.0  # constant: singular, robust se stays NaN
    plain = cox_screen(F, time, event, C)
    rob = cox_screen(F, time, event, C, robust=True)
    rob32 = cox_screen(F, time, event, C, robust=True, dtype=torch.float32)
    np.testing.assert_array_equal(rob["beta"], plain["beta"])
    assert rob["singular"][2] and np.isnan(rob["se"][2])
    assert not np.allclose(rob["se"][:2], plain["se"][:2], rtol=1e-3)
    np.testing.assert_allclose(rob32["se"][:2], rob["se"][:2], rtol=1e-4)


def test_chunk_copies_follow_padded_efron_tensors():
    from gpureg._device import auto_chunk_size
    from gpureg.cox import _cox_setup, _tie_weights, chunk_copies
    rng = np.random.default_rng(3)
    n = 20000
    time, event = rng.integers(1, 200, size=n).astype(float), (rng.random(n) < 0.5) * 1.0
    cpu = torch.device("cpu")
    _, ev, rs = _cox_setup(time, event, None, None, cpu, torch.float64)
    X = torch.zeros(1, n, 4, dtype=torch.float64)
    valid = _tie_weights(ev, torch.ones(1, n, dtype=torch.float64), rs, True, X)[1]
    assert rs.max_ties > 30 and valid.shape[1:] == (rs.event_groups.numel(), rs.max_ties)
    assert chunk_copies(rs, n, False) == pytest.approx(8 + 3 * valid[0].numel() / n)
    assert chunk_copies(rs, n, True) == pytest.approx(12 + 5 * valid[0].numel() / n)
    assert auto_chunk_size(n, 4, torch.float64, cpu, chunk_copies(rs, n, False)) >= 16
