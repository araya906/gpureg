"""Synthetic-data validation of gpureg against reference implementations.

Usage:  python validation/run_checks.py [--quick] [--out validation/results]
Writes latest.json / latest.md and appends to history.jsonl. Exit code 0 only if every check passes.
The float32-vs-float64 accuracy report is informational and never affects the exit code.
"""
import argparse
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
import statsmodels.api as sm
import torch
from scipy import stats
from sklearn.linear_model import Lasso, LogisticRegression

from gpureg import cox_screen, glm_screen, lasso_cv

try:  # optional reference for the Cox lasso (pip install scikit-survival)
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.util import Surv
except ImportError:
    CoxnetSurvivalAnalysis = None

# scikit-learn >= 1.8 deprecates `penalty` (removed in 1.10); l1_ratio=1 selects pure L1.
_SK_L1 = ({"l1_ratio": 1.0} if tuple(int(v) for v in sklearn.__version__.split(".")[:2]) >= (1, 8)
          else {"penalty": "l1"})
TOL = {"beta_abs": 1e-5, "se_rel": 1e-4, "log10p_abs": 1e-3, "lasso_gauss_abs": 1e-5,
       "lasso_binom_abs": 1e-3, "lasso_cox_abs": 1e-4, "lasso_f32_abs": 1e-4, "screen_f32_beta_abs": 1e-4,
       "screen_f32_log10p_rel": 1e-4, "lasso_strong_abs": 1e-8, "lasso_kkt_abs": 1e-6}
SM_FAMILIES = {"gaussian": sm.families.Gaussian(), "binomial": sm.families.Binomial(),
               "poisson": sm.families.Poisson()}


def simulate_screen(kind, n, m, seed, missing=0.05):
    """Covariates C, features F (with NaNs), outcome: y, or (time, event) with ties for cox."""
    rng = np.random.default_rng(seed)
    C, F = rng.normal(size=(n, 3)), rng.normal(size=(n, m))
    eta = 0.2 + C @ np.array([0.4, -0.3, 0.2]) + 0.5 * F[:, 0] - 0.3 * F[:, 1]
    if kind == "gaussian":
        out = eta + rng.normal(size=n)
    elif kind == "binomial":
        out = rng.binomial(1, 1 / (1 + np.exp(-eta))).astype(float)
    elif kind == "poisson":
        out = rng.poisson(np.exp(0.5 * eta)).astype(float)
    else:
        t_event = np.ceil(rng.exponential(1 / np.exp(eta - 0.2)) * 30)
        t_cens = np.ceil(rng.exponential(2.0, size=n) * 30)
        out = (np.minimum(t_event, t_cens), (t_event <= t_cens).astype(float))
    F[rng.random(F.shape) < missing] = np.nan
    return C, F, out


def reference_screen(kind, C, F, out, ties):
    """statsmodels fit per feature on that feature's complete rows: (beta, se, p) rows."""
    rows = []
    for j in range(F.shape[1]):
        keep = ~np.isnan(F[:, j])
        X = np.column_stack([C[keep], F[keep, j]])
        if kind == "cox":
            fit = sm.PHReg(out[0][keep], X, status=out[1][keep], ties=ties).fit()
        else:
            fit = sm.GLM(out[keep], sm.add_constant(X), family=SM_FAMILIES[kind]).fit()
        rows.append((fit.params[-1], fit.bse[-1], fit.pvalues[-1]))
    return np.array(rows)


def _neg_log10(p):
    return -np.log10(np.clip(p, 1e-300, 1.0))


def compare_screen(res, ref):
    beta_err = float(np.abs(res["beta"].to_numpy() - ref[:, 0]).max())
    se_err = float((np.abs(res["se"].to_numpy() - ref[:, 1]) / ref[:, 1]).max())
    p_err = float(np.abs(_neg_log10(res["p"].to_numpy()) - _neg_log10(ref[:, 2])).max())
    ok = (beta_err <= TOL["beta_abs"] and se_err <= TOL["se_rel"]
          and p_err <= TOL["log10p_abs"] and bool(res["converged"].all())
          and not res["infinite"].any())
    return {"beta_max_abs_err": beta_err, "se_max_rel_err": se_err,
            "log10p_max_abs_err": p_err, "all_converged": bool(res["converged"].all()),
            "n_infinite": int(res["infinite"].sum()), "pass": ok}


def _fit_screen(kind, C, F, out, ties, device, dtype=torch.float64):
    if kind == "cox":
        return cox_screen(F, out[0], out[1], C, ties=ties, device=device, dtype=dtype)
    return glm_screen(F, out, C, family=kind, device=device, dtype=dtype)


def check_screen_poisson_offset(device, n, m, seed=4):
    """Poisson rate screen with a log person-time offset vs statsmodels GLM(offset=...)."""
    C, F, _ = simulate_screen("poisson", n, m, seed)
    rng = np.random.default_rng(seed + 100)
    offset = np.log(rng.uniform(0.5, 20.0, size=n))
    eta = -2.0 + C @ np.array([0.3, -0.2, 0.1]) + 0.4 * np.nan_to_num(F[:, 0]) + offset
    y = rng.poisson(np.exp(eta)).astype(float)
    start = time.perf_counter()
    res = glm_screen(F, y, C, family="poisson", device=device, offset=offset)
    t_ours = time.perf_counter() - start
    rows = []
    for j in range(m):
        keep = ~np.isnan(F[:, j])
        X = sm.add_constant(np.column_stack([C[keep], F[keep, j]]))
        fit = sm.GLM(y[keep], X, family=SM_FAMILIES["poisson"], offset=offset[keep]).fit()
        rows.append((fit.params[-1], fit.bse[-1], fit.pvalues[-1]))
    return {"check": "screen_poisson_offset", "n": n, "m": m, **compare_screen(res, np.array(rows)),
            "time_gpureg_s": t_ours}


def check_screen_glm_robust(kind, device, n, m, seed=7):
    """HC0 sandwich SEs under misspecification vs statsmodels GLM(...).fit(cov_type="HC0").

    gaussian: heteroskedastic noise; binomial: omitted random effect; poisson: negative binomial.
    """
    C, F, _ = simulate_screen(kind, n, m, seed)
    rng = np.random.default_rng(seed + 100)
    eta = 0.2 + C @ np.array([0.4, -0.3, 0.2]) + 0.5 * np.nan_to_num(F[:, 0])
    if kind == "gaussian":
        y = eta + rng.normal(size=n) * (0.5 + np.abs(C[:, 0]))
    elif kind == "binomial":
        y = rng.binomial(1, 1 / (1 + np.exp(-eta - rng.normal(size=n)))).astype(float)
    else:
        y = rng.negative_binomial(2, 2 / (2 + np.exp(0.5 * eta))).astype(float)
    start = time.perf_counter()
    res = glm_screen(F, y, C, family=kind, device=device, robust=True)
    t_ours = time.perf_counter() - start
    rows = []
    for j in range(m):
        keep = ~np.isnan(F[:, j])
        X = sm.add_constant(np.column_stack([C[keep], F[keep, j]]))
        fit = sm.GLM(y[keep], X, family=SM_FAMILIES[kind]).fit(cov_type="HC0")
        rows.append((fit.params[-1], fit.bse[-1], fit.pvalues[-1]))
    return {"check": f"screen_{kind}_robust", "n": n, "m": m,
            **compare_screen(res, np.array(rows)), "time_gpureg_s": t_ours}


def check_screen_cox_strata(device, n, m, ties, seed=5):
    """Stratified Cox screen (8 strata, different baseline hazards) vs statsmodels PHReg(strata=)."""
    C, F, _ = simulate_screen("cox", n, m, seed)
    rng = np.random.default_rng(seed + 100)
    strata = rng.integers(0, 8, size=n)
    eta = C @ np.array([0.4, -0.3, 0.2]) + 0.5 * np.nan_to_num(F[:, 0]) + 0.4 * strata - 1.5
    t_event = np.ceil(rng.exponential(1 / np.exp(eta)) * 30)
    t_cens = np.ceil(rng.exponential(2.0, size=n) * 30)
    time_, event = np.minimum(t_event, t_cens), (t_event <= t_cens).astype(float)
    start = time.perf_counter()
    res = cox_screen(F, time_, event, C, ties=ties, device=device, strata=strata)
    t_ours = time.perf_counter() - start
    rows = []
    for j in range(m):
        keep = ~np.isnan(F[:, j])
        X = np.column_stack([C[keep], F[keep, j]])
        fit = sm.PHReg(time_[keep], X, status=event[keep], ties=ties, strata=strata[keep]).fit()
        rows.append((fit.params[-1], fit.bse[-1], fit.pvalues[-1]))
    return {"check": f"screen_cox_{ties}_strata", "n": n, "m": m,
            **compare_screen(res, np.array(rows)), "time_gpureg_s": t_ours}


def check_screen_cox_entry(device, n, m, ties, seed=6):
    """Left-truncated Cox screen (entry < time, 4 strata) vs statsmodels PHReg(entry=, strata=).

    Entry times are non-integer so none equals an event time: statsmodels counts a row as at risk
    at its entry time, gpureg (like R's survival) does not; elsewhere both use entry < t <= time.
    """
    C, F, (time_, event) = simulate_screen("cox", n, m, seed)
    rng = np.random.default_rng(seed + 100)
    entry = np.where(rng.random(n) < 0.6, rng.uniform(0.0, 0.9, n) * time_, 0.0)
    strata = rng.integers(0, 4, size=n)
    start = time.perf_counter()
    res = cox_screen(F, time_, event, C, ties=ties, device=device, strata=strata, entry=entry)
    t_ours = time.perf_counter() - start
    rows = []
    for j in range(m):
        keep = ~np.isnan(F[:, j])
        X = np.column_stack([C[keep], F[keep, j]])
        fit = sm.PHReg(time_[keep], X, status=event[keep], ties=ties, strata=strata[keep],
                       entry=entry[keep]).fit()
        rows.append((fit.params[-1], fit.bse[-1], fit.pvalues[-1]))
    return {"check": f"screen_cox_{ties}_entry", "n": n, "m": m,
            **compare_screen(res, np.array(rows)), "time_gpureg_s": t_ours}


def _no_event_at_risk(time_, event, entry, strata):
    """Rows whose (entry, time] window holds no event time of their stratum."""
    out = np.ones(len(time_), dtype=bool)
    for s in np.unique(strata):
        rows, t_ev = strata == s, np.unique(time_[(strata == s) & (event > 0)])
        upto = np.searchsorted(t_ev, time_[rows], "right")
        out[rows] = upto == np.searchsorted(t_ev, entry[rows], "right")
    return out


def _phreg_robust_row(time_, event, entry, strata, X, ties):
    """(beta, se, p) of the last term with the Lin-Wei sandwich from statsmodels' score residuals.

    Rows never at risk at an event time have zero residuals (as in R); statsmodels 0.15 returns NaN
    or, when (entry, time] holds no event time, a spurious non-zero value for them.
    """
    model = sm.PHReg(time_, X, status=event, ties=ties, strata=strata, entry=entry)
    fit = model.fit()
    resid = model.score_residuals(fit.params)
    resid[_no_event_at_risk(time_, event, entry, strata)] = 0.0
    cov = np.linalg.inv(-model.hessian(fit.params))
    se = np.sqrt((cov @ resid.T @ resid @ cov)[-1, -1])
    return fit.params[-1], se, 2 * stats.norm.sf(abs(fit.params[-1] / se))


def check_screen_cox_robust(device, n, m, ties, seed=9):
    """Lin-Wei robust SEs (omitted frailty, 4 strata, delayed entry) vs statsmodels PHReg score
    residuals. These are Breslow-form, so Efron is checked on untied times, where both forms agree
    (smaller n, m: statsmodels loops over every event time); tied Efron is tested in tests/.
    """
    n, m = (n, m) if ties == "breslow" else (min(n, 2000), min(m, 30))
    C, F, _ = simulate_screen("cox", n, m, seed)
    rng = np.random.default_rng(seed + 100)
    strata = rng.integers(0, 4, size=n)
    eta = C @ np.array([0.4, -0.3, 0.2]) + 0.5 * np.nan_to_num(F[:, 0]) + rng.normal(size=n)
    time_ = np.minimum(rng.exponential(1 / np.exp(eta)), rng.exponential(2.0, size=n)) * 30
    event = (rng.random(n) < 0.7).astype(float)
    time_ = np.ceil(time_) if ties == "breslow" else time_
    entry = np.where(rng.random(n) < 0.5, rng.uniform(0.0, 0.9, n) * time_, 0.0)
    start = time.perf_counter()
    res = cox_screen(F, time_, event, C, ties=ties, device=device, strata=strata, entry=entry,
                     robust=True)
    t_ours = time.perf_counter() - start
    rows = []
    for j in range(m):
        keep = ~np.isnan(F[:, j])
        X = np.column_stack([C[keep], F[keep, j]])
        rows.append(_phreg_robust_row(time_[keep], event[keep], entry[keep], strata[keep], X, ties))
    return {"check": f"screen_cox_{ties}_robust", "n": n, "m": m,
            **compare_screen(res, np.array(rows)), "time_gpureg_s": t_ours}


def _screen_name(kind, ties):
    return f"screen_{kind}" + (f"_{ties}" if kind == "cox" else "")


def check_screen(kind, n, m, seed, device, ties="efron"):
    C, F, out = simulate_screen(kind, n, m, seed)
    start = time.perf_counter()
    res = _fit_screen(kind, C, F, out, ties, device)
    t_ours = time.perf_counter() - start
    start = time.perf_counter()
    ref = reference_screen(kind, C, F, out, ties)
    t_ref = time.perf_counter() - start
    return {"check": _screen_name(kind, ties), "n": n, "m": m, **compare_screen(res, ref),
            "time_gpureg_s": t_ours, "time_reference_s": t_ref}


def _standardized(n, p, seed):
    X = np.random.default_rng(seed).normal(size=(n, p))
    return (X - X.mean(0)) / X.std(0)


def _lasso_gaussian_path(device, n, p, dtype=torch.float64):
    X = _standardized(n, p, 10)
    y = X[:, :4] @ np.array([1.0, -0.8, 0.5, 0.3]) + np.random.default_rng(11).normal(size=n)
    res = lasso_cv(X, y, family="gaussian", n_folds=0, n_lambda=20, device=device,
                   dtype=dtype, tol=1e-10, max_iter=20000)
    return X, y, res


def check_lasso_gaussian(device, n=1000, p=50):
    X, y, res = _lasso_gaussian_path(device, n, p)
    err = 0.0
    for i in (2, 10, 19):
        ref = Lasso(alpha=res.lambdas[i], tol=1e-12, max_iter=200000).fit(X, y)
        err = max(err, float(np.abs(res.coef[i] - ref.coef_).max()),
                  abs(float(res.intercept[i] - ref.intercept_)))
    return {"check": "lasso_gaussian_vs_sklearn", "n": n, "m": p, "coef_max_abs_err": err,
            "pass": err <= TOL["lasso_gauss_abs"]}


def check_lasso_binomial(device, n=1000, p=20):
    X = _standardized(n, p, 12)
    eta = X[:, :3] @ np.array([1.2, -1.0, 0.8])
    y = np.random.default_rng(13).binomial(1, 1 / (1 + np.exp(-eta))).astype(float)
    res = lasso_cv(X, y, family="binomial", n_folds=0, n_lambda=10, device=device,
                   tol=1e-10, max_iter=50000)
    err = 0.0
    for i in (3, 9):
        ref = LogisticRegression(**_SK_L1, C=1 / (n * res.lambdas[i]), solver="saga",
                                 tol=1e-10, max_iter=200000).fit(X, y)
        err = max(err, float(np.abs(res.coef[i] - ref.coef_[0]).max()))
    return {"check": "lasso_binomial_vs_sklearn", "n": n, "m": p, "coef_max_abs_err": err,
            "pass": err <= TOL["lasso_binom_abs"]}


def check_lasso_cox_recovery(device, n=2000, p=50):
    """No exact reference installed: check CV recovers the simulated signals."""
    X = _standardized(n, p, 14)
    rng = np.random.default_rng(15)
    t_event = rng.exponential(1 / np.exp(X[:, :3] @ np.array([0.9, -0.7, 0.6])))
    t_cens = rng.exponential(2.0, size=n)
    res = lasso_cv(X, family="cox", time=np.minimum(t_event, t_cens),
                   event=(t_event <= t_cens).astype(float), n_folds=5, n_lambda=30, device=device)
    chosen = set(res.selected("min"))
    return {"check": "lasso_cox_signal_recovery", "n": n, "m": p, "n_selected": len(chosen),
            "pass": {"x0", "x1", "x2"} <= chosen}


def check_lasso_cox(device, n=1000, p=30):
    """Breslow Cox lasso path (tied times) vs scikit-survival Coxnet; skipped if not installed."""
    row = {"check": "lasso_cox_vs_sksurv", "n": n, "m": p}
    if CoxnetSurvivalAnalysis is None:
        return {**row, "skipped": "scikit-survival not installed", "pass": True}
    X = _standardized(n, p, 16)
    rng = np.random.default_rng(17)
    t_event = rng.exponential(1 / np.exp(X[:, :3] @ np.array([0.9, -0.7, 0.6])))
    t_cens = rng.exponential(2.0, size=n)
    time_, event = np.ceil(np.minimum(t_event, t_cens) * 20), t_event <= t_cens
    res = lasso_cv(X, family="cox", time=time_, event=event.astype(float), n_folds=0,
                   n_lambda=10, device=device, tol=1e-10, max_iter=50000)
    err = 0.0
    for i in (3, 6, 9):
        ref = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[res.lambdas[i]], tol=1e-12,
                                     max_iter=1000000).fit(X, Surv.from_arrays(event, time_))
        err = max(err, float(np.abs(res.coef[i] - ref.coef_[:, 0]).max()))
    return {**row, "coef_max_abs_err": err, "pass": err <= TOL["lasso_cox_abs"]}


def _phreg_lasso_kkt(model, beta, lam):
    """Max violation of the lasso KKT conditions, using statsmodels' stratified score."""
    g = model.score(beta) / model.nobs
    nz = beta != 0
    return max(np.abs(g[nz] - lam * np.sign(beta[nz])).max(initial=0.0),
               (np.abs(g[~nz]) - lam).max(initial=0.0))


def check_lasso_cox_strata(device, n=1000, p=20):
    """Stratified Breslow Cox lasso (4 strata, tied times) vs statsmodels
    PHReg(strata=).fit_regularized(L1_wt=1), whose objective matches on standardized X.

    Gated: gpureg's KKT residual from statsmodels' score at every lambda, and coefficients at the
    lambdas where statsmodels' coordinate descent itself satisfies KKT (it can leave a violating
    coefficient at zero; such lambdas are counted in ref_not_optimal).
    """
    X = _standardized(n, p, 18)
    rng = np.random.default_rng(19)
    strata = rng.integers(0, 4, size=n)
    eta = X[:, :3] @ np.array([0.8, -0.6, 0.5]) + 0.5 * strata
    t_event = np.ceil(rng.exponential(1 / np.exp(eta)) * 20)
    t_cens = np.ceil(rng.exponential(2.0, size=n) * 20)
    time_, event = np.minimum(t_event, t_cens), (t_event <= t_cens).astype(float)
    res = lasso_cv(X, family="cox", time=time_, event=event, strata=strata, n_folds=0,
                   n_lambda=10, device=device, tol=1e-10, max_iter=50000)
    model = sm.PHReg(time_, X, status=event, ties="breslow", strata=strata)
    err = kkt = 0.0
    not_opt = 0
    for lam, coef in zip(res.lambdas, res.coef):
        kkt = max(kkt, _phreg_lasso_kkt(model, coef, lam))
        ref = model.fit_regularized(alpha=lam, L1_wt=1.0).params
        if _phreg_lasso_kkt(model, ref, lam) > TOL["lasso_kkt_abs"]:
            not_opt += 1
            continue
        err = max(err, float(np.abs(coef - ref).max()))
    return {"check": "lasso_cox_strata_vs_statsmodels", "n": n, "m": p, "coef_max_abs_err": err,
            "kkt_max_abs": kkt, "ref_not_optimal": not_opt,
            "pass": err <= TOL["lasso_cox_abs"] and kkt <= TOL["lasso_kkt_abs"]
            and not_opt < len(res.lambdas)}


def check_lasso_strong_rules(device, n=1000, p=200):
    """Binomial CV path with and without strong-rule screening: same coefficients and CV curve."""
    X = _standardized(n, p, 18)
    eta = X[:, :5] @ np.array([1.0, -0.8, 0.6, 0.5, -0.4])
    y = np.random.default_rng(19).binomial(1, 1 / (1 + np.exp(-eta))).astype(float)
    fits = [lasso_cv(X, y, family="binomial", n_folds=5, n_lambda=30, device=device,
                     unpenalized=[p - 1], strong_rules=sr) for sr in (True, False)]
    err = max(float(np.abs(fits[0].coef - fits[1].coef).max()),
              float(np.abs(fits[0].cv_mean - fits[1].cv_mean).max()))
    return {"check": "lasso_strong_rules_vs_full", "n": n, "m": p, "max_abs_err": err,
            "mean_cols_fitted": float(fits[0].n_fitted.mean()),
            "pass": err <= TOL["lasso_strong_abs"]}


def float32_screen(kind, n, m, seed, device, ties="efron"):
    """Same data fitted in float32 and float64: max discrepancies, no pass/fail."""
    C, F, out = simulate_screen(kind, n, m, seed)
    lo = _fit_screen(kind, C, F, out, ties, device, torch.float32)
    hi = _fit_screen(kind, C, F, out, ties, device, torch.float64)
    beta, se = lo["beta"].to_numpy(np.float64), lo["se"].to_numpy(np.float64)
    p_lo, p_hi = lo["p"].to_numpy(np.float64), hi["p"].to_numpy()
    shown = (p_lo > 0) & (p_hi > 0)  # p is float64 for any fit dtype; 0 only on true underflow
    p_err = np.abs(_neg_log10(p_lo[shown]) - _neg_log10(p_hi[shown]))
    p_rel = p_err / np.maximum(_neg_log10(p_hi[shown]), 1.0)
    return {"check": _screen_name(kind, ties), "n": n, "m": m,
            "beta_max_abs_err": float(np.abs(beta - hi["beta"].to_numpy()).max()),
            "se_max_rel_err": float((np.abs(se - hi["se"].to_numpy()) / hi["se"].to_numpy()).max()),
            "log10p_max_abs_err": float(p_err.max()) if p_err.size else 0.0,
            "log10p_max_rel_err": float(p_rel.max()) if p_rel.size else 0.0,
            "p_underflow": int(((p_lo == 0) & (p_hi > 0)).sum()), "converged_frac": float(lo["converged"].mean())}


def check_lasso_gaussian_float32(device, n=1000, p=50):
    """float32 lasso path vs float64, without hitting max_iter (gated: slack and tol dtype-aware)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lo = _lasso_gaussian_path(device, n, p, torch.float32)[2]
    hit_max_iter = any("max_iter" in str(w.message) for w in caught)
    hi = _lasso_gaussian_path(device, n, p)[2]
    err = float(np.abs(lo.coef.astype(np.float64) - hi.coef).max())
    return {"check": "lasso_gaussian_float32_vs_float64", "n": n, "m": p, "coef_max_abs_err": err,
            "hit_max_iter": hit_max_iter, "pass": err <= TOL["lasso_f32_abs"] and not hit_max_iter}


def check_screen_binomial_float32(device, n, m):
    """float32 binomial screen: every model converged and beta close to float64 (gated)."""
    row = float32_screen("binomial", n, m, 1, device)
    ok = row["converged_frac"] == 1.0 and row["beta_max_abs_err"] <= TOL["screen_f32_beta_abs"]
    return {"check": "screen_binomial_float32_vs_float64", "n": n, "m": m,
            "beta_max_abs_err": row["beta_max_abs_err"], "converged_frac": row["converged_frac"],
            "pass": ok}


def check_screen_gaussian_float32_p(device, n, m):
    """float32 gaussian screen: p-values (some below float32 tiny) match float64 (gated)."""
    row = float32_screen("gaussian", n, m, 1, device)
    ok = row["p_underflow"] == 0 and row["log10p_max_rel_err"] <= TOL["screen_f32_log10p_rel"]
    return {"check": "screen_gaussian_float32_pvalues", "n": n, "m": m,
            "log10p_max_rel_err": row["log10p_max_rel_err"], "p_underflow": row["p_underflow"],
            "pass": ok}


def degenerate_features(kind, out, C, seed):
    """Columns: ordinary, constant, collinear with a covariate, constant after NaN drop, then
    separating and quasi-separating.

    The separating column is 1 exactly where y == 1 (binomial), y == 0 (poisson) or event == 1
    (cox); the quasi-separating one is 1 on a random half of those rows (cox: of censored rows,
    a monotone likelihood). Both MLEs are infinite; gaussian has no such columns.
    """
    rng = np.random.default_rng(seed)
    n = C.shape[0]
    half_nan = np.where(rng.random(n) < 0.5, np.nan, 3.0)
    cols = [rng.normal(size=n), np.ones(n), 2.0 * C[:, 0] + 1.0, half_nan]
    if kind == "gaussian":
        return np.column_stack(cols)
    sep = out[1] if kind == "cox" else (out == (1.0 if kind == "binomial" else 0.0)).astype(float)
    quasi = (out[1] == 0) if kind == "cox" else (sep == 1)
    return np.column_stack(cols + [sep, (quasi & (rng.random(n) < 0.5)).astype(float)])


def _degenerate_ok(res):
    """Ordinary column clean; aliased columns singular with NaN se/p; separated columns flagged
    infinite (or singular) and not significant."""
    singular, conv, se, p, inf = (res[c].to_numpy()
                                  for c in ("singular", "converged", "se", "p", "infinite"))
    ordinary = conv[0] and not singular[0] and not inf[0] and np.isfinite(p[0])
    aliased = singular[1:4].all() and np.isnan(se[1:4]).all() and np.isnan(p[1:4]).all()
    separated = all((inf[j] or singular[j]) and not (conv[j] and not singular[j] and p[j] < 0.5)
                    for j in range(4, len(p)))
    return bool(ordinary and aliased and separated)


def check_degenerate(kind, device, n=500):
    """Constant, collinear and separating features in float64 and float32 (gated)."""
    C, _, out = simulate_screen(kind, n, 2, 3)
    F = degenerate_features(kind, out, C, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the 'infinite' warning is expected here
        ok = {f"pass_{name}": _degenerate_ok(_fit_screen(kind, C, F, out, "efron", device, dt))
              for name, dt in (("float64", torch.float64), ("float32", torch.float32))}
    return {"check": f"degenerate_{kind}", "n": n, "m": F.shape[1], **ok, "pass": all(ok.values())}


def run_float32(quick, device):
    n, m = (500, 10) if quick else (5000, 100)
    rows = [float32_screen(kind, n, m, 1, device) for kind in ("gaussian", "binomial", "poisson")]
    return rows + [float32_screen("cox", n, m, 2, device, ties) for ties in ("efron", "breslow")]


def run_all(quick, device):
    sizes = [(500, 10)] if quick else [(500, 10), (5000, 100)]
    checks = []
    for n, m in sizes:
        for kind in ("gaussian", "binomial", "poisson"):
            checks.append(check_screen(kind, n, m, 1, device))
        for ties in ("efron", "breslow"):
            checks.append(check_screen("cox", n, m, 2, device, ties))
    checks.append(check_screen_poisson_offset(device, *sizes[-1]))
    for kind in ("gaussian", "binomial", "poisson"):
        checks.append(check_screen_glm_robust(kind, device, *sizes[-1]))
    for ties in ("efron", "breslow"):
        checks.append(check_screen_cox_strata(device, *sizes[-1], ties))
        checks.append(check_screen_cox_entry(device, *sizes[-1], ties))
        checks.append(check_screen_cox_robust(device, *sizes[-1], ties))
    checks.append(check_screen_binomial_float32(device, *sizes[-1]))
    checks.append(check_screen_gaussian_float32_p(device, *sizes[-1]))
    for kind in ("gaussian", "binomial", "poisson", "cox"):
        checks.append(check_degenerate(kind, device))
    for fn in (check_lasso_gaussian, check_lasso_gaussian_float32, check_lasso_binomial,
               check_lasso_cox, check_lasso_cox_strata, check_lasso_cox_recovery,
               check_lasso_strong_rules):
        checks.append(fn(device))
    return checks


def _float32_lines(rows):
    keys = ("beta_max_abs_err", "se_max_rel_err", "log10p_max_abs_err", "log10p_max_rel_err",
            "p_underflow",
            "converged_frac")
    lines = ["", "## float32 vs float64 (informational, not gated)", "",
             "p is returned in float64 for every fit dtype; underflow counts p == 0 in float32 "
             "fits only.", "",
             "| fit | n | m | beta max abs | se max rel | -log10 p max abs | -log10 p max rel "
             "| p underflow | converged |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        cells = [f"{r[k]:.2g}" if k in r else "-" for k in keys]
        lines.append(f"| {r['check']} | {r['n']} | {r['m']} | " + " | ".join(cells) + " |")
    return lines


def _pass_cell(check):
    if check.get("skipped"):
        return f"skipped ({check['skipped']})"
    return "yes" if check["pass"] else "NO"


def write_report(checks, out_dir, device, float32=()):
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "device": device, "torch": torch.__version__, "python": platform.python_version(),
              "all_pass": all(c["pass"] for c in checks), "checks": checks,
              "float32_accuracy": list(float32)}
    (out_dir / "latest.json").write_text(json.dumps(report, indent=2))
    with open(out_dir / "history.jsonl", "a") as fh:
        fh.write(json.dumps({k: report[k] for k in ("timestamp", "device", "all_pass")}) + "\n")
    lines = [f"# Validation {report['timestamp']} ({device})", "", "| check | n | m | pass |",
             "|---|---|---|---|"]
    lines += [f"| {c['check']} | {c['n']} | {c['m']} | {_pass_cell(c)} |" for c in checks]
    if float32:
        lines += _float32_lines(float32)
    (out_dir / "latest.md").write_text("\n".join(lines) + "\n")
    return report["all_pass"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default="validation/results")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ok = write_report(run_all(args.quick, device), Path(args.out), device,
                      run_float32(args.quick, device))
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED: see latest.md")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
