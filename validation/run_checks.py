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
from sklearn.linear_model import Lasso, LogisticRegression

from gpureg import cox_screen, glm_screen, lasso_cv

# scikit-learn >= 1.8 deprecates `penalty` (removed in 1.10); l1_ratio=1 selects pure L1.
_SK_L1 = ({"l1_ratio": 1.0} if tuple(int(v) for v in sklearn.__version__.split(".")[:2]) >= (1, 8)
          else {"penalty": "l1"})
TOL = {"beta_abs": 1e-5, "se_rel": 1e-4, "log10p_abs": 1e-3, "lasso_gauss_abs": 1e-5,
       "lasso_binom_abs": 1e-3}
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
          and p_err <= TOL["log10p_abs"] and bool(res["converged"].all()))
    return {"beta_max_abs_err": beta_err, "se_max_rel_err": se_err,
            "log10p_max_abs_err": p_err, "all_converged": bool(res["converged"].all()), "pass": ok}


def _fit_screen(kind, C, F, out, ties, device, dtype=torch.float64):
    if kind == "cox":
        return cox_screen(F, out[0], out[1], C, ties=ties, device=device, dtype=dtype)
    return glm_screen(F, out, C, family=kind, device=device, dtype=dtype)


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


def float32_screen(kind, n, m, seed, device, ties="efron"):
    """Same data fitted in float32 and float64: max discrepancies, no pass/fail."""
    C, F, out = simulate_screen(kind, n, m, seed)
    lo = _fit_screen(kind, C, F, out, ties, device, torch.float32)
    hi = _fit_screen(kind, C, F, out, ties, device, torch.float64)
    beta, se = lo["beta"].to_numpy(np.float64), lo["se"].to_numpy(np.float64)
    p_lo, p_hi = lo["p"].to_numpy(np.float64), hi["p"].to_numpy()
    shown = p_hi >= np.finfo(np.float32).tiny  # smaller p underflow in float32 output
    p_err = np.abs(_neg_log10(p_lo[shown]) - _neg_log10(p_hi[shown]))
    return {"check": _screen_name(kind, ties), "n": n, "m": m,
            "beta_max_abs_err": float(np.abs(beta - hi["beta"].to_numpy()).max()),
            "se_max_rel_err": float((np.abs(se - hi["se"].to_numpy()) / hi["se"].to_numpy()).max()),
            "log10p_max_abs_err": float(p_err.max()) if p_err.size else 0.0,
            "p_underflow": int((~shown).sum()), "converged_frac": float(lo["converged"].mean())}


def float32_lasso_gaussian(device, n=1000, p=50):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lo = _lasso_gaussian_path(device, n, p, torch.float32)[2]
    hi = _lasso_gaussian_path(device, n, p)[2]
    err = float(np.abs(lo.coef.astype(np.float64) - hi.coef).max())
    return {"check": "lasso_gaussian_path", "n": n, "m": p, "beta_max_abs_err": err}


def run_float32(quick, device):
    n, m = (500, 10) if quick else (5000, 100)
    rows = [float32_screen(kind, n, m, 1, device) for kind in ("gaussian", "binomial", "poisson")]
    rows += [float32_screen("cox", n, m, 2, device, ties) for ties in ("efron", "breslow")]
    return rows + [float32_lasso_gaussian(device)]


def run_all(quick, device):
    sizes = [(500, 10)] if quick else [(500, 10), (5000, 100)]
    checks = []
    for n, m in sizes:
        for kind in ("gaussian", "binomial", "poisson"):
            checks.append(check_screen(kind, n, m, 1, device))
        for ties in ("efron", "breslow"):
            checks.append(check_screen("cox", n, m, 2, device, ties))
    for fn in (check_lasso_gaussian, check_lasso_binomial, check_lasso_cox_recovery):
        checks.append(fn(device))
    return checks


def _float32_lines(rows):
    keys = ("beta_max_abs_err", "se_max_rel_err", "log10p_max_abs_err", "p_underflow",
            "converged_frac")
    lines = ["", "## float32 vs float64 (informational, not gated)", "",
             "-log10 p compared where p >= float32 tiny; smaller p counted as underflow.", "",
             "| fit | n | m | beta max abs | se max rel | -log10 p max abs | p underflow "
             "| converged |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        cells = [f"{r[k]:.2g}" if k in r else "-" for k in keys]
        lines.append(f"| {r['check']} | {r['n']} | {r['m']} | " + " | ".join(cells) + " |")
    return lines


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
    lines += [f"| {c['check']} | {c['n']} | {c['m']} | {'yes' if c['pass'] else 'NO'} |" for c in checks]
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
