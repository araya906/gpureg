import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
import run_checks  # noqa: E402


@pytest.mark.parametrize("kind", ["gaussian", "cox"])
def test_float32_screen_reports_small_errors(kind):
    row = run_checks.float32_screen(kind, 300, 4, 0, "cpu")
    assert row["converged_frac"] == 1.0  # default tol is raised to the float32 rounding floor
    assert row["beta_max_abs_err"] < 1e-3
    assert row["se_max_rel_err"] < 1e-3
    assert row["log10p_max_abs_err"] < 1e-2


def test_float32_binomial_screen_converges():
    # Before the dtype-aware tol/halving slack, 2 of these 100 float32 fits stalled at iteration 3.
    row = run_checks.check_screen_binomial_float32("cpu", 5000, 100)
    assert row["converged_frac"] == 1.0 and row["pass"]


def test_float32_pvalues_do_not_underflow():
    # Gaussian n=5000 has p below float32 tiny (~1e-38); float32 p used to underflow to 0.
    row = run_checks.check_screen_gaussian_float32_p("cpu", 5000, 100)
    assert row["p_underflow"] == 0 and row["pass"]


def test_float32_report_does_not_gate(tmp_path):
    checks = [{"check": "ok", "n": 1, "m": 1, "pass": True}]
    bad = [{"check": "lasso_gaussian_path", "n": 1, "m": 1, "beta_max_abs_err": 1.0}]
    assert run_checks.write_report(checks, tmp_path, "cpu", bad)
    report = json.loads((tmp_path / "latest.json").read_text())
    assert report["all_pass"] and report["float32_accuracy"] == bad
    row = "| lasso_gaussian_path | 1 | 1 | 1 | - | - | - | - | - |"
    assert row in (tmp_path / "latest.md").read_text()


def test_lasso_cox_matches_sksurv():
    pytest.importorskip("sksurv")
    row = run_checks.check_lasso_cox("cpu", n=400, p=10)
    assert "skipped" not in row and row["pass"]


def test_lasso_cox_skips_without_sksurv(monkeypatch):
    monkeypatch.setattr(run_checks, "CoxnetSurvivalAnalysis", None)
    row = run_checks.check_lasso_cox("cpu")
    assert row["pass"] and row["skipped"]


def test_skipped_check_is_marked_in_report(tmp_path):
    checks = [{"check": "opt", "n": 1, "m": 1, "skipped": "x not installed", "pass": True}]
    assert run_checks.write_report(checks, tmp_path, "cpu")
    assert "| opt | 1 | 1 | skipped (x not installed) |" in (tmp_path / "latest.md").read_text()


@pytest.mark.parametrize("kind", ["gaussian", "binomial", "poisson", "cox"])
def test_degenerate_features_check_passes(kind):
    row = run_checks.check_degenerate(kind, "cpu", n=300)
    assert row["pass_float64"] and row["pass_float32"]


def test_poisson_offset_check_passes():
    row = run_checks.check_screen_poisson_offset("cpu", 400, 5)
    assert row["pass"] and row["beta_max_abs_err"] < 1e-6


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_cox_strata_check_passes(ties):
    row = run_checks.check_screen_cox_strata("cpu", 400, 5, ties)
    assert row["pass"] and row["beta_max_abs_err"] < 1e-6


def test_lasso_strong_rules_check_passes():
    row = run_checks.check_lasso_strong_rules("cpu", n=300, p=40)
    assert row["pass"] and row["mean_cols_fitted"] < 40
