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


def test_float32_report_does_not_gate(tmp_path):
    checks = [{"check": "ok", "n": 1, "m": 1, "pass": True}]
    bad = [{"check": "lasso_gaussian_path", "n": 1, "m": 1, "beta_max_abs_err": 1.0}]
    assert run_checks.write_report(checks, tmp_path, "cpu", bad)
    report = json.loads((tmp_path / "latest.json").read_text())
    assert report["all_pass"] and report["float32_accuracy"] == bad
    row = "| lasso_gaussian_path | 1 | 1 | 1 | - | - | - | - |"
    assert row in (tmp_path / "latest.md").read_text()
