# Nightly log

- 2026-11-09: initial package, validation harness and nightly brief (not yet executed).
- 2026-11-09: first end-to-end run (CPU): 13/13 validation checks and 13/13 tests pass; switched sklearn L1 reference to l1_ratio=1 (penalty deprecated in 1.8).
- 2026-11-10: 13/13 checks, 16/16 tests (CPU). Added informational float32-vs-float64 report to validation; screens agree to <=4e-5 in beta, lasso float32 path stalls (0.6 error) -> new ROADMAP items.
- 2026-11-11: 14/14 checks, 17/17 tests (CPU). Lasso backtracking slack now 10*finfo(dtype).eps*|f| (was fixed 1e-12): float32 path error vs float64 0.6 -> 3e-7; float32 lasso check moved from informational to gated (tol 1e-4) plus unit test.
- 2026-11-12: 15/15 checks, 18/18 tests (CPU). Newton tol and step-halving slack now floored at 10*finfo(dtype).eps (float64 unchanged): float32 binomial screen converged 98% -> 100%, beta error vs float64 2.9e-5 -> 1.1e-7; new gated check screen_binomial_float32_vs_float64. Split remaining float32 p / lasso tol work into ROADMAP items.
