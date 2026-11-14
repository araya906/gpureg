# Nightly log

- 2026-11-09: initial package, validation harness and nightly brief (not yet executed).
- 2026-11-09: first end-to-end run (CPU): 13/13 validation checks and 13/13 tests pass; switched sklearn L1 reference to l1_ratio=1 (penalty deprecated in 1.8).
- 2026-11-10: 13/13 checks, 16/16 tests (CPU). Added informational float32-vs-float64 report to validation; screens agree to <=4e-5 in beta, lasso float32 path stalls (0.6 error) -> new ROADMAP items.
- 2026-11-11: 14/14 checks, 17/17 tests (CPU). Lasso backtracking slack now 10*finfo(dtype).eps*|f| (was fixed 1e-12): float32 path error vs float64 0.6 -> 3e-7; float32 lasso check moved from informational to gated (tol 1e-4) plus unit test.
- 2026-11-12: 15/15 checks, 18/18 tests (CPU). Newton tol and step-halving slack now floored at 10*finfo(dtype).eps (float64 unchanged): float32 binomial screen converged 98% -> 100%, beta error vs float64 2.9e-5 -> 1.1e-7; new gated check screen_binomial_float32_vs_float64. Split remaining float32 p / lasso tol work into ROADMAP items.
- 2026-11-13: 16/16 checks, 20/20 tests (CPU). Wald z and p now computed in float64 on the CPU for any fit dtype: float32 screens no longer underflow p to 0 (was 1-2 of 100), -log10 p rel error vs float64 <=2.2e-6; new gated check screen_gaussian_float32_pvalues plus unit tests.
- 2026-11-14: 16/16 checks, 21/21 tests (CPU). lasso_cv tol now floored at 10*finfo(dtype).eps like _newton (float64 unchanged): float32 tol=1e-10 path no longer runs to max_iter/warns, coef error vs float64 1.0e-6 (gate 1e-4); float32 lasso check now also fails on max_iter, plus unit test.
