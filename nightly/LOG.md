# Nightly log

- 2026-11-09: initial package, validation harness and nightly brief (not yet executed).
- 2026-11-09: first end-to-end run (CPU): 13/13 validation checks and 13/13 tests pass; switched sklearn L1 reference to l1_ratio=1 (penalty deprecated in 1.8).
- 2026-11-10: 13/13 checks, 16/16 tests (CPU). Added informational float32-vs-float64 report to validation; screens agree to <=4e-5 in beta, lasso float32 path stalls (0.6 error) -> new ROADMAP items.
- 2026-11-11: 14/14 checks, 17/17 tests (CPU). Lasso backtracking slack now 10*finfo(dtype).eps*|f| (was fixed 1e-12): float32 path error vs float64 0.6 -> 3e-7; float32 lasso check moved from informational to gated (tol 1e-4) plus unit test.
