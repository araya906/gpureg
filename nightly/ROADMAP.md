# Roadmap (nightly runs take the first unchecked item)

- [x] Make the first validation run pass end to end (harness and package were written without being executed).
- [x] Add a float32 accuracy report to the validation (max errors vs float64) without failing the gate.
- [x] Lasso float32: the fixed `1e-12 * |f|` acceptance slack in `_backtrack` is below float32 rounding, so L keeps doubling and the path stalls (coef error 0.6 vs float64 in the report). Make the slack dtype-aware (e.g. a few `finfo(dtype).eps`); a quick trial gave ~1e-6.
- [ ] float32 screens: default `tol=1e-9` is below float32 resolution, so `converged` can be False on good fits (binomial 98%); scale the default tol by dtype. Also consider returning p in float64 (float32 p underflows below ~1e-38). Same for lasso: a float32 fit with tol=1e-10 runs to max_iter and warns, although coefficients match float64 to ~3e-7.
- [ ] Add a scikit-survival `CoxnetSurvivalAnalysis` reference for Cox lasso (optional dependency; skip if unavailable).
- [ ] Add a separation / constant-feature scenario and check that `singular` and `converged` flags are set sensibly.
- [ ] GLM: support offsets (e.g. log person-time for poisson).
- [ ] Cox: support strata.
- [ ] Cox: support start-stop (left-truncated) data.
- [ ] Lasso: strong-rule screening to skip inactive features along the path.
- [ ] Optional sandwich (robust) standard errors for GLM and Cox.
- [ ] Benchmark the automatic chunk-size heuristic on larger simulated data (CPU) and document results.
