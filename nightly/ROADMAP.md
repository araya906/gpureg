# Roadmap (nightly runs take the first unchecked item)

- [x] Make the first validation run pass end to end (harness and package were written without being executed).
- [ ] Add a float32 accuracy report to the validation (max errors vs float64) without failing the gate.
- [ ] Add a scikit-survival `CoxnetSurvivalAnalysis` reference for Cox lasso (optional dependency; skip if unavailable).
- [ ] Add a separation / constant-feature scenario and check that `singular` and `converged` flags are set sensibly.
- [ ] GLM: support offsets (e.g. log person-time for poisson).
- [ ] Cox: support strata.
- [ ] Cox: support start-stop (left-truncated) data.
- [ ] Lasso: strong-rule screening to skip inactive features along the path.
- [ ] Optional sandwich (robust) standard errors for GLM and Cox.
- [ ] Benchmark the automatic chunk-size heuristic on larger simulated data (CPU) and document results.
