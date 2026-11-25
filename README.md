# gpureg

Batched GLM, Cox and lasso fitting on GPU (PyTorch), with standard errors and p-values.
Falls back to CPU when no CUDA device is available.

## Install

Install the PyTorch build matching your CUDA version first (see pytorch.org), then:

```bash
pip install -e .            # package
pip install -e ".[test]"    # plus statsmodels / scikit-learn for the tests
```

## Usage

```python
from gpureg import glm_screen, cox_screen, glm_fit, cox_fit, lasso_cv

# One model per feature: y ~ 1 + covariates + feature_j (all features batched on the GPU)
res = glm_screen(proteins, y, covariates, family="binomial")
res = cox_screen(proteins, time, event, covariates, ties="efron")

# Single models with a full coefficient table (diagnostics in .attrs)
tab = glm_fit(X, y, family="poisson")
# Rates: offset enters the linear predictor with coefficient 1 (glm_screen too)
tab = glm_fit(X, events, family="poisson", offset=np.log(person_years))
# Sandwich (HC0) standard errors, robust to a misspecified variance (glm_screen too)
tab = glm_fit(X, y, family="poisson", robust=True)
tab = cox_fit(X, time, event)
# Stratified Cox: one baseline hazard per stratum (cox_screen too)
tab = cox_fit(X, time, event, strata=centre)
# Start-stop data (left truncation, time-varying covariates): at risk on (entry, time]
tab = cox_fit(X, stop, event, entry=start)
# Lin-Wei robust standard errors, as coxph(robust=TRUE) (cox_screen too)
tab = cox_fit(X, time, event, robust=True)

# Lasso / elastic net with 10-fold CV; folds are fitted together as one batch
fit = lasso_cv(proteins_and_covs, family="cox", time=time, event=event,
               unpenalized=["age", "sex"], alpha=1.0, n_folds=10)
fit.selected("1se"); fit.coef_at("min"); fit.path_frame()
```

Inputs can be numpy arrays or pandas objects; DataFrame column names are kept.

## Design

| Model | Algorithm | Matches |
|---|---|---|
| GLM (gaussian, binomial, poisson; canonical links) | Batched Newton with step halving | `glm()` / statsmodels `GLM` |
| Cox (Efron default, Breslow; optional strata and start-stop times) | Batched Newton with step halving | `coxph()` / statsmodels `PHReg` |
| Lasso / elastic net (gaussian, binomial, Cox-Breslow) | Batched FISTA with backtracking and restart | `glmnet` scaling, scikit-learn `Lasso` |

- **Screening** builds a `(models, n, k)` design per chunk; chunk size is set from free GPU memory (`chunk_size=` overrides). On CPU the default is a 256 MB budget and at most 64 models per chunk. A CPU benchmark (`python validation/bench_chunks.py`, results in `validation/results/bench_chunks.md`; 2 threads, float64, n = 5,000-100,000) puts these auto chunks (3-64 models) within about 0-30% of the fastest fixed size for GLMs and untied Cox (the former 2 GB budget was 1.3-2.2x slower). The memory estimate is close for GLMs and untied Cox (measured peak 0.9-1.3x the estimate at the auto size) but 4-70x too high for Cox with many tied event times (`copies` grows with 2 x the largest tie count), so heavily tied Cox screens still get 1-3 models per chunk; there `chunk_size=4`-`16` was up to 1.4x faster at n = 5,000-20,000. GPU timing was not measured.
- **Missing feature values** (NaN) are handled per model with zero weights, so each model uses its own complete rows. Outcome and covariates must be complete.
- **Precision**: `float64` by default. Consumer GPUs run float64 much slower than float32; pass `dtype=torch.float32` for speed at the cost of accuracy in SEs and very small p-values.
- **Diagnostics** per model: `converged`, `n_iter`, `singular` (information matrix rank-deficient, e.g. a constant or collinear feature, or in Cox one constant within every stratum; `se`, `z` and `p` are then NaN), `n_obs`, `infinite` (an estimate diverges: complete or quasi-separation in a GLM, monotone likelihood in Cox; detected as in R's `coxph` from the Newton step still left at the end, and reported with a warning; such fits have very large SEs with p near 1). `glm_fit`/`cox_fit` give `infinite` per term.
- **Robust SEs** (`robust=True`): GLM uses the HC0 sandwich `I⁻¹ (Σ rᵢ² xᵢxᵢ') I⁻¹` with `r = y − μ`, no dispersion or small-sample factor (statsmodels `cov_type="HC0"`, R `sandwich::vcovHC(type="HC0")`). Cox uses the Lin–Wei sandwich `D'D` with dfbeta `D = U I⁻¹` from per-row score residuals `U` (Efron-weighted for tied deaths, combining with strata and entry), as R `coxph(robust=TRUE)` without `cluster`. Estimates are unchanged; `se`, `z` and `p` use the robust covariance.
- **Lasso standardization** uses population SDs on the full data (also for CV folds); coefficients are returned on the original scale. CV error is held-out deviance per observation; Cox uses the Verweij–van Houwelingen cross-validated partial likelihood.
- **Lasso strong rules** (`strong_rules=True`, default): each lambda is fitted only on the sequential strong set (union over CV folds), then the KKT conditions are checked on the dropped columns and violators are added and refitted, so the path is unchanged. `fit.n_fitted` gives the columns fitted per lambda. The saving is largest early in the path and when few features enter; in CPU trials (5 folds, p up to 3000) it saved 0-23% of the time.

## Caveats

- Lasso selects variables but its coefficients have no valid p-values, and refitting the selected set with `glm_fit`/`cox_fit` gives over-optimistic p-values. Use post-selection inference or sample splitting if you need tests.
- Cox: no case weights; `lasso_cv` has no strata or entry times. With `entry=`, a row is at risk on `(entry, time]` as in R's `survival`; statsmodels `PHReg(entry=)` also counts it at its entry time, so results differ when an entry time equals an event time.
- GPU speed-up comes from batching many models or large `n`; a single small model is faster on CPU.

## Tests

`tests/` compare results against statsmodels (GLM, PHReg) and scikit-learn (`Lasso`), the Cox lasso against scikit-survival (`CoxnetSurvivalAnalysis`, optional: `pip install -e ".[survival]"`, skipped if absent), and check that CV lasso recovers simulated signals. Run with `pytest` (CPU is enough).
