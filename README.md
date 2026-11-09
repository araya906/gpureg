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
tab = cox_fit(X, time, event)

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
| Cox (Efron default, Breslow) | Batched Newton with step halving | `coxph()` / statsmodels `PHReg` |
| Lasso / elastic net (gaussian, binomial, Cox-Breslow) | Batched FISTA with backtracking and restart | `glmnet` scaling, scikit-learn `Lasso` |

- **Screening** builds a `(models, n, k)` design per chunk; chunk size is set from free GPU memory (`chunk_size=` overrides).
- **Missing feature values** (NaN) are handled per model with zero weights, so each model uses its own complete rows. Outcome and covariates must be complete.
- **Precision**: `float64` by default. Consumer GPUs run float64 much slower than float32; pass `dtype=torch.float32` for speed at the cost of accuracy in SEs and very small p-values.
- **Diagnostics** per model: `converged`, `n_iter`, `singular` (information matrix not invertible, SEs unreliable), `n_obs`.
- **Lasso standardization** uses population SDs on the full data (also for CV folds); coefficients are returned on the original scale. CV error is held-out deviance per observation; Cox uses the Verweij–van Houwelingen cross-validated partial likelihood.

## Caveats

- Lasso selects variables but its coefficients have no valid p-values, and refitting the selected set with `glm_fit`/`cox_fit` gives over-optimistic p-values. Use post-selection inference or sample splitting if you need tests.
- Cox: right-censored data only (no strata, start-stop times or case weights).
- GPU speed-up comes from batching many models or large `n`; a single small model is faster on CPU.

## Tests

`tests/` compare results against statsmodels (GLM, PHReg) and scikit-learn (`Lasso`), and check that CV lasso recovers simulated signals. Run with `pytest` (CPU is enough).
