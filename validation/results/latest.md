# Validation 2026-11-13T01:54:34+00:00 (cpu)

| check | n | m | pass |
|---|---|---|---|
| screen_gaussian | 500 | 10 | yes |
| screen_binomial | 500 | 10 | yes |
| screen_poisson | 500 | 10 | yes |
| screen_cox_efron | 500 | 10 | yes |
| screen_cox_breslow | 500 | 10 | yes |
| screen_gaussian | 5000 | 100 | yes |
| screen_binomial | 5000 | 100 | yes |
| screen_poisson | 5000 | 100 | yes |
| screen_cox_efron | 5000 | 100 | yes |
| screen_cox_breslow | 5000 | 100 | yes |
| screen_binomial_float32_vs_float64 | 5000 | 100 | yes |
| screen_gaussian_float32_pvalues | 5000 | 100 | yes |
| lasso_gaussian_vs_sklearn | 1000 | 50 | yes |
| lasso_gaussian_float32_vs_float64 | 1000 | 50 | yes |
| lasso_binomial_vs_sklearn | 1000 | 20 | yes |
| lasso_cox_signal_recovery | 2000 | 50 | yes |

## float32 vs float64 (informational, not gated)

p is returned in float64 for every fit dtype; underflow counts p == 0 in float32 fits only.

| fit | n | m | beta max abs | se max rel | -log10 p max abs | -log10 p max rel | p underflow | converged |
|---|---|---|---|---|---|---|---|---|
| screen_gaussian | 5000 | 100 | 1.7e-08 | 2.2e-07 | 2.8e-05 | 5.9e-07 | 0 | 1 |
| screen_binomial | 5000 | 100 | 1.1e-07 | 1.6e-07 | 1.4e-05 | 6.2e-07 | 0 | 1 |
| screen_poisson | 5000 | 100 | 1.3e-08 | 1.4e-07 | 8.6e-06 | 6.3e-07 | 0 | 1 |
| screen_cox_efron | 5000 | 100 | 1.5e-07 | 2e-07 | 0.00011 | 1.9e-06 | 0 | 1 |
| screen_cox_breslow | 5000 | 100 | 1.7e-07 | 1.8e-07 | 5.8e-05 | 2.2e-06 | 0 | 1 |
