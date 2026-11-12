# Validation 2026-11-12T01:45:23+00:00 (cpu)

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
| lasso_gaussian_vs_sklearn | 1000 | 50 | yes |
| lasso_gaussian_float32_vs_float64 | 1000 | 50 | yes |
| lasso_binomial_vs_sklearn | 1000 | 20 | yes |
| lasso_cox_signal_recovery | 2000 | 50 | yes |

## float32 vs float64 (informational, not gated)

-log10 p compared where p >= float32 tiny; smaller p counted as underflow.

| fit | n | m | beta max abs | se max rel | -log10 p max abs | p underflow | converged |
|---|---|---|---|---|---|---|---|
| screen_gaussian | 5000 | 100 | 1.7e-08 | 2.2e-07 | 5.7e-07 | 2 | 1 |
| screen_binomial | 5000 | 100 | 1.1e-07 | 1.6e-07 | 1.1e-05 | 1 | 1 |
| screen_poisson | 5000 | 100 | 1.3e-08 | 1.4e-07 | 3.2e-06 | 1 | 1 |
| screen_cox_efron | 5000 | 100 | 1.5e-07 | 2e-07 | 1.9e-06 | 2 | 1 |
| screen_cox_breslow | 5000 | 100 | 1.7e-07 | 1.8e-07 | 2.1e-06 | 2 | 1 |
