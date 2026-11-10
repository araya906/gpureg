# Validation 2026-11-10T01:48:07+00:00 (cpu)

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
| lasso_gaussian_vs_sklearn | 1000 | 50 | yes |
| lasso_binomial_vs_sklearn | 1000 | 20 | yes |
| lasso_cox_signal_recovery | 2000 | 50 | yes |

## float32 vs float64 (informational, not gated)

-log10 p compared where p >= float32 tiny; smaller p counted as underflow.

| fit | n | m | beta max abs | se max rel | -log10 p max abs | p underflow | converged |
|---|---|---|---|---|---|---|---|
| screen_gaussian | 5000 | 100 | 1.7e-08 | 2.2e-07 | 5e-07 | 2 | 1 |
| screen_binomial | 5000 | 100 | 2.9e-05 | 4.2e-05 | 0.00074 | 1 | 0.98 |
| screen_poisson | 5000 | 100 | 4.1e-06 | 5.3e-06 | 0.0017 | 1 | 1 |
| screen_cox_efron | 5000 | 100 | 1.6e-06 | 2.2e-07 | 0.0001 | 2 | 1 |
| screen_cox_breslow | 5000 | 100 | 1.2e-05 | 7.3e-07 | 0.00012 | 2 | 1 |
| lasso_gaussian_path | 1000 | 50 | 0.6 | - | - | - | - |
