# Chunk-size benchmark 2026-11-26T01:51:19+00:00 (CPU, 2 threads, torch 2.15.0+cu132)

float64, 3 covariates, 5% missing feature values (validation/run_checks.py simulate_screen; cox times are integers 1..~300, cox_untied jitters them apart); peak MB is the RSS high-water mark added by the screen (fresh process per row, data already in memory); heuristic MB is what auto_chunk_size assumes the chunk needs (models/chunk x n x k x 8 bytes x copies; CPU default: budget 256e6 bytes, at most 64 models).

| model | n | m | max ties | chunk | models/chunk | seconds | ms/model | peak MB | heuristic MB |
|---|---|---|---|---|---|---|---|---|---|
| binomial | 5000 | 2000 | 0 | auto | 64 | 0.86 | 0.43 | 96 | 73 |
| binomial | 5000 | 2000 | 0 | 16 | 16 | 1.02 | 0.51 | 47 | 18 |
| binomial | 5000 | 2000 | 0 | 64 | 64 | 0.90 | 0.45 | 95 | 73 |
| binomial | 5000 | 2000 | 0 | 256 | 256 | 1.05 | 0.52 | 239 | 293 |
| binomial | 5000 | 2000 | 0 | 1024 | 1024 | 2.22 | 1.11 | 852 | 1172 |
| binomial | 20000 | 1000 | 0 | auto | 53 | 1.82 | 1.82 | 209 | 243 |
| binomial | 20000 | 1000 | 0 | 16 | 16 | 1.70 | 1.70 | 80 | 73 |
| binomial | 20000 | 1000 | 0 | 64 | 64 | 1.82 | 1.82 | 257 | 293 |
| binomial | 20000 | 1000 | 0 | 256 | 256 | 4.19 | 4.19 | 841 | 1172 |
| binomial | 100000 | 200 | 0 | auto | 10 | 1.85 | 9.23 | 190 | 229 |
| binomial | 100000 | 200 | 0 | 16 | 16 | 2.14 | 10.71 | 329 | 366 |
| binomial | 100000 | 200 | 0 | 64 | 64 | 3.92 | 19.62 | 1106 | 1465 |
| cox | 5000 | 1000 | 221 | auto | 64 | 3.98 | 3.98 | 314 | 243 |
| cox | 5000 | 1000 | 221 | 16 | 16 | 3.23 | 3.23 | 107 | 61 |
| cox | 5000 | 1000 | 221 | 64 | 64 | 4.60 | 4.60 | 317 | 243 |
| cox | 5000 | 1000 | 221 | 256 | 256 | 8.79 | 8.79 | 1019 | 970 |
| cox | 20000 | 400 | 907 | auto | 12 | 8.90 | 22.26 | 312 | 230 |
| cox | 20000 | 400 | 907 | 16 | 16 | 15.20 | 38.00 | 366 | 307 |
| cox | 20000 | 400 | 907 | 64 | 64 | 18.21 | 45.54 | 1093 | 1227 |
| cox | 20000 | 400 | 907 | 256 | 256 | 24.49 | 61.23 | 3973 | 4906 |
| cox | 50000 | 100 | 2261 | auto | 4 | 5.97 | 59.71 | 301 | 211 |
| cox | 50000 | 100 | 2261 | 16 | 16 | 11.79 | 117.91 | 945 | 844 |
| cox | 50000 | 100 | 2261 | 64 | 64 | 15.54 | 155.37 | 2786 | 3378 |
| cox_untied | 5000 | 1000 | 1 | auto | 64 | 1.67 | 1.67 | 123 | 97 |
| cox_untied | 5000 | 1000 | 1 | 16 | 16 | 1.86 | 1.86 | 60 | 24 |
| cox_untied | 5000 | 1000 | 1 | 64 | 64 | 1.71 | 1.71 | 128 | 97 |
| cox_untied | 5000 | 1000 | 1 | 256 | 256 | 2.48 | 2.48 | 367 | 389 |
| cox_untied | 20000 | 400 | 1 | auto | 40 | 3.16 | 7.91 | 259 | 243 |
| cox_untied | 20000 | 400 | 1 | 16 | 16 | 2.68 | 6.71 | 118 | 97 |
| cox_untied | 20000 | 400 | 1 | 64 | 64 | 3.69 | 9.22 | 393 | 389 |
| cox_untied | 20000 | 400 | 1 | 256 | 256 | 7.21 | 18.02 | 1162 | 1556 |
| cox_untied | 50000 | 100 | 1 | auto | 16 | 2.41 | 24.06 | 244 | 243 |
| cox_untied | 50000 | 100 | 1 | 16 | 16 | 2.17 | 21.68 | 246 | 243 |
| cox_untied | 50000 | 100 | 1 | 64 | 64 | 3.75 | 37.51 | 793 | 973 |
