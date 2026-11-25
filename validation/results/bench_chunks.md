# Chunk-size benchmark 2026-11-25T01:50:58+00:00 (CPU, 2 threads, torch 2.15.0+cu132)

float64, 3 covariates, 5% missing feature values (validation/run_checks.py simulate_screen; cox times are integers 1..~300, cox_untied jitters them apart); peak MB is the RSS high-water mark added by the screen (fresh process per row, data already in memory); heuristic MB is what auto_chunk_size assumes the chunk needs (models/chunk x n x k x 8 bytes x copies; CPU default: budget 256e6 bytes, at most 64 models).

| model | n | m | max ties | chunk | models/chunk | seconds | ms/model | peak MB | heuristic MB |
|---|---|---|---|---|---|---|---|---|---|
| binomial | 5000 | 2000 | 0 | auto | 64 | 0.97 | 0.48 | 93 | 73 |
| binomial | 5000 | 2000 | 0 | 16 | 16 | 1.14 | 0.57 | 50 | 18 |
| binomial | 5000 | 2000 | 0 | 64 | 64 | 1.04 | 0.52 | 91 | 73 |
| binomial | 5000 | 2000 | 0 | 256 | 256 | 1.49 | 0.75 | 288 | 293 |
| binomial | 5000 | 2000 | 0 | 1024 | 1024 | 2.56 | 1.28 | 852 | 1172 |
| binomial | 20000 | 1000 | 0 | auto | 53 | 2.41 | 2.41 | 233 | 243 |
| binomial | 20000 | 1000 | 0 | 16 | 16 | 2.04 | 2.04 | 82 | 73 |
| binomial | 20000 | 1000 | 0 | 64 | 64 | 2.46 | 2.46 | 249 | 293 |
| binomial | 20000 | 1000 | 0 | 256 | 256 | 5.16 | 5.16 | 855 | 1172 |
| binomial | 100000 | 200 | 0 | auto | 10 | 2.45 | 12.23 | 214 | 229 |
| binomial | 100000 | 200 | 0 | 16 | 16 | 2.79 | 13.97 | 343 | 366 |
| binomial | 100000 | 200 | 0 | 64 | 64 | 5.22 | 26.09 | 1049 | 1465 |
| cox | 5000 | 1000 | 221 | auto | 3 | 4.88 | 4.88 | 54 | 206 |
| cox | 5000 | 1000 | 221 | 16 | 16 | 3.44 | 3.44 | 109 | 1099 |
| cox | 5000 | 1000 | 221 | 64 | 64 | 4.76 | 4.76 | 314 | 4395 |
| cox | 5000 | 1000 | 221 | 256 | 256 | 9.10 | 9.10 | 970 | 17578 |
| cox | 20000 | 400 | 907 | auto | 1 | 10.83 | 27.07 | 64 | 1112 |
| cox | 20000 | 400 | 907 | 16 | 16 | 16.15 | 40.38 | 366 | 17793 |
| cox | 20000 | 400 | 907 | 64 | 64 | 19.10 | 47.76 | 1102 | 71172 |
| cox | 20000 | 400 | 907 | 256 | 256 | 26.86 | 67.14 | 3895 | 284688 |
| cox | 50000 | 100 | 2261 | auto | 1 | 6.28 | 62.75 | 100 | 6912 |
| cox | 50000 | 100 | 2261 | 16 | 16 | 12.35 | 123.50 | 945 | 110596 |
| cox | 50000 | 100 | 2261 | 64 | 64 | 17.27 | 172.70 | 2737 | 442383 |
| cox_untied | 5000 | 1000 | 1 | auto | 64 | 1.89 | 1.89 | 128 | 98 |
| cox_untied | 5000 | 1000 | 1 | 16 | 16 | 1.97 | 1.97 | 62 | 24 |
| cox_untied | 5000 | 1000 | 1 | 64 | 64 | 2.03 | 2.03 | 128 | 98 |
| cox_untied | 5000 | 1000 | 1 | 256 | 256 | 2.82 | 2.82 | 374 | 391 |
| cox_untied | 20000 | 400 | 1 | auto | 40 | 4.22 | 10.55 | 259 | 244 |
| cox_untied | 20000 | 400 | 1 | 16 | 16 | 3.29 | 8.22 | 118 | 98 |
| cox_untied | 20000 | 400 | 1 | 64 | 64 | 4.23 | 10.59 | 366 | 391 |
| cox_untied | 20000 | 400 | 1 | 256 | 256 | 7.46 | 18.64 | 1162 | 1562 |
| cox_untied | 50000 | 100 | 1 | auto | 16 | 2.34 | 23.41 | 244 | 244 |
| cox_untied | 50000 | 100 | 1 | 16 | 16 | 2.19 | 21.85 | 246 | 244 |
| cox_untied | 50000 | 100 | 1 | 64 | 64 | 4.34 | 43.45 | 799 | 977 |
