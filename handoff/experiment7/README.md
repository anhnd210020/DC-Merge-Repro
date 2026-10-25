# Experiment 7: controls, label-free method and wide grid for per-depth lane coefficients

Code: `code/` (run_exp7.py, launcher, implementation guide, CPU smoke test, verify_threads7.py).
Results: `results/experiment7/` (summaries *.md/*.csv/*.json and per-context result/cache files),
`results/logs/` (run logs, progress bars removed), `results/ops/` (scheduler and status script).

| Part | Verdict |
|---|---|
| 7a gate | IN_BETWEEN (7c/7e core only, capped at PARTIAL) |
| 7b controls | TASK-SPECIFIC DEPTH (core). 6b +8.22/+10.86 vs depth_only +0.48/+3.94 and scalar_matched +2.88/+0.39 (test, core avg). Advantage small in C5-C7 (stanford_cars, high baseline). |
| 7c label-free | PARTIAL: agree_floor +2.66/+5.61 (core), ~1/3 of 6b layer_fair, no context loses >1 pp worst |
| 7d wide grid | descriptive: +1.74/+1.44 extra over 6b on test; svhn late layers -> 0 in all chosen vectors |
| 7e entropy | NOT VIABLE (-6.01/-10.19 entropy_mean) |
| all8 | not run (needs 7c VIABLE or STRONG) |

Must-hold: replay 0.000 pp vs archive; checks 1-5 PASS in all 7 contexts; each chosen vector scored once on test.
Run note: default 32 CPU threads (verified identical to the cache); 1 thread changes validation accuracy by up to
0.24 pp, see results/logs/verify_*.log. The 26 tracebacks in the logs are CUDA OOM crashes from an earlier
launch layout; every job was resumed from its cache and completed.
