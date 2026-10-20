# Experiment 5 signal check and gate decision

The uploaded package was extracted separately at `/workspace/experiment5_package_20261019/DC-Merge`. The Experiment 3 `lanes.py` arithmetic fix and patched `run_oracle.py` were copied into this extraction only; their SHA-256 hashes match `/workspace/DC-Merge-Repro/handoff/experiment3/code/lane_harm/`. The working repository and Experiment 2–4 result directories were not used as output destinations.

The guide's step 1 completed for all three core contexts in tmux session `exp5-check-20261019`. It evaluated the 78, 57, and 82 vectors already in the Experiment 3 validation caches. Entropy used seed 0 and 500 unlabeled validation images per task; the exact selected counts, split sizes, and leading indices are recorded in `check_metadata.json`. The check's accuracy correlations use Experiment 3's cached validation accuracies as diagnostics.

| Context | ρ(−mean H, mean accuracy) | ρ(−max H, worst accuracy) | Mean-entropy pick vs. dc_rank (validation mean, pp) | Fair entropy pick vs. dc_rank (validation worst, pp) |
|---|---:|---:|---:|---:|
| dtd-eurosat-gtsrb-svhn | 0.35 | 0.09 | 68.72 vs. 71.16 (−2.44) | 54.12 vs. 62.08 (−7.96) |
| eurosat-gtsrb-mnist-svhn | 0.43 | 0.12 | 68.89 vs. 73.74 (−4.85) | 48.33 vs. 53.70 (−5.37) |
| eurosat-gtsrb-sun397-svhn | 0.01 | −0.28 | 70.77 vs. 75.83 (−5.06) | 47.64 vs. 62.04 (−14.40) |

## Decision

**STOP at the signal-check gate.** The entropy-selected mean vector scores below `dc_rank` on validation in all three core contexts, and the fair correlation is negative in the third. Experiment 5's guide says to stop when the entropy pick is worse than `dc_rank` or a correlation is nonpositive. Therefore no coefficient search, no chosen entropy vectors, and no test evaluations were run; the four-context phase was not started. This preserves the guide's gate rather than treating the failed signal as a reason to continue.

`check_summary.md`, `check.json`, the completed per-context entropy caches, and `check.log` are the final Experiment 5 outputs for this gated run. There is intentionally no `entropy_summary.md`, `entropy.csv`, or per-context `entropy_result.json`, because the guide's stop condition was met before step 2.
