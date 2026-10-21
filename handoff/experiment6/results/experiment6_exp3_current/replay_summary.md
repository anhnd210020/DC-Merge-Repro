# Experiment 3 current-environment replay (for Experiment 6)

Archive: `/workspace/ducanh/repo-exp5/handoff/experiment3/results/experiment3/contexts`. Experiment 3's chosen vectors re-evaluated on this machine with Experiment 3's code path; nothing re-selected. Differences are this machine minus archive, in accuracy pp.

| context | vector | max |val diff| | max |test diff| | test mean norm: archive → replay | test worst norm: archive → replay |
|---|---|---:|---:|---|---|
| dtd-eurosat-gtsrb-svhn | dc_rank | 0.079 | 0.053 | 71.19 → 71.20 | 62.20 → 62.20 |
| dtd-eurosat-gtsrb-svhn | oracle_mean | 0.106 | 0.053 | 73.85 → 73.84 | 57.99 → 58.01 |
| dtd-eurosat-gtsrb-svhn | oracle_fair | 0.148 | 0.053 | 74.14 → 74.16 | 62.01 → 61.99 |
| eurosat-gtsrb-mnist-svhn | dc_rank | 0.119 | 0.188 | 73.82 → 73.83 | 55.45 → 55.64 |
| eurosat-gtsrb-mnist-svhn | oracle_mean | 0.198 | 0.075 | 77.11 → 77.11 | 56.21 → 56.25 |
| eurosat-gtsrb-mnist-svhn | oracle_fair | 0.198 | 0.075 | 77.11 → 77.11 | 56.21 → 56.25 |
| eurosat-gtsrb-sun397-svhn | dc_rank | 0.154 | 0.074 | 76.68 → 76.66 | 63.60 → 63.52 |
| eurosat-gtsrb-sun397-svhn | oracle_mean | 0.158 | 0.053 | 79.51 → 79.51 | 57.84 → 57.78 |
| eurosat-gtsrb-sun397-svhn | oracle_fair | 0.148 | 0.185 | 78.61 → 78.68 | 63.42 → 63.45 |
| stanford_cars-eurosat-gtsrb-mnist | dc_rank | 0.158 | 0.100 | 81.59 → 81.60 | 72.16 → 72.20 |
| stanford_cars-eurosat-gtsrb-mnist | oracle_mean | 0.187 | 0.087 | 84.09 → 84.10 | 77.92 → 77.92 |
| stanford_cars-eurosat-gtsrb-mnist | oracle_fair | 0.187 | 0.087 | 84.09 → 84.10 | 77.92 → 77.92 |
| stanford_cars-eurosat-gtsrb-resisc45 | dc_rank | 0.062 | 0.074 | 85.50 → 85.51 | 80.58 → 80.53 |
| stanford_cars-eurosat-gtsrb-resisc45 | oracle_mean | 0.079 | 0.032 | 85.54 → 85.54 | 80.45 → 80.41 |
| stanford_cars-eurosat-gtsrb-resisc45 | oracle_fair | 0.062 | 0.074 | 85.50 → 85.51 | 80.58 → 80.53 |
| stanford_cars-eurosat-mnist-resisc45 | dc_rank | 0.124 | 0.148 | 87.68 → 87.61 | 85.93 → 85.77 |
| stanford_cars-eurosat-mnist-resisc45 | oracle_mean | 0.185 | 0.037 | 87.69 → 87.70 | 75.08 → 75.06 |
| stanford_cars-eurosat-mnist-resisc45 | oracle_fair | 0.124 | 0.148 | 87.68 → 87.61 | 85.93 → 85.77 |
| stanford_cars-eurosat-mnist-sun397 | dc_rank | 0.100 | 0.138 | 89.61 → 89.57 | 84.76 → 84.67 |
| stanford_cars-eurosat-mnist-sun397 | oracle_mean | 0.185 | 0.062 | 90.52 → 90.55 | 82.95 → 83.01 |
| stanford_cars-eurosat-mnist-sun397 | oracle_fair | 0.111 | 0.111 | 90.32 → 90.31 | 86.70 → 86.72 |

Determinism (same evaluation twice on this machine): max diff 0.000 pp.
