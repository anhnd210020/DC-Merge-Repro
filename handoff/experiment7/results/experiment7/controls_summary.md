# Experiment 7b: controls for the 6b per-depth oracle gain (TEST, Δ vs dc_rank)

Selected on validation with labels; each chosen vector scored once on test. Reference rows (dc_rank, 6b layer oracles, Exp 3 scalar oracle_fair) are the `current` reference.

| context | method | test mean | test worst (task) | Δmean | Δworst | val Δmean | val Δworst |
|---|---|---:|---|---:|---:|---:|---:|
| dtd-eurosat-gtsrb-svhn | dc_rank | 71.20 | 62.20 | +0.00 | +0.00 | 0.00 | 0.00 |
| dtd-eurosat-gtsrb-svhn | depth_only_mean | 71.83 | 56.28 (eurosat) | +0.63 | -5.92 | 0.46 | -7.03 |
| dtd-eurosat-gtsrb-svhn | depth_only_fair | 71.20 | 62.20 (eurosat) | +0.00 | +0.00 | 0.00 | 0.00 |
| dtd-eurosat-gtsrb-svhn | scalar_matched_mean | 74.82 | 52.36 (svhn) | +3.63 | -9.84 | 3.72 | -8.68 |
| dtd-eurosat-gtsrb-svhn | scalar_matched_fair | 74.25 | 63.34 (svhn) | +3.06 | +1.14 | 3.00 | 1.66 |
| dtd-eurosat-gtsrb-svhn | layer_mean | 78.95 | 67.30 | +7.76 | +5.10 | 7.71 | 4.68 |
| dtd-eurosat-gtsrb-svhn | layer_fair | 78.82 | 65.06 | +7.62 | +2.86 | 7.50 | 2.53 |
| dtd-eurosat-gtsrb-svhn | scalar_fair | 74.16 | 61.99 | +2.97 | -0.21 | 2.82 | 0.23 |
| eurosat-gtsrb-mnist-svhn | dc_rank | 73.83 | 55.64 | +0.00 | +0.00 | 0.00 | 0.00 |
| eurosat-gtsrb-mnist-svhn | depth_only_mean | 75.19 | 62.84 (eurosat) | +1.36 | +7.20 | 1.57 | 7.44 |
| eurosat-gtsrb-mnist-svhn | depth_only_fair | 75.19 | 62.84 (eurosat) | +1.36 | +7.20 | 1.57 | 7.44 |
| eurosat-gtsrb-mnist-svhn | scalar_matched_mean | 77.39 | 56.15 (svhn) | +3.57 | +0.52 | 3.97 | 3.77 |
| eurosat-gtsrb-mnist-svhn | scalar_matched_fair | 77.39 | 56.15 (svhn) | +3.57 | +0.52 | 3.97 | 3.77 |
| eurosat-gtsrb-mnist-svhn | layer_mean | 82.80 | 76.20 | +8.97 | +20.56 | 9.34 | 22.36 |
| eurosat-gtsrb-mnist-svhn | layer_fair | 82.69 | 75.30 | +8.86 | +19.67 | 9.29 | 21.37 |
| eurosat-gtsrb-mnist-svhn | scalar_fair | 77.11 | 56.25 | +3.28 | +0.61 | 3.70 | 3.81 |
| eurosat-gtsrb-sun397-svhn | dc_rank | 76.66 | 63.52 | +0.00 | +0.00 | 0.00 | 0.00 |
| eurosat-gtsrb-sun397-svhn | depth_only_mean | 76.75 | 62.61 (eurosat) | +0.09 | -0.91 | 0.35 | -0.94 |
| eurosat-gtsrb-sun397-svhn | depth_only_fair | 76.74 | 68.12 (eurosat) | +0.08 | +4.60 | 0.71 | 5.45 |
| eurosat-gtsrb-sun397-svhn | scalar_matched_mean | 79.66 | 55.01 (svhn) | +3.00 | -8.51 | 3.99 | -6.46 |
| eurosat-gtsrb-sun397-svhn | scalar_matched_fair | 78.68 | 63.03 (svhn) | +2.02 | -0.49 | 3.15 | 1.38 |
| eurosat-gtsrb-sun397-svhn | layer_mean | 84.58 | 70.44 | +7.92 | +6.92 | 8.69 | 8.17 |
| eurosat-gtsrb-sun397-svhn | layer_fair | 84.53 | 69.26 | +7.87 | +5.74 | 8.44 | 6.68 |
| eurosat-gtsrb-sun397-svhn | scalar_fair | 78.68 | 63.45 | +2.02 | -0.07 | 2.96 | 1.85 |
| stanford_cars-eurosat-gtsrb-mnist | dc_rank | 81.60 | 72.20 | +0.00 | +0.00 | 0.00 | 0.00 |
| stanford_cars-eurosat-gtsrb-mnist | depth_only_mean | 82.95 | 75.25 (eurosat) | +1.34 | +3.06 | 1.40 | 4.47 |
| stanford_cars-eurosat-gtsrb-mnist | depth_only_fair | 82.95 | 75.25 (eurosat) | +1.34 | +3.06 | 1.40 | 4.47 |
| stanford_cars-eurosat-gtsrb-mnist | scalar_matched_mean | 83.90 | 76.96 (stanford_cars) | +2.29 | +4.77 | 3.23 | 7.38 |
| stanford_cars-eurosat-gtsrb-mnist | scalar_matched_fair | 83.90 | 76.96 (stanford_cars) | +2.29 | +4.77 | 3.23 | 7.38 |
| stanford_cars-eurosat-gtsrb-mnist | layer_mean | 86.14 | 84.63 | +4.54 | +12.44 | 5.58 | 14.56 |
| stanford_cars-eurosat-gtsrb-mnist | layer_fair | 85.63 | 79.09 | +4.03 | +6.90 | 5.13 | 9.41 |
| stanford_cars-eurosat-gtsrb-mnist | scalar_fair | 84.10 | 77.92 | +2.50 | +5.72 | 3.28 | 8.11 |
| stanford_cars-eurosat-gtsrb-resisc45 | dc_rank | 85.51 | 80.53 | +0.00 | +0.00 | 0.00 | 0.00 |
| stanford_cars-eurosat-gtsrb-resisc45 | depth_only_mean | 87.36 | 84.21 (stanford_cars) | +1.85 | +3.67 | 2.06 | 5.03 |
| stanford_cars-eurosat-gtsrb-resisc45 | depth_only_fair | 87.36 | 84.21 (stanford_cars) | +1.85 | +3.67 | 2.06 | 5.03 |
| stanford_cars-eurosat-gtsrb-resisc45 | scalar_matched_mean | 85.39 | 77.02 (stanford_cars) | -0.12 | -3.51 | 0.48 | -2.35 |
| stanford_cars-eurosat-gtsrb-resisc45 | scalar_matched_fair | 85.54 | 80.41 (stanford_cars) | +0.03 | -0.12 | 0.42 | 0.81 |
| stanford_cars-eurosat-gtsrb-resisc45 | layer_mean | 87.82 | 83.29 | +2.32 | +2.76 | 3.03 | 4.95 |
| stanford_cars-eurosat-gtsrb-resisc45 | layer_fair | 88.11 | 81.43 | +2.60 | +0.89 | 2.97 | 2.43 |
| stanford_cars-eurosat-gtsrb-resisc45 | scalar_fair | 85.51 | 80.53 | +0.00 | +0.00 | 0.00 | 0.00 |
| stanford_cars-eurosat-mnist-resisc45 | dc_rank | 87.61 | 85.77 | +0.00 | +0.00 | 0.00 | 0.00 |
| stanford_cars-eurosat-mnist-resisc45 | depth_only_mean | 88.42 | 84.23 (eurosat) | +0.81 | -1.55 | 0.84 | -1.54 |
| stanford_cars-eurosat-mnist-resisc45 | depth_only_fair | 87.61 | 85.77 (eurosat) | +0.00 | +0.00 | 0.00 | 0.00 |
| stanford_cars-eurosat-mnist-resisc45 | scalar_matched_mean | 87.31 | 72.80 (stanford_cars) | -0.30 | -12.97 | 0.87 | -9.81 |
| stanford_cars-eurosat-mnist-resisc45 | scalar_matched_fair | 87.61 | 85.77 (eurosat) | +0.00 | +0.00 | 0.00 | 0.00 |
| stanford_cars-eurosat-mnist-resisc45 | layer_mean | 89.47 | 81.37 | +1.86 | -4.41 | 2.29 | -3.89 |
| stanford_cars-eurosat-mnist-resisc45 | layer_fair | 87.88 | 85.21 | +0.27 | -0.57 | 0.58 | 0.15 |
| stanford_cars-eurosat-mnist-resisc45 | scalar_fair | 87.61 | 85.77 | +0.00 | +0.00 | 0.00 | 0.00 |
| stanford_cars-eurosat-mnist-sun397 | dc_rank | 89.57 | 84.67 | +0.00 | +0.00 | 0.00 | 0.00 |
| stanford_cars-eurosat-mnist-sun397 | depth_only_mean | 90.89 | 86.16 (stanford_cars) | +1.32 | +1.48 | 1.43 | 1.62 |
| stanford_cars-eurosat-mnist-sun397 | depth_only_fair | 90.89 | 86.16 (stanford_cars) | +1.32 | +1.48 | 1.43 | 1.62 |
| stanford_cars-eurosat-mnist-sun397 | scalar_matched_mean | 90.49 | 82.00 (stanford_cars) | +0.92 | -2.68 | 1.45 | -1.62 |
| stanford_cars-eurosat-mnist-sun397 | scalar_matched_fair | 90.38 | 84.51 (stanford_cars) | +0.80 | -0.16 | 0.82 | 0.49 |
| stanford_cars-eurosat-mnist-sun397 | layer_mean | 91.46 | 85.61 | +1.89 | +0.93 | 2.60 | 1.87 |
| stanford_cars-eurosat-mnist-sun397 | layer_fair | 91.34 | 83.78 | +1.76 | -0.89 | 2.22 | 0.49 |
| stanford_cars-eurosat-mnist-sun397 | scalar_fair | 90.31 | 86.72 | +0.74 | +2.05 | 0.98 | 2.76 |

## Averages over 3 contexts (core; test, Δ vs dc_rank)

| method | Δmean | Δworst |
|---|---:|---:|
| depth_only_mean | +0.70 | +0.13 |
| depth_only_fair | +0.48 | +3.94 |
| scalar_matched_mean | +3.40 | -5.95 |
| scalar_matched_fair | +2.88 | +0.39 |
| layer_mean | +8.22 | +10.86 |
| layer_fair | +8.12 | +9.42 |
| scalar_fair | +2.75 | +0.11 |

**Verdict (core, pre-registered): TASK-SPECIFIC DEPTH**

Best of the two objectives per arm (larger Δworst, ties by Δmean): 6b = `layer_mean` (+8.22 / +10.86); depth_only = `depth_only_fair` (+0.48 / +3.94); scalar_matched = `scalar_matched_fair` (+2.88 / +0.39)

## Averages over 7 contexts (all; test, Δ vs dc_rank)

| method | Δmean | Δworst |
|---|---:|---:|
| depth_only_mean | +1.06 | +1.01 |
| depth_only_fair | +0.85 | +2.86 |
| scalar_matched_mean | +1.85 | -4.60 |
| scalar_matched_fair | +1.68 | +0.81 |
| layer_mean | +5.04 | +6.33 |
| layer_fair | +4.72 | +4.94 |
| scalar_fair | +1.64 | +1.16 |

## Chosen vectors

| context | arm | coordinates |
|---|---|---|
| dtd-eurosat-gtsrb-svhn | depth_only_mean | 0.4,0.6,0.5 |
| dtd-eurosat-gtsrb-svhn | depth_only_fair | 0.5,0.5,0.5 |
| dtd-eurosat-gtsrb-svhn | scalar_matched_mean | 0.5,0.8,0.7,0.35 |
| dtd-eurosat-gtsrb-svhn | scalar_matched_fair | 0.35,0.75,0.65,0.5 |
| dtd-eurosat-gtsrb-svhn | scalar_matched budget | 341/341 distinct val evals, 2 starts |
| eurosat-gtsrb-mnist-svhn | depth_only_mean | 0.4,0.5,0.7 |
| eurosat-gtsrb-mnist-svhn | depth_only_fair | 0.4,0.5,0.7 |
| eurosat-gtsrb-mnist-svhn | scalar_matched_mean | 1,0.75,0.4,0.5 |
| eurosat-gtsrb-mnist-svhn | scalar_matched_fair | 1,0.75,0.4,0.5 |
| eurosat-gtsrb-mnist-svhn | scalar_matched budget | 360/360 distinct val evals, 2 starts |
| eurosat-gtsrb-sun397-svhn | depth_only_mean | 0.5,0.6,0.6 |
| eurosat-gtsrb-sun397-svhn | depth_only_fair | 0.4,0.5,0.7 |
| eurosat-gtsrb-sun397-svhn | scalar_matched_mean | 0.95,0.75,0.55,0.45 |
| eurosat-gtsrb-sun397-svhn | scalar_matched_fair | 1,0.75,0.35,0.6 |
| eurosat-gtsrb-sun397-svhn | scalar_matched budget | 360/360 distinct val evals, 2 starts |
| stanford_cars-eurosat-gtsrb-mnist | depth_only_mean | 0.4,0.4,0.7 |
| stanford_cars-eurosat-gtsrb-mnist | depth_only_fair | 0.4,0.4,0.7 |
| stanford_cars-eurosat-gtsrb-mnist | scalar_matched_mean | 0.25,0.85,0.7,0.5 |
| stanford_cars-eurosat-gtsrb-mnist | scalar_matched_fair | 0.25,0.85,0.7,0.5 |
| stanford_cars-eurosat-gtsrb-mnist | scalar_matched budget | 296/296 distinct val evals, 2 starts |
| stanford_cars-eurosat-gtsrb-resisc45 | depth_only_mean | 0.6,0.5,0.9 |
| stanford_cars-eurosat-gtsrb-resisc45 | depth_only_fair | 0.6,0.5,0.9 |
| stanford_cars-eurosat-gtsrb-resisc45 | scalar_matched_mean | 0.45,0.7,0.7,0.6 |
| stanford_cars-eurosat-gtsrb-resisc45 | scalar_matched_fair | 0.6,0.7,0.7,0.6 |
| stanford_cars-eurosat-gtsrb-resisc45 | scalar_matched budget | 330/330 distinct val evals, 2 starts |
| stanford_cars-eurosat-mnist-resisc45 | depth_only_mean | 0.4,0.6,0.7 |
| stanford_cars-eurosat-mnist-resisc45 | depth_only_fair | 0.6,0.6,0.6 |
| stanford_cars-eurosat-mnist-resisc45 | scalar_matched_mean | 0.2,0.65,0.7,0.7 |
| stanford_cars-eurosat-mnist-resisc45 | scalar_matched_fair | 0.6,0.6,0.6,0.6 |
| stanford_cars-eurosat-mnist-resisc45 | scalar_matched budget | 330/330 distinct val evals, 2 starts |
| stanford_cars-eurosat-mnist-sun397 | depth_only_mean | 0.7,0.5,0.8 |
| stanford_cars-eurosat-mnist-sun397 | depth_only_fair | 0.7,0.5,0.8 |
| stanford_cars-eurosat-mnist-sun397 | scalar_matched_mean | 0.55,0.85,0.8,0.55 |
| stanford_cars-eurosat-mnist-sun397 | scalar_matched_fair | 0.7,0.85,0.8,0.55 |
| stanford_cars-eurosat-mnist-sun397 | scalar_matched budget | 317/317 distinct val evals, 1 starts |
