# Experiment 6b-2: per-(task, layer-group) oracle (TEST), grouping = depth

Coefficients searched on validation (greedy, 2 passes, grid 0.3–1.0 plus α); each vector evaluated once on test. `scalar_*` are Experiment 3's per-task oracles (copied). Shrinkage = validation gain minus test gain, both vs dc_rank.

## Per context (test)

| context | method | mean norm | worst norm (task) | Δmean | Δworst | val Δmean | val Δworst | shrink mean | shrink worst |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| dtd-eurosat-gtsrb-svhn | dc_rank | 71.20 | 62.20 (eurosat) | +0.00 | +0.00 |  |  |  |  |
| dtd-eurosat-gtsrb-svhn | layer_mean | 78.95 | 67.30 (svhn) | +7.76 | +5.10 | +7.71 | +4.68 | -0.05 | -0.42 |
| dtd-eurosat-gtsrb-svhn | layer_fair | 78.82 | 65.06 (svhn) | +7.62 | +2.86 | +7.50 | +2.53 | -0.12 | -0.33 |
| dtd-eurosat-gtsrb-svhn | scalar_mean | 73.84 | 58.01 (svhn) | +2.64 | -4.19 | +3.04 | -3.67 | +0.39 | +0.52 |
| dtd-eurosat-gtsrb-svhn | scalar_fair | 74.16 | 61.99 (svhn) | +2.97 | -0.21 | +2.82 | +0.23 | -0.15 | +0.44 |
| dtd-eurosat-gtsrb-svhn | ct | 69.41 | 55.28 (svhn) | -1.78 | -6.92 |  |  |  |  |
| eurosat-gtsrb-mnist-svhn | dc_rank | 73.83 | 55.64 (eurosat) | +0.00 | +0.00 |  |  |  |  |
| eurosat-gtsrb-mnist-svhn | layer_mean | 82.80 | 76.20 (svhn) | +8.97 | +20.56 | +9.34 | +22.36 | +0.37 | +1.80 |
| eurosat-gtsrb-mnist-svhn | layer_fair | 82.69 | 75.30 (svhn) | +8.86 | +19.67 | +9.29 | +21.37 | +0.43 | +1.70 |
| eurosat-gtsrb-mnist-svhn | scalar_mean | 77.11 | 56.25 (svhn) | +3.28 | +0.61 | +3.70 | +3.81 | +0.42 | +3.20 |
| eurosat-gtsrb-mnist-svhn | scalar_fair | 77.11 | 56.25 (svhn) | +3.28 | +0.61 | +3.70 | +3.81 | +0.42 | +3.20 |
| eurosat-gtsrb-mnist-svhn | ct | 70.24 | 53.34 (svhn) | -3.58 | -2.30 |  |  |  |  |
| eurosat-gtsrb-sun397-svhn | dc_rank | 76.66 | 63.52 (eurosat) | +0.00 | +0.00 |  |  |  |  |
| eurosat-gtsrb-sun397-svhn | layer_mean | 84.58 | 70.44 (svhn) | +7.92 | +6.92 | +8.69 | +8.17 | +0.77 | +1.26 |
| eurosat-gtsrb-sun397-svhn | layer_fair | 84.53 | 69.26 (svhn) | +7.87 | +5.74 | +8.44 | +6.68 | +0.57 | +0.94 |
| eurosat-gtsrb-sun397-svhn | scalar_mean | 79.51 | 57.78 (svhn) | +2.86 | -5.74 | +3.57 | -4.07 | +0.71 | +1.66 |
| eurosat-gtsrb-sun397-svhn | scalar_fair | 78.68 | 63.45 (svhn) | +2.02 | -0.07 | +2.96 | +1.85 | +0.94 | +1.92 |
| eurosat-gtsrb-sun397-svhn | ct | 73.44 | 61.09 (svhn) | -3.21 | -2.43 |  |  |  |  |
| stanford_cars-eurosat-gtsrb-mnist | dc_rank | 81.60 | 72.20 (eurosat) | +0.00 | +0.00 |  |  |  |  |
| stanford_cars-eurosat-gtsrb-mnist | layer_mean | 86.14 | 84.63 (stanford_cars) | +4.54 | +12.44 | +5.58 | +14.56 | +1.04 | +2.12 |
| stanford_cars-eurosat-gtsrb-mnist | layer_fair | 85.63 | 79.09 (stanford_cars) | +4.03 | +6.90 | +5.13 | +9.41 | +1.10 | +2.51 |
| stanford_cars-eurosat-gtsrb-mnist | scalar_mean | 84.10 | 77.92 (stanford_cars) | +2.50 | +5.72 | +3.28 | +8.11 | +0.78 | +2.39 |
| stanford_cars-eurosat-gtsrb-mnist | scalar_fair | 84.10 | 77.92 (stanford_cars) | +2.50 | +5.72 | +3.28 | +8.11 | +0.78 | +2.39 |
| stanford_cars-eurosat-gtsrb-mnist | ct | 80.27 | 67.90 (gtsrb) | -1.33 | -4.30 |  |  |  |  |
| stanford_cars-eurosat-gtsrb-resisc45 | dc_rank | 85.51 | 80.53 (stanford_cars) | +0.00 | +0.00 |  |  |  |  |
| stanford_cars-eurosat-gtsrb-resisc45 | layer_mean | 87.82 | 83.29 (stanford_cars) | +2.32 | +2.76 | +3.03 | +4.95 | +0.71 | +2.19 |
| stanford_cars-eurosat-gtsrb-resisc45 | layer_fair | 88.11 | 81.43 (stanford_cars) | +2.60 | +0.89 | +2.97 | +2.43 | +0.37 | +1.54 |
| stanford_cars-eurosat-gtsrb-resisc45 | scalar_mean | 85.54 | 80.41 (stanford_cars) | +0.03 | -0.12 | +0.42 | +0.81 | +0.39 | +0.93 |
| stanford_cars-eurosat-gtsrb-resisc45 | scalar_fair | 85.51 | 80.53 (stanford_cars) | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| stanford_cars-eurosat-gtsrb-resisc45 | ct | 82.01 | 79.70 (gtsrb) | -3.49 | -0.84 |  |  |  |  |
| stanford_cars-eurosat-mnist-resisc45 | dc_rank | 87.61 | 85.77 (eurosat) | +0.00 | +0.00 |  |  |  |  |
| stanford_cars-eurosat-mnist-resisc45 | layer_mean | 89.47 | 81.37 (stanford_cars) | +1.86 | -4.41 | +2.29 | -3.89 | +0.43 | +0.52 |
| stanford_cars-eurosat-mnist-resisc45 | layer_fair | 87.88 | 85.21 (eurosat) | +0.27 | -0.57 | +0.58 | +0.15 | +0.31 | +0.72 |
| stanford_cars-eurosat-mnist-resisc45 | scalar_mean | 87.70 | 75.06 (stanford_cars) | +0.09 | -10.72 | +0.77 | -8.92 | +0.68 | +1.80 |
| stanford_cars-eurosat-mnist-resisc45 | scalar_fair | 87.61 | 85.77 (eurosat) | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| stanford_cars-eurosat-mnist-resisc45 | ct | 86.05 | 84.55 (stanford_cars) | -1.56 | -1.22 |  |  |  |  |
| stanford_cars-eurosat-mnist-sun397 | dc_rank | 89.57 | 84.67 (stanford_cars) | +0.00 | +0.00 |  |  |  |  |
| stanford_cars-eurosat-mnist-sun397 | layer_mean | 91.46 | 85.61 (stanford_cars) | +1.89 | +0.93 | +2.60 | +1.87 | +0.71 | +0.93 |
| stanford_cars-eurosat-mnist-sun397 | layer_fair | 91.34 | 83.78 (stanford_cars) | +1.76 | -0.89 | +2.22 | +0.49 | +0.46 | +1.38 |
| stanford_cars-eurosat-mnist-sun397 | scalar_mean | 90.55 | 83.01 (stanford_cars) | +0.98 | -1.66 | +1.26 | -0.57 | +0.28 | +1.10 |
| stanford_cars-eurosat-mnist-sun397 | scalar_fair | 90.31 | 86.72 (stanford_cars) | +0.74 | +2.05 | +0.98 | +2.76 | +0.25 | +0.71 |
| stanford_cars-eurosat-mnist-sun397 | ct | 89.44 | 83.09 (stanford_cars) | -0.13 | -1.58 |  |  |  |  |

## Averages over 3 contexts (core; test, Δ vs dc_rank)

| method | Δmean | Δworst | Δworst − scalar_fair Δworst | mean shrink | worst shrink | verdict |
|---|---:|---:|---:|---:|---:|---|
| layer_mean | +8.22 | +10.86 | +10.75 | +0.36 | +0.88 | LAYER LEVER: predictor of c_{i,g} and a layer-aware method |
| layer_fair | +8.12 | +9.42 | +9.31 | +0.29 | +0.77 | LAYER LEVER: predictor of c_{i,g} and a layer-aware method |
| scalar_mean | +2.93 | -3.10 | -3.21 | +0.51 | +1.79 |  |
| scalar_fair | +2.75 | +0.11 | +0.00 | +0.41 | +1.85 |  |
| ct | -2.86 | -3.88 | -3.99 | — | — |  |

**Verdict (core, best of the two layer oracles = `layer_mean`): LAYER LEVER** — predictor of c_{i,g} and a layer-aware method.

Per context, does the layer oracle beat the scalar `oracle_fair` on test?

| context | best layer oracle | Δmean vs scalar_fair | Δworst vs scalar_fair |
|---|---|---:|---:|
| dtd-eurosat-gtsrb-svhn | layer_mean | +4.79 | +5.31 |
| eurosat-gtsrb-mnist-svhn | layer_mean | +5.69 | +19.95 |
| eurosat-gtsrb-sun397-svhn | layer_mean | +5.90 | +6.99 |

## Averages over 7 contexts (all; test, Δ vs dc_rank)

| method | Δmean | Δworst | Δworst − scalar_fair Δworst | mean shrink | worst shrink | verdict |
|---|---:|---:|---:|---:|---:|---|
| layer_mean | +5.04 | +6.33 | +5.17 | +0.57 | +1.20 | LAYER LEVER: predictor of c_{i,g} and a layer-aware method |
| layer_fair | +4.72 | +4.94 | +3.79 | +0.45 | +1.21 | LAYER LEVER: predictor of c_{i,g} and a layer-aware method |
| scalar_mean | +1.77 | -2.30 | -3.46 | +0.52 | +1.66 |  |
| scalar_fair | +1.64 | +1.16 | +0.00 | +0.32 | +1.24 |  |
| ct | -2.16 | -2.80 | -3.96 | — | — |  |

**Verdict (all, best of the two layer oracles = `layer_mean`): LAYER LEVER** — predictor of c_{i,g} and a layer-aware method.

Per context, does the layer oracle beat the scalar `oracle_fair` on test?

| context | best layer oracle | Δmean vs scalar_fair | Δworst vs scalar_fair |
|---|---|---:|---:|
| dtd-eurosat-gtsrb-svhn | layer_mean | +4.79 | +5.31 |
| eurosat-gtsrb-mnist-svhn | layer_mean | +5.69 | +19.95 |
| eurosat-gtsrb-sun397-svhn | layer_mean | +5.90 | +6.99 |
| stanford_cars-eurosat-gtsrb-mnist | layer_mean | +2.04 | +6.72 |
| stanford_cars-eurosat-gtsrb-resisc45 | layer_mean | +2.32 | +2.76 |
| stanford_cars-eurosat-mnist-resisc45 | layer_mean | +1.86 | -4.41 |
| stanford_cars-eurosat-mnist-sun397 | layer_mean | +1.15 | -1.12 |

Decision rule (pre-registered): LAYER LEVER if Δworst ≥ +3 pp AND Δworst exceeds scalar oracle_fair's Δworst by ≥ +2 pp AND Δmean ≥ +1 pp; NO LEVER if Δworst is within 1 pp of the scalar oracle's (or below it); otherwise SMALL GAIN. "Best of the two" is chosen on test among two oracles, so it is an optimistic upper bound.

## Chosen coefficient tables (validation-selected)

**dtd-eurosat-gtsrb-svhn — layer_mean** (α = 0.5)

| task | early | mid | late | scalar oracle_fair |
|---|---:|---:|---:|---:|
| dtd | 0.3 | 0.6 | 0.3 | 0.3 |
| eurosat | 0.4 | 1 | 1 | 0.8 |
| gtsrb | 0.3 | 0.7 | 0.8 | 0.7 |
| svhn | 0.9 | 0.4 | 0.3 | 0.5 |

**dtd-eurosat-gtsrb-svhn — layer_fair** (α = 0.5)

| task | early | mid | late | scalar oracle_fair |
|---|---:|---:|---:|---:|
| dtd | 0.3 | 0.5 | 0.4 | 0.3 |
| eurosat | 0.6 | 1 | 1 | 0.8 |
| gtsrb | 0.4 | 0.7 | 0.8 | 0.7 |
| svhn | 0.9 | 0.4 | 0.3 | 0.5 |

**eurosat-gtsrb-mnist-svhn — layer_mean** (α = 0.6)

| task | early | mid | late | scalar oracle_fair |
|---|---:|---:|---:|---:|
| eurosat | 0.5 | 1 | 1 | 1 |
| gtsrb | 0.3 | 0.5 | 1 | 0.7 |
| mnist | 0.4 | 0.3 | 0.7 | 0.4 |
| svhn | 0.9 | 0.6 | 0.3 | 0.5 |

**eurosat-gtsrb-mnist-svhn — layer_fair** (α = 0.6)

| task | early | mid | late | scalar oracle_fair |
|---|---:|---:|---:|---:|
| eurosat | 0.6 | 1 | 1 | 1 |
| gtsrb | 0.3 | 0.5 | 1 | 0.7 |
| mnist | 0.5 | 0.4 | 0.6 | 0.4 |
| svhn | 0.9 | 0.6 | 0.3 | 0.5 |

**eurosat-gtsrb-sun397-svhn — layer_mean** (α = 0.6)

| task | early | mid | late | scalar oracle_fair |
|---|---:|---:|---:|---:|
| eurosat | 0.4 | 1 | 1 | 1 |
| gtsrb | 0.4 | 0.6 | 1 | 0.7 |
| sun397 | 0.8 | 0.3 | 0.9 | 0.3 |
| svhn | 0.9 | 0.5 | 0.3 | 0.6 |

**eurosat-gtsrb-sun397-svhn — layer_fair** (α = 0.6)

| task | early | mid | late | scalar oracle_fair |
|---|---:|---:|---:|---:|
| eurosat | 0.3 | 1 | 1 | 1 |
| gtsrb | 0.3 | 0.6 | 1 | 0.7 |
| sun397 | 0.4 | 0.3 | 0.9 | 0.3 |
| svhn | 1 | 0.4 | 0.3 | 0.6 |

**stanford_cars-eurosat-gtsrb-mnist — layer_mean** (α = 0.6)

| task | early | mid | late | scalar oracle_fair |
|---|---:|---:|---:|---:|
| stanford_cars | 0.4 | 0.3 | 0.7 | 0.3 |
| eurosat | 0.5 | 1 | 1 | 0.9 |
| gtsrb | 0.7 | 0.5 | 1 | 0.7 |
| mnist | 0.8 | 0.4 | 0.6 | 0.5 |

**stanford_cars-eurosat-gtsrb-mnist — layer_fair** (α = 0.6)

| task | early | mid | late | scalar oracle_fair |
|---|---:|---:|---:|---:|
| stanford_cars | 0.3 | 0.3 | 0.5 | 0.3 |
| eurosat | 0.9 | 1 | 1 | 0.9 |
| gtsrb | 0.7 | 0.6 | 1 | 0.7 |
| mnist | 0.5 | 0.4 | 0.7 | 0.5 |

**stanford_cars-eurosat-gtsrb-resisc45 — layer_mean** (α = 0.7)

| task | early | mid | late | scalar oracle_fair |
|---|---:|---:|---:|---:|
| stanford_cars | 0.3 | 0.3 | 0.9 | 0.7 |
| eurosat | 0.5 | 0.8 | 0.8 | 0.7 |
| gtsrb | 0.7 | 0.6 | 0.8 | 0.7 |
| resisc45 | 0.6 | 0.3 | 0.9 | 0.7 |

**stanford_cars-eurosat-gtsrb-resisc45 — layer_fair** (α = 0.7)

| task | early | mid | late | scalar oracle_fair |
|---|---:|---:|---:|---:|
| stanford_cars | 0.3 | 0.3 | 0.7 | 0.7 |
| eurosat | 0.5 | 0.7 | 0.8 | 0.7 |
| gtsrb | 0.6 | 0.6 | 0.8 | 0.7 |
| resisc45 | 0.6 | 0.4 | 0.9 | 0.7 |

**stanford_cars-eurosat-mnist-resisc45 — layer_mean** (α = 0.6)

| task | early | mid | late | scalar oracle_fair |
|---|---:|---:|---:|---:|
| stanford_cars | 0.3 | 0.3 | 0.6 | 0.6 |
| eurosat | 0.5 | 0.6 | 0.8 | 0.6 |
| mnist | 0.7 | 0.6 | 0.9 | 0.6 |
| resisc45 | 0.9 | 0.4 | 0.8 | 0.6 |

**stanford_cars-eurosat-mnist-resisc45 — layer_fair** (α = 0.6)

| task | early | mid | late | scalar oracle_fair |
|---|---:|---:|---:|---:|
| stanford_cars | 0.5 | 0.6 | 0.6 | 0.6 |
| eurosat | 0.6 | 0.6 | 0.6 | 0.6 |
| mnist | 0.7 | 0.6 | 0.6 | 0.6 |
| resisc45 | 0.4 | 0.5 | 0.7 | 0.6 |

**stanford_cars-eurosat-mnist-sun397 — layer_mean** (α = 0.7)

| task | early | mid | late | scalar oracle_fair |
|---|---:|---:|---:|---:|
| stanford_cars | 0.3 | 0.3 | 0.9 | 0.6 |
| eurosat | 0.8 | 0.8 | 0.9 | 0.7 |
| mnist | 0.7 | 0.7 | 0.7 | 0.6 |
| sun397 | 0.7 | 0.3 | 0.8 | 0.5 |

**stanford_cars-eurosat-mnist-sun397 — layer_fair** (α = 0.7)

| task | early | mid | late | scalar oracle_fair |
|---|---:|---:|---:|---:|
| stanford_cars | 0.3 | 0.3 | 0.7 | 0.6 |
| eurosat | 0.6 | 0.7 | 0.7 | 0.7 |
| mnist | 0.7 | 0.6 | 0.7 | 0.6 |
| sun397 | 0.7 | 0.3 | 0.9 | 0.5 |

