# Experiment 7c/7e: label-free per-depth lane coefficients (TEST, Δ vs dc_rank)

7c = agreement objectives (agree_floor primary, agree_relfair secondary); 7e = entropy objective on per-depth lanes (identical 12-coordinate search; not AdaMerging). Selection uses only unlabeled images and the experts — **not data-free**. Each chosen vector scored once on test.
7a gate: **IN_BETWEEN** (result capped at PARTIAL).

| context | method | test mean | test worst (task) | Δmean | Δworst | share of 6b layer_fair Δmean | share of 6b layer_mean Δmean | val Δmean (diag.) |
|---|---|---:|---|---:|---:|---:|---:|---:|
| dtd-eurosat-gtsrb-svhn | dc_rank | 71.20 | 62.20 | +0.00 | +0.00 |  |  |  |
| dtd-eurosat-gtsrb-svhn | agree_floor | 71.20 | 62.20 (eurosat) | +0.00 | +0.00 | 0% | 0% | 0.00 |
| dtd-eurosat-gtsrb-svhn | agree_relfair | 71.20 | 62.20 (eurosat) | +0.00 | +0.00 | 0% | 0% | 0.00 |
| dtd-eurosat-gtsrb-svhn | entropy_mean | 66.34 | 50.06 (svhn) | -4.86 | -12.14 | -64% | -63% | -4.63 |
| dtd-eurosat-gtsrb-svhn | entropy_fair | 69.21 | 59.89 (gtsrb) | -1.98 | -2.31 | -26% | -26% | -1.87 |
| dtd-eurosat-gtsrb-svhn | layer_mean | 78.95 | 67.30 | +7.76 | +5.10 |  |  |  |
| dtd-eurosat-gtsrb-svhn | layer_fair | 78.82 | 65.06 | +7.62 | +2.86 |  |  |  |
| dtd-eurosat-gtsrb-svhn | scalar_fair | 74.16 | 61.99 | +2.97 | -0.21 |  |  |  |
| eurosat-gtsrb-mnist-svhn | dc_rank | 73.83 | 55.64 | +0.00 | +0.00 |  |  |  |
| eurosat-gtsrb-mnist-svhn | agree_floor | 78.39 | 64.43 (eurosat) | +4.56 | +8.79 | 51% | 51% | 4.57 |
| eurosat-gtsrb-mnist-svhn | agree_relfair | 74.74 | 54.39 (eurosat) | +0.91 | -1.24 | 10% | 10% | 1.07 |
| eurosat-gtsrb-mnist-svhn | entropy_mean | 67.40 | 50.33 (svhn) | -6.43 | -5.31 | -73% | -72% | -5.45 |
| eurosat-gtsrb-mnist-svhn | entropy_fair | 67.32 | 57.75 (svhn) | -6.51 | +2.12 | -73% | -73% | -6.54 |
| eurosat-gtsrb-mnist-svhn | layer_mean | 82.80 | 76.20 | +8.97 | +20.56 |  |  |  |
| eurosat-gtsrb-mnist-svhn | layer_fair | 82.69 | 75.30 | +8.86 | +19.67 |  |  |  |
| eurosat-gtsrb-mnist-svhn | scalar_fair | 77.11 | 56.25 | +3.28 | +0.61 |  |  |  |
| eurosat-gtsrb-sun397-svhn | dc_rank | 76.66 | 63.52 | +0.00 | +0.00 |  |  |  |
| eurosat-gtsrb-sun397-svhn | agree_floor | 80.09 | 71.55 (eurosat) | +3.43 | +8.03 | 44% | 43% | 3.45 |
| eurosat-gtsrb-sun397-svhn | agree_relfair | 77.69 | 64.16 (eurosat) | +1.04 | +0.64 | 13% | 13% | 1.08 |
| eurosat-gtsrb-sun397-svhn | entropy_mean | 69.93 | 50.40 (gtsrb) | -6.73 | -13.12 | -86% | -85% | -5.66 |
| eurosat-gtsrb-sun397-svhn | entropy_fair | 74.42 | 59.99 (gtsrb) | -2.24 | -3.53 | -28% | -28% | -1.71 |
| eurosat-gtsrb-sun397-svhn | layer_mean | 84.58 | 70.44 | +7.92 | +6.92 |  |  |  |
| eurosat-gtsrb-sun397-svhn | layer_fair | 84.53 | 69.26 | +7.87 | +5.74 |  |  |  |
| eurosat-gtsrb-sun397-svhn | scalar_fair | 78.68 | 63.45 | +2.02 | -0.07 |  |  |  |

## Averages over 3 contexts (core; test, Δ vs dc_rank)

| method | Δmean | Δworst | contexts with worst drop > 1 pp | verdict |
|---|---:|---:|---:|---|
| agree_floor | +2.66 | +5.61 | 0 | PARTIAL |
| agree_relfair | +0.65 | -0.20 | 1 | PARTIAL |
| entropy_mean | -6.01 | -10.19 | 3 | NOT VIABLE (same rule, for comparison) |
| entropy_fair | -3.58 | -1.24 | 2 | NOT VIABLE (same rule, for comparison) |
| layer_mean | +8.22 | +10.86 | 0 |  |
| layer_fair | +8.12 | +9.42 | 0 |  |
| scalar_fair | +2.75 | +0.11 | 0 |  |

**Primary verdict (core, agree_floor): PARTIAL**; secondary agree_relfair: PARTIAL

Rules: STRONG = Δmean ≥ 50% of 6b layer_fair Δmean AND Δworst ≥ +3 pp AND no context loses > 1 pp on its worst task; VIABLE = same with Δworst > 0; PARTIAL = positive Δmean otherwise; NOT VIABLE = Δmean ≤ 0. The decision is on all 7 contexts; core is reported separately.

## Chosen coefficients (task × depth group)

**dtd-eurosat-gtsrb-svhn — agree_floor** (α = 0.5)

| task | early | mid | late |
|---|---:|---:|---:|
| dtd | 0.5 | 0.5 | 0.5 |
| eurosat | 0.5 | 0.5 | 0.5 |
| gtsrb | 0.5 | 0.5 | 0.5 |
| svhn | 0.5 | 0.5 | 0.5 |

**dtd-eurosat-gtsrb-svhn — agree_relfair** (α = 0.5)

| task | early | mid | late |
|---|---:|---:|---:|
| dtd | 0.5 | 0.5 | 0.5 |
| eurosat | 0.5 | 0.5 | 0.5 |
| gtsrb | 0.5 | 0.5 | 0.5 |
| svhn | 0.5 | 0.5 | 0.5 |

**dtd-eurosat-gtsrb-svhn — entropy_mean** (α = 0.5)

| task | early | mid | late |
|---|---:|---:|---:|
| dtd | 0.6 | 0.3 | 0.3 |
| eurosat | 1 | 1 | 0.3 |
| gtsrb | 0.5 | 0.3 | 0.3 |
| svhn | 0.6 | 0.3 | 0.3 |

**dtd-eurosat-gtsrb-svhn — entropy_fair** (α = 0.5)

| task | early | mid | late |
|---|---:|---:|---:|
| dtd | 0.7 | 0.4 | 0.3 |
| eurosat | 0.8 | 1 | 0.3 |
| gtsrb | 0.3 | 0.5 | 0.3 |
| svhn | 1 | 0.7 | 0.3 |

**eurosat-gtsrb-mnist-svhn — agree_floor** (α = 0.6)

| task | early | mid | late |
|---|---:|---:|---:|
| eurosat | 0.6 | 0.9 | 0.7 |
| gtsrb | 0.4 | 0.6 | 0.9 |
| mnist | 0.3 | 0.6 | 0.6 |
| svhn | 0.6 | 0.6 | 0.6 |

**eurosat-gtsrb-mnist-svhn — agree_relfair** (α = 0.6)

| task | early | mid | late |
|---|---:|---:|---:|
| eurosat | 0.6 | 0.6 | 0.6 |
| gtsrb | 0.3 | 0.6 | 0.7 |
| mnist | 0.9 | 0.6 | 0.6 |
| svhn | 0.6 | 0.6 | 0.6 |

**eurosat-gtsrb-mnist-svhn — entropy_mean** (α = 0.6)

| task | early | mid | late |
|---|---:|---:|---:|
| eurosat | 1 | 1 | 0.3 |
| gtsrb | 0.5 | 0.3 | 0.3 |
| mnist | 0.3 | 0.3 | 0.3 |
| svhn | 0.6 | 0.3 | 0.3 |

**eurosat-gtsrb-mnist-svhn — entropy_fair** (α = 0.6)

| task | early | mid | late |
|---|---:|---:|---:|
| eurosat | 0.7 | 1 | 0.3 |
| gtsrb | 0.6 | 0.9 | 0.3 |
| mnist | 1 | 0.8 | 0.3 |
| svhn | 1 | 0.3 | 0.3 |

**eurosat-gtsrb-sun397-svhn — agree_floor** (α = 0.6)

| task | early | mid | late |
|---|---:|---:|---:|
| eurosat | 0.6 | 0.6 | 0.6 |
| gtsrb | 0.3 | 0.6 | 0.6 |
| sun397 | 0.7 | 0.3 | 0.7 |
| svhn | 0.6 | 0.6 | 0.3 |

**eurosat-gtsrb-sun397-svhn — agree_relfair** (α = 0.6)

| task | early | mid | late |
|---|---:|---:|---:|
| eurosat | 0.6 | 0.6 | 0.6 |
| gtsrb | 0.6 | 0.6 | 0.6 |
| sun397 | 0.6 | 0.6 | 0.8 |
| svhn | 0.6 | 0.6 | 0.5 |

**eurosat-gtsrb-sun397-svhn — entropy_mean** (α = 0.6)

| task | early | mid | late |
|---|---:|---:|---:|
| eurosat | 1 | 1 | 0.3 |
| gtsrb | 0.3 | 0.3 | 0.3 |
| sun397 | 0.8 | 0.3 | 0.3 |
| svhn | 0.5 | 0.3 | 0.3 |

**eurosat-gtsrb-sun397-svhn — entropy_fair** (α = 0.6)

| task | early | mid | late |
|---|---:|---:|---:|
| eurosat | 0.6 | 1 | 0.3 |
| gtsrb | 0.3 | 0.5 | 0.3 |
| sun397 | 1 | 0.3 | 0.3 |
| svhn | 1 | 0.5 | 0.3 |

