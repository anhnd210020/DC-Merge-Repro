# Experiment 6b-1: where in depth is EuroSAT's harm created? (VALIDATION, grouping = depth)

All other lanes at the context's selected α. `harm` = EuroSAT accuracy with the neighbour lane(s) fully off minus with all lanes on (pp). *only g*: neighbour on only in group g; the harm it creates there is (neighbour off) − (only g). *off g*: neighbour off only in g; the harm it removes is (off g) − (all on). Shares are relative to the total harm.

## dtd-eurosat-gtsrb-svhn (α = 0.5); EuroSAT with all lanes on: 61.19%

| neighbour | EuroSAT if fully off | total harm (pp) | early: created / share (only g) | mid: created / share (only g) | late: created / share (only g) | early: removed / share (off g) | mid: removed / share (off g) | late: removed / share (off g) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| svhn | 81.30 | +20.11 | -0.41 / -2% | +13.85 / 69% | +3.44 / 17% | +1.30 / 6% | +16.22 / 81% | +5.33 / 27% |
| gtsrb | 73.70 | +12.52 | -1.56 / -12% | +6.78 / 54% | +2.44 / 20% | -0.63 / -5% | +11.22 / 90% | +6.33 / 51% |
| svhn+gtsrb | 85.52 | +24.33 | +0.04 / 0% | +15.74 / 65% | +2.67 / 11% | -0.93 / -4% | +23.89 / 98% | +10.59 / 44% |

SVHN × GTSRB interaction on EuroSAT (reference: both fully off; interaction = harm(both in g) − harm(SVHN in g) − harm(GTSRB in g)):

| scope | harm SVHN | harm GTSRB | harm both | interaction (pp) |
|---|---:|---:|---:|---:|
| only early | +0.41 | -0.81 | +0.04 | +0.44 |
| only mid | +8.11 | +3.93 | +15.74 | +3.70 |
| only late | +0.44 | +0.00 | +2.67 | +2.22 |
| all groups | | | | +8.30 |

EuroSAT's own lane boosted ×1.5 (validation accuracy, Δ vs all lanes on):

| boosted in | dtd | eurosat | gtsrb | svhn | Δ mean normalized |
|---|---:|---:|---:|---:|---:|
| early | 53.14 (-0.37) | 64.78 (+3.59) | 72.33 (-1.70) | 62.45 (-3.23) | -0.49 |
| mid | 53.03 (-0.48) | 69.93 (+8.74) | 72.21 (-1.82) | 64.66 (-1.02) | +1.31 |
| late | 53.30 (-0.21) | 70.19 (+9.00) | 71.85 (-2.18) | 65.17 (-0.50) | +1.52 |
| all | 51.91 (-1.60) | 80.41 (+19.22) | 67.97 (-6.06) | 60.81 (-4.86) | +1.49 |

All four tasks' accuracies for every configuration are in `mechanism_result.json`.

## eurosat-gtsrb-mnist-svhn (α = 0.6); EuroSAT with all lanes on: 52.89%

| neighbour | EuroSAT if fully off | total harm (pp) | early: created / share (only g) | mid: created / share (only g) | late: created / share (only g) | early: removed / share (off g) | mid: removed / share (off g) | late: removed / share (off g) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| svhn | 83.11 | +30.22 | +1.00 / 3% | +19.89 / 66% | +3.59 / 12% | +5.07 / 17% | +25.48 / 84% | +5.93 / 20% |
| gtsrb | 77.11 | +24.22 | -1.30 / -5% | +11.67 / 48% | +4.41 / 18% | +2.63 / 11% | +20.81 / 86% | +10.30 / 43% |
| svhn+gtsrb | 88.85 | +35.96 | +1.00 / 3% | +18.33 / 51% | +0.19 / 1% | +5.85 / 16% | +34.59 / 96% | +15.33 / 43% |

SVHN × GTSRB interaction on EuroSAT (reference: both fully off; interaction = harm(both in g) − harm(SVHN in g) − harm(GTSRB in g)):

| scope | harm SVHN | harm GTSRB | harm both | interaction (pp) |
|---|---:|---:|---:|---:|
| only early | +0.89 | -0.26 | +1.00 | +0.37 |
| only mid | +8.52 | +4.07 | +18.33 | +5.74 |
| only late | -0.19 | -0.26 | +0.19 | +0.63 |
| all groups | | | | +18.48 |

EuroSAT's own lane boosted ×1.5 (validation accuracy, Δ vs all lanes on):

| boosted in | eurosat | gtsrb | mnist | svhn | Δ mean normalized |
|---|---:|---:|---:|---:|---:|
| early | 55.63 (+2.74) | 72.05 (-1.15) | 94.10 (-0.60) | 64.54 (-4.88) | -1.01 |
| mid | 62.30 (+9.41) | 71.10 (-2.10) | 93.75 (-0.95) | 67.31 (-2.11) | +1.07 |
| late | 62.74 (+9.85) | 69.52 (-3.68) | 94.45 (-0.25) | 68.25 (-1.17) | +1.20 |
| all | 75.70 (+22.81) | 65.84 (-7.36) | 93.40 (-1.30) | 61.99 (-7.43) | +1.67 |

All four tasks' accuracies for every configuration are in `mechanism_result.json`.

## eurosat-gtsrb-sun397-svhn (α = 0.6); EuroSAT with all lanes on: 61.11%

| neighbour | EuroSAT if fully off | total harm (pp) | early: created / share (only g) | mid: created / share (only g) | late: created / share (only g) | early: removed / share (off g) | mid: removed / share (off g) | late: removed / share (off g) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| svhn | 88.63 | +27.52 | +2.00 / 7% | +15.56 / 57% | +1.33 / 5% | +3.44 / 13% | +23.81 / 87% | +8.78 / 32% |
| gtsrb | 84.04 | +22.93 | -1.85 / -8% | +12.07 / 53% | +4.00 / 17% | +0.93 / 4% | +20.96 / 91% | +10.15 / 44% |
| svhn+gtsrb | 91.89 | +30.78 | +0.74 / 2% | +13.41 / 44% | +0.04 / 0% | +3.37 / 11% | +29.63 / 96% | +16.52 / 54% |

SVHN × GTSRB interaction on EuroSAT (reference: both fully off; interaction = harm(both in g) − harm(SVHN in g) − harm(GTSRB in g)):

| scope | harm SVHN | harm GTSRB | harm both | interaction (pp) |
|---|---:|---:|---:|---:|
| only early | +0.81 | -0.37 | +0.74 | +0.30 |
| only mid | +3.63 | +1.74 | +13.41 | +8.04 |
| only late | +0.11 | -0.26 | +0.04 | +0.19 |
| all groups | | | | +19.67 |

EuroSAT's own lane boosted ×1.5 (validation accuracy, Δ vs all lanes on):

| boosted in | eurosat | gtsrb | sun397 | svhn | Δ mean normalized |
|---|---:|---:|---:|---:|---:|
| early | 65.70 (+4.59) | 76.56 (-2.26) | 66.12 (-0.30) | 65.71 (-3.19) | -0.34 |
| mid | 71.81 (+10.70) | 76.60 (-2.22) | 65.54 (-0.88) | 67.40 (-1.50) | +1.47 |
| late | 72.56 (+11.44) | 74.27 (-4.55) | 65.84 (-0.58) | 68.08 (-0.83) | +1.34 |
| all | 83.48 (+22.37) | 69.95 (-8.87) | 65.16 (-1.26) | 63.20 (-5.70) | +1.52 |

All four tasks' accuracies for every configuration are in `mechanism_result.json`.

