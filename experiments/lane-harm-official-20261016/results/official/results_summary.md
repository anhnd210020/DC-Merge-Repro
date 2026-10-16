# Lane-harm experiment on official DC-Merge: results summary

Operator: top-k = **official**, alpha = 0.4, official commit 089af77.
Harm reference: **standalone**.  Contexts: 8.  All consistency checks passed: True.
Baseline merge identical to official `dc_merge` in every row: True.

## Target: eurosat

| context | kind | harm vs standalone | harm vs solo | protect_lane | fix_conflict | undilute | block_injection | restore_own_update | control_matched_lane | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dtd-eurosat-gtsrb-svhn | eurosat+gtsrb+svhn | 37.45 | 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 16.00 / 0.73 | 0.00 / 0.00 | partial: restore_own_update removes 43% (control 0%) |
| eurosat-gtsrb-mnist-svhn | eurosat+gtsrb+svhn | 34.97 | -2.48 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 15.93 / 0.17 | 0.00 / 0.00 | partial: restore_own_update removes 46% (control 0%) |
| eurosat-gtsrb-sun397-svhn | eurosat+gtsrb+svhn | 35.26 | -2.19 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 16.04 / 0.92 | 0.00 / 0.00 | partial: restore_own_update removes 45% (control 0%) |
| stanford_cars-eurosat-gtsrb-mnist | eurosat+one | 32.86 | -4.59 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 16.67 / 0.16 | 0.00 / 0.00 | cause = restore_own_update (removes 51%, control 0%) |
| stanford_cars-eurosat-gtsrb-resisc45 | eurosat+one | 28.26 | -9.19 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 14.33 / 0.61 | 0.00 / 0.00 | cause = restore_own_update (removes 51%, control 0%) |
| stanford_cars-eurosat-mnist-resisc45 | eurosat+neither | 27.63 | -9.81 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 13.89 / 0.45 | 0.00 / 0.00 | cause = restore_own_update (removes 50%, control 0%) |
| stanford_cars-eurosat-mnist-sun397 | eurosat+neither | 31.60 | -5.85 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 16.52 / 0.42 | 0.00 / 0.00 | cause = restore_own_update (removes 52%, control 0%) |
| all8 | all8 | 39.23 | 1.78 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 7.78 / 0.89 | 0.00 / 0.00 | no isolated mechanism (cross-lane readout or downstream) |

Cells: benefit to target (pp) / mean cost to the other tasks (pp).

## Target: stanford_cars

| context | kind | harm vs standalone | harm vs solo | protect_lane | fix_conflict | undilute | block_injection | restore_own_update | control_matched_lane | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stanford_cars-eurosat-gtsrb-mnist | eurosat+one | 13.74 | -1.87 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 4.17 / 1.68 | 0.00 / 0.00 | partial: restore_own_update removes 30% (control 0%) |
| stanford_cars-eurosat-gtsrb-resisc45 | eurosat+one | 13.74 | -1.87 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 3.86 / 1.29 | 0.00 / 0.00 | partial: restore_own_update removes 28% (control 0%) |
| stanford_cars-eurosat-mnist-resisc45 | eurosat+neither | 13.37 | -2.24 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 3.92 / 2.11 | 0.00 / 0.00 | partial: restore_own_update removes 29% (control 0%) |
| stanford_cars-eurosat-mnist-sun397 | eurosat+neither | 13.18 | -2.43 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 3.17 / 2.08 | 0.00 / 0.00 | partial: restore_own_update removes 24% (control 0%) |
| all8 | all8 | 14.30 | -1.31 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.12 / 1.38 | 0.00 / 0.00 | no isolated mechanism (cross-lane readout or downstream) |

Cells: benefit to target (pp) / mean cost to the other tasks (pp).

## Target: dtd

| context | kind | harm vs standalone | harm vs solo | protect_lane | fix_conflict | undilute | block_injection | restore_own_update | control_matched_lane | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dtd-eurosat-gtsrb-svhn | eurosat+gtsrb+svhn | 20.26 | -1.28 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 6.91 / 1.49 | 0.00 / 0.00 | partial: restore_own_update removes 34% (control 0%) |
| all8 | all8 | 18.67 | -2.87 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 1.76 / 0.46 | 0.00 / 0.00 | no isolated mechanism (cross-lane readout or downstream) |

Cells: benefit to target (pp) / mean cost to the other tasks (pp).

## Target: gtsrb

| context | kind | harm vs standalone | harm vs solo | protect_lane | fix_conflict | undilute | block_injection | restore_own_update | control_matched_lane | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dtd-eurosat-gtsrb-svhn | eurosat+gtsrb+svhn | 45.33 | -13.18 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 23.63 / 2.04 | 0.00 / 0.00 | cause = restore_own_update (removes 52%, control 0%) |
| eurosat-gtsrb-mnist-svhn | eurosat+gtsrb+svhn | 45.41 | -13.10 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 22.76 / 1.41 | 0.00 / 0.00 | cause = restore_own_update (removes 50%, control 0%) |
| eurosat-gtsrb-sun397-svhn | eurosat+gtsrb+svhn | 45.13 | -13.38 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 23.00 / 1.44 | 0.00 / 0.00 | cause = restore_own_update (removes 51%, control 0%) |
| stanford_cars-eurosat-gtsrb-mnist | eurosat+one | 45.81 | -12.71 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 21.02 / 1.97 | 0.00 / 0.00 | partial: restore_own_update removes 46% (control 0%) |
| stanford_cars-eurosat-gtsrb-resisc45 | eurosat+one | 45.17 | -13.34 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 19.91 / 1.93 | 0.00 / 0.00 | partial: restore_own_update removes 44% (control 0%) |
| all8 | all8 | 39.99 | -18.53 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 5.54 / 1.45 | 0.00 / 0.00 | no isolated mechanism (cross-lane readout or downstream) |

Cells: benefit to target (pp) / mean cost to the other tasks (pp).

## Target: mnist

| context | kind | harm vs standalone | harm vs solo | protect_lane | fix_conflict | undilute | block_injection | restore_own_update | control_matched_lane | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| eurosat-gtsrb-mnist-svhn | eurosat+gtsrb+svhn | 25.65 | -18.65 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 16.60 / 1.88 | 0.00 / 0.00 | cause = restore_own_update (removes 65%, control 0%) |
| stanford_cars-eurosat-gtsrb-mnist | eurosat+one | 28.75 | -15.55 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 14.80 / 2.42 | 0.00 / 0.00 | cause = restore_own_update (removes 51%, control 0%) |
| stanford_cars-eurosat-mnist-resisc45 | eurosat+neither | 27.05 | -17.25 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 14.55 / 2.06 | 0.00 / 0.00 | cause = restore_own_update (removes 54%, control 0%) |
| stanford_cars-eurosat-mnist-sun397 | eurosat+neither | 27.10 | -17.20 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 15.65 / 2.27 | 0.00 / 0.00 | cause = restore_own_update (removes 58%, control 0%) |
| all8 | all8 | 13.40 | -30.90 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 4.10 / 1.23 | 0.00 / 0.00 | partial: restore_own_update removes 31% (control 0%) |

Cells: benefit to target (pp) / mean cost to the other tasks (pp).

## Target: resisc45

| context | kind | harm vs standalone | harm vs solo | protect_lane | fix_conflict | undilute | block_injection | restore_own_update | control_matched_lane | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stanford_cars-eurosat-gtsrb-resisc45 | eurosat+one | 22.72 | -6.76 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 10.56 / 0.26 | 0.00 / 0.00 | partial: restore_own_update removes 46% (control 0%) |
| stanford_cars-eurosat-mnist-resisc45 | eurosat+neither | 23.11 | -6.37 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 10.95 / -0.53 | 0.00 / 0.00 | partial: restore_own_update removes 47% (control 0%) |
| all8 | all8 | 18.84 | -10.63 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 3.40 / 0.48 | 0.00 / 0.00 | no isolated mechanism (cross-lane readout or downstream) |

Cells: benefit to target (pp) / mean cost to the other tasks (pp).

## Target: sun397

| context | kind | harm vs standalone | harm vs solo | protect_lane | fix_conflict | undilute | block_injection | restore_own_update | control_matched_lane | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| eurosat-gtsrb-sun397-svhn | eurosat+gtsrb+svhn | 8.81 | -1.13 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 2.57 / -0.34 | 0.00 / 0.00 | partial: restore_own_update removes 29% (control 0%) |
| stanford_cars-eurosat-mnist-sun397 | eurosat+neither | 8.71 | -1.23 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 2.24 / 0.26 | 0.00 / 0.00 | partial: restore_own_update removes 26% (control 0%) |
| all8 | all8 | 7.96 | -1.99 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.71 / 0.94 | 0.00 / 0.00 | no isolated mechanism (cross-lane readout or downstream) |

Cells: benefit to target (pp) / mean cost to the other tasks (pp).

## Target: svhn

| context | kind | harm vs standalone | harm vs solo | protect_lane | fix_conflict | undilute | block_injection | restore_own_update | control_matched_lane | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dtd-eurosat-gtsrb-svhn | eurosat+gtsrb+svhn | 45.93 | -11.77 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 16.33 / 2.23 | 0.00 / 0.00 | partial: restore_own_update removes 36% (control 0%) |
| eurosat-gtsrb-mnist-svhn | eurosat+gtsrb+svhn | 47.41 | -10.30 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 17.77 / 0.33 | 0.00 / 0.00 | partial: restore_own_update removes 37% (control 0%) |
| eurosat-gtsrb-sun397-svhn | eurosat+gtsrb+svhn | 46.83 | -10.87 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 15.67 / 2.70 | 0.00 / 0.00 | partial: restore_own_update removes 33% (control 0%) |
| all8 | all8 | 38.76 | -18.94 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 7.49 / 1.54 | 0.00 / 0.00 | no isolated mechanism (cross-lane readout or downstream) |

Cells: benefit to target (pp) / mean cost to the other tasks (pp).
