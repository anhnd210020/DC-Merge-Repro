# Lane-harm experiment on official DC-Merge: results summary

Operator: top-k = **rank**, alpha = 0.4, official commit 089af77.
Harm reference: **solo**.  Contexts: 7.  All consistency checks passed: True.

## Target: eurosat

| context | kind | harm vs standalone | harm vs solo | protect_lane | fix_conflict | undilute | block_injection | restore_own_update | control_matched_lane | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dtd-eurosat-gtsrb-svhn | eurosat+gtsrb+svhn | 35.23 | 21.67 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 3.67 / 0.58 | 0.00 / 0.00 | no isolated mechanism (cross-lane readout or downstream) |
| eurosat-gtsrb-mnist-svhn | eurosat+gtsrb+svhn | 34.15 | 20.59 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 4.00 / 0.82 | 0.00 / 0.00 | no isolated mechanism (cross-lane readout or downstream) |
| eurosat-gtsrb-sun397-svhn | eurosat+gtsrb+svhn | 29.37 | 15.81 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 3.30 / 0.54 | 0.00 / 0.00 | partial: restore_own_update removes 21% (control 0%) |
| stanford_cars-eurosat-gtsrb-mnist | eurosat+one | 28.23 | 14.67 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 2.96 / 0.28 | 0.00 / 0.00 | partial: restore_own_update removes 20% (control 0%) |
| stanford_cars-eurosat-gtsrb-resisc45 | eurosat+one | 18.04 | 4.48 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 2.70 / 0.36 | 0.00 / 0.00 | cause = restore_own_update (removes 60%, control 0%) |
| stanford_cars-eurosat-mnist-resisc45 | eurosat+neither | 18.89 | 5.33 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 2.22 / 0.04 | 0.00 / 0.00 | partial: restore_own_update removes 42% (control 0%) |
| stanford_cars-eurosat-mnist-sun397 | eurosat+neither | 20.60 | 7.04 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 2.04 / 0.39 | 0.00 / 0.00 | partial: restore_own_update removes 29% (control 0%) |

Cells: benefit to target (pp) / mean cost to the other tasks (pp).

## Target: stanford_cars

| context | kind | harm vs standalone | harm vs solo | protect_lane | fix_conflict | undilute | block_injection | restore_own_update | control_matched_lane | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stanford_cars-eurosat-gtsrb-mnist | eurosat+one | 10.95 | 2.55 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.81 / 1.04 | 0.00 / 0.00 | restore_own_update: real cause, but fixing it moves the harm to others |
| stanford_cars-eurosat-gtsrb-resisc45 | eurosat+one | 12.63 | 4.23 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.19 / 0.55 | 0.00 / 0.00 | no isolated mechanism (cross-lane readout or downstream) |
| stanford_cars-eurosat-mnist-resisc45 | eurosat+neither | 11.51 | 3.11 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.44 / 1.02 | 0.00 / 0.00 | no isolated mechanism (cross-lane readout or downstream) |
| stanford_cars-eurosat-mnist-sun397 | eurosat+neither | 10.01 | 1.62 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.31 / 0.95 | 0.00 / 0.00 | no isolated mechanism (cross-lane readout or downstream) |

Cells: benefit to target (pp) / mean cost to the other tasks (pp).

## Target: dtd

| context | kind | harm vs standalone | harm vs solo | protect_lane | fix_conflict | undilute | block_injection | restore_own_update | control_matched_lane | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dtd-eurosat-gtsrb-svhn | eurosat+gtsrb+svhn | 15.37 | 3.24 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.69 / 0.01 | 0.00 / 0.00 | partial: restore_own_update removes 21% (control 0%) |

Cells: benefit to target (pp) / mean cost to the other tasks (pp).

## Target: gtsrb

| context | kind | harm vs standalone | harm vs solo | protect_lane | fix_conflict | undilute | block_injection | restore_own_update | control_matched_lane | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dtd-eurosat-gtsrb-svhn | eurosat+gtsrb+svhn | 30.01 | 7.84 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 4.16 / 0.67 | 0.00 / 0.00 | cause = restore_own_update (removes 53%, control 0%) |
| eurosat-gtsrb-mnist-svhn | eurosat+gtsrb+svhn | 31.48 | 9.30 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 2.61 / 0.48 | 0.00 / 0.00 | partial: restore_own_update removes 28% (control 0%) |
| eurosat-gtsrb-sun397-svhn | eurosat+gtsrb+svhn | 28.59 | 6.41 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 4.35 / 0.65 | 0.00 / 0.00 | cause = restore_own_update (removes 68%, control 0%) |
| stanford_cars-eurosat-gtsrb-mnist | eurosat+one | 32.03 | 9.86 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 1.98 / 0.68 | 0.00 / 0.00 | partial: restore_own_update removes 20% (control 0%) |
| stanford_cars-eurosat-gtsrb-resisc45 | eurosat+one | 31.28 | 9.11 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 2.85 / 0.56 | 0.00 / 0.00 | partial: restore_own_update removes 31% (control 0%) |

Cells: benefit to target (pp) / mean cost to the other tasks (pp).

## Target: mnist

| context | kind | harm vs standalone | harm vs solo | protect_lane | fix_conflict | undilute | block_injection | restore_own_update | control_matched_lane | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| eurosat-gtsrb-mnist-svhn | eurosat+gtsrb+svhn | 10.60 | 1.10 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 2.50 / 0.80 | 0.00 / 0.00 | cause = restore_own_update (removes 227%, control 0%) |
| stanford_cars-eurosat-gtsrb-mnist | eurosat+one | 16.95 | 7.45 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 2.05 / 0.87 | 0.00 / 0.00 | partial: restore_own_update removes 28% (control 0%) |
| stanford_cars-eurosat-mnist-resisc45 | eurosat+neither | 15.20 | 5.70 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 1.40 / 0.25 | 0.00 / 0.00 | partial: restore_own_update removes 25% (control 0%) |
| stanford_cars-eurosat-mnist-sun397 | eurosat+neither | 14.55 | 5.05 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 1.45 / 0.85 | 0.00 / 0.00 | partial: restore_own_update removes 29% (control 0%) |

Cells: benefit to target (pp) / mean cost to the other tasks (pp).

## Target: resisc45

| context | kind | harm vs standalone | harm vs solo | protect_lane | fix_conflict | undilute | block_injection | restore_own_update | control_matched_lane | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stanford_cars-eurosat-gtsrb-resisc45 | eurosat+one | 15.57 | 4.27 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 1.51 / 0.50 | 0.00 / 0.00 | partial: restore_own_update removes 35% (control 0%) |
| stanford_cars-eurosat-mnist-resisc45 | eurosat+neither | 15.27 | 3.97 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 1.57 / 0.26 | 0.00 / 0.00 | partial: restore_own_update removes 40% (control 0%) |

Cells: benefit to target (pp) / mean cost to the other tasks (pp).

## Target: sun397

| context | kind | harm vs standalone | harm vs solo | protect_lane | fix_conflict | undilute | block_injection | restore_own_update | control_matched_lane | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| eurosat-gtsrb-sun397-svhn | eurosat+gtsrb+svhn | 6.87 | 0.65 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.15 / 0.62 | 0.00 / 0.00 | restore_own_update: real cause, but fixing it moves the harm to others |
| stanford_cars-eurosat-mnist-sun397 | eurosat+neither | 7.02 | 0.81 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.30 / 0.91 | 0.00 / 0.00 | restore_own_update: real cause, but fixing it moves the harm to others |

Cells: benefit to target (pp) / mean cost to the other tasks (pp).

## Target: svhn

| context | kind | harm vs standalone | harm vs solo | protect_lane | fix_conflict | undilute | block_injection | restore_own_update | control_matched_lane | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dtd-eurosat-gtsrb-svhn | eurosat+gtsrb+svhn | 36.34 | -0.04 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 5.28 / 0.91 | 0.00 / 0.00 | no harm to explain |
| eurosat-gtsrb-mnist-svhn | eurosat+gtsrb+svhn | 36.09 | -0.29 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 5.40 / 1.64 | 0.00 / 0.00 | no harm to explain |
| eurosat-gtsrb-sun397-svhn | eurosat+gtsrb+svhn | 36.07 | -0.31 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 3.90 / 1.03 | 0.00 / 0.00 | no harm to explain |

Cells: benefit to target (pp) / mean cost to the other tasks (pp).
