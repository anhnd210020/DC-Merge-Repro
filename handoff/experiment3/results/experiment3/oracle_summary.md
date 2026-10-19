# Experiment 3: oracle per-task coefficients vs dc_rank and CT-Merging (TEST)

Coefficients searched on validation; each vector evaluated once on test.

| context | method | coefficients | mean norm | worst norm (task) | EuroSAT acc |
|---|---|---|---:|---|---:|
| dtd-eurosat-gtsrb-svhn | dc_rank | dtd=0.5, eurosa=0.5, gtsrb=0.5, svhn=0.5 | 71.19 | 62.20 (eurosat) | 61.07 |
| dtd-eurosat-gtsrb-svhn | oracle_mean | dtd=0.4, eurosa=0.7, gtsrb=0.6, svhn=0.4 | 73.85 | 57.99 (svhn) | 79.52 |
| dtd-eurosat-gtsrb-svhn | oracle_fair | dtd=0.3, eurosa=0.8, gtsrb=0.7, svhn=0.5 | 74.14 | 62.01 (svhn) | 76.70 |
| dtd-eurosat-gtsrb-svhn | ct | gamma=0.7 | 69.41 | 55.28 (svhn) | 66.93 |
| eurosat-gtsrb-mnist-svhn | dc_rank | eurosa=0.6, gtsrb=0.6, mnist=0.6, svhn=0.6 | 73.82 | 55.45 (eurosat) | 54.44 |
| eurosat-gtsrb-mnist-svhn | oracle_mean | eurosa=1, gtsrb=0.7, mnist=0.4, svhn=0.5 | 77.11 | 56.21 (svhn) | 86.22 |
| eurosat-gtsrb-mnist-svhn | oracle_fair | eurosa=1, gtsrb=0.7, mnist=0.4, svhn=0.5 | 77.11 | 56.21 (svhn) | 86.22 |
| eurosat-gtsrb-mnist-svhn | ct | gamma=0.6 | 70.24 | 53.34 (svhn) | 70.52 |
| eurosat-gtsrb-sun397-svhn | dc_rank | eurosa=0.6, gtsrb=0.6, sun397=0.6, svhn=0.6 | 76.68 | 63.60 (eurosat) | 62.44 |
| eurosat-gtsrb-sun397-svhn | oracle_mean | eurosa=1, gtsrb=0.7, sun397=0.4, svhn=0.5 | 79.51 | 57.84 (svhn) | 89.15 |
| eurosat-gtsrb-sun397-svhn | oracle_fair | eurosa=1, gtsrb=0.7, sun397=0.3, svhn=0.6 | 78.61 | 63.42 (svhn) | 83.74 |
| eurosat-gtsrb-sun397-svhn | ct | gamma=0.6 | 73.44 | 61.09 (svhn) | 69.41 |
| stanford_cars-eurosat-gtsrb-mnist | dc_rank | stanfo=0.6, eurosa=0.6, gtsrb=0.6, mnist=0.6 | 81.59 | 72.16 (eurosat) | 70.85 |
| stanford_cars-eurosat-gtsrb-mnist | oracle_mean | stanfo=0.3, eurosa=0.9, gtsrb=0.7, mnist=0.5 | 84.09 | 77.92 (stanford_cars) | 89.81 |
| stanford_cars-eurosat-gtsrb-mnist | oracle_fair | stanfo=0.3, eurosa=0.9, gtsrb=0.7, mnist=0.5 | 84.09 | 77.92 (stanford_cars) | 89.81 |
| stanford_cars-eurosat-gtsrb-mnist | ct | gamma=0.6 | 80.27 | 67.90 (gtsrb) | 77.89 |
| stanford_cars-eurosat-gtsrb-resisc45 | dc_rank | stanfo=0.7, eurosa=0.7, gtsrb=0.7, resisc=0.7 | 85.50 | 80.58 (stanford_cars) | 87.00 |
| stanford_cars-eurosat-gtsrb-resisc45 | oracle_mean | stanfo=0.6, eurosa=0.7, gtsrb=0.7, resisc=0.6 | 85.54 | 80.45 (stanford_cars) | 88.22 |
| stanford_cars-eurosat-gtsrb-resisc45 | oracle_fair | stanfo=0.7, eurosa=0.7, gtsrb=0.7, resisc=0.7 | 85.50 | 80.58 (stanford_cars) | 87.00 |
| stanford_cars-eurosat-gtsrb-resisc45 | ct | gamma=0.7 | 82.01 | 79.70 (gtsrb) | 84.15 |
| stanford_cars-eurosat-mnist-resisc45 | dc_rank | stanfo=0.6, eurosa=0.6, mnist=0.6, resisc=0.6 | 87.68 | 85.93 (eurosat) | 84.37 |
| stanford_cars-eurosat-mnist-resisc45 | oracle_mean | stanfo=0.3, eurosa=0.7, mnist=0.7, resisc=0.7 | 87.69 | 75.08 (stanford_cars) | 90.67 |
| stanford_cars-eurosat-mnist-resisc45 | oracle_fair | stanfo=0.6, eurosa=0.6, mnist=0.6, resisc=0.6 | 87.68 | 85.93 (eurosat) | 84.37 |
| stanford_cars-eurosat-mnist-resisc45 | ct | gamma=0.6 | 86.05 | 84.55 (stanford_cars) | 85.78 |
| stanford_cars-eurosat-mnist-sun397 | dc_rank | stanfo=0.7, eurosa=0.7, mnist=0.7, sun397=0.7 | 89.61 | 84.76 (stanford_cars) | 86.33 |
| stanford_cars-eurosat-mnist-sun397 | oracle_mean | stanfo=0.5, eurosa=0.8, mnist=0.7, sun397=0.5 | 90.52 | 82.95 (stanford_cars) | 93.11 |
| stanford_cars-eurosat-mnist-sun397 | oracle_fair | stanfo=0.6, eurosa=0.7, mnist=0.6, sun397=0.5 | 90.32 | 86.70 (stanford_cars) | 90.81 |
| stanford_cars-eurosat-mnist-sun397 | ct | gamma=0.7 | 89.44 | 83.09 (stanford_cars) | 87.81 |

## Averages over 7 contexts (test)

| method | mean norm | worst norm | Δmean vs dc_rank | Δworst vs dc_rank | verdict |
|---|---:|---:|---:|---:|---|
| dc_rank | 80.87 | 72.09 | +0.00 | +0.00 |  |
| oracle_mean | 82.62 | 69.78 | +1.75 | -2.32 | SMALL GAIN: below the thresholds, discuss |
| oracle_fair | 82.49 | 73.25 | +1.63 | +1.16 | SMALL GAIN: below the thresholds, discuss |
| ct | 78.70 | 69.28 | -2.17 | -2.82 |  |

## Per-context deltas vs dc_rank (test)

| context | oracle_mean Δmean / Δworst | oracle_fair Δmean / Δworst |
|---|---:|---:|
| dtd-eurosat-gtsrb-svhn | +2.66 / -4.21 | +2.95 / -0.19 |
| eurosat-gtsrb-mnist-svhn | +3.30 / +0.76 | +3.30 / +0.76 |
| eurosat-gtsrb-sun397-svhn | +2.83 / -5.76 | +1.93 / -0.18 |
| stanford_cars-eurosat-gtsrb-mnist | +2.50 / +5.76 | +2.50 / +5.76 |
| stanford_cars-eurosat-gtsrb-resisc45 | +0.05 / -0.12 | +0.00 / +0.00 |
| stanford_cars-eurosat-mnist-resisc45 | +0.02 / -10.85 | +0.00 / +0.00 |
| stanford_cars-eurosat-mnist-sun397 | +0.91 / -1.81 | +0.71 / +1.95 |

Checks: baseline reproduces Experiment 2a dc_rank within 0.000 pp (val) and 0.000 pp (test); lane sum error ≤ 1.7e-07.


## Baseline provenance warning

Primary `dc_rank` comparisons use the current-environment direct replay recorded in `/srv/dcmerge/results/experiment2-current-baseline/`; the archived Experiment 2 `compare/` files remain unchanged. Against the Oct 16 archived values, the largest selected-alpha per-task difference is +0.158353 pp on validation (GTSRB, `dtd-eurosat-gtsrb-svhn`) and -0.148148 pp on test (EuroSAT, `stanford_cars-eurosat-gtsrb-resisc45`). A repeated same-input current GPU evaluation was exact, while the historical 0.32 pp internal difference compared different arithmetic paths; this supports numerical sensitivity near decision boundaries but does not prove stochastic GPU variation or establish an archived run-to-run tolerance. See `baseline_replay_warning.md` and `/srv/dcmerge/results/experiment2-recreated/replay-diagnosis.md` for verified inputs and remaining historical uncertainties.
