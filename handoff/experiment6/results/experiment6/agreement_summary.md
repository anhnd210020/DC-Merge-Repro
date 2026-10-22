# Experiment 6a: expert-agreement lane coefficients (TEST), proxy = `top1`

The search uses only unlabeled validation images and the experts; labels score the chosen vectors once on test. **Not data-free** (needs unlabeled images and the per-task experts).
Gate verdict: **PASS**.

| context | method | coefficients | mean norm | worst norm (task) | Δmean | Δworst |
|---|---|---|---:|---|---:|---:|
| dtd-eurosat-gtsrb-svhn | dc_rank | dtd=0.5, eurosa=0.5, gtsrb=0.5, svhn=0.5 | 71.20 | 62.20 (eurosat) | +0.00 | +0.00 |
| dtd-eurosat-gtsrb-svhn | agreement_mean | dtd=0.5, eurosa=0.9, gtsrb=0.7, svhn=0.3 | 75.05 | 47.20 (svhn) | +3.85 | -15.00 |
| dtd-eurosat-gtsrb-svhn | agreement_fair | dtd=0.5, eurosa=0.5, gtsrb=0.5, svhn=0.5 | 71.20 | 62.20 (eurosat) | +0.00 | +0.00 |
| dtd-eurosat-gtsrb-svhn | oracle_fair | dtd=0.3, eurosa=0.8, gtsrb=0.7, svhn=0.5 | 74.16 | 61.99 (svhn) | +2.97 | -0.21 |
| dtd-eurosat-gtsrb-svhn | oracle_mean | dtd=0.4, eurosa=0.7, gtsrb=0.6, svhn=0.4 | 73.84 | 58.01 (svhn) | +2.64 | -4.19 |
| dtd-eurosat-gtsrb-svhn | ct | gamma=0.7 | 69.41 | 55.28 (svhn) | -1.78 | -6.92 |
| eurosat-gtsrb-mnist-svhn | dc_rank | eurosa=0.6, gtsrb=0.6, mnist=0.6, svhn=0.6 | 73.83 | 55.64 (eurosat) | +0.00 | +0.00 |
| eurosat-gtsrb-mnist-svhn | agreement_mean | eurosa=1, gtsrb=0.7, mnist=0.4, svhn=0.5 | 77.11 | 56.25 (svhn) | +3.28 | +0.61 |
| eurosat-gtsrb-mnist-svhn | agreement_fair | eurosa=0.8, gtsrb=0.6, mnist=0.6, svhn=0.6 | 74.72 | 66.62 (svhn) | +0.89 | +10.98 |
| eurosat-gtsrb-mnist-svhn | oracle_fair | eurosa=1, gtsrb=0.7, mnist=0.4, svhn=0.5 | 77.11 | 56.25 (svhn) | +3.28 | +0.61 |
| eurosat-gtsrb-mnist-svhn | oracle_mean | eurosa=1, gtsrb=0.7, mnist=0.4, svhn=0.5 | 77.11 | 56.25 (svhn) | +3.28 | +0.61 |
| eurosat-gtsrb-mnist-svhn | ct | gamma=0.6 | 70.24 | 53.34 (svhn) | -3.58 | -2.30 |
| eurosat-gtsrb-sun397-svhn | dc_rank | eurosa=0.6, gtsrb=0.6, sun397=0.6, svhn=0.6 | 76.66 | 63.52 (eurosat) | +0.00 | +0.00 |
| eurosat-gtsrb-sun397-svhn | agreement_mean | eurosa=0.9, gtsrb=0.7, sun397=0.4, svhn=0.6 | 78.79 | 65.55 (svhn) | +2.14 | +2.03 |
| eurosat-gtsrb-sun397-svhn | agreement_fair | eurosa=0.7, gtsrb=0.6, sun397=0.4, svhn=0.6 | 78.05 | 70.52 (svhn) | +1.39 | +7.00 |
| eurosat-gtsrb-sun397-svhn | oracle_fair | eurosa=1, gtsrb=0.7, sun397=0.3, svhn=0.6 | 78.68 | 63.45 (svhn) | +2.02 | -0.07 |
| eurosat-gtsrb-sun397-svhn | oracle_mean | eurosa=1, gtsrb=0.7, sun397=0.4, svhn=0.5 | 79.51 | 57.78 (svhn) | +2.86 | -5.74 |
| eurosat-gtsrb-sun397-svhn | ct | gamma=0.6 | 73.44 | 61.09 (svhn) | -3.21 | -2.43 |
| stanford_cars-eurosat-gtsrb-mnist | dc_rank | stanfo=0.6, eurosa=0.6, gtsrb=0.6, mnist=0.6 | 81.60 | 72.20 (eurosat) | +0.00 | +0.00 |
| stanford_cars-eurosat-gtsrb-mnist | agreement_mean | stanfo=0.3, eurosa=0.9, gtsrb=0.8, mnist=0.5 | 83.50 | 76.31 (stanford_cars) | +1.90 | +4.12 |
| stanford_cars-eurosat-gtsrb-mnist | agreement_fair | stanfo=0.7, eurosa=0.6, gtsrb=0.5, mnist=0.3 | 79.56 | 72.82 (mnist) | -2.04 | +0.62 |
| stanford_cars-eurosat-gtsrb-mnist | oracle_fair | stanfo=0.3, eurosa=0.9, gtsrb=0.7, mnist=0.5 | 84.10 | 77.92 (stanford_cars) | +2.50 | +5.72 |
| stanford_cars-eurosat-gtsrb-mnist | oracle_mean | stanfo=0.3, eurosa=0.9, gtsrb=0.7, mnist=0.5 | 84.10 | 77.92 (stanford_cars) | +2.50 | +5.72 |
| stanford_cars-eurosat-gtsrb-mnist | ct | gamma=0.6 | 80.27 | 67.90 (gtsrb) | -1.33 | -4.30 |
| stanford_cars-eurosat-gtsrb-resisc45 | dc_rank | stanfo=0.7, eurosa=0.7, gtsrb=0.7, resisc=0.7 | 85.51 | 80.53 (stanford_cars) | +0.00 | +0.00 |
| stanford_cars-eurosat-gtsrb-resisc45 | agreement_mean | stanfo=0.3, eurosa=0.6, gtsrb=0.7, resisc=0.6 | 84.79 | 73.57 (stanford_cars) | -0.72 | -6.96 |
| stanford_cars-eurosat-gtsrb-resisc45 | agreement_fair | stanfo=1, eurosa=0.6, gtsrb=0.6, resisc=0.5 | 82.25 | 78.61 (eurosat) | -3.25 | -1.93 |
| stanford_cars-eurosat-gtsrb-resisc45 | oracle_fair | stanfo=0.7, eurosa=0.7, gtsrb=0.7, resisc=0.7 | 85.51 | 80.53 (stanford_cars) | +0.00 | +0.00 |
| stanford_cars-eurosat-gtsrb-resisc45 | oracle_mean | stanfo=0.6, eurosa=0.7, gtsrb=0.7, resisc=0.6 | 85.54 | 80.41 (stanford_cars) | +0.03 | -0.12 |
| stanford_cars-eurosat-gtsrb-resisc45 | ct | gamma=0.7 | 82.01 | 79.70 (gtsrb) | -3.49 | -0.84 |
| stanford_cars-eurosat-mnist-resisc45 | dc_rank | stanfo=0.6, eurosa=0.6, mnist=0.6, resisc=0.6 | 87.61 | 85.77 (eurosat) | +0.00 | +0.00 |
| stanford_cars-eurosat-mnist-resisc45 | agreement_mean | stanfo=0.3, eurosa=0.7, mnist=0.7, resisc=0.8 | 87.55 | 73.41 (stanford_cars) | -0.06 | -12.36 |
| stanford_cars-eurosat-mnist-resisc45 | agreement_fair | stanfo=0.8, eurosa=0.6, mnist=0.4, resisc=0.4 | 84.69 | 78.32 (mnist) | -2.92 | -7.46 |
| stanford_cars-eurosat-mnist-resisc45 | oracle_fair | stanfo=0.6, eurosa=0.6, mnist=0.6, resisc=0.6 | 87.61 | 85.77 (eurosat) | +0.00 | +0.00 |
| stanford_cars-eurosat-mnist-resisc45 | oracle_mean | stanfo=0.3, eurosa=0.7, mnist=0.7, resisc=0.7 | 87.70 | 75.06 (stanford_cars) | +0.09 | -10.72 |
| stanford_cars-eurosat-mnist-resisc45 | ct | gamma=0.6 | 86.05 | 84.55 (stanford_cars) | -1.56 | -1.22 |
| stanford_cars-eurosat-mnist-sun397 | dc_rank | stanfo=0.7, eurosa=0.7, mnist=0.7, sun397=0.7 | 89.57 | 84.67 (stanford_cars) | +0.00 | +0.00 |
| stanford_cars-eurosat-mnist-sun397 | agreement_mean | stanfo=0.3, eurosa=0.7, mnist=0.7, sun397=0.7 | 89.37 | 76.11 (stanford_cars) | -0.20 | -8.56 |
| stanford_cars-eurosat-mnist-sun397 | agreement_fair | stanfo=1, eurosa=0.7, mnist=0.7, sun397=0.7 | 87.39 | 80.91 (eurosat) | -2.19 | -3.76 |
| stanford_cars-eurosat-mnist-sun397 | oracle_fair | stanfo=0.6, eurosa=0.7, mnist=0.6, sun397=0.5 | 90.31 | 86.72 (stanford_cars) | +0.74 | +2.05 |
| stanford_cars-eurosat-mnist-sun397 | oracle_mean | stanfo=0.5, eurosa=0.8, mnist=0.7, sun397=0.5 | 90.55 | 83.01 (stanford_cars) | +0.98 | -1.66 |
| stanford_cars-eurosat-mnist-sun397 | ct | gamma=0.7 | 89.44 | 83.09 (stanford_cars) | -0.13 | -1.58 |

## Averages over 3 contexts (core; test, Δ vs dc_rank)

| method | Δmean | Δworst | contexts with worst drop > 1 pp | share of oracle_fair Δmean | verdict |
|---|---:|---:|---:|---:|---|
| dc_rank | +0.00 | +0.00 | 0 |  |  |
| agreement_mean | +3.09 | -4.12 | 1 | 112% | PARTIAL: gains below the bar or a context with worst-task drop > 1 pp |
| agreement_fair | +0.76 | +5.99 | 0 | 28% | PARTIAL: gains below the bar or a context with worst-task drop > 1 pp |
| oracle_fair | +2.75 | +0.11 | 0 |  |  |
| oracle_mean | +2.93 | -3.10 | 2 |  |  |
| ct | -2.86 | -3.88 | 3 |  |  |

## Averages over 7 contexts (all; test, Δ vs dc_rank)

| method | Δmean | Δworst | contexts with worst drop > 1 pp | share of oracle_fair Δmean | verdict |
|---|---:|---:|---:|---:|---|
| dc_rank | +0.00 | +0.00 | 0 |  |  |
| agreement_mean | +1.45 | -5.16 | 4 | 89% | PARTIAL: gains below the bar or a context with worst-task drop > 1 pp |
| agreement_fair | -1.16 | +0.78 | 3 | -71% | NOT VIABLE: no mean gain over dc_rank |
| oracle_fair | +1.64 | +1.16 | 0 |  |  |
| oracle_mean | +1.77 | -2.30 | 4 |  |  |
| ct | -2.16 | -2.80 | 6 |  |  |
