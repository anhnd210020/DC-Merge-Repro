# Experiment 2a: DC-Merge vs CT-Merging (test split, coefficient selected on validation)

| context | method | coef | mean acc | mean norm | worst norm (task) | EuroSAT acc |
|---|---|---:|---:|---:|---|---:|
| dtd-eurosat-gtsrb-svhn | dc_official | 0.7 | 59.08 | 66.04 | 61.07 (eurosat) | 59.96 |
| dtd-eurosat-gtsrb-svhn | dc_rank | 0.5 | 63.52 | 71.22 | 62.20 (eurosat) | 61.07 |
| dtd-eurosat-gtsrb-svhn | ct | 0.7 | 61.51 | 69.41 | 55.28 (svhn) | 66.93 |
| eurosat-gtsrb-mnist-svhn | dc_official | 0.8 | 70.19 | 71.54 | 62.46 (eurosat) | 61.33 |
| eurosat-gtsrb-mnist-svhn | dc_rank | 0.6 | 72.44 | 73.83 | 55.56 (eurosat) | 54.56 |
| eurosat-gtsrb-mnist-svhn | ct | 0.6 | 68.98 | 70.24 | 53.34 (svhn) | 70.52 |
| eurosat-gtsrb-sun397-svhn | dc_official | 0.8 | 63.69 | 71.01 | 63.71 (eurosat) | 62.56 |
| eurosat-gtsrb-sun397-svhn | dc_rank | 0.6 | 69.14 | 76.65 | 63.48 (eurosat) | 62.33 |
| eurosat-gtsrb-sun397-svhn | ct | 0.6 | 66.11 | 73.44 | 61.09 (svhn) | 69.41 |
| stanford_cars-eurosat-gtsrb-mnist | dc_official | 0.9 | 72.39 | 77.98 | 68.56 (gtsrb) | 70.44 |
| stanford_cars-eurosat-gtsrb-mnist | dc_rank | 0.6 | 75.63 | 81.60 | 72.20 (eurosat) | 70.89 |
| stanford_cars-eurosat-gtsrb-mnist | ct | 0.6 | 74.46 | 80.27 | 67.90 (gtsrb) | 77.89 |
| stanford_cars-eurosat-gtsrb-resisc45 | dc_official | 0.9 | 72.49 | 79.04 | 72.68 (gtsrb) | 82.00 |
| stanford_cars-eurosat-gtsrb-resisc45 | dc_rank | 0.7 | 78.76 | 85.55 | 80.55 (stanford_cars) | 87.15 |
| stanford_cars-eurosat-gtsrb-resisc45 | ct | 0.7 | 75.36 | 82.01 | 79.70 (gtsrb) | 84.15 |
| stanford_cars-eurosat-mnist-resisc45 | dc_official | 0.9 | 77.14 | 83.68 | 80.88 (stanford_cars) | 80.70 |
| stanford_cars-eurosat-mnist-resisc45 | dc_rank | 0.6 | 80.66 | 87.60 | 85.81 (eurosat) | 84.26 |
| stanford_cars-eurosat-mnist-resisc45 | ct | 0.6 | 79.24 | 86.05 | 84.55 (stanford_cars) | 85.78 |
| stanford_cars-eurosat-mnist-sun397 | dc_official | 0.9 | 72.92 | 84.28 | 78.31 (eurosat) | 76.89 |
| stanford_cars-eurosat-mnist-sun397 | dc_rank | 0.7 | 77.77 | 89.64 | 84.76 (stanford_cars) | 86.41 |
| stanford_cars-eurosat-mnist-sun397 | ct | 0.7 | 77.74 | 89.44 | 83.09 (stanford_cars) | 87.81 |
| all8 | dc_official | 0.4 | 64.13 | 73.86 | 58.28 (gtsrb) | 61.15 |
| all8 | ct | 0.4 | 63.18 | 72.95 | 51.00 (svhn) | 65.04 |

## Averages over the 7 four-task contexts

| method | mean norm | mean worst norm | mean EuroSAT acc | contexts |
|---|---:|---:|---:|---:|
| dc_official | 76.22 | 69.67 | 70.56 | 7 |
| dc_rank | 80.87 | 72.08 | 72.38 | 7 |
| ct | 78.70 | 69.28 | 77.50 | 7 |
