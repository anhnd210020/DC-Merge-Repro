# Experiment 7d: widened grid {0.0,…,1.5} for the per-depth oracle (core, descriptive)

Started from the 6b chosen vectors; passes until no improvement (cap 5). Extra = wide − 6b, test. Shrinkage = validation gain − test gain vs dc_rank.

| context | objective | passes | 6b test mean / worst | wide test mean / worst | extra Δmean | extra Δworst | shrink mean | shrink worst |
|---|---|---:|---|---|---:|---:|---:|---:|
| dtd-eurosat-gtsrb-svhn | layer_mean | 5 | 78.95 / 67.30 | 80.79 / 64.22 | +1.84 | -3.08 | -0.33 | -0.40 |
| dtd-eurosat-gtsrb-svhn | layer_fair | 3 | 78.82 / 65.06 | 81.51 / 66.40 | +2.70 | +1.33 | -0.75 | -0.07 |
| eurosat-gtsrb-mnist-svhn | layer_mean | 5 | 82.80 / 76.20 | 84.76 / 75.80 | +1.96 | -0.40 | +0.38 | +1.90 |
| eurosat-gtsrb-mnist-svhn | layer_fair | 2 | 82.69 / 75.30 | 84.00 / 76.28 | +1.31 | +0.98 | +0.30 | +1.74 |
| eurosat-gtsrb-sun397-svhn | layer_mean | 5 | 84.58 / 70.44 | 85.92 / 74.42 | +1.34 | +3.98 | +0.80 | +0.99 |
| eurosat-gtsrb-sun397-svhn | layer_fair | 4 | 84.53 / 69.26 | 85.84 / 75.07 | +1.31 | +5.81 | +0.88 | +1.24 |

Average extra test gain over 6b: +1.74 mean / +1.44 worst.

Chosen vectors:

**dtd-eurosat-gtsrb-svhn — wide_mean**

| task | early | mid | late |
|---|---:|---:|---:|
| dtd | 0 | 0.7 | 0.6 |
| eurosat | 0.6 | 1.1 | 1 |
| gtsrb | 0.3 | 0.6 | 1 |
| svhn | 0.9 | 0.4 | 0 |

**dtd-eurosat-gtsrb-svhn — wide_fair**

| task | early | mid | late |
|---|---:|---:|---:|
| dtd | 0 | 0.6 | 0.8 |
| eurosat | 0.5 | 1 | 1.3 |
| gtsrb | 0.4 | 0.6 | 1.1 |
| svhn | 1.1 | 0.4 | 0.1 |

**eurosat-gtsrb-mnist-svhn — wide_mean**

| task | early | mid | late |
|---|---:|---:|---:|
| eurosat | 0.3 | 1 | 1.4 |
| gtsrb | 0.3 | 0.5 | 1.1 |
| mnist | 0.3 | 0.5 | 0.7 |
| svhn | 0.8 | 0.6 | 0 |

**eurosat-gtsrb-mnist-svhn — wide_fair**

| task | early | mid | late |
|---|---:|---:|---:|
| eurosat | 0.4 | 1 | 1.3 |
| gtsrb | 0.3 | 0.6 | 1 |
| mnist | 0.5 | 0.4 | 0.6 |
| svhn | 0.9 | 0.6 | 0.3 |

**eurosat-gtsrb-sun397-svhn — wide_mean**

| task | early | mid | late |
|---|---:|---:|---:|
| eurosat | 0.4 | 0.8 | 1.2 |
| gtsrb | 0.3 | 0.5 | 1 |
| sun397 | 0 | 0 | 1 |
| svhn | 0.8 | 0.6 | 0 |

**eurosat-gtsrb-sun397-svhn — wide_fair**

| task | early | mid | late |
|---|---:|---:|---:|
| eurosat | 0.3 | 0.7 | 1.3 |
| gtsrb | 0.3 | 0.5 | 1 |
| sun397 | 0.2 | 0 | 1 |
| svhn | 0.8 | 0.6 | 0 |

