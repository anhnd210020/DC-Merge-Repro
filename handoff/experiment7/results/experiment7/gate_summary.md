# Experiment 7a: agreement gate on the per-depth family (VALIDATION, Exp 6 6b vectors)

Agreement = top-1 agreement with each task's expert on the 500 unlabeled validation images, seed 0 (labels never read). Accuracies = Exp 6's cached 6b validation accuracies, normalized. No new labelled evaluation.

| context | vectors | ρ(mean agree, mean acc) | ρ(min rel, worst acc) | ρ(min raw agree, worst acc) | dc_rank mean / worst | best cached mean / best cached worst |
|---|---:|---:|---:|---:|---|---|
| dtd-eurosat-gtsrb-svhn | 341 | 0.98 | 0.24 | 0.11 | 71.13 / 62.08 | 78.84 / 72.52 |
| eurosat-gtsrb-mnist-svhn | 360 | 0.98 | 0.47 | 0.98 | 73.74 / 53.66 | 83.08 / 77.59 |
| eurosat-gtsrb-sun397-svhn | 360 | 0.99 | 0.67 | 0.94 | 75.86 / 62.00 | 84.55 / 76.84 |

Picks (validation mean / worst normalized; ✓ = beats dc_rank):

| context | rule | vector | val mean | val worst | beats mean | beats worst |
|---|---|---|---:|---:|:---:|:---:|
| dtd-eurosat-gtsrb-svhn | (i) max mean agree | `0.3,0.6,0.3,0.4,1,1,0.3,0.7,0.9,0.9,0.4,0.3` | 78.72 | 67.14 | ✓ | ✓ |
| dtd-eurosat-gtsrb-svhn | (ii) max mean agree, rel ≥ 1 | `0.3,0.6,0.8,0.4,1,0.9,0.4,0.7,0.8,0.9,0.4,0.3` | 77.62 | 66.38 | ✓ | ✓ |
| dtd-eurosat-gtsrb-svhn | (iii) max min rel | `0.3,0.6,0.9,0.4,1,0.9,0.4,0.7,0.8,0.9,0.4,0.3` | 77.42 | 66.46 | ✓ | ✓ |
| dtd-eurosat-gtsrb-svhn | 6b oracle layer_mean (labels) | `0.3,0.6,0.3,0.4,1,1,0.3,0.7,0.8,0.9,0.4,0.3` | 78.84 | 66.76 | | |
| dtd-eurosat-gtsrb-svhn | 6b oracle layer_fair (labels) | `0.3,0.5,0.4,0.6,1,1,0.4,0.7,0.8,0.9,0.4,0.3` | 78.63 | 64.61 | | |
| eurosat-gtsrb-mnist-svhn | (i) max mean agree | `0.5,1,1,0.3,0.5,1,0.4,0.3,0.7,1,0.6,0.3` | 82.95 | 76.42 | ✓ | ✓ |
| eurosat-gtsrb-mnist-svhn | (ii) max mean agree, rel ≥ 1 | `0.5,1,1,0.3,0.5,1,0.4,0.3,0.9,0.8,0.6,0.3` | 82.51 | 74.79 | ✓ | ✓ |
| eurosat-gtsrb-mnist-svhn | (iii) max min rel | `0.3,1,1,0.3,0.5,1,0.4,0.3,1,0.6,0.6,0.6` | 80.28 | 73.50 | ✓ | ✓ |
| eurosat-gtsrb-mnist-svhn | 6b oracle layer_mean (labels) | `0.5,1,1,0.3,0.5,1,0.4,0.3,0.7,0.9,0.6,0.3` | 83.08 | 76.02 | | |
| eurosat-gtsrb-mnist-svhn | 6b oracle layer_fair (labels) | `0.6,1,1,0.3,0.5,1,0.5,0.4,0.6,0.9,0.6,0.3` | 83.02 | 75.03 | | |
| eurosat-gtsrb-sun397-svhn | (i) max mean agree | `0.4,1,1,0.4,0.6,1,0.8,0.3,0.8,0.8,0.5,0.3` | 84.23 | 69.28 | ✓ | ✓ |
| eurosat-gtsrb-sun397-svhn | (ii) max mean agree, rel ≥ 1 | `0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6` | 75.86 | 62.00 | ✗ | ✗ |
| eurosat-gtsrb-sun397-svhn | (iii) max min rel | `0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6` | 75.86 | 62.00 | ✗ | ✗ |
| eurosat-gtsrb-sun397-svhn | 6b oracle layer_mean (labels) | `0.4,1,1,0.4,0.6,1,0.8,0.3,0.9,0.9,0.5,0.3` | 84.55 | 70.18 | | |
| eurosat-gtsrb-sun397-svhn | 6b oracle layer_fair (labels) | `0.3,1,1,0.3,0.6,1,0.4,0.3,0.9,1,0.4,0.3` | 84.30 | 68.69 | | |

## Gate (pre-registered, core contexts, validation)

**IN_BETWEEN** — neither the PASS nor the FAIL condition holds

IN BETWEEN -> run 7c/7e on core only; label the result PARTIAL at most

Alternative readings: PASS if (ii)/(iii) may differ per context = False; FAIL if any core context has no worst-task win = False.
