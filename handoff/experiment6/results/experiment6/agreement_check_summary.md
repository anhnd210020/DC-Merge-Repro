# Experiment 6a check: does agreement with the experts pick high accuracy? (validation, Exp. 3 vectors)

Proxies: 500 unlabeled validation images per task (seed 0, Experiment 5's subset); labels are not read by the proxy. Accuracies: Experiment 3's cached validation accuracies, normalized by the single-task (expert) accuracy. **Not data-free:** needs unlabeled images and the experts.

| context | proxy | vectors | ρ(−mean loss, mean acc) | ρ(−max loss, worst acc) | mean acc: best / pick / dc_rank | worst acc: best / fair pick / dc_rank | pick > dc_rank? |
|---|---|---:|---:|---:|---|---|:---:|
| dtd-eurosat-gtsrb-svhn | kl | 78 | -0.20 | -0.20 | 74.55 / 66.71 / 71.16 | 68.32 / 56.07 / 62.08 | no |
| dtd-eurosat-gtsrb-svhn | top1 | 78 | 0.95 | 0.52 | 74.55 / 74.55 / 71.16 | 68.32 / 63.54 / 62.08 | yes |
| dtd-eurosat-gtsrb-svhn | cos | 78 | -0.25 | -0.14 | 74.55 / 61.02 / 71.16 | 68.32 / 56.07 / 62.08 | no |
| dtd-eurosat-gtsrb-svhn | entropy (Exp 5) | 78 | 0.35 | 0.09 | 74.55 / 68.72 / 71.16 | 68.32 / 54.12 / 62.08 | no |
| eurosat-gtsrb-mnist-svhn | kl | 57 | -0.14 | 0.39 | 77.47 / 71.98 / 73.74 | 69.32 / 53.38 / 53.70 | no |
| eurosat-gtsrb-mnist-svhn | top1 | 57 | 0.98 | 0.99 | 77.47 / 77.47 / 73.74 | 69.32 / 69.32 / 53.70 | yes |
| eurosat-gtsrb-mnist-svhn | cos | 57 | -0.29 | 0.06 | 77.47 / 69.06 / 73.74 | 69.32 / 54.00 / 53.70 | no |
| eurosat-gtsrb-mnist-svhn | entropy (Exp 5) | 57 | 0.43 | 0.12 | 77.47 / 68.89 / 73.74 | 69.32 / 48.33 / 53.70 | no |
| eurosat-gtsrb-sun397-svhn | kl | 82 | 0.27 | 0.38 | 79.60 / 76.54 / 75.83 | 70.57 / 59.86 / 62.04 | yes |
| eurosat-gtsrb-sun397-svhn | top1 | 82 | 0.97 | 0.99 | 79.60 / 79.49 / 75.83 | 70.57 / 69.34 / 62.04 | yes |
| eurosat-gtsrb-sun397-svhn | cos | 82 | 0.10 | 0.38 | 79.60 / 76.54 / 75.83 | 70.57 / 60.89 / 62.04 | yes |
| eurosat-gtsrb-sun397-svhn | entropy (Exp 5) | 82 | 0.01 | -0.28 | 79.60 / 70.77 / 75.83 | 70.57 / 47.64 / 62.04 | no |

Proxy losses: `kl` = KL(expert || merged); `top1` = 1 - top-1 agreement; `cos` = 1 - feature cosine. The entropy rows are Experiment 5's reference on the same vectors.

## Gate (pre-registered, core contexts, validation)

**PASS** — top1: rho_mean >= 0.5 and pick > dc_rank in all three core contexts

GATE PASSES -> run the coordinate search with the selected proxy (core, then four)

| proxy | ρ_mean per core context | min ρ_mean | pick beats dc_rank | passes | fails (per-proxy reading) |
|---|---|---:|---|:---:|:---:|
| kl | -0.20, -0.14, 0.27 | -0.20 | no, no, yes | no | yes |
| top1 | 0.95, 0.98, 0.97 | 0.95 | yes, yes, yes | yes | no |
| cos | -0.25, -0.29, 0.10 | -0.29 | no, no, yes | no | yes |

FAIL uses the literal reading (*every* proxy has a core ρ_mean ≤ 0.3, *or* *every* proxy's pick is below dc_rank in all three core contexts). Under the stricter per-proxy reading (each proxy fails at least one of the two), the gate would not fail.
