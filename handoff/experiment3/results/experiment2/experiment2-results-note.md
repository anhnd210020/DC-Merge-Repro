# Experiment 2 results

Run completed 2026-10-16 with the supplied official snapshot and one seed. Compare and decomposition stages both completed; all 23 compare rows and 12 decomposition rows are saved.

## Checks

- `all8 / dc_official` selected α = 0.4 and reached **73.856% mean normalized test accuracy**, matching the 73.86% target.
- `all8 / ct` selected γ = 0.4 and scored **72.95%**, 0.91 pp below `dc_official`, a close and slightly lower result.
- All six official-operator lane checks report `max_abs_diff_vs_official = 0.0`. Maximum lane-sum relative error across all rows was 1.71e-7; maximum CT basis orthonormality error was 2.63e-6 (the limit is 1e-4).
- The decomposition sum identity held to within 1.5e-14 pp in all 12 tables. Per the guide, the official operator’s “merged alone” frame/harm values are not interpreted; the causal decomposition below uses `rank`. No run errors or assertion failures appeared.
- The archived Experiment 1 per-task validation outputs were not present in the shared results directory, so the direct historical α=0.4 comparison could not be made. As an internal cross-check, `dc_official` α=0.4 validation accuracies from 2a and the full-lane official α=0.4 configurations in 2b differed by at most 0.32 pp across the three core contexts, consistent with GPU evaluation variation.

## 2a: CT-Merging and EuroSAT

CT-Merging increases EuroSAT test accuracy over `dc_rank` by **5.85, 15.96, and 7.07 pp** in the three core contexts. The guide's joint rescue condition also asks for a better worst-task score. That condition is not met: the worst task is SVHN under CT in each context, and its normalized accuracy is 2.22–6.92 pp below the `dc_rank` worst-task score. The mean normalized accuracy is also 1.81–3.59 pp lower under CT. CT therefore rescues EuroSAT's task accuracy while shifting the bottleneck to SVHN and lowering the context mean.

| context | selected α (`dc_rank`) | selected γ (`ct`) | EuroSAT: rank → CT | mean norm: rank → CT | worst norm: rank → CT |
|---|---:|---:|---:|---:|---:|
| `dtd-eurosat-gtsrb-svhn` | 0.5 | 0.7 | 61.07 → 66.93 (+5.85 pp) | 71.22 → 69.41 | 62.20 → 55.28 |
| `eurosat-gtsrb-mnist-svhn` | 0.6 | 0.6 | 54.56 → 70.52 (+15.96 pp) | 73.83 → 70.24 | 55.56 → 53.34 |
| `eurosat-gtsrb-sun397-svhn` | 0.6 | 0.6 | 62.33 → 69.41 (+7.07 pp) | 76.65 → 73.44 | 63.48 → 61.09 |

Across the seven four-task contexts, mean normalized accuracy is 80.87 for `dc_rank` and 78.70 for CT; mean EuroSAT accuracy is 72.38 and 77.50, respectively. The complete table is in `compare_summary.md`.

## 2b: selected-α `rank` decomposition

Accuracy values and harm terms are in percentage points. Neighbor share includes all single, pair, and triple neighbor terms; the interaction column is the GTSRB+SVHN pair term.

| context | α | frame cost | GTSRB alone | SVHN alone | GTSRB+SVHN interaction | total harm vs merged-alone | neighbor share | pair interaction / harm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dtd-eurosat-gtsrb-svhn` | 0.5 | 2.44 | 0.52 | 6.30 | 9.89 | 28.44 | 91.4% | 34.8% |
| `eurosat-gtsrb-mnist-svhn` | 0.6 | 3.07 | 1.15 | 5.19 | 16.15 | 40.26 | 92.4% | 40.1% |
| `eurosat-gtsrb-sun397-svhn` | 0.6 | 2.59 | 1.22 | 7.04 | 18.22 | 32.11 | 92.0% | 56.7% |

The decomposition supports a neighbor-driven account in all three contexts. Frame costs are small, while the GTSRB+SVHN interaction alone accounts for at least 34.8% of total harm in each context. The full term breakdown is in `decompose_summary.md`.

## 2c: dose result

No tested **neighbor-only** partial dose met the guide's joint criterion of EuroSAT gain at least twice the summed losses on the other tasks and a positive mean. The best neighbor-only setting by gain/loss ratio was `rank`, `eurosat-gtsrb-sun397-svhn`, selected α = 0.6, with SVHN at λ = 0.75: EuroSAT **+11.11 pp**, other-task losses **7.84 pp**, mean of four **+1.85 pp** (1.42×). Turning both GTSRB and SVHN to λ = 0.25 could recover more EuroSAT—for example +34.63 pp in the selected-α MNIST context—but cost 63.53 pp across the other tasks and changed the mean by −7.22 pp.

The simpler own-lane boost did pass the 2× test in α=0.4 controls. The largest mean gain was `official`, `eurosat-gtsrb-mnist-svhn`, α = 0.4, EuroSAT lane ×1.5: EuroSAT **+10.48 pp**, other-task losses **1.00 pp**, mean of four **+2.46 pp**. On `rank` in `dtd-eurosat-gtsrb-svhn` at α = 0.4, EuroSAT ×1.5 gave **+16.22 pp**, losses **6.98 pp**, and mean **+2.31 pp**. No selected-α own-lane boost met the 2× criterion. The tested neighbor rescaling alone therefore misses the target; at fixed α=0.4, boosting the victim's lane is the cheaper successful lever in these validation runs.

## Files

- [Compare summary](compare/compare_summary.md) and [compare CSV](compare/compare.csv)
- [Decomposition and dose summary](decompose/decompose_summary.md) and [decompose CSV](decompose/decompose.csv)
- [Compare log](../experiment2-compare.log) and [decompose log](../experiment2-decompose.log)
- Per-configuration JSON files are in `compare/` and `decompose/`.

This is a one-seed validation experiment across three victim contexts; it is sufficient for the stated go/no-go check and not a publication-level estimate.
