# Experiment 4 report: task coefficient transfer

## Result

Experiment 4 tests whether validation-selected task coefficients transfer across
contexts. It reuses the Experiment 3 `oracle_fair` vectors, averages each task's
`c_i / alpha` ratio over its other contexts, and multiplies that ratio by the
held-out context's own alpha. The held-out context was excluded from its ratio;
test scores were not used to form coefficients. There was no search: the run made
7 validation and 7 test evaluations and left the Experiment 3 baseline and oracle
records unchanged.

Seven-context test averages (normalized accuracy, percent):

| Method | Mean | Worst task | Δ mean vs `dc_rank` | Δ worst vs `dc_rank` |
|---|---:|---:|---:|---:|
| `dc_rank` | 80.866 | 72.094 | — | — |
| `loco_transfer` | 81.230 | 72.161 | +0.364 pp | +0.068 pp |
| `oracle_fair` | 82.492 | 73.251 | +1.626 pp | +1.157 pp |

On validation, `loco_transfer` averages +0.651 pp mean and +1.405 pp worst-task
over `dc_rank`; `oracle_fair` averages +1.937 pp and +2.335 pp. The transfer rule
recovers about 22% of the oracle's mean test gain and about 6% of its worst-task
gain. Results vary by context: the transferred vector improves test mean in four
of seven contexts and worst-task accuracy in five of seven. All three core
contexts improve both metrics, while two of the four additional contexts show
substantial losses.

## Held-out vectors and test changes

Coefficients are shown in task order from each context's result JSON. Deltas are
percentage points versus that context's `dc_rank`.

| Context | `loco_transfer` coefficients | Δ mean | Δ worst |
|---|---|---:|---:|
| `dtd-eurosat-gtsrb-svhn` | dtd 0.5000, eurosat 0.6528, gtsrb 0.5625, svhn 0.4583 | +2.320 | +0.187 |
| `eurosat-gtsrb-mnist-svhn` | eurosat 0.7767, gtsrb 0.7100, mnist 0.5381, svhn 0.6000 | +1.587 | +6.526 |
| `eurosat-gtsrb-sun397-svhn` | eurosat 0.7767, gtsrb 0.7100, sun397 0.4286, svhn 0.5500 | +2.114 | +1.678 |
| `stanford_cars-eurosat-gtsrb-mnist` | stanford_cars 0.5714, eurosat 0.7933, gtsrb 0.7100, mnist 0.5048 | +1.557 | +7.582 |
| `stanford_cars-eurosat-gtsrb-resisc45` | stanford_cars 0.5500, eurosat 0.9839, gtsrb 0.8575, resisc45 0.7000 | -2.754 | -10.003 |
| `stanford_cars-eurosat-mnist-resisc45` | stanford_cars 0.4714, eurosat 0.8433, mnist 0.4714, resisc45 0.6000 | -1.589 | -6.185 |
| `stanford_cars-eurosat-mnist-sun397` | stanford_cars 0.5833, eurosat 0.9839, mnist 0.5833, sun397 0.3500 | -0.688 | +0.690 |

DTD is unsupported by cross-context evidence in this selected set: no other context
contains DTD, so the method uses its flagged neutral ratio. EuroSAT has six other
contexts; GTSRB four; MNIST three; SVHN two; and Stanford Cars, RESISC45, and SUN397
only one or three, as listed in `results/transfer_summary.md`.

## Checks and limitations

- All seven per-context JSON records agree with `aggregate.json`; the reported
  validation/test averages recompute from those seven records.
- All seven context directories contain validation and test caches. The supplied
  `SHA256SUMS` covers every packaged file except itself.
- The baseline is the saved Experiment 3 baseline evaluated in this environment.
  The Oct 16 archived-versus-current discrepancy is documented in the Experiment
  3 handoff; this experiment does not resolve it. Comparisons here are internal to
  the current-environment Experiment 3 baseline and its saved oracle.
- This is a transfer check for the selected seven contexts, not a new coefficient
  search or evidence that coefficients transfer universally. Support counts differ
  by task, and DTD has no cross-context support.

See `results/transfer_summary.md` for the full validation table, all context-level
comparisons against both `dc_rank` and `oracle_fair`, and support counts.
