# Experiment 3: per-task coefficient oracle — mathematical report

## Question and result

Experiment 3 asks whether task-specific lane coefficients selected on validation can outperform Experiment 2a’s single `dc_rank` coefficient on held-out test data, considering both average normalized accuracy and the worst task. The three EuroSAT “core” contexts are the decisive subset; the seven-context set adds four controls.

The selected `oracle_fair` result is **SMALL GAIN** for the core set and for all seven contexts. It improves mean test accuracy by about 2.72 pp in core and 1.63 pp across seven contexts, but its worst-task gain is only 0.13 pp and 1.16 pp respectively, below the guide’s +3 pp headroom criterion. `oracle_mean` improves the mean more, while lowering average worst-task accuracy. The guide therefore says to report and discuss the result before deciding; it does not call for building the data-free predictor yet.

## Merge decomposition

For each task \(i\), let \(\Delta_i\) be its fine-tuned parameter update. The experiment retains the task’s 16 leading singular directions and applies DC-Merge linear singular-value smoothing. In a context, the smoothed updates are projected into a shared left/right cover frame \(U,V\):

\[
M_i = U^\top \widetilde{\Delta}_i V.
\]

The projected matrices are sparsified with the experiment’s top-16 rule and combined with TIES into aggregate \(A\). A structural block-diagonal mask \(S\) removes off-lane blocks. For the context’s diagonal block \(B_i\), define the task lane

\[
L_i = U_{:,B_i} A_{B_i,B_i} V^\top_{B_i,:},\qquad
\Delta_{\rm merge}=U(A\odot S)V^\top=\sum_i L_i.
\]

The candidate weights are

\[
W(c)=W_0+\sum_i c_iL_i.
\]

Setting every \(c_i=\alpha\) returns the corresponding global-\(\alpha\) `dc_rank` merge. The implementation preserves the direct merged-matrix arithmetic path when all coefficients are equal, avoiding a different floating-point summation order. For unequal coefficients it composes the task lane matrices. The maximum measured relative lane-sum error was \(1.7320\times10^{-7}\).

For task \(i\), normalized accuracy is \(n_i(c)=100\,a_i(c)/r_i\), where \(a_i\) is merged accuracy and \(r_i\) is the single-task reference. The objective summaries are

\[
\operatorname{mean}(c)=\frac{1}{T}\sum_i n_i(c),\qquad
\operatorname{worst}(c)=\min_i n_i(c).
\]

## Validation search and test protocol

Each search starts at \(c_i=\alpha\), where \(\alpha\) is the selected Experiment 2a `dc_rank` coefficient. For each task in fixed context order, it tries every value in \(\{0.3,0.4,\ldots,1.0\}\) plus the starting \(\alpha\), accepting strict improvements. It makes at most two passes and stops early when a pass accepts no change. Both objectives use the same validation cache:

- `oracle_mean` greedily maximizes mean normalized validation accuracy.
- `oracle_fair` greedily maximizes that mean while requiring worst-task normalized validation accuracy to be at least the baseline worst-task value.

The test split is used after coefficient selection: the `dc_rank`, `oracle_mean`, and `oracle_fair` vectors are evaluated and CT-Merging’s test values are read from Experiment 2a. Search caches and complete per-context search logs are included.

## Aggregate findings

| Set | Method | Δ mean test norm (pp) | Δ worst test norm (pp) | Guide verdict |
|---|---|---:|---:|---|
| 3 core contexts | `oracle_mean` | +2.93 | -3.07 | SMALL GAIN: below thresholds, discuss |
| 3 core contexts | `oracle_fair` | +2.72 | +0.13 | SMALL GAIN: below thresholds, discuss |
| 7 contexts | `oracle_mean` | +1.75 | -2.32 | SMALL GAIN: below thresholds, discuss |
| 7 contexts | `oracle_fair` | +1.63 | +1.16 | SMALL GAIN: below thresholds, discuss |

The guide calls **HEADROOM** only when the chosen oracle’s mean gain is at least +1 pp and worst-task gain at least +3 pp. `oracle_fair` meets the mean threshold in both sets, but not the worst-task threshold. `oracle_mean` misses the fairness threshold and has a negative worst-task delta. The resulting verdict is SMALL GAIN for both sets.

The per-context search timers sum to 431.8 minutes (about 7 h 12 min; core 3 h 09 min and four 4 h 03 min), excluding setup and baseline-replay work. Final verification recorded 60.77 GB used and 25.13 GB free on the 85.90 GB filesystem.

## Per-context `oracle_fair` vectors

Coefficients are shown in task order. Direction is relative to that context’s single `dc_rank` coefficient \(\alpha\). Deltas are test normalized-accuracy percentage points relative to the current-environment `dc_rank` replay.

| Context | \(\alpha\) | Chosen vector \(c\) | Change vs \(\alpha\) | Δ mean | Δ worst |
|---|---:|---|---|---:|---:|
| dtd-eurosat-gtsrb-svhn | 0.5 | dtd=0.3, eurosat=0.8, gtsrb=0.7, svhn=0.5 | up: eurosat, gtsrb; down: dtd; same: svhn | +2.95 | -0.19 |
| eurosat-gtsrb-mnist-svhn | 0.6 | eurosat=1, gtsrb=0.7, mnist=0.4, svhn=0.5 | up: eurosat, gtsrb; down: mnist, svhn | +3.30 | +0.76 |
| eurosat-gtsrb-sun397-svhn | 0.6 | eurosat=1, gtsrb=0.7, sun397=0.3, svhn=0.6 | up: eurosat, gtsrb; down: sun397; same: svhn | +1.93 | -0.18 |
| stanford_cars-eurosat-gtsrb-mnist | 0.6 | stanford_cars=0.3, eurosat=0.9, gtsrb=0.7, mnist=0.5 | up: eurosat, gtsrb; down: stanford_cars, mnist | +2.50 | +5.76 |
| stanford_cars-eurosat-gtsrb-resisc45 | 0.7 | stanford_cars=0.7, eurosat=0.7, gtsrb=0.7, resisc45=0.7 | same: stanford_cars, eurosat, gtsrb, resisc45 | +0.00 | +0.00 |
| stanford_cars-eurosat-mnist-resisc45 | 0.6 | stanford_cars=0.6, eurosat=0.6, mnist=0.6, resisc45=0.6 | same: stanford_cars, eurosat, mnist, resisc45 | +0.00 | +0.00 |
| stanford_cars-eurosat-mnist-sun397 | 0.7 | stanford_cars=0.6, eurosat=0.7, mnist=0.6, sun397=0.5 | down: stanford_cars, mnist, sun397; same: eurosat | +0.71 | +1.95 |

## Baseline provenance and validity

The primary comparison is against the current-environment direct `dc_rank` replay at the archived selected coefficient. For all seven contexts the Experiment 3 baseline reproduced that replay within 0.000 pp on validation and test. The archived Experiment 2 `compare/` files are preserved unchanged. The lane-sum reconstruction error stayed below \(1.74\times10^{-7}\), the validation objective constraints passed, all seven result records and 28 CSV rows are present, and the core/four wrappers both exited successfully.

The separate Oct 16 archive comparison is a warning: the largest absolute task-level difference is +0.158353 pp on validation (GTSRB, `dtd-eurosat-gtsrb-svhn`) and -0.148148 pp on test (EuroSAT, `stanford_cars-eurosat-gtsrb-resisc45`). A repeat of identical current-GPU inputs was exact; the historical 0.32 pp note compared different arithmetic paths. This is consistent with sensitivity of near-boundary predictions, but it does not prove that the historical discrepancy is normal stochastic run-to-run GPU variation. The original runtime/hardware, archived per-example predictions/logits, and complete image-byte hashes remain unavailable. These uncertainties concern exact historical reproduction; the seven Experiment 3 comparisons are internally valid against their current-environment baseline.

## Interpretation and limits

This is a validation-label oracle, not a deployable coefficient predictor. The greedy coordinate search can miss better vectors; two passes over the stated grid do not prove a global optimum or prove that no better vector exists. Results use one checkpoint set and the selected four-task contexts in a single run. Treat modest test gains as exploratory and do not interpret the validation oracle as an unbiased estimate of a predictor’s generalization. The guide’s next step for SMALL GAIN is to report and discuss before deciding.
