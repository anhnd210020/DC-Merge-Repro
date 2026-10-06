# Final experiment report: task merge demand prediction

**Final-test completion:** 2026-10-06 UTC

**Decision basis:** frozen validation selection in `validation_model_selection_frozen.md`; no final-test result was used to choose or change a model.

**Reported prediction metrics:** `aggregate_final_test` from `output_retry2/final_test/grouped_prediction_results.json` (not `aggregate_oof`).

## Research-question findings

1. **Is demand heterogeneous across tasks? Yes.** Between-task sums of squares account for 0.562 of total density demand variation, 0.902 for rank, and 0.623 for scale across eight tasks and 118 task/context observations. Rank demand is especially task-specific; density and scale also vary substantially.
2. **Does demand depend on context? Yes.** The within-task/context sums of squares are 811.3 (density), 184.7 (rank), and 1,088.8 (scale). The density and scale results show substantial variation for the same task across its contexts; rank has less context variation relative to its strong between-task component, but is not context-invariant.
3. **Do the three resources reveal one shared demand structure? No.** Final-test global Spearman correlations are density–rank -0.371, density–scale 0.781, and rank–scale -0.716 (n=118 each). The mixed signs support resource-specific or multidimensional demand rather than one universal scalar.
4. **Do pre-evaluation structural features predict unseen-task demand? Partly, with resource-specific strength.** The frozen Phase A leave-one-task-out features achieve final-test Spearman 0.825 for density, 0.905 for rank, and 0.860 for scale (n=56 each). Rank and scale remain strong in Phases B/C. Density falls to 0.247 in C (n=22), so the frozen density signal is not stable across the larger-context check. These scores evaluate the validation-frozen models; no feature was reselected from final-test results.

## Aggregate final-test prediction metrics

The frozen single-feature Ridge models are `module_norm_mean` (density), `energy_rank_90` (rank), and `spectral_entropy` (scale), as fixed by validation. The constant baseline predicts demand 1.0. Pearson and Spearman are undefined for this constant prediction.

| Resource | Phase (n) | Candidate | normalized_demand_mae | pairwise_ordering_accuracy | Pearson | Spearman |
|---|---:|---|---:|---:|---:|---:|
| Density | A (56) | Frozen: `module_norm_mean` | 0.198 | 0.808 | 0.523 | 0.825 |
| Density | A (56) | `composite_ridge` | 0.283 | 0.709 | 0.473 | 0.634 |
| Density | A (56) | `constant_one_baseline` | 0.260 | 0.500 | undefined | undefined |
| Density | B (40) | Frozen: `module_norm_mean` | 0.148 | 0.726 | 0.725 | 0.731 |
| Density | B (40) | `composite_ridge` | 0.129 | 0.803 | 0.848 | 0.831 |
| Density | B (40) | `constant_one_baseline` | 0.477 | 0.500 | undefined | undefined |
| Density | C (22) | Frozen: `module_norm_mean` | 0.214 | 0.558 | 0.308 | 0.247 |
| Density | C (22) | `composite_ridge` | 0.187 | 0.753 | 0.604 | 0.692 |
| Density | C (22) | `constant_one_baseline` | 0.534 | 0.500 | undefined | undefined |
| Rank | A (56) | Frozen: `energy_rank_90` | 0.101 | 0.854 | 0.923 | 0.905 |
| Rank | A (56) | `composite_ridge` | 0.194 | 0.704 | 0.660 | 0.610 |
| Rank | A (56) | `constant_one_baseline` | 0.432 | 0.500 | undefined | undefined |
| Rank | B (40) | Frozen: `energy_rank_90` | 0.088 | 0.842 | 0.883 | 0.892 |
| Rank | B (40) | `composite_ridge` | 0.083 | 0.860 | 0.885 | 0.905 |
| Rank | B (40) | `constant_one_baseline` | 0.310 | 0.500 | undefined | undefined |
| Rank | C (22) | Frozen: `energy_rank_90` | 0.089 | 0.857 | 0.918 | 0.913 |
| Rank | C (22) | `composite_ridge` | 0.100 | 0.892 | 0.902 | 0.940 |
| Rank | C (22) | `constant_one_baseline` | 0.381 | 0.500 | undefined | undefined |
| Scale | A (56) | Frozen: `spectral_entropy` | 0.144 | 0.825 | 0.844 | 0.860 |
| Scale | A (56) | `composite_ridge` | 0.231 | 0.740 | 0.710 | 0.682 |
| Scale | A (56) | `constant_one_baseline` | 0.381 | 0.500 | undefined | undefined |
| Scale | B (40) | Frozen: `spectral_entropy` | 0.124 | 0.776 | 0.859 | 0.791 |
| Scale | B (40) | `composite_ridge` | 0.083 | 0.799 | 0.940 | 0.817 |
| Scale | B (40) | `constant_one_baseline` | 0.448 | 0.500 | undefined | undefined |
| Scale | C (22) | Frozen: `spectral_entropy` | 0.116 | 0.844 | 0.813 | 0.871 |
| Scale | C (22) | `composite_ridge` | 0.108 | 0.844 | 0.872 | 0.879 |
| Scale | C (22) | `constant_one_baseline` | 0.450 | 0.500 | undefined | undefined |

## Decision

**Weak Go — pursue resource-specific, multidimensional demand estimation.** This follows the validation-frozen feature choices and the Experiment Card's Weak Go framing when demand can be predicted by resource but the resources do not share a universal scalar. The final test supports rank and scale strongly in the primary Phase A setting and supports the frozen density feature there, while the density Phase C result limits the generalization claim. A subsequent allocator study should retain separate resource demands and prospectively test the frozen predictors under a fixed budget. Do not claim one context-independent scalar demand.

## Completion and integrity

- The final-test wrapper exit file contains `0`. The final-test completion manifest records 1,360 configurations; there are 1,360 run directories, 1,360 `result.json` files, and 1,360 `unit.json` files. Validation has the same run and file counts.
- Validation and final-test completion manifests each list 2,733 files. Every listed file is present and matches its SHA-256. Both per-split completion seals, both preflight and DONE marker seals, the final-test top-level completion marker, and `ALL_DONE.json` verify. The markers record real mode and passed preflight.
- Validation and final-test reports are present, with five figures for each split. All expected status markers are present; no RUNNING or FAILED marker was found.
- Integrity caveat: `final_test/COMPLETE.json` labels 96 rows as `focal_context_rows`, the A+B subtotal (56+40). The demand table contains 354 final-test rows (118 per resource), including the 22 Phase C focal rows; the run inventory and checksums also include Phase C. The broad completion-field name therefore understates the full row count, but Phase C is present.

## Limitations

- Only eight tasks were evaluated. Phase A has 56 observations across eight held-out tasks; Phases B and C have 40 rows across 25 contexts and 22 rows across nine contexts. The smaller B/C samples, especially C, limit precision and broad-context claims.
- Aggregate prediction metrics are pooled and have no reported confidence intervals; observations are grouped by task/context and are not independent samples.
- The frozen density feature weakens in Phase C, where the composite is stronger. Because selection was frozen on validation, this is reported as a limitation rather than used to switch models.
- The audit notes incomplete input identity for archived rank raw observations and invalid archived density curves because per-module rounding collapsed requested levels. Synthetic results are not scientific evidence.
- The validation review notes that the Experiment Card did not fully specify a winner/tie-break procedure among the constant, 30 single-feature models, and eligible composite. The frozen validation note records the accepted Phase A Spearman selection rule and its tie decisions; this report preserves that decision.

## Audit references

The paths below resolve relative to this report in the full audit archive after extraction.

- Frozen selection: `../../validation_model_selection_frozen.md`
- Validation selection review: `../../validation_model_selection_review.md`
- Validation report: `../../output_retry2/validation/REPORT.md`
- Final-test report and metrics: `../../output_retry2/final_test/REPORT.md` and `../../output_retry2/final_test/grouped_prediction_results.json`
- Experiment Card: `../docs/experiment_card_pre_evaluation_task_merge_demand.docx`
