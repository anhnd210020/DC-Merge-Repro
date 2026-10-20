# DC-Merge Experiments 1–5: implemented methods, mathematics, and evidence

**Supervisor research report**  
**Prepared:** 20 October 2026  
**Evidence boundary:** saved repository code, saved result records, run metadata, and the baseline audit. No experiment was run to prepare this report.

## Executive summary

This five-experiment sequence moves from reproducing the official DC-Merge merge to asking whether task-specific coefficients can reduce cross-task damage, whether those coefficients transfer, and whether unlabeled images can select them.

- A surviving official-checkpoint reproduction record reports **α = 0.4**, **64.149% mean test accuracy**, and **73.879% mean normalized test accuracy**, close to the paper target of 64.17% / 73.90%. The complete original Experiment 1 per-task validation output and standalone run log are unavailable; the repository explicitly records that gap. A separate self-trained checkpoint result is not an Experiment 1 result and is not used below. ([reproduction comparison](../../../reproduction/selftrained_b32_r16_8task/results/reproduction_comparison.md), [artifact availability note](../../experiments1-4-diagnostics/README.md), [Experiment 2 note](../../experiment3/results/experiment2/experiment2-results-note.md))
- In the **archived 16 October Experiment 2** results, `dc_rank` improved the mean normalized test score over `dc_official` by **4.65 pp** across the seven four-task contexts (80.87 vs. 76.22); CT-Merging increased EuroSAT accuracy in the three core contexts, but lowered the seven-context mean and mean worst-task score. These are archived values, not the current replay. ([archived comparison summary](../../experiment3/results/experiment2/archive_compare/compare_summary.md))
- Experiment 3’s validation-label oracle improved test mean normalized accuracy over the **current-environment direct `dc_rank` replay** by **+2.72 pp in the three core contexts** and **+1.63 pp over all seven** using `oracle_fair`. The corresponding worst-task gains were only **+0.13 pp** and **+1.16 pp**, so both sets receive the guide’s **SMALL GAIN** verdict. This is an oracle upper-bound experiment, not a deployable label-free method. ([oracle summary](../../experiment3/results/experiment3/oracle_summary.md), [final verification](../../experiment3/results/experiment3/final_verification.json))
- Experiment 4’s leave-one-context-out transfer kept **+0.36 pp mean / +0.07 pp worst** over current `dc_rank`, about **22% of the oracle’s seven-context mean gain**. It was harmful in both RESISC45 control contexts; the largest drop was **−10.00 pp** in mean worst-task score. ([transfer summary](../../experiment4/results/transfer_summary.md), [aggregate records](../../experiment4/results/aggregate.json))
- Experiment 5 tested the entropy signal on three core contexts and found weak correlations; the entropy-picked validation vectors scored below `dc_rank` in all three. The guide’s negative gate was met. **No Experiment 5 coefficient search, test evaluation, four-context phase, or seven-context method verdict exists.** ([signal-check summary](../results/check_summary.md), [check records](../results/check.json), [gate report](../report.md))

The strongest supported claim is therefore narrow: validation-selected per-task lane coefficients expose modest test headroom relative to current `dc_rank` in these contexts, but the tested cross-context transfer is weak and the tested entropy signal failed its pre-search gate. The historical Oct 16 versus current replay discrepancy remains unexplained; numerical sensitivity is plausible, but normal GPU variation has not been established.

## 1. What the current code actually does

### 1.1 Shared model and evaluation pipeline

The runs use a CLIP ViT-B/32 base, eight task-specific rank-16 LoRA adapters, one CLIP classification head per task, and the eight datasets Stanford Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, and SVHN. The runners load the official `vitB32_r16_8task` configuration, replace its asset paths with the run’s paths, load the heads and task data, and extract each LoRA update as `B @ A`. The merge is added to the base model state by `apply_merge`. The source code is in [`run_official.py`](../../experiment3/code/lane_harm/run_official.py#L55-L87) and [`ft_handlers.py`](../../experiment3/code/vision_lora_merge/ft_handlers.py#L71-L104).

The main context set has seven four-task combinations. The first three are the EuroSAT/GTSRB/SVHN “core” contexts; the remaining four are controls. The exact ordered definitions are in [`run_official.py`](../../experiment3/code/lane_harm/run_official.py#L27-L45). `evaluate_cliphead` normalizes image features and computes task-head scores before `argmax`; the accuracy evaluator runs without gradients and uses autocast. Experiment 5 separately uses a scale of 100 for entropy probabilities, because probability entropy depends on logit scale. ([accuracy evaluator](../../experiment3/code/vision_lora_merge/utils.py#L213-L260), [entropy implementation](../code/lane_harm/run_entropy.py#L84-L114))

### 1.2 Official DC-Merge, `dc_rank`, and lane analysis

**Official DC-Merge (`dc_official`).** For each module, the code takes each task’s leading 16 singular directions, replaces their singular values with a smoothed distribution whose ratio is capped at 5, constructs shared left/right cover frames, projects and sparsifies the task matrices, applies the TIES aggregation, masks out off-diagonal task blocks, and reconstructs the merged update. The official runner uses linear smoothing with `rho=5.0`. The implementation is [`merging_functions.py::dc_merge`](../../experiment3/code/vision_lora_merge/merging_functions.py#L154-L205), with the top-percent filter and TIES aggregation below it at [lines 208–232](../../experiment3/code/vision_lora_merge/merging_functions.py#L208-L232).

**`dc_rank`.** This is a clearly labeled experiment variant, not the official operator. It keeps exactly 16 largest-magnitude coefficients per task in the projected matrix instead of the official 0.1% filter. The rank-aware path is implemented in [`dc_trace.py`](../../experiment3/code/lane_harm/dc_trace.py#L12-L17) and [lines 29–99](../../experiment3/code/lane_harm/dc_trace.py#L29-L99). `run_week1.py::build_merge` selects official DC-Merge, `dc_rank`, or CT-Merging; the alpha sweep uses validation mean normalized accuracy and evaluates the selected coefficient once on test. ([merge selection and sweep](../../experiment3/code/lane_harm/run_week1.py#L51-L69), [compare loop](../../experiment3/code/lane_harm/run_week1.py#L75-L109))

**Lane analysis.** The lane code exposes the diagonal task blocks of the same full-context TIES aggregate. It checks that their sum reconstructs the merge, then tests targeted interventions such as protecting a task’s lane, fixing conflict, blocking injection, and reducing one or more companion lanes. [`lanes.py::LaneMerge`](../../experiment3/code/lane_harm/lanes.py#L24-L74) constructs and recomposes lanes; [`lane_harm.py`](../../experiment3/code/lane_harm/lane_harm.py#L1-L50) defines the diagnostics and interventions. `run_week1.py::cmd_decompose` uses all subsets of companion lanes and a Möbius inclusion–exclusion decomposition to separate frame cost, individual companion effects, and interactions. ([decomposition implementation](../../experiment3/code/lane_harm/run_week1.py#L153-L180), [evaluation and exact-sum check](../../experiment3/code/lane_harm/run_week1.py#L183-L276))

### 1.3 Experiment-by-experiment pipeline

| Experiment | Implemented operation | Label use and saved outcome |
|---|---|---|
| **1 — official reproduction and lane study** | Runs official DC-Merge on released checkpoints, selects α from validation in the official runner, and contains lane-level attribution/intervention code. | Validation labels select α. A surviving official-checkpoint aggregate is available; the complete original per-task Experiment 1 outputs and standalone run log were not found. The later self-trained reproduction in `reproduction/` is a different checkpoint set. |
| **2 — merge comparison and harm decomposition** | `run_week1.py compare` compares `dc_official`, `dc_rank`, and CT-Merging on seven four-task contexts plus all8; `decompose` evaluates companion-lane subsets and doses in the three core contexts. | α / γ are selected on validation; scores are reported on test, with decomposition and dose scores on validation. The saved Oct 16 files are explicitly treated as archived results. |
| **3 — per-task oracle** | `run_oracle.py` starts from each `dc_rank` α and greedily searches a coefficient per lane for `oracle_mean` and `oracle_fair`; it evaluates chosen vectors on test once. | Validation labels choose the coefficients. Primary comparisons use a separately saved current-environment `dc_rank` baseline overlay. Seven contexts completed and passed its current-baseline checks. |
| **4 — cross-context transfer** | `run_loco_task_transfer.py` averages each task’s `oracle_fair` coefficient-to-α ratios over other contexts containing that task, multiplies by the held-out context’s α, and evaluates that vector once on validation and test. | Ratios come from Experiment 3’s validation-selected coefficients; no new search and no test-score use in coefficient construction. Seven contexts completed. |
| **5 — entropy signal check** | `run_entropy.py check` computes entropy for the coefficient vectors already in Experiment 3’s validation caches, then correlates entropy with their cached validation accuracies. | Entropy uses 500 unlabeled validation images per task and seed 0. Labels enter only the diagnostic accuracy correlation and comparisons. The negative gate stopped the run before search or test. |

The exact Experiment 1 reproduction aggregate and limitations are recorded in [`reproduction_comparison.md`](../../../reproduction/selftrained_b32_r16_8task/results/reproduction_comparison.md) and the [diagnostics inventory](../../experiments1-4-diagnostics/README.md). The Experiment 3 and Experiment 4 execution code and outputs are preserved under [`handoff/experiment3`](../../experiment3/README.md) and [`handoff/experiment4`](../../experiment4/README.md); Experiment 5’s code, metadata, log, and signal-check cache are under [`handoff/experiment5`](../README.md).

## 2. Mathematical formulation as implemented

### 2.1 Task updates and the DC-Merge operator

Let task `i` have LoRA factors `Aᵢ` and `Bᵢ`. For module `ℓ`, the code’s task update is

```text
Δᵢ,ℓ = Bᵢ,ℓ Aᵢ,ℓ
```

and the merged model adds a merged update to pretrained weights `W₀`. For each task and module, a rank-16 SVD gives `Δᵢ,ℓ ≈ Uᵢ Sᵢ Vᵢᵀ`. Under linear smoothing, the top-16 singular values are replaced by their summed energy times a normalized linear vector from `min(5, sᵢ,1 / sᵢ,16)` to 1. Call the resulting matrix `Δ̃ᵢ,ℓ`.

Concatenated task singular directions are orthogonalized into shared cover frames `U` and `V`. The code projects each smoothed update into that frame:

```text
Mᵢ = Uᵀ Δ̃ᵢ V
```

It then filters each projected matrix. `dc_official` retains the largest-magnitude 0.1% (with the implementation’s threshold-tie rule); `dc_rank` retains 16 entries per task. TIES uses the sign of the sum as the reference, keeps entries agreeing with that sign, averages the selected nonzero entries, and zeros entries whose signed sum is zero. Let `A` be that aggregate. A block-diagonal structural mask `S` removes all off-lane blocks. If `Bᵢ` denotes task `i`’s block offset, then

```text
Δmerge = U (A ⊙ S) Vᵀ = ∑ᵢ Lᵢ
Lᵢ = U[:, Bᵢ] A[Bᵢ, Bᵢ] Vᵀ[Bᵢ, :]
W(c) = W₀ + ∑ᵢ cᵢ Lᵢ
```

The last equation is the Experiment 3–5 parameterization. Setting every `cᵢ = α` gives `W₀ + α Δmerge`, which is the global-coefficient special case used by `dc_rank`. In the patched `LaneMerge.compose`, uniform coefficients take the direct `α × merged_update` path; unequal coefficients scale and sum the lane terms. The maximum recorded relative lane-sum reconstruction error was `1.7320 × 10⁻⁷`. ([merge arithmetic](../../experiment3/code/lane_harm/lanes.py#L53-L74), [per-context checks](../../experiment3/results/experiment3/final_verification.json))

### 2.2 Classification and normalized accuracy

For task `i`, let `aᵢ(c)` be its percentage accuracy under `W(c)`, and let `rᵢ` be the saved single-task reference accuracy percentage on the same split. The code defines

```text
nᵢ(c) = 100 × aᵢ(c) / rᵢ
mean_norm(c) = (1/T) ∑ᵢ nᵢ(c)
worst_norm(c) = minᵢ nᵢ(c)
```

Here `T` is the number of tasks in the context. A context’s `worst_norm` is its minimum task-normalized accuracy. The seven-context aggregate “mean worst” is the arithmetic mean of those seven context minima; it is not the single minimum across all 28 task-context entries. The implemented normalization is [`run_week1.py::norm`](../../experiment3/code/lane_harm/run_week1.py#L44-L45); Experiment 3 and 4 use the same ratio in [`run_oracle.py`](../../experiment3/code/lane_harm/run_oracle.py#L61-L75) and [`run_loco_task_transfer.py`](../../experiment4/results/run_loco_task_transfer.py#L39-L47).

### 2.3 Experiment 3 oracle objectives and search

For each context, Experiment 3 reads α selected for `dc_rank` on validation and starts at `cᵢ = α` for all tasks. Its grid is `{0.3, 0.4, …, 1.0}` plus α, and it makes at most two greedy coordinate passes in fixed task order.

```text
oracle_mean: maximize mean_norm_val(c)
oracle_fair: maximize mean_norm_val(c), subject to
             worst_norm_val(c) ≥ worst_norm_val_val(dc_rank)
```

The `oracle_fair` constraint is enforced during validation search; it is not a test-set constraint. The search is greedy and can miss a better vector. The code uses on-disk accuracy caches, verifies the uniform-α baseline, then evaluates `dc_rank`, `oracle_mean`, and `oracle_fair` on test. ([search definition](../../experiment3/code/lane_harm/run_oracle.py#L74-L106), [baseline, objective calls, and test evaluations](../../experiment3/code/lane_harm/run_oracle.py#L109-L179))

### 2.4 Experiment 4 leave-one-context-out rule

Let `Cᵢ` be the set of saved contexts containing task `i`; let `c*ᵢ,c′` be the Experiment 3 `oracle_fair` coefficient for task `i` in context `c′`, and let `αc′` be that context’s `dc_rank` coefficient. For held-out context `c`, the rule is

```text
rᵢ,−c = average over c′ ∈ Cᵢ \ {c} of (c*ᵢ,c′ / αc′)
cᵢ,transfer(c) = αc × rᵢ,−c
```

The held-out coefficient does not contribute to its own ratio. If there is no other context containing the task, the implemented fallback is ratio 1.0; DTD has this case. The saved metadata records zero searches, seven validation evaluations, seven test evaluations, and no use of test accuracy to form coefficients. ([formula and support counts](../../experiment4/results/run_loco_task_transfer.py#L90-L116), [run manifest](../../experiment4/results/run_manifest.json))

### 2.5 Experiment 5 entropy objectives and signal check

For task `i`, the code takes a seeded subset `Xᵢ` of up to 500 validation images, discards labels in the entropy loader, normalizes the image feature `fθ(x; W(c))` and head vectors `hᵢ,k`, and forms

```text
pᵢ,k(x; c) = softmaxₖ(100 × fθ(x; W(c))ᵀ hᵢ,k)
Hᵢ(c) = average over x ∈ Xᵢ of [−∑ₖ pᵢ,k(x; c) log pᵢ,k(x; c) / log Kᵢ]
```

`Kᵢ` is the number of classes. Thus `Hᵢ` is normalized to [0, 1] and lower values mean more confident predictions, not necessarily more accurate predictions. The planned `entropy_mean` objective minimizes `(1/T)∑ᵢ Hᵢ`; `entropy_fair` minimizes `maxᵢ Hᵢ`, breaking ties by lower mean entropy. The planned coordinate search uses the same 0.3–1.0 grid plus the starting α and a two-pass cap; these objectives and search are implemented, but the coefficient-search command was not run. ([unlabeled loader and entropy](../code/lane_harm/run_entropy.py#L68-L114), [grid and temperature](../code/lane_harm/run_entropy.py#L39-L43), [objective and coordinate search](../code/lane_harm/run_entropy.py#L130-L160), [two-pass default](../code/lane_harm/run_entropy.py#L360-L369))

The executed `check` phase evaluates entropy on Experiment 3’s already-cached coefficient vectors, selects the lowest-mean-entropy and lowest-maximum-entropy vector for diagnostics, and computes Spearman correlations against their cached validation accuracy scores. Validation labels are therefore absent from the entropy objective but are used in this signal-check diagnostic. ([check implementation](../code/lane_harm/run_entropy.py#L199-L245))

### 2.6 Where labels enter

| Stage | Validation labels used? | Test labels used? |
|---|---|---|
| Experiment 1 / Experiment 2 α or γ selection | Yes, to select the scalar coefficient. | No for selection; test is scored after selection. |
| Experiment 2 lane decomposition and dose table | Yes, to measure validation accuracy after interventions. | No. |
| Experiment 3 oracle | Yes, directly to search `c` and calculate the fairness floor. | No for selection; each chosen vector is scored once on test. |
| Experiment 4 transfer | Indirectly: source-context `oracle_fair` vectors were selected on validation. The held-out search uses no labels because there is no search. | No test scores enter ratio construction; transferred vectors are scored once on each split. |
| Experiment 5 entropy calculation | No labels are read while calculating entropy or choosing a vector. | No test evaluation occurred. |
| Experiment 5 signal check | Yes, only to correlate entropy with cached accuracy and compare the entropy pick with `dc_rank`. | No. |

## 3. What improvement over DC-Merge is supported by the saved results?

### 3.1 Experiment 1: surviving aggregate, incomplete original records

The repository’s saved official-checkpoint reproduction summary reports selected `α = 0.4`, mean absolute test accuracy `64.149%`, and mean normalized test accuracy `73.879%`, against paper Table 1 values `64.17%` and `73.90%`. These are aggregate reproduction metrics; the archived standalone Experiment 1 per-task validation result set and run log are unavailable. The later independently self-trained adapters in the same reproduction directory report a different α and result and are not substituted for the official-checkpoint record. ([comparison](../../../reproduction/selftrained_b32_r16_8task/results/reproduction_comparison.md), [reproduction README](../../../reproduction/selftrained_b32_r16_8task/README.md), [availability note](../../experiments1-4-diagnostics/README.md))

### 3.2 Experiment 2: archived 16 October results

The table below is **only the archived Oct 16 Experiment 2a summary**. α / γ were selected on validation; the listed normalized accuracies and task minima are test results. “Mean worst” in the aggregate is the mean of per-context worst-task normalized scores.

| Four-task context | `dc_rank` α | `dc_rank` mean / worst norm | CT mean / worst norm |
|---|---:|---:|---:|
| C1: DTD–EuroSAT–GTSRB–SVHN | 0.5 | 71.22 / 62.20 | 69.41 / 55.28 |
| C2: EuroSAT–GTSRB–MNIST–SVHN | 0.6 | 73.83 / 55.56 | 70.24 / 53.34 |
| C3: EuroSAT–GTSRB–SUN397–SVHN | 0.6 | 76.65 / 63.48 | 73.44 / 61.09 |
| C4: Cars–EuroSAT–GTSRB–MNIST | 0.6 | 81.60 / 72.20 | 80.27 / 67.90 |
| C5: Cars–EuroSAT–GTSRB–RESISC45 | 0.7 | 85.55 / 80.55 | 82.01 / 79.70 |
| C6: Cars–EuroSAT–MNIST–RESISC45 | 0.6 | 87.60 / 85.81 | 86.05 / 84.55 |
| C7: Cars–EuroSAT–MNIST–SUN397 | 0.7 | 89.64 / 84.76 | 89.44 / 83.09 |
| **Seven-context mean** | — | **80.87 / 72.08** | **78.70 / 69.28** |

Across the seven archived contexts, `dc_rank` scored 80.87 mean normalized test accuracy versus 76.22 for `dc_official` (a 4.65 pp difference); mean context-worst was 72.08 versus 69.67. CT-Merging averaged 78.70 mean / 69.28 mean-worst. For the separate all8 archived setting, `dc_official` selected α = 0.4 and reached 73.856% mean normalized test accuracy; the code notes `dc_rank` is identical to the official top-k count at eight tasks, so it is not separately swept there. ([archived comparison table](../../experiment3/results/experiment2/archive_compare/compare_summary.md), [archived result note](../../experiment3/results/experiment2/experiment2-results-note.md), [eight-task equality note](../../experiment3/code/lane_harm/run_week1.py#L75-L109))

In the three core contexts, CT raised raw EuroSAT test accuracy over archived `dc_rank` by `+5.85`, `+15.96`, and `+7.07` pp. But CT lowered the mean normalized score by `1.81`–`3.59` pp, and the worst task became SVHN with a lower score than the `dc_rank` worst. The saved decomposition attributes roughly 91–92% of EuroSAT harm to neighbor lanes; the GTSRB+SVHN pair interaction alone accounts for 34.8–56.7% of the total harm in the three core contexts. This supports a context interaction, not an additive “victim lane only” account. ([Experiment 2 note](../../experiment3/results/experiment2/experiment2-results-note.md))

| Core context | EuroSAT test accuracy: archived `dc_rank` → CT | Neighbour share of harm | GTSRB+SVHN interaction / total harm |
|---|---:|---:|---:|
| C1 | 61.07 → 66.93 (+5.85 pp) | 91.4% | 34.8% |
| C2 | 54.56 → 70.52 (+15.96 pp) | 92.4% | 40.1% |
| C3 | 62.33 → 69.41 (+7.07 pp) | 92.0% | 56.7% |

### 3.3 Experiment 3: current-environment baseline and test oracle results

This table is **not the Oct 16 archive**. `dc_rank` values are the direct current-environment replay at each archived selected α; oracle values are the current-environment test evaluations of the validation-selected vectors. Deltas are percentage points of normalized accuracy relative to that current-environment `dc_rank` result. `oracle_mean` optimizes mean validation accuracy only; `oracle_fair` enforces the validation worst-task floor. ([current-baseline manifest](../../experiment3/results/experiment2/current_baseline/manifest.json), [per-context oracle summary](../../experiment3/results/experiment3/oracle_summary.md), [raw context records](../../experiment3/results/experiment3/contexts/))

| Context | Current `dc_rank` mean / worst | `oracle_mean` Δ mean / worst | `oracle_fair` Δ mean / worst |
|---|---:|---:|---:|
| C1 | 71.19 / 62.20 | +2.66 / −4.21 | +2.95 / −0.19 |
| C2 | 73.82 / 55.45 | +3.30 / +0.76 | +3.30 / +0.76 |
| C3 | 76.68 / 63.60 | +2.83 / −5.76 | +1.93 / −0.18 |
| C4 | 81.59 / 72.16 | +2.50 / +5.76 | +2.50 / +5.76 |
| C5 | 85.50 / 80.58 | +0.05 / −0.12 | +0.00 / +0.00 |
| C6 | 87.68 / 85.93 | +0.02 / −10.85 | +0.00 / +0.00 |
| C7 | 89.61 / 84.76 | +0.91 / −1.81 | +0.71 / +1.95 |

| Evaluation set | Method | Mean normalized test | Mean context-worst test | Δ mean / Δ worst vs current `dc_rank` | Guide verdict |
|---|---|---:|---:|---:|---|
| Core (3) | `dc_rank` | 73.90 | 60.41 | — | — |
| Core (3) | `oracle_mean` | 76.82 | 57.35 | +2.93 / −3.07 | SMALL GAIN |
| Core (3) | `oracle_fair` | 76.62 | 60.55 | +2.72 / +0.13 | SMALL GAIN |
| All seven | `dc_rank` | 80.87 | 72.09 | — | — |
| All seven | `oracle_mean` | 82.62 | 69.78 | +1.75 / −2.32 | SMALL GAIN |
| All seven | `oracle_fair` | 82.49 | 73.25 | +1.63 / +1.16 | SMALL GAIN |

`oracle_fair` substantially improves EuroSAT’s raw test accuracy in the three core contexts—61.07→76.70%, 54.44→86.22%, and 62.44→83.74%—but the two first and third contexts still have a slightly lower test worst-task score than their current baseline. Averaged over three core contexts, fair-oracle mean improves +2.72 pp and worst improves only +0.13 pp; over all seven the corresponding gains are +1.63 and +1.16 pp. The guide’s HEADROOM rule requires at least +1 pp mean **and** +3 pp worst, so the saved conclusion is SMALL GAIN, not a decision to build a predictor. ([oracle summary](../../experiment3/results/experiment3/oracle_summary.md), [final verdict record](../../experiment3/results/experiment3/final_verification.json))

For the three core contexts, the raw EuroSAT test accuracy gains in this table use the current-environment direct replay as the baseline. The saved `oracle_fair` vectors, in each context’s task order, are:

| Context | `oracle_fair` vector |
|---|---|
| C1 | DTD 0.3, EuroSAT 0.8, GTSRB 0.7, SVHN 0.5 |
| C2 | EuroSAT 1.0, GTSRB 0.7, MNIST 0.4, SVHN 0.5 |
| C3 | EuroSAT 1.0, GTSRB 0.7, SUN397 0.3, SVHN 0.6 |
| C4 | Cars 0.3, EuroSAT 0.9, GTSRB 0.7, MNIST 0.5 |
| C5 | Cars 0.7, EuroSAT 0.7, GTSRB 0.7, RESISC45 0.7 (unchanged) |
| C6 | Cars 0.6, EuroSAT 0.6, MNIST 0.6, RESISC45 0.6 (unchanged) |
| C7 | Cars 0.6, EuroSAT 0.7, MNIST 0.6, SUN397 0.5 |

Across all seven contexts, the lane-sum relative error was at most `1.7320 × 10⁻⁷`; all seven current-replay baseline checks were exactly 0.000 pp on validation and test; all validation objective constraints passed; seven results and 28 CSV rows are present. ([final verification](../../experiment3/results/experiment3/final_verification.json), [logs](../../experiment3/results/experiment3/logs/))

### 3.4 Experiment 4: task coefficients do not transfer reliably

The next table reports test deltas for the leave-one-context-out transferred vector relative to the same current-environment `dc_rank` baseline. The ratio vectors themselves come from other contexts’ validation-selected `oracle_fair` coefficients; DTD has no supporting other context and uses the explicit neutral fallback. No search was run. ([transfer summary](../../experiment4/results/transfer_summary.md), [transfer coefficients](../../experiment4/results/transfer_coefficients.json), [run manifest](../../experiment4/results/run_manifest.json))

| Context | Transferred coefficients | Δ mean / worst vs `dc_rank` (test) |
|---|---|---:|
| C1 | DTD 0.5000, EuroSAT 0.6528, GTSRB 0.5625, SVHN 0.4583 | +2.320 / +0.187 |
| C2 | EuroSAT 0.7767, GTSRB 0.7100, MNIST 0.5381, SVHN 0.6000 | +1.587 / +6.526 |
| C3 | EuroSAT 0.7767, GTSRB 0.7100, SUN397 0.4286, SVHN 0.5500 | +2.114 / +1.678 |
| C4 | Cars 0.5714, EuroSAT 0.7933, GTSRB 0.7100, MNIST 0.5048 | +1.557 / +7.582 |
| C5 — RESISC45 control | Cars 0.5500, EuroSAT 0.9839, GTSRB 0.8575, RESISC45 0.7000 | **−2.754 / −10.003** |
| C6 — RESISC45 control | Cars 0.4714, EuroSAT 0.8433, MNIST 0.4714, RESISC45 0.6000 | **−1.589 / −6.185** |
| C7 | Cars 0.5833, EuroSAT 0.9839, MNIST 0.5833, SUN397 0.3500 | −0.688 / +0.690 |
| **Seven-context mean** | — | **+0.364 / +0.068** |

The transfer kept about `0.364 / 1.626 = 22.4%` of the oracle-fair mean gain across seven contexts. The two RESISC45 contexts are the negative controls: the transferred coefficients reduced both mean and worst-task test scores, with the C5 worst-task decline of about 10 pp. The transfer result is therefore evidence against a simple task-only coefficient rule for these contexts; it does not show that every possible transfer rule fails. ([aggregate records](../../experiment4/results/aggregate.json), [guide-level interpretation](../../experiment4/REPORT.md))

### 3.5 Experiment 5: signal check fails the pre-search gate

The signal check used 500 unlabeled validation images per task, seed 0, and the 78, 57, and 82 coefficient vectors cached by Experiment 3 in the three core contexts. The accuracy comparisons below are normalized validation accuracy diagnostics: `best / entropy-picked / dc_rank`. `entropy_mean` picks the lowest mean entropy; the “fair pick” picks the lowest maximum task entropy. ([check summary](../results/check_summary.md), [subset metadata](../results/check_metadata.json), [run metadata](../config/run_metadata.json))

| Core context | ρ(−mean H, mean normalized accuracy) | ρ(−max H, worst normalized accuracy) | Mean normalized validation accuracy: best / entropy pick / `dc_rank` | Worst-task normalized validation accuracy: best / fair pick / `dc_rank` |
|---|---:|---:|---:|---:|
| C1 | 0.35 | 0.09 | 74.55 / 68.72 / 71.16 | 68.32 / 54.12 / 62.08 |
| C2 | 0.43 | 0.12 | 77.47 / 68.89 / 73.74 | 69.32 / 48.33 / 53.70 |
| C3 | 0.01 | −0.28 | 79.60 / 70.77 / 75.83 | 70.57 / 47.64 / 62.04 |

Every entropy-selected mean-normalized-accuracy diagnostic was worse than `dc_rank` on validation; the fair entropy pick was also below the baseline worst-task score in each context; and the fair correlation was negative in C3. The guide explicitly stops when entropy selection is worse than `dc_rank` or a required correlation is nonpositive. The saved state is therefore **signal-check complete, negative gate**. It contains no coefficient-search results, no test score for either entropy method, no `four` phase, and no seven-context Experiment 5 verdict. ([gate report](../report.md), [check log](../logs/check.log), [metadata](../config/run_metadata.json))

## 4. Baseline discrepancy audit: archived Oct 16 versus current replay

The exact archived-versus-current cause is **not established**. The archived/current comparison at the archived selected α found a maximum absolute validation difference of **0.158353 pp** (GTSRB in C1) and a maximum absolute test difference of **0.148148 pp** (EuroSAT in C5). These deltas are per-task accuracy differences, not changes in the Experiment 3 current-baseline checks. ([task-level manifest](../../experiment3/results/experiment2/current_baseline/manifest.json), [audit](../../experiment3/audit/replay-diagnosis.md), [supervisor Q&A](../../experiment3/BASELINE_DISCREPANCY_QA.md))

The audit verified that the available runner and preprocessing source files matched the Oct 16 handoff snapshot; the current model, adapters, and heads matched saved Hub revisions and content hashes; the dataset repository revision and split/config inputs matched; and repeated preprocessing of sampled inputs matched. A controlled same-input GTSRB validation repeat on the current runtime produced no changed predictions. A separate current-runtime diagnostic found 30 prediction changes between AMP and FP32 and 19 between old lane accumulation and the fixed uniform arithmetic path, concentrated at very small or tied margins. Those diagnostics make numerical sensitivity plausible; they do not identify the archived flip examples.

The Experiment 2 note’s reported difference up to **0.32 pp** does not independently explain the archived/current mismatch: it compared distinct evaluation arithmetic paths, not two repeated evaluations through one fixed path. The lane-arithmetic fix is a separate, verified correction: for uniform coefficients, `LaneMerge.compose` now scales the completed direct merge; the current direct Experiment 2 replay itself does not use this composition method. ([code fix](../../experiment3/code/lane_harm/lanes.py#L53-L74), [audit assessment](../../experiment3/audit/replay-diagnosis.md), [Oct 16 note](../../experiment3/results/experiment2/experiment2-results-note.md))

What remains unavailable is decisive for a historical attribution: the archived per-example logits/predictions, a complete Oct 16 runtime/GPU/driver record, and cryptographic hashes of every extracted dataset image. Therefore the evidence supports “numerical sensitivity is plausible; historical cause and normality are unproven,” not “this is normal GPU variation.” This gap limits exact historical reproduction. It does not invalidate Experiment 3’s internal comparison: the current-environment baseline overlay was used consistently, and all seven baseline checks against that overlay were 0.000 pp on validation and test. ([unverified items](../../experiment3/audit/replay-diagnosis.md), [E3 final checks](../../experiment3/results/experiment3/final_verification.json))

<!-- PAGEBREAK -->

## 5. Reproducibility record

- **Experiment 1 / later reproduction artifacts:** [`reproduction/`](../../../reproduction/selftrained_b32_r16_8task/README.md) distinguishes its official-checkpoint reproduction from separately self-trained adapters. The standalone original Experiment 1 per-task validation output is unavailable.
- **Experiment 2 archived results:** [`archive_compare/compare_summary.md`](../../experiment3/results/experiment2/archive_compare/compare_summary.md) and [`experiment2-results-note.md`](../../experiment3/results/experiment2/experiment2-results-note.md). The archive is preserved separately from the current baseline overlay.
- **Experiment 3:** per-context `oracle_result.json`, `val_cache.json`, and `test_cache.json` under [`results/experiment3/contexts/`](../../experiment3/results/experiment3/contexts/), aggregate [`oracle_summary.md`](../../experiment3/results/experiment3/oracle_summary.md), [`oracle.csv`](../../experiment3/results/experiment3/oracle.csv), [`final_verification.json`](../../experiment3/results/experiment3/final_verification.json), and core/four logs under [`results/experiment3/logs/`](../../experiment3/results/experiment3/logs/). The packaged reproduction instructions are [`REPRODUCIBILITY.md`](../../experiment3/REPRODUCIBILITY.md). The captured environment was Python 3.9.23, PyTorch 2.8.0+cu128, CUDA runtime 12.8, RTX 3090, driver 580.142; the original historical runtime remains incomplete.
- **Experiment 4:** seven context result JSONs and caches, [`aggregate.json`](../../experiment4/results/aggregate.json), [`transfer_summary.md`](../../experiment4/results/transfer_summary.md), [`run_manifest.json`](../../experiment4/results/run_manifest.json), and [`run.log`](../../experiment4/results/run.log). The manifest records seven new validation and seven new test evaluations, with zero searches.
- **Experiment 5:** [`run_entropy.py`](../code/lane_harm/run_entropy.py), [official guide](../docs/IMPLEMENTATION_GUIDE_EXPERIMENT5.md), [`check.json`](../results/check.json), [`check_summary.md`](../results/check_summary.md), three [`entropy_cache_n500_s0.json`](../results/entropy_caches/) files, [`check.log`](../logs/check.log), and [`run_metadata.json`](../config/run_metadata.json). Metadata records the input ZIP hash, code hashes, environment, seed, subset sizes, and the explicit stop-before-search gate.

The E3 archived/current overlay does not overwrite the archived Oct 16 `compare/` files. No datasets, model weights, adapters, heads, credentials, or large caches are part of this report package.

## 6. Limitations and research conclusion

These results use one ViT-B/32 checkpoint set, one seed, eight tasks, and seven selected four-task contexts. Experiment 3’s two-pass coordinate search is greedy and uses validation labels; its test improvements describe oracle headroom, not expected performance of a predictor. Experiment 4 transfers only among this small, unevenly connected context set; DTD has no cross-context source, while RESISC45 has one. Experiment 5 uses one seed-0 subset of 500 validation images per task, and entropy can reward confidence without correctness. The original Oct 16 runtime and predictions are incomplete, and Experiment 1’s standalone per-task run record is missing.

The evidence supports three conclusions. First, within the archived Experiment 2 comparison, the `dc_rank` top-k variant outperformed `dc_official` on mean and mean-worst normalized test accuracy across the seven contexts, while CT-Merging’s EuroSAT gains came with lower aggregate scores. Second, validation-tuned lane coefficients exposed modest test headroom over current `dc_rank`, but they did not meet the guide’s fairness threshold, so they remain an upper bound. Third, the tested task-only transfer retained little average gain and failed in both RESISC45 controls; the tested entropy signal was worse than baseline and failed its gate before any search or test.

The next step specified by the saved Experiment 3 guide is to **report and discuss the SMALL GAIN result before deciding whether to build a data-free predictor**. The Experiment 5 guide says to **stop and report** after its negative signal gate. On the current evidence, a label-free entropy coefficient method is not supported; it should not be described as viable, and no Experiment 5 test-performance claim can be made.
