# Experiment 5: can unlabeled images choose the per-task coefficients?

*Experiment guide for the mentee. It needs the outputs of Experiment 3 ([`IMPLEMENTATION_GUIDE_EXPERIMENT3.md`](IMPLEMENTATION_GUIDE_EXPERIMENT3.md)), and Experiment 2's assets and environment.*

---

## Part I: the story

### What we know

- **Per-task coefficients have real headroom.** In Experiment 3, choosing one coefficient per task (using validation labels) beat `dc_rank`'s single α on test: **+1.6 pp mean over 7 contexts, +2.7 pp in the 3 EuroSAT contexts**.
- **The right coefficients depend on the combination.** In Experiment 4, coefficients carried over from *other* contexts kept only about 22% of that gain. In the two RESISC45 contexts they did real damage (−10 pp on the worst task). EuroSAT needs a boost only when GTSRB and SVHN are present.
- **The harm is non-additive and arises through the network's depth** (Experiment 2). Inside one module the lanes are orthogonal, so any score computed from weights alone adds up exactly and **cannot see** the interaction.

### The method idea

So the coefficients must be chosen **for each merge**, by looking at how the **actual merged network** behaves. Labels aren't available at merge time, but **unlabeled images** usually are.

> **Label-free lane coefficients:** choose $c_i$ for each task so the merged model is **confident** on each task's unlabeled images, i.e. its prediction entropy is low. The *fair* version minimises the **worst** task's entropy.

$$W(c) = W_0 + \sum_i c_i L_i,\qquad
\textbf{entropy\_mean: } \min_c \tfrac1T\sum_i H_i(c),\qquad
\textbf{entropy\_fair: } \min_c \max_i H_i(c)$$

Here $H_i$ is the average normalized prediction entropy on a fixed set of **500 unlabeled validation images** of task $i$. It ranges from 0 (fully confident) to 1 (uniform). Logits use CLIP's standard scale, 100, on unit-norm heads.

### Why this could work, and why it could fail

- **For:** evaluating the real network captures the cross-layer interaction that weight-only rules miss. It re-estimates the coefficients for every combination, so it shouldn't repeat Experiment 4's RESISC45 mistake. And it adjusts only 4 numbers, so it can't over-fit much.
- **Against:** entropy can be made low by a model that is **confidently wrong**. A cheap check (Part II, step 1) tests this directly before the full search is run.

### What we compare (all on test)

| Arm | Uses labels to choose? | Role |
|---|---|---|
| `dc_rank` (global α) | validation labels, for α only | baseline |
| **`entropy_mean`**, **`entropy_fair`** | **no** | **the method** |
| `oracle_fair` (Exp. 3) | validation labels | upper bound |
| CT-Merging (Exp. 2) | validation labels, for γ only | data-free competitor |

Prior work using the same idea: AdaMerging (ICLR 2024, entropy on task arithmetic) and AdaRank (ICLR 2026, entropy to select singular components). If this experiment succeeds, both become **mandatory baselines**.

---

## Part II: running it

### 0. Package and paths

`DC-Merge-official-experiment5.zip` = the Experiment 3 package plus `lane_harm/run_entropy.py` and this guide. No other file changed.

```bash
cd DC-Merge
EXP3=/srv/dcmerge/results/experiment3      # Experiment 3 output (per-context oracle_result.json, val_cache.json)
OUT=/srv/dcmerge/results/experiment5       # new folder
ARGS="--repo $PWD --model $MODEL --ft-dir $FT_DIR --heads $HEADS --data $DATA --exp3 $EXP3 --device cuda"
```

`PYTHON`, `MODEL`, `FT_DIR`, `HEADS` and `DATA` are set as in Experiments 2–3.

### 1. The signal check: run first (~30–60 min for the 3 core contexts)

```bash
"$PYTHON" lane_harm/run_entropy.py check $ARGS --out "$OUT" --contexts core 2>&1 | tee "$OUT-check.log"
```

This takes the 30–90 coefficient vectors that Experiment 3 already evaluated per context (with validation accuracy) and computes their entropy on the unlabeled subset. Labels are not used for the entropy. It reports:
- **ρ(−mean H, mean acc):** does lower average entropy go with higher average accuracy? (Spearman rank correlation)
- **ρ(−max H, worst acc):** does lower worst-task entropy go with a higher worst-task accuracy?
- The validation accuracy of the vector entropy would pick, next to the best cached vector and `dc_rank`.

| Check result (3 core contexts) | Do next |
|---|---|
| ρ ≥ **0.5** in all three, and the entropy pick beats `dc_rank` on mean accuracy | Signal is good → **step 2** |
| ρ between 0 and 0.5, or mixed | Weak signal → still run step 2 on core only, and expect PARTIAL |
| ρ ≤ 0, or the entropy pick is worse than `dc_rank` | Entropy prefers confidently-wrong merges → **stop**, report the check; no full run needed |

This step reuses the entropy cache, so step 2 doesn't recompute these vectors.

### 2. The entropy search and one test evaluation

```bash
# the 3 EuroSAT contexts first, then the other 4 (finished contexts are skipped)
"$PYTHON" lane_harm/run_entropy.py run $ARGS --out "$OUT" --contexts core 2>&1 | tee "$OUT-core.log"
"$PYTHON" lane_harm/run_entropy.py run $ARGS --out "$OUT" --contexts four 2>&1 | tee "$OUT-four.log"
```

**Per context, the script:**
1. Builds the `dc_rank` lanes.
2. Runs the coordinate search, with the **same grid and passes as Experiment 3**, for both entropy objectives, using **unlabeled images only**.
3. Evaluates each chosen vector **once on test**: the result. It also scores it on validation with labels, as a diagnostic only.
4. Reads `dc_rank`, `oracle_fair`, `oracle_mean` and CT-Merging test numbers from Experiment 3, so they are the same baselines.

**Cost (estimate):** entropy on 500 images × 4 tasks is much cheaper than a full validation evaluation. Expect roughly **1–3 h for core** and **2–4 h more for the other 4**.

Options:
- `--n-unlabeled` (default 500) sets the number of unlabeled images per task.
- `--seed` (default 0) chooses which images.
- Keep both at their defaults for the main run.

To rebuild the summary only: `"$PYTHON" lane_harm/run_entropy.py summarize --out "$OUT"`.

### 3. Output

| File | Content |
|---|---|
| `$OUT/check_summary.md` | Step 1 table |
| `$OUT/entropy_summary.md` | **Main result:** per context and method, test mean/worst and Δ vs `dc_rank`; averages, share of the oracle's gain, verdict |
| `$OUT/entropy.csv` | The same numbers |
| `$OUT/<context>/entropy_result.json` | Unlabeled subset used, search log, chosen vectors, entropies, test and validation scores |
| `$OUT/<context>/entropy_cache_*.json` | Every evaluated vector's per-task entropy |

---

## Part III: expected outcomes and decision

### Must hold

- The log for each context shows `unlabeled images per task: … = 500`, or the whole split if it's smaller.
- `entropy_result.json` → `lane_checks.max_rel_sum_error` is below 1e-5, as in Experiments 2–3.
- The `dc_rank`, `oracle_fair` and `ct` rows equal Experiment 3's numbers. They are copied, not recomputed.

### What we expect

- **Honestly uncertain.** Entropy objectives are known to work reasonably for task-arithmetic merging (AdaMerging), but they can also reward over-confident merges. In our synthetic code test, entropy did **not** track accuracy at all, so this is a real risk. The step-1 check is there to catch it cheaply.
- **If it works,** expect the chosen vectors to resemble the oracle's: EuroSAT and GTSRB up in the core contexts, DTD/SUN397/Cars down, and few or no changes in the RESISC45 contexts.

### Decision (computed in `entropy_summary.md`, from the 7-context averages)

| Result on test vs `dc_rank` | Verdict | Next step |
|---|---|---|
| Entropy arm keeps **≥ 50% of `oracle_fair`'s mean gain** and **no context** has a worst-task drop > 1 pp | **METHOD VIABLE** | Gradient version (coefficients per task × per layer), AdaMerging and AdaRank baselines, 8/12/16-task benchmarks |
| Gains, but below 50%, or some context drops the worst task by > 1 pp | **PARTIAL** | Discuss: try a safety rule (accept a change only if no task's entropy rises), or stop |
| No mean gain | **NOT VIABLE** | Write the analysis paper (Experiments 1–4) |

---

## Part IV: what to send back

1. `check_summary.md`, `entropy_summary.md`, `entropy.csv`, and the logs.
2. A short note:
   - The check's ρ values per context.
   - The verdict for core and for all 7.
   - For each context: did the entropy search change coefficients in the same direction as the oracle?

## Limitations (state them when reporting)

- One unlabeled subset (seed 0, 500 images per task), taken from the validation split. Labels are never read during the search.
- Same greedy search and grid as Experiment 3, so it can miss better vectors.
- One checkpoint set, 7 selected contexts, a single run.
- A positive result makes this a **candidate method**. It is not yet compared with AdaMerging or AdaRank.
