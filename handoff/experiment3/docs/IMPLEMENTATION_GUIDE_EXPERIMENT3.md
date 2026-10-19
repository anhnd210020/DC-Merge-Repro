# Experiment 3: can one coefficient per task beat a single global α?

*Experiment guide for the mentee. It builds on Experiment 2 ([`IMPLEMENTATION_GUIDE.md`](IMPLEMENTATION_GUIDE.md)) and needs its `compare/` results.*

---

## Part I: the story

### What Experiment 2 showed

- **`dc_rank` (keep 16) is the best merge so far.** Over 7 four-task contexts it has the highest mean (80.87) and the highest worst-task score (72.08) on test.
- **CT-Merging lifts EuroSAT, but moves the damage to SVHN.** The worst task gets worse.
- **EuroSAT's harm comes from its neighbours** (about 91% of it), mostly from GTSRB and SVHN *together*.
- **Changing one lane at a time didn't give a clean rescue.** But turning SVHN down to 0.75 did raise EuroSAT by +11 pp *and* the mean by +1.85 pp (validation).
- **Boosting EuroSAT's own lane only helped at α = 0.4.** At the per-context α selected on validation, it didn't.

### The question now

All merges so far use **one global coefficient α** for every task. That is a compromise: tuning α to maximise the mean also increases EuroSAT's harm. Experiment 2 only changed **one lane at a time**. It never searched all tasks' coefficients **together**.

> **If each task gets its own coefficient $c_i$, and we choose the best vector on validation, does it beat `dc_rank`'s single α on test, for both the mean and the worst task?**

The merged model is

$$W(c) = W_0 + \sum_i c_i\,L_i ,$$

where $L_i$ is task $i$'s lane from Experiment 2. Setting all $c_i = \alpha$ gives back `dc_rank` exactly.

### Why this is the decisive experiment

This is an **oracle**: it uses validation accuracy, which a data-free method won't have. It is therefore an *upper bound* on what any method that predicts per-task coefficients could achieve.

- **If the oracle can't beat `dc_rank` on test,** no data-free predictor will. Then we stop the "save the victim" method and write the analysis paper (non-additive harm).
- **If it can,** there is real headroom, and the next step is predicting $c_i$ from the weights alone.

---

## Part II: running it

### 1. Package

`DC-Merge-official-experiment3.zip` = the Experiment 2 package plus:

```
lane_harm/run_oracle.py                    ← this experiment
docs/IMPLEMENTATION_GUIDE_EXPERIMENT3.md   ← this guide
```

It uses the same lane code (`lanes.py`) as Experiment 2. No other file changed.

### 2. What the script does, per context

1. Reads `dc_rank`'s selected α from Experiment 2a: `compare/<context>__dc_rank.json`.
2. **Sanity check:** with every $c_i = \alpha$, validation and test accuracies must reproduce Experiment 2a's `dc_rank` numbers.
3. **Coordinate search on validation**, starting from all $c_i = \alpha$. One task at a time, try $c_i \in \{0.3, 0.4, \dots, 1.0\}$ (plus α) with the others fixed, and keep any improvement. At most 2 passes over the tasks. Two objectives:
   - **`oracle_mean`:** maximise mean normalized accuracy.
   - **`oracle_fair`:** maximise mean normalized accuracy, **subject to the worst task's normalized accuracy not falling below `dc_rank`'s**.
4. **One test evaluation** for each of: `dc_rank`, `oracle_mean`, `oracle_fair`. CT-Merging's test result is taken from Experiment 2a.

Every validation evaluation is cached on disk, so if the run stops, rerunning the same command resumes the search where it left off.

### 3. Commands

Same environment and assets as Experiments 1–2 (`PYTHON`, `MODEL`, `FT_DIR`, `HEADS`, `DATA` set as in Experiment 2). Unpack into a new folder.

```bash
cd DC-Merge
EXP2=/srv/dcmerge/results/experiment2           # the Experiment 2 output (contains compare/)
OUT=/srv/dcmerge/results/experiment3            # new folder
ARGS="--repo $PWD --model $MODEL --ft-dir $FT_DIR --heads $HEADS --data $DATA --exp2 $EXP2 --device cuda"

# Step 1: the 3 EuroSAT contexts first (decisive, ~2-4 h)
"$PYTHON" lane_harm/run_oracle.py run $ARGS --out "$OUT" --contexts core 2>&1 | tee "$OUT-core.log"

# Step 2: all 7 four-task contexts (finished ones are skipped)
"$PYTHON" lane_harm/run_oracle.py run $ARGS --out "$OUT" --contexts four 2>&1 | tee "$OUT-four.log"
```

To rebuild the summary only: `"$PYTHON" lane_harm/run_oracle.py summarize --out "$OUT"`.

**Cost (estimate):** about 30–90 validation merges per context (the two objectives share cached evaluations), plus 3 test merges. Each merge is evaluated on the context's 4 tasks; there's no SVD work beyond building the lanes once per context.
- Step 1: roughly **2–4 hours**.
- Step 2: roughly **4–8 hours more**, on one RTX 3090/4090.

### 4. Output

| File | Content |
|---|---|
| `$OUT/oracle_summary.md` | **Read first.** Per context: coefficient vectors, test mean/worst normalized, EuroSAT. Then averages with Δ vs `dc_rank` and the verdict |
| `$OUT/oracle.csv` | The same numbers, one row per context and method |
| `$OUT/<context>/oracle_result.json` | Full record: search log (every accepted step), chosen vectors, validation and test scores, checks |
| `$OUT/<context>/val_cache.json`, `test_cache.json` | Every evaluated coefficient vector and its accuracies |

---

## Part III: expected outcomes and decision

### Must hold

| Check | Expected |
|---|---|
| Log line per context | `baseline vs Experiment 2a: val diff 0.000 pp, test diff 0.000 pp` (up to ~0.05 pp is acceptable: lanes are summed in a different order, which can flip a handful of predictions) |
| `oracle_fair` on validation | Worst normalized ≥ `dc_rank`'s worst (enforced by the search) |
| Both oracles on validation | Mean normalized ≥ `dc_rank`'s (the search starts there and only accepts improvements) |

If the diff is larger than 0.1 pp, stop and report: the run then differs from Experiment 2a.

### What we expect

- **On validation**, both oracles should improve on `dc_rank`. Experiment 2c already found single-lane changes that raised the mean (e.g. SVHN at 0.75: +1.85 pp).
- **The interesting question is test.** The oracle picks the best of dozens of vectors on validation, so part of its validation gain is luck. Test shows how much is real.
- **Likely direction of the vectors:** EuroSAT's coefficient **up**, SVHN's (and possibly GTSRB's) **down** in the 3 EuroSAT contexts. If the oracle mostly changes *other* tasks' coefficients, note it; that's a surprise worth reporting.

### Decision (applied to the averages over the contexts; written in the summary)

| Test result vs `dc_rank` (best of the two oracles) | Verdict | Next step |
|---|---|---|
| Δmean ≥ **+1 pp** and Δworst ≥ **+3 pp** | **HEADROOM** | Build the data-free predictor of $c_i$ |
| Δworst > 0 but Δmean < 0 | **FAIRNESS ONLY** | Fairness costs average; weaker story, discuss |
| Δmean ≤ 0 and Δworst ≤ 0 | **NO HEADROOM** | Drop the method; write the analysis paper |
| Anything else (gains below the thresholds) | **SMALL GAIN** | Report and discuss before deciding |

Also look at the **per-context** table: does the oracle help in all 3 EuroSAT contexts, or only one?

---

## Part IV: what to send back

1. `oracle_summary.md`, `oracle.csv`, and the logs.
2. A short note:
   - Did every "must hold" check pass?
   - The verdict on the 3 core contexts, then on all 7.
   - For each context, the chosen `oracle_fair` vector. Which tasks went up, and which went down?

## Limitations (state them when reporting)

- **This is an oracle,** tuned with validation labels. It bounds what a method could reach; it is not itself a method.
- **The search is a greedy coordinate search** on a grid of 0.3–1.0 with ≤ 2 passes. It can miss better vectors, so a negative result means "no headroom found by this search", not a proof that none exists.
- One checkpoint set, the selected 4-task contexts, and a single run.
