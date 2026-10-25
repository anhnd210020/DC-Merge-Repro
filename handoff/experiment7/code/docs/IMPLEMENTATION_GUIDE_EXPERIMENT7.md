# Experiment 7: can unlabeled images choose the per-depth coefficients?

*Experiment guide for the mentee. It builds on Experiment 6 ([`IMPLEMENTATION_GUIDE_EXPERIMENT6.md`](IMPLEMENTATION_GUIDE_EXPERIMENT6.md)) and needs its results (branch `codex/experiment6`, commit `b97f466`).*

---

## Part I: the story

### What Experiment 6 showed

- **Per-depth coefficients are a real lever.** One coefficient per (task, depth group), chosen on validation, beat `dc_rank` on test by **+8.22 mean / +10.86 worst** in the 3 core contexts. Over all 7 contexts the gain was +5.04 / +6.33. The per-task oracle of Experiment 3 reached only +2.75 / +0.11 (core) and +1.64 / +1.16 (all 7).
- **Top-1 agreement with the experts ranks accuracy almost perfectly.** Spearman ρ was 0.95 / 0.98 / 0.97. The 6a search failed only because of its objectives: the raw `agreement_fair` always treated the weak experts (Stanford Cars, DTD) as the worst task.

### The questions

**7a — Does agreement still rank accuracy inside the per-depth family?** This is checked on the ~300–360 vectors Experiment 6 already evaluated, without any new labelled evaluation.

**7b — Why did the 6b oracle win?** Two controls test alternative explanations:

- **The depth schedule alone (`depth_only`):** one coefficient per depth group, shared by all tasks.
- **More search, not more granularity (`scalar_matched`):** one coefficient per task with a finer grid and the same number of validation evaluations as 6b.

**7c — The method.** Choose the 12 per-depth coefficients with agreement only, no labels. Two objectives, fixed in advance:

```latex
\textbf{agree\_floor: } \max_c \tfrac1T\sum_i \mathrm{agree}_i(c) \;\; \text{s.t.}\;\; \mathrm{agree}_i(c) \ge \mathrm{agree}_i(\alpha)\ \forall i
\qquad
\textbf{agree\_relfair: } \max_c \min_i \frac{\mathrm{agree}_i(c)}{\mathrm{agree}_i(\alpha)}
```

The ratio rel_i = agree_i(c) / agree_i(α) fixes 6a's failure: a weak expert no longer counts as the worst task just because its raw agreement is low.

**7d — Is 6b a lower bound?** The labelled oracle is rerun on a wider grid {0.0, …, 1.5}, starting from the 6b vectors.

**7e — Is it the objective or the machinery?** Experiment 5's entropy objective is run through the identical 12-coordinate search. Label it "entropy objective on per-depth lanes", not "AdaMerging".

> **7c is not data-free.** It needs the per-task experts and unlabeled in-distribution images (the AdaMerging / AdaRank regime). Its design was chosen after seeing Experiment 6, so the hypothesis is not independent of it, even though it is pre-registered here.

### Definitions

- `agree_i(c)`: the fraction of the 500 unlabeled validation images of task i (seed 0, the Experiment 5/6 subset) where argmax W(c) equals argmax of expert i. This is exactly 6a's `top1`. The experts' predictions are **loaded from Experiment 6's cache** (`experiment6/experts/*.pt`, hashes logged).
- `entropy_i(c)`: Experiment 5's normalized prediction entropy, logit scale 100, on the same images.
- W(c) = W0 + Σ_{i,g} c_{i,g} L_i|g, with the 6b groups: early = blocks 0–3, mid = 4–7, late = 8–11. The 12 coordinates are ordered `task@early, task@mid, task@late, …`.

---

## Part II: running it

### 0. Package and paths

`DC-Merge-experiment7.zip` adds three things and changes no existing file:

- `lane_harm/run_exp7.py` (all subcommands)
- `scripts/run_exp7.sh` (launcher)
- this guide

It also includes the CPU smoke test in `tests/`.

Assemble the runnable tree from the Experiment 6 branch:

```bash
git clone --branch codex/experiment6 https://github.com/anhnd210020/DC-Merge-Repro /workspace/DC-Merge-Repro
R=/workspace/DC-Merge-Repro
mkdir -p /workspace/ducanh && cp -r $R/handoff/experiment6/code /workspace/ducanh/DC-Merge
cp -r $R/handoff/experiment3/code/vision_lora_merge/dataset/shuffled_idxs /workspace/ducanh/DC-Merge/vision_lora_merge/dataset/
cd /workspace/ducanh && unzip -o DC-Merge-experiment7.zip      # adds the Exp 7 files
```

Environment file `/workspace/ducanh/exp7_env.sh`:

```bash
export PYTHON=/venv/dcmerge_vision/bin/python
export MODEL=/workspace/models/clip-vit-base-patch32
export FT_DIR=/workspace/models/DC-Merge-Vision-B32-r16-8task/lora_checkpoints
export HEADS=/workspace/models/DC-Merge-Vision-B32-r16-8task/lora_heads
export DATA=/workspace/datasets8
export EXP3R=/workspace/DC-Merge-Repro/handoff/experiment6/results/experiment6_exp3_current
export EXP6=/workspace/DC-Merge-Repro/handoff/experiment6/results/experiment6
export OUT=/workspace/ducanh/results/experiment7
export REFERENCE=current        # see "Reference mode" below
```

Every run prints the paths it uses.

**Parallel runs.** Different steps may run at the same time on the same context: the validation, test and proxy caches are merged under a file lock, so no entry is lost. The smoke test confirmed that concurrent and sequential runs give identical results and caches. Never start the same step on the same context twice at once.

### Reference mode (decide once, before step 1)

The comparisons need `dc_rank`, the Experiment 3 scalar oracles and the 6b oracles evaluated **on the same machine** as Experiment 7.

- **`REFERENCE=archive`** compares against the numbers saved by Experiment 6. Checks 1–2 must hold within 0.05 pp, or the run aborts. Use it only on the machine that ran Experiment 6.
- **`REFERENCE=current`** uses `replay` first, which re-evaluates those vectors on this machine without selecting anything. Every difference from the archive is recorded. The replay aborts above 0.5 pp, or if a repeated evaluation is not identical. Checks 1–2 then compare against the replay (expected 0.000 pp).

This is the rule Experiment 6 adopted when it moved machines. On a new server use `current`, and report `replay_summary.md`. The mode is stored in `OUT/run_config.json` and cannot change within one OUT.

### 1. Replay and checks (~1 h)

```bash
source /workspace/ducanh/exp7_env.sh && cd /workspace/ducanh/DC-Merge
bash scripts/run_exp7.sh replay four        # REFERENCE=current only
EXTRA_ARGS=--recompute-experts bash scripts/run_exp7.sh checks four
```

### 2. 7a gate (core; ~30–60 min; STOP afterwards)

```bash
bash scripts/run_exp7.sh gate core
```

This computes agreement for every vector in Experiment 6's `layer_val_cache.json`, about 300–360 per core context. The mechanism configurations are included. The accuracies are Experiment 6's cached validation accuracies, so no new labelled evaluation is made.

**Send `gate_summary.md` before running anything expensive.**

### 3. 7b controls (core first; ~3–6 h per context, run the three in parallel)

```bash
for c in dtd-eurosat-gtsrb-svhn eurosat-gtsrb-mnist-svhn eurosat-gtsrb-sun397-svhn; do
  tmux new -d -s e7b-$c "source /workspace/ducanh/exp7_env.sh && cd /workspace/ducanh/DC-Merge && bash scripts/run_exp7.sh controls $c"
done
```

- **`depth_only`:** 3 coordinates, grid {0.3,…,1.0} plus α, at most 2 passes, objectives `mean` and `fair`.
- **`scalar_matched`:** 4 coordinates, grid {0.20, 0.25, …, 1.20} plus α. It runs passes until no improvement for both objectives, then seeded random restarts (seed 0, starting points drawn uniformly from the fine grid). It stops when its distinct validation evaluations equal Experiment 6's `val_evals` for that context (`layer_oracle_result.json`).
- For each objective, the chosen vector is the best of **everything evaluated within the budget**. This equals the best finished run's endpoint, and also counts a run the budget cut off. It is generous to the control, so it is conservative for the 6b claim.

### 4. 7c method and 7e entropy (only if the gate allows; ~1–2 h per context)

```bash
bash scripts/run_exp7.sh method core   && bash scripts/run_exp7.sh entropy core
bash scripts/run_exp7.sh method four   && bash scripts/run_exp7.sh entropy four    # gate PASS only
```

Both scripts read `OUT/gate.json` and refuse to run against the rule. Both use the 12 coordinates, the exact 6b grid plus α, task-major order, at most 2 passes, and start from all c = α. Agreement and entropy come from one forward pass on the unlabeled images and share one cache, so 7e costs little extra. Each chosen vector is scored once on test, plus once on labelled validation as a diagnostic only.

### 5. 7d wide grid (core, descriptive; can take 5–20 h per context, run last)

```bash
bash scripts/run_exp7.sh wide core
```

This runs the labelled oracle with grid {0.0, …, 1.5} plus α, starting from the 6b chosen vectors, for up to 5 passes. It costs up to about 190 evaluations per pass.

### 6. All 7 contexts, optional all8, summaries

```bash
bash scripts/run_exp7.sh controls four       # parallel per context, as above
bash scripts/run_exp7.sh all8                # only if 7c (all 7) is VIABLE or STRONG; refuses otherwise
bash scripts/run_exp7.sh summarize
```

`all8` runs `agree_floor` on 8 tasks × 3 groups = 24 coordinates. Its `dc_rank` baseline (α = 0.4) is evaluated on this machine, and the archived `dc_official` all8 value is quoted.

### 7. Output

| File | Content |
|---|---|
| `replay_summary.md` | this machine vs archive for every reference vector |
| `<context>/checks.json` | checks 1–5 with all numbers |
| `gate_summary.md`, `gate.csv`, `gate.json` | 7a: ρ values, picks of rules (i)–(iii), the 6b picks, verdict |
| `controls_summary.md`, `controls.csv` | 7b: per context and averages, chosen vectors, budget used, verdict |
| `method_summary.md`, `method.csv`, `method.json` | 7c + 7e: per context, averages, shares of the 6b gain, task × group tables, verdicts |
| `wide_summary.md` | 7d: extra gain over 6b and shrinkage |
| `all8_summary.md` | optional all8 run |
| `<context>/*_result.json` | full records: search logs, test logs (check 6), chosen vectors |
| `$OUT-<step>-<contexts>.log` | logs; every number is tagged with its split by a `[split]` line |

---

## Part III: must-hold checks and decision rules

### Must hold (the script aborts otherwise)

| # | Check | Expected |
|---|---|---|
| 1 | all c = α reproduces `dc_rank` validation and test | ≤ 0.05 pp (0.000 with `current`) |
| 2 | the 6b `layer_mean` / `layer_fair` vectors reproduce their validation accuracies | ≤ 0.05 pp (0.000 with `current`) |
| 3 | lane sums | `max_rel_sum_error` < 1e-5 |
| 4 | unlabeled subset identical to Experiments 5/6 (size, split size, image indices) and agreement for Experiment 3 vectors equals 6a's cached values | identical, 0 images different |
| 5 | expert predictions loaded from Experiment 6's cache (sha256 logged); with `--recompute-experts`, recomputed on this machine | 0 argmax mismatches |
| 6 | each chosen vector evaluated once on test | counts in every result file's `test_log` and the "done" log line |

If check 4 fails only because agreement differs by one or two images, **stop and report.** That can be near-tie flips on a new machine. Rerunning with `EXTRA_ARGS="--agree-tol-images 2"` is a protocol change for the supervisor to approve.

### 7a gate (pre-registered; core, validation)

| Result | Verdict | Do next |
|---|---|---|
| ρ_mean ≥ 0.7 in all 3 core contexts AND the pick of rule (ii) or rule (iii) beats `dc_rank` on both mean and worst in all 3 | **PASS** | 7c and 7e, core then all 7 |
| ρ_mean < 0.5 in any core context, OR no rule's pick beats `dc_rank` on the worst task in any core context | **FAIL** | No 7c/7e. Report. 7b and 7d still run |
| anything else | **IN BETWEEN** | 7c/7e on core only, at most PARTIAL |

The rules are:

- (i) max mean agreement;
- (ii) max mean agreement with rel_i ≥ 1 for every task;
- (iii) max min_i rel_i, ties broken by mean agreement.

"Rule (ii) or (iii)" means one rule wins in all three contexts. The summary also prints the verdict under the looser reading, where the rules may differ per context. FAIL's second clause is read literally ("in none of the three"); the stricter reading is printed beside it.

### 7b controls (pre-registered; core averages, test, Δ vs `dc_rank`)

"Best of two" per arm means the objective with the larger Δworst, ties broken by Δmean.

| Result | Verdict |
|---|---|
| 6b beats BOTH controls by ≥ +2 pp worst and ≥ +1 pp mean | **TASK-SPECIFIC DEPTH** |
| `depth_only` reaches ≥ 70% of the 6b mean gain AND is within 1 pp of it on worst | **DEPTH SCHEDULE SUFFICES** |
| `scalar_matched` within 1 pp of 6b on worst AND on mean | **SEARCH BUDGET EXPLAINS IT** |
| none of these | **MIXED** (if two hold, both are printed) |

### 7c verdict (pre-registered; all 7, test, Δ vs `dc_rank`; `agree_floor` primary)

| Result | Verdict |
|---|---|
| Δmean ≥ 50% of the 6b `layer_fair` Δmean AND Δworst ≥ +3 pp AND no context loses > 1 pp on its worst task | **STRONG** |
| the same, with average Δworst > 0 | **VIABLE** |
| positive Δmean, but a condition above fails | **PARTIAL** |
| Δmean ≤ 0 | **NOT VIABLE** |

Core is reported separately. An IN BETWEEN gate caps the result at PARTIAL. The 7e rows get the same rule, for comparison only.

### 7d (descriptive)

If the average extra test gain over 6b exceeds 2 pp, the summary states that the 6b numbers are a lower bound.

---

## Part IV: what to send back

1. `replay_summary.md`, every `checks vs …` log line, and `gate_summary.md` (immediately after step 2).
2. `controls_summary.md`, `method_summary.md`, `wide_summary.md`, the CSVs, `all8_summary.md` if run, and all logs.
3. A short note:
   - every must-hold check,
   - the 7a verdict per context,
   - the 7b verdict,
   - the 7c verdict (core and all 7) with the chosen task × group tables,
   - how 7e compares with 7c,
   - the 7d extra gain.

## Limitations (state them when reporting)

- 7c's design was chosen after seeing Experiment 6. It is pre-registered now, but the hypothesis is not independent of Experiment 6.
- Greedy coordinate search on a coarse grid; a negative result means "none found by this search".
- One checkpoint set, one seed, 500 unlabeled images per task, selected contexts.
- 7c is not data-free: it needs the experts and unlabeled in-distribution images.
- The 7a gate pairs this machine's agreement with Experiment 6's cached validation accuracies, which come from a different machine (near-tie flips are at most 0.2 pp).
- With `REFERENCE=current`, all comparisons are against this machine's replay of the reference vectors.
