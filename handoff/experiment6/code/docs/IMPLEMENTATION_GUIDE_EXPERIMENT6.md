# Experiment 6: a better label-free signal (6a) and a finer coefficient (6b)

*Experiment guide for the mentee. It needs the outputs of Experiment 3 ([`IMPLEMENTATION_GUIDE_EXPERIMENT3.md`](IMPLEMENTATION_GUIDE_EXPERIMENT3.md)) and Experiment 5 ([`IMPLEMENTATION_GUIDE_EXPERIMENT5.md`](IMPLEMENTATION_GUIDE_EXPERIMENT5.md)), and the same assets and environment as Experiments 2–5.*

---

## Part I: the story

### What we know

- **Per-task coefficients have some headroom, mostly on the mean.** Experiment 3's validation oracle (`oracle_fair`) beat `dc_rank` on test by **+2.72 mean / +0.13 worst** in the 3 core contexts and **+1.63 / +1.16** over all 7. That is SMALL GAIN: the worst task barely moves.
- **Entropy cannot choose them.** Experiment 5's check failed its gate: ρ(−mean H, mean acc) = 0.35 / 0.43 / 0.01, and the entropy picks were 2.4–5.1 pp below `dc_rank`. In the saved caches, SVHN and MNIST sit near maximum entropy (H ≈ 0.94–0.99), and their entropy *rises* as their accuracy rises. So entropy pushes exactly the wrong lanes down.

### Two questions

**6a — Is the label-free signal the bottleneck?** We have each task's fine-tuned expert. On the same unlabeled images, a merged model that behaves like the expert is probably right where the expert is right. Three proxies, all compared with the expert $E_i$ (base CLIP + task $i$'s own LoRA update):

$$\text{kl}_i(c) = \tfrac1n\sum_x \mathrm{KL}\big(p_{E_i}(x)\,\|\,p_{M(c)}(x)\big),\quad
\text{top1}_i(c) = 1-\tfrac1n\sum_x [\arg\max p_{E_i} = \arg\max p_{M(c)}],\quad
\text{cos}_i(c) = 1-\tfrac1n\sum_x \cos\big(f_{E_i}(x), f_{M(c)}(x)\big).$$

Logits use CLIP's scale 100 on unit-norm features and heads (as in Experiment 5); $f$ is the image embedding after the visual projection, before the head. All three are written as losses, so **lower is better**.

> This proxy needs **unlabeled images and the experts**, so it is **not data-free**. It is the same regime as AdaMerging and AdaRank.

**6b — Is the coefficient too coarse?** One number per task scales all 12 blocks of its lane together. If the harm to EuroSAT is created in one part of the network, one coefficient per **(task, layer group)** could protect EuroSAT there without giving up SVHN/GTSRB elsewhere:

$$W(c) = W_0 + \sum_{i,g} c_{i,g}\, L_i\big|_g ,\qquad g \in \{\text{early (blocks 0–3)},\ \text{mid (4–7)},\ \text{late (8–11)}\}.$$

With $c_{i,g}=c_i$ for every $g$ this is exactly Experiment 3; with every $c_{i,g}=\alpha$ it is `dc_rank`.

**Important fact about the lanes.** The official config's LoRA adapters target only `q_proj, k_proj, v_proj, out_proj`. Each task therefore has **48 lane modules (12 blocks × 4 attention projections) and no MLP lanes**. The depth grouping is unaffected. The secondary "attention vs MLP" split is impossible, so the optional secondary grouping is **`qk_vo`** (query/key vs value/output) instead. `--grouping attn_mlp` exists but exits with this explanation. No module lies outside the 12 blocks; if one did, it would receive the scalar α for every task (the script prints the count).

---

## Part II: running it

### 0. Package

The Experiment 5 package plus:

```
lane_harm/run_agreement.py                 ← 6a
lane_harm/run_layer_oracle.py              ← 6b
lane_harm/rebaseline_exp3.py               ← current-environment replay of Exp 3 (step 0b)
scripts/run_exp6.sh                        ← launcher (all steps resumable)
docs/IMPLEMENTATION_GUIDE_EXPERIMENT6.md   ← this guide
```

No existing file changed. `run_layer_oracle.py` extends `lanes.LaneMerge` through a subclass defined inside the new file.

```bash
cd DC-Merge
export EXP3=/srv/dcmerge/results/experiment3     # per-context oracle_result.json, val_cache.json, test_cache.json
export EXP5=/srv/dcmerge/results/experiment5     # check_metadata.json and <context>/entropy_cache_n500_s0.json
export OUT=/srv/dcmerge/results/experiment6      # new folder
# PYTHON, MODEL, FT_DIR, HEADS, DATA as in Experiments 2-5 (or source /workspace/dcmerge_env.sh)
```

### 0b. Current-environment replay of Experiment 3 (required on a new machine; ~30–40 min)

On a different machine, a few near-tie predictions flip relative to the archived Experiment 3 numbers. The first real 6b run hit exactly this: 0.079 pp = 2 GTSRB validation images, 0.053 pp = 1 DTD test image, 0.148 pp = 4 EuroSAT images. The 6b checks exist to prove that the new composition code reproduces Experiment 3's model, and the smoke test already proves that at 0.000 pp. Against another machine's numbers, a 0.05 pp tolerance would instead demand bit-identical hardware behaviour. Experiment 3 solved the same problem with a current-environment baseline overlay; Experiment 6 does the same:

```bash
bash scripts/run_exp6.sh rebaseline
```

`lane_harm/rebaseline_exp3.py` uses Experiment 3's own code path (`LaneMerge.compose` and `run_oracle.Evaluator`). It re-evaluates Experiment 3's **already-chosen** vectors (`dc_rank`, `oracle_mean`, `oracle_fair`) on validation and test on this machine; nothing is re-selected. It writes an Experiment-3-shaped folder (`$EXP3_CURRENT`, default `${OUT}_exp3_current`) and `replay_summary.md` with every per-task difference from the archive. It aborts if:
- any difference exceeds 0.5 pp (that would be more than numerical flips), or
- a repeated identical evaluation on this machine is not exactly equal (non-determinism).

Which reference each step uses (the launcher does this automatically):

| step | `--exp3` | why |
|---|---|---|
| `6a-check` | archived Experiment 3 | the gate needs all of Experiment 3's cached vectors, which are internally consistent with each other |
| `6a-run-*`, all `6b-*` | `$EXP3_CURRENT` (this machine) | must-hold checks and the `dc_rank` / scalar-oracle comparisons are made on the same machine. The tolerance stays **0.05 pp** and the expected value is 0.000. |

Report `replay_summary.md`, and state in the write-up that 6a-run and 6b compare against the current-environment replay. CT rows are still the archived Experiment 2 records.

### 1. 6a: the expert-agreement gate (run first; ~1 h)

```bash
bash scripts/run_exp6.sh 6a-check
```

It does three things:

1. **Expert sanity check (labels used only here).** Each expert's full-validation accuracy must match `single_task/our_finetuned/ViT-B-32/val_acc.json`. If an expert misses by more than 1 pp, the script stops, because the expert construction would be wrong.
2. **Same unlabeled images as Experiment 5.** The subset comes from Experiment 5's own loader (500 validation images per task, seed 0). The script compares sizes and first indices with `$EXP5/check_metadata.json` and stops on any mismatch.
3. **The gate.** For every vector already in Experiment 3's `val_cache.json`, it computes the three proxies (no new labeled evaluation). It then reports Spearman correlations against the cached validation accuracies, and the vector each proxy would pick, next to the best cached vector, `dc_rank`, and Experiment 5's entropy on the same vectors.

Expert outputs are cached once per task (`$OUT/experts/`). Proxy values are cached per vector (`$OUT/<context>/agreement_cache_n500_s0.json`), so a rerun resumes.

**Stop here and send `agreement_check_summary.md` before running anything else for 6a.**

### 2. 6a: the search (only if the gate allows it)

```bash
bash scripts/run_exp6.sh 6a-run-core     # gate PASS or IN_BETWEEN
bash scripts/run_exp6.sh 6a-run-four     # gate PASS only
```

The script reads the gate from `$OUT/agreement_check.json` and refuses to run against the rule below. It searches with the gate's proxy, using the same grid, passes and greedy order as Experiment 3, for two objectives:

- `agreement_mean`: minimise the mean proxy loss.
- `agreement_fair`: minimise the worst task's proxy loss.

Each chosen vector is evaluated **once on test**, plus once on validation as a diagnostic. The `dc_rank`, `oracle_fair`, `oracle_mean` and CT rows are copied from Experiment 3.

### 3. 6b-1: mechanism (core, validation; ~1–2 h)

```bash
bash scripts/run_exp6.sh 6b-mechanism
```

All other lanes stay at the context's α. For SVHN, GTSRB, and both together, and for each group g, it evaluates:

| configuration | meaning |
|---|---|
| `off[N]` | neighbour(s) N fully off |
| `only[N]@g` | N on only in group g |
| `off[N]@g` | N off only in group g |
| `only[svhn]@g\|gtsrb_off`, `only[gtsrb]@g\|svhn_off` | for the per-group interaction (reference: both fully off) |
| `boost[eurosat]@g`, `boost[eurosat]@all` | EuroSAT's own lane × 1.5 in one group / everywhere |

That is 33 validation merges per context. All four tasks' accuracies are saved. The summary reports, for EuroSAT:

- **total harm:** EuroSAT with N fully off, minus all on.
- **created in g:** (N fully off) − (N only in g). This is the harm N creates when it exists only in g.
- **removed by g:** (N off only in g) − (all on).
- **interaction per g:** harm(both in g) − harm(SVHN in g) − harm(GTSRB in g).

### 4. 6b-2: the per-(task, group) oracle (~2.5–5 h per context)

```bash
tmux new -d -s exp6b "bash scripts/run_exp6.sh 6b-oracle-core"
# then
tmux new -d -s exp6b4 "bash scripts/run_exp6.sh 6b-oracle-four"
# optional, core only, if time allows
tmux new -d -s exp6q "bash scripts/run_exp6.sh 6b-qkvo-core"
```

There are 12 coordinates (4 tasks × 3 groups), searched in task-major order (`task@early, task@mid, task@late, …`). The grid is {0.3, …, 1.0} plus α, with at most 2 passes and the objectives `mean` and `fair` as in Experiment 3. The fair floor is `dc_rank`'s validation worst-task score. Each chosen vector is evaluated **once on test**.

**Cost.** In the CPU smoke test the search used 170–205 merges per context, against 57–105 in Experiment 3. At Experiment 3's pace (about 0.8 min per validation merge) that is roughly **2.5–3 h per context**, and up to about 5 h in the worst case. Budget about 8–15 h for core and about 10–20 h for the other four.

The two experiments can run at the same time on one GPU (each moves its model off the GPU between evaluations).

### 5. Output

| File | Content |
|---|---|
| `$OUT/agreement_check_summary.md`, `agreement_check.csv`, `agreement_check.json` | 6a gate: per proxy and context, ρ values and picks; the verdict and selected proxy |
| `$OUT/agreement_summary.md`, `agreement.csv`, `<context>/agreement_result.json` | 6a search result (only if run) |
| `$OUT/mechanism_summary.md`, `mechanism.csv`, `<context>/mechanism_result.json` | 6b-1 |
| `$OUT/layer_oracle_summary.md`, `layer_oracle.csv`, `<context>/layer_oracle_result.json` | **6b-2 main result**, with the chosen task × group tables |
| `$OUT/<context>/checks.json` | 6b must-hold checks |
| `$OUT/<context>/layer_val_cache.json`, `layer_test_cache.json` | every 6b evaluation (resume) |
| `$OUT/grouping_qk_vo/…`, `layer_oracle_summary_qk_vo.md` | optional secondary grouping |
| `$OUT-6a-check.log`, `$OUT-6b-*.log` | logs; each run prints `[split]` lines saying which split each number comes from |

---

## Part III: must-hold checks and decision rules

### Must hold (the scripts abort otherwise)

| Check | Expected |
|---|---|
| 6a expert check | each expert's full-validation accuracy within 0.5 pp of `val_acc.json`. Above 0.5 pp the script warns; above 1 pp it aborts. |
| 6a subset | `unlabeled subset identical to Experiment 5` for each core context |
| Replay | `replay_summary.md` exists; every difference vs the archive ≤ 0.5 pp; determinism repeat 0.000 pp |
| 6b: all $c_{i,g}=\alpha$ | validation **and** test accuracy equal the **current-environment** Experiment 3 `dc_rank` within **0.05 pp** (expected 0.000: same float path, same machine) |
| 6b: $c_{i,g}=c_i$ with Exp 3's `oracle_fair` | validation accuracy equals the current-environment replay within **0.05 pp** (expected 0.000) |
| 6b lane sums | `max_rel_sum_error < 1e-5` |
| Log | `[split]` lines present; no test number used for selection |

### 6a gate (pre-registered; core contexts, validation)

| Result | Verdict | Do next |
|---|---|---|
| Some proxy has ρ_mean ≥ **0.5** in all 3 core contexts **and** its pick beats `dc_rank`'s mean in all 3 | **PASS** | Search with that proxy (if several pass: most contexts where the pick beats `dc_rank`, then highest minimum ρ_mean, then the order kl, top1, cos). Core, then four. |
| Every proxy has ρ_mean ≤ **0.3** in some core context, **or** every proxy's pick is below `dc_rank` in all 3 | **FAIL** | Stop. Report the check. The label-free proxy is not the bottleneck to pursue. |
| Anything else | **IN BETWEEN** | Search on core only; the result is capped at **PARTIAL** |

The FAIL row uses the literal reading of the rule. The summary also states whether the stricter per-proxy reading (each proxy fails at least one condition) would fail.

**Search verdict** (from `agreement_summary.md`, averages vs `dc_rank` on test):

- **VIABLE:** keeps ≥ 50% of `oracle_fair`'s mean gain and no context loses > 1 pp on the worst task.
- **PARTIAL:** smaller gains, or some context drops > 1 pp on the worst task.
- **NOT VIABLE:** no mean gain.

### 6b-2 decision (pre-registered; test averages vs `dc_rank`, best of the two layer oracles)

| Result | Verdict | Next step |
|---|---|---|
| Δworst ≥ **+3 pp** AND Δworst exceeds scalar `oracle_fair`'s Δworst by ≥ **+2 pp** AND Δmean ≥ **+1 pp** | **LAYER LEVER** | Predictor of $c_{i,g}$; a layer-aware method |
| Δworst within **1 pp** of the scalar oracle's (or below it) | **NO LEVER** | The fairness ceiling is not a coefficient-granularity problem |
| Otherwise (above the scalar oracle but below the bars) | **SMALL GAIN** | Report and discuss |

The verdict is computed for core and for all 7 separately. "Best of the two" picks between `layer_mean` and `layer_fair` on test, so it is an optimistic upper bound.

**Overfitting check.** Twelve coefficients chosen on validation can overfit. The summary reports *shrinkage* (validation gain − test gain, both vs `dc_rank`) for the layer oracles and for Experiment 3's scalar oracles in the same contexts. If the layer oracle's validation gain is large but its shrinkage is too, the extra freedom is fitting validation noise.

### What we expect

- **6a:** agreement should track accuracy much better than entropy, because it is anchored to each task's own correct behaviour, not to confidence. The risk is that it tracks accuracy *too* well to add anything over the scalar oracle's ceiling. The gate only tells us whether the proxy can *find* the headroom, not whether the headroom is large.
- **6b-1:** if EuroSAT's harm is created mostly in one group (say, SVHN's late lanes), 6b-2 has something to exploit. If it is spread evenly, a per-group coefficient has little to work with.
- **6b-2:** a real gain must show up on the **worst task**. Experiment 3's ceiling was the worst-task score, not the mean.

---

## Part IV: what to send back

1. `agreement_check_summary.md` (immediately after step 1), then any 6a search outputs.
2. `mechanism_summary.md`, `layer_oracle_summary.md`, `layer_oracle.csv`, the `checks.json` files, and all `$OUT-6*.log` files.
3. A short note:
   - Did every must-hold check pass? Quote the `checks:` log lines.
   - The 6a gate verdict, per proxy and per context.
   - 6b-1: where in depth EuroSAT's harm is created, and whether the SVHN × GTSRB interaction sits in one group.
   - 6b-2: the verdict for core and all 7, the per-context table, the chosen task × group tables, and the shrinkage numbers.

## Limitations (state them when reporting)

- Greedy coordinate search on a coarse grid. A negative result means "none found by this search", not "none exists".
- 500 unlabeled validation images per task, one seed (0), Experiment 5's subset.
- One checkpoint set, one run, the selected seven four-task contexts.
- **6a is not data-free**: it needs unlabeled images and the per-task experts.
- 6b uses validation for 12 coefficients per context, and the "best of two" verdict is chosen on test. Read the shrinkage table before trusting a gain.
- There are no MLP lanes, so the guide's attention-vs-MLP split was replaced by the optional `qk_vo` split.
