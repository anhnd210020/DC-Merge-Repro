# Experiment 2: who harms the victim, and can turning them down save it?

*Experiment guide and story for the mentee. It builds on Experiment 1 ([`IMPLEMENTATION_GUIDE_OFFICIAL.md`](IMPLEMENTATION_GUIDE_OFFICIAL.md)). The background is in [`MENTEE_METHOD_GUIDE.md`](MENTEE_METHOD_GUIDE.md).*

---

## Part I: the story

### Where we are

Experiment 1 ran on official DC-Merge with the authors' checkpoints. It reproduced the paper (73.86% vs 73.90% normalized test accuracy) and found four things:

1. **Keeping 16 instead of 4 coefficients per task** raises accuracy in every one of 28 task–context pairs (+8.30 pp on average).
2. **EuroSAT is still a victim** when merged with GTSRB and SVHN. In context `dtd-eurosat-gtsrb-svhn` it scores 63.3%, against 85.0% when merged alone: 21.7 pp of harm.
3. **Conflict inside EuroSAT's own lane is not the cause.** All four TIES-level interventions recover 0 pp.
4. Replacing EuroSAT's own lane with its smoothed update recovers 17–21% of its harm, but this intervention changes several stages at once and has no matched control for that replacement. It does not isolate a cause.

At least roughly **80% of the harm remains after that replacement**. A hypothesis to test is that **neighbours' lane terms**, particularly GTSRB's and SVHN's, change shared layers that EuroSAT's images also pass through. The results so far do not establish this hypothesis.

### Why that matters for a paper

The closest competitor, **CT-Merging** (arXiv 2607.20561), scales each task by **its own** energy and never looks at who it is merged with. If harm comes from the neighbours, an own-energy rule can't protect a victim. Rescaling the **neighbours** can. Turning the harmful neighbours down a little might save the victim and still raise the average, because strong tasks sit on a plateau where a small cut costs little.

### The three questions in this experiment

| Part | Question | Why it matters |
|---|---|---|
| **2a: Compare** | Does CT-Merging already rescue EuroSAT? | If yes, our gap is much smaller; we must know first |
| **2b: Decompose** | Exactly how much of EuroSAT's harm comes from each neighbour, and from neighbours *together*? | Turns "probably the neighbours" into a measured fact |
| **2c: Dose** | If we turn the harmful neighbours **down** (not off), does EuroSAT gain more than they lose? | The go/no-go test for a "save the victim, keep the average" method |

### How 2b works: an exact decomposition

A DC-Merge update is **exactly** a sum of one term per task, its lane:

$$\Delta_{\text{merge}} = L_{\text{eurosat}} + L_{\text{gtsrb}} + L_{\text{svhn}} + L_{\text{dtd}}.$$

We keep the real merge's shared frame and simply leave lanes out. For every subset of neighbours (2³ = 8 merges) we measure EuroSAT's accuracy $f(\text{subset})$. EuroSAT's harm then splits into parts that **add up exactly**, with no unexplained remainder:

- **Frame cost** = merged alone − own lane only in the shared frame.
- **Per-neighbour harm**, e.g. *GTSRB alone* = $f(\varnothing) - f(\{\text{gtsrb}\})$.
- **Interactions**, e.g. *GTSRB + SVHN* = the extra harm when both are present, beyond the sum of each alone.

### How 2c works: a dose test

Scale one neighbour's lane by λ = 0.25, 0.5, 0.75 (with λ = 0 coming from 2b), and both GTSRB and SVHN together. Also try **boosting EuroSAT's own lane** to 1.25 and 1.5. For each setting we record all four tasks, so we see EuroSAT's gain, the neighbour's own loss, and the change in the mean.

---

## Part II: running it

### 1. Package

`DC-Merge-official-experiment2.zip` = the official DC-Merge (commit `089af77`, no official file changed) plus:

```
lane_harm/
├── run_week1.py     ← this experiment: compare / decompose / summarize
├── ct_merging.py    CT-Merging, implemented from the paper (no official code exists)
├── lanes.py         per-task lane terms of a DC-Merge merge (remove / rescale)
├── dc_trace.py      official dc_merge with intermediates (from Experiment 1)
├── run_official.py  Experiment 1 runner; also used for asset loading
└── lane_harm.py     Experiment 1 method
```

**Built-in safeguards** (the run stops if any fails):
- the lane terms add back up to the real merge;
- on the official operator, the merge is **bit-for-bit identical** to official `dc_merge`;
- the decomposition parts sum exactly to the harm;
- CT-Merging's bases are orthonormal.
- at α = 0.4, the official validation scores match the sealed Experiment 1 reference (within 0.5 pp); the all8 official winner is α = 0.4 and its normalized test score matches within 0.5 pp. The decomposition's full-context scores are likewise checked against the corresponding references.

### 2. Setup

Same environment and assets as Experiment 1 (`IMPLEMENTATION_GUIDE_OFFICIAL.md`, section 3). Unpack the new zip into a **new** folder, activate the same Python environment (`PYTHON` as in Experiment 1), then:

```bash
cd DC-Merge
REPO=$PWD
MODEL=$ASSETS/clip-vit-base-patch32
FT_DIR=$ASSETS/lora_checkpoints
HEADS=$ASSETS/lora_heads
DATA=$ASSETS/datasets
OUT=/srv/dcmerge/results/experiment2          # new folder
COMMON="--repo $REPO --model $MODEL --ft-dir $FT_DIR --heads $HEADS --data $DATA --out $OUT --device cuda"
```

### 3. Run (in this order, inside `tmux`)

**2a: Compare** (DC-Merge official, DC-Merge rank variant, CT-Merging; 7 four-task contexts + the 8-task merge)

```bash
"$PYTHON" lane_harm/run_week1.py compare $COMMON 2>&1 | tee "$OUT-compare.log"
```

- The merge coefficient (α for DC-Merge, γ for CT-Merging) is chosen per method and context on **validation**, from one shared grid: 0.1 … 1.0, 1.3, 1.5, 2.0, 2.5, 3.0. It contains CT-Merging's grid and α = 0.4.
- Each method is then evaluated **once on test** at its chosen value.

**2b + 2c: Decompose and dose** (the 3 EuroSAT + GTSRB + SVHN contexts; run after 2a)

```bash
"$PYTHON" lane_harm/run_week1.py decompose $COMMON 2>&1 | tee "$OUT-decompose.log"
```

Defaults: both operators (`rank`, `official`), and two α values per context:
- the one **selected in 2a** for that operator and context;
- **0.4**, for comparison with Experiment 1.

Validation split.

- Both commands verify completed result files before skipping them on rerun; after an interruption, run the same command again.
- To rebuild the summaries: `"$PYTHON" lane_harm/run_week1.py summarize --out "$OUT"`.

**Cost (rough estimate):**
- 2a: about 350 validation evaluations of 4 tasks (8 for `all8`), plus 23 test evaluations. There is no attribution step, so it's faster per merge than Experiment 1.
- 2b+2c: 2 operators × 2 α × 3 contexts × 22 lane configurations, plus one solo evaluation per setting ≈ 276 evaluations. If the selected coefficient is 0.4, its duplicate setting runs once.
- **Expect roughly 5–10 hours in total on one RTX 3090/4090.**

### 4. Output

| File | Content |
|---|---|
| `$OUT/compare/compare_summary.md` | **Read first.** Per context and method: coefficient, test mean, mean normalized, worst normalized task, EuroSAT. Then averages over the 7 contexts |
| `$OUT/compare/compare.csv`, `<context>__<method>.json` | All numbers, including the full validation sweep |
| `$OUT/decompose/decompose_summary.md` | Per context, operator and α: the decomposition table and the dose table |
| `$OUT/decompose/decompose.csv`, `*.json` | All numbers, including every configuration's 4 accuracies |

---

## Part III: expected outcomes and how to decide

### Must hold (otherwise stop and report)

| Check | Expected |
|---|---|
| `all8 / dc_official` in 2a | Selected α = **0.4**, test mean normalized ≈ **73.86** (Experiment 1) |
| `dc_official` rows, Experiment 1 contexts | At α = 0.4 the validation accuracies in the sweep equal Experiment 1's official baseline accuracies (up to GPU evaluation nondeterminism) |
| Decompose log | No error. The lane and official checks raise if they fail, and `lane_checks.max_abs_diff_vs_official` = 0.0 for the official operator |
| Decomposition table | Each table's parts sum to its "total harm" (enforced) |

### 2a: Does CT-Merging rescue EuroSAT?

- **All 8 tasks:** Table 2 of the CT-Merging paper reports 72.55 normalized for CT-Merging and 72.86 for DC-Merge on ViT-B/32, 8 tasks, DC-Merge adapters. **Expect CT-Merging close to `dc_official`**; its precise value can differ with our selected coefficient grid and evaluation setup. A large gap either way calls for checking our independent CT-Merging implementation before drawing a conclusion.
- **4-task contexts:** compare EuroSAT and worst-task accuracy against `dc_rank`.

| Result (3 EuroSAT core contexts) | Meaning |
|---|---|
| CT-Merging's EuroSAT ≥ `dc_rank`'s EuroSAT **+ 5 pp** in all three, with a better worst-task | CT-Merging largely rescues the victim. **Our "neighbour-blind" gap is weak**; report, and we rethink |
| Otherwise | The gap stands: own-energy scaling doesn't protect the victim |

### 2b: Where does the harm come from? (read the `rank` operator at the selected α first)

**Predicted:** small frame cost; DTD, MNIST and SUN397 each small; GTSRB and SVHN each clearly larger; and a **positive GTSRB + SVHN interaction** (together they hurt more than separately).

| Result | Meaning |
|---|---|
| Neighbour terms (all except frame cost) ≥ 60% of total harm | **Harm is neighbour-driven** → the method should rescale neighbours |
| Frame cost ≥ 50% of total harm | Harm mostly comes from sharing the frame, not from what neighbours add → rescaling neighbours won't help much |
| \|GTSRB + SVHN interaction\| ≥ 25% of total harm | **Interaction matters**: pairwise rules (like Net Utility's penalty) can't capture it. A strong point for the paper |

**About the official operator.** Its "merged alone" keeps only one coefficient per module, so its harm-vs-solo numbers aren't meaningful. Read its dose table only: the changes in accuracy are valid.

### 2c: Is there a cheap lever?

For each dose setting, compare EuroSAT's gain with the **summed losses of the other three tasks**.

| Result | Meaning |
|---|---|
| Some setting gives EuroSAT gain ≥ **2×** the others' summed loss, **and** mean of 4 > 0 | **Favourable trade: the method is viable.** We can save the victim and raise the average |
| EuroSAT gain ≈ others' loss | Redistribution only: fairness is possible, but no average gain |
| EuroSAT gains < 2 pp even with GTSRB+SVHN at 0.5 | **No lever:** neighbour strength isn't where the harm can be fixed |
| Boosting EuroSAT's own lane helps as much as turning neighbours down | The simpler lever (boost the victim) is enough; report which is cheaper |

---

## Part IV: what to send back

1. `compare_summary.md`, `compare.csv`, `decompose_summary.md`, `decompose.csv`.
2. The two logs.
3. A short note:
   - Did every "must hold" check pass?
   - The 2a verdict: does CT-Merging rescue EuroSAT?
   - For each of the 3 contexts (`rank` operator, selected α): frame cost, GTSRB alone, SVHN alone, GTSRB+SVHN interaction, total harm.
   - The best dose setting: EuroSAT gain, others' summed loss, change in the mean.

## Known limitations

- **CT-Merging is our implementation** from the paper's equations; no official code exists. Unstated details are handled conservatively and noted in `ct_merging.py`: the eigen-solver (an equivalent SVD route, checked), degenerate residual directions, and which split γ is selected on (we use validation).
- **Removing a lane keeps the full merge's frame and TIES votes.** That is intended: it isolates each neighbour's *contribution* to the real merge, rather than re-merging without that neighbour. The exact additive decomposition is an accounting identity over subset accuracies in this fixed frame; interaction terms do not by themselves prove a unique biological or causal mechanism. At k = 16 the TIES interventions inside EuroSAT's lane had no measured accuracy effect in Experiment 1.
- The packaged `lane_harm/experiment1_reference.json` was extracted from the sealed Experiment 1 archive (hash recorded inside the file). The 0.5 pp tolerance is an explicit allowance for evaluation nondeterminism, not an equivalence proof when that margin is reached.
- One seed, the validation split for 2b/2c, 3 contexts. Enough to decide whether to build the method; not enough to publish.
