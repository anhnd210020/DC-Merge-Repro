# Who pays for a merge? Method guide

*Problem statement, motivation and proposed method. There is no code here; see [`IMPLEMENTATION_GUIDE_OFFICIAL.md`](IMPLEMENTATION_GUIDE_OFFICIAL.md) for that.*

---

## 1. Problem statement

**Model merging** combines several task-specific fine-tuned models into one model, using only their weights: no training data and no further training. For each task *i*, the **task vector** is the change from the pretrained weights. A merging method turns a set of task vectors into one merged update.

Subspace-based methods (TSV-M, Iso-CTS, DC-Merge) are the current state of the art. They decompose each task's update into singular directions and give each task a share of a common low-dimensional space. Every such method spends limited shared resources on each task:

- **rank**: how many directions a task keeps;
- **scale**: how strongly the task's update enters the merge;
- **support**: in DC-Merge, how many coefficients survive sparsification.

These methods are judged by **average** accuracy across tasks. An average can rise while one task is badly damaged.

> **Problem.** In a multi-task merge, which task is harmed, by how much, and by which step of the merging operator? Can the merge's shared resources be reallocated so that the worst-off tasks gain a lot while the strongest lose little, without any task data?

**Harm** has a precise definition here. The harm to task *i* in a context *S* (the set of tasks merged together) is

> harm*ᵢ*(S) = accuracy of *i* merged **alone** through the same operator − accuracy of *i* in the merge of *S*.

Comparing against "merged alone" separates the damage caused by the **other tasks** from the damage the operator does to every task anyway (truncation, smoothing, scaling).

---

## 2. Motivation

### 2.1 Evidence from our own experiments

- **A fixed operator can still hide a victim.** DC-Merge's original sparsification rule keeps about `0.001 × R²` coefficients per task. That suits the published 8-task setting, where it equals the rank. In our 4-task contexts it keeps about **one coefficient per task per module**. Changing it to "keep *k = r* coefficients" raised mean retention from 66.9% to 81.8%, and worst-task retention from 53.4% to 68.9%. It won in 45/45 contexts (validation, one seed).
- **The improvement was not uniform.** In the three contexts containing **EuroSAT, GTSRB and SVHN**, EuroSAT lost about **16–19 pp** even though the context average rose.
- **What this suggests:**
  - average-based evaluation misses victims;
  - whether a task becomes a victim depends on *which companions* it is merged with;
  - the merging operator's own budgets decide who loses.

### 2.2 What existing work does not cover

| Work | What it does | What it leaves open |
|---|---|---|
| SERA (2026), PDRA (ours) | Allocate rank per task from the task's **own** spectrum | Ignores the companions; optimises the mean |
| Net Utility / Budget-Aware LoRA Merging (2026) | Data-free direction selection with a **pairwise** interference penalty | Separable surrogate; no worst-task objective |
| AdaRank (ICLR 2026) | Learns per-task ranks | Needs unlabeled test data; optimises entropy |
| Iso-C / Iso-CTS | Flattens the spectrum for "balanced" merging | Same treatment for every task; does not target victims |
| METIS (ICML 2026) | Reweights tasks by loss gap; helps the worst task | Needs data and several rounds of fine-tuning |

No existing method explains **which operator step creates the harm**, predicts victims from weights alone in a given context, *and* reallocates resources with an explicit worst-task objective.

### 2.3 Why we need a mechanism first

Our earlier allocator (PDRA) predicted rank demand from one spectral feature and assigned ranks by a fixed template. That is a heuristic: it says *what* to do but not *why*. Its labels were also measured under the broken one-coefficient operator.

Before building any allocator, we must know **where harm comes from**. Otherwise we risk optimising a proxy that has nothing to do with the cause.

---

## 3. Proposed method

The method has three stages. **Your current task is Stage 1.** Stages 2 and 3 build on its result.

### Stage 1: Causal harm attribution (current experiment)

**Idea.** In DC-Merge every task owns a separate block, its **lane**, in a shared orthogonal coordinate frame (the *cover space*). The merged update is exactly a sum of the lanes' contributions. A task's update reaches the merged model through a fixed chain of steps:

1. scale by the merge coefficient α;
2. truncate to rank *r*;
3. smooth the singular values;
4. **project into the cover frame**: part of the update can fall outside the task's own lane, where it is discarded (**spill / rotation**);
5. sparsify (keep *k* coefficients);
6. **TIES sign voting inside the lane**: companions can average with it (**dilution**), out-vote it (**conflict**), or write where it had nothing (**injection**);
7. **reconstruct**: other tasks' lanes can also overlap with what this task reads (**cross-lane readout**).

Steps 4, 6 and 7 are where the companion tasks act. They are the candidate causes of context-dependent harm.

**Weight-space accounting.** For each task we measure how much error each step adds, relative to the task's own update and read on the task's own input directions. The step contributions add up exactly to the total. We also record how much energy each companion places in each task's lane, separately for every pair of tasks, so we see who affects whom.

**Causal interventions.** Correlations aren't enough, so for a chosen target task *p* we remove one mechanism at a time and re-evaluate accuracy:

| Intervention | Removes |
|---|---|
| Protect lane | all companion influence in *p*'s lane |
| Fix conflict / Undilute / Block injection | one TIES effect each |
| Restore own update | spill and rotation: *p*'s lane is replaced by its own update |
| **Matched control** | the same kind and amount of edit, applied to a *different* task's lane |

Every intervention except "restore own update" changes **only** *p*'s lane, and this is verified on each run. Any effect on the other tasks is therefore a genuine side-effect and is measured as **cost**.

**What this tells us.**

| Result | Meaning |
|---|---|
| An intervention removes most of *p*'s harm, and the control does not | That step **causes** the harm |
| The control helps as much | The effect is only "less energy", not a specific cause |
| Fixing the cause costs the other tasks as much as it gains | The harm can be moved but not reduced |
| Nothing removes much harm | Harm comes from cross-lane readout or from the network's nonlinearity |

**Theoretical anchor.** If the tasks' subspaces are exactly orthogonal, every context term is zero: no spill, no leakage, no TIES damage, no cross-lane readout. All context harm therefore comes from **overlap between tasks**, and the attribution shows which form that overlap takes.

### Stage 2: Predicting harm without data (next)

Once the cause is known, build a predictor of each task's harm in a context from weights alone. It combines:

- **own-spectrum features**: how much a task needs, e.g. how spread out its singular values are;
- **companion-geometry features** tied to the identified cause: e.g. the task's spill share, or the energy a specific companion writes into its lane.

The predictor is fitted once on validation, frozen, and tested on **held-out task combinations** and leave-one-task-out. The pipeline is PDRA's, but the target is harm instead of rank demand, and the features are chosen by mechanism instead of by search.

### Stage 3: Fair allocation (later)

Choose each task's rank, scale and support under fixed total budgets to maximise a **β-fair welfare**:

- **β = 0**: maximise the average (what current methods do);
- **β = 1**: proportional fairness;
- **β → ∞**: maximise the worst task.

One parameter moves along the trade-off "a little less for strong tasks, much more for weak ones." With concave predicted utilities, the max-min case is solved by raising a common floor until the budget runs out (water-filling). The cost of fairness, i.e. how much average is lost per point of worst-task gain, can be bounded by how strongly the tasks overlap, and that bound is a testable prediction.

The allocator is then applied to DC-Merge, and later TSV-M and Iso-CTS, through each method's own resources.

### What would end the project (decided in advance)

- **Stage 1** finds no cause that removes ≥ 20% of a victim's harm with a clean control. Then the operator offers no lever, and an allocator would only be a heuristic.
- **Stage 3's premise fails:** with an oracle allocation, raising the worst task by about 5 pp costs more than about 1 pp of average. Then there is nothing cheap to redistribute.

---

## 4. Scope of the current task

- **Code:** the official DC-Merge repository (commit `089af77`), unmodified. ViT-B/32, LoRA rank 16, the authors' 8-task checkpoints.
- **Operator:** the official DC-Merge (rank 16 per task, top-k = 0.1% of the cover space, linear smoothing ρ = 5). For comparison, our rank-aware variant (keep 16 coefficients per task) is also run.
- **Merge coefficient:** α is chosen with the official validation sweep on the 8-task merge. That step also checks we reproduce the paper's Table 1.
- **Contexts:** the published 8-task merge, the 3 four-task contexts with EuroSAT + GTSRB + SVHN, and 4 control contexts. **Validation split only.**
- **Out of scope for now:** final test of the interventions, other backbones, and changes to the merge code.
