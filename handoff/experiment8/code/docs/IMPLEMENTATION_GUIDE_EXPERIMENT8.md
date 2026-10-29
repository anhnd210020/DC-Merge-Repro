# Experiment 8: where in depth is merging harm created? Implementation plan

Oct 27, 2026 · @ihb

*Living version: https://claude.ai/artifact/66bD6tirQZgkUEYDiMysPc (includes the pipeline diagram).*

## Summary

Experiment 8 is a confirmatory test of the project's main finding: **merging harm to a victim task is created in the middle transformer blocks**. Experiment 6 discovered this for one victim (EuroSAT), one operator (DC-Merge) and LoRA adapters. Experiment 8 tests it for every victim task, under three operators (task arithmetic, Iso-CTS, TSV-M), on the standard full fine-tuned CLIP ViT-B/32 checkpoints, with a ViT-L/14 replication.

It tests five claims in a fixed order: the harm is **localized** mid-depth; the localization is **specific** to interference rather than generic sensitivity; the victim's task **information is lost** there; the mid-depth harm is **amplified** by later blocks; and removing the harmers only mid-depth **repairs** the victim at a smaller cost to the harmers.

- **Design.** Depth is split into three equal thirds, as in Experiment 6. For each (victim, context) pair, a full factorial of 8 layer-swap hybrids gives an exact Shapley allocation of the harm to the thirds, the same tool we use for "who is responsible".
- **Statistics.** No fixed percentage thresholds. Each claim is a contrast with an explicit null (equal allocation, a norm-matched random perturbation, or equal cost to victim and harmers). Inference uses an exact sign-flip test with victims as the unit, a fixed-sequence procedure for multiplicity, and cluster-bootstrap intervals.
- **No winner's curse.** Pairs and harmers are chosen on one half of each test split; every reported number uses the other half.
- **Cost.** About 9-10 GPU-hours on one RTX 3090, or about 5-6 hours in a reduced configuration (one context per victim, no ViT-L/14).
- **Code.** It extends the existing `exp8` package. Stage A (selection) reuses its steps; Stage B adds the depth steps and a new `stats.py`; the old `depth`, `patch` and `weightspace` steps and the fixed-threshold verdicts are removed.

## Background and claims

Experiment 6 (DC-Merge rank, LoRA on attention only, three EuroSAT contexts) found that switching GTSRB and SVHN off **only in blocks 4-7** removed 96-98% of EuroSAT's harm, while switching them on only there recreated just 44-65%. That result was exploratory: one victim, one operator, one checkpoint type, and the depth pattern was noticed after the data were in. Experiment 8 turns it into a pre-specified, confirmatory test. The depth partition (thirds), the contrasts and the tests below are fixed before any Stage B data exist.

**What changes relative to Experiment 6:**

|  | Experiment 6 | Experiment 8 |
| --- | --- | --- |
| Victims | EuroSAT only | Every task that is significantly harmed in some context |
| Operator | DC-Merge rank (lanes) | Task arithmetic, Iso-CTS, TSV-M (no lanes needed) |
| Checkpoints | Rank-16 LoRA, attention only | Full fine-tuning (attention and MLP) |
| Backbone | ViT-B/32 | ViT-B/32, plus ViT-L/14 (24 blocks) replication |
| Depth allocation | Switch-off and switch-on shares | Exact Shapley allocation over the 3 thirds (8-hybrid factorial) |
| Control | None | Norm- and spectrum-matched random perturbation per third |
| Mechanism | None | Linear probes, activation patching, cross-term generation vs transport |
| Inference | Point estimates | Pre-specified tests with explicit nulls, victim-level exact tests, cluster-bootstrap intervals |

**The five claims, in the order they are tested** (each presupposes the ones before it):

1. **Localization.** More of the harm is allocated to the middle third than to the early third, and more than to the late third.
2. **Specificity.** The excess of real harm over the harm of a random perturbation of the same size and spectrum is also concentrated in the middle third. This rules out "the middle blocks are just where the network is most sensitive".
3. **Information loss.** The gap in linearly decodable victim information, between the merge with and without the harmers, grows more across the middle third than across the early or late third.
4. **Amplification.** The middle third's harm is larger when the other thirds also carry the harmers: removing the harmers only mid-depth fixes more than adding them only there creates.
5. **Repair separation.** Removing the harmers only in the middle third recovers a larger fraction of the victim's harm than the fraction of their own accuracy it costs the harmers.

Claim 5 is the bridge to a method. It explains, beyond DC-Merge, why per-depth coefficients break the worst-task ceiling that per-task coefficients hit.

## Design at a glance

&#91;embedded content: Experiment 8 pipeline · two halves, two stages\]

Stage A works only on the selection half and ends by freezing the pairs and harmers. Stage B measures them on the report half; the probes are trained on selection-half images but scored on the report half. `depthstats` combines both, using the selection-half harm as each victim's fixed normalizer. The ViT-L/14 replication repeats `factorial` and `random` for task arithmetic.

## Pairs, splits and depth units

### Splits

Each task's test split is divided once (seed 0) into two disjoint halves, as in the existing package:

- **Selection half.** Used for the coefficient α, all 162 subset merges, the Shapley responsibilities, the choice of pairs and harmers, and the normalizers used in the tests (capped at 5,000 images per task, the existing `screen` split).
- **Report half.** Every Stage B number is computed here, and nothing on it is used to choose anything.

Probes are the one place the selection half is used again in Stage B: they are *trained* on it and *evaluated* on the report half. This is legitimate, because the probes do not choose pairs, harmers or any quantity a test uses.

### Pairs: one rule, stratified by victim

For each operator o, victim task v and each of the 35 four-task contexts S that contain v, using only the selection half:

1. **Harmers** N\_h = the neighbours of v with positive Shapley responsibility in S (from the existing subset table). If N\_h is empty, the pair is not a candidate.
2. **Harm from the harmers** Γ\_sel = f\_v(M(S \\ N\_h)) − f\_v(M(S)), in pp of normalized accuracy. Both merges are already in the subset table.
3. **Eligibility:** (a) the lower end of the 95% paired image-bootstrap interval of Γ\_sel (2,000 resamples) is above 0. That is the standard requirement that the harm is distinguishable from sampling noise; no size threshold is imposed. (b) The harmers' summed full-removal loss Σ\_h [f\_h(M(S)) − f\_h(M(S \\ N\_h))] on the selection half is above 0, because that sum is the denominator of ρ\_h in claim 5. This is a well-definedness condition, not a size threshold; it almost always holds, since removing a task's own vector costs it most of its fine-tuning gain. A single harmer whose own loss is negative is allowed and logged.
4. **Choice:** among the eligible contexts for (o, v), the k = 2 with the largest Γ\_sel (ties broken by the task order in `config.TASKS`).

This gives at most 8 victims × 2 contexts × 3 operators = 48 pairs. Every victim with at least one eligible pair enters the analysis; victims with none are listed with their largest Γ\_sel and its interval. The harmer set N\_h and Γ\_sel are frozen in `selection.json` and never recomputed on the report half.

Why stratify by victim: the 10 most-harmed pairs overall would likely be one or two victims (as with EuroSAT before), and a depth profile could then be a property of one task. With one or two pairs per victim, the victim becomes the unit of generalization.

### Depth units and thirds

The vision tower is split into units: **pre** (patch, class and position embeddings and the pre-LayerNorm), **blocks 0 to L−1**, and **post** (the final LayerNorm). The visual projection and text heads are pretrained and shared, so they are not merged.

The three players of the depth factorial are equal thirds of the blocks, with the two non-block units attached to the adjacent third:

| Third | ViT-B/32 (L = 12) | ViT-L/14 (L = 24) |
| --- | --- | --- |
| Early (E) | pre + blocks 0-3 | pre + blocks 0-7 |
| Middle (M) | blocks 4-7 | blocks 8-15 |
| Late (Lt) | blocks 8-11 + post | blocks 16-23 + post |

Equal thirds are the coarsest partition that can separate "middle" from both ends, and they are the partition under which Experiment 6 made its finding. The block-level scan (step B2) measures pre and post separately, so we can check that attaching them does not drive the result.

Within each block, parameters are further labelled **attention** (q, k, v, out projections and `layer_norm1`) or **MLP** (`fc1`, `fc2` and `layer_norm2`), for the descriptive attention-versus-MLP split.

## Quantities measured

All accuracies are normalized (merged ÷ expert × 100, in pp) and computed on the report half, unless marked raw. For a pair (v, S, N\_h) under operator M with coefficient α:

```latex
A = \theta_0 + \alpha\, M(S), \qquad B = \theta_0 + \alpha\, M(S \setminus N_h)
```

For a set of thirds G ⊆ {E, M, Lt}, the **hybrid** H\_G takes the parameters of A in the units of G and those of B elsewhere, so H\_∅ = B and H\_{E,M,Lt} = A. f\_t(X) is task t's normalized accuracy under parameters X.

### 1. Depth allocation of the harm (factorial over thirds)

The harm present when the harmers are in the thirds G:

```latex
u(G) = f_v(B) - f_v(H_G), \qquad u(\varnothing) = 0, \qquad u(\{E,M,Lt\}) = \Gamma = f_v(B) - f_v(A)
```

All 8 hybrids are evaluated, so u is known on every subset. The **Shapley allocation** φ\_g of Γ to third g and the **Harsanyi dividends** d(T) are the same quantities used for "who", with thirds as the players:

```latex
\varphi_g = \sum_{T \ni g} \frac{d(T)}{|T|}, \qquad d(T) = \sum_{R \subseteq T} (-1)^{|T|-|R|}\, u(R), \qquad \sum_g \varphi_g = \Gamma \ \text{exactly}
```

The **depth share** of third g, pooled over a set of pairs P, is a ratio of sums (never a mean of per-pair ratios, which is unstable when a pair's Γ is small):

```latex
\sigma_g = \frac{\sum_{p \in P} \varphi_{g,p}}{\sum_{p \in P} \Gamma_p}
```

**Creation and removal by one third.** Creation c\_g = u({g}): the harm when the harmers are present only in g. Removal r\_g = Γ − u({E,M,Lt} \\ {g}): the harm removed when the harmers are taken out only in g. Their difference for the middle third is exactly the sum of the interaction dividends that involve it, which is the amplification quantity of claim 4:

```latex
\kappa_M = r_M - c_M = d(\{E,M\}) + d(\{M,Lt\}) + d(\{E,M,Lt\})
```

### 2. Random-perturbation reference (specificity)

For each third g, let D\_g = (A − B) restricted to the units of g. A **matched random perturbation** P\_g has, for every 2-D weight matrix, the same singular values as the corresponding block of D\_g with random orthonormal singular vectors; every other tensor is Gaussian noise rescaled to the same Frobenius norm. With s = 1, 2, 3 random seeds:

```latex
\text{rand}_g = \frac{1}{3}\sum_{s=1}^{3} \big[ f_v(B) - f_v(B + P_g^{(s)}) \big], \qquad e_g = c_g - \text{rand}_g
```

e\_g is the **excess harm** of the real harmers in third g over a perturbation of exactly the same size and spectrum in the same place. Both terms add something to B in third g only, so they are directly comparable.

### 3. Information loss (linear probes)

For model X ∈ {A, B, expert of v, θ\_0} and depth index ℓ = 0 (input to block 0, after the pre-LayerNorm) to L (output of the last block), a multinomial logistic probe for v's classes is trained on the class-token residual state of the selection half and evaluated on the report half. Let acc^X\_ℓ be its raw accuracy (%). The **probe gap** and its increment across third g (boundaries at ℓ = 0, L/3, 2L/3, L) are:

```latex
\Delta_\ell = \text{acc}^{B}_\ell - \text{acc}^{A}_\ell, \qquad \lambda_E = \Delta_{L/3} - \Delta_0, \quad \lambda_M = \Delta_{2L/3} - \Delta_{L/3}, \quad \lambda_{Lt} = \Delta_L - \Delta_{2L/3}
```

The **information-loss fraction** ψ = Δ\_L ÷ Γ\_raw compares the final probe gap with the head-accuracy gap, both raw. ψ near 1 means the harm is loss of linearly decodable information. ψ near 0 means the information is still present but the head misreads it, a late readout problem.

### 4. Repair by removing the harmers only mid-depth

The hybrid H\_{E,Lt} keeps the harmers everywhere except the middle third. Over all tasks of the context:

```latex
\rho_v = \frac{f_v(H_{E,Lt}) - f_v(A)}{\Gamma}, \qquad \rho_h = \frac{\sum_{h \in N_h} [ f_h(A) - f_h(H_{E,Lt}) ]}{\sum_{h \in N_h} [ f_h(A) - f_h(B) ]}
```

ρ\_v is the fraction of the victim's harm recovered, and ρ\_h is the fraction of the harmers' accuracy lost, relative to removing them entirely (in B the harmers are absent, so their accuracy falls close to zero-shot). If a task's skill and the harm it causes were spread over depth in the same way, ρ\_v = ρ\_h. The contrast ρ\_v − ρ\_h measures how far depth separates the two. The same quantities are also recorded for the other two single-third removals and for the mean and worst normalized accuracy of the four tasks (descriptive).

## Statistical analysis plan

Every confirmatory claim is a contrast against an explicit null, tested with victims as the unit of inference and ordered so that the familywise error rate is 5%. No percentage-of-pairs thresholds are used anywhere; effect sizes are reported with intervals whatever the test says.

### Units and structure

The data are nested: images within pairs, pairs (up to 2 contexts × 3 operators) within victims. The claim is about victim tasks in general, so **the victim is the unit of inference**. Contexts and operators sharpen each victim's estimate but are not independent replications. Victims are not fully independent either: a victim in one pair can be a harmer in another, which is a stated limitation.

### Victim-level statistic

For each pair p, x\_p is its contrast in pp (table below), computed on the report half. Each victim's statistic sums its pairs within an operator, divides by their **selection-half** harm, and averages over the operators where it has pairs:

```latex
z_v = \frac{1}{|O_v|} \sum_{o \in O_v} \frac{\sum_{p \in P(o,v)} x_p}{\sum_{p \in P(o,v)} \Gamma^{\mathrm{sel}}_p}
```

The normalizer comes from data that the test never sees, and it is positive by the eligibility rule. That avoids two common faults: a ratio whose denominator is itself noisy, and a normalization that depends on the outcome. For claim 5, x\_p is already a fraction, so z\_v is the plain average over the victim's pairs and operators, with each fraction formed as a ratio of sums within (o, v).

### Confirmatory hypotheses

| Order | Claim | Pair-level contrast x\_p | Null hypothesis | Test |
| --- | --- | --- | --- | --- |
| 1 | Localization | φ\_M − φ\_E and φ\_M − φ\_Lt | The middle third receives no more of the harm than the early third, or than the late third | Intersection-union of two one-sided sign-flip tests |
| 2 | Specificity | e\_M − e\_E and e\_M − e\_Lt | The excess of real over matched random harm is not larger mid-depth | Intersection-union, as above |
| 3 | Information loss | λ\_M − λ\_E and λ\_M − λ\_Lt | The probe gap grows no more across the middle third than across the others | Intersection-union, as above |
| 4 | Amplification | κ\_M = r\_M − c\_M | The middle third does not interact with the other thirds (removal = creation) | One-sided sign-flip test |
| 5 | Repair separation | ρ\_v − ρ\_h | Removing the harmers mid-depth costs them the same fraction as it recovers for the victim | One-sided sign-flip test |

**Sign-flip test.** With n victims and statistic T = mean of z\_v, the p-value is the share of all 2^n sign patterns s for which the mean of s\_v z\_v is at least T, enumerated exactly. It is valid when each z\_v is symmetric about 0 under the null, and makes no normality or variance assumption, which matters with about 8 units. Where the null is a boundary ("no more than"), the test is conservative.

**Intersection-union.** Claims 1-3 hold only if the middle third beats *both* other thirds. The two one-sided p-values are combined by their maximum, which is a valid level-α test of the joint claim with no adjustment.

**Multiplicity: fixed sequence.** The five claims are tested in the order above, each at α = 0.05, stopping at the first that is not rejected. This controls the familywise error rate at 5% without splitting α. The order follows logical dependence: specificity, information loss and amplification each presuppose localization, and repair is the most derived claim. Claims after the first failure are still reported, with estimates and intervals, but labelled exploratory.

**Feasibility.** With n victims, the smallest attainable p-value is 2^−n. At least 5 eligible victims are needed (1/32 = 0.031); with fewer, no confirmatory claim is possible and the results are reported as estimates only. This is checked and printed by `select`.

### Effect sizes and intervals

- **Estimands:** the depth shares σ\_E, σ\_M, σ\_Lt; the pooled contrasts behind each claim; the interaction share κ\_M ÷ Γ; the repair fractions ρ\_v and ρ\_h; and ψ. All are ratios of sums over pairs.
- **Intervals:** 95% percentile intervals from a two-stage cluster bootstrap with 2,000 resamples. Each resample draws victims with replacement, then each drawn victim's pairs with replacement, then each drawn pair's report-half images with replacement. The image draw is shared by all conditions of that pair (the 8 hybrids, the random controls, and the probe predictions), so the paired structure is kept. With about 8 clusters, percentile intervals cover only approximately; the exact test is the basis for the claims and the intervals describe the size of the effect.
- **Per operator:** the same estimands with intervals, for each operator separately, plus the operator-to-operator difference in σ\_M with a victim-paired bootstrap interval. These are reported, not tested.

### "Who is responsible" (secondary family)

Stage A also yields the neighbour decomposition for the same pairs. It is tested as a separate family with Holm's procedure over two tests:

- **W1, interactions exist:** victim-level sum of interaction dividends, normalized by selection-half context harm. Null: zero. Two-sided sign-flip test.
- **W2, harm is about task content:** real neighbours' harm minus harm from norm- and spectrum-matched fake neighbours. Null: no difference. One-sided sign-flip test.

Also reported, without tests: the interaction share as a ratio of sums, the agreement of neighbour responsibility across operators (mean Spearman correlation of Shapley vectors, with a victim-cluster interval), and the breadth distribution of context harm over all 280 (victim, context) pairs on the selection half.

### Pre-declared sensitivity analyses

These are reported next to the main results; none changes a confirmatory verdict:

1. one context per victim (k = 1) instead of 2;
2. leave one victim out: the range of each statistic, and whether any sign changes;
3. the mean of per-pair ratios instead of the ratio of sums;
4. each operator separately;
5. the random reference with one seed instead of three;
6. the size of the pre and post units' single-unit effects (from B2), to show that attaching them to the end thirds does not drive claim 1.

### Exploratory analyses (no p-values)

Block-level creation and removal profiles; attention versus MLP within each third; where the victim's own update matters (the same factorial between M({v}) and θ\_0); activation patching by block; cross-term generation versus transport by block; the weight-space interference profile (TSV-M's singular task interference per third) against σ; the information-loss fraction ψ; repair by removing the early or late third; and the ViT-L/14 replication.

### Reporting rules

- Every selected pair is analysed and reported, including any whose harm on the report half turns out to be zero or negative. Nothing is excluded after selection.
- Every test is reported with n, the statistic, the exact p-value and the interval, including failures.
- The analysis code is written and tested on synthetic data before Stage B runs. `select` records a hash of `config.py`, and `depthstats` refuses to run if the configuration changed after selection.

## Stage A: selection

Stage A runs on the selection half and reuses the existing `exp8` package. It produces α per operator, the 162-subset table, the frozen list of pairs and harmers, and the "who" results for those pairs.

| Step | Status | What it does | Split |
| --- | --- | --- | --- |
| `prepare-data` | Unchanged | Downloads each test split once; caches uint8 images; writes the two halves | — |
| `check` | Extended | Existing asset, determinism and operator checks, plus three new ones: the block runner reproduces the model's hidden states; feature extraction matches the forward pass; and throughput in images/s is measured and logged, so the cost table can be recalibrated | full / selection |
| `alpha` | Unchanged | Global α per operator on 10 fixed random contexts, official grids | selection |
| `subsets` | Unchanged | All 162 subsets of size 1-4, each evaluated on its member tasks | selection |
| `analyze` | Reduced | Writes every (victim, context) pair's neighbour decomposition and Shapley vector, plus descriptive breadth. **The fixed-threshold verdicts (old H1-H3) are removed**, and so is the old top-10 selection | selection |
| `select` | **New** | The victim-stratified rule above; extra evaluations of each harmer task on B; writes `selection.json` with the config hash and the feasibility check (n victims ≥ 5) | selection |
| `report` | Changed | Runs on the pairs in `selection.json` with N\_h frozen: the 8-subset neighbour decomposition on the report half, the all-fake and culprit-fake controls, and per-image correctness kept for the bootstrap | report |

### `select` in detail

Inputs: `alpha.json`, `analyze.json` and the evaluation cache of the operator.

1. For every (v, S) pair in `analyze.json`: take N\_h = {j : φ\_j > 0} from the selection-half Shapley vector. Skip the pair if N\_h is empty.
2. Load the per-image correctness of v under M(S) and M(S \\ N\_h) on the `screen` split; both are cached by `subsets`. Compute Γ\_sel and a 2,000-resample paired image bootstrap of it (normalizing by the expert's correctness on the same resampled images).
3. Interval condition (rule 3a): the 2.5th percentile is above 0.
4. For each candidate that meets it, evaluate every harmer h on B = M(S \\ N\_h) on the `screen` split (h is not a member of B, so `subsets` has not evaluated it). Store each harmer's full-removal loss f\_h(A) − f\_h(B); their sum is the normalizer for claim 5 and must be positive (rule 3b). Rank the candidates that meet both conditions by Γ\_sel and keep the top k per victim (default 2, flag `--k-per-victim`).
5. Write `selection.json` with: operator, α, pairs (victim, context, harmers, Γ\_sel and its interval, harmer losses), victims without an eligible pair (with their best Γ\_sel and interval), n\_victims, and `config_sha256`.

`select --all-ops` then writes a combined `selection_all.json`, with the union of victims across operators, which `depthstats` uses for the pooled test.

### `report` changes

- It iterates over `selection.json`, not over the old top-10 plus ordinary pairs.
- The harmer set is read from `selection.json`; it no longer recomputes harmers from report-half Shapley values.
- The fake controls (all neighbours faked; main culprit faked) are kept unchanged; they feed W2.

## Stage B: the depth steps

Every Stage B step reads `selection.json`, runs on the report half, stores per-image correctness in the operator's evaluation cache (so each step resumes and can be bootstrapped), and writes one JSON file per operator. "Victim eval" means one evaluation of the victim on its report half.

### B1 `factorial`: Shapley allocation over thirds (claims 1, 4, 5)

- **Conditions:** the 8 hybrids H\_G, G ⊆ {E, M, Lt}. H\_∅ = B and H\_all = A are the plain merges.
- **Evaluated on:** all four tasks of the context (the victim, the harmers and the remaining neighbour). That is up to 32 evaluations, fewer where A and B are already cached by `report`.
- **Computes:** u(G), φ\_g, d(T), c\_g, r\_g and κ\_M for the victim. For each single-third removal H\_{all \\ {g}}: ρ\_v, ρ\_h, and the mean and worst normalized accuracy of the four tasks. For each task t, the Shapley allocation over thirds of the accuracy change between A and B (where each task's accuracy depends on the harmers' depth placement).
- **Checks:** H\_∅ and H\_all reproduce B and A per image, exactly; Σφ\_g = Γ within 1e-9.
- **Output:** `factorial.json`.

### B2 `blocks`: block-level profile (exploratory)

- **Conditions:** for each of the L + 2 units (pre, blocks, post), removal (that unit from B, the rest from A) and creation (that unit from A, the rest from B), so 2(L + 2) victim evals. Then attention-only and MLP-only versions of each third's removal and creation, another 12 victim evals.
- **Computes:** per-unit removal and creation as fractions of Γ; the sum of single-unit removals (whose distance from 1 measures non-additivity over depth); the attention and MLP shares of each third.
- **Output:** `blocks.json`.

### B3 `random`: matched random reference (claim 2)

- For each third g: compute D\_g = A − B on the units of g. For each 2-D matrix, take its SVD (cached per pair) and build U′ S V′ᵀ with random orthonormal U′ and V′. Every other tensor becomes Gaussian noise rescaled to the tensor's Frobenius norm. The seeds are fixed: seed = 10,000 × pair index + 10 × third index + s, for s = 1, 2, 3.
- **Conditions:** B + P\_g^(s), 9 victim evals.
- **Computes:** rand\_g, e\_g = c\_g − rand\_g (c\_g from B1), and the Frobenius norm of D\_g per third and per block (descriptive).
- **Checks:** the singular values of each random matrix match those of D\_g (relative error below 1e-5), and the norms of the other tensors match.
- **Output:** `random.json`.

### B4 `own`: where the victim's own update matters (exploratory)

- **Conditions:** the same 8-hybrid factorial between O = θ\_0 + α M({v}) and θ\_0, so 6 new victim evals. It is computed once per (operator, victim) and shared by that victim's pairs.
- **Computes:** the Shapley allocation over thirds of the victim's own gain f\_v(O) − f\_v(θ\_0). Comparing it with σ tells whether the harm lands where the victim's skill is stored.
- **Output:** `own.json`.

### B5 `probes`: information loss (claim 3)

- **Models:** A and B per pair; the victim's expert and θ\_0 per victim (cached, shared).
- **Features:** the class-token residual state at ℓ = 0, …, L, in fp32. Training images are the victim's selection half, capped at 5,000 with a fixed seeded subset. Test images are the report half, capped the same way.
- **Probe:** features standardized with the training mean and standard deviation. Multinomial logistic regression with an L2 penalty λ ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1}, chosen by accuracy on a fixed 20% hold-out of the training images and then refit on all training images. Full-batch L-BFGS in torch on the GPU, fp32, at most 200 iterations. The grid and hold-out are standard and identical for every model and depth.
- **Computes:** acc^X\_ℓ for the 4 models, Δ\_ℓ, λ\_E, λ\_M, λ\_Lt and ψ. Per-image probe correctness on the test images is stored for the bootstrap.
- **Sanity (warning, not abort):** the expert's final-depth probe should not fall more than 2 pp below the expert's head accuracy, and θ\_0's probe should be below the expert's for most victims.
- **Output:** `probes.json`.

### B6 `patch`: activation patching (exploratory)

- On a fixed patch subset of at most 2,000 report-half images: for each block ℓ, A's forward pass with B's block-ℓ output (all tokens) substituted.
- **Computes:** restored share s\_ℓ = (acc\_patched − acc\_A) ÷ (acc\_B − acc\_A), raw.
- **Check:** substituting A's own block output reproduces A's predictions exactly.
- **Output:** `patch.json`.

### B7 `crossterm`: where the interaction is generated (exploratory)

- Pairs with at least two harmers. p and q are the two harmers with the largest selection-half Shapley values; R = S \\ {p, q}.
- **Corners:** M(R), M(R + p), M(R + q) and M(S). The residual state (all tokens, fp32) is taken at every block input h^X\_ℓ.
- **Computes**, with F^X\_ℓ the block-ℓ map of model X and h̄ = h^{R+p} + h^{R+q} − h^R:

```latex
I_\ell = h^{S}_\ell - h^{R+p}_\ell - h^{R+q}_\ell + h^{R}_\ell, \qquad G_\ell = F^{S}_\ell(\bar h) - F^{R+p}_\ell(\bar h) - F^{R+q}_\ell(\bar h) + F^{R}_\ell(\bar h), \qquad T_\ell = F^{S}_\ell(h^{S}_\ell) - F^{S}_\ell(\bar h)
```

M\_ℓ is computed explicitly from the marginal-path mismatches, so that I\_{ℓ+1} = G\_ℓ + T\_ℓ + M\_ℓ. Reported per block: the mean norms of I, G, T and M (class token and all tokens), and the transport gain ‖T\_ℓ‖ ÷ ‖I\_ℓ‖. At the output: the victim's accuracy under the actual merge and under the additive logit prediction.

- **Check:** ‖I\_{ℓ+1} − (G\_ℓ + T\_ℓ + M\_ℓ)‖ ÷ ‖I\_{ℓ+1}‖ below 1e-4 at every block. This catches state mix-ups in the code.
- **Note:** with full fine-tuning the embeddings change too, so I\_0 is not zero (unlike the LoRA setting of the *Orientation, Not Magnitude* analysis).
- **Output:** `crossterm.json`.

### B8 `weightspace`: weight-space reference (exploratory)

- **Computes:** TSV-M's singular task interference per block and per third, using rank 1/|S| per task, as in the TSV-M paper; the victim-harmer subspace overlap per third; and the Frobenius share of D per third.
- **Compares:** the Spearman correlation between the per-block interference and the per-block removal profile from B2, and the depth at which each peaks. No GPU evaluations.
- **Output:** `weightspace.json`.

### B9 `depthstats`: the analysis

- Refuses to run if `config_sha256` differs from the one in `selection_all.json`.
- Computes, in this order: the victim-level z for every contrast; the exact sign-flip tests and intersection-union p-values; the fixed-sequence verdicts; the cluster-bootstrap intervals; the per-operator estimates; the six sensitivity analyses; and W1 and W2 with Holm's procedure.
- **Output:** `depth_summary.md`, `depth_summary.json`, `who_summary.md`, `who_summary.json`.

### B10 `replicate`: ViT-L/14 (task arithmetic only, exploratory)

1. Run `check` and `alpha` with `--backbone ViT-L-14 --tau-device cpu --alpha-contexts 5 --screen-cap 2000`. Task vectors stay on the CPU, since 8 × 1.2 GB would not fit next to the images on 24 GB.
2. **Candidates:** every ViT-B/32 task-arithmetic pair in `selection.json`. For each, the 8 neighbour subsets are evaluated on the victim (L/14, selection half, cap 2,000), and N\_h and Γ\_sel are re-derived with the same rule as `select`.
3. Keep up to 6 eligible pairs, preferring distinct victims. Run B1 and B3 on the 24-block thirds.
4. **Report:** σ\_E, σ\_M and σ\_Lt with cluster-bootstrap intervals, next to the ViT-B/32 task-arithmetic values. Too few units for a test; the question is whether the middle third keeps the largest share at a different depth scale.

- **Output:** `vitl14/` with the same files.

## Code map

The package is `experiment8/` (Python 3.10+, torch, transformers 4.x or 5.x, scipy, datasets). No new dependencies: probes use torch.

| File | Content |
| --- | --- |
| `exp8/config.py` | Every fixed setting: tasks, assets, FusionBench reference accuracies (ViT-B/32 and ViT-L/14), operator grids, split caps (`SCREEN_CAP` 5,000, `PROBE_TEST_CAP` 5,000, `PATCH_CAP` 2,000, `CROSSTERM_CAP` 500), `BOOTSTRAP` 2,000, `DEPTH` (k per victim 2, α 0.05, 3 random seeds, probe grid, 5 minimum victims, 6 replication pairs), `THIRDS`, `HYPOTHESIS_ORDER`, `thirds(L)`, `boundaries(L)`, `config_sha256()`. No percentage thresholds |
| `exp8/data.py` | Image cache; splits `sel`, `screen`, `rep`, and the seeded report-half subsets `probe`, `patch`, `crossterm` |
| `exp8/models.py` | Assets (task vectors optionally on CPU via `--tau-device`), parameter `unit_of` and `kind_of` (attn, mlp, other), the evaluator, `hidden_all`, `run_block` (transformers 4.x and 5.x), `logits_with_patch` |
| `exp8/operators.py` | TA, Iso-CTS, TSV-M (official ports), fake neighbours, `hybrid_sel(A, B, units, kinds, predicate)`, `in_units`, `matched_random_like` |
| `exp8/harm.py` | Neighbour decomposition, `shapley_from_value` (any player set, used for thirds) |
| `exp8/stats.py` | `sign_flip_test`, `iut`, `fixed_sequence`, `holm`, `percentile_ci`, `ratio_of_sums`, `multinomial_counts` |
| `exp8/probes.py` | Deterministic multinomial logistic probes with hold-out choice of the L2 penalty |
| `exp8/pipeline.py` | Shared runner (evaluation cache, merges) and Stage A: `check`, `alpha`, `subsets`, `analyze`, `select`, `select --all-ops`, `report` |
| `exp8/depth.py` | Stage B: `factorial`, `blocks`, `random`, `own`, `probes`, `patch`, `crossterm`, `weightspace`, and the replication's candidate re-derivation |
| `exp8/depthstats.py` | The analysis (B9) and the replication summary (B10). Reads per-image correctness from the cache and recomputes every statistic from it, so the bootstrap can resample images |
| `exp8/run.py` | Command line (below) |
| `scripts/run_exp8.sh` | The run order, with the pause after task arithmetic |
| `tests/test_exp8.py`, `tests/official_reference.py` | Unit and end-to-end tests |

### Command line

`python -m exp8.run <step> [--op ta|iso_cts|tsvm] --out OUT --data-dir DATA [flags]`. Steps: `prepare-data`, `check`, `alpha`, `subsets`, `analyze`, `select`, `report`, `factorial`, `blocks`, `random`, `own`, `probes`, `patch`, `crossterm`, `weightspace`, `depthstats`, `replicate`, and `all` (Stage A then Stage B for one operator).

| Flag | Default | Use |
| --- | --- | --- |
| `--k-per-victim` | 2 | contexts kept per (operator, victim) |
| `--all-ops` | off | `select`: write the pooled `selection_all.json` |
| `--tau-device` | same as `--device` | `cpu` keeps task vectors off the GPU (ViT-L/14) |
| `--alpha-contexts`, `--screen-cap` | 10, 5,000 | smaller values for the replication |
| `--probe-depths` | `all` | `boundaries` probes only ℓ = 0, L/3, 2L/3, L (enough for claim 3; halves probe time) |
| `--skip-svd-check` | off | `check`: skip the TSV-M and Iso-CTS identities |
| `--bootstrap` | 2,000 | `depthstats` resamples |
| `--exploratory` | off | `depthstats --op X --exploratory`: one operator, written to `depth_summary_exploratory_X.md`, verdicts suppressed |
| `--alpha` | none | fix α instead of running `alpha` |
| `--batch-size`, `--gpu-cache-gb`, `--merge-cache` | 500, 8, 3 | memory controls |

### Interfaces to keep stable

- **Evaluation cache key:** Stage A keys (`expert`, `zeroshot`, `merge`) are unchanged. Stage B conditions use `kind: "cond"` with a `cond` dict whose `step` is `factorial` (pair, third set), `unit` or `part` (direction, unit or third + kind), `random` (pair, third, seed) or `own` (victim, third set); probes use `kind: "probe"` (model tag, task, depth). The key includes the backbone and asset ids, so the ViT-L/14 run cannot collide with ViT-B/32.
- **Pair id:** `"{op}|{victim}|{context joined by '-'}"`, used in every JSON output and by the bootstrap to keep a pair's conditions together.
- **JSON outputs:** one object per pair, keyed by pair id, with all numbers in normalized pp unless the field name ends in `_raw`.

## Must-hold checks and tests

A failed must-hold check aborts the step. Warnings are logged and reported but do not abort.

### Runtime checks

| Step | Check | Tolerance |
| --- | --- | --- |
| `check` | Experts and θ\_0 match FusionBench test accuracy on all 8 tasks | ±1 pp |
| `check` | Two identical evaluations give identical predictions | exact |
| `check` | Task arithmetic of one task equals its expert; TSV-M of one task reconstructs it; Iso-CTS output spectrum is isotropic | as in the existing package |
| `check` | Calling the blocks one by one from the ℓ = 0 state reproduces the model's hidden states at every depth | relative 1e-5 |
| `select` | No report-half image index is used; the halves are disjoint | exact |
| `select` | Summed harmer full-removal loss is positive (part of eligibility, rule 3b; individual negative losses are logged) | > 0 |
| `factorial` | H\_∅ = B and H\_all = A, per image | exact |
| `factorial` | Shapley efficiency: Σφ\_g = Γ | 1e-9 |
| `random` | Random matrices keep the singular values of D\_g; other tensors keep their norm | relative 1e-5 |
| `patch` | Patching A's own block output reproduces A | exact |
| `crossterm` | Block runner reproduces the merge's hidden states, and I\_{ℓ+1} = G\_ℓ + T\_ℓ + M\_ℓ | relative 1e-4 |
| `depthstats` | `config_sha256` unchanged since `select` | exact |
| `probes` | Expert's final probe within 2 pp of its head accuracy; θ\_0 probe below the expert's | warning only |

Every step runs with TF32 off, deterministic cuDNN and fp32 weights and activations. The cross-term analysis in particular cancels four large terms, and the *Orientation, Not Magnitude* paper showed that bf16 estimates of such terms are mostly rounding noise.

### Tests (CPU, before any GPU run)

`python -m pytest -q tests` runs seven tests in about 5 minutes on CPU, with no downloads.

1. **Operators:** TSV-M and Iso-CTS equal the official code on random tensors, for 1 to 4 tasks.
2. **Decompositions:** neighbour decomposition exactness; with the three thirds as players, Shapley efficiency, κ\_M = r\_M − c\_M = d(EM) + d(ML) + d(EML), and Shapley = single effects for an additive value.
3. **Statistics:** all-positive z with n = 5 gives p = 1/32; a mixed-sign case equals brute-force enumeration; `iut` returns the maximum and propagates missing values; `fixed_sequence` stops at the first failure; Holm on two textbook cases.
4. **Hybrids and random reference:** `hybrid_sel` picks exactly the intended parameters for unit and attention/MLP predicates; `matched_random_like` keeps singular values and norms and is deterministic for a seed.
5. **Probes:** a separable 3-class problem is recovered above 95%.
6. **End-to-end positive control.** A tiny CLIP vision model (6 blocks, 5 tasks, piecewise-constant images; class heads built from each expert's centred features). Two tasks share a planted component only in blocks 2-3 (the middle third). Every step runs for all three operators, then `select --all-ops`, `depthstats` (with the minimum-victim count lowered to 2 for 5 tasks), the exploratory mode and `replicate`. The test asserts that every must-hold check passes; the largest-harm pair has the planted harmers; rerunning `subsets`, `factorial`, `blocks` and `random` adds no evaluations; σ\_M is the largest share for task arithmetic; `depthstats`' Γ and φ recomputed from the per-image cache equal `factorial.json` (two independent code paths); and `depthstats` refuses to run when the config hash changes.
7. **End-to-end negative control.** Near-zero task updates: no candidate is eligible, every victim is listed as ineligible, and `depthstats` reports an infeasible, empty analysis instead of inventing victims.

On the synthetic positive control the pooled σ\_M was 0.81 (95% interval 0.68 to 0.95) and removing the middle third alone recovered 71% of the harm: the pipeline finds a planted mid-depth interaction. The synthetic model is not meant to reproduce the other claims (its planted component is itself a random matrix, so the matched random reference is as harmful as it is).

## Compute budget and run order

The estimate is about **10 GPU-hours** on one RTX 3090, or about **6 hours** with one context per victim and no ViT-L/14. It assumes about 3,000 images/s for ViT-B/32 evaluation from the uint8 cache, about 650 images/s for ViT-L/14, and about 15-20 s per four-task Iso-CTS merge on the GPU. `check` measures the real throughput; rescale the table with it before starting.

| Phase | Work | ViT-B/32 GPU time |
| --- | --- | --- |
| Setup | `prepare-data` (CPU, once, \~14 GB) and `check` | \~0.2 h, plus 20-40 min CPU |
| Stage A | `alpha` for 3 operators | \~0.7 h |
| Stage A | `subsets` (task arithmetic \~0.4 h, TSV-M \~0.6 h, Iso-CTS \~1.2 h) | \~2.2 h |
| Stage A | `analyze`, `select`, `report` (mostly merges for the 48 pairs) | \~1 h |
| Stage B | Per pair: B1 32 evals, B2 40, B3 9, B4 6 (per victim), probes \~1-2 min, patch and cross-term \~1-2 min, so about 5-6 min; 48 pairs | \~4.5 h |
| Stage B | `weightspace`, `depthstats` | CPU, \~20 min |
| Replication | ViT-L/14 task arithmetic: `check`, `alpha` (5 contexts, 2,000 images), candidate re-derivation, B1 and B3 on 6 pairs; \~11 GB of checkpoints | \~1.3 h |
| **Total** |  | **\~10 h** |

**Reduced configuration** (`--k-per-victim 1`, no replication): Stage B falls to \~2.3 h and the total to \~6 h. With k = 1 the pooled test still has up to 8 victims, but each victim's estimate rests on one context.

### Run order on one GPU

```bash
OUT=/srv/exp8/results; DATA=/srv/exp8/data
C="--out $OUT --data-dir $DATA --device cuda"
python -m exp8.run prepare-data $C
python -m exp8.run check $C
for OP in ta iso_cts tsvm; do
  for S in alpha subsets analyze select report factorial blocks random own probes patch crossterm weightspace; do
    python -m exp8.run $S --op $OP $C
  done
done
python -m exp8.run select --all-ops $C          # pooled selection file and the feasibility check
python -m exp8.run depthstats $C                # once, after all operators
python -m exp8.run replicate --op ta --backbone ViT-L-14 --tau-device cpu $C
```

`scripts/run_exp8.sh` runs the same sequence and **pauses after task arithmetic**. At the pause, read only the must-hold results and the number of eligible victims in `ta/selection.json`. If task arithmetic has fewer than 5 eligible victims, stop and report: that is a futility stop and does not bias anything. Do not run `depthstats` until all three operators are finished. If you want to look at task arithmetic alone, `depthstats --op ta --exploratory` writes a separate file labelled exploratory, which does not replace the confirmatory analysis.

Every step is resumable from the evaluation cache. After an interruption, rerun the same command.

## Outputs and what each outcome means

### Files

| File | Content |
| --- | --- |
| `$OUT/depth_summary.md`, `.json` | **Read first.** The five confirmatory claims with n, statistic, exact p, fixed-sequence verdict and interval; σ per third (pooled and per operator); the sensitivity analyses; the exploratory figures' data |
| `$OUT/who_summary.md`, `.json` | W1 and W2 with Holm; interaction share; cross-operator agreement; breadth |
| `$OUT/selection_all.json` | Pooled pairs, victims, feasibility, config hash |
| `$OUT/<op>/selection.json` | Per-operator pairs, harmers, Γ\_sel with intervals, ineligible victims |
| `$OUT/<op>/factorial.json`, `blocks.json`, `random.json`, `own.json`, `probes.json`, `patch.json`, `crossterm.json`, `weightspace.json` | One object per pair |
| `$OUT/<op>/evals/` | Per-image correctness for every condition (keep it; the bootstrap and any later reanalysis need it) |
| `$OUT/vitl14/` | The replication, same layout, plus `replicate_summary.md`/`.json` (σ per third with intervals, next to the ViT-B/32 task-arithmetic values) |
| `$OUT/depth_summary_exploratory_<op>.md` | Only if you ran the exploratory single-operator analysis; never a confirmatory result |

### What to send back

1. Both summaries, `selection_all.json`, every per-operator JSON (not `evals/`), the ViT-L/14 folder and the full log with timings.
2. A short note covering:
   - whether every must-hold check passed, and the asset accuracies against FusionBench;
   - how many victims were eligible under each operator, and which were not (with their best Γ\_sel);
   - for each claim: the estimate, interval, p-value and verdict, and where the fixed sequence stopped;
   - σ\_E, σ\_M and σ\_Lt per operator, and the ViT-L/14 values;
   - ψ and the probe-gap curve for the three largest-harm pairs;
   - whether the Experiment 2 pattern (GTSRB + SVHN harming EuroSAT) reappears under any operator.

### Outcomes and claims for the paper

| Outcome | Claim we can make |
| --- | --- |
| Claims 1-5 confirmed | Merging harm is created mid-depth, specifically by interference, by destroying decodable task information, and is amplified downstream; and depth separates victim harm from harmer skill, which is why per-depth coefficients work. All of this holds beyond DC-Merge |
| 1-2 confirmed, 3 fails, ψ near 0 | Harm arises mid-depth but the information survives; the victim is misread at the output. The repair should target late readout alignment, not the middle blocks |
| 1 confirmed, 2 fails | The middle blocks are where the network is most sensitive to any change of this size; interference lands there but is not special to it. A weaker but honest claim |
| 1 fails | The mid-depth finding was specific to DC-Merge with LoRA on attention. The paper leads with "who is responsible", and depth becomes a DC-Merge result |
| 1-4 confirmed, 5 fails | The mechanism generalizes, but depth does not separate victim and harmer; a per-depth method would need another lever |
| σ\_M largest per operator, but heterogeneous | Report the operator differences; isotropic merging may move or soften the depth profile |
| ViT-L/14 agrees | The middle third keeps the largest share at a different depth, so the effect is relative depth, not a fixed block index |

## Limitations and threats to validity

- **Few independent units.** About 8 victims carry the inference, and they are not fully independent: tasks recur as victims and harmers, and the three operators share the same checkpoints. The exact test is valid under the stated symmetry assumption, but power is limited and the bootstrap intervals are approximate.
- **Coarse depth.** Thirds can separate middle from ends but cannot pinpoint a block. The block-level scan is descriptive only, by design, to keep the confirmatory family small.
- **Hybrids are off-manifold.** Mixing blocks from two merged models creates models that neither operator would produce. Measuring both removal and creation, and checking the random-perturbation reference, limits but does not remove this concern.
- **Re-merge semantics.** For TSV-M and Iso-CTS, removing harmers re-runs the operator, which also changes the shared subspace seen by the victim. That is the operator-agnostic counterfactual a practitioner would use, but it is not the clean task-vector removal of task arithmetic. Task arithmetic is the reference case.
- **One checkpoint set per backbone,** full fine-tuning only. LoRA is covered by Experiment 6 only through DC-Merge.
- **The probes measure linear decodability** on at most 5,000 training images. SUN397 (397 classes) has about 12 training images per class, so its probe accuracies are low; this affects all models equally, but makes its gaps noisier.
- **The random reference** matches size and spectrum, not alignment with the pretrained weights. A perturbation in the pretrained weights' own subspace could hurt more than a random one. This is stated, not controlled.
- **The q, k, v split** for the SVD operators (Hugging Face stores them separately, unlike the papers' open\_clip code), so absolute accuracies differ slightly from the papers.
- **The fixed coefficient** α per operator is tuned on 10 random contexts; harm is measured at the deployed coefficient, not at each context's best one.
- **The ViT-L/14 replication** is small (up to 6 pairs, one operator) and descriptive.
