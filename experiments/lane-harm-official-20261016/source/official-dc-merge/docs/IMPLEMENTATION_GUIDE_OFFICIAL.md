# Lane-harm experiment on the official DC-Merge: implementation guide

Read [`MENTEE_METHOD_GUIDE.md`](MENTEE_METHOD_GUIDE.md) first for what the experiment tests and why.

This package runs the experiment on the **official** DC-Merge code ([github.com/Tobeginwith/DC-Merge](https://github.com/Tobeginwith/DC-Merge), commit `089af77`, 29 Apr 2026), with the **official checkpoints**. No official file is modified.

## 1. Official DC-Merge vs the PDRA repository

The official repository was compared file by file with `vision_lora_merge/` in DC-Merge-PDRA:

| | Official | DC-Merge-PDRA |
|---|---|---|
| `dc_merge` | Rank fixed at **16 per task**; top-k = `max(1, int(0.001 × (16N)²))`; returns merged matrices only | Rewritten: per-task ranks, `topk_policy`, task scales/densities, intermediates |
| Kept coefficients per task per module | **4 tasks: 4**; 8 tasks: 16 | 4 tasks at rank 8 (legacy): **1**; with `topk_policy="rank"`: 8 |
| Other merging functions (TIES, TSV-M, Iso-CTS, WUDI, `keep_topk_percent`, `ties_small`, …) | — | Identical |
| `ft_handlers.py`, `utils.py`, configs | Paths are empty strings to fill in | Same code, plus environment-variable paths, a dict-input loader and a self-trained config; Windows line endings |

**What this means.** Results from the PDRA repository were measured on a *modified* operator (rank 8, 1 kept coefficient) and on *self-trained* adapters. Here we measure the operator exactly as published, on the authors' checkpoints. EuroSAT may or may not be a victim here, and either answer is a result.

## 2. Package contents

```
DC-Merge/                         official repository at commit 089af77 (unchanged)
├── vision_lora_merge/ ...        official LoRA merging code
├── lane_harm/
│   ├── lane_harm.py              the method: attribution + six interventions
│   ├── dc_trace.py               official dc_merge, statement by statement, also returning intermediates
│   └── run_official.py           runner: select-alpha / run / summarize
└── docs/
    ├── MENTEE_METHOD_GUIDE.md
    └── IMPLEMENTATION_GUIDE_OFFICIAL.md   (this file)
```

**Why `dc_trace.py` exists.** The official `dc_merge` returns only the merged matrices, but the interventions need the internal quantities. `dc_trace.py` repeats the official computation line for line, using the official helper functions. On **every** baseline merge the runner checks that its output is **bit-for-bit identical** to the official `dc_merge`, and stops if not.

To rebuild the package yourself instead of using the zip:

```bash
git clone https://github.com/Tobeginwith/DC-Merge.git && cd DC-Merge && git checkout 089af77
# then copy lane_harm/ and docs/ from the zip into this folder
```

## 3. Setup

### 3.1 Environment

```bash
cd DC-Merge
conda env create -f environment_vision.yml      # env "dcmerge_vision"
conda activate dcmerge_vision
PYTHON=$(which python)
"$PYTHON" -c 'import importlib.util; names=("torch","numpy","torchvision","transformers","peft","PIL","tqdm","sklearn","scipy"); m=[n for n in names if importlib.util.find_spec(n) is None]; print("missing modules:", m); raise SystemExit(bool(m))'
```

**Expected:** `missing modules: []`.

### 3.2 Assets (all official, from the README)

| Asset | Source | Variable |
|---|---|---|
| CLIP ViT-B/32 | `huggingface-cli download openai/clip-vit-base-patch32 --local-dir $ASSETS/clip-vit-base-patch32` | `MODEL` |
| LoRA checkpoints (Table 1) | Google Drive link in the official README, section "Checkpoints" → `lora_checkpoints/<task>/` | `FT_DIR` |
| Classification heads (Table 1) | Google Drive link in the official README → `lora_heads/ViT-B-32/<task>_head.pt` | `HEADS` |
| 8 datasets | Official README, section "Datasets" (ModelScope mirror available), layout as shown there | `DATA` |

```bash
REPO=$PWD
ASSETS=/srv/dcmerge/assets
MODEL=$ASSETS/clip-vit-base-patch32
FT_DIR=$ASSETS/lora_checkpoints
HEADS=$ASSETS/lora_heads
DATA=$ASSETS/datasets
OUT=/srv/dcmerge/results/lane-harm-official      # new folder, outside the repo
COMMON="--repo $REPO --model $MODEL --ft-dir $FT_DIR --heads $HEADS --data $DATA --device cuda"
```

You don't need to edit the official config files: the runner sets these paths in memory. If the 8-task datasets use a different file layout, follow the official README layout.

## 4. Run: three steps

### Step A: reproduce the official result and pick α (≈ 1–2 GPU-hours)

```bash
"$PYTHON" lane_harm/run_official.py select-alpha $COMMON --out "$OUT" --with-test 2>&1 | tee "$OUT-alpha.log"
```

This runs the official protocol on the full 8-task merge: α from 0.1 to 3.0 on validation, the best mean normalized accuracy chosen, then the test split at that α. Output: `$OUT/alpha_selection_all8_official.json`.

### Step B: lane-harm on the official operator (main result)

```bash
ALPHA=<best_alpha from step A>
"$PYTHON" lane_harm/run_official.py run $COMMON --out "$OUT/official" --topk official \
    --alpha $ALPHA --preset eurosat --preset all8 2>&1 | tee "$OUT-official.log"
```

### Step C: the rank-aware variant, for comparison with the PDRA result

```bash
"$PYTHON" lane_harm/run_official.py run $COMMON --out "$OUT/rank" --topk rank \
    --alpha $ALPHA --preset eurosat 2>&1 | tee "$OUT-rank.log"
```

`--topk rank` keeps r = 16 coefficients per task instead of 0.1%. **This is our modification, not the official operator,** and its summary says so. On the 8-task merge both rules keep 16, so `all8` is not repeated.

**Contexts (`--preset eurosat`):**

| Group | Contexts |
|---|---|
| EuroSAT + GTSRB + SVHN | `dtd-eurosat-gtsrb-svhn`, `eurosat-gtsrb-mnist-svhn`, `eurosat-gtsrb-sun397-svhn` |
| EuroSAT + one of GTSRB/SVHN | `stanford_cars-eurosat-gtsrb-mnist`, `stanford_cars-eurosat-gtsrb-resisc45` |
| EuroSAT + neither | `stanford_cars-eurosat-mnist-resisc45`, `stanford_cars-eurosat-mnist-sun397` |

`--preset all8` adds the published 8-task merge.

**Cost:**
- Each 4-task context: 1 baseline + 4 targets × 6 interventions = 25 merges, each evaluated on 4 validation sets.
- `all8`: 1 + 8 × 6 = 49 merges on 8 sets.
- Solo merges: about 8 in total, cached.

Each merge also takes about 1–2 CPU-minutes for the attribution. **Estimate: 7–10 hours for step B, 6–8 hours for step C.**

If interrupted, rerun the same command with `--resume`. The runner refuses to mix a different `--topk` or `--alpha` into an existing `--out`. To rebuild only the summary: `"$PYTHON" lane_harm/run_official.py summarize --repo $REPO --out "$OUT/official"`.

## 5. Output

| File | Content |
|---|---|
| `$OUT/alpha_selection_all8_official.json` | Alpha sweep, best α, test accuracies at best α |
| `$OUT/<official\|rank>/results_summary.md` | **Start here:** per target, harm and benefit / cost per intervention, plus a verdict |
| `$OUT/<official\|rank>/interventions.csv` | All numbers, one row per (context, target, intervention) |
| `$OUT/<official\|rank>/<context>/context_report.json` | Accuracies, accounting, baseline verification |
| `$OUT/<official\|rank>/run_meta.json` | Operator, α, harm reference, official commit |

**Harm reference** (used for `share_removed` and the verdict):
- `--topk official` uses **standalone** by default: the authors' single-task validation accuracy, from `single_task/our_finetuned/ViT-B-32/val_acc.json`. A "solo merge" is not a fair reference here: the official rule with one task keeps only `max(1, int(0.001·16²)) = 1` coefficient per module, so the solo merge is itself crippled.
- `--topk rank` uses **solo** by default, as in the PDRA experiment.

Both harms are always reported, in the columns `harm_vs_standalone_pp` and `harm_vs_solo_pp`.

## 6. Expected outcomes

### 6.1 Must hold (otherwise stop and report)

| Check | Expected |
|---|---|
| Module check | `missing modules: []` |
| **Step A: reproduce Table 1** | Test mean normalized accuracy ≈ **73.9** and mean accuracy ≈ **64.2** (paper: 73.90 / 64.17). Our earlier reproduction with the official checkpoints gave **73.88 at α = 0.4**. |
| Step B log, each context | `baseline vs official dc_merge: identical` |
| Step B summary header | `All consistency checks passed: True` and `Baseline merge identical to official dc_merge in every row: True` |
| Step C log | `baseline vs official dc_merge: not compared (rank variant)` — expected, since it's a different operator |
| `all8` baseline accuracies | Equal to the step A validation accuracies at the same α |

If step A doesn't reproduce about 73.9, the assets or environment differ from the paper's. Fix that before interpreting step B.

### 6.2 Predicted mechanism

1. **4-task contexts on the official operator keep 4 coefficients per task per module** (`kept_per_task_total` = 4 × number of modules). TIES-lane effects should be small (`leakage_ratio` ≈ 0), because the few kept entries fall on each task's own diagonal.
2. **`restore_own_update` should give the largest benefit** wherever there is context harm: spill/rotation is the remaining context-dependent loss.
3. **Official vs rank variant.** Keeping 16 instead of 4 coefficients should increase every task's accuracy in the 4-task contexts. In our PDRA data, going from 1 to 8 coefficients raised mean retention by about 15 pp. **Expect a smaller gain here,** because 4 is already more than 1.
4. **EuroSAT.** With official checkpoints and rank 16, EuroSAT need not be the victim. Read the tables for **all** targets: the most harmed task in each context is the one to analyse.

### 6.3 Decision rule (per context and target, applied automatically)

| Observation | Verdict |
|---|---|
| Best intervention removes ≥ 50% of harm, control < 10% | **Cause identified** |
| The control removes ≥ half as much as the best intervention | Not specific (an energy effect) |
| Mean cost to the others ≥ 80% of the benefit | Real cause, but fixing it moves the harm |
| No intervention removes ≥ 20% | No isolated mechanism |
| Otherwise | Partial |

## 7. What to send back

1. `alpha_selection_all8_official.json` and `$OUT-alpha.log`.
2. Both `results_summary.md` files, both `interventions.csv` files, and the logs.
3. A note:
   - Did every row of 6.1 hold?
   - Which task is most harmed in each context?
   - Which verdict does it get under the official operator and under the rank variant?

## 8. Common problems

| Symptom | Fix |
|---|---|
| Step A far from 73.9 | Check the checkpoints (Table 1 LoRA set, not KnOTS), the heads, the CLIP snapshot, and the dataset layout. |
| `ModuleNotFoundError: utils` / `merging_functions` | `--repo` must be the repository root, the folder containing `vision_lora_merge/`. |
| `consistency check failed` or a baseline not identical to official | Don't work around it; send the log and the failing `.summary.json`. |
| `... was started with {...}; use a new --out` | You changed `--topk` or `--alpha`; use a fresh output folder. |
| `pdb` prompt during loading | The official loader stops in `pdb` if a `.pt` checkpoint doesn't match. Use the official `lora_checkpoints` folders (PEFT format). |
