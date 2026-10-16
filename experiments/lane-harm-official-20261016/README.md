# Lane-harm experiment: official DC-Merge and rank variant

This directory preserves the source snapshot and results from the lane-harm run completed on 2026-10-16. The run used the official DC-Merge code at [`089af77a378562272461e0f041217ec440a4a230`](https://github.com/Tobeginwith/DC-Merge/commit/089af77a378562272461e0f041217ec440a4a230). The local official source tree was not a Git checkout, so it was compared file by file with that commit: all 207 upstream tracked files matched byte for byte. The run-specific additions are in `source/official-dc-merge/lane_harm/` and the two guides in `source/official-dc-merge/docs/`.

The source used for the run was `/workspace/ducanh/DC-Merge`. `/workspace/DC-Merge-Repro` was a separate checkout and did not produce these results. The complete captured source snapshot is included here, with exclusions documented in [EXCLUDED_FILES.md](EXCLUDED_FILES.md) and SHA-256 checksums in [SOURCE_SHA256SUMS](source/official-dc-merge/SOURCE_SHA256SUMS).

## Run configuration

- Backbone: CLIP ViT-B/32; official eight-task LoRA checkpoints, rank 16 and LoRA alpha 16.
- Runtime: Python 3.9.23 in the `dcmerge_vision` environment; CUDA device; config `vitB32_r16_8task`; linear smoothing with `rho=5.0`.
- Alpha selection: the official operator swept alpha 0.1 through 3.0 in 0.1 increments on the all8 validation context. It selected **alpha=0.4** (mean normalized validation accuracy 73.627).
- Seeds: the run metadata records CLI seed 0 (the default, used for intervention randomization). The runner also calls the official `set_seed(400)` for model/data evaluation.
- Official top-k: the published rule keeps `max(1, int(0.001 * (16N)^2))` coefficients per task's cover-space matrix: 4 per task for four-task contexts and 16 for all8. The baseline is checked against official `dc_merge` on every official run row.
- Rank variant top-k: the experiment modification keeps 16 coefficients per task, regardless of context size. Its harm reference is the solo merge; the official operator uses standalone accuracy.

The official run covered the seven EuroSAT contexts below plus all8. The rank variant covered only the seven EuroSAT contexts; **it was not run for all8**. The alpha-selection test result (mean accuracy 64.13%, mean normalized accuracy 73.86%) is an **official-baseline reproduction**, not a rank-variant test result. Lane-harm intervention evaluations in both summaries use validation data.

| Context group | Contexts |
| --- | --- |
| EuroSAT + GTSRB + SVHN | `dtd-eurosat-gtsrb-svhn`, `eurosat-gtsrb-mnist-svhn`, `eurosat-gtsrb-sun397-svhn` |
| EuroSAT + one of GTSRB/SVHN | `stanford_cars-eurosat-gtsrb-mnist`, `stanford_cars-eurosat-gtsrb-resisc45` |
| EuroSAT + neither | `stanford_cars-eurosat-mnist-resisc45`, `stanford_cars-eurosat-mnist-sun397` |
| Official only | `all8` |

The all8 task order is Stanford Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, and SVHN.

## Reproduction commands

The original shell command lines were not captured in the logs. These commands give the exact recorded options, paths, and configuration, using the run's checked-in source snapshot in place of the original source path. The dataset, CLIP model, heads, and LoRA checkpoints remain external assets and must be supplied at the recorded paths or changed to their new locations.

~~~bash
cd /path/to/DC-Merge-Repro/experiments/lane-harm-official-20261016
REPO="$PWD/source/official-dc-merge"
PYTHON=/venv/dcmerge_vision/bin/python
MODEL=/workspace/models/clip-vit-base-patch32
FT_DIR=/workspace/models/DC-Merge-Vision-B32-r16-8task/lora_checkpoints
HEADS=/workspace/models/DC-Merge-Vision-B32-r16-8task/lora_heads
DATA=/workspace/datasets8
OUT=/workspace/results/lane-harm-official
COMMON=(--repo "$REPO" --model "$MODEL" --ft-dir "$FT_DIR" --heads "$HEADS" --data "$DATA" --device cuda)

cd "$REPO"
"$PYTHON" lane_harm/run_official.py select-alpha "${COMMON[@]}" \
  --out "$OUT" --with-test 2>&1 | tee "$OUT-alpha.log"

"$PYTHON" lane_harm/run_official.py run "${COMMON[@]}" \
  --out "$OUT/official" --topk official --alpha 0.4 \
  --preset eurosat --preset all8 2>&1 | tee "$OUT-official.log"

"$PYTHON" lane_harm/run_official.py run "${COMMON[@]}" \
  --out "$OUT/rank" --topk rank --alpha 0.4 \
  --preset eurosat 2>&1 | tee "$OUT-rank.log"
~~~

`--seed` is omitted, so the runner's CLI/intervention seed is 0. The alpha-selection command defaults to context `all8` and top-k `official`.

## Results

- [Alpha selection JSON](results/alpha_selection_all8_official.json)
- [Official operator summary](results/official/results_summary.md)
- [Official operator CSV](results/official/interventions.csv)
- [Rank variant summary](results/rank/results_summary.md)
- [Rank variant CSV](results/rank/interventions.csv)
- [Alpha selection log](results/logs/lane-harm-official-alpha.log)
- [Official operator log](results/logs/lane-harm-official-official.log)
- [Rank variant log](results/logs/lane-harm-official-rank.log)
- [Complete result archive](results/lane-harm-results-20261016.tar.gz)

The archive contains 845 files (all per-context JSON/JSONL outputs, both `interventions.csv` files, both `results_summary.md` files, alpha selection JSON, and all three logs). Its SHA-256 is:

~~~text
ad630671fdb22ac9fd82ebef33ea2dbca9891175072346843bee0cd1fc036030
~~~

## Experiment 2 source (added after the Experiment 1 run)

The next experiment's code is now in the **same source snapshot** at
`source/official-dc-merge/lane_harm/{ct_merging.py,lanes.py,run_week1.py}`.
Its instructions are in `source/official-dc-merge/docs/IMPLEMENTATION_GUIDE.md`.
The supplied `DC-Merge-official-experiment2.zip` has SHA-256
`42ecfcf62112dbe76c33b51167e30bfe8b81df662a3606a2d66e8b49c88652cc`;
all common files were byte-identical to the published Experiment 1 snapshot.
Only these three new modules, the guide, the sealed Experiment 1 reference JSON,
and small CPU-independent checks were added. The official merging code was not
modified. The reference JSON was derived from the Experiment 1 result archive
above, and carries that archive's SHA-256.

Experiment 2a compares official DC-Merge, the rank top-k variant, and a local
implementation of CT-Merging on validation-selected coefficients, evaluating
each chosen merge once on test. Experiment 2b/2c measures all eight subsets of
three companion lanes in the fixed full-context frame and then tests partial
lane scales. The runner checks Experiment 1's validation baseline at alpha 0.4
and the all-eight official selected coefficient/test score, allowing 0.5
percentage points for evaluation nondeterminism. It checks existing result
files before resuming. See the guide for the complete commands and caveats.

No Experiment 2 GPU run or result is included in this commit. Running it
requires the external model, checkpoints, heads, datasets, and Python/CUDA
environment described above; this publication contains code and a run plan.
The Git snapshot also omits the five upstream dataset split-index `.pt` files
listed in `EXCLUDED_FILES.md`; restore those byte-identical files from the
official DC-Merge commit before using this source snapshot as `--repo`.
The supplied Experiment 2 ZIP contains them, so its independent unpacked
source can be used directly with the external assets.
