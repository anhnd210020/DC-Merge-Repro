# PDRA local freeze and GPU runbook

The PDRA runner wraps `task_demand_prediction.runtime.RealEvaluator`, which calls the repository's existing `dc_merge` with the same `smoothing_strategy='linear'`, `rho=5.0`, task scale `1.0`, task density `1.0`, and `task_ranks` intervention. It keeps the existing top-k and TIES behavior and applies the existing `alpha=0.8` merge. The laptop workflow freezes and checks files only; it does not load a model, evaluate a dataset, or require CUDA.

## Frozen protocol

- The task pool and order are the eight tasks in `task_demand_prediction.design.TASKS`.
- The primary manifest contains all 45 four-task combinations and all 19 six-task combinations that are absent from the prior experiment's 25 unique four-task and 9 unique six-task memberships. Memberships are enumerated in registered task order. Every task appears in both fresh context sizes. No reused memberships are included in the primary evaluation.
- Uniform ranks are `[8, 8, 8, 8]` and `[8, 8, 8, 8, 8, 8]`. PDRA uses `[12, 8, 8, 4]` and `[12, 12, 8, 8, 4, 4]`, assigned in demand order. Equal demands are ordered by the registered task order. Reverse-PDRA orders by increasing predicted demand and uses the same tie rule.
- The fitted predictor is one standardized `Ridge(alpha=1.0)` using `energy_rank_90`. It was fit once on the 118 previous-experiment validation context/task rank-demand rows. Median imputation and standardization are frozen in the artifact. The feature values are constant per task and bound to the adapter weight hashes in the artifact.
- Oracle-Ranked uses a same-split rank-response sweep only for that diagnostic method. Oracle demand does not enter the predictor, PDRA, Reverse-PDRA, feature selection, or tuning. The final-test oracle sweep uses final-test labels to set only the Oracle-Ranked diagnostic order and compute oracle headroom; it must not be used to choose or tune a method. Oracle-Ranked is not guaranteed to be a mathematical upper bound.
- Rank demand reuses the rank curve `{2, 4, 8, 12, 16}` and its normalized axis `(rank - 2) / (16 - 2)`. The code obtains the rank response points by calling the current `RealEvaluator`, whose `dc_merge` invocation passes `task_ranks` through the established rank-reduction path.

## Work and output counts

There are 64 contexts. The allocation stage has 4 methods per context, or 256 allocation evaluation units per split. Oracle demand needs one rank-16 baseline plus four non-baseline rank evaluations per task: 17 extra units per four-task context and 25 per six-task context, or 1,240 oracle sweep units per split. The frozen plan therefore runs 1,496 evaluation units per split and 2,992 across validation and final test. The runner retains a separate sealed result for each method/context and does not deduplicate method-labeled units with coincident rank maps.

## Local commands

Rebuild either artifact from the verified prior audit archive if needed. Both commands verify its full SHA-256 and refuse to replace a different frozen artifact:

```powershell
& 'D:\envs\dev\Scripts\python.exe' -m task_demand_prediction pdra-freeze-contexts --audit-archive 'C:\Users\Admin\Downloads\DC-Merge-Repro-final-audit-20261006.tar.gz'
& 'D:\envs\dev\Scripts\python.exe' -m task_demand_prediction pdra-fit-predictor --audit-archive 'C:\Users\Admin\Downloads\DC-Merge-Repro-final-audit-20261006.tar.gz'
```

The local static preflight checks the frozen context and predictor artifacts and a proposed external output path without loading the evaluator:

```powershell
$Out = Join-Path $env:TEMP 'dcmerge-pdra-run'
& 'D:\envs\dev\Scripts\python.exe' -m task_demand_prediction pdra-preflight --static-only --output-dir $Out
& 'D:\envs\dev\Scripts\python.exe' -m unittest discover -s tests -v
```

## Verified local asset inventory (2026-10-07)

These paths were checked on the preparation machine. The datasets, adapters, heads, references, and exact CLIP base-model snapshot are available. The `DCMERGE_*` environment variables are unset, so use explicit paths after transfer.

| Asset | Local path and result |
|---|---|
| CLIP ViT-B/32 base model | **Available** at `D:\pdra-assets\clip-vit-base-patch32-d2bce4b`. Restored from `openai/clip-vit-base-patch32` at revision `d2bce4bc6684fb94eb627c995ba3164d618ef0f7`; all six files match the frozen sizes and SHA-256 values below. |
| Eight datasets | **Available** at `D:\Research\DC_Merge\datasets8`. All preflight layout checks passed for Cars (`cars`), DTD (`dtd`), EuroSAT (`eurosat`), GTSRB (`gtsrb`), MNIST (`MNIST\raw`), RESISC45 (`resisc45`, 45 class directories), SUN397 (`sun397`), and SVHN (`svhn`). The dataset metadata identity matches the frozen predictor. |
| Eight adapters and eight heads | **Available** under `C:\Users\Admin\Downloads\DC-Merge-Repro-model-weights-20261007`. The extracted tree has `artifacts\selftrained_b32_r16_8task\checkpoints\<task>\{adapter_config.json,adapter_model.bin}` and `assets\heads\ViT-B-32\<task>_head.pt`. All 24 files match the per-file hashes in `pdra_predictor.json`. |
| Validation/test references | **Available** at `D:\Research\DC-Merge-PDRA\reproduction\selftrained_b32_r16_8task\results\selftrained_{val,test}_acc.json` (also present identically in the sibling reproduction repo). Validation SHA-256 `ea41870ab35b9aec08e9a35af68adc124e7be22a3419c203f90697d318750ff4` matches the predictor. Test SHA-256 is `b597460ce56127fda5cb3edc0abe1b8463c68e8cb4c1555f29808ae32bf7be96`. |
| Prior audit archive | **Available** at `C:\Users\Admin\Downloads\DC-Merge-Repro-final-audit-20261006.tar.gz`. SHA-256 `7b01616a8e75774802ce8b95fd03e900c494003a7d0c16fb228153879e3f04ad` matches the pinned `AUDIT_ARCHIVE_SHA256`. It contains the predictor source members but no model, dataset, adapter, head, or reference files. |
| Adapter/head archive | **Available** at `C:\Users\Admin\Downloads\DC-Merge-Repro-model-weights-20261007.tar.gz`. Measured SHA-256 `ba5ccb94b489aa863295a324d22fbb70d05af8cb5d703f2df5d6ffa3510a0a4e`; it contains the 8 configs, 8 adapter weights, and 8 heads, and no base model, datasets, or references. The archive itself has no pinned checksum sidecar; the 24 member hashes do match the frozen predictor. |

The exact snapshot recorded in `task_demand_prediction/pdra_predictor.json` under `payload.source.source_model_files` was recovered from the official Hugging Face repository `openai/clip-vit-base-patch32` at immutable revision `d2bce4bc6684fb94eb627c995ba3164d618ef0f7`. The local copy is at `D:\pdra-assets\clip-vit-base-patch32-d2bce4b`. Each downloaded file was checked against both the frozen size and SHA-256 below:

| File | Size | SHA-256 |
|---|---:|---|
| `config.json` | 4,186 | `b575ef3c36f2a057fa19e221650105052d61cc9c1a972ec15019c6261ec98770` |
| `merges.txt` | 524,657 | `f526393189112391ce6f9795d4695f704121ce452c3aad1f5335cc41337eba85` |
| `preprocessor_config.json` | 316 | `910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd` |
| `pytorch_model.bin` | 605,247,071 | `a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f` |
| `tokenizer.json` | 2,224,041 | `b556ac8c99757ffb677208af34bc8c6721572114111a6e0aaf5fa69ff0b8d842` |
| `vocab.json` | 862,328 | `5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363` |

Do not substitute a similarly named CLIP checkpoint. The server preflight compares the model file identities to the predictor provenance. The preparation environment also lacks `transformers`, `peft`, and `clip`; no packages were installed. The server Python environment must already include all modules in `pairwise_rank_density.preflight.REQUIRED_MODULES` before full preflight.

## Windows-to-Linux transfer

The commands below transfer available experiment inputs. They do not transfer generated results or the prior audit archive, which is provenance-only and not needed by `pdra-preflight`. Set `$Server` to the SSH destination.

```powershell
$Server = 'user@gpu-host'
$Remote = '/srv/pdra'
$Weights = 'C:\Users\Admin\Downloads\DC-Merge-Repro-model-weights-20261007.tar.gz'
$Data = 'D:\Research\DC_Merge\datasets8'
$References = 'D:\Research\DC-Merge-PDRA\reproduction\selftrained_b32_r16_8task\results'
$Model = 'D:\pdra-assets\clip-vit-base-patch32-d2bce4b'

ssh $Server "mkdir -p $Remote/transfer $Remote/assets/references $Remote/assets/clip-vit-base-patch32 $Remote/results"
scp -p $Weights "${Server}:$Remote/transfer/"
scp -r $Data "${Server}:$Remote/assets/"
scp -p "$References\selftrained_val_acc.json" "$References\selftrained_test_acc.json" "${Server}:$Remote/assets/references/"
scp -p "$Model\config.json" "$Model\merges.txt" "$Model\preprocessor_config.json" "$Model\pytorch_model.bin" "$Model\tokenizer.json" "$Model\vocab.json" "${Server}:$Remote/assets/clip-vit-base-patch32/"
```

On the Linux host, clone the pushed experiment branch, verify and unpack the adapter/head archive, and confirm the references:

```bash
set -euo pipefail
REPO=/srv/pdra/DC-Merge-PDRA
REMOTE=/srv/pdra
ASSETS=$REMOTE/assets
WEIGHTS=$REMOTE/transfer/DC-Merge-Repro-model-weights-20261007.tar.gz

git clone --branch codex/pdra-experiment https://github.com/anhnd210020/DC-Merge-Repro.git "$REPO"
printf '%s  %s\n' 'ba5ccb94b489aa863295a324d22fbb70d05af8cb5d703f2df5d6ffa3510a0a4e' "$WEIGHTS" | sha256sum -c -
tar -xzf "$WEIGHTS" -C "$ASSETS"
printf '%s  %s\n' 'ea41870ab35b9aec08e9a35af68adc124e7be22a3419c203f90697d318750ff4' "$ASSETS/references/selftrained_val_acc.json" | sha256sum -c -
printf '%s  %s\n' 'b597460ce56127fda5cb3edc0abe1b8463c68e8cb4c1555f29808ae32bf7be96' "$ASSETS/references/selftrained_test_acc.json" | sha256sum -c -
(cd "$ASSETS/clip-vit-base-patch32" && sha256sum -c <<'SHA256'
b575ef3c36f2a057fa19e221650105052d61cc9c1a972ec15019c6261ec98770  config.json
f526393189112391ce6f9795d4695f704121ce452c3aad1f5335cc41337eba85  merges.txt
910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd  preprocessor_config.json
a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f  pytorch_model.bin
b556ac8c99757ffb677208af34bc8c6721572114111a6e0aaf5fa69ff0b8d842  tokenizer.json
5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363  vocab.json
SHA256
)
```

Before full preflight, confirm the data copy is at `$ASSETS/datasets8` and each reference SHA-256 matches this inventory. The model transfer above targets `$ASSETS/clip-vit-base-patch32` and verifies all six frozen file hashes on the Linux host.

## Linux server preflight and run

Use an already provisioned CUDA Python environment; do not proceed if a required module or asset is missing. Full `pdra-preflight` checks dependencies, paths, hashes, disk space, and CUDA availability but does not evaluate examples. `$OUT` must be outside the repository and its parent must already exist.

```bash
PYTHON=/opt/pdra/env/bin/python
REPO=/srv/pdra/DC-Merge-PDRA
REMOTE=/srv/pdra
ASSETS=$REMOTE/assets
MODEL=$ASSETS/clip-vit-base-patch32
DATA=$ASSETS/datasets8
ADAPTERS=$ASSETS/artifacts/selftrained_b32_r16_8task/checkpoints
HEADS=$ASSETS/assets/heads
REFERENCES=$ASSETS/references
OUT=/srv/pdra/results/dcmerge-pdra-20261007

"$PYTHON" -c 'import importlib.util; names=("torch","numpy","torchvision","transformers","peft","PIL","tqdm","sklearn","scipy","clip"); missing=[n for n in names if importlib.util.find_spec(n) is None]; print("missing modules:", missing); raise SystemExit(bool(missing))'
"$PYTHON" -m task_demand_prediction pdra-preflight --repo-root "$REPO" --output-dir "$OUT" --model-dir "$MODEL" --data-dir "$DATA" --adapter-dir "$ADAPTERS" --head-dir "$HEADS" --reference-dir "$REFERENCES" --device cuda --confirm-oracle-policy same-split-posthoc
```

Do not start evaluation until the full server preflight passes and the experiment owner authorizes the GPU run. When authorized, run validation first; final test is gated on its sealed completion:

```bash
"$PYTHON" -m task_demand_prediction pdra-run --repo-root "$REPO" --output-dir "$OUT" --model-dir "$MODEL" --data-dir "$DATA" --adapter-dir "$ADAPTERS" --head-dir "$HEADS" --reference-dir "$REFERENCES" --device cuda --stage validation --confirm-oracle-policy same-split-posthoc
"$PYTHON" -m task_demand_prediction pdra-run --repo-root "$REPO" --output-dir "$OUT" --model-dir "$MODEL" --data-dir "$DATA" --adapter-dir "$ADAPTERS" --head-dir "$HEADS" --reference-dir "$REFERENCES" --device cuda --stage final-test --confirm-oracle-policy same-split-posthoc
```

The `--confirm-oracle-policy same-split-posthoc` flag records acceptance of the split-specific Oracle-Ranked diagnostic. Resume only with the same output directory and matching registered identity. A result without its integrity marker is not overwritten. If a process stops after writing `<stage>/COMPLETE.json` but before `status/<stage>.DONE.json`, `--resume` verifies the registration, exact frozen run set, all run and analysis hashes, and only then creates the missing DONE marker. A present invalid DONE or RUNNING marker is preserved and causes resume to stop.

The result directory contains registration, sealed per-unit records, stage summaries, and completion markers. It does not contain copies of the model or datasets.

## Protocol choice to confirm before the GPU run

The implementation uses same-split oracle sweeps to make the Oracle-Ranked and headroom diagnostics meaningful within each split. This costs 1,240 extra sweeps on final test and uses final-test labels only for the post-hoc Oracle-Ranked comparator. Confirm that this isolated diagnostic use is acceptable. A validation-only oracle order would remove those 1,240 final-test sweeps, but it would no longer measure final-test oracle headroom. The PDRA and Reverse-PDRA orders remain frozen from the validation-only predictor under either policy.
