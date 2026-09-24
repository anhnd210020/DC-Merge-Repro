# EuroSAT/SUN397 post-smoothing task-scale ablation

This experiment is a post-hoc intervention on the already-trained eight LoRA
adapters. It never invokes an optimizer and never retrains an adapter.

## Intervention and diagnostic

For task `i`, DC-Merge first computes the rank-16 SVD and the existing linear
smoothed singular distribution. The optional task scale multiplies the full
smoothed singular-value vector immediately afterward:

```text
smoothed_singular_i = sum(original_singular_i)
                      * linear_distribution_i
                      * task_scale_i
```

The shared left/right cover bases are constructed solely from the singular
directions, so this scalar does not change a task's directions, its normalized
smoothed singular distribution, or the cover bases. It is applied before
cover-space projection, top-0.1% filtering, TIES sign consensus, and
aggregation. Those cross-task operations may therefore change, as intended.
This mechanism is an ablation and is not part of the original DC-Merge method.

Realized effective weight uses
`dcmerge-diagonal-block-squared-frobenius-v1`: for each of 48 modules, take the
squared Frobenius norm of each task-indexed rank-16 diagonal block in the
post-top-k, post-TIES cover-space aggregate after applying the structural block
mask; sum by task across modules; normalize the eight sums to 100%. It is not a
learned task coefficient and is not TIES source attribution. Because this is a
squared-norm quantity, input amplitude and realized weight are not linearly
related even before top-k/TIES.

## Preregistered configurations

The full named mappings live in `task_scale_configs.json`.

| Run | EuroSAT | SUN397 | Other six tasks |
|---|---:|---:|---:|
| `baseline` | 1.00 | 1.00 | 1.00 |
| `partial_balance` | 2.00 | 0.80 | 1.00 |
| `full_balance_initial` | 3.07 | 0.60 | 1.00 |
| `strong_reallocation` | 4.00 | 0.42 | 1.00 |
| `strong_reversal` | 4.50 | 0.32 | 1.00 |

The last three pairs are preregistered linear-proxy choices, not promises about
realized squared-Frobenius weights.

## Server commands

Set paths only if they differ from the defaults in the launcher. The validation
stage evaluates all five configurations at alphas 0.1 through 3.0, using only
validation data, then freezes `selected_for_test.json`.

```bash
export DCMERGE_SELFTRAIN_ROOT=/workspace/selftrained_b32_r16_8task
export DCMERGE_MODEL_DIR=/workspace/models/clip-vit-base-patch32
export DCMERGE_DATA_DIR=/workspace/datasets8
export DCMERGE_TASK_SCALE_OUTPUT=/workspace/analysis_outputs/task_scale_ablation

tmux new -s dcmerge-scales
bash launch_task_scale_ablation.sh validation
```

After reviewing and accepting the frozen manifest, final test is a separate
command. It reconstructs only the manifest-approved merges at their frozen
alphas and does not run an alpha sweep.

```bash
bash launch_task_scale_ablation.sh final-test
```

Equivalent direct commands are:

```bash
python run_task_scale_ablation.py \
  --config vitB32_r16_8task_selftrained \
  --scale-config task_scale_configs.json \
  --stage validation \
  --artifact-root /workspace/selftrained_b32_r16_8task \
  --model-dir /workspace/models/clip-vit-base-patch32 \
  --data-dir /workspace/datasets8 \
  --output-dir /workspace/analysis_outputs/task_scale_ablation \
  --resume

python run_task_scale_ablation.py \
  --config vitB32_r16_8task_selftrained \
  --scale-config task_scale_configs.json \
  --stage final-test \
  --selection-manifest /workspace/analysis_outputs/task_scale_ablation/selected_for_test.json \
  --artifact-root /workspace/selftrained_b32_r16_8task \
  --model-dir /workspace/models/clip-vit-base-patch32 \
  --data-dir /workspace/datasets8 \
  --output-dir /workspace/analysis_outputs/task_scale_ablation \
  --resume
```

The launcher uses `set -Eeuo pipefail`, writes timestamped combined logs, runs
a complete preflight before model loading, resumes successful units, and writes
a stage `DONE` marker only after the Python process exits successfully.

## Selection and leakage controls

For every run, alpha is selected by the same verified objective: the unweighted
mean over eight tasks of merged validation accuracy divided by that task's
single-task validation accuracy. A strict `>` comparison makes the earliest
alpha win ties. Test denominators and test loaders are not accessed during
validation. Final test refuses unknown scales, changed scales, changed task
order, invalid alphas, or a modified manifest hash.

The all-ones run is a gate. Before any scaled run continues, it must match the
legacy/default merged delta, the stored effective weights, and selected alpha
0.8. During final test, baseline per-task accuracy must match the stored result
within 0.05 percentage points.

## Outputs

The output root contains preflight records, validation/final aggregate CSV and
JSON, the frozen manifest, validation-only PNG/PDF dose-response figures, logs,
and stage status files. Every run directory contains immutable metadata,
effective weights, the full alpha sweep, selected validation metrics, task-level
CSV/JSON, summaries, and completion markers. Final-test files have distinct
names and do not overwrite validation artifacts.
