# Task merge demand prediction

This package implements the supplied Experiment Card without training
adapters. The Card at `docs/experiment_card_pre_evaluation_task_merge_demand.docx`
was read in full (104 paragraphs, five tables, no comments or tracked changes);
its SHA-256 is recorded in `preregistration.json`.

## Frozen contexts and counts

- Phase A: all 28 unordered pairs among eight tasks, with both focal directions.
- Phase B: five distinct four-task contexts per task (40 focal rows; 25 unique contexts).
- Phase C: nine unique six-task memberships and 22 focal rows from the Experiment 3 context-selection registration. Only memberships are reused.
- Total: 62 contexts, 118 focal/context rows, 1,360 configurations per split, and 4,654 task/configuration cells per split.

## Oracle definitions

For focal task `i` in a fixed context, `R_i` is normalized accuracy retention
in percent. Signed deficits follow the Card:

- Scale: `R_i(1.5) - R_i(s)` for `s = .5, .75, 1, 1.25, 1.5`.
- Rank: `R_i(16) - R_i(r)` for `r = 2, 4, 8, 12, 16`.
- Density: `R_i(1) - R_i(d)` for `d = .25, .5, .75, 1`.
- Scale secondary metric: `[R_i(1.25)-R_i(.75)]/(1.25-.75)` in retention percentage-points per scale multiplier.

The operational normalized-AUC convention maps each tested resource's minimum
to `x=0` and maximum to `x=1`, then computes the signed trapezoidal integral
without dividing by interval width again. Thus scale uses `(s-.5)/(1.5-.5)`,
rank `(r-2)/(16-2)`, and density `(d-.25)/(1-.25)`. This places tested ranges
on a common unit axis; another width divisor would compute an average instead
of the requested signed integral. Negative values are preserved. The Card's
resource-specific deficit references remain `R_i(1.5)`, `R_i(16)`, and
`R_i(1)` for scale, rank, and density.

## Density operational rule

For a focal task, let `N` be the number of coordinates in its baseline support
after projection and top-k. Density `d` requests `K=round-half-up(d*N)` retained
coordinates across all modules. They are selected globally by descending
absolute value, then stable module name, then flattened coordinate index. At
`d=1`, the original baseline support is reproduced exactly. Diagnostics record
requested density, target and actual retained counts, and per-module counts.
The prior archive's per-module rounding retained 0/2,688 coordinates at 0.25
and 2,688/2,688 at both 0.50 and 0.75. Those density curves are invalid and
excluded from reuse.

## Features and evaluation

Features are computed before evaluation from task vectors and context geometry:
norms, raw spectral entropy and effective rank, energy retention, module norm
summaries, projection/top-k survival, partner cosine, subspace overlap, focal
energy outside the companion span, and retained-coordinate sign conflict and
coverage. Full definitions are in `features.py` and `FEATURES.json`.

Phase A primary grouped evaluation leaves one task out; leave-context-out is
secondary. Validation is the only split used to fit/select prediction features
and estimators. The Card permits a composite only when a single-feature signal
is reasonable, without giving a numeric threshold. The preregistered
operational rule is to enable the composite only when at least one individual
feature has a validation whole-context-bootstrap Spearman 95% interval that
excludes zero. Final-test labels do not participate in this gate, feature
selection, or estimator selection. Final-test grouped predictions fit on
validation labels while excluding the held task/context, then score the
matching final-test group.

## Prior results and input identity

Archived rank curves cover the exact 28 Phase A pairs, both splits, and levels
2/4/8/12/16. Raw retention observations may be reused only for those exact
contexts and splits after all input identity checks pass. The archive's old
`r/16` AUC must be recomputed from raw observations using `(r-2)/(16-2)`; do not
reuse pre-integrated AUC values. All eight adapter-weight and both reference
hashes match. Adapter-config hashes were not recorded; base model, heads,
dataset, and split files are unavailable locally. Therefore no historical rank
output is certified for import yet. Never transfer observations across
contexts. See `prior_result_audit.json` for archive inventory and comparisons.

## Local CPU verification

```powershell
python -m unittest discover -s tests -p 'test_task_demand_prediction.py' -v
python -m task_demand_prediction simulate --device cpu --output-dir "$env:TEMP/dcmerge-demand-simulation"
```

Simulation outputs are synthetic only and are not experimental results.

## Server preflight and run commands

The protocol is frozen. Before server evaluation, supply all assets below and
complete preflight. Preflight is an asset/hash check. Review validation outputs
before the final-test command.

```bash
export DCMERGE_REPO="/work/DC-Merge-Repro"
export DCMERGE_MODEL_DIR="/assets/clip-vit-base-patch32"
export DCMERGE_DATA_DIR="/assets/datasets8"
export DCMERGE_ADAPTER_DIR="/assets/selftrained/checkpoints"
export DCMERGE_HEAD_DIR="/assets/selftrained/heads"
export DCMERGE_REFERENCE_DIR="$DCMERGE_REPO/reproduction/selftrained_b32_r16_8task/results"
export DCMERGE_OUTPUT_DIR="/results/task-demand-prediction"
cd "$DCMERGE_REPO"
python -m task_demand_prediction preflight --device cuda \
  --model-dir "$DCMERGE_MODEL_DIR" --data-dir "$DCMERGE_DATA_DIR" \
  --adapter-dir "$DCMERGE_ADAPTER_DIR" --head-dir "$DCMERGE_HEAD_DIR" \
  --reference-dir "$DCMERGE_REFERENCE_DIR" --output-dir "$DCMERGE_OUTPUT_DIR"
python -m task_demand_prediction run --stage validation --device cuda \
  --model-dir "$DCMERGE_MODEL_DIR" --data-dir "$DCMERGE_DATA_DIR" \
  --adapter-dir "$DCMERGE_ADAPTER_DIR" --head-dir "$DCMERGE_HEAD_DIR" \
  --reference-dir "$DCMERGE_REFERENCE_DIR" --output-dir "$DCMERGE_OUTPUT_DIR"
# After validation outputs are reviewed:
python -m task_demand_prediction run --stage final-test --device cuda \
  --model-dir "$DCMERGE_MODEL_DIR" --data-dir "$DCMERGE_DATA_DIR" \
  --adapter-dir "$DCMERGE_ADAPTER_DIR" --head-dir "$DCMERGE_HEAD_DIR" \
  --reference-dir "$DCMERGE_REFERENCE_DIR" --output-dir "$DCMERGE_OUTPUT_DIR"
```

No inference or server connection was performed during this reconciliation.
