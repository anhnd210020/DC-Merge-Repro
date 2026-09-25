# State-dependent task drift

This isolated package implements the four-task pilot without changing the existing task-scale experiment. The conditional checkpoint stores only the fresh target adapter, so it represents `Delta_(A|B,alpha)` relative to its hashed anchor, never relative to `M0`.

Run-level resume is supported. A complete run is skipped only with `--resume`; otherwise overwrite is refused. Failed or incomplete runs restart from the beginning when `--resume` is supplied. Optimizer, scheduler, scaler, and RNG state are not resumed mid-training.

The primary drift is the unweighted macro mean over valid modules. The secondary energy-weighted diagnostic weights each module by the mean of the two compared squared Frobenius norms. Zero-rank comparisons are marked degenerate and excluded. Pairwise DC-Merge calls the original repository implementation with `task_scales=None`, selects alpha using only the two validation tasks, freezes the selection, and only then reads test data.

`dry-run` and `list-runs` do not import Transformers/PEFT, instantiate models or datasets, load checkpoints, use CUDA, or access the network. `preflight` inspects real local resources but never downloads them. The existing SVHN loader has `download=True`; preflight therefore requires both `.mat` files before execution so that torchvision has nothing to download.

Paper-style projected DirSim is deliberately recorded as `not_implemented_exact_definition_unverified`: no exact repository implementation was found, and this package does not relabel an approximation as DirSim.

## Repair-pass integrity rules

Analysis accepts only runs whose status and metadata are both `COMPLETE`, whose on-disk adapter checksum matches both metadata and completion status, and whose deterministic `run_definition_hash` matches the current requested configuration. Conditional validation reconstructs the expected anchor identity from the current canonical B adapter and checksum. Real training records the Git SHA and `git_dirty`; commit the implementation before scientific execution so the SHA identifies the executed code.

Saved adapters are reloaded from the actual PEFT adapter files before completion. Effective deltas reconstructed from disk are compared with the validation-selected in-memory deltas. Conditional runs additionally verify the module-level identity `M0 + alpha*Delta_B + Delta_A_given_B`. This checkpoint layer is dataset-free; attachment to a second full CLIP instance remains part of the one-step server smoke test rather than the laptop unit suite.

Per-module, per-layer, overall, stochastic-seed, directed-pair, and symmetric unordered-pair outputs remain separate. Standard deviations use sample standard deviation (`N-1`), with zero for a single observation. The aggregate stage writes `summaries/pair_level_state_vs_merge.{json,csv}` and makes no causal claim.

Every pairwise alpha is evaluated after exactly restoring the affected M0 weights. A targeted `run-pilot --target-task A --anchor-task B` automatically includes `A@M0 seed 420` and `B@M0 seed 420` baseline dependencies; if a different target seed is selected, that target baseline is included too.

## Precision

The checked-in pilot configuration uses `fp32`. All full scientific baseline, conditional, pairwise, and orchestration commands must use `--precision fp32` or omit the flag. Reduced steps/batches are engineering smoke controls; another precision would be a non-scientific engineering change and is not the recommended pilot recipe. The scientific path disables autocast for FP32 forward and evaluation computations, while retaining the original reproduction's CUDA `GradScaler` behavior during training. The repository's legacy `evaluate_cliphead()` uses CUDA autocast, so its evaluation is not expected to be bit-for-bit numerically identical to this explicitly FP32 path.

## Command examples

Engineering smoke test for one directed pair (use a separate smoke output root):

```bash
python -m state_dependent_task_drift.cli run-pilot \
  --config state_dependent_task_drift/pilot_config.json \
  --pretrained-model <MODEL_PATH> --dataset-root <DATASET_ROOT> --head-root <HEAD_ROOT> \
  --output-root <SMOKE_OUTPUT_ROOT> --device cuda --precision fp32 --workers 8 \
  --target-task stanford_cars --anchor-task eurosat --target-seed 420 \
  --max-train-steps 1 --max-eval-batches 1
```

Full scientific pilot:

```bash
python -m state_dependent_task_drift.cli run-pilot \
  --config state_dependent_task_drift/pilot_config.json \
  --pretrained-model <MODEL_PATH> --dataset-root <DATASET_ROOT> --head-root <HEAD_ROOT> \
  --output-root <OUTPUT_ROOT> --device cuda --precision fp32 --workers 8
```

Run-level resume uses the identical command plus `--resume`. Incompatible current configuration is rejected instead of skipped.
