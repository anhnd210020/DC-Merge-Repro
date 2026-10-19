# Experiment 4: leave-one-context-out task coefficient transfer

This package contains the exact Experiment 4 evaluation driver and wrapper, the
execution path configuration, all seven context results and caches, the complete
run log, the machine-readable aggregate and run manifest, and SHA-256 checksums.
No dataset, model weight, adapter, head, authentication token, or download cache is
included.

## Protocol

For each task in a held-out context, the transferred ratio is the unweighted mean
of `oracle_fair c_i / alpha` over the other Experiment 3 contexts that contain
that task. The held-out context's own `dc_rank` alpha scales this ratio. The held-out
context contributes no coefficient-selection information. Test results do not
form the transferred coefficients. No new search was run; there were seven new
validation and seven new test evaluations. DTD has no other occurrence in the
seven-context set, so its ratio is neutral `1.0` and is explicitly flagged as
unsupported.

## Contents

- `results/run_loco_task_transfer.py`, `results/run_loco_task_transfer.sh`: exact
  execution source and wrapper used for this run.
- `results/run_manifest.json`: context list, formula, runtime, evaluation counts,
  source hashes, and unchanged-baseline/oracle status.
- `results/transfer_coefficients.json`, `results/aggregate.json`, and
  `results/transfer_summary.md`: coefficients and aggregate/context reporting.
- `results/<context>/transfer_result.json`, `val_cache.json`, and
  `test_cache.json`: complete per-context records and caches.
- `results/run.log`: full run log; byte-identical to the server's top-level
  `experiment4_task_transfer_20261019.log` at packaging time.
- `config/dcmerge_env.sh`: exact non-secret path/environment-name configuration.
- `REPORT.md`: concise findings and caveats.
- `SHA256SUMS`: SHA-256 for every file in this package except `SHA256SUMS` itself.

## Reproduction

The run used the Experiment 3 result directory as its source of validation-selected
coefficients and baseline evaluations. It used the same eight-task model, adapters,
heads, and datasets from the prior experiments. These large assets are not hosted
in this repository; use the paths or equivalent verified copies recorded in
`results/run_manifest.json` and `config/dcmerge_env.sh`.

The original invocation is preserved in `results/run_loco_task_transfer.sh`. For a
new machine, run the packaged Python driver with the same arguments shown there,
adjusting only asset paths and the output directory. The corresponding environment
specification is `../experiment3/code/environment_vision.yml`. Runtime versions
and exact source-file hashes are in `results/run_manifest.json`.

## Integrity check

From this directory, run:

```bash
sha256sum -c SHA256SUMS
```
