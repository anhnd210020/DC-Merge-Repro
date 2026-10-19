# Baseline replay audit files

- `replay-diagnosis.md`: verified inputs and findings from the archived-versus-current replay investigation.
- `baseline_replay_warning.md`: concise provenance note included with Experiment 3 outputs.
- `replay_prediction_diagnostic.json`: per-example prediction comparison and diagnostics.
- `sequential_path_diagnostic.json` / `.log`: arithmetic-path checks.
- `same_input_gpu_check.log`: repeatability check using identical current-environment inputs.

The audit supports numerical sensitivity as a plausible explanation, but does not prove that the Oct 16 difference is normal stochastic GPU variation. Historical hardware/runtime details, archived per-example logits/predictions, and full image-byte hashes remain unavailable. Experiment 3 used the current direct replay as its primary baseline; all seven current-baseline reproduction checks passed at 0.000 pp.
