# Experiment 3 handoff package

This directory packages the completed seven-context Experiment 3 oracle run with the exact runner-side code, compact outputs, its current-environment baseline inputs, archived Experiment 2 comparisons, and the baseline audit. It contains no model weights, adapters, dataset images, authentication material, or large raw logs.

## Start here

1. Read [`report.md`](report.md) for the equations, search protocol, quantitative findings, fair vectors, and limitations.
2. Read [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the run commands and required external assets.
3. Read [`results/experiment3/oracle_summary.md`](results/experiment3/oracle_summary.md) for the original per-context table, and [`results/experiment3/final_verification.json`](results/experiment3/final_verification.json) for machine-readable completion checks.
4. Read [`audit/replay-diagnosis.md`](audit/replay-diagnosis.md) and [`audit/baseline_replay_warning.md`](audit/baseline_replay_warning.md) for the Oct 16 discrepancy audit.

## Contents

- `code/`: Experiment 3 runner and the exact official vision code/data-loader definitions used by it, plus the environment recipe. Small shuffled-index tensors are included because the official loader uses them. No image/model/adaptor/head assets are present.
- `docs/`: Experiment 2 and Experiment 3 implementation guides.
- `results/experiment3/`: summary, CSV, final verification, per-context result records and resumable validation/test caches, plus concise excerpts of the core/four run logs.
- `results/experiment2/archive_compare/`: unchanged archived `compare/` evidence (25 files).
- `results/experiment2/current_baseline/`: direct current-environment `dc_rank` and CT inputs consumed for the Experiment 3 comparisons, plus provenance manifest.
- `audit/`: saved audit notes and compact diagnostic evidence.

Full progress-bar logs remain on the server at `/srv/dcmerge/results/experiment3-core.log`, `/srv/dcmerge/results/experiment3-four.log`, and `/srv/dcmerge/results/experiment3-tmux.log`; their useful context/check/completion lines are in this package’s `results/experiment3/logs/` excerpts.
