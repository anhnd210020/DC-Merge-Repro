# Experiment 5 handoff

This handoff preserves the completed Experiment 5 **signal check only**. The check evaluated all Experiment 3 cached validation vectors in the three core contexts and hit the guide's negative gate. **Coefficient search, test evaluation, and the four-context phase were not run.** See [report.md](report.md) first.

## Contents

- `code/lane_harm/`: the Experiment 5 runner and its compact lane-harm runtime dependencies. `lanes.py` and `run_oracle.py` are the same patched files as the Experiment 3 handoff; their hashes are in `run_metadata.json`.
- `docs/IMPLEMENTATION_GUIDE_EXPERIMENT5.md`: the uploaded experiment protocol.
- `results/`: the check table/JSON, seeded subset metadata, and all three resumable entropy caches.
- `logs/check.log`: the original signal-check log.
- `config/run_metadata.json`: source ZIP, code/runtime hashes and the exact run setup.
- `SHA256SUMS`: hashes of the handoff files (sidecar `SHA256SUMS.sha256` hashes that manifest).

## Re-run command

On the original Vast environment, set the paths in `handoff/experiment4/config/dcmerge_env.sh` and run `scripts/run_check.sh`. It writes to a fresh timestamped directory under `/srv/dcmerge` by default. The code needs the Experiment 3 result directory and model/data paths configured there. Re-running the signal check does not start the coefficient search; the guide's gate decision remains explicit in `report.md`.
