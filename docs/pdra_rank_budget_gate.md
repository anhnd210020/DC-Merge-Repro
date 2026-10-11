# Uniform-32 rank budget gate (validation only)

This follow-up tests whether giving every task a post-projection budget equal
to its assigned rank improves the four-task Uniform baseline. It is a new
operator variant, independent of the sealed PDRA validation run. It does not
evaluate PDRA allocations, rerun the rank-demand oracle, or open final-test.

Each of the 45 fresh four-task contexts uses ranks `[8,8,8,8]`, task scales
`1.0`, task densities `1.0`, linear smoothing with `rho=5.0`, and the existing
cover-space, TIES, structural mask, and model scaling paths. The new
`topk_policy='rank'` targets eight projected entries per task per module
before TIES, using the legacy magnitude threshold and its tie behavior.
Exact ties can retain more than eight; the run records actual support counts.
The legacy policy remains the default. The global task-wide density filter
uses the same rank budget when density is 1.0.

The baseline is the completed, sealed archive
`PDRA-validation-backup-20261011.tar.gz` with SHA-256
`77dd34ac25fad0fb57ddc5e2f59a37bea709c2ffb879542a07fc520b5b25d070`.
The gate verifies all 1,496 archived run results and unit markers, completion
and DONE markers, frozen artifacts, and three analysis hashes before reading
its 45 original Uniform results. It then compares the new results context by
context with the old results on the same validation references. The summary
reports paired means, wins/losses, and worst-task retention; a GO/NO-GO
decision requires review rather than an automatic per-context threshold.

## Laptop CPU checks (PowerShell)

Run from the repository on `codex/pdra-experiment`, with the existing `dev`
environment. The commands below call its Python executable directly; do not
create or install another environment. The source backup is a **baseline
input**, never an output directory.

```powershell
$Python = 'D:\envs\dev\Scripts\python.exe'
$Repo = 'D:\Research\DC-Merge-PDRA'
$Archive = Join-Path $env:USERPROFILE 'Downloads\PDRA-validation-backup-20261011.tar.gz'
Set-Location $Repo
& $Python -B -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'CPU tests failed' }
& $Python -m task_demand_prediction.rank_budget_gate preflight `
  --repo-root $Repo --baseline-archive $Archive --static-only
if ($LASTEXITCODE -ne 0) { throw 'Static gate failed' }
```

The PyTorch operator tests must **run and pass** on the laptop. A skipped
PyTorch test means the operator has not yet been verified.

## GPU server: asset checks and run

Transfer this branch and the verified archive to the server. Restore the
same eight adapters, eight heads, ViT-B/32 snapshot, eight datasets, and
validation references pinned by the PDRA predictor. Use a fresh output path
outside the repository. The commands below are templates; set paths to the
actual verified assets. No original PDRA output is needed on the server.

```bash
REPO=/srv/pdra/DC-Merge-PDRA
PYTHON=/opt/pdra/env/bin/python
ARCHIVE=/srv/pdra/assets/PDRA-validation-backup-20261011.tar.gz
OUT=/srv/pdra/results/uniform32-rank-budget-v1
MODEL=/srv/pdra/assets/clip-vit-base-patch32
DATA=/srv/pdra/assets/datasets8
ADAPTERS=/srv/pdra/assets/artifacts/selftrained_b32_r16_8task/checkpoints
HEADS=/srv/pdra/assets/assets/heads
REFERENCES=/srv/pdra/assets/references

"$PYTHON" -m task_demand_prediction.rank_budget_gate preflight \
  --repo-root "$REPO" --baseline-archive "$ARCHIVE" --output-dir "$OUT" \
  --model-dir "$MODEL" --data-dir "$DATA" --adapter-dir "$ADAPTERS" \
  --head-dir "$HEADS" --reference-dir "$REFERENCES" --device cuda

"$PYTHON" -m task_demand_prediction.rank_budget_gate run \
  --repo-root "$REPO" --baseline-archive "$ARCHIVE" --output-dir "$OUT" \
  --model-dir "$MODEL" --data-dir "$DATA" --adapter-dir "$ADAPTERS" \
  --head-dir "$HEADS" --reference-dir "$REFERENCES" --device cuda
```

To resume after a stopped process, ensure no copy of it is still running,
then repeat the `run` command with `--resume`. The output identity includes
the asset hashes, source hashes, frozen baseline archive and completion hash.
It refuses mismatched completed unit markers. To verify after completion,
repeat the command with `verify` in place of `run`.

Review `validation/summary.json` for the mean paired difference, losses,
and changes to worst-task retention. Changing the operator changes the
meaning of the old oracle sweep, so its rankings and predictor should not
be treated as calibrated under this variant.
