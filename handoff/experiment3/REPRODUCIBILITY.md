# Reproducing Experiment 3

## Required inputs

The package intentionally omits the large or sensitive assets. Supply the same eight-task ViT-B/32 base model, eight LoRA adapters, eight CLIP heads, and eight datasets (Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, SVHN) used by the original experiment. The original instance used:

- model: `/workspace/models/clip-vit-base-patch32`
- adapter directory: `/workspace/models/DC-Merge-Vision-B32-r16-8task/lora_checkpoints`
- heads: `/workspace/models/DC-Merge-Vision-B32-r16-8task/lora_heads`
- data root: `/workspace/datasets8`
- micromamba environment: `dcmerge_vision`

Set these paths from your own secure environment; do not place credentials or asset files in this repository. `code/environment_vision.yml` is the package environment recipe. It is not a full lockfile for the historical runtime.

## Exact saved-baseline replay

Run from the repository root, after activating the compatible environment and setting `MODEL`, `FT_DIR`, `HEADS`, `DATA`, and `OUT` to valid paths. These commands read the packaged direct-replay inputs and write to a new output directory; keep the captured results directory intact.

```bash
REPO="$PWD/handoff/experiment3/code"
EXP2="$PWD/handoff/experiment3/results/experiment2/current_baseline"
OUT="${OUT:-/srv/dcmerge/results/experiment3-reproduction}"
PYTHON="${PYTHON:-python}"
ARGS=(--repo "$REPO" --model "$MODEL" --ft-dir "$FT_DIR" --heads "$HEADS" \
      --data "$DATA" --exp2 "$EXP2" --out "$OUT" --device cuda)

"$PYTHON" "$REPO/lane_harm/run_oracle.py" run "${ARGS[@]}" --contexts core \
  2>&1 | tee "$OUT-core.log"
"$PYTHON" "$REPO/lane_harm/run_oracle.py" run "${ARGS[@]}" --contexts four \
  2>&1 | tee "$OUT-four.log"
"$PYTHON" "$REPO/lane_harm/run_oracle.py" summarize --out "$OUT"
```

The `four` command contains all seven contexts and skips any already-completed result record in that output directory. Validation and test caches make an interrupted run resumable. `scripts/run_phase.sh core|four` provides the same phases and saves a phase log. For detached durable sessions, start `tmux new-session -d -s experiment3-core` (then send the `run_phase.sh core` command into it), wait for completion, and repeat with session `experiment3-four` and phase `four`. The `four` phase rebuilds the summary.

The included baseline directory contains only the seven current-environment `dc_rank` records and their CT comparison records, which are the inputs the oracle runner consumes. The archived Experiment 2 comparisons are kept separately and unchanged. Comparing the direct replay with those archived Oct 16 values produces the documented historical discrepancy; do not replace the current baseline or alter the protocol to hide it.

## Verification of this handoff

`results/experiment3/final_verification.json` records seven result records, 28 method/context CSV rows, exact current-replay baseline checks (0.000 pp for validation and test), passing validation constraints, maximum lane-sum error, unchanged archive compare hash, and successful core/four wrapper exits. Per-context details are in `results/experiment3/contexts/*/oracle_result.json`.
