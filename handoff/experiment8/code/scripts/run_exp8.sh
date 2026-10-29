#!/usr/bin/env bash
# Experiment 8 on one GPU, in the order of the guide. Every step is resumable: rerun this
# script after an interruption and finished work is skipped.
# Usage: bash scripts/run_exp8.sh <OUT> <DATA> [extra args, e.g. --device cuda:0 --k-per-victim 1]
# The script pauses after task arithmetic. Continue with EXP8_CONTINUE=1.
# Set EXP8_SKIP_REPLICATE=1 to skip the ViT-L/14 replication.
set -euo pipefail
OUT=${1:?results folder}
DATA=${2:?image cache folder}
shift 2
COMMON="--out $OUT --data-dir $DATA $*"
cd "$(dirname "$0")/.."

python -m exp8.run prepare-data $COMMON
python -m exp8.run check $COMMON

for OP in ta iso_cts tsvm; do
  echo "=================== $OP ==================="
  for STEP in alpha subsets analyze select report factorial blocks random own probes patch crossterm weightspace; do
    if [ "$STEP" = "alpha" ] && [ -f "$OUT/$OP/alpha.json" ]; then continue; fi
    python -m exp8.run $STEP --op $OP $COMMON
  done
  if [ "$OP" = "ta" ] && [ -z "${EXP8_CONTINUE:-}" ]; then
    echo
    echo "Task arithmetic finished. Read ONLY $OUT/_shared/check.json and the number of victims in"
    echo "$OUT/ta/selection.json (\"n_victims\"). Fewer than 5: stop and report (futility stop)."
    echo "Otherwise continue with: EXP8_CONTINUE=1 bash scripts/run_exp8.sh $OUT $DATA $*"
    exit 0
  fi
done

python -m exp8.run select --all-ops $COMMON
python -m exp8.run depthstats $COMMON
if [ -z "${EXP8_SKIP_REPLICATE:-}" ]; then
  python -m exp8.run replicate --op ta --backbone ViT-L-14 --tau-device cpu $COMMON
fi
echo "Done. Read $OUT/depth_summary.md, then $OUT/who_summary.md and $OUT/vitl14/replicate_summary.md"
