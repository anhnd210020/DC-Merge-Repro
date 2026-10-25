#!/usr/bin/env bash
# Experiment 7 launcher. Every step resumes from its on-disk cache: rerun the same command to continue.
#
#   bash scripts/run_exp7.sh <step> [contexts]
#
#   step      replay | checks | gate | controls | method | entropy | wide | all8 | summarize
#   contexts  core (default) | four (all 7) | a comma list of context ids | all8 (step all8 only)
#
# Order: replay four -> checks four -> gate core (STOP, report the gate) -> controls core -> method/entropy core
#        (only if the gate allows) -> wide core -> controls/method/entropy four -> summarize.
# Parallel: any steps on any contexts may run at the same time in separate tmux sessions (caches are
#           merged under a file lock), but never the SAME step on the SAME context twice at once.
#           Summaries are rebuilt at the end of every step and by `summarize`.
#
# Required environment (e.g. from /workspace/ducanh/exp7_env.sh):
#   PYTHON MODEL FT_DIR HEADS DATA EXP3R EXP6 OUT and REFERENCE (archive | current; see the guide)
set -euo pipefail

: "${PYTHON:?set PYTHON}"; : "${MODEL:?set MODEL}"; : "${FT_DIR:?set FT_DIR}"; : "${HEADS:?set HEADS}"
: "${DATA:?set DATA}"; : "${EXP3R:?set EXP3R}"; : "${EXP6:?set EXP6}"; : "${OUT:?set OUT}"
: "${REFERENCE:?set REFERENCE to archive or current (see docs/IMPLEMENTATION_GUIDE_EXPERIMENT7.md)}"

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
mkdir -p "$OUT"
step=${1:?usage: run_exp7.sh <step> [contexts]}
ctx=${2:-core}
[[ "$step" == "all8" ]] && ctx=all8
tag=$(echo "$ctx" | tr ',' '+' | cut -c1-60)

if [[ "$step" == "summarize" ]]; then
  exec $PYTHON lane_harm/run_exp7.py summarize --out "$OUT"
fi

$PYTHON lane_harm/run_exp7.py "$step" \
  --repo "$REPO" --model "$MODEL" --ft-dir "$FT_DIR" --heads "$HEADS" --data "$DATA" \
  --exp3r "$EXP3R" --exp6 "$EXP6" --out "$OUT" --reference "$REFERENCE" --contexts "$ctx" \
  --device cuda ${EXTRA_ARGS:-} 2>&1 | tee -a "$OUT-$step-$tag.log"
