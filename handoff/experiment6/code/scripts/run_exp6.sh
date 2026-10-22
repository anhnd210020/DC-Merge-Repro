#!/usr/bin/env bash
# Experiment 6 launcher. Every step is resumable: rerun the same step to continue.
#
#   bash scripts/run_exp6.sh rebaseline    # replay Exp 3's reference vectors on THIS machine (~30-40 min)
#   bash scripts/run_exp6.sh 6a-check      # cheap gate, core only; STOP and read the gate
#   bash scripts/run_exp6.sh 6a-run-core   # only if the gate is PASS or IN_BETWEEN
#   bash scripts/run_exp6.sh 6a-run-four   # only if the gate is PASS
#   bash scripts/run_exp6.sh 6b-mechanism  # core, validation
#   bash scripts/run_exp6.sh 6b-oracle-core
#   bash scripts/run_exp6.sh 6b-oracle-four
#   bash scripts/run_exp6.sh 6b-qkvo-core  # optional, secondary grouping
#   bash scripts/run_exp6.sh 6b-summarize
#
# Long steps: run inside tmux, e.g.
#   tmux new -d -s exp6b "bash scripts/run_exp6.sh 6b-oracle-core"
set -euo pipefail

# --- environment (same variables as Experiments 2-5) -------------------------
if [[ -f /workspace/dcmerge_env.sh ]]; then source /workspace/dcmerge_env.sh; fi
PYTHON=${PYTHON:-"$DCMERGE_MICROMAMBA run -n $DCMERGE_ENV_NAME python"}
MODEL=${MODEL:-$DCMERGE_MODEL_DIR}
FT_DIR=${FT_DIR:-$DCMERGE_FT_DIR}
HEADS=${HEADS:-$DCMERGE_HEAD_DIR}
DATA=${DATA:-$DCMERGE_DATA_DIR}
EXP3=${EXP3:-/srv/dcmerge/results/experiment3}
EXP5=${EXP5:-/srv/dcmerge/results/experiment5}
# Current-environment replay of Exp 3 (rebaseline step). 6a-check reads the ARCHIVED $EXP3;
# 6a-run and all 6b steps read $EXP3_CURRENT, so their checks compare runs on the same machine.
EXP3_CURRENT=${EXP3_CURRENT:-${OUT:-/srv/dcmerge/results/experiment6}_exp3_current}
OUT=${OUT:-/srv/dcmerge/results/experiment6}

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
mkdir -p "$OUT"
ASSETS="--repo $REPO --model $MODEL --ft-dir $FT_DIR --heads $HEADS --data $DATA --device cuda"
BASE="$ASSETS --exp3 $EXP3"                 # archived Exp 3 (6a gate only)
BASE_CUR="$ASSETS --exp3 $EXP3_CURRENT"     # this machine's replay (6a run, 6b)
need_current() {
  if [[ ! -f "$EXP3_CURRENT/replay_summary.md" ]]; then
    echo "missing $EXP3_CURRENT/replay_summary.md: run 'bash scripts/run_exp6.sh rebaseline' first" >&2; exit 2
  fi
}

step=${1:?usage: run_exp6.sh <step>}
case "$step" in
  rebaseline)
    $PYTHON lane_harm/rebaseline_exp3.py $ASSETS --exp3 "$EXP3" --out "$EXP3_CURRENT" --contexts four 2>&1 | tee -a "$OUT-rebaseline.log" ;;
  6a-check)
    $PYTHON lane_harm/run_agreement.py check $BASE --exp5 "$EXP5" --out "$OUT" --contexts core 2>&1 | tee -a "$OUT-6a-check.log" ;;
  6a-run-core)
    need_current; $PYTHON lane_harm/run_agreement.py run $BASE_CUR --exp5 "$EXP5" --out "$OUT" --contexts core 2>&1 | tee -a "$OUT-6a-core.log" ;;
  6a-run-four)
    need_current; $PYTHON lane_harm/run_agreement.py run $BASE_CUR --exp5 "$EXP5" --out "$OUT" --contexts four 2>&1 | tee -a "$OUT-6a-four.log" ;;
  6b-mechanism)
    need_current; $PYTHON lane_harm/run_layer_oracle.py mechanism $BASE_CUR --out "$OUT" --contexts core 2>&1 | tee -a "$OUT-6b-mechanism.log" ;;
  6b-oracle-core)
    need_current; $PYTHON lane_harm/run_layer_oracle.py oracle $BASE_CUR --out "$OUT" --contexts core 2>&1 | tee -a "$OUT-6b-core.log" ;;
  6b-oracle-four)
    need_current; $PYTHON lane_harm/run_layer_oracle.py oracle $BASE_CUR --out "$OUT" --contexts four 2>&1 | tee -a "$OUT-6b-four.log" ;;
  6b-qkvo-core)
    need_current; $PYTHON lane_harm/run_layer_oracle.py oracle $BASE_CUR --out "$OUT" --contexts core --grouping qk_vo 2>&1 | tee -a "$OUT-6b-qkvo.log" ;;
  6b-summarize)
    $PYTHON lane_harm/run_layer_oracle.py summarize --out "$OUT"
    if [[ -d "$OUT/grouping_qk_vo" ]]; then $PYTHON lane_harm/run_layer_oracle.py summarize --out "$OUT" --grouping qk_vo; fi ;;
  *)
    echo "unknown step: $step" >&2; exit 2 ;;
esac
