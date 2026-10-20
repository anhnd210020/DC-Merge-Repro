#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$ROOT/handoff/experiment4/config/dcmerge_env.sh"
REPO="$ROOT/handoff/experiment3/code"
E5_CODE="$ROOT/handoff/experiment5/code/lane_harm"
EXP3="$ROOT/handoff/experiment3/results/experiment3"
OUT="${DCMERGE_EXP5_OUT:-/srv/dcmerge/results/experiment5-rerun-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
cd "$E5_CODE"
"$DCMERGE_MICROMAMBA" run -n "$DCMERGE_ENV_NAME" python run_entropy.py check \
  --repo "$REPO" --model "$DCMERGE_MODEL_DIR" --ft-dir "$DCMERGE_FT_DIR" \
  --heads "$DCMERGE_HEAD_DIR" --data "$DCMERGE_DATA_DIR" --exp3 "$EXP3" \
  --out "$OUT" --device cuda --contexts core 2>&1 | tee "$OUT/check.log"
