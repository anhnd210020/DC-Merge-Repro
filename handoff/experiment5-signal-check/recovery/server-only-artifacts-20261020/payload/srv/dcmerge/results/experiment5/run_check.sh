#!/usr/bin/env bash
set -euo pipefail
source /workspace/dcmerge_env.sh
REPO=/workspace/experiment5_package_20261019/DC-Merge
OUT=/srv/dcmerge/results/experiment5
cd "$REPO"
"$DCMERGE_MICROMAMBA" run -n "$DCMERGE_ENV_NAME" python lane_harm/run_entropy.py check \
  --repo "$REPO" --model "$DCMERGE_MODEL_DIR" --ft-dir "$DCMERGE_FT_DIR" \
  --heads "$DCMERGE_HEAD_DIR" --data "$DCMERGE_DATA_DIR" --exp3 /srv/dcmerge/results/experiment3 \
  --out "$OUT" --device cuda --contexts core 2>&1 | tee "$OUT/check.log"
