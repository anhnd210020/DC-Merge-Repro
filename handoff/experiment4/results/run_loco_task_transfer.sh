#!/usr/bin/env bash
set -euo pipefail
source /workspace/dcmerge_env.sh
out=/srv/dcmerge/results/experiment4_task_transfer_20261019
log=/srv/dcmerge/results/experiment4_task_transfer_20261019.log
exec "$DCMERGE_MICROMAMBA" run -n "$DCMERGE_ENV_NAME" \
  python /workspace/run_loco_task_transfer.py \
  --repo /workspace/ducanh/DC-Merge \
  --model "$DCMERGE_MODEL_DIR" \
  --ft-dir "$DCMERGE_FT_DIR" \
  --heads "$DCMERGE_HEAD_DIR" \
  --data "$DCMERGE_DATA_DIR" \
  --exp3 /srv/dcmerge/results/experiment3 \
  --out "$out" \
  --device cuda > "$log" 2>&1
