#!/usr/bin/env bash
set -euo pipefail

source /workspace/dcmerge_env.sh
cd /workspace/ducanh/DC-Merge
exec "$DCMERGE_MICROMAMBA" run -n "$DCMERGE_ENV_NAME" \
  python /workspace/dcmerge_oracle_baseline_gate.py \
  --repo /workspace/ducanh/DC-Merge \
  --model "$DCMERGE_MODEL_DIR" \
  --ft-dir "$DCMERGE_FT_DIR" \
  --heads "$DCMERGE_HEAD_DIR" \
  --data "$DCMERGE_DATA_DIR" \
  --exp2 /srv/dcmerge/results/experiment2 \
  --out /srv/dcmerge/results/experiment3 \
  --contexts four --device cuda
