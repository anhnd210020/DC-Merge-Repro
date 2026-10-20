#!/usr/bin/env bash
set -euo pipefail
source /workspace/dcmerge_env.sh
cd "$DCMERGE_REPO/vision_lora_merge"
exec "$DCMERGE_MICROMAMBA" run -n "$DCMERGE_ENV_NAME" \
  python eval.py \
  --config vitB32_r16_8task \
  --method dc_merge \
  --smoothing linear \
  "$@"
