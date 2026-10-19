#!/usr/bin/env bash
set -euo pipefail

phase="${1:?usage: run_phase.sh core|four}"
case "$phase" in core|four) ;; *) echo "phase must be core or four" >&2; exit 2 ;; esac
: "${MODEL:?set MODEL to the base ViT-B/32 model directory}"
: "${FT_DIR:?set FT_DIR to the LoRA adapter root}"
: "${HEADS:?set HEADS to the CLIP head root}"
: "${DATA:?set DATA to the eight-dataset root}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../code" && pwd)"
package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exp2="$package_root/results/experiment2/current_baseline"
out="${OUT:-/srv/dcmerge/results/experiment3-reproduction}"
python="${PYTHON:-python}"
mkdir -p "$out"
args=(--repo "$repo" --model "$MODEL" --ft-dir "$FT_DIR" --heads "$HEADS"
      --data "$DATA" --exp2 "$exp2" --out "$out" --device cuda)
"$python" "$repo/lane_harm/run_oracle.py" run "${args[@]}" --contexts "$phase" \
  2>&1 | tee "$out-$phase.log"
if [[ "$phase" == four ]]; then
  "$python" "$repo/lane_harm/run_oracle.py" summarize --out "$out"
fi
