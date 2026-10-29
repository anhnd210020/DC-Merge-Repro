#!/usr/bin/env bash
set -uo pipefail
source /workspace/exp8_env.sh
cd /workspace/experiment8
C="--out $OUT --data-dir $DATA --device cuda"
run(){ echo "[$(date -u +%F' '%T)] START $*"; python -m exp8.run "$@" $C; rc=$?; echo "[$(date -u +%F' '%T)] END rc=$rc $*"; return $rc; }
must(){ run "$@" || { echo "CONFIRMATORY STEP FAILED: $*"; exit 1; }; }
try(){ run "$@" || echo "EXPLORATORY STEP FAILED (continuing): $*"; }
must prepare-data
must check
for OP in ta iso_cts tsvm; do
  [ -f "$OUT/$OP/alpha.json" ] || must alpha --op $OP
  for S in subsets analyze select report factorial random probes; do must $S --op $OP; done
  if [ "$OP" = "ta" ]; then
    n=$(python -c "import json;print(json.load(open('$OUT/ta/selection.json'))['n_victims'])")
    echo "ta n_victims=$n"
    if [ "$n" -lt 5 ]; then echo "FUTILITY STOP: fewer than 5 eligible victims under task arithmetic"; exit 2; fi
  fi
done
must select --all-ops
for OP in ta iso_cts tsvm; do
  for S in blocks own patch crossterm weightspace; do try $S --op $OP; done
done
must depthstats
try replicate --op ta --backbone ViT-L-14 --tau-device cpu
echo "ALL DONE"
