#!/usr/bin/env bash
set -Eeuo pipefail

repo="${DCMERGE_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
python_bin="${DCMERGE_PYTHON:-python3}"
export DCMERGE_REPO="$repo"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"

for name in DCMERGE_MODEL_DIR DCMERGE_DATA_DIR DCMERGE_ADAPTER_DIR DCMERGE_HEAD_DIR DCMERGE_OUTPUT_DIR; do
  if [[ -z "${!name:-}" ]]; then
    printf 'Missing required environment variable %s. See task_demand_prediction/README.md.\n' "$name" >&2
    exit 2
  fi
done

logdir="${DCMERGE_OUTPUT_DIR}.launcher-logs"
mkdir -p -- "$logdir"
log="$logdir/launch-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$log") 2>&1
on_error() {
  code=$?
  marker="$logdir/launch.FAILED"
  tmp="$marker.tmp"
  printf 'exit_code=%s\nline=%s\n' "$code" "${BASH_LINENO[0]:-unknown}" > "$tmp"
  mv -f -- "$tmp" "$marker"
  exit "$code"
}
trap on_error ERR

cd -- "$repo"
"$python_bin" -m task_demand_prediction run --stage both --repo-root "$repo" --device cuda
mkdir -p -- "$DCMERGE_OUTPUT_DIR/logs"
cp -- "$log" "$DCMERGE_OUTPUT_DIR/logs/"
rm -f -- "$logdir/launch.FAILED"
