#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/workspace/selftrained_b32_r16_8task
STATE="$ROOT/state"
LOGS="$ROOT/logs"
BASELINE_LOG=/workspace/dcmerge_b32_8task.log
RECORD="$STATE/baseline_process.json"
MASTER="$ROOT/run_selftrain_pipeline.sh"

mkdir -p "$STATE" "$LOGS"
exec >>"$LOGS/queue.log" 2>&1
exec 8>"$STATE/queue.lock"
if ! flock -n 8; then
    echo "QUEUE_ALREADY_LOCKED timestamp=$(date --iso-8601=seconds)"
    exit 0
fi
if [[ -f "$STATE/COMPLETE" ]]; then
    echo "QUEUE_SEES_COMPLETE timestamp=$(date --iso-8601=seconds)"
    exit 0
fi
if [[ ! -s "$RECORD" ]]; then
    echo "QUEUE_ERROR missing baseline process record: $RECORD"
    exit 1
fi

baseline_pid=$(jq -er '.pid' "$RECORD")
baseline_start_ticks=$(jq -er '.start_ticks' "$RECORD")
baseline_command=$(jq -er '.command_line' "$RECORD")
echo "QUEUE_START=$(date --iso-8601=seconds) pid=$baseline_pid"
echo "BASELINE_COMMAND=$baseline_command"

same_baseline_process() {
    kill -0 "$baseline_pid" 2>/dev/null || return 1
    [[ -r "/proc/$baseline_pid/stat" ]] || return 1
    current_start_ticks=$(awk '{print $22}' "/proc/$baseline_pid/stat" 2>/dev/null) || return 1
    [[ "$current_start_ticks" == "$baseline_start_ticks" ]]
}

while same_baseline_process; do
    sleep 20
done

completion_timestamp=$(date --iso-8601=seconds)
total_seconds_found=false
sleep 10
for _ in $(seq 1 8); do
    if grep -aq 'TOTAL_SECONDS=' "$BASELINE_LOG"; then
        total_seconds_found=true
        break
    fi
    sleep 20
done

{
    echo "baseline_completion_timestamp=$completion_timestamp"
    echo "baseline_log=$BASELINE_LOG"
    echo "total_seconds_found=$total_seconds_found"
} >"$STATE/baseline_completion.tmp"
mv "$STATE/baseline_completion.tmp" "$STATE/baseline_completion"

remaining=$(python3 "$ROOT/baseline_processes.py" --all)
if [[ "$remaining" != "[]" ]]; then
    echo "QUEUE_ERROR matching baseline processes remain: $remaining"
    exit 1
fi
if tmux has-session -t dcmerge-selftrain 2>/dev/null; then
    echo "QUEUE_NOOP dcmerge-selftrain already exists"
    exit 0
fi
if [[ -f "$STATE/COMPLETE" ]]; then
    echo "QUEUE_NOOP self-training is COMPLETE"
    exit 0
fi

tmux new-session -d -s dcmerge-selftrain \
    "bash -lc 'exec /workspace/selftrained_b32_r16_8task/run_selftrain_pipeline.sh >> /workspace/selftrained_b32_r16_8task/logs/master.log 2>&1'"
echo "SELFTRAIN_SESSION_STARTED=$(date --iso-8601=seconds)"
