#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/workspace/selftrained_b32_r16_8task
REPO=/workspace/DC-Merge-Repro
MAMBA=/workspace/bin/micromamba
ENV_NAME=dcmerge_vision
STATE="$ROOT/state"
LOGS="$ROOT/logs"
TASKS=(stanford_cars dtd eurosat gtsrb mnist resisc45 sun397 svhn)

mkdir -p "$ROOT/checkpoints" "$ROOT/heads" "$LOGS" "$ROOT/results" "$STATE"
exec 9>"$STATE/selftrain.lock"
if ! flock -n 9; then
    echo "SELFTRAIN_ALREADY_LOCKED timestamp=$(date --iso-8601=seconds)"
    exit 0
fi
if [[ -f "$STATE/COMPLETE" ]]; then
    echo "SELFTRAIN_ALREADY_COMPLETE timestamp=$(date --iso-8601=seconds)"
    exit 0
fi

source /workspace/dcmerge_env.sh
export DCMERGE_SELFTRAIN_ROOT="$ROOT"
export DCMERGE_HEAD_DIR="$ROOT/heads"
export DCMERGE_FT_DIR="$ROOT/checkpoints"
export PYTHONPATH="$ROOT:$REPO/vision_lora_merge${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO/vision_lora_merge"

pipeline_start_epoch=$(date +%s)
printf 'RUNNING\n' >"$STATE/pipeline.status.tmp"
mv "$STATE/pipeline.status.tmp" "$STATE/pipeline.status"

on_exit() {
    rc=$?
    if [[ $rc -ne 0 ]]; then
        printf 'FAILED\n' >"$STATE/pipeline.status.tmp"
        mv "$STATE/pipeline.status.tmp" "$STATE/pipeline.status"
        echo "PIPELINE_FAILED rc=$rc timestamp=$(date --iso-8601=seconds)"
    fi
}
trap on_exit EXIT

echo "PIPELINE_START=$(date --iso-8601=seconds)"
echo "GIT_COMMIT=$(git -C "$REPO" rev-parse HEAD)"
echo "COMMAND=$0 $*"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

required_free_bytes=$((10 * 1024 * 1024 * 1024))
available_free_bytes=$(df -B1 --output=avail /workspace | tail -n 1 | tr -d ' ')
echo "DISK_PREFLIGHT available_bytes=$available_free_bytes required_bytes=$required_free_bytes"
if [[ ! "$available_free_bytes" =~ ^[0-9]+$ ]] || (( available_free_bytes < required_free_bytes )); then
    echo "DISK_PREFLIGHT_ABORT: /workspace requires at least 10 GiB free; available_bytes=$available_free_bytes"
    exit 1
fi
echo "DISK_PREFLIGHT_PASS"

CUDA_VISIBLE_DEVICES='' "$MAMBA" run -n "$ENV_NAME" \
    python "$ROOT/generate_heads.py" --device cpu --force 2>&1 | tee -a "$LOGS/heads.log"
CUDA_VISIBLE_DEVICES='' "$MAMBA" run -n "$ENV_NAME" \
    python "$ROOT/compare_heads.py" 2>&1 | tee -a "$LOGS/heads.log"

for task in "${TASKS[@]}"; do
    status_file="$STATE/$task.status"
    metadata_file="$STATE/$task.metadata.json"
    checkpoint="$ROOT/checkpoints/$task"
    if [[ -f "$status_file" ]] \
       && [[ "$(<"$status_file")" == DONE ]] \
       && [[ -s "$checkpoint/adapter_config.json" ]] \
       && [[ -s "$checkpoint/adapter_model.bin" ]] \
       && [[ -s "$metadata_file" ]] \
       && [[ "$(jq -r '.status // empty' "$metadata_file")" == DONE ]] \
       && [[ "$(jq -r '.verification.inference_succeeded // false' "$metadata_file")" == true ]] \
       && [[ "$(jq -r '.verification.loaded_tensor_comparison.exact_equal // false' "$metadata_file")" == true ]]; then
        set +e
        "$MAMBA" run -n "$ENV_NAME" python "$ROOT/validate_checkpoint.py" --task "$task" \
            2>&1 | tee -a "$LOGS/$task.log"
        validation_rc=${PIPESTATUS[0]}
        set -e
        if (( validation_rc == 0 )); then
            echo "TASK_SKIP_VALIDATED=$task"
            continue
        fi
        echo "TASK_PRIOR_DONE_INVALID=$task rc=$validation_rc; retraining"
        printf 'PENDING\n' >"$status_file.tmp"
        mv "$status_file.tmp" "$status_file"
    fi

    printf 'RUNNING\n' >"$status_file.tmp"
    mv "$status_file.tmp" "$status_file"
    echo "ATTEMPT_START=$(date --iso-8601=seconds) task=$task" | tee -a "$LOGS/$task.log"
    set +e
    "$MAMBA" run -n "$ENV_NAME" python "$ROOT/train_task.py" --task "$task" \
        2>&1 | tee -a "$LOGS/$task.log"
    train_rc=${PIPESTATUS[0]}
    set -e
    if (( train_rc != 0 )); then
        printf 'FAILED\n' >"$status_file.tmp"
        mv "$status_file.tmp" "$status_file"
        echo "TASK_FAILED=$task rc=$train_rc"
        exit "$train_rc"
    fi
    echo "ATTEMPT_END=$(date --iso-8601=seconds) task=$task" | tee -a "$LOGS/$task.log"

    set +e
    "$MAMBA" run -n "$ENV_NAME" python "$ROOT/validate_checkpoint.py" --task "$task" \
        2>&1 | tee -a "$LOGS/$task.log"
    validation_rc=${PIPESTATUS[0]}
    set -e
    if (( validation_rc != 0 )); then
        printf 'FAILED\n' >"$status_file.tmp"
        mv "$status_file.tmp" "$status_file"
        echo "TASK_VALIDATION_FAILED=$task rc=$validation_rc"
        exit "$validation_rc"
    fi
    printf 'DONE\n' >"$status_file.tmp"
    mv "$status_file.tmp" "$status_file"
    echo "TASK_DONE=$task"
done

"$MAMBA" run -n "$ENV_NAME" python "$ROOT/evaluate_single_tasks.py" 2>&1 | tee -a "$LOGS/single_task_eval.log"
dcmerge_start_epoch=$(date +%s)
"$MAMBA" run -n "$ENV_NAME" python "$ROOT/evaluate_dcmerge.py" 2>&1 | tee -a "$LOGS/dcmerge.log"
dcmerge_end_epoch=$(date +%s)
"$MAMBA" run -n "$ENV_NAME" python "$ROOT/make_comparison.py" 2>&1 | tee -a "$LOGS/comparison.log"

pipeline_end_epoch=$(date +%s)
{
    echo "completed_at=$(date --iso-8601=seconds)"
    echo "total_pipeline_seconds=$((pipeline_end_epoch - pipeline_start_epoch))"
    echo "dcmerge_stage_wall_seconds=$((dcmerge_end_epoch - dcmerge_start_epoch))"
} >"$STATE/COMPLETE.tmp"
mv "$STATE/COMPLETE.tmp" "$STATE/COMPLETE"
printf 'COMPLETE\n' >"$STATE/pipeline.status.tmp"
mv "$STATE/pipeline.status.tmp" "$STATE/pipeline.status"
trap - EXIT
echo "PIPELINE_COMPLETE=$(date --iso-8601=seconds)"
