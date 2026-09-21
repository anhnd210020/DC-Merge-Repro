#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/workspace/selftrained_b32_r16_8task
REPO=/workspace/DC-Merge-Repro
MAMBA=/workspace/bin/micromamba
ENV_NAME=dcmerge_vision
STATE="$ROOT/state"
LOGS="$ROOT/logs"
TASKS=(stanford_cars dtd eurosat gtsrb mnist resisc45 sun397 svhn)
VALIDATE_ONLY=false

if [[ $# -gt 1 ]] || [[ $# -eq 1 && "${1:-}" != "--validate-only" ]]; then
    echo "usage: $0 [--validate-only]" >&2
    exit 2
fi
if [[ ${1:-} == "--validate-only" ]]; then
    VALIDATE_ONLY=true
fi

mkdir -p "$LOGS" "$ROOT/results" "$STATE"
exec 9>"$STATE/selftrain.lock"
if ! flock -n 9; then
    echo "POST_TRAINING_RESUME_ALREADY_LOCKED timestamp=$(date --iso-8601=seconds)"
    exit 1
fi
if [[ -f "$STATE/COMPLETE" ]]; then
    echo "PIPELINE_ALREADY_COMPLETE timestamp=$(date --iso-8601=seconds)"
    exit 0
fi

source /workspace/dcmerge_env.sh
export DCMERGE_SELFTRAIN_ROOT="$ROOT"
export DCMERGE_HEAD_DIR="$ROOT/heads"
export DCMERGE_FT_DIR="$ROOT/checkpoints"
export PYTHONPATH="$ROOT:$REPO/vision_lora_merge${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO/vision_lora_merge"

pipeline_start_epoch=$(date +%s)
printf 'VALIDATING_RESUME_INPUTS\n' >"$STATE/pipeline.status.tmp"
mv "$STATE/pipeline.status.tmp" "$STATE/pipeline.status"

on_exit() {
    rc=$?
    if [[ $rc -ne 0 ]]; then
        printf 'FAILED\n' >"$STATE/pipeline.status.tmp"
        mv "$STATE/pipeline.status.tmp" "$STATE/pipeline.status"
        echo "POST_TRAINING_PIPELINE_FAILED rc=$rc timestamp=$(date --iso-8601=seconds)"
    fi
}
trap on_exit EXIT

echo "POST_TRAINING_RESUME_START=$(date --iso-8601=seconds)"
echo "TRAINING_DISABLED=true"

# This entry point has no training command or fallback. Require all previously
# completed task states and adapter files without modifying any of them.
for task in "${TASKS[@]}"; do
    status_file="$STATE/$task.status"
    if [[ ! -f "$status_file" ]] || [[ "$(<"$status_file")" != DONE ]]; then
        echo "CHECKPOINT_GATE_ABORT task=$task reason=status_not_DONE"
        exit 1
    fi
    for adapter_file in adapter_config.json adapter_model.bin; do
        if [[ ! -s "$ROOT/checkpoints/$task/$adapter_file" ]]; then
            echo "CHECKPOINT_GATE_ABORT task=$task reason=missing_$adapter_file"
            exit 1
        fi
    done
done
echo "ALL_DONE_CHECKPOINTS_PRESENT=true"

expected_tasks_json='["stanford_cars","dtd","eurosat","gtsrb","mnist","resisc45","sun397","svhn"]'
single_task_results_valid() {
    local result_file result_path
    for result_file in selftrained_val_acc.json selftrained_test_acc.json; do
        result_path="$ROOT/results/$result_file"
        [[ -s "$result_path" ]] || return 1
        jq -e --argjson expected "$expected_tasks_json" \
            'type == "object"
             and ((keys | sort) == ($expected | sort))
             and all(.[]; type == "number")' \
            "$result_path" >/dev/null || return 1
    done
}

if single_task_results_valid; then
    echo "SINGLE_TASK_EVALUATION_SKIPPED=valid_existing_val_test_json"
else
    printf 'VALIDATING_CHECKPOINTS\n' >"$STATE/pipeline.status.tmp"
    mv "$STATE/pipeline.status.tmp" "$STATE/pipeline.status"
    for task in "${TASKS[@]}"; do
        "$MAMBA" run -n "$ENV_NAME" python "$ROOT/validate_checkpoint.py" --task "$task" \
            2>&1 | tee -a "$LOGS/post_training_checkpoint_validation.log"
    done
    echo "ALL_DONE_CHECKPOINTS_VALIDATED=true"

    "$MAMBA" run -n "$ENV_NAME" python "$ROOT/compare_heads.py" \
        2>&1 | tee -a "$LOGS/post_training_checkpoint_validation.log"

    printf 'RUNNING_SINGLE_TASK_EVALUATION\n' >"$STATE/pipeline.status.tmp"
    mv "$STATE/pipeline.status.tmp" "$STATE/pipeline.status"
    "$MAMBA" run -n "$ENV_NAME" python "$ROOT/evaluate_single_tasks.py" \
        2>&1 | tee -a "$LOGS/single_task_eval.log"

    if ! single_task_results_valid; then
        echo "SINGLE_TASK_RESULTS_ABORT invalid_or_incomplete_results"
        exit 1
    fi
fi
echo "SELFTRAINED_VAL_TEST_JSON_PASS=true"

printf 'VALIDATING_DCMERGE_CONFIG\n' >"$STATE/pipeline.status.tmp"
mv "$STATE/pipeline.status.tmp" "$STATE/pipeline.status"
"$MAMBA" run -n "$ENV_NAME" python "$ROOT/evaluate_dcmerge.py" --smoke-config \
    2>&1 | tee -a "$LOGS/dcmerge_config_smoke.log"

printf 'DCMERGE_READY\n' >"$STATE/pipeline.status.tmp"
mv "$STATE/pipeline.status.tmp" "$STATE/pipeline.status"
echo "DCMERGE_READY=true"
if [[ "$VALIDATE_ONLY" == true ]]; then
    trap - EXIT
    echo "VALIDATION_ONLY_COMPLETE=$(date --iso-8601=seconds)"
    exit 0
fi

printf 'RUNNING_DCMERGE\n' >"$STATE/pipeline.status.tmp"
mv "$STATE/pipeline.status.tmp" "$STATE/pipeline.status"
dcmerge_start_epoch=$(date +%s)
"$MAMBA" run -n "$ENV_NAME" python "$ROOT/evaluate_dcmerge.py" \
    2>&1 | tee -a "$LOGS/dcmerge.log"
dcmerge_end_epoch=$(date +%s)

printf 'GENERATING_COMPARISON\n' >"$STATE/pipeline.status.tmp"
mv "$STATE/pipeline.status.tmp" "$STATE/pipeline.status"
"$MAMBA" run -n "$ENV_NAME" python "$ROOT/make_comparison.py" \
    2>&1 | tee -a "$LOGS/comparison.log"

pipeline_end_epoch=$(date +%s)
{
    echo "completed_at=$(date --iso-8601=seconds)"
    echo "total_post_training_seconds=$((pipeline_end_epoch - pipeline_start_epoch))"
    echo "dcmerge_stage_wall_seconds=$((dcmerge_end_epoch - dcmerge_start_epoch))"
} >"$STATE/COMPLETE.tmp"
mv "$STATE/COMPLETE.tmp" "$STATE/COMPLETE"
printf 'COMPLETE\n' >"$STATE/pipeline.status.tmp"
mv "$STATE/pipeline.status.tmp" "$STATE/pipeline.status"
trap - EXIT
echo "POST_TRAINING_PIPELINE_COMPLETE=$(date --iso-8601=seconds)"
