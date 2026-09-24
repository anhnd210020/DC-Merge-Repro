#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STAGE="${1:?usage: $0 validation|final-test [extra runner arguments]}"
shift

case "${STAGE}" in
  validation|final-test) ;;
  *) echo "invalid stage: ${STAGE}" >&2; exit 2 ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python}"
ARTIFACT_ROOT="${DCMERGE_SELFTRAIN_ROOT:-/workspace/selftrained_b32_r16_8task}"
MODEL_DIR="${DCMERGE_MODEL_DIR:-/workspace/models/clip-vit-base-patch32}"
DATA_DIR="${DCMERGE_DATA_DIR:-/workspace/datasets8}"
OUTPUT_DIR="${DCMERGE_TASK_SCALE_OUTPUT:-/workspace/analysis_outputs/task_scale_ablation}"
SCALE_CONFIG="${DCMERGE_TASK_SCALE_CONFIG:-${SCRIPT_DIR}/task_scale_configs_primary_calibrated.json}"
SELECTION_MANIFEST="${DCMERGE_SELECTION_MANIFEST:-${OUTPUT_DIR}/selected_for_test.json}"
LOG_DIR="${OUTPUT_DIR}/logs"
STATUS_DIR="${OUTPUT_DIR}/status"
mkdir -p "${LOG_DIR}" "${STATUS_DIR}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="${LOG_DIR}/${STAGE}-${STAMP}.log"
DONE_PATH="${STATUS_DIR}/${STAGE}.DONE"
FAILED_PATH="${STATUS_DIR}/${STAGE}.FAILED"
rm -f -- "${FAILED_PATH}"

on_error() {
  status=$?
  printf 'stage=%s\nfailed_at=%s\nexit_status=%s\nlog=%s\n' \
    "${STAGE}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${status}" "${LOG_PATH}" \
    > "${FAILED_PATH}"
  exit "${status}"
}
trap on_error ERR

COMMON_ARGS=(
  --config vitB32_r16_8task_selftrained
  --scale-config "${SCALE_CONFIG}"
  --stage "${STAGE}"
  --output-dir "${OUTPUT_DIR}"
  --artifact-root "${ARTIFACT_ROOT}"
  --model-dir "${MODEL_DIR}"
  --data-dir "${DATA_DIR}"
  --repo-root "${SCRIPT_DIR}"
  --device cuda
)

if [[ "${STAGE}" == "final-test" ]]; then
  COMMON_ARGS+=(--selection-manifest "${SELECTION_MANIFEST}")
fi

echo "Preflight: ${STAGE}" | tee -a "${LOG_PATH}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_task_scale_ablation.py" \
  "${COMMON_ARGS[@]}" --preflight-only "$@" 2>&1 | tee -a "${LOG_PATH}"

echo "Run: ${STAGE}" | tee -a "${LOG_PATH}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_task_scale_ablation.py" \
  "${COMMON_ARGS[@]}" --resume "$@" 2>&1 | tee -a "${LOG_PATH}"

printf 'stage=%s\ncompleted_at=%s\nlog=%s\n' \
  "${STAGE}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${LOG_PATH}" > "${DONE_PATH}"
rm -f -- "${FAILED_PATH}"
echo "DONE ${STAGE}; log=${LOG_PATH}"
