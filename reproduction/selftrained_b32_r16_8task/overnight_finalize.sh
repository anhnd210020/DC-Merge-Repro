#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/workspace/selftrained_b32_r16_8task
REPO=/workspace/DC-Merge-Repro
MAMBA=/workspace/bin/micromamba
ENV_NAME=dcmerge_vision
ACTIVE_SESSION=dcmerge-selftrained
LOG="$ROOT/logs/overnight_finalize.log"
STATUS_FILE="$ROOT/state/pipeline.status"
BACKUP=/workspace/DCMerge_final_backup
ARCHIVE=/workspace/DCMerge_final_backup.tar.gz
CHECKSUM_FILE=/workspace/DCMerge_final_backup.tar.gz.sha256

mkdir -p "$ROOT/logs" "$ROOT/results"
if [[ ${OVERNIGHT_FINALIZER_LOGGED:-false} != true ]]; then
    export OVERNIGHT_FINALIZER_LOGGED=true
    exec >>"$LOG" 2>&1
fi

finalizer_complete_emitted=false
on_exit() {
    local rc=$?
    if [[ "$finalizer_complete_emitted" != true ]]; then
        echo "FINALIZER_COMPLETE=$(date --iso-8601=seconds) rc=$rc"
    fi
}
trap on_exit EXIT

echo "FINALIZER_START=$(date --iso-8601=seconds)"
echo "WAITING_FOR_DCMERGE=true"
while tmux has-session -t "$ACTIVE_SESSION" 2>/dev/null; do
    sleep 60
done

echo "DCMERGE_SESSION_ENDED=$(date --iso-8601=seconds)"
sleep 30

if [[ -f "$STATUS_FILE" ]]; then
    pipeline_status=$(tr -d '\r\n' <"$STATUS_FILE")
else
    pipeline_status=NOT_FOUND
fi
echo "PIPELINE_STATUS=$pipeline_status"

source /workspace/dcmerge_env.sh
export DCMERGE_SELFTRAIN_ROOT="$ROOT"
export DCMERGE_HEAD_DIR="$ROOT/heads"
export DCMERGE_FT_DIR="$ROOT/checkpoints"
export PYTHONPATH="$ROOT:$REPO/vision_lora_merge${PYTHONPATH:+:$PYTHONPATH}"

report_mode=failure
if [[ "$pipeline_status" == COMPLETE ]]; then
    report_mode=success
fi

report_rc=0
"$MAMBA" run -n "$ENV_NAME" python - "$report_mode" "$pipeline_status" <<'PY' || report_rc=$?
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODE = sys.argv[1]
PIPELINE_STATUS = sys.argv[2]
ROOT = Path("/workspace/selftrained_b32_r16_8task")
REPO = Path("/workspace/DC-Merge-Repro")
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
TASKS = (
    "stanford_cars",
    "dtd",
    "eurosat",
    "gtsrb",
    "mnist",
    "resisc45",
    "sun397",
    "svhn",
)
NOT_FOUND = "NOT FOUND"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command(args: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            args, cwd=cwd, text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception:
        return NOT_FOUND


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except Exception:
        return ""


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def found(value: Any) -> Any:
    return NOT_FOUND if value is None else value


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def average(values: Any) -> Any:
    if not isinstance(values, dict):
        return NOT_FOUND
    selected = [values.get(task) for task in TASKS]
    if not all(finite_number(value) for value in selected):
        return NOT_FOUND
    return sum(selected) / len(selected)


def nested(mapping: Any, *keys: str) -> Any:
    value = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return NOT_FOUND
        value = value[key]
    return value


def number_from(pattern: str, text: str, flags: int = 0) -> Any:
    match = re.search(pattern, text, flags)
    return float(match.group(1)) if match else NOT_FOUND


def markdown_value(value: Any) -> str:
    if finite_number(value):
        return f"{value:.9f}".rstrip("0").rstrip(".")
    return str(value)


git_commit = command(["git", "rev-parse", "HEAD"], REPO)
git_status = command(["git", "status", "--short"], REPO)
if git_status == "":
    git_status = "CLEAN"

task_statuses = {}
for task in TASKS:
    status_path = ROOT / "state" / f"{task}.status"
    task_statuses[task] = read_text(status_path).strip() or NOT_FOUND

val_path = RESULTS / "selftrained_val_acc.json"
test_path = RESULTS / "selftrained_test_acc.json"
dcmerge_path = RESULTS / "selftrained_dcmerge_results.json"
comparison_path = RESULTS / "reproduction_comparison.md"
val = load_json(val_path)
test = load_json(test_path)
dcmerge = load_json(dcmerge_path)
val = val if isinstance(val, dict) else {}
test = test if isinstance(test, dict) else {}
dcmerge = dcmerge if isinstance(dcmerge, dict) else {}

single_val = {task: found(val.get(task)) for task in TASKS}
single_test = {task: found(test.get(task)) for task in TASKS}
val_average = average(val)
test_average = average(test)

dcmerge_best_alpha = found(dcmerge.get("selected_alpha"))
dcmerge_validation_normalized = found(
    dcmerge.get("best_validation_average_normalized_accuracy_percent")
)
dcmerge_final_average = nested(dcmerge, "test", "average_accuracy_percent")
dcmerge_final_normalized = nested(
    dcmerge, "test", "average_normalized_accuracy_percent"
)
dcmerge_absolute = nested(dcmerge, "test", "accuracy_percent")
dcmerge_normalized = nested(dcmerge, "test", "normalized_accuracy_percent")
if not isinstance(dcmerge_absolute, dict):
    dcmerge_absolute = {}
if not isinstance(dcmerge_normalized, dict):
    dcmerge_normalized = {}
dcmerge_per_task = {
    task: {
        "accuracy_percent": found(dcmerge_absolute.get(task)),
        "normalized_accuracy_percent": found(dcmerge_normalized.get(task)),
    }
    for task in TASKS
}

baseline_log = Path("/workspace/dcmerge_b32_8task.log")
baseline_text = read_text(baseline_log).replace("\r", "\n")
final_baseline = baseline_text.rsplit("Evaluating Merged Model on test set", 1)[-1]
alpha_matches = re.findall(r"Best Alpha: ([0-9.eE+-]+)", baseline_text)
official = {
    "best_alpha": float(alpha_matches[-1]) if alpha_matches else NOT_FOUND,
    "average_accuracy_percent": number_from(
        r"Average Accuracy is ([0-9.]+)", final_baseline
    ),
    "average_normalized_accuracy_percent": number_from(
        r"Average Normalized Accuracy is ([0-9.]+)", final_baseline
    ),
    "source": str(baseline_log) if baseline_log.is_file() else NOT_FOUND,
}

comparison_text = read_text(comparison_path)
paper = {
    "average_accuracy_percent": number_from(
        r"\| DC-Merge absolute average \|\s*([0-9.]+)", comparison_text
    ),
    "average_normalized_accuracy_percent": number_from(
        r"\| DC-Merge normalized average \|\s*([0-9.]+)", comparison_text
    ),
    "source": str(comparison_path) if comparison_path.is_file() else NOT_FOUND,
}


def difference(left: Any, right: Any) -> Any:
    if finite_number(left) and finite_number(right):
        return left - right
    return NOT_FOUND


differences = {
    "selftrained_minus_official_reproduction": {
        "average_accuracy_percentage_points": difference(
            dcmerge_final_average, official["average_accuracy_percent"]
        ),
        "average_normalized_accuracy_percentage_points": difference(
            dcmerge_final_normalized,
            official["average_normalized_accuracy_percent"],
        ),
    },
    "selftrained_minus_paper_table1": {
        "average_accuracy_percentage_points": difference(
            dcmerge_final_average, paper["average_accuracy_percent"]
        ),
        "average_normalized_accuracy_percentage_points": difference(
            dcmerge_final_normalized,
            paper["average_normalized_accuracy_percent"],
        ),
    },
}

smoke_jsons = sorted(
    {
        *RESULTS.glob("*smoke*.json"),
        *RESULTS.glob("*preflight*.json"),
        *RESULTS.glob("*serializer*.json"),
    }
)
root_python = sorted(ROOT.glob("*.py"))
root_shell = sorted(ROOT.glob("*.sh"))
checkpoint_checks = {
    task: all(
        (ROOT / "checkpoints" / task / name).is_file()
        and (ROOT / "checkpoints" / task / name).stat().st_size > 0
        for name in ("adapter_config.json", "adapter_model.bin")
    )
    for task in TASKS
}
head_checks = {
    task: (
        ROOT / "heads" / "ViT-B-32" / f"{task}_head.pt"
    ).is_file()
    for task in TASKS
}
artifact_checks = {
    "pipeline_status_complete": PIPELINE_STATUS == "COMPLETE",
    "all_task_statuses_done": all(value == "DONE" for value in task_statuses.values()),
    "selftrained_val_json": val_path.is_file(),
    "selftrained_test_json": test_path.is_file(),
    "dcmerge_result_json": dcmerge_path.is_file(),
    "reproduction_comparison_markdown": comparison_path.is_file(),
    "smoke_config_jsons": bool(smoke_jsons),
    "final_dcmerge_log": (LOGS / "dcmerge_selftrained.log").is_file(),
    "master_training_log": (LOGS / "master.log").is_file(),
    "all_checkpoints": all(checkpoint_checks.values()),
    "all_heads": all(head_checks.values()),
    "python_scripts": bool(root_python),
    "shell_scripts": bool(root_shell),
    "selftrained_config": (
        REPO / "vision_lora_merge/configs/vitB32_r16_8task_selftrained.py"
    ).is_file(),
}

artifact_paths = {
    "single_task_val": str(val_path),
    "single_task_test": str(test_path),
    "dcmerge_results": str(dcmerge_path),
    "comparison": str(comparison_path),
    "smoke_config_jsons": [str(path) for path in smoke_jsons],
    "final_dcmerge_log": str(LOGS / "dcmerge_selftrained.log"),
    "master_training_log": str(LOGS / "master.log"),
    "checkpoints": str(ROOT / "checkpoints"),
    "heads": str(ROOT / "heads"),
    "scripts": [str(path) for path in root_python + root_shell],
    "selftrained_config": str(
        REPO / "vision_lora_merge/configs/vitB32_r16_8task_selftrained.py"
    ),
}

warning_pattern = re.compile(
    r"traceback|\berror\b|\bfailed\b|exception|warning", re.IGNORECASE
)
warnings = []
for log_path in sorted(LOGS.glob("*.log")):
    for line_number, line in enumerate(
        read_text(log_path).replace("\r", "\n").splitlines(), start=1
    ):
        if warning_pattern.search(line):
            warnings.append(f"{log_path}:{line_number}: {line[:500]}")
            if len(warnings) == 200:
                warnings.append("WARNING SCAN TRUNCATED AFTER 200 MATCHES")
                break
    if len(warnings) > 200:
        break

summary = {
    "generated_at": now(),
    "pipeline_status": PIPELINE_STATUS,
    "git_commit": git_commit,
    "git_status": git_status,
    "task_statuses": task_statuses,
    "single_task_val": single_val,
    "single_task_test": single_test,
    "single_task_val_average": val_average,
    "single_task_test_average": test_average,
    "dcmerge_best_alpha": dcmerge_best_alpha,
    "dcmerge_validation_normalized_average": dcmerge_validation_normalized,
    "dcmerge_final_average_accuracy": dcmerge_final_average,
    "dcmerge_final_average_normalized_accuracy": dcmerge_final_normalized,
    "dcmerge_per_task": dcmerge_per_task,
    "official_reproduction": official,
    "paper_target": paper,
    "differences": differences,
    "artifact_paths": artifact_paths,
    "artifact_checks": artifact_checks,
    "all_expected_artifacts_exist": all(artifact_checks.values()),
    "warnings": warnings,
}

if MODE == "success":
    (RESULTS / "final_handoff.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Final DC-Merge self-trained handoff",
        "",
        f"- Timestamp: {summary['generated_at']}",
        f"- Git commit: {git_commit}",
        f"- Pipeline status: {PIPELINE_STATUS}",
        f"- All expected artifacts exist: {summary['all_expected_artifacts_exist']}",
        "",
        "## Task statuses and single-task accuracy",
        "",
        "| Task | Status | Validation (%) | Test (%) |",
        "|---|---|---:|---:|",
    ]
    for task in TASKS:
        lines.append(
            f"| {task} | {task_statuses[task]} | "
            f"{markdown_value(single_val[task])} | {markdown_value(single_test[task])} |"
        )
    lines.extend(
        [
            "",
            f"- Average validation accuracy: {markdown_value(val_average)}%",
            f"- Average test accuracy: {markdown_value(test_average)}%",
            "",
            "## Self-trained DC-Merge",
            "",
            f"- Best alpha: {markdown_value(dcmerge_best_alpha)}",
            f"- Best validation normalized average: {markdown_value(dcmerge_validation_normalized)}%",
            f"- Final average absolute accuracy: {markdown_value(dcmerge_final_average)}%",
            f"- Final average normalized accuracy: {markdown_value(dcmerge_final_normalized)}%",
            "",
            "| Task | Final absolute (%) | Final normalized (%) |",
            "|---|---:|---:|",
        ]
    )
    for task in TASKS:
        lines.append(
            f"| {task} | {markdown_value(dcmerge_per_task[task]['accuracy_percent'])} | "
            f"{markdown_value(dcmerge_per_task[task]['normalized_accuracy_percent'])} |"
        )
    lines.extend(
        [
            "",
            "## References and differences",
            "",
            f"- Official reproduction best alpha: {markdown_value(official['best_alpha'])}",
            f"- Official reproduction average accuracy: {markdown_value(official['average_accuracy_percent'])}%",
            f"- Official reproduction average normalized accuracy: {markdown_value(official['average_normalized_accuracy_percent'])}%",
            f"- Paper Table-1 average accuracy: {markdown_value(paper['average_accuracy_percent'])}%",
            f"- Paper Table-1 average normalized accuracy: {markdown_value(paper['average_normalized_accuracy_percent'])}%",
            f"- Self-trained minus official, absolute: {markdown_value(differences['selftrained_minus_official_reproduction']['average_accuracy_percentage_points'])} pp",
            f"- Self-trained minus official, normalized: {markdown_value(differences['selftrained_minus_official_reproduction']['average_normalized_accuracy_percentage_points'])} pp",
            f"- Self-trained minus paper, absolute: {markdown_value(differences['selftrained_minus_paper_table1']['average_accuracy_percentage_points'])} pp",
            f"- Self-trained minus paper, normalized: {markdown_value(differences['selftrained_minus_paper_table1']['average_normalized_accuracy_percentage_points'])} pp",
            "",
            "## Artifact checks",
            "",
        ]
    )
    lines.extend(f"- {key}: {value}" for key, value in artifact_checks.items())
    lines.extend(["", "## Final artifact paths", ""])
    for key, value in artifact_paths.items():
        if isinstance(value, list):
            lines.append(f"- {key}:")
            lines.extend(f"  - `{item}`" for item in value)
        else:
            lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Log warnings/errors", ""])
    lines.extend(f"- {item}" for item in warnings) if warnings else lines.append("- None found")
    lines.extend(["", "## Git status", "", "```text", git_status, "```", ""])
    (RESULTS / "FINAL_HANDOFF.md").write_text("\n".join(lines))
else:
    active_log = LOGS / "dcmerge_selftrained.log"
    log_lines = read_text(active_log).replace("\r", "\n").splitlines()
    last_200 = log_lines[-200:]
    traceback_lines: list[str] = []
    traceback_starts = [
        index for index, line in enumerate(log_lines) if "Traceback (most recent call last):" in line
    ]
    if traceback_starts:
        traceback_lines = log_lines[traceback_starts[-1] : traceback_starts[-1] + 200]

    stage_markers = re.compile(
        r"(ALL_DONE_CHECKPOINTS_PRESENT|SINGLE_TASK_EVALUATION_SKIPPED|"
        r"SELFTRAINED_VAL_TEST_JSON_PASS|CONFIG_SMOKE_TEST_PASS|DCMERGE_READY|"
        r"SELECTED_ALPHA|POST_TRAINING_PIPELINE_COMPLETE)"
    )
    successful_markers = [line for line in log_lines if stage_markers.search(line)]
    last_successful_stage = successful_markers[-1] if successful_markers else NOT_FOUND
    suspected_failure = NOT_FOUND
    if traceback_lines:
        candidates = [line.strip() for line in traceback_lines if line.strip()]
        suspected_failure = candidates[-1] if candidates else NOT_FOUND
    elif PIPELINE_STATUS != NOT_FOUND:
        suspected_failure = f"pipeline.status ended at {PIPELINE_STATUS}"

    current_results = sorted(str(path) for path in RESULTS.iterdir()) if RESULTS.is_dir() else []
    intact = [key for key, value in artifact_checks.items() if value]
    failure_lines = [
        "# Final DC-Merge failure handoff",
        "",
        f"- Timestamp: {now()}",
        f"- Pipeline status: {PIPELINE_STATUS}",
        f"- Git commit: {git_commit}",
        f"- Last successful stage: {last_successful_stage}",
        f"- Suspected failure location: {suspected_failure}",
        "",
        "## Task statuses",
        "",
    ]
    failure_lines.extend(f"- {task}: {task_statuses[task]}" for task in TASKS)
    failure_lines.extend(["", "## Current result files", ""])
    failure_lines.extend(f"- `{path}`" for path in current_results)
    failure_lines.extend(["", "## Artifacts confirmed intact", ""])
    failure_lines.extend(f"- {item}" for item in intact)
    failure_lines.extend(["", "## Git status", "", "```text", git_status, "```", ""])
    failure_lines.extend(["## Python traceback", "", "```text"])
    failure_lines.extend(traceback_lines or [NOT_FOUND])
    failure_lines.extend(["```", "", "## Last 200 lines of dcmerge_selftrained.log", "", "```text"])
    failure_lines.extend(last_200 or [NOT_FOUND])
    failure_lines.extend(["```", ""])
    (RESULTS / "FINAL_FAILURE_HANDOFF.md").write_text("\n".join(failure_lines))
PY

if [[ $report_rc -eq 0 ]]; then
    echo "FINAL_HANDOFF_CREATED=true"
else
    echo "FINAL_HANDOFF_CREATED=false"
    echo "FINAL_HANDOFF_ERROR_RC=$report_rc"
fi

backup_created=false
backup_sha256=NOT_CREATED
if [[ -e "$BACKUP" || -e "$ARCHIVE" || -e "$CHECKSUM_FILE" ]]; then
    echo "BACKUP_REFUSED=one_or_more_exact_backup_targets_already_exist"
else
    mkdir "$BACKUP"
    for directory in results state logs checkpoints heads; do
        if [[ -d "$ROOT/$directory" ]]; then
            cp -a "$ROOT/$directory" "$BACKUP/"
        fi
    done

    find "$ROOT" -maxdepth 1 -type f \( -name '*.py' -o -name '*.sh' \) \
        -exec cp -a -t "$BACKUP" {} +
    if [[ -f "$ROOT/REPRODUCTION_RECIPE.md" ]]; then
        cp -a "$ROOT/REPRODUCTION_RECIPE.md" "$BACKUP/"
    fi
    mkdir "$BACKUP/config"
    cp -a "$REPO/vision_lora_merge/configs/vitB32_r16_8task_selftrained.py" \
        "$BACKUP/config/"

    git -C "$REPO" rev-parse HEAD >"$BACKUP/repo_git_commit.txt" 2>&1 || true
    git -C "$REPO" status --short >"$BACKUP/repo_git_status.txt" 2>&1 || true
    git -C "$REPO" diff >"$BACKUP/repo_git_diff.patch" 2>&1 || true
    "$MAMBA" run -n "$ENV_NAME" python --version \
        >"$BACKUP/environment_python_version.txt" 2>&1 || true
    "$MAMBA" run -n "$ENV_NAME" python -m pip freeze \
        >"$BACKUP/environment_pip_freeze.txt" 2>&1 || true
    nvidia-smi >"$BACKUP/nvidia_smi.txt" 2>&1 || true
    {
        echo "DISK_USAGE_CAPTURED_AT=$(date --iso-8601=seconds)"
        df -h /workspace
        du -sh "$ROOT"
        du -sh "$ROOT"/results "$ROOT"/state "$ROOT"/logs \
            "$ROOT"/checkpoints "$ROOT"/heads
    } >"$BACKUP/disk_usage.txt" 2>&1 || true

    # Refresh the finalizer log copy immediately before archiving it.
    cp -a "$LOG" "$BACKUP/logs/overnight_finalize.log"
    tar -czf "$ARCHIVE" -C /workspace DCMerge_final_backup
    backup_sha256=$(sha256sum "$ARCHIVE" | tee "$CHECKSUM_FILE" | awk '{print $1}')
    backup_created=true
    echo "BACKUP_ARCHIVE_SIZE=$(du -h "$ARCHIVE" | awk '{print $1}')"
fi

echo "BACKUP_CREATED=$backup_created"
echo "BACKUP_SHA256=$backup_sha256"
finalizer_complete_emitted=true
echo "FINALIZER_COMPLETE=$(date --iso-8601=seconds)"
