"""Validation, hashing, and metric helpers for the task-scale ablation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Mapping, Sequence

from dcmerge_effective_weights import validate_named_task_scales


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
ALPHA_CANDIDATES = tuple(index / 10.0 for index in range(1, 31))
RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_hash(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_scale_configs(path: Path, expected_tasks: Sequence[str] = TASKS) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported task-scale config schema")
    task_order = payload.get("task_order")
    if task_order != list(expected_tasks):
        raise ValueError(
            f"task_order must exactly equal {list(expected_tasks)}, found {task_order}"
        )
    configs = payload.get("configs")
    if not isinstance(configs, list) or not configs:
        raise ValueError("configs must be a non-empty list")
    run_ids = []
    validated = []
    for entry in configs:
        if not isinstance(entry, dict) or set(entry) != {"run_id", "task_scales"}:
            raise ValueError("each config must contain exactly run_id and task_scales")
        run_id = entry["run_id"]
        if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError(f"invalid run_id: {run_id!r}")
        scales = validate_named_task_scales(expected_tasks, entry["task_scales"])
        run_ids.append(run_id)
        validated.append(
            {
                "run_id": run_id,
                "task_scales": scales,
                "ordered_scale_vector": [scales[task] for task in expected_tasks],
            }
        )
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("run_id values must be unique")
    return {
        "schema_version": 1,
        "task_order": list(expected_tasks),
        "configs": validated,
        "config_hash": config_hash(payload),
    }


def select_best_alpha(validation_results: Sequence[Mapping]) -> Mapping:
    if not validation_results:
        raise ValueError("validation results are empty")
    best = validation_results[0]
    for candidate in validation_results[1:]:
        if (
            candidate["average_normalized_accuracy_percent"]
            > best["average_normalized_accuracy_percent"]
        ):
            best = candidate
    return best


def summarize_split(
    merged: Mapping[str, float],
    single_task: Mapping[str, float],
    tasks: Sequence[str] = TASKS,
) -> dict:
    if set(merged) != set(tasks) or set(single_task) != set(tasks):
        raise ValueError("metric mappings must contain exactly the expected tasks")
    normalized = {}
    drops = {}
    for task in tasks:
        denominator = float(single_task[task])
        value = float(merged[task])
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise ValueError(f"invalid single-task denominator for {task}")
        if not math.isfinite(value):
            raise ValueError(f"invalid merged accuracy for {task}")
        normalized[task] = value / denominator * 100.0
        drops[task] = denominator - value
    return {
        "accuracy_percent": {task: float(merged[task]) for task in tasks},
        "normalized_accuracy_percent": normalized,
        "retention_percent": dict(normalized),
        "absolute_drop_pp": drops,
        "average_accuracy_percent": sum(merged[task] for task in tasks) / len(tasks),
        "average_normalized_accuracy_percent": sum(normalized.values()) / len(tasks),
        "average_retention_percent": sum(normalized.values()) / len(tasks),
        "worst_task_accuracy_percent": min(merged[task] for task in tasks),
        "worst_task_accuracy_task": min(tasks, key=lambda task: merged[task]),
        "worst_task_retention_percent": min(normalized.values()),
        "worst_task_retention_task": min(tasks, key=lambda task: normalized[task]),
    }


def task_metric_rows(
    run_id: str,
    scales: Mapping[str, float],
    effective_weights: Mapping[str, float],
    single_val: Mapping[str, float],
    merged_val: Mapping[str, float],
    selected_alpha: float,
    *,
    single_test: Mapping[str, float] | None = None,
    merged_test: Mapping[str, float] | None = None,
    tasks: Sequence[str] = TASKS,
) -> list[dict]:
    val = summarize_split(merged_val, single_val, tasks)
    test = None
    if single_test is not None or merged_test is not None:
        if single_test is None or merged_test is None:
            raise ValueError("single and merged test metrics must be provided together")
        test = summarize_split(merged_test, single_test, tasks)
    rows = []
    for task in tasks:
        row = {
            "run_id": run_id,
            "task": task,
            "task_scale": scales[task],
            "realized_effective_weight_pct": effective_weights[task],
            "single_task_val_accuracy": single_val[task],
            "merged_val_accuracy": merged_val[task],
            "val_retention_pct": val["retention_percent"][task],
            "val_drop_pp": val["absolute_drop_pp"][task],
            "selected_global_alpha": selected_alpha,
        }
        if test is not None:
            row.update(
                {
                    "single_task_test_accuracy": single_test[task],
                    "merged_test_accuracy": merged_test[task],
                    "test_retention_pct": test["retention_percent"][task],
                    "test_drop_pp": test["absolute_drop_pp"][task],
                }
            )
        rows.append(row)
    return rows
