"""Deterministic synthetic evaluator used only for workflow dry runs."""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np

from .design import TASKS


class SimulationEvaluator:
    mode = "simulation"

    def __call__(self, configuration: Mapping[str, Any], split: str) -> dict[str, Any]:
        per_task = {}
        for task in configuration["tasks"]:
            digest = hashlib.sha256(f"{configuration['run_id']}|{task}|{split}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            count = 64
            labels = np.arange(count, dtype=np.int32) % 4
            baseline = 52 + TASKS.index(task) % 5
            if configuration["variant"] == "rank":
                delta = (16 - int(configuration["level"])) * 0.35
            elif configuration["variant"] == "density":
                delta = (1.0 - float(configuration["level"])) * 3.0
            else:
                delta = 0.0
            focal = task == configuration.get("focal_task")
            accuracy = max(10, min(90, baseline - delta if focal else baseline + 0.1 * delta))
            correct_count = round(count * accuracy / 100)
            predictions = labels.copy()
            wrong = np.arange(correct_count, count)
            predictions[wrong] = (predictions[wrong] + 1) % 4
            rng.shuffle(predictions)
            # Use label permutation with an exact correct-count mask after the shuffle.
            predictions = labels.copy()
            wrong_indices = rng.permutation(count)[correct_count:]
            predictions[wrong_indices] = (predictions[wrong_indices] + 1) % 4
            per_task[task] = {
                "accuracy_percent": float(100 * np.mean(predictions == labels)),
                "labels": labels,
                "predictions": predictions,
                "examples": count,
            }
        density = {
            task: {"nominal_density": float(configuration["density_by_task"][task]),
                   "baseline_mask_count": 1000, "target_mask_count": round(1000 * configuration["density_by_task"][task]),
                   "retained_mask_count": round(1000 * configuration["density_by_task"][task]),
                   "retained_nonzero_count": round(1000 * configuration["density_by_task"][task]),
                   "total_coordinates": 1_000_000,
                   "actual_mask_fraction": round(1000 * configuration["density_by_task"][task]) / 1_000_000,
                   "actual_nonzero_fraction": round(1000 * configuration["density_by_task"][task]) / 1_000_000,
                   "actual_baseline_support_fraction": configuration["density_by_task"][task]}
            for task in configuration["tasks"]
        }
        return {"per_task": per_task, "density_by_task": density,
                "resources": {"device": "synthetic", "evaluated_tasks": list(configuration["tasks"]),
                              "simulation": True}}
