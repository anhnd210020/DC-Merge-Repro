#!/usr/bin/env python3
"""Deterministically calibrate the primary pair-total task-scale series.

This program loads only LoRA adapter tensors.  It never loads CLIP, datasets,
heads, validation accuracy, or test accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from dcmerge_effective_weights import (
    EFFECTIVE_WEIGHT_DEFINITION_VERSION,
    compute_effective_weights,
    effective_weights_from_projection_cache,
    prepare_projection_cache,
)
from task_scale_ablation_common import TASKS


PAIR_TASKS = ("eurosat", "sun397")
PAIR_PENALTY = 4.0
LOG_SCALE_BOUNDS = (math.log(0.25), math.log(4.0))
COARSE_GRID_POINTS_PER_AXIS = 9


def full_scales(eurosat: float, sun397: float) -> dict[str, float]:
    scales = {task: 1.0 for task in TASKS}
    scales["eurosat"] = float(eurosat)
    scales["sun397"] = float(sun397)
    return scales


def residuals(cache, log_scales, target, pair_total):
    eurosat, sun397 = np.exp(np.asarray(log_scales, dtype=float))
    weights = effective_weights_from_projection_cache(
        cache, full_scales(eurosat, sun397)
    )
    pair_error = weights["eurosat"] + weights["sun397"] - pair_total
    return np.asarray(
        [
            weights["eurosat"] - target["eurosat"],
            weights["sun397"] - target["sun397"],
            math.sqrt(PAIR_PENALTY) * pair_error,
        ],
        dtype=float,
    )


def calibrate_target(cache, target, pair_total):
    """Deterministic log-space grid search followed by bounded least squares."""
    grid = np.linspace(
        LOG_SCALE_BOUNDS[0], LOG_SCALE_BOUNDS[1], COARSE_GRID_POINTS_PER_AXIS
    )
    candidates = []
    for log_eurosat in grid:
        for log_sun397 in grid:
            value = residuals(
                cache, (log_eurosat, log_sun397), target, pair_total
            )
            candidates.append(
                (float(value @ value), float(log_eurosat), float(log_sun397))
            )
    _, initial_eurosat, initial_sun397 = min(candidates)
    result = least_squares(
        lambda values: residuals(cache, values, target, pair_total),
        x0=np.asarray([initial_eurosat, initial_sun397]),
        bounds=LOG_SCALE_BOUNDS,
        ftol=1e-13,
        xtol=1e-13,
        gtol=1e-13,
        max_nfev=200,
        method="trf",
    )
    if not result.success:
        raise RuntimeError(f"calibration did not converge: {result.message}")
    eurosat, sun397 = np.exp(result.x)
    return full_scales(eurosat, sun397), {
        "coarse_grid_best_log_scales": [initial_eurosat, initial_sun397],
        "least_squares_nfev": int(result.nfev),
        "least_squares_status": int(result.status),
    }


def calibration_record(target, scales, achieved, pair_total, search):
    pair_achieved = achieved["eurosat"] + achieved["sun397"]
    errors = {
        task: achieved[task] - target[task] for task in PAIR_TASKS
    }
    objective = (
        errors["eurosat"] ** 2
        + errors["sun397"] ** 2
        + PAIR_PENALTY * (pair_achieved - pair_total) ** 2
    )
    return {
        "target_effective_weight_percent": target,
        "task_scales": scales,
        "achieved_effective_weight_percent": {
            task: achieved[task] for task in PAIR_TASKS
        },
        "achieved_pair_total_percent": pair_achieved,
        "pair_total_error_percent": pair_achieved - pair_total,
        "per_task_target_error_percent": errors,
        "target_rmse_percent": math.sqrt(
            (errors["eurosat"] ** 2 + errors["sun397"] ** 2) / 2.0
        ),
        "calibration_objective": objective,
        "search_diagnostics": search,
    }


def build_config(artifact_root: Path) -> dict:
    checkpoint_dirs = {
        task: artifact_root / "checkpoints" / task for task in TASKS
    }
    cache = prepare_projection_cache(checkpoint_dirs, TASKS, progress=True)
    baseline_scales = full_scales(1.0, 1.0)
    baseline = compute_effective_weights(
        checkpoint_dirs, TASKS, baseline_scales
    )["effective_block_weight_percent"]
    pair_total = baseline["eurosat"] + baseline["sun397"]
    targets = [
        (
            "baseline",
            "Baseline allocation",
            {"eurosat": baseline["eurosat"], "sun397": baseline["sun397"]},
        ),
        (
            "controlled_moderate",
            "Moderate pair-total-preserving redistribution",
            {"eurosat": 10.0, "sun397": pair_total - 10.0},
        ),
        (
            "controlled_balanced",
            "Balanced pair-total-preserving redistribution",
            {"eurosat": pair_total / 2.0, "sun397": pair_total / 2.0},
        ),
        (
            "controlled_reversal",
            "Reversed pair-total-preserving redistribution",
            {"eurosat": pair_total - 10.0, "sun397": 10.0},
        ),
    ]
    configs = []
    for run_id, label, target in targets:
        if run_id == "baseline":
            scales = baseline_scales
            search = {"method": "identity baseline", "least_squares_nfev": 0}
        else:
            scales, search = calibrate_target(cache, target, pair_total)
        # Record achieved values from the full, uncached diagnostic path.  No
        # accuracy or model evaluation is available to this program.
        achieved = compute_effective_weights(
            checkpoint_dirs, TASKS, scales
        )["effective_block_weight_percent"]
        record = calibration_record(target, scales, achieved, pair_total, search)
        configs.append({"run_id": run_id, "label": label, **record})
    return {
        "schema_version": 2,
        "series_id": "primary_diagnostic_calibrated_pair_total",
        "series_role": "primary_controlled_redistribution",
        "description": (
            "Primary causal series calibrated only on the offline effective-weight "
            "diagnostic to preserve the baseline EuroSAT+SUN397 pair total."
        ),
        "frozen_before_accuracy_evaluation": True,
        "task_order": list(TASKS),
        "calibration": {
            "uses_accuracy": False,
            "effective_weight_definition_version": EFFECTIVE_WEIGHT_DEFINITION_VERSION,
            "baseline_pair_total_percent": pair_total,
            "pair_penalty": PAIR_PENALTY,
            "objective": (
                "(w_eurosat-target_eurosat)^2 + (w_sun397-target_sun397)^2 "
                "+ pair_penalty*((w_eurosat+w_sun397)-baseline_pair_total)^2"
            ),
            "algorithm": (
                "deterministic 9x9 log-space grid over positive EuroSAT/SUN397 "
                "scales in [0.25,4.0], followed by bounded scipy least_squares; "
                "all other task scales fixed at 1.0"
            ),
        },
        "configs": configs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/selftrained_b32_r16_8task"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = build_config(args.artifact_root.resolve())
    rendered = json.dumps(payload, indent=2, sort_keys=False, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
