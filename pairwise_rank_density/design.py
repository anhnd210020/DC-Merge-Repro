"""Immutable registration and analysis definitions for the pairwise study."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

TASKS = (
    "stanford_cars", "dtd", "eurosat", "gtsrb", "mnist", "resisc45", "sun397", "svhn"
)
RANK_LEVELS = (2, 4, 8, 12, 16)
DENSITY_LEVELS = (0.25, 0.5, 0.75, 1.0)
ALPHA = 0.8
TOP_PERCENT = 0.001
RHO = 5.0
SEED = 400


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def density_token(value: float) -> str:
    return format(float(value), ".8g").replace(".", "p")


def pair_configurations(task_a: str, task_b: str) -> list[dict[str, Any]]:
    if task_a not in TASKS or task_b not in TASKS or TASKS.index(task_a) >= TASKS.index(task_b):
        raise ValueError("pair tasks must follow the registered task order")
    pair_id = f"{task_a}--{task_b}"
    configs: list[dict[str, Any]] = []

    def add(kind: str, focal: str | None, value: int | float | None) -> None:
        ranks = {task_a: 16, task_b: 16}
        densities = {task_a: 1.0, task_b: 1.0}
        if kind == "rank":
            ranks[focal] = int(value)
            run_id = f"{pair_id}__rank__{focal}__r{int(value)}"
        elif kind == "density":
            densities[focal] = float(value)
            run_id = f"{pair_id}__density__{focal}__d{density_token(float(value))}"
        else:
            run_id = f"{pair_id}__baseline"
        configs.append({
            "run_id": run_id,
            "pair_id": pair_id,
            "tasks": [task_a, task_b],
            "variant": kind,
            "focal_task": focal,
            "partner_task": (task_b if focal == task_a else task_a) if focal else None,
            "level": value,
            "rank_by_task": ranks,
            "density_by_task": densities,
        })

    add("baseline", None, None)
    for focal in (task_a, task_b):
        for level in RANK_LEVELS:
            if level != 16:
                add("rank", focal, level)
    for focal in (task_a, task_b):
        for level in DENSITY_LEVELS:
            if level != 1.0:
                add("density", focal, level)
    return configs


def make_design() -> dict[str, Any]:
    pairs = [list(pair) for pair in itertools.combinations(TASKS, 2)]
    configurations = [config for pair in pairs for config in pair_configurations(*pair)]
    return {
        "schema_version": 1,
        "experiment": "pairwise-rank-density-demand",
        "task_order": list(TASKS),
        "pairs": pairs,
        "levels": {"rank": list(RANK_LEVELS), "density": list(DENSITY_LEVELS)},
        "method": {
            "algorithm": "repository_dc_merge",
            "smoothing": "linear",
            "rho": RHO,
            "alpha": ALPHA,
            "top_percent": TOP_PERCENT,
            "ties": "ties_small",
            "structural_mask": "task-specific diagonal blocks",
            "seed": SEED,
            "density_rounding": "floor(d * baseline_mask_count + 0.5), half-up",
            "density_tie_break": "descending absolute magnitude, ascending flattened coordinate",
            "demand": {
                "metric": "signed trapezoidal AUC of focal retention deficit",
                "deficit_pp": "focal pair-baseline normalized retention - focal normalized retention at level x",
                "formula": "sum((x[k+1]-x[k]) * (deficit[k]+deficit[k+1]) / 2)",
                "rank_axis": "retained rank / 16",
                "rank_levels": list(RANK_LEVELS),
                "density_axis": "nominal density",
                "density_levels": list(DENSITY_LEVELS),
                "include_shared_full_resource_endpoint": True,
                "auc_units": "percentage_points × normalized_resource_fraction",
                "partner_interference_pp": "partner pair-baseline normalized retention - partner normalized retention under focal intervention",
            },
        },
        "counts": {
            "unordered_pairs": len(pairs),
            "configurations_per_pair": 15,
            "unique_configurations": len(configurations),
            "task_configuration_cells_per_split": 2 * len(configurations),
        },
        "configurations": configurations,
    }


def validate_design(design: Mapping[str, Any]) -> None:
    expected = make_design()
    if digest(design) != digest(expected):
        raise ValueError("design differs from the immutable pairwise rank/density registration")


def _sample_variance(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def trapezoidal_auc(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    """Signed trapezoidal area for an ordered response curve."""
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("AUC inputs must have equal length >= 2")
    if any(not math.isfinite(float(x)) for x in x_values) or any(
        not math.isfinite(float(y)) for y in y_values
    ):
        raise ValueError("AUC inputs must be finite")
    if any(float(right) <= float(left) for left, right in zip(x_values, x_values[1:])):
        raise ValueError("AUC x values must be strictly increasing")
    return sum(
        (float(x1) - float(x0)) * (float(y0) + float(y1)) / 2.0
        for x0, x1, y0, y1 in zip(x_values, x_values[1:], y_values, y_values[1:])
    )


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for index, _ in ordered[start:end]:
            ranks[index] = average_rank
        start = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman inputs must have equal length >= 2")
    if any(not math.isfinite(float(x)) for x in (*left, *right)):
        raise ValueError("Spearman inputs must be finite")
    x, y = _average_ranks(left), _average_ranks(right)
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    dx = [v - mean_x for v in x]
    dy = [v - mean_y for v in y]
    denominator = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    return sum(a * b for a, b in zip(dx, dy)) / denominator if denominator else None


def build_response_rows(design: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]], split: str) -> list[dict[str, Any]]:
    """Create per-level deficits and partner interference (positive means harm)."""
    validate_design(design)
    rows: list[dict[str, Any]] = []
    for pair in design["pairs"]:
        task_a, task_b = pair
        baseline_id = f"{task_a}--{task_b}__baseline"
        baseline = results[baseline_id]["metrics"][split]
        for focal, partner in ((task_a, task_b), (task_b, task_a)):
            for kind, level, x in (
                ("rank", 16, 1.0), ("density", 1.0, 1.0),
            ):
                focal_retention = baseline[focal]["normalized_retention_percent"]
                partner_retention = baseline[partner]["normalized_retention_percent"]
                rows.append({
                    "run_id": baseline_id, "pair_id": f"{task_a}--{task_b}",
                    "variant": kind, "focal_task": focal, "partner_task": partner,
                    "level": level, "resource_x": x, "is_shared_baseline_endpoint": True,
                    "focal_baseline_retention_percent": focal_retention,
                    "focal_retention_percent": focal_retention,
                    "self_retention_change_pp": 0.0, "deficit_pp": 0.0,
                    "self_retention_loss_demand_pp": 0.0,
                    "partner_baseline_retention_percent": partner_retention,
                    "partner_retention_percent": partner_retention,
                    "partner_retention_change_pp": 0.0,
                    "partner_interference_pp": 0.0, "split": split,
                })
        for config in design["configurations"]:
            if config["pair_id"] != f"{task_a}--{task_b}" or config["variant"] == "baseline":
                continue
            current = results[config["run_id"]]["metrics"][split]
            focal, partner = config["focal_task"], config["partner_task"]
            baseline_focal = baseline[focal]["normalized_retention_percent"]
            baseline_partner = baseline[partner]["normalized_retention_percent"]
            focal_retention = current[focal]["normalized_retention_percent"]
            partner_retention = current[partner]["normalized_retention_percent"]
            self_change = current[focal]["normalized_retention_percent"] - baseline[focal]["normalized_retention_percent"]
            partner_change = current[partner]["normalized_retention_percent"] - baseline[partner]["normalized_retention_percent"]
            rows.append({
                "run_id": config["run_id"], "pair_id": config["pair_id"],
                "variant": config["variant"], "focal_task": focal, "partner_task": partner,
                "level": config["level"],
                "resource_x": (float(config["level"]) / 16.0 if config["variant"] == "rank"
                               else float(config["level"])),
                "is_shared_baseline_endpoint": False,
                "focal_baseline_retention_percent": baseline_focal,
                "focal_retention_percent": focal_retention,
                "self_retention_change_pp": self_change,
                "deficit_pp": baseline_focal - focal_retention,
                # Retained as a compatibility alias for existing raw-response consumers.
                "self_retention_loss_demand_pp": baseline_focal - focal_retention,
                "partner_baseline_retention_percent": baseline_partner,
                "partner_retention_percent": partner_retention,
                "partner_retention_change_pp": partner_change,
                "partner_interference_pp": baseline_partner - partner_retention,
                "split": split,
            })
    return rows


def summarize_demand(response_rows: Sequence[Mapping[str, Any]], split: str) -> dict[str, Any]:
    by_task_kind_partner: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in response_rows:
        if row["split"] != split:
            continue
        by_task_kind_partner[(str(row["focal_task"]), str(row["variant"]), str(row["partner_task"]))].append(row)
    pair_demands = []
    task_summary: dict[str, dict[str, Any]] = {}
    axes = {
        "rank": {"axis": "retained rank / 16", "axis_unit": "fraction of full rank",
                 "levels": list(RANK_LEVELS), "x_levels": [rank / 16.0 for rank in RANK_LEVELS]},
        "density": {"axis": "nominal density", "axis_unit": "fraction of baseline support",
                    "levels": list(DENSITY_LEVELS), "x_levels": list(DENSITY_LEVELS)},
    }
    for task in TASKS:
        task_summary[task] = {}
        for kind in ("rank", "density"):
            partners = [other for other in TASKS if other != task]
            partner_demands = []
            for partner in partners:
                curve = by_task_kind_partner.get((task, kind, partner), [])
                if not curve:
                    raise ValueError(f"missing {kind} demand for {task} with partner {partner}")
                curve = sorted(curve, key=lambda row: float(row["resource_x"]))
                actual_x = [float(row["resource_x"]) for row in curve]
                expected_x = axes[kind]["x_levels"]
                if actual_x != expected_x:
                    raise ValueError(
                        f"{kind} response curve for {task}/{partner} has x={actual_x}; expected {expected_x}"
                    )
                auc = trapezoidal_auc(actual_x, [float(row["deficit_pp"]) for row in curve])
                summary_row = {
                    "task": task, "partner": partner, "variant": kind,
                    "demand_auc_pp_resource_fraction": auc,
                    "resource_axis": axes[kind]["axis"],
                    "axis_unit": axes[kind]["axis_unit"],
                    "resource_levels": axes[kind]["levels"],
                    "x_levels": expected_x,
                    "formula": "sum((x[k+1]-x[k]) * (deficit[k]+deficit[k+1]) / 2)",
                    "auc_units": "percentage_points × normalized_resource_fraction",
                    "intervention_count": len(curve) - 1,
                }
                partner_demands.append(summary_row)
                pair_demands.append(partner_demands[-1])
            aucs = [row["demand_auc_pp_resource_fraction"] for row in partner_demands]
            for row in partner_demands:
                row["deviation_from_task_mean_auc_pp_resource_fraction"] = row["demand_auc_pp_resource_fraction"] - sum(aucs) / len(aucs)
            task_summary[task][kind] = {
                **axes[kind],
                "formula": "sum((x[k+1]-x[k]) * (deficit[k]+deficit[k+1]) / 2)",
                "auc_units": "percentage_points × normalized_resource_fraction",
                "variance_units": "(percentage_points × normalized_resource_fraction)^2",
                "mean_demand_auc_pp_resource_fraction": sum(aucs) / len(aucs),
                "sample_variance_across_partners_auc_units_squared": _sample_variance(aucs),
                "partner_count": len(aucs),
                "partner_demands": partner_demands,
            }
    rank_order = [task_summary[task]["rank"]["mean_demand_auc_pp_resource_fraction"] for task in TASKS]
    density_order = [task_summary[task]["density"]["mean_demand_auc_pp_resource_fraction"] for task in TASKS]
    return {
        "split": split,
        "task_order": list(TASKS),
        "demand_definition": "signed trapezoidal AUC of focal performance deficit versus normalized resource level, including the shared full-resource baseline endpoint",
        "variance_definition": "sample variance (N-1) across the seven pair-specific AUCs",
        "partner_interference_definition": "partner pair-baseline normalized retention minus partner normalized retention under the focal intervention; positive means harm to the partner",
        "pair_demands": pair_demands,
        "task_summary": task_summary,
        "spearman_rank_vs_density_demand_auc_order": spearman(rank_order, density_order),
        "spearman_task_values": {"rank_demand_auc_pp_resource_fraction": rank_order,
                                 "density_demand_auc_pp_resource_fraction": density_order},
    }
