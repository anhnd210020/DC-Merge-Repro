"""Frozen task/context registration and configuration generation."""
from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any, Mapping

TASKS = ("stanford_cars", "dtd", "eurosat", "gtsrb", "mnist", "resisc45", "sun397", "svhn")
SCALE_LEVELS = (0.50, 0.75, 1.00, 1.25, 1.50)
RANK_LEVELS = (2, 4, 8, 12, 16)
DENSITY_LEVELS = (0.25, 0.50, 0.75, 1.00)
ACTIVE_RESOURCES = ("scale", "rank", "density")
ALPHA = 0.8
RHO = 5.0
TOP_PERCENT = 0.001
SEED = 400


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def make_context_manifest() -> dict[str, Any]:
    """Register all pairs and five deterministic four-task contexts per focal task."""
    phase_a = [list(c) for c in itertools.combinations(TASKS, 2)]
    phase_b_rows: list[dict[str, Any]] = []
    for focal in TASKS:
        companions = [task for task in TASKS if task != focal]
        # Select across the lexicographic combination list to spread contexts.
        candidates = list(itertools.combinations(companions, 3))
        indices = [round(i * (len(candidates) - 1) / 4) for i in range(5)]
        for ordinal, index in enumerate(indices, 1):
            members = sorted((focal, *candidates[index]), key=TASKS.index)
            phase_b_rows.append({
                "focal_task": focal,
                "context_id": "ctx4-" + "-".join(members),
                "tasks": members,
                "selection_ordinal": ordinal,
            })
    # Exact six-task context memberships and the pair members used as focal
    # tasks in the completed Experiment 3 context-selection run. The old
    # jointly-scaled responses are not imported as oracle observations.
    phase_c_specs = [
        (("stanford_cars","dtd","eurosat","gtsrb","mnist","svhn"), ("eurosat","dtd")),
        (("dtd","eurosat","gtsrb","mnist","resisc45","sun397"), ("eurosat","gtsrb")),
        (("stanford_cars","eurosat","mnist","resisc45","sun397","svhn"), ("eurosat","stanford_cars")),
        (("stanford_cars","gtsrb","mnist","resisc45","sun397","svhn"), ("gtsrb","svhn")),
        (("stanford_cars","dtd","eurosat","resisc45","sun397","svhn"), ("stanford_cars","sun397")),
        (("stanford_cars","dtd","eurosat","gtsrb","mnist","svhn"), ("dtd","eurosat")),
        (("stanford_cars","dtd","eurosat","gtsrb","mnist","resisc45"), ("dtd","gtsrb")),
        (("dtd","eurosat","gtsrb","mnist","resisc45","sun397"), ("dtd","sun397")),
        (("eurosat","gtsrb","mnist","resisc45","sun397","svhn"), ("mnist","resisc45")),
        (("stanford_cars","dtd","mnist","resisc45","sun397","svhn"), ("dtd","mnist")),
        (("stanford_cars","dtd","mnist","resisc45","sun397","svhn"), ("resisc45","svhn")),
        (("stanford_cars","dtd","eurosat","gtsrb","sun397","svhn"), ("dtd","svhn")),
    ]
    phase_c_map: dict[tuple[str, ...], set[str]] = {}
    for tasks, focals in phase_c_specs:
        ordered = sorted(tasks, key=TASKS.index)
        phase_c_map.setdefault(tuple(ordered), set()).update(focals)
    phase_c = [{"context_id":"ctx6-"+"-".join(tasks), "tasks":list(tasks),
                "focal_tasks":sorted(focals,key=TASKS.index),
                "source":"Experiment 3 exact context membership; no response reuse"}
               for tasks, focals in sorted(phase_c_map.items(), key=lambda x: tuple(TASKS.index(t) for t in x[0]))]
    return {
        "schema_version": 1,
        "experiment": "can-task-merge-demand-be-predicted-before-evaluation",
        "task_order": list(TASKS),
        "phase_a_contexts": [
            {"context_id": "ctx2-" + "-".join(c), "tasks": c} for c in phase_a
        ],
        "phase_b_focal_contexts": phase_b_rows,
        "phase_c_reused_contexts": phase_c,
        "phase_c_status": "12 exact six-task Experiment 3 contexts registered; historical joint-scale responses not reused",
        "counts": {
            "phase_a_contexts": len(phase_a),
            "phase_a_focal_context_rows": len(phase_a) * 2,
            "phase_b_focal_context_rows": len(phase_b_rows),
            "phase_b_unique_contexts": len({r["context_id"] for r in phase_b_rows}),
            "phase_c_reused_contexts": len(phase_c),
            "phase_c_focal_context_rows": sum(len(r["focal_tasks"]) for r in phase_c),
            "total_focal_context_rows_ab": len(phase_a) * 2 + len(phase_b_rows),
        },
    }


def make_design(manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    manifest = dict(manifest or make_context_manifest())
    contexts: dict[str, dict[str, Any]] = {}
    for row in manifest["phase_a_contexts"]:
        contexts[row["context_id"]] = {"phase": "A", "focals": list(row["tasks"]), **row}
    for row in manifest["phase_b_focal_contexts"]:
        context = contexts.setdefault(row["context_id"], {
            "phase": "B", "context_id": row["context_id"], "tasks": row["tasks"], "focals": []
        })
        if context["phase"] != "B": raise ValueError("context id reused across phases")
        if row["focal_task"] not in context["focals"]: context["focals"].append(row["focal_task"])
    for row in manifest.get("phase_c_reused_contexts", []):
        contexts[row["context_id"]] = {"phase":"C", "context_id":row["context_id"],
            "tasks":row["tasks"], "focals":list(row["focal_tasks"])}
    configurations = []
    for context_id, context in contexts.items():
        members = context["tasks"]
        def add(resource: str, focal: str | None, level: float | int | None) -> None:
            scales = {t: 1.0 for t in members}
            ranks = {t: 16 for t in members}
            densities = {t: 1.0 for t in members}
            if focal is not None:
                if resource == "scale": scales[focal] = float(level)
                elif resource == "rank": ranks[focal] = int(level)
                elif resource == "density": densities[focal] = float(level)
            token = "baseline" if focal is None else f"{resource}-{focal}-{level}"
            configurations.append({
                "run_id": f"{context_id}__{token}", "context_id": context_id,
                "phase": context["phase"], "tasks": list(members), "resource": resource,
                "focal_task": focal, "level": level,
                "scale_by_task": scales, "rank_by_task": ranks,
                "density_by_task": densities,
            })
        add("baseline", None, None)
        for focal in context["focals"]:
            for resource, levels, baseline in (("scale", SCALE_LEVELS, 1.0), ("rank", RANK_LEVELS, 16), ("density", DENSITY_LEVELS, 1.0)):
                for level in levels:
                    if level != baseline:
                        add(resource, focal, level)
    design = {
        "schema_version": 1,
        "experiment": manifest["experiment"],
        "task_order": list(TASKS),
        "context_manifest": manifest,
        "resource_levels": {"scale": list(SCALE_LEVELS), "rank": list(RANK_LEVELS), "density": list(DENSITY_LEVELS)},
        "active_resources": list(ACTIVE_RESOURCES),
        "method": {
            "algorithm": "existing repository dc_merge",
            "smoothing": "linear", "rho": RHO, "alpha": ALPHA,
            "top_percent": TOP_PERCENT, "aggregation": "ties_small",
            "rank": "merge-time truncation of rank-16 adapter delta",
            "density": "task-wide budget K=round-half-up(d*N) over focal-task baseline post-projection/top-k support; select globally by descending absolute value, then stable module name and flattened coordinate index; d=1 preserves baseline support exactly; record requested, target, actual, and per-module counts",
            "prediction_features_use_candidate_accuracy": False,
            "oracle_definition_status": "Experiment Card response deficits and user-frozen operational conventions applied",
            "auc_axis": "linearly map each tested resource minimum to x=0 and maximum to x=1; signed trapezoidal integral with no further interval-width division",
            "auc_rationale": "This makes each resource's tested range comparable on a unit axis while preserving signed area; dividing again by interval width would average rather than integrate over the normalized range.",
            "protocol_frozen": True,
            "unresolved_protocol_decisions": [],
        },
        "configurations": configurations,
        "counts": {
            "unique_contexts": len(contexts), "unique_configurations": len(configurations),
            "phase_a_unique_configurations": sum(c["phase"] == "A" for c in configurations),
            "phase_b_unique_configurations": sum(c["phase"] == "B" for c in configurations),
            "phase_c_unique_configurations": sum(c["phase"] == "C" for c in configurations),
            "phase_a_task_evaluation_cells": sum(len(c["tasks"]) for c in configurations if c["phase"] == "A"),
            "phase_b_task_evaluation_cells": sum(len(c["tasks"]) for c in configurations if c["phase"] == "B"),
            "phase_c_task_evaluation_cells": sum(len(c["tasks"]) for c in configurations if c["phase"] == "C"),
            "task_evaluation_cells_per_split": sum(len(c["tasks"]) for c in configurations),
            "phase_a_focal_context_rows": 56,
            "phase_b_focal_context_rows": 40,
            "phase_c_focal_context_rows": sum(len(r["focal_tasks"]) for r in manifest.get("phase_c_reused_contexts", [])),
            "registered_focal_context_rows": 56 + 40 + sum(len(r["focal_tasks"]) for r in manifest.get("phase_c_reused_contexts", [])),
        },
    }
    design["design_sha256"] = sha256_json(design)
    return design


def validate_design(design: Mapping[str, Any]) -> None:
    supplied_sha = design.get("design_sha256")
    unsigned = {k:v for k,v in design.items() if k != "design_sha256"}
    if supplied_sha != sha256_json(unsigned):
        raise ValueError("design checksum mismatch")
    manifest = design["context_manifest"]
    if len(manifest["phase_a_contexts"]) != 28:
        raise ValueError("Phase A must contain all 28 unordered pairs")
    rows = manifest["phase_b_focal_contexts"]
    if len(rows) < 40 or any(len(r["tasks"]) != 4 or r["focal_task"] not in r["tasks"] for r in rows):
        raise ValueError("Phase B needs at least 40 valid four-task focal/context rows")
    for row in manifest.get("phase_c_reused_contexts", []):
        if len(row["tasks"]) != 6 or not set(row["focal_tasks"]).issubset(row["tasks"]):
            raise ValueError("Phase C contexts must be six-task sets with registered focal tasks")
    if any(sum(r["focal_task"] == t for r in rows) < 5 for t in TASKS):
        raise ValueError("each task needs at least five Phase B contexts")
    phase_c = manifest.get("phase_c_reused_contexts", [])
    if any(len(r["tasks"]) not in (6, 8) or not set(r.get("focal_tasks", [])).issubset(r["tasks"]) for r in phase_c):
        raise ValueError("Phase C requires valid six- or eight-task contexts and registered focal tasks")
    if len({r["context_id"] for r in phase_c}) != len(phase_c):
        raise ValueError("Phase C context memberships must be unique")
    ids = [c["run_id"] for c in design["configurations"]]
    if len(ids) != len(set(ids)) or len(ids) != design["counts"]["unique_configurations"]:
        raise ValueError("configuration ids/count are inconsistent")
    if not any(c["resource"] == "density" and c["level"] != 1.0 for c in design["configurations"]):
        raise ValueError("density sweep must include non-baseline levels")
    if design["counts"]["registered_focal_context_rows"] != 56 + len(rows) + sum(len(r["focal_tasks"]) for r in phase_c):
        raise ValueError("registered focal/context count is inconsistent with manifest")
    for c in design["configurations"]:
        if c["focal_task"] is None:
            continue
        if c["focal_task"] not in c["tasks"]:
            raise ValueError(f"focal task absent from context: {c['run_id']}")
        untouched = [t for t in c["tasks"] if t != c["focal_task"]]
        if any(c["scale_by_task"][t] != 1.0 or c["rank_by_task"][t] != 16 or c["density_by_task"][t] != 1.0 for t in untouched):
            raise ValueError(f"non-focal resource changed: {c['run_id']}")
