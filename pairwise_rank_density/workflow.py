"""Registered, resumable validation -> frozen manifest -> final-test workflow."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping

from .design import ALPHA, TASKS, TOP_PERCENT, build_response_rows, make_design, summarize_demand, validate_design
from .storage import (
    atomic_csv, atomic_json, atomic_predictions, output_lock, read_json,
    sealed, sha256_file, verify_sealed,
)


def _result_path(output: Path, split: str, run_id: str) -> Path:
    return output / split / "runs" / run_id / "result.json"


def _unit_hashes(output: Path, split: str, run_id: str) -> dict[str, str]:
    folder = _result_path(output, split, run_id).parent
    paths = sorted(path for path in folder.iterdir() if path.is_file() and path.name != "unit.json")
    return {str(path.relative_to(output)): sha256_file(path) for path in paths}


def _verify_unit(output: Path, split: str, run_id: str) -> dict[str, Any]:
    folder = _result_path(output, split, run_id).parent
    marker = folder / "unit.json"
    if not marker.is_file():
        raise ValueError(f"incomplete {split} run {run_id}: missing unit integrity marker")
    payload = verify_sealed(read_json(marker))
    current = _unit_hashes(output, split, run_id)
    if current != payload["files"]:
        raise ValueError(f"integrity failure for {split} run {run_id}")
    result = read_json(_result_path(output, split, run_id))
    if result.get("run_id") != run_id or result.get("split") != split:
        raise ValueError(f"run metadata mismatch for {split} run {run_id}")
    return result


def _write_unit(output: Path, split: str, configuration: Mapping[str, Any], raw: Mapping[str, Any],
                references: Mapping[str, float], identity_hash: str, mode: str,
                stage_reference_sha256: str | None = None) -> dict[str, Any]:
    run_id = configuration["run_id"]
    folder = _result_path(output, split, run_id).parent
    folder.mkdir(parents=True, exist_ok=True)
    per_task = raw["per_task"]
    metrics: dict[str, Any] = {}
    prediction_meta = {}
    for task in configuration["tasks"]:
        cell = per_task[task]
        accuracy = float(cell["accuracy_percent"])
        reference = float(references[task])
        if not 0 <= accuracy <= 100 or reference <= 0:
            raise ValueError(f"invalid accuracy/reference in {run_id}/{task}")
        metrics[task] = {
            "accuracy_percent": accuracy,
            "reference_accuracy_percent": reference,
            "normalized_retention_percent": 100.0 * accuracy / reference,
            "examples": int(cell["examples"]),
        }
        prediction_path = folder / f"{task}_predictions.npz"
        atomic_predictions(prediction_path, cell["labels"], cell["predictions"])
        prediction_meta[task] = {
            "path": str(prediction_path.relative_to(output)),
            "examples": int(cell["examples"]),
            "sha256": sha256_file(prediction_path),
        }
    result = {
        "schema_version": 1,
        "mode": mode,
        "run_id": run_id,
        "pair_id": configuration["pair_id"],
        "split": split,
        "tasks": configuration["tasks"],
        "variant": configuration["variant"],
        "focal_task": configuration["focal_task"],
        "partner_task": configuration["partner_task"],
        "level": configuration["level"],
        "rank_by_task": configuration["rank_by_task"],
        "density_by_task": configuration["density_by_task"],
        "method": {"alpha": ALPHA, "smoothing": "linear", "rho": 5.0,
                   "top_percent": TOP_PERCENT, "ties": "ties_small"},
        "metrics": {split: metrics},
        "density_diagnostics_by_task": raw.get("density_by_task", {}),
        "resources": raw.get("resources", {}),
        "predictions": prediction_meta,
        "identity_sha256": identity_hash,
        "stage_reference_sha256": stage_reference_sha256,
    }
    atomic_json(_result_path(output, split, run_id), result)
    hashes = _unit_hashes(output, split, run_id)
    atomic_json(folder / "unit.json", sealed({"split": split, "run_id": run_id, "files": hashes}))
    return result


def _load_stage(output: Path, split: str, design: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {configuration["run_id"]: _verify_unit(output, split, configuration["run_id"])
            for configuration in design["configurations"]}


def _write_analysis(output: Path, split: str, design: Mapping[str, Any], results: Mapping[str, Any]) -> None:
    response = build_response_rows(design, results, split)
    summary = summarize_demand(response, split)
    stage_dir = output / split
    atomic_csv(stage_dir / "responses.csv", response)
    atomic_csv(stage_dir / "pair_demands.csv", summary["pair_demands"])
    task_rows = []
    for task in TASKS:
        for kind in ("rank", "density"):
            item = summary["task_summary"][task][kind]
            task_rows.append({"task": task, "variant": kind,
                              "resource_axis": item["axis"], "axis_unit": item["axis_unit"],
                              "resource_levels": item["levels"], "x_levels": item["x_levels"],
                              "formula": item["formula"], "auc_units": item["auc_units"],
                              "variance_units": item["variance_units"],
                              "mean_demand_auc_pp_resource_fraction": item["mean_demand_auc_pp_resource_fraction"],
                              "sample_variance_across_partners_auc_units_squared": item["sample_variance_across_partners_auc_units_squared"],
                              "partner_count": item["partner_count"]})
    atomic_csv(stage_dir / "task_demand_summary.csv", task_rows)
    atomic_json(stage_dir / "demand_summary.json", summary)


def register(output: Path, identity: Mapping[str, Any], *, mode: str = "real") -> str:
    design = make_design()
    validate_design(design)
    registration = sealed({"design": design, "identity": dict(identity), "mode": mode})
    target = output / "registration.json"
    if target.exists():
        existing = verify_sealed(read_json(target))
        if existing != registration["payload"]:
            raise ValueError(f"existing registration differs; refusing to overwrite {target}")
    else:
        atomic_json(target, registration)
    return registration["sha256"]


def validation_manifest(output: Path, design: Mapping[str, Any], registration_sha256: str,
                        identity_sha256: str, *, mode: str = "real") -> dict[str, Any]:
    results = _load_stage(output, "validation", design)
    payload = {
        "schema_version": 1,
        "mode": mode,
        "registration_sha256": registration_sha256,
        "identity_sha256": identity_sha256,
        "configuration_count": len(design["configurations"]),
        "task_configuration_cells": 2 * len(design["configurations"]),
        "validation_units": {
            config["run_id"]: read_json(_result_path(output, "validation", config["run_id"]).parent / "unit.json")["sha256"]
            for config in design["configurations"]
        },
        "validation_complete": len(results) == 420,
    }
    if not payload["validation_complete"]:
        raise ValueError("cannot freeze validation: registered configuration set is incomplete")
    return sealed(payload)


def verify_validation_manifest(output: Path, design: Mapping[str, Any], registration_sha256: str,
                               identity_sha256: str, *, mode: str = "real") -> None:
    marker = output / "frozen_validation.json"
    if not marker.is_file():
        raise ValueError("final-test blocked: frozen validation manifest is missing")
    payload = verify_sealed(read_json(marker))
    if (payload["mode"] != mode or payload["registration_sha256"] != registration_sha256
            or payload["identity_sha256"] != identity_sha256
            or payload["configuration_count"] != len(design["configurations"])
            or not payload["validation_complete"]):
        raise ValueError("final-test blocked: validation manifest does not match registration/identity")
    _load_stage(output, "validation", design)
    actual = {
        config["run_id"]: read_json(_result_path(output, "validation", config["run_id"]).parent / "unit.json")["sha256"]
        for config in design["configurations"]
    }
    if actual != payload["validation_units"]:
        raise ValueError("final-test blocked: validation unit manifest integrity failure")


def run_stage(output: Path, stage: str, identity: Mapping[str, Any], references: Mapping[str, float],
              evaluator: Callable[[Mapping[str, Any], str], Mapping[str, Any]], *,
              mode: str = "real", force: bool = False,
              stage_reference_sha256: str | None = None) -> None:
    if stage not in ("validation", "final-test"):
        raise ValueError("stage must be validation or final-test")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output):
        design = make_design()
        registration_sha = register(output, identity, mode=mode)
        identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if stage == "final-test":
            verify_validation_manifest(output, design, registration_sha, identity_sha, mode=mode)
        for configuration in design["configurations"]:
            run_id = configuration["run_id"]
            result_path = _result_path(output, stage, run_id)
            if result_path.exists() and not force:
                marker = result_path.parent / "unit.json"
                if marker.is_file():
                    _verify_unit(output, stage, run_id)
                    continue
                # A crash before the atomic unit marker leaves an uncommitted
                # partial unit. Remove only this run's files and recompute it.
                shutil.rmtree(result_path.parent)
            log_path = result_path.parent / "failure.log"
            try:
                raw = evaluator(configuration, "val" if stage == "validation" else "test")
                if log_path.exists():
                    log_path.unlink()
                _write_unit(output, stage, configuration, raw, references, identity_sha, mode,
                            stage_reference_sha256)
            except Exception:
                result_path.parent.mkdir(parents=True, exist_ok=True)
                logging.exception("%s run failed: %s", stage, run_id)
                log_path.write_text(traceback.format_exc(), encoding="utf-8")
                raise
        results = _load_stage(output, stage, design)
        _write_analysis(output, stage, design, results)
        if stage == "validation":
            manifest = validation_manifest(output, design, registration_sha, identity_sha, mode=mode)
            target = output / "frozen_validation.json"
            atomic_json(target, manifest)
        else:
            atomic_json(output / "final_test_complete.json", sealed({
                "mode": mode, "registration_sha256": registration_sha,
                "identity_sha256": identity_sha, "configuration_count": 420,
                "task_configuration_cells": 840,
                "reference_sha256": stage_reference_sha256,
            }))


def validate_output_integrity(output: Path, stage: str = "validation") -> None:
    design = make_design()
    _load_stage(Path(output), stage, design)

