"""Predicted Demand-Ranked Allocation experiment and frozen provenance artifacts.

The runner delegates every real merge/evaluation to ``RealEvaluator``.  This
module only supplies task-rank assignments, frozen inputs, checkpointing, and
analysis around the repository's existing DC-Merge implementation.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import tarfile
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .design import ALPHA, RHO, TOP_PERCENT, TASKS
from .storage import atomic_csv, atomic_json, atomic_text, hash_file, output_lock, seal, unseal

AUDIT_ARCHIVE_SHA256 = "7b01616a8e75774802ce8b95fd03e900c494003a7d0c16fb228153879e3f04ad"
AUDIT_FEATURES_MEMBER = "output_retry2/features.csv"
AUDIT_VALIDATION_ORACLE_MEMBER = "output_retry2/validation/oracle_responses.csv"
AUDIT_CONTEXT_MEMBER = "output_retry2/context_manifest.json"
AUDIT_REGISTRATION_MEMBER = "output_retry2/registration.json"
AUDIT_MODEL_SELECTION_MEMBER = "validation_model_selection_frozen.md"
PREDICTOR_FILENAME = "pdra_predictor.json"
MANIFEST_FILENAME = "pdra_context_manifest.json"
FEATURE_NAME = "energy_rank_90"
RANK_LEVELS = (2, 4, 8, 12, 16)
METHODS = ("Uniform", "PDRA", "Reverse-PDRA", "Oracle-Ranked")
TEMPLATES = {4: (12, 8, 8, 4), 6: (12, 12, 8, 8, 4, 4)}
UNIFORM = {4: (8, 8, 8, 8), 6: (8, 8, 8, 8, 8, 8)}
TIE_RULE = "equal demand is ordered by the registered TASKS order"
ORACLE_POLICY = "same-split post-hoc diagnostic; oracle labels only set Oracle-Ranked order"
ORACLE_POLICY_TOKEN = "same-split-posthoc"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sealed(payload: Any) -> dict[str, Any]:
    return {"payload": payload, "sha256": sha256_json(payload)}


def _read_member(archive: tarfile.TarFile, name: str) -> bytes:
    try:
        stream = archive.extractfile(name)
    except KeyError as exc:
        raise ValueError(f"required audit archive member is missing: {name}") from exc
    if stream is None:
        raise ValueError(f"required audit archive member is unreadable: {name}")
    return stream.read()


def _load_prior_manifest(raw: bytes) -> dict[str, Any]:
    value = json.loads(raw.decode("utf-8"))
    if "payload" in value and "sha256" in value:
        return unseal(value)
    return value


def build_context_manifest(prior_manifest: Mapping[str, Any], *, archive_sha256: str = AUDIT_ARCHIVE_SHA256) -> dict[str, Any]:
    """Select every lexicographically ordered membership absent from prior B/C."""
    prior_four = {tuple(row["tasks"]) for row in prior_manifest["phase_b_focal_contexts"]}
    prior_six = {tuple(row["tasks"]) for row in prior_manifest.get("phase_c_reused_contexts", [])}
    combos = {
        4: list(itertools.combinations(TASKS, 4)),
        6: list(itertools.combinations(TASKS, 6)),
    }
    fresh_four = [c for c in combos[4] if c not in prior_four]
    fresh_six = [c for c in combos[6] if c not in prior_six]
    contexts = {
        "four_task": [{"context_id": "pdra4-" + "-".join(c), "tasks": list(c)} for c in fresh_four],
        "six_task": [{"context_id": "pdra6-" + "-".join(c), "tasks": list(c)} for c in fresh_six],
    }
    coverage = {
        size: {task: sum(task in row["tasks"] for row in rows) for task in TASKS}
        for size, rows in contexts.items()
    }
    payload = {
        "schema_version": 1,
        "experiment": "predicted-demand-ranked-allocation",
        "task_order": list(TASKS),
        "selection_rule": "all combinations of the registered 8-task pool not present in previous Phase B (4-task) or Phase C (6-task); lexicographic TASKS order",
        "prior_context_source": AUDIT_CONTEXT_MEMBER,
        "prior_context_manifest_sha256": hashlib.sha256(canonical_json(dict(prior_manifest)).encode()).hexdigest(),
        "prior_reused_context_counts": {"four_task": len(prior_four), "six_task": len(prior_six)},
        "prior_memberships": {"four_task": [list(c) for c in sorted(prior_four)], "six_task": [list(c) for c in sorted(prior_six)]},
        "fresh_contexts": contexts,
        "fresh_context_counts": {"four_task": len(fresh_four), "six_task": len(fresh_six)},
        "task_coverage": coverage,
        "source_audit_archive_sha256": archive_sha256,
        "allocation": {
            "rank_levels": list(RANK_LEVELS), "uniform": {str(k): list(v) for k, v in UNIFORM.items()},
            "pdra_template": {str(k): list(v) for k, v in TEMPLATES.items()},
            "tie_break": TIE_RULE,
        },
        "oracle_diagnostic": ORACLE_POLICY,
        "frozen_before_allocation_evaluation": True,
    }
    return _sealed(payload)


def build_context_manifest_from_archive(audit_archive: Path) -> dict[str, Any]:
    archive_path = Path(audit_archive)
    archive_hash = hash_file(archive_path)
    if archive_hash != AUDIT_ARCHIVE_SHA256:
        raise ValueError(f"prior audit archive hash mismatch: {archive_hash}")
    with tarfile.open(archive_path, mode="r:gz") as archive:
        prior = _load_prior_manifest(_read_member(archive, AUDIT_CONTEXT_MEMBER))
    result = build_context_manifest(prior, archive_sha256=archive_hash)
    validate_context_manifest(result)
    return result


def validate_context_manifest(value: Mapping[str, Any]) -> None:
    if value.get("sha256") != sha256_json(value.get("payload")):
        raise ValueError("PDRA context manifest checksum mismatch")
    payload = value["payload"]
    if payload.get("task_order") != list(TASKS):
        raise ValueError("PDRA context manifest task order differs from the registered order")
    prior = {4: {tuple(c) for c in payload["prior_memberships"]["four_task"]},
             6: {tuple(c) for c in payload["prior_memberships"]["six_task"]}}
    expected_counts = {4: math.comb(8, 4) - len(prior[4]), 6: math.comb(8, 6) - len(prior[6])}
    for size, key in ((4, "four_task"), (6, "six_task")):
        rows = payload["fresh_contexts"][key]
        members = [tuple(row["tasks"]) for row in rows]
        if len(members) != expected_counts[size] or len(set(members)) != len(members):
            raise ValueError(f"invalid {size}-task fresh context count or duplicate membership")
        if any(len(row) != size or tuple(sorted(row, key=TASKS.index)) != row for row in members):
            raise ValueError(f"invalid {size}-task context task order")
        if set(members) & prior[size]:
            raise ValueError(f"fresh {size}-task contexts overlap prior exact memberships")
        expected = [c for c in itertools.combinations(TASKS, size) if c not in prior[size]]
        if members != expected:
            raise ValueError(f"fresh {size}-task contexts do not follow the frozen exhaustive selection rule")
        if payload["fresh_context_counts"][key] != expected_counts[size]:
            raise ValueError("context manifest count summary mismatch")
        if any(payload["task_coverage"][key][task] <= 0 for task in TASKS):
            raise ValueError(f"not all tasks are represented among fresh {size}-task contexts")
    if not payload.get("frozen_before_allocation_evaluation"):
        raise ValueError("context manifest is not frozen")


def load_context_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_context_manifest(value)
    return value


def _read_csv(raw: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(raw.decode("utf-8-sig", errors="strict").splitlines()))


def _build_validation_rows(features_raw: bytes, oracle_raw: bytes) -> tuple[list[dict[str, Any]], dict[str, float]]:
    feature_rows = _read_csv(features_raw)
    feature_map: dict[tuple[str, str], dict[str, str]] = {}
    task_features: dict[str, float] = {}
    for row in feature_rows:
        if row.get("feature_split") not in ("pre_evaluation_shared_across_validation_and_test", "validation"):
            raise ValueError("feature table contains a non-pre-evaluation or non-validation row")
        key = (row["context_id"], row["focal_task"])
        if key in feature_map:
            raise ValueError(f"duplicate feature row: {key}")
        feature_map[key] = row
        task, value = row["focal_task"], float(row[FEATURE_NAME])
        if task in task_features and task_features[task] != value:
            raise ValueError(f"{FEATURE_NAME} is not task-self-only for {task}")
        task_features[task] = value
    if set(task_features) != set(TASKS):
        raise ValueError("prior feature table does not cover the registered tasks")

    response_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(oracle_raw):
        if row.get("split") != "validation":
            raise ValueError("only previous-experiment validation responses are allowed for predictor fitting")
        if row.get("resource") == "rank":
            response_groups[(row["context_id"], row["focal_task"])].append(row)
    if set(response_groups) != set(feature_map):
        raise ValueError("validation rank response cells and feature rows differ")

    order = {task: i for i, task in enumerate(TASKS)}
    training_rows: list[dict[str, Any]] = []
    expected_levels = list(RANK_LEVELS)
    for (context_id, task), curve in sorted(response_groups.items(), key=lambda item: (item[0][0], order[item[0][1]])):
        by_level = {int(float(r["level"])): r for r in curve}
        if set(by_level) != set(expected_levels) or len(curve) != len(expected_levels):
            raise ValueError(f"incomplete/duplicate validation rank curve for {context_id}/{task}")
        retentions = [float(by_level[level]["normalized_retention_percent"]) for level in expected_levels]
        x = [(level - expected_levels[0]) / (expected_levels[-1] - expected_levels[0]) for level in expected_levels]
        deficits = [retentions[-1] - value for value in retentions]
        target = math.fsum((x[i + 1] - x[i]) * (deficits[i + 1] + deficits[i]) / 2 for i in range(len(x) - 1))
        feature_row = feature_map[(context_id, task)]
        training_rows.append({
            "split": "validation", "context_id": context_id, "phase": feature_row["phase"],
            "focal_task": task, "feature_name": FEATURE_NAME,
            "feature_value": task_features[task], "target_demand_auc_signed": target,
            "rank_axis": "(rank - 2) / (16 - 2)",
            "rank_response_curve": [{"rank": level, "rank_x": x[i],
                "normalized_retention_percent": retentions[i], "deficit_pp": deficits[i]}
                for i, level in enumerate(expected_levels)],
        })
    return training_rows, task_features


def build_predictor_artifact(audit_archive: Path) -> dict[str, Any]:
    """Fit the frozen one-feature Ridge model on prior validation labels only."""
    archive_path = Path(audit_archive)
    archive_hash = hash_file(archive_path)
    if archive_hash != AUDIT_ARCHIVE_SHA256:
        raise ValueError(f"prior audit archive hash mismatch: {archive_hash}")
    with tarfile.open(archive_path, mode="r:gz") as archive:
        features_raw = _read_member(archive, AUDIT_FEATURES_MEMBER)
        oracle_raw = _read_member(archive, AUDIT_VALIDATION_ORACLE_MEMBER)
        prior_manifest_raw = _read_member(archive, AUDIT_CONTEXT_MEMBER)
        registration_raw = _read_member(archive, AUDIT_REGISTRATION_MEMBER)
        registration = json.loads(registration_raw.decode("utf-8"))
        model_selection_raw = _read_member(archive, AUDIT_MODEL_SELECTION_MEMBER)
    prior_payload = _load_prior_manifest(prior_manifest_raw)
    prior_adapter_hashes = registration.get("payload", registration).get("identity", {}).get("adapter_model_sha256")
    prior_identity = registration.get("payload", registration).get("identity", {})
    if not prior_adapter_hashes or set(prior_adapter_hashes) != set(TASKS):
        raise ValueError("prior sealed run does not identify the eight source adapter checkpoints")
    source_dataset = prior_identity.get("dataset", {})
    source_dataset_identity = {key: source_dataset[key] for key in
        ("required_file_sha256", "mnist_source", "resisc45_class_dirs") if key in source_dataset}
    if not source_dataset_identity or not prior_identity.get("model_files"):
        raise ValueError("prior sealed run lacks base-model or dataset file identities")
    rows, task_features = _build_validation_rows(features_raw, oracle_raw)
    if len(rows) != 118:
        raise ValueError(f"expected the frozen 118 prior validation rank rows, got {len(rows)}")

    # Reuse the existing sklearn model family and its SciPy compatibility shim.
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from .analysis import _scipy_sym_pos_compat

    X = np.asarray([[row["feature_value"]] for row in rows], dtype=float)
    y = np.asarray([row["target_demand_auc_signed"] for row in rows], dtype=float)
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    with _scipy_sym_pos_compat():
        model.fit(X, y)
    imputer = model.named_steps["simpleimputer"]
    scaler = model.named_steps["standardscaler"]
    ridge = model.named_steps["ridge"]
    payload = {
        "schema_version": 1,
        "predictor": "validation-only standardized one-feature Ridge",
        "selected_feature": FEATURE_NAME,
        "fit_split": "validation",
        "fit_protocol": "fit once on all 118 previous-experiment validation rank context/task labels after frozen Phase A OOF selection; no allocation-test data; no tuning on final test",
        "estimator": {
            "class": "sklearn.linear_model.Ridge",
            "alpha": 1.0,
            "preprocessing": "SimpleImputer(strategy=median) then StandardScaler",
            "median_imputer_statistic": float(imputer.statistics_[0]),
            "standard_scaler_mean": float(scaler.mean_[0]),
            "standard_scaler_scale": float(scaler.scale_[0]),
            "ridge_coefficient_standardized": float(ridge.coef_[0]),
            "ridge_intercept": float(ridge.intercept_),
        },
        "task_features": {task: task_features[task] for task in TASKS},
        "training_rows": rows,
        "training_rows_sha256": sha256_json(rows),
        "training_row_count": len(rows),
        "source": {
            "audit_archive_sha256": archive_hash,
            "registration_member_sha256": hashlib.sha256(registration_raw).hexdigest(),
            "features_member": AUDIT_FEATURES_MEMBER,
            "features_member_sha256": hashlib.sha256(features_raw).hexdigest(),
            "validation_oracle_member": AUDIT_VALIDATION_ORACLE_MEMBER,
            "validation_oracle_member_sha256": hashlib.sha256(oracle_raw).hexdigest(),
            "prior_context_manifest_sha256": hashlib.sha256(prior_manifest_raw).hexdigest(),
            "prior_context_manifest_payload_sha256": sha256_json(prior_payload),
            "model_selection_decision_member": AUDIT_MODEL_SELECTION_MEMBER,
            "model_selection_decision_sha256": hashlib.sha256(model_selection_raw).hexdigest(),
            "source_adapter_model_sha256": {task: prior_adapter_hashes[task] for task in TASKS},
            "source_adapter_config_sha256": {task: prior_identity["adapter_config_sha256"][task] for task in TASKS},
            "source_head_sha256": {task: prior_identity["head_sha256"][task] for task in TASKS},
            "source_model_files": prior_identity["model_files"],
            "source_dataset_identity": source_dataset_identity,
            "source_validation_reference_sha256": prior_identity["validation_reference_sha256"],
            "source_validation_reference_values": prior_identity["validation_reference_values"],
            "read_members": [AUDIT_FEATURES_MEMBER, AUDIT_VALIDATION_ORACLE_MEMBER, AUDIT_CONTEXT_MEMBER,
                             AUDIT_REGISTRATION_MEMBER, AUDIT_MODEL_SELECTION_MEMBER],
            "test_label_files_read": [],
        },
        "final_test_labels_used": False,
        "allocation_test_labels_used_for_fit_or_tuning": False,
    }
    return _sealed(payload)


def validate_predictor_artifact(value: Mapping[str, Any]) -> None:
    payload = value.get("payload")
    if value.get("sha256") != sha256_json(payload):
        raise ValueError("PDRA predictor artifact checksum mismatch")
    if payload.get("fit_split") != "validation" or payload.get("selected_feature") != FEATURE_NAME:
        raise ValueError("PDRA predictor does not match the frozen validation feature/model selection")
    if payload.get("training_row_count") != len(payload.get("training_rows", [])) or payload.get("training_row_count") != 118:
        raise ValueError("PDRA predictor training row count is invalid")
    rows = payload["training_rows"]
    if any(row.get("split") != "validation" for row in rows):
        raise ValueError("PDRA predictor contains non-validation training rows")
    if sha256_json(rows) != payload.get("training_rows_sha256"):
        raise ValueError("PDRA predictor training row checksum mismatch")
    if payload.get("final_test_labels_used") or payload.get("allocation_test_labels_used_for_fit_or_tuning"):
        raise ValueError("PDRA predictor records forbidden test-label use")
    source = payload.get("source", {})
    if source.get("test_label_files_read") != [] or AUDIT_VALIDATION_ORACLE_MEMBER not in source.get("read_members", []):
        raise ValueError("PDRA predictor validation provenance is incomplete")
    if set(payload.get("task_features", {})) != set(TASKS) or set(source.get("source_adapter_model_sha256", {})) != set(TASKS):
        raise ValueError("PDRA predictor does not cover the registered task/adapter set")
    if any(set(source.get(key, {})) != set(TASKS) for key in
           ("source_adapter_config_sha256", "source_head_sha256")):
        raise ValueError("PDRA predictor does not bind the task adapter configuration and heads")
    if not source.get("source_model_files") or not source.get("source_dataset_identity") or not source.get("source_validation_reference_sha256"):
        raise ValueError("PDRA predictor does not bind its base model, validation data, and reference inputs")
    estimator = payload.get("estimator", {})
    required = ("alpha", "median_imputer_statistic", "standard_scaler_mean", "standard_scaler_scale",
                "ridge_coefficient_standardized", "ridge_intercept")
    if estimator.get("alpha") != 1.0 or any(k not in estimator or not math.isfinite(float(estimator[k])) for k in required):
        raise ValueError("PDRA predictor preprocessing or coefficients are invalid")
    if estimator["standard_scaler_scale"] <= 0:
        raise ValueError("PDRA predictor scaler has invalid scale")


def load_predictor_artifact(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_predictor_artifact(value)
    return value


def predict_demands(predictor: Mapping[str, Any], task_features: Mapping[str, float | None] | None = None) -> dict[str, float]:
    validate_predictor_artifact(predictor)
    payload = predictor["payload"]
    estimator = payload["estimator"]
    features = task_features or payload["task_features"]
    if set(features) != set(TASKS):
        raise ValueError("PDRA task features must cover exactly the registered tasks")
    output = {}
    for task in TASKS:
        raw = features[task]
        x = estimator["median_imputer_statistic"] if raw is None else float(raw)
        if not math.isfinite(x):
            raise ValueError(f"non-finite {FEATURE_NAME} for {task}")
        standardized = (x - estimator["standard_scaler_mean"]) / estimator["standard_scaler_scale"]
        output[task] = estimator["ridge_intercept"] + estimator["ridge_coefficient_standardized"] * standardized
    return output


def _demand_order(tasks: Sequence[str], demand: Mapping[str, float], *, descending: bool) -> list[str]:
    if len(tasks) not in TEMPLATES or len(set(tasks)) != len(tasks) or not set(tasks).issubset(TASKS):
        raise ValueError("allocation contexts must contain unique registered 4-task or 6-task memberships")
    if set(demand) != set(tasks):
        raise ValueError("demand map must contain exactly the context tasks")
    if any(not math.isfinite(float(demand[task])) for task in tasks):
        raise ValueError("demand values must be finite")
    task_index = {task: i for i, task in enumerate(TASKS)}
    if descending:
        return sorted(tasks, key=lambda task: (-float(demand[task]), task_index[task]))
    return sorted(tasks, key=lambda task: (float(demand[task]), task_index[task]))


def assign_ranks(tasks: Sequence[str], predicted_demand: Mapping[str, float], *, method: str,
                 oracle_demand: Mapping[str, float] | None = None) -> dict[str, int]:
    tasks = tuple(tasks)
    if method not in METHODS:
        raise ValueError(f"unknown PDRA allocation method: {method}")
    if method == "Uniform":
        if len(tasks) not in UNIFORM or len(set(tasks)) != len(tasks) or not set(tasks).issubset(TASKS):
            raise ValueError("allocation contexts must contain unique registered 4-task or 6-task memberships")
        ranks = dict(zip(tasks, UNIFORM[len(tasks)]))
    elif method == "Oracle-Ranked":
        if oracle_demand is None:
            raise ValueError("Oracle-Ranked requires explicit same-split oracle demand values")
        order = _demand_order(tasks, oracle_demand, descending=True)
        ranks = dict(zip(order, TEMPLATES[len(tasks)]))
    else:
        descending = method == "PDRA"
        order = _demand_order(tasks, predicted_demand, descending=descending)
        ranks = dict(zip(order, TEMPLATES[len(tasks)]))
    budget = 32 if len(tasks) == 4 else 48
    if sum(ranks.values()) != budget or not set(ranks.values()).issubset(RANK_LEVELS):
        raise AssertionError(f"{method} allocation violates its frozen rank budget")
    return ranks


def captured_gain(pdra: float, uniform: float, oracle: float, *, tolerance: float = 1e-12) -> dict[str, Any]:
    headroom = oracle - uniform
    improvement = pdra - uniform
    if abs(headroom) <= tolerance:
        return {"value": None, "status": "zero_headroom", "headroom_pp": headroom}
    if headroom < 0:
        return {"value": None, "status": "negative_headroom", "headroom_pp": headroom}
    return {"value": improvement / headroom, "status": "positive_headroom", "headroom_pp": headroom}


def summarize_context(task_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in task_rows:
        grouped[str(row["method"])].append(float(row["retention_percent"]))
    if set(grouped) != set(METHODS) or len({len(values) for values in grouped.values()}) != 1:
        raise ValueError("context summary needs the same task cells for all four methods")
    means = {method: math.fsum(grouped[method]) / len(grouped[method]) for method in METHODS}
    worst = {method: min(grouped[method]) for method in METHODS}
    gain = captured_gain(means["PDRA"], means["Uniform"], means["Oracle-Ranked"])
    return {
        "mean_retention_percent": means,
        "worst_task_retention_percent": worst,
        "pdra_minus_uniform_pp": means["PDRA"] - means["Uniform"],
        "pdra_minus_reverse_pp": means["PDRA"] - means["Reverse-PDRA"],
        "oracle_headroom_pp": gain["headroom_pp"],
        "captured_gain": gain["value"],
        "captured_gain_status": gain["status"],
    }


def summarize_paired(context_rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    values = [float(row[metric]) for row in context_rows]
    if not values:
        raise ValueError("paired summary needs at least one context")
    return {"context_count": len(values), "mean_paired_improvement_pp": math.fsum(values) / len(values),
            "win_rate": sum(value > 0 for value in values) / len(values),
            "tie_rate": sum(value == 0 for value in values) / len(values)}


def make_oracle_sweep_units(contexts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    units = []
    for context in contexts:
        tasks = tuple(context["tasks"])
        configurations: list[tuple[str | None, int | None]] = [(None, None)]
        configurations.extend((task, rank) for task in tasks for rank in RANK_LEVELS[:-1])
        for focal, rank in configurations:
            ranks = {task: 16 for task in tasks}
            if focal is not None:
                ranks[focal] = int(rank)
            token = "baseline" if focal is None else f"{focal}-rank{rank}"
            units.append({"run_id": f"{context['context_id']}__oracle_sweep__{token}",
                "kind": "oracle_rank_sweep", "context_id": context["context_id"], "tasks": list(tasks),
                "focal_task": focal, "level": rank, "rank_by_task": ranks,
                "scale_by_task": {task: 1.0 for task in tasks},
                "density_by_task": {task: 1.0 for task in tasks}, "resource": "rank"})
    return units


def oracle_demands_from_sweep(contexts: Sequence[Mapping[str, Any]], sweep_results: Mapping[str, Mapping[str, Any]],
                              references: Mapping[str, float]) -> dict[str, dict[str, float]]:
    demands: dict[str, dict[str, float]] = {}
    for context in contexts:
        context_id = context["context_id"]
        rows = [row for row in make_oracle_sweep_units([context])]
        by_focal: dict[str, dict[int, float]] = {task: {} for task in context["tasks"]}
        for unit in rows:
            result = sweep_results[unit["run_id"]]
            metrics = result.get("per_task", {})
            focal = unit["focal_task"]
            if focal is None:
                for task in context["tasks"]:
                    accuracy = float(metrics[task]["accuracy_percent"])
                    by_focal[task][16] = 100.0 * accuracy / float(references[task])
            else:
                accuracy = float(metrics[focal]["accuracy_percent"])
                by_focal[focal][int(unit["level"])] = 100.0 * accuracy / float(references[focal])
        context_demands: dict[str, float] = {}
        xs = [(rank - 2) / 14 for rank in RANK_LEVELS]
        for task, curve in by_focal.items():
            if set(curve) != set(RANK_LEVELS):
                raise ValueError(f"incomplete oracle sweep for {context_id}/{task}")
            deficits = [curve[16] - curve[rank] for rank in RANK_LEVELS]
            context_demands[task] = math.fsum((xs[i + 1] - xs[i]) * (deficits[i + 1] + deficits[i]) / 2
                                               for i in range(len(xs) - 1))
        demands[context_id] = context_demands
    return demands


def allocation_units(contexts: Sequence[Mapping[str, Any]], predicted: Mapping[str, float],
                     oracle: Mapping[str, Mapping[str, float]]) -> list[dict[str, Any]]:
    units = []
    for context in contexts:
        tasks = tuple(context["tasks"])
        predicted_context = {task: float(predicted[task]) for task in tasks}
        for method in METHODS:
            ranks = assign_ranks(tasks, predicted_context, method=method,
                                 oracle_demand=oracle[context["context_id"]] if method == "Oracle-Ranked" else None)
            units.append({"run_id": f"{context['context_id']}__allocation__{method.lower().replace('-', '_')}",
                "kind": "allocation", "context_id": context["context_id"], "tasks": list(tasks),
                "method": method, "rank_by_task": ranks,
                "scale_by_task": {task: 1.0 for task in tasks},
                "density_by_task": {task: 1.0 for task in tasks},
                "resource": "allocation", "level": None, "focal_task": None})
    return units


def run_unit(output: Path, stage: str, unit: Mapping[str, Any], evaluator: Callable[[Mapping[str, Any]], Any],
             *, identity_sha256: str, resume: bool = False) -> dict[str, Any]:
    """Write a unit atomically; resume skips only hash-verified completed units."""
    root = Path(output) / stage / "runs"
    run_id = str(unit["run_id"])
    if any(ch in run_id for ch in ("/", "\\", "..")):
        raise ValueError("unsafe run id")
    result_path = root / f"{run_id}.json"
    marker_path = root / f"{run_id}.unit.json"
    if result_path.exists() or marker_path.exists():
        if not (result_path.is_file() and marker_path.is_file()):
            raise ValueError(f"incomplete unsealed run exists for {run_id}; refusing to overwrite it")
        marker = unseal(json.loads(marker_path.read_text(encoding="utf-8")))
        if marker.get("result_sha256") != hash_file(result_path) or marker.get("identity_sha256") != identity_sha256:
            raise ValueError(f"run integrity failure: {run_id}")
        old = json.loads(result_path.read_text(encoding="utf-8"))
        if old.get("unit_sha256") != sha256_json(dict(unit)):
            raise ValueError(f"run configuration changed: {run_id}")
        if not resume:
            raise FileExistsError(f"completed run exists; pass --resume to keep and skip {run_id}")
        return old
    result_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        raw = evaluator(unit)
        result = {"run_id": run_id, "unit": dict(unit), "unit_sha256": sha256_json(dict(unit)),
                  "identity_sha256": identity_sha256, "result": raw}
        atomic_json(result_path, result)
        atomic_json(marker_path, seal({"result_sha256": hash_file(result_path), "identity_sha256": identity_sha256}))
        return result
    except Exception:
        atomic_text(result_path.parent / f"{run_id}.failure.txt", traceback.format_exc())
        raise


def _read_unit(output: Path, stage: str, run_id: str, identity_sha256: str) -> dict[str, Any]:
    result_path = Path(output) / stage / "runs" / f"{run_id}.json"
    marker_path = Path(output) / stage / "runs" / f"{run_id}.unit.json"
    if not result_path.is_file() or not marker_path.is_file():
        raise ValueError(f"missing sealed {stage} unit: {run_id}")
    marker = unseal(json.loads(marker_path.read_text(encoding="utf-8")))
    if marker != {"result_sha256": hash_file(result_path), "identity_sha256": identity_sha256}:
        raise ValueError(f"{stage} run integrity failure: {run_id}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("identity_sha256") != identity_sha256:
        raise ValueError(f"{stage} run identity mismatch: {run_id}")
    return result


def _derive_task_rows(contexts: Sequence[Mapping[str, Any]], units: Sequence[Mapping[str, Any]], output: Path,
                      stage: str, identity_sha: str, references: Mapping[str, float],
                      predicted: Mapping[str, float], oracle: Mapping[str, Mapping[str, float]]) -> list[dict[str, Any]]:
    rows = []
    for unit in units:
        if unit["kind"] != "allocation":
            continue
        result = _read_unit(output, stage, unit["run_id"], identity_sha)["result"]
        for task in unit["tasks"]:
            accuracy = float(result["per_task"][task]["accuracy_percent"])
            standalone = float(references[task])
            if not (0 <= accuracy <= 100 and 0 < standalone <= 100):
                raise ValueError(f"invalid task accuracy/reference for {task}")
            rows.append({"split": stage, "dataset_split": "val" if stage == "validation" else "test",
                "context_id": unit["context_id"], "method": unit["method"], "task": task,
                "predicted_demand": predicted[task],
                "oracle_demand_same_split_diagnostic": oracle[unit["context_id"]][task],
                "assigned_rank": unit["rank_by_task"][task], "merged_accuracy_percent": accuracy,
                "standalone_accuracy_percent": standalone, "retention_percent": 100.0 * accuracy / standalone,
                "run_id": unit["run_id"]})
    return rows


def _summarize_results(contexts: Sequence[Mapping[str, Any]], task_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_context: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in task_rows:
        by_context[row["context_id"]].append(row)
    context_summaries = []
    for context in contexts:
        context_id = context["context_id"]
        summary = summarize_context(by_context[context_id])
        context_summaries.append({"context_id": context_id, "tasks": context["tasks"], **summary})
    capture = [row["captured_gain"] for row in context_summaries if row["captured_gain"] is not None]
    return {"context_count": len(context_summaries), "contexts": context_summaries,
        "paired_pdra_vs_uniform": summarize_paired(context_summaries, "pdra_minus_uniform_pp"),
        "paired_pdra_vs_reverse": summarize_paired(context_summaries, "pdra_minus_reverse_pp"),
        "oracle_captured_gain": {"valid_context_count": len(capture),
            "mean_captured_gain": math.fsum(capture) / len(capture) if capture else None,
            "invalid_headroom_contexts": {status: sum(row["captured_gain_status"] == status for row in context_summaries)
                for status in ("zero_headroom", "negative_headroom")}},
        "oracle_diagnostic_policy": ORACLE_POLICY,
        "final_test_labels_used_for_pdra_prediction_or_assignment": False}


def validate_output_path(repo: Path, output: Path) -> Path:
    repo, output = Path(repo).resolve(), Path(output).resolve()
    if output == repo or repo in output.parents:
        raise ValueError("PDRA run output must be outside the repository")
    if not output.parent.exists():
        raise FileNotFoundError(f"output parent does not exist: {output.parent}")
    return output


def static_preflight(repo_root: Path) -> dict[str, Any]:
    """Validate frozen local artifacts and expected work; never imports the model runtime."""
    repo = Path(repo_root).resolve()
    manifest = load_context_manifest(repo / "task_demand_prediction" / MANIFEST_FILENAME)
    predictor = load_predictor_artifact(repo / "task_demand_prediction" / PREDICTOR_FILENAME)
    contexts = manifest["payload"]["fresh_contexts"]
    context_count = sum(len(rows) for rows in contexts.values())
    oracle_units = sum(len(make_oracle_sweep_units(rows)) for rows in contexts.values())
    allocations = context_count * len(METHODS)
    if context_count != 64 or oracle_units != 1240 or allocations != 256:
        raise ValueError("frozen PDRA design counts differ from the expected 64/1,240/256")
    if (ALPHA, RHO, TOP_PERCENT) != (0.8, 5.0, 0.001):
        raise ValueError("registered DC-Merge alpha/rho/top-k settings changed from the frozen experiment")
    return {"status": "passed", "inference_started": False, "gpu_required": False,
        "manifest_sha256": manifest["sha256"], "predictor_artifact_sha256": predictor["sha256"],
        "context_count": context_count, "fresh_context_counts": manifest["payload"]["fresh_context_counts"],
        "prior_reused_context_counts": manifest["payload"]["prior_reused_context_counts"],
        "allocation_units_per_split": allocations, "oracle_sweep_units_per_split": oracle_units,
        "evaluation_units_per_split": allocations + oracle_units,
        "evaluation_units_both_splits": 2 * (allocations + oracle_units),
        "rank_levels": list(RANK_LEVELS), "methods": list(METHODS), "merge_settings_changed": False,
        "oracle_policy": ORACLE_POLICY,
        "protocol_choice_to_confirm_before_gpu": ORACLE_POLICY_TOKEN}


def _register_output(output: Path, identity: Mapping[str, Any], manifest: Mapping[str, Any],
                     predictor: Mapping[str, Any]) -> str:
    output.mkdir(parents=True, exist_ok=True)
    identity_sha = sha256_json(dict(identity))
    registration = {"schema_version": 1, "identity": dict(identity), "identity_sha256": identity_sha,
                    "manifest_sha256": manifest["sha256"], "predictor_sha256": predictor["sha256"],
                    "oracle_policy": ORACLE_POLICY}
    for filename, value in (("registration.json", seal(registration)), ("context_manifest.json", manifest),
                            ("predictor.json", predictor)):
        path = output / filename
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != value:
                raise ValueError(f"existing PDRA output registration differs: {path}")
        else:
            atomic_json(path, value)
    return identity_sha


def normalize_stage(stage: str) -> tuple[str, str, str]:
    """Return the one canonical CLI name, stored folder name, and runtime split."""
    aliases = {
        "validation": ("validation", "validation", "val"),
        "final-test": ("final-test", "final_test", "test"),
        "final_test": ("final-test", "final_test", "test"),
    }
    try:
        return aliases[stage]
    except KeyError as exc:
        raise ValueError(f"unsupported PDRA stage: {stage}") from exc


def _expected_run_ids(output: Path, manifest: Mapping[str, Any] | None = None) -> set[str]:
    manifest = manifest or load_context_manifest(Path(output) / "context_manifest.json")
    contexts = manifest["payload"]["fresh_contexts"]["four_task"] + manifest["payload"]["fresh_contexts"]["six_task"]
    sweep_ids = {unit["run_id"] for unit in make_oracle_sweep_units(contexts)}
    allocation_ids = {
        f"{context['context_id']}__allocation__{method.lower().replace('-', '_')}"
        for context in contexts for method in METHODS
    }
    return sweep_ids | allocation_ids


def _verify_completed_stage(output: Path, stage: str, identity_sha: str, *, recover_missing_done: bool = False) -> dict[str, Any]:
    complete_path = Path(output) / stage / "COMPLETE.json"
    if not complete_path.is_file():
        raise ValueError(f"{stage} stage is not complete")
    completion = unseal(json.loads(complete_path.read_text(encoding="utf-8")))
    if completion.get("identity_sha256") != identity_sha or completion.get("stage") != stage:
        raise ValueError(f"{stage} completion identity differs")
    status_path = Path(output) / "status" / f"{stage}.DONE.json"
    expected_status = {"identity_sha256": identity_sha, "completion_sha256": hash_file(complete_path)}
    if status_path.is_file():
        status = unseal(json.loads(status_path.read_text(encoding="utf-8")))
        if status != expected_status:
            raise ValueError(f"{stage} DONE status does not match completion")
    elif not recover_missing_done:
        raise ValueError(f"{stage} DONE status is missing")
    running_path = Path(output) / "status" / f"{stage}.RUNNING.json"
    if running_path.is_file():
        running = unseal(json.loads(running_path.read_text(encoding="utf-8")))
        if running != {"stage": stage, "identity_sha256": identity_sha, "inference_started": True}:
            raise ValueError(f"{stage} RUNNING status does not match the registered stage")

    manifest = load_context_manifest(Path(output) / "context_manifest.json")
    predictor = load_predictor_artifact(Path(output) / "predictor.json")
    registration = unseal(json.loads((Path(output) / "registration.json").read_text(encoding="utf-8")))
    if (registration.get("identity_sha256") != identity_sha
            or sha256_json(registration.get("identity")) != identity_sha
            or registration.get("manifest_sha256") != manifest["sha256"]
            or registration.get("predictor_sha256") != predictor["sha256"]
            or registration.get("oracle_policy") != ORACLE_POLICY):
        raise ValueError(f"{stage} output registration is invalid")
    if (completion.get("context_manifest_sha256") != manifest["sha256"]
            or completion.get("predictor_sha256") != predictor["sha256"]
            or completion.get("oracle_policy") != ORACLE_POLICY
            or completion.get("final_test_labels_used_for_pdra_prediction_or_assignment") is not False
            or completion.get("test_labels_used_for_oracle_ranked_diagnostic") != (stage == "final_test")):
        raise ValueError(f"{stage} completion provenance differs from its registered artifacts")

    expected_ids = _expected_run_ids(output, manifest)
    run_hashes = completion.get("run_result_sha256")
    if not isinstance(run_hashes, dict) or set(run_hashes) != expected_ids:
        raise ValueError(f"{stage} completion does not list the exact expected run set")
    if (completion.get("configuration_count") != len(expected_ids)
            or completion.get("allocation_unit_count") != len(expected_ids) - 1240
            or completion.get("oracle_sweep_unit_count") != 1240
            or completion.get("status") != "complete"):
        raise ValueError(f"{stage} completion counts or status are invalid")

    # Verify all data before considering the status marker. This permits a narrowly
    # scoped recovery when a process stopped after sealing COMPLETE but before DONE.
    for run_id, expected_hash in run_hashes.items():
        result_path = Path(output) / stage / "runs" / f"{run_id}.json"
        if not result_path.is_file() or hash_file(result_path) != expected_hash:
            raise ValueError(f"verified {stage} result changed: {run_id}")
        marker_path = Path(output) / stage / "runs" / f"{run_id}.unit.json"
        if not marker_path.is_file() or unseal(json.loads(marker_path.read_text(encoding="utf-8"))) != {
                "result_sha256": expected_hash, "identity_sha256": identity_sha}:
            raise ValueError(f"verified {stage} unit marker changed: {run_id}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        unit = result.get("unit")
        if (result.get("run_id") != run_id or result.get("identity_sha256") != identity_sha
                or not isinstance(unit, dict) or unit.get("run_id") != run_id
                or result.get("unit_sha256") != sha256_json(unit)):
            raise ValueError(f"verified {stage} run record is invalid: {run_id}")
    for filename, expected_hash in (("task_results.csv", completion.get("task_results_sha256")),
                                    ("summary.json", completion.get("summary_sha256")),
                                    ("oracle_demand_same_split.json", completion.get("oracle_demand_sha256"))):
        path = Path(output) / stage / filename
        if not path.is_file() or hash_file(path) != expected_hash:
            raise ValueError(f"verified {stage} analysis changed: {filename}")

    if not status_path.is_file() and recover_missing_done:
        # Callers hold output_lock. Never replace an existing (possibly tampered) marker.
        atomic_json(status_path, seal(expected_status))
    if running_path.is_file():
        # A verified completion makes its matching RUNNING marker stale.
        running_path.unlink()
    return completion


def _verify_validation_completion(output: Path, identity_sha: str) -> None:
    try:
        _verify_completed_stage(output, "validation", identity_sha, recover_missing_done=True)
    except ValueError as exc:
        raise ValueError(f"final-test blocked: {exc}") from exc


def _load_server_preflight(args, stage: str):
    from pairwise_rank_density.preflight import preflight, resolve_paths

    if not getattr(args, "output_dir", None):
        raise ValueError("full PDRA preflight/run requires an explicit external --output-dir")
    validate_output_path(args.repo_root, args.output_dir)
    paths = resolve_paths(args)
    identity, operational, references = preflight(paths, stage=stage, device=args.device)
    predictor = load_predictor_artifact(Path(args.repo_root).resolve() / "task_demand_prediction" / PREDICTOR_FILENAME)
    source = predictor["payload"]["source"]
    identity_checks = {
        "adapter_model_sha256": source["source_adapter_model_sha256"],
        "adapter_config_sha256": source["source_adapter_config_sha256"],
        "head_sha256": source["source_head_sha256"],
        "model_files": source["source_model_files"],
        "validation_reference_sha256": source["source_validation_reference_sha256"],
    }
    mismatches = [key for key, expected in identity_checks.items() if identity.get(key) != expected]
    current_dataset = identity.get("dataset", {})
    for key, expected in source["source_dataset_identity"].items():
        if current_dataset.get(key) != expected:
            mismatches.append(f"dataset.{key}")
    if mismatches:
        raise ValueError("PDRA predictor provenance inputs differ from the previous validation experiment: " + ", ".join(mismatches))
    pdra_code = [Path(__file__), Path(__file__).with_name("__main__.py"), Path(__file__).with_name("runtime.py"),
                 Path(__file__).with_name("design.py"), Path(__file__).with_name("storage.py"),
                 Path(__file__).with_name(MANIFEST_FILENAME), Path(__file__).with_name(PREDICTOR_FILENAME)]
    identity["pdra_artifacts_sha256"] = {path.name: hash_file(path) for path in pdra_code}
    identity["pdra_manifest_sha256"] = load_context_manifest(Path(args.repo_root).resolve() / "task_demand_prediction" / MANIFEST_FILENAME)["sha256"]
    identity["pdra_predictor_sha256"] = predictor["sha256"]
    identity["pdra_oracle_policy"] = ORACLE_POLICY
    return paths, identity, operational, references, predictor


def full_preflight(args, *, stage: str = "validation") -> dict[str, Any]:
    if getattr(args, "confirm_oracle_policy", None) != ORACLE_POLICY_TOKEN:
        raise ValueError("review the same-split Oracle-Ranked policy and pass --confirm-oracle-policy same-split-posthoc")
    static = static_preflight(Path(args.repo_root))
    cli_stage, _stored_stage, _runtime_split = normalize_stage(stage)
    paths, identity, operational, _references, _predictor = _load_server_preflight(args, cli_stage)
    return {"static": static, "operational": operational, "identity_sha256": sha256_json(identity),
            "adapter_hashes_match_predictor_features": True, "output": str(paths.output),
            "oracle_policy_confirmation": ORACLE_POLICY_TOKEN,
            "inference_started": False, "status": "passed"}


def _make_real_evaluator(paths, device: str):
    from .runtime import RealEvaluator

    return RealEvaluator(paths, device)


def run_experiment(args) -> dict[str, Any]:
    """Execute one explicit stage; never performs training or downloads assets."""
    if getattr(args, "confirm_oracle_policy", None) != ORACLE_POLICY_TOKEN:
        raise ValueError("review the same-split Oracle-Ranked policy and pass --confirm-oracle-policy same-split-posthoc")
    cli_stage, stored_stage, runtime_split = normalize_stage(args.stage)
    paths, identity, _operational, references, predictor = _load_server_preflight(args, cli_stage)
    output = paths.output
    manifest = load_context_manifest(Path(args.repo_root).resolve() / "task_demand_prediction" / MANIFEST_FILENAME)
    contexts = manifest["payload"]["fresh_contexts"]["four_task"] + manifest["payload"]["fresh_contexts"]["six_task"]
    predicted = predict_demands(predictor)
    with output_lock(output):
        # Registration and the validation gate share the same lock as stage startup.
        # A concurrent validation writer therefore cannot race this check.
        identity_sha = _register_output(output, identity, manifest, predictor)
        if stored_stage == "final_test":
            _verify_validation_completion(output, identity_sha)

        complete_path = output / stored_stage / "COMPLETE.json"
        if complete_path.is_file():
            _verify_completed_stage(output, stored_stage, identity_sha, recover_missing_done=True)
            return {"status": "already_complete", "stage": stored_stage, "completion": complete_path}

        status = output / "status"
        running_path = status / f"{stored_stage}.RUNNING.json"
        expected_running = {"stage": stored_stage, "identity_sha256": identity_sha,
                            "inference_started": True}
        had_running_marker = running_path.exists()
        if had_running_marker:
            if not running_path.is_file():
                raise ValueError(f"{stored_stage} RUNNING marker is not a file; refusing to resume")
            try:
                existing_running = unseal(json.loads(running_path.read_text(encoding="utf-8")))
            except (OSError, ValueError) as exc:
                raise ValueError(f"{stored_stage} RUNNING marker is invalid; refusing to resume") from exc
            if existing_running != expected_running:
                raise ValueError(f"{stored_stage} RUNNING marker does not match the current stage identity; refusing to resume")
            if not args.resume:
                raise ValueError(f"stale {stored_stage} RUNNING marker exists; inspect it and use --resume to continue")

        evaluator = _make_real_evaluator(paths, args.device)

        def evaluate_unit(unit):
            return evaluator(unit, runtime_split)

        status.mkdir(parents=True, exist_ok=True)
        if not had_running_marker:
            atomic_json(running_path, seal(expected_running))
        try:
            if stored_stage == "validation" or ORACLE_POLICY.startswith("same-split"):
                sweep_units = make_oracle_sweep_units(contexts)
                for unit in sweep_units:
                    run_unit(output, stored_stage, unit, evaluate_unit, identity_sha256=identity_sha, resume=args.resume)
                sweep_results = {unit["run_id"]: _read_unit(output, stored_stage, unit["run_id"], identity_sha)["result"]
                                 for unit in sweep_units}
                oracle = oracle_demands_from_sweep(contexts, sweep_results, references)
                oracle_payload = {"stage": stored_stage, "oracle_policy": ORACLE_POLICY,
                    "demands": oracle, "scope": "Oracle-Ranked only; not used by PDRA predictions or PDRA/Reverse-PDRA assignments",
                    "test_labels_used_for_oracle_ranked_diagnostic": stored_stage == "final_test",
                    "identity_sha256": identity_sha}
                oracle_path = output / stored_stage / "oracle_demand_same_split.json"
                if oracle_path.exists():
                    if json.loads(oracle_path.read_text(encoding="utf-8")) != seal(oracle_payload):
                        raise ValueError("existing same-split oracle demand differs from completed rank sweeps")
                else:
                    atomic_json(oracle_path, seal(oracle_payload))
            else:
                raise ValueError("unsupported oracle protocol; PDRA runner requires same-split post-hoc sweeps")

            units = allocation_units(contexts, predicted, oracle)
            for unit in units:
                run_unit(output, stored_stage, unit, evaluate_unit, identity_sha256=identity_sha, resume=args.resume)
            task_rows = _derive_task_rows(contexts, units, output, stored_stage, identity_sha,
                                          references, predicted, oracle)
            summary = _summarize_results(contexts, task_rows)
            folder = output / stored_stage
            atomic_csv(folder / "task_results.csv", task_rows)
            atomic_json(folder / "summary.json", summary)
            run_hashes = {unit["run_id"]: hash_file(output / stored_stage / "runs" / f"{unit['run_id']}.json")
                          for unit in [*sweep_units, *units]}
            completion = {"stage": stored_stage, "identity_sha256": identity_sha,
                "context_manifest_sha256": manifest["sha256"], "predictor_sha256": predictor["sha256"],
                "oracle_policy": ORACLE_POLICY, "configuration_count": len(run_hashes),
                "allocation_unit_count": len(units), "oracle_sweep_unit_count": len(sweep_units),
                "run_result_sha256": run_hashes, "task_results_sha256": hash_file(folder / "task_results.csv"),
                "summary_sha256": hash_file(folder / "summary.json"),
                "oracle_demand_sha256": hash_file(folder / "oracle_demand_same_split.json"),
                "final_test_labels_used_for_pdra_prediction_or_assignment": False,
                "test_labels_used_for_oracle_ranked_diagnostic": stored_stage == "final_test",
                "status": "complete"}
            atomic_json(complete_path, seal(completion))
            atomic_json(status / f"{stored_stage}.DONE.json", seal({"identity_sha256": identity_sha,
                        "completion_sha256": hash_file(complete_path)}))
            (status / f"{stored_stage}.RUNNING.json").unlink(missing_ok=True)
            (status / f"{stored_stage}.FAILED.json").unlink(missing_ok=True)
            return {"status": "complete", "stage": stored_stage, "configuration_count": len(run_hashes),
                    "allocation_unit_count": len(units), "oracle_sweep_unit_count": len(sweep_units),
                    "completion": str(complete_path), "summary": str(folder / "summary.json")}
        except Exception:
            atomic_json(status / f"{stored_stage}.FAILED.json", seal({"stage": stored_stage,
                        "identity_sha256": identity_sha, "error": traceback.format_exc()}))
            raise


def write_artifact_if_frozen(path: Path, artifact: Mapping[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != artifact:
            raise FileExistsError(f"frozen artifact already exists with different content: {path}")
        return
    atomic_json(path, artifact)
