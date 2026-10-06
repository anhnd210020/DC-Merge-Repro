"""Signed demand curves, grouped inference, and leakage-aware model evaluation."""
from __future__ import annotations

import math
import random
from contextlib import contextmanager
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

RESOURCE_LEVELS = {"scale": (0.5, 0.75, 1.0, 1.25, 1.5), "rank": (2, 4, 8, 12, 16), "density": (0.25, 0.5, 0.75, 1.0)}
RESOURCE_X = {name: (lambda v, lo=levels[0], hi=levels[-1]: (float(v)-lo)/(hi-lo))
              for name, levels in RESOURCE_LEVELS.items()}


@contextmanager
def _scipy_sym_pos_compat():
    """Temporarily adapt sklearn's removed SciPy sym_pos keyword."""
    import inspect
    import scipy.linalg

    original_solve = scipy.linalg.solve
    if "sym_pos" in inspect.signature(original_solve).parameters:
        yield
        return

    def compatible_solve(a, b, *args, sym_pos=None, **kwargs):
        if sym_pos is True:
            kwargs.setdefault("assume_a", "pos")
        elif sym_pos is False:
            kwargs.setdefault("assume_a", "gen")
        return original_solve(a, b, *args, **kwargs)

    scipy.linalg.solve = compatible_solve
    try:
        yield
    finally:
        scipy.linalg.solve = original_solve


def _fit_predict_ridge(model, x_train, y_train, x_test):
    # sklearn 0.24 still passes sym_pos=True; map it to SciPy's identical assume_a='pos'.
    with _scipy_sym_pos_compat():
        model.fit(x_train, y_train)
    return model.predict(x_test)


def trapezoidal_auc(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("AUC needs equally sized x/y arrays with at least two points")
    if any(float(b) <= float(a) for a, b in zip(x, x[1:])):
        raise ValueError("resource levels must be strictly increasing")
    return float(sum((float(x[i+1])-float(x[i])) * (float(y[i+1])+float(y[i])) / 2 for i in range(len(x)-1)))


def _ranks(values: Sequence[float]) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]: j += 1
        ranks[order[i:j]] = (i + j - 1) / 2 + 1
        i = j
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2: return None
    rx, ry = _ranks(x), _ranks(y)
    if np.std(rx) == 0 or np.std(ry) == 0: return None
    return float(np.corrcoef(rx, ry)[0, 1])


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2: return None
    if np.std(x) == 0 or np.std(y) == 0: return None
    return float(np.corrcoef(x, y)[0, 1])


def build_oracle_rows(design: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]], split: str) -> list[dict[str, Any]]:
    """Store Card response curves and signed AUC over the frozen unit axis.

    For each resource, deficit is normalized retention at the maximum resource
    level minus current normalized retention. AUC is integrated over a linear
    min-max mapping to [0,1], without another interval-width divisor.
    """
    by_context = defaultdict(list)
    for config in design["configurations"]:
        result = results.get(config["run_id"])
        if result is None: continue
        metrics = result.get("metrics", {}).get(split, result.get("metrics", {}))
        by_context[config["context_id"]].append((config, metrics))
    rows = []
    for context_id, items in by_context.items():
        baselines = [(c, m) for c, m in items if c["resource"] == "baseline"]
        if len(baselines) != 1: raise ValueError(f"expected one baseline in {context_id}")
        baseline_config, baseline_metrics = baselines[0]
        baseline_result = results[baseline_config["run_id"]]
        registered_focals = list(dict.fromkeys(c["focal_task"] for c,_ in items if c["focal_task"] is not None))
        if baseline_config["phase"] == "A": registered_focals = list(baseline_config["tasks"])
        active_resources = tuple(design.get("active_resources", ("scale", "rank")))
        for focal in registered_focals:
            for resource in active_resources:
                if resource == "rank": level = 16
                elif resource == "scale": level = 1.0
                elif resource == "density": level = 1.0
                else: continue
                retention=float(baseline_metrics[focal]["normalized_retention_percent"])
                rows.append({"split":split,"context_id":context_id,"phase":baseline_config["phase"],
                    "tasks_json":__import__("json").dumps(baseline_config["tasks"],separators=(",",":")),
                    "focal_task":focal,"resource":resource,"level":level,"resource_x":RESOURCE_X[resource](level),
                    "max_resource_normalized_retention_percent":None,"normalized_retention_percent":retention,
                    "deficit_pp":None,"signed_response_change_pp":None,"run_id":baseline_config["run_id"],
                    "density_diagnostics":__import__("json").dumps(baseline_result.get("density_diagnostics_by_task",{}).get(focal,{}),sort_keys=True),
                    "auc_axis_status":"protocol_frozen"})
        for config, metrics in items:
            if config["resource"] == "baseline": continue
            result = results[config["run_id"]]
            focal = config["focal_task"]
            current = metrics[focal]["normalized_retention_percent"]
            baseline = baseline_metrics[focal]["normalized_retention_percent"]
            if config["resource"] not in active_resources: continue
            rows.append({
                "split": split, "context_id": context_id, "phase": config["phase"],
                "tasks_json": __import__("json").dumps(config["tasks"], separators=(",", ":")),
                "focal_task": focal, "resource": config["resource"], "level": config["level"],
                "resource_x": RESOURCE_X[config["resource"]](config["level"]),
                "max_resource_normalized_retention_percent": None,
                "normalized_retention_percent": current,
                "deficit_pp": None,
                "signed_response_change_pp": None,
                "run_id": config["run_id"],
                "density_diagnostics": __import__("json").dumps(result.get("density_diagnostics_by_task", {}).get(focal, {}), sort_keys=True),
                "auc_axis_status":"protocol_frozen",
            })
    groups = defaultdict(list)
    for row in rows: groups[(row["context_id"], row["focal_task"], row["resource"])].append(row)
    for (context, focal, resource), curve in groups.items():
        levels = RESOURCE_LEVELS[resource]
        have = {r["level"]: r for r in curve}
        if any(level not in have for level in levels):
            # A partial raw sweep is useful to resume, but cannot create an AUC.
            auc = None
        else:
            ordered = [have[level] for level in levels]
            maximum = float(ordered[-1]["normalized_retention_percent"])
            for row in ordered:
                row["max_resource_normalized_retention_percent"] = maximum
                row["deficit_pp"] = maximum - float(row["normalized_retention_percent"])
                row["signed_response_change_pp"] = float(row["normalized_retention_percent"]) - maximum
            auc = trapezoidal_auc([RESOURCE_X[resource](x) for x in levels], [r["deficit_pp"] for r in ordered])
        local_slope = None
        if resource == "scale" and 0.75 in have and 1.25 in have:
            local_slope = (float(have[1.25]["normalized_retention_percent"]) -
                           float(have[0.75]["normalized_retention_percent"])) / (1.25 - 0.75)
        for row in curve:
            row["demand_auc_signed_pp_resource_unit"] = auc
            row["scale_local_slope_retention_pp_per_multiplier"] = local_slope
            row["auc_response_direction"] = "maximum_resource_retention_minus_current; negative means net retention gain below maximum resource"
            row["auc_axis_normalization"] = "linear min-max mapping of tested levels to [0,1]; signed trapezoidal integral; no further interval-width division"
            row["auc_axis_status"] = "protocol_frozen"
    return sorted(rows, key=lambda r: (r["phase"], r["context_id"], r["focal_task"], r["resource"], r["resource_x"]))


def aggregate_demands(oracle_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for row in oracle_rows:
        auc = row.get("demand_auc_signed_pp_resource_unit")
        if auc is not None: values[(row["split"], row["context_id"], row["focal_task"], row["resource"])] = row
    return [{"split": s, "context_id": c, "focal_task": t, "resource": r, "demand_auc_signed": float(row["demand_auc_signed_pp_resource_unit"]),
             "oracle_status":"protocol_frozen",
             "scale_local_slope_retention_pp_per_multiplier":row.get("scale_local_slope_retention_pp_per_multiplier"),
             "demand_units": "percentage_point x resource-axis unit", "sign_preserved": True}
            for (s,c,t,r),row in sorted(values.items())]


def cross_resource_correlations(demands: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(demands); resources = tuple(dict.fromkeys(r["resource"] for r in rows))
    by_key = {(r["split"], r["context_id"], r["focal_task"], r["resource"]): r["demand_auc_signed"] for r in rows}
    out = {"global": {}, "within_context": {}}
    for i, left in enumerate(resources):
        for right in resources[i+1:]:
            pairs = [(v, by_key[k[:-1]+(right,)]) for k,v in by_key.items() if k[-1] == left and k[:-1]+(right,) in by_key]
            key = f"{left}__{right}"
            out["global"][key] = {"n": len(pairs), "spearman": spearman([p[0] for p in pairs], [p[1] for p in pairs])}
            context_values = defaultdict(lambda: ([], []))
            for k,v in by_key.items():
                if k[-1] == left and k[:-1]+(right,) in by_key:
                    a,b=context_values[(k[0], k[1])]; a.append(v); b.append(by_key[k[:-1]+(right,)])
            out["within_context"][key] = {f"{c[1]}:{c[0]}": {"n": len(a), "spearman": spearman(a,b)} for c,(a,b) in context_values.items()}
    return out


def demand_variation(demands: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Descriptive one-way task/context SS decomposition per resource."""
    rows = list(demands); out = {}
    resources = tuple(dict.fromkeys(r["resource"] for r in rows))
    for resource in resources:
        subset = [r for r in rows if r["resource"] == resource]
        vals = [float(r["demand_auc_signed"]) for r in subset]
        mean = float(np.mean(vals)) if vals else None
        ss_total = float(sum((v-mean)**2 for v in vals)) if vals else None
        groups = defaultdict(list)
        for r in subset: groups[r["focal_task"]].append(float(r["demand_auc_signed"]))
        ss_between = sum(len(v)*(float(np.mean(v))-mean)**2 for v in groups.values()) if vals else None
        ss_within = sum(sum((x-float(np.mean(v)))**2 for x in v) for v in groups.values()) if vals else None
        context_groups = defaultdict(list)
        for r in subset: context_groups[(r["focal_task"],r["context_id"])].append(float(r["demand_auc_signed"]))
        out[resource] = {"n": len(vals), "mean_signed_demand": mean, "between_task_ss": ss_between,
                         "within_task_context_ss": ss_within, "total_ss": ss_total,
                         "between_task_fraction_of_total": ss_between/ss_total if ss_total else None,
                         "task_count": len(groups), "task_context_cell_count": len(context_groups),
                         "interpretation": "descriptive decomposition of signed values; task/context rows are clustered"}
    return out


def grouped_bootstrap_spearman(x: Sequence[float], y: Sequence[float], groups: Sequence[str], *, iterations: int = 2000, seed: int = 400) -> dict[str, Any]:
    if not (len(x) == len(y) == len(groups)): raise ValueError("x/y/groups lengths differ")
    members = defaultdict(list)
    for i, group in enumerate(groups): members[str(group)].append(i)
    names = sorted(members)
    if len(names) < 2: return {"estimate": spearman(x,y), "ci95": None, "groups": len(names), "iterations": iterations}
    rng = random.Random(seed); samples=[]
    for _ in range(iterations):
        chosen = [rng.choice(names) for _ in names]
        idx = [i for name in chosen for i in members[name]]
        value = spearman([x[i] for i in idx],[y[i] for i in idx])
        if value is not None: samples.append(value)
    ci = [float(np.quantile(samples,.025)), float(np.quantile(samples,.975))] if samples else None
    return {"estimate": spearman(x,y), "ci95": ci, "groups": len(names), "iterations": iterations, "resampled_unit": "whole pair/context cluster"}


def pairwise_order_accuracy(predicted: Sequence[float], actual: Sequence[float]) -> float | None:
    if len(predicted) != len(actual) or len(actual) < 2: return None
    total = concordant = 0
    for i in range(len(actual)):
        for j in range(i+1,len(actual)):
            da, dp = actual[i]-actual[j], predicted[i]-predicted[j]
            if da == 0: continue
            total += 1
            if dp == 0: concordant += .5
            elif da*dp > 0: concordant += 1
    return concordant/total if total else None


def predictability_metrics(predicted: Sequence[float], actual: Sequence[float]) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=float); predicted = np.asarray(predicted, dtype=float)
    scale = float(np.ptp(actual))
    return {"n": len(actual), "spearman": spearman(predicted,actual), "pearson": pearson(predicted,actual),
            "normalized_demand_mae": float(np.mean(np.abs(predicted-actual))/scale) if scale else None,
            "pairwise_ordering_accuracy": pairwise_order_accuracy(predicted,actual)}


def evaluate_grouped_splits(rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str], target: str, *, group_field: str, constant_value: float = 1.0) -> dict[str, Any]:
    """LO-task and leave-group-out ridge predictions; rows must already be validation only."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.asarray([[np.nan if r.get(f) is None else float(r[f]) for f in feature_names] for r in rows])
    y = np.asarray([float(r[target]) for r in rows])
    predictions = np.full(len(rows), np.nan); constant_predictions=np.full(len(rows),np.nan); forest_predictions=np.full(len(rows),np.nan)
    folds=[]
    for held in sorted({str(r[group_field]) for r in rows}):
        test = np.asarray([i for i,r in enumerate(rows) if str(r[group_field]) == held])
        train = np.asarray([i for i in range(len(rows)) if i not in set(test)])
        if not len(train) or not len(test): continue
        available=[j for j in range(len(feature_names)) if not np.isnan(X[train,j]).all()]
        if available:
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
            pred = _fit_predict_ridge(model, X[train][:,available], y[train], X[test][:,available])
        else:
            pred=np.full(len(test),float(constant_value))
        predictions[test]=pred
        constant_predictions[test]=float(constant_value)
        if len(available)>1:
            from sklearn.ensemble import RandomForestRegressor
            imputer=SimpleImputer(strategy='median'); forest=RandomForestRegressor(n_estimators=100,min_samples_leaf=5,random_state=400,n_jobs=1)
            forest.fit(imputer.fit_transform(X[train][:,available]),y[train]); forest_predictions[test]=forest.predict(imputer.transform(X[test][:,available]))
        folds.append({"held_out": held,"features_used":[feature_names[j] for j in available],
                      "fit_status":"fit" if available else "constant_one_fallback_no_training_observations",
                      **predictability_metrics(pred,y[test])})
    ok = np.isfinite(predictions)
    return {"target": target, "features": list(feature_names), "group_field": group_field,
            "folds": folds, "aggregate_oof": predictability_metrics(predictions[ok],y[ok]) if ok.any() else None,
            "constant_demand_baseline_oof":predictability_metrics(constant_predictions[ok],y[ok]) if ok.any() else None,
            "random_forest_secondary_oof":predictability_metrics(forest_predictions[ok],y[ok]) if ok.any() and len(feature_names)>1 else None,
            "oof_actual":y[ok].tolist(),"oof_ridge_prediction":predictions[ok].tolist(),
            "training_rows": int(ok.sum()), "train_correlation_used_as_primary": False,
            "estimator":"standardized ridge alpha=1; random forest 100 trees secondary; fixed constant baseline Dhat=1"}


def evaluate_final_test_grouped(validation_rows: Sequence[Mapping[str, Any]], test_rows: Sequence[Mapping[str, Any]],
                                feature_names: Sequence[str], target: str, *, group_field: str,
                                constant_value: float = 1.0) -> dict[str, Any]:
    """Fit only on validation labels, excluding each held task/context, score final test once."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    def matrix(rows):
        return np.asarray([[np.nan if r.get(f) is None else float(r[f]) for f in feature_names] for r in rows], dtype=float)
    predictions=[]; actual=[]; groups=[]; folds=[]
    val_groups={str(r[group_field]) for r in validation_rows}
    for held in sorted({str(r[group_field]) for r in test_rows}):
        train=[r for r in validation_rows if str(r[group_field]) != held]
        test=[r for r in test_rows if str(r[group_field]) == held]
        if held not in val_groups or not train or not test: continue
        y_train=np.asarray([float(r[target]) for r in train]); y_test=np.asarray([float(r[target]) for r in test])
        available=[j for j in range(len(feature_names)) if not np.isnan(matrix(train)[:,j]).all()]
        if available:
            model=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),Ridge(alpha=1.0))
            pred = _fit_predict_ridge(model, matrix(train)[:,available], y_train, matrix(test)[:,available])
        else: pred=np.full(len(test),float(constant_value))
        predictions.extend(pred.tolist()); actual.extend(y_test.tolist()); groups.extend([held]*len(test))
        folds.append({"held_out":held,"features_used":[feature_names[j] for j in available],
                      "fit_status":"fit" if available else "constant_one_fallback_no_training_observations",
                      **predictability_metrics(pred,y_test)})
    return {"target":target,"features":list(feature_names),"group_field":group_field,"folds":folds,
            "aggregate_final_test":predictability_metrics(predictions,actual) if actual else None,
            "constant_demand_baseline":float(constant_value),"training_split":"validation only",
            "final_test_labels_used_for_fit_or_selection":False,"training_validation_groups_excluded_per_fold":True,
            "oof_actual":actual,"oof_prediction":predictions,"training_rows":len(validation_rows),
            "estimator":"standardized ridge alpha=1; fixed constant baseline Dhat=1"}


def feature_correlations(features: Sequence[Mapping[str, Any]], demands: Sequence[Mapping[str, Any]], feature_names: Sequence[str], *, iterations: int = 2000) -> list[dict[str, Any]]:
    demand_map = {(r["split"],r["context_id"],r["focal_task"],r["resource"]): r["demand_auc_signed"] for r in demands}
    feature_map = {(r["context_id"],r["focal_task"]): r for r in features}
    output=[]
    for resource in tuple(dict.fromkeys(r["resource"] for r in demands)):
        matched=[]
        for (split,context,focal,res), target in demand_map.items():
            if res != resource or (context,focal) not in feature_map: continue
            matched.append((split,context,focal,target,feature_map[(context,focal)]))
        for name in feature_names:
            values=[(x[3],x[4].get(name),x[0],x[1]) for x in matched if x[4].get(name) is not None]
            ci=grouped_bootstrap_spearman([v[1] for v in values],[v[0] for v in values],[v[3] for v in values],iterations=iterations) if len(values)>1 else {"estimate":None,"ci95":None,"groups":0,"iterations":iterations}
            output.append({"resource":resource,"feature":name,"n":len(values),"spearman":ci["estimate"],
                           "bootstrap_ci95":ci["ci95"],"bootstrap_groups":ci["groups"],
                           "bootstrap_resampled_unit":"whole context","split":values[0][2] if values else None})
    return output
