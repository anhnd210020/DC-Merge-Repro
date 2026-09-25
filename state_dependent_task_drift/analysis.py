from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .lora_delta import load_adapter
from .metrics import (
    aggregate_layer_rows,
    compare_adapter_subspaces,
    mean_and_sample_std,
    safe_metric_aggregate,
    static_module_metrics,
)
from .run_plan import conditional_run_dir, ordered_task_pairs, seed_pairs, unordered_task_pairs
from .run_validation import validate_complete_run
from .state_io import atomic_write_csv, atomic_write_json
from .training import resolve_baseline_adapter


def _conditional_adapter(
    cfg: Mapping[str, Any], target: str, anchor: str, seed: int,
    reuse_existing_baseline: bool = False,
    max_train_steps: int | None = None,
    max_eval_batches: int | None = None,
) -> Path:
    anchor_adapter = resolve_baseline_adapter(
        cfg, anchor, int(cfg["anchor_seed"]), reuse_existing_baseline,
        max_train_steps, max_eval_batches,
    )
    run_dir = conditional_run_dir(Path(cfg["output_root"]), target, anchor, cfg["anchor_alpha"], seed)
    validate_complete_run(
        cfg, run_dir, target_task=target, target_seed=seed, anchor_task=anchor,
        anchor_adapter=anchor_adapter, max_train_steps=max_train_steps,
        max_eval_batches=max_eval_batches,
    )
    return run_dir / "adapter"


def _compare(
    left: Path, right: Path, rank: int, context: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lc, ls = load_adapter(left)
    rc, rs = load_adapter(right)
    rows, summary = compare_adapter_subspaces(lc, ls, rc, rs, rank)
    return [{**context, **row} for row in rows], {**context, **summary}


def _seed_drift_summaries(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for category, keys in (
        ("baseline_same_anchor_seed_drift", ("target_task",)),
        ("conditional_same_anchor_seed_drift", ("target_task", "anchor_task")),
    ):
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for observation in observations:
            if observation["category"] == category:
                groups.setdefault(tuple(observation[key] for key in keys), []).append(observation)
        for group, values in groups.items():
            ordered = sorted(values, key=lambda row: (row["seed_left"], row["seed_right"]))
            raw = [
                {
                    "seed_left": row["seed_left"], "seed_right": row["seed_right"],
                    "macro_mean_drift": row["macro_mean_drift"],
                }
                for row in ordered
            ]
            numeric = [float(row["macro_mean_drift"]) for row in ordered if row["macro_mean_drift"] is not None]
            mean, std = mean_and_sample_std(numeric)
            summaries.append(
                {
                    "category": category, **dict(zip(keys, group)),
                    "seed_pair_observations": raw, "count": len(numeric),
                    "mean_macro_drift": mean, "sample_std_macro_drift": std,
                    "std_convention": "sample standard deviation (N-1); zero for a single observation",
                }
            )
    return summaries


def _directed_pair_aggregates(
    cfg: Mapping[str, Any], observations: list[dict[str, Any]],
    directed_pairs: tuple[tuple[str, str], ...],
) -> list[dict[str, Any]]:
    results = []
    for target, anchor in directed_pairs:
        selected = [
            row for row in observations
            if row["category"] == "cross_state_matched_seed_drift"
            and row["target_task"] == target and row["anchor_task"] == anchor
        ]
        macro = [float(row["macro_mean_drift"]) for row in selected if row["macro_mean_drift"] is not None]
        weighted = [
            float(row["energy_weighted_mean_drift"])
            for row in selected if row["energy_weighted_mean_drift"] is not None
        ]
        macro_mean, macro_std = mean_and_sample_std(macro)
        weighted_mean, weighted_std = mean_and_sample_std(weighted)
        results.append(
            {
                "target_task": target, "anchor_task": anchor,
                "anchor_alpha": cfg["anchor_alpha"],
                "target_seeds": [row["target_seed"] for row in selected],
                "seed_observation_count": len(macro),
                "macro_drift_mean": macro_mean,
                "macro_drift_sample_std": macro_std,
                "energy_weighted_drift_mean": weighted_mean,
                "energy_weighted_drift_sample_std": weighted_std,
                "std_convention": "sample standard deviation (N-1); zero for a single observation",
            }
        )
    return results


def _symmetric_state_drift(
    tasks: tuple[str, ...], directed: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    lookup = {(row["target_task"], row["anchor_task"]): row for row in directed}
    results = []
    for task_a, task_b in unordered_task_pairs(tasks):
        a_from_b, b_from_a = lookup.get((task_a, task_b)), lookup.get((task_b, task_a))
        if not a_from_b or not b_from_a:
            continue
        a_mean, b_mean = a_from_b["macro_drift_mean"], b_from_a["macro_drift_mean"]
        state = (float(a_mean) + float(b_mean)) / 2.0 if a_mean is not None and b_mean is not None else None
        results.append(
            {
                "task_a": task_a, "task_b": task_b,
                "D_A_from_B_mean": a_mean,
                "D_A_from_B_std": a_from_b["macro_drift_sample_std"],
                "D_B_from_A_mean": b_mean,
                "D_B_from_A_std": b_from_a["macro_drift_sample_std"],
                "D_AB_state": state,
            }
        )
    return results


def compute_drift(
    cfg: Mapping[str, Any], reuse_existing_baseline: bool = False, *,
    only_pair: tuple[str, str] | None = None, only_seed: int | None = None,
    max_train_steps: int | None = None, max_eval_batches: int | None = None,
) -> dict[str, Any]:
    root, rank = Path(cfg["output_root"]), int(cfg["lora_rank"])
    module_rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    directed_pairs = ordered_task_pairs(cfg["tasks"])
    if only_pair is not None:
        directed_pairs = tuple(pair for pair in directed_pairs if pair == only_pair)
        if len(directed_pairs) != 1:
            raise ValueError(f"invalid directed pilot pair: {only_pair}")
    seeds = (only_seed,) if only_seed is not None else tuple(cfg["seeds"])

    for target, anchor in directed_pairs:
        for seed in seeds:
            context = {
                "category": "cross_state_matched_seed_drift", "target_task": target,
                "anchor_task": anchor, "target_seed": seed, "anchor_seed": cfg["anchor_seed"],
                "anchor_alpha": cfg["anchor_alpha"], "seed_left": seed, "seed_right": seed,
            }
            baseline = resolve_baseline_adapter(
                cfg, target, seed, reuse_existing_baseline, max_train_steps, max_eval_batches
            )
            conditional = _conditional_adapter(
                cfg, target, anchor, seed, reuse_existing_baseline,
                max_train_steps, max_eval_batches,
            )
            rows, summary = _compare(baseline, conditional, rank, context)
            module_rows.extend(rows)
            observations.append(summary)

    stochastic_targets = tuple(dict.fromkeys(target for target, _anchor in directed_pairs))
    if only_seed is None:
        for target in stochastic_targets:
            for left_seed, right_seed in seed_pairs(cfg["seeds"]):
                context = {
                    "category": "baseline_same_anchor_seed_drift", "target_task": target,
                    "anchor_task": None, "target_seed": None, "anchor_seed": None,
                    "anchor_alpha": 0.0, "seed_left": left_seed, "seed_right": right_seed,
                }
                left = resolve_baseline_adapter(
                    cfg, target, left_seed, reuse_existing_baseline, max_train_steps, max_eval_batches
                )
                right = resolve_baseline_adapter(
                    cfg, target, right_seed, reuse_existing_baseline, max_train_steps, max_eval_batches
                )
                rows, summary = _compare(left, right, rank, context)
                module_rows.extend(rows)
                observations.append(summary)
        for target, anchor in directed_pairs:
            for left_seed, right_seed in seed_pairs(cfg["seeds"]):
                context = {
                    "category": "conditional_same_anchor_seed_drift", "target_task": target,
                    "anchor_task": anchor, "target_seed": None, "anchor_seed": cfg["anchor_seed"],
                    "anchor_alpha": cfg["anchor_alpha"], "seed_left": left_seed, "seed_right": right_seed,
                }
                left = _conditional_adapter(
                    cfg, target, anchor, left_seed, reuse_existing_baseline,
                    max_train_steps, max_eval_batches,
                )
                right = _conditional_adapter(
                    cfg, target, anchor, right_seed, reuse_existing_baseline,
                    max_train_steps, max_eval_batches,
                )
                rows, summary = _compare(left, right, rank, context)
                module_rows.extend(rows)
                observations.append(summary)

    pair_aggregates = _directed_pair_aggregates(cfg, observations, directed_pairs)
    seed_summaries = _seed_drift_summaries(observations)
    symmetric = _symmetric_state_drift(tuple(cfg["tasks"]), pair_aggregates)
    layer_rows = aggregate_layer_rows(module_rows)
    metrics = root / "metrics"
    atomic_write_json(metrics / "layer_drift" / "module_drift.json", module_rows)
    csv_rows = [dict(row) for row in module_rows]
    for row in csv_rows:
        for key in ("baseline_singular_values", "conditional_singular_values"):
            row[key] = json.dumps(row[key])
    atomic_write_csv(metrics / "layer_drift" / "module_drift.csv", csv_rows)
    atomic_write_json(metrics / "layer_drift" / "layer_aggregates.json", layer_rows)
    atomic_write_csv(metrics / "layer_drift" / "layer_aggregates.csv", layer_rows)
    atomic_write_json(metrics / "seed_drift" / "observations.json", observations)
    atomic_write_csv(metrics / "seed_drift" / "observations.csv", observations)
    atomic_write_json(metrics / "seed_drift" / "stochastic_summaries.json", seed_summaries)
    seed_csv = [{**row, "seed_pair_observations": json.dumps(row["seed_pair_observations"])} for row in seed_summaries]
    atomic_write_csv(metrics / "seed_drift" / "stochastic_summaries.csv", seed_csv)
    atomic_write_json(metrics / "pair_drift" / "matched_seed_pair_aggregates.json", pair_aggregates)
    atomic_write_csv(metrics / "pair_drift" / "matched_seed_pair_aggregates.csv", pair_aggregates)
    atomic_write_json(metrics / "pair_drift" / "symmetric_state_drift.json", symmetric)
    atomic_write_csv(metrics / "pair_drift" / "symmetric_state_drift.csv", symmetric)
    result = {
        "module_rows": len(module_rows), "layer_rows": len(layer_rows),
        "observations": len(observations), "pair_aggregates": pair_aggregates,
        "stochastic_drift_summaries": seed_summaries,
        "symmetric_state_drift": symmetric,
        "std_convention": "sample standard deviation (N-1); zero for a single observation",
        "completed_run_validation_required": True,
    }
    atomic_write_json(metrics / "drift_summary.json", result)
    return result


def compute_static_metrics(
    cfg: Mapping[str, Any], reuse_existing_baseline: bool = False, *,
    only_pair: tuple[str, str] | None = None,
    max_train_steps: int | None = None, max_eval_batches: int | None = None,
) -> dict[str, Any]:
    root = Path(cfg["output_root"]) / "metrics" / "static_metrics"
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    pairs = unordered_task_pairs(cfg["tasks"])
    if only_pair is not None:
        pairs = tuple(pair for pair in pairs if set(pair) == set(only_pair))
        if len(pairs) != 1:
            raise ValueError(f"invalid unordered pilot pair: {only_pair}")
    for left_task, right_task in pairs:
        left_path = resolve_baseline_adapter(
            cfg, left_task, int(cfg["anchor_seed"]), reuse_existing_baseline,
            max_train_steps, max_eval_batches,
        )
        right_path = resolve_baseline_adapter(
            cfg, right_task, int(cfg["anchor_seed"]), reuse_existing_baseline,
            max_train_steps, max_eval_batches,
        )
        lc, ls = load_adapter(left_path)
        rc, rs = load_adapter(right_path)
        rows = static_module_metrics(lc, ls, rc, rs, int(cfg["lora_rank"]))
        enriched = [{"task_a": left_task, "task_b": right_task, **row} for row in rows]
        all_rows.extend(enriched)
        aggregate_metrics = {
            name: safe_metric_aggregate(enriched, name)
            for name in (
                "raw_effective_delta_cosine", "raw_delta_sign_conflict",
                "left_subspace_overlap", "right_subspace_overlap",
                "left_frobenius_norm", "right_frobenius_norm", "left_energy", "right_energy",
            )
        }
        summaries.append({"task_a": left_task, "task_b": right_task, "metrics": aggregate_metrics})
    definition = {
        "raw_effective_delta_cosine": "per-module cosine after flattening scaling*(B@A); aggregate excludes unavailable/nonfinite/degenerate modules",
        "raw_delta_sign_conflict": "fraction of jointly nonzero dense effective-delta elements with unequal signs; zeros excluded",
        "subspace_overlap": "projector overlap (1/k)||U_a^T U_b||_F^2 and analogous right-space value",
        "aggregate_null_safety": "each metric records valid_count, skipped_count, degenerate_count; no valid values produce null, never zero",
        "paper_projected_DirSim": "not_implemented_exact_definition_unverified",
    }
    atomic_write_csv(root / "module_metrics.csv", all_rows)
    atomic_write_json(root / "module_metrics.json", all_rows)
    summary_csv = [
        {"task_a": row["task_a"], "task_b": row["task_b"], "metrics": json.dumps(row["metrics"], sort_keys=True)}
        for row in summaries
    ]
    atomic_write_csv(root / "pair_summaries.csv", summary_csv)
    atomic_write_json(root / "summary.json", {"definitions": definition, "pairs": summaries})
    return {"definitions": definition, "pairs": summaries}
