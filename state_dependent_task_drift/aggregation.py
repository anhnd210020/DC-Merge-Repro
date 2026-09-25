from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .run_plan import unordered_task_pairs
from .state_io import atomic_write_csv, atomic_write_json, read_json


def build_pair_level_join(
    tasks: tuple[str, ...], state_rows: list[Mapping[str, Any]], merge_rows: list[Mapping[str, Any]],
    *, require_all_pairs: bool = True,
) -> list[dict[str, Any]]:
    expected = unordered_task_pairs(tasks)
    state_lookup = {frozenset((row["task_a"], row["task_b"])): row for row in state_rows}
    merge_lookup = {frozenset((row["task_a"], row["task_b"])): row for row in merge_rows}
    if len(state_lookup) != len(state_rows):
        raise RuntimeError("duplicate unordered pairs in symmetric state-drift rows")
    if len(merge_lookup) != len(merge_rows):
        raise RuntimeError("duplicate unordered pairs in pairwise merge rows")
    output = []
    for task_a, task_b in expected:
        key = frozenset((task_a, task_b))
        state, merge = state_lookup.get(key), merge_lookup.get(key)
        if state is None or merge is None:
            if require_all_pairs:
                raise RuntimeError(
                    f"pair-level join missing {'state drift' if state is None else 'pairwise merge'} for {task_a}, {task_b}"
                )
            continue
        if state["task_a"] == task_a:
            a_from_b, b_from_a = state["D_A_from_B_mean"], state["D_B_from_A_mean"]
        else:
            a_from_b, b_from_a = state["D_B_from_A_mean"], state["D_A_from_B_mean"]
        if merge["task_a"] == task_a:
            retention_a, retention_b = merge["Retention_A"], merge["Retention_B"]
        else:
            retention_a, retention_b = merge["Retention_B"], merge["Retention_A"]
        output.append(
            {
                "task_a": task_a, "task_b": task_b,
                "D_AB_state": state["D_AB_state"],
                "D_A_from_B_mean": a_from_b,
                "D_B_from_A_mean": b_from_a,
                "G_AB": merge["G_AB"],
                "retention_A": retention_a,
                "retention_B": retention_b,
                "selected_merge_alpha": merge["selected_alpha"],
                "canonical_pairwise_baseline_seed": merge["canonical_seed"],
            }
        )
    if require_all_pairs and len(output) != len(expected):
        raise RuntimeError(f"pair-level join produced {len(output)} rows; expected {len(expected)}")
    return output


def aggregate_outputs(cfg: Mapping[str, Any], *, require_all_pairs: bool = True) -> dict[str, Any]:
    root = Path(cfg["output_root"])
    paths = {
        "drift": root / "metrics" / "drift_summary.json",
        "static_metrics": root / "metrics" / "static_metrics" / "summary.json",
        "pairwise_merge": root / "metrics" / "pairwise_merge" / "summary.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing and require_all_pairs:
        raise FileNotFoundError(f"required aggregate inputs are missing: {missing}")
    inputs: dict[str, Any] = {
        name: read_json(path) if path.is_file() else {"status": "missing", "path": str(path)}
        for name, path in paths.items()
    }
    state_rows = inputs["drift"].get("symmetric_state_drift", [])
    merge_rows = inputs["pairwise_merge"].get("pairs", [])
    for merge in merge_rows:
        run_dir = root / "metrics" / "pairwise_merge" / f"{merge['task_a']}--{merge['task_b']}"
        status_path, result_path = run_dir / "status.json", run_dir / "result.json"
        if not status_path.is_file() or read_json(status_path).get("status") != "COMPLETE":
            raise RuntimeError(f"pairwise merge run is not COMPLETE: {run_dir}")
        if not result_path.is_file() or read_json(result_path) != merge:
            raise RuntimeError(f"pairwise merge summary/result mismatch: {run_dir}")
    joined = build_pair_level_join(
        tuple(cfg["tasks"]), state_rows, merge_rows, require_all_pairs=require_all_pairs
    )
    result = {
        "categories_kept_separate": [
            "cross_state_matched_seed_drift",
            "baseline_same_anchor_seed_drift",
            "conditional_same_anchor_seed_drift",
        ],
        "pair_level_join": joined,
        "pair_level_join_count": len(joined),
        "causal_claim": None,
        "inputs": inputs,
    }
    destination = root / "summaries"
    atomic_write_json(destination / "pair_level_state_vs_merge.json", joined)
    atomic_write_csv(destination / "pair_level_state_vs_merge.csv", joined)
    atomic_write_json(destination / "pilot_summary.json", result)
    return result
