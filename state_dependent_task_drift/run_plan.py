from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping


def ordered_task_pairs(tasks: Iterable[str]) -> tuple[tuple[str, str], ...]:
    values = tuple(tasks)
    return tuple((target, anchor) for target in values for anchor in values if target != anchor)


def unordered_task_pairs(tasks: Iterable[str]) -> tuple[tuple[str, str], ...]:
    return tuple(combinations(tuple(tasks), 2))


def seed_pairs(seeds: Iterable[int]) -> tuple[tuple[int, int], ...]:
    return tuple(combinations(tuple(int(seed) for seed in seeds), 2))


def alpha_token(alpha: float) -> str:
    return format(float(alpha), ".12g").replace("-", "m").replace(".", "p")


def baseline_run_id(task: str, seed: int) -> str:
    return f"baseline--target-{task}--seed-{int(seed)}"


def conditional_run_id(target: str, anchor: str, alpha: float, seed: int, anchor_seed: int) -> str:
    return (
        f"conditional--target-{target}--anchor-{anchor}--alpha-{alpha_token(alpha)}"
        f"--target-seed-{int(seed)}--anchor-seed-{int(anchor_seed)}"
    )


def baseline_run_dir(output_root: Path, task: str, seed: int) -> Path:
    return output_root / "checkpoints" / "baseline_seed_runs" / f"target_{task}" / f"seed_{seed}"


def conditional_run_dir(output_root: Path, target: str, anchor: str, alpha: float, seed: int) -> Path:
    return (
        output_root
        / "checkpoints"
        / "conditional_runs"
        / f"target_{target}"
        / f"anchor_{anchor}"
        / f"alpha_{alpha_token(alpha)}"
        / f"seed_{seed}"
    )


def enumerate_runs(cfg: Mapping[str, object]) -> list[dict[str, object]]:
    tasks = tuple(cfg["tasks"])
    seeds = tuple(cfg["seeds"])
    alpha = float(cfg["anchor_alpha"])
    anchor_seed = int(cfg["anchor_seed"])
    output_root = Path(cfg["output_root"]) if cfg.get("output_root") else Path("analysis_outputs/state_dependent_task_drift")
    runs: list[dict[str, object]] = []
    for task in tasks:
        for seed in seeds:
            runs.append({
                "kind": "baseline", "target_task": task, "target_seed": seed,
                "run_id": baseline_run_id(task, seed),
                "output_dir": str(baseline_run_dir(output_root, task, seed)),
            })
    for target, anchor in ordered_task_pairs(tasks):
        for seed in seeds:
            runs.append({
                "kind": "conditional", "target_task": target, "anchor_task": anchor,
                "anchor_alpha": alpha, "target_seed": seed, "anchor_seed": anchor_seed,
                "run_id": conditional_run_id(target, anchor, alpha, seed, anchor_seed),
                "output_dir": str(conditional_run_dir(output_root, target, anchor, alpha, seed)),
            })
    return runs


def run_pilot_baseline_dependencies(
    tasks: Iterable[str], seeds: Iterable[int], anchor_seed: int,
    target_task: str | None = None, anchor_task: str | None = None,
    target_seed: int | None = None,
) -> tuple[tuple[str, int], ...]:
    """Return ordered baseline dependencies, including B@M0 canonical anchor."""
    task_values, seed_values = tuple(tasks), tuple(int(seed) for seed in seeds)
    if target_task is None:
        return tuple((task, seed) for task in task_values for seed in seed_values)
    if target_task not in task_values:
        raise ValueError(f"unknown target task: {target_task}")
    selected_seeds = (int(target_seed),) if target_seed is not None else seed_values
    dependencies = [(target_task, seed) for seed in selected_seeds]
    canonical_target = (target_task, int(anchor_seed))
    if canonical_target not in dependencies:
        dependencies.append(canonical_target)
    anchor_tasks = (anchor_task,) if anchor_task is not None else tuple(
        task for task in task_values if task != target_task
    )
    for required_anchor in anchor_tasks:
        if required_anchor not in task_values or required_anchor == target_task:
            raise ValueError(f"invalid anchor task for {target_task}: {required_anchor}")
        anchor_dependency = (required_anchor, int(anchor_seed))
        if anchor_dependency not in dependencies:
            dependencies.append(anchor_dependency)
    return tuple(dependencies)
