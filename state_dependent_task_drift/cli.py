from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .aggregation import aggregate_outputs
from .analysis import compute_drift, compute_static_metrics
from .config import jsonable_config, load_config
from .pairwise_merge import run_pairwise_merge
from .preflight import dataset_requirements, head_path, inspect_resources
from .run_plan import (
    enumerate_runs, ordered_task_pairs, run_pilot_baseline_dependencies,
    unordered_task_pairs,
)
from .state_io import atomic_write_json
from .training import train_run


DEFAULT_CONFIG = Path(__file__).with_name("pilot_config.json")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--pretrained-model", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--head-root", type=Path)
    parser.add_argument("--baseline-adapter-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"))
    parser.add_argument("--workers", type=int)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--seed-list", nargs="+", type=int)
    parser.add_argument("--anchor-seed", type=int)
    parser.add_argument("--anchor-alpha", type=float)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-alpha", type=float)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-existing-baseline", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m state_dependent_task_drift.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("preflight", "check exact local resources without downloading"),
        ("list-runs", "list deterministic baseline and conditional runs"),
        ("dry-run", "resolve and validate the plan without model/data/network/CUDA access"),
        ("train-baseline", "train target adapters from M0"),
        ("train-conditional", "train fresh target adapters from a frozen shifted anchor"),
        ("compute-drift", "compute matched-seed and stochastic subspace drift"),
        ("pairwise-merge", "run pairwise baseline-only DC-Merge validation selection and final test"),
        ("compute-static-metrics", "compute baseline task-to-task static metrics"),
        ("aggregate", "aggregate already-computed outputs"),
        ("run-pilot", "orchestrate the complete pilot"),
    ):
        child = sub.add_parser(name, help=help_text)
        _add_common(child)
        if name in {"train-baseline", "train-conditional", "run-pilot"}:
            child.add_argument("--target-task", choices=("stanford_cars", "eurosat", "svhn", "dtd"))
            child.add_argument("--target-seed", type=int, choices=(420, 421, 422))
        if name == "pairwise-merge":
            child.add_argument("--target-task", choices=("stanford_cars", "eurosat", "svhn", "dtd"))
        if name in {"train-conditional", "pairwise-merge", "run-pilot"}:
            child.add_argument("--anchor-task", choices=("stanford_cars", "eurosat", "svhn", "dtd"))
    return parser


def _config(args: argparse.Namespace) -> dict[str, Any]:
    overrides = {
        "repository_root": args.repository_root,
        "pretrained_model": args.pretrained_model,
        "dataset_root": args.dataset_root,
        "head_root": args.head_root,
        "baseline_adapter_root": args.baseline_adapter_root,
        "output_root": args.output_root,
        "precision": args.precision,
        "workers": args.workers,
        "tasks": args.tasks,
        "seeds": args.seed_list,
        "anchor_seed": args.anchor_seed,
        "anchor_alpha": args.anchor_alpha,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
    }
    return load_config(args.config, overrides)


def _dry_resource_report(cfg: dict[str, Any]) -> dict[str, Any]:
    expected: list[dict[str, Any]] = []
    for key in ("pretrained_model", "dataset_root", "head_root"):
        value = cfg.get(key)
        expected.append({"kind": key, "path": str(value) if value else "<not configured>", "exists": bool(value and Path(value).exists())})
    data, heads = cfg.get("dataset_root"), cfg.get("head_root")
    if data:
        for task in cfg["tasks"]:
            for path in dataset_requirements(Path(data), Path(cfg["repository_root"]), task):
                expected.append({"kind": f"dataset:{task}", "path": str(path), "exists": path.exists()})
    if heads:
        for task in cfg["tasks"]:
            path = head_path(Path(heads), task)
            expected.append({"kind": f"head:{task}", "path": str(path), "exists": path.exists()})
    return {"resources": expected, "checkpoint_contents_loaded": False}


def _selected_tasks(cfg: dict[str, Any], target: str | None) -> tuple[str, ...]:
    return (target,) if target else tuple(cfg["tasks"])


def _selected_seeds(cfg: dict[str, Any], seed: int | None) -> tuple[int, ...]:
    return (seed,) if seed is not None else tuple(cfg["seeds"])


def _train_baselines(cfg, args, *, include_run_pilot_dependencies: bool = False):
    results = []
    steps = 1 if args.smoke_test else args.max_train_steps
    batches = 1 if args.smoke_test else args.max_eval_batches
    if include_run_pilot_dependencies:
        dependencies = run_pilot_baseline_dependencies(
            cfg["tasks"], cfg["seeds"], cfg["anchor_seed"],
            args.target_task, args.anchor_task, args.target_seed,
        )
    else:
        dependencies = tuple(
            (task, seed)
            for task in _selected_tasks(cfg, args.target_task)
            for seed in _selected_seeds(cfg, args.target_seed)
        )
    for task, seed in dependencies:
        results.append(train_run(
            cfg, target_task=task, target_seed=seed, resume=args.resume,
            max_train_steps=steps, max_eval_batches=batches, device=args.device,
            reuse_existing_baseline=args.reuse_existing_baseline,
        ))
    return results


def _train_conditionals(cfg, args):
    results = []
    steps = 1 if args.smoke_test else args.max_train_steps
    batches = 1 if args.smoke_test else args.max_eval_batches
    pairs = ordered_task_pairs(cfg["tasks"])
    if args.target_task:
        pairs = tuple(pair for pair in pairs if pair[0] == args.target_task)
    if args.anchor_task:
        pairs = tuple(pair for pair in pairs if pair[1] == args.anchor_task)
    if args.target_task and args.anchor_task and args.target_task == args.anchor_task:
        raise ValueError("target_task and anchor_task must differ")
    for target, anchor in pairs:
        for seed in _selected_seeds(cfg, args.target_seed):
            results.append(train_run(
                cfg, target_task=target, target_seed=seed, anchor_task=anchor,
                resume=args.resume, max_train_steps=steps, max_eval_batches=batches,
                device=args.device, reuse_existing_baseline=args.reuse_existing_baseline,
            ))
    return results


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _config(args)
    command = args.command
    if command in {"dry-run", "list-runs"}:
        runs = enumerate_runs(cfg)
        result = {
            "command": command, "resolved_config": jsonable_config(cfg), "runs": runs,
            "run_count": len(runs), "ordered_pair_count": len(ordered_task_pairs(cfg["tasks"])),
            "unordered_pair_count": len(unordered_task_pairs(cfg["tasks"])),
            "side_effect_free": True,
        }
        if command == "dry-run":
            result["resource_preview"] = _dry_resource_report(cfg)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    output = cfg.get("output_root")
    if output is None:
        raise ValueError("output_root is required for execution commands")
    Path(output).mkdir(parents=True, exist_ok=True)
    atomic_write_json(Path(output) / "config" / "resolved_config.json", jsonable_config(cfg))
    if command == "preflight":
        result = inspect_resources(
            cfg,
            require_baselines=args.reuse_existing_baseline or cfg.get("baseline_adapter_root") is not None,
        )
        atomic_write_json(Path(output) / "state" / "preflight.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ok"] else 2
    if command == "train-baseline":
        print(json.dumps(_train_baselines(cfg, args), indent=2, sort_keys=True))
    elif command == "train-conditional":
        print(json.dumps(_train_conditionals(cfg, args), indent=2, sort_keys=True))
    elif command == "compute-drift":
        print(json.dumps(compute_drift(
            cfg, args.reuse_existing_baseline,
            max_train_steps=args.max_train_steps,
            max_eval_batches=args.max_eval_batches,
        ), indent=2, sort_keys=True))
    elif command == "pairwise-merge":
        only_pair = (args.target_task, args.anchor_task) if getattr(args, "target_task", None) and args.anchor_task else None
        print(json.dumps(run_pairwise_merge(
            cfg, device=args.device, resume=args.resume,
            max_eval_batches=1 if args.smoke_test else args.max_eval_batches,
            max_train_steps=1 if args.smoke_test else args.max_train_steps,
            reuse_existing_baseline=args.reuse_existing_baseline, only_pair=only_pair,
        ), indent=2, sort_keys=True))
    elif command == "compute-static-metrics":
        print(json.dumps(compute_static_metrics(
            cfg, args.reuse_existing_baseline,
            max_train_steps=args.max_train_steps,
            max_eval_batches=args.max_eval_batches,
        ), indent=2, sort_keys=True))
    elif command == "aggregate":
        print(json.dumps(aggregate_outputs(cfg), indent=2, sort_keys=True))
    elif command == "run-pilot":
        if (args.target_task is None) != (args.anchor_task is None):
            raise ValueError(
                "targeted run-pilot requires both --target-task A and --anchor-task B"
            )
        resources = inspect_resources(cfg, require_baselines=False)
        if not resources["ok"]:
            raise RuntimeError(f"preflight failed: {resources['missing']}")
        _train_baselines(cfg, args, include_run_pilot_dependencies=True)
        _train_conditionals(cfg, args)
        pair = (args.target_task, args.anchor_task) if args.target_task and args.anchor_task else None
        effective_steps = 1 if args.smoke_test else args.max_train_steps
        effective_batches = 1 if args.smoke_test else args.max_eval_batches
        compute_drift(
            cfg, args.reuse_existing_baseline, only_pair=pair,
            only_seed=args.target_seed if args.target_task else None,
            max_train_steps=effective_steps, max_eval_batches=effective_batches,
        )
        run_pairwise_merge(
            cfg, device=args.device, resume=args.resume,
            max_eval_batches=effective_batches, max_train_steps=effective_steps,
            reuse_existing_baseline=args.reuse_existing_baseline, only_pair=pair,
        )
        compute_static_metrics(
            cfg, args.reuse_existing_baseline, only_pair=pair,
            max_train_steps=effective_steps, max_eval_batches=effective_batches,
        )
        print(json.dumps(
            aggregate_outputs(cfg, require_all_pairs=args.target_task is None),
            indent=2, sort_keys=True,
        ))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
