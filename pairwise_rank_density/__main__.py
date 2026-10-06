"""CLI: python -m pairwise_rank_density {preflight,run,simulate}."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .design import make_design
from .preflight import preflight, resolve_paths
from .runtime import RealEvaluator
from .simulation import SimulationEvaluator
from .storage import atomic_json, sealed
from .workflow import run_stage


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Pairwise Rank & Density Demand experiment")
    sub = result.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run", "simulate"):
        item = sub.add_parser(command)
        item.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
        item.add_argument("--model-dir")
        item.add_argument("--data-dir")
        item.add_argument("--adapter-dir")
        item.add_argument("--head-dir")
        item.add_argument("--reference-dir")
        item.add_argument("--output-dir")
        item.add_argument("--min-free-gib", type=float, default=10.0)
        item.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        if command == "run":
            item.add_argument("--stage", choices=("validation", "final-test", "both"), required=True)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    design = make_design()
    if args.command == "simulate":
        output = Path(args.output_dir or "").resolve()
        if not args.output_dir:
            raise SystemExit("simulate requires --output-dir (use a temporary directory)")
        repo = Path(__file__).resolve().parents[1]
        if output == repo or repo in output.parents or output in repo.parents:
            raise SystemExit("simulation output must be outside the repository tree")
        if output.exists() and any(output.iterdir()):
            raise SystemExit("simulation output directory must be new or empty")
        identity = {"mode": "simulation", "design_sha256": __import__("pairwise_rank_density.design", fromlist=["digest"]).digest(design)}
        references = {task: 80.0 for task in design["task_order"]}
        run_stage(output, "validation", identity, references, SimulationEvaluator(), mode="simulation")
        run_stage(output, "final-test", identity, references, SimulationEvaluator(), mode="simulation")
        atomic_json(output / "SIMULATION_ONLY.json", sealed({"simulation": True, "scientific_results": False}))
        print(json.dumps({"mode": "simulation", "validation": 420, "final_test": 420,
                          "cells_per_split": 840, "output": str(output)}, indent=2))
        return 0

    paths = resolve_paths(args)
    stage = "validation" if args.command == "preflight" else ("validation" if args.stage in ("validation", "both") else "final-test")
    identity, operational, references = preflight(paths, stage=stage, device=args.device)
    print(json.dumps({"status": "preflight passed", "identity": identity,
                      "operational": operational, "design_counts": design["counts"]}, indent=2))
    if args.command == "preflight":
        return 0
    evaluator = RealEvaluator(paths, args.device)
    if args.stage in ("validation", "both"):
        run_stage(paths.output, "validation", identity, references, evaluator,
                  stage_reference_sha256=operational["stage_reference_sha256"])
    if args.stage in ("final-test", "both"):
        if args.stage == "both":
            identity, operational, references = preflight(paths, stage="final-test", device=args.device)
            test_reference_sha256 = operational["stage_reference_sha256"]
        else:
            test_reference_sha256 = operational["stage_reference_sha256"]
        run_stage(paths.output, "final-test", identity, references, evaluator,
                  stage_reference_sha256=test_reference_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
