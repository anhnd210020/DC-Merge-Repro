"""Validation-only Uniform-32 gate for a rank-tied post-projection budget.

This is a separate experiment.  It never opens the original PDRA output for
writing and never provides a final-test command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tarfile
import traceback
from pathlib import Path

from . import pdra
from .storage import atomic_json, hash_file, output_lock, seal, unseal

BASELINE_ARCHIVE_SHA256 = "77dd34ac25fad0fb57ddc5e2f59a37bea709c2ffb879542a07fc520b5b25d070"
GATE_NAME = "uniform32_rank_budget"


def _json_member(archive: tarfile.TarFile, names: set[str], name: str):
    if name not in names:
        raise ValueError(f"baseline archive is missing {name}")
    member = archive.extractfile(name)
    if member is None:
        raise ValueError(f"cannot read baseline archive member {name}")
    raw = member.read()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def load_baseline(archive_path: Path, manifest: dict, predictor: dict) -> dict:
    """Verify the sealed validation stage and retain only its Uniform rows."""
    if hash_file(archive_path) != BASELINE_ARCHIVE_SHA256:
        raise ValueError("baseline archive SHA-256 does not match the frozen validation backup")
    contexts = manifest["payload"]["fresh_contexts"]["four_task"]
    with tarfile.open(archive_path, "r:gz") as archive:
        names = set(archive.getnames())
        suffix = "/validation/COMPLETE.json"
        roots = [name[:-len(suffix)] for name in names if name.endswith(suffix)]
        if len(roots) != 1:
            raise ValueError("baseline archive must contain exactly one validation completion")
        root = roots[0]

        def read(name):
            return _json_member(archive, names, f"{root}/{name}")

        complete, complete_hash = read("validation/COMPLETE.json")
        done, _ = read("status/validation.DONE.json")
        registration, _ = read("registration.json")
        archived_manifest, _ = read("context_manifest.json")
        archived_predictor, _ = read("predictor.json")
        complete, done, registration = map(unseal, (complete, done, registration))
        pdra.validate_context_manifest(archived_manifest)
        pdra.validate_predictor_artifact(archived_predictor)
        if archived_manifest != manifest or archived_predictor != predictor:
            raise ValueError("baseline archive frozen artifacts differ from current repository")
        identity = registration["identity_sha256"]
        if (complete.get("stage") != "validation" or complete.get("status") != "complete"
                or complete.get("identity_sha256") != identity
                or registration.get("identity_sha256") != identity
                or pdra.sha256_json(registration.get("identity")) != identity
                or done != {"identity_sha256": identity, "completion_sha256": complete_hash}
                or complete.get("context_manifest_sha256") != manifest["sha256"]
                or complete.get("predictor_sha256") != predictor["sha256"]
                or complete.get("configuration_count") != 1496):
            raise ValueError("baseline registration, completion, and DONE markers disagree")
        hashes = complete.get("run_result_sha256")
        if not isinstance(hashes, dict) or len(hashes) != 1496:
            raise ValueError("baseline completion does not register all 1496 results")
        baseline = {}
        context_ids = {ctx["context_id"] for ctx in contexts}
        for run_id, expected_hash in hashes.items():
            result, actual_hash = read(f"validation/runs/{run_id}.json")
            marker, _ = read(f"validation/runs/{run_id}.unit.json")
            if (actual_hash != expected_hash
                    or unseal(marker) != {"result_sha256": actual_hash, "identity_sha256": identity}
                    or result.get("identity_sha256") != identity
                    or result.get("run_id") != run_id
                    or result.get("unit_sha256") != pdra.sha256_json(result.get("unit"))):
                raise ValueError(f"baseline run integrity failed: {run_id}")
            unit = result["unit"]
            if (unit.get("kind") == "allocation" and unit.get("method") == "Uniform"
                    and unit.get("context_id") in context_ids):
                baseline[unit["context_id"]] = result["result"]
        if set(baseline) != context_ids:
            raise ValueError("baseline archive does not contain all 45 four-task Uniform results")
        for name, field in (("task_results.csv", "task_results_sha256"),
                            ("summary.json", "summary_sha256"),
                            ("oracle_demand_same_split.json", "oracle_demand_sha256")):
            member = archive.extractfile(f"{root}/validation/{name}")
            if member is None or hashlib.sha256(member.read()).hexdigest() != complete.get(field):
                raise ValueError(f"baseline analysis integrity failed: {name}")
        return {"completion_sha256": complete_hash, "results": baseline,
                "archived_identity_sha256": identity}


def _gate_units(contexts):
    return [{"run_id": f"{ctx['context_id']}__uniform_rank_budget",
             "kind": "allocation", "method": "Uniform-rank-budget",
             "context_id": ctx["context_id"], "tasks": list(ctx["tasks"]),
             "rank_by_task": {task: 8 for task in ctx["tasks"]},
             "scale_by_task": {task: 1.0 for task in ctx["tasks"]},
             "density_by_task": {task: 1.0 for task in ctx["tasks"]}}
            for ctx in contexts]


def _summarize(units, new_results, baseline_results, references):
    contexts = []
    for unit in units:
        ctx = unit["context_id"]
        old = baseline_results[ctx]["per_task"]
        new = new_results[unit["run_id"]]["per_task"]
        if set(old) != set(unit["tasks"]) or set(new) != set(old):
            raise ValueError(f"context task sets differ: {ctx}")
        old_ret = {task: 100.0 * float(old[task]["accuracy_percent"]) / references[task]
                   for task in unit["tasks"]}
        new_ret = {task: 100.0 * float(new[task]["accuracy_percent"]) / references[task]
                   for task in unit["tasks"]}
        old_mean = math.fsum(old_ret.values()) / 4
        new_mean = math.fsum(new_ret.values()) / 4
        contexts.append({"context_id": ctx, "tasks": unit["tasks"],
                         "original_uniform_mean_retention_percent": old_mean,
                         "rank_budget_uniform_mean_retention_percent": new_mean,
                         "rank_budget_minus_original_pp": new_mean - old_mean,
                         "original_uniform_worst_retention_percent": min(old_ret.values()),
                         "rank_budget_uniform_worst_retention_percent": min(new_ret.values()),
                         "original_uniform_task_retention_percent": old_ret,
                         "rank_budget_uniform_task_retention_percent": new_ret})
    deltas = [row["rank_budget_minus_original_pp"] for row in contexts]
    return {"contexts": contexts, "context_count": len(contexts),
            "mean_paired_difference_pp": math.fsum(deltas) / len(deltas),
            "wins": sum(delta > 0 for delta in deltas),
            "losses": sum(delta < 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
            "mean_worst_task_difference_pp": math.fsum(
                row["rank_budget_uniform_worst_retention_percent"]
                - row["original_uniform_worst_retention_percent"] for row in contexts) / len(contexts),
            "interpretation": "validation-only Uniform comparison; no PDRA allocation or final-test evaluation"}


def _registration(args, identity, manifest, predictor, baseline):
    from vision_lora_merge import merging_functions
    import task_demand_prediction.runtime as runtime
    return {"protocol": GATE_NAME, "schema_version": 1,
            "topk_policy": "rank", "context_size": 4, "method": "Uniform",
            "allocation_ranks": [8, 8, 8, 8], "unit_count": 45,
            "baseline_archive_sha256": BASELINE_ARCHIVE_SHA256,
            "baseline_completion_sha256": baseline["completion_sha256"],
            "baseline_identity_sha256": baseline["archived_identity_sha256"],
            "manifest_sha256": manifest["sha256"], "predictor_sha256": predictor["sha256"],
            "asset_identity_sha256": pdra.sha256_json(identity),
            "operator_sha256": hash_file(Path(merging_functions.__file__)),
            "runtime_sha256": hash_file(Path(runtime.__file__)),
            "gate_sha256": hash_file(Path(__file__))}


def _verify_finished(output: Path, registration: dict, units: list[dict]):
    done = unseal(json.loads((output / "status" / f"{GATE_NAME}.DONE.json").read_text()))
    completion_path = output / "validation" / "COMPLETE.json"
    completion = unseal(json.loads(completion_path.read_text()))
    expected = {unit["run_id"] for unit in units}
    if (done != {"registration_sha256": pdra.sha256_json(registration),
                 "completion_sha256": hash_file(completion_path)}
            or completion.get("registration_sha256") != pdra.sha256_json(registration)
            or set(completion.get("run_result_sha256", {})) != expected
            or completion.get("summary_sha256") != hash_file(output / "validation" / "summary.json")):
        raise ValueError("gate completion and DONE disagree")
    for run_id, expected_hash in completion["run_result_sha256"].items():
        result = output / "validation" / "runs" / f"{run_id}.json"
        marker = output / "validation" / "runs" / f"{run_id}.unit.json"
        if (hash_file(result) != expected_hash or unseal(json.loads(marker.read_text())) !=
                {"result_sha256": expected_hash, "identity_sha256": pdra.sha256_json(registration)}):
            raise ValueError(f"gate result changed: {run_id}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run", "verify"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--baseline-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--head-dir", type=Path)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--min-free-gib", type=float, default=10)
    args = parser.parse_args(argv)
    if args.command != "preflight" and args.static_only:
        parser.error("--static-only applies to preflight")
    manifest = pdra.load_context_manifest(args.repo_root / "task_demand_prediction" / pdra.MANIFEST_FILENAME)
    predictor = pdra.load_predictor_artifact(args.repo_root / "task_demand_prediction" / pdra.PREDICTOR_FILENAME)
    contexts = manifest["payload"]["fresh_contexts"]["four_task"]
    baseline = load_baseline(args.baseline_archive, manifest, predictor)
    if args.command == "preflight" and args.static_only:
        print(json.dumps({"status": "static passed", "contexts": len(contexts),
                          "baseline_archive_sha256": BASELINE_ARCHIVE_SHA256,
                          "inference_started": False}, indent=2))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required for full preflight, run, and verify")
    pdra.validate_output_path(args.repo_root, args.output_dir)
    paths, identity, operational, references, _ = pdra._load_server_preflight(args, "validation")
    # The gate is identified separately from the original PDRA stage.
    registration = _registration(args, identity, manifest, predictor, baseline)
    units = _gate_units(contexts)
    if args.command == "preflight":
        print(json.dumps({"status": "passed", "inference_started": False,
                          "registration_sha256": pdra.sha256_json(registration),
                          "units": len(units), "operational": operational}, indent=2, default=str))
        return 0
    output = paths.output
    if args.command == "verify":
        existing = unseal(json.loads((output / "registration.json").read_text()))
        if existing != registration:
            raise ValueError("gate registration differs from current assets or code")
        _verify_finished(output, registration, units)
        print(json.dumps({"status": "verified", "units": len(units),
                          "summary": str(output / "validation" / "summary.json")}, indent=2))
        return 0
    with output_lock(output):
        reg_path = output / "registration.json"
        if reg_path.exists():
            if unseal(json.loads(reg_path.read_text())) != registration:
                raise ValueError("existing output registration differs; use a new output directory")
        else:
            if set(output.iterdir()) != {output / ".workflow.lock"}:
                raise ValueError("output directory is not empty; choose a new one")
            atomic_json(reg_path, seal(registration))
        complete_path = output / "validation" / "COMPLETE.json"
        if complete_path.exists():
            _verify_finished(output, registration, units)
            print(json.dumps({"status": "already_verified", "units": len(units)}))
            return 0
        status_dir = output / "status"
        running_path = status_dir / f"{GATE_NAME}.RUNNING.json"
        run_identity = pdra.sha256_json(registration)
        expected_running = {"protocol": GATE_NAME, "identity_sha256": run_identity}
        if running_path.exists():
            if unseal(json.loads(running_path.read_text())) != expected_running or not args.resume:
                raise ValueError("existing RUNNING marker differs or requires --resume")
        elif (output / "validation" / "runs").exists() and not args.resume:
            raise ValueError("run records exist; inspect them and pass --resume")
        from .runtime import RealEvaluator
        evaluator = RealEvaluator(paths, args.device, topk_policy="rank")
        status_dir.mkdir(exist_ok=True)
        if not running_path.exists():
            atomic_json(running_path, seal(expected_running))
        try:
            for unit in units:
                pdra.run_unit(output, "validation", unit,
                              lambda item: evaluator(item, "validation"),
                              identity_sha256=run_identity, resume=args.resume)
            results = {unit["run_id"]: pdra._read_unit(output, "validation", unit["run_id"],
                                                        run_identity)["result"] for unit in units}
            summary = _summarize(units, results, baseline["results"], references)
            summary_path = output / "validation" / "summary.json"
            if summary_path.exists():
                if json.loads(summary_path.read_text()) != summary:
                    raise ValueError("existing gate summary differs")
            else:
                atomic_json(summary_path, summary)
            complete = {"protocol": GATE_NAME, "registration_sha256": run_identity,
                        "run_result_sha256": {unit["run_id"]: hash_file(output / "validation" / "runs" /
                                                                          f"{unit['run_id']}.json") for unit in units},
                        "summary_sha256": hash_file(summary_path), "status": "complete"}
            atomic_json(complete_path, seal(complete))
            atomic_json(status_dir / f"{GATE_NAME}.DONE.json",
                        seal({"registration_sha256": run_identity,
                              "completion_sha256": hash_file(complete_path)}))
            running_path.unlink()
            (status_dir / f"{GATE_NAME}.FAILED.json").unlink(missing_ok=True)
            print(json.dumps({"status": "complete", "units": len(units),
                              "mean_paired_difference_pp": summary["mean_paired_difference_pp"],
                              "summary": str(summary_path)}, indent=2))
            return 0
        except Exception:
            atomic_json(status_dir / f"{GATE_NAME}.FAILED.json",
                        seal({"identity_sha256": run_identity, "error": traceback.format_exc()}))
            raise


if __name__ == "__main__":
    raise SystemExit(main())
