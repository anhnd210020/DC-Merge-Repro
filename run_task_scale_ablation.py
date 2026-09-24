#!/usr/bin/env python3
"""Server entry point for the preregistered DC-Merge task-scale ablation.

This runner never trains an adapter.  Validation performs the complete alpha
sweep and freezes a manifest; final-test accepts only that manifest and never
creates or tunes a scale configuration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


REPO_DEFAULT = Path(__file__).resolve().parent
BASELINE_WEIGHT_TOLERANCE_PCT = 2e-5
BASELINE_TEST_TOLERANCE_PP = 0.05
CONFIG_NAME_DEFAULT = "vitB32_r16_8task_selftrained"
SMOOTHING = "linear"
RHO = 5.0
RANK = 16
TOP_PERCENT = 1e-3
SEED = 400


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv_atomic(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL
    ).strip()


def require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty {label}: {path}")


def require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"missing {label}: {path}")


def require_any(paths: Sequence[Path], label: str) -> None:
    if not any(path.exists() for path in paths):
        raise FileNotFoundError(
            f"missing {label}; expected one of: " + ", ".join(map(str, paths))
        )


def validate_dataset_layout(data_dir: Path) -> dict[str, list[str]]:
    """Fail before model loading if any of the eight dataset layouts is absent."""
    required_directories = {
        "stanford_cars": [
            data_dir / "cars" / "devkit",
            data_dir / "cars" / "cars_train",
            data_dir / "cars" / "cars_test",
        ],
        "dtd": [data_dir / "dtd" / split for split in ("train", "val", "test")],
        "eurosat": [
            data_dir / "eurosat" / split for split in ("train", "val", "test")
        ],
        "gtsrb": [
            data_dir / "gtsrb" / "GTSRB" / "Training",
            data_dir / "gtsrb" / "GTSRB" / "Final_Test" / "Images",
        ],
        "resisc45": [data_dir / "resisc45" / "NWPU-RESISC45"],
        "sun397": [data_dir / "sun397" / split for split in ("train", "val")],
    }
    recorded = {}
    for task, paths in required_directories.items():
        for path in paths:
            require_directory(path, f"{task} dataset directory")
        recorded[task] = [str(path.resolve()) for path in paths]
    require_file(data_dir / "gtsrb" / "GT-final_test.csv", "GTSRB labels")
    for split in ("train", "val", "test"):
        require_file(
            data_dir / "resisc45" / f"resisc45-{split}.txt",
            f"RESISC45 {split} split list",
        )
    mnist_options = [data_dir / "MNIST" / "raw", data_dir / "MNIST" / "processed"]
    svhn_options = [
        data_dir / "svhn" / "train_32x32.mat",
        data_dir / "svhn" / "test_32x32.mat",
    ]
    require_any(mnist_options, "MNIST local files")
    for path in svhn_options:
        require_file(path, "SVHN local file")
    recorded["mnist"] = [str(path.resolve()) for path in mnist_options if path.exists()]
    recorded["svhn"] = [str(path.resolve()) for path in svhn_options]
    return recorded


def configure_environment(args) -> None:
    os.environ["DCMERGE_REPO"] = str(args.repo_root)
    os.environ["DCMERGE_DATA_DIR"] = str(args.data_dir)
    os.environ["DCMERGE_MODEL_DIR"] = str(args.model_dir)
    os.environ["DCMERGE_CACHE_DIR"] = str(args.model_dir)
    os.environ["DCMERGE_SELFTRAIN_ROOT"] = str(args.artifact_root)


def preflight(args, scale_config, tasks) -> dict:
    require_directory(args.repo_root / "vision_lora_merge", "vision source tree")
    require_directory(args.model_dir, "base CLIP directory")
    require_file(args.model_dir / "config.json", "base CLIP config")
    require_file(args.model_dir / "preprocessor_config.json", "CLIP preprocessor config")
    require_any(
        [args.model_dir / "model.safetensors", args.model_dir / "pytorch_model.bin"],
        "base CLIP weights",
    )
    require_directory(args.data_dir, "dataset root")
    dataset_paths = validate_dataset_layout(args.data_dir)

    checkpoint_paths = {}
    head_paths = {}
    checkpoint_sha256 = {}
    for task in tasks:
        checkpoint = args.artifact_root / "checkpoints" / task
        config_path = checkpoint / "adapter_config.json"
        weights_path = checkpoint / "adapter_model.bin"
        require_file(config_path, f"{task} adapter config")
        require_file(weights_path, f"{task} adapter weights")
        adapter_config = read_json(config_path)
        if adapter_config.get("r") != RANK or adapter_config.get("lora_alpha") != RANK:
            raise RuntimeError(f"unexpected rank/alpha for {task}: {adapter_config}")
        checkpoint_paths[task] = str(checkpoint.resolve())
        checkpoint_sha256[task] = sha256_file(weights_path)
        head = args.artifact_root / "heads" / "ViT-B-32" / f"{task}_head.pt"
        require_file(head, f"{task} classification head")
        head_paths[task] = str(head.resolve())

    result_paths = {
        "val": args.artifact_root / "results" / "selftrained_val_acc.json",
        "test": args.artifact_root / "results" / "selftrained_test_acc.json",
    }
    require_file(result_paths["val"], "single-task validation denominators")
    if args.stage == "final-test":
        require_file(result_paths["test"], "single-task test denominators")
    for split, path in result_paths.items():
        # Validation records the test path for provenance but never reads test
        # denominators or accuracies.
        if split == "test" and args.stage != "final-test":
            continue
        values = read_json(path)
        if set(values) != set(tasks) or any(
            not isinstance(values[task], (int, float))
            or not math.isfinite(values[task])
            or values[task] <= 0.0
            for task in tasks
        ):
            raise RuntimeError(f"invalid {split} denominator file: {path}")

    vision_dir = args.repo_root / "vision_lora_merge"
    if str(vision_dir) not in sys.path:
        sys.path.insert(0, str(vision_dir))
    from utils import get_config_from_name

    old_cwd = Path.cwd()
    os.chdir(vision_dir)
    try:
        raw = get_config_from_name(args.config, device=args.device)
    finally:
        os.chdir(old_cwd)
    names = [item["name"] for item in raw["dataset"]]
    if names != list(tasks):
        raise RuntimeError(f"configured task order differs from preregistration: {names}")
    actual_checkpoints = [str(Path(path).resolve()) for path in raw["model"]["bases"]]
    if actual_checkpoints != [checkpoint_paths[task] for task in tasks]:
        raise RuntimeError("config checkpoint order/path mapping is incorrect")
    actual_heads = [str(Path(item["clip_encodings"]).resolve()) for item in raw["dataset"]]
    if actual_heads != [head_paths[task] for task in tasks]:
        raise RuntimeError("config head order/path mapping is incorrect")
    for item in raw["dataset"]:
        shuffled = item.get("shuffled_idxs")
        if shuffled:
            require_file(Path(shuffled), f"{item['name']} fixed split indices")

    import peft
    import torch
    import transformers

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA stage requested but torch.cuda.is_available() is false")
    try:
        branch = git_value(args.repo_root, "branch", "--show-current")
        commit = git_value(args.repo_root, "rev-parse", "HEAD")
        dirty = bool(git_value(args.repo_root, "status", "--porcelain"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        branch, commit, dirty = "unknown", "unknown", None

    return {
        "checked_at": utc_now(),
        "stage": args.stage,
        "config": args.config,
        "scale_config_path": str(args.scale_config.resolve()),
        "scale_config_hash": scale_config["config_hash"],
        "task_order": list(tasks),
        "checkpoint_paths": checkpoint_paths,
        "checkpoint_adapter_sha256": checkpoint_sha256,
        "head_paths": head_paths,
        "dataset_paths": dataset_paths,
        "model_path": str(args.model_dir.resolve()),
        "denominator_paths": {
            split: str(path.resolve()) for split, path in result_paths.items()
        },
        "git_branch": branch,
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "device": args.device,
        "cuda_available": torch.cuda.is_available(),
        "training_invocations": [],
    }


def load_runtime(args, tasks):
    vision_dir = args.repo_root / "vision_lora_merge"
    if str(vision_dir) not in sys.path:
        sys.path.insert(0, str(vision_dir))
    import torch
    from ft_handlers import aggregate_deltas, apply_merge, get_ft_parameters_delta
    from merging_functions import dc_merge
    from utils import (
        evaluate_cliphead,
        get_clip_encodings,
        get_config_from_name,
        prepare_experiment_config,
        set_seed,
    )

    old_cwd = Path.cwd()
    os.chdir(vision_dir)
    try:
        set_seed(SEED)
        raw = get_config_from_name(args.config, device=args.device)
        heads = [get_clip_encodings(item["clip_encodings"]) for item in raw["dataset"]]
        prepared = prepare_experiment_config(raw)
    finally:
        os.chdir(old_cwd)
    names = [item["name"] for item in raw["dataset"]]
    if names != list(tasks):
        raise RuntimeError("runtime task order changed after preflight")
    if len(prepared["models"]["bases"]) != len(tasks) or len(prepared["data"]) != len(tasks):
        raise RuntimeError("runtime did not prepare exactly eight models and datasets")
    models = [model.cpu() for model in prepared["models"]["bases"]]
    with torch.no_grad():
        deltas = [get_ft_parameters_delta(model) for model in models]
        aggregated = aggregate_deltas(deltas)
    return {
        "torch": torch,
        "apply_merge": apply_merge,
        "dc_merge": dc_merge,
        "evaluate": evaluate_cliphead,
        "prepared": prepared,
        "heads": heads,
        "aggregated": aggregated,
    }


def evaluate_split(runtime, merged_deltas, alpha, reference, split, tasks, device):
    with runtime["torch"].no_grad():
        model = runtime["apply_merge"](
            runtime["prepared"]["models"]["new"],
            merged_deltas,
            scaling_coeffs=alpha,
        )
    model.to(device)
    per_task = {}
    try:
        for index, task in enumerate(tasks):
            loader = runtime["prepared"]["data"][index]["test"][split]
            accuracy = 100.0 * runtime["evaluate"](
                model,
                loader,
                class_vectors=runtime["heads"][index].to(device),
            )
            per_task[task] = accuracy
            normalized = accuracy / reference[task] * 100.0
            print(
                f"{split.upper()} task={task} accuracy={accuracy:.8f} "
                f"normalized={normalized:.8f}",
                flush=True,
            )
    finally:
        model.cpu()
        del model
        if device == "cuda":
            runtime["torch"].cuda.empty_cache()
    from task_scale_ablation_common import summarize_split

    return summarize_split(per_task, reference, tasks)


def merged_delta_comparison(left, right) -> dict:
    if list(left) != list(right):
        raise AssertionError("merged-delta keys differ")
    maximum = 0.0
    exact = True
    for key in left:
        maximum = max(maximum, float((left[key] - right[key]).abs().max()))
        exact = exact and left[key].equal(right[key])
    if not exact and maximum > 1e-7:
        raise AssertionError(f"all-ones merge differs from default; max abs={maximum}")
    return {"exact_equal": exact, "max_absolute_difference": maximum}


def baseline_reference(repo_root: Path) -> dict:
    return read_json(repo_root / "baseline_task_scale_regression.json")


def assert_baseline_weights(diagnostic: Mapping, reference: Mapping) -> None:
    actual = diagnostic["effective_block_weight_percent"]
    expected = reference["effective_block_weight_percent"]
    errors = {task: abs(actual[task] - expected[task]) for task in expected}
    if max(errors.values()) > BASELINE_WEIGHT_TOLERANCE_PCT:
        raise AssertionError(f"baseline effective weights regressed: {errors}")


def mapping_rows(args, raw_config, scales, tasks) -> list[dict]:
    rows = []
    for index, (task, item, checkpoint) in enumerate(
        zip(tasks, raw_config["dataset"], raw_config["model"]["bases"])
    ):
        row = {
            "index": index,
            "task": task,
            "checkpoint": str(Path(checkpoint).resolve()),
            "head": str(Path(item["clip_encodings"]).resolve()),
            "dataset": item["name"],
            "dataset_root": str(args.data_dir.resolve()),
            "task_scale": scales[task],
        }
        rows.append(row)
        print(
            "MAPPING " + " ".join(f"{key}={value}" for key, value in row.items()),
            flush=True,
        )
    return rows


def validation_run(args, runtime, entry, scale_config, preflight_record, references, tasks):
    from dcmerge_effective_weights import (
        EFFECTIVE_WEIGHT_DEFINITION_VERSION,
        compute_effective_weights,
    )
    from task_scale_ablation_common import select_best_alpha, task_metric_rows

    run_id = entry["run_id"]
    scales = entry["task_scales"]
    run_dir = args.output_dir / run_id
    complete_path = run_dir / "VALIDATION_COMPLETE.json"
    run_identity = {
        "run_id": run_id,
        "task_order": list(tasks),
        "task_scales": scales,
        "ordered_scale_vector": entry["ordered_scale_vector"],
        "scale_config_hash": scale_config["config_hash"],
        "git_commit": preflight_record["git_commit"],
        "smoothing": SMOOTHING,
        "rho": RHO,
        "rank": RANK,
        "seed": SEED,
        "alpha_candidates": list(__import__("task_scale_ablation_common").ALPHA_CANDIDATES),
    }
    from task_scale_ablation_common import config_hash

    run_hash = config_hash(run_identity)
    if complete_path.exists():
        completed = read_json(complete_path)
        if completed.get("run_hash") != run_hash:
            raise RuntimeError(f"completed run hash mismatch for {run_id}")
        if not args.resume:
            raise FileExistsError(
                f"{run_id} is already complete; pass --resume to reuse it"
            )
        print(f"RESUME completed validation run={run_id}", flush=True)
        summary = read_json(run_dir / "run_summary_validation.json")
        if run_id == "baseline":
            if summary["selected_global_alpha"] != baseline_reference(args.repo_root)[
                "selected_alpha"
            ]:
                raise AssertionError("completed baseline has the wrong selected alpha")
            assert_baseline_weights(
                read_json(run_dir / "effective_weights.json"),
                baseline_reference(args.repo_root),
            )
        return summary

    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"incomplete output exists for {run_id}; inspect it and pass --resume"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    mapping = []
    for index, task in enumerate(tasks):
        row = {
            "index": index,
            "task": task,
            "checkpoint": preflight_record["checkpoint_paths"][task],
            "head": preflight_record["head_paths"][task],
            "dataset": task,
            "dataset_root": str(args.data_dir.resolve()),
            "task_scale": scales[task],
        }
        mapping.append(row)
        print(
            "MAPPING " + " ".join(f"{key}={value}" for key, value in row.items()),
            flush=True,
        )
    metadata_path = run_dir / "run_metadata.json"
    metadata = {
        **run_identity,
        "run_hash": run_hash,
        "created_at": utc_now(),
        "mapping": mapping,
        "checkpoint_sha256": preflight_record["checkpoint_adapter_sha256"],
        "dataset_paths": preflight_record["dataset_paths"],
        "effective_weight_definition_version": EFFECTIVE_WEIGHT_DEFINITION_VERSION,
        "validation_objective": "mean normalized validation accuracy over eight tasks",
        "training_invocations": [],
    }
    if metadata_path.exists():
        existing_metadata = read_json(metadata_path)
        if existing_metadata.get("run_hash") != run_hash:
            raise RuntimeError(f"incomplete run metadata hash mismatch for {run_id}")
    else:
        write_json_atomic(metadata_path, metadata)

    checkpoint_dirs = {
        task: Path(preflight_record["checkpoint_paths"][task]) for task in tasks
    }
    diagnostic_path = run_dir / "effective_weights.json"
    if diagnostic_path.exists() and args.resume:
        diagnostic = read_json(diagnostic_path)
    else:
        diagnostic = compute_effective_weights(
            checkpoint_dirs,
            tasks,
            scales,
            rank=RANK,
            rho=RHO,
            top_percent=TOP_PERCENT,
            progress=True,
        )
        write_json_atomic(diagnostic_path, diagnostic)
    if not math.isclose(
        sum(diagnostic["effective_block_weight_percent"].values()),
        100.0,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise AssertionError("realized effective weights do not sum to 100%")

    vector = entry["ordered_scale_vector"]
    with runtime["torch"].no_grad():
        merged_deltas = runtime["dc_merge"](
            runtime["aggregated"],
            smoothing_strategy=SMOOTHING,
            rho=RHO,
            task_scales=vector,
        )
    if run_id == "baseline":
        if any(value != 1.0 for value in vector):
            raise AssertionError("baseline must use all-one task scales")
        with runtime["torch"].no_grad():
            default_deltas = runtime["dc_merge"](
                runtime["aggregated"], smoothing_strategy=SMOOTHING, rho=RHO
            )
        comparison = merged_delta_comparison(default_deltas, merged_deltas)
        write_json_atomic(run_dir / "all_ones_delta_regression.json", comparison)
        assert_baseline_weights(diagnostic, baseline_reference(args.repo_root))

    partial_path = run_dir / "validation_alpha_sweep.partial.json"
    validations = read_json(partial_path) if partial_path.exists() and args.resume else []
    by_alpha = {float(item["alpha"]): item for item in validations}
    from task_scale_ablation_common import ALPHA_CANDIDATES

    for alpha in ALPHA_CANDIDATES:
        if alpha in by_alpha:
            print(f"RESUME validation run={run_id} alpha={alpha}", flush=True)
            continue
        print(f"VALIDATION run={run_id} alpha={alpha}", flush=True)
        metrics = evaluate_split(
            runtime, merged_deltas, alpha, references["val"], "val", tasks, args.device
        )
        metrics["alpha"] = alpha
        by_alpha[alpha] = metrics
        validations = [by_alpha[value] for value in ALPHA_CANDIDATES if value in by_alpha]
        write_json_atomic(partial_path, validations)
    validations = [by_alpha[alpha] for alpha in ALPHA_CANDIDATES]
    best = select_best_alpha(validations)
    selected_alpha = float(best["alpha"])
    if run_id == "baseline" and selected_alpha != baseline_reference(args.repo_root)["selected_alpha"]:
        raise AssertionError(
            f"baseline alpha regressed: expected 0.8, found {selected_alpha}"
        )
    write_json_atomic(run_dir / "validation_alpha_sweep.json", validations)
    write_json_atomic(run_dir / "selected_validation_metrics.json", best)

    rows = task_metric_rows(
        run_id,
        scales,
        diagnostic["effective_block_weight_percent"],
        references["val"],
        best["accuracy_percent"],
        selected_alpha,
        tasks=tasks,
    )
    write_json_atomic(run_dir / "task_metrics_validation.json", rows)
    write_csv_atomic(run_dir / "task_metrics_validation.csv", rows)
    summary = {
        "run_id": run_id,
        "run_hash": run_hash,
        "task_scales": scales,
        "ordered_scale_vector": vector,
        "realized_effective_weight_percent": diagnostic[
            "effective_block_weight_percent"
        ],
        "selected_global_alpha": selected_alpha,
        "validation": best,
        "effective_weight_definition_version": diagnostic["definition_version"],
    }
    write_json_atomic(run_dir / "run_summary_validation.json", summary)
    write_json_atomic(
        complete_path,
        {"completed_at": utc_now(), "run_hash": run_hash, "status": "complete"},
    )
    return summary


def create_validation_figure(output_dir: Path, summaries: Sequence[Mapping]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [summary["run_id"] for summary in summaries]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    colors = {"eurosat": "#1f77b4", "sun397": "#d95f02"}
    display = {"eurosat": "EuroSAT", "sun397": "SUN397"}
    for task in ("eurosat", "sun397"):
        x = [summary["realized_effective_weight_percent"][task] for summary in summaries]
        retention = [
            summary["validation"]["retention_percent"][task] for summary in summaries
        ]
        accuracy = [summary["validation"]["accuracy_percent"][task] for summary in summaries]
        for axis, y in zip(axes, (retention, accuracy)):
            axis.plot(x, y, marker="o", linewidth=2.0, color=colors[task], label=display[task])
            for x_value, y_value, label in zip(x, y, labels):
                axis.annotate(
                    label,
                    (x_value, y_value),
                    xytext=(4, 5),
                    textcoords="offset points",
                    fontsize=7,
                )
    axes[0].set_ylabel("Validation retention (%)")
    axes[1].set_ylabel("Validation accuracy (%)")
    for axis in axes:
        axis.set_xlabel("Realized effective weight (%)")
        axis.grid(alpha=0.25)
        axis.legend(frameon=True)
    axes[0].set_title("A. Retention dose-response", weight="bold")
    axes[1].set_title("B. Accuracy dose-response", weight="bold")
    fig.suptitle("EuroSAT / SUN397 task-scale intervention (validation only)", weight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "eurosat_sun397_validation_dose_response.png", dpi=300)
    fig.savefig(output_dir / "eurosat_sun397_validation_dose_response.pdf")
    plt.close(fig)


def freeze_manifest(args, scale_config, summaries, preflight_record, tasks) -> Path:
    from task_scale_ablation_common import config_hash

    path = args.output_dir / "selected_for_test.json"
    body = {
        "schema_version": 1,
        "created_at": utc_now(),
        "purpose": "frozen preregistered configurations approved before test evaluation",
        "scale_config_path": str(args.scale_config.resolve()),
        "scale_config_hash": scale_config["config_hash"],
        "experiment_config_hash": config_hash(
            {
                "config": args.config,
                "tasks": list(tasks),
                "smoothing": SMOOTHING,
                "rho": RHO,
                "rank": RANK,
                "seed": SEED,
                "alpha_candidates": list(__import__("task_scale_ablation_common").ALPHA_CANDIDATES),
            }
        ),
        "git_commit": preflight_record["git_commit"],
        "task_order": list(tasks),
        "validation_objective": "mean normalized validation accuracy over eight tasks",
        "approved_runs": [
            {
                "run_id": summary["run_id"],
                "run_hash": summary["run_hash"],
                "task_scales": summary["task_scales"],
                "ordered_scale_vector": summary["ordered_scale_vector"],
                "selected_global_alpha": summary["selected_global_alpha"],
                "validation": summary["validation"],
                "realized_effective_weight_percent": summary[
                    "realized_effective_weight_percent"
                ],
            }
            for summary in summaries
        ],
    }
    body["manifest_hash"] = config_hash(body)
    if path.exists():
        existing = read_json(path)
        comparable_existing = dict(existing)
        comparable_existing.pop("created_at", None)
        comparable_existing.pop("manifest_hash", None)
        comparable_body = dict(body)
        comparable_body.pop("created_at", None)
        comparable_body.pop("manifest_hash", None)
        if comparable_existing != comparable_body:
            raise FileExistsError(f"refusing to overwrite a different frozen manifest: {path}")
        return path
    write_json_atomic(path, body)
    return path


def run_validation(args, runtime, scale_config, preflight_record, references, tasks):
    summaries = []
    entries = scale_config["configs"]
    if entries[0]["run_id"] != "baseline":
        raise RuntimeError("baseline must be the first validation configuration")
    for entry in entries:
        summaries.append(
            validation_run(
                args, runtime, entry, scale_config, preflight_record, references, tasks
            )
        )
    rows = []
    for summary in summaries:
        validation = summary["validation"]
        rows.append(
            {
                "run_id": summary["run_id"],
                "selected_global_alpha": summary["selected_global_alpha"],
                "mean_validation_accuracy": validation["average_accuracy_percent"],
                "mean_normalized_validation_accuracy": validation[
                    "average_normalized_accuracy_percent"
                ],
                "mean_validation_retention": validation["average_retention_percent"],
                "worst_task_validation_accuracy": validation[
                    "worst_task_accuracy_percent"
                ],
                "worst_task_validation_retention": validation[
                    "worst_task_retention_percent"
                ],
            }
        )
    write_json_atomic(args.output_dir / "validation_run_summaries.json", summaries)
    write_csv_atomic(args.output_dir / "validation_run_summary.csv", rows)
    create_validation_figure(args.output_dir, summaries)
    manifest = freeze_manifest(args, scale_config, summaries, preflight_record, tasks)
    print(f"VALIDATION_COMPLETE manifest={manifest}", flush=True)


def validate_manifest(path: Path, scale_config, tasks) -> dict:
    from task_scale_ablation_common import config_hash

    manifest = read_json(path)
    supplied_hash = manifest.get("manifest_hash")
    unhashed = dict(manifest)
    unhashed.pop("manifest_hash", None)
    if supplied_hash != config_hash(unhashed):
        raise RuntimeError("selection manifest hash is invalid")
    if manifest.get("scale_config_hash") != scale_config["config_hash"]:
        raise RuntimeError("selection manifest does not match the scale config")
    if manifest.get("task_order") != list(tasks):
        raise RuntimeError("selection manifest task order is invalid")
    configured = {entry["run_id"]: entry for entry in scale_config["configs"]}
    approved = manifest.get("approved_runs")
    if not isinstance(approved, list) or not approved:
        raise RuntimeError("selection manifest approves no runs")
    approved_ids = [entry.get("run_id") for entry in approved]
    if len(approved_ids) != len(set(approved_ids)):
        raise RuntimeError("selection manifest contains duplicate run IDs")
    for entry in approved:
        run_id = entry["run_id"]
        if run_id not in configured:
            raise RuntimeError(f"manifest contains unknown run: {run_id}")
        if entry["task_scales"] != configured[run_id]["task_scales"]:
            raise RuntimeError(f"manifest scales differ for {run_id}")
        if entry.get("ordered_scale_vector") != configured[run_id]["ordered_scale_vector"]:
            raise RuntimeError(f"manifest ordered scale vector differs for {run_id}")
        alpha = entry.get("selected_global_alpha")
        if alpha not in __import__("task_scale_ablation_common").ALPHA_CANDIDATES:
            raise RuntimeError(f"manifest alpha is invalid for {run_id}: {alpha}")
    return manifest


def run_final_test(args, runtime, scale_config, preflight_record, references, tasks):
    from task_scale_ablation_common import task_metric_rows

    if args.selection_manifest is None:
        raise ValueError("--selection-manifest is required for final-test")
    manifest = validate_manifest(args.selection_manifest, scale_config, tasks)
    if (
        manifest.get("git_commit") not in (None, "unknown")
        and preflight_record.get("git_commit") != manifest["git_commit"]
    ):
        raise RuntimeError(
            "final-test source commit differs from the frozen validation manifest"
        )
    final_summaries = []
    for approved in manifest["approved_runs"]:
        run_id = approved["run_id"]
        run_dir = args.output_dir / run_id
        validation_summary_path = run_dir / "run_summary_validation.json"
        require_file(validation_summary_path, f"{run_id} frozen validation summary")
        validation_summary = read_json(validation_summary_path)
        if validation_summary["run_hash"] != approved["run_hash"]:
            raise RuntimeError(f"validation summary hash mismatch for {run_id}")
        complete_path = run_dir / "FINAL_TEST_COMPLETE.json"
        if complete_path.exists():
            completed = read_json(complete_path)
            if completed.get("manifest_hash") != manifest["manifest_hash"]:
                raise RuntimeError(f"final-test manifest hash mismatch for {run_id}")
            if not args.resume:
                raise FileExistsError(
                    f"final test for {run_id} is complete; pass --resume to reuse it"
                )
            final_summaries.append(read_json(run_dir / "run_summary_final_test.json"))
            continue
        vector = approved["ordered_scale_vector"]
        with runtime["torch"].no_grad():
            merged_deltas = runtime["dc_merge"](
                runtime["aggregated"],
                smoothing_strategy=SMOOTHING,
                rho=RHO,
                task_scales=vector,
            )
        alpha = float(approved["selected_global_alpha"])
        print(f"FINAL_TEST run={run_id} frozen_alpha={alpha}", flush=True)
        test = evaluate_split(
            runtime, merged_deltas, alpha, references["test"], "test", tasks, args.device
        )
        if run_id == "baseline":
            expected = baseline_reference(args.repo_root)["merged_test_accuracy_percent"]
            errors = {
                task: abs(test["accuracy_percent"][task] - expected[task]) for task in tasks
            }
            if max(errors.values()) > BASELINE_TEST_TOLERANCE_PP:
                raise AssertionError(f"baseline final metrics regressed: {errors}")
        rows = task_metric_rows(
            run_id,
            approved["task_scales"],
            approved["realized_effective_weight_percent"],
            references["val"],
            approved["validation"]["accuracy_percent"],
            alpha,
            single_test=references["test"],
            merged_test=test["accuracy_percent"],
            tasks=tasks,
        )
        write_json_atomic(run_dir / "task_metrics_final_test.json", rows)
        write_csv_atomic(run_dir / "task_metrics_final_test.csv", rows)
        summary = {
            **validation_summary,
            "frozen_manifest_hash": manifest["manifest_hash"],
            "test": test,
        }
        write_json_atomic(run_dir / "run_summary_final_test.json", summary)
        write_json_atomic(
            complete_path,
            {
                "completed_at": utc_now(),
                "manifest_hash": manifest["manifest_hash"],
                "status": "complete",
            },
        )
        final_summaries.append(summary)

    rows = []
    for summary in final_summaries:
        test = summary["test"]
        rows.append(
            {
                "run_id": summary["run_id"],
                "selected_global_alpha": summary["selected_global_alpha"],
                "mean_test_accuracy": test["average_accuracy_percent"],
                "mean_normalized_test_accuracy": test[
                    "average_normalized_accuracy_percent"
                ],
                "mean_test_retention": test["average_retention_percent"],
                "worst_task_test_accuracy": test["worst_task_accuracy_percent"],
                "worst_task_test_retention": test["worst_task_retention_percent"],
            }
        )
    write_json_atomic(args.output_dir / "final_test_run_summaries.json", final_summaries)
    write_csv_atomic(args.output_dir / "final_test_run_summary.csv", rows)
    print("FINAL_TEST_COMPLETE=true", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_NAME_DEFAULT)
    parser.add_argument(
        "--scale-config", type=Path, default=REPO_DEFAULT / "task_scale_configs.json"
    )
    parser.add_argument("--stage", required=True, choices=("validation", "final-test"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/analysis_outputs/task_scale_ablation"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/workspace/selftrained_b32_r16_8task"),
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/workspace/models/clip-vit-base-patch32")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("/workspace/datasets8"))
    parser.add_argument("--repo-root", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main():
    started = time.monotonic()
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.scale_config = args.scale_config.resolve()
    args.output_dir = args.output_dir.resolve()
    args.artifact_root = args.artifact_root.resolve()
    args.model_dir = args.model_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    if args.selection_manifest is not None:
        args.selection_manifest = args.selection_manifest.resolve()
    configure_environment(args)

    if str(args.repo_root) not in sys.path:
        sys.path.insert(0, str(args.repo_root))
    from task_scale_ablation_common import TASKS, load_scale_configs

    scale_config = load_scale_configs(args.scale_config, TASKS)
    if args.config != CONFIG_NAME_DEFAULT:
        raise ValueError(
            f"this controlled runner requires --config {CONFIG_NAME_DEFAULT}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preflight_record = preflight(args, scale_config, TASKS)
    preflight_record["elapsed_seconds"] = time.monotonic() - started
    write_json_atomic(
        args.output_dir / f"preflight_{args.stage.replace('-', '_')}.json",
        preflight_record,
    )
    print("PREFLIGHT_PASS=true", flush=True)
    if args.preflight_only:
        return

    references = {"val": read_json(Path(preflight_record["denominator_paths"]["val"]))}
    if args.stage == "final-test":
        references["test"] = read_json(
            Path(preflight_record["denominator_paths"]["test"])
        )
    runtime = load_runtime(args, TASKS)
    if args.stage == "validation":
        run_validation(args, runtime, scale_config, preflight_record, references, TASKS)
    else:
        run_final_test(args, runtime, scale_config, preflight_record, references, TASKS)


if __name__ == "__main__":
    main()
