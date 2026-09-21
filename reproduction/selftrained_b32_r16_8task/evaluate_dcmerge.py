#!/usr/bin/env python3
"""Run the authors' full validation-alpha search and final test protocol."""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from pipeline_common import ROOT, TASKS, VISION_DIR, require_environment, utc_now, write_json


CONFIG_NAME = "vitB32_r16_8task_selftrained"
SEED = 400
METHOD = "dc_merge"
SMOOTHING = "linear"
RHO = 5.0
ALPHA_CANDIDATES = [index / 10.0 for index in range(1, 31)]


def evaluate_split(model, dataloaders, names, heads, reference, split, device, evaluator):
    per_task = {}
    normalized = {}
    for index, loader_dict in enumerate(dataloaders):
        task = names[index]
        accuracy = 100.0 * evaluator(
            model.to(device), loader_dict["test"][split], heads[index].to(device)
        )
        per_task[task] = accuracy
        normalized[task] = accuracy / reference[task] * 100.0
        print(
            f"{split.upper()} TASK={task} ACCURACY={accuracy:.8f} "
            f"NORMALIZED={normalized[task]:.8f}",
            flush=True,
        )
    return {
        "accuracy_percent": per_task,
        "normalized_accuracy_percent": normalized,
        "average_accuracy_percent": sum(per_task.values()) / len(per_task),
        "average_normalized_accuracy_percent": sum(normalized.values()) / len(normalized),
    }


def load_single_task_references():
    references = {}
    for split in ("val", "test"):
        path = ROOT / "results" / f"selftrained_{split}_acc.json"
        values = json.loads(path.read_text())
        if set(values) != set(TASKS) or any(
            not isinstance(values[task], (int, float)) for task in TASKS
        ):
            raise RuntimeError(f"self-trained single-task result file is invalid: {path}")
        references[split] = values
    return references


def validate_raw_config(config, device):
    if config.get("device") != device:
        raise RuntimeError(
            f"runtime device was not injected correctly: {config.get('device')!r}"
        )

    names = [item["name"] for item in config["dataset"]]
    if tuple(names) != TASKS:
        raise RuntimeError(f"unexpected task order: {names}")

    expected_checkpoints = [ROOT / "checkpoints" / task for task in TASKS]
    actual_checkpoints = [Path(path).resolve() for path in config["model"]["bases"]]
    if actual_checkpoints != [path.resolve() for path in expected_checkpoints]:
        raise RuntimeError(f"unexpected checkpoint paths: {actual_checkpoints}")
    for task, checkpoint in zip(TASKS, actual_checkpoints):
        required = (checkpoint / "adapter_config.json", checkpoint / "adapter_model.bin")
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            raise RuntimeError(f"incomplete self-trained checkpoint for {task}: {checkpoint}")
        status = (ROOT / "state" / f"{task}.status").read_text().strip()
        if status != "DONE":
            raise RuntimeError(f"task status is not DONE for {task}: {status!r}")

    expected_heads = [
        ROOT / "heads" / "ViT-B-32" / f"{task}_head.pt" for task in TASKS
    ]
    actual_heads = [Path(item["clip_encodings"]).resolve() for item in config["dataset"]]
    if actual_heads != [path.resolve() for path in expected_heads]:
        raise RuntimeError(f"unexpected head paths: {actual_heads}")
    if any(not path.is_file() or path.stat().st_size == 0 for path in actual_heads):
        raise RuntimeError("one or more self-trained CLIP heads are missing or empty")

    return names, actual_checkpoints, actual_heads


def load_and_prepare(device, get_config_from_name, get_clip_encodings, prepare_experiment_config):
    # Match vision_lora_merge/eval.py: load by config name while injecting the
    # runtime-selected device, then prepare that augmented config.
    raw_config = get_config_from_name(CONFIG_NAME, device=device)
    names, checkpoints, head_paths = validate_raw_config(raw_config, device)
    references = load_single_task_references()
    heads = [get_clip_encodings(item["clip_encodings"]) for item in raw_config["dataset"]]
    prepared = prepare_experiment_config(raw_config)
    if len(prepared["models"]["bases"]) != len(TASKS):
        raise RuntimeError("prepare_experiment_config did not load all eight adapters")
    if len(prepared["data"]) != len(TASKS):
        raise RuntimeError("prepare_experiment_config did not load all eight datasets")
    return raw_config, prepared, heads, references, names, checkpoints, head_paths


def smoke_result(device, prepared, names, checkpoints, head_paths):
    result = {
        "validated_at": utc_now(),
        "config": CONFIG_NAME,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "tasks": names,
        "task_count": len(names),
        "checkpoint_paths": [str(path) for path in checkpoints],
        "checkpoint_count": len(checkpoints),
        "head_paths": [str(path) for path in head_paths],
        "head_count": len(head_paths),
        "task_statuses": {
            task: (ROOT / "state" / f"{task}.status").read_text().strip()
            for task in TASKS
        },
        "seed": SEED,
        "method": METHOD,
        "smoothing": SMOOTHING,
        "rho": RHO,
        "alpha_candidates": ALPHA_CANDIDATES,
        "prepared_model_count": len(prepared["models"]["bases"]),
        "prepared_dataset_count": len(prepared["data"]),
        "single_task_reference_splits": ["val", "test"],
    }
    write_json(ROOT / "results" / "dcmerge_config_smoke.json", result)
    print("CONFIG_SMOKE_TEST_PASS=true", flush=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-config",
        action="store_true",
        help="load and validate the complete model/data config, then exit before merging",
    )
    args = parser.parse_args()

    require_environment()
    os.chdir(VISION_DIR)
    from ft_handlers import aggregate_deltas, apply_merge, get_ft_parameters_delta
    from merging_functions import dc_merge
    from utils import (
        evaluate_cliphead,
        get_clip_encodings,
        get_config_from_name,
        prepare_experiment_config,
        set_seed,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required for queued DC-Merge evaluation")
    set_seed(SEED)
    started = time.monotonic()
    (
        _raw_config,
        prepared,
        heads,
        references,
        names,
        checkpoints,
        head_paths,
    ) = load_and_prepare(
        device, get_config_from_name, get_clip_encodings, prepare_experiment_config
    )
    smoke_result(device, prepared, names, checkpoints, head_paths)
    if args.smoke_config:
        return

    dataloaders = prepared["data"]
    val_reference = references["val"]
    test_reference = references["test"]

    models = [model.cpu() for model in prepared["models"]["bases"]]
    deltas = [get_ft_parameters_delta(model) for model in models]
    merged_deltas = dc_merge(
        aggregate_deltas(deltas), smoothing_strategy=SMOOTHING, rho=RHO
    )

    validations = []
    best_alpha = 0.1
    best_normalized = 0.0
    for alpha in ALPHA_CANDIDATES:
        print(f"USING_ALPHA={alpha}", flush=True)
        merged_model = apply_merge(
            prepared["models"]["new"], merged_deltas, scaling_coeffs=alpha
        )
        metrics = evaluate_split(
            merged_model,
            dataloaders,
            names,
            heads,
            val_reference,
            "val",
            device,
            evaluate_cliphead,
        )
        metrics["alpha"] = alpha
        validations.append(metrics)
        if metrics["average_normalized_accuracy_percent"] > best_normalized:
            best_normalized = metrics["average_normalized_accuracy_percent"]
            best_alpha = alpha
        del merged_model
        torch.cuda.empty_cache()

    print(f"SELECTED_ALPHA={best_alpha}", flush=True)
    merged_model = apply_merge(
        prepared["models"]["new"], merged_deltas, scaling_coeffs=best_alpha
    )
    test_metrics = evaluate_split(
        merged_model,
        dataloaders,
        names,
        heads,
        test_reference,
        "test",
        device,
        evaluate_cliphead,
    )
    result = {
        "generated_at": utc_now(),
        "config": CONFIG_NAME,
        "method": METHOD,
        "smoothing": SMOOTHING,
        "rho": RHO,
        "validation_objective": "mean normalized accuracy over eight tasks",
        "alpha_candidates": ALPHA_CANDIDATES,
        "validation_results": validations,
        "selected_alpha": best_alpha,
        "best_validation_average_normalized_accuracy_percent": best_normalized,
        "test": test_metrics,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(ROOT / "results" / "selftrained_dcmerge_results.json", result)


if __name__ == "__main__":
    main()
