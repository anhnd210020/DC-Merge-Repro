from __future__ import annotations

import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch

from .anchor import canonical_hash, materialize_anchor
from .lora_delta import adapter_checksum, effective_deltas, load_adapter
from .run_plan import unordered_task_pairs
from .preflight import dataset_requirements, head_path, missing_model_resources
from .state_io import atomic_write_json, read_json, run_action, set_status
from .training import VisionClassifier, _task_config, evaluate, resolve_baseline_adapter, utc_now


def pairwise_run_definition(
    cfg: Mapping[str, Any], task_a: str, task_b: str,
    adapter_paths: Mapping[str, Path], max_eval_batches: int | None,
) -> dict[str, Any]:
    return {
        "run_kind": "pairwise_dc_merge",
        "task_a": task_a,
        "task_b": task_b,
        "canonical_seed": int(cfg["anchor_seed"]),
        "base_model_identifier": str(cfg.get("base_model_identifier", cfg["pretrained_model"])),
        "baseline_a_adapter_sha256": adapter_checksum(Path(adapter_paths[task_a])),
        "baseline_b_adapter_sha256": adapter_checksum(Path(adapter_paths[task_b])),
        "merge_configuration": {
            "implementation": "repository_dc_merge",
            "smoothing_strategy": "linear",
            "rho": 5.0,
            "task_scales": None,
            "alpha_candidates": [float(alpha) for alpha in cfg["merge_alphas"]],
            "alpha_evaluation_base_state": "clean M0 restored exactly before every candidate",
        },
        "evaluation_configuration": {
            "precision": str(cfg.get("precision", "fp32")),
            "batch_size": int(cfg.get("batch_size", 32)),
            "workers": int(cfg.get("workers", 0)),
            "max_eval_batches": max_eval_batches,
            "selection_split": "validation",
            "selection_objective": "mean normalized validation accuracy over exactly the two pair tasks",
            "test_used_for_selection": False,
        },
    }


def pairwise_run_definition_hash(definition: Mapping[str, Any]) -> str:
    return canonical_hash(definition)


def validate_complete_pairwise_run(
    run_dir: Path, expected_definition: Mapping[str, Any]
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    paths = {
        "status": run_dir / "status.json",
        "result": run_dir / "result.json",
        "frozen selection": run_dir / "frozen_selection.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"COMPLETE pairwise run is missing required outputs: {missing}")
    status = read_json(paths["status"])
    result = read_json(paths["result"])
    frozen = read_json(paths["frozen selection"])
    if status.get("status") != "COMPLETE":
        raise RuntimeError(f"pairwise run is not COMPLETE: {run_dir}")

    expected = dict(expected_definition)
    expected_hash = pairwise_run_definition_hash(expected)
    for label, artifact in (("status", status), ("result", result), ("frozen selection", frozen)):
        if artifact.get("pairwise_run_definition_hash") != expected_hash:
            raise RuntimeError(
                f"pairwise run definition is incompatible with current configuration in {label}: "
                f"stored={artifact.get('pairwise_run_definition_hash')!r}, expected={expected_hash!r}"
            )
        if artifact.get("pairwise_run_definition") != expected:
            raise RuntimeError(f"stored pairwise run definition differs in {label}: {run_dir}")

    task_a, task_b = expected["task_a"], expected["task_b"]
    selected_alpha = frozen.get("selected_alpha")
    required_values = {
        "result task_a": (result.get("task_a"), task_a),
        "result task_b": (result.get("task_b"), task_b),
        "result canonical_seed": (result.get("canonical_seed"), expected["canonical_seed"]),
        "frozen selection_tasks": (frozen.get("selection_tasks"), [task_a, task_b]),
        "frozen selection_split": (frozen.get("selection_split"), "validation"),
        "frozen selection_objective": (
            frozen.get("selection_objective"),
            expected["evaluation_configuration"]["selection_objective"],
        ),
        "result selected_alpha": (result.get("selected_alpha"), selected_alpha),
        "status selected_alpha": (status.get("selected_alpha"), selected_alpha),
        "result validation_sweep": (result.get("validation_sweep"), frozen.get("validation_sweep")),
        "result smoothing": (result.get("dc_merge_smoothing"), "linear"),
        "result rho": (result.get("dc_merge_rho"), 5.0),
        "result task_scales": (result.get("task_scales"), None),
        "frozen task_scales": (frozen.get("task_scales"), None),
        "result alpha base state": (
            result.get("alpha_evaluation_base_state"),
            expected["merge_configuration"]["alpha_evaluation_base_state"],
        ),
        "result test selection isolation": (
            result.get("test_read_only_after_selection_frozen"), True
        ),
    }
    mismatches = {
        label: {"actual": actual, "expected": wanted}
        for label, (actual, wanted) in required_values.items() if actual != wanted
    }
    candidates = expected["merge_configuration"]["alpha_candidates"]
    if selected_alpha not in candidates:
        mismatches["selected_alpha candidate"] = {
            "actual": selected_alpha, "expected": candidates
        }
    sweep_alphas = [entry.get("alpha") for entry in frozen.get("validation_sweep", [])]
    if sweep_alphas != candidates:
        mismatches["validation sweep alphas"] = {
            "actual": sweep_alphas, "expected": candidates
        }
    for label, artifact, key in (
        ("result", result, "task_scales"),
        ("frozen selection", frozen, "task_scales"),
    ):
        if key not in artifact:
            mismatches[f"{label} {key} presence"] = {"actual": "missing", "expected": None}
    if mismatches:
        raise RuntimeError(f"inconsistent COMPLETE pairwise artifacts for {run_dir}: {mismatches}")
    return {"status": status, "result": result, "frozen_selection": frozen}


def capture_clean_target_weights(
    model: torch.nn.Module, deltas: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    mapped = materialize_anchor(model, deltas, 0.0)
    parameters = dict(model.named_parameters())
    return {name: parameters[name].detach().clone() for name in mapped.values()}


def restore_clean_target_weights(
    model: torch.nn.Module, clean_weights: Mapping[str, torch.Tensor]
) -> None:
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in clean_weights.items():
            if name not in parameters:
                raise KeyError(f"clean target weight no longer exists: {name}")
            parameters[name].copy_(value.to(device=parameters[name].device, dtype=parameters[name].dtype))


def materialize_from_clean(
    model: torch.nn.Module, clean_weights: Mapping[str, torch.Tensor],
    deltas: Mapping[str, torch.Tensor], alpha: float,
) -> None:
    restore_clean_target_weights(model, clean_weights)
    materialize_anchor(model, deltas, alpha)


def _build_runtime(cfg: Mapping[str, Any], tasks: tuple[str, str], device: str):
    from transformers import CLIPModel
    try:
        from transformers import CLIPImageProcessor as ImageProcessor
    except ImportError:
        from transformers import CLIPFeatureExtractor as ImageProcessor

    required = []
    for task in tasks:
        required.extend(dataset_requirements(Path(cfg["dataset_root"]), Path(cfg["repository_root"]), task))
        required.append(head_path(Path(cfg["head_root"]), task))
    model_root = Path(cfg["pretrained_model"])
    missing = [str(path) for path in required if not path.exists()]
    missing.extend(missing_model_resources(model_root))
    if missing:
        raise FileNotFoundError("required local resources are missing; refusing auto-download: " + ", ".join(missing))
    model_path = str(cfg["pretrained_model"])
    clip = CLIPModel.from_pretrained(model_path, local_files_only=True)
    processor = ImageProcessor.from_pretrained(model_path, local_files_only=True)
    for parameter in clip.parameters():
        parameter.requires_grad_(False)
    model = VisionClassifier(clip.vision_model, clip.visual_projection).to(device).eval()
    preprocess = lambda image: processor(image, return_tensors="pt")
    repo = Path(cfg["repository_root"])
    vision_dir = repo / "vision_lora_merge"
    if str(vision_dir) not in os.sys.path:
        os.sys.path.insert(0, str(vision_dir))
    from utils import prepare_data

    loaders, heads = {}, {}
    for task in tasks:
        data_cfg = _task_config(cfg, task, preprocess, preprocess)
        loaders[task] = prepare_data(data_cfg, device=device)["test"]
        heads[task] = torch.load(data_cfg["clip_encodings"], map_location=device, weights_only=True)
    return model, loaders, heads


def _evaluate_pair(model, loaders, heads, tasks, split, device, max_batches, references=None):
    accuracies = {task: evaluate(model, loaders[task][split], heads[task], device, max_batches) for task in tasks}
    normalized = None
    if references is not None:
        normalized = {task: accuracies[task] / references[task] if references[task] else None for task in tasks}
    return {
        "accuracy": accuracies,
        "normalized_accuracy": normalized,
        "mean_accuracy": sum(accuracies.values()) / len(tasks),
        "mean_normalized_accuracy": sum(normalized.values()) / len(tasks) if normalized else None,
    }


def run_pairwise_merge(
    cfg: Mapping[str, Any], *, device: str, resume: bool = False,
    max_eval_batches: int | None = None, reuse_existing_baseline: bool = False,
    only_pair: tuple[str, str] | None = None, max_train_steps: int | None = None,
) -> dict[str, Any]:
    repo = Path(cfg["repository_root"])
    vision_dir = repo / "vision_lora_merge"
    if str(vision_dir) not in os.sys.path:
        os.sys.path.insert(0, str(vision_dir))
    from merging_functions import dc_merge

    pair_summaries = []
    pairs = unordered_task_pairs(cfg["tasks"])
    if only_pair is not None:
        pairs = tuple(pair for pair in pairs if set(pair) == set(only_pair))
        if len(pairs) != 1:
            raise ValueError(f"invalid unordered pilot pair: {only_pair}")
    for task_a, task_b in pairs:
        run_dir = Path(cfg["output_root"]) / "metrics" / "pairwise_merge" / f"{task_a}--{task_b}"
        paths = {
            task: resolve_baseline_adapter(
                cfg, task, int(cfg["anchor_seed"]), reuse_existing_baseline,
                max_train_steps, max_eval_batches,
            )
            for task in (task_a, task_b)
        }
        definition = pairwise_run_definition(
            cfg, task_a, task_b, paths, max_eval_batches
        )
        definition_hash = pairwise_run_definition_hash(definition)
        action = run_action(run_dir, resume)
        if action == "skip":
            validated = validate_complete_pairwise_run(run_dir, definition)
            pair_summaries.append(validated["result"])
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        set_status(
            run_dir, "RUNNING", started_at=utc_now(),
            restart_from_beginning=action == "retry",
            pairwise_run_definition=definition,
            pairwise_run_definition_hash=definition_hash,
        )
        started = time.monotonic()
        try:
            deltas = {}
            for task, path in paths.items():
                adapter_cfg, adapter_state = load_adapter(path)
                deltas[task] = effective_deltas(adapter_cfg, adapter_state)
            if set(deltas[task_a]) != set(deltas[task_b]):
                raise ValueError("pair adapter modules do not match")
            grouped = OrderedDict((name, [deltas[task_a][name], deltas[task_b][name]]) for name in deltas[task_a])
            merged = dc_merge(grouped, smoothing_strategy="linear", rho=5.0, task_scales=None)
            model, loaders, heads = _build_runtime(cfg, (task_a, task_b), device)
            clean_weights = capture_clean_target_weights(model.vision_model, merged)
            baseline_val = {}
            for task in (task_a, task_b):
                materialize_from_clean(model.vision_model, clean_weights, deltas[task], 1.0)
                baseline_val[task] = evaluate(model, loaders[task]["val"], heads[task], device, max_eval_batches)

            validation_sweep = []
            for alpha in cfg["merge_alphas"]:
                materialize_from_clean(model.vision_model, clean_weights, merged, float(alpha))
                metrics = _evaluate_pair(model, loaders, heads, (task_a, task_b), "val", device, max_eval_batches, baseline_val)
                metrics["alpha"] = float(alpha)
                validation_sweep.append(metrics)
            selected = max(validation_sweep, key=lambda item: item["mean_normalized_accuracy"])
            selected_alpha = float(selected["alpha"])
            manifest = {
                "selection_split": "validation", "selection_tasks": [task_a, task_b],
                "selection_objective": "mean normalized validation accuracy over exactly the two pair tasks",
                "selected_alpha": selected_alpha, "task_scales": None,
                "validation_sweep": validation_sweep, "frozen_at": utc_now(),
                "pairwise_run_definition": definition,
                "pairwise_run_definition_hash": definition_hash,
            }
            atomic_write_json(run_dir / "frozen_selection.json", manifest)

            baseline_test = {}
            for task in (task_a, task_b):
                materialize_from_clean(model.vision_model, clean_weights, deltas[task], 1.0)
                baseline_test[task] = evaluate(model, loaders[task]["test"], heads[task], device, max_eval_batches)
            materialize_from_clean(model.vision_model, clean_weights, merged, selected_alpha)
            merged_test = _evaluate_pair(model, loaders, heads, (task_a, task_b), "test", device, max_eval_batches, baseline_test)
            gap = (baseline_test[task_a] + baseline_test[task_b]) / 2.0 - merged_test["mean_accuracy"]
            result = {
                "task_a": task_a, "task_b": task_b, "canonical_seed": int(cfg["anchor_seed"]),
                "dc_merge_smoothing": "linear", "dc_merge_rho": 5.0, "task_scales": None,
                "selected_alpha": selected_alpha, "validation_sweep": validation_sweep,
                "baseline_validation_accuracy": baseline_val, "baseline_test_accuracy": baseline_test,
                "merged_validation": selected, "merged_test": merged_test,
                "G_AB": gap,
                "Retention_A": merged_test["accuracy"][task_a] / baseline_test[task_a],
                "Retention_B": merged_test["accuracy"][task_b] / baseline_test[task_b],
                "test_read_only_after_selection_frozen": True,
                "alpha_evaluation_base_state": "clean M0 restored exactly before every candidate",
                "elapsed_seconds": time.monotonic() - started,
                "pairwise_run_definition": definition,
                "pairwise_run_definition_hash": definition_hash,
            }
            atomic_write_json(run_dir / "result.json", result)
            set_status(
                run_dir, "COMPLETE", completed_at=utc_now(), selected_alpha=selected_alpha,
                pairwise_run_definition=definition,
                pairwise_run_definition_hash=definition_hash,
            )
            validate_complete_pairwise_run(run_dir, definition)
            pair_summaries.append(result)
        except Exception as error:
            set_status(
                run_dir, "FAILED", failed_at=utc_now(), error=repr(error),
                pairwise_run_definition=definition,
                pairwise_run_definition_hash=definition_hash,
            )
            raise
    summary = {"pair_count": len(pair_summaries), "pairs": pair_summaries}
    atomic_write_json(Path(cfg["output_root"]) / "metrics" / "pairwise_merge" / "summary.json", summary)
    return summary
