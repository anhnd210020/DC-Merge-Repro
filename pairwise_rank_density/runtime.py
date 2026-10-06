"""Lazy real evaluator: existing adapter/delta/merge/data helpers, pair-only inference."""
from __future__ import annotations

import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

from .design import ALPHA, RHO, TASKS, TOP_PERCENT
from .preflight import Paths, _normalized_adapter_state


def force_offline_environment(paths: Paths) -> None:
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
        WANDB_MODE="disabled", DCMERGE_DATA_DIR=str(paths.data),
        DCMERGE_MODEL_DIR=str(paths.model), DCMERGE_CACHE_DIR=str(paths.model),
    )


class RealEvaluator:
    def __init__(self, paths: Paths, device: str):
        self.paths = paths
        self.device = device
        self.runtime = None

    def _load(self) -> dict[str, Any]:
        if self.runtime is not None:
            return self.runtime
        force_offline_environment(self.paths)
        vision = self.paths.repo / "vision_lora_merge"
        os.chdir(vision)
        if str(vision) not in sys.path:
            sys.path.insert(0, str(vision))
        import torch
        from ft_handlers import aggregate_deltas, get_ft_parameters_delta
        from utils import get_clip_encodings, get_config_from_name, prepare_experiment_config, set_seed

        set_seed(400)
        raw = get_config_from_name("vitB32_r16_8task_selftrained", device="cpu")
        raw["model"]["cachedir"] = str(self.paths.model)
        raw["model"]["bases"] = []
        for task_config in raw["dataset"]:
            task_config["clip_encodings"] = str(
                self.paths.heads / "ViT-B-32" / f"{task_config['name']}_head.pt"
            )
        heads = {task: get_clip_encodings(self.paths.heads / "ViT-B-32" / f"{task}_head.pt").to(self.device)
                 for task in TASKS}
        prepared = prepare_experiment_config(raw)
        if [item["name"] for item in raw["dataset"]] != list(TASKS):
            raise RuntimeError("runtime dataset order differs from the registered task order")

        deltas_by_task = {}
        for task in TASKS:
            weight_path = self.paths.adapters / task / "adapter_model.bin"
            state = torch.load(weight_path, map_location="cpu", weights_only=True)
            deltas = get_ft_parameters_delta(_normalized_adapter_state(state))
            deltas_by_task[task] = OrderedDict((key, value.cpu()) for key, value in deltas.items())
            del state, deltas
        reference_keys = tuple(deltas_by_task[TASKS[0]])
        for task in TASKS[1:]:
            if tuple(deltas_by_task[task]) != reference_keys:
                raise RuntimeError(f"adapter module keys/order differ for {task}")
        pair_deltas = {}
        for index, task_a in enumerate(TASKS):
            for task_b in TASKS[index + 1:]:
                pair_deltas[(task_a, task_b)] = aggregate_deltas([
                    deltas_by_task[task_a], deltas_by_task[task_b]
                ])
        prepared["models"]["bases"] = []
        self.runtime = {
            "torch": torch,
            "prepared": prepared,
            "heads": heads,
            "pair_deltas": pair_deltas,
            "load_seconds": None,
        }
        return self.runtime

    def __call__(self, configuration: Mapping[str, Any], split: str) -> dict[str, Any]:
        runtime = self._load()
        torch = runtime["torch"]
        tasks = tuple(configuration["tasks"])
        grouped = runtime["pair_deltas"][tasks]
        vision = self.paths.repo / "vision_lora_merge"
        if str(vision) not in sys.path:
            sys.path.insert(0, str(vision))
        from ft_handlers import apply_merge
        from merging_functions import dc_merge
        from utils import evaluate_cliphead

        with torch.no_grad():
            merged, diagnostics = dc_merge(
                grouped,
                smoothing_strategy="linear",
                rho=RHO,
                task_ranks=tuple(configuration["rank_by_task"][task] for task in tasks),
                task_densities=tuple(configuration["density_by_task"][task] for task in tasks),
                return_intermediates=True,
            )
            model = apply_merge(runtime["prepared"]["models"]["new"], merged, scaling_coeffs=ALPHA)
        model.to(self.device)
        per_task: dict[str, Any] = {}
        density_by_task = {task: {"nominal_density": configuration["density_by_task"][task],
                                  "baseline_mask_count": 0, "target_mask_count": 0,
                                  "retained_mask_count": 0, "retained_nonzero_count": 0,
                                  "total_coordinates": 0} for task in tasks}
        for module, details in diagnostics.items():
            for index, task in enumerate(tasks):
                stats = details["density"][index]
                row = density_by_task[task]
                for key in ("baseline_mask_count", "target_mask_count", "retained_mask_count",
                            "retained_nonzero_count", "total_coordinates"):
                    row[key] += int(stats[key])
        try:
            for task in tasks:
                row = density_by_task[task]
                row["actual_mask_fraction"] = row["retained_mask_count"] / row["total_coordinates"]
                row["actual_nonzero_fraction"] = row["retained_nonzero_count"] / row["total_coordinates"]
                row["actual_baseline_support_fraction"] = (
                    row["retained_mask_count"] / row["baseline_mask_count"]
                    if row["baseline_mask_count"] else 0.0
                )
                loader = runtime["prepared"]["data"][TASKS.index(task)]["test"][split]
                evaluation = evaluate_cliphead(
                    model, loader, class_vectors=runtime["heads"][task], return_predictions=True,
                )
                per_task[task] = {"accuracy_percent": 100.0 * float(evaluation["accuracy"]),
                                  "labels": evaluation["labels"], "predictions": evaluation["predictions"],
                                  "examples": int(evaluation["examples"])}
        finally:
            model.cpu()
            del model, merged, diagnostics
            if self.device == "cuda":
                torch.cuda.empty_cache()
        return {
            "per_task": per_task,
            "density_by_task": density_by_task,
            "resources": {"device": self.device, "evaluated_tasks": list(tasks)},
        }
