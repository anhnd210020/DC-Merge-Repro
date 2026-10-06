"""Synthetic evaluator for CPU-only pipeline checks; never scientific evidence."""
from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping
from .design import TASKS


class SimulationEvaluator:
    mode = "simulation"

    def __call__(self, config: Mapping[str, Any], split: str) -> dict[str, Any]:
        per_task = {}
        for task in config["tasks"]:
            digest = hashlib.sha256(f"{config['run_id']}|{split}|{task}".encode()).digest()
            base = 56.0 + (TASKS.index(task) % 5)
            change = 0.0
            if config["focal_task"] == task:
                if config["resource"] == "rank": change = (16 - int(config["level"])) * .25
                elif config["resource"] == "density": change = (1 - float(config["level"])) * 2
                elif config["resource"] == "scale": change = abs(1 - float(config["level"])) * .5
            # Deterministic context variation makes synthetic tests exercise nonconstant curves.
            jitter = (digest[0] / 255 - .5) * .3
            per_task[task] = {"accuracy_percent": max(1, min(99, base + jitter - change)), "examples": 100}
        density = {}
        for task in config["tasks"]:
            requested=float(config["density_by_task"][task]); baseline=100
            target=int(math.floor(requested*baseline+.5))
            per_module={f"module_{i}":min(25,max(0,target-i*25)) for i in range(4)}
            density[task] = {"nominal_density": requested, "requested_density": requested,
                             "baseline_mask_count": baseline, "target_mask_count": target,
                             "retained_mask_count": target, "retained_nonzero_count": target,
                             "total_coordinates": 10000, "actual_mask_fraction": target/10000,
                             "actual_nonzero_fraction": target/10000,
                             "actual_baseline_support_fraction": target/baseline,
                             "per_module_counts": {name:{"baseline_mask_count":25,"target_mask_count":count,
                                 "retained_mask_count":count} for name,count in per_module.items()}}
        return {"mode":"simulation","per_task": per_task, "density_by_task": density,
                "resources": {"device": "synthetic", "evaluated_tasks": list(config["tasks"]), "simulation": True}}


def synthetic_features(design: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows=[]; seen=set()
    for config in design["configurations"]:
        key=(config["context_id"],tuple(config["tasks"]))
        if key in seen: continue
        seen.add(key)
        if config['phase']=='A': focals=list(config['tasks'])
        elif config['phase']=='B': focals=[r['focal_task'] for r in design['context_manifest']['phase_b_focal_contexts'] if r['context_id']==config['context_id']]
        else: focals=next(r['focal_tasks'] for r in design['context_manifest']['phase_c_reused_contexts'] if r['context_id']==config['context_id'])
        for focal in dict.fromkeys(focals):
            n=TASKS.index(focal)+1
            rows.append({"context_id":config["context_id"],"phase":config["phase"],"focal_task":focal,
                         "context_tasks_json":__import__("json").dumps(config["tasks"]),
                         "frobenius_norm":float(n),"squared_frobenius_norm":float(n*n),
                         "nuclear_norm":float(n*2),"spectral_entropy":.5,"effective_rank":2.,
                         "energy_rank_25":1,"energy_rank_50":1,"energy_rank_75":2,"energy_rank_90":2,
                         "module_norm_mean":float(n),"module_norm_std":0.,"module_norm_max":float(n),"module_norm_cv":0.,
                         "projection_survival":.8,"projection_loss":.2,"topk_survival":.5,"total_survival":.4,
                         "pairwise_cosine_mean":.1*n/8,"pairwise_cosine_min":-.1,"pairwise_cosine_max":.4,
                         "pairwise_cosine_std":.2,
                         "subspace_overlap_mean":.25,"unique_direction_score":.75,"shared_direction_score":.25,
                         "sign_conflict":None,"sign_coverage":None,"feature_mode":"simulation"})
    return rows
