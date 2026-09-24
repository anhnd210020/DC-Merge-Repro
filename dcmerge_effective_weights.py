"""Pure, adapter-only DC-Merge effective-weight diagnostics.

The effective-weight definition is versioned here so the standalone analysis
and task-scale ablation runner use the same implementation.  It mirrors the
rank-16 released merge while avoiding dense 768x768 SVDs.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch


EFFECTIVE_WEIGHT_DEFINITION_VERSION = "dcmerge-diagonal-block-squared-frobenius-v1"
RANK = 16
RHO = 5.0
TOP_PERCENT = 1e-3


def load_state(path: Path) -> dict[str, torch.Tensor]:
    """Load one trusted local adapter state without arbitrary pickle objects."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before the weights_only keyword
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint object in {path}")
    return state


def load_scaling(config_path: Path, rank: int = RANK) -> float:
    import json

    config = json.loads(config_path.read_text(encoding="utf-8"))
    configured_rank = config.get("r")
    alpha = config.get("lora_alpha")
    if configured_rank != rank:
        raise ValueError(
            f"Expected LoRA rank {rank}, found {configured_rank} in {config_path}"
        )
    if alpha is None:
        raise ValueError(f"lora_alpha missing in {config_path}")
    return float(alpha) / float(rank)


def adapter_prefixes(state: Mapping[str, torch.Tensor]) -> list[str]:
    suffix = ".lora_A.weight"
    prefixes = sorted(key[: -len(suffix)] for key in state if key.endswith(suffix))
    if len(prefixes) != 48:
        raise ValueError(f"Expected 48 LoRA modules, found {len(prefixes)}")
    for prefix in prefixes:
        if prefix + ".lora_B.weight" not in state:
            raise KeyError(f"Missing lora_B tensor for {prefix}")
    return prefixes


def low_rank_svd(a: torch.Tensor, b: torch.Tensor, scaling: float):
    """Exact thin SVD of ``scaling * (B @ A)`` through a rank-sized core."""
    b = b.to(dtype=torch.float64) * math.sqrt(scaling)
    a = a.to(dtype=torch.float64) * math.sqrt(scaling)
    qb, rb = torch.linalg.qr(b, mode="reduced")
    qa, ra = torch.linalg.qr(a.T, mode="reduced")
    core = rb @ ra.T
    p, singular, qh = torch.linalg.svd(core, full_matrices=False)
    return qb @ p, singular, qh @ qa.T


def linear_distribution(num_values: int, ratio: float) -> torch.Tensor:
    values = torch.linspace(ratio, 1.0, num_values, dtype=torch.float64)
    return values / values.sum()


def keep_topk_percent(
    matrices: Sequence[torch.Tensor], percent: float
) -> list[torch.Tensor]:
    filtered = []
    for matrix in matrices:
        flattened = matrix.abs().reshape(-1)
        k = max(1, int(percent * flattened.numel()))
        threshold = torch.topk(flattened, k).values.min()
        filtered.append(matrix * (matrix.abs() >= threshold))
    return filtered


def ties_small_with_sources(matrices: Sequence[torch.Tensor]):
    """Released ``ties_small`` plus each post-sign-selection source addend."""
    stacked = torch.stack(list(matrices), dim=0)
    summed = stacked.sum(dim=0)
    summed_sign = torch.sign(summed)
    mask = torch.sign(stacked) == summed_sign.unsqueeze(0)
    count = mask.sum(dim=0).clamp(min=1)
    selected = stacked * mask
    sources = selected / count.unsqueeze(0)
    merged = sources.sum(dim=0)
    zero = summed == 0
    merged = torch.where(zero, torch.zeros_like(merged), merged)
    sources = torch.where(zero.unsqueeze(0), torch.zeros_like(sources), sources)
    return merged, sources


def layer_from_prefix(prefix: str) -> int:
    found = re.search(r"encoder\.layers\.(\d+)\.", prefix)
    if not found:
        raise ValueError(f"Cannot identify transformer layer from {prefix}")
    return int(found.group(1))


def normalise(values: Mapping[str, float]) -> dict[str, float]:
    total = float(sum(values.values()))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Cannot normalise a non-finite or non-positive total")
    result = {key: float(value / total) for key, value in values.items()}
    if any(not math.isfinite(value) or value < 0.0 for value in result.values()):
        raise ValueError("Normalised effective weights must be finite and nonnegative")
    return result


def validate_named_task_scales(
    task_names: Sequence[str], task_scales: Mapping[str, float] | None
) -> dict[str, float]:
    if len(task_names) != len(set(task_names)):
        raise ValueError("task names must be unique")
    if task_scales is None:
        return {task: 1.0 for task in task_names}
    if set(task_scales) != set(task_names):
        missing = sorted(set(task_names) - set(task_scales))
        unknown = sorted(set(task_scales) - set(task_names))
        raise ValueError(f"task-scale names mismatch: missing={missing}, unknown={unknown}")
    scales = {task: float(task_scales[task]) for task in task_names}
    if any(not math.isfinite(value) or value <= 0.0 for value in scales.values()):
        raise ValueError("every task scale must be finite and greater than zero")
    return scales


def compute_effective_weights(
    checkpoint_dirs: Mapping[str, Path],
    task_names: Sequence[str],
    task_scales: Mapping[str, float] | None = None,
    *,
    rank: int = RANK,
    rho: float = RHO,
    top_percent: float = TOP_PERCENT,
    progress: bool = False,
) -> dict:
    """Compute realized weights using the existing baseline definition.

    For task ``i`` and module ``m``, the effective energy is
    ``||B_i * T_m||_F^2``, where ``T_m`` is the post-top-k, post-TIES
    cover-space aggregate and ``B_i`` is the task-indexed structural diagonal
    block mask.  Energies are summed over all modules and normalized over tasks.
    """
    names = tuple(task_names)
    if not names:
        raise ValueError("at least one task is required")
    scales = validate_named_task_scales(names, task_scales)
    if set(checkpoint_dirs) != set(names):
        raise ValueError("checkpoint mapping must contain exactly the task names")

    states: dict[str, dict[str, torch.Tensor]] = {}
    lora_scalings: dict[str, float] = {}
    for task in names:
        directory = Path(checkpoint_dirs[task])
        states[task] = load_state(directory / "adapter_model.bin")
        lora_scalings[task] = load_scaling(directory / "adapter_config.json", rank)
    if len(set(lora_scalings.values())) != 1:
        raise ValueError(f"Adapters have inconsistent LoRA scaling: {lora_scalings}")
    lora_scaling = next(iter(lora_scalings.values()))

    prefixes = adapter_prefixes(states[names[0]])
    for task in names[1:]:
        if adapter_prefixes(states[task]) != prefixes:
            raise ValueError(f"LoRA module layout differs for task {task}")

    raw_energy = defaultdict(float)
    smoothed_energy = defaultdict(float)
    block_energy = defaultdict(float)
    source_credit_numerator = defaultdict(float)
    layer_energy = defaultdict(lambda: defaultdict(float))
    module_diagnostics = []

    for module_index, prefix in enumerate(prefixes, start=1):
        svds = {}
        for task in names:
            a = states[task][prefix + ".lora_A.weight"]
            b = states[task][prefix + ".lora_B.weight"]
            u, singular, vh = low_rank_svd(a, b, lora_scaling)
            if singular.numel() != rank:
                raise ValueError(f"Unexpected rank at {prefix} for {task}")
            svds[task] = (u, singular, vh)
            raw_energy[task] += float(singular.square().sum())

        # Cover-space inputs contain directions only.  Task-scaled magnitudes
        # cannot affect this shared polar basis.
        sum_u = torch.cat([svds[task][0][:, :rank] for task in names], dim=1)
        sum_v = torch.cat([svds[task][2][:rank, :] for task in names], dim=0)
        uu, _, vhu = torch.linalg.svd(sum_u, full_matrices=False)
        uv, _, vhv = torch.linalg.svd(sum_v, full_matrices=False)
        cover_u = uu @ vhu
        cover_vt = uv @ vhv

        projected = []
        ratios = {}
        for task in names:
            u, singular, vh = svds[task]
            tail = float(singular[rank - 1])
            observed_ratio = (
                math.inf if abs(tail) < 1e-15 else float(singular[0] / singular[rank - 1])
            )
            ratio = min(rho, observed_ratio)
            distribution = linear_distribution(rank, ratio)
            smoothed_singular = singular.sum() * distribution
            if scales[task] != 1.0:
                smoothed_singular = smoothed_singular * scales[task]
            smoothed_energy[task] += float(smoothed_singular.square().sum())
            left = cover_u.T @ u[:, :rank]
            right = vh[:rank, :] @ cover_vt.T
            projected.append(left @ torch.diag(smoothed_singular) @ right)
            ratios[task] = ratio

        filtered = keep_topk_percent(projected, top_percent)
        merged_cover, source_cover = ties_small_with_sources(filtered)
        structural_mask = torch.zeros_like(merged_cover)
        for i, task in enumerate(names):
            block = torch.zeros_like(merged_cover)
            start, end = i * rank, (i + 1) * rank
            block[start:end, start:end] = 1
            structural_mask += block
            kept = merged_cover * block
            energy = float(kept.square().sum())
            block_energy[task] += energy
            layer_energy[layer_from_prefix(prefix)][task] += energy

        final_cover = merged_cover * structural_mask
        final_energy = float(final_cover.square().sum())
        if final_energy > 0.0:
            for i, task in enumerate(names):
                task_source = source_cover[i] * structural_mask
                source_credit_numerator[task] += float((task_source * final_cover).sum())
        module_diagnostics.append(
            {
                "module": prefix,
                "mean_capped_smoothing_ratio": float(np.mean(list(ratios.values()))),
                "merged_cover_energy": final_energy,
            }
        )
        if progress:
            print(f"[{module_index:02d}/{len(prefixes):02d}] {prefix}", flush=True)

    raw_weight = normalise({task: raw_energy[task] for task in names})
    smoothed_weight = normalise({task: smoothed_energy[task] for task in names})
    block_weight = normalise({task: block_energy[task] for task in names})
    effective_percent = {task: 100.0 * block_weight[task] for task in names}
    if not math.isclose(sum(effective_percent.values()), 100.0, rel_tol=0.0, abs_tol=1e-10):
        raise AssertionError("effective task weights do not sum to 100%")

    total_final_energy = float(sum(block_energy.values()))
    source_credit = {
        task: source_credit_numerator[task] / total_final_energy for task in names
    }
    layer_weights = {}
    for layer, values in sorted(layer_energy.items()):
        normalized = normalise({task: values[task] for task in names})
        layer_weights[str(layer)] = {task: 100.0 * normalized[task] for task in names}

    return {
        "definition_version": EFFECTIVE_WEIGHT_DEFINITION_VERSION,
        "definition": (
            "Across modules, sum the squared Frobenius norm of each task-indexed "
            "rank-16 diagonal block of the post-top-k, post-TIES cover-space "
            "aggregate after the structural block mask, then normalize over tasks."
        ),
        "task_order": list(names),
        "task_scales": scales,
        "rank": rank,
        "rho": rho,
        "top_percent": top_percent,
        "lora_scaling": lora_scaling,
        "raw_energy_weight_percent": {
            task: 100.0 * raw_weight[task] for task in names
        },
        "smoothed_energy_weight_percent": {
            task: 100.0 * smoothed_weight[task] for task in names
        },
        "effective_block_weight_percent": effective_percent,
        "ties_source_credit_percent": {
            task: 100.0 * source_credit[task] for task in names
        },
        "block_energy": {task: block_energy[task] for task in names},
        "layer_effective_weight_percent": layer_weights,
        "module_diagnostics": module_diagnostics,
    }
