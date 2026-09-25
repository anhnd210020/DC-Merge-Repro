from __future__ import annotations

import math
import re
from collections import OrderedDict
from typing import Any, Mapping

import torch

from .lora_delta import lora_scaling, paired_factors


LAYER_MODULE_RE = re.compile(
    r"(?:^|\.)encoder\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|out_proj)$"
)


def numerical_tolerance(singular_values: torch.Tensor, rows: int, columns: int, dtype: torch.dtype) -> float:
    if singular_values.numel() == 0:
        return 0.0
    real_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
    epsilon = torch.finfo(real_dtype).eps
    return float(epsilon * max(rows, columns) * float(singular_values.max()))


def thin_svd(a: torch.Tensor, b: torch.Tensor, scaling: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact economy SVD of ``scaling * B @ A`` using only a rank-by-rank core."""
    if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
        raise ValueError(f"invalid factors: A={tuple(a.shape)}, B={tuple(b.shape)}")
    work_dtype = torch.float64 if a.dtype == torch.float64 or b.dtype == torch.float64 else torch.float32
    aa, bb = a.to(work_dtype), b.to(work_dtype)
    qb, rb = torch.linalg.qr(bb, mode="reduced")
    qa, ra = torch.linalg.qr(aa.T, mode="reduced")
    core = float(scaling) * (rb @ ra.T)
    uc, singular, vhc = torch.linalg.svd(core, full_matrices=False)
    return qb @ uc, singular, vhc @ qa.T


def factor_svd(config: Mapping[str, Any], state: Mapping[str, torch.Tensor]) -> OrderedDict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    scale = lora_scaling(config)
    return OrderedDict((name, thin_svd(a, b, scale)) for name, (a, b) in paired_factors(state).items())


def _module_subspace(
    module: str,
    baseline: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    conditional: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    requested_rank: int,
) -> dict[str, Any]:
    ub, sb, vhb = baseline
    uc, sc, vhc = conditional
    tol_b = numerical_tolerance(sb, ub.shape[0], vhb.shape[1], sb.dtype)
    tol_c = numerical_tolerance(sc, uc.shape[0], vhc.shape[1], sc.dtype)
    rank_b = int((sb > tol_b).sum())
    rank_c = int((sc > tol_c).sum())
    k = min(rank_b, rank_c, int(requested_rank))
    norm_b = float(torch.linalg.vector_norm(sb))
    norm_c = float(torch.linalg.vector_norm(sc))
    row: dict[str, Any] = {
        "module": module,
        "requested_rank": int(requested_rank),
        "baseline_numerical_rank": rank_b,
        "conditional_numerical_rank": rank_c,
        "comparison_rank": k,
        "baseline_tolerance": tol_b,
        "conditional_tolerance": tol_c,
        "baseline_singular_values": [float(x) for x in sb],
        "conditional_singular_values": [float(x) for x in sc],
        "baseline_frobenius_norm": norm_b,
        "conditional_frobenius_norm": norm_c,
        "baseline_energy": norm_b * norm_b,
        "conditional_energy": norm_c * norm_c,
        "degenerate": k == 0,
        "S_U": None,
        "S_V": None,
        "D": None,
    }
    if k:
        su = float(torch.linalg.matrix_norm(ub[:, :k].T @ uc[:, :k], ord="fro") ** 2 / k)
        sv = float(torch.linalg.matrix_norm(vhb[:k] @ vhc[:k].T, ord="fro") ** 2 / k)
        row.update({"S_U": su, "S_V": sv, "D": 1.0 - (su + sv) / 2.0})
    return row


def compare_adapter_subspaces(
    baseline_config: Mapping[str, Any], baseline_state: Mapping[str, torch.Tensor],
    conditional_config: Mapping[str, Any], conditional_state: Mapping[str, torch.Tensor],
    requested_rank: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    left, right = factor_svd(baseline_config, baseline_state), factor_svd(conditional_config, conditional_state)
    if set(left) != set(right):
        raise ValueError(f"adapter module mismatch: baseline-only={sorted(set(left)-set(right))}, conditional-only={sorted(set(right)-set(left))}")
    rows = [_module_subspace(name, left[name], right[name], requested_rank) for name in left]
    valid = [row for row in rows if not row["degenerate"]]
    if not valid:
        return rows, {"valid_modules": 0, "macro_mean_drift": None, "energy_weighted_mean_drift": None}
    macro = sum(float(row["D"]) for row in valid) / len(valid)
    weights = [(float(row["baseline_energy"]) + float(row["conditional_energy"])) / 2.0 for row in valid]
    total_weight = sum(weights)
    weighted = sum(float(row["D"]) * weight for row, weight in zip(valid, weights)) / total_weight if total_weight else None
    return rows, {
        "valid_modules": len(valid),
        "degenerate_modules": len(rows) - len(valid),
        "macro_mean_drift": macro,
        "energy_weighted_mean_drift": weighted,
        "energy_weighting_definition": "module weight = mean(baseline Frobenius norm squared, comparison Frobenius norm squared)",
    }


def static_module_metrics(
    left_config: Mapping[str, Any], left_state: Mapping[str, torch.Tensor],
    right_config: Mapping[str, Any], right_state: Mapping[str, torch.Tensor], requested_rank: int,
) -> list[dict[str, Any]]:
    lf, rf = paired_factors(left_state), paired_factors(right_state)
    if set(lf) != set(rf):
        raise ValueError("adapter modules do not match")
    ls, rs = lora_scaling(left_config), lora_scaling(right_config)
    svd_l, svd_r = factor_svd(left_config, left_state), factor_svd(right_config, right_state)
    rows = []
    for name in lf:
        la, lb = lf[name]
        ra, rb = rf[name]
        dl, dr = ls * (lb @ la), rs * (rb @ ra)
        denominator = float(torch.linalg.vector_norm(dl) * torch.linalg.vector_norm(dr))
        cosine = float((dl * dr).sum()) / denominator if denominator else None
        jointly_nonzero = (dl != 0) & (dr != 0)
        count = int(jointly_nonzero.sum())
        conflict = float(((torch.sign(dl) != torch.sign(dr)) & jointly_nonzero).sum()) / count if count else None
        subspace = _module_subspace(name, svd_l[name], svd_r[name], requested_rank)
        rows.append({
            "module": name,
            "raw_effective_delta_cosine": cosine,
            "raw_delta_sign_conflict": conflict,
            "jointly_nonzero_elements": count,
            "left_subspace_overlap": subspace["S_U"],
            "right_subspace_overlap": subspace["S_V"],
            "left_frobenius_norm": float(torch.linalg.vector_norm(dl)),
            "right_frobenius_norm": float(torch.linalg.vector_norm(dr)),
            "left_energy": float((dl * dl).sum()),
            "right_energy": float((dr * dr).sum()),
            "comparison_rank": subspace["comparison_rank"],
            "degenerate": subspace["degenerate"],
        })
    return rows


def mean_and_sample_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def logical_layer(module_name: str) -> tuple[int, str]:
    match = LAYER_MODULE_RE.search(module_name)
    if not match:
        raise ValueError(
            "LoRA module name does not match the validated CLIP layer convention "
            f"'encoder.layers.<index>.self_attn.<target>': {module_name!r}"
        )
    return int(match.group(1)), match.group(2)


def aggregate_layer_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    context_keys = (
        "category", "target_task", "anchor_task", "target_seed", "anchor_seed",
        "anchor_alpha", "seed_left", "seed_right",
    )
    for row in rows:
        layer_index, _target_module = logical_layer(str(row["module"]))
        key = tuple(row.get(name) for name in context_keys) + (layer_index,)
        grouped.setdefault(key, []).append(row)
    aggregates: list[dict[str, Any]] = []
    for key, modules in grouped.items():
        context = dict(zip(context_keys, key[:-1]))
        layer_index = key[-1]
        valid = [row for row in modules if not row["degenerate"] and row["D"] is not None]
        degenerate_count = len(modules) - len(valid)
        energies = [
            (float(row["baseline_energy"]) + float(row["conditional_energy"])) / 2.0
            for row in valid
        ]
        total_energy = sum(energies)
        aggregates.append(
            {
                **context,
                "layer_index": layer_index,
                "layer_name": f"encoder.layers.{layer_index}",
                "module_count": len(modules),
                "valid_module_count": len(valid),
                "degenerate_module_count": degenerate_count,
                "macro_mean_S_U": sum(float(row["S_U"]) for row in valid) / len(valid) if valid else None,
                "macro_mean_S_V": sum(float(row["S_V"]) for row in valid) / len(valid) if valid else None,
                "macro_mean_D": sum(float(row["D"]) for row in valid) / len(valid) if valid else None,
                "energy_weighted_D": (
                    sum(float(row["D"]) * weight for row, weight in zip(valid, energies)) / total_energy
                    if valid and total_energy > 0 else None
                ),
                "mean_baseline_numerical_rank": (
                    sum(int(row["baseline_numerical_rank"]) for row in valid) / len(valid) if valid else None
                ),
                "mean_conditional_numerical_rank": (
                    sum(int(row["conditional_numerical_rank"]) for row in valid) / len(valid) if valid else None
                ),
                "mean_comparison_rank": (
                    sum(int(row["comparison_rank"]) for row in valid) / len(valid) if valid else None
                ),
                "mean_baseline_frobenius_norm": (
                    sum(float(row["baseline_frobenius_norm"]) for row in valid) / len(valid) if valid else None
                ),
                "mean_conditional_frobenius_norm": (
                    sum(float(row["conditional_frobenius_norm"]) for row in valid) / len(valid) if valid else None
                ),
                "energy_weighting_definition": (
                    "module weight = mean(baseline Frobenius norm squared, comparison Frobenius norm squared)"
                ),
            }
        )
    return aggregates


def safe_metric_aggregate(
    rows: list[Mapping[str, Any]], metric: str, *, degenerate_key: str = "degenerate"
) -> dict[str, Any]:
    valid: list[float] = []
    degenerate_count = 0
    skipped_count = 0
    for row in rows:
        if bool(row.get(degenerate_key, False)):
            degenerate_count += 1
            continue
        value = row.get(metric)
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            skipped_count += 1
            continue
        valid.append(float(value))
    return {
        "value": sum(valid) / len(valid) if valid else None,
        "valid_count": len(valid),
        "skipped_count": skipped_count,
        "degenerate_count": degenerate_count,
        "status": "ok" if valid else "no_valid_observations",
    }
