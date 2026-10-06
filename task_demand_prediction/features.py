"""Accuracy-free task-vector and context-geometry feature calculations."""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

ENERGY_CUTS = (0.25, 0.50, 0.75, 0.90)
FEATURE_EPS = 1e-12
FEATURE_DEFINITIONS = {
    "frobenius_norm": "sqrt(sum of squared entries over all modules); parameter units",
    "squared_frobenius_norm": "sum of squared entries over all modules; parameter-units squared",
    "nuclear_norm": "sum of singular values over every module; parameter units",
    "spectral_entropy": "raw Shannon entropy H=-sum(p_k log p_k) of squared singular-value energy probabilities; nats",
    "effective_rank": "exp(Shannon entropy of normalized squared singular values); component count",
    "energy_rank_k": "minimum concatenated module singular components retaining fraction k of energy; component count",
    "energy_rank_*": "concatenate every module's singular values, sort squared values globally, then count to threshold; component count",
    "energy_rank_fraction_*": "energy_rank_* divided by the total number of singular components; unitless",
    "module_norm_*": "population mean/std (ddof=0), maximum and std/mean CV across module Frobenius norms; parameter units or unitless CV",
    "projection_survival": "sum projected-matrix squared energy / (sum smoothed-input squared energy + 1e-12); unitless",
    "topk_survival": "sum post-top-k squared energy / (sum projected squared energy + 1e-12); unitless",
    "total_survival": "sum post-top-k squared energy / (sum smoothed-input squared energy + 1e-12); unitless",
    "pairwise_cosine_*": "flatten corresponding shared-key modules and concatenate, compute focal-vs-companion cosine; mean/min/max/population-std across companions; unitless [-1,1]",
    "subspace_overlap_mean": "per module average of left and right normalized squared principal-angle cosines, average modules equally, then average companions equally; unitless [0,1]",
    "unique_direction_score": "focal squared energy outside the linear span of companion task vectors, divided by focal squared energy; unitless [0,1]",
    "shared_direction_score": "1 - unique_direction_score; unitless [0,1]",
    "sign_conflict": "focal retained nonzero coordinates disagreeing with an unambiguous companion sign majority / coordinates with such a majority; null when none",
    "sign_coverage": "focal retained nonzero coordinates also retained by at least one companion / focal retained nonzero coordinates; null when focal has no retained coordinates",
}


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    out = np.asarray(value, dtype=np.float64)
    if not np.isfinite(out).all():
        raise ValueError("feature inputs must be finite")
    return out


def _modules(state: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if not state:
        raise ValueError("task vector has no modules")
    return {str(k): _array(v) for k, v in sorted(state.items())}


def compute_task_svd_cache(modules: Mapping[str, Any], rank: int = 16) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    cache={}
    for key,matrix in _modules(modules).items():
        u,s,v=np.linalg.svd(matrix,full_matrices=False)
        k=min(rank,len(s)); cache[key]=(u[:,:k],s,v[:k,:])
    return cache


def self_structure(task: str, modules: Mapping[str, Any], svd_cache: Mapping[str, tuple[np.ndarray,np.ndarray,np.ndarray]]|None=None) -> dict[str, Any]:
    matrices = _modules(modules)
    module_norms = []
    singular_values = []
    squared = 0.0
    nuclear = 0.0
    for key, matrix in matrices.items():
        norm = float(np.linalg.norm(matrix))
        module_norms.append(norm)
        squared += norm * norm
        s = svd_cache[key][1] if svd_cache is not None else np.linalg.svd(matrix, compute_uv=False)
        singular_values.extend(s.tolist())
        nuclear += float(s.sum())
    spectrum = np.asarray(singular_values, dtype=np.float64)
    energy = spectrum * spectrum
    total_energy = float(energy.sum())
    probabilities = energy / total_energy if total_energy > 0 else np.zeros_like(energy)
    positive = probabilities[probabilities > 0]
    entropy = float(-(positive * np.log(positive)).sum()) if positive.size else 0.0
    normalized_entropy = entropy
    rank = float(math.exp(entropy)) if positive.size else 0.0
    order = np.sort(energy)[::-1]
    cumulative = np.cumsum(order)
    row: dict[str, Any] = {
        "task": task,
        "frobenius_norm": math.sqrt(squared),
        "squared_frobenius_norm": squared,
        "nuclear_norm": nuclear,
        "spectral_entropy": normalized_entropy,
        "effective_rank": rank,
        "module_norm_mean": float(np.mean(module_norms)),
        "module_norm_std": float(np.std(module_norms, ddof=0)),
        "module_norm_max": float(np.max(module_norms)),
        "module_norm_cv": float(np.std(module_norms, ddof=0) / np.mean(module_norms)) if np.mean(module_norms) else 0.0,
        "module_count": len(matrices),
        "singular_component_count": len(spectrum),
        "total_spectral_energy": total_energy,
    }
    for cut in ENERGY_CUTS:
        row[f"energy_rank_{int(cut*100)}"] = int(np.searchsorted(cumulative, cut * total_energy, side="left") + 1) if total_energy else 0
        row[f"energy_rank_fraction_{int(cut*100)}"] = row[f"energy_rank_{int(cut*100)}"] / len(spectrum) if len(spectrum) else 0.0
    return row


def _flat_cosine(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray]) -> float | None:
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    av = np.concatenate([a[k].ravel() for k in keys])
    bv = np.concatenate([b[k].ravel() for k in keys])
    den = float(np.linalg.norm(av) * np.linalg.norm(bv))
    return float(np.dot(av, bv) / den) if den else None


def _subspace_overlap(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray], rank: int = 16,
                      cache: Mapping[str, Mapping[str, tuple[np.ndarray,np.ndarray,np.ndarray]]]|None=None,
                      task_a: str|None=None, task_b: str|None=None) -> float | None:
    values = []
    for key in sorted(set(a) & set(b)):
        if cache is None:
            left, _, right = np.linalg.svd(a[key], full_matrices=False)
            left_b, _, right_b = np.linalg.svd(b[key], full_matrices=False)
        else:
            left,_,right=cache[task_a][key];left_b,_,right_b=cache[task_b][key]
        k = min(rank, left.shape[1], left_b.shape[1], right.shape[0], right_b.shape[0])
        if k == 0:
            continue
        # Average left/right normalized squared cosines; this is invariant to basis rotation.
        lo = float(np.linalg.norm(left[:, :k].T @ left_b[:, :k], ord="fro") ** 2 / k)
        ro = float(np.linalg.norm(right[:k, :] @ right_b[:k, :].T, ord="fro") ** 2 / k)
        values.append((lo + ro) / 2.0)
    return float(np.mean(values)) if values else None


def context_interactions(
    focal: str,
    context_tasks: list[str] | tuple[str, ...],
    task_vectors: Mapping[str, Mapping[str, Any]],
    projected: Mapping[str, Mapping[str, Any]] | None = None,
    filtered: Mapping[str, Mapping[str, Any]] | None = None,
    preprojection: Mapping[str, Mapping[str, Any]] | None = None,
    subspace_cache: Mapping[str, Mapping[str, tuple[np.ndarray,np.ndarray,np.ndarray]]]|None=None,
) -> dict[str, Any]:
    if focal not in context_tasks or set(context_tasks) - set(task_vectors):
        raise ValueError("context task membership does not match task-vector inputs")
    arrays = {task: _modules(task_vectors[task]) for task in context_tasks}
    companions = [task for task in context_tasks if task != focal]
    cosines = [_flat_cosine(arrays[focal], arrays[t]) for t in companions]
    cosines = [x for x in cosines if x is not None]
    overlaps = [_subspace_overlap(arrays[focal], arrays[t],cache=subspace_cache,task_a=focal,task_b=t) for t in companions]
    overlaps = [x for x in overlaps if x is not None]
    projection_survival = topk_survival = total_survival = None
    if projected is not None and filtered is not None:
        source = _modules(preprojection[focal]) if preprojection is not None else arrays[focal]
        base = sum(float(np.sum(x*x)) for x in source.values())
        proj_arrays = _modules(projected[focal])
        filt_arrays = _modules(filtered[focal])
        proj = sum(float(np.sum(x*x)) for x in proj_arrays.values())
        filt = sum(float(np.sum(x*x)) for x in filt_arrays.values())
        projection_survival = proj / (base + FEATURE_EPS)
        topk_survival = filt / (proj + FEATURE_EPS)
        total_survival = filt / (base + FEATURE_EPS)
    conflict = coverage = None
    if filtered is not None and companions:
        focal_kept = _modules(filtered[focal])
        others = [_modules(filtered[t]) for t in companions]
        eligible = conflict_count = covered = focal_count = 0
        for key, a in focal_kept.items():
            za = a != 0
            focal_count += int(za.sum())
            votes = np.zeros(a.shape, dtype=np.int32)
            for other in others:
                b = other.get(key)
                if b is None: continue
                votes += (b > 0).astype(np.int32) - (b < 0).astype(np.int32)
            any_other = np.zeros(a.shape, dtype=bool)
            for other in others:
                if key in other: any_other |= other[key] != 0
            covered += int(np.logical_and(za, any_other).sum())
            consensus = votes != 0
            compared = np.logical_and(za, consensus)
            eligible += int(compared.sum())
            conflict_count += int(np.logical_and(compared, np.sign(a) != np.sign(votes)).sum())
        conflict = conflict_count / eligible if eligible else None
        coverage = covered / focal_count if focal_count else None
    # Flatten compatible module coordinates; companion columns span the
    # context subspace and a small QR avoids a parameter-sized Gram matrix.
    unique = shared_direction = None
    keys = sorted(set(arrays[focal]).intersection(*(set(arrays[t]) for t in companions))) if companions else []
    if keys:
        focal_vec = np.concatenate([arrays[focal][k].ravel() for k in keys])
        companion_vecs = [np.concatenate([arrays[t][k].ravel() for k in keys]) for t in companions]
        energy = float(np.dot(focal_vec, focal_vec))
        basis = np.column_stack(companion_vecs)
        q, r = np.linalg.qr(basis, mode="reduced")
        keep = np.abs(np.diag(r)) > np.finfo(float).eps * max(basis.shape) * (np.linalg.norm(basis, ord=2) if basis.size else 1.0)
        q = q[:, keep]
        residual = focal_vec - q @ (q.T @ focal_vec) if q.shape[1] else focal_vec
        unique = float(np.dot(residual, residual) / energy) if energy else None
        shared_direction = 1.0 - unique if unique is not None else None
    return {
        "context_size": len(context_tasks), "companion_count": len(companions),
        "projection_survival": projection_survival, "projection_loss": 1.0 - projection_survival if projection_survival is not None else None,
        "topk_survival": topk_survival, "total_survival": total_survival,
        "pairwise_cosine_mean": float(np.mean(cosines)) if cosines else None,
        "pairwise_cosine_min": float(np.min(cosines)) if cosines else None,
        "pairwise_cosine_max": float(np.max(cosines)) if cosines else None,
        "pairwise_cosine_std": float(np.std(cosines, ddof=0)) if cosines else None,
        "subspace_overlap_mean": float(np.mean(overlaps)) if overlaps else None,
        "unique_direction_score": unique,
        "shared_direction_score": shared_direction,
        "sign_conflict": conflict, "sign_coverage": coverage,
    }


def build_feature_row(context_id: str, phase: str, focal: str, tasks: list[str], vectors: Mapping[str, Mapping[str, Any]], *, self_feature_cache: Mapping[str,Mapping[str,Any]]|None=None, **kwargs: Any) -> dict[str, Any]:
    return {
        "feature_split": "pre_evaluation_shared_across_validation_and_test",
        "context_id": context_id, "phase": phase, "focal_task": focal,
        "context_tasks_json": json_array(tasks),
        **(self_feature_cache[focal] if self_feature_cache is not None else self_structure(focal, vectors[focal])),
        **context_interactions(focal, tasks, vectors, **kwargs),
    }


def json_array(values: Any) -> str:
    import json
    return json.dumps(list(values), separators=(",", ":"))
