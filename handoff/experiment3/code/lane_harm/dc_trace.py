"""Traced copy of the official DC-Merge LoRA operator.

The official ``vision_lora_merge/merging_functions.py::dc_merge`` (commit
089af77) merges every module in one function and returns only the merged
matrices.  ``lane_harm`` needs the intermediate quantities (cover frame,
projected and sparsified matrices, TIES aggregate, mask).  This file repeats the
official per-module computation statement by statement, using the official
helpers ``generate_linear_distribution``, ``keep_topk_percent`` and
``ties_small`` imported from the official module, and additionally returns the
intermediates.

``topk="official"`` is the official operator.  The runner checks on every merge
that its output is identical to the official ``dc_merge``.

``topk="rank"`` is *our* variant: keep exactly r = 16 coefficients per task per
module (same magnitude-threshold tie rule) instead of 0.1% of the cover space.
It is not part of the official code, and is reported as such.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

OFFICIAL_RANK = 16          # official: ``low_rank_per_task = 16``
OFFICIAL_TOP_PERCENT = 1e-3  # official: ``keep_topk_percent(Ms, 1e-3)``


def keep_topk_count(mat_list, count):
    """Keep the ``count`` largest-magnitude entries (official threshold/tie rule)."""
    out = []
    for t in mat_list:
        flat_abs = t.abs().view(-1)
        threshold = torch.topk(flat_abs, count).values.min()
        out.append(t * (t.abs() >= threshold))
    return out


def make_backend(mf):
    """Return an object with the interface lane_harm expects.

    ``mf`` is the official ``merging_functions`` module.
    """

    def group(vecs, smoothing_strategy="avg", rho=5.0, topk="official",
              return_intermediates=False, task_densities=None):
        if task_densities is not None:
            raise ValueError("the official operator has no task densities")
        if topk not in ("official", "rank"):
            raise ValueError("topk must be 'official' or 'rank'")
        N = len(vecs)
        low_rank_per_task = OFFICIAL_RANK
        if N * low_rank_per_task > min(vecs[0].shape):
            raise ValueError(f"{N} tasks x rank {low_rank_per_task} exceed module size {tuple(vecs[0].shape)}")
        smoothed_vecs, dirs_u, dirs_v, energies = [], [], [], []

        # ---- official dc_merge, lines 169-188 -------------------------------
        for i in range(N):
            u, s, v = torch.linalg.svd(vecs[i], full_matrices=False)
            if i == 0:
                sum_u = torch.zeros_like(u)
                sum_v = torch.zeros_like(v)

            sum_u[:, i * low_rank_per_task: (i + 1) * low_rank_per_task] = u[:, :low_rank_per_task]
            sum_v[i * low_rank_per_task: (i + 1) * low_rank_per_task, :] = v[:low_rank_per_task, :]

            orig_energy = s[:low_rank_per_task].clone()
            if smoothing_strategy == 'linear':
                smoothed_ratio = min(rho, orig_energy[0] / orig_energy[-1])
                smoothed_energy_dist = mf.generate_linear_distribution(low_rank_per_task, smoothed_ratio)
            elif smoothing_strategy == 'avg':
                smoothed_energy_dist = torch.ones_like(orig_energy) / len(orig_energy)
            else:
                raise ValueError("Invalid smoothing strategy")

            smoothed_energy = orig_energy.sum() * smoothed_energy_dist
            smoothed_vecs.append(u[:, :low_rank_per_task] @ torch.diag(smoothed_energy) @ v[:low_rank_per_task, :])
            dirs_u.append(u[:, :low_rank_per_task])
            dirs_v.append(v[:low_rank_per_task, :])
            energies.append(smoothed_energy)

        # ---- official dc_merge, lines 190-203 -------------------------------
        u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
        u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
        cover_space_u = (u_u @ v_u)[:, :N * low_rank_per_task]
        cover_space_vT = (u_v @ v_v)[:N * low_rank_per_task, :]

        Ms = [torch.linalg.multi_dot((cover_space_u.T, smoothed_vecs[i], cover_space_vT.T, )) for i in range(N)]
        if topk == "official":
            filtered_Ms = mf.keep_topk_percent(Ms, OFFICIAL_TOP_PERCENT)
        else:
            filtered_Ms = keep_topk_count(Ms, low_rank_per_task)
        agg_M = mf.ties_small(filtered_Ms)
        mask_M = torch.zeros_like(agg_M)
        d_per_task = mask_M.shape[0] // N
        for i in range(N):
            mask_M[i * d_per_task: (i + 1) * d_per_task, i * d_per_task: (i + 1) * d_per_task] = 1

        merged = torch.linalg.multi_dot((cover_space_u, agg_M * mask_M, cover_space_vT, ))
        if not return_intermediates:
            return merged
        return merged, {
            "cover_space_u": cover_space_u,
            "cover_space_vT": cover_space_vT,
            "projected_matrices": Ms,
            "filtered_matrices": filtered_Ms,
            "ties_aggregate": agg_M,
            "structural_mask": mask_M,
            "rank_offsets": tuple((i * d_per_task, (i + 1) * d_per_task) for i in range(N)),
            "task_ranks": (low_rank_per_task,) * N,
            "singular_directions_u": dirs_u,
            "singular_directions_v": dirs_v,
            "smoothed_singular_values": energies,
        }

    return SimpleNamespace(_dc_merge_matrix_group=group, ties_small=mf.ties_small,
                           dc_merge=mf.dc_merge)
