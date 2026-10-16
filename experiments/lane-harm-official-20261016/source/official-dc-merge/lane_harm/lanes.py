"""Per-task lane terms of a DC-Merge merge, for removal and rescaling.

The DC-Merge update of one module is an exact sum over tasks:

    Delta_merge = U~ (A * Mask) V~^T = sum_j  U~_j A_jj V~_j^T  =: sum_j L_j

because the mask keeps only the diagonal blocks and U~, V~ have orthonormal
columns.  ``LaneMerge`` computes the L_j once per context (with the full
context's cover frame and TIES aggregate) and composes

    Delta(lambda) = sum_j lambda_j L_j .

lambda_j = 0 removes task j's contribution entirely, lambda_j in (0, 1)
turns it down.  The frame and the aggregate stay those of the real merge, so
changing lambda changes nothing except how much of each task's lane is added.
"""
from __future__ import annotations

from collections import OrderedDict

import torch


class LaneMerge:
    def __init__(self, grouped, backend, task_names, dc_kwargs, official_dc_merge=None):
        self.tasks = list(task_names)
        self.lanes = OrderedDict()          # module -> [L_j]
        self.checks = {"max_rel_sum_error": 0.0, "max_abs_diff_vs_official": None}
        merged_ref = official_dc_merge(grouped, **dc_kwargs) if official_dc_merge else None
        for key, vecs in grouped.items():
            merged, inter = backend._dc_merge_matrix_group(list(vecs), return_intermediates=True, **dc_kwargs)
            U, Vt = inter["cover_space_u"], inter["cover_space_vT"]
            A = inter["ties_aggregate"] * inter["structural_mask"]
            terms = []
            for (s, e) in inter["rank_offsets"]:
                terms.append(U[:, s:e] @ A[s:e, s:e] @ Vt[s:e, :])
            total = sum(terms)
            rel = float((total - merged).norm() / merged.norm().clamp_min(1e-30))
            self.checks["max_rel_sum_error"] = max(self.checks["max_rel_sum_error"], rel)
            if merged_ref is not None:
                d = float((merged_ref[key] - merged).abs().max())
                prev = self.checks["max_abs_diff_vs_official"] or 0.0
                self.checks["max_abs_diff_vs_official"] = max(prev, d)
            self.lanes[key] = terms
        if self.checks["max_rel_sum_error"] > 1e-5:
            raise AssertionError(f"lane terms do not sum to the merge: {self.checks['max_rel_sum_error']:g}")
        if self.checks["max_abs_diff_vs_official"] not in (None, 0.0):
            raise AssertionError(f"traced merge differs from official dc_merge: "
                                 f"{self.checks['max_abs_diff_vs_official']:g}")

    def compose(self, scales=None):
        """scales: {task_name: lambda}; tasks not listed keep lambda = 1."""
        scales = scales or {}
        unknown = set(scales) - set(self.tasks)
        if unknown:
            raise ValueError(f"unknown tasks {unknown}")
        lam = [float(scales.get(t, 1.0)) for t in self.tasks]
        out = OrderedDict()
        for key, terms in self.lanes.items():
            acc = torch.zeros_like(terms[0])
            for l, term in zip(lam, terms):
                if l != 0.0:
                    acc = acc + l * term
            out[key] = acc
        return OrderedDict(sorted(out.items()))
