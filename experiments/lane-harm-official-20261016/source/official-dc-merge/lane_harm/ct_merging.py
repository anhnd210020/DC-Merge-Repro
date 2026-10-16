"""CT-Merging (Ryum & Kang, arXiv:2607.20561), implemented from the paper.

No official code is released.  This follows Sections 3.1-3.3 and 4.1:

  1. For each task, Delta_t = U_t S_t V_t^T (16 retained directions).
  2. Consensus left directions: top-k eigenvectors of P = (1/T) sum_t U_t U_t^T   (Eqs. 2-3).
  3. Right consensus directions: SVD of the averaged projected update
        (1/T) sum_t U_c^T Delta_t = P_c Sigma_c V_c^T,   U_c <- U_c P_c          (Eq. 5).
  4. Residuals: for each of the leading r left directions u_{t,j} of task t,
        z = (I - U_c U_c^T) u,  a = ||z||,  u_bar = z / a,  s_bar = a * s        (Eqs. 6-8).
  5. U = [U_c | U_bar_1 ... U_bar_T],  V = [V_c | V_1 ... V_T]; whiten both with
        W(X) = L R^T  for  X = L S R^T  (TSV-style).
  6. Scales: E_t = sum_j s_bar_{t,j}^2, rho_t = sqrt(E_t / r);  E_c = ||Sigma_c||_F^2,
        rho_c = sqrt(E_c / k);  D = diag(rho_c I_k, rho_1 I_r, ..., rho_T I_r)    (Eqs. 10-13).
  7. Delta_merge = gamma * U~ D V~^T                                              (Eq. 14).

This function returns U~ D V~^T; gamma is the merge coefficient passed to
``apply_merge`` (selected on validation from the paper's grid).

Paper settings for ViT-B: k = 8, r = 15.  Where the paper is not explicit
(eigen-solver, sign conventions, degenerate residuals) the choices are noted
inline; they do not change the span or the scales.
"""
from __future__ import annotations

from collections import OrderedDict

import torch

PAPER_K_VITB = 8
PAPER_R_VITB = 15
PAPER_GAMMA_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.3, 1.5)


def _whiten(x):
    l, _s, rt = torch.linalg.svd(x, full_matrices=False)
    return l @ rt


def ct_merge_module(vecs, k=PAPER_K_VITB, r=PAPER_R_VITB, rank=16, eps=1e-12,
                    return_parts=False):
    T = len(vecs)
    if r > rank:
        raise ValueError("r must not exceed the retained rank")
    us, ss, vs = [], [], []
    for d in vecs:
        u, s, vh = torch.linalg.svd(d, full_matrices=False)
        us.append(u[:, :rank]); ss.append(s[:rank]); vs.append(vh[:rank, :].T)   # v: n x rank
    m, n = vecs[0].shape
    if k + T * r > min(m, n):
        raise ValueError(f"k + T*r = {k + T * r} exceeds module size {(m, n)}")

    # (2)-(3) top-k eigenvectors of P = (1/T) sum U U^T, via SVD of [U_1..U_T]/sqrt(T)
    stacked = torch.cat(us, dim=1) / (T ** 0.5)
    lu, _, _ = torch.linalg.svd(stacked, full_matrices=False)
    u_c = lu[:, :k]

    # (5) right consensus directions and rotation of U_c
    avg_proj = sum(u_c.T @ d for d in vecs) / T                 # k x n
    p_c, sigma_c, vch = torch.linalg.svd(avg_proj, full_matrices=False)
    u_c = u_c @ p_c
    v_c = vch.T                                                 # n x k

    # (6)-(8) residual directions
    proj = torch.eye(m, dtype=u_c.dtype, device=u_c.device) - u_c @ u_c.T
    res_u, res_v, rhos = [], [], []
    for t in range(T):
        z = proj @ us[t][:, :r]                                 # m x r
        a = z.norm(dim=0)
        u_bar = z / a.clamp_min(eps)                            # degenerate column -> ~0 (not in paper)
        s_bar = a * ss[t][:r]
        res_u.append(u_bar); res_v.append(vs[t][:, :r])
        rhos.append(torch.sqrt((s_bar ** 2).sum() / r))         # (10)-(11)
    rho_c = torch.sqrt((sigma_c ** 2).sum() / k)                # (12)

    # (5') whitening and (13)-(14)
    U = _whiten(torch.cat([u_c] + res_u, dim=1))
    V = _whiten(torch.cat([v_c] + res_v, dim=1))
    diag = torch.cat([rho_c.repeat(k)] + [rho.repeat(r) for rho in rhos])
    merged = U @ torch.diag(diag) @ V.T
    if not return_parts:
        return merged
    return merged, {"U": U, "V": V, "rho_c": float(rho_c), "rho_t": [float(x) for x in rhos],
                    "orthonormality_error": max(float((U.T @ U - torch.eye(U.shape[1], dtype=U.dtype)).abs().max()),
                                                float((V.T @ V - torch.eye(V.shape[1], dtype=V.dtype)).abs().max()))}


def ct_merge(deltas_dict, k=PAPER_K_VITB, r=PAPER_R_VITB):
    """Merge every module; returns (merged OrderedDict, diagnostics)."""
    out, diag = OrderedDict(), {"max_orthonormality_error": 0.0, "rho_t_by_module": {}}
    for key, vecs in deltas_dict.items():
        merged, parts = ct_merge_module(vecs, k=k, r=r, return_parts=True)
        out[key] = merged
        diag["max_orthonormality_error"] = max(diag["max_orthonormality_error"], parts["orthonormality_error"])
        diag["rho_t_by_module"][key] = parts["rho_t"]
    return OrderedDict(sorted(out.items())), diag
