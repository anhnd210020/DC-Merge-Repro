"""Merging operators: Task Arithmetic, TSV-M and Iso-CTS, plus the fake-neighbour control.

Each operator maps a list of task vectors (dicts name -> tensor) to one merged delta.
The merged model is theta_0 + alpha * delta. TSV-M and Iso-CTS are ported statement by
statement from the official code (github.com/danielm1405/iso-merging, src/utils/iso.py and
src/utils/TSVM_utils.py; Iso-CTS's TSV-M file is the one from Task Singular Vectors):
  * a parameter is treated as a matrix iff it is 2-D and not "text_projection";
  * every other parameter (biases, LayerNorms, class embedding, 4-D patch conv) is averaged.
One unavoidable difference from the papers' numbers: they merge open_clip checkpoints, whose
attention stores q, k, v as one fused (3d x d) matrix; Hugging Face CLIP stores three d x d
matrices, so the SVD-based operators act on q, k and v separately.
"""
import torch


def is_matrix(name, t):
    return t.dim() == 2 and "text_projection" not in name


def _svd(x):
    return torch.linalg.svd(x, full_matrices=False)


class SVDCache:
    """Full thin SVD of every task matrix, kept on CPU, computed once per (task, key)."""

    def __init__(self, tau):
        self.tau = tau
        self.c = {}

    def get(self, task, name, device):
        k = (task, name)
        if k not in self.c:
            u, s, vh = _svd(self.tau[task][name].float())
            self.c[k] = (u.cpu(), s.cpu(), vh.cpu())
        u, s, vh = self.c[k]
        return u.to(device), s.to(device), vh.to(device)


@torch.no_grad()
def merge_ta(tvs, names):
    return {n: sum(tv[n] for tv in tvs) for n in names}


@torch.no_grad()
def merge_tsvm(tvs, names, task_ids=None, svd_cache=None):
    """TSV-M (compute_and_sum_svd_mem_reduction), optionally using cached per-task SVDs."""
    T = len(tvs)
    sv_reduction = 1 / T
    out = {}
    for n in names:
        first = tvs[0][n]
        if not is_matrix(n, first):
            acc = tvs[0][n].clone()
            for i in range(1, T):
                acc += (tvs[i][n] - acc) / (i + 1)
            out[n] = acc
            continue
        dev = first.device
        for i, tv in enumerate(tvs):
            if svd_cache is not None and task_ids is not None:
                u, s, v = svd_cache.get(task_ids[i], n, dev)
            else:
                u, s, v = _svd(tv[n])
            if i == 0:
                sum_u = torch.zeros_like(u)
                sum_s = torch.zeros_like(s)
                sum_v = torch.zeros_like(v)
            r = int(s.shape[0] * sv_reduction)
            sum_u[:, i * r:(i + 1) * r] = u[:, :r]
            sum_s[i * r:(i + 1) * r] = s[:r]
            sum_v[i * r:(i + 1) * r, :] = v[:r, :]
        u_u, _, v_u = _svd(sum_u)
        u_v, _, v_v = _svd(sum_v)
        out[n] = torch.linalg.multi_dot((u_u, v_u, torch.diag(sum_s), u_v, v_v))
    return out


@torch.no_grad()
def merge_iso_cts(tvs, names, common_space_fraction=0.8):
    """Iso-CTS (iso_cts in src/utils/iso.py)."""
    T = len(tvs)
    out = {}
    for n in names:
        shape_ = tvs[0][n].shape
        if not is_matrix(n, tvs[0][n]):
            acc = tvs[0][n].clone()
            for i in range(1, T):
                acc += (tvs[i][n] - acc) / (i + 1)
            out[n] = acc
            continue
        combined_w = sum(tv[n] for tv in tvs)
        common = int(min(shape_) * common_space_fraction)
        ts_total = round((min(shape_) - common) / T) * T
        common = min(shape_) - ts_total
        u, s, v = _svd(combined_w)
        cu, cs, cv = u[:, :common], s[:common], v[:common, :]
        n_dims = int((min(shape_) - common) / T)
        for i, tv in enumerate(tvs):
            w = tv[n]
            w_ts = w - cu @ cu.T @ w
            u_ts, s_ts, v_ts = _svd(w_ts)
            if i == 0:
                U = torch.zeros_like(u_ts)
                S = torch.zeros_like(s_ts)
                V = torch.zeros_like(v_ts)
            U[:, i * n_dims:(i + 1) * n_dims] = u_ts[:, :n_dims]
            S[i * n_dims:(i + 1) * n_dims] = s_ts[:n_dims]
            V[i * n_dims:(i + 1) * n_dims, :] = v_ts[:n_dims, :]
        U[:, T * n_dims:T * n_dims + common] = cu
        S[T * n_dims:T * n_dims + common] = cs
        V[T * n_dims:T * n_dims + common, :] = cv
        uu, _, vu = _svd(U)
        uv, _, vv = _svd(V)
        U = uu @ vu
        V = uv @ vv
        S = torch.ones_like(S) * S.mean()
        out[n] = torch.linalg.multi_dot((U, torch.diag(S), V))
    return out


def merge(op, tvs, names, task_ids=None, svd_cache=None):
    if op == "ta":
        return merge_ta(tvs, names)
    if op == "tsvm":
        return merge_tsvm(tvs, names, task_ids=task_ids, svd_cache=svd_cache)
    if op == "iso_cts":
        from .config import ISO_CTS_COMMON_FRACTION
        return merge_iso_cts(tvs, names, ISO_CTS_COMMON_FRACTION)
    raise ValueError(op)


@torch.no_grad()
def fake_task_vector(tv, names, seed, svd_cache=None, task_id=None):
    """Norm-matched fake neighbour (control for 'harm is about task content').

    Matrices keep the task's singular values but get random orthonormal singular vectors;
    every other tensor is replaced by Gaussian noise with the same Frobenius norm.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    out = {}
    for n in names:
        t = tv[n]
        if is_matrix(n, t):
            if svd_cache is not None and task_id is not None:
                _, s, _ = svd_cache.get(task_id, n, t.device)
            else:
                s = torch.linalg.svdvals(t)
            m, k = t.shape
            r = s.shape[0]
            qa, _ = torch.linalg.qr(torch.randn(m, r, generator=g).to(t.device))
            qb, _ = torch.linalg.qr(torch.randn(k, r, generator=g).to(t.device))
            out[n] = (qa * s) @ qb.T
        else:
            z = torch.randn(t.shape, generator=g).to(t.device)
            out[n] = z * (t.norm() / z.norm().clamp_min(1e-12))
    return out


def hybrid(params_a, params_b, units, from_a):
    """Parameters taken from A for depth units in `from_a`, from B everywhere else."""
    from_a = set(from_a)
    return {n: (params_a[n] if units[n] in from_a else params_b[n]) for n in params_a}


def hybrid_sel(params_a, params_b, units, kinds, select_a):
    """Parameters from A where select_a(unit, kind) is true, from B elsewhere."""
    return {n: (params_a[n] if select_a(units[n], kinds[n]) else params_b[n]) for n in params_a}


def in_units(unit_set, kind=None):
    """Predicate: parameter's unit is in unit_set (and, if given, its kind matches)."""
    us = set(unit_set)
    if kind is None:
        return lambda u, k: u in us
    return lambda u, k: (u in us) and (k == kind)


@torch.no_grad()
def spectra(D):
    """Singular values (float64) of every matrix in D; computed once and reused across seeds."""
    return {n: torch.linalg.svdvals(D[n].double()) for n in D if is_matrix(n, D[n])}


@torch.no_grad()
def matched_random_like(D, seed, svals=None):
    """Spectrum- and norm-matched random perturbation of a dict of tensors.

    2-D matrices (as in `is_matrix`) keep their singular values with random orthonormal
    singular vectors; every other tensor becomes Gaussian noise with the same Frobenius norm.
    Returns (perturbation dict, max relative spectrum error, max relative norm error).
    """
    if svals is None:
        svals = spectra(D)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    out, spec_err, norm_err = {}, 0.0, 0.0
    for n in sorted(D):
        t = D[n]
        if n in svals:
            s = svals[n]
            m, k = t.shape
            r = s.shape[0]
            qa, _ = torch.linalg.qr(torch.randn(m, r, generator=g, dtype=torch.float64).to(t.device))
            qb, _ = torch.linalg.qr(torch.randn(k, r, generator=g, dtype=torch.float64).to(t.device))
            p = ((qa * s) @ qb.T).to(t.dtype)
            if float(s.max()) > 0:
                s2 = torch.linalg.svdvals(p.double())
                spec_err = max(spec_err, float((s2 - s).abs().max() / s.max()))
            out[n] = p
        else:
            z = torch.randn(t.shape, generator=g, dtype=torch.float64).to(t.device)
            tn = float(t.double().norm())
            if tn > 0:
                p = (z * (tn / max(float(z.norm()), 1e-30))).to(t.dtype)
                norm_err = max(norm_err, abs(float(p.double().norm()) - tn) / tn)
            else:
                p = torch.zeros_like(t)
            out[n] = p
    return out, spec_err, norm_err
