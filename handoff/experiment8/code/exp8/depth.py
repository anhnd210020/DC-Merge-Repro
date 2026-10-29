"""Stage B: the depth steps (report half). Mixed into the Runner (see run.py).

Notation (guide, "Quantities measured"): A = merge(S), B = merge(S minus harmers); the hybrid H_G
takes A's parameters in the thirds G and B's elsewhere; u(G) = f_v(B) - f_v(H_G).
"""
import os

import numpy as np
import torch

from . import config as C
from .harm import shapley_from_value, subsets
from .operators import hybrid_sel, in_units, is_matrix, matched_random_like, spectra
from .pipeline import log, pair_id
from .probes import fit_probe
from .store import read_json, write_json

THIRDS = C.THIRDS


def gkey(G):
    return "+".join(t for t in THIRDS if t in G) or "none"


class DepthSteps:
    # ------------------------------------------------------------------ helpers
    def _AB(self, op, p):
        a = p["alpha"]
        S = tuple(p["context"])
        sub_b = tuple(t for t in S if t not in p["harmers"])
        return self.merged_params(op, S, a), self.merged_params(op, sub_b, a), S, sub_b

    def _thirds(self):
        return C.thirds(self.n_layers)

    def _units_of(self, G):
        th = self._thirds()
        return [u for g in THIRDS if g in G for u in th[g]]

    def _hyb(self, A, B, pred):
        return hybrid_sel(A, B, self.assets.units, self.assets.kinds, pred)

    def _json(self, op, name):
        path = os.path.join(self.op_dir(op), name)
        return path, (read_json(path) if os.path.exists(path) else {"pairs": {}})

    def factorial_cond(self, p, G):
        return {"step": "factorial", "pair": p["id"], "G": gkey(G)}

    # ------------------------------------------------------------------ B1 factorial
    def step_factorial(self, op):
        path, res = self._json(op, "factorial.json")
        res["n_layers"] = self.n_layers
        for p in self.pairs(op):
            if p["id"] in res["pairs"]:
                continue
            A, B, S, sub_b = self._AB(op, p)
            v, tasks = p["victim"], list(S)
            fG = {}
            for G in subsets(THIRDS):
                units = self._units_of(G)
                cs = self.cond_correct(op, self.factorial_cond(p, G), tasks, "rep",
                                       lambda units=units: self._hyb(A, B, in_units(units)))
                fG[gkey(G)] = {t: self.nacc(cs[t], t, "rep") for t in tasks}
                if not G:
                    ref = self.merge_correct(op, sub_b, p["alpha"], [v], "rep")[v]
                    if not np.array_equal(ref, cs[v]):
                        raise SystemExit(f"must-hold: H_empty != B for {p['id']}")
                if len(G) == 3:
                    ref = self.merge_correct(op, S, p["alpha"], [v], "rep")[v]
                    if not np.array_equal(ref, cs[v]):
                        raise SystemExit(f"must-hold: H_all != A for {p['id']}")
            row = summarize_factorial(fG, v, p["harmers"], tasks)
            res["pairs"][p["id"]] = dict(row, id=p["id"], victim=v, context=list(S), harmers=p["harmers"])
            write_json(path, res)
            log(f"[{op}] factorial {p['id']}: Gamma={row['Gamma']:.2f} "
                f"phi E/M/Lt = {row['phi']['E']:.2f}/{row['phi']['M']:.2f}/{row['phi']['Lt']:.2f}")

    # ------------------------------------------------------------------ B2 blocks
    def step_blocks(self, op):
        path, res = self._json(op, "blocks.json")
        L = self.n_layers
        fac = self._json(op, "factorial.json")[1]["pairs"]
        for p in self.pairs(op):
            if p["id"] in res["pairs"]:
                continue
            A, B, S, sub_b = self._AB(op, p)
            v = p["victim"]
            fr = fac[p["id"]]
            fA, fB, Gam = fr["f"]["E+M+Lt"][v], fr["f"]["none"][v], fr["Gamma"]
            units = ["pre"] + list(range(L)) + ["post"]
            row = {"units": [str(u) for u in units], "removed": {}, "created": {}, "parts": {}}

            def ev(cond, pred):
                c = self.cond_correct(op, dict(cond, pair=p["id"]), [v], "rep",
                                      lambda: self._hyb(A, B, pred))[v]
                return self.nacc(c, v, "rep")

            for u in units:
                fr_ = ev({"step": "unit", "dir": "removed", "u": str(u)}, lambda uu, k, u=u: uu != u)
                fc_ = ev({"step": "unit", "dir": "created", "u": str(u)}, lambda uu, k, u=u: uu == u)
                row["removed"][str(u)] = (fr_ - fA) / Gam if abs(Gam) > 1e-9 else None
                row["created"][str(u)] = (fB - fc_) / Gam if abs(Gam) > 1e-9 else None
            th = self._thirds()
            for g in THIRDS:
                for kind in ("attn", "mlp"):
                    us = set(th[g])
                    fr_ = ev({"step": "part", "dir": "removed", "g": g, "kind": kind},
                             lambda uu, k, us=us, kind=kind: not (uu in us and k == kind))
                    fc_ = ev({"step": "part", "dir": "created", "g": g, "kind": kind},
                             lambda uu, k, us=us, kind=kind: (uu in us and k == kind))
                    row["parts"][f"{g}:{kind}"] = {
                        "removed": (fr_ - fA) / Gam if abs(Gam) > 1e-9 else None,
                        "created": (fB - fc_) / Gam if abs(Gam) > 1e-9 else None,
                        "removed_pp": fr_ - fA, "created_pp": fB - fc_}
            rs = [x for x in row["removed"].values() if x is not None]
            row["sum_single_removed"] = float(np.sum(rs)) if rs else None
            row["Gamma"] = Gam
            res["pairs"][p["id"]] = dict(row, id=p["id"], victim=v)
            write_json(path, res)
            log(f"[{op}] blocks {p['id']} done")

    # ------------------------------------------------------------------ B3 random
    @torch.no_grad()
    def step_random(self, op):
        path, res = self._json(op, "random.json")
        fac = self._json(op, "factorial.json")[1]["pairs"]
        th = self._thirds()
        names = self.assets.names
        for pi, p in enumerate(self.pairs(op)):
            if p["id"] in res["pairs"]:
                continue
            A, B, S, sub_b = self._AB(op, p)
            v = p["victim"]
            fr = fac[p["id"]]
            fB = fr["f"]["none"][v]
            row = {"rand": {}, "rand_by_seed": {}, "excess": {}, "frob": {}, "frob_by_block": {}}
            for gi, g in enumerate(THIRDS):
                us = set(th[g])
                D = {n: (A[n] - B[n]) for n in names if self.assets.units[n] in us}
                row["frob"][g] = float(torch.sqrt(sum((d.double() ** 2).sum() for d in D.values())))
                sv = spectra(D)
                vals = []
                for s in range(1, C.DEPTH["random_seeds"] + 1):
                    seed = 10000 * pi + 10 * gi + s
                    cond = {"step": "random", "pair": p["id"], "g": g, "seed": seed}
                    key = self.cond_key(op, cond, v, "rep")
                    c = self.store(op).get(key)
                    if c is None:
                        P, se, ne = matched_random_like(D, seed, svals=sv)
                        if se > 1e-5 or ne > 1e-5:
                            raise SystemExit(f"must-hold: random perturbation mismatch ({se:.2e}, {ne:.2e})")
                        params = {n: (B[n] + P[n]) if n in P else B[n] for n in names}
                        c = self.cond_correct(op, cond, [v], "rep", lambda params=params: params)[v]
                        del P, params
                    vals.append(fB - self.nacc(c, v, "rep"))
                row["rand_by_seed"][g] = vals
                row["rand"][g] = float(np.mean(vals))
                row["excess"][g] = fr["c"][g] - row["rand"][g]
                del D, sv
            for u in ["pre"] + list(range(self.n_layers)) + ["post"]:
                row["frob_by_block"][str(u)] = float(torch.sqrt(sum(
                    ((A[n] - B[n]).double() ** 2).sum() for n in names if self.assets.units[n] == u)))
            res["pairs"][p["id"]] = dict(row, id=p["id"], victim=v)
            write_json(path, res)
            log(f"[{op}] random {p['id']}: excess E/M/Lt = "
                + "/".join(f"{row['excess'][g]:.2f}" for g in THIRDS))

    # ------------------------------------------------------------------ B4 own
    def step_own(self, op):
        path, res = self._json(op, "own.json")
        a = self.alpha(op)
        victims = sorted({p["victim"] for p in self.pairs(op)}, key=C.TASKS.index)
        for v in victims:
            if v in res["pairs"]:
                continue
            O = self.merged_params(op, (v,), a)
            Z = self.assets.theta0
            f = {}
            for G in subsets(THIRDS):
                units = self._units_of(G)
                c = self.cond_correct(op, {"step": "own", "victim": v, "G": gkey(G)}, [v], "rep",
                                      lambda units=units: self._hyb(O, Z, in_units(units)))[v]
                f[G] = self.nacc(c, v, "rep")
            w = {G: f[G] - f[frozenset()] for G in f}
            d, phi = shapley_from_value(w, THIRDS)
            gain = w[frozenset(THIRDS)]
            res["pairs"][v] = {"victim": v, "gain": gain, "phi": phi,
                               "share": {g: (phi[g] / gain if abs(gain) > 1e-9 else None) for g in THIRDS}}
            write_json(path, res)
            log(f"[{op}] own {v}: gain {gain:.2f}, shares " +
                "/".join(f"{phi[g] / gain:.2f}" if abs(gain) > 1e-9 else "-" for g in THIRDS))

    # ------------------------------------------------------------------ B5 probes
    @torch.no_grad()
    def _cls_states(self, params, task, split):
        self.ev.load(params, tag=None)
        x, _ = self.ev.images(task, split)
        feats = None
        for s in range(0, len(x), self.args.batch_size):
            hs, _ = self.ev.hidden_all(task, split, s, min(len(x), s + self.args.batch_size))
            st = torch.stack([h[:, 0, :] for h in hs]).float()
            feats = st if feats is None else torch.cat([feats, st], 1)
        return feats  # (L+1, n, d)

    def _probe_model(self, op, tag, params, v, depths):
        """Probe correctness per depth for one model, cached in the store."""
        keys = {l: self._base_key(kind="probe", op=op, model=tag, task=v, depth=l, split="probe") for l in depths}
        st = self.store(op)
        have = {l: st.get(k) for l, k in keys.items()}
        if all(c is not None for c in have.values()):
            return have, None
        Xtr = self._cls_states(params, v, "screen")
        Xte = self._cls_states(params, v, "probe")
        td = self.ev.taskdata(v)
        ytr, yte = td.labels[td.indices("screen")], td.labels[td.indices("probe")]
        P = C.DEPTH["probe"]
        lams = {}
        for l in depths:
            if have[l] is None:
                lam, corr = fit_probe(Xtr[l], ytr, Xte[l], yte, self.assets.n_classes[v], P["lambdas"],
                                      P["holdout"], P["iters"], seed=C.SEED)
                st.put(keys[l], corr)
                have[l] = corr
                lams[l] = lam
        return have, lams

    def step_probes(self, op):
        path, res = self._json(op, "probes.json")
        L = self.n_layers
        depths = list(range(L + 1)) if self.args.probe_depths == "all" else sorted(
            {b for be in C.boundaries(L).values() for b in be})
        for p in self.pairs(op):
            if p["id"] in res["pairs"]:
                continue
            A, B, S, sub_b = self._AB(op, p)
            v = p["victim"]
            cA, lamA = self._probe_model(op, "A|" + p["id"], A, v, depths)
            cB, lamB = self._probe_model(op, "B|" + p["id"], B, v, depths)
            cE, _ = self._probe_model(op, "expert|" + v, self.assets.expert_params(v), v, depths)
            cZ, _ = self._probe_model(op, "zeroshot|" + v, self.assets.theta0, v, depths)
            acc = {m: {str(l): 100.0 * float(c[l].mean()) for l in depths}
                   for m, c in (("A", cA), ("B", cB), ("expert", cE), ("zeroshot", cZ))}
            delta = {str(l): acc["B"][str(l)] - acc["A"][str(l)] for l in depths}
            bnd = C.boundaries(L)
            lam = {g: delta[str(b1)] - delta[str(b0)] for g, (b0, b1) in bnd.items()
                   if str(b0) in delta and str(b1) in delta}
            pos = self.ev.taskdata(v).positions_in_rep("probe")
            hA = self.stored_cond(op, self.factorial_cond(p, frozenset(THIRDS)), v, "rep")[pos]
            hB = self.stored_cond(op, self.factorial_cond(p, frozenset()), v, "rep")[pos]
            gam_raw = 100.0 * float(hB.mean() - hA.mean())
            head_E = 100.0 * float(self.expert_correct(v, "rep")[pos].mean())
            warn = []
            if acc["expert"][str(L)] < head_E - 2.0:
                warn.append(f"expert final probe {acc['expert'][str(L)]:.1f} < head {head_E:.1f} - 2")
            if acc["zeroshot"][str(L)] >= acc["expert"][str(L)]:
                warn.append("zero-shot probe not below expert probe")
            res["pairs"][p["id"]] = {"id": p["id"], "victim": v, "depths": depths, "acc": acc,
                                     "delta": delta, "lambda": lam, "gamma_raw_probe": gam_raw,
                                     "psi": (delta[str(L)] / gam_raw) if abs(gam_raw) > 1e-9 else None,
                                     "l2_chosen": {"A": lamA, "B": lamB}, "warnings": warn}
            write_json(path, res)
            for w_ in warn:
                log(f"[{op}] probes WARNING {p['id']}: {w_}")
            log(f"[{op}] probes {p['id']}: lambda E/M/Lt = " + "/".join(f"{lam.get(g, float('nan')):.2f}" for g in THIRDS))

    # ------------------------------------------------------------------ B6 patch
    @torch.no_grad()
    def step_patch(self, op):
        path, res = self._json(op, "patch.json")
        L = self.n_layers
        for p in self.pairs(op):
            if p["id"] in res["pairs"]:
                continue
            A, B, S, sub_b = self._AB(op, p)
            v = p["victim"]
            y = torch.as_tensor(self.ev.labels(v, "patch"), device=self.device)
            n = len(y)
            restored = np.zeros(L)
            accA = accB = 0.0
            ident_ok = True
            for s in range(0, n, self.args.batch_size):
                e = min(n, s + self.args.batch_size)
                self.ev.load(B, tag=None)
                hb, lb = self.ev.hidden_all(v, "patch", s, e)
                accB += (lb.argmax(-1) == y[s:e]).sum().item()
                self.ev.load(A, tag=None)
                ha, la = self.ev.hidden_all(v, "patch", s, e)
                accA += (la.argmax(-1) == y[s:e]).sum().item()
                lid = self.ev.logits_with_patch(v, "patch", s, e, L // 2, ha[L // 2 + 1])
                ident_ok &= bool(torch.equal(lid.argmax(-1), la.argmax(-1)))
                for l in range(L):
                    lg = self.ev.logits_with_patch(v, "patch", s, e, l, hb[l + 1])
                    restored[l] += (lg.argmax(-1) == y[s:e]).sum().item()
            if not ident_ok:
                raise SystemExit(f"must-hold: patching A with its own state changed predictions ({p['id']})")
            accA, accB = accA / n, accB / n
            gap = accB - accA
            share = [((r / n) - accA) / gap if abs(gap) > 1e-9 else None for r in restored]
            res["pairs"][p["id"]] = {"id": p["id"], "victim": v, "acc_A_raw": accA, "acc_B_raw": accB,
                                     "restored_share_by_block": share, "n_images": n}
            write_json(path, res)
            log(f"[{op}] patch {p['id']} done")

    # ------------------------------------------------------------------ B7 cross-term
    @torch.no_grad()
    def step_crossterm(self, op):
        path, res = self._json(op, "crossterm.json")
        L = self.n_layers
        a = self.alpha(op)
        old_cache = self.args.merge_cache
        self.args.merge_cache = max(old_cache, 5)
        for p in self.pairs(op):
            if p["id"] in res["pairs"]:
                continue
            if len(p["harmers"]) < 2:
                res["pairs"][p["id"]] = {"id": p["id"], "skipped": "fewer than two harmers"}
                write_json(path, res)
                continue
            v, S = p["victim"], tuple(p["context"])
            pq = sorted(p["harmers"], key=lambda j: -p["shapley_sel"][j])[:2]
            R = tuple(t for t in S if t not in pq)
            corners = {"R": R, "p": R + (pq[0],), "q": R + (pq[1],), "S": S}
            params = {k: self.merged_params(op, sub, a) for k, sub in corners.items()}
            y = torch.as_tensor(self.ev.labels(v, "crossterm"), device=self.device)
            n, bs = len(y), 50
            acc = {"I_cls": np.zeros(L + 1), "I_all": np.zeros(L + 1), "G": np.zeros(L), "T": np.zeros(L),
                   "M": np.zeros(L), "id_err": 0.0, "runner_err": 0.0}
            correct_actual = correct_add = correct_R = 0
            for s in range(0, n, bs):
                e = min(n, s + bs)
                hs, lg = {}, {}
                for k in corners:
                    self.ev.load(params[k], tag=None)
                    hs[k], lg[k] = self.ev.hidden_all(v, "crossterm", s, e)
                I = [hs["S"][l] - hs["p"][l] - hs["q"][l] + hs["R"][l] for l in range(L + 1)]
                for l in range(L + 1):
                    acc["I_cls"][l] += I[l][:, 0].norm(dim=-1).sum().item()
                    acc["I_all"][l] += I[l].flatten(1).norm(dim=-1).sum().item()
                hbar = [hs["p"][l] + hs["q"][l] - hs["R"][l] for l in range(L)]
                F = {}
                for k in corners:
                    self.ev.load(params[k], tag=None)
                    F[k] = [self.ev.run_block(l, hbar[l]) for l in range(L)]
                    if k == "S":
                        for l in range(L):
                            out = self.ev.run_block(l, hs["S"][l])
                            acc["runner_err"] = max(acc["runner_err"], float(
                                (out - hs["S"][l + 1]).norm() / hs["S"][l + 1].norm().clamp_min(1e-12)))
                for l in range(L):
                    G_ = F["S"][l] - F["p"][l] - F["q"][l] + F["R"][l]
                    T_ = hs["S"][l + 1] - F["S"][l]
                    M_ = (F["p"][l] - hs["p"][l + 1]) + (F["q"][l] - hs["q"][l + 1]) - (F["R"][l] - hs["R"][l + 1])
                    acc["G"][l] += G_.flatten(1).norm(dim=-1).sum().item()
                    acc["T"][l] += T_.flatten(1).norm(dim=-1).sum().item()
                    acc["M"][l] += M_.flatten(1).norm(dim=-1).sum().item()
                    acc["id_err"] = max(acc["id_err"], float((I[l + 1] - (G_ + T_ + M_)).norm()
                                                           / I[l + 1].norm().clamp_min(1e-12)))
                add = lg["p"] + lg["q"] - lg["R"]
                correct_actual += (lg["S"].argmax(-1) == y[s:e]).sum().item()
                correct_add += (add.argmax(-1) == y[s:e]).sum().item()
                correct_R += (lg["R"].argmax(-1) == y[s:e]).sum().item()
            if acc["runner_err"] > 1e-4 or acc["id_err"] > 1e-4:
                raise SystemExit(f"must-hold: cross-term check failed for {p['id']} "
                                 f"(runner {acc['runner_err']:.2e}, identity {acc['id_err']:.2e})")
            mean = {k: (acc[k] / n).tolist() for k in ("I_cls", "I_all", "G", "T", "M")}
            gain = [mean["T"][l] / mean["I_all"][l] if mean["I_all"][l] > 0 else None for l in range(L)]
            res["pairs"][p["id"]] = {"id": p["id"], "victim": v, "pair": pq, "n_images": n, "norms": mean,
                                     "transport_gain": gain, "runner_rel_error": acc["runner_err"],
                                     "identity_rel_error": acc["id_err"],
                                     "acc_actual_raw": correct_actual / n, "acc_additive_raw": correct_add / n,
                                     "acc_without_pair_raw": correct_R / n}
            write_json(path, res)
            log(f"[{op}] crossterm {p['id']} done")
        self.args.merge_cache = old_cache

    # ------------------------------------------------------------------ B8 weight space
    @torch.no_grad()
    def step_weightspace(self, op):
        from scipy.stats import spearmanr
        path, res = self._json(op, "weightspace.json")
        blocks = self._json(op, "blocks.json")[1]["pairs"]
        A_ = self.assets
        L = self.n_layers
        th = self._thirds()
        mats = [n for n in A_.names if is_matrix(n, A_.theta0[n]) and isinstance(A_.units[n], int)]
        for p in self.pairs(op):
            if p["id"] in res["pairs"]:
                continue
            S, v, Nh = p["context"], p["victim"], p["harmers"]
            sti, ovl, cnt = np.zeros(L), np.zeros(L), np.zeros(L)
            for n in mats:
                b = A_.units[n]
                T = len(S)
                Us, Ss, Vs = [], [], []
                for t in S:
                    u, s, vh = self.svd.get(t, n, self.device)
                    k = max(1, int(s.shape[0] / T))
                    Us.append(u[:, :k]); Ss.append(s[:k]); Vs.append(vh[:k, :].T)
                U, Sd, V = torch.cat(Us, 1), torch.cat(Ss), torch.cat(Vs, 1)
                I = torch.eye(U.shape[1], device=U.device)
                sti[b] += float(((U.T @ U - I) @ torch.diag(Sd) @ (V.T @ V - I)).abs().sum())
                uv = Us[S.index(v)]
                ovl[b] += float(np.mean([float((uv.T @ Us[S.index(j)]).pow(2).sum() / uv.shape[1]) for j in Nh]))
                cnt[b] += 1
            ovl = ovl / np.maximum(cnt, 1)
            row = {"id": p["id"], "victim": v, "sti_by_block": sti.tolist(), "overlap_by_block": ovl.tolist(),
                   "sti_by_third": {g: float(sum(sti[u] for u in th[g] if isinstance(u, int))) for g in THIRDS}}
            br = blocks.get(p["id"])
            if br:
                rem = np.array([br["removed"][str(l)] if br["removed"][str(l)] is not None else np.nan
                                for l in range(L)], dtype=float)
                ok = np.isfinite(rem)
                if ok.sum() >= 3 and np.ptp(sti[ok]) > 0 and np.ptp(rem[ok]) > 0:
                    rho = spearmanr(sti[ok], rem[ok]).correlation
                    row["spearman_sti_vs_removed"] = None if np.isnan(rho) else float(rho)
                row["sti_peak_block"] = int(np.argmax(sti))
                row["removed_peak_block"] = int(np.nanargmax(rem)) if ok.any() else None
            res["pairs"][p["id"]] = row
            write_json(path, res)
        log(f"[{op}] weightspace: {len(res['pairs'])} pairs")

    # ------------------------------------------------------------------ B10 replication
    def step_replicate_select(self, op, source_selection):
        """Re-derive harmers and eligibility on this backbone for the ViT-B/32 pairs."""
        from .harm import decompose
        a = self.alpha(op)
        src = read_json(source_selection)["pairs"]
        cands = []
        for q in src:
            v, S = q["victim"], tuple(q["context"])
            N, f, corr = self._f_table(op, a, v, S, "screen")
            r = decompose(f, N)
            Nh = [j for j in N if r["shapley"][j] > 0]
            if not Nh:
                continue
            sub_b = tuple(t for t in S if t not in Nh)
            cA = corr[frozenset(N)]
            cB = self.merge_correct(op, sub_b, a, [v], "screen")[v]
            e = self.expert_correct(v, "screen")
            g = self.nacc(cB, v, "screen") - self.nacc(cA, v, "screen")
            cnt = np.random.RandomState(C.SEED).multinomial(len(e), np.full(len(e), 1.0 / len(e)),
                                                             size=C.BOOTSTRAP).astype(np.float64)
            gb = 100.0 * (cnt @ (cB.astype(float) - cA.astype(float))) / np.maximum(cnt @ e.astype(float), 1e-12)
            lo, hi = float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5))
            if lo > 0:
                loss = self._harmer_loss(op, a, S, Nh)
                if sum(loss.values()) > 0:
                    cands.append({"victim": v, "context": list(S), "harmers": Nh, "gamma_sel": g,
                                  "gamma_sel_ci": [lo, hi], "shapley_sel": r["shapley"], "H_sel": r["H"],
                                  "harmer_loss_sel": loss})
        cands.sort(key=lambda c: -c["gamma_sel"])
        chosen, seen = [], set()
        for c in cands:                      # first the best pair of each distinct victim
            if c["victim"] not in seen:
                chosen.append(c); seen.add(c["victim"])
        for c in cands:                      # then fill remaining slots
            if len(chosen) >= C.DEPTH["replicate_max_pairs"]:
                break
            if c not in chosen:
                chosen.append(c)
        chosen = chosen[:C.DEPTH["replicate_max_pairs"]]
        pairs = []
        for rank, c in enumerate(chosen, 1):
            pairs.append(dict(c, rank=rank, op=op, alpha=a, id=pair_id(op, c["victim"], c["context"])))
        write_json(os.path.join(self.op_dir(op), "selection.json"),
                   {"op": op, "alpha": a, "pairs": pairs, "victims": sorted({p["victim"] for p in pairs}),
                    "n_candidates": len(src), "n_eligible": len(cands), "config_sha256": C.config_sha256(),
                    "split": "screen", "source": source_selection})
        log(f"[{op}] replicate select: {len(pairs)} of {len(cands)} eligible pairs kept")


def summarize_factorial(fG, v, harmers, tasks):
    """All factorial quantities for one pair from its 8-condition table of normalized accuracies."""
    full, none = "E+M+Lt", "none"
    fv = {G: fG[gkey(G)][v] for G in subsets(THIRDS)}
    fB, fA = fv[frozenset()], fv[frozenset(THIRDS)]
    Gam = fB - fA
    u = {G: fB - fv[G] for G in fv}
    d, phi = shapley_from_value(u, THIRDS)
    c = {g: u[frozenset([g])] for g in THIRDS}
    r = {g: Gam - u[frozenset(t for t in THIRDS if t != g)] for g in THIRDS}
    kappa = r["M"] - c["M"]
    repair = {}
    for g in THIRDS:
        k = gkey(frozenset(t for t in THIRDS if t != g))
        gain = fG[k][v] - fA
        hl = sum(fG[full][h] - fG[k][h] for h in harmers)
        hf = sum(fG[full][h] - fG[none][h] for h in harmers)
        accs = [fG[k][t] for t in tasks]
        repair[g] = {"victim_gain": gain, "rho_v": gain / Gam if abs(Gam) > 1e-9 else None,
                     "harmer_loss": hl, "harmer_full_loss_rep": hf,
                     "rho_h": hl / hf if abs(hf) > 1e-9 else None,
                     "mean": float(np.mean(accs)), "worst": float(min(accs))}
    task_phi = {}
    for t in tasks:
        ut = {G: fG[none][t] - fG[gkey(G)][t] for G in subsets(THIRDS)}
        task_phi[t] = shapley_from_value(ut, THIRDS)[1]
    return {"f": fG, "Gamma": Gam, "u": {gkey(G): x for G, x in u.items()}, "phi": phi,
            "dividends": {gkey(T): x for T, x in d.items()}, "c": c, "r": r, "kappa_M": kappa,
            "sigma": {g: (phi[g] / Gam if abs(Gam) > 1e-9 else None) for g in THIRDS},
            "repair": repair, "task_phi": task_phi,
            "mean_A": float(np.mean([fG[full][t] for t in tasks])),
            "worst_A": float(min(fG[full][t] for t in tasks))}
