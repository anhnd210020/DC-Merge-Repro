"""The Experiment 8 analysis: five confirmatory depth claims, the 'who' family, sensitivity
and exploratory summaries. Reads per-image correctness from the evaluation caches; loads no model.

Victim-level statistic for contrast x (pp) under operator o:
    z_{v,o} = sum_{pairs of (o,v)} x_p / sum_{pairs of (o,v)} Gamma_sel_p ;  z_v = mean_o z_{v,o}
Claim 5 uses z_{v,o} = sum gain_mid / sum Gamma_sel - sum harmer_loss_mid / sum harmer_full_loss_sel.
"""
import os

import numpy as np

from . import config as C
from .harm import shapley_from_value, subsets
from .stats import (fixed_sequence, holm, iut, multinomial_counts, percentile_ci,
                    ratio_of_sums, sign_flip_test)
from .store import read_json, write_json

THIRDS = C.THIRDS
CONTRASTS = {  # hypothesis -> list of (contrast name, alternative)
    "localization": [("phi_M-phi_E", "greater"), ("phi_M-phi_Lt", "greater")],
    "specificity": [("e_M-e_E", "greater"), ("e_M-e_Lt", "greater")],
    "information": [("lam_M-lam_E", "greater"), ("lam_M-lam_Lt", "greater")],
    "amplification": [("kappa_M", "greater")],
    "repair": [("repair", "greater")],
}


def gkey(G):
    return "+".join(t for t in THIRDS if t in G) or "none"


class PairData:
    """All per-image arrays one pair needs, and its statistics under image weights."""

    def __init__(self, runner, op, p, L, n_seeds):
        self.p, self.op = p, op
        self.v, self.harmers = p["victim"], p["harmers"]
        self.gamma_sel = p["gamma_sel"]
        self.harmer_full_sel = sum(p["harmer_loss_sel"].values()) if "harmer_loss_sel" in p else None
        cond = lambda G: {"step": "factorial", "pair": p["id"], "G": gkey(G)}
        tasks = [self.v] + list(self.harmers)
        self.e = {t: runner.expert_correct(t, "rep").astype(np.float64) for t in tasks}
        self.fac = {t: {gkey(G): runner.stored_cond(op, cond(G), t, "rep").astype(np.float64)
                        for G in subsets(THIRDS)} for t in tasks}
        self.n = {t: len(self.e[t]) for t in tasks}
        self.rand = {}
        for gi, g in enumerate(THIRDS):
            self.rand[g] = []
            for s in range(1, n_seeds + 1):
                seed = 10000 * p["_index"] + 10 * gi + s
                key = runner.cond_key(op, {"step": "random", "pair": p["id"], "g": g, "seed": seed}, self.v, "rep")
                c = runner.store(op).get(key)
                self.rand[g].append(None if c is None else c.astype(np.float64))
        self.has_random = all(x is not None for g in THIRDS for x in self.rand[g])
        self.bnd = C.boundaries(L)
        depths = sorted({b for be in self.bnd.values() for b in be})
        self.probe = {}
        for m, tag in (("A", "A|" + p["id"]), ("B", "B|" + p["id"])):
            self.probe[m] = {}
            for l in depths:
                c = runner.store(op).get(runner._base_key(kind="probe", op=op, model=tag, task=self.v,
                                                           depth=l, split="probe"))
                self.probe[m][l] = None if c is None else c.astype(np.float64)
        self.has_probe = all(c is not None for m in self.probe for c in self.probe[m].values())
        self.pos = runner.taskdata(self.v).positions_in_rep("probe")

    def stats(self, w=None, seeds=None):
        """w: dict task -> image weights (None = observed data)."""
        def f(arr, t):
            ww = None if w is None else w[t]
            if ww is None:
                return 100.0 * arr.mean() / self.e[t].mean()
            return 100.0 * (ww @ arr) / max(ww @ self.e[t], 1e-12)

        v = self.v
        fv = {G: f(self.fac[v][gkey(G)], v) for G in subsets(THIRDS)}
        fB, fA = fv[frozenset()], fv[frozenset(THIRDS)]
        Gam = fB - fA
        u = {G: fB - fv[G] for G in fv}
        d, phi = shapley_from_value(u, THIRDS)
        c = {g: u[frozenset([g])] for g in THIRDS}
        r = {g: Gam - u[frozenset(t for t in THIRDS if t != g)] for g in THIRDS}
        out = {"Gamma": Gam, "phi": phi, "c": c, "r": r, "kappa_M": r["M"] - c["M"]}
        if self.has_random:
            sidx = seeds or list(range(len(self.rand["E"])))
            rand = {g: float(np.mean([fB - f(self.rand[g][i], v) for i in sidx])) for g in THIRDS}
            out["e"] = {g: c[g] - rand[g] for g in THIRDS}
        if self.has_probe:
            ww = None if w is None else w[v][self.pos]

            def acc(arr):
                return 100.0 * (arr.mean() if ww is None else (ww @ arr) / max(ww.sum(), 1e-12))

            delta = {l: acc(self.probe["B"][l]) - acc(self.probe["A"][l]) for l in self.probe["A"]}
            out["lam"] = {g: delta[b1] - delta[b0] for g, (b0, b1) in self.bnd.items()}
            hA = self.fac[v][gkey(THIRDS)][self.pos]
            hB = self.fac[v]["none"][self.pos]
            out["Delta_L"] = delta[max(delta)]
            out["Gamma_raw_probe"] = acc(hB) - acc(hA)
        k = gkey(frozenset(["E", "Lt"]))
        out["gain_mid"] = f(self.fac[v][k], v) - fA
        out["harmer_loss_mid"] = sum(f(self.fac[h]["E+M+Lt"], h) - f(self.fac[h][k], h) for h in self.harmers)
        out["harmer_full_rep"] = sum(f(self.fac[h]["E+M+Lt"], h) - f(self.fac[h]["none"], h)
                                     for h in self.harmers)
        return out

    def draw_weights(self, rng):
        return {t: multinomial_counts(rng, n) for t, n in self.n.items()}


def contrast_value(name, s):
    if name.startswith("phi_M-"):
        return s["phi"]["M"] - s["phi"][name.split("-")[1][4:]]
    if name.startswith("e_M-"):
        return s["e"]["M"] - s["e"][name.split("-")[1][2:]] if "e" in s else None
    if name.startswith("lam_M-"):
        return s["lam"]["M"] - s["lam"][name.split("-")[1][4:]] if "lam" in s else None
    if name == "kappa_M":
        return s["kappa_M"]
    raise ValueError(name)


def victim_z(items, name):
    """items: list of (PairData, stats). Returns {victim: z_v}."""
    by = {}
    for pdat, s in items:
        by.setdefault(pdat.v, {}).setdefault(pdat.op, []).append((pdat, s))
    z = {}
    for v, ops in by.items():
        vals = []
        for op, lst in ops.items():
            if name == "repair":
                den_h = sum(pd.harmer_full_sel for pd, _ in lst)
                vals.append(sum(s["gain_mid"] for _, s in lst) / sum(pd.gamma_sel for pd, _ in lst)
                            - sum(s["harmer_loss_mid"] for _, s in lst) / den_h)
            else:
                xs = [contrast_value(name, s) for _, s in lst]
                if any(x is None for x in xs):
                    continue
                vals.append(sum(xs) / sum(pd.gamma_sel for pd, _ in lst))
        if vals:
            z[v] = float(np.mean(vals))
    return z


def pooled(items):
    """Pooled effect sizes (ratios of sums over pairs)."""
    S = lambda key, sub=None: [ (s[key][sub] if sub else s[key]) for _, s in items if key in s]
    G = S("Gamma")
    out = {"n_pairs": len(items),
           "sigma": {g: ratio_of_sums(S("phi", g), G) for g in THIRDS},
           "kappa_share": ratio_of_sums(S("kappa_M"), G),
           "rho_v_mid": ratio_of_sums(S("gain_mid"), G),
           "rho_h_mid": ratio_of_sums(S("harmer_loss_mid"), S("harmer_full_rep"))}
    if all("e" in s for _, s in items) and items:
        out["excess_share"] = {g: ratio_of_sums(S("e", g), G) for g in THIRDS}
    if all("lam" in s for _, s in items) and items:
        out["lambda_mean_pp"] = {g: float(np.mean(S("lam", g))) for g in THIRDS}
        out["psi"] = ratio_of_sums(S("Delta_L"), S("Gamma_raw_probe"))
    return out


def test_family(items):
    res, ordered = {}, []
    for h in C.HYPOTHESIS_ORDER:
        comps = []
        for name, alt in CONTRASTS[h]:
            z = victim_z(items, name)
            t = sign_flip_test(list(z.values()), alternative=alt)
            comps.append(dict(t, contrast=name, z=z))
        p = iut(*[c["p"] for c in comps]) if all(c["p"] is not None for c in comps) else None
        res[h] = {"p": p, "components": comps}
        ordered.append((h, p))
    verdicts = fixed_sequence(ordered, C.DEPTH["alpha_level"])
    for h in res:
        res[h]["verdict"] = verdicts[h]
    return res


def mean_z(items, name):
    z = victim_z(items, name)
    return float(np.mean(list(z.values()))) if z else None


def cluster_bootstrap(pdatas, B, seed=0):
    """Two-stage cluster bootstrap: victims with replacement, then pairs within each
    (operator, victim) with replacement, then images (multinomial weights) within each pair.
    A victim drawn twice counts as two clusters."""
    rng = np.random.RandomState(seed)
    by = {}
    for pd_ in pdatas:
        by.setdefault(pd_.v, {}).setdefault(pd_.op, []).append(pd_)
    victims = sorted(by)
    names = [name for h in C.HYPOTHESIS_ORDER for name, _ in CONTRASTS[h]]
    draws = []
    for _ in range(B):
        clusters, items = [], []
        for vi in rng.randint(0, len(victims), len(victims)):
            cl = []
            for op in sorted(by[victims[vi]]):
                lst = by[victims[vi]][op]
                for j in rng.randint(0, len(lst), len(lst)):
                    pd_ = lst[j]
                    cl.append((pd_, pd_.stats(pd_.draw_weights(rng))))
            clusters.append(cl)
            items += cl
        d = {"pooled": pooled(items)}
        for name in names:
            zs = [z for cl in clusters for z in victim_z(cl, name).values()]
            d[name] = float(np.mean(zs)) if zs else None
        draws.append(d)
    return draws


def ci_of(draws, getter):
    vals = []
    for d in draws:
        try:
            x = getter(d)
        except (KeyError, TypeError):
            x = None
        vals.append(x)
    return percentile_ci(vals)


def run(runner, args):
    out = runner.out
    exploratory = bool(getattr(args, "exploratory", False))
    sel_path = os.path.join(out, "selection_all.json")
    if exploratory and getattr(args, "op", None):
        sel_path = os.path.join(out, args.op, "selection.json")
    elif getattr(args, "op", None):
        raise SystemExit("`depthstats --op` is only allowed with --exploratory (the confirmatory "
                         "analysis pools all operators)")
    if not os.path.exists(sel_path):
        raise SystemExit("run `select --all-ops` first" if not exploratory else f"missing {sel_path}")
    sel = read_json(sel_path)
    if sel["config_sha256"] != C.config_sha256():
        raise SystemExit("config.py changed after `select`; the pre-specified analysis cannot run")
    ops = [args.op] if getattr(args, "op", None) else sel["ops"]
    L = runner.n_layers_from_disk(ops)
    n_seeds = C.DEPTH["random_seeds"]
    pdatas = []
    for op in ops:
        for i, p in enumerate(runner.pairs(op)):
            p = dict(p, _index=i)
            pdatas.append(PairData(runner, op, p, L, n_seeds))
    if not pdatas:
        res = {"feasible": False, "note": "no eligible pairs; nothing to test", "n_victims": 0}
        _write(out, res, exploratory, ops)
        return res
    items = [(pd_, pd_.stats()) for pd_ in pdatas]
    victims = sorted({pd_.v for pd_ in pdatas}, key=C.TASKS.index)
    feasible = len(victims) >= C.DEPTH["min_victims"]
    res = {"ops": ops, "n_pairs": len(items), "victims": victims, "n_victims": len(victims),
           "feasible": feasible, "min_attainable_p": 2.0 ** (-len(victims)),
           "exploratory_run": exploratory, "split": "rep",
           "confirmatory": test_family(items), "estimates": pooled(items)}
    if not feasible or exploratory:
        why = (f"fewer than {C.DEPTH['min_victims']} victims; descriptive only" if not feasible
               else "exploratory run; not a confirmatory verdict")
        for h in res["confirmatory"]:
            res["confirmatory"][h]["verdict"] = f"not tested ({why})"
    draws = cluster_bootstrap(pdatas, args.bootstrap or C.BOOTSTRAP)
    res["intervals"] = {"sigma": {g: ci_of(draws, lambda d, g=g: d["pooled"]["sigma"][g]) for g in THIRDS},
                        "kappa_share": ci_of(draws, lambda d: d["pooled"]["kappa_share"]),
                        "rho_v_mid": ci_of(draws, lambda d: d["pooled"]["rho_v_mid"]),
                        "rho_h_mid": ci_of(draws, lambda d: d["pooled"]["rho_h_mid"]),
                        "psi": ci_of(draws, lambda d: d["pooled"]["psi"]),
                        "contrasts": {name: ci_of(draws, lambda d, n=name: d[n])
                                      for h in C.HYPOTHESIS_ORDER for name, _ in CONTRASTS[h]}}
    for h in C.HYPOTHESIS_ORDER:
        for comp in res["confirmatory"][h]["components"]:
            comp["ci"] = res["intervals"]["contrasts"][comp["contrast"]]
    # per operator (estimates and intervals; not tested)
    res["per_operator"] = {}
    for op in ops:
        sub = [pd_ for pd_ in pdatas if pd_.op == op]
        if not sub:
            continue
        it = [(pd_, pd_.stats()) for pd_ in sub]
        dr = cluster_bootstrap(sub, max(200, (args.bootstrap or C.BOOTSTRAP) // 4), seed=1)
        res["per_operator"][op] = {
            "estimates": pooled(it),
            "sigma_ci": {g: ci_of(dr, lambda d, g=g: d["pooled"]["sigma"][g]) for g in THIRDS},
            "contrasts": {name: {"T": mean_z(it, name), "ci": ci_of(dr, lambda d, n=name: d[n])}
                          for h in C.HYPOTHESIS_ORDER for name, _ in CONTRASTS[h]}}
    res["sensitivity"] = sensitivity(runner, ops, pdatas, items)
    res["exploratory"] = exploratory_summaries(runner, ops, pdatas, L)
    res["who"] = who_family(runner, ops, pdatas)
    _write(out, res, exploratory, ops)
    return res


def sensitivity(runner, ops, pdatas, items):
    s = {}
    k1 = [(pd_, st) for pd_, st in items if pd_.p.get("rank", 1) == 1]
    s["k1"] = {h: v["p"] for h, v in test_family(k1).items()} if k1 else None
    s["k1_T"] = {name: mean_z(k1, name) for h in C.HYPOTHESIS_ORDER for name, _ in CONTRASTS[h]} if k1 else None
    loo = {}
    victims = sorted({pd_.v for pd_ in pdatas})
    for h in C.HYPOTHESIS_ORDER:
        for name, _ in CONTRASTS[h]:
            full = mean_z(items, name)
            vals = [mean_z([it for it in items if it[0].v != v], name) for v in victims]
            vals = [x for x in vals if x is not None]
            loo[name] = {"T": full, "min": min(vals) if vals else None, "max": max(vals) if vals else None,
                         "sign_changes": bool(vals) and full is not None and any(np.sign(x) != np.sign(full) for x in vals)}
    s["leave_one_victim_out"] = loo
    ratios = [{g: st["phi"][g] / st["Gamma"] for g in THIRDS} for _, st in items if st["Gamma"] > 0]
    s["sigma_mean_of_ratios"] = {g: float(np.mean([r[g] for r in ratios])) for g in THIRDS} if ratios else None
    one = [(pd_, pd_.stats(seeds=[0])) for pd_ in pdatas if pd_.has_random]
    if one:
        s["specificity_one_seed"] = {name: sign_flip_test(list(victim_z(one, name).values()), alt)["p"]
                                     for name, alt in CONTRASTS["specificity"]}
    pp = {"pre": {"removed": [], "created": []}, "post": {"removed": [], "created": []}}
    for op in ops:
        bl = _load(runner.out, op, "blocks.json")
        for row in bl.values():
            for u in ("pre", "post"):
                for d in ("removed", "created"):
                    x = row[d].get(u)
                    if x is not None:
                        pp[u][d].append(x)
    s["pre_post_single_unit_mean_fraction"] = {u: {d: (float(np.mean(x)) if x else None) for d, x in v.items()}
                                               for u, v in pp.items()}
    return s


def _load(out, op, name):
    p = os.path.join(out, op, name)
    return read_json(p)["pairs"] if os.path.exists(p) else {}


def exploratory_summaries(runner, ops, pdatas, L):
    e = {}
    units = ["pre"] + [str(l) for l in range(L)] + ["post"]
    rem, cre, parts, patch, gain, ws, own, task_share = {}, {}, {}, [], [], [], [], {}
    for op in ops:
        bl, fac = _load(runner.out, op, "blocks.json"), _load(runner.out, op, "factorial.json")
        for pid, row in bl.items():
            G = row["Gamma"]
            for u in units:
                if row["removed"].get(u) is not None:
                    rem.setdefault(u, [0.0, 0.0]); rem[u][0] += row["removed"][u] * G; rem[u][1] += G
                    cre.setdefault(u, [0.0, 0.0]); cre[u][0] += row["created"][u] * G; cre[u][1] += G
            for k, x in row["parts"].items():
                parts.setdefault(k, [0.0, 0.0, 0.0]); parts[k][0] += x["removed_pp"]
                parts[k][1] += x["created_pp"]; parts[k][2] += G
        for pid, row in fac.items():
            for h in row["harmers"]:
                ph = row["task_phi"][h]
                tot = sum(ph.values())
                if abs(tot) > 1e-9:
                    for g in THIRDS:
                        task_share.setdefault(g, []).append(ph[g] / tot)
        for row in _load(runner.out, op, "patch.json").values():
            patch.append(row["restored_share_by_block"])
        for row in _load(runner.out, op, "crossterm.json").values():
            if "transport_gain" in row:
                gain.append(row["transport_gain"])
        for row in _load(runner.out, op, "weightspace.json").values():
            ws.append(row)
        for row in _load(runner.out, op, "own.json").values():
            own.append(row)
    e["block_removed_fraction"] = {u: v[0] / v[1] for u, v in rem.items() if abs(v[1]) > 1e-9}
    e["block_created_fraction"] = {u: v[0] / v[1] for u, v in cre.items() if abs(v[1]) > 1e-9}
    e["attn_mlp"] = {k: {"removed": v[0] / v[2], "created": v[1] / v[2]} for k, v in parts.items() if abs(v[2]) > 1e-9}
    e["harmer_skill_share_by_third"] = {g: float(np.mean(v)) for g, v in task_share.items()}

    def colmean(rows):
        if not rows:
            return None
        arr = np.array([[np.nan if x is None else x for x in r] for r in rows], dtype=float)
        return [None if np.all(np.isnan(c)) else float(np.nanmean(c)) for c in arr.T]

    e["patch_restored_share_by_block"] = colmean(patch)
    e["crossterm_transport_gain_by_block"] = colmean(gain)
    sp = [r.get("spearman_sti_vs_removed") for r in ws if r.get("spearman_sti_vs_removed") is not None]
    e["weightspace_mean_spearman_sti_vs_removed"] = float(np.mean(sp)) if sp else None
    e["weightspace_sti_peak_blocks"] = [r.get("sti_peak_block") for r in ws]
    e["weightspace_removed_peak_blocks"] = [r.get("removed_peak_block") for r in ws]
    e["own_share_by_third"] = {g: float(np.mean([o["share"][g] for o in own if o["share"][g] is not None]))
                               for g in THIRDS} if own else None
    rep = {}
    for op in ops:
        for row in _load(runner.out, op, "factorial.json").values():
            for g, x in row["repair"].items():
                rep.setdefault(g, {"gain": 0.0, "Gamma": 0.0, "hl": 0.0, "hf": 0.0})
                rep[g]["gain"] += x["victim_gain"]; rep[g]["Gamma"] += row["Gamma"]
                rep[g]["hl"] += x["harmer_loss"]; rep[g]["hf"] += x["harmer_full_loss_rep"]
    e["repair_by_removed_third"] = {g: {"rho_v": v["gain"] / v["Gamma"] if v["Gamma"] else None,
                                        "rho_h": v["hl"] / v["hf"] if v["hf"] else None} for g, v in rep.items()}
    return e


def who_family(runner, ops, pdatas):
    rows = []
    for op in ops:
        rep = _load(runner.out, op, "report.json")
        for pd_ in pdatas:
            if pd_.op == op and pd_.p["id"] in rep:
                rows.append((pd_, rep[pd_.p["id"]]))
    if not rows:
        return None

    def z(fn):
        by = {}
        for pd_, r in rows:
            by.setdefault(pd_.v, {}).setdefault(pd_.op, []).append((pd_, r))
        return {v: float(np.mean([sum(fn(r) for _, r in lst) / sum(pd_.gamma_sel for pd_, _ in lst)
                                  for lst in ops_.values()])) for v, ops_ in by.items()}

    w1 = sign_flip_test(list(z(lambda r: r["interaction"]).values()), "two-sided")
    w2 = sign_flip_test(list(z(lambda r: r["H"] - r["fake_all_harm"]).values()), "greater")
    hm = holm({"W1_interactions_exist": w1["p"], "W2_content_not_size": w2["p"]}, C.DEPTH["alpha_level"])
    from scipy.stats import spearmanr
    agree = []
    by_key = {}
    for pd_, r in rows:
        by_key.setdefault((pd_.v, tuple(pd_.p["context"])), {})[pd_.op] = pd_.p["shapley_sel"]
    for k, d in by_key.items():
        os_ = sorted(d)
        for i in range(len(os_)):
            for j in range(i + 1, len(os_)):
                N = sorted(d[os_[i]])
                rho = spearmanr([d[os_[i]][x] for x in N], [d[os_[j]][x] for x in N]).correlation
                if not np.isnan(rho):
                    agree.append(rho)
    breadth = {op: read_json(os.path.join(runner.out, op, "analyze.json"))["breadth"]
               for op in ops if os.path.exists(os.path.join(runner.out, op, "analyze.json"))}
    return {"W1_interactions_exist": dict(w1, holm=hm["W1_interactions_exist"]),
            "W2_content_not_size": dict(w2, holm=hm["W2_content_not_size"]),
            "interaction_share": ratio_of_sums([r["interaction"] for _, r in rows], [r["H"] for _, r in rows]),
            "fake_over_real": ratio_of_sums([r["fake_all_harm"] for _, r in rows], [r["H"] for _, r in rows]),
            "cross_operator_shapley_spearman_mean": float(np.mean(agree)) if agree else None,
            "cross_operator_n": len(agree), "breadth": breadth}


def _fmt(x, d=3):
    if x is None:
        return "–"
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return f"[{_fmt(x[0], d)}, {_fmt(x[1], d)}]"
    if isinstance(x, float):
        return f"{x:.{d}f}"
    return str(x)


def _write(out, res, exploratory, ops):
    tag = "_exploratory_" + "_".join(ops) if exploratory else ""
    write_json(os.path.join(out, f"depth_summary{tag}.json"), res)
    md = ["# Experiment 8: depth summary" + (" (EXPLORATORY run)" if exploratory else ""), ""]
    if not res.get("confirmatory"):
        md += [res.get("note", "nothing to report")]
    else:
        md += [f"Operators: {', '.join(res['ops'])}. Pairs: {res['n_pairs']}. Victims ({res['n_victims']}): "
               f"{', '.join(res['victims'])}. Smallest attainable p: {_fmt(res['min_attainable_p'], 4)}. "
               f"Feasible (>= {C.DEPTH['min_victims']} victims): {res['feasible']}.", "",
               "## Confirmatory claims (fixed sequence, alpha = 0.05; victims as units, exact sign-flip tests)", "",
               "| Order | Claim | Contrast | n | T (mean z_v) | 95% CI | p | Claim p (IUT) | Verdict |",
               "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for i, h in enumerate(C.HYPOTHESIS_ORDER, 1):
            r = res["confirmatory"][h]
            for j, c in enumerate(r["components"]):
                md.append(f"| {i if j == 0 else ''} | {h if j == 0 else ''} | {c['contrast']} | {c['n']} | "
                          f"{_fmt(c['T'])} | {_fmt(c.get('ci'))} | {_fmt(c['p'], 4)} | "
                          f"{_fmt(r['p'], 4) if j == 0 else ''} | {r['verdict'] if j == 0 else ''} |")
        est, iv = res["estimates"], res["intervals"]
        md += ["", "## Effect sizes (pooled, ratios of sums; two-stage cluster-bootstrap 95% intervals)", "",
               "| Quantity | Estimate | 95% CI |", "| --- | --- | --- |"]
        for g in THIRDS:
            md.append(f"| depth share sigma_{g} | {_fmt(est['sigma'][g])} | {_fmt(iv['sigma'][g])} |")
        md += [f"| amplification share kappa_M / Gamma | {_fmt(est['kappa_share'])} | {_fmt(iv['kappa_share'])} |",
               f"| victim recovery rho_v (mid removed) | {_fmt(est['rho_v_mid'])} | {_fmt(iv['rho_v_mid'])} |",
               f"| harmer cost rho_h (mid removed) | {_fmt(est['rho_h_mid'])} | {_fmt(iv['rho_h_mid'])} |",
               f"| information-loss fraction psi | {_fmt(est.get('psi'))} | {_fmt(iv['psi'])} |"]
        md += ["", "## Per operator (estimates with intervals; not tested)", "",
               "| Operator | pairs | sigma_E | sigma_M | sigma_Lt |", "| --- | --- | --- | --- | --- |"]
        for op, r in res["per_operator"].items():
            s, ci = r["estimates"]["sigma"], r["sigma_ci"]
            md.append(f"| {op} | {r['estimates']['n_pairs']} | " + " | ".join(
                f"{_fmt(s[g])} {_fmt(ci[g])}" for g in THIRDS) + " |")
        sv = res["sensitivity"]
        md += ["", "## Sensitivity analyses (pre-declared; do not change verdicts)", "",
               f"- k = 1 (top context per victim): claim p-values {sv.get('k1')}",
               f"- mean of per-pair ratios, sigma: {sv.get('sigma_mean_of_ratios')}",
               f"- specificity with one random seed: {sv.get('specificity_one_seed')}",
               f"- pre/post single-unit mean fractions: {sv.get('pre_post_single_unit_mean_fraction')}",
               "- leave one victim out (T, min, max, sign change):"]
        for k, v in sv["leave_one_victim_out"].items():
            md.append(f"    - {k}: {_fmt(v['T'])}, {_fmt(v['min'])}, {_fmt(v['max'])}, {v['sign_changes']}")
        ex = res["exploratory"]
        md += ["", "## Exploratory (no p-values)", "",
               f"- harmers' own-skill share by third: {ex.get('harmer_skill_share_by_third')}",
               f"- victim's own-update share by third: {ex.get('own_share_by_third')}",
               f"- repair by removed third (rho_v, rho_h): {ex.get('repair_by_removed_third')}",
               f"- attention vs MLP (fraction of Gamma): {ex.get('attn_mlp')}",
               f"- block removed fraction: {ex.get('block_removed_fraction')}",
               f"- block created fraction: {ex.get('block_created_fraction')}",
               f"- activation patching restored share by block: {ex.get('patch_restored_share_by_block')}",
               f"- cross-term transport gain by block: {ex.get('crossterm_transport_gain_by_block')}",
               f"- weight space: mean Spearman(STI, removed) {_fmt(ex.get('weightspace_mean_spearman_sti_vs_removed'))}; "
               f"STI peaks {ex.get('weightspace_sti_peak_blocks')}; removal peaks {ex.get('weightspace_removed_peak_blocks')}"]
        if res.get("who"):
            w = res["who"]
            md += ["", "## Who is responsible (secondary family, Holm)", "",
                   f"- W1 interactions exist (two-sided): n={w['W1_interactions_exist']['n']}, "
                   f"T={_fmt(w['W1_interactions_exist']['T'])}, p={_fmt(w['W1_interactions_exist']['p'], 4)}, "
                   f"Holm {w['W1_interactions_exist']['holm']}",
                   f"- W2 harm exceeds fake neighbours: n={w['W2_content_not_size']['n']}, "
                   f"T={_fmt(w['W2_content_not_size']['T'])}, p={_fmt(w['W2_content_not_size']['p'], 4)}, "
                   f"Holm {w['W2_content_not_size']['holm']}",
                   f"- interaction share (ratio of sums): {_fmt(w['interaction_share'])}; fake/real harm: "
                   f"{_fmt(w['fake_over_real'])}; cross-operator Shapley Spearman: "
                   f"{_fmt(w['cross_operator_shapley_spearman_mean'])} (n={w['cross_operator_n']})"]
    with open(os.path.join(out, f"depth_summary{tag}.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    if res.get("who"):
        w = res["who"]
        write_json(os.path.join(out, f"who_summary{tag}.json"), w)
        wm = ["# Experiment 8: who is responsible" + (" (EXPLORATORY run)" if exploratory else ""), "",
              "Secondary family (Holm, alpha = 0.05); victims as units; report half.", "",
              "| Claim | n | T (mean z_v) | p | Holm-adjusted p | Rejected |", "| --- | --- | --- | --- | --- | --- |"]
        for k in ("W1_interactions_exist", "W2_content_not_size"):
            x = w[k]
            wm.append(f"| {k} ({x.get('alternative')}) | {x['n']} | {_fmt(x['T'])} | {_fmt(x['p'], 4)} | "
                      f"{_fmt(x['holm'][0], 4)} | {x['holm'][1]} |")
        wm += ["", f"- Interaction share of harm (ratio of sums): {_fmt(w['interaction_share'])}",
               f"- Fake-neighbour harm / real harm: {_fmt(w['fake_over_real'])}",
               f"- Cross-operator Spearman of selection-half Shapley values: "
               f"{_fmt(w['cross_operator_shapley_spearman_mean'])} (n = {w['cross_operator_n']})", "",
               "## Breadth (selection half, descriptive)", ""]
        for op, b in w["breadth"].items():
            wm.append(f"- {op}: mean of context means {_fmt(b['mean_of_mean'], 2)}, mean of worst "
                      f"{_fmt(b['mean_of_worst'], 2)}, harm quantiles {b['harm_quantiles']}")
        with open(os.path.join(out, f"who_summary{tag}.md"), "w") as f:
            f.write("\n".join(wm) + "\n")


def replicate_summary(runner, op, source_out, B):
    """sigma per third on the replication backbone, next to the source backbone's values."""
    L = runner.n_layers_from_disk([op])
    pd_rep = [PairData(runner, op, dict(p, _index=i), L, C.DEPTH["random_seeds"])
              for i, p in enumerate(runner.pairs(op))]
    res = {"op": op, "n_pairs": len(pd_rep), "victims": sorted({p.v for p in pd_rep}, key=C.TASKS.index)}
    if pd_rep:
        it = [(p, p.stats()) for p in pd_rep]
        dr = cluster_bootstrap(pd_rep, B, seed=2)
        res["estimates"] = pooled(it)
        res["sigma_ci"] = {g: ci_of(dr, lambda d, g=g: d["pooled"]["sigma"][g]) for g in THIRDS}
        res["excess_share_ci"] = {g: ci_of(dr, lambda d, g=g: d["pooled"]["excess_share"][g]) for g in THIRDS}
    src = os.path.join(source_out, "depth_summary.json")
    if os.path.exists(src):
        s = read_json(src)
        res["source_sigma"] = s.get("per_operator", {}).get(op, {}).get("estimates", {}).get("sigma")
        res["source_sigma_ci"] = s.get("per_operator", {}).get(op, {}).get("sigma_ci")
    write_json(os.path.join(runner.out, "replicate_summary.json"), res)
    md = [f"# Experiment 8: replication ({runner.backbone}, {op}; exploratory, no tests)", "",
          f"Pairs: {res['n_pairs']}; victims: {', '.join(res['victims'])}.", "",
          "| Third | sigma (this backbone) | 95% CI | sigma (source) | 95% CI (source) |",
          "| --- | --- | --- | --- | --- |"]
    for g in THIRDS:
        md.append(f"| {g} | {_fmt(res.get('estimates', {}).get('sigma', {}).get(g))} | "
                  f"{_fmt(res.get('sigma_ci', {}).get(g))} | {_fmt((res.get('source_sigma') or {}).get(g))} | "
                  f"{_fmt((res.get('source_sigma_ci') or {}).get(g))} |")
    with open(os.path.join(runner.out, "replicate_summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    return res
