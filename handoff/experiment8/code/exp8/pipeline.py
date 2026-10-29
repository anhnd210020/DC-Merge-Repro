"""Experiment 8 runner: shared machinery and Stage A (selection half).

Every evaluation is cached on disk as per-image correctness; rerunning a step resumes it.
Stage B (report half) lives in exp8/depth.py, the analysis in exp8/depthstats.py.
"""
import os
import random
import time
from collections import OrderedDict

import numpy as np
import torch

from . import config as C
from .harm import bootstrap_decomposition, decompose, norm_acc, subsets
from .models import Assets, Evaluator, set_determinism
from .operators import SVDCache, fake_task_vector, hybrid, hybrid_sel, is_matrix, merge
from .store import EvalStore, read_json, write_json


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def pair_id(op, victim, context):
    return f"{op}|{victim}|{'-'.join(context)}"


class Runner:
    def __init__(self, args):
        self.args = args
        self.tasks = list(args.tasks) if args.tasks else list(C.TASKS)
        self.out = args.out
        self.backbone = args.backbone
        bb = C.BACKBONES.get(args.backbone, {})
        self.base_id = args.base_id or bb.get("base")
        self.ft_pattern = args.ft_pattern or bb.get("finetuned")
        self.device = args.device
        if getattr(args, "screen_cap", None):
            C.SCREEN_CAP = int(args.screen_cap)
        os.makedirs(self.out, exist_ok=True)
        self.shared = EvalStore(os.path.join(self.out, "_shared"))
        self._assets = None
        self._ev = None
        self._svd = None
        self._merge_cache = OrderedDict()
        self._fake_cache = OrderedDict()
        set_determinism()

    # ------------------------------------------------------------------ assets
    @property
    def assets(self):
        if self._assets is None:
            log(f"loading assets: base={self.base_id} ft={self.ft_pattern}")
            self._assets = Assets(self.base_id, self.ft_pattern, self.tasks, self.device,
                                  os.path.join(self.out, "_shared"),
                                  tau_device=getattr(self.args, "tau_device", None))
        return self._assets

    @property
    def ev(self):
        if self._ev is None:
            self._ev = Evaluator(self.assets, self.args.data_dir, self.args.batch_size,
                                 self.args.gpu_cache_gb)
        return self._ev

    @property
    def svd(self):
        if self._svd is None:
            self._svd = SVDCache(self.assets.tau)
        return self._svd

    @property
    def n_layers(self):
        return self.assets.n_layers

    def n_layers_from_disk(self, ops=()):
        """Number of blocks, read from a factorial.json if one exists (avoids loading models)."""
        for op in list(ops) + list(C.OPERATORS):
            f = os.path.join(self.out, op, "factorial.json")
            if os.path.exists(f):
                L = read_json(f).get("n_layers")
                if L:
                    return int(L)
        return self.n_layers

    def op_dir(self, op):
        d = os.path.join(self.out, op)
        os.makedirs(d, exist_ok=True)
        return d

    def store(self, op):
        return EvalStore(self.op_dir(op))

    def alpha(self, op):
        p = os.path.join(self.op_dir(op), "alpha.json")
        if self.args.alpha is not None:
            return float(self.args.alpha)
        if not os.path.exists(p):
            raise SystemExit(f"[{op}] run the `alpha` step first (or pass --alpha)")
        return float(read_json(p)["alpha"])

    def pairs(self, op):
        p = os.path.join(self.op_dir(op), "selection.json")
        if not os.path.exists(p):
            raise SystemExit(f"[{op}] run `select` first")
        return read_json(p)["pairs"]

    def taskdata(self, task):
        from .data import TaskData
        return TaskData(self.args.data_dir, task)

    # ------------------------------------------------------------------ evaluations
    def _base_key(self, **kw):
        k = {"backbone": self.backbone, "base": self.base_id, "ft": self.ft_pattern}
        k.update(kw)
        if kw.get("split") == "screen":
            k["screen_cap"] = C.SCREEN_CAP
        return k

    def expert_correct(self, task, split):
        key = self._base_key(kind="expert", task=task, split=split)
        c = self.shared.get(key)
        if c is None:
            self.ev.load(self.assets.expert_params(task), tag=("expert", task))
            c = self.ev.correct(task, split)
            self.shared.put(key, c)
        return c

    def zeroshot_correct(self, task, split):
        key = self._base_key(kind="zeroshot", task=task, split=split)
        c = self.shared.get(key)
        if c is None:
            self.ev.load(self.assets.theta0, tag=("zeroshot",))
            c = self.ev.correct(task, split)
            self.shared.put(key, c)
        return c

    def fake(self, task):
        if task not in self._fake_cache:
            seed = C.SEED * 1000 + self.tasks.index(task) + 17
            self._fake_cache[task] = fake_task_vector(self.assets.tau[task], self.assets.names, seed,
                                                      svd_cache=self.svd, task_id=task)
            while len(self._fake_cache) > 3:
                self._fake_cache.popitem(last=False)
        return self._fake_cache[task]

    def merged_params(self, op, subset, alpha, fakes=()):
        subset = C.canon(subset)
        key = (op, subset, alpha, tuple(sorted(fakes)))
        if key in self._merge_cache:
            self._merge_cache.move_to_end(key)
            return self._merge_cache[key]
        tvs = [self.fake(t) if t in fakes else self.assets.tau[t] for t in subset]
        ids = [None if t in fakes else t for t in subset]
        use_cache = op == "tsvm" and not fakes
        delta = merge(op, tvs, self.assets.names, task_ids=ids if use_cache else None,
                      svd_cache=self.svd if use_cache else None)
        params = self.assets.params_with_delta(delta, alpha)
        del delta
        self._merge_cache[key] = params
        while len(self._merge_cache) > self.args.merge_cache:
            self._merge_cache.popitem(last=False)
        return params

    def merge_correct(self, op, subset, alpha, tasks, split, fakes=()):
        """Correctness arrays of the merge of `subset` on each of `tasks`."""
        st = self.store(op)
        subset = C.canon(subset)
        out, todo = {}, []
        for t in tasks:
            key = self._base_key(kind="merge", op=op, alpha=alpha, subset=list(subset),
                                 fakes=sorted(fakes), task=t, split=split)
            c = st.get(key)
            if c is None:
                todo.append((t, key))
            else:
                out[t] = c
        if todo:
            p = self.merged_params(op, subset, alpha, fakes)
            self.ev.load(p, tag=("merge", op, subset, alpha, tuple(sorted(fakes))))
            for t, key in todo:
                c = self.ev.correct(t, split)
                st.put(key, c)
                out[t] = c
        return out

    def cond_key(self, op, cond, task, split):
        return self._base_key(kind="cond", op=op, cond=cond, task=task, split=split)

    def cond_correct(self, op, cond, tasks, split, params_fn):
        """Cached evaluation of an arbitrary condition (hybrid, perturbation, ...).

        cond: JSON-able dict that identifies the condition; params_fn builds its parameters
        only if some task still needs evaluating.
        """
        st = self.store(op)
        out, todo = {}, []
        for t in tasks:
            key = self.cond_key(op, cond, t, split)
            c = st.get(key)
            if c is None:
                todo.append((t, key))
            else:
                out[t] = c
        if todo:
            p = params_fn()
            self.ev.load(p, tag=None)
            for t, key in todo:
                c = self.ev.correct(t, split)
                st.put(key, c)
                out[t] = c
        return out

    def stored_cond(self, op, cond, task, split):
        c = self.store(op).get(self.cond_key(op, cond, task, split))
        if c is None:
            raise SystemExit(f"missing evaluation {cond} on {task}/{split}; run the step that makes it")
        return c

    def nacc(self, correct, task, split):
        return norm_acc(correct, self.expert_correct(task, split))

    # ------------------------------------------------------------------ Stage A
    def step_check(self):
        """Must-hold checks: assets, determinism, operator identities, block runner, throughput."""
        res = {"assets": {}, "pass": True, "problems": []}
        ref_e = C.REF_EXPERT.get(self.backbone, {})
        ref_z = C.REF_ZEROSHOT.get(self.backbone, {})
        for t in self.tasks:
            e = 100 * self.expert_correct(t, "all").mean()
            z = 100 * self.zeroshot_correct(t, "all").mean()
            res["assets"][t] = {"expert": e, "zeroshot": z, "ref_expert": ref_e.get(t),
                                "ref_zeroshot": ref_z.get(t)}
            for name, val, ref in (("expert", e, ref_e.get(t)), ("zeroshot", z, ref_z.get(t))):
                if ref is not None and abs(val - ref) > self.args.ref_tol:
                    res["pass"] = False
                    res["problems"].append(f"{t} {name} {val:.2f} vs reference {ref:.2f}")
            log(f"check {t}: expert {e:.2f} (ref {ref_e.get(t)}), zero-shot {z:.2f} (ref {ref_z.get(t)})")
        t0 = self.tasks[0]
        # determinism + throughput
        self.ev.load(self.assets.expert_params(t0), tag=None)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        tic = time.time()
        a = self.ev.correct(t0, "screen")
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        res["throughput_images_per_s"] = float(len(a) / max(time.time() - tic, 1e-9))
        self.ev.load(self.assets.expert_params(t0), tag=None)
        b = self.ev.correct(t0, "screen")
        res["determinism_identical"] = bool((a == b).all())
        if not res["determinism_identical"]:
            res["pass"] = False
            res["problems"].append("two identical evaluations differ")
        # block runner reproduces the hidden states
        hs, _ = self.ev.hidden_all(t0, "screen", 0, min(16, len(a)))
        h, worst = hs[0], 0.0
        for l in range(self.n_layers):
            h = self.ev.run_block(l, h)
            worst = max(worst, float((h - hs[l + 1]).norm() / hs[l + 1].norm().clamp_min(1e-12)))
        res["block_runner_rel_error"] = worst
        if worst > 1e-5:
            res["pass"] = False
            res["problems"].append(f"block runner differs from the model (rel {worst:.2e})")
        # operator identities
        names, A = self.assets.names, self.assets
        tau0 = {n: A.tau[t0][n].to(self.device) for n in names}
        ta1 = merge("ta", [tau0], names)
        res["ta_single_is_expert_maxabs"] = max(float((ta1[n] - tau0[n]).abs().max()) for n in names)
        ok_ops = res["ta_single_is_expert_maxabs"] == 0.0
        if self.args.check_svd_ops:
            ts1 = merge("tsvm", [tau0], names)
            rel = max(float((ts1[n] - tau0[n]).norm() / tau0[n].norm().clamp_min(1e-12)) for n in names)
            res["tsvm_single_reconstructs_rel"] = rel
            tau1 = {n: A.tau[self.tasks[1]][n].to(self.device) for n in names}
            iso = merge("iso_cts", [tau0, tau1], names)
            spread = 0.0
            for n in names:
                if is_matrix(n, iso[n]):
                    s = torch.linalg.svdvals(iso[n])
                    spread = max(spread, float((s.max() - s.min()) / s.mean()))
            res["iso_cts_isotropic_spread"] = spread
            nonmat = [n for n in names if not is_matrix(n, tau0[n])]
            res["iso_cts_nonmatrix_is_mean_maxabs"] = max(
                float((iso[n] - (tau0[n] + tau1[n]) / 2).abs().max()) for n in nonmat)
            ok_ops = ok_ops and rel < 1e-3 and spread < 1e-2 and res["iso_cts_nonmatrix_is_mean_maxabs"] < 1e-6
        pa, pb = A.params_with_delta(ta1, 1.0), A.theta0
        h_all = hybrid_sel(pa, pb, A.units, A.kinds, lambda u, k: True)
        h_none = hybrid_sel(pa, pb, A.units, A.kinds, lambda u, k: False)
        res["hybrid_identities"] = all(h_all[n] is pa[n] and h_none[n] is pb[n] for n in names)
        if not (ok_ops and res["hybrid_identities"]):
            res["pass"] = False
            res["problems"].append("operator or hybrid identity failed")
        write_json(os.path.join(self.out, "_shared", "check.json"), res)
        log(f"check: {'PASS' if res['pass'] else 'FAIL'} {res['problems']} "
            f"(throughput {res['throughput_images_per_s']:.0f} images/s)")
        if not res["pass"] and not self.args.force:
            raise SystemExit("must-hold check failed; see _shared/check.json")
        return res

    def step_alpha(self, op, n_contexts=None):
        ctxs = C.all_contexts(self.tasks)
        rng = random.Random(C.SEED)
        n_contexts = n_contexts or self.args.alpha_contexts or C.N_ALPHA_CONTEXTS
        sample = rng.sample(ctxs, min(n_contexts, len(ctxs)))
        grid = self.args.alpha_grid or C.ALPHA_GRID[op]
        table = {}
        for a in grid:
            scores = []
            for S in sample:
                cs = self.merge_correct(op, S, a, S, "screen")
                scores.append(np.mean([self.nacc(cs[t], t, "screen") for t in S]))
            table[str(a)] = float(np.mean(scores))
            log(f"[{op}] alpha {a}: mean normalized {table[str(a)]:.2f}")
        best = max(grid, key=lambda a: (table[str(a)], -a))
        write_json(os.path.join(self.op_dir(op), "alpha.json"),
                   {"alpha": best, "grid": table, "contexts": [list(s) for s in sample], "split": "screen"})
        log(f"[{op}] selected alpha = {best}")

    def step_subsets(self, op):
        a = self.alpha(op)
        subs = C.all_subsets_up_to(self.tasks)
        t0 = time.time()
        for i, S in enumerate(subs):
            self.merge_correct(op, S, a, S, "screen")
            if (i + 1) % 10 == 0 or i + 1 == len(subs):
                log(f"[{op}] subsets {i + 1}/{len(subs)} ({time.time() - t0:.0f}s)")

    def _f_table(self, op, alpha, victim, S, split):
        N = [t for t in S if t != victim]
        f, corr = {}, {}
        for T in subsets(N):
            sub = C.canon((victim,) + tuple(T))
            c = self.merge_correct(op, sub, alpha, [victim], split)[victim]
            corr[T] = c
            f[T] = self.nacc(c, victim, split)
        return N, f, corr

    def step_analyze(self, op):
        """Neighbour decomposition of every (victim, context) pair and descriptive breadth."""
        a = self.alpha(op)
        pairs, contexts = [], []
        for S in C.all_contexts(self.tasks):
            accs = {}
            for v in S:
                N, f, _ = self._f_table(op, a, v, S, "screen")
                r = decompose(f, N)
                assert abs(r["check_sum"]) < 1e-6, "decomposition is not exact"
                accs[v] = f[frozenset(N)]
                pairs.append({"victim": v, "context": list(S), "neighbours": N,
                              "alone": f[frozenset()], "in_context": f[frozenset(N)],
                              "H": r["H"], "interaction": r["interaction"],
                              "interaction_share": r["interaction_share"],
                              "sum_singles": r["sum_singles"], "singles": r["singles"],
                              "by_order": r["by_order"], "shapley": r["shapley"]})
            contexts.append({"context": list(S), "mean": float(np.mean(list(accs.values()))),
                             "worst": float(min(accs.values()))})
        Hs = [p["H"] for p in pairs]
        breadth = {"n_contexts": len(contexts), "n_pairs": len(pairs),
                   "mean_of_mean": float(np.mean([c["mean"] for c in contexts])),
                   "mean_of_worst": float(np.mean([c["worst"] for c in contexts])),
                   "harm_quantiles": {str(q): float(np.percentile(Hs, q)) for q in (10, 25, 50, 75, 90, 99)}}
        write_json(os.path.join(self.op_dir(op), "analyze.json"),
                   {"alpha": a, "split": "screen", "breadth": breadth, "contexts": contexts, "pairs": pairs})
        log(f"[{op}] analyze: {len(pairs)} pairs; median context harm {breadth['harm_quantiles']['50']:.2f} pp")

    def step_select(self, op):
        """Victim-stratified selection on the selection half; frozen for Stage B."""
        a = self.alpha(op)
        an = read_json(os.path.join(self.op_dir(op), "analyze.json"))
        k = self.args.k_per_victim or C.DEPTH["k_per_victim"]
        cand = {}
        for p in an["pairs"]:
            v, S = p["victim"], tuple(p["context"])
            Nh = [j for j in p["neighbours"] if p["shapley"][j] > 0]
            if not Nh:
                continue
            sub_b = tuple(t for t in S if t not in Nh)
            cA = self.merge_correct(op, S, a, [v], "screen")[v]
            cB = self.merge_correct(op, sub_b, a, [v], "screen")[v]
            e = self.expert_correct(v, "screen")
            g = self.nacc(cB, v, "screen") - self.nacc(cA, v, "screen")
            rng = np.random.RandomState(C.SEED)
            cnt = rng.multinomial(len(e), np.full(len(e), 1.0 / len(e)), size=C.BOOTSTRAP).astype(np.float64)
            den = cnt @ e.astype(np.float64)
            gb = 100.0 * (cnt @ (cB.astype(np.float64) - cA.astype(np.float64))) / np.maximum(den, 1e-12)
            lo, hi = float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5))
            c = {"victim": v, "context": list(S), "harmers": Nh, "gamma_sel": float(g),
                 "gamma_sel_ci": [lo, hi], "H_sel": p["H"], "shapley_sel": p["shapley"]}
            if lo > 0:
                c["harmer_loss_sel"] = self._harmer_loss(op, a, S, Nh)
            c["eligible"] = bool(lo > 0 and sum(c["harmer_loss_sel"].values()) > 0)
            cand.setdefault(v, []).append(c)
        pairs, ineligible = [], []
        for v in self.tasks:
            cs = sorted(cand.get(v, []), key=lambda c: (-c["gamma_sel"], C.TASKS.index(v), c["context"]))
            el = [c for c in cs if c["eligible"]]
            if not el:
                best = cs[0] if cs else None
                ineligible.append({"victim": v, "best_gamma_sel": best["gamma_sel"] if best else None,
                                   "best_ci": best["gamma_sel_ci"] if best else None,
                                   "best_context": best["context"] if best else None})
                continue
            for rank, c in enumerate(el[:k], start=1):
                c = dict(c, rank=rank, op=op, alpha=a, id=pair_id(op, v, c["context"]))
                neg = [h for h, x in c["harmer_loss_sel"].items() if x <= 0]
                if neg:
                    log(f"[{op}] note {c['id']}: harmers {neg} do not lose when removed (sum is positive)")
                pairs.append(c)
        victims = sorted({p["victim"] for p in pairs}, key=C.TASKS.index)
        n_el = sum(c["eligible"] for cs in cand.values() for c in cs)
        res = {"op": op, "alpha": a, "k_per_victim": k, "pairs": pairs, "victims": victims,
               "n_victims": len(victims), "ineligible_victims": ineligible,
               "n_candidates": sum(len(v) for v in cand.values()), "n_eligible_candidates": int(n_el),
               "config_sha256": C.config_sha256(), "split": "screen"}
        write_json(os.path.join(self.op_dir(op), "selection.json"), res)
        log(f"[{op}] select: {len(pairs)} pairs over {len(victims)} victims ({n_el} eligible candidates); "
            f"not eligible: {[x['victim'] for x in ineligible]}")

    def _harmer_loss(self, op, a, S, Nh):
        """Harmers' own normalized-accuracy loss when all harmers are removed (selection half)."""
        sub_b = tuple(t for t in S if t not in Nh)
        fA = self.merge_correct(op, S, a, Nh, "screen")
        fB = self.merge_correct(op, sub_b, a, Nh, "screen")
        return {h: float(self.nacc(fA[h], h, "screen") - self.nacc(fB[h], h, "screen")) for h in Nh}

    def step_select_all(self):
        ops = [o for o in C.OPERATORS if os.path.exists(os.path.join(self.out, o, "selection.json"))]
        sels = {o: read_json(os.path.join(self.out, o, "selection.json")) for o in ops}
        hashes = {s["config_sha256"] for s in sels.values()}
        if len(hashes) != 1 or C.config_sha256() not in hashes:
            raise SystemExit("config.py changed between operators' `select` runs; rerun select")
        pairs = [p for o in ops for p in sels[o]["pairs"]]
        victims = sorted({p["victim"] for p in pairs}, key=C.TASKS.index)
        feasible = len(victims) >= C.DEPTH["min_victims"]
        res = {"ops": ops, "pairs": pairs, "victims": victims, "n_victims": len(victims),
               "feasible": feasible, "min_attainable_p": 2.0 ** (-len(victims)) if victims else None,
               "config_sha256": C.config_sha256()}
        write_json(os.path.join(self.out, "selection_all.json"), res)
        log(f"select --all-ops: {len(pairs)} pairs, {len(victims)} victims, feasible={feasible} "
            f"(smallest attainable p {res['min_attainable_p']})")

    def step_report(self, op):
        """'Who' decomposition of the selected pairs on the report half, with fake controls."""
        a = self.alpha(op)
        path = os.path.join(self.op_dir(op), "report.json")
        out = read_json(path)["pairs"] if os.path.exists(path) else {}
        for i, s in enumerate(self.pairs(op)):
            if s["id"] in out:
                continue
            v, S = s["victim"], s["context"]
            N, f, corr = self._f_table(op, a, v, S, "rep")
            r = decompose(f, N)
            boot = bootstrap_decomposition(corr, self.expert_correct(v, "rep"), N, B=C.BOOTSTRAP)
            all_fake = self.merge_correct(op, (v,) + tuple(N), a, [v], "rep", fakes=tuple(N))[v]
            f0 = f[frozenset()]
            f_all_fake = self.nacc(all_fake, v, "rep")
            culprit = max(N, key=lambda j: s["shapley_sel"][j])
            one_fake = self.merge_correct(op, (v, culprit), a, [v], "rep", fakes=(culprit,))[v]
            H = r["H"]
            out[s["id"]] = dict(id=s["id"], victim=v, context=S, neighbours=N, harmers=s["harmers"],
                                H=H, alone=f0, in_context=f[frozenset(N)], **boot,
                                interaction=r["interaction"], interaction_share=r["interaction_share"],
                                sum_singles=r["sum_singles"], singles=r["singles"], shapley=r["shapley"],
                                by_order=r["by_order"], dividends=r["dividends"],
                                fake_all_harm=f0 - f_all_fake, culprit=culprit,
                                culprit_single=r["singles"][culprit],
                                fake_culprit_single=f0 - self.nacc(one_fake, v, "rep"))
            write_json(path, {"alpha": a, "split": "rep", "pairs": out})
            log(f"[{op}] report {i + 1}: {s['id']} H={H:.1f} interaction={r['interaction']:.1f} "
                f"fake_all={out[s['id']]['fake_all_harm']:.1f}")
        write_json(path, {"alpha": a, "split": "rep", "pairs": out})
