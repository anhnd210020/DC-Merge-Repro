#!/usr/bin/env python3
"""Experiment 6a: can agreement with each task's fine-tuned expert choose lane coefficients?

Same merged model and search space as Experiments 3 and 5,

    W(c) = W0 + sum_i c_i L_i      (dc_rank lanes from lanes.py),

but the label-free objective compares the merged model M(c) with task i's expert
E_i = base CLIP + task i's own full LoRA update (B @ A, LoRA scale 16/16 = 1), on the
SAME 500 unlabeled validation images per task (seed 0) that Experiment 5 used:

    kl    : mean_x KL( softmax(E_i logits) || softmax(M(c) logits) )        lower = better
    top1  : 1 - mean_x [argmax E_i == argmax M(c)]                          lower = better
    cos   : 1 - mean_x cos( emb_E_i(x), emb_M(c)(x) )                       lower = better

Logits are tau * <f/|f|, h_k/|h_k|> with tau = 100 (as in Experiment 5); the embedding
is the model output after the visual projection, before the head.  Every proxy is
stored as a per-task LOSS, so "lower is better" for all three.

The proxy needs unlabeled images AND the experts, so it is NOT data-free (same regime
as AdaMerging / AdaRank).

Subcommands
  check      gate: proxies on Experiment 3's cached vectors vs their cached validation
             accuracy (no new labeled evaluation).  Writes agreement_check_summary.md,
             agreement_check.csv, agreement_check.json (with the pre-registered verdict).
  run        only if the gate allows it: coordinate search with the gate's proxy, both
             objectives, each chosen vector evaluated ONCE on test.
  summarize  rebuild agreement_summary.md / agreement.csv from finished contexts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_official as ro  # noqa: E402
import run_entropy as rent  # noqa: E402  (same unlabeled subset, same spearman, same grid)

GRID = rent.GRID
CORE = rent.CORE
FOUR = rent.FOUR
TAU = rent.TAU
PROXIES = ("kl", "top1", "cos")
PROXY_LABEL = {"kl": "KL(expert || merged)", "top1": "1 - top-1 agreement", "cos": "1 - feature cosine"}
OBJECTIVES = ("agreement_mean", "agreement_fair")

# pre-registered gate thresholds (core contexts, validation)
RHO_PASS = 0.5
RHO_FAIL = 0.3
# expert sanity check: expert full-validation accuracy vs single_task/our_finetuned/val_acc.json
EXPERT_WARN_PP = 0.5
EXPERT_ABORT_PP = 1.0

write_json = rent.write_json
key_of = rent.key_of
spearman = rent.spearman


def contexts_of(spec):
    return rent.contexts_of(spec)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_note(stage, text):
    """One log line per number source, as the guide requires."""
    print(f"[split] {stage}: {text}", flush=True)


# ----------------------------------------------------------------------------
# forward passes: embeddings and logits on the unlabeled subsets
# ----------------------------------------------------------------------------
def make_forward(rt, tasks, loaders, device):
    """Return fn(merged_update) -> {task: (emb_unit [n,d] float32 cpu, logits [n,K] float32 cpu)}."""
    torch = rt["torch"]
    heads = {t: rt["heads"][t].float() for t in tasks}
    heads = {t: (h / h.norm(dim=-1, keepdim=True)).to(device) for t, h in heads.items()}

    def forward(merged):
        with torch.no_grad():
            model = rt["apply_merge"](rt["new"], merged, scaling_coeffs=1.0).to(device)
        model.eval()
        out = {}
        try:
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                for t in tasks:
                    embs, logits = [], []
                    for batch in loaders[t]:
                        x = batch[0] if isinstance(batch, (list, tuple)) else batch   # labels ignored
                        f = model(x.to(device)).float()
                        f = f / f.norm(dim=-1, keepdim=True)
                        embs.append(f.cpu())
                        logits.append((TAU * f @ heads[t].T).float().cpu())
                    out[t] = (torch.cat(embs), torch.cat(logits))
        finally:
            model.cpu()
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        return out
    return forward


def proxy_values(torch, merged_out, expert_out):
    """Per-task raw proxies: kl (>=0), top1_agree in [0,1], feat_cos in [-1,1]."""
    emb_m, log_m = merged_out
    emb_e, log_e = expert_out
    if emb_m.shape != emb_e.shape or log_m.shape != log_e.shape:
        raise RuntimeError("merged and expert outputs have different shapes; unlabeled subset mismatch")
    lp_m = torch.log_softmax(log_m.double(), dim=-1)
    lp_e = torch.log_softmax(log_e.double(), dim=-1)
    kl = (lp_e.exp() * (lp_e - lp_m)).sum(-1).mean()
    top1 = (log_m.argmax(-1) == log_e.argmax(-1)).double().mean()
    cos = (emb_m.double() * emb_e.double()).sum(-1).mean()
    return {"kl": float(kl), "top1_agree": float(top1), "feat_cos": float(cos)}


def task_loss(rec, proxy):
    """Convert a task's raw proxies to a loss where lower is better."""
    if proxy == "kl":
        return rec["kl"]
    if proxy == "top1":
        return 1.0 - rec["top1_agree"]
    if proxy == "cos":
        return 1.0 - rec["feat_cos"]
    raise ValueError(proxy)


def losses(P, tasks, proxy):
    vals = [task_loss(P[t], proxy) for t in tasks]
    return sum(vals) / len(vals), max(vals)


# ----------------------------------------------------------------------------
# experts (do not depend on c): computed once per task, cached on disk
# ----------------------------------------------------------------------------
class Experts:
    def __init__(self, rt, args, loaders, info):
        self.rt, self.args, self.loaders, self.info = rt, args, loaders, info
        self.dir = args.out / "experts"
        self.mem = {}

    def path(self, t):
        return self.dir / f"{t}_n{self.args.n_unlabeled}_s{self.args.seed}.pt"

    def get(self, t):
        torch = self.rt["torch"]
        if t in self.mem:
            return self.mem[t]
        p = self.path(t)
        subset = self.info[t]
        if p.exists():
            d = torch.load(p, map_location="cpu")
            if d["first_indices"] != subset["first_indices"] or d["n"] != subset["n"] or d["tau"] != TAU:
                raise RuntimeError(f"cached expert outputs {p} were built on a different subset/tau; delete it")
        else:
            fwd = make_forward(self.rt, [t], {t: self.loaders[t]}, self.args.device)
            emb, logits = fwd(self.rt["deltas"][t])[t]          # base + task t's own full LoRA update
            d = {"emb": emb, "logits": logits, "n": subset["n"], "first_indices": subset["first_indices"],
                 "tau": TAU, "task": t}
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            torch.save(d, tmp)
            tmp.replace(p)
            print(f"  expert {t}: cached {tuple(emb.shape)} embeddings and {tuple(logits.shape)} logits", flush=True)
        self.mem[t] = (d["emb"], d["logits"])
        return self.mem[t]


def check_experts(rt, args, tasks):
    """Expert full-validation accuracy must match the single-task reference (sanity only)."""
    path = args.out / "experts" / "expert_val_check.json"
    rec = json.loads(path.read_text()) if path.exists() else {}
    for t in tasks:
        if t not in rec:
            acc = ro.make_evaluate(rt, [t], args.device, "val")(rt["deltas"][t], 1.0)[t]
            rec[t] = {"expert_val_acc": acc, "reference_val_acc": rt["refs"]["val"][t],
                      "diff_pp": acc - rt["refs"]["val"][t]}
            write_json(path, rec)
        d = rec[t]["diff_pp"]
        state = "ok" if abs(d) <= EXPERT_WARN_PP else ("WARNING" if abs(d) <= EXPERT_ABORT_PP else "FAIL")
        print(f"  expert check {t}: full-val acc {rec[t]['expert_val_acc']:.2f} vs reference "
              f"{rec[t]['reference_val_acc']:.2f} ({d:+.2f} pp) {state}", flush=True)
        if state == "FAIL" and not args.skip_expert_check:
            raise SystemExit(f"expert for {t} does not reproduce its single-task accuracy "
                             f"({d:+.2f} pp > {EXPERT_ABORT_PP} pp); the expert construction is wrong. "
                             f"Stop and report (or rerun with --skip-expert-check to override, and say so).")
    return rec


class AgreementCache:
    def __init__(self, lm, tasks, forward, experts, torch, path):
        self.lm, self.tasks, self.forward, self.experts, self.torch = lm, tasks, forward, experts, torch
        self.path = path
        self.cache = json.loads(path.read_text()) if path.exists() else {}
        self.new = 0

    def __call__(self, c):
        k = key_of(c)
        if k not in self.cache:
            out = self.forward(self.lm.compose({t: ci for t, ci in zip(self.tasks, c)}))
            self.cache[k] = {t: proxy_values(self.torch, out[t], self.experts.get(t)) for t in self.tasks}
            self.new += 1
            write_json(self.path, self.cache)
        return self.cache[k]


def check_subset_matches_exp5(args, cid, info):
    """The unlabeled subset must be Experiment 5's (same seed, same first indices)."""
    meta = None
    for p in (args.exp5 / "check_metadata.json", args.exp5 / "results" / "check_metadata.json"):
        if p.exists():
            meta = json.loads(p.read_text())
            break
    if meta is None:
        print(f"[{cid}] note: Experiment 5 check_metadata.json not found under {args.exp5}; "
              f"subset identity relies on the shared loader code (seed {args.seed})", flush=True)
        return "not_checked"
    ref = meta.get("contexts", {}).get(cid)
    if ref is None:
        print(f"[{cid}] note: context not in Experiment 5 metadata (it ran core only); same loader code used", flush=True)
        return "not_in_exp5"
    if meta.get("n_unlabeled") != args.n_unlabeled or meta.get("seed") != args.seed:
        raise SystemExit(f"Experiment 5 used n={meta.get('n_unlabeled')}, seed={meta.get('seed')}; "
                         f"this run uses n={args.n_unlabeled}, seed={args.seed}")
    for t, r in ref.items():
        mine = {k: info[t][k] for k in ("n", "of", "first_indices")}
        if mine != {k: r[k] for k in ("n", "of", "first_indices")}:
            raise SystemExit(f"[{cid}] unlabeled subset for {t} differs from Experiment 5: {mine} vs {r}")
    print(f"[{cid}] unlabeled subset identical to Experiment 5 (n, split size and first indices match)", flush=True)
    return "identical"


def setup_context(rt, args, cid, experts_by_task):
    import lanes
    tasks = list(ro.CONTEXTS[cid])
    with rt["torch"].no_grad():
        lm = lanes.LaneMerge(ro.group(rt["deltas"], tasks), rt["backend"], tasks, ro.dc_kwargs("rank"))
    loaders, info = rent.unlabeled_loaders(rt, tasks, args.n_unlabeled, args.seed)
    subset_state = check_subset_matches_exp5(args, cid, info)
    experts = experts_by_task(loaders, info)
    for t in tasks:
        experts.get(t)
    fwd = make_forward(rt, tasks, loaders, args.device)
    cache = AgreementCache(lm, tasks, fwd, experts, rt["torch"],
                           args.out / cid / f"agreement_cache_n{args.n_unlabeled}_s{args.seed}.json")
    exp3 = json.loads((args.exp3 / cid / "oracle_result.json").read_text())
    return tasks, lm, cache, info, exp3, subset_state


def make_experts_factory(rt, args):
    shared = {}

    def factory(loaders, info):
        # One Experts object per run; loaders/info for a task are identical across contexts
        # (seed + task index), so expert outputs are shared.
        if "obj" not in shared:
            shared["obj"] = Experts(rt, args, dict(loaders), dict(info))
        else:
            for t in loaders:
                if t in shared["obj"].info and shared["obj"].info[t] != info[t]:
                    raise RuntimeError(f"subset for {t} changed between contexts")
                shared["obj"].loaders.setdefault(t, loaders[t])
                shared["obj"].info.setdefault(t, info[t])
        return shared["obj"]
    return factory


# ----------------------------------------------------------------------------
# entropy reference (Experiment 5)
# ----------------------------------------------------------------------------
def load_entropy_cache(args, cid):
    for p in (args.exp5 / cid / f"entropy_cache_n{args.n_unlabeled}_s{args.seed}.json",
              args.exp5 / "results" / "entropy_caches" / f"{cid}.json",
              args.exp5 / "entropy_caches" / f"{cid}.json"):
        if p.exists():
            return json.loads(p.read_text()), p
    return None, None


# ----------------------------------------------------------------------------
# check (the gate)
# ----------------------------------------------------------------------------
def pick_rows(keys, acc_mean, acc_worst, loss_mean, loss_max):
    """Spearman correlations and the validation accuracy of the vector the proxy would pick."""
    i_mean = min(range(len(keys)), key=lambda i: (loss_mean[i], loss_max[i]))
    i_fair = min(range(len(keys)), key=lambda i: (loss_max[i], loss_mean[i]))
    return {
        "rho_mean": spearman([-x for x in loss_mean], acc_mean),
        "rho_worst": spearman([-x for x in loss_max], acc_worst),
        "pick_mean_vector": keys[i_mean],
        "pick_mean_val_mean": acc_mean[i_mean],
        "pick_mean_val_worst": acc_worst[i_mean],
        "pick_fair_vector": keys[i_fair],
        "pick_fair_val_mean": acc_mean[i_fair],
        "pick_fair_val_worst": acc_worst[i_fair],
    }


def gate_decision(rows):
    """Pre-registered decision on the three core contexts (validation).

    PASS  : some proxy has rho_mean >= 0.5 in all three core contexts AND its pick beats
            dc_rank's mean normalized accuracy in all three.
    FAIL  : (every proxy has rho_mean <= 0.3 in at least one core context) OR (every
            proxy's pick is below dc_rank in all three core contexts).  This is the
            literal reading of the guide; the per-proxy reading is recorded alongside.
    else  : IN_BETWEEN -> run on core only, result labelled PARTIAL.
    """
    by = {}
    for r in rows:
        if r["context"] in CORE and r["proxy"] in PROXIES:
            by.setdefault(r["proxy"], {})[r["context"]] = r
    complete = all(set(by.get(p, {})) == set(CORE) for p in PROXIES)
    if not complete:
        return {"verdict": "INCOMPLETE", "reason": "the gate needs all three core contexts for every proxy",
                "proxy": None}
    per = {}
    for p in PROXIES:
        rs = [by[p][c] for c in CORE]
        rhos = [r["rho_mean"] for r in rs]
        rhos_ok = [x if x == x else -1.0 for x in rhos]             # NaN counts as failing
        beats = [r["pick_mean_val_mean"] > r["dc_rank_val_mean"] for r in rs]
        below = [r["pick_mean_val_mean"] < r["dc_rank_val_mean"] for r in rs]
        per[p] = {"rho_mean": dict(zip(CORE, rhos)), "min_rho_mean": min(rhos_ok),
                  "pick_beats_dc_rank": dict(zip(CORE, beats)), "pick_below_dc_rank": dict(zip(CORE, below)),
                  "passes": min(rhos_ok) >= RHO_PASS and all(beats),
                  "low_rho": min(rhos_ok) <= RHO_FAIL, "pick_below_all": all(below)}
        per[p]["fails_per_proxy_reading"] = per[p]["low_rho"] or per[p]["pick_below_all"]
    rank = lambda p: (sum(per[p]["pick_beats_dc_rank"].values()), per[p]["min_rho_mean"], -PROXIES.index(p))
    passing = [p for p in PROXIES if per[p]["passes"]]
    literal_fail = all(per[p]["low_rho"] for p in PROXIES) or all(per[p]["pick_below_all"] for p in PROXIES)
    per_proxy_fail = all(per[p]["fails_per_proxy_reading"] for p in PROXIES)
    if passing:
        verdict, proxy = "PASS", max(passing, key=rank)
        reason = f"{proxy}: rho_mean >= {RHO_PASS} and pick > dc_rank in all three core contexts"
    elif literal_fail:
        verdict, proxy = "FAIL", None
        reason = ("every proxy has rho_mean <= 0.3 in some core context"
                  if all(per[p]["low_rho"] for p in PROXIES) else
                  "every proxy's pick is below dc_rank in all three core contexts")
    else:
        verdict, proxy = "IN_BETWEEN", max(PROXIES, key=rank)
        reason = (f"no proxy passes, the literal FAIL condition is not met; best proxy by "
                  f"(contexts where pick beats dc_rank, min rho_mean) = {proxy}")
    return {"verdict": verdict, "proxy": proxy, "reason": reason, "per_proxy": per,
            "per_proxy_reading_would_fail": per_proxy_fail,
            "thresholds": {"rho_pass": RHO_PASS, "rho_fail": RHO_FAIL}}


GATE_TEXT = {
    "PASS": "GATE PASSES -> run the coordinate search with the selected proxy (core, then four)",
    "FAIL": "GATE FAILS -> stop; report the check; the label-free proxy is not the bottleneck to pursue further",
    "IN_BETWEEN": "IN BETWEEN -> run on core only; the result is labelled PARTIAL",
    "INCOMPLETE": "INCOMPLETE -> run the check on all three core contexts",
}


def cmd_check(args):
    rt = ro.load_runtime(args)
    split_note("check", "proxies use the fixed unlabeled VALIDATION subset (labels never read); "
                        "accuracies are Experiment 3's cached VALIDATION accuracies; no test split is touched")
    factory = make_experts_factory(rt, args)
    rows, meta = [], {"n_unlabeled": args.n_unlabeled, "seed": args.seed, "tau": TAU, "contexts": {}}
    expert_tasks = []
    for cid in contexts_of(args.contexts):
        expert_tasks += [t for t in ro.CONTEXTS[cid] if t not in expert_tasks]
    print("expert sanity check (full VALIDATION split, labels used only for this check):", flush=True)
    meta["expert_check"] = check_experts(rt, args, expert_tasks)
    for cid in contexts_of(args.contexts):
        t0 = time.time()
        tasks, lm, agree, info, exp3, subset_state = setup_context(rt, args, cid, factory)
        meta["contexts"][cid] = {"subset": info, "subset_vs_exp5": subset_state, "lane_checks": lm.checks}
        vcache = json.loads((args.exp3 / cid / "val_cache.json").read_text())
        refs = rt["refs"]["val"]
        keys = sorted(vcache)
        if args.max_vectors and len(keys) > args.max_vectors:
            keys = keys[: args.max_vectors]
        acc_mean, acc_worst, P = [], [], {}
        for k in keys:
            c = [float(x) for x in k.split(",")]
            P[k] = agree(c)
            n = {t: 100.0 * vcache[k][t] / refs[t] for t in tasks}
            acc_mean.append(sum(n.values()) / len(n))
            acc_worst.append(min(n.values()))
        base = {"context": cid, "vectors": len(keys),
                "best_val_mean": max(acc_mean), "best_val_worst": max(acc_worst),
                "dc_rank_val_mean": exp3["val_baseline"]["mean_norm"],
                "dc_rank_val_worst": exp3["val_baseline"]["worst_norm"],
                "oracle_fair_val_mean": exp3["val_chosen"]["fair"]["val_mean_norm"],
                "oracle_fair_val_worst": exp3["val_chosen"]["fair"]["val_worst_norm"]}
        for proxy in PROXIES:
            lm_, lx_ = zip(*(losses(P[k], tasks, proxy) for k in keys))
            rows.append({**base, "proxy": proxy, **pick_rows(keys, acc_mean, acc_worst, list(lm_), list(lx_))})
        # entropy reference on the same vectors (Experiment 5)
        H, hp = load_entropy_cache(args, cid)
        if H is not None:
            ks = [i for i, k in enumerate(keys) if k in H]
            if len(ks) == len(keys):
                hm = [sum(H[k].values()) / len(H[k]) for k in keys]
                hx = [max(H[k].values()) for k in keys]
                rows.append({**base, "proxy": "entropy (Exp 5)", "source": str(hp),
                             **pick_rows(keys, acc_mean, acc_worst, hm, hx)})
            else:
                print(f"[{cid}] entropy cache {hp} covers {len(ks)}/{len(keys)} vectors; reference row skipped",
                      flush=True)
        else:
            print(f"[{cid}] Experiment 5 entropy cache not found under {args.exp5}; reference row skipped", flush=True)
        for r in rows[-(len(PROXIES) + (1 if rows[-1]["proxy"].startswith("entropy") else 0)):]:
            print(f"[{cid}] {r['proxy']:16s} rho_mean={r['rho_mean']:+.2f} rho_worst={r['rho_worst']:+.2f} "
                  f"pick mean={r['pick_mean_val_mean']:.2f} vs dc_rank {r['dc_rank_val_mean']:.2f}", flush=True)
        print(f"[{cid}] {len(keys)} vectors in {(time.time()-t0)/60:.1f} min ({agree.new} new proxy evaluations)",
              flush=True)
    gate = gate_decision(rows)
    meta["code_sha256"] = {f: sha256(HERE / f) for f in ("run_agreement.py", "run_entropy.py", "lanes.py",
                                                          "run_oracle.py", "run_official.py", "dc_trace.py")}
    meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(args.out / "agreement_check.json", {"rows": rows, "gate": gate, "metadata": meta})
    write_check_summary(args.out, rows, gate, args.n_unlabeled, args.seed)
    print(f"GATE: {gate['verdict']} ({gate['reason']})", flush=True)


def fmt(x, nd=2):
    return "—" if x is None or x != x else f"{x:.{nd}f}"


def write_check_summary(out, rows, gate, n_unlabeled, seed):
    lines = ["# Experiment 6a check: does agreement with the experts pick high accuracy? "
             "(validation, Exp. 3 vectors)", "",
             f"Proxies: {n_unlabeled} unlabeled validation images per task (seed {seed}, Experiment 5's subset); labels are not "
             "read by the proxy. Accuracies: Experiment 3's cached validation accuracies, normalized by the "
             "single-task (expert) accuracy. **Not data-free:** needs unlabeled images and the experts.", "",
             "| context | proxy | vectors | ρ(−mean loss, mean acc) | ρ(−max loss, worst acc) | "
             "mean acc: best / pick / dc_rank | worst acc: best / fair pick / dc_rank | pick > dc_rank? |",
             "|---|---|---:|---:|---:|---|---|:---:|"]
    for r in rows:
        lines.append(
            f"| {r['context']} | {r['proxy']} | {r['vectors']} | {fmt(r['rho_mean'])} | {fmt(r['rho_worst'])} | "
            f"{fmt(r['best_val_mean'])} / {fmt(r['pick_mean_val_mean'])} / {fmt(r['dc_rank_val_mean'])} | "
            f"{fmt(r['best_val_worst'])} / {fmt(r['pick_fair_val_worst'])} / {fmt(r['dc_rank_val_worst'])} | "
            f"{'yes' if r['pick_mean_val_mean'] > r['dc_rank_val_mean'] else 'no'} |")
    lines += ["", "Proxy losses: " + "; ".join(f"`{p}` = {PROXY_LABEL[p]}" for p in PROXIES) +
              ". The entropy rows are Experiment 5's reference on the same vectors.", "",
              "## Gate (pre-registered, core contexts, validation)", "",
              f"**{gate['verdict']}** — {gate['reason']}", "", GATE_TEXT[gate["verdict"]], ""]
    if gate.get("per_proxy"):
        lines += ["| proxy | ρ_mean per core context | min ρ_mean | pick beats dc_rank | passes | "
                  "fails (per-proxy reading) |", "|---|---|---:|---|:---:|:---:|"]
        for p, d in gate["per_proxy"].items():
            lines.append(f"| {p} | " + ", ".join(fmt(v) for v in d["rho_mean"].values()) +
                         f" | {fmt(d['min_rho_mean'])} | " +
                         ", ".join("yes" if v else "no" for v in d["pick_beats_dc_rank"].values()) +
                         f" | {'yes' if d['passes'] else 'no'} | {'yes' if d['fails_per_proxy_reading'] else 'no'} |")
        lines += ["", "FAIL uses the literal reading (*every* proxy has a core ρ_mean ≤ 0.3, *or* *every* proxy's "
                  "pick is below dc_rank in all three core contexts). Under the stricter per-proxy reading "
                  f"(each proxy fails at least one of the two), the gate would "
                  f"{'also FAIL' if gate['per_proxy_reading_would_fail'] else 'not fail'}."]
    out.mkdir(parents=True, exist_ok=True)
    (out / "agreement_check_summary.md").write_text("\n".join(lines) + "\n")
    with open(out / "agreement_check.csv", "w", newline="") as fh:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("context", "proxy"), k))
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out/'agreement_check_summary.md'} and {out/'agreement_check.csv'}")


# ----------------------------------------------------------------------------
# run: search with the gated proxy + one test evaluation per chosen vector
# ----------------------------------------------------------------------------
def objective_value(P, tasks, proxy, objective):
    """Higher is better (we minimise the proxy loss)."""
    mean, worst = losses(P, tasks, proxy)
    return (-mean, -worst) if objective == "agreement_mean" else (-worst, -mean)


def coordinate_search(agree, tasks, start, proxy, objective, max_passes, log):
    """Same greedy coordinate search as Experiments 3 and 5 (grid plus the start value)."""
    c = list(start)
    best = objective_value(agree(c), tasks, proxy, objective)
    for p in range(max_passes):
        improved = False
        for i, t in enumerate(tasks):
            for v in sorted(set(GRID) | {start[i]}):
                if v == c[i]:
                    continue
                trial = list(c)
                trial[i] = v
                val = objective_value(agree(trial), tasks, proxy, objective)
                if val > tuple(b + 1e-9 for b in best):
                    c, best, improved = trial, val, True
            m, w = losses(agree(c), tasks, proxy)
            log.append({"objective": objective, "pass": p + 1, "after_task": t, "c": list(c),
                        "mean_loss": m, "max_loss": w})
            print(f"    [{objective}] pass {p+1} after {t:14s} c={key_of(c):28s} "
                  f"mean {proxy}={m:.4f} max {proxy}={w:.4f}", flush=True)
        if not improved:
            break
    return c


def read_gate(out):
    p = out / "agreement_check.json"
    if not p.exists():
        raise SystemExit(f"{p} not found: run `check` on the core contexts first")
    return json.loads(p.read_text())["gate"]


def cmd_run(args):
    import run_oracle as rx
    gate = read_gate(args.out)
    if gate["verdict"] in ("FAIL", "INCOMPLETE"):
        raise SystemExit(f"gate verdict is {gate['verdict']} ({gate['reason']}); the pre-registered rule "
                         f"says not to run the search")
    contexts = contexts_of(args.contexts)
    if gate["verdict"] == "IN_BETWEEN" and any(c not in CORE for c in contexts):
        raise SystemExit("gate is IN_BETWEEN: the pre-registered rule allows the core contexts only")
    proxy = gate["proxy"]
    print(f"gate {gate['verdict']}: searching with proxy '{proxy}' ({PROXY_LABEL[proxy]})", flush=True)
    split_note("run", "search uses the unlabeled VALIDATION subset only; chosen vectors are scored once on TEST "
                      "(result) and once on VALIDATION (diagnostic); dc_rank/oracle/CT rows are copied from Exp 3")
    rt = ro.load_runtime(args)
    factory = make_experts_factory(rt, args)
    for cid in contexts:
        res_path = args.out / cid / "agreement_result.json"
        if res_path.exists():
            prev = json.loads(res_path.read_text())
            if prev.get("proxy") != proxy or prev.get("gate_verdict") != gate["verdict"]:
                raise SystemExit(f"[{cid}] existing {res_path} used proxy {prev.get('proxy')} / gate "
                                 f"{prev.get('gate_verdict')}, but the current gate says {proxy} / "
                                 f"{gate['verdict']}; move the old results away before rerunning")
            print(f"[{cid}] done, skipping", flush=True)
            continue
        t0 = time.time()
        tasks, lm, agree, info, exp3, subset_state = setup_context(rt, args, cid, factory)
        alpha = float(exp3["dc_rank_alpha"])
        start = [alpha] * len(tasks)
        print(f"[{cid}] dc_rank alpha = {alpha}; unlabeled images per task: "
              + ", ".join(f"{t}={info[t]['n']}" for t in tasks), flush=True)
        log, chosen = [], {}
        for obj in OBJECTIVES:
            chosen[obj] = coordinate_search(agree, tasks, start, proxy, obj, args.max_passes, log)
        # labels enter only here
        test = rx.Evaluator(rt, lm, tasks, args.device, "test", rt["refs"]["test"], args.out / cid / "test_cache.json")
        val = rx.Evaluator(rt, lm, tasks, args.device, "val", rt["refs"]["val"], args.out / cid / "val_cache.json")
        results = {}
        for name in OBJECTIVES:
            c = chosen[name]
            acc, n = test(c)
            vacc, vn = val(c)
            results[name] = {"c": dict(zip(tasks, c)), "test_acc": acc, "test_norm": n,
                             "test_mean_norm": sum(n.values()) / len(n), "test_worst_norm": min(n.values()),
                             "test_worst_task": min(n, key=n.get),
                             "val_mean_norm": sum(vn.values()) / len(vn), "val_worst_norm": min(vn.values()),
                             "proxies": agree(c)}
        for name in ("dc_rank", "oracle_fair", "oracle_mean", "ct"):
            results[name] = dict(exp3["test"][name])
        results["dc_rank"]["proxies"] = agree(start)
        write_json(res_path, {"context": cid, "tasks": tasks, "dc_rank_alpha": alpha, "grid": list(GRID),
                              "proxy": proxy, "gate_verdict": gate["verdict"], "tau": TAU,
                              "n_unlabeled": args.n_unlabeled, "seed": args.seed, "unlabeled_subset": info,
                              "subset_vs_exp5": subset_state, "max_passes": args.max_passes,
                              "proxy_evals": len(agree.cache), "lane_checks": lm.checks,
                              "search_log": log, "results": results})
        print(f"[{cid}] done in {(time.time()-t0)/60:.1f} min ({len(agree.cache)} proxy evaluations)", flush=True)
    summarize(args.out)


def verdict(dm_method, dm_oracle, worst_drops, capped):
    retention = dm_method / dm_oracle if dm_oracle > 0 else float("nan")
    if dm_oracle > 0 and retention >= 0.5 and worst_drops == 0:
        v = "VIABLE: keeps >= 50% of oracle_fair's mean gain, no context loses > 1 pp on the worst task"
        if capped:
            v = "PARTIAL (gate was IN BETWEEN; would otherwise be VIABLE)"
        return retention, v
    if dm_method > 0:
        return retention, "PARTIAL: gains below the bar or a context with worst-task drop > 1 pp"
    return retention, "NOT VIABLE: no mean gain over dc_rank"


def summarize(out):
    files = sorted(out.glob("*/agreement_result.json"))
    if not files:
        raise SystemExit(f"no agreement_result.json in {out}")
    methods = ("dc_rank", "agreement_mean", "agreement_fair", "oracle_fair", "oracle_mean", "ct")
    first = json.loads(files[0].read_text())
    capped = first["gate_verdict"] == "IN_BETWEEN"
    lines = [f"# Experiment 6a: expert-agreement lane coefficients (TEST), proxy = `{first['proxy']}`", "",
             "The search uses only unlabeled validation images and the experts; labels score the chosen vectors "
             "once on test. **Not data-free** (needs unlabeled images and the per-task experts).",
             f"Gate verdict: **{first['gate_verdict']}**." + (" Result capped at PARTIAL." if capped else ""), "",
             "| context | method | coefficients | mean norm | worst norm (task) | Δmean | Δworst |",
             "|---|---|---|---:|---|---:|---:|"]
    agg = {m: [] for m in methods}
    drops = {m: 0 for m in methods}
    rows = []
    for p in files:
        d = json.loads(p.read_text())
        b = d["results"]["dc_rank"]
        for m in methods:
            r = d["results"][m]
            dm, dw = r["test_mean_norm"] - b["test_mean_norm"], r["test_worst_norm"] - b["test_worst_norm"]
            cs = ", ".join(f"{t[:6]}={v:g}" for t, v in r["c"].items())
            lines.append(f"| {d['context']} | {m} | {cs} | {r['test_mean_norm']:.2f} | "
                         f"{r['test_worst_norm']:.2f} ({r['test_worst_task']}) | {dm:+.2f} | {dw:+.2f} |")
            agg[m].append((dm, dw))
            drops[m] += dw < -1.0
            rows.append({"context": d["context"], "method": m, "test_mean_norm": r["test_mean_norm"],
                         "test_worst_norm": r["test_worst_norm"], "d_mean": dm, "d_worst": dw,
                         **{f"c_{t}": v for t, v in r["c"].items()}})
    core_files = [f for f in files if f.parent.name in CORE]
    for label, sel in (("core", core_files), ("all", files)):
        if not sel or (label == "all" and len(files) == len(core_files)):
            continue
        idx = [files.index(f) for f in sel]
        n = len(idx)
        lines += ["", f"## Averages over {n} contexts ({label}; test, Δ vs dc_rank)", "",
                  "| method | Δmean | Δworst | contexts with worst drop > 1 pp | share of oracle_fair Δmean | verdict |",
                  "|---|---:|---:|---:|---:|---|"]
        dm_or = sum(agg["oracle_fair"][i][0] for i in idx) / n
        for m in methods:
            dm = sum(agg[m][i][0] for i in idx) / n
            dw = sum(agg[m][i][1] for i in idx) / n
            dr = sum(1 for i in idx if agg[m][i][1] < -1.0)
            if m.startswith("agreement"):
                ret, v = verdict(dm, dm_or, dr, capped)
                share = f"{100*ret:.0f}%" if ret == ret else "—"
            else:
                share, v = "", ""
            lines.append(f"| {m} | {dm:+.2f} | {dw:+.2f} | {dr} | {share} | {v} |")
    (out / "agreement_summary.md").write_text("\n".join(lines) + "\n")
    with open(out / "agreement.csv", "w", newline="") as fh:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("context", "method"), k))
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out/'agreement_summary.md'} and {out/'agreement.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "run"):
        p = sub.add_parser(name)
        for a in ("--repo", "--model", "--ft-dir", "--heads", "--data", "--exp3", "--exp5"):
            p.add_argument(a, type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--contexts", default="core")
        p.add_argument("--n-unlabeled", type=int, default=500, help="unlabeled images per task (keep 500)")
        p.add_argument("--seed", type=int, default=0, help="keep 0 (Experiment 5's subset)")
        p.add_argument("--device", default="cuda")
        p.add_argument("--skip-expert-check", action="store_true",
                       help="continue even if an expert misses its single-task accuracy by > 1 pp (report it)")
        if name == "run":
            p.add_argument("--max-passes", type=int, default=2)
        else:
            p.add_argument("--max-vectors", type=int, default=0, help="limit cached vectors per context (0 = all)")
    s = sub.add_parser("summarize")
    s.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out = args.out.resolve()
    if args.cmd == "summarize":
        summarize(args.out)
        return
    for a in ("repo", "model", "ft_dir", "heads", "data", "exp3", "exp5"):
        setattr(args, a, getattr(args, a).resolve())
    (cmd_check if args.cmd == "check" else cmd_run)(args)


if __name__ == "__main__":
    main()
