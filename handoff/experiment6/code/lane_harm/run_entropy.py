#!/usr/bin/env python3
"""Experiment 5: choose per-task lane coefficients WITHOUT labels (entropy), test once.

Same search space as Experiment 3 (W(c) = W0 + sum_i c_i L_i on the dc_rank lanes, grid
0.3..1.0 plus the dc_rank alpha, greedy coordinate search, <= 2 passes) but the
objective uses only unlabeled images:

    H_i(c) = mean over a fixed unlabeled image subset of task i of the normalized
             prediction entropy  -sum_k p_k log p_k / log K,
             p = softmax(tau * <f(x)/|f(x)|, h_k/|h_k|>),  tau = 100 (CLIP logit scale)

    entropy_mean : minimise (1/T) sum_i H_i(c)
    entropy_fair : minimise max_i H_i(c)       (ties broken by the mean)

The unlabeled subset is a fixed, seeded sample of each task's VALIDATION images; labels
are never read during the search.  Accuracy (labels) is computed only for the chosen
vectors: once on test (the result) and once on validation (diagnostic).

Subcommands
  check      label-free signal check on Experiment 3's cached vectors: does low entropy
             pick high validation accuracy?  (entropy evaluations only, cheap)
  run        the entropy search per context + one test evaluation
  summarize  rebuild the summary
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_official as ro  # noqa: E402

GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
FOUR = list(ro.CONTEXTS)[:7]
CORE = list(ro.CONTEXTS)[:3]
TAU = 100.0
OBJECTIVES = ("entropy_mean", "entropy_fair")


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def key_of(c):
    return ",".join(f"{x:g}" for x in c)


def contexts_of(spec):
    ids = CORE if spec == "core" else (FOUR if spec == "four" else spec.split(","))
    bad = [c for c in ids if c not in ro.CONTEXTS]
    if bad:
        raise SystemExit(f"unknown contexts {bad}")
    return ids


# ----------------------------------------------------------------------------
# unlabeled subsets and entropy
# ----------------------------------------------------------------------------
def unlabeled_loaders(rt, tasks, n_per_task, seed):
    """Fixed seeded subset of each task's validation images; labels are dropped."""
    import torch
    from torch.utils.data import DataLoader, Subset
    out, info = {}, {}
    for t in tasks:
        base = rt["data"][ro.TASKS.index(t)]["test"]["val"]
        ds = base.dataset
        g = torch.Generator().manual_seed(seed + ro.TASKS.index(t))
        idx = torch.randperm(len(ds), generator=g)[: min(n_per_task, len(ds))].tolist()
        out[t] = DataLoader(Subset(ds, idx), batch_size=getattr(base, "batch_size", 32) or 32,
                            shuffle=False, num_workers=getattr(base, "num_workers", 0))
        info[t] = {"n": len(idx), "of": len(ds), "first_indices": idx[:5]}
    return out, info


def make_entropy(rt, tasks, loaders, device):
    torch = rt["torch"]
    heads = {t: rt["heads"][t].float() for t in tasks}
    heads = {t: (h / h.norm(dim=-1, keepdim=True)).to(device) for t, h in heads.items()}

    def entropy(merged):
        with torch.no_grad():
            model = rt["apply_merge"](rt["new"], merged, scaling_coeffs=1.0).to(device)
        model.eval()
        out = {}
        try:
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                for t in tasks:
                    tot, n = 0.0, 0
                    k = heads[t].shape[0]
                    for batch in loaders[t]:
                        x = batch[0] if isinstance(batch, (list, tuple)) else batch   # labels ignored
                        f = model(x.to(device)).float()
                        f = f / f.norm(dim=-1, keepdim=True)
                        logp = torch.log_softmax(TAU * f @ heads[t].T, dim=-1)
                        h = -(logp.exp() * logp).sum(-1) / math.log(k)
                        tot += float(h.sum())
                        n += h.numel()
                    out[t] = tot / max(n, 1)
        finally:
            model.cpu()
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        return out
    return entropy


class EntropyCache:
    def __init__(self, lm, tasks, fn, path):
        self.lm, self.tasks, self.fn, self.path = lm, tasks, fn, path
        self.cache = json.loads(path.read_text()) if path.exists() else {}

    def __call__(self, c):
        k = key_of(c)
        if k not in self.cache:
            self.cache[k] = self.fn(self.lm.compose({t: ci for t, ci in zip(self.tasks, c)}))
            write_json(self.path, self.cache)
        return self.cache[k]


def objective_value(H, objective):
    """Higher is better (we minimise entropy)."""
    vals = list(H.values())
    mean, worst = sum(vals) / len(vals), max(vals)
    if objective == "entropy_mean":
        return (-mean, -worst)
    return (-worst, -mean)


def coordinate_search(ent, tasks, start, objective, max_passes, log):
    c = list(start)
    best = objective_value(ent(c), objective)
    for p in range(max_passes):
        improved = False
        for i, t in enumerate(tasks):
            for v in sorted(set(GRID) | {start[i]}):
                if v == c[i]:
                    continue
                trial = list(c)
                trial[i] = v
                val = objective_value(ent(trial), objective)
                if val > tuple(b + 1e-9 for b in best):
                    c, best, improved = trial, val, True
            H = ent(c)
            log.append({"objective": objective, "pass": p + 1, "after_task": t, "c": list(c),
                        "mean_entropy": sum(H.values()) / len(H), "max_entropy": max(H.values())})
            print(f"    [{objective}] pass {p+1} after {t:14s} c={key_of(c):28s} "
                  f"mean H={sum(H.values())/len(H):.4f} max H={max(H.values()):.4f}", flush=True)
        if not improved:
            break
    return c


def setup_context(rt, args, cid):
    import lanes
    tasks = list(ro.CONTEXTS[cid])
    with rt["torch"].no_grad():
        lm = lanes.LaneMerge(ro.group(rt["deltas"], tasks), rt["backend"], tasks, ro.dc_kwargs("rank"))
    loaders, info = unlabeled_loaders(rt, tasks, args.n_unlabeled, args.seed)
    ent = EntropyCache(lm, tasks, make_entropy(rt, tasks, loaders, args.device),
                       args.out / cid / f"entropy_cache_n{args.n_unlabeled}_s{args.seed}.json")
    exp3 = json.loads((args.exp3 / cid / "oracle_result.json").read_text())
    return tasks, lm, ent, info, exp3


# ----------------------------------------------------------------------------
# check: does entropy rank Experiment 3's cached vectors like accuracy does?
# ----------------------------------------------------------------------------
def spearman(a, b):
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb) if va and vb else float("nan")


def cmd_check(args):
    rt = ro.load_runtime(args)
    rows = []
    for cid in contexts_of(args.contexts):
        tasks, lm, ent, info, exp3 = setup_context(rt, args, cid)
        vcache = json.loads((args.exp3 / cid / "val_cache.json").read_text())
        refs = rt["refs"]["val"]
        keys = sorted(vcache)
        if args.max_vectors and len(keys) > args.max_vectors:
            keys = keys[: args.max_vectors]
        t0 = time.time()
        acc_mean, acc_worst, h_mean, h_max = [], [], [], []
        for k in keys:
            c = [float(x) for x in k.split(",")]
            H = ent(c)
            n = {t: 100.0 * vcache[k][t] / refs[t] for t in tasks}
            acc_mean.append(sum(n.values()) / len(n))
            acc_worst.append(min(n.values()))
            h_mean.append(sum(H.values()) / len(H))
            h_max.append(max(H.values()))
        i_mean = min(range(len(keys)), key=lambda i: h_mean[i])
        i_fair = min(range(len(keys)), key=lambda i: (h_max[i], h_mean[i]))
        row = {"context": cid, "vectors": len(keys),
               "spearman(-mean H, val mean acc)": spearman([-x for x in h_mean], acc_mean),
               "spearman(-max H, val worst acc)": spearman([-x for x in h_max], acc_worst),
               "best val mean acc among cached": max(acc_mean),
               "val mean acc of lowest-mean-H vector": acc_mean[i_mean],
               "val worst acc of lowest-max-H vector": acc_worst[i_fair],
               "best val worst acc among cached": max(acc_worst),
               "dc_rank val mean acc": exp3["val_baseline"]["mean_norm"],
               "dc_rank val worst acc": exp3["val_baseline"]["worst_norm"]}
        rows.append(row)
        print(f"[{cid}] {len(keys)} vectors in {(time.time()-t0)/60:.1f} min  "
              f"rho(mean)={row['spearman(-mean H, val mean acc)']:.2f}  "
              f"rho(worst)={row['spearman(-max H, val worst acc)']:.2f}", flush=True)
    lines = ["# Experiment 5 check: does low entropy pick high accuracy? (validation, Exp. 3 vectors)", "",
             "| context | vectors | ρ(−mean H, mean acc) | ρ(−max H, worst acc) | mean acc: best / entropy pick / dc_rank | worst acc: best / entropy pick / dc_rank |",
             "|---|---:|---:|---:|---|---|"]
    for r in rows:
        lines.append(f"| {r['context']} | {r['vectors']} | {r['spearman(-mean H, val mean acc)']:.2f} | "
                     f"{r['spearman(-max H, val worst acc)']:.2f} | "
                     f"{r['best val mean acc among cached']:.2f} / {r['val mean acc of lowest-mean-H vector']:.2f} / {r['dc_rank val mean acc']:.2f} | "
                     f"{r['best val worst acc among cached']:.2f} / {r['val worst acc of lowest-max-H vector']:.2f} / {r['dc_rank val worst acc']:.2f} |")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "check_summary.md").write_text("\n".join(lines) + "\n")
    write_json(args.out / "check.json", rows)
    print(f"wrote {args.out/'check_summary.md'}")


# ----------------------------------------------------------------------------
# run: entropy search + one test evaluation
# ----------------------------------------------------------------------------
def cmd_run(args):
    import run_oracle as rx
    rt = ro.load_runtime(args)
    for cid in contexts_of(args.contexts):
        res_path = args.out / cid / "entropy_result.json"
        if res_path.exists():
            print(f"[{cid}] done, skipping", flush=True)
            continue
        t0 = time.time()
        tasks, lm, ent, info, exp3 = setup_context(rt, args, cid)
        alpha = float(exp3["dc_rank_alpha"])
        start = [alpha] * len(tasks)
        print(f"[{cid}] dc_rank alpha = {alpha}; unlabeled images per task: "
              + ", ".join(f"{t}={info[t]['n']}" for t in tasks), flush=True)
        log, chosen = [], {}
        for obj in OBJECTIVES:
            chosen[obj] = coordinate_search(ent, tasks, start, obj, args.max_passes, log)
        # labels enter only here: test (result) and validation (diagnostic)
        test = rx.Evaluator(rt, lm, tasks, args.device, "test", rt["refs"]["test"], args.out / cid / "test_cache.json")
        val = rx.Evaluator(rt, lm, tasks, args.device, "val", rt["refs"]["val"], args.out / cid / "val_cache.json")
        results = {}
        for name in OBJECTIVES:
            c = chosen[name]
            acc, n = test(c)
            vacc, vn = val(c)
            H = ent(c)
            results[name] = {"c": dict(zip(tasks, c)), "test_acc": acc, "test_norm": n,
                             "test_mean_norm": sum(n.values()) / len(n), "test_worst_norm": min(n.values()),
                             "test_worst_task": min(n, key=n.get),
                             "val_mean_norm": sum(vn.values()) / len(vn), "val_worst_norm": min(vn.values()),
                             "entropy": H}
        H0 = ent(start)
        for name in ("dc_rank", "oracle_fair", "oracle_mean", "ct"):
            results[name] = dict(exp3["test"][name])
        results["dc_rank"]["entropy"] = H0
        write_json(res_path, {"context": cid, "tasks": tasks, "dc_rank_alpha": alpha, "grid": list(GRID),
                              "tau": TAU, "n_unlabeled": args.n_unlabeled, "seed": args.seed,
                              "unlabeled_subset": info, "max_passes": args.max_passes,
                              "entropy_evals": len(ent.cache), "lane_checks": lm.checks,
                              "search_log": log, "results": results})
        print(f"[{cid}] done in {(time.time()-t0)/60:.1f} min ({len(ent.cache)} entropy evaluations)", flush=True)
    summarize(args.out)


def verdict(dm_entropy, dm_oracle, worst_drops):
    retention = dm_entropy / dm_oracle if dm_oracle > 0 else float("nan")
    if dm_oracle > 0 and retention >= 0.5 and worst_drops == 0:
        return retention, "METHOD VIABLE: build the gradient version, add AdaMerging/AdaRank baselines"
    if dm_entropy > 0:
        return retention, "PARTIAL: gains below the bar or a context with worst drop > 1 pp; discuss"
    return retention, "NOT VIABLE: entropy does not recover the headroom; write the analysis paper"


def summarize(out):
    files = sorted(out.glob("*/entropy_result.json"))
    if not files:
        raise SystemExit(f"no results in {out}")
    methods = ("dc_rank", "entropy_mean", "entropy_fair", "oracle_fair", "ct")
    lines = ["# Experiment 5: label-free lane coefficients (TEST)", "",
             "Entropy objectives use only unlabeled validation images; labels are used only to score the chosen vectors.", "",
             "| context | method | coefficients | mean norm | worst norm (task) | Δmean | Δworst |",
             "|---|---|---|---:|---|---:|---:|"]
    agg = {m: [] for m in methods}
    worst_drops = {m: 0 for m in methods}
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
            if dw < -1.0:
                worst_drops[m] += 1
            rows.append({"context": d["context"], "method": m, "test_mean_norm": r["test_mean_norm"],
                         "test_worst_norm": r["test_worst_norm"], "d_mean": dm, "d_worst": dw,
                         **{f"c_{t}": v for t, v in r["c"].items()}})
    n = len(files)
    lines += ["", f"## Averages over {n} contexts (test, Δ vs dc_rank)", "",
              "| method | Δmean | Δworst | contexts with worst drop > 1 pp | share of oracle_fair Δmean | verdict |",
              "|---|---:|---:|---:|---:|---|"]
    dm_or = sum(x[0] for x in agg["oracle_fair"]) / n
    for m in methods:
        dm = sum(x[0] for x in agg[m]) / n
        dw = sum(x[1] for x in agg[m]) / n
        if m.startswith("entropy"):
            ret, v = verdict(dm, dm_or, worst_drops[m])
            share = f"{100*ret:.0f}%" if ret == ret else "—"
        else:
            share, v = "", ""
        lines.append(f"| {m} | {dm:+.2f} | {dw:+.2f} | {worst_drops[m]} | {share} | {v} |")
    (out / "entropy_summary.md").write_text("\n".join(lines) + "\n")
    with open(out / "entropy.csv", "w", newline="") as fh:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("context", "method"), k))
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out/'entropy_summary.md'} and {out/'entropy.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "run"):
        p = sub.add_parser(name)
        for a in ("--repo", "--model", "--ft-dir", "--heads", "--data", "--exp3"):
            p.add_argument(a, type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--contexts", default="core")
        p.add_argument("--n-unlabeled", type=int, default=500, help="unlabeled images per task")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--device", default="cuda")
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
    for a in ("repo", "model", "ft_dir", "heads", "data", "exp3"):
        setattr(args, a, getattr(args, a).resolve())
    (cmd_check if args.cmd == "check" else cmd_run)(args)


if __name__ == "__main__":
    main()
