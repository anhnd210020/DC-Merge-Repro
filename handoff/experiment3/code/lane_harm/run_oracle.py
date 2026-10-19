#!/usr/bin/env python3
"""Experiment 3: is there an oracle per-task coefficient that beats dc_rank on test?

Operator: DC-Merge with the rank-aware top-k (``dc_rank``).  Its update is an exact
sum of per-task lanes, so a per-task coefficient vector c gives

    W(c) = W0 + sum_i c_i L_i ,

and the usual global alpha is the special case c_i = alpha for all i.

For each context the script
  1. starts from dc_rank's validation-selected alpha (read from Experiment 2a),
  2. runs a coordinate search over c on VALIDATION for two objectives
       mean : maximise the mean normalized accuracy
       fair : the same, subject to  min_i n_i(c) >= min_i n_i(baseline)
  3. evaluates the baseline and both oracle vectors ONCE on TEST, and
  4. compares with dc_rank and CT-Merging from Experiment 2a (test).

Every evaluation is cached on disk, so an interrupted search resumes where it stopped.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_official as ro  # noqa: E402

GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
FOUR = list(ro.CONTEXTS)[:7]
CORE = list(ro.CONTEXTS)[:3]
OBJECTIVES = ("mean", "fair")


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def key_of(c):
    return ",".join(f"{x:g}" for x in c)


class Evaluator:
    """Evaluate W0 + sum_i c_i L_i on a split, with an on-disk cache."""

    def __init__(self, rt, lm, tasks, device, split, refs, cache_path):
        self.rt, self.lm, self.tasks, self.refs = rt, lm, tasks, refs
        self.fn = ro.make_evaluate(rt, tasks, device, split)
        self.cache_path = cache_path
        self.cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        self.new_evals = 0

    def __call__(self, c):
        k = key_of(c)
        if k not in self.cache:
            merged = self.lm.compose({t: ci for t, ci in zip(self.tasks, c)})
            acc = self.fn(merged, 1.0)                    # coefficients are already inside the lanes
            self.cache[k] = acc
            self.new_evals += 1
            write_json(self.cache_path, self.cache)
        acc = self.cache[k]
        n = {t: 100.0 * acc[t] / self.refs[t] for t in self.tasks}
        return acc, n


def score(n):
    return sum(n.values()) / len(n), min(n.values())


def coordinate_search(ev, tasks, start, objective, floor, max_passes, log):
    """Greedy coordinate ascent over GRID (plus the start value) per task."""
    def value(c):
        _, n = ev(c)
        mean, worst = score(n)
        feasible = objective == "mean" or worst >= floor - 1e-9
        return (mean if feasible else -1e9), mean, worst

    c = list(start)
    best = value(c)
    for p in range(max_passes):
        improved = False
        for i, t in enumerate(tasks):
            cand_vals = sorted(set(GRID) | {start[i]})
            for v in cand_vals:
                if v == c[i]:
                    continue
                trial = list(c)
                trial[i] = v
                val = value(trial)
                if val[0] > best[0] + 1e-9:
                    c, best, improved = trial, val, True
            log.append({"objective": objective, "pass": p + 1, "after_task": t, "c": list(c),
                        "val_mean_norm": best[1], "val_worst_norm": best[2]})
            print(f"    [{objective}] pass {p+1} after {t:14s} c={key_of(c):28s} "
                  f"val mean={best[1]:.2f} worst={best[2]:.2f}", flush=True)
        if not improved:
            break
    return c, best


def run(args):
    import lanes
    rt = ro.load_runtime(args)
    torch = rt["torch"]
    cmp_dir = args.exp2 / "compare"
    out = args.out
    contexts = CORE if args.contexts == "core" else (FOUR if args.contexts == "four" else args.contexts.split(","))
    for cid in contexts:
        if cid not in ro.CONTEXTS:
            raise SystemExit(f"unknown context {cid}")
        res_path = out / cid / "oracle_result.json"
        if res_path.exists():
            print(f"[{cid}] done, skipping", flush=True)
            continue
        tasks = list(ro.CONTEXTS[cid])
        rank_json = json.loads((cmp_dir / f"{cid}__dc_rank.json").read_text())
        ct_json = json.loads((cmp_dir / f"{cid}__ct.json").read_text())
        alpha = float(rank_json["selected_coef"])
        t0 = time.time()
        print(f"[{cid}] dc_rank alpha = {alpha}", flush=True)
        with torch.no_grad():
            lm = lanes.LaneMerge(ro.group(rt["deltas"], tasks), rt["backend"], tasks, ro.dc_kwargs("rank"))
        val = Evaluator(rt, lm, tasks, args.device, "val", rt["refs"]["val"], out / cid / "val_cache.json")
        test = Evaluator(rt, lm, tasks, args.device, "test", rt["refs"]["test"], out / cid / "test_cache.json")

        # sanity: uniform c = alpha must reproduce Experiment 2a's dc_rank numbers
        base = [alpha] * len(tasks)
        acc_v, n_v = val(base)
        exp2_val = next(s["acc"] for s in rank_json["val_sweep"] if abs(s["coef"] - alpha) < 1e-9)
        val_diff = max(abs(acc_v[t] - exp2_val[t]) for t in tasks)
        print(f"[{cid}] baseline validation diff vs Experiment 2a: {val_diff:.3f} pp", flush=True)
        if val_diff > 0.1:
            print(f"[{cid}] WARNING: validation baseline differs by {val_diff:.3f} pp (> 0.1 pp); continuing as authorized, using the current-environment baseline for primary comparisons", flush=True)

        # Verify test reproduction before any coordinate search. The later dc_rank
        # test result reuses this cached baseline evaluation.
        base_test_acc, _ = test(base)
        test_diff = max(abs(base_test_acc[t] - rank_json["test_acc"][t]) for t in tasks)
        print(f"[{cid}] baseline test diff vs Experiment 2a: {test_diff:.3f} pp", flush=True)
        if test_diff > 0.1:
            print(f"[{cid}] WARNING: test baseline differs by {test_diff:.3f} pp (> 0.1 pp); continuing as authorized, using the current-environment baseline for primary comparisons", flush=True)

        base_mean, base_worst = score(n_v)

        log, chosen = [], {}
        for obj in OBJECTIVES:
            c, best = coordinate_search(val, tasks, base, obj, base_worst, args.max_passes, log)
            chosen[obj] = {"c": c, "val_mean_norm": best[1], "val_worst_norm": best[2]}

        # one test evaluation per vector
        results = {}
        for name, c in [("dc_rank", base)] + [(f"oracle_{o}", chosen[o]["c"]) for o in OBJECTIVES]:
            acc, n = test(c)
            mean, worst = score(n)
            results[name] = {"c": dict(zip(tasks, c)), "test_acc": acc, "test_norm": n,
                             "test_mean_norm": mean, "test_worst_norm": worst,
                             "test_worst_task": min(n, key=n.get)}
        test_diff = max(abs(results["dc_rank"]["test_acc"][t] - rank_json["test_acc"][t]) for t in tasks)
        results["ct"] = {"c": {"gamma": ct_json["selected_coef"]}, "test_acc": ct_json["test_acc"],
                         "test_norm": ct_json["test_norm"], "test_mean_norm": ct_json["test_mean_norm"],
                         "test_worst_norm": ct_json["test_worst_norm"],
                         "test_worst_task": ct_json["test_worst_task"]}
        write_json(res_path, {
            "context": cid, "tasks": tasks, "dc_rank_alpha": alpha, "grid": list(GRID),
            "max_passes": args.max_passes, "val_evals": len(val.cache),
            "checks": {"max_val_acc_diff_vs_exp2_pp": val_diff, "max_test_acc_diff_vs_exp2_pp": test_diff,
                       "lane_checks": lm.checks},
            "val_baseline": {"mean_norm": base_mean, "worst_norm": base_worst},
            "val_chosen": chosen, "search_log": log, "test": results})
        print(f"[{cid}] done in {(time.time()-t0)/60:.1f} min; {len(val.cache)} validation merges; "
              f"baseline vs Experiment 2a: val diff {val_diff:.3f} pp, test diff {test_diff:.3f} pp", flush=True)
    summarize(out)


def verdict(d_mean, d_worst):
    if d_mean >= 1.0 and d_worst >= 3.0:
        return "HEADROOM: build the data-free predictor"
    if d_worst > 0 and d_mean < 0:
        return "FAIRNESS ONLY: worst up, mean down"
    if d_mean <= 0 and d_worst <= 0:
        return "NO HEADROOM: drop the method, write the analysis paper"
    return "SMALL GAIN: below the thresholds, discuss"


def summarize(out):
    files = sorted(out.glob("*/oracle_result.json"))
    if not files:
        raise SystemExit(f"no results in {out}")
    methods = ("dc_rank", "oracle_mean", "oracle_fair", "ct")
    lines = ["# Experiment 3: oracle per-task coefficients vs dc_rank and CT-Merging (TEST)", "",
             "Coefficients searched on validation; each vector evaluated once on test.", "",
             "| context | method | coefficients | mean norm | worst norm (task) | EuroSAT acc |",
             "|---|---|---|---:|---|---:|"]
    rows, agg = [], {m: {"mean": [], "worst": []} for m in methods}
    checks = []
    for p in files:
        d = json.loads(p.read_text())
        checks.append(d["checks"])
        for m in methods:
            r = d["test"][m]
            cs = ", ".join(f"{t[:6]}={v:g}" for t, v in r["c"].items())
            eu = r["test_acc"].get("eurosat")
            lines.append(f"| {d['context']} | {m} | {cs} | {r['test_mean_norm']:.2f} | "
                         f"{r['test_worst_norm']:.2f} ({r['test_worst_task']}) | {'—' if eu is None else f'{eu:.2f}'} |")
            agg[m]["mean"].append(r["test_mean_norm"])
            agg[m]["worst"].append(r["test_worst_norm"])
            rows.append({"context": d["context"], "method": m, **{f"c_{t}": v for t, v in r["c"].items()},
                         "test_mean_norm": r["test_mean_norm"], "test_worst_norm": r["test_worst_norm"],
                         "test_worst_task": r["test_worst_task"], "eurosat_acc": eu})
    n = len(files)
    avg = {m: (sum(agg[m]["mean"]) / n, sum(agg[m]["worst"]) / n) for m in methods}
    lines += ["", f"## Averages over {n} contexts (test)", "",
              "| method | mean norm | worst norm | Δmean vs dc_rank | Δworst vs dc_rank | verdict |",
              "|---|---:|---:|---:|---:|---|"]
    for m in methods:
        dm, dw = avg[m][0] - avg["dc_rank"][0], avg[m][1] - avg["dc_rank"][1]
        v = verdict(dm, dw) if m.startswith("oracle") else ""
        lines.append(f"| {m} | {avg[m][0]:.2f} | {avg[m][1]:.2f} | {dm:+.2f} | {dw:+.2f} | {v} |")
    lines += ["", "## Per-context deltas vs dc_rank (test)", "",
              "| context | oracle_mean Δmean / Δworst | oracle_fair Δmean / Δworst |", "|---|---:|---:|"]
    for p in files:
        d = json.loads(p.read_text())
        b = d["test"]["dc_rank"]
        cells = []
        for m in ("oracle_mean", "oracle_fair"):
            r = d["test"][m]
            cells.append(f"{r['test_mean_norm']-b['test_mean_norm']:+.2f} / {r['test_worst_norm']-b['test_worst_norm']:+.2f}")
        lines.append(f"| {d['context']} | " + " | ".join(cells) + " |")
    worst_val = max(c["max_val_acc_diff_vs_exp2_pp"] for c in checks)
    worst_test = max(c["max_test_acc_diff_vs_exp2_pp"] for c in checks)
    lines += ["", f"Checks: baseline reproduces Experiment 2a dc_rank within {worst_val:.3f} pp (val) and "
              f"{worst_test:.3f} pp (test); lane sum error ≤ "
              f"{max(c['lane_checks']['max_rel_sum_error'] for c in checks):.1e}."]
    (out / "oracle_summary.md").write_text("\n".join(lines) + "\n")
    keys = sorted({k for r in rows for k in r}, key=lambda k: (not k in ("context", "method"), k))
    with open(out / "oracle.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out/'oracle_summary.md'} and {out/'oracle.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    for name in ("--repo", "--model", "--ft-dir", "--heads", "--data"):
        r.add_argument(name, type=Path, required=True)
    r.add_argument("--exp2", type=Path, required=True, help="Experiment 2 output folder (contains compare/)")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--contexts", default="core", help="core (3 EuroSAT contexts), four (all 7), or comma list")
    r.add_argument("--max-passes", type=int, default=2)
    r.add_argument("--device", default="cuda")
    s = sub.add_parser("summarize")
    s.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out = args.out.resolve()
    if args.cmd == "summarize":
        summarize(args.out)
        return
    for name in ("repo", "model", "ft_dir", "heads", "data", "exp2"):
        setattr(args, name, getattr(args, name).resolve())
    run(args)


if __name__ == "__main__":
    main()
