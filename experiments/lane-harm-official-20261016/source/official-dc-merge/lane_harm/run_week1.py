#!/usr/bin/env python3
"""Experiment 2 on official DC-Merge: CT-Merging baseline, companion-lane decomposition, dose test.

  compare     DC-Merge (official), DC-Merge (rank variant) and CT-Merging on the 7 four-task
              contexts and the 8-task merge.  Coefficient selected on validation from a shared
              grid, then evaluated once on test.
  decompose   For the 3 EuroSAT+GTSRB+SVHN contexts: switch companion lanes off in every
              combination (exact decomposition of EuroSAT's harm) and turn them down
              gradually (dose test).  Validation split.  Run after ``compare``.

Shares asset loading, contexts and the official-equivalence check with run_official.py.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_official as ro  # noqa: E402

GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0)
METHODS = ("dc_official", "dc_rank", "ct")
FOUR = list(ro.CONTEXTS)[:7]
EUROSAT_CORE = list(ro.CONTEXTS)[:3]
DOSE_LEVELS = (0.25, 0.5, 0.75)
SELF_BOOST = (1.25, 1.5)
TARGET = "eurosat"
REFERENCE = HERE / "experiment1_reference.json"
REFERENCE_TOLERANCE_PP = 0.5  # allowance for GPU evaluation nondeterminism


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def norm(acc, refs):
    return {t: 100.0 * acc[t] / refs[t] for t in acc}


def check_experiment1_reference(row, reference, tolerance=REFERENCE_TOLERANCE_PP):
    """Stop on a failed Experiment 1 reproduction before sealing a comparison row."""
    if row["method"] != "dc_official":
        return
    cid = row["context"]
    if cid not in reference["validation_accuracy_by_context"]:
        raise AssertionError(f"missing Experiment 1 reference for {cid}")
    at_point_four = next((x for x in row["val_sweep"] if x["coef"] == 0.4), None)
    if at_point_four is None:
        raise AssertionError("the coefficient grid must contain 0.4")
    expected = reference["validation_accuracy_by_context"][cid]
    if set(at_point_four["acc"]) != set(expected):
        raise AssertionError(f"{cid}: validation tasks differ from Experiment 1")
    diffs = {t: abs(at_point_four["acc"][t] - expected[t]) for t in expected}
    if max(diffs.values()) > tolerance:
        raise AssertionError(f"{cid}: validation at alpha=0.4 differs from Experiment 1 by "
                             f"{diffs} pp (tolerance {tolerance} pp)")
    if cid == "all8":
        if row["selected_coef"] != reference["all8_selected_coefficient"]:
            raise AssertionError(f"all8: expected selected alpha "
                                 f"{reference['all8_selected_coefficient']}, got {row['selected_coef']}")
        delta = abs(row["test_mean_norm"] - reference["all8_test_mean_normalized"])
        if delta > tolerance:
            raise AssertionError(f"all8: normalized test result differs from Experiment 1 "
                                 f"by {delta:.3f} pp (tolerance {tolerance} pp)")


def check_comparison_row(row, cid, method, reference):
    if row.get("context") != cid or row.get("method") != method or row.get("grid") != list(GRID):
        raise AssertionError(f"{cid} / {method}: existing comparison has incompatible metadata")
    if len(row.get("val_sweep", [])) != len(GRID) or set(row.get("test_acc", {})) != set(row.get("tasks", [])):
        raise AssertionError(f"{cid} / {method}: existing comparison is incomplete")
    if [s["coef"] for s in row["val_sweep"]] != list(GRID):
        raise AssertionError(f"{cid} / {method}: comparison sweep differs from the prescribed grid")
    if any(set(s["acc"]) != set(row["tasks"]) for s in row["val_sweep"]):
        raise AssertionError(f"{cid} / {method}: comparison has incomplete validation scores")
    check_experiment1_reference(row, reference)


# ----------------------------------------------------------------------------
# merges
# ----------------------------------------------------------------------------
def build_merge(rt, method, tasks):
    import ct_merging
    torch = rt["torch"]
    grouped = ro.group(rt["deltas"], tasks)
    info = {}
    with torch.no_grad():
        if method == "dc_official":
            merged = rt["mf"].dc_merge(grouped, smoothing_strategy="linear", rho=5.0)
        elif method == "dc_rank":
            merged = OrderedDict(sorted((k, rt["backend"]._dc_merge_matrix_group(v, **ro.dc_kwargs("rank")))
                                        for k, v in grouped.items()))
        elif method == "ct":
            merged, diag = ct_merging.ct_merge(grouped)
            info["max_orthonormality_error"] = diag["max_orthonormality_error"]
            if diag["max_orthonormality_error"] > 1e-4:
                raise AssertionError(f"CT-Merging bases not orthonormal: {diag['max_orthonormality_error']:g}")
        else:
            raise ValueError(method)
    return merged, info


# ----------------------------------------------------------------------------
# compare
# ----------------------------------------------------------------------------
def cmd_compare(args):
    reference = json.loads(REFERENCE.read_text())
    rt = ro.load_runtime(args)
    out = args.out / "compare"
    ctxs = FOUR + (["all8"] if not args.skip_all8 else [])
    for cid in ctxs:
        tasks = list(ro.CONTEXTS[cid])
        for method in args.methods:
            if method == "dc_rank" and cid == "all8":
                continue                     # identical to dc_official at 8 tasks (k = 16 both)
            path = out / f"{cid}__{method}.json"
            if path.exists():
                check_comparison_row(json.loads(path.read_text()), cid, method, reference)
                print(f"[{cid} | {method}] done, skipping", flush=True)
                continue
            t0 = time.time()
            merged, info = build_merge(rt, method, tasks)
            val_eval = ro.make_evaluate(rt, tasks, args.device, "val")
            sweep, best = [], (None, -1.0)
            for c in GRID:
                acc = val_eval(merged, c)
                nm = norm(acc, rt["refs"]["val"])
                mean_n = sum(nm.values()) / len(nm)
                sweep.append({"coef": c, "acc": acc, "mean_norm": mean_n, "worst_norm": min(nm.values())})
                if mean_n > best[1]:
                    best = (c, mean_n)
            test_acc = ro.make_evaluate(rt, tasks, args.device, "test")(merged, best[0])
            tn = norm(test_acc, rt["refs"]["test"])
            worst_task = min(tn, key=tn.get)
            row = {"context": cid, "method": method, "tasks": tasks, "grid": list(GRID),
                              "selected_coef": best[0], "val_best_mean_norm": best[1], "val_sweep": sweep,
                              "test_acc": test_acc, "test_norm": tn,
                              "test_mean_acc": sum(test_acc.values()) / len(tasks),
                              "test_mean_norm": sum(tn.values()) / len(tasks),
                              "test_worst_norm": tn[worst_task], "test_worst_task": worst_task, **info}
            check_comparison_row(row, cid, method, reference)
            write_json(path, row)
            print(f"[{cid} | {method}] coef={best[0]}  test mean norm={sum(tn.values())/len(tasks):.2f}  "
                  f"worst={worst_task} {tn[worst_task]:.2f}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    summarize_compare(out)


def summarize_compare(out):
    rows = [json.loads(p.read_text()) for p in sorted(out.glob("*__*.json"))]
    if not rows:
        raise SystemExit(f"no results in {out}")
    with open(out / "compare.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["context", "method", "selected_coef", "test_mean_acc", "test_mean_norm",
                    "test_worst_norm", "test_worst_task", "eurosat_test_acc"] + list(ro.TASKS))
        for r in rows:
            w.writerow([r["context"], r["method"], r["selected_coef"], f"{r['test_mean_acc']:.2f}",
                        f"{r['test_mean_norm']:.2f}", f"{r['test_worst_norm']:.2f}", r["test_worst_task"],
                        f"{r['test_acc'].get('eurosat', float('nan')):.2f}"]
                       + [f"{r['test_acc'][t]:.2f}" if t in r["test_acc"] else "" for t in ro.TASKS])
    by = {(r["context"], r["method"]): r for r in rows}
    lines = ["# Experiment 2a: DC-Merge vs CT-Merging (test split, coefficient selected on validation)", "",
             "| context | method | coef | mean acc | mean norm | worst norm (task) | EuroSAT acc |",
             "|---|---|---:|---:|---:|---|---:|"]
    for cid in ro.CONTEXTS:
        for m in METHODS:
            r = by.get((cid, m))
            if r:
                eu = r["test_acc"].get("eurosat")
                lines.append(f"| {cid} | {m} | {r['selected_coef']} | {r['test_mean_acc']:.2f} | "
                             f"{r['test_mean_norm']:.2f} | {r['test_worst_norm']:.2f} ({r['test_worst_task']}) | "
                             f"{'—' if eu is None else f'{eu:.2f}'} |")
    lines += ["", "## Averages over the 7 four-task contexts", "",
              "| method | mean norm | mean worst norm | mean EuroSAT acc | contexts |", "|---|---:|---:|---:|---:|"]
    for m in METHODS:
        rs = [by[(c, m)] for c in FOUR if (c, m) in by]
        if rs:
            lines.append(f"| {m} | {sum(r['test_mean_norm'] for r in rs)/len(rs):.2f} | "
                         f"{sum(r['test_worst_norm'] for r in rs)/len(rs):.2f} | "
                         f"{sum(r['test_acc']['eurosat'] for r in rs)/len(rs):.2f} | {len(rs)} |")
    (out / "compare_summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out/'compare.csv'} and {out/'compare_summary.md'}")


# ----------------------------------------------------------------------------
# decompose + dose
# ----------------------------------------------------------------------------
def configurations(tasks):
    comps = [t for t in tasks if t != TARGET]
    confs = []
    for n in range(len(comps) + 1):                       # all subsets of companions switched ON
        for on in itertools.combinations(comps, n):
            confs.append(("subset", "+".join(on) or "none", {c: 0.0 for c in comps if c not in on}))
    for c in comps:                                       # dose: one companion turned down
        for lam in DOSE_LEVELS:
            confs.append(("dose", f"{c}={lam}", {c: lam}))
    if {"gtsrb", "svhn"} <= set(comps):                   # dose: both suspects together
        for lam in DOSE_LEVELS:
            confs.append(("dose", f"gtsrb+svhn={lam}", {"gtsrb": lam, "svhn": lam}))
    for lam in SELF_BOOST:                                # victim's own lane turned up
        confs.append(("self", f"{TARGET}={lam}", {TARGET: lam}))
    return comps, confs


def mobius(f, comps):
    """h(T) = harm attributable to the set T of companions (exact; sums to f(none) - f(all))."""
    h = {}
    for n in range(1, len(comps) + 1):
        for T in itertools.combinations(comps, n):
            m = 0.0
            for k in range(len(T) + 1):
                for S in itertools.combinations(T, k):
                    m += (-1) ** (len(T) - len(S)) * f["+".join(sorted(S, key=comps.index)) or "none"]
            h["+".join(T)] = -m
    return h


def check_decomposition_row(row, reference, comparison=None, tolerance=REFERENCE_TOLERANCE_PP):
    """Validate a sealed or newly computed row, including its full-context baseline."""
    cid, op, alpha = row["context"], row["operator"], row["alpha"]
    comps = [t for t in row["tasks"] if t != TARGET]
    if cid not in EUROSAT_CORE or op not in ("rank", "official") or row["companions"] != comps:
        raise AssertionError("decomposition metadata differs from the prescribed context")
    checks = row["lane_checks"]
    if checks["max_rel_sum_error"] > 1e-5:
        raise AssertionError(f"{cid}: lane terms do not sum to the merge")
    if op == "official" and checks["max_abs_diff_vs_official"] != 0.0:
        raise AssertionError(f"{cid}: official merge differs from dc_merge")
    _comps, expected_configs = configurations(row["tasks"])
    keys = [(r["kind"], r["name"]) for r in row["results"]]
    expected_keys = [(kind, name) for kind, name, _ in expected_configs]
    if (keys != expected_keys or any(set(r["acc"]) != set(row["tasks"]) for r in row["results"])
            or any(r.get("scales") != scales for r, (_, _, scales) in zip(row["results"], expected_configs))):
        raise AssertionError(f"{cid}: expected every lane configuration exactly once with all task scores")
    f = {r["name"]: r["acc"][TARGET] for r in row["results"] if r["kind"] == "subset"}
    expected_subsets = {"+".join(on) or "none" for n in range(len(comps) + 1)
                        for on in itertools.combinations(comps, n)}
    if set(f) != expected_subsets or len(f) != len(expected_subsets):
        raise AssertionError(f"{cid}: expected every companion subset exactly once")
    full = next(r["acc"] for r in row["results"] if r["kind"] == "subset"
                and r["name"] == "+".join(comps))
    if set(full) != set(row["tasks"]):
        raise AssertionError(f"{cid}: missing full-context task accuracies")
    frame = row["solo_acc"] - f["none"]
    residual = (row["solo_acc"] - f["+".join(comps)]) - frame - sum(mobius(f, comps).values())
    if abs(residual) > 1e-6:
        raise AssertionError(f"{cid}: decomposition has residual {residual} pp")
    if alpha == reference["validation_coefficient"]:
        key = "validation_accuracy_by_context" if op == "official" else "rank_validation_accuracy_by_context"
        expected = reference[key][cid]
    elif comparison is not None and alpha == comparison["selected_coef"]:
        selected = next((s for s in comparison["val_sweep"] if s["coef"] == alpha), None)
        if selected is None:
            raise AssertionError(f"{cid}: selected coefficient absent from comparison")
        expected = selected["acc"]
    else:
        expected = None
    if expected is not None:
        if set(expected) != set(full):
            raise AssertionError(f"{cid}: baseline task set differs from the reference")
        diffs = {t: abs(full[t] - expected[t]) for t in full}
        if max(diffs.values()) > tolerance:
            raise AssertionError(f"{cid}: full-context validation differs by {diffs} pp "
                                 f"(tolerance {tolerance} pp)")


def cmd_decompose(args):
    import lanes
    reference = json.loads(REFERENCE.read_text())
    rt = ro.load_runtime(args)
    torch = rt["torch"]
    out = args.out / "decompose"
    cmp_dir = args.out / "compare"
    for op in args.operators:
        for cid in EUROSAT_CORE:
            tasks = list(ro.CONTEXTS[cid])
            alphas = []
            for a in args.alphas:
                if a == "selected":
                    p = cmp_dir / f"{cid}__dc_{op}.json"
                    if not p.exists():
                        raise SystemExit(f"{p} missing: run `compare` first or pass numeric --alphas")
                    comparison = json.loads(p.read_text())
                    check_comparison_row(comparison, cid, f"dc_{op}", reference)
                    alphas.append(("selected", float(comparison["selected_coef"])))
                else:
                    alphas.append((a, float(a)))
            alphas = list({alpha: (tag, alpha) for tag, alpha in reversed(alphas)}.values())[::-1]
            for tag, alpha in alphas:
                path = out / f"{cid}__{op}__alpha{alpha:g}.json"
                cmp_path = cmp_dir / f"{cid}__dc_{op}.json"
                comparison = json.loads(cmp_path.read_text()) if cmp_path.exists() else None
                if comparison is not None:
                    check_comparison_row(comparison, cid, f"dc_{op}", reference)
                if path.exists():
                    existing = json.loads(path.read_text())
                    if existing["context"] != cid or existing["operator"] != op or existing["alpha"] != alpha:
                        raise AssertionError(f"{path}: incompatible existing run")
                    check_decomposition_row(existing, reference, comparison)
                    print(f"[{cid} | {op} | alpha={alpha:g}] done, skipping", flush=True)
                    continue
                t0 = time.time()
                with torch.no_grad():
                    lm = lanes.LaneMerge(ro.group(rt["deltas"], tasks), rt["backend"], tasks, ro.dc_kwargs(op),
                                         official_dc_merge=rt["mf"].dc_merge if op == "official" else None)
                evaluate = ro.make_evaluate(rt, tasks, args.device, "val")
                comps, confs = configurations(tasks)
                results = []
                for kind, name, scales in confs:
                    acc = evaluate(lm.compose(scales), alpha)
                    results.append({"kind": kind, "name": name, "scales": scales, "acc": acc})
                    print(f"  {kind:6s} {name:28s} EuroSAT {acc[TARGET]:.2f}", flush=True)
                with torch.no_grad():
                    solo_merged = OrderedDict((k, rt["backend"]._dc_merge_matrix_group(v, **ro.dc_kwargs(op)))
                                              for k, v in ro.group(rt["deltas"], [TARGET]).items())
                solo = ro.make_evaluate(rt, [TARGET], args.device, "val")(solo_merged, alpha)[TARGET]
                row = {"context": cid, "operator": op, "alpha": alpha, "alpha_source": tag,
                                  "tasks": tasks, "companions": comps, "target": TARGET,
                                  "solo_acc": solo, "standalone_acc": rt["refs"]["val"][TARGET],
                                  "lane_checks": lm.checks, "results": results}
                check_decomposition_row(row, reference, comparison)
                write_json(path, row)
                print(f"[{cid} | {op} | alpha={alpha:g}] done in {(time.time()-t0)/60:.1f} min", flush=True)
    summarize_decompose(out)


def summarize_decompose(out):
    files = sorted(out.glob("*__*__alpha*.json"))
    if not files:
        raise SystemExit(f"no results in {out}")
    lines = ["# Experiment 2b/2c: where EuroSAT's harm comes from, and the dose test (validation)", ""]
    csv_rows = []
    for p in files:
        d = json.loads(p.read_text())
        comps, res = d["companions"], d["results"]
        f = {r["name"]: r["acc"][TARGET] for r in res if r["kind"] == "subset"}
        full = f["+".join(comps)]
        h = mobius(f, comps)
        frame = d["solo_acc"] - f["none"]
        harm_solo = d["solo_acc"] - full
        resid = harm_solo - frame - sum(h.values())
        if abs(resid) > 1e-6:
            raise AssertionError(f"decomposition does not sum: {resid}")
        lines += [f"## {d['context']}: operator **{d['operator']}**, α = {d['alpha']:g} ({d['alpha_source']})", "",
                  f"EuroSAT accuracy: standalone {d['standalone_acc']:.2f}, merged alone {d['solo_acc']:.2f}, "
                  f"own lane only (shared frame) {f['none']:.2f}, full context {full:.2f}.", "",
                  "**Decomposition of harm vs merged-alone** (pp; terms add up exactly):", "",
                  "| part | pp |", "|---|---:|", f"| frame cost (merged alone → own lane in shared frame) | {frame:.2f} |"]
        for T, v in sorted(h.items(), key=lambda kv: (kv[0].count("+"), kv[0])):
            label = f"{T} alone" if "+" not in T else f"interaction {T}"
            lines.append(f"| {label} | {v:.2f} |")
        lines += [f"| **total harm vs merged alone** | **{harm_solo:.2f}** |", "",
                  "**Dose test** (change vs full context, pp):", "",
                  "| setting | EuroSAT | " + " | ".join(comps) + " | mean of 4 |", "|---|---:|" + "---:|" * len(comps) + "---:|"]
        base = next(r["acc"] for r in res if r["kind"] == "subset" and r["name"] == "+".join(comps))
        for r in res:
            if r["kind"] in ("dose", "self") or (r["kind"] == "subset" and r["name"].count("+") == len(comps) - 2):
                dl = {t: r["acc"][t] - base[t] for t in d["tasks"]}
                mean = sum(dl.values()) / len(dl)
                lines.append(f"| {r['name'] if r['kind'] != 'subset' else 'off: ' + ','.join(c for c in comps if c not in r['name'].split('+'))} | "
                             f"{dl[TARGET]:+.2f} | " + " | ".join(f"{dl[c]:+.2f}" for c in comps) + f" | {mean:+.2f} |")
                csv_rows.append({"context": d["context"], "operator": d["operator"], "alpha": d["alpha"],
                                 "setting": r["name"], "kind": r["kind"], "eurosat_delta": dl[TARGET],
                                 "mean_delta": mean, **{f"{c}_delta": dl[c] for c in comps}})
        lines.append("")
        csv_rows.append({"context": d["context"], "operator": d["operator"], "alpha": d["alpha"],
                         "setting": "DECOMPOSITION", "kind": "decomposition", "frame_cost": frame,
                         "harm_vs_solo": harm_solo, **{f"h[{T}]": v for T, v in h.items()}})
    keys = sorted({k for r in csv_rows for k in r}, key=lambda k: (k not in ("context", "operator", "alpha", "setting", "kind"), k))
    with open(out / "decompose.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(csv_rows)
    (out / "decompose_summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out/'decompose.csv'} and {out/'decompose_summary.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--repo", type=Path, required=True)
        p.add_argument("--model", type=Path, required=True)
        p.add_argument("--ft-dir", type=Path, required=True)
        p.add_argument("--heads", type=Path, required=True)
        p.add_argument("--data", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--device", default="cuda")

    c = sub.add_parser("compare")
    common(c)
    c.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    c.add_argument("--skip-all8", action="store_true")
    d = sub.add_parser("decompose")
    common(d)
    d.add_argument("--operators", nargs="+", default=["rank", "official"], choices=["rank", "official"])
    d.add_argument("--alphas", nargs="+", default=["selected", "0.4"],
                   help="'selected' (from compare) and/or numbers")
    s = sub.add_parser("summarize")
    s.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out = args.out.resolve()
    if args.cmd == "summarize":
        if (args.out / "compare").exists():
            summarize_compare(args.out / "compare")
        if (args.out / "decompose").exists():
            summarize_decompose(args.out / "decompose")
        return
    args.repo = args.repo.resolve()
    for name in ("model", "ft_dir", "heads", "data"):
        setattr(args, name, getattr(args, name).resolve())
    (cmd_compare if args.cmd == "compare" else cmd_decompose)(args)


if __name__ == "__main__":
    main()
