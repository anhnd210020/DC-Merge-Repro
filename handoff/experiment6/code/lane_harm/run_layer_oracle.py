#!/usr/bin/env python3
"""Experiment 6b: does giving each task one coefficient PER LAYER GROUP beat the scalar oracle?

Generalised merged model, on the same dc_rank lanes as Experiments 3-5:

    W(c) = W0 + sum_{i,g} c_{i,g} * L_i|g ,

where L_i|g is task i's lane restricted to the modules of layer group g.  With
c_{i,g} = c_i for every g this is exactly Experiment 3's W(c); with every c_{i,g} = alpha
it is dc_rank.

Lane modules.  The official config's LoRA targets only the attention projections
(q_proj, k_proj, v_proj, out_proj), so each task has 12 blocks x 4 = 48 lane modules and
NO MLP lanes.  Groupings:
  depth  (primary)  early = blocks 0-3, mid = 4-7, late = 8-11
  qk_vo  (optional) {q_proj, k_proj} vs {v_proj, out_proj}; a substitute for the guide's
                    "attention vs MLP", which is impossible without MLP lanes
  attn_mlp          implemented, but exits with an explanation when no MLP lane exists
Modules outside the 12 transformer blocks (none are expected) receive the context's
scalar alpha for every task; their count is printed.

Subcommands
  mechanism  6b-1, core contexts, validation: where in depth do SVHN/GTSRB create EuroSAT's harm?
  oracle     6b-2: coordinate search over the (task, group) coefficients on validation,
             each chosen vector evaluated ONCE on test
  summarize  rebuild the summaries

Every evaluation is cached on disk; rerunning the same command resumes.  Each
subcommand first runs the must-hold checks for the context and aborts on failure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_official as ro  # noqa: E402

GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)          # same as Experiment 3
FOUR = list(ro.CONTEXTS)[:7]
CORE = list(ro.CONTEXTS)[:3]
OBJECTIVES = ("mean", "fair")
CHECK_TOL_PP = 0.05
LANE_SUM_TOL = 1e-5
N_BLOCKS = 12

BLOCK_RE = re.compile(r"encoder\.layers\.(\d+)\.")
PROJ_RE = re.compile(r"\.(q_proj|k_proj|v_proj|out_proj|fc1|fc2)(?:\.|$)")

GROUPINGS = {
    "depth": ("early", "mid", "late"),
    "qk_vo": ("qk", "vo"),
    "attn_mlp": ("attn", "mlp"),
}


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def fkey(vals):
    return ",".join(f"{x:g}" for x in vals)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_note(stage, text):
    print(f"[split] {stage}: {text}", flush=True)


def gdir(out, grouping):
    """Primary (depth) results live directly under OUT, as the guide asks."""
    return out if grouping == "depth" else out / f"grouping_{grouping}"


def contexts_of(spec):
    ids = CORE if spec == "core" else (FOUR if spec == "four" else spec.split(","))
    bad = [c for c in ids if c not in ro.CONTEXTS or c == "all8"]
    if bad:
        raise SystemExit(f"unknown or unsupported contexts {bad}")
    return ids


# ----------------------------------------------------------------------------
# module -> block -> group
# ----------------------------------------------------------------------------
def module_groups(keys, grouping):
    """Return ({module_key: group or None}, table rows).  None = outside the blocks."""
    groups = GROUPINGS[grouping]
    mapping, rows = OrderedDict(), []
    blocks = set()
    for k in keys:
        mb, mp = BLOCK_RE.search(k), PROJ_RE.search(k)
        block = int(mb.group(1)) if mb else None
        proj = mp.group(1) if mp else None
        if block is None:
            g = None
        elif grouping == "depth":
            g = groups[min(block // 4, 2)]
        elif grouping == "qk_vo":
            g = "qk" if proj in ("q_proj", "k_proj") else ("vo" if proj in ("v_proj", "out_proj") else None)
        else:
            g = "mlp" if proj in ("fc1", "fc2") else ("attn" if proj else None)
        if block is not None:
            blocks.add(block)
        mapping[k] = g
        rows.append({"module": k, "block": block, "proj": proj, "group": g})
    if blocks != set(range(N_BLOCKS)):
        raise SystemExit(f"expected transformer blocks 0..{N_BLOCKS-1}, found {sorted(blocks)}")
    used = {g for g in mapping.values() if g is not None}
    missing = [g for g in groups if g not in used]
    if missing:
        raise SystemExit(f"grouping '{grouping}' has empty groups {missing}: the LoRA adapters have no lanes "
                         f"there (target modules found: {sorted({r['proj'] for r in rows if r['proj']})}). "
                         f"Use --grouping depth or qk_vo.")
    return mapping, rows


def print_mapping(rows, grouping):
    print(f"module -> block -> group mapping ({grouping}); {len(rows)} lane modules:", flush=True)
    print(f"  {'block':>5} {'proj':9} {'group':6} module", flush=True)
    for r in sorted(rows, key=lambda r: (r["block"] if r["block"] is not None else 99, r["proj"] or "", r["module"])):
        b = "-" if r["block"] is None else str(r["block"])
        print(f"  {b:>5} {str(r['proj']):9} {str(r['group']):6} {r['module']}", flush=True)
    counts = {}
    for r in rows:
        counts[r["group"]] = counts.get(r["group"], 0) + 1
    print("  modules per group: " + ", ".join(f"{g}={n}" for g, n in counts.items()), flush=True)
    nonblock = counts.get(None, 0)
    print(f"  non-block modules: {nonblock} (choice: they receive the context's scalar alpha for every task)",
          flush=True)


# ----------------------------------------------------------------------------
# per-(task, group) composition, via a subclass (lanes.py is not edited)
# ----------------------------------------------------------------------------
def make_group_lane_merge():
    import lanes

    class GroupLaneMerge(lanes.LaneMerge):
        """LaneMerge whose lanes can be scaled per (task, module group).

        Per module, the coefficient vector over tasks is lam = [c_{t, g(module)}]_t.
        As in LaneMerge.compose, a module whose coefficients are all equal uses the
        direct path lam[0] * merged (the same float path as dc_rank); otherwise the
        lane terms are scaled and summed in task order.  With c_{i,g} = c_i for all g
        this is operation-for-operation identical to LaneMerge.compose.
        """

        def set_groups(self, mapping, groups, default):
            if not hasattr(self, "merged"):
                raise RuntimeError("lanes.py lacks the uniform-coefficient fix (self.merged); "
                                   "use the Experiment 3 lanes.py")
            if set(mapping) != set(self.lanes):
                raise RuntimeError("module keys of the grouping and the lanes differ")
            self.module_group = mapping
            self.groups = tuple(groups)
            self.default = float(default)

        def compose_groups(self, coef):
            """coef: {task: {group: c}}."""
            torch = __import__("torch")
            out = OrderedDict()
            for key, terms in self.lanes.items():
                g = self.module_group[key]
                lam = [float(coef[t][g]) if g is not None else self.default for t in self.tasks]
                if all(v == lam[0] for v in lam):
                    out[key] = lam[0] * self.merged[key]
                    continue
                acc = torch.zeros_like(terms[0])
                for l, term in zip(lam, terms):
                    if l != 0.0:
                        acc = acc + l * term
                out[key] = acc
            return OrderedDict(sorted(out.items()))

    return GroupLaneMerge


class Space:
    """Flat coordinate order: task-major, then group (e.g. eurosat@early, eurosat@mid, ...)."""

    def __init__(self, tasks, groups):
        self.tasks, self.groups = list(tasks), list(groups)
        self.coords = [(t, g) for t in self.tasks for g in self.groups]

    def flat(self, coef):
        return [float(coef[t][g]) for t, g in self.coords]

    def nested(self, vec):
        out = {t: {} for t in self.tasks}
        for (t, g), v in zip(self.coords, vec):
            out[t][g] = float(v)
        return out

    def uniform(self, value):
        return [float(value)] * len(self.coords)

    def from_task_vector(self, c):
        return [float(c[self.tasks.index(t)]) for t, _ in self.coords]

    def table(self, vec):
        lines = ["| task | " + " | ".join(self.groups) + " |", "|---|" + "---:|" * len(self.groups)]
        n = self.nested(vec)
        for t in self.tasks:
            lines.append(f"| {t} | " + " | ".join(f"{n[t][g]:g}" for g in self.groups) + " |")
        return lines


class Evaluator:
    """Accuracy of W0 + sum c_{i,g} L_i|g on a split, cached on disk by the flat vector."""

    def __init__(self, rt, lm, space, device, split, refs, cache_path):
        self.lm, self.space, self.refs = lm, space, refs
        self.fn = ro.make_evaluate(rt, space.tasks, device, split)
        self.cache_path = cache_path
        self.cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        self.new_evals = 0

    def __call__(self, vec):
        k = fkey(vec)
        if k not in self.cache:
            merged = self.lm.compose_groups(self.space.nested(vec))
            self.cache[k] = self.fn(merged, 1.0)          # coefficients are inside the lanes
            self.new_evals += 1
            write_json(self.cache_path, self.cache)
        acc = self.cache[k]
        n = {t: 100.0 * acc[t] / self.refs[t] for t in self.space.tasks}
        return acc, n


def score(n):
    return sum(n.values()) / len(n), min(n.values())


# ----------------------------------------------------------------------------
# context setup and must-hold checks
# ----------------------------------------------------------------------------
_MAPPING_PRINTED = set()


def setup_context(rt, args, cid):
    GroupLaneMerge = make_group_lane_merge()
    tasks = list(ro.CONTEXTS[cid])
    exp3 = json.loads((args.exp3 / cid / "oracle_result.json").read_text())
    alpha = float(exp3["dc_rank_alpha"])
    grouped = ro.group(rt["deltas"], tasks)
    mapping, rows = module_groups(list(grouped), args.grouping)
    if args.grouping not in _MAPPING_PRINTED:
        print_mapping(rows, args.grouping)
        _MAPPING_PRINTED.add(args.grouping)
    with rt["torch"].no_grad():
        lm = GroupLaneMerge(grouped, rt["backend"], tasks, ro.dc_kwargs("rank"))
    lm.set_groups(mapping, GROUPINGS[args.grouping], alpha)
    space = Space(tasks, GROUPINGS[args.grouping])
    d = gdir(args.out, args.grouping) / cid
    val = Evaluator(rt, lm, space, args.device, "val", rt["refs"]["val"], d / "layer_val_cache.json")
    test = Evaluator(rt, lm, space, args.device, "test", rt["refs"]["test"], d / "layer_test_cache.json")
    return tasks, alpha, exp3, lm, space, val, test, rows


def run_checks(args, cid, tasks, alpha, exp3, lm, space, val, test):
    """Abort unless (1) uniform alpha reproduces dc_rank val+test, (2) constant-over-group
    oracle_fair reproduces Exp 3 validation, (3) lane sums reconstruct the merge."""
    e3v = json.loads((args.exp3 / cid / "val_cache.json").read_text())
    e3t = json.loads((args.exp3 / cid / "test_cache.json").read_text())
    base_task = [alpha] * len(tasks)
    fair_task = [float(x) for x in exp3["val_chosen"]["fair"]["c"]]
    rec = {"context": cid, "tolerance_pp": CHECK_TOL_PP, "grouping": args.grouping}

    def diff(acc, ref):
        per = {t: acc[t] - ref[t] for t in tasks}
        return max(abs(v) for v in per.values()), per

    acc_v, _ = val(space.uniform(alpha))
    acc_t, _ = test(space.uniform(alpha))
    rec["uniform_alpha_val"] = diff(acc_v, e3v[fkey(base_task)])
    rec["uniform_alpha_test"] = diff(acc_t, e3t[fkey(base_task)])
    acc_f, _ = val(space.from_task_vector(fair_task))
    if fkey(fair_task) not in e3v:
        raise SystemExit(f"[{cid}] Exp 3 val cache lacks the oracle_fair vector {fair_task}")
    rec["oracle_fair_val"] = diff(acc_f, e3v[fkey(fair_task)])
    rec["lane_checks"] = lm.checks
    ok = (rec["uniform_alpha_val"][0] <= CHECK_TOL_PP and rec["uniform_alpha_test"][0] <= CHECK_TOL_PP
          and rec["oracle_fair_val"][0] <= CHECK_TOL_PP and lm.checks["max_rel_sum_error"] < LANE_SUM_TOL)
    rec["passed"] = ok
    write_json(gdir(args.out, args.grouping) / cid / "checks.json", rec)
    print(f"[{cid}] checks: uniform alpha={alpha:g} val diff {rec['uniform_alpha_val'][0]:.3f} pp, "
          f"test diff {rec['uniform_alpha_test'][0]:.3f} pp (vs Exp 3 dc_rank); constant-over-group "
          f"oracle_fair val diff {rec['oracle_fair_val'][0]:.3f} pp (vs Exp 3); lane sum error "
          f"{lm.checks['max_rel_sum_error']:.2e} -> {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        raise SystemExit(f"[{cid}] must-hold check failed (tolerance {CHECK_TOL_PP} pp / {LANE_SUM_TOL}); "
                         f"see {gdir(args.out, args.grouping) / cid / 'checks.json'}. Stop and report.")
    return rec


# ----------------------------------------------------------------------------
# 6b-1 mechanism
# ----------------------------------------------------------------------------
NEIGHBOUR_SETS = (("svhn",), ("gtsrb",), ("svhn", "gtsrb"))
VICTIM = "eurosat"


def mechanism_configs(space, alpha, boost):
    """Named coefficient vectors.  All other lanes stay at alpha."""
    G = space.groups

    def vec(**over):
        coef = {t: {g: alpha for g in G} for t in space.tasks}
        for t, gv in over.items():
            for g, v in gv.items():
                coef[t][g] = v
        return space.flat(coef)

    def lanes_of(names, on_groups, on_value):
        return {t: {g: (on_value if g in on_groups else 0.0) for g in G} for t in names}

    cfg = OrderedDict()
    cfg["baseline"] = vec()
    for N in NEIGHBOUR_SETS:
        tag = "+".join(N)
        cfg[f"off[{tag}]"] = vec(**lanes_of(N, (), alpha))
        for g in G:
            cfg[f"only[{tag}]@{g}"] = vec(**lanes_of(N, (g,), alpha))
            cfg[f"off[{tag}]@{g}"] = vec(**{t: {h: (0.0 if h == g else alpha) for h in G} for t in N})
    # interaction states, reference = both neighbours fully off
    for g in G:
        cfg[f"only[svhn]@{g}|gtsrb_off"] = vec(**lanes_of(("svhn",), (g,), alpha), **lanes_of(("gtsrb",), (), alpha))
        cfg[f"only[gtsrb]@{g}|svhn_off"] = vec(**lanes_of(("gtsrb",), (g,), alpha), **lanes_of(("svhn",), (), alpha))
    # victim's own lane boosted in one group at a time, and everywhere
    b = round(boost * alpha, 10)
    for g in G:
        cfg[f"boost[{VICTIM}]@{g}"] = vec(**{VICTIM: {g: b}})
    cfg[f"boost[{VICTIM}]@all"] = vec(**{VICTIM: {g: b for g in G}})
    return cfg


def mechanism_analysis(acc, G):
    """EuroSAT harm localisation from the named configurations' validation accuracies."""
    E = lambda name: acc[name][VICTIM]
    base = E("baseline")
    out = {"eurosat_baseline": base, "neighbours": {}, "interaction": {}}
    for N in NEIGHBOUR_SETS:
        tag = "+".join(N)
        off = E(f"off[{tag}]")
        harm = off - base                       # EuroSAT accuracy lost to these lanes (pp)
        per = {}
        for g in G:
            only = E(f"only[{tag}]@{g}")
            offg = E(f"off[{tag}]@{g}")
            created = off - only                # harm when the neighbour exists only in g
            recovered = offg - base             # harm removed by switching the neighbour off only in g
            per[g] = {"eurosat_only_g": only, "eurosat_off_g": offg,
                      "harm_created_only_g": created, "harm_recovered_off_g": recovered,
                      "share_only_g": created / harm if harm else float("nan"),
                      "share_off_g": recovered / harm if harm else float("nan")}
        out["neighbours"][tag] = {"eurosat_neighbour_off": off, "harm_total": harm, "per_group": per,
                                  "sum_share_only": sum(p["share_only_g"] for p in per.values()),
                                  "sum_share_off": sum(p["share_off_g"] for p in per.values())}
    R0 = E("off[svhn+gtsrb]")
    I_all = (R0 - base) - (R0 - E("off[gtsrb]")) - (R0 - E("off[svhn]"))
    out["interaction"]["all_groups"] = {"reference_both_off": R0, "interaction_pp": I_all}
    for g in G:
        hs = R0 - E(f"only[svhn]@{g}|gtsrb_off")
        hg = R0 - E(f"only[gtsrb]@{g}|svhn_off")
        hsg = R0 - E(f"only[svhn+gtsrb]@{g}")
        out["interaction"][g] = {"harm_svhn_only_g": hs, "harm_gtsrb_only_g": hg, "harm_both_only_g": hsg,
                                 "interaction_pp": hsg - hs - hg}
    return out


def write_run_metadata(args, cmd):
    write_json(gdir(args.out, args.grouping) / f"run_metadata_{cmd}.json", {
        "experiment": "6b", "subcommand": cmd, "grouping": args.grouping, "contexts": args.contexts,
        "grid": list(GRID), "check_tolerance_pp": CHECK_TOL_PP, "lane_sum_tolerance": LANE_SUM_TOL,
        "boost": getattr(args, "boost", None), "max_passes": getattr(args, "max_passes", None),
        "argv": sys.argv, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_sha256": {f: sha256(HERE / f) for f in ("run_layer_oracle.py", "lanes.py", "run_oracle.py",
                                                      "run_official.py", "dc_trace.py")}})


def cmd_mechanism(args):
    write_run_metadata(args, "mechanism")
    rt = ro.load_runtime(args)
    split_note("mechanism", "every accuracy is VALIDATION (labels used to measure, nothing is selected); "
                            "the uniform-alpha check additionally evaluates TEST once")
    contexts = contexts_of(args.contexts)
    for cid in contexts:
        if not {VICTIM, "svhn", "gtsrb"} <= set(ro.CONTEXTS[cid]):
            print(f"[{cid}] skipped: mechanism needs eurosat, svhn and gtsrb", flush=True)
            continue
        res_path = gdir(args.out, args.grouping) / cid / "mechanism_result.json"
        t0 = time.time()
        tasks, alpha, exp3, lm, space, val, test, _ = setup_context(rt, args, cid)
        checks = run_checks(args, cid, tasks, alpha, exp3, lm, space, val, test)
        cfg = mechanism_configs(space, alpha, args.boost)
        acc, norm = {}, {}
        for i, (name, vec) in enumerate(cfg.items()):
            a, n = val(vec)
            acc[name], norm[name] = a, n
            print(f"  [{cid}] {i+1:2d}/{len(cfg)} {name:28s} " +
                  " ".join(f"{t}={a[t]:.2f}" for t in tasks), flush=True)
        analysis = mechanism_analysis(acc, space.groups)
        write_json(res_path, {"context": cid, "tasks": tasks, "alpha": alpha, "grouping": args.grouping,
                              "groups": space.groups, "boost_factor": args.boost, "split": "val",
                              "configs": {k: dict(zip([f"{t}@{g}" for t, g in space.coords], v))
                                          for k, v in cfg.items()},
                              "val_acc": acc, "val_norm": norm, "analysis": analysis, "checks": checks,
                              "new_val_evals": val.new_evals})
        print(f"[{cid}] mechanism done in {(time.time()-t0)/60:.1f} min ({val.new_evals} new validation merges)",
              flush=True)
    summarize_mechanism(args.out, args.grouping)


def pct(x):
    return "—" if x != x else f"{100*x:.0f}%"


def summarize_mechanism(out, grouping):
    files = sorted((gdir(out, grouping)).glob("*/mechanism_result.json"))
    if not files:
        return
    lines = [f"# Experiment 6b-1: where in depth is EuroSAT's harm created? (VALIDATION, grouping = {grouping})", "",
             "All other lanes at the context's selected α. `harm` = EuroSAT accuracy with the neighbour lane(s) "
             "fully off minus with all lanes on (pp). *only g*: neighbour on only in group g; the harm it "
             "creates there is (neighbour off) − (only g). *off g*: neighbour off only in g; the harm it removes "
             "is (off g) − (all on). Shares are relative to the total harm.", ""]
    rows = []
    for p in files:
        d = json.loads(p.read_text())
        a, G = d["analysis"], d["groups"]
        lines += [f"## {d['context']} (α = {d['alpha']:g}); EuroSAT with all lanes on: {a['eurosat_baseline']:.2f}%", "",
                  "| neighbour | EuroSAT if fully off | total harm (pp) | " +
                  " | ".join(f"{g}: created / share (only g)" for g in G) + " | " +
                  " | ".join(f"{g}: removed / share (off g)" for g in G) + " |",
                  "|---|---:|---:|" + "---:|" * (2 * len(G))]
        for tag, nb in a["neighbours"].items():
            pg = nb["per_group"]
            lines.append(f"| {tag} | {nb['eurosat_neighbour_off']:.2f} | {nb['harm_total']:+.2f} | " +
                         " | ".join(f"{pg[g]['harm_created_only_g']:+.2f} / {pct(pg[g]['share_only_g'])}" for g in G)
                         + " | " +
                         " | ".join(f"{pg[g]['harm_recovered_off_g']:+.2f} / {pct(pg[g]['share_off_g'])}" for g in G)
                         + " |")
            for g in G:
                rows.append({"context": d["context"], "neighbour": tag, "group": g,
                             "harm_total": nb["harm_total"], **pg[g]})
        ia = a["interaction"]
        lines += ["", "SVHN × GTSRB interaction on EuroSAT (reference: both fully off; interaction = "
                  "harm(both in g) − harm(SVHN in g) − harm(GTSRB in g)):", "",
                  "| scope | harm SVHN | harm GTSRB | harm both | interaction (pp) |", "|---|---:|---:|---:|---:|"]
        for g in G:
            r = ia[g]
            lines.append(f"| only {g} | {r['harm_svhn_only_g']:+.2f} | {r['harm_gtsrb_only_g']:+.2f} | "
                         f"{r['harm_both_only_g']:+.2f} | {r['interaction_pp']:+.2f} |")
        lines.append(f"| all groups | | | | {ia['all_groups']['interaction_pp']:+.2f} |")
        acc, tasks = d["val_acc"], d["tasks"]
        lines += ["", f"EuroSAT's own lane boosted ×{d['boost_factor']:g} (validation accuracy, Δ vs all lanes on):", "",
                  "| boosted in | " + " | ".join(tasks) + " | Δ mean normalized |", "|---|" + "---:|" * (len(tasks) + 1)]
        bn = d["val_norm"]["baseline"]
        for g in list(G) + ["all"]:
            name = f"boost[{VICTIM}]@{g}"
            n = d["val_norm"][name]
            dm = sum(n.values()) / len(n) - sum(bn.values()) / len(bn)
            lines.append(f"| {g} | " + " | ".join(f"{acc[name][t]:.2f} ({acc[name][t]-acc['baseline'][t]:+.2f})"
                                                  for t in tasks) + f" | {dm:+.2f} |")
        lines += ["", "All four tasks' accuracies for every configuration are in `mechanism_result.json`.", ""]
    suffix = "" if grouping == "depth" else f"_{grouping}"
    (out / f"mechanism_summary{suffix}.md").write_text("\n".join(lines) + "\n")
    with open(out / f"mechanism{suffix}.csv", "w", newline="") as fh:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("context", "neighbour", "group"), k))
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out / f'mechanism_summary{suffix}.md'}")


# ----------------------------------------------------------------------------
# 6b-2 oracle
# ----------------------------------------------------------------------------
def coordinate_search(ev, space, start, objective, floor, max_passes, log):
    """Greedy coordinate ascent over GRID (plus the start value) per (task, group); as Exp 3."""
    def value(vec):
        _, n = ev(vec)
        mean, worst = score(n)
        feasible = objective == "mean" or worst >= floor - 1e-9
        return (mean if feasible else -1e9), mean, worst

    c = list(start)
    best = value(c)
    for p in range(max_passes):
        improved = False
        for i, (t, g) in enumerate(space.coords):
            for v in sorted(set(GRID) | {start[i]}):
                if v == c[i]:
                    continue
                trial = list(c)
                trial[i] = v
                val = value(trial)
                if val[0] > best[0] + 1e-9:
                    c, best, improved = trial, val, True
            log.append({"objective": objective, "pass": p + 1, "after": f"{t}@{g}", "c": list(c),
                        "val_mean_norm": best[1], "val_worst_norm": best[2]})
            print(f"    [{objective}] pass {p+1} after {t+'@'+g:20s} c={fkey(c)} "
                  f"val mean={best[1]:.2f} worst={best[2]:.2f}", flush=True)
        if not improved:
            break
    return c, best


def cmd_oracle(args):
    write_run_metadata(args, "oracle")
    rt = ro.load_runtime(args)
    split_note("oracle", "coefficients are selected on VALIDATION; each chosen vector is evaluated ONCE on TEST; "
                         "scalar oracle rows are copied from Exp 3 (validation-selected, test-evaluated)")
    for cid in contexts_of(args.contexts):
        res_path = gdir(args.out, args.grouping) / cid / "layer_oracle_result.json"
        if res_path.exists():
            print(f"[{cid}] done, skipping", flush=True)
            continue
        t0 = time.time()
        tasks, alpha, exp3, lm, space, val, test, _ = setup_context(rt, args, cid)
        checks = run_checks(args, cid, tasks, alpha, exp3, lm, space, val, test)
        start = space.uniform(alpha)
        _, n0 = val(start)
        base_mean, base_worst = score(n0)
        print(f"[{cid}] alpha={alpha:g}; {len(space.coords)} coordinates; dc_rank val mean={base_mean:.2f} "
              f"worst={base_worst:.2f}", flush=True)
        log, chosen = [], {}
        for obj in OBJECTIVES:
            c, best = coordinate_search(val, space, start, obj, base_worst, args.max_passes, log)
            chosen[obj] = {"c": c, "val_mean_norm": best[1], "val_worst_norm": best[2]}
        results = {}
        for name, vec in [("dc_rank", start)] + [(f"layer_{o}", chosen[o]["c"]) for o in OBJECTIVES]:
            acc, n = test(vec)                 # one test evaluation per vector (dc_rank's came from the check)
            vacc, vn = val(vec)
            m, w = score(n)
            vm, vw = score(vn)
            results[name] = {"c": space.nested(vec), "test_acc": acc, "test_norm": n,
                             "test_mean_norm": m, "test_worst_norm": w, "test_worst_task": min(n, key=n.get),
                             "val_mean_norm": vm, "val_worst_norm": vw}
        for o in OBJECTIVES:                   # Exp 3 scalar oracles, copied
            r = dict(exp3["test"][f"oracle_{o}"])
            r["val_mean_norm"] = exp3["val_chosen"][o]["val_mean_norm"]
            r["val_worst_norm"] = exp3["val_chosen"][o]["val_worst_norm"]
            results[f"scalar_{o}"] = r
        results["ct"] = dict(exp3["test"]["ct"])
        b = results["dc_rank"]
        for name, r in results.items():
            if "val_mean_norm" in r and name != "dc_rank":
                dvm, dvw = r["val_mean_norm"] - b["val_mean_norm"], r["val_worst_norm"] - b["val_worst_norm"]
                dtm, dtw = r["test_mean_norm"] - b["test_mean_norm"], r["test_worst_norm"] - b["test_worst_norm"]
                r["shrinkage"] = {"val_d_mean": dvm, "test_d_mean": dtm, "shrink_mean": dvm - dtm,
                                  "val_d_worst": dvw, "test_d_worst": dtw, "shrink_worst": dvw - dtw}
        write_json(res_path, {"context": cid, "tasks": tasks, "groups": space.groups, "grouping": args.grouping,
                              "coords": [f"{t}@{g}" for t, g in space.coords], "dc_rank_alpha": alpha,
                              "grid": list(GRID), "max_passes": args.max_passes, "val_evals": len(val.cache),
                              "checks": checks, "val_baseline": {"mean_norm": base_mean, "worst_norm": base_worst},
                              "val_chosen": chosen, "search_log": log, "test": results})
        print(f"[{cid}] done in {(time.time()-t0)/60:.1f} min; {len(val.cache)} validation merges", flush=True)
    summarize_oracle(args.out, args.grouping)


VERDICT_RANK = {"LAYER LEVER": 2, "SMALL GAIN": 1, "NO LEVER": 0}


def layer_verdict(dm, dw, dw_scalar):
    """Pre-registered rule (test averages vs dc_rank)."""
    gap = dw - dw_scalar
    if dw >= 3.0 and gap >= 2.0 and dm >= 1.0:
        return "LAYER LEVER", "predictor of c_{i,g} and a layer-aware method"
    if abs(gap) <= 1.0:
        return "NO LEVER", "the fairness ceiling is not a coefficient-granularity problem"
    if gap > 1.0:
        return "SMALL GAIN", "report and discuss"
    return "NO LEVER", "layer oracle is worse than the scalar oracle on the worst task"


def summarize_oracle(out, grouping):
    files = sorted(gdir(out, grouping).glob("*/layer_oracle_result.json"))
    if not files:
        return
    methods = ("dc_rank", "layer_mean", "layer_fair", "scalar_mean", "scalar_fair", "ct")
    suffix = "" if grouping == "depth" else f"_{grouping}"
    lines = [f"# Experiment 6b-2: per-(task, layer-group) oracle (TEST), grouping = {grouping}", "",
             "Coefficients searched on validation (greedy, 2 passes, grid 0.3–1.0 plus α); each vector evaluated "
             "once on test. `scalar_*` are Experiment 3's per-task oracles (copied). Shrinkage = validation gain "
             "minus test gain, both vs dc_rank.", "",
             "## Per context (test)", "",
             "| context | method | mean norm | worst norm (task) | Δmean | Δworst | val Δmean | val Δworst | "
             "shrink mean | shrink worst |", "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    data, rows = {}, []
    for p in files:
        d = json.loads(p.read_text())
        data[d["context"]] = d
        b = d["test"]["dc_rank"]
        for m in methods:
            r = d["test"][m]
            dm, dw = r["test_mean_norm"] - b["test_mean_norm"], r["test_worst_norm"] - b["test_worst_norm"]
            s = r.get("shrinkage", {})
            g = lambda k: f"{s[k]:+.2f}" if k in s else ""
            lines.append(f"| {d['context']} | {m} | {r['test_mean_norm']:.2f} | {r['test_worst_norm']:.2f} "
                         f"({r['test_worst_task']}) | {dm:+.2f} | {dw:+.2f} | {g('val_d_mean')} | {g('val_d_worst')} | "
                         f"{g('shrink_mean')} | {g('shrink_worst')} |")
            row = {"context": d["context"], "method": m, "test_mean_norm": r["test_mean_norm"],
                   "test_worst_norm": r["test_worst_norm"], "test_worst_task": r["test_worst_task"],
                   "d_mean": dm, "d_worst": dw, **{k: v for k, v in s.items()}}
            if m.startswith("layer") or m == "dc_rank":
                for t, gv in r["c"].items():
                    for gname, v in gv.items():
                        row[f"c_{t}@{gname}"] = v
            elif isinstance(r.get("c"), dict):
                row.update({f"c_{t}": v for t, v in r["c"].items()})
            rows.append(row)

    def avg(ids, m, key):
        vals = []
        for cid in ids:
            d = data[cid]
            vals.append(d["test"][m][key] - d["test"]["dc_rank"][key])
        return sum(vals) / len(vals)

    for label, ids in (("core", [c for c in CORE if c in data]), ("all", list(data))):
        if not ids:
            continue
        lines += ["", f"## Averages over {len(ids)} contexts ({label}; test, Δ vs dc_rank)", "",
                  "| method | Δmean | Δworst | Δworst − scalar_fair Δworst | mean shrink | worst shrink | verdict |",
                  "|---|---:|---:|---:|---:|---:|---|"]
        dw_sf = avg(ids, "scalar_fair", "test_worst_norm")
        verdicts = {}
        for m in methods:
            if m == "dc_rank":
                continue
            dm, dw = avg(ids, m, "test_mean_norm"), avg(ids, m, "test_worst_norm")
            sh = [data[c]["test"][m].get("shrinkage") for c in ids]
            smean = sum(s["shrink_mean"] for s in sh) / len(sh) if all(sh) else float("nan")
            sworst = sum(s["shrink_worst"] for s in sh) / len(sh) if all(sh) else float("nan")
            v = ""
            if m.startswith("layer"):
                verdicts[m] = layer_verdict(dm, dw, dw_sf) + (dm, dw)
                v = f"{verdicts[m][0]}: {verdicts[m][1]}"
            f2 = lambda x: "—" if x != x else f"{x:+.2f}"
            lines.append(f"| {m} | {dm:+.2f} | {dw:+.2f} | {dw - dw_sf:+.2f} | {f2(smean)} | {f2(sworst)} | {v} |")
        best = max(verdicts, key=lambda m: (VERDICT_RANK[verdicts[m][0]], verdicts[m][3]))
        lines += ["", f"**Verdict ({label}, best of the two layer oracles = `{best}`): {verdicts[best][0]}** — "
                  f"{verdicts[best][1]}.", "",
                  "Per context, does the layer oracle beat the scalar `oracle_fair` on test?", "",
                  "| context | best layer oracle | Δmean vs scalar_fair | Δworst vs scalar_fair |", "|---|---|---:|---:|"]
        for cid in ids:
            d = data[cid]["test"]
            lines.append(f"| {cid} | {best} | {d[best]['test_mean_norm'] - d['scalar_fair']['test_mean_norm']:+.2f} | "
                         f"{d[best]['test_worst_norm'] - d['scalar_fair']['test_worst_norm']:+.2f} |")
    lines += ["", "Decision rule (pre-registered): LAYER LEVER if Δworst ≥ +3 pp AND Δworst exceeds scalar "
              "oracle_fair's Δworst by ≥ +2 pp AND Δmean ≥ +1 pp; NO LEVER if Δworst is within 1 pp of the scalar "
              "oracle's (or below it); otherwise SMALL GAIN. \"Best of the two\" is chosen on test among two oracles, "
              "so it is an optimistic upper bound.", "", "## Chosen coefficient tables (validation-selected)", ""]
    for cid, d in data.items():
        groups = d["groups"]
        for m in ("layer_mean", "layer_fair"):
            c = d["test"][m]["c"]
            lines += [f"**{cid} — {m}** (α = {d['dc_rank_alpha']:g})", "",
                      "| task | " + " | ".join(groups) + " | scalar oracle_fair |",
                      "|---|" + "---:|" * (len(groups) + 1)]
            sf = d["test"]["scalar_fair"]["c"]
            for t in d["tasks"]:
                lines.append(f"| {t} | " + " | ".join(f"{c[t][g]:g}" for g in groups) + f" | {sf[t]:g} |")
            lines.append("")
    (out / f"layer_oracle_summary{suffix}.md").write_text("\n".join(lines) + "\n")
    with open(out / f"layer_oracle{suffix}.csv", "w", newline="") as fh:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("context", "method"), k))
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out / f'layer_oracle_summary{suffix}.md'} and {out / f'layer_oracle{suffix}.csv'}")


def cmd_summarize(args):
    summarize_mechanism(args.out, args.grouping)
    summarize_oracle(args.out, args.grouping)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("mechanism", "oracle"):
        p = sub.add_parser(name)
        for a in ("--repo", "--model", "--ft-dir", "--heads", "--data", "--exp3"):
            p.add_argument(a, type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--contexts", default="core", help="core, four, or a comma list")
        p.add_argument("--grouping", choices=tuple(GROUPINGS), default="depth")
        p.add_argument("--device", default="cuda")
        if name == "mechanism":
            p.add_argument("--boost", type=float, default=1.5, help="EuroSAT own-lane boost factor (x alpha)")
        else:
            p.add_argument("--max-passes", type=int, default=2)
    s = sub.add_parser("summarize")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--grouping", choices=tuple(GROUPINGS), default="depth")
    args = ap.parse_args()
    args.out = args.out.resolve()
    if args.cmd == "summarize":
        cmd_summarize(args)
        return
    for a in ("repo", "model", "ft_dir", "heads", "data", "exp3"):
        setattr(args, a, getattr(args, a).resolve())
    if args.cmd == "mechanism" and args.contexts not in ("core",) and \
            any(c not in CORE for c in contexts_of(args.contexts)):
        print("note: the guide runs the mechanism test on the core contexts only", flush=True)
    {"mechanism": cmd_mechanism, "oracle": cmd_oracle}[args.cmd](args)


if __name__ == "__main__":
    main()
