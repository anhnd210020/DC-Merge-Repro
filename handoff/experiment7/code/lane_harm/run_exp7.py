#!/usr/bin/env python3
"""Experiment 7: label-free per-depth lane coefficients, with controls.

Model (as in 6b): W(c) = W0 + sum_{i,g} c_{i,g} L_i|g, groups early = blocks 0-3,
mid = 4-7, late = 8-11, dc_rank lanes (topk="rank", k=16), 12 coordinates per
four-task context in task-major order (task@early, task@mid, task@late, ...).

Label-free signals, on the SAME 500 unlabeled validation images per task (seed 0) as
Experiments 5 and 6, against each task's expert (base + its own LoRA update), whose
predictions are loaded from Experiment 6's cache:
    agree_i(c)   = fraction of images where argmax W(c) == argmax expert_i  (6a's top1)
    rel_i(c)     = agree_i(c) / agree_i(alpha)
    entropy_i(c) = mean normalized prediction entropy (Experiment 5's definition, tau 100)

Subcommands (run in this order; every step resumes from its on-disk caches)
  replay    re-evaluate the reference vectors (dc_rank, Exp 3 scalar oracles, Exp 6 6b
            oracles) on THIS machine; needed for --reference current
  checks    must-hold checks 1-5 per context (also run automatically by every step below)
  gate      7a: agreement on every vector of Exp 6's 6b validation caches; pre-registered gate
  controls  7b: depth_only (3 coords) and scalar_matched (4 coords, finer grid, matched budget)
  method    7c: agree_floor (primary) and agree_relfair, label-free search; test once
  entropy   7e: entropy_mean and entropy_fair with the identical search; test once
  wide      7d: labelled oracle on a widened grid, core only, descriptive
  all8      optional: agree_floor on all8 (24 coords), only if 7c is VIABLE or STRONG
  summarize rebuild every summary from the per-context result files

Reference numbers (dc_rank, scalar oracle_fair, 6b oracles) come from
  --reference archive : the archived Exp 6 replay / Exp 6 results (checks 1-2 must hold
                        within 0.05 pp on this machine, else abort), or
  --reference current : this machine's `replay` (the rule Experiment 6 adopted when it
                        moved machines; archive differences are recorded, abort > 0.5 pp).
The choice is recorded in OUT/run_config.json and cannot change within one OUT.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_official as ro  # noqa: E402
import run_entropy as rent  # noqa: E402
import run_agreement as ra  # noqa: E402
import run_layer_oracle as rl  # noqa: E402

GROUPS = rl.GROUPINGS["depth"]                       # ("early", "mid", "late")
GRID_6B = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
GRID_FINE = tuple(round(0.20 + 0.05 * k, 2) for k in range(21))   # 0.20 .. 1.20
GRID_WIDE = tuple(round(0.1 * k, 1) for k in range(16))           # 0.0 .. 1.5
CORE = list(ro.CONTEXTS)[:3]
FOUR = list(ro.CONTEXTS)[:7]
TAU = ra.TAU
CHECK_TOL_PP = 0.05
REPLAY_ABORT_PP = 0.5
LANE_SUM_TOL = 1e-5
EPS = 1e-9
WIDE_MAX_PASSES = 5
UNTIL_CONVERGED_CAP = 50          # safety cap for "passes until no improvement"
ALL8_DC_OFFICIAL_TEST_MEAN = 73.856   # archived Exp 2 all8 dc_official (alpha 0.4), test mean norm

fkey = rl.fkey


# ----------------------------------------------------------------------------
# concurrency-safe files: several processes (different steps on the same context, or summaries
# written at the end of every step) may touch the same file. Writes use a unique temporary
# name + atomic replace; caches are merged under an exclusive lock, so no entry is ever lost.
# ----------------------------------------------------------------------------
import fcntl  # noqa: E402
import os  # noqa: E402


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}


def locked_merge(path, key, value):
    """Add key -> value to the JSON cache at path without losing other processes' entries."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(f".{path.name}.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            disk = _read(path)
            disk.setdefault(key, value)
            write_json(path, disk)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)
    return disk


class SharedEvaluator(rl.Evaluator):
    """rl.Evaluator whose on-disk cache is shared safely between concurrent processes."""

    def refresh(self, k):
        if k not in self.cache:
            self.cache.update(_read(self.cache_path))

    def __call__(self, vec):
        k = fkey(vec)
        self.refresh(k)
        if k not in self.cache:
            merged = self.lm.compose_groups(self.space.nested(vec))
            value = self.fn(merged, 1.0)
            self.new_evals += 1
            self.cache = locked_merge(self.cache_path, k, value)
        acc = self.cache[k]
        n = {t: 100.0 * acc[t] / self.refs[t] for t in self.space.tasks}
        return acc, n
sha256 = rl.sha256
spearman = rent.spearman


def split_note(stage, text):
    print(f"[split] {stage}: {text}", flush=True)


def contexts_of(spec, allow_all8=False):
    ids = CORE if spec == "core" else (FOUR if spec == "four" else spec.split(","))
    bad = [c for c in ids if c not in ro.CONTEXTS or (c == "all8" and not allow_all8)]
    if bad:
        raise SystemExit(f"unknown or unsupported contexts {bad}")
    return ids


def grid_with(base, alpha):
    return sorted(set(base) | {round(float(alpha), 10)})


def mean_worst(n):
    v = list(n.values())
    return sum(v) / len(v), min(v)


def better(a, b):
    """Lexicographic 'a strictly better than b' with tolerance (higher is better)."""
    for x, y in zip(a, b):
        if x > y + EPS:
            return True
        if x < y - EPS:
            return False
    return False


# ----------------------------------------------------------------------------
# run configuration and paths
# ----------------------------------------------------------------------------
def record_config(args):
    p = args.out / "run_config.json"
    cfg = {"reference": args.reference, "n_unlabeled": args.n_unlabeled, "seed": args.seed,
           "exp6": str(args.exp6), "exp3r": str(args.exp3r)}
    if p.exists():
        old = json.loads(p.read_text())
        diff = {k: (old.get(k), v) for k, v in cfg.items() if old.get(k) != v}
        if diff:
            raise SystemExit(f"{p} was written with different settings {diff}; use a new --out "
                             f"or the original settings (the reference mode cannot change within one OUT)")
    else:
        write_json(p, {**cfg, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print("paths used:", flush=True)
    for name in ("repo", "model", "ft_dir", "heads", "data", "exp3r", "exp6", "out"):
        print(f"  {name:7s} = {getattr(args, name)}", flush=True)
    print(f"  reference mode = {args.reference}; unlabeled = {args.n_unlabeled} per task, seed {args.seed}",
          flush=True)


# ----------------------------------------------------------------------------
# per-context runtime: lanes, space, labelled evaluators, proxy cache
# ----------------------------------------------------------------------------
class Experts:
    """Expert argmax predictions on the unlabeled subset, loaded from Experiment 6's cache."""

    def __init__(self, rt, args):
        self.rt, self.args = rt, args
        self.dir = args.exp6 / "experts"
        self.mem, self.meta = {}, {}

    def get(self, t, info_t):
        torch = self.rt["torch"]
        if t in self.mem:
            return self.mem[t]
        p = self.dir / f"{t}_n{self.args.n_unlabeled}_s{self.args.seed}.pt"
        if not p.exists():
            raise SystemExit(f"expert cache {p} not found (Experiment 6 must have cached task {t}); "
                             f"check --exp6 points at the experiment6 results folder")
        d = torch.load(p, map_location="cpu")
        if d["n"] != info_t["n"] or list(d["first_indices"]) != list(info_t["first_indices"]) or d["tau"] != TAU:
            raise SystemExit(f"[check 4] expert cache {p} was built on a different subset/tau: "
                             f"cached n={d['n']} first={d['first_indices']} vs this run n={info_t['n']} "
                             f"first={info_t['first_indices']}")
        self.mem[t] = d["logits"].argmax(-1)
        self.meta[t] = {"path": str(p), "sha256": sha256(p), "n": d["n"],
                        "first_indices": list(d["first_indices"])}
        print(f"  expert {t}: loaded {p.name} sha256={self.meta[t]['sha256'][:16]}", flush=True)
        return self.mem[t]


class ProxyCache:
    """Per-task agreement and entropy of W(c) on the unlabeled images, cached by the 12-vector."""

    def __init__(self, rt, lm, space, loaders, experts, info, device, path):
        self.torch = rt["torch"]
        self.lm, self.space, self.path = lm, space, path
        self.forward = ra.make_forward(rt, space.tasks, loaders, device)
        self.expert = {t: experts.get(t, info[t]) for t in space.tasks}
        self.cache = json.loads(path.read_text()) if path.exists() else {}
        self.new = 0

    def __call__(self, vec):
        k = fkey(vec)
        if k not in self.cache:
            self.cache.update(_read(self.path))
        if k not in self.cache:
            out = self.forward(self.lm.compose_groups(self.space.nested(vec)))
            rec = {}
            for t in self.space.tasks:
                _emb, logits = out[t]
                agree = float((logits.argmax(-1) == self.expert[t]).double().mean())
                lp = self.torch.log_softmax(logits.double(), dim=-1)
                h = -(lp.exp() * lp).sum(-1) / math.log(logits.shape[-1])
                rec[t] = {"agree": agree, "entropy": float(h.mean())}
            self.new += 1
            self.cache = locked_merge(self.path, k, rec)
        return self.cache[k]


class TestCounter:
    """Test evaluator wrapper: each chosen vector is evaluated once (the cache enforces it);
    counts are logged per search (must-hold check 6)."""

    def __init__(self, ev):
        self.ev, self.log = ev, []

    def __call__(self, vec, search):
        k = fkey(vec)
        self.ev.refresh(k)
        new = k not in self.ev.cache
        acc, n = self.ev(vec)
        self.log.append({"search": search, "vector": k, "new_test_evaluation": new})
        return acc, n


class Ctx:
    def __init__(self, rt, args, cid, alpha=None):
        GroupLaneMerge = rl.make_group_lane_merge()
        self.cid, self.args, self.rt = cid, args, rt
        self.tasks = list(ro.CONTEXTS[cid])
        if alpha is None:
            e3 = json.loads((args.exp3r / cid / "oracle_result.json").read_text())
            alpha = float(e3["dc_rank_alpha"])
        self.alpha = float(alpha)
        grouped = ro.group(rt["deltas"], self.tasks)
        mapping, rows = rl.module_groups(list(grouped), "depth")
        if not getattr(args, "_mapping_printed", False):
            rl.print_mapping(rows, "depth")
            args._mapping_printed = True
        with rt["torch"].no_grad():
            self.lm = GroupLaneMerge(grouped, rt["backend"], self.tasks, ro.dc_kwargs("rank"))
        self.lm.set_groups(mapping, GROUPS, self.alpha)
        self.space = rl.Space(self.tasks, GROUPS)
        self.dir = args.out / cid
        self.val = SharedEvaluator(rt, self.lm, self.space, args.device, "val", rt["refs"]["val"],
                                self.dir / "val_cache.json")
        self.test = TestCounter(SharedEvaluator(rt, self.lm, self.space, args.device, "test",
                                             rt["refs"]["test"], self.dir / "test_cache.json"))
        self.start = self.space.uniform(self.alpha)
        self._proxy = None

    def proxy(self, experts):
        if self._proxy is None:
            loaders, info = rent.unlabeled_loaders(self.rt, self.tasks, self.args.n_unlabeled, self.args.seed)
            self.info = info
            self._proxy = ProxyCache(self.rt, self.lm, self.space, loaders, experts, info, self.args.device,
                                     self.dir / f"proxy_cache_n{self.args.n_unlabeled}_s{self.args.seed}.json")
        return self._proxy

    def to12_task(self, c4):
        return self.space.from_task_vector([float(x) for x in c4])

    def norm(self, acc, split):
        refs = self.rt["refs"][split]
        return {t: 100.0 * acc[t] / refs[t] for t in self.tasks}


# ----------------------------------------------------------------------------
# references: archive or this machine's replay
# ----------------------------------------------------------------------------
def archive_reference(args, cid, tasks):
    """dc_rank + Exp 3 scalar oracles (from the Exp 6 replay of Exp 3) and the Exp 6 6b oracles."""
    e3 = json.loads((args.exp3r / cid / "oracle_result.json").read_text())
    v3 = json.loads((args.exp3r / cid / "val_cache.json").read_text())
    t3 = json.loads((args.exp3r / cid / "test_cache.json").read_text())
    alpha = float(e3["dc_rank_alpha"])
    ref = OrderedDict()
    for name, c4 in (("dc_rank", [alpha] * len(tasks)),
                     ("scalar_mean", e3["val_chosen"]["mean"]["c"]),
                     ("scalar_fair", e3["val_chosen"]["fair"]["c"])):
        k4 = fkey([float(x) for x in c4])
        c12 = [float(c4[tasks.index(t)]) for t in tasks for _ in GROUPS]
        ref[name] = {"c12": c12, "val_acc": v3[k4], "test_acc": t3[k4], "source": f"exp3r:{k4}"}
    p6 = args.exp6 / cid / "layer_oracle_result.json"
    if p6.exists():
        r6 = json.loads(p6.read_text())
        v6 = json.loads((args.exp6 / cid / "layer_val_cache.json").read_text())
        t6 = json.loads((args.exp6 / cid / "layer_test_cache.json").read_text())
        if r6["coords"] != [f"{t}@{g}" for t in tasks for g in GROUPS]:
            raise SystemExit(f"[{cid}] Exp 6 coordinate order differs: {r6['coords']}")
        for o in ("mean", "fair"):
            c12 = [float(x) for x in r6["val_chosen"][o]["c"]]
            k = fkey(c12)
            ref[f"layer_{o}"] = {"c12": c12, "val_acc": v6[k], "test_acc": t6[k], "source": f"exp6:{k}"}
        ref["_exp6_val_evals"] = int(r6["val_evals"])
    return alpha, ref


def load_reference(args, cid, tasks):
    alpha, arch = archive_reference(args, cid, tasks)
    if args.reference == "archive":
        return alpha, arch
    p = args.out / cid / "reference.json"
    if not p.exists():
        raise SystemExit(f"[{cid}] {p} missing: run `replay` first (required for --reference current)")
    cur = json.loads(p.read_text())["reference"]
    cur["_exp6_val_evals"] = arch.get("_exp6_val_evals")
    return alpha, cur


def ref_scores(ctx, ref, name, split):
    acc = ref[name][f"{split}_acc"]
    return mean_worst(ctx.norm(acc, split))


def cmd_replay(args):
    rt = ro.load_runtime(args)
    split_note("replay", "reference vectors re-evaluated on VALIDATION and TEST on this machine; nothing selected")
    for cid in contexts_of(args.contexts):
        p = args.out / cid / "reference.json"
        if p.exists():
            print(f"[{cid}] replay done, skipping", flush=True)
            continue
        ctx = Ctx(rt, args, cid)
        alpha, arch = archive_reference(args, cid, ctx.tasks)
        cur, diffs, worst = OrderedDict(), {}, 0.0
        for name, r in arch.items():
            if name.startswith("_"):
                continue
            acc_v, _ = ctx.val(r["c12"])
            acc_t, _ = ctx.test.ev(r["c12"])
            dv = {t: acc_v[t] - r["val_acc"][t] for t in ctx.tasks}
            dt = {t: acc_t[t] - r["test_acc"][t] for t in ctx.tasks}
            diffs[name] = {"val": dv, "test": dt}
            worst = max(worst, *(abs(x) for x in dv.values()), *(abs(x) for x in dt.values()))
            cur[name] = {"c12": r["c12"], "val_acc": acc_v, "test_acc": acc_t, "source": "this machine"}
            print(f"  [{cid}] {name:12s} max |val diff| {max(abs(x) for x in dv.values()):.3f} pp, "
                  f"max |test diff| {max(abs(x) for x in dt.values()):.3f} pp vs archive", flush=True)
        fn = ro.make_evaluate(rt, ctx.tasks, args.device, "val")
        a1 = fn(ctx.lm.compose_groups(ctx.space.nested(ctx.start)), 1.0)
        det = max(abs(a1[t] - cur["dc_rank"]["val_acc"][t]) for t in ctx.tasks)
        print(f"  [{cid}] determinism: uncached repeat of dc_rank validation differs by {det:.3f} pp", flush=True)
        rec = {"context": cid, "alpha": alpha, "reference": cur, "diffs_vs_archive": diffs,
               "max_abs_diff_pp": worst, "determinism_pp": det, "lane_checks": ctx.lm.checks,
               "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if det != 0.0:
            write_json(args.out / cid / "replay_FAILED.json", rec)
            raise SystemExit(f"[{cid}] repeated evaluation is not deterministic ({det} pp); stop and report")
        if worst > REPLAY_ABORT_PP:
            write_json(args.out / cid / "replay_FAILED.json", rec)
            raise SystemExit(f"[{cid}] replay differs from the archive by {worst:.3f} pp (> {REPLAY_ABORT_PP}); "
                             f"more than near-tie flips; stop and report")
        write_json(p, rec)
        print(f"[{cid}] replay done; max |diff| vs archive {worst:.3f} pp", flush=True)
    summarize_replay(args.out)


def summarize_replay(out):
    files = sorted(out.glob("*/reference.json"))
    if not files:
        return
    lines = ["# Experiment 7: reference vectors re-evaluated on this machine", "",
             "dc_rank and Exp 3 scalar oracles vs the archived Exp 6 replay; 6b oracles vs Exp 6. "
             "Accuracy differences (this machine − archive), pp.", "",
             "| context | vector | max abs val diff | max abs test diff |", "|---|---|---:|---:|"]
    for p in files:
        d = json.loads(p.read_text())
        for name, df in d["diffs_vs_archive"].items():
            lines.append(f"| {d['context']} | {name} | {max(abs(x) for x in df['val'].values()):.3f} | "
                         f"{max(abs(x) for x in df['test'].values()):.3f} |")
    (out / "replay_summary.md").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------
# must-hold checks 1-5 (check 6 is the test counter, logged by every search)
# ----------------------------------------------------------------------------
def run_checks(args, ctx, experts):
    p = ctx.dir / "checks.json"
    if p.exists():
        rec = json.loads(p.read_text())
        if rec.get("passed") and rec.get("reference") == args.reference:
            return rec
    cid, tasks = ctx.cid, ctx.tasks
    rec = {"context": cid, "reference": args.reference, "tolerance_pp": CHECK_TOL_PP}
    is_all8 = cid == "all8"
    if not is_all8:
        alpha, ref = load_reference(args, cid, tasks)
        if abs(alpha - ctx.alpha) > 1e-12:
            raise SystemExit(f"[{cid}] alpha mismatch {alpha} vs {ctx.alpha}")
        # check 1: uniform alpha reproduces dc_rank (val and test)
        acc_v, _ = ctx.val(ctx.start)
        acc_t, _ = ctx.test.ev(ctx.start)
        d1v = max(abs(acc_v[t] - ref["dc_rank"]["val_acc"][t]) for t in tasks)
        d1t = max(abs(acc_t[t] - ref["dc_rank"]["test_acc"][t]) for t in tasks)
        rec["check1_dc_rank"] = {"val_diff_pp": d1v, "test_diff_pp": d1t, "pass": max(d1v, d1t) <= CHECK_TOL_PP}
        # check 2: 6b vectors reproduce Exp 6 validation
        d2 = {}
        for o in ("layer_mean", "layer_fair"):
            if o in ref:
                a, _ = ctx.val(ref[o]["c12"])
                d2[o] = max(abs(a[t] - ref[o]["val_acc"][t]) for t in tasks)
        rec["check2_6b_val"] = {"diff_pp": d2, "pass": bool(d2) and max(d2.values()) <= CHECK_TOL_PP}
    # check 3: lane sums
    rec["check3_lane_sum"] = {"max_rel_sum_error": ctx.lm.checks["max_rel_sum_error"],
                              "pass": ctx.lm.checks["max_rel_sum_error"] < LANE_SUM_TOL}
    # check 4: unlabeled subset identical to Exp 5/6, agree equals 6a's cached values
    prox = ctx.proxy(experts)
    idx_hash = {t: hashlib.sha256(json.dumps(ctx.info[t]["first_indices"]).encode()).hexdigest()[:16]
                for t in tasks}
    sub_ok, sub_notes = True, {}
    meta6 = {}
    p6 = args.exp6 / "agreement_check.json"
    if p6.exists():
        meta6 = json.loads(p6.read_text()).get("metadata", {}).get("contexts", {}).get(cid, {}).get("subset", {})
    for t in tasks:
        mine = {k: ctx.info[t][k] for k in ("n", "of", "first_indices")}
        e = experts.meta[t]
        ok = mine["n"] == e["n"] and mine["first_indices"] == e["first_indices"]
        if t in meta6:
            ok = ok and mine == {k: meta6[t][k] for k in ("n", "of", "first_indices")}
        sub_ok &= ok
        sub_notes[t] = {"this_run": mine, "matches_expert_cache": ok, "in_exp6_gate_metadata": t in meta6}
    a6p = args.exp6 / cid / f"agreement_cache_n{args.n_unlabeled}_s{args.seed}.json"
    agree_cmp = {}
    if a6p.exists() and not is_all8:
        a6 = json.loads(a6p.read_text())
        keys = [fkey([ctx.alpha] * len(tasks))]
        e3 = json.loads((args.exp3r / cid / "oracle_result.json").read_text())
        keys += [fkey([float(x) for x in e3["val_chosen"][o]["c"]]) for o in ("mean", "fair")]
        keys += sorted(a6)
        keys = [k for k in dict.fromkeys(keys) if k in a6][: args.check_agree_vectors]
        for k in keys:
            got = prox(ctx.to12_task([float(x) for x in k.split(",")]))
            agree_cmp[k] = {t: {"exp6": a6[k][t]["top1_agree"], "exp7": got[t]["agree"],
                                "diff_images": round((got[t]["agree"] - a6[k][t]["top1_agree"]) * ctx.info[t]["n"])}
                            for t in tasks}
    max_img = max((abs(v["diff_images"]) for r in agree_cmp.values() for v in r.values()), default=0)
    agree_ok = max_img <= args.agree_tol_images and \
        (args.agree_tol_images > 0 or all(abs(v["exp6"] - v["exp7"]) < 1e-12 for r in agree_cmp.values()
                                          for v in r.values()))
    rec["check4_subset_and_agree"] = {"subset": sub_notes, "subset_index_hash": idx_hash, "agree_vs_6a": agree_cmp,
                                      "agree_vectors_compared": len(agree_cmp), "max_diff_images": max_img,
                                      "tolerance_images": args.agree_tol_images,
                                      "pass": sub_ok and agree_ok and (len(agree_cmp) > 0 or is_all8)}
    # check 5: expert predictions from the Exp 6 cache (hash logged), optionally recomputed
    recompute = {}
    if args.recompute_experts:
        loaders, _ = rent.unlabeled_loaders(ctx.rt, tasks, args.n_unlabeled, args.seed)
        for t in tasks:
            out = ra.make_forward(ctx.rt, [t], {t: loaders[t]}, args.device)(ctx.rt["deltas"][t])[t][1]
            recompute[t] = int((out.argmax(-1) != experts.mem[t]).sum())
    rec["check5_experts"] = {"files": {t: experts.meta[t] for t in tasks}, "recomputed_argmax_mismatches": recompute,
                             "pass": all(v == 0 for v in recompute.values())}
    keys = [k for k in rec if k.startswith("check")]
    rec["passed"] = all(rec[k]["pass"] for k in keys)
    write_json(p, rec)
    line = "; ".join(f"{k.split('_')[0]}={'PASS' if rec[k]['pass'] else 'FAIL'}" for k in keys)
    extra = ""
    if "check1_dc_rank" in rec:
        extra = (f" (dc_rank val {rec['check1_dc_rank']['val_diff_pp']:.3f} / test "
                 f"{rec['check1_dc_rank']['test_diff_pp']:.3f} pp; 6b val "
                 + ", ".join(f"{k} {v:.3f}" for k, v in rec['check2_6b_val']['diff_pp'].items()) + " pp"
                 f"; agree vs 6a on {len(agree_cmp)} vectors, max diff {max_img} images)")
    print(f"[{cid}] checks vs {args.reference} reference: {line}{extra} -> {'PASS' if rec['passed'] else 'FAIL'}",
          flush=True)
    if not rec["passed"]:
        raise SystemExit(f"[{cid}] must-hold check failed; see {p}. Stop and report.")
    return rec


def cmd_checks(args):
    rt = ro.load_runtime(args)
    experts = Experts(rt, args)
    for cid in contexts_of(args.contexts):
        run_checks(args, Ctx(rt, args, cid), experts)


# ----------------------------------------------------------------------------
# generic coordinate search
# ----------------------------------------------------------------------------
class Budget(Exception):
    pass


def coordinate_ascent(value, start, grids, max_passes, log, tag, labels):
    """Greedy coordinate ascent (Exp 3 / 6b rules): coordinates in order, every grid value,
    keep strict improvements, stop after a pass without improvement or after max_passes."""
    c = list(start)
    best = value(c)
    passes = 0
    for p in range(max_passes):
        passes = p + 1
        improved = False
        for i in range(len(c)):
            for v in grids[i]:
                if v == c[i]:
                    continue
                trial = list(c)
                trial[i] = v
                val = value(trial)
                if better(val, best):
                    c, best, improved = trial, val, True
            log.append({"search": tag, "pass": p + 1, "after": labels[i], "c": list(c), "value": list(best)})
        print(f"    [{tag}] pass {p+1}: c={fkey(c)} value={tuple(round(x, 4) for x in best)}", flush=True)
        if not improved:
            break
    return c, best, passes


def labelled_value(ctx, to12, objective, floor, budget=None):
    """Validation-label objective of Exp 3 / 6b: mean normalized, fair = subject to worst >= floor."""
    def value(coords):
        vec = to12(coords)
        if budget is not None:
            budget.touch(fkey(vec))
        _, n = ctx.val(vec)
        m, w = mean_worst(n)
        feasible = objective == "mean" or w >= floor - EPS
        return (m if feasible else -1e9,)
    return value


class BudgetTracker:
    def __init__(self, limit):
        self.limit, self.visited = limit, []
        self._set = set()

    def touch(self, key):
        if key in self._set:
            return
        if len(self.visited) >= self.limit:
            raise Budget()
        self._set.add(key)
        self.visited.append(key)


# ----------------------------------------------------------------------------
# 7a gate
# ----------------------------------------------------------------------------
def gate_rows(ctx, prox, vcache, ref, r6):
    tasks = ctx.tasks
    keys = sorted(vcache)
    base_key = fkey(ctx.start)
    if base_key not in vcache:
        raise SystemExit(f"[{ctx.cid}] Exp 6 validation cache lacks the dc_rank vector {base_key}")
    a0 = prox(ctx.start)
    rows = []
    for k in keys:
        vec = [float(x) for x in k.split(",")]
        P = prox(vec)
        n = ctx.norm(vcache[k], "val")
        m, w = mean_worst(n)
        ag = {t: P[t]["agree"] for t in tasks}
        rel = {t: (ag[t] / a0[t]["agree"]) if a0[t]["agree"] > 0 else float("nan") for t in tasks}
        rows.append({"key": k, "val_mean": m, "val_worst": w, "mean_agree": sum(ag.values()) / len(ag),
                     "min_agree": min(ag.values()), "min_rel": min(rel.values()),
                     "all_rel_ge1": all(rel[t] >= 1.0 - 1e-12 for t in tasks)})
    return rows


def gate_context(ctx, rows, r6):
    base = next(r for r in rows if r["key"] == fkey(ctx.start))
    pick = {}
    pick["i_max_mean_agree"] = max(rows, key=lambda r: (r["mean_agree"], r["min_rel"]))
    feas = [r for r in rows if r["all_rel_ge1"]]
    pick["ii_max_mean_agree_rel_ge1"] = max(feas, key=lambda r: (r["mean_agree"], r["min_rel"]))
    pick["iii_max_min_rel"] = max(rows, key=lambda r: (r["min_rel"], r["mean_agree"]))
    out = {"context": ctx.cid, "vectors": len(rows),
           "rho_mean": spearman([r["mean_agree"] for r in rows], [r["val_mean"] for r in rows]),
           "rho_worst_rel": spearman([r["min_rel"] for r in rows], [r["val_worst"] for r in rows]),
           "rho_worst_raw": spearman([r["min_agree"] for r in rows], [r["val_worst"] for r in rows]),
           "dc_rank": {"val_mean": base["val_mean"], "val_worst": base["val_worst"]},
           "best_cached_mean": max(rows, key=lambda r: r["val_mean"]),
           "best_cached_worst": max(rows, key=lambda r: r["val_worst"]),
           "feasible_rel_ge1": len(feas), "picks": {}}
    for name, r in pick.items():
        out["picks"][name] = {"vector": r["key"], "val_mean": r["val_mean"], "val_worst": r["val_worst"],
                              "beats_mean": r["val_mean"] > base["val_mean"] + EPS,
                              "beats_worst": r["val_worst"] > base["val_worst"] + EPS}
    out["exp6_oracle_picks"] = {o: {"vector": fkey(r6["val_chosen"][o]["c"]),
                                    "val_mean": r6["val_chosen"][o]["val_mean_norm"],
                                    "val_worst": r6["val_chosen"][o]["val_worst_norm"]} for o in ("mean", "fair")}
    return out


def gate_decision(per):
    """Pre-registered (core, validation).
    PASS : rho_mean >= 0.7 in all three core contexts AND the pick of rule (ii) or rule (iii)
           beats dc_rank on BOTH mean and worst in all three (one rule across all three).
    FAIL : rho_mean < 0.5 in any core context, OR no rule's pick beats dc_rank on the worst
           task in any core context (literal reading: in none of the three).
    else : IN_BETWEEN -> 7c on core only, at most PARTIAL.
    Alternative readings are recorded beside the verdict."""
    got = {d["context"]: d for d in per if d["context"] in CORE}
    if set(got) != set(CORE):
        return {"verdict": "INCOMPLETE", "reason": "the gate needs all three core contexts"}
    rhos = [got[c]["rho_mean"] for c in CORE]
    rhos_ok = [x if x == x else -1.0 for x in rhos]
    both = lambda c, r: got[c]["picks"][r]["beats_mean"] and got[c]["picks"][r]["beats_worst"]
    rule_pass = {r: all(both(c, r) for c in CORE) for r in ("ii_max_mean_agree_rel_ge1", "iii_max_min_rel")}
    either_per_context = all(any(both(c, r) for r in rule_pass) for c in CORE)
    worst_any = {c: any(p["beats_worst"] for p in got[c]["picks"].values()) for c in CORE}
    fail_literal = min(rhos_ok) < 0.5 or not any(worst_any.values())
    fail_strict = min(rhos_ok) < 0.5 or not all(worst_any.values())
    passes = min(rhos_ok) >= 0.7 and any(rule_pass.values())
    if passes:
        v, why = "PASS", "rho_mean >= 0.7 everywhere and rule " + \
            " / ".join(r for r, ok in rule_pass.items() if ok) + " beats dc_rank on mean and worst in all three"
    elif fail_literal:
        v = "FAIL"
        why = "rho_mean < 0.5 in a core context" if min(rhos_ok) < 0.5 else \
            "no rule's pick beats dc_rank on the worst task in any core context"
    else:
        v, why = "IN_BETWEEN", "neither the PASS nor the FAIL condition holds"
    return {"verdict": v, "reason": why, "rho_mean": dict(zip(CORE, rhos)), "rule_passes_all_three": rule_pass,
            "worst_beaten_by_some_rule": worst_any,
            "alt_pass_if_rules_may_differ_per_context": min(rhos_ok) >= 0.7 and either_per_context,
            "alt_fail_if_any_context_has_no_worst_win": fail_strict,
            "thresholds": {"rho_pass": 0.7, "rho_fail": 0.5}}


GATE_TEXT = {"PASS": "GATE PASS -> run 7c (method) and 7e (entropy), core then all 7",
             "FAIL": "GATE FAIL -> do not run 7c/7e; report. 7b and 7d still run",
             "IN_BETWEEN": "IN BETWEEN -> run 7c/7e on core only; label the result PARTIAL at most",
             "INCOMPLETE": "INCOMPLETE -> run the gate on all three core contexts"}


def cmd_gate(args):
    rt = ro.load_runtime(args)
    experts = Experts(rt, args)
    split_note("gate", "agreement uses only the unlabeled VALIDATION subset; accuracies are Exp 6's cached "
                       "6b VALIDATION accuracies (no new labelled evaluation); no test split is touched")
    for cid in contexts_of(args.contexts):
        p = args.out / cid / "gate_result.json"
        if p.exists():
            print(f"[{cid}] gate done, skipping", flush=True)
            continue
        t0 = time.time()
        ctx = Ctx(rt, args, cid)
        run_checks(args, ctx, experts)
        prox = ctx.proxy(experts)
        vcache = json.loads((args.exp6 / cid / "layer_val_cache.json").read_text())
        r6 = json.loads((args.exp6 / cid / "layer_oracle_result.json").read_text())
        _, ref = load_reference(args, cid, ctx.tasks)
        rows = gate_rows(ctx, prox, vcache, ref, r6)
        res = gate_context(ctx, rows, r6)
        write_json(p, {**res, "rows": rows})
        print(f"[{cid}] gate: {len(rows)} vectors in {(time.time()-t0)/60:.1f} min ({prox.new} new proxy evaluations); "
              f"rho_mean={res['rho_mean']:.2f} rho_worst(rel)={res['rho_worst_rel']:.2f} "
              f"rho_worst(raw)={res['rho_worst_raw']:.2f} [validation]", flush=True)
    summarize_gate(args.out)


def read_gate(out):
    p = out / "gate.json"
    if not p.exists():
        raise SystemExit(f"{p} not found: run `gate` on the core contexts first")
    return json.loads(p.read_text())["decision"]


def fmt(x, nd=2):
    return "—" if x is None or x != x else f"{x:.{nd}f}"


def summarize_gate(out):
    files = sorted(out.glob("*/gate_result.json"))
    if not files:
        return
    per = [json.loads(p.read_text()) for p in files]
    dec = gate_decision(per)
    names = {"i_max_mean_agree": "(i) max mean agree", "ii_max_mean_agree_rel_ge1": "(ii) max mean agree, rel ≥ 1",
             "iii_max_min_rel": "(iii) max min rel"}
    cfg = json.loads((out / "run_config.json").read_text()) if (out / "run_config.json").exists() else {}
    lines = ["# Experiment 7a: agreement gate on the per-depth family (VALIDATION, Exp 6 6b vectors)", "",
             f"Agreement = top-1 agreement with each task's expert on the {cfg.get('n_unlabeled', 500)} unlabeled "
             f"validation images, seed {cfg.get('seed', 0)} (labels never "
             "read). Accuracies = Exp 6's cached 6b validation accuracies, normalized. No new labelled evaluation.", "",
             "| context | vectors | ρ(mean agree, mean acc) | ρ(min rel, worst acc) | ρ(min raw agree, worst acc) | "
             "dc_rank mean / worst | best cached mean / best cached worst |", "|---|---:|---:|---:|---:|---|---|"]
    rows = []
    for d in per:
        lines.append(f"| {d['context']} | {d['vectors']} | {fmt(d['rho_mean'])} | {fmt(d['rho_worst_rel'])} | "
                     f"{fmt(d['rho_worst_raw'])} | {fmt(d['dc_rank']['val_mean'])} / {fmt(d['dc_rank']['val_worst'])} | "
                     f"{fmt(d['best_cached_mean']['val_mean'])} / {fmt(d['best_cached_worst']['val_worst'])} |")
    lines += ["", "Picks (validation mean / worst normalized; ✓ = beats dc_rank):", "",
              "| context | rule | vector | val mean | val worst | beats mean | beats worst |",
              "|---|---|---|---:|---:|:---:|:---:|"]
    for d in per:
        for r, pk in d["picks"].items():
            lines.append(f"| {d['context']} | {names[r]} | `{pk['vector']}` | {fmt(pk['val_mean'])} | "
                         f"{fmt(pk['val_worst'])} | {'✓' if pk['beats_mean'] else '✗'} | {'✓' if pk['beats_worst'] else '✗'} |")
            rows.append({"context": d["context"], "rule": r, "vector": pk["vector"], "val_mean": pk["val_mean"],
                         "val_worst": pk["val_worst"], "beats_mean": pk["beats_mean"], "beats_worst": pk["beats_worst"],
                         "rho_mean": d["rho_mean"], "rho_worst_rel": d["rho_worst_rel"],
                         "rho_worst_raw": d["rho_worst_raw"], "dc_rank_val_mean": d["dc_rank"]["val_mean"],
                         "dc_rank_val_worst": d["dc_rank"]["val_worst"]})
        for o, pk in d["exp6_oracle_picks"].items():
            lines.append(f"| {d['context']} | 6b oracle layer_{o} (labels) | `{pk['vector']}` | {fmt(pk['val_mean'])} | "
                         f"{fmt(pk['val_worst'])} | | |")
    lines += ["", "## Gate (pre-registered, core contexts, validation)", "",
              f"**{dec['verdict']}** — {dec['reason']}", "", GATE_TEXT[dec["verdict"]], ""]
    if dec["verdict"] != "INCOMPLETE":
        lines += [f"Alternative readings: PASS if (ii)/(iii) may differ per context = "
                  f"{dec['alt_pass_if_rules_may_differ_per_context']}; FAIL if any core context has no worst-task win = "
                  f"{dec['alt_fail_if_any_context_has_no_worst_win']}."]
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate_summary.md").write_text("\n".join(lines) + "\n")
    write_json(out / "gate.json", {"decision": dec, "contexts": [d["context"] for d in per]})
    _csv(out / "gate.csv", rows, ("context", "rule"))
    print(f"GATE: {dec['verdict']} ({dec['reason']}); wrote {out / 'gate_summary.md'}", flush=True)


def _csv(path, rows, first):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in first, first.index(k) if k in first else 0, k))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ----------------------------------------------------------------------------
# result records shared by 7b / 7c / 7d / 7e
# ----------------------------------------------------------------------------
def score_chosen(ctx, vec, search, with_val=True):
    acc_t, n_t = ctx.test(vec, search)
    m, w = mean_worst(n_t)
    rec = {"c12": list(vec), "test_acc": acc_t, "test_norm": n_t, "test_mean_norm": m, "test_worst_norm": w,
           "test_worst_task": min(n_t, key=n_t.get)}
    if with_val:
        acc_v, n_v = ctx.val(vec)
        rec["val_mean_norm"], rec["val_worst_norm"] = mean_worst(n_v)
    return rec


def ref_block(ctx, ref):
    out = {}
    for name in ("dc_rank", "scalar_mean", "scalar_fair", "layer_mean", "layer_fair"):
        if name in ref:
            tm, tw = ref_scores(ctx, ref, name, "test")
            vm, vw = ref_scores(ctx, ref, name, "val")
            out[name] = {"c12": ref[name]["c12"], "test_mean_norm": tm, "test_worst_norm": tw,
                         "val_mean_norm": vm, "val_worst_norm": vw}
    return out


# ----------------------------------------------------------------------------
# 7b controls
# ----------------------------------------------------------------------------
def cmd_controls(args):
    rt = ro.load_runtime(args)
    experts = Experts(rt, args)
    split_note("controls", "coefficients selected on VALIDATION (labels); each chosen vector scored ONCE on TEST")
    for cid in contexts_of(args.contexts):
        p = args.out / cid / "controls_result.json"
        if p.exists():
            print(f"[{cid}] controls done, skipping", flush=True)
            continue
        t0 = time.time()
        ctx = Ctx(rt, args, cid)
        run_checks(args, ctx, experts)
        _, ref = load_reference(args, cid, ctx.tasks)
        budget_n = ref.get("_exp6_val_evals")
        if not budget_n:
            raise SystemExit(f"[{cid}] Exp 6 val_evals not found; cannot match the budget")
        _, n0 = ctx.val(ctx.start)
        floor = mean_worst(n0)[1]
        log, chosen, meta = [], {}, {}
        # depth_only: 3 coordinates d_g shared by all tasks
        to12_d = lambda d: [float(d[GROUPS.index(g)]) for _t, g in ctx.space.coords]
        grids_d = [grid_with(GRID_6B, ctx.alpha)] * 3
        for o in ("mean", "fair"):
            c, best, passes = coordinate_ascent(labelled_value(ctx, to12_d, o, floor), [ctx.alpha] * 3, grids_d, 2,
                                                log, f"depth_only_{o}", list(GROUPS))
            chosen[f"depth_only_{o}"] = {"coords": c, "c12": to12_d(c), "passes": passes}
        # scalar_matched: 4 coordinates, fine grid, until converged, then seeded restarts to the 6b budget
        to12_s = lambda c: ctx.to12_task(c)
        grids_s = [grid_with(GRID_FINE, ctx.alpha)] * len(ctx.tasks)
        bt = BudgetTracker(budget_n)
        best_s = {o: None for o in ("mean", "fair")}
        rng = random.Random(0)
        runs = 0
        try:
            starts = [[ctx.alpha] * len(ctx.tasks)]
            while True:
                s = starts.pop(0) if starts else [rng.choice(GRID_FINE) for _ in ctx.tasks]
                for o in ("mean", "fair"):
                    c, val, passes = coordinate_ascent(labelled_value(ctx, to12_s, o, floor, bt), s, grids_s,
                                                       UNTIL_CONVERGED_CAP, log, f"scalar_matched_{o}_run{runs}",
                                                       ctx.tasks)
                    if best_s[o] is None or better(val, best_s[o][1]):
                        best_s[o] = (c, val, runs)
                runs += 1
        except Budget:
            pass
        # Keep the best over everything evaluated within the budget (all restarts, both objectives' paths):
        # for greedy ascent this equals the best endpoint whenever a run finished, and it also counts the
        # progress of a run cut off by the budget. Generous to the control, i.e. conservative for 6b.
        for o in ("mean", "fair"):
            val_o = labelled_value(ctx, lambda v: v, o, floor)
            cand = []
            for k in bt.visited:
                vec = [float(x) for x in k.split(",")]
                cand.append((val_o(vec), k))
            bv, bk = cand[0]
            for v, k in cand[1:]:
                if better(v, bv):
                    bv, bk = v, k
            vec = [float(x) for x in bk.split(",")]
            coords = [vec[ctx.space.coords.index((t, GROUPS[0]))] for t in ctx.tasks]
            chosen[f"scalar_matched_{o}"] = {"coords": coords, "c12": vec,
                                             "best_endpoint_run": best_s[o][2] if best_s[o] else None,
                                             "val_objective": list(bv)}
        meta["scalar_matched"] = {"budget": budget_n, "distinct_val_evaluations": len(bt.visited),
                                  "runs_started": runs + 1, "seed": 0}
        results = {name: {**score_chosen(ctx, ch["c12"], name), **{k: v for k, v in ch.items() if k != "c12"}}
                   for name, ch in chosen.items()}
        write_json(p, {"context": cid, "tasks": ctx.tasks, "alpha": ctx.alpha, "floor_val_worst": floor,
                       "results": results, "reference": ref_block(ctx, ref), "meta": meta,
                       "test_log": ctx.test.log, "search_log": log, "reference_mode": args.reference})
        new_t = sum(1 for e in ctx.test.log if e["new_test_evaluation"])
        print(f"[{cid}] controls done in {(time.time()-t0)/60:.1f} min; scalar_matched used "
              f"{len(bt.visited)}/{budget_n} distinct validation evaluations; test: {len(results)} chosen vectors, "
              f"{new_t} new test evaluations", flush=True)
    summarize_controls(args.out)


def best_of_two(avg, prefix):
    """Best of the two objectives = larger core-average Δworst, ties by Δmean."""
    a, b = avg[f"{prefix}_mean"], avg[f"{prefix}_fair"]
    return (f"{prefix}_mean", a) if (a[1], a[0]) >= (b[1], b[0]) else (f"{prefix}_fair", b)


def averages(recs, names):
    out = {}
    for m in names:
        vals = [(r["all"][m]["test_mean_norm"] - r["all"]["dc_rank"]["test_mean_norm"],
                 r["all"][m]["test_worst_norm"] - r["all"]["dc_rank"]["test_worst_norm"])
                for r in recs if m in r["all"]]
        if len(vals) == len(recs) and vals:
            out[m] = (sum(v[0] for v in vals) / len(vals), sum(v[1] for v in vals) / len(vals))
    return out


def controls_verdict(avg):
    ok = all(k in avg for k in ("layer_mean", "layer_fair", "depth_only_mean", "depth_only_fair",
                                "scalar_matched_mean", "scalar_matched_fair"))
    if not ok:
        return "INCOMPLETE", {}
    n6, l6 = best_of_two(avg, "layer")
    nd, d = best_of_two(avg, "depth_only")
    ns, s = best_of_two(avg, "scalar_matched")
    hold = []
    if l6[1] - d[1] >= 2 and l6[1] - s[1] >= 2 and l6[0] - d[0] >= 1 and l6[0] - s[0] >= 1:
        hold.append("TASK-SPECIFIC DEPTH")
    if l6[0] > 0 and d[0] >= 0.7 * l6[0] and d[1] >= l6[1] - 1:
        hold.append("DEPTH SCHEDULE SUFFICES")
    if s[1] >= l6[1] - 1 and s[0] >= l6[0] - 1:
        hold.append("SEARCH BUDGET EXPLAINS IT")
    return (" + ".join(hold) if hold else "MIXED"), {"6b": (n6, l6), "depth_only": (nd, d), "scalar_matched": (ns, s)}


def summarize_controls(out):
    files = sorted(out.glob("*/controls_result.json"))
    if not files:
        return
    arms = ("depth_only_mean", "depth_only_fair", "scalar_matched_mean", "scalar_matched_fair")
    refs = ("layer_mean", "layer_fair", "scalar_fair")
    recs = []
    for p in files:
        d = json.loads(p.read_text())
        d["all"] = {**d["reference"], **d["results"]}
        recs.append(d)
    lines = ["# Experiment 7b: controls for the 6b per-depth oracle gain (TEST, Δ vs dc_rank)", "",
             "Selected on validation with labels; each chosen vector scored once on test. Reference rows (dc_rank, "
             f"6b layer oracles, Exp 3 scalar oracle_fair) are the `{recs[0]['reference_mode']}` reference.", "",
             "| context | method | test mean | test worst (task) | Δmean | Δworst | val Δmean | val Δworst |",
             "|---|---|---:|---|---:|---:|---:|---:|"]
    rows = []
    for d in recs:
        b = d["all"]["dc_rank"]
        for m in ("dc_rank",) + arms + refs:
            if m not in d["all"]:
                continue
            r = d["all"][m]
            dm, dw = r["test_mean_norm"] - b["test_mean_norm"], r["test_worst_norm"] - b["test_worst_norm"]
            vdm = r.get("val_mean_norm", float("nan")) - b["val_mean_norm"]
            vdw = r.get("val_worst_norm", float("nan")) - b["val_worst_norm"]
            lines.append(f"| {d['context']} | {m} | {r['test_mean_norm']:.2f} | {r['test_worst_norm']:.2f}"
                         f"{' (' + r['test_worst_task'] + ')' if 'test_worst_task' in r else ''} | {dm:+.2f} | {dw:+.2f} | "
                         f"{fmt(vdm)} | {fmt(vdw)} |")
            rows.append({"context": d["context"], "method": m, "test_mean_norm": r["test_mean_norm"],
                         "test_worst_norm": r["test_worst_norm"], "d_mean": dm, "d_worst": dw,
                         "c12": fkey(r["c12"])})
    for label, sel in (("core", [d for d in recs if d["context"] in CORE]), ("all", recs)):
        if not sel or (label == "all" and len(sel) == len([d for d in recs if d["context"] in CORE])):
            continue
        avg = averages(sel, arms + refs)
        lines += ["", f"## Averages over {len(sel)} contexts ({label}; test, Δ vs dc_rank)", "",
                  "| method | Δmean | Δworst |", "|---|---:|---:|"]
        for m, (a, w) in avg.items():
            lines.append(f"| {m} | {a:+.2f} | {w:+.2f} |")
        if label == "core":
            v, parts = controls_verdict(avg)
            lines += ["", f"**Verdict (core, pre-registered): {v}**", ""]
            if parts:
                lines.append("Best of the two objectives per arm (larger Δworst, ties by Δmean): " +
                             "; ".join(f"{k} = `{n}` ({x[0]:+.2f} / {x[1]:+.2f})" for k, (n, x) in parts.items()))
    lines += ["", "## Chosen vectors", "", "| context | arm | coordinates |", "|---|---|---|"]
    for d in recs:
        for m in arms:
            r = d["results"][m]
            lines.append(f"| {d['context']} | {m} | {fkey(r['coords'])} |")
        sm = d["meta"]["scalar_matched"]
        lines.append(f"| {d['context']} | scalar_matched budget | {sm['distinct_val_evaluations']}/{sm['budget']} "
                     f"distinct val evals, {sm['runs_started']} starts |")
    (out / "controls_summary.md").write_text("\n".join(lines) + "\n")
    _csv(out / "controls.csv", rows, ("context", "method"))
    print(f"wrote {out / 'controls_summary.md'}", flush=True)


# ----------------------------------------------------------------------------
# 7c method and 7e entropy (label-free search, identical machinery)
# ----------------------------------------------------------------------------
PROXY_OBJECTIVES = {"method": ("agree_floor", "agree_relfair"), "entropy": ("entropy_mean", "entropy_fair")}


def proxy_value(prox, to12, tasks, objective, base):
    def value(coords):
        P = prox(to12(coords))
        ag = [P[t]["agree"] for t in tasks]
        mean_ag = sum(ag) / len(ag)
        if objective == "agree_floor":
            ok = all(P[t]["agree"] >= base[t]["agree"] - 1e-12 for t in tasks)
            return (mean_ag if ok else -1e9,)
        if objective == "agree_relfair":
            rel = [P[t]["agree"] / base[t]["agree"] if base[t]["agree"] > 0 else 0.0 for t in tasks]
            return (min(rel), mean_ag)
        H = [P[t]["entropy"] for t in tasks]
        if objective == "entropy_mean":
            return (-sum(H) / len(H), -max(H))
        if objective == "entropy_fair":
            return (-max(H), -sum(H) / len(H))
        raise ValueError(objective)
    return value


def gate_allows(args, cid):
    g = read_gate(args.out)
    v = g["verdict"]
    if v in ("FAIL", "INCOMPLETE"):
        raise SystemExit(f"7a gate verdict is {v} ({g.get('reason')}); the pre-registered rule says not to run 7c/7e")
    if v == "IN_BETWEEN" and cid not in CORE:
        raise SystemExit("7a gate is IN_BETWEEN: the pre-registered rule allows the core contexts only")
    return v


def cmd_proxy_search(args, kind):
    rt = ro.load_runtime(args)
    experts = Experts(rt, args)
    split_note(kind, "search uses ONLY the unlabeled VALIDATION subset (no labels); chosen vectors scored ONCE on "
                     "TEST (result) and on VALIDATION with labels (diagnostic only)")
    todo = contexts_of(args.contexts, allow_all8=(kind == "all8"))
    if kind != "all8":
        for cid in todo:                 # refuse before any work if a context is not allowed
            gate_allows(args, cid)
    for cid in todo:
        if kind == "all8":
            gv = gate_allows(args, CORE[0])
            mv = json.loads((args.out / "method.json").read_text())["verdict_all"] \
                if (args.out / "method.json").exists() else None
            if mv not in ("STRONG", "VIABLE"):
                raise SystemExit(f"all8 runs only if 7c (all 7) is VIABLE or STRONG; it is {mv}")
            objectives, fname = ("agree_floor",), "all8_result.json"
        else:
            gv = gate_allows(args, cid)
            objectives, fname = PROXY_OBJECTIVES[kind], f"{kind}_result.json"
        p = args.out / cid / fname
        if p.exists():
            print(f"[{cid}] {kind} done, skipping", flush=True)
            continue
        t0 = time.time()
        ctx = Ctx(rt, args, cid, alpha=(args.all8_alpha if cid == "all8" else None))
        run_checks(args, ctx, experts)
        prox = ctx.proxy(experts)
        base = prox(ctx.start)
        grids = [grid_with(GRID_6B, ctx.alpha)] * len(ctx.space.coords)
        labels = [f"{t}@{g}" for t, g in ctx.space.coords]
        log, results = [], {}
        for o in objectives:
            c, val, passes = coordinate_ascent(proxy_value(prox, lambda v: v, ctx.tasks, o, base), ctx.start, grids,
                                               2, log, o, labels)
            results[o] = {**score_chosen(ctx, c, o), "passes": passes, "objective_value": list(val),
                          "proxies": prox(c)}
        if cid == "all8":
            ref = {"dc_rank": score_chosen(ctx, ctx.start, "all8_dc_rank")}
            refb = {"dc_rank": ref["dc_rank"], "dc_official_archived_test_mean": ALL8_DC_OFFICIAL_TEST_MEAN}
        else:
            _, refr = load_reference(args, cid, ctx.tasks)
            refb = ref_block(ctx, refr)
        refb.setdefault("dc_rank", {})["proxies"] = base
        write_json(p, {"context": cid, "tasks": ctx.tasks, "alpha": ctx.alpha, "kind": kind, "gate_verdict": gv,
                       "grid": grids[0], "max_passes": 2, "proxy_evals": len(prox.cache), "results": results,
                       "reference": refb, "test_log": ctx.test.log, "search_log": log,
                       "reference_mode": args.reference})
        new_t = sum(1 for e in ctx.test.log if e["new_test_evaluation"])
        print(f"[{cid}] {kind} done in {(time.time()-t0)/60:.1f} min ({len(prox.cache)} proxy evaluations); test: "
              f"{len(results)} chosen vectors, {new_t} new test evaluations", flush=True)
    summarize_method(args.out)


def method_verdict(dm, dm_6b_fair, dw, drops, capped):
    if dm <= 0:
        return "NOT VIABLE"
    if dm_6b_fair > 0 and dm >= 0.5 * dm_6b_fair and drops == 0:
        v = "STRONG" if dw >= 3.0 else ("VIABLE" if dw > 0 else "PARTIAL")
        return "PARTIAL" if capped and v in ("STRONG", "VIABLE") else v
    return "PARTIAL"


def summarize_method(out):
    summarize_all8(out)
    recs = {}
    for kind in ("method", "entropy"):
        for p in sorted(out.glob(f"*/{kind}_result.json")):
            d = json.loads(p.read_text())
            r = recs.setdefault(d["context"], {"context": d["context"], "tasks": d["tasks"], "alpha": d["alpha"],
                                               "gate": d["gate_verdict"], "all": dict(d["reference"])})
            r["all"].update(d["results"])
    if not recs:
        return
    order = [c for c in FOUR if c in recs]
    rl_ = [recs[c] for c in order]
    methods = ("agree_floor", "agree_relfair", "entropy_mean", "entropy_fair")
    refs = ("layer_mean", "layer_fair", "scalar_fair")
    capped = any(r["gate"] == "IN_BETWEEN" for r in rl_)
    lines = ["# Experiment 7c/7e: label-free per-depth lane coefficients (TEST, Δ vs dc_rank)", "",
             "7c = agreement objectives (agree_floor primary, agree_relfair secondary); 7e = entropy objective on "
             "per-depth lanes (identical 12-coordinate search; not AdaMerging). Selection uses only unlabeled images and "
             "the experts — **not data-free**. Each chosen vector scored once on test.",
             f"7a gate: **{rl_[0]['gate']}**" + (" (result capped at PARTIAL)" if capped else "") + ".", "",
             "| context | method | test mean | test worst (task) | Δmean | Δworst | share of 6b layer_fair Δmean | "
             "share of 6b layer_mean Δmean | val Δmean (diag.) |", "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    rows = []
    for r in rl_:
        a, b = r["all"], r["all"]["dc_rank"]
        f6 = a.get("layer_fair", {}).get("test_mean_norm", float("nan")) - b["test_mean_norm"]
        m6 = a.get("layer_mean", {}).get("test_mean_norm", float("nan")) - b["test_mean_norm"]
        for m in ("dc_rank",) + methods + refs:
            if m not in a:
                continue
            x = a[m]
            dm, dw = x["test_mean_norm"] - b["test_mean_norm"], x["test_worst_norm"] - b["test_worst_norm"]
            sh = lambda den: f"{100*dm/den:.0f}%" if den and den == den and den > 0 and m in methods else ""
            vdm = x.get("val_mean_norm", float("nan")) - b.get("val_mean_norm", float("nan"))
            lines.append(f"| {r['context']} | {m} | {x['test_mean_norm']:.2f} | {x['test_worst_norm']:.2f}"
                         f"{' (' + x['test_worst_task'] + ')' if 'test_worst_task' in x else ''} | {dm:+.2f} | {dw:+.2f} | "
                         f"{sh(f6)} | {sh(m6)} | {fmt(vdm) if m in methods else ''} |")
            rows.append({"context": r["context"], "method": m, "test_mean_norm": x["test_mean_norm"],
                         "test_worst_norm": x["test_worst_norm"], "d_mean": dm, "d_worst": dw, "c12": fkey(x["c12"])})
    verdicts = {}
    core_sel = [r for r in rl_ if r["context"] in CORE]
    for label, sel in (("core", core_sel), ("all", rl_)):
        if not sel or (label == "all" and len(sel) == len(core_sel)):
            continue
        if label == "all" and len(sel) != 7:
            lines += ["", f"(all-7 verdict pending: {len(sel)} of 7 contexts finished)"]
        avg = averages(sel, methods + refs)
        lines += ["", f"## Averages over {len(sel)} contexts ({label}; test, Δ vs dc_rank)", "",
                  "| method | Δmean | Δworst | contexts with worst drop > 1 pp | verdict |", "|---|---:|---:|---:|---|"]
        for m, (a_, w_) in avg.items():
            drops = sum(1 for r in sel if r["all"][m]["test_worst_norm"] - r["all"]["dc_rank"]["test_worst_norm"] < -1.0)
            v = ""
            if m in methods and "layer_fair" in avg:
                v = method_verdict(a_, avg["layer_fair"][0], w_, drops, capped)
                verdicts[(label, m)] = v
                if m.startswith("entropy"):
                    v += " (same rule, for comparison)"
            lines.append(f"| {m} | {a_:+.2f} | {w_:+.2f} | {drops} | {v} |")
        if ("all" if label == "all" else "core", "agree_floor") in verdicts:
            lines += ["", f"**Primary verdict ({label}, agree_floor): {verdicts[(label, 'agree_floor')]}**"
                      + (f"; secondary agree_relfair: {verdicts.get((label, 'agree_relfair'), '—')}" ) ]
    lines += ["", "Rules: STRONG = Δmean ≥ 50% of 6b layer_fair Δmean AND Δworst ≥ +3 pp AND no context loses > 1 pp on "
              "its worst task; VIABLE = same with Δworst > 0; PARTIAL = positive Δmean otherwise; NOT VIABLE = "
              "Δmean ≤ 0. The decision is on all 7 contexts; core is reported separately.", "",
              "## Chosen coefficients (task × depth group)", ""]
    for r in rl_:
        for m in methods:
            if m not in r["all"]:
                continue
            sp = rl.Space(r["tasks"], GROUPS)
            lines += [f"**{r['context']} — {m}** (α = {r['alpha']:g})", ""] + sp.table(r["all"][m]["c12"]) + [""]
    (out / "method_summary.md").write_text("\n".join(lines) + "\n")
    _csv(out / "method.csv", rows, ("context", "method"))
    write_json(out / "method.json", {"verdict_all": verdicts.get(("all", "agree_floor")) if len(rl_) == 7 else None,
                                     "verdict_core": verdicts.get(("core", "agree_floor")),
                                     "verdicts": {f"{k[0]}:{k[1]}": v for k, v in verdicts.items()},
                                     "contexts": order})
    print(f"wrote {out / 'method_summary.md'}", flush=True)


def summarize_all8(out):
    p = out / "all8" / "all8_result.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    r, b = d["results"]["agree_floor"], d["reference"]["dc_rank"]
    lines = ["# Optional: agree_floor on all8 (24 coordinates, TEST)", "",
             f"| method | test mean | test worst | Δmean vs dc_rank | Δworst vs dc_rank |", "|---|---:|---:|---:|---:|",
             f"| dc_rank (this machine, α = {d['alpha']:g}) | {b['test_mean_norm']:.2f} | {b['test_worst_norm']:.2f} | | |",
             f"| agree_floor | {r['test_mean_norm']:.2f} | {r['test_worst_norm']:.2f} | "
             f"{r['test_mean_norm'] - b['test_mean_norm']:+.2f} | {r['test_worst_norm'] - b['test_worst_norm']:+.2f} |",
             "", f"Archived dc_official all8 (Exp 2, α = 0.4): test mean normalized {ALL8_DC_OFFICIAL_TEST_MEAN}."]
    (out / "all8_summary.md").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------
# 7d wide grid (descriptive)
# ----------------------------------------------------------------------------
def cmd_wide(args):
    rt = ro.load_runtime(args)
    experts = Experts(rt, args)
    split_note("wide", "labelled oracle: selected on VALIDATION, each chosen vector scored ONCE on TEST; descriptive")
    for cid in contexts_of(args.contexts):
        if cid not in CORE:
            raise SystemExit("7d is core only")
        p = args.out / cid / "wide_result.json"
        if p.exists():
            print(f"[{cid}] wide done, skipping", flush=True)
            continue
        t0 = time.time()
        ctx = Ctx(rt, args, cid)
        run_checks(args, ctx, experts)
        _, ref = load_reference(args, cid, ctx.tasks)
        _, n0 = ctx.val(ctx.start)
        floor = mean_worst(n0)[1]
        grids = [grid_with(GRID_WIDE, ctx.alpha)] * len(ctx.space.coords)
        labels = [f"{t}@{g}" for t, g in ctx.space.coords]
        log, results = [], {}
        for o in ("mean", "fair"):
            start = ref[f"layer_{o}"]["c12"]
            c, val, passes = coordinate_ascent(labelled_value(ctx, lambda v: v, o, floor), start, grids,
                                               WIDE_MAX_PASSES, log, f"wide_{o}", labels)
            results[f"wide_{o}"] = {**score_chosen(ctx, c, f"wide_{o}"), "passes": passes, "start": start}
        write_json(p, {"context": cid, "tasks": ctx.tasks, "alpha": ctx.alpha, "grid": grids[0],
                       "max_passes": WIDE_MAX_PASSES, "val_evals": len(ctx.val.cache), "results": results,
                       "reference": ref_block(ctx, ref), "test_log": ctx.test.log, "search_log": log,
                       "reference_mode": args.reference})
        print(f"[{cid}] wide done in {(time.time()-t0)/60:.1f} min", flush=True)
    summarize_wide(args.out)


def summarize_wide(out):
    files = sorted(out.glob("*/wide_result.json"))
    if not files:
        return
    lines = ["# Experiment 7d: widened grid {0.0,…,1.5} for the per-depth oracle (core, descriptive)", "",
             "Started from the 6b chosen vectors; passes until no improvement (cap 5). Extra = wide − 6b, test. "
             "Shrinkage = validation gain − test gain vs dc_rank.", "",
             "| context | objective | passes | 6b test mean / worst | wide test mean / worst | extra Δmean | extra Δworst | "
             "shrink mean | shrink worst |", "|---|---|---:|---|---|---:|---:|---:|---:|"]
    extra = []
    for p in files:
        d = json.loads(p.read_text())
        b, ref = d["reference"]["dc_rank"], d["reference"]
        for o in ("mean", "fair"):
            w, s6 = d["results"][f"wide_{o}"], ref[f"layer_{o}"]
            em, ew = w["test_mean_norm"] - s6["test_mean_norm"], w["test_worst_norm"] - s6["test_worst_norm"]
            sm = (w["val_mean_norm"] - b["val_mean_norm"]) - (w["test_mean_norm"] - b["test_mean_norm"])
            sw = (w["val_worst_norm"] - b["val_worst_norm"]) - (w["test_worst_norm"] - b["test_worst_norm"])
            extra.append((em, ew))
            lines.append(f"| {d['context']} | layer_{o} | {w['passes']} | {s6['test_mean_norm']:.2f} / "
                         f"{s6['test_worst_norm']:.2f} | {w['test_mean_norm']:.2f} / {w['test_worst_norm']:.2f} | "
                         f"{em:+.2f} | {ew:+.2f} | {sm:+.2f} | {sw:+.2f} |")
    am = sum(e[0] for e in extra) / len(extra)
    aw = sum(e[1] for e in extra) / len(extra)
    lines += ["", f"Average extra test gain over 6b: {am:+.2f} mean / {aw:+.2f} worst."]
    if max(am, aw) > 2.0:
        lines.append("**The extra test gain exceeds 2 pp: the 6b numbers are a lower bound on the per-depth oracle.**")
    lines += ["", "Chosen vectors:", ""]
    for p in files:
        d = json.loads(p.read_text())
        sp = rl.Space(d["tasks"], GROUPS)
        for o in ("mean", "fair"):
            lines += [f"**{d['context']} — wide_{o}**", ""] + sp.table(d["results"][f"wide_{o}"]["c12"]) + [""]
    (out / "wide_summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'wide_summary.md'}", flush=True)


def cmd_summarize(args):
    summarize_replay(args.out)
    summarize_gate(args.out)
    summarize_controls(args.out)
    summarize_method(args.out)
    summarize_wide(args.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("replay", "checks", "gate", "controls", "method", "entropy", "wide", "all8"):
        p = sub.add_parser(name)
        for a in ("--repo", "--model", "--ft-dir", "--heads", "--data", "--exp3r", "--exp6", "--out"):
            p.add_argument(a, type=Path, required=True)
        p.add_argument("--contexts", default="all8" if name == "all8" else "core")
        p.add_argument("--reference", choices=("archive", "current"), required=True)
        p.add_argument("--n-unlabeled", type=int, default=500)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--device", default="cuda")
        p.add_argument("--check-agree-vectors", type=int, default=8,
                       help="Exp 3 vectors whose agreement is compared with 6a's cache (check 4)")
        p.add_argument("--recompute-experts", action="store_true",
                       help="check 5: recompute expert predictions on this machine and require identical argmax")
        p.add_argument("--agree-tol-images", type=int, default=0,
                       help="check 4: allowed |agreement difference| vs 6a's cache, in images (default 0 = exact; "
                            "raise only after reporting a failure, e.g. near-tie flips on a new machine)")
        p.add_argument("--all8-alpha", type=float, default=0.4)
    s = sub.add_parser("summarize")
    s.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out = args.out.resolve()
    if args.cmd == "summarize":
        cmd_summarize(args)
        return
    for a in ("repo", "model", "ft_dir", "heads", "data", "exp3r", "exp6"):
        setattr(args, a, getattr(args, a).resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    record_config(args)
    {"replay": cmd_replay, "checks": cmd_checks, "gate": cmd_gate, "controls": cmd_controls,
     "method": lambda a: cmd_proxy_search(a, "method"), "entropy": lambda a: cmd_proxy_search(a, "entropy"),
     "wide": cmd_wide, "all8": lambda a: cmd_proxy_search(a, "all8")}[args.cmd](args)


if __name__ == "__main__":
    main()
