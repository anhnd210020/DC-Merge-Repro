#!/usr/bin/env python
"""Experiment 7 operational check: does the CPU thread count change any accuracy?

The lane setup (DC-Merge SVDs) runs on the CPU, and a different thread count changes its last
floating-point bits. This script rebuilds the lanes with the CURRENT thread setting and
recomputes, WITHOUT the cache, validation accuracy of vectors whose cached values were computed
earlier on this machine with the default (64-thread) setting. Read-only for every Exp 7 cache;
never evaluates on test. Writes only OUT/<cid>/thread_check_t<T>.json.
Exit code 0 = every recomputed accuracy identical to its cached value; 3 = some difference.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_exp7 as r7  # noqa: E402

ro, fkey = r7.ro, r7.fkey


def threads(rt):
    f = getattr(rt["torch"], "get_num_threads", None)
    return f() if f else 0


def check_context(args, rt, cid):
    nthreads = threads(rt)
    t0 = time.time()
    ctx = r7.Ctx(rt, args, cid)
    setup_min = (time.time() - t0) / 60
    _, ref = r7.load_reference(args, cid, ctx.tasks)
    cache = ctx.val.cache
    vecs = [("dc_rank_uniform_alpha", ctx.start)]
    for name in ("layer_mean", "layer_fair"):
        if name in ref:
            vecs.append((f"exp6_{name}", [float(x) for x in ref[name]["c12"]]))
    seen = {fkey(v) for _, v in vecs}
    for k in sorted(cache):
        if len(vecs) >= args.max_vectors:
            break
        v = [float(x) for x in k.split(",")]
        if len(v) == 12 and k not in seen:
            vecs.append(("cached", v))
            seen.add(k)
    sizes = {}
    for t in ctx.tasks:
        try:
            sizes[t] = len(rt["data"][ro.TASKS.index(t)]["test"]["val"].dataset)
        except Exception:
            sizes[t] = 0
    rows, worst_pp = [], 0.0
    for name, v in vecs:
        k = fkey(v)
        if k not in cache:
            print(f"  [{cid}] {name}: not in the validation cache, skipped", flush=True)
            continue
        merged = ctx.lm.compose_groups(ctx.space.nested(v))
        acc = ctx.val.fn(merged, 1.0)
        diff = {t: acc[t] - cache[k][t] for t in ctx.tasks}
        imgs = {t: round(abs(diff[t]) * sizes[t] / 100.0) for t in ctx.tasks}
        mx = max(abs(x) for x in diff.values())
        worst_pp = max(worst_pp, mx)
        rows.append({"name": name, "key": k, "diff_pp": diff, "diff_images": imgs})
        print(f"  [{cid}] {name:22s} max |val diff| {mx:.4f} pp, images {sum(imgs.values())}", flush=True)
    rec = {"context": cid, "torch_threads": nthreads, "lane_setup_minutes": setup_min,
           "lane_sum_rel_error": ctx.lm.checks["max_rel_sum_error"], "vectors": rows,
           "max_abs_diff_pp": worst_pp, "identical": worst_pp == 0.0}
    r7.write_json(args.out / cid / f"thread_check_t{nthreads}.json", rec)
    print(f"[{cid}] threads={nthreads} lane setup {setup_min:.1f} min; "
          f"{len(rows)} vectors recomputed; max |diff| {worst_pp:.4f} pp -> "
          f"{'IDENTICAL' if worst_pp == 0.0 else 'DIFFERENT'}", flush=True)
    return worst_pp == 0.0


def main():
    ap = argparse.ArgumentParser()
    for a in ("--repo", "--model", "--ft-dir", "--heads", "--data", "--exp3r", "--exp6", "--out"):
        ap.add_argument(a, type=Path, required=True)
    ap.add_argument("--contexts", default="eurosat-gtsrb-sun397-svhn")
    ap.add_argument("--reference", choices=("archive", "current"), required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-vectors", type=int, default=6)
    ap.add_argument("--n-unlabeled", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for a in ("repo", "model", "ft_dir", "heads", "data", "exp3r", "exp6", "out"):
        setattr(args, a, getattr(args, a).resolve())
    rt = ro.load_runtime(args)
    print(f"torch threads = {threads(rt)}", flush=True)
    ok = all([check_context(args, rt, cid) for cid in r7.contexts_of(args.contexts)])
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
