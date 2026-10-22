#!/usr/bin/env python3
"""Current-environment replay of Experiment 3's reference vectors (for Experiment 6).

Why: the 6b must-hold checks compare against Experiment 3's numbers within 0.05 pp.  On a
different machine a few near-tie predictions flip (e.g. 2 GTSRB validation images = 0.079
pp), exactly the numerical sensitivity documented in the Experiment 3 baseline audit.
Experiment 3 handled the same problem with a current-environment baseline overlay; this
script does the same for Experiment 6.

What it does, per context, with Experiment 3's OWN code path (lanes.LaneMerge.compose and
run_oracle.Evaluator, unchanged):
  * evaluates on THIS machine the three Experiment 3 reference vectors - dc_rank (uniform
    alpha), oracle_mean and oracle_fair (Exp 3's validation-chosen vectors, NOT re-searched) -
    on validation and test;
  * writes an Experiment-3-shaped folder (oracle_result.json, val_cache.json,
    test_cache.json) whose dc_rank / oracle_mean / oracle_fair numbers are the replayed ones
    (CT stays the archived Experiment 2 record, marked as such);
  * records every per-task difference from the archive, and aborts if any exceeds
    --max-archive-diff (default 0.5 pp: larger than numerical flips, so something else
    would be wrong);
  * repeats one evaluation without the cache to confirm this machine is deterministic.

Nothing is selected here and no test number is used for any choice.  Use the output folder
as --exp3 for run_layer_oracle.py (both subcommands) and for run_agreement.py `run`.  Keep
the ARCHIVED Experiment 3 folder for run_agreement.py `check` (the gate needs all of Exp 3's
cached vectors, which are internally consistent with each other).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_official as ro  # noqa: E402

FOUR = list(ro.CONTEXTS)[:7]
CORE = list(ro.CONTEXTS)[:3]


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def key_of(c):
    return ",".join(f"{x:g}" for x in c)


def contexts_of(spec):
    ids = CORE if spec == "core" else (FOUR if spec == "four" else spec.split(","))
    bad = [c for c in ids if c not in FOUR]
    if bad:
        raise SystemExit(f"unknown contexts {bad}")
    return ids


def runtime_info(torch):
    info = {"torch": getattr(torch, "__version__", "?")}
    try:
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001  (CPU smoke test)
        pass
    return info


def cmd_run(args):
    import lanes
    import run_oracle as rx
    rt = ro.load_runtime(args)
    torch = rt["torch"]
    print("[split] replay: Experiment 3's already-chosen vectors are re-evaluated on VALIDATION and TEST "
          "on this machine; nothing is selected", flush=True)
    det_done = False
    for cid in contexts_of(args.contexts):
        res_path = args.out / cid / "oracle_result.json"
        if res_path.exists():
            print(f"[{cid}] done, skipping", flush=True)
            continue
        t0 = time.time()
        src = args.exp3 / cid
        arch = json.loads((src / "oracle_result.json").read_text())
        arch_v = json.loads((src / "val_cache.json").read_text())
        arch_t = json.loads((src / "test_cache.json").read_text())
        tasks = list(arch["tasks"])
        if tasks != list(ro.CONTEXTS[cid]):
            raise SystemExit(f"[{cid}] task order differs from run_official.CONTEXTS")
        alpha = float(arch["dc_rank_alpha"])
        with torch.no_grad():
            lm = lanes.LaneMerge(ro.group(rt["deltas"], tasks), rt["backend"], tasks, ro.dc_kwargs("rank"))
        val = rx.Evaluator(rt, lm, tasks, args.device, "val", rt["refs"]["val"], args.out / cid / "val_cache.json")
        test = rx.Evaluator(rt, lm, tasks, args.device, "test", rt["refs"]["test"], args.out / cid / "test_cache.json")
        vecs = {"dc_rank": [alpha] * len(tasks),
                "oracle_mean": [float(x) for x in arch["val_chosen"]["mean"]["c"]],
                "oracle_fair": [float(x) for x in arch["val_chosen"]["fair"]["c"]]}
        new = copy.deepcopy(arch)
        diffs, worst = {}, {"val": 0.0, "test": 0.0}
        for name, c in vecs.items():
            k = key_of(c)
            acc_v, n_v = val(c)
            acc_t, n_t = test(c)
            dv = {t: acc_v[t] - arch_v[k][t] for t in tasks}
            dt = {t: acc_t[t] - arch_t[k][t] for t in tasks}
            diffs[name] = {"vector": k, "val_acc_diff_pp": dv, "test_acc_diff_pp": dt}
            worst["val"] = max(worst["val"], max(abs(x) for x in dv.values()))
            worst["test"] = max(worst["test"], max(abs(x) for x in dt.values()))
            mv, wv = sum(n_v.values()) / len(n_v), min(n_v.values())
            mt, wt = sum(n_t.values()) / len(n_t), min(n_t.values())
            new["test"][name].update({"test_acc": acc_t, "test_norm": n_t, "test_mean_norm": mt,
                                      "test_worst_norm": wt, "test_worst_task": min(n_t, key=n_t.get)})
            if name == "dc_rank":
                new["val_baseline"] = {"mean_norm": mv, "worst_norm": wv}
            else:
                o = name.split("_")[1]
                new["val_chosen"][o]["val_mean_norm"] = mv
                new["val_chosen"][o]["val_worst_norm"] = wv
            print(f"  [{cid}] {name:12s} c={k:20s} max |val diff| {max(abs(x) for x in dv.values()):.3f} pp, "
                  f"max |test diff| {max(abs(x) for x in dt.values()):.3f} pp vs archive", flush=True)
        fair_ok = new["val_chosen"]["fair"]["val_worst_norm"] >= new["val_baseline"]["worst_norm"] - 1e-9
        det = None
        if not det_done:
            # identical evaluation twice, bypassing the cache: this machine must be deterministic
            fn = ro.make_evaluate(rt, tasks, args.device, "val")
            merged = lm.compose({t: alpha for t in tasks})
            a1, a2 = fn(merged, 1.0), fn(merged, 1.0)
            det = {"vector": key_of(vecs["dc_rank"]), "max_abs_diff_pp": max(abs(a1[t] - a2[t]) for t in tasks),
                   "matches_cached": max(abs(a1[t] - val.cache[key_of(vecs["dc_rank"])][t]) for t in tasks)}
            det_done = True
            print(f"  [{cid}] determinism: repeat {det['max_abs_diff_pp']:.3f} pp, vs cached "
                  f"{det['matches_cached']:.3f} pp", flush=True)
            if det["max_abs_diff_pp"] != 0.0 or det["matches_cached"] != 0.0:
                raise SystemExit(f"[{cid}] this machine is not deterministic for a repeated evaluation; stop and report")
        new["replay"] = {
            "source": str(src), "note": "dc_rank/oracle_mean/oracle_fair re-evaluated on this machine with "
            "Experiment 3's code path; ct is the archived Experiment 2 record (not re-evaluated)",
            "runtime": runtime_info(torch), "lane_checks": lm.checks, "diffs_vs_archive": diffs,
            "max_abs_val_diff_pp": worst["val"], "max_abs_test_diff_pp": worst["test"],
            "oracle_fair_still_meets_val_floor": fair_ok, "determinism": det,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if max(worst.values()) > args.max_archive_diff:
            write_json(args.out / cid / "replay_FAILED.json", new["replay"])
            raise SystemExit(f"[{cid}] replay differs from the archive by {max(worst.values()):.3f} pp "
                             f"(> {args.max_archive_diff} pp): more than numerical flips; stop and report")
        write_json(res_path, new)
        print(f"[{cid}] replay done in {(time.time()-t0)/60:.1f} min; max |diff| vs archive: "
              f"val {worst['val']:.3f} pp, test {worst['test']:.3f} pp; oracle_fair still meets its "
              f"validation floor: {fair_ok}", flush=True)
    summarize(args.out, args.exp3)


def summarize(out, exp3):
    files = sorted(out.glob("*/oracle_result.json"))
    lines = ["# Experiment 3 current-environment replay (for Experiment 6)", "",
             f"Archive: `{exp3}`. Experiment 3's chosen vectors re-evaluated on this machine with Experiment 3's "
             "code path; nothing re-selected. Differences are this machine minus archive, in accuracy pp.", "",
             "| context | vector | max |val diff| | max |test diff| | test mean norm: archive → replay | "
             "test worst norm: archive → replay |", "|---|---|---:|---:|---|---|"]
    for p in files:
        d = json.loads(p.read_text())
        a = json.loads((exp3 / d["context"] / "oracle_result.json").read_text())
        for name, df in d["replay"]["diffs_vs_archive"].items():
            mv = max(abs(x) for x in df["val_acc_diff_pp"].values())
            mt = max(abs(x) for x in df["test_acc_diff_pp"].values())
            lines.append(f"| {d['context']} | {name} | {mv:.3f} | {mt:.3f} | "
                         f"{a['test'][name]['test_mean_norm']:.2f} → {d['test'][name]['test_mean_norm']:.2f} | "
                         f"{a['test'][name]['test_worst_norm']:.2f} → {d['test'][name]['test_worst_norm']:.2f} |")
    det = [json.loads(p.read_text())["replay"]["determinism"] for p in files]
    det = [x for x in det if x]
    if det:
        lines += ["", f"Determinism (same evaluation twice on this machine): max diff "
                  f"{det[0]['max_abs_diff_pp']:.3f} pp."]
    (out / "replay_summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'replay_summary.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ("--repo", "--model", "--ft-dir", "--heads", "--data", "--exp3", "--out"):
        ap.add_argument(a, type=Path, required=True)
    ap.add_argument("--contexts", default="four", help="core, four (all 7), or a comma list")
    ap.add_argument("--max-archive-diff", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    for a in ("repo", "model", "ft_dir", "heads", "data", "exp3", "out"):
        setattr(args, a, getattr(args, a).resolve())
    if args.out == args.exp3:
        raise SystemExit("--out must be a new folder, not the archived Experiment 3 folder")
    cmd_run(args)


if __name__ == "__main__":
    main()
