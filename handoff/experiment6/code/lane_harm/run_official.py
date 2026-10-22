#!/usr/bin/env python3
"""Lane-harm experiment on the OFFICIAL DC-Merge repository (LoRA, ViT-B/32, 8 tasks).

Uses only the official code: config ``vitB32_r16_8task``, ``prepare_experiment_config``,
``get_ft_parameters_delta``, ``apply_merge``, ``evaluate_cliphead`` and the official
``dc_merge`` helpers (through ``dc_trace``, verified identical on every merge).

Subcommands
  select-alpha   official alpha sweep (0.1..3.0, mean normalized val accuracy) for one
                 context; with --with-test also evaluates the test split at the best alpha
                 (for the 8-task merge this reproduces the paper's Table 1 protocol).
  run            baseline merge + six interventions per target task, per context.
  summarize      rebuild interventions.csv and results_summary.md from finished contexts.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS = ("stanford_cars", "dtd", "eurosat", "gtsrb", "mnist", "resisc45", "sun397", "svhn")
CONFIG_NAME = "vitB32_r16_8task"
CONTEXTS = OrderedDict([
    # EuroSAT + GTSRB + SVHN: the cases where EuroSAT was hurt in our earlier runs
    ("dtd-eurosat-gtsrb-svhn", ("dtd", "eurosat", "gtsrb", "svhn")),
    ("eurosat-gtsrb-mnist-svhn", ("eurosat", "gtsrb", "mnist", "svhn")),
    ("eurosat-gtsrb-sun397-svhn", ("eurosat", "gtsrb", "sun397", "svhn")),
    # controls: EuroSAT + one of GTSRB/SVHN
    ("stanford_cars-eurosat-gtsrb-mnist", ("stanford_cars", "eurosat", "gtsrb", "mnist")),
    ("stanford_cars-eurosat-gtsrb-resisc45", ("stanford_cars", "eurosat", "gtsrb", "resisc45")),
    # controls: EuroSAT + neither
    ("stanford_cars-eurosat-mnist-resisc45", ("stanford_cars", "eurosat", "mnist", "resisc45")),
    ("stanford_cars-eurosat-mnist-sun397", ("stanford_cars", "eurosat", "mnist", "sun397")),
    # the published setting
    ("all8", TASKS),
])
PRESETS = {
    "eurosat": list(CONTEXTS)[:7],
    "all8": ["all8"],
}
MODES = ("protect_lane", "fix_conflict", "undilute", "block_injection",
         "restore_own_update", "control_matched_lane")
ALPHAS = None  # set from numpy at runtime: np.arange(0.1, 3.1, 0.1), exactly as eval.py


# ----------------------------------------------------------------------------
# official runtime
# ----------------------------------------------------------------------------
def load_runtime(args):
    import torch
    vision = args.repo / "vision_lora_merge"
    os.chdir(vision)                       # official configs resolve shuffled_idxs via os.getcwd()
    sys.path.insert(0, str(vision))
    sys.path.insert(0, str(HERE))
    from utils import (get_config_from_name, prepare_experiment_config, set_seed,
                       evaluate_cliphead, get_clip_encodings)
    from ft_handlers import get_ft_parameters_delta, apply_merge
    import merging_functions as mf
    import dataset.configs as dataset_configs
    import dc_trace

    set_seed(400)                          # official BIGSEED
    for t in TASKS:                        # official file has BASE_DIR = '' (fill-in)
        getattr(dataset_configs, t)["root"] = str(args.data)
    raw = get_config_from_name(CONFIG_NAME, device=args.device)
    raw["model"]["cachedir"] = str(args.model)
    raw["model"]["bases"] = [str(args.ft_dir / t) for t in TASKS]
    for cfg in raw["dataset"]:
        cfg["clip_encodings"] = str(args.heads / "ViT-B-32" / f"{cfg['name']}_head.pt")
    if [c["name"] for c in raw["dataset"]] != list(TASKS):
        raise RuntimeError("official config task order changed")
    heads = {c["name"]: get_clip_encodings(c["clip_encodings"]) for c in raw["dataset"]}
    config = prepare_experiment_config(raw)
    with torch.no_grad():
        deltas = {t: get_ft_parameters_delta(m.cpu()) for t, m in zip(TASKS, config["models"]["bases"])}
    refs = json.loads((vision / "single_task" / "our_finetuned" / "ViT-B-32" / "val_acc.json").read_text())
    refs_test = json.loads((vision / "single_task" / "our_finetuned" / "ViT-B-32" / "test_acc.json").read_text())
    return {"torch": torch, "mf": mf, "backend": dc_trace.make_backend(mf), "deltas": deltas,
            "heads": heads, "data": config["data"], "new": config["models"]["new"],
            "apply_merge": apply_merge, "evaluate_cliphead": evaluate_cliphead,
            "refs": {"val": refs, "test": refs_test}}


def group(deltas, tasks):
    keys = list(deltas[tasks[0]])
    return OrderedDict(sorted((k, [deltas[t][k] for t in tasks]) for k in keys))


def make_evaluate(rt, tasks, device, split="val"):
    torch = rt["torch"]

    def evaluate(merged, alpha):
        with torch.no_grad():
            model = rt["apply_merge"](rt["new"], merged, scaling_coeffs=alpha).to(device)
        out = {}
        try:
            for t in tasks:
                loader = rt["data"][TASKS.index(t)]["test"][split]
                acc = rt["evaluate_cliphead"](model, loader, class_vectors=rt["heads"][t].to(device))
                out[t] = 100.0 * float(acc)
        finally:
            model.cpu()
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        return out
    return evaluate


def dc_kwargs(topk):
    kw = {"smoothing_strategy": "linear", "rho": 5.0}   # README default for ViT-B-32 8-task
    if topk == "rank":
        kw["topk"] = "rank"
    return kw


def selected(args):
    ids = []
    for p in args.preset or []:
        ids += PRESETS[p]
    ids += args.context or []
    ids = list(dict.fromkeys(ids))
    bad = [c for c in ids if c not in CONTEXTS]
    if bad:
        raise SystemExit(f"unknown contexts {bad}; choose from {list(CONTEXTS)}")
    if not ids:
        raise SystemExit("no contexts selected (use --preset or --context)")
    return ids


# ----------------------------------------------------------------------------
# select-alpha (official protocol)
# ----------------------------------------------------------------------------
def cmd_select_alpha(args):
    import numpy as np
    rt = load_runtime(args)
    torch = rt["torch"]
    tasks = list(CONTEXTS[args.context_for_alpha])
    grouped = group(rt["deltas"], tasks)
    with torch.no_grad():
        if args.topk == "official":
            merged = rt["mf"].dc_merge(grouped, smoothing_strategy="linear", rho=5.0)
        else:
            merged = OrderedDict((k, rt["backend"]._dc_merge_matrix_group(v, **dc_kwargs("rank")))
                                 for k, v in grouped.items())
    rows, best = [], (None, -1.0)
    for a in np.arange(0.1, 3.1, 0.1):
        acc = make_evaluate(rt, tasks, args.device, "val")(merged, a)
        norm = sum(acc[t] / rt["refs"]["val"][t] * 100 for t in tasks) / len(tasks)
        rows.append({"alpha": round(float(a), 1), "mean_norm_val": norm, **acc})
        print(f"alpha={a:.1f}  mean normalized val = {norm:.3f}", flush=True)
        if norm > best[1]:
            best = (float(a), norm)
    result = {"context": args.context_for_alpha, "topk": args.topk,
              "best_alpha": round(best[0], 1), "best_mean_norm_val": best[1], "sweep": rows}
    if args.with_test:
        acc = make_evaluate(rt, tasks, args.device, "test")(merged, best[0])
        result["test_accuracy"] = acc
        result["test_mean_acc"] = sum(acc.values()) / len(acc)
        result["test_mean_norm"] = sum(acc[t] / rt["refs"]["test"][t] * 100 for t in tasks) / len(tasks)
        print(f"TEST at alpha={best[0]:.1f}: mean acc {result['test_mean_acc']:.2f}, "
              f"mean normalized {result['test_mean_norm']:.2f}")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"alpha_selection_{args.context_for_alpha}_{args.topk}.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"best alpha = {result['best_alpha']}  (wrote {path})")


# ----------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------
def cmd_run(args):
    ids = selected(args)
    args.out.mkdir(parents=True, exist_ok=True)
    meta_path = args.out / "run_meta.json"
    meta = {"topk": args.topk, "alpha": args.alpha, "seed": args.seed, "config": CONFIG_NAME,
            "official_commit": args.commit, "harm_reference":
            args.harm_reference or ("standalone" if args.topk == "official" else "solo")}
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if {k: old.get(k) for k in ("topk", "alpha")} != {k: meta[k] for k in ("topk", "alpha")}:
            raise SystemExit(f"{args.out} was started with {old}; use a new --out")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"contexts ({len(ids)}): {', '.join(ids)}   topk={args.topk}  alpha={args.alpha}", flush=True)

    rt = load_runtime(args)
    torch = rt["torch"]
    sys.path.insert(0, str(HERE))
    import lane_harm as lh

    solo_path = args.out / "solo_accuracy.json"
    solo = json.loads(solo_path.read_text()) if solo_path.exists() else {}
    for cid in ids:
        tasks = list(CONTEXTS[cid])
        if args.resume and (args.out / cid / "context_report.json").exists():
            print(f"[{cid}] already done, skipping", flush=True)
            continue
        t0 = time.time()
        with torch.no_grad():
            rep = lh.run_context(group(rt["deltas"], tasks), {t: group(rt["deltas"], [t]) for t in tasks},
                                 rt["backend"], tasks, alpha=args.alpha,
                                 evaluate=make_evaluate(rt, tasks, args.device, "val"),
                                 out_dir=args.out / cid, dc_kwargs=dc_kwargs(args.topk),
                                 seed=args.seed, solo_accuracy=solo, modes=MODES)
        solo.update(rep["solo_accuracy"])
        solo_path.write_text(json.dumps(solo, indent=2))
        bv = rep["baseline_verification"]
        eq = bv.get("max_abs_diff_vs_dc_merge")
        print(f"[{cid}] done in {(time.time() - t0) / 60:.1f} min; baseline vs official dc_merge: "
              + ("identical" if eq == 0.0 else ("not compared (rank variant)" if eq is None else f"max diff {eq:g}")),
              flush=True)
    refs = json.loads((args.repo / "vision_lora_merge" / "single_task" / "our_finetuned"
                       / "ViT-B-32" / "val_acc.json").read_text())
    summarize(args.out, refs)


# ----------------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------------
def kind(tasks):
    s = set(tasks)
    if len(s) == 8:
        return "all8"
    if "eurosat" not in s:
        return "no-eurosat"
    return {2: "eurosat+gtsrb+svhn", 1: "eurosat+one", 0: "eurosat+neither"}[len({"gtsrb", "svhn"} & s)]


def verdict(rows, share_key):
    share = {m: (rows[m][share_key] if rows[m][share_key] is not None else 0.0) for m in rows}
    ctrl = share.get("control_matched_lane", 0.0)
    best = max((m for m in rows if m != "control_matched_lane"), key=lambda m: share[m])
    if rows[best]["harm_ref_pp"] <= 0:
        return "no harm to explain"
    if share[best] < 0.20:
        return "no isolated mechanism (cross-lane readout or downstream)"
    if ctrl >= 0.5 * share[best]:
        return f"{best}: not specific (control removes {ctrl:.0%})"
    benefit, cost = rows[best]["benefit_target_pp"], rows[best]["cost_mean_others_pp"]
    if benefit > 0 and cost >= 0.8 * benefit:
        return f"{best}: real cause, but fixing it moves the harm to others"
    if share[best] >= 0.5 and ctrl < 0.10:
        return f"cause = {best} (removes {share[best]:.0%}, control {ctrl:.0%})"
    return f"partial: {best} removes {share[best]:.0%} (control {ctrl:.0%})"


def summarize(out: Path, refs: dict):
    meta = json.loads((out / "run_meta.json").read_text())
    ref_kind = meta["harm_reference"]
    reports = sorted(out.glob("*/context_report.json"))
    if not reports:
        raise SystemExit(f"no context_report.json under {out}")
    csv_rows, by_ct = [], defaultdict(dict)
    for path in reports:
        cid, rep = path.parent.name, json.loads(path.read_text())
        for r in rep["interventions"]:
            t = r["target"]
            costs = list(r["cost_to_others_pp"].values())
            ws = r["weight_space_target"]
            ctx_acc = rep["context_accuracy"][t]
            harm_solo = r["harm_baseline_target"]
            harm_std = refs[t] - ctx_acc
            harm_ref = harm_std if ref_kind == "standalone" else harm_solo
            ben = r["benefit_target_pp"]
            row = {
                "context": cid, "kind": kind(rep["tasks"]), "target": t, "mode": r["mode"],
                "standalone_acc": refs[t], "solo_merge_acc": rep["solo_accuracy"][t], "context_acc": ctx_acc,
                "harm_vs_standalone_pp": harm_std, "harm_vs_solo_pp": harm_solo,
                "harm_ref_pp": harm_ref,
                "benefit_target_pp": ben,
                "share_removed": (ben / harm_ref) if harm_ref > 0 else None,
                "cost_mean_others_pp": sum(costs) / len(costs) if costs else 0.0,
                "cost_max_others_pp": max(costs) if costs else 0.0,
                "mean_change_pp": r["mean_change_pp"],
                "spill_increment": ws["stage_increment"]["lane_proj"],
                "ties_increment": ws["stage_increment"]["lane_ties"],
                "cross_lane_increment": ws["stage_increment"]["merged"],
                "leakage_ratio": ws["leakage_ratio"],
                "kept_per_task_total": ws["support_kept_total"],
                "checks_passed": r["verification"]["passed"],
                "baseline_equal_to_official": (rep["baseline_verification"].get("max_abs_diff_vs_dc_merge") == 0.0
                                               if meta["topk"] == "official" else None),
            }
            csv_rows.append(row)
            by_ct[(cid, t)][r["mode"]] = row
    with open(out / "interventions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)

    def f(x):
        return "—" if x is None else f"{x:.2f}"

    lines = ["# Lane-harm experiment on official DC-Merge: results summary", "",
             f"Operator: top-k = **{meta['topk']}**, alpha = {meta['alpha']}, official commit {meta['official_commit']}.",
             f"Harm reference: **{ref_kind}**.  Contexts: {len(reports)}.  "
             f"All consistency checks passed: {all(r['checks_passed'] for r in csv_rows)}."]
    if meta["topk"] == "official":
        lines.append("Baseline merge identical to official `dc_merge` in every row: "
                     f"{all(r['baseline_equal_to_official'] for r in csv_rows)}.")
    lines.append("")
    order = {"eurosat+gtsrb+svhn": 0, "eurosat+one": 1, "eurosat+neither": 2, "all8": 3, "no-eurosat": 4}
    for target in ["eurosat"] + [t for t in TASKS if t != "eurosat"]:
        keys = sorted([k for k in by_ct if k[1] == target], key=lambda k: (order[by_ct[k][MODES[0]]["kind"]], k[0]))
        if not keys:
            continue
        lines += [f"## Target: {target}", "",
                  "| context | kind | harm vs standalone | harm vs solo | " + " | ".join(MODES) + " | verdict |",
                  "|---|---|---:|---:|" + "---:|" * len(MODES) + "---|"]
        for cid, _ in keys:
            rows = by_ct[(cid, target)]
            r0 = rows[MODES[0]]
            cells = [f"{f(rows[m]['benefit_target_pp'])} / {f(rows[m]['cost_mean_others_pp'])}" for m in MODES]
            lines.append(f"| {cid} | {r0['kind']} | {f(r0['harm_vs_standalone_pp'])} | {f(r0['harm_vs_solo_pp'])} | "
                         + " | ".join(cells) + f" | {verdict(rows, 'share_removed')} |")
        lines += ["", "Cells: benefit to target (pp) / mean cost to the other tasks (pp).", ""]
    (out / "results_summary.md").write_text("\n".join(lines))
    print(f"wrote {out / 'interventions.csv'} and {out / 'results_summary.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--repo", type=Path, required=True, help="official DC-Merge clone (root)")
        p.add_argument("--model", type=Path, required=True, help="local clip-vit-base-patch32 directory")
        p.add_argument("--ft-dir", type=Path, required=True, help="lora_checkpoints dir with <task>/ subfolders")
        p.add_argument("--heads", type=Path, required=True, help="lora_heads dir containing ViT-B-32/<task>_head.pt")
        p.add_argument("--data", type=Path, required=True, help="dataset root (BASE_DIR)")
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--topk", choices=("official", "rank"), default="official")
        p.add_argument("--device", default="cuda")

    a = sub.add_parser("select-alpha")
    common(a)
    a.add_argument("--context-for-alpha", default="all8", choices=list(CONTEXTS))
    a.add_argument("--with-test", action="store_true")
    r = sub.add_parser("run")
    common(r)
    r.add_argument("--alpha", type=float, required=True, help="from select-alpha")
    r.add_argument("--preset", action="append", choices=sorted(PRESETS))
    r.add_argument("--context", action="append")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--resume", action="store_true")
    r.add_argument("--harm-reference", choices=("standalone", "solo"), default=None,
                   help="default: standalone for --topk official, solo for --topk rank")
    r.add_argument("--commit", default="089af77", help="official commit hash (recorded only)")
    s = sub.add_parser("summarize")
    s.add_argument("--repo", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.repo = args.repo.resolve()
    if args.cmd == "summarize":
        refs = json.loads((args.repo / "vision_lora_merge" / "single_task" / "our_finetuned"
                           / "ViT-B-32" / "val_acc.json").read_text())
        summarize(args.out.resolve(), refs)
        return
    args.out = args.out.resolve()
    for name in ("model", "ft_dir", "heads", "data"):
        setattr(args, name, getattr(args, name).resolve())
    (cmd_select_alpha if args.cmd == "select-alpha" else cmd_run)(args)


if __name__ == "__main__":
    main()
