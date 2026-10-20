#!/usr/bin/env python3
"""Run only Experiment 3's baseline reproduction checks; do not search."""
from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--ft-dir", type=Path, required=True)
    ap.add_argument("--heads", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--exp2", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--contexts", choices=("core", "four"), default="four")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    for name in ("repo", "model", "ft_dir", "heads", "data", "exp2", "out"):
        setattr(args, name, getattr(args, name).resolve())

    lane_dir = args.repo / "lane_harm"
    sys.path.insert(0, str(lane_dir))
    import lanes
    import run_oracle as oracle

    runtime_args = Namespace(
        repo=args.repo,
        model=args.model,
        ft_dir=args.ft_dir,
        heads=args.heads,
        data=args.data,
        device=args.device,
    )
    rt = oracle.ro.load_runtime(runtime_args)
    contexts = oracle.CORE if args.contexts == "core" else oracle.FOUR
    failures = []

    for cid in contexts:
        tasks = list(oracle.ro.CONTEXTS[cid])
        rank_path = args.exp2 / "compare" / f"{cid}__dc_rank.json"
        rank_json = json.loads(rank_path.read_text())
        alpha = float(rank_json["selected_coef"])
        print(f"[{cid}] baseline alpha = {alpha:g}", flush=True)

        with rt["torch"].no_grad():
            lm = lanes.LaneMerge(
                oracle.ro.group(rt["deltas"], tasks),
                rt["backend"],
                tasks,
                oracle.ro.dc_kwargs("rank"),
            )
        val = oracle.Evaluator(
            rt, lm, tasks, args.device, "val", rt["refs"]["val"],
            args.out / cid / "val_cache.json",
        )
        test = oracle.Evaluator(
            rt, lm, tasks, args.device, "test", rt["refs"]["test"],
            args.out / cid / "test_cache.json",
        )
        base = [alpha] * len(tasks)

        acc_v, _ = val(base)
        exp2_val = next(
            row["acc"] for row in rank_json["val_sweep"]
            if abs(float(row["coef"]) - alpha) < 1e-9
        )
        val_diffs = {t: abs(acc_v[t] - exp2_val[t]) for t in tasks}
        val_task, val_diff = max(val_diffs.items(), key=lambda item: item[1])
        if val_diff > 0.1:
            message = (
                f"[{cid}] BASELINE GATE FAILED: val diff {val_diff:.3f} pp "
                f"({val_task}); limit is 0.1 pp"
            )
            print(message, flush=True)
            return 2

        acc_t, _ = test(base)
        test_diffs = {t: abs(acc_t[t] - rank_json["test_acc"][t]) for t in tasks}
        test_task, test_diff = max(test_diffs.items(), key=lambda item: item[1])
        print(
            f"[{cid}] baseline vs Experiment 2a: val diff {val_diff:.3f} pp, "
            f"test diff {test_diff:.3f} pp; lane sum error "
            f"{lm.checks['max_rel_sum_error']:.2e}",
            flush=True,
        )
        if test_diff > 0.1:
            print(
                f"[{cid}] BASELINE GATE FAILED: test diff {test_diff:.3f} pp "
                f"({test_task}); limit is 0.1 pp",
                flush=True,
            )
            return 2

    print(f"BASELINE GATE PASS: {len(contexts)} contexts", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
