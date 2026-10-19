#!/usr/bin/env python3
"""Leave-one-context-out transfer of Experiment 3 oracle_fair lane ratios.

For each held-out context, average c_i / alpha over the other saved Experiment 3
contexts containing task i, multiply by the held-out context's dc_rank alpha, and
evaluate that single vector on validation and test. No coefficient search occurs.
When a task appears only in the held-out context (DTD), use the neutral ratio 1.0
and mark it as unsupported by cross-context evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def key_of(values):
    return ",".join(f"{float(x):g}" for x in values)


def score(acc, tasks, refs):
    norm = {t: 100.0 * float(acc[t]) / float(refs[t]) for t in tasks}
    return {
        "accuracy": {t: float(acc[t]) for t in tasks},
        "normalized_accuracy": norm,
        "mean_norm": sum(norm.values()) / len(tasks),
        "worst_norm": min(norm.values()),
        "worst_task": min(norm, key=norm.get),
    }


def vector_equal(a, b, tasks, tol=1e-12):
    return all(abs(float(a[t]) - float(b[t])) <= tol for t in tasks)


def atomic_cache_write(path, cache):
    write_json(path, cache)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path("/workspace/ducanh/DC-Merge"))
    ap.add_argument("--model", type=Path, default=Path("/workspace/models/clip-vit-base-patch32"))
    ap.add_argument("--ft-dir", type=Path, default=Path("/workspace/models/DC-Merge-Vision-B32-r16-8task/lora_checkpoints"))
    ap.add_argument("--heads", type=Path, default=Path("/workspace/models/DC-Merge-Vision-B32-r16-8task/lora_heads"))
    ap.add_argument("--data", type=Path, default=Path("/workspace/datasets8"))
    ap.add_argument("--exp3", type=Path, default=Path("/srv/dcmerge/results/experiment3"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    repo, exp3, out = args.repo.resolve(), args.exp3.resolve(), args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to write into non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    lane_dir = repo / "lane_harm"
    sys.path.insert(0, str(lane_dir))
    import lanes
    import run_official as ro

    contexts = list(ro.CONTEXTS)[:7]
    records = {}
    for cid in contexts:
        p = exp3 / cid / "oracle_result.json"
        if not p.is_file():
            raise SystemExit(f"missing completed Experiment 3 context result: {p}")
        records[cid] = read_json(p)
    if set(records) != set(contexts):
        raise SystemExit("Experiment 3 contexts do not match the expected seven-context protocol")

    # Derive every held-out coefficient using only other contexts' saved oracle_fair
    # coefficients and their selected alpha. The original coefficients/results are read-only.
    transfer = {}
    for cid in contexts:
        d = records[cid]
        alpha = float(d["dc_rank_alpha"])
        cvec = {}
        details = {}
        for task in d["tasks"]:
            sources = []
            for other in contexts:
                if other == cid:
                    continue
                od = records[other]
                if task in od["tasks"]:
                    ratio = float(od["test"]["oracle_fair"]["c"][task]) / float(od["dc_rank_alpha"])
                    sources.append({"context": other, "ratio": ratio})
            mean_ratio = sum(x["ratio"] for x in sources) / len(sources) if sources else 1.0
            cvec[task] = alpha * mean_ratio
            details[task] = {
                "source_contexts": sources,
                "source_count": len(sources),
                "mean_c_over_alpha": mean_ratio,
                "transfer_coefficient": cvec[task],
                "fallback": "neutral ratio 1.0; no other context contains this task" if not sources else None,
            }
        transfer[cid] = {"alpha": alpha, "tasks": list(d["tasks"]), "coefficients": cvec, "ratios": details}

    # Record provenance and protocol before loading the model/runtime.
    source_hashes = {}
    for rel in ("lane_harm/run_oracle.py", "lane_harm/run_official.py", "lane_harm/lanes.py"):
        p = repo / rel
        source_hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    try:
        source_commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        source_commit = None
    metadata = {
        "experiment": "leave-one-context-out task-coefficient transfer",
        "coefficient_source": "oracle_fair",
        "formula": "c_transfer(task, heldout) = alpha_heldout * mean_{other contexts containing task}(c_oracle_fair / alpha_other)",
        "missing_history_policy": "neutral ratio 1.0; flagged; no value from the held-out context used",
        "contexts": contexts,
        "validation_and_test": "one evaluation of each transferred vector per split, reusing exact saved evaluations only when the coefficient vector matches an existing baseline/oracle vector",
        "searches_run": 0,
        "baseline_or_oracle_results_modified": False,
        "source_repo": str(repo),
        "source_commit": source_commit,
        "source_sha256": source_hashes,
        "experiment3_results": str(exp3),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "running",
    }
    write_json(out / "run_manifest.json", metadata)
    write_json(out / "transfer_coefficients.json", transfer)

    ns = SimpleNamespace(repo=repo, model=args.model.resolve(), ft_dir=args.ft_dir.resolve(),
                         heads=args.heads.resolve(), data=args.data.resolve(), device=args.device)
    rt = ro.load_runtime(ns)
    torch = rt["torch"]
    metadata["runtime"] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
    }
    write_json(out / "run_manifest.json", metadata)

    summary_rows = []
    for index, cid in enumerate(contexts, 1):
        d = records[cid]
        tasks = list(d["tasks"])
        alpha = float(d["dc_rank_alpha"])
        cvec = transfer[cid]["coefficients"]
        cvals = [float(cvec[t]) for t in tasks]
        base_vec = {t: alpha for t in tasks}
        fair_vec = {t: float(d["test"]["oracle_fair"]["c"][t]) for t in tasks}
        mean_vec = {t: float(d["test"]["oracle_mean"]["c"][t]) for t in tasks}

        # Load Experiment 3 caches read-only and verify their baseline/oracle entries.
        old_val = read_json(exp3 / cid / "val_cache.json")
        old_test = read_json(exp3 / cid / "test_cache.json")
        val_base_key = key_of([base_vec[t] for t in tasks])
        val_fair_key = key_of([fair_vec[t] for t in tasks])
        val_mean_key = key_of([mean_vec[t] for t in tasks])
        test_base_key = key_of([base_vec[t] for t in tasks])
        test_fair_key = key_of([fair_vec[t] for t in tasks])
        test_mean_key = key_of([mean_vec[t] for t in tasks])
        for cache, key, desc in ((old_val, val_base_key, "val baseline"), (old_val, val_fair_key, "val oracle_fair"),
                                 (old_val, val_mean_key, "val oracle_mean"), (old_test, test_base_key, "test baseline"),
                                 (old_test, test_fair_key, "test oracle_fair"), (old_test, test_mean_key, "test oracle_mean")):
            if key not in cache:
                raise SystemExit(f"{cid}: saved Experiment 3 cache lacks {desc} vector {key}")

        val_baseline = score(old_val[val_base_key], tasks, rt["refs"]["val"])
        val_fair = score(old_val[val_fair_key], tasks, rt["refs"]["val"])
        val_mean = score(old_val[val_mean_key], tasks, rt["refs"]["val"])
        test_baseline = score(old_test[test_base_key], tasks, rt["refs"]["test"])
        test_fair = score(old_test[test_fair_key], tasks, rt["refs"]["test"])
        test_mean = score(old_test[test_mean_key], tasks, rt["refs"]["test"])

        # Verify saved selected metrics agree with the original result JSON.
        for t in tasks:
            if abs(test_baseline["accuracy"][t] - d["test"]["dc_rank"]["test_acc"][t]) > 1e-10:
                raise SystemExit(f"{cid}: saved test baseline cache/result mismatch for {t}")
            if abs(test_fair["accuracy"][t] - d["test"]["oracle_fair"]["test_acc"][t]) > 1e-10:
                raise SystemExit(f"{cid}: saved oracle_fair cache/result mismatch for {t}")

        # Construct exactly one new LaneMerge for the held-out context.
        with torch.no_grad():
            lm = lanes.LaneMerge(ro.group(rt["deltas"], tasks), rt["backend"], tasks, ro.dc_kwargs("rank"))

        context_out = out / cid
        context_out.mkdir(parents=True, exist_ok=True)
        split_caches = {}
        sources = {}
        for split in ("val", "test"):
            transfer_cache_path = context_out / f"{split}_cache.json"
            transfer_cache = read_json(transfer_cache_path) if transfer_cache_path.exists() else {}
            k = key_of(cvals)
            if k in transfer_cache:
                acc = transfer_cache[k]
                sources[split] = "transfer cache"
            else:
                # Reuse an exact pre-existing baseline or oracle evaluation if the full
                # vector matches. No rounded-key cache hit is accepted on its own.
                known = [("dc_rank", base_vec), ("oracle_fair", fair_vec), ("oracle_mean", mean_vec)]
                match = next((name for name, vec in known if vector_equal(cvec, vec, tasks)), None)
                if match is not None:
                    old_cache = old_val if split == "val" else old_test
                    known_key = key_of([({"dc_rank": base_vec, "oracle_fair": fair_vec, "oracle_mean": mean_vec}[match])[t]
                                       for t in tasks])
                    acc = old_cache[known_key]
                    sources[split] = f"exact saved Experiment 3 {match} cache"
                else:
                    evaluate = ro.make_evaluate(rt, tasks, args.device, split)
                    merged = lm.compose(cvec)
                    acc = evaluate(merged, 1.0)
                    sources[split] = "new evaluation"
                transfer_cache[k] = acc
                atomic_cache_write(transfer_cache_path, transfer_cache)
            split_caches[split] = score(acc, tasks, rt["refs"][split])

        result = {
            "context": cid,
            "tasks": tasks,
            "alpha": alpha,
            "transfer_coefficients": cvec,
            "coefficient_ratios": transfer[cid]["ratios"],
            "evaluation_source": sources,
            "lane_checks": lm.checks,
            "validation": {
                "dc_rank": val_baseline,
                "loco_transfer": split_caches["val"],
                "oracle_fair": val_fair,
                "oracle_mean": val_mean,
                "delta_mean_norm_vs_dc_rank": split_caches["val"]["mean_norm"] - val_baseline["mean_norm"],
                "delta_worst_norm_vs_dc_rank": split_caches["val"]["worst_norm"] - val_baseline["worst_norm"],
                "delta_mean_norm_vs_oracle_fair": split_caches["val"]["mean_norm"] - val_fair["mean_norm"],
                "delta_worst_norm_vs_oracle_fair": split_caches["val"]["worst_norm"] - val_fair["worst_norm"],
            },
            "test": {
                "dc_rank": test_baseline,
                "loco_transfer": split_caches["test"],
                "oracle_fair": test_fair,
                "oracle_mean": test_mean,
                "delta_mean_norm_vs_dc_rank": split_caches["test"]["mean_norm"] - test_baseline["mean_norm"],
                "delta_worst_norm_vs_dc_rank": split_caches["test"]["worst_norm"] - test_baseline["worst_norm"],
                "delta_mean_norm_vs_oracle_fair": split_caches["test"]["mean_norm"] - test_fair["mean_norm"],
                "delta_worst_norm_vs_oracle_fair": split_caches["test"]["worst_norm"] - test_fair["worst_norm"],
            },
        }
        write_json(context_out / "transfer_result.json", result)
        summary_rows.append(result)
        print(f"[{index}/7] {cid}: val Δmean={result['validation']['delta_mean_norm_vs_dc_rank']:+.3f} "
              f"Δworst={result['validation']['delta_worst_norm_vs_dc_rank']:+.3f}; "
              f"test Δmean={result['test']['delta_mean_norm_vs_dc_rank']:+.3f} "
              f"Δworst={result['test']['delta_worst_norm_vs_dc_rank']:+.3f}; sources={sources}", flush=True)

    methods = ("dc_rank", "loco_transfer", "oracle_fair")
    avg = {}
    for split in ("validation", "test"):
        avg[split] = {}
        for method in methods:
            avg[split][method] = {
                metric: sum(row[split][method][metric] for row in summary_rows) / len(summary_rows)
                for metric in ("mean_norm", "worst_norm")
            }
    lines = [
        "# Leave-one-context-out transfer of task coefficients",
        "",
        "Ratios are the unweighted mean of saved `oracle_fair` c_i/alpha values in other contexts containing task i. "
        "The held-out context's alpha scales that mean. No new search was run. DTD has no other context in the seven-context set, so its ratio is neutral 1.0 and is flagged as unsupported.",
        "",
        "## Test results",
        "",
        "All accuracy scores below are mean or worst-task normalized accuracy; deltas are percentage points.",
        "",
        "| Context | Transfer coefficients | Transfer Δ mean vs dc_rank | Transfer Δ worst vs dc_rank | Transfer Δ mean vs oracle_fair | Transfer Δ worst vs oracle_fair |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        coeffs = ", ".join(f"{t}={row['transfer_coefficients'][t]:.4f}" for t in row["tasks"])
        x = row["test"]
        lines.append(f"| {row['context']} | {coeffs} | {x['delta_mean_norm_vs_dc_rank']:+.3f} | "
                     f"{x['delta_worst_norm_vs_dc_rank']:+.3f} | {x['delta_mean_norm_vs_oracle_fair']:+.3f} | "
                     f"{x['delta_worst_norm_vs_oracle_fair']:+.3f} |")
    lines += ["", "## Seven-context averages", "",
              "| Split | Method | Mean normalized | Worst normalized | Δ mean vs dc_rank | Δ worst vs dc_rank |",
              "|---|---|---:|---:|---:|---:|"]
    for split, label in (("validation", "Validation"), ("test", "Test")):
        base = avg[split]["dc_rank"]
        for method in methods:
            m = avg[split][method]
            lines.append(f"| {label} | {method} | {m['mean_norm']:.3f} | {m['worst_norm']:.3f} | "
                         f"{m['mean_norm']-base['mean_norm']:+.3f} | {m['worst_norm']-base['worst_norm']:+.3f} |")
    lines += ["", "## Interpretation limits", "",
              "This is a transfer check of validation-selected coefficients, not another search. Coefficients are estimated from other contexts' `oracle_fair` vectors; task-context counts differ, and some tasks have only one prior context. The held-out test results assess transfer for these seven overlapping contexts only.", ""]
    (out / "transfer_summary.md").write_text("\n".join(lines))
    write_json(out / "aggregate.json", {"averages": avg, "contexts": contexts, "results": summary_rows})
    metadata["status"] = "complete"
    metadata["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metadata["new_evaluations"] = {
        split: sum(1 for row in summary_rows if row["evaluation_source"][split] == "new evaluation")
        for split in ("val", "test")
    }
    write_json(out / "run_manifest.json", metadata)
    print(f"complete: wrote {out / 'transfer_summary.md'}; new evaluations={metadata['new_evaluations']}", flush=True)


if __name__ == "__main__":
    main()
