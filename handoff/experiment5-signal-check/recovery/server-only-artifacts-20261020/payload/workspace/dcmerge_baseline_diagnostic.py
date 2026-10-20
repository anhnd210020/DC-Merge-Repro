#!/usr/bin/env python3
"""Read-only one-context comparison of direct and lane uniform-alpha baselines."""
import json
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, "/workspace/ducanh/DC-Merge/lane_harm")
import lanes
import run_oracle as oracle
import run_week1 as week1

repo = Path("/workspace/ducanh/DC-Merge")
args = Namespace(
    repo=repo,
    model=Path("/workspace/models/clip-vit-base-patch32"),
    ft_dir=Path("/workspace/models/DC-Merge-Vision-B32-r16-8task/lora_checkpoints"),
    heads=Path("/workspace/models/DC-Merge-Vision-B32-r16-8task/lora_heads"),
    data=Path("/workspace/datasets8"),
    device="cuda",
)
rt = oracle.ro.load_runtime(args)
import utils
utils.tqdm = lambda iterable, *a, **kw: iterable

cid = "dtd-eurosat-gtsrb-svhn"
tasks = list(oracle.ro.CONTEXTS[cid])
rank = json.loads(Path("/srv/dcmerge/results/experiment2/compare", f"{cid}__dc_rank.json").read_text())
alpha = float(rank["selected_coef"])
print("versions", rt["torch"].__version__)
print("tasks", tasks, "alpha", alpha)
for task in tasks:
    for split in ("val", "test"):
        loader = rt["data"][oracle.ro.TASKS.index(task)]["test"][split]
        print("loader", task, split, "n", len(loader.dataset), "batch", loader.batch_size)

direct, _ = week1.build_merge(rt, "dc_rank", tasks)
with rt["torch"].no_grad():
    lm = lanes.LaneMerge(
        oracle.ro.group(rt["deltas"], tasks), rt["backend"], tasks,
        oracle.ro.dc_kwargs("rank"),
    )
lane = lm.compose({t: alpha for t in tasks})
max_abs = 0.0
for key in direct:
    max_abs = max(max_abs, float((direct[key] - lane[key]).abs().max()))
print("lane_checks", lm.checks, "uniform_lane_max_abs_vs_direct", max_abs)

for split in ("val", "test"):
    evaluate = oracle.ro.make_evaluate(rt, tasks, "cuda", split)
    direct_runs = [evaluate(direct, alpha) for _ in range(2)]
    lane_run = evaluate(lane, 1.0)
    baseline = (next(x["acc"] for x in rank["val_sweep"] if abs(x["coef"] - alpha) < 1e-9)
                if split == "val" else rank["test_acc"])
    for label, acc in [("direct_1", direct_runs[0]), ("direct_2", direct_runs[1]), ("lane", lane_run)]:
        diffs = {t: abs(acc[t] - baseline[t]) for t in tasks}
        print(split, label, "acc", json.dumps(acc, sort_keys=True),
              "max_diff_vs_exp2", max(diffs.values()),
              "task", max(diffs, key=diffs.get))
    print(split, "repeat_diffs", json.dumps({t: abs(direct_runs[0][t] - direct_runs[1][t]) for t in tasks}, sort_keys=True))
