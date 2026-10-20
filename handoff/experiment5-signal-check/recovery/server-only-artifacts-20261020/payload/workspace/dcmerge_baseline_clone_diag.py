#!/usr/bin/env python3
"""Compare the saved E2 baseline with the current reproduction checkout."""
import json
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, "/workspace/ducanh/DC-Merge/lane_harm")
import run_official as ro
import run_week1 as week1

repo = Path("/workspace/DC-Merge-Repro")
args = Namespace(
    repo=repo,
    model=Path("/workspace/models/clip-vit-base-patch32"),
    ft_dir=Path("/workspace/models/DC-Merge-Vision-B32-r16-8task/lora_checkpoints"),
    heads=Path("/workspace/models/DC-Merge-Vision-B32-r16-8task/lora_heads"),
    data=Path("/workspace/datasets8"),
    device="cuda",
)
rt = ro.load_runtime(args)
cid = "dtd-eurosat-gtsrb-svhn"
tasks = list(ro.CONTEXTS[cid])
rank = json.loads(Path("/srv/dcmerge/results/experiment2/compare", f"{cid}__dc_rank.json").read_text())
alpha = float(rank["selected_coef"])
merged, _ = week1.build_merge(rt, "dc_rank", tasks)
for split in ("val", "test"):
    acc = ro.make_evaluate(rt, tasks, "cuda", split)(merged, alpha)
    reference = (next(x["acc"] for x in rank["val_sweep"] if abs(x["coef"] - alpha) < 1e-9)
                 if split == "val" else rank["test_acc"])
    diffs = {t: abs(acc[t] - reference[t]) for t in tasks}
    print(split, json.dumps({"acc": acc, "diffs": diffs,
                             "max_diff": max(diffs.values()),
                             "max_task": max(diffs, key=diffs.get)}, sort_keys=True), flush=True)
