#!/usr/bin/env python3
"""Verify a uniform lane vector matches the direct merged update tensor-wise."""
import json
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, "/workspace/ducanh/DC-Merge/lane_harm")
import lanes
import run_oracle as oracle
import run_week1 as week1

args = Namespace(
    repo=Path("/workspace/ducanh/DC-Merge"),
    model=Path("/workspace/models/clip-vit-base-patch32"),
    ft_dir=Path("/workspace/models/DC-Merge-Vision-B32-r16-8task/lora_checkpoints"),
    heads=Path("/workspace/models/DC-Merge-Vision-B32-r16-8task/lora_heads"),
    data=Path("/workspace/datasets8"),
    device="cuda",
)
rt = oracle.ro.load_runtime(args)
cid = "dtd-eurosat-gtsrb-svhn"
tasks = list(oracle.ro.CONTEXTS[cid])
rank = json.loads(Path("/srv/dcmerge/results/experiment2/compare", f"{cid}__dc_rank.json").read_text())
alpha = float(rank["selected_coef"])
direct, _ = week1.build_merge(rt, "dc_rank", tasks)
with rt["torch"].no_grad():
    lm = lanes.LaneMerge(oracle.ro.group(rt["deltas"], tasks), rt["backend"], tasks,
                         oracle.ro.dc_kwargs("rank"))
uniform = lm.compose({task: alpha for task in tasks})
max_abs = max(float((direct[key] * alpha - uniform[key]).abs().max()) for key in direct)
print(json.dumps({"context": cid, "alpha": alpha, "max_abs_vs_direct_scaled": max_abs,
                  "lane_checks": lm.checks}, sort_keys=True))
