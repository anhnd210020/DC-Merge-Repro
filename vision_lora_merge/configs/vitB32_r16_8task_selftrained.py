"""ViT-B/32 rank-16 eight-task config for independently trained adapters."""

import os
from copy import deepcopy

from configs.vitB32_r16_8task import config as _benchmark_config


EXPERIMENT_ROOT = os.environ.get(
    "DCMERGE_SELFTRAIN_ROOT", "/workspace/selftrained_b32_r16_8task"
)

config = deepcopy(_benchmark_config)
config["model"]["cachedir"] = os.environ.get(
    "DCMERGE_MODEL_DIR", "/workspace/models/clip-vit-base-patch32"
)
config["model"]["bases"] = [
    os.path.join(EXPERIMENT_ROOT, "checkpoints", dataset["name"])
    for dataset in config["dataset"]
]

for dataset in config["dataset"]:
    dataset["clip_encodings"] = os.path.join(
        EXPERIMENT_ROOT,
        "heads",
        "ViT-B-32",
        f"{dataset['name']}_head.pt",
    )
