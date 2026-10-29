"""Fixed settings for Experiment 8. Do not edit after `select` has run: `select` records a
hash of this file and `depthstats` refuses to run if it changed.

There are no fixed percentage-of-pairs thresholds anywhere. Every claim is a contrast against
an explicit null, tested with an exact sign-flip test (victims as units) in a fixed sequence
(see exp8/stats.py and the guide).
"""
import hashlib
import os
from itertools import combinations

TASKS = ["sun397", "stanford-cars", "resisc45", "eurosat", "svhn", "gtsrb", "mnist", "dtd"]
SHORT = {"sun397": "SUN", "stanford-cars": "Cars", "resisc45": "RES", "eurosat": "Euro",
         "svhn": "SVHN", "gtsrb": "GTSRB", "mnist": "MNIST", "dtd": "DTD"}

# Hugging Face assets (the FusionBench full fine-tuned CLIP encoders).
BACKBONES = {
    "ViT-B-32": {"base": "openai/clip-vit-base-patch32",
                 "finetuned": "tanganke/clip-vit-base-patch32_{task}"},
    "ViT-L-14": {"base": "openai/clip-vit-large-patch14",
                 "finetuned": "tanganke/clip-vit-large-patch14_{task}"},
}

# Test split of each task (FusionBench dataset configs). (path, config name or None, split)
DATASETS = {
    "sun397": ("tanganke/sun397", None, "test"),
    "stanford-cars": ("tanganke/stanford_cars", None, "test"),
    "resisc45": ("tanganke/resisc45", None, "test"),
    "eurosat": ("tanganke/eurosat", None, "test"),
    "svhn": ("ufldl-stanford/svhn", "cropped_digits", "test"),
    "gtsrb": ("tanganke/gtsrb", None, "test"),
    "mnist": ("ylecun/mnist", None, "test"),
    "dtd": ("tanganke/dtd", None, "test"),
}

# Published FusionBench test accuracies (%), used only by the must-hold asset check.
REF_EXPERT = {
    "ViT-B-32": {"sun397": 75.0, "stanford-cars": 78.3, "resisc45": 95.2, "eurosat": 99.0,
                 "svhn": 97.3, "gtsrb": 98.9, "mnist": 99.6, "dtd": 79.7},
    "ViT-L-14": {"sun397": 82.8, "stanford-cars": 92.9, "resisc45": 97.4, "eurosat": 99.2,
                 "svhn": 97.9, "gtsrb": 99.2, "mnist": 99.8, "dtd": 85.5},
}
REF_ZEROSHOT = {
    "ViT-B-32": {"sun397": 63.2, "stanford-cars": 59.8, "resisc45": 60.7, "eurosat": 46.0,
                 "svhn": 31.6, "gtsrb": 32.5, "mnist": 48.3, "dtd": 43.9},
    "ViT-L-14": {"sun397": 68.3, "stanford-cars": 77.8, "resisc45": 71.0, "eurosat": 58.9,
                 "svhn": 58.4, "gtsrb": 50.6, "mnist": 76.4, "dtd": 55.5},
}

OPERATORS = ["ta", "iso_cts", "tsvm"]
ISO_CTS_COMMON_FRACTION = 0.8  # official default

# Coefficient grids: the official ranges (TA in (0,1], SVD methods in (0,3]), coarsened.
ALPHA_GRID = {
    "ta": [round(0.1 * i, 2) for i in range(1, 11)],
    "iso_cts": [round(0.2 * i, 2) for i in range(1, 16)],
    "tsvm": [round(0.2 * i, 2) for i in range(1, 16)],
}

SEED = 0
CONTEXT_SIZE = 4
N_ALPHA_CONTEXTS = 10       # random 4-task contexts used to pick alpha
SCREEN_CAP = 5000           # max images per task on the selection half (runtime-overridable)
PROBE_TEST_CAP = 5000       # max report-half images per task for the probes
PATCH_CAP = 2000            # max report-half images per task for activation patching
CROSSTERM_CAP = 500         # max report-half images per task for the cross-term
BOOTSTRAP = 2000

DEPTH = {
    "k_per_victim": 2,          # contexts kept per (operator, victim)
    "alpha_level": 0.05,
    "random_seeds": 3,
    "probe": {"lambdas": [1e-4, 1e-3, 1e-2, 1e-1, 1.0], "holdout": 0.2, "iters": 200},
    "min_victims": 5,           # smallest n with 2^-n <= 0.05
    "replicate_max_pairs": 6,
}
THIRDS = ["E", "M", "Lt"]
HYPOTHESIS_ORDER = ["localization", "specificity", "information", "amplification", "repair"]


def thirds(L):
    """Depth units of each third: equal thirds of the blocks; pre with E, post with Lt."""
    a, b = L // 3, 2 * L // 3
    return {"E": ["pre"] + list(range(0, a)), "M": list(range(a, b)), "Lt": list(range(b, L)) + ["post"]}


def boundaries(L):
    """Probe depths at the edges of the thirds (hidden-state index: 0 = input to block 0)."""
    return {"E": (0, L // 3), "M": (L // 3, 2 * L // 3), "Lt": (2 * L // 3, L)}


def all_contexts(tasks=TASKS, size=CONTEXT_SIZE):
    return [tuple(c) for c in combinations(tasks, size)]


def all_subsets_up_to(tasks=TASKS, size=CONTEXT_SIZE):
    out = []
    for k in range(1, size + 1):
        out += [tuple(c) for c in combinations(tasks, k)]
    return out


def canon(subset):
    """Canonical order of a task subset (the TASKS order)."""
    return tuple(t for t in TASKS if t in set(subset))


def config_sha256():
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
