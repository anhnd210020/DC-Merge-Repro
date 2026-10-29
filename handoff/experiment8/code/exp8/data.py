"""Image cache and evaluation splits.

Each task's test split is converted once to uint8 tensors (CLIP resize + center crop,
no normalization) and stored as a memory-mapped .npy file. Normalization happens on
the GPU, so evaluation is a pure forward pass.

The test split is divided once (seed 0) into two disjoint halves:
  * "sel": used to measure breadth (E1-E3) and to SELECT the pairs we study;
  * "rep": used to REPORT every number about the selected pairs (E2 report, E4, E5).
Selecting on one half and reporting on the other prevents the most-harmed task from
being chosen by the same noise that measures its harm.
"""
import json
import os

import numpy as np

from . import config as C
from .config import DATASETS, SEED

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def task_dir(data_dir, task):
    return os.path.join(data_dir, task)


def make_splits(n, seed=SEED):
    perm = np.random.RandomState(seed).permutation(n)
    half = n // 2
    return {
        "n": int(n),
        "sel_order": [int(i) for i in perm[:half]],   # random order; screening = prefix
        "rep": sorted(int(i) for i in perm[half:]),
        "seed": seed,
    }


def prepare_task(task, data_dir, image_size=224, batch=256):
    """Download the test split of `task` and write images.npy / labels.npy / splits.json."""
    from datasets import load_dataset
    try:  # transformers >= 5: force the PIL backend (the one FusionBench's numbers used)
        from transformers import CLIPImageProcessorPil as CLIPImageProcessor
    except ImportError:  # transformers 4.x: CLIPImageProcessor is PIL-based
        from transformers import CLIPImageProcessor

    out = task_dir(data_dir, task)
    if os.path.exists(os.path.join(out, "splits.json")):
        print(f"[data] {task}: cached")
        return
    os.makedirs(out, exist_ok=True)
    path, name, split = DATASETS[task]
    ds = load_dataset(path, name, split=split) if name else load_dataset(path, split=split)
    proc = CLIPImageProcessor(size={"shortest_edge": image_size},
                              crop_size={"height": image_size, "width": image_size},
                              do_rescale=False, do_normalize=False)
    n = len(ds)
    img_path = os.path.join(out, "images.npy")
    images = np.lib.format.open_memmap(img_path + ".tmp", mode="w+", dtype=np.uint8,
                                       shape=(n, 3, image_size, image_size))
    labels = np.zeros(n, dtype=np.int64)
    for start in range(0, n, batch):
        rows = ds[start:start + batch]
        imgs = [im.convert("RGB") for im in rows["image"]]
        px = proc(images=imgs, return_tensors="np")["pixel_values"]
        images[start:start + len(imgs)] = np.clip(np.rint(px), 0, 255).astype(np.uint8)
        lab = rows["label"]
        labels[start:start + len(imgs)] = [int(x) for x in lab]
        print(f"[data] {task}: {min(start + batch, n)}/{n}", end="\r")
    images.flush()
    del images
    os.replace(img_path + ".tmp", img_path)
    np.save(os.path.join(out, "labels.npy"), labels)
    with open(os.path.join(out, "splits.json"), "w") as f:
        json.dump(make_splits(n), f)
    print(f"\n[data] {task}: {n} images written")


class TaskData:
    """Lazy access to one task's cached images and its split indices."""

    def __init__(self, data_dir, task):
        d = task_dir(data_dir, task)
        self.task = task
        self.images = np.load(os.path.join(d, "images.npy"), mmap_mode="r")
        self.labels = np.load(os.path.join(d, "labels.npy"))
        with open(os.path.join(d, "splits.json")) as f:
            self.splits = json.load(f)

    def indices(self, split):
        """Sorted index array for 'sel', 'screen', 'rep', 'probe', 'patch', 'crossterm' or 'all'.

        'screen' (selection half, capped) and 'rep' (report half) are disjoint by construction;
        'probe', 'patch' and 'crossterm' are fixed seeded subsets of 'rep'.
        """
        s = self.splits
        if split == "all":
            return np.arange(s["n"])
        if split == "sel":
            return np.array(sorted(s["sel_order"]))
        if split == "screen":
            return np.array(sorted(s["sel_order"][:C.SCREEN_CAP]))
        rep = np.array(s["rep"])
        if split == "rep":
            return rep
        caps = {"probe": (C.PROBE_TEST_CAP, 2), "patch": (C.PATCH_CAP, 1), "crossterm": (C.CROSSTERM_CAP, 3)}
        if split in caps:
            cap, off = caps[split]
            if len(rep) <= cap:
                return rep
            pick = np.random.RandomState(SEED + off).permutation(len(rep))[:cap]
            return np.sort(rep[pick])
        raise ValueError(split)

    def positions_in_rep(self, split):
        """Positions of a report-half subset inside the sorted report-half index array."""
        rep = np.array(self.splits["rep"])
        return np.searchsorted(rep, self.indices(split))
