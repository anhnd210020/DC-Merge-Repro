"""On-disk cache of per-image correctness, so every step is resumable and bootstrappable."""
import hashlib
import json
import os

import numpy as np


def _key(d):
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()


class EvalStore:
    def __init__(self, root):
        self.root = os.path.join(root, "evals")
        os.makedirs(self.root, exist_ok=True)
        self.n_new = 0

    def path(self, key):
        h = _key(key)
        return os.path.join(self.root, h[:2], h + ".npz")

    def get(self, key):
        p = self.path(key)
        if not os.path.exists(p):
            return None
        z = np.load(p)
        return np.unpackbits(z["bits"], count=int(z["n"])).astype(bool)

    def put(self, key, correct):
        p = self.path(key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp.npz"
        np.savez_compressed(tmp, bits=np.packbits(correct.astype(np.uint8)), n=len(correct),
                            key=json.dumps(key, sort_keys=True))
        os.replace(tmp, p)
        self.n_new += 1


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)
