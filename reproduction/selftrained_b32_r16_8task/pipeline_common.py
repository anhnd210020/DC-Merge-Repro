#!/usr/bin/env python3
"""Shared constants and helpers for the isolated self-trained reproduction."""

from __future__ import annotations

import importlib
import json
import os
import random
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


ROOT = Path("/workspace/selftrained_b32_r16_8task")
REPO = Path("/workspace/DC-Merge-Repro")
VISION_DIR = REPO / "vision_lora_merge"
MODEL_DIR = Path("/workspace/models/clip-vit-base-patch32")
DATA_DIR = Path("/workspace/datasets8")
TASKS = (
    "stanford_cars",
    "dtd",
    "eurosat",
    "gtsrb",
    "mnist",
    "resisc45",
    "sun397",
    "svhn",
)

TRAINING = {
    "epochs": 10000,
    "max_steps": 10000,
    "early_stopping_patience": 5,
    "early_stopping_min_delta": 1e-3,
    "eval_freq": 2000,
    "lora_rank": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"],
    "bias": "none",
    "optimizer": "AdamW",
    "lr": 3e-4,
    "weight_decay": 1e-1,
    "scheduler": "cosine with linear warmup",
    "warm_up": 500,
    "label_smoothing": 0.0,
    "gradient_accumulation_steps": 1,
    "mixed_precision": "torch.cuda.amp.GradScaler",
    "gradient_clipping": None,
    "seed": 420,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_repo_to_path() -> None:
    path = str(VISION_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)


def raw_config() -> Dict[str, Any]:
    add_repo_to_path()
    module = importlib.import_module("configs.vitB32_r16_8task_selftrained")
    return deepcopy(module.config)


def task_config(task: str) -> Dict[str, Any]:
    if task not in TASKS:
        raise ValueError(f"unsupported task: {task}")
    for item in raw_config()["dataset"]:
        if item["name"] == task:
            return item
    raise KeyError(task)


def seed_everything(seed: int = 420) -> None:
    # This intentionally matches authors' set_seed: no deterministic-algorithm
    # toggle and no explicit CUDA seed beyond torch.manual_seed.
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def nvidia_info() -> Dict[str, str]:
    fields = "name,driver_version"
    result = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().split(", ", 1)
    return {"gpu_model": result[0], "driver": result[1]}


def runtime_versions() -> Dict[str, str]:
    import peft
    import transformers

    versions = {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "transformers": transformers.__version__,
        "peft": peft.__version__,
    }
    versions.update(nvidia_info())
    return versions


def adapter_weight_file(checkpoint: Path) -> Optional[Path]:
    candidate = checkpoint / "adapter_model.bin"
    if candidate.is_file() and candidate.stat().st_size > 0:
        return candidate
    return None


def checkpoint_files_valid(checkpoint: Path) -> bool:
    return (
        (checkpoint / "adapter_config.json").is_file()
        and adapter_weight_file(checkpoint) is not None
    )


def require_environment() -> None:
    required = [REPO, MODEL_DIR, DATA_DIR]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required paths: " + ", ".join(missing))
    os.environ["DCMERGE_REPO"] = str(REPO)
    os.environ["DCMERGE_DATA_DIR"] = str(DATA_DIR)
    os.environ["DCMERGE_MODEL_DIR"] = str(MODEL_DIR)
    os.environ["DCMERGE_CACHE_DIR"] = str(MODEL_DIR)
    os.environ["DCMERGE_SELFTRAIN_ROOT"] = str(ROOT)
