from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


PILOT_TASKS = ("stanford_cars", "eurosat", "svhn", "dtd")
PILOT_SEEDS = (420, 421, 422)
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "out_proj")


def _resolve_path(value: Any, base: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(path: Path, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    cfg = deepcopy(raw)
    for key, value in (overrides or {}).items():
        if value is not None:
            cfg[key] = value

    repo_value = cfg.get("repository_root", ".")
    repo = _resolve_path(repo_value, path.parent)
    assert repo is not None
    cfg["repository_root"] = repo
    for key in (
        "pretrained_model",
        "dataset_root",
        "head_root",
        "baseline_adapter_root",
        "output_root",
    ):
        cfg[key] = _resolve_path(cfg.get(key), repo)

    cfg["tasks"] = tuple(cfg.get("tasks", PILOT_TASKS))
    cfg["seeds"] = tuple(int(seed) for seed in cfg.get("seeds", PILOT_SEEDS))
    cfg["anchor_seed"] = int(cfg.get("anchor_seed", 420))
    cfg["anchor_alpha"] = float(cfg.get("anchor_alpha", 0.5))
    cfg["lora_rank"] = int(cfg.get("lora_rank", 16))
    cfg["lora_alpha"] = float(cfg.get("lora_alpha", 16))
    cfg["target_modules"] = tuple(cfg.get("target_modules", TARGET_MODULES))
    cfg["merge_alphas"] = tuple(float(x) for x in cfg.get("merge_alphas", []))
    validate_config(cfg)
    return cfg


def validate_config(cfg: Mapping[str, Any]) -> None:
    tasks = tuple(cfg["tasks"])
    if tasks != PILOT_TASKS:
        raise ValueError(f"pilot tasks and order must be {PILOT_TASKS}, got {tasks}")
    if tuple(cfg["seeds"]) != PILOT_SEEDS:
        raise ValueError(f"pilot seeds must be {PILOT_SEEDS}")
    if cfg["anchor_seed"] != 420:
        raise ValueError("pilot anchor seed must be 420")
    if cfg["anchor_alpha"] < 0:
        raise ValueError("anchor_alpha must be nonnegative")
    if cfg["lora_rank"] <= 0 or cfg["lora_alpha"] <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    if not cfg["target_modules"]:
        raise ValueError("target_modules cannot be empty")
    expected = tuple(index / 10 for index in range(1, 31))
    if tuple(cfg["merge_alphas"]) != expected:
        raise ValueError("merge_alphas must be exactly 0.1 through 3.0 in steps of 0.1")
    if cfg.get("precision", "fp32") not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32, fp16, or bf16")


def jsonable_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in cfg.items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, tuple):
            result[key] = list(value)
        else:
            result[key] = value
    return result

