from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .run_validation import validate_reuse_adapter


def missing_model_resources(root: Path) -> list[str]:
    root = Path(root)
    missing = [str(root / name) for name in ("config.json", "preprocessor_config.json") if not (root / name).is_file()]
    direct = [root / "model.safetensors", root / "pytorch_model.bin"]
    indexes = [root / "model.safetensors.index.json", root / "pytorch_model.bin.index.json"]
    if any(path.is_file() for path in direct):
        return missing
    existing_indexes = [path for path in indexes if path.is_file()]
    if not existing_indexes:
        missing.append("one of: " + ", ".join(str(path) for path in direct + indexes))
        return missing
    try:
        index = json.loads(existing_indexes[0].read_text(encoding="utf-8"))
        shards = sorted(set(index["weight_map"].values()))
    except Exception as error:
        missing.append(f"valid Hugging Face weight index {existing_indexes[0]} ({error!r})")
        return missing
    missing.extend(str(root / shard) for shard in shards if not (root / shard).is_file())
    return missing


def head_path(root: Path, task: str) -> Path:
    direct = root / f"{task}_head.pt"
    nested = root / "ViT-B-32" / f"{task}_head.pt"
    return direct if direct.is_file() else nested


def dataset_requirements(root: Path, repo: Path, task: str) -> list[Path]:
    if task == "stanford_cars":
        return [
            root / "cars" / "devkit" / "cars_train_annos.mat",
            root / "cars" / "devkit" / "cars_meta.mat",
            root / "cars" / "cars_train",
            root / "cars" / "cars_test_annos_withlabels.mat",
            root / "cars" / "cars_test",
            repo / "vision_lora_merge" / "dataset" / "shuffled_idxs" / "cars_shuffled_idxs.pt",
        ]
    if task in {"eurosat", "dtd"}:
        return [root / task / split for split in ("train", "val", "test")]
    if task == "svhn":
        return [
            root / "svhn" / "train_32x32.mat",
            root / "svhn" / "test_32x32.mat",
            repo / "vision_lora_merge" / "dataset" / "shuffled_idxs" / "svhn_shuffled_idxs.pt",
        ]
    raise ValueError(task)


def inspect_resources(cfg: Mapping[str, Any], require_baselines: bool = True) -> dict[str, Any]:
    missing: list[dict[str, str]] = []
    present: list[str] = []
    repo = Path(cfg["repository_root"])

    def check(path: Path | None, kind: str) -> None:
        if path is None:
            missing.append({"kind": kind, "path": "<not configured>"})
        elif not path.exists():
            missing.append({"kind": kind, "path": str(path)})
        else:
            present.append(str(path))

    model = cfg.get("pretrained_model")
    check(model, "pretrained_model")
    if model is not None:
        for item in missing_model_resources(Path(model)):
            missing.append({"kind": "pretrained_model_file", "path": item})
    data = cfg.get("dataset_root")
    heads = cfg.get("head_root")
    check(data, "dataset_root")
    check(heads, "head_root")
    if data is not None:
        for task in cfg["tasks"]:
            for path in dataset_requirements(Path(data), repo, task):
                check(path, f"dataset:{task}")
    if heads is not None:
        for task in cfg["tasks"]:
            check(head_path(Path(heads), task), f"head:{task}")

    adapters: dict[str, Any] = {}
    baseline_root = cfg.get("baseline_adapter_root")
    if require_baselines:
        check(baseline_root, "baseline_adapter_root")
        if baseline_root is not None:
            for task in cfg["tasks"]:
                directory = Path(baseline_root) / task
                try:
                    validation = validate_reuse_adapter(
                        cfg, directory, task, int(cfg["anchor_seed"])
                    )
                    adapters[task] = {
                        **validation["adapter_report"],
                        "reuse_metadata_valid": True,
                    }
                except Exception as error:
                    missing.append({
                        "kind": f"baseline_adapter_reuse:{task}",
                        "path": str(directory),
                        "error": repr(error),
                    })
    return {"ok": not missing, "missing": missing, "present": present, "adapter_reports": adapters, "auto_download": False}
