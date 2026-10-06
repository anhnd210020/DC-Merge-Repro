"""Offline filesystem, dependency, CUDA, and artifact checks; never evaluates a model."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .design import TASKS, digest, make_design, validate_design
from .storage import sha256_file

EXPECTED_CLASSES = {
    "stanford_cars": 196, "dtd": 47, "eurosat": 10, "gtsrb": 43,
    "mnist": 10, "resisc45": 45, "sun397": 397, "svhn": 10,
}
EXPECTED_ADAPTER_MODULES = {
    f"vision_model.base_model.model.encoder.layers.{layer}.self_attn.{projection}.weight"
    for layer in range(12)
    for projection in ("q_proj", "k_proj", "v_proj", "out_proj")
}
REQUIRED_MODULES = ("torch", "numpy", "torchvision", "transformers", "peft", "PIL", "tqdm", "sklearn", "scipy", "clip")


@dataclass(frozen=True)
class Paths:
    repo: Path
    model: Path
    data: Path
    adapters: Path
    heads: Path
    references: Path
    output: Path
    min_free_gib: float = 10.0


def resolve_paths(args) -> Paths:
    repo = Path(args.repo_root).resolve()
    artifact_root = repo / "artifacts" / "selftrained_b32_r16_8task"
    reference_root = repo / "reproduction" / "selftrained_b32_r16_8task" / "results"
    model_value = args.model_dir or os.environ.get("DCMERGE_MODEL_DIR")
    data_value = args.data_dir or os.environ.get("DCMERGE_DATA_DIR")
    if not model_value or not data_value:
        raise ValueError("set DCMERGE_MODEL_DIR and DCMERGE_DATA_DIR (or pass --model-dir/--data-dir)")
    return Paths(
        repo=repo,
        model=Path(model_value).resolve(),
        data=Path(data_value).resolve(),
        adapters=Path(args.adapter_dir or os.environ.get("DCMERGE_ADAPTER_DIR", artifact_root / "checkpoints")).resolve(),
        heads=Path(args.head_dir or os.environ.get("DCMERGE_HEAD_DIR", artifact_root / "heads")).resolve(),
        references=Path(args.reference_dir or os.environ.get("DCMERGE_REFERENCE_DIR", reference_root)).resolve(),
        output=Path(args.output_dir or os.environ.get("DCMERGE_OUTPUT_DIR", repo / "analysis_outputs" / "pairwise_rank_density")).resolve(),
        min_free_gib=float(args.min_free_gib),
    )


def check_output_path(paths: Paths) -> None:
    repo = paths.repo.resolve()
    output = paths.output.resolve()
    forbidden = [
        repo / ".git", repo / "artifacts", repo / "reproduction", repo / "vision_lora_merge",
        repo / "state_dependent_task_drift", repo / "local_scale_sweep", repo / "handoff",
        repo / "analysis_outputs" / "dcmerge_task_weights", repo / "analysis_outputs" / "experiment1_cpu",
        repo / "analysis_outputs" / "state_dependent_task_drift", repo / "analysis_outputs" / "task_scale_ablation",
    ]
    forbidden.extend([paths.model, paths.data, paths.adapters, paths.heads, paths.references])
    if output == repo or any(output == root or root in output.parents for root in forbidden if str(root)):
        raise ValueError(f"output directory overlaps protected repository code, prior results, or input assets: {output}")
    if not output.parent.exists():
        raise FileNotFoundError(f"output parent does not exist: {output.parent}")
    if output.exists() and any(output.iterdir()) and not (output / "registration.json").is_file():
        raise FileExistsError(f"output is nonempty and is not a registered pairwise experiment: {output}")


def _need_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"empty {label}: {path}")
    return path


def _need_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _model_files(model_dir: Path) -> list[Path]:
    ancillary = [
        _need_file(model_dir / "config.json", "CLIP model config"),
        _need_file(model_dir / "preprocessor_config.json", "CLIP image preprocessor config"),
    ]
    tokenizer_ok = (model_dir / "tokenizer.json").is_file() or (
        (model_dir / "vocab.json").is_file() and (model_dir / "merges.txt").is_file()
    )
    if not tokenizer_ok:
        raise FileNotFoundError(f"missing CLIP tokenizer.json or vocab.json + merges.txt under {model_dir}")
    model_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    vision_config = model_config.get("vision_config", {})
    if (model_config.get("model_type") != "clip" or vision_config.get("hidden_size") != 768
            or vision_config.get("image_size") != 224 or vision_config.get("patch_size") != 32):
        raise ValueError("base model must be CLIP ViT-B/32 with 224px images and 32px patches")
    ancillary.extend(path for path in (model_dir / "tokenizer.json", model_dir / "vocab.json", model_dir / "merges.txt") if path.is_file())
    weight_files = [
        path for path in (
            model_dir / "model.safetensors", model_dir / "pytorch_model.bin",
            model_dir / "model.safetensors.index.json", model_dir / "pytorch_model.bin.index.json",
        ) if path.is_file()
    ]
    if not weight_files:
        raise FileNotFoundError(f"missing CLIP model weights or a shard index under {model_dir}")
    shards_found = []
    for index_path in (model_dir / "model.safetensors.index.json", model_dir / "pytorch_model.bin.index.json"):
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            shards = sorted(set(index.get("weight_map", {}).values()))
            if not shards:
                raise ValueError(f"model shard index contains no weight_map: {index_path}")
            for shard in shards:
                shards_found.append(_need_file(model_dir / shard, "CLIP model shard"))
    return sorted(set([*ancillary, *weight_files, *shards_found]))


def _dataset_layout(root: Path) -> dict[str, Any]:
    directories = {
        "cars": ["cars/devkit", "cars/cars_train", "cars/cars_test"],
        "dtd": [f"dtd/{split}" for split in ("train", "val", "test")],
        "eurosat": [f"eurosat/{split}" for split in ("train", "val", "test")],
        "gtsrb": ["gtsrb/GTSRB/Training", "gtsrb/GTSRB/Final_Test/Images"],
        "sun397": ["sun397/train", "sun397/val"],
    }
    for task, relpaths in directories.items():
        for rel in relpaths:
            folder = _need_dir(root / rel, f"{task} dataset directory")
            if not any(folder.iterdir()):
                raise ValueError(f"empty dataset directory: {folder}")
    required = [
        "cars/devkit/cars_train_annos.mat", "cars/devkit/cars_meta.mat",
        "cars/cars_test_annos_withlabels.mat", "gtsrb/GT-final_test.csv",
        "svhn/train_32x32.mat", "svhn/test_32x32.mat",
    ]
    for split in ("train", "val", "test"):
        required.append(f"resisc45/resisc45-{split}.txt")
    for rel in required:
        _need_file(root / rel, f"dataset metadata {rel}")
    mnist_raw = root / "MNIST" / "raw"
    mnist_processed = root / "MNIST" / "processed"
    if mnist_raw.is_dir():
        for name in ("train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"):
            _need_file(mnist_raw / name, "MNIST raw file")
    elif mnist_processed.is_dir():
        for name in ("training.pt", "test.pt"):
            _need_file(mnist_processed / name, "MNIST processed file")
    else:
        raise FileNotFoundError(f"missing MNIST/raw or MNIST/processed under {root}")
    class_dirs = sorted(path.name for path in (root / "resisc45").iterdir() if path.is_dir())
    if len(class_dirs) != 45 or "NWPU-RESISC45" in class_dirs:
        raise ValueError("RESISC45 requires its 45 class directories directly beneath data/resisc45")
    if not any(path.is_dir() for path in (root / "dtd" / "train").iterdir()):
        raise ValueError("DTD training split contains no class directories")
    return {
        "resisc45_class_dirs": class_dirs,
        "required_file_sha256": {rel: sha256_file(root / rel) for rel in required},
        "mnist_source": "raw" if mnist_raw.is_dir() else "processed",
        "dataset_root": str(root),
    }


def _normalized_adapter_state(state: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    for key, value in state.items():
        key = str(key)
        if key.startswith("base_model.model."):
            key = "vision_model." + key
        key = key.replace(".lora_A.weight", ".lora_A.default.weight")
        key = key.replace(".lora_B.weight", ".lora_B.default.weight")
        normalized[key] = value
    return normalized


def _validate_adapter_assets(paths: Paths) -> dict[str, Any]:
    import torch
    vision = paths.repo / "vision_lora_merge"
    sys.path.insert(0, str(vision)) if str(vision) not in sys.path else None
    from ft_handlers import get_ft_parameters_delta
    weights: dict[str, str] = {}
    shapes: dict[str, Any] = {}
    common_keys = None
    for task in TASKS:
        folder = _need_dir(paths.adapters / task, f"{task} adapter directory")
        config_path = _need_file(folder / "adapter_config.json", f"{task} adapter config")
        weights_path = _need_file(folder / "adapter_model.bin", f"{task} adapter weights")
        adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
        if adapter_config.get("r") != 16 or adapter_config.get("lora_alpha") != 16:
            raise ValueError(f"{task} adapter must remain rank/alpha 16: {adapter_config}")
        if set(adapter_config.get("target_modules", ())) != {"q_proj", "k_proj", "v_proj", "out_proj"}:
            raise ValueError(f"{task} adapter targets differ from the registered ViT-B/32 recipe")
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        deltas = get_ft_parameters_delta(_normalized_adapter_state(state))
        keys = tuple(deltas)
        if set(keys) != EXPECTED_ADAPTER_MODULES:
            missing = sorted(EXPECTED_ADAPTER_MODULES - set(keys))
            extra = sorted(set(keys) - EXPECTED_ADAPTER_MODULES)
            raise ValueError(f"{task} adapter target modules are incompatible; missing={missing[:3]}, extra={extra[:3]}")
        if any(tuple(value.shape) != (768, 768) for value in deltas.values()):
            raise ValueError(f"{task} adapter delta matrices must all be 768x768")
        if common_keys is not None and keys != common_keys:
            raise ValueError(f"adapter module order/keys differ for {task}")
        common_keys = keys
        shapes[task] = {key: list(value.shape) for key, value in deltas.items()}
        if any(not torch.isfinite(value).all() for value in deltas.values()):
            raise ValueError(f"non-finite LoRA delta found in {task}")
        weights[task] = sha256_file(weights_path)
        del state, deltas
    return {"adapter_model_sha256": weights, "adapter_delta_shapes": shapes}


def _validate_heads(paths: Paths) -> dict[str, str]:
    import torch
    hashes = {}
    for task in TASKS:
        path = _need_file(paths.heads / "ViT-B-32" / f"{task}_head.pt", f"{task} ViT-B-32 classification head")
        try:
            tensor = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise ValueError(f"cannot load classification head {path}: {error}") from error
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (EXPECTED_CLASSES[task], 512):
            raise ValueError(f"{task} head must have shape ({EXPECTED_CLASSES[task]}, 512), got {getattr(tensor, 'shape', None)}")
        hashes[task] = sha256_file(path)
    return hashes


def _read_references(path: Path, split: str) -> dict[str, float]:
    values = json.loads(_need_file(path, f"single-task {split} reference metrics").read_text(encoding="utf-8"))
    if set(values) != set(TASKS):
        raise ValueError(f"{split} reference tasks differ from registered task order: {path}")
    out = {task: float(values[task]) for task in TASKS}
    if any(not math.isfinite(value) or not 0 < value <= 100 for value in out.values()):
        raise ValueError(f"invalid {split} reference accuracy values: {path}")
    return out


def preflight(paths: Paths, *, stage: str, device: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
    registered_design = make_design()
    validate_design(registered_design)
    check_output_path(paths)
    required = [module for module in REQUIRED_MODULES if importlib.util.find_spec(module) is None]
    if required:
        raise RuntimeError("missing Python dependencies: " + ", ".join(required))
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    _need_dir(paths.model, "local CLIP model directory")
    model_files = _model_files(paths.model)
    dataset = _dataset_layout(paths.data)
    adapter_identity = _validate_adapter_assets(paths)
    head_hashes = _validate_heads(paths)
    adapter_config_hashes = {
        task: sha256_file(paths.adapters / task / "adapter_config.json")
        for task in TASKS
    }
    val_path = paths.references / "selftrained_val_acc.json"
    val_references = _read_references(val_path, "validation")
    test_path = paths.references / "selftrained_test_acc.json"
    _need_file(test_path, "single-task test reference metrics")
    if stage == "final-test":
        test_references = _read_references(test_path, "test")
        references = test_references
    else:
        # Deliberately do not read test denominators before the validation manifest is frozen.
        references = val_references
    free_bytes = shutil.disk_usage(paths.output.parent).free
    free_gib = free_bytes / (1024 ** 3)
    if free_gib < paths.min_free_gib:
        raise OSError(f"output volume has {free_gib:.2f} GiB free; need at least {paths.min_free_gib:.2f} GiB")
    code_paths = [
        paths.repo / "vision_lora_merge" / name
        for name in ("merging_functions.py", "ft_handlers.py", "utils.py", "eval.py")
    ]
    code_paths.extend([
        paths.repo / "vision_lora_merge" / "configs" / "vitB32_r16_8task_selftrained.py",
        paths.repo / "vision_lora_merge" / "configs" / "vitB32_r16_8task.py",
        paths.repo / "vision_lora_merge" / "models" / "huggingface_clip.py",
        paths.repo / "vision_lora_merge" / "dataset" / "shuffled_idxs" / "cars_shuffled_idxs.pt",
    ])
    code_paths.extend(sorted((paths.repo / "vision_lora_merge" / "dataset").glob("*.py")))
    code_paths.extend(sorted((Path(__file__).parent).glob("*.py")))
    code_paths.append(Path(__file__).with_name("config.json"))
    code_hashes = {str(path.relative_to(paths.repo)): sha256_file(path) for path in code_paths if path.is_file()}
    model_hashes = {
        str(path.relative_to(paths.model)): {"size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in model_files
    }
    identity = {
        "mode": "real",
        "design_sha256": digest(registered_design),
        "task_order": list(TASKS),
        "code_sha256": code_hashes,
        "model_dir": str(paths.model),
        "model_files": model_hashes,
        "dataset": dataset,
        **adapter_identity,
        "adapter_config_sha256": adapter_config_hashes,
        "head_sha256": head_hashes,
        "validation_reference_sha256": sha256_file(val_path),
        # Bind the final-test denominator file without parsing its values until
        # the final-test stage, after validation has been frozen.
        "test_reference_sha256": sha256_file(test_path),
        "validation_reference_values": val_references,
        "reference_root": str(paths.references),
    }
    operational = {
        "python": sys.version.split()[0],
        "dependencies": {
            name: importlib.metadata.version({"PIL": "Pillow", "sklearn": "scikit-learn", "clip": "openai-clip"}.get(name, name))
            for name in REQUIRED_MODULES
        },
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "output": str(paths.output),
        "output_free_gib": free_gib,
        "min_free_gib": paths.min_free_gib,
        "registered_pairs": 28,
        "registered_configurations": 420,
        "stage_reference_sha256": sha256_file(test_path) if stage == "final-test" else sha256_file(val_path),
    }
    return identity, operational, references
