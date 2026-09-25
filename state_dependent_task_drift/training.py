from __future__ import annotations

import contextlib
import json
import os
import random
import shutil
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .anchor import (
    assert_optimizer_matches,
    configure_fresh_lora_trainability,
    materialize_anchor,
)
from .config import jsonable_config
from .lora_delta import adapter_report, effective_deltas, load_adapter, validate_saved_adapter
from .preflight import dataset_requirements, head_path, missing_model_resources
from .run_plan import baseline_run_dir, baseline_run_id, conditional_run_dir, conditional_run_id
from .run_validation import (
    effective_training_recipe,
    expected_anchor_definition,
    expected_run_definition,
    run_definition_hash,
    validate_complete_run,
    validate_reuse_adapter,
)
from .state_io import atomic_write_json, run_action, set_status


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def git_dirty(repo: Path) -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=normal"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(output.strip())
    except Exception:
        return None


def apply_reproduction_scheduler_step(scheduler: Any, updates_completed: int) -> int:
    """Apply the original zero-based post-update schedule and return the new count."""
    if updates_completed < 0:
        raise ValueError("updates_completed must be nonnegative")
    scheduler(updates_completed)
    return updates_completed + 1


def grad_scaler_enabled(device: str) -> bool:
    """Match the original unconditional CUDA GradScaler without enabling it on CPU."""
    return torch.device(device).type == "cuda"


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _task_config(cfg: Mapping[str, Any], task: str, train_preprocess: Any, eval_preprocess: Any) -> dict[str, Any]:
    repo = Path(cfg["repository_root"])
    result: dict[str, Any] = {
        "name": task,
        "root": str(cfg["dataset_root"]),
        "batch_size": int(cfg.get("batch_size", 32)),
        "num_workers": int(cfg.get("workers", 0)),
        "train_preprocess": train_preprocess,
        "eval_preprocess": eval_preprocess,
        "clip_encodings": str(head_path(Path(cfg["head_root"]), task)),
        "shuffle_train": True,
        "crop_ratio": 1.0,
    }
    if task in {"stanford_cars", "svhn"}:
        stem = "cars" if task == "stanford_cars" else task
        result.update({
            "val_fraction": 0.2,
            "shuffled_idxs": str(repo / "vision_lora_merge" / "dataset" / "shuffled_idxs" / f"{stem}_shuffled_idxs.pt"),
        })
    return result


class VisionClassifier(torch.nn.Module):
    def __init__(self, vision_model: torch.nn.Module, projection: torch.nn.Module):
        super().__init__()
        self.vision_model = vision_model
        self.vision_head = projection

    def forward(self, inputs: Any) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            inputs = {"pixel_values": inputs}
        if inputs["pixel_values"].ndim == 5:
            inputs["pixel_values"] = inputs["pixel_values"].squeeze(1)
        return self.vision_head(self.vision_model(**inputs)[1])


def build_fresh_model(cfg: Mapping[str, Any], device: str, anchor_adapter: Path | None = None) -> tuple[VisionClassifier, dict[str, Any] | None, Any, Any]:
    from peft import LoraConfig, get_peft_model
    from transformers import CLIPModel
    try:
        from transformers import CLIPImageProcessor as ImageProcessor
    except ImportError:
        from transformers import CLIPFeatureExtractor as ImageProcessor

    model_path = str(cfg["pretrained_model"])
    clip = CLIPModel.from_pretrained(model_path, local_files_only=True)
    processor = ImageProcessor.from_pretrained(model_path, local_files_only=True)
    anchor_info = None
    if anchor_adapter is not None:
        adapter_cfg, adapter_state = load_adapter(anchor_adapter)
        deltas = effective_deltas(adapter_cfg, adapter_state)
        mapped = materialize_anchor(clip.vision_model, deltas, float(cfg["anchor_alpha"]))
        anchor_info = {"adapter_report": adapter_report(anchor_adapter), "mapped_weights": mapped}

    for parameter in clip.parameters():
        parameter.requires_grad_(False)
    lora = LoraConfig(
        r=int(cfg["lora_rank"]), lora_alpha=float(cfg["lora_alpha"]),
        target_modules=list(cfg["target_modules"]), lora_dropout=float(cfg.get("lora_dropout", 0.1)),
        bias=str(cfg.get("lora_bias", "none")),
    )
    vision = get_peft_model(clip.vision_model, lora)
    model = VisionClassifier(vision, clip.visual_projection).to(device)
    configure_fresh_lora_trainability(model)
    preprocess = lambda image: processor(image, return_tensors="pt")
    return model, anchor_info, preprocess, preprocess


def _move_inputs(inputs: Any, device: str) -> Any:
    return inputs.to(device) if hasattr(inputs, "to") else {key: value.to(device) for key, value in inputs.items()}


def evaluate(model: torch.nn.Module, loader: Any, class_vectors: torch.Tensor, device: str, max_batches: int | None) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for index, (inputs, labels) in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            encodings = model(_move_inputs(inputs, device))
            encodings = encodings / encodings.norm(dim=-1, keepdim=True)
            predictions = (encodings @ class_vectors.T).argmax(dim=1)
            correct += int((predictions == labels.to(device)).sum())
            total += int(labels.numel())
    if total == 0:
        raise RuntimeError("evaluation consumed zero examples")
    return correct / total


def _save_adapter_atomic(peft_model: torch.nn.Module, destination: Path) -> None:
    candidate = destination.with_name(destination.name + ".candidate")
    previous = destination.with_name(destination.name + ".previous")
    for path in (candidate, previous):
        if path.exists():
            shutil.rmtree(path)
    candidate.mkdir(parents=True)
    peft_model.save_pretrained(candidate, safe_serialization=False)
    adapter_report(candidate)
    if destination.exists():
        destination.replace(previous)
    candidate.replace(destination)
    if previous.exists():
        shutil.rmtree(previous)


def _validate_fresh_peft_reload(
    cfg: Mapping[str, Any], checkpoint: Path, anchor_adapter: Path | None,
    reference_model: VisionClassifier, reference_device: str,
) -> dict[str, Any]:
    """Attach the saved adapter to a freshly reconstructed local M0/anchor."""
    from peft import PeftModel
    from transformers import CLIPModel

    clip = CLIPModel.from_pretrained(str(cfg["pretrained_model"]), local_files_only=True)
    if anchor_adapter is not None:
        anchor_cfg, anchor_state = load_adapter(anchor_adapter)
        materialize_anchor(
            clip.vision_model,
            effective_deltas(anchor_cfg, anchor_state),
            float(cfg["anchor_alpha"]),
        )
    reloaded_vision = PeftModel.from_pretrained(
        clip.vision_model, str(checkpoint), is_trainable=False
    )
    reloaded = VisionClassifier(reloaded_vision, clip.visual_projection).cpu().eval()
    reference_model.eval()
    image_size = int(getattr(clip.config.vision_config, "image_size", 224))
    pixels = torch.zeros((1, 3, image_size, image_size), dtype=torch.float32)
    with torch.no_grad():
        expected = reference_model({"pixel_values": pixels.to(reference_device)}).detach().cpu().float()
        actual = reloaded({"pixel_values": pixels}).detach().cpu().float()
    maximum = float((expected - actual).abs().max())
    if not torch.allclose(expected, actual, rtol=1e-4, atol=1e-5):
        raise RuntimeError(
            f"fresh PEFT reload synthetic forward mismatch: max_abs={maximum}"
        )
    return {
        "succeeded": True,
        "fresh_local_base_reconstructed": True,
        "exact_anchor_reconstructed": anchor_adapter is not None,
        "peft_from_pretrained_used": True,
        "synthetic_input_shape": list(pixels.shape),
        "output_shape": list(actual.shape),
        "maximum_absolute_difference": maximum,
        "rtol": 1e-4,
        "atol": 1e-5,
    }


def _autocast(device: str, precision: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=torch.device(device).type, dtype=dtype)


def _baseline_adapter_for_anchor(
    cfg: Mapping[str, Any], task: str, seed: int, reuse_existing: bool,
    max_train_steps: int | None = None, max_eval_batches: int | None = None,
) -> Path:
    output = Path(cfg["output_root"])
    generated = baseline_run_dir(output, task, seed)
    metadata_path = generated / "metadata.json"
    if metadata_path.is_file() and (generated / "status.json").is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        status = json.loads((generated / "status.json").read_text(encoding="utf-8"))
        if status.get("status") == "COMPLETE" and metadata.get("target_training_seed") == seed:
            validate_complete_run(
                cfg, generated, target_task=task, target_seed=seed,
                max_train_steps=max_train_steps, max_eval_batches=max_eval_batches,
            )
            return generated / "adapter"
    if not reuse_existing:
        raise FileNotFoundError(f"verified generated baseline missing for {task} seed {seed}: {generated}")
    root = cfg.get("baseline_adapter_root")
    if root is None:
        raise FileNotFoundError("baseline_adapter_root is not configured")
    candidate = Path(root) / task
    validate_reuse_adapter(cfg, candidate, task, seed)
    return candidate


def train_run(
    cfg: Mapping[str, Any], *, target_task: str, target_seed: int,
    anchor_task: str | None = None, resume: bool = False,
    max_train_steps: int | None = None, max_eval_batches: int | None = None,
    device: str = "cpu", reuse_existing_baseline: bool = False,
) -> dict[str, Any]:
    if anchor_task is None:
        run_dir = baseline_run_dir(Path(cfg["output_root"]), target_task, target_seed)
        run_id = baseline_run_id(target_task, target_seed)
        anchor_adapter = None
    else:
        run_dir = conditional_run_dir(Path(cfg["output_root"]), target_task, anchor_task, cfg["anchor_alpha"], target_seed)
        run_id = conditional_run_id(target_task, anchor_task, cfg["anchor_alpha"], target_seed, cfg["anchor_seed"])
        anchor_adapter = _baseline_adapter_for_anchor(
            cfg, anchor_task, int(cfg["anchor_seed"]), reuse_existing_baseline,
            max_train_steps=max_train_steps, max_eval_batches=max_eval_batches,
        )
    definition = expected_run_definition(
        cfg, target_task=target_task, target_seed=target_seed, anchor_task=anchor_task,
        anchor_adapter=anchor_adapter, max_train_steps=max_train_steps,
        max_eval_batches=max_eval_batches,
    )
    action = run_action(run_dir, resume)
    if action == "skip":
        validate_complete_run(
            cfg, run_dir, target_task=target_task, target_seed=target_seed,
            anchor_task=anchor_task, anchor_adapter=anchor_adapter,
            max_train_steps=max_train_steps, max_eval_batches=max_eval_batches,
        )
        return {"run_id": run_id, "status": "SKIPPED", "output_dir": str(run_dir)}
    run_dir.mkdir(parents=True, exist_ok=True)
    set_status(run_dir, "RUNNING", run_id=run_id, started_at=utc_now(), restart_from_beginning=action == "retry")
    started = time.monotonic()
    metadata: dict[str, Any] = {
        "run_id": run_id, "status": "RUNNING", "target_task": target_task,
        "target_training_seed": int(target_seed), "anchor_task": anchor_task,
        "anchor_alpha": float(cfg["anchor_alpha"]) if anchor_task else None,
        "anchor_seed": int(cfg["anchor_seed"]) if anchor_task else None,
        "base_model_identifier": str(cfg.get("base_model_identifier", cfg["pretrained_model"])),
        "resolved_paths": jsonable_config(cfg), "git_commit": git_commit(Path(cfg["repository_root"])),
        "git_dirty": git_dirty(Path(cfg["repository_root"])),
        "timestamp": utc_now(), "completion_status": "RUNNING",
        "training_configuration": effective_training_recipe(cfg, max_train_steps, max_eval_batches),
        "run_definition": definition,
        "run_definition_hash": run_definition_hash(definition),
    }
    atomic_write_json(run_dir / "resolved_config.json", jsonable_config(cfg))
    try:
        required_paths = dataset_requirements(Path(cfg["dataset_root"]), Path(cfg["repository_root"]), target_task)
        required_paths.append(head_path(Path(cfg["head_root"]), target_task))
        model_root = Path(cfg["pretrained_model"])
        missing = [str(path) for path in required_paths if not path.exists()]
        missing.extend(missing_model_resources(model_root))
        if missing:
            raise FileNotFoundError(
                "required local resources are missing; refusing model/dataset auto-download: " + ", ".join(missing)
            )
        seed_everything(target_seed)
        model, anchor_info, train_preprocess, eval_preprocess = build_fresh_model(cfg, device, anchor_adapter)
        if anchor_adapter is not None:
            report = anchor_info["adapter_report"]
            expected_modules = set(cfg["target_modules"])
            if (
                report["rank"] != int(cfg["lora_rank"])
                or report["lora_alpha"] != float(cfg["lora_alpha"])
                or set(report["target_modules"]) != expected_modules
            ):
                raise RuntimeError(f"anchor adapter LoRA recipe mismatch: {report}")
            anchor_identity = expected_anchor_definition(cfg, anchor_task, anchor_adapter)
            metadata.update({
                "anchor_adapter_path": str(anchor_adapter.resolve()), "anchor_adapter_sha256": report["sha256"],
                "anchor_definition": anchor_identity,
                "anchor_definition_hash": definition["anchor_definition_hash"],
            })
        repo = Path(cfg["repository_root"])
        vision_dir = repo / "vision_lora_merge"
        if str(vision_dir) not in os.sys.path:
            os.sys.path.insert(0, str(vision_dir))
        from utils import cosine_lr, prepare_data

        data_cfg = _task_config(cfg, target_task, train_preprocess, eval_preprocess)
        loaders = prepare_data(data_cfg, device=device)
        train_loader, val_loader, test_loader = loaders["train"]["full"], loaders["test"]["val"], loaders["test"]["test"]
        class_vectors = torch.load(data_cfg["clip_encodings"], map_location=device, weights_only=True)
        trainable = configure_fresh_lora_trainability(model)
        optimizer = torch.optim.AdamW(trainable, lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
        assert_optimizer_matches(optimizer, trainable)
        effective_steps = int(max_train_steps if max_train_steps is not None else cfg["max_train_steps"])
        scheduler = cosine_lr(optimizer, float(cfg["learning_rate"]), int(cfg["warmup_steps"]), effective_steps)
        scaler = torch.cuda.amp.GradScaler(enabled=grad_scaler_enabled(device))
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=float(cfg.get("label_smoothing", 0.0)))
        best_val, best_step, updates = -1.0, None, 0
        best_state = None
        training_events: list[dict[str, Any]] = []
        patience = 0
        checkpoint = run_dir / "adapter"
        stop = False
        for _epoch in range(int(cfg.get("epochs", 10000))):
            for inputs, labels in train_loader:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device, str(cfg.get("precision", "fp32"))):
                    encodings = model(_move_inputs(inputs, device))
                    encodings = encodings / encodings.norm(dim=-1, keepdim=True)
                    loss = loss_fn(100.0 * encodings @ class_vectors.T, labels.to(device))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                updates = apply_reproduction_scheduler_step(scheduler, updates)
                should_eval = updates % int(cfg["eval_frequency"]) == 0 or updates >= effective_steps
                if should_eval:
                    val = evaluate(model, val_loader, class_vectors, device, max_eval_batches)
                    training_events.append({"step": updates, "validation_accuracy": val, "timestamp": utc_now()})
                    atomic_write_json(run_dir / "training_log.json", training_events)
                    if val > best_val + float(cfg.get("early_stopping_min_delta", 1e-3)):
                        best_val, best_step, patience = val, updates, 0
                        best_state = {
                            name: tensor.detach().cpu().clone()
                            for name, tensor in model.vision_model.state_dict().items()
                            if ".lora_A." in name or ".lora_B." in name
                        }
                        _save_adapter_atomic(model.vision_model, checkpoint)
                    else:
                        patience += 1
                    if patience >= int(cfg.get("early_stopping_patience", 5)):
                        stop = True
                if updates >= effective_steps or stop:
                    break
            if updates >= effective_steps or stop:
                break
        if best_step is None or best_state is None:
            raise RuntimeError("training ended without a validation-selected checkpoint")
        report = validate_saved_adapter(
            checkpoint, best_state, int(cfg["lora_rank"]), float(cfg["lora_alpha"]),
            cfg["target_modules"],
        )
        load_result = model.vision_model.load_state_dict(best_state, strict=False)
        if load_result.unexpected_keys:
            raise RuntimeError(f"could not restore validation-selected adapter: {load_result.unexpected_keys}")
        report["fresh_peft_reload"] = _validate_fresh_peft_reload(
            cfg, checkpoint, anchor_adapter, model, device
        )
        if anchor_adapter is not None:
            anchor_cfg, anchor_state = load_adapter(anchor_adapter)
            saved_cfg, saved_state = load_adapter(checkpoint)
            selected_cfg = {"r": int(cfg["lora_rank"]), "lora_alpha": float(cfg["lora_alpha"])}
            selected_delta = effective_deltas(selected_cfg, best_state)
            reloaded_delta = effective_deltas(saved_cfg, saved_state)
            anchor_delta = effective_deltas(anchor_cfg, anchor_state)
            maximum = 0.0
            for name in selected_delta:
                expected_weight_delta = float(cfg["anchor_alpha"]) * anchor_delta[name] + selected_delta[name]
                reloaded_weight_delta = float(cfg["anchor_alpha"]) * anchor_delta[name] + reloaded_delta[name]
                difference = float((expected_weight_delta - reloaded_weight_delta).abs().max())
                maximum = max(maximum, difference)
                if not torch.allclose(expected_weight_delta, reloaded_weight_delta, rtol=1e-6, atol=1e-7):
                    raise RuntimeError(f"conditional reconstruction differs after disk reload for {name}")
            report["conditional_anchor_relative_round_trip"] = {
                "succeeded": True,
                "identity": "M0 + alpha*Delta_B + Delta_A_given_B",
                "maximum_absolute_difference": maximum,
                "validation_level": "effective target-module weights; no dataset required",
            }
        test_accuracy = evaluate(model, test_loader, class_vectors, device, max_eval_batches)
        metadata.update({
            "status": "COMPLETE", "completion_status": "COMPLETE", "optimizer_updates": updates,
            "best_validation_accuracy": best_val, "best_step": best_step,
            "test_accuracy_after_frozen_selection": test_accuracy,
            "checkpoint_path": str(checkpoint.resolve()), "checkpoint_validation": report,
            "lora_rank": report["rank"], "lora_alpha": report["lora_alpha"],
            "effective_lora_scaling": report["scaling"], "lora_target_modules": report["target_modules"],
            "conditional_checkpoint_semantics": "fresh target adapter relative to exact anchor" if anchor_task else "baseline target adapter relative to M0",
            "elapsed_seconds": time.monotonic() - started,
        })
        atomic_write_json(run_dir / "metadata.json", metadata)
        set_status(run_dir, "COMPLETE", run_id=run_id, completed_at=utc_now(), checkpoint_sha256=report["sha256"])
        validate_complete_run(
            cfg, run_dir, target_task=target_task, target_seed=target_seed,
            anchor_task=anchor_task, anchor_adapter=anchor_adapter,
            max_train_steps=max_train_steps, max_eval_batches=max_eval_batches,
        )
        return metadata
    except Exception as error:
        metadata.update({"status": "FAILED", "completion_status": "FAILED", "error": repr(error), "traceback": traceback.format_exc()})
        atomic_write_json(run_dir / "metadata.json", metadata)
        set_status(run_dir, "FAILED", run_id=run_id, failed_at=utc_now(), error=repr(error))
        raise


def resolve_baseline_adapter(
    cfg: Mapping[str, Any], task: str, seed: int, reuse_existing: bool = False,
    max_train_steps: int | None = None, max_eval_batches: int | None = None,
) -> Path:
    return _baseline_adapter_for_anchor(
        cfg, task, seed, reuse_existing, max_train_steps, max_eval_batches
    )
