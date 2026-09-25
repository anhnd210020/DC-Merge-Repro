from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .anchor import anchor_definition, anchor_definition_hash, canonical_hash
from .lora_delta import adapter_checksum, adapter_report
from .state_io import read_json


def effective_training_recipe(
    cfg: Mapping[str, Any],
    max_train_steps: int | None = None,
    max_eval_batches: int | None = None,
) -> dict[str, Any]:
    return {
        "optimizer": "AdamW",
        "lr": float(cfg["learning_rate"]),
        "weight_decay": float(cfg["weight_decay"]),
        "warmup_steps": int(cfg["warmup_steps"]),
        "max_steps": int(max_train_steps if max_train_steps is not None else cfg["max_train_steps"]),
        "eval_frequency": int(cfg["eval_frequency"]),
        "early_stopping_patience": int(cfg.get("early_stopping_patience", 5)),
        "early_stopping_min_delta": float(cfg.get("early_stopping_min_delta", 1e-3)),
        "label_smoothing": float(cfg.get("label_smoothing", 0.0)),
        "batch_size": int(cfg.get("batch_size", 32)),
        "precision": str(cfg.get("precision", "fp32")),
        "max_eval_batches": max_eval_batches,
        "training_autocast": "disabled_when_precision_is_fp32",
        "gradient_scaler": "enabled_on_cuda_matching_original_reproduction",
        "scheduler_index_origin": 0,
        "scheduler_application": "after_optimizer_update_before_update_count_increment",
        "checkpoint_selection": "validation_accuracy",
        "test_used_for_selection": False,
        "fresh_optimizer_state": True,
    }


def expected_anchor_definition(
    cfg: Mapping[str, Any], anchor_task: str, anchor_adapter: Path
) -> dict[str, Any]:
    report = adapter_report(anchor_adapter)
    expected_scaling = float(cfg["lora_alpha"]) / int(cfg["lora_rank"])
    if (
        report["rank"] != int(cfg["lora_rank"])
        or report["lora_alpha"] != float(cfg["lora_alpha"])
        or report["scaling"] != expected_scaling
        or set(report["target_modules"]) != set(cfg["target_modules"])
    ):
        raise RuntimeError(f"canonical anchor adapter LoRA configuration is incompatible: {report}")
    return anchor_definition(
        base_model_id=str(cfg.get("base_model_identifier", cfg["pretrained_model"])),
        adapter_sha256=report["sha256"],
        anchor_task=anchor_task,
        anchor_seed=int(cfg["anchor_seed"]),
        alpha=float(cfg["anchor_alpha"]),
        rank=report["rank"],
        lora_alpha=report["lora_alpha"],
        scaling=report["scaling"],
        target_modules=report["target_modules"],
    )


def expected_run_definition(
    cfg: Mapping[str, Any],
    *,
    target_task: str,
    target_seed: int,
    anchor_task: str | None = None,
    anchor_adapter: Path | None = None,
    max_train_steps: int | None = None,
    max_eval_batches: int | None = None,
) -> dict[str, Any]:
    definition: dict[str, Any] = {
        "run_kind": "conditional" if anchor_task else "baseline",
        "target_task": target_task,
        "target_training_seed": int(target_seed),
        "base_model_identifier": str(cfg.get("base_model_identifier", cfg["pretrained_model"])),
        "lora_rank": int(cfg["lora_rank"]),
        "lora_alpha": float(cfg["lora_alpha"]),
        "effective_lora_scaling": float(cfg["lora_alpha"]) / int(cfg["lora_rank"]),
        "lora_target_modules": list(cfg["target_modules"]),
        "lora_dropout": float(cfg.get("lora_dropout", 0.1)),
        "lora_bias": str(cfg.get("lora_bias", "none")),
        "training_configuration": effective_training_recipe(cfg, max_train_steps, max_eval_batches),
    }
    if anchor_task is not None:
        if anchor_adapter is None:
            raise ValueError("conditional run definition requires anchor_adapter")
        anchor = expected_anchor_definition(cfg, anchor_task, anchor_adapter)
        definition.update(
            {
                "anchor_task": anchor_task,
                "anchor_seed": int(cfg["anchor_seed"]),
                "anchor_alpha": float(cfg["anchor_alpha"]),
                "anchor_adapter_sha256": anchor["anchor_adapter_sha256"],
                "anchor_definition_hash": anchor_definition_hash(anchor),
            }
        )
    return definition


def run_definition_hash(definition: Mapping[str, Any]) -> str:
    return canonical_hash(definition)


def validate_complete_run(
    cfg: Mapping[str, Any],
    run_dir: Path,
    *,
    target_task: str,
    target_seed: int,
    anchor_task: str | None = None,
    anchor_adapter: Path | None = None,
    max_train_steps: int | None = None,
    max_eval_batches: int | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    status_path, metadata_path = run_dir / "status.json", run_dir / "metadata.json"
    if not status_path.is_file():
        raise RuntimeError(f"run status is missing: {status_path}")
    status = read_json(status_path)
    if status.get("status") != "COMPLETE":
        raise RuntimeError(f"run is not COMPLETE: {run_dir} status={status.get('status')!r}")
    if not metadata_path.is_file():
        raise RuntimeError(f"run metadata is missing: {metadata_path}")
    metadata = read_json(metadata_path)
    if metadata.get("completion_status") != "COMPLETE" or metadata.get("status") != "COMPLETE":
        raise RuntimeError(
            f"run metadata is not COMPLETE: {run_dir} status={metadata.get('status')!r} "
            f"completion_status={metadata.get('completion_status')!r}"
        )
    if metadata.get("target_task") != target_task:
        raise RuntimeError(f"target task mismatch for {run_dir}: {metadata.get('target_task')!r} != {target_task!r}")
    if metadata.get("target_training_seed") != int(target_seed):
        raise RuntimeError(
            f"target training seed mismatch for {run_dir}: {metadata.get('target_training_seed')!r} != {target_seed}"
        )
    checkpoint = run_dir / "adapter"
    report = adapter_report(checkpoint)
    expected_scaling = float(cfg["lora_alpha"]) / int(cfg["lora_rank"])
    if (
        report["rank"] != int(cfg["lora_rank"])
        or report["lora_alpha"] != float(cfg["lora_alpha"])
        or report["scaling"] != expected_scaling
        or set(report["target_modules"]) != set(cfg["target_modules"])
    ):
        raise RuntimeError(f"completed adapter LoRA configuration is incompatible for {run_dir}: {report}")
    stored_checksum = metadata.get("checkpoint_validation", {}).get("sha256")
    if stored_checksum != report["sha256"]:
        raise RuntimeError(
            f"adapter checksum mismatch for {run_dir}: metadata={stored_checksum!r}, actual={report['sha256']!r}"
        )
    checkpoint_validation = metadata.get("checkpoint_validation", {})
    if checkpoint_validation.get("disk_reload_succeeded") is not True:
        raise RuntimeError(f"completed run lacks successful disk reload validation: {run_dir}")
    if checkpoint_validation.get("fresh_peft_reload", {}).get("succeeded") is not True:
        raise RuntimeError(f"completed run lacks successful fresh PEFT reload validation: {run_dir}")
    if status.get("checkpoint_sha256") != report["sha256"]:
        raise RuntimeError(
            f"completion checksum mismatch for {run_dir}: status={status.get('checkpoint_sha256')!r}, "
            f"actual={report['sha256']!r}"
        )
    expected = expected_run_definition(
        cfg,
        target_task=target_task,
        target_seed=target_seed,
        anchor_task=anchor_task,
        anchor_adapter=anchor_adapter,
        max_train_steps=max_train_steps,
        max_eval_batches=max_eval_batches,
    )
    expected_hash = run_definition_hash(expected)
    if metadata.get("run_definition_hash") != expected_hash:
        raise RuntimeError(
            f"run definition is incompatible with current configuration for {run_dir}: "
            f"stored={metadata.get('run_definition_hash')!r}, expected={expected_hash!r}"
        )
    if metadata.get("run_definition") != expected:
        raise RuntimeError(f"stored run definition differs from current requested definition: {run_dir}")
    if anchor_task is not None:
        if checkpoint_validation.get("conditional_anchor_relative_round_trip", {}).get("succeeded") is not True:
            raise RuntimeError(f"conditional run lacks anchor-relative round-trip validation: {run_dir}")
        if metadata.get("anchor_task") != anchor_task:
            raise RuntimeError(f"anchor task mismatch for {run_dir}")
        if metadata.get("anchor_seed") != int(cfg["anchor_seed"]):
            raise RuntimeError(f"anchor seed mismatch for {run_dir}")
        if metadata.get("anchor_alpha") != float(cfg["anchor_alpha"]):
            raise RuntimeError(f"anchor alpha mismatch for {run_dir}")
        expected_anchor = expected_anchor_definition(cfg, anchor_task, Path(anchor_adapter))
        expected_anchor_hash = anchor_definition_hash(expected_anchor)
        if metadata.get("anchor_definition_hash") != expected_anchor_hash:
            raise RuntimeError(
                f"anchor_definition_hash mismatch for {run_dir}: stored={metadata.get('anchor_definition_hash')!r}, "
                f"expected={expected_anchor_hash!r}"
            )
        if metadata.get("anchor_adapter_sha256") != adapter_checksum(Path(anchor_adapter)):
            raise RuntimeError(f"anchor adapter checksum mismatch for {run_dir}")
    return {"metadata": metadata, "status": status, "adapter_report": report, "run_definition": expected}


def validate_reuse_adapter(
    cfg: Mapping[str, Any], directory: Path, task: str, seed: int
) -> dict[str, Any]:
    directory = Path(directory)
    report = adapter_report(directory)
    metadata_path = directory / "state_drift_reuse_metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"missing baseline reuse metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = {
        "target_task": task,
        "target_training_seed": int(seed),
        "base_model_identifier": str(cfg.get("base_model_identifier", cfg["pretrained_model"])),
        "lora_rank": int(cfg["lora_rank"]),
        "lora_alpha": float(cfg["lora_alpha"]),
        "effective_lora_scaling": float(cfg["lora_alpha"]) / int(cfg["lora_rank"]),
        "lora_target_modules": list(cfg["target_modules"]),
        "completion_status": "COMPLETE",
        "adapter_sha256": report["sha256"],
    }
    mismatches = {key: {"actual": metadata.get(key), "expected": value} for key, value in required.items() if metadata.get(key) != value}
    recipe = metadata.get("training_configuration", {})
    expected_recipe = effective_training_recipe(cfg)
    compatibility_keys = (
        "optimizer", "lr", "weight_decay", "warmup_steps", "max_steps", "eval_frequency",
        "early_stopping_patience", "early_stopping_min_delta", "label_smoothing", "batch_size",
        "precision", "checkpoint_selection", "test_used_for_selection",
        "training_autocast", "gradient_scaler", "scheduler_index_origin",
        "scheduler_application",
    )
    for key in compatibility_keys:
        if recipe.get(key) != expected_recipe[key]:
            mismatches[f"training_configuration.{key}"] = {
                "actual": recipe.get(key), "expected": expected_recipe[key]
            }
    if mismatches:
        raise RuntimeError(f"baseline reuse metadata is incompatible for {directory}: {mismatches}")
    return {"metadata": metadata, "adapter_report": report}
