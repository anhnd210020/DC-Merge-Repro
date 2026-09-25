from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import torch


def canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def anchor_definition(
    *, base_model_id: str, adapter_sha256: str, anchor_task: str, anchor_seed: int,
    alpha: float, rank: int, lora_alpha: float, scaling: float, target_modules: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    return {
        "base_model_identifier": base_model_id,
        "anchor_adapter_sha256": adapter_sha256,
        "anchor_task": anchor_task,
        "anchor_seed": int(anchor_seed),
        "anchor_alpha": float(alpha),
        "lora_rank": int(rank),
        "lora_alpha": float(lora_alpha),
        "effective_lora_scaling": float(scaling),
        "target_modules": list(target_modules),
    }


def anchor_definition_hash(definition: Mapping[str, Any]) -> str:
    return canonical_hash(definition)


def validate_anchor_hash(metadata: Mapping[str, Any], expected_definition: Mapping[str, Any]) -> None:
    expected = anchor_definition_hash(expected_definition)
    actual = metadata.get("anchor_definition_hash")
    if actual != expected:
        raise RuntimeError(f"anchor_definition_hash mismatch: checkpoint={actual!r}, expected={expected!r}")


def _candidate_weight_names(module_name: str) -> tuple[str, ...]:
    stripped = module_name
    for prefix in ("base_model.model.", "model."):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
    return (
        f"{stripped}.weight", f"vision_model.{stripped}.weight",
        f"base_model.model.{stripped}.weight", f"{module_name}.weight",
    )


def materialize_anchor(model: torch.nn.Module, deltas: Mapping[str, torch.Tensor], alpha: float) -> dict[str, str]:
    parameters = dict(model.named_parameters())
    mapped: dict[str, str] = {}
    with torch.no_grad():
        for module_name, delta in deltas.items():
            matches = [name for name in _candidate_weight_names(module_name) if name in parameters]
            if not matches:
                suffixes = tuple(_candidate_weight_names(module_name))
                matches = [name for name in parameters if any(name.endswith(suffix) for suffix in suffixes)]
            matches = list(dict.fromkeys(matches))
            if len(matches) != 1:
                raise KeyError(f"could not uniquely map {module_name} to a base weight; matches={matches}")
            name = matches[0]
            parameter = parameters[name]
            if tuple(parameter.shape) != tuple(delta.shape):
                raise ValueError(f"shape mismatch for {name}: base={tuple(parameter.shape)}, delta={tuple(delta.shape)}")
            parameter.add_(delta.to(device=parameter.device, dtype=parameter.dtype), alpha=float(alpha))
            mapped[module_name] = name
    return mapped


def configure_fresh_lora_trainability(model: torch.nn.Module) -> tuple[torch.nn.Parameter, ...]:
    trainable = []
    for name, parameter in model.named_parameters():
        qualified = f".{name}"
        enabled = ".lora_A." in qualified or ".lora_B." in qualified
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError("fresh model exposes no trainable LoRA parameters")
    return tuple(trainable)


def assert_optimizer_matches(optimizer: torch.optim.Optimizer, expected: tuple[torch.nn.Parameter, ...]) -> None:
    actual_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    expected_ids = {id(p) for p in expected}
    if actual_ids != expected_ids:
        raise AssertionError(f"optimizer parameter mismatch: missing={len(expected_ids-actual_ids)}, extra={len(actual_ids-expected_ids)}")
