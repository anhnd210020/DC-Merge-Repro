#!/usr/bin/env python3
"""Strict PEFT 0.3-compatible serialization for the copied CLIP LoRA model."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Mapping, Optional

import torch


LORA_KEY = re.compile(
    r"^base_model\.model\.encoder\.layers\.(\d+)\.self_attn\."
    r"(q_proj|k_proj|v_proj|out_proj)\.lora_([AB])(?:\.default)?\.weight$"
)
EXPECTED_LAYERS = 12
EXPECTED_MODULES = ("q_proj", "k_proj", "v_proj", "out_proj")
EXPECTED_TENSORS = EXPECTED_LAYERS * len(EXPECTED_MODULES) * 2


def canonical_lora_key(key: str) -> str:
    return key.replace(".lora_A.default.weight", ".lora_A.weight").replace(
        ".lora_B.default.weight", ".lora_B.weight"
    )


def validate_lora_state(state: Mapping[str, torch.Tensor]) -> None:
    if len(state) != EXPECTED_TENSORS:
        raise ValueError(f"expected {EXPECTED_TENSORS} LoRA tensors, found {len(state)}")
    seen = set()
    for key, tensor in state.items():
        match = LORA_KEY.fullmatch(key)
        if not match:
            raise ValueError(f"unexpected LoRA key: {key}")
        layer, module, factor = int(match.group(1)), match.group(2), match.group(3)
        if layer not in range(EXPECTED_LAYERS):
            raise ValueError(f"unexpected layer in {key}")
        expected_shape = (16, 768) if factor == "A" else (768, 16)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"wrong shape for {key}: {tuple(tensor.shape)} != {expected_shape}"
            )
        if tensor.dtype != torch.float32:
            raise ValueError(f"wrong dtype for {key}: {tensor.dtype}")
        seen.add((layer, module, factor))
    expected = {
        (layer, module, factor)
        for layer in range(EXPECTED_LAYERS)
        for module in EXPECTED_MODULES
        for factor in ("A", "B")
    }
    if seen != expected:
        raise ValueError(f"LoRA tensor coverage mismatch: missing={sorted(expected - seen)}")


def extract_lora_state(peft_model) -> "OrderedDict[str, torch.Tensor]":
    extracted = OrderedDict()
    for key, value in peft_model.state_dict().items():
        if ".lora_A." in key or ".lora_B." in key:
            canonical = canonical_lora_key(key)
            if canonical in extracted:
                raise ValueError(f"duplicate canonical LoRA key: {canonical}")
            extracted[canonical] = value.detach().cpu().float().contiguous().clone()
    validate_lora_state(extracted)
    return extracted


def expected_adapter_config() -> Dict[str, object]:
    # This deliberately matches the released PEFT 0.3 adapter schema.
    return {
        "base_model_name_or_path": None,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": 16,
        "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"],
        "task_type": None,
    }


def validate_adapter_directory(directory: Path) -> "OrderedDict[str, torch.Tensor]":
    config_path = directory / "adapter_config.json"
    weights_path = directory / "adapter_model.bin"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not weights_path.is_file() or weights_path.stat().st_size == 0:
        raise FileNotFoundError(weights_path)
    config = json.loads(config_path.read_text())
    if config != expected_adapter_config():
        raise ValueError(f"adapter config mismatch: {config}")
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"adapter weights must be a dict, got {type(state)}")
    ordered = OrderedDict((key, value) for key, value in state.items())
    validate_lora_state(ordered)
    return ordered


def save_adapter_atomic(peft_model, destination: Path):
    candidate = destination.with_name(destination.name + ".candidate")
    backup = destination.with_name(destination.name + ".previous")
    shutil.rmtree(candidate, ignore_errors=True)
    shutil.rmtree(backup, ignore_errors=True)
    candidate.mkdir(parents=True)
    trained_state = extract_lora_state(peft_model)
    (candidate / "adapter_config.json").write_text(
        json.dumps(expected_adapter_config(), indent=2) + "\n"
    )
    torch.save(trained_state, candidate / "adapter_model.bin")
    saved_state = validate_adapter_directory(candidate)
    compare_lora_states(trained_state, saved_state, require_exact=True)
    if destination.exists():
        destination.replace(backup)
    candidate.replace(destination)
    shutil.rmtree(backup, ignore_errors=True)
    return trained_state


def compare_lora_states(
    expected: Mapping[str, torch.Tensor],
    actual: Mapping[str, torch.Tensor],
    require_exact: bool = True,
) -> Dict[str, object]:
    validate_lora_state(expected)
    validate_lora_state(actual)
    if set(expected) != set(actual):
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise ValueError(
            f"LoRA key mismatch: missing={missing}, unexpected={unexpected}"
        )
    max_abs_difference = 0.0
    exact = True
    allclose = True
    for key in expected:
        if expected[key].shape != actual[key].shape:
            raise ValueError(f"shape mismatch for {key}")
        difference = float((expected[key] - actual[key]).abs().max())
        max_abs_difference = max(max_abs_difference, difference)
        exact = exact and torch.equal(expected[key], actual[key])
        allclose = allclose and torch.allclose(
            expected[key], actual[key], rtol=0.0, atol=0.0
        )
    if require_exact and not exact:
        raise ValueError(f"LoRA round-trip is not exact; max diff={max_abs_difference}")
    return {
        "tensor_count": len(expected),
        "exact_equal": exact,
        "allclose": allclose,
        "max_absolute_difference": max_abs_difference,
    }


def tensor_digests(state: Mapping[str, torch.Tensor]) -> Dict[str, str]:
    validate_lora_state(state)
    return {
        key: hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
        for key, value in state.items()
    }
