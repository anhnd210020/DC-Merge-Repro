from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch


FACTOR_RE = re.compile(r"^(.*)\.lora_([AB])(?:\.default)?\.weight$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_checksum(directory: Path) -> str:
    directory = Path(directory)
    files = [directory / "adapter_config.json"]
    files.extend(path for path in (directory / "adapter_model.bin", directory / "adapter_model.safetensors") if path.is_file())
    if len(files) != 2 or not all(path.is_file() for path in files):
        raise FileNotFoundError(f"adapter requires config and exactly one weights file: {directory}")
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def load_adapter(directory: Path) -> tuple[dict[str, Any], OrderedDict[str, torch.Tensor]]:
    directory = Path(directory)
    config = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    binary = directory / "adapter_model.bin"
    safe = directory / "adapter_model.safetensors"
    if binary.is_file() and safe.is_file():
        raise ValueError(f"ambiguous adapter weights in {directory}")
    if binary.is_file():
        state = torch.load(binary, map_location="cpu", weights_only=True)
    elif safe.is_file():
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise RuntimeError("safetensors is required to load this adapter") from error
        state = load_file(str(safe), device="cpu")
    else:
        raise FileNotFoundError(f"missing adapter weights in {directory}")
    if not isinstance(state, Mapping):
        raise TypeError("adapter state must be a tensor mapping")
    return config, OrderedDict(sorted((str(k), v.detach().cpu()) for k, v in state.items()))


def lora_scaling(config: Mapping[str, Any]) -> float:
    rank = int(config["r"])
    alpha = float(config["lora_alpha"])
    if rank <= 0:
        raise ValueError("LoRA rank must be positive")
    return alpha / rank


def paired_factors(state: Mapping[str, torch.Tensor]) -> OrderedDict[str, tuple[torch.Tensor, torch.Tensor]]:
    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in state.items():
        match = FACTOR_RE.fullmatch(key)
        if match:
            grouped.setdefault(match.group(1), {})[match.group(2)] = tensor
    if not grouped:
        raise ValueError("adapter contains no LoRA A/B factors")
    result: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
    for name in sorted(grouped):
        factors = grouped[name]
        if set(factors) != {"A", "B"}:
            raise ValueError(f"unpaired LoRA factors for {name}: {sorted(factors)}")
        a, b = factors["A"], factors["B"]
        if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
            raise ValueError(f"invalid LoRA shapes for {name}: A={tuple(a.shape)}, B={tuple(b.shape)}")
        result[name] = (a, b)
    return result


def effective_deltas(config: Mapping[str, Any], state: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    scale = lora_scaling(config)
    rank = int(config["r"])
    result = OrderedDict()
    for name, (a, b) in paired_factors(state).items():
        if a.shape[0] != rank:
            raise ValueError(f"configured rank {rank} does not match {name} factor rank {a.shape[0]}")
        result[name] = scale * (b @ a)
    return result


def adapter_report(directory: Path) -> dict[str, Any]:
    config, state = load_adapter(directory)
    factors = paired_factors(state)
    configured_targets = set(config.get("target_modules", []))
    if not configured_targets:
        raise ValueError("adapter config has no target_modules")
    modules = []
    for name, (a, b) in factors.items():
        if a.shape[0] != int(config["r"]):
            raise ValueError(f"configured rank does not match factors for {name}")
        if name.rsplit(".", 1)[-1] not in configured_targets:
            raise ValueError(f"LoRA factor module is not declared in adapter config: {name}")
        modules.append({"module": name, "rank": int(a.shape[0]), "a_shape": list(a.shape), "b_shape": list(b.shape)})
    return {
        "path": str(Path(directory).resolve()), "sha256": adapter_checksum(directory),
        "rank": int(config["r"]), "lora_alpha": float(config["lora_alpha"]),
        "scaling": lora_scaling(config), "target_modules": list(config.get("target_modules", [])),
        "modules": modules,
    }


def lora_state_from_model_state(state: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    result = OrderedDict(
        sorted(
            (str(key), value.detach().cpu().clone())
            for key, value in state.items()
            if FACTOR_RE.fullmatch(str(key))
        )
    )
    paired_factors(result)
    return result


def validate_saved_adapter(
    directory: Path,
    selected_state: Mapping[str, torch.Tensor],
    expected_rank: int,
    expected_alpha: float,
    expected_target_modules: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Reload an adapter from disk and compare its effective deltas to selection state."""
    config, reloaded_state = load_adapter(directory)
    report = adapter_report(directory)
    expected_scaling = float(expected_alpha) / int(expected_rank)
    if report["rank"] != int(expected_rank):
        raise RuntimeError(f"saved adapter rank mismatch: {report['rank']} != {expected_rank}")
    if report["lora_alpha"] != float(expected_alpha):
        raise RuntimeError(f"saved adapter alpha mismatch: {report['lora_alpha']} != {expected_alpha}")
    if report["scaling"] != expected_scaling:
        raise RuntimeError(f"saved adapter scaling mismatch: {report['scaling']} != {expected_scaling}")
    if set(report["target_modules"]) != set(expected_target_modules):
        raise RuntimeError(
            f"saved adapter target modules mismatch: {report['target_modules']} != {list(expected_target_modules)}"
        )
    selected_lora = lora_state_from_model_state(selected_state)
    expected_config = {"r": int(expected_rank), "lora_alpha": float(expected_alpha)}
    selected_deltas = effective_deltas(expected_config, selected_lora)
    reloaded_deltas = effective_deltas(config, reloaded_state)
    if set(selected_deltas) != set(reloaded_deltas):
        raise RuntimeError(
            f"saved adapter effective-delta module mismatch: selected={sorted(selected_deltas)}, "
            f"reloaded={sorted(reloaded_deltas)}"
        )
    maximum = 0.0
    for name in selected_deltas:
        difference = float((selected_deltas[name] - reloaded_deltas[name]).abs().max())
        maximum = max(maximum, difference)
        if not torch.allclose(selected_deltas[name], reloaded_deltas[name], rtol=1e-6, atol=1e-7):
            raise RuntimeError(f"saved adapter effective delta differs for {name}: max_abs={difference}")
    return {
        **report,
        "disk_reload_succeeded": True,
        "effective_delta_module_count": len(reloaded_deltas),
        "effective_delta_max_absolute_difference": maximum,
        "effective_delta_allclose_rtol": 1e-6,
        "effective_delta_allclose_atol": 1e-7,
        "validation_level": "actual adapter file reload and effective-delta comparison",
    }
