#!/usr/bin/env python3
"""Independent DONE/skip gate for a saved self-trained adapter."""

import argparse
import json

import torch

from adapter_io import (
    compare_lora_states,
    extract_lora_state,
    tensor_digests,
    validate_adapter_directory,
)
from pipeline_common import ROOT, TASKS, utc_now, write_json
from train_task import LoadedVisionAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA validation requested but unavailable")

    checkpoint = ROOT / "checkpoints" / args.task
    metadata_path = ROOT / "state" / f"{args.task}.metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("status") != "DONE":
        raise RuntimeError(f"metadata status is not DONE: {metadata.get('status')}")

    saved_state = validate_adapter_directory(checkpoint)
    expected_digests = metadata.get("trained_lora_sha256")
    actual_digests = tensor_digests(saved_state)
    if not expected_digests or actual_digests != expected_digests:
        raise RuntimeError("saved tensors do not match trained-tensor digests")

    model = LoadedVisionAdapter(checkpoint, args.device).eval()
    loaded_state = extract_lora_state(model.vision_model)
    comparison = compare_lora_states(saved_state, loaded_state, require_exact=True)
    batch = {"pixel_values": torch.randn(1, 3, 224, 224, device=args.device)}
    with torch.no_grad():
        outputs = model(batch)
    if tuple(outputs.shape) != (1, 512) or not torch.isfinite(outputs).all():
        raise RuntimeError(f"inference failed: shape={tuple(outputs.shape)}")

    result = {
        "task": args.task,
        "validated_at": utc_now(),
        "adapter_config_exists": True,
        "adapter_model_bin_exists": True,
        "saved_keys_shapes_valid": True,
        "saved_matches_trained_digests": True,
        "fresh_load_succeeded": True,
        "round_trip": comparison,
        "inference_succeeded": True,
        "inference_output_shape": list(outputs.shape),
    }
    write_json(ROOT / "state" / f"{args.task}.done_validation.json", result)
    print(
        f"CHECKPOINT_VALIDATION_PASS task={args.task} tensors={comparison['tensor_count']} "
        f"exact={comparison['exact_equal']} max_abs_diff={comparison['max_absolute_difference']} "
        f"inference_shape={tuple(outputs.shape)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
