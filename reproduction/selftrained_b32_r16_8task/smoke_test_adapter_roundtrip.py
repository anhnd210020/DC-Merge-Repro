#!/usr/bin/env python3
"""One short GPU optimizer/save/load/inference round-trip smoke test."""

import gc
import os
import sys
import tempfile
from pathlib import Path

import torch
from peft import LoraConfig

from adapter_io import (
    compare_lora_states,
    extract_lora_state,
    save_adapter_atomic,
    tensor_digests,
    validate_adapter_directory,
)
from pipeline_common import MODEL_DIR, ROOT, TRAINING, VISION_DIR, seed_everything, utc_now, write_json
from train_task import LoadedVisionAdapter


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU smoke test requires CUDA")
    sys.path.insert(0, str(VISION_DIR))
    os.chdir(VISION_DIR)
    from models.huggingface_clip import HFLoRACLIPVisionModel

    device = "cuda"
    seed_everything(12345)
    config = LoraConfig(
        r=TRAINING["lora_rank"],
        lora_alpha=TRAINING["lora_alpha"],
        target_modules=TRAINING["target_modules"],
        lora_dropout=TRAINING["lora_dropout"],
        bias=TRAINING["bias"],
    )
    model = HFLoRACLIPVisionModel(
        model_name=str(MODEL_DIR),
        cache_dir=str(MODEL_DIR),
        lora_config=config.__dict__,
        device=device,
    )
    peft_model = model.vision_model
    peft_config = peft_model.peft_config["default"]
    trained_object_type = f"{type(peft_model).__module__}.{type(peft_model).__name__}"
    trained_config_type = f"{type(peft_config).__module__}.{type(peft_config).__name__}"
    trained_peft_type = repr(peft_config.peft_type)
    trained_config_rank = peft_config.r
    trained_config_targets = peft_config.target_modules
    initial_state = extract_lora_state(peft_model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=TRAINING["lr"],
        weight_decay=TRAINING["weight_decay"],
    )
    loss_function = torch.nn.CrossEntropyLoss()
    pixels = torch.randn(2, 3, 224, 224, device=device)
    labels = torch.tensor([0, 1], device=device)
    class_vectors = torch.randn(10, 512, device=device)
    class_vectors /= class_vectors.norm(dim=-1, keepdim=True)
    losses = []
    for _ in range(3):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        encodings = model({"pixel_values": pixels})
        encodings = encodings / encodings.norm(dim=-1, keepdim=True)
        loss = loss_function(10.0 * encodings @ class_vectors.T, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    trained_state = extract_lora_state(peft_model)
    changed_tensors = sum(
        not torch.equal(initial_state[key], trained_state[key]) for key in trained_state
    )
    if changed_tensors == 0:
        raise RuntimeError("optimizer smoke test did not change any LoRA tensor")

    smoke_parent = ROOT / "state"
    with tempfile.TemporaryDirectory(prefix="adapter_roundtrip_", dir=smoke_parent) as temp:
        checkpoint = Path(temp) / "adapter"
        pre_save_state = save_adapter_atomic(peft_model, checkpoint)
        saved_state = validate_adapter_directory(checkpoint)
        saved_comparison = compare_lora_states(
            pre_save_state, saved_state, require_exact=True
        )
        saved_digests = tensor_digests(saved_state)

        del optimizer, peft_model, model
        gc.collect()
        torch.cuda.empty_cache()

        loaded = LoadedVisionAdapter(checkpoint, device).eval()
        loaded_state = extract_lora_state(loaded.vision_model)
        loaded_comparison = compare_lora_states(
            pre_save_state, loaded_state, require_exact=True
        )
        with torch.no_grad():
            inference = loaded({"pixel_values": pixels[:1]})
        if tuple(inference.shape) != (1, 512) or not torch.isfinite(inference).all():
            raise RuntimeError(f"invalid inference result: {tuple(inference.shape)}")

        result = {
            "completed_at": utc_now(),
            "temporary_checkpoint_only": True,
            "temporary_checkpoint_removed_on_exit": True,
            "optimizer_steps": 3,
            "losses": losses,
            "changed_lora_tensors": changed_tensors,
            "peft_object_type": trained_object_type,
            "trained_copied_object_diagnostic": {
                "object_type": trained_object_type,
                "adapter_name": "default",
                "peft_config_type": trained_config_type,
                "peft_type_after_authors_deepcopy": trained_peft_type,
                "r_after_authors_deepcopy": trained_config_rank,
                "target_modules_after_authors_deepcopy": trained_config_targets,
            },
            "saved_tensor_comparison": saved_comparison,
            "loaded_tensor_comparison": loaded_comparison,
            "tensor_count": len(saved_state),
            "all_saved_tensor_digests_present": len(saved_digests) == 96,
            "inference_succeeded": True,
            "inference_output_shape": list(inference.shape),
        }
        print(
            f"SERIALIZER_SMOKE_PASS steps=3 changed_tensors={changed_tensors} "
            f"saved_exact={saved_comparison['exact_equal']} "
            f"loaded_exact={loaded_comparison['exact_equal']} "
            f"tensor_count={len(saved_state)} inference_shape={tuple(inference.shape)}",
            flush=True,
        )
    write_json(ROOT / "results" / "serializer_smoke_test.json", result)


if __name__ == "__main__":
    main()
