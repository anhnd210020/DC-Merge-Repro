#!/usr/bin/env python3
"""Train and validate one fresh LoRA adapter using the authors' recipe."""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel
from torch.cuda.amp import GradScaler
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from adapter_io import (
    compare_lora_states,
    extract_lora_state,
    save_adapter_atomic,
    tensor_digests,
    validate_adapter_directory,
)
from pipeline_common import (
    MODEL_DIR,
    ROOT,
    TASKS,
    TRAINING,
    VISION_DIR,
    adapter_weight_file,
    git_commit,
    require_environment,
    runtime_versions,
    seed_everything,
    task_config,
    utc_now,
    write_json,
)


class LoadedVisionAdapter(torch.nn.Module):
    def __init__(self, checkpoint: Path, device: str):
        super().__init__()
        clip_model = CLIPModel.from_pretrained(str(MODEL_DIR))
        self.vision_model = PeftModel.from_pretrained(
            clip_model.vision_model, str(checkpoint)
        ).to(device)
        self.vision_head = clip_model.visual_projection.to(device)
        self.vision_head.weight.requires_grad = False

    def forward(self, inputs):
        if len(inputs["pixel_values"].shape) == 5:
            inputs["pixel_values"] = inputs["pixel_values"].squeeze(1)
        return self.vision_head(self.vision_model(**inputs)[1])


def verify_checkpoint(
    task: str, checkpoint: Path, trained_state, cfg, loader_dict, device: str
):
    saved_state = validate_adapter_directory(checkpoint)
    saved_comparison = compare_lora_states(
        trained_state, saved_state, require_exact=True
    )
    model = LoadedVisionAdapter(checkpoint, device).eval()
    loaded_state = extract_lora_state(model.vision_model)
    loaded_comparison = compare_lora_states(
        trained_state, loaded_state, require_exact=True
    )
    batch, labels = next(iter(loader_dict["test"]["test"]))
    with torch.no_grad():
        outputs = model(batch.to(device))
    if outputs.shape != (labels.shape[0], 512) or not torch.isfinite(outputs).all():
        raise RuntimeError(f"invalid verification output: shape={tuple(outputs.shape)}")
    return {
        "adapter_config": str(checkpoint / "adapter_config.json"),
        "adapter_weights": str(adapter_weight_file(checkpoint)),
        "saved_tensor_comparison": saved_comparison,
        "loaded_tensor_comparison": loaded_comparison,
        "saved_lora_keys_and_shapes_valid": True,
        "inference_batch_size": int(labels.shape[0]),
        "inference_output_shape": list(outputs.shape),
        "inference_succeeded": True,
        "verified_at": utc_now(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASKS)
    args = parser.parse_args()
    require_environment()
    os.chdir(VISION_DIR)
    from models.huggingface_clip import HFLoRACLIPVisionModel
    from utils import cosine_lr, evaluate_cliphead, prepare_data

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required for queued LoRA training")

    started_wall = time.monotonic()
    cfg = task_config(args.task)
    effective_training = dict(TRAINING)
    effective_training.update(
        {
            "batch_size": cfg["batch_size"],
            "num_workers": cfg["num_workers"],
            "configured_shuffle_train": cfg.get("shuffle_train"),
            "effective_shuffle_train": args.task != "dtd",
            "val_fraction": cfg.get("val_fraction"),
            "shuffled_idxs": cfg.get("shuffled_idxs"),
            "classification_head": cfg["clip_encodings"],
            "preprocessing": json.loads(
                (MODEL_DIR / "preprocessor_config.json").read_text()
            ),
        }
    )
    metadata = {
        "task": args.task,
        "status": "RUNNING",
        "start_timestamp": utc_now(),
        "command_line": " ".join(sys.argv),
        "git_commit": git_commit(),
        "training": effective_training,
        "versions": runtime_versions(),
    }
    metadata_path = ROOT / "state" / f"{args.task}.metadata.json"
    write_json(metadata_path, metadata)

    try:
        seed_everything(TRAINING["seed"])
        lora_config = LoraConfig(
            r=TRAINING["lora_rank"],
            lora_alpha=TRAINING["lora_alpha"],
            target_modules=TRAINING["target_modules"],
            lora_dropout=TRAINING["lora_dropout"],
            bias=TRAINING["bias"],
        )
        model = HFLoRACLIPVisionModel(
            model_name=str(MODEL_DIR),
            cache_dir=str(MODEL_DIR),
            lora_config=lora_config.__dict__,
            device=device,
        )
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"TRAINABLE_PARAMETERS={trainable}", flush=True)
        print(f"ALL_PARAMETERS={total}", flush=True)

        cfg["train_preprocess"] = model.train_preprocess
        cfg["eval_preprocess"] = model.val_preprocess
        loader_dict = prepare_data(cfg, device=device)
        train_loader = loader_dict["train"]["full"]
        val_loader = loader_dict["test"]["val"]
        test_loader = loader_dict["test"]["test"]
        metadata["dataset_sizes"] = {
            "train": len(train_loader.dataset),
            "validation": len(val_loader.dataset),
            "test": len(test_loader.dataset),
        }
        class_vectors = torch.load(
            cfg["clip_encodings"], map_location=device, weights_only=True
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=TRAINING["lr"],
            weight_decay=TRAINING["weight_decay"],
        )
        scheduler = cosine_lr(
            optimizer, TRAINING["lr"], TRAINING["warm_up"], TRAINING["max_steps"]
        )
        scaler = GradScaler()
        loss_fn = CrossEntropyLoss(label_smoothing=TRAINING["label_smoothing"])
        checkpoint = ROOT / "checkpoints" / args.task
        best_val = 0.0
        best_test = 0.0
        best_train = 0.0
        early_counter = 0
        updates = 0
        stop = False
        best_lora_state = None

        for epoch in tqdm(range(TRAINING["epochs"]), desc=f"training {args.task}"):
            for index, (inputs, labels) in enumerate(train_loader):
                model.train()
                step = index + epoch * len(train_loader)
                optimizer.zero_grad(set_to_none=True)
                encodings = model(inputs.to(device))
                encodings = encodings / encodings.norm(dim=-1, keepdim=True)
                logits = 100.0 * encodings @ class_vectors.T
                loss = loss_fn(logits, labels.to(device))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler(step)
                updates += 1

                if (step + 1) % TRAINING["eval_freq"] == 0:
                    train_acc, train_loss = evaluate_cliphead(
                        model, train_loader, class_vectors, return_loss=True
                    )
                    val_acc, val_loss = evaluate_cliphead(
                        model, val_loader, class_vectors, return_loss=True
                    )
                    test_acc, test_loss = evaluate_cliphead(
                        model, test_loader, class_vectors, return_loss=True
                    )
                    print(
                        f"STEP={step + 1} TRAIN_ACC={train_acc * 100:.6f} "
                        f"VAL_ACC={val_acc * 100:.6f} TEST_ACC={test_acc * 100:.6f} "
                        f"TRAIN_LOSS={train_loss:.8f} VAL_LOSS={val_loss:.8f} "
                        f"TEST_LOSS={test_loss:.8f}",
                        flush=True,
                    )
                    if val_acc > best_val + TRAINING["early_stopping_min_delta"]:
                        best_val, best_test, best_train = val_acc, test_acc, train_acc
                        best_lora_state = save_adapter_atomic(
                            model.vision_model, checkpoint
                        )
                        early_counter = 0
                    else:
                        early_counter += 1
                        if early_counter >= TRAINING["early_stopping_patience"]:
                            print(f"EARLY_STOP_STEP={step + 1}", flush=True)
                            stop = True
                            break

                # This preserves the authors' inclusive check (10,001 updates possible).
                if step >= TRAINING["max_steps"]:
                    stop = True
                    break
            if stop:
                break

        if best_lora_state is None:
            raise RuntimeError("training ended without a validation-selected checkpoint")

        # Release the training model before independently reloading the saved adapter.
        del optimizer, model
        torch.cuda.empty_cache()
        verification = verify_checkpoint(
            args.task,
            checkpoint,
            best_lora_state,
            cfg,
            loader_dict,
            device,
        )
        metadata.update(
            {
                "status": "DONE",
                "trainable_parameters": trainable,
                "all_parameters": total,
                "optimizer_updates": updates,
                "best_validation_accuracy_percent": best_val * 100,
                "test_accuracy_at_best_validation_percent": best_test * 100,
                "training_accuracy_at_best_validation_percent": best_train * 100,
                "trained_lora_sha256": tensor_digests(best_lora_state),
                "verification": verification,
            }
        )
    except Exception as error:
        metadata["status"] = "FAILED"
        metadata["error"] = repr(error)
        metadata["traceback"] = traceback.format_exc()
        raise
    finally:
        metadata["end_timestamp"] = utc_now()
        metadata["elapsed_seconds"] = time.monotonic() - started_wall
        write_json(metadata_path, metadata)


if __name__ == "__main__":
    main()
