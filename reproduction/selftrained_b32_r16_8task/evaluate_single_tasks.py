#!/usr/bin/env python3
"""Measure validation and test accuracy for the self-trained adapters."""

import argparse
import gc
import os
import time

import torch
from transformers import CLIPProcessor

from pipeline_common import (
    MODEL_DIR,
    ROOT,
    TASKS,
    VISION_DIR,
    checkpoint_files_valid,
    require_environment,
    task_config,
    utc_now,
    write_json,
)
from train_task import LoadedVisionAdapter


def inspect_batch(inputs, labels):
    """Print the concrete batch contract used by the CLIP vision wrapper."""
    pixel_values = inputs["pixel_values"]
    print(f"BATCH type(inputs)={type(inputs)!r}", flush=True)
    print(
        "BATCH type(inputs['pixel_values'])="
        f"{type(pixel_values)!r}",
        flush=True,
    )
    if isinstance(pixel_values, list):
        print(f"BATCH pixel_values_list_length={len(pixel_values)}", flush=True)
        if pixel_values:
            element = pixel_values[0]
            print(f"BATCH pixel_values_element_type={type(element)!r}", flush=True)
            print(
                f"BATCH pixel_values_element_shape={getattr(element, 'shape', None)}",
                flush=True,
            )
    else:
        print(f"BATCH pixel_values_shape={pixel_values.shape}", flush=True)
        print(f"BATCH pixel_values_dtype={pixel_values.dtype}", flush=True)
    print(f"BATCH labels_type={type(labels)!r}", flush=True)
    print(f"BATCH labels_shape={labels.shape}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=TASKS,
        help="Evaluate only one task and write a smoke-result JSON, not the 8-task JSONs.",
    )
    parser.add_argument(
        "--smoke-first-batch",
        action="store_true",
        help="Inspect and run one batch before the complete split evaluations.",
    )
    args = parser.parse_args()

    require_environment()
    os.chdir(VISION_DIR)
    from utils import evaluate_cliphead, prepare_data

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required for queued evaluation")
    processor = CLIPProcessor.from_pretrained(str(MODEL_DIR))
    val_results = {}
    test_results = {}
    started = time.monotonic()
    tasks = (args.task,) if args.task else TASKS

    for task in tasks:
        checkpoint = ROOT / "checkpoints" / task
        if not checkpoint_files_valid(checkpoint):
            raise RuntimeError(f"invalid self-trained checkpoint: {checkpoint}")
        cfg = task_config(task)
        # Match HFLoRACLIPVisionModel.train_preprocess/val_preprocess exactly.
        # Without return_tensors="pt", BatchFeature.pixel_values remains a list;
        # with it, DataLoader produces [B, 1, C, H, W], which the authors'
        # forward method deliberately squeezes to [B, C, H, W].
        preprocess = lambda image: processor.image_processor(  # noqa: E731
            image, return_tensors="pt"
        )
        cfg["train_preprocess"] = preprocess
        cfg["eval_preprocess"] = preprocess
        data = prepare_data(cfg, device=device)
        head = torch.load(cfg["clip_encodings"], map_location=device, weights_only=True)
        model = LoadedVisionAdapter(checkpoint, device).eval()
        if args.smoke_first_batch:
            inputs, labels = next(iter(data["test"]["val"]))
            inspect_batch(inputs, labels)
            with torch.no_grad(), torch.cuda.amp.autocast():
                encodings = model(inputs.to(device))
                normed_encodings = encodings / encodings.norm(dim=-1, keepdim=True)
                logits = normed_encodings @ head.T
            expected_shape = (labels.shape[0], head.shape[0])
            if tuple(logits.shape) != expected_shape or not torch.isfinite(logits).all():
                raise RuntimeError(
                    "first-batch inference failed: "
                    f"shape={tuple(logits.shape)} expected={expected_shape}"
                )
            print(
                f"FIRST_BATCH_INFERENCE_PASS task={task} "
                f"batch_size={labels.shape[0]} encoding_shape={tuple(encodings.shape)} "
                f"logits_shape={tuple(logits.shape)}",
                flush=True,
            )
        val_results[task] = round(
            100.0 * evaluate_cliphead(model, data["test"]["val"], head), 8
        )
        test_results[task] = round(
            100.0 * evaluate_cliphead(model, data["test"]["test"], head), 8
        )
        print(
            f"SINGLE_TASK={task} VAL={val_results[task]:.8f} "
            f"TEST={test_results[task]:.8f}",
            flush=True,
        )
        del model, data, head
        gc.collect()
        torch.cuda.empty_cache()

    if args.task:
        write_json(
            ROOT / "results" / f"{args.task}_evaluation_smoke.json",
            {
                "generated_at": utc_now(),
                "task": args.task,
                "pretrained_model": str(MODEL_DIR),
                "adapter_checkpoint": str(ROOT / "checkpoints" / args.task),
                "classification_head": task_config(args.task)["clip_encodings"],
                "first_batch_inference_passed": args.smoke_first_batch,
                "validation_accuracy_percent": val_results[args.task],
                "test_accuracy_percent": test_results[args.task],
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        return

    write_json(ROOT / "results" / "selftrained_val_acc.json", val_results)
    write_json(ROOT / "results" / "selftrained_test_acc.json", test_results)
    write_json(
        ROOT / "results" / "selftrained_single_task_summary.json",
        {
            "generated_at": utc_now(),
            "validation_accuracy_percent": val_results,
            "test_accuracy_percent": test_results,
            "individual_average_test_accuracy_percent": sum(test_results.values())
            / len(test_results),
            "elapsed_seconds": time.monotonic() - started,
        },
    )


if __name__ == "__main__":
    main()
