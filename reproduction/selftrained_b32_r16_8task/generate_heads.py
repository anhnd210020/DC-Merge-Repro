#!/usr/bin/env python3
"""Regenerate the authors' deterministic zero-shot classification heads."""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from pipeline_common import MODEL_DIR, ROOT, TASKS, VISION_DIR, require_environment, task_config


def build_head(model, tokenizer, classnames, templates, device):
    weights = []
    with torch.no_grad():
        for classname in tqdm(classnames, desc="classes"):
            embeddings = []
            for template in templates:
                tokens = tokenizer(template(classname))
                tokens = {
                    key: torch.tensor(value, device=device).reshape(1, -1)
                    for key, value in tokens.items()
                }
                embedding = model.text_projection(model.text_model(**tokens)[1])
                embeddings.append(embedding)
            embeddings = torch.cat(embeddings, dim=0)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            embedding = embeddings.mean(dim=0, keepdim=True)
            embedding /= embedding.norm()
            weights.append(embedding)
        weights = torch.stack(weights, dim=0).transpose(0, 2)
        weights *= model.logit_scale.exp()
        return weights.squeeze().float().transpose(0, 1).cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    require_environment()
    # Dataset configs construct shuffled-index paths relative to this directory.
    import os
    os.chdir(VISION_DIR)
    from dataset.templates import get_templates
    from utils import prepare_data

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA head generation was requested, but CUDA is unavailable")

    model = CLIPModel.from_pretrained(str(MODEL_DIR)).eval().to(device)
    processor = CLIPProcessor.from_pretrained(str(MODEL_DIR))
    output_dir = ROOT / "heads" / "ViT-B-32"
    output_dir.mkdir(parents=True, exist_ok=True)

    for task in TASKS:
        output = output_dir / f"{task}_head.pt"
        if not args.force and output.is_file() and output.stat().st_size > 0:
            print(f"HEAD_SKIP_VALID={task} path={output}", flush=True)
            continue
        cfg = task_config(task)
        cfg["train_preprocess"] = processor.image_processor
        cfg["eval_preprocess"] = processor.image_processor
        loaders = prepare_data(cfg, device=device)
        head = build_head(
            model,
            processor.tokenizer,
            loaders["test"]["class_names"],
            get_templates(task),
            device,
        )
        temporary = Path(str(output) + ".tmp")
        torch.save(head, temporary)
        temporary.replace(output)
        print(
            f"HEAD_DONE={task} device={device} shape={tuple(head.shape)} "
            f"dtype={head.dtype} path={output}",
            flush=True,
        )


if __name__ == "__main__":
    main()
