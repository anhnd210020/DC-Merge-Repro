#!/usr/bin/env python3
"""Compare regenerated heads with released heads without using them as inputs."""

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as functional

from pipeline_common import ROOT, TASKS, utc_now, write_json


OFFICIAL = Path(
    "/workspace/models/DC-Merge-Vision-B32-r16-8task/lora_heads/ViT-B-32"
)
REGENERATED = ROOT / "heads" / "ViT-B-32"
EXPECTED_CLASSES = {
    "stanford_cars": 196,
    "dtd": 47,
    "eurosat": 10,
    "gtsrb": 43,
    "mnist": 10,
    "resisc45": 45,
    "sun397": 397,
    "svhn": 10,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()
    results = {}
    failures = []

    for task in TASKS:
        regenerated_path = REGENERATED / f"{task}_head.pt"
        official_path = OFFICIAL / f"{task}_head.pt"
        if not regenerated_path.is_file() or not official_path.is_file():
            missing = [
                str(path)
                for path in (regenerated_path, official_path)
                if not path.is_file()
            ]
            failures.append(f"{task}: missing head(s): {', '.join(missing)}")
            continue
        regenerated = torch.load(
            regenerated_path, map_location="cpu", weights_only=True
        )
        official = torch.load(official_path, map_location="cpu", weights_only=True)
        shapes_match = regenerated.shape == official.shape
        dtypes_match = regenerated.dtype == official.dtype
        expected_shape = (EXPECTED_CLASSES[task], 512)
        shape_is_expected = tuple(regenerated.shape) == expected_shape
        if shapes_match:
            difference = (regenerated.to(torch.float64) - official.to(torch.float64)).abs()
            max_difference = float(difference.max())
            mean_difference = float(difference.mean())
            cosine = float(
                functional.cosine_similarity(
                    regenerated.reshape(1, -1).to(torch.float64),
                    official.reshape(1, -1).to(torch.float64),
                ).item()
            )
            allclose = bool(
                torch.allclose(regenerated, official, rtol=args.rtol, atol=args.atol)
            )
        else:
            max_difference = math.inf
            mean_difference = math.inf
            cosine = float("nan")
            allclose = False
        passed = shapes_match and shape_is_expected and dtypes_match and allclose
        result = {
            "regenerated_path": str(regenerated_path),
            "official_path": str(official_path),
            "shape": list(regenerated.shape),
            "official_shape": list(official.shape),
            "dtype": str(regenerated.dtype),
            "official_dtype": str(official.dtype),
            "max_absolute_difference": max_difference,
            "mean_absolute_difference": mean_difference,
            "cosine_similarity": cosine,
            "allclose": allclose,
            "rtol": args.rtol,
            "atol": args.atol,
            "passed": passed,
        }
        results[task] = result
        print(
            f"HEAD_PREFLIGHT task={task} shape={tuple(regenerated.shape)} "
            f"dtype={regenerated.dtype} max_abs_diff={max_difference:.12g} "
            f"mean_abs_diff={mean_difference:.12g} cosine={cosine:.12g} "
            f"allclose={allclose} rtol={args.rtol} atol={args.atol}",
            flush=True,
        )
        if not passed:
            failures.append(
                f"{task}: substantial head mismatch "
                f"(shape_match={shapes_match}, expected_shape={shape_is_expected}, "
                f"dtype_match={dtypes_match}, allclose={allclose}, "
                f"max_abs_diff={max_difference:.12g}, "
                f"mean_abs_diff={mean_difference:.12g}, cosine={cosine:.12g})"
            )

    report = {
        "generated_at": utc_now(),
        "purpose": "reproducibility comparison only; official heads are not training inputs",
        "tolerance": {"rtol": args.rtol, "atol": args.atol},
        "tasks": results,
        "passed": not failures,
        "failures": failures,
    }
    write_json(ROOT / "results" / "head_preflight.json", report)
    if failures:
        print("HEAD_PREFLIGHT_ABORT: " + " | ".join(failures), flush=True)
        raise SystemExit(1)
    print("HEAD_PREFLIGHT_PASS", flush=True)


if __name__ == "__main__":
    main()
