"""Class names and prompt templates for the 8 standard tasks.

The per-task files are vendored verbatim from FusionBench, so the zero-shot heads
match the heads the FusionBench fine-tuned CLIP encoders were trained against.
"""
import importlib

_MODULES = {
    "sun397": "sun397",
    "stanford-cars": "stanford_cars",
    "resisc45": "resisc45",
    "eurosat": "eurosat",
    "svhn": "svhn",
    "gtsrb": "gtsrb",
    "mnist": "mnist",
    "dtd": "dtd",
}


def get_classnames_and_templates(task: str):
    mod = importlib.import_module(f"exp8.zeroshot.{_MODULES[task]}")
    return list(mod.classnames), list(mod.templates)
