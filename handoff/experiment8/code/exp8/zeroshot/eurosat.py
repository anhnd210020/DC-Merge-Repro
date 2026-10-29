# Vendored verbatim from FusionBench (MIT License, (c) 2024 Anke Tang):
# fusion_bench/tasks/clip_classification/eurosat.py
classnames = [
    "annual crop land",
    "forest",
    "brushland or shrubland",
    "highway or road",
    "industrial buildings or commercial buildings",
    "pasture land",
    "permanent crop land",
    "residential buildings or homes or apartments",
    "river",
    "lake or sea",
]

templates = [
    lambda c: f"a centered satellite photo of {c}.",
    lambda c: f"a centered satellite photo of a {c}.",
    lambda c: f"a centered satellite photo of the {c}.",
]
