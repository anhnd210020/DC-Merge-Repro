---
tags:
  - clip
  - lora
  - model-merging
  - reproducibility
---

# ViT-B/32 rank-16 eight-task DC-Merge reproduction

This directory records two separate experiments performed from this repository:

- **Experiment A — official-checkpoint reproduction:** evaluates the authors'
  released LoRA checkpoints with the repository's DC-Merge implementation.
- **Experiment B — independently self-trained reproduction:** trains eight new
  LoRA adapters from the pretrained OpenAI CLIP ViT-B/32 base model, selects
  each adapter using validation accuracy, runs the DC-Merge validation-alpha
  sweep, and evaluates the selected merge once on test.

The Experiment B adapters are independently trained reproduction checkpoints.
They are **not** the authors' official released LoRA weights. Binary adapters,
self-generated classification heads, detailed logs, and the final backup archive
are stored in the private Hugging Face model repository
`anhnd210020/DC-Merge-Selftrained-B32-r16-8task`; they are intentionally not
committed to Git.

## Model and tasks

- Base model: pretrained OpenAI CLIP ViT-B/32
  (`openai/clip-vit-base-patch32`)
- LoRA rank/alpha/dropout: 16 / 16 / 0.1
- LoRA targets: vision-attention `q_proj`, `k_proj`, `v_proj`, and `out_proj`
- Datasets: Stanford Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, SVHN
- Independent task-training seed: 420
- DC-Merge seed: 400
- Merge: linear smoothing, `rho=5.0`
- Alpha search: 0.1 through 3.0 inclusive in increments of 0.1

## Important training recipe

Each task starts from the same pretrained CLIP vision encoder and trains only
LoRA parameters. The optimizer is AdamW with learning rate `3e-4` and weight
decay `1e-1`; the schedule uses 500-step linear warmup followed by cosine decay
toward the 10,000-step limit. Batch size is 32, label smoothing is 0, and model
selection uses validation accuracy with patience 5 and minimum improvement
0.001. Fixed zero-shot CLIP text heads are generated independently using the
repository's dataset prompts and class ordering.

The exact effective recipe, dataset splits, preprocessing, source ambiguities,
and validation gates are documented in
[`REPRODUCTION_RECIPE.md`](REPRODUCTION_RECIPE.md).

## Reproduction repairs and validation

Three implementation issues were isolated without changing the intended
experiment:

1. **Adapter serialization.** With the legacy PEFT version, the authors'
   deep-copy path did not preserve usable adapter configuration metadata for a
   reliable `save_pretrained` round trip. `adapter_io.py` explicitly extracts
   the 96 expected rank-16 LoRA A/B tensors, writes the PEFT-compatible adapter
   config and weights atomically, and validates exact tensor equality after a
   fresh load. `serializer_smoke_test.json` records the passing round trip.
2. **Evaluation preprocessing.** The CLIP image processor must be called with
   `return_tensors="pt"` so the DataLoader produces the tensor shape expected by
   the repository model wrapper. `evaluate_single_tasks.py` now matches the
   model's training/validation preprocessing and verifies a first batch before
   full evaluation.
3. **Runtime config/device loading.** Static configs intentionally contain no
   `device` key. The official evaluator calls
   `get_config_from_name(name, device=device)`, which injects the runtime device
   before `prepare_experiment_config`. `evaluate_dcmerge.py` now uses that same
   path and validates all eight intended adapters, heads, and datasets before
   merging. `dcmerge_config_smoke.json` records the passing CUDA configuration.

No adapter was retrained to address evaluation or merge wiring issues.

## Final results

| Metric | Experiment A: official checkpoints | Experiment B: self-trained | Paper Table 1 |
|---|---:|---:|---:|
| Selected alpha | 0.4 | 0.8 | — |
| DC-Merge average accuracy (%) | 64.149 | 64.424 | 64.17 |
| DC-Merge average normalized accuracy (%) | 73.879 | 75.119 | 73.90 |

The self-trained adapters average 86.966% single-task test accuracy. Full
per-task values and unrounded machine-readable metrics are available in the
result artifacts below.

## Artifact locations

- Final human-readable handoff: [`results/FINAL_HANDOFF.md`](results/FINAL_HANDOFF.md)
- Machine-readable handoff: [`results/final_handoff.json`](results/final_handoff.json)
- DC-Merge alpha sweep and final test: [`results/selftrained_dcmerge_results.json`](results/selftrained_dcmerge_results.json)
- Single-task validation: [`results/selftrained_val_acc.json`](results/selftrained_val_acc.json)
- Single-task test: [`results/selftrained_test_acc.json`](results/selftrained_test_acc.json)
- Aggregate comparison: [`results/reproduction_comparison.md`](results/reproduction_comparison.md)
- Head validation: [`results/head_preflight.json`](results/head_preflight.json)
- Serializer validation: [`results/serializer_smoke_test.json`](results/serializer_smoke_test.json)
- Runtime config validation: [`results/dcmerge_config_smoke.json`](results/dcmerge_config_smoke.json)
- Self-trained static config: [`../../vision_lora_merge/configs/vitB32_r16_8task_selftrained.py`](../../vision_lora_merge/configs/vitB32_r16_8task_selftrained.py)

The source repository is
[`anhnd210020/DC-Merge-Repro`](https://github.com/anhnd210020/DC-Merge-Repro).
