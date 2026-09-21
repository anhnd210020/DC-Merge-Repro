# Extracted ViT-B/32 rank-16 eight-task training recipe

This is an evidence record for the queued experiment. Values below come from
repository commit `7a0644b55951845d8b2624f2782518de271dc3e4`; they are not inferred from the
released adapter files. Appendix E.2 of the paper explicitly documents the
pretrained CLIP family, LoRA structure, optimizer family, scheduler family,
loss, learning rate, weight decay, and label smoothing. Batch size, warmup,
step limits, evaluation frequency, early stopping, and seed below are **code
defaults** from `vision_lora_merge/vision_training.py` and
`configs/8vision_train.py`, not paper-specified values.

## Effective recipe

- Pretrained model: Hugging Face `openai/clip-vit-base-patch32`, loaded from
  `/workspace/models/clip-vit-base-patch32`. Its downloaded snapshot metadata
  records revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.
- LoRA: rank 16, alpha 16, dropout 0.1, bias `none`; target every vision
  attention `q_proj`, `k_proj`, `v_proj`, and `out_proj` in all 12 layers.
  PEFT freezes the pretrained vision parameters; the CLIP visual projection is
  also explicitly frozen. The trainable tensors are only LoRA A/B parameters
  (1,179,648 parameters for ViT-B/32). PEFT's default initialization applies:
  A is initialized and B starts at zero.
- Objective: normalize image embeddings, multiply their dot product with the
  fixed zero-shot class vectors by an additional factor of 100, and use cross
  entropy with label smoothing 0.0.
- Optimizer: PyTorch AdamW over `model.parameters()`, learning rate `3e-4`,
  weight decay `1e-1`; frozen parameters receive no gradients or updates.
- Schedule: repository `cosine_lr`, 500-step linear warmup and cosine decay to
  `max_steps=10000`. The scheduler is called after each optimizer update. Thus
  the first update uses the optimizer's initial `3e-4`; scheduler step zero then
  assigns `6e-7` for the next update. This ordering is preserved.
- Duration: `epochs=10000` as an outer cap and `max_steps=10000`. The source
  tests `step >= max_steps` after the update, so absent early stopping it can
  perform updates numbered 0 through 10000 (10,001 updates).
- Batch size: 32 for every task. There is no gradient accumulation (one
  optimizer step per batch), clipping, or distributed training. AMP uses
  `torch.cuda.amp.GradScaler`; evaluation uses autocast.
- Seed: 420 before each independent task in the repaired sequential runner,
  using exactly the source's `torch.manual_seed`, `random.seed`, and
  `numpy.random.seed`. CUDA deterministic algorithms and cuDNN determinism are
  not enabled by the authors' code.
- Checkpoint selection: evaluate train, validation, and test every 2,000 global
  steps. Save when validation accuracy is greater than the previous best plus
  0.001. Stop after five consecutive evaluation points without such an
  improvement. The test score is observed at every selection point but is not
  used to choose the checkpoint. The queued repair saves the selected PEFT
  adapter directory rather than the source script's incompatible full `.pt`
  state dict.
- Training preprocessing and evaluation preprocessing are the same
  `CLIPProcessor.image_processor`: resize to 224, bicubic resampling, center
  crop 224, RGB conversion, and normalization with mean
  `[0.48145466, 0.4578275, 0.40821073]` and standard deviation
  `[0.26862954, 0.26130258, 0.27577711]`. No random image augmentation is
  present. Training loaders shuffle except DTD, whose loader omits
  `shuffle=True` even though the config says `shuffle_train=True`.

## Data splits and loaders

| Task | Train | Validation | Test | Workers | Split rule |
|---|---:|---:|---:|---:|---|
| stanford_cars | 8,144 | 1,608 | 6,433 | 16 | Fixed shuffled index file partitions the official 8,041-image test set 20/80 |
| dtd | 1,880 | 1,880 | 1,880 | 16 | Existing `train`, `val`, and `test` folders; training is sequential, not shuffled |
| eurosat | 21,600 | 2,700 | 2,700 | 16 | Existing `train`, `val`, and `test` folders |
| gtsrb | 26,640 | 2,526 | 10,104 | 16 | Fixed shuffled index file partitions the official 12,630-image test set 20/80 |
| mnist | 60,000 | 2,000 | 8,000 | 8 | Fixed shuffled index file partitions the official 10,000-image test set 20/80 |
| resisc45 | 18,900 | 6,300 | 6,300 | 16 | Repository-provided filename lists define train/val/test |
| sun397 | 19,850 | 3,970 | 15,880 | 16 | Fixed shuffled index file partitions the provided 19,850-image `val` folder 20/80 |
| svhn | 73,257 | 5,206 | 20,826 | 8 | Fixed shuffled index file partitions the official 26,032-image test set 20/80 |

The fixed index files are the repository's files under
`vision_lora_merge/dataset/shuffled_idxs`; they are not regenerated.

## Classification heads and evaluation

The head is not a learned classifier. For each dataset class name, the authors'
dataset-specific text templates are encoded by the pretrained CLIP text model.
Each prompt embedding is normalized, embeddings are averaged and normalized
again, and the vector is multiplied by `exp(model.logit_scale)`. This process is
deterministic and independent of any released LoRA adapter, so the queued
pipeline regenerates heads under `heads/ViT-B-32` using the authors' templates
and class-name ordering.

Before any GPU training, the master pipeline requires at least 10 GiB free on
`/workspace`, hides CUDA, forcibly regenerates all eight heads on CPU, and
compares them with the released heads using `torch.allclose` at `rtol=1e-4` and
`atol=1e-5`. The released heads are read only for this reproducibility check and
are never copied into the experiment. Any shape, dtype, or numerical mismatch
aborts the pipeline before the first adapter is trained; a successful check
prints `HEAD_PREFLIGHT_PASS` to the master log and writes detailed metrics to
`results/head_preflight.json`.

Single-task validation and test accuracy are measured from each saved adapter.
For DC-Merge, seed 400 is set, LoRA `B @ A` deltas are merged with `dc_merge`,
linear smoothing, and rho 5.0. Alphas 0.1 through 3.0 inclusive in increments
of 0.1 are evaluated on validation. The alpha maximizing the mean of each
merged validation accuracy normalized by that task's self-trained validation
accuracy is selected; strict `>` means the first alpha wins ties. That alpha is
then evaluated once on test, with normalized test accuracy divided by each
self-trained adapter's test accuracy.

## Ambiguities and source defects

- **AMBIGUOUS:** The paper does not state warmup length, number of steps,
  evaluation frequency, early stopping, batch size, seed, or augmentation.
  Those values above are code defaults. The paper does explicitly state AdamW,
  cosine scheduling, cross entropy, learning rate `3e-4`, weight decay `1e-1`,
  label smoothing 0, and all listed LoRA settings.
- **AMBIGUOUS:** No justification is given for evaluating the held-out test
  split every 2,000 steps during training. The code selects only by validation
  accuracy, but repeatedly observes test performance.
- **AMBIGUOUS:** The source does not request deterministic CUDA algorithms,
  seed DataLoader workers explicitly, or state a reproducibility tolerance.
- **AMBIGUOUS:** The exact historical software/hardware environment used to
  create Table 1 is not tied to a checkpoint manifest. This run records its own
  versions and hardware per task.
- `vision_training.py` currently hard-codes `CONFIG_NAME='16vision_train'`,
  leaves cache/output paths blank, and comments out the `--task_name` filter;
  as written, one invocation loops over every configured task.
- `configs/8vision_train.py` currently selects ViT-B/16, not B/32, despite
  containing commented architecture alternatives.
- The training source saves a full model `state_dict` to a `.pt` file, while
  the released checkpoints and merge loader use PEFT directories containing
  `adapter_config.json` and adapter weights. The queued runner preserves the
  optimization and selection logic but saves the best PEFT adapter in the
  format the merge code actually consumes.
- `generate_clip_heads.py` has blank paths and hard-codes the 16-task config;
  its helper also relies on a global `dataset_name`. The queued generator fixes
  only those wiring defects while preserving the head mathematics.
- `crop_ratio` and `shuffle_train` appear in config dictionaries but are not
  generally consumed by loaders; in particular DTD training remains unshuffled.
