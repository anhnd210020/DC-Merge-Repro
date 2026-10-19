# Execution configuration

`dcmerge_env.sh` is the exact non-secret project path configuration used by the
Experiment 4 shell wrapper. The wrapper invoked the packaged driver with these
effective settings:

```text
repo:   /workspace/ducanh/DC-Merge
model:  /workspace/models/clip-vit-base-patch32
ft-dir: /workspace/models/DC-Merge-Vision-B32-r16-8task/lora_checkpoints
heads:  /workspace/models/DC-Merge-Vision-B32-r16-8task/lora_heads
data:   /workspace/datasets8
exp3:   /srv/dcmerge/results/experiment3
out:    /srv/dcmerge/results/experiment4_task_transfer_20261019
device: cuda
```

Runtime: NVIDIA GeForce RTX 3090, Python 3.9.23, PyTorch 2.8.0+cu128, CUDA 12.8.
The source paths and hashes, exact context list, elapsed time, and evaluation
counts are recorded in `../results/run_manifest.json`. No authentication token
was required or recorded by this evaluation.
