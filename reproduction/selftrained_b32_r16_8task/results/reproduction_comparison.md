# DC-Merge ViT-B/32 rank-16 eight-task reproduction comparison

Generated: 2026-09-21T16:15:03.823295+00:00

## Single-task test accuracy

| Task | Authors (%) | Self-trained (%) | Difference (pp) |
|---|---:|---:|---:|
| stanford_cars | 76.610 | 73.263 | -3.347 |
| dtd | 67.340 | 67.660 | +0.320 |
| eurosat | 98.190 | 98.444 | +0.254 |
| gtsrb | 98.290 | 96.892 | -1.398 |
| mnist | 99.150 | 99.138 | -0.013 |
| resisc45 | 93.970 | 93.079 | -0.891 |
| sun397 | 72.490 | 70.995 | -1.495 |
| svhn | 96.500 | 96.255 | -0.245 |

## Aggregate comparison

| Metric | Table 1 (%) | Official-checkpoint reproduction (%) | Self-trained (%) |
|---|---:|---:|---:|
| Individual average | 87.82 | n/a | 86.966 |
| DC-Merge absolute average | 64.17 | 64.149 | 64.424 |
| DC-Merge normalized average | 73.90 | 73.879 | 75.119 |

## Run details

- Self-trained selected alpha: 0.8
- Official-checkpoint reproduction selected alpha: 0.4
- Total adapter training and per-task verification time: 9598.724 seconds
- Self-trained DC-Merge evaluation time: 2295.899 seconds
- Official-checkpoint reproduction time: 2374.0 seconds
- Self-trained single-task evaluation time: 271.767 seconds

Authors' single-task values are read only as reference from the repository's `single_task/our_finetuned/ViT-B-32/test_acc.json`. All self-trained values come from independently initialized adapters under this experiment root.
