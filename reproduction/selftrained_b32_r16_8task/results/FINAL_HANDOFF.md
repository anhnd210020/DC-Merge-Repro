# Final DC-Merge self-trained handoff

- Timestamp: 2026-09-21T16:16:06.295127+00:00
- Git commit: 7a0644b55951845d8b2624f2782518de271dc3e4
- Pipeline status: COMPLETE
- All expected artifacts exist: True

## Task statuses and single-task accuracy

| Task | Status | Validation (%) | Test (%) |
|---|---|---:|---:|
| stanford_cars | DONE | 75.43532338 | 73.26286336 |
| dtd | DONE | 67.5 | 67.65957447 |
| eurosat | DONE | 98.51851852 | 98.44444444 |
| gtsrb | DONE | 96.79334917 | 96.89231987 |
| mnist | DONE | 99.5 | 99.1375 |
| resisc45 | DONE | 93.98412698 | 93.07936508 |
| sun397 | DONE | 72.36775819 | 70.99496222 |
| svhn | DONE | 96.44640799 | 96.25468165 |

- Average validation accuracy: 87.568185529%
- Average test accuracy: 86.965713886%

## Self-trained DC-Merge

- Best alpha: 0.8
- Best validation normalized average: 75.18056652%
- Final average absolute accuracy: 64.424346229%
- Final average normalized accuracy: 75.118902946%

| Task | Final absolute (%) | Final normalized (%) |
|---|---:|---:|
| stanford_cars | 63.065443805 | 86.081052409 |
| dtd | 50.319148936 | 74.37106918 |
| eurosat | 46.185185185 | 46.914973667 |
| gtsrb | 68.873713381 | 71.08273749 |
| mnist | 84.6 | 85.3360232 |
| resisc45 | 76.174603175 | 81.838335607 |
| sun397 | 65.579345088 | 92.371828983 |
| svhn | 60.59733026 | 62.955203032 |

## References and differences

- Official reproduction best alpha: 0.4
- Official reproduction average accuracy: 64.149%
- Official reproduction average normalized accuracy: 73.879%
- Paper Table-1 average accuracy: 64.17%
- Paper Table-1 average normalized accuracy: 73.9%
- Self-trained minus official, absolute: 0.275346229 pp
- Self-trained minus official, normalized: 1.239902946 pp
- Self-trained minus paper, absolute: 0.254346229 pp
- Self-trained minus paper, normalized: 1.218902946 pp

## Artifact checks

- pipeline_status_complete: True
- all_task_statuses_done: True
- selftrained_val_json: True
- selftrained_test_json: True
- dcmerge_result_json: True
- reproduction_comparison_markdown: True
- smoke_config_jsons: True
- final_dcmerge_log: True
- master_training_log: True
- all_checkpoints: True
- all_heads: True
- python_scripts: True
- shell_scripts: True
- selftrained_config: True

## Final artifact paths

- single_task_val: `/workspace/selftrained_b32_r16_8task/results/selftrained_val_acc.json`
- single_task_test: `/workspace/selftrained_b32_r16_8task/results/selftrained_test_acc.json`
- dcmerge_results: `/workspace/selftrained_b32_r16_8task/results/selftrained_dcmerge_results.json`
- comparison: `/workspace/selftrained_b32_r16_8task/results/reproduction_comparison.md`
- smoke_config_jsons:
  - `/workspace/selftrained_b32_r16_8task/results/dcmerge_config_smoke.json`
  - `/workspace/selftrained_b32_r16_8task/results/head_preflight.json`
  - `/workspace/selftrained_b32_r16_8task/results/serializer_smoke_test.json`
  - `/workspace/selftrained_b32_r16_8task/results/stanford_cars_evaluation_smoke.json`
- final_dcmerge_log: `/workspace/selftrained_b32_r16_8task/logs/dcmerge_selftrained.log`
- master_training_log: `/workspace/selftrained_b32_r16_8task/logs/master.log`
- checkpoints: `/workspace/selftrained_b32_r16_8task/checkpoints`
- heads: `/workspace/selftrained_b32_r16_8task/heads`
- scripts:
  - `/workspace/selftrained_b32_r16_8task/adapter_io.py`
  - `/workspace/selftrained_b32_r16_8task/baseline_processes.py`
  - `/workspace/selftrained_b32_r16_8task/compare_heads.py`
  - `/workspace/selftrained_b32_r16_8task/evaluate_dcmerge.py`
  - `/workspace/selftrained_b32_r16_8task/evaluate_single_tasks.py`
  - `/workspace/selftrained_b32_r16_8task/generate_heads.py`
  - `/workspace/selftrained_b32_r16_8task/make_comparison.py`
  - `/workspace/selftrained_b32_r16_8task/pipeline_common.py`
  - `/workspace/selftrained_b32_r16_8task/smoke_test_adapter_roundtrip.py`
  - `/workspace/selftrained_b32_r16_8task/train_task.py`
  - `/workspace/selftrained_b32_r16_8task/validate_checkpoint.py`
  - `/workspace/selftrained_b32_r16_8task/overnight_finalize.sh`
  - `/workspace/selftrained_b32_r16_8task/resume_post_training.sh`
  - `/workspace/selftrained_b32_r16_8task/run_selftrain_pipeline.sh`
  - `/workspace/selftrained_b32_r16_8task/watch_baseline_then_train.sh`
- selftrained_config: `/workspace/DC-Merge-Repro/vision_lora_merge/configs/vitB32_r16_8task_selftrained.py`

## Log warnings/errors

- /workspace/selftrained_b32_r16_8task/logs/dcmerge.log:2: Traceback (most recent call last):
- /workspace/selftrained_b32_r16_8task/logs/dcmerge.log:180: /workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/dcmerge_selftrained.log:295: /workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/dtd.log:5: /workspace/selftrained_b32_r16_8task/train_task.py:171: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/dtd.log:41: training dtd:   0%|          | 33/10000 [03:07<15:46:42,  5.70s/it]/workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/dtd.log:150: Traceback (most recent call last):
- /workspace/selftrained_b32_r16_8task/logs/dtd.log:167: /workspace/selftrained_b32_r16_8task/train_task.py:174: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/dtd.log:203: training dtd:   0%|          | 33/10000 [03:06<15:40:56,  5.66s/it]/workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/eurosat.log:5: /workspace/selftrained_b32_r16_8task/train_task.py:171: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/eurosat.log:10: training eurosat:   0%|          | 2/10000 [01:41<141:21:51, 50.90s/it]/workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/eurosat.log:453: Traceback (most recent call last):
- /workspace/selftrained_b32_r16_8task/logs/eurosat.log:470: /workspace/selftrained_b32_r16_8task/train_task.py:174: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/eurosat.log:475: training eurosat:   0%|          | 2/10000 [01:43<143:26:17, 51.65s/it]/workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/gtsrb.log:6: /workspace/selftrained_b32_r16_8task/train_task.py:171: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/gtsrb.log:11: training gtsrb:   0%|          | 2/10000 [02:06<175:49:04, 63.31s/it]/workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/gtsrb.log:646: Traceback (most recent call last):
- /workspace/selftrained_b32_r16_8task/logs/gtsrb.log:664: /workspace/selftrained_b32_r16_8task/train_task.py:174: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/gtsrb.log:669: training gtsrb:   0%|          | 2/10000 [02:06<175:26:07, 63.17s/it]/workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/master.log:607: /workspace/selftrained_b32_r16_8task/train_task.py:171: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/master.log:617: training stanford_cars:   0%|          | 7/10000 [02:20<55:53:05, 20.13s/it]/workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/master.log:912: Traceback (most recent call last):
- /workspace/selftrained_b32_r16_8task/logs/master.log:930: /workspace/selftrained_b32_r16_8task/train_task.py:171: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/master.log:966: training dtd:   0%|          | 33/10000 [03:07<15:46:42,  5.70s/it]/workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/master.log:1075: Traceback (most recent call last):
- /workspace/selftrained_b32_r16_8task/logs/master.log:1093: /workspace/selftrained_b32_r16_8task/train_task.py:171: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/master.log:1098: training eurosat:   0%|          | 2/10000 [01:41<141:21:51, 50.90s/it]/workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/master.log:1541: Traceback (most recent call last):
- /workspace/selftrained_b32_r16_8task/logs/master.log:1560: /workspace/selftrained_b32_r16_8task/train_task.py:171: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/master.log:1565: training gtsrb:   0%|          | 2/10000 [02:06<175:49:04, 63.31s/it]/workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/master.log:2200: Traceback (most recent call last):
- /workspace/selftrained_b32_r16_8task/logs/master.log:2219: /workspace/selftrained_b32_r16_8task/train_task.py:171: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/master.log:2223: training mnist:   0%|          | 1/10000 [02:20<389:39:54, 140.29s/it]/workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/master.log:3336: Traceback (most recent call last):
- /workspace/selftrained_b32_r16_8task/logs/master.log:3354: /workspace/selftrained_b32_r16_8task/train_task.py:171: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/master.log:3360: training resisc45:   0%|          | 3/10000 [02:15<126:04:39, 45.40s/it]/workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/master.log:3873: Traceback (most recent call last):
- /workspace/selftrained_b32_r16_8task/logs/master.log:3892: /workspace/selftrained_b32_r16_8task/train_task.py:171: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/master.log:4498: /workspace/selftrained_b32_r16_8task/train_task.py:174: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/master.log:4508: training stanford_cars:   0%|          | 7/10000 [02:20<55:44:14, 20.08s/it]/workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/master.log:5988: /workspace/selftrained_b32_r16_8task/train_task.py:174: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/master.log:6024: training dtd:   0%|          | 33/10000 [03:06<15:40:56,  5.66s/it]/workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/master.log:6720: /workspace/selftrained_b32_r16_8task/train_task.py:174: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/master.log:6725: training eurosat:   0%|          | 2/10000 [01:43<143:26:17, 51.65s/it]/workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/master.log:8946: /workspace/selftrained_b32_r16_8task/train_task.py:174: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/master.log:8951: training gtsrb:   0%|          | 2/10000 [02:06<175:26:07, 63.17s/it]/workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/master.log:12126: /workspace/selftrained_b32_r16_8task/train_task.py:174: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/master.log:12130: training mnist:   0%|          | 1/10000 [02:19<388:45:19, 139.97s/it]/workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/master.log:17704: /workspace/selftrained_b32_r16_8task/train_task.py:174: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/master.log:17710: training resisc45:   0%|          | 3/10000 [02:17<127:44:19, 46.00s/it]/workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/master.log:20292: /workspace/selftrained_b32_r16_8task/train_task.py:174: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/master.log:20298: training sun397:   0%|          | 3/10000 [03:37<192:15:55, 69.24s/it]/workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/master.log:24530: /workspace/selftrained_b32_r16_8task/train_task.py:174: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/master.log:24533: training svhn:   0%|          | 0/10000 [00:00<?, ?it/s]/workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/master.log:32421: /workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/master.log:32426: Traceback (most recent call last):
- /workspace/selftrained_b32_r16_8task/logs/mnist.log:6: /workspace/selftrained_b32_r16_8task/train_task.py:171: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/mnist.log:10: training mnist:   0%|          | 1/10000 [02:20<389:39:54, 140.29s/it]/workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/mnist.log:1123: Traceback (most recent call last):
- /workspace/selftrained_b32_r16_8task/logs/mnist.log:1141: /workspace/selftrained_b32_r16_8task/train_task.py:174: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/mnist.log:1145: training mnist:   0%|          | 1/10000 [02:19<388:45:19, 139.97s/it]/workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/post_training.log:23: /workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/post_training.log:932: Traceback (most recent call last):
- /workspace/selftrained_b32_r16_8task/logs/resisc45.log:5: /workspace/selftrained_b32_r16_8task/train_task.py:171: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/resisc45.log:11: training resisc45:   0%|          | 3/10000 [02:15<126:04:39, 45.40s/it]/workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/resisc45.log:524: Traceback (most recent call last):
- /workspace/selftrained_b32_r16_8task/logs/resisc45.log:541: /workspace/selftrained_b32_r16_8task/train_task.py:174: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/resisc45.log:547: training resisc45:   0%|          | 3/10000 [02:17<127:44:19, 46.00s/it]/workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/serializer_smoke_test.log:1: Traceback (most recent call last):
- /workspace/selftrained_b32_r16_8task/logs/single_task_eval.log:3: /workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/single_task_eval.log:8: Traceback (most recent call last):
- /workspace/selftrained_b32_r16_8task/logs/single_task_eval.log:24: /workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/stanford_cars.log:6: /workspace/selftrained_b32_r16_8task/train_task.py:171: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/stanford_cars.log:16: training stanford_cars:   0%|          | 7/10000 [02:20<55:53:05, 20.13s/it]/workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/stanford_cars.log:311: Traceback (most recent call last):
- /workspace/selftrained_b32_r16_8task/logs/stanford_cars.log:329: /workspace/selftrained_b32_r16_8task/train_task.py:174: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/stanford_cars.log:339: training stanford_cars:   0%|          | 7/10000 [02:20<55:44:14, 20.08s/it]/workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/stanford_cars_postfix_smoke.log:9: /workspace/selftrained_b32_r16_8task/evaluate_single_tasks.py:98: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/stanford_cars_postfix_smoke.log:12: /workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/sun397.log:6: /workspace/selftrained_b32_r16_8task/train_task.py:171: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/sun397.log:14: /workspace/selftrained_b32_r16_8task/train_task.py:174: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/sun397.log:20: training sun397:   0%|          | 3/10000 [03:37<192:15:55, 69.24s/it]/workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/svhn.log:6: /workspace/selftrained_b32_r16_8task/train_task.py:174: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
- /workspace/selftrained_b32_r16_8task/logs/svhn.log:9: training svhn:   0%|          | 0/10000 [00:00<?, ?it/s]/workspace/DC-Merge-Repro/vision_lora_merge/utils.py:227: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.

## Git status

```text
?? vision_lora_merge/__pycache__/
?? vision_lora_merge/configs/__pycache__/
?? vision_lora_merge/configs/vitB32_r16_8task_selftrained.py
?? vision_lora_merge/dataset/__pycache__/
?? vision_lora_merge/logs/
?? vision_lora_merge/models/__pycache__/
```
