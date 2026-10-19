# Experiment 2 replay audit

## Assessment

The current replay differences are small (maximum absolute difference 0.158353 percentage points on validation and 0.148148 points on test across the seven selected `dc_rank` context baselines). The evidence is consistent with numerical evaluation sensitivity on predictions near a class boundary. It does not establish that the exact archived-versus-current changes are random GPU nondeterminism, so “normal evaluation variation” is plausible but not proven.

Experiment 2’s reported 0.32 pp cross-check does not independently explain the current archived mismatch. That comparison was between two different arithmetic paths (`dc_official` and full-lane composition), rather than repeated executions of one fixed evaluation path. The old lane implementation accumulated separately scaled task lanes; the current uniform path scales the completed update. A tensor-level check of the fixed path matched the direct update exactly, and an identical-input current-runtime test gave zero prediction changes across two runs. Thus the 0.32 pp is evidence that low-margin predictions can respond to evaluation arithmetic, not a measured same-path run-to-run tolerance.

## Verified

- Same-input, same-runtime GTSRB validation repeat: 1,872/2,526 correct in both passes, zero changed predictions, identical input hashes and labels. The official evaluator returned the same 74.109264% score.
- A sequential current-runtime diagnostic using the evaluator’s `argmax` found 30 changed GTSRB predictions between AMP and FP32 and 19 between the old lane accumulation and fixed uniform update; the old-lane and fixed paths had the same aggregate accuracy in this check. The affected examples have zero or one-FP16-step top-two margins. This supports sensitivity to arithmetic/precision near ties, but is not an archived per-example comparison.
- Current direct replay compared at each archived selected alpha: max absolute validation difference 0.158353 pp (`dtd-eurosat-gtsrb-svhn`, GTSRB); max absolute test difference 0.148148 pp (`stanford_cars-eurosat-gtsrb-resisc45`, EuroSAT). Complete per-task deltas are in `experiment2-current-baseline/manifest.json`.
- The source files for `run_week1.py`, `run_official.py`, `vision_lora_merge/utils.py`, and `vision_lora_merge/models/huggingface_clip.py` match the Oct 16 handoff snapshot by SHA-256. The current direct E2 replay does not use the changed `lanes.py` composition path.
- All eight local adapter and head files match the Hugging Face model bundle’s recorded LFS/Xet hashes. The local CLIP checkpoint matches the recorded model-file hash and is pinned to commit `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`; the adapter/head repository revision is `fcde80f8dccb513997c97e7215ff02a589b2994a`.
- The current datasets repository is pinned at `42fc343338f31c2f41cfa85e97d20e22dfbf9d45`; relevant split index files and task source/config files match the handoff package. Current preprocessing uses the same saved slow CLIP image processor and same size/crop/normalization configuration, with repeated sample preprocessing matching exactly.
- Current runtime: Python 3.9.23, Torch 2.8.0+cu128, torchvision 0.23.0+cu128, Transformers 4.55.2, Pillow 11.3.0, NumPy 1.26.3; GPU RTX 3090, driver 580.142. The old logs do not provide a complete package lock or GPU/driver record.

## Unverified or unavailable

- Oct 16 per-example logits/predictions were not saved, so the exact changed sample IDs and flip counts between the archived run and current replay cannot be recovered from aggregate accuracy JSON.
- The archived run lacks exact model/data file hashes and a complete runtime/hardware record. Current model/checkpoint hashes and repository revisions are verified, but there are no retained Oct 16 weights or runtime to compare directly.
- The extracted dataset tree was not cryptographically re-hashed against the original dataset tar archives. The dataset repository and split inputs are pinned/matched, making a dataset change less likely, but byte-for-byte identity of every extracted image remains unverified.
- The CPU/GPU library component responsible for the residual difference cannot be isolated from the available archived records. Current repeat stability rules out observed same-path nondeterminism in the controlled GTSRB check, not all possible cross-run effects.
- An earlier exploratory diagnostic used `topk(2)[:,0]` rather than the canonical `argmax` and mixed model/precision paths; it is not used as evidence. The corrected sequential diagnostic above is the valid current-path comparison.

## Experiment 3 baseline handling

Per supervisor authorization, the coefficient search continues against a separate current-environment baseline overlay. It retains each archived selected alpha and replaces only that `dc_rank` validation accuracy row and test accuracy with the current direct replay at the same alpha. CT rows and all original Experiment 2 files remain untouched. The oracle retains its original coefficient grid, objectives, two-pass limit, and strict 0.1 pp reproduction check; the check is relative to the current-environment baseline. Archived-versus-current differences remain documented as limitations.

Overlay details and all 28 task-level archived/current comparisons: `/srv/dcmerge/results/experiment2-current-baseline/README.md` and `manifest.json`.
