# Leave-one-context-out transfer of task coefficients

Ratios are the unweighted mean of validation-selected `oracle_fair` c_i/alpha values in other contexts containing task i. The held-out context's alpha scales that mean. Test scores were not used to form ratios, and no new search was run. DTD has no other context in the seven-context set, so its ratio is neutral 1.0 and is flagged as unsupported.

## Test results

All accuracy scores below are mean or worst-task normalized accuracy; deltas are percentage points.

| Context | Transfer coefficients | Transfer Δ mean vs dc_rank | Transfer Δ worst vs dc_rank | Transfer Δ mean vs oracle_fair | Transfer Δ worst vs oracle_fair |
|---|---|---:|---:|---:|---:|
| dtd-eurosat-gtsrb-svhn | dtd=0.5000, eurosat=0.6528, gtsrb=0.5625, svhn=0.4583 | +2.320 | +0.187 | -0.625 | +0.373 |
| eurosat-gtsrb-mnist-svhn | eurosat=0.7767, gtsrb=0.7100, mnist=0.5381, svhn=0.6000 | +1.587 | +6.526 | -1.709 | +5.766 |
| eurosat-gtsrb-sun397-svhn | eurosat=0.7767, gtsrb=0.7100, sun397=0.4286, svhn=0.5500 | +2.114 | +1.678 | +0.187 | +1.856 |
| stanford_cars-eurosat-gtsrb-mnist | stanford_cars=0.5714, eurosat=0.7933, gtsrb=0.7100, mnist=0.5048 | +1.557 | +7.582 | -0.946 | +1.823 |
| stanford_cars-eurosat-gtsrb-resisc45 | stanford_cars=0.5500, eurosat=0.9839, gtsrb=0.8575, resisc45=0.7000 | -2.754 | -10.003 | -2.754 | -10.003 |
| stanford_cars-eurosat-mnist-resisc45 | stanford_cars=0.4714, eurosat=0.8433, mnist=0.4714, resisc45=0.6000 | -1.589 | -6.185 | -1.589 | -6.185 |
| stanford_cars-eurosat-mnist-sun397 | stanford_cars=0.5833, eurosat=0.9839, mnist=0.5833, sun397=0.3500 | -0.688 | +0.690 | -1.400 | -1.258 |

## Seven-context averages

| Split | Method | Mean normalized | Worst normalized | Δ mean vs dc_rank | Δ worst vs dc_rank |
|---|---|---:|---:|---:|---:|
| Validation | dc_rank | 80.266 | 70.449 | +0.000 | +0.000 |
| Validation | loco_transfer | 80.917 | 71.854 | +0.651 | +1.405 |
| Validation | oracle_fair | 82.203 | 72.784 | +1.937 | +2.335 |
| Test | dc_rank | 80.866 | 72.094 | +0.000 | +0.000 |
| Test | loco_transfer | 81.230 | 72.161 | +0.364 | +0.068 |
| Test | oracle_fair | 82.492 | 73.251 | +1.626 | +1.157 |

## Interpretation limits

This is a transfer check of validation-selected coefficients, not another search. Coefficients are estimated from other contexts' `oracle_fair` vectors; task-context counts differ, and some tasks have only one prior context. The held-out test results assess transfer for these seven overlapping contexts only.

## Validation results

| Context | Transfer Δ mean vs dc_rank | Transfer Δ worst vs dc_rank |
|---|---:|---:|
| dtd-eurosat-gtsrb-svhn | +2.290 | +0.664 |
| eurosat-gtsrb-mnist-svhn | +1.307 | +6.914 |
| eurosat-gtsrb-sun397-svhn | +2.749 | +3.266 |
| stanford_cars-eurosat-gtsrb-mnist | +2.083 | +9.131 |
| stanford_cars-eurosat-gtsrb-resisc45 | -1.952 | -7.867 |
| stanford_cars-eurosat-mnist-resisc45 | -1.447 | -3.895 |
| stanford_cars-eurosat-mnist-sun397 | -0.477 | +1.622 |

## Transfer support by task

The counts below are other contexts contributing a ratio for each held-out task. They are based on the seven saved contexts only.

| Task | Other-context count |
|---|---:|
| dtd | 0 |
| eurosat | 6 |
| gtsrb | 4 |
| mnist | 3 |
| resisc45 | 1 |
| stanford_cars | 3 |
| sun397 | 1 |
| svhn | 2 |
