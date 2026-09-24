# Task-scale ablation interpretation template

> Fill this document only after validation is complete and, where explicitly
> authorized by the frozen manifest, final test is complete. Do not infer
> unobserved results.

## Experiment identity

- Git commit: `<commit>`
- Series: `<primary diagnostic-calibrated controlled redistribution | secondary heuristic stress test>`
- Scale-config hash: `<hash>`
- Frozen-manifest hash: `<hash>`
- Validation objective: mean normalized accuracy across eight tasks
- Reported split: `<validation | frozen final test>`

## Observed intervention

- EuroSAT input scale range: `<...>`
- EuroSAT realized effective-weight range: `<...>`
- SUN397 input scale range: `<...>`
- SUN397 realized effective-weight range: `<...>`
- Any non-monotonic input-scale/realized-weight behavior: `<...>`

## Performance response

- EuroSAT retention by preregistered point: `<...>`
- SUN397 retention by preregistered point: `<...>`
- Mean and worst-task performance by point: `<...>`
- Selected global alpha by point: `<...>`

## Supported pattern

Select and justify only the pattern supported by the completed results.

For causal allocation language, use the primary pair-total-controlled series.
Report the original heuristic configurations separately as secondary stress
tests because their realized EuroSAT+SUN397 pair total increases substantially.

1. **Allocation-supported pattern:** increasing realized EuroSAT contribution
   consistently improves EuroSAT retention while SUN397 decreases in a
   controlled tradeoff. Evidence: `<...>`
2. **Conflict-supported pattern:** realized EuroSAT weight increases
   substantially but EuroSAT performance does not recover, indicating that
   contribution magnitude alone is insufficient. Evidence: `<...>`
3. **Mixed/nonlinear pattern:** performance is non-monotonic, consistent with
   task scaling changing TIES sign consensus or interference. Evidence: `<...>`
4. **Global-cost pattern:** EuroSAT improves while aggregate or worst-task
   performance degrades strongly, indicating redistribution rather than free
   recovery. Evidence: `<...>`

## Qualified conclusion

`<State what this controlled intervention supports in this specific rank-16,
eight-task setting. Do not write “low weight causes low accuracy.” If the
allocation hypothesis is supported, describe it as evidence from this ablation,
not a general causal law or a claim about the original DC-Merge algorithm.>`

## Limitations

- Five preregistered task-scale points and one model/task setting.
- Realized effective weight is a post-hoc squared-Frobenius block diagnostic,
  not a learned DC-Merge quantity.
- Positive uniform task scaling preserves that task's independently selected
  top-k support apart from numerical/tie edge cases. It can still alter TIES
  sign consensus because consensus uses the magnitude-weighted task sum, so the
  intervention is not a pure change to an otherwise fixed final coefficient.
- `<Additional observed limitations>`
