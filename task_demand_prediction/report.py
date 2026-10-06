"""Write a stage-specific report aligned with the registered Experiment Card."""
from __future__ import annotations
from pathlib import Path
from .storage import atomic_text

def write_report(output: Path, stage: str, variation, correlations, predictive) -> Path:
    path=Path(output)/stage/'REPORT.md'
    if stage=='validation':
        label='Validation'; caution='Validation only. Freeze feature and estimator choices before opening final-test outputs.'
    else:
        label='Final test'; caution='Final-test labels are scored only after prediction; grouped models were fit on validation rows excluding each held task/context.'
    lines=[f'# Task merge demand prediction ? {label}', '', caution, '',
        '**Card formulas:** Signed deficits use `R_i(1.5)`, `R_i(16)`, and `R_i(1)` as the scale, rank, and density references. AUC maps each tested range to [0,1] and integrates signed trapezoids without another width divisor. Density uses a task-wide rounded budget and deterministic global top-K allocation.', '',
        '## Findings', '',
        '1. **Demand heterogeneity.** Between-task sums of squares as fractions of total: '+', '.join(f'{k}={v["between_task_fraction_of_total"]:.3f}' if v['between_task_fraction_of_total'] is not None else f'{k}=not estimable' for k,v in variation.items())+'.',
        '2. **Context dependence.** Within-task/context sums of squares: '+', '.join(f'{k}={v["within_task_context_ss"]:.4g}' if v['within_task_context_ss'] is not None else f'{k}=not estimable' for k,v in variation.items())+'. Contexts are compared within focal task.',
        '3. **Cross-resource association.** Global signed-demand Spearman correlations: '+', '.join(f'{k} ?={v["spearman"]:.3f} (n={v["n"]})' if v['spearman'] is not None else f'{k} not estimable (n={v["n"]})' for k,v in correlations.get('global',{}).items())+'.',
        '4. **Grouped prediction.** Per-fold, aggregate, constant-one and individual-feature metrics are in `grouped_prediction_results.json`. Validation is the model-selection split; final-test labels do not fit or select models.', '',
        '## Interpretation limits', '',
        '- Phase A contains all exact pairs; Phase B contains 25 frozen four-task contexts and 40 focal/context rows; Phase C contains nine exact six-task memberships and 22 focal rows. No measurements transfer between contexts.',
        '- Archived rank raw observations cover the 28 exact Phase A contexts and both splits. Recompute AUC with the frozen axis; input identity is still incomplete. Archived density curves are invalid because per-module rounding collapsed multiple requested levels.',
        '- Synthetic results are not scientific evidence.', '']
    atomic_text(path,'\n'.join(lines)); return path
