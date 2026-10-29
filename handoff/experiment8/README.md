# Experiment 8: where in depth is merging harm created?

Confirmatory test of "merging harm is created mid-depth" on full fine-tuned CLIP ViT-B/32 (8 tasks),
under task arithmetic, Iso-CTS and TSV-M. Code: `code/` (exp8 package, tests, guide, run script).
Results: `results/` (depth_summary.md, who_summary.md, per-operator JSONs, selection files), logs in
`results/logs/`. Per-image `evals/` caches and `heads.pt` are not included.

## Confirmatory result (fixed sequence, victims as units, n = 8, 38 pairs)

| Claim | Result | Verdict |
|---|---|---|
| 1 Localization | phi_M-phi_Lt +0.30 (p=0.004), phi_M-phi_E +0.02 (p=0.26) | NOT CONFIRMED, sequence stops |
| 2 Specificity | e_M-e_Lt +0.18, e_M-e_E -0.05 | exploratory |
| 3 Information loss | probe gap grows most in late blocks (negative contrasts) | exploratory |
| 4 Amplification | kappa_M +0.47 (p=0.004) | exploratory |
| 5 Repair separation | +0.28 (p=0.004) | exploratory |

Depth shares sigma E/M/Lt: TA 0.54/0.36/0.10, Iso-CTS 0.58 mid (3 victims), TSV-M 0.31/0.45/0.24.
Block-level removal peaks at blocks 3-6 (across the early/middle boundary); late blocks matter least.
Who family (Holm): interactions exist (p=0.008, 46% of harm) and harm is task content, not size (p=0.004).
Repair: removing harmers in the early third recovers 63% of victim harm at 13% harmer cost.

Must-hold: 7/7 tests passed; ViT-B/32 asset check PASS (all within 1 pp of FusionBench); all
confirmatory steps completed. ViT-L/14 replication not run: its check failed on EuroSAT zero-shot
(61.26 vs 58.90) and TSV-M single-task reconstruction (1.02e-3 vs 1e-3 limit); it is exploratory.
