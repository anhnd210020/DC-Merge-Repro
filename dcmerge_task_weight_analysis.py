#!/usr/bin/env python3
"""Analyse how DC-Merge allocates effective contribution across tasks.

This script mirrors the released ``dc_merge`` implementation for the
ViT-B/32, rank-16, eight-task LoRA experiment:
  linear singular-value smoothing (rho=5), cover-space construction,
  top-0.1% filtering, TIES sign consensus, and the structural block mask.

It needs only the eight saved LoRA adapters and final_handoff.json.  It does
not load CLIP, datasets, or classification heads, and it does not evaluate
any model.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, rankdata, spearmanr

from dcmerge_effective_weights import compute_effective_weights


TASKS = [
    "stanford_cars", "dtd", "eurosat", "gtsrb",
    "mnist", "resisc45", "sun397", "svhn",
]
DISPLAY = {
    "stanford_cars": "Cars", "dtd": "DTD", "eurosat": "EuroSAT",
    "gtsrb": "GTSRB", "mnist": "MNIST", "resisc45": "RESISC45",
    "sun397": "SUN397", "svhn": "SVHN",
}
RANK = 16
RHO = 5.0
TOP_PERCENT = 1e-3


def correlation_stats(x: np.ndarray, y: np.ndarray, label: str) -> dict:
    """Correlation with an exact two-sided permutation p-value (n=8)."""
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_rho, spearman_p = spearmanr(x, y)
    x_rank = rankdata(x).astype(float)
    y_rank = rankdata(y).astype(float)
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denominator = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    observed = float(np.dot(x_centered, y_centered) / denominator)
    extreme = 0
    total = 0
    for permutation in itertools.permutations(y_centered):
        candidate = float(np.dot(x_centered, permutation) / denominator)
        extreme += abs(candidate) >= abs(observed) - 1e-12
        total += 1
    exact_p = extreme / total

    rng = np.random.default_rng(20260922)
    boot = []
    for _ in range(5000):
        idx = rng.integers(0, len(x), len(x))
        if np.unique(x[idx]).size > 1 and np.unique(y[idx]).size > 1:
            boot.append(float(spearmanr(x[idx], y[idx]).statistic))
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "relationship": label,
        "n_tasks": int(len(x)),
        "pearson_r": float(pearson_r),
        "pearson_p_asymptotic": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p_asymptotic": float(spearman_p),
        "spearman_p_exact_permutation": float(exact_p),
        "spearman_bootstrap_95ci": [float(low), float(high)],
    }


def annotate_scatter(ax, x, y, labels, title, x_label, stats):
    ax.scatter(x, y, color="#2d6a9f", s=62, zorder=3)
    if len(np.unique(x)) > 1:
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(min(x), max(x), 100)
        ax.plot(grid, slope * grid + intercept, color="#d95f02", linewidth=1.7)
    for x_value, y_value, name in zip(x, y, labels):
        ax.annotate(name, (x_value, y_value), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Effective block weight (%)")
    ax.grid(alpha=0.22)
    ax.text(
        0.03, 0.97,
        f"Spearman rho = {stats['spearman_rho']:.3f}\n"
        f"exact permutation p = {stats['spearman_p_exact_permutation']:.3f}",
        transform=ax.transAxes, va="top", fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#999999"},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root", type=Path,
        default=Path("artifacts/selftrained_b32_r16_8task"),
        help="Downloaded Hugging Face selftrained_b32_r16_8task folder.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("analysis_outputs/dcmerge_task_weights"),
    )
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    checkpoints = root / "checkpoints"
    handoff_path = root / "results" / "final_handoff.json"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not checkpoints.is_dir() or not handoff_path.is_file():
        raise FileNotFoundError("Checkpoint folder or results/final_handoff.json was not found.")

    diagnostic = compute_effective_weights(
        {task: checkpoints / task for task in TASKS},
        TASKS,
        rank=RANK,
        rho=RHO,
        top_percent=TOP_PERCENT,
        progress=True,
    )
    raw_weight = {
        task: diagnostic["raw_energy_weight_percent"][task] / 100.0
        for task in TASKS
    }
    smoothed_weight = {
        task: diagnostic["smoothed_energy_weight_percent"][task] / 100.0
        for task in TASKS
    }
    block_weight = {
        task: diagnostic["effective_block_weight_percent"][task] / 100.0
        for task in TASKS
    }
    source_credit = {
        task: diagnostic["ties_source_credit_percent"][task] / 100.0
        for task in TASKS
    }
    diagnostics = diagnostic["module_diagnostics"]

    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    rows = []
    for task in TASKS:
        individual = float(handoff["single_task_test"][task])
        merged = float(handoff["dcmerge_per_task"][task]["accuracy_percent"])
        retention = merged / individual * 100.0
        rows.append({
            "task": task,
            "display_task": DISPLAY[task],
            "raw_energy_weight_percent": 100 * raw_weight[task],
            "smoothed_energy_weight_percent": 100 * smoothed_weight[task],
            "effective_block_weight_percent": 100 * block_weight[task],
            "ties_source_credit_percent": 100 * source_credit[task],
            "individual_test_accuracy_percent": individual,
            "merged_test_accuracy_percent": merged,
            "merged_retention_percent": retention,
            "fine_tuning_difficulty_percent": 100 - individual,
            "merge_difficulty_percent": 100 - retention,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "task_weight_summary.csv", index=False, float_format="%.6f")

    layer_rows = []
    for layer, weights in sorted(
        diagnostic["layer_effective_weight_percent"].items(),
        key=lambda item: int(item[0]),
    ):
        for task in TASKS:
            layer_rows.append({"layer": int(layer), "task": task, "weight_percent": weights[task]})
    layer_frame = pd.DataFrame(layer_rows)
    layer_frame.to_csv(output / "task_layer_weight.csv", index=False, float_format="%.6f")
    pd.DataFrame(diagnostics).to_csv(output / "module_diagnostics.csv", index=False, float_format="%.6f")

    # Plot 1: raw, smoothed, and effective allocation.
    sns.set_theme(style="whitegrid", context="notebook")
    melted = summary.melt(
        id_vars=["display_task"],
        value_vars=["raw_energy_weight_percent", "smoothed_energy_weight_percent", "effective_block_weight_percent"],
        var_name="stage", value_name="weight_percent",
    )
    labels = {
        "raw_energy_weight_percent": "Raw LoRA energy",
        "smoothed_energy_weight_percent": "After smoothing",
        "effective_block_weight_percent": "Effective DC-Merge block",
    }
    melted["stage"] = melted["stage"].map(labels)
    fig, ax = plt.subplots(figsize=(12.3, 5.4))
    sns.barplot(data=melted, x="display_task", y="weight_percent", hue="stage", ax=ax, palette=["#9ecae1", "#6baed6", "#2171b5"])
    ax.axhline(100 / len(TASKS), color="#d95f02", linestyle="--", linewidth=1.4, label="Equal 12.5%")
    ax.set(title="Task-weight allocation across DC-Merge stages", xlabel="Task", ylabel="Energy / effective allocation (%)")
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    fig.savefig(output / "task_weight_distribution.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "task_weight_distribution.pdf", bbox_inches="tight")
    plt.close(fig)

    # Plot 2: layer-level allocation.
    heatmap = layer_frame.pivot(index="task", columns="layer", values="weight_percent").reindex(TASKS)
    heatmap.index = [DISPLAY[t] for t in heatmap.index]
    fig, ax = plt.subplots(figsize=(12.2, 5.6))
    sns.heatmap(heatmap, cmap="YlGnBu", annot=True, fmt=".1f", linewidths=.35, cbar_kws={"label": "Within-layer task weight (%)"}, ax=ax)
    ax.set(title="Effective task-weight allocation by Transformer layer", xlabel="Transformer layer", ylabel="Task")
    plt.tight_layout()
    fig.savefig(output / "task_layer_weight_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "task_layer_weight_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    x_fine = summary["fine_tuning_difficulty_percent"].to_numpy()
    x_merge = summary["merge_difficulty_percent"].to_numpy()
    y_weight = summary["effective_block_weight_percent"].to_numpy()
    fine_stats = correlation_stats(x_fine, y_weight, "effective weight vs fine-tuning difficulty")
    merge_stats = correlation_stats(x_merge, y_weight, "effective weight vs merge difficulty")
    stats = {"fine_tuning_difficulty": fine_stats, "merge_difficulty": merge_stats}
    (output / "weight_difficulty_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # Plot 3: relationship with the two pre-registered difficulty proxies.
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.2), sharey=True)
    names = summary["display_task"].tolist()
    annotate_scatter(axes[0], x_fine, y_weight, names, "Fine-tuning difficulty", "Single-task error: 100 - individual accuracy (%)", fine_stats)
    annotate_scatter(axes[1], x_merge, y_weight, names, "Merge difficulty", "Retention loss: 100 - merged / individual (%)", merge_stats)
    fig.suptitle("Does effective DC-Merge weight relate to task difficulty?", y=1.02, fontsize=13, weight="bold")
    plt.tight_layout()
    fig.savefig(output / "weight_vs_task_difficulty.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "weight_vs_task_difficulty.pdf", bbox_inches="tight")
    plt.close(fig)

    report = [
        "# DC-Merge task-weight analysis",
        "",
        "This analysis exactly follows the released dc_merge mechanics for rank-16 LoRA: linear smoothing with rho=5, cover-space whitening, top-0.1% filtering, TIES sign consensus, and the structural block mask.",
        "",
        "## Interpretation",
        "",
        "- `effective_block_weight_percent` is the main answer to how the merged update is allocated across the eight task-indexed structural blocks.",
        "- `ties_source_credit_percent` is a complementary attribution: the share of the final masked cover-space update contributed by each task after TIES sign selection. It sums to 100% up to numerical precision.",
        "- Correlations use only eight tasks. They are descriptive, not causal evidence; the exact permutation p-value is the primary significance check.",
        "",
        "## Correlation results",
        "",
        f"- Fine-tuning difficulty: Spearman rho={fine_stats['spearman_rho']:.3f}, exact p={fine_stats['spearman_p_exact_permutation']:.3f}, bootstrap 95% CI={fine_stats['spearman_bootstrap_95ci']}.",
        f"- Merge difficulty: Spearman rho={merge_stats['spearman_rho']:.3f}, exact p={merge_stats['spearman_p_exact_permutation']:.3f}, bootstrap 95% CI={merge_stats['spearman_bootstrap_95ci']}.",
        "",
        (
            "For fine-tuning difficulty, this eight-task run provides evidence of a positive monotonic association with effective block weight. "
            "This is descriptive evidence only: it does not establish that DC-Merge deliberately prioritises harder tasks, because adapter geometry and task-vector energy may confound the association."
            if fine_stats["spearman_p_exact_permutation"] < 0.05
            else
            "For fine-tuning difficulty, this eight-task run does not provide statistically significant evidence of a monotonic association with effective block weight."
        ),
        (
            "For merge difficulty, the negative association is statistically significant by the exact permutation test but should be described as borderline: its bootstrap interval crosses zero under resampling of only eight tasks."
            if merge_stats["spearman_p_exact_permutation"] < 0.05
            else
            "For merge difficulty, this eight-task run does not provide statistically significant evidence of a monotonic association with effective block weight."
        ),
    ]
    (output / "analysis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\nAnalysis complete.")
    print(f"Output directory: {output}")
    print(summary[["task", "effective_block_weight_percent", "ties_source_credit_percent"]].to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
