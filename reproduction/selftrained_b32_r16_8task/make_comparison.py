#!/usr/bin/env python3
"""Create the requested comparison report from measured output files."""

import json
import re
from pathlib import Path

from pipeline_common import REPO, ROOT, TASKS, utc_now


TABLE1 = {
    "individual_average_accuracy_percent": 87.82,
    "dcmerge_absolute_average_accuracy_percent": 64.17,
    "dcmerge_normalized_average_accuracy_percent": 73.90,
}
BASELINE_LOG = Path("/workspace/dcmerge_b32_8task.log")


def parse_official_baseline():
    text = BASELINE_LOG.read_text(errors="replace").replace("\r", "\n")
    marker = "Evaluating Merged Model on test set"
    if marker not in text:
        raise RuntimeError("official-checkpoint baseline log has no final test section")
    final = text.rsplit(marker, 1)[1]
    absolute = re.search(r"Average Accuracy is ([0-9.]+)", final)
    normalized = re.search(r"Average Normalized Accuracy is ([0-9.]+)", final)
    alpha_matches = re.findall(r"Best Alpha: ([0-9.eE+-]+)", text)
    total_matches = re.findall(r"TOTAL_SECONDS=([0-9.]+)", text)
    if not absolute or not normalized:
        raise RuntimeError("could not parse final baseline averages")
    return {
        "absolute_average_accuracy_percent": float(absolute.group(1)),
        "normalized_average_accuracy_percent": float(normalized.group(1)),
        "selected_alpha": float(alpha_matches[-1]) if alpha_matches else None,
        "elapsed_seconds": float(total_matches[-1]) if total_matches else None,
        "log": str(BASELINE_LOG),
    }


def main():
    authors_path = (
        REPO
        / "vision_lora_merge"
        / "single_task"
        / "our_finetuned"
        / "ViT-B-32"
        / "test_acc.json"
    )
    authors = json.loads(authors_path.read_text())
    ours = json.loads((ROOT / "results" / "selftrained_test_acc.json").read_text())
    single_summary = json.loads(
        (ROOT / "results" / "selftrained_single_task_summary.json").read_text()
    )
    dcmerge = json.loads(
        (ROOT / "results" / "selftrained_dcmerge_results.json").read_text()
    )
    baseline = parse_official_baseline()
    training_seconds = 0.0
    for task in TASKS:
        metadata = json.loads((ROOT / "state" / f"{task}.metadata.json").read_text())
        if metadata.get("status") != "DONE":
            raise RuntimeError(f"task metadata is not DONE: {task}")
        training_seconds += float(metadata["elapsed_seconds"])

    rows = []
    for task in TASKS:
        rows.append(
            f"| {task} | {authors[task]:.3f} | {ours[task]:.3f} | "
            f"{ours[task] - authors[task]:+.3f} |"
        )
    ours_average = sum(ours[task] for task in TASKS) / len(TASKS)
    test = dcmerge["test"]
    lines = [
        "# DC-Merge ViT-B/32 rank-16 eight-task reproduction comparison",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Single-task test accuracy",
        "",
        "| Task | Authors (%) | Self-trained (%) | Difference (pp) |",
        "|---|---:|---:|---:|",
        *rows,
        "",
        "## Aggregate comparison",
        "",
        "| Metric | Table 1 (%) | Official-checkpoint reproduction (%) | Self-trained (%) |",
        "|---|---:|---:|---:|",
        f"| Individual average | {TABLE1['individual_average_accuracy_percent']:.2f} | n/a | {ours_average:.3f} |",
        f"| DC-Merge absolute average | {TABLE1['dcmerge_absolute_average_accuracy_percent']:.2f} | {baseline['absolute_average_accuracy_percent']:.3f} | {test['average_accuracy_percent']:.3f} |",
        f"| DC-Merge normalized average | {TABLE1['dcmerge_normalized_average_accuracy_percent']:.2f} | {baseline['normalized_average_accuracy_percent']:.3f} | {test['average_normalized_accuracy_percent']:.3f} |",
        "",
        "## Run details",
        "",
        f"- Self-trained selected alpha: {dcmerge['selected_alpha']}",
        f"- Official-checkpoint reproduction selected alpha: {baseline['selected_alpha']}",
        f"- Total adapter training and per-task verification time: {training_seconds:.3f} seconds",
        f"- Self-trained DC-Merge evaluation time: {dcmerge['elapsed_seconds']:.3f} seconds",
        f"- Official-checkpoint reproduction time: {baseline['elapsed_seconds']} seconds",
        f"- Self-trained single-task evaluation time: {single_summary['elapsed_seconds']:.3f} seconds",
        "",
        "Authors' single-task values are read only as reference from the repository's "
        "`single_task/our_finetuned/ViT-B-32/test_acc.json`. All self-trained values "
        "come from independently initialized adapters under this experiment root.",
        "",
    ]
    (ROOT / "results" / "reproduction_comparison.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
