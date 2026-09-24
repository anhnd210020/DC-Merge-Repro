import json
import importlib
import sys
import unittest
from collections import OrderedDict
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
VISION = REPO / "vision_lora_merge"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(VISION))

from merging_functions import (  # noqa: E402
    _dc_merge_matrix_group,
    dc_merge,
    generate_linear_distribution,
    keep_topk_percent,
    ties_small,
)
from dcmerge_effective_weights import (  # noqa: E402
    adapter_prefixes,
    effective_weights_from_projection_cache,
    load_state,
    prepare_projection_cache,
)
from task_scale_ablation_common import (  # noqa: E402
    TASKS,
    load_scale_configs,
    select_best_alpha,
    summarize_split,
)


def legacy_dc_merge(deltas_dict, smoothing_strategy="avg", rho=5.0):
    """Verbatim mathematical path from commit 1a9817e, kept only as an oracle."""
    dc_dict = {}
    for key, vecs in deltas_dict.items():
        num_tasks = len(vecs)
        rank = 16
        smoothed_vecs = []
        for index in range(num_tasks):
            u, singular, vh = torch.linalg.svd(vecs[index], full_matrices=False)
            if index == 0:
                sum_u = torch.zeros_like(u)
                sum_v = torch.zeros_like(vh)
            sum_u[:, index * rank : (index + 1) * rank] = u[:, :rank]
            sum_v[index * rank : (index + 1) * rank, :] = vh[:rank, :]
            original = singular[:rank].clone()
            if smoothing_strategy == "linear":
                ratio = min(rho, original[0] / original[-1])
                distribution = generate_linear_distribution(rank, ratio)
            elif smoothing_strategy == "avg":
                distribution = torch.ones_like(original) / len(original)
            else:
                raise ValueError
            smoothed = original.sum() * distribution
            smoothed_vecs.append(
                u[:, :rank] @ torch.diag(smoothed) @ vh[:rank, :]
            )
        uu, _, vhu = torch.linalg.svd(sum_u, full_matrices=False)
        uv, _, vhv = torch.linalg.svd(sum_v, full_matrices=False)
        cover_u = (uu @ vhu)[:, : num_tasks * rank]
        cover_vt = (uv @ vhv)[: num_tasks * rank, :]
        projected = [
            torch.linalg.multi_dot(
                (cover_u.T, smoothed_vecs[index], cover_vt.T)
            )
            for index in range(num_tasks)
        ]
        aggregate = ties_small(keep_topk_percent(projected, 1e-3))
        mask = torch.zeros_like(aggregate)
        block = mask.shape[0] // num_tasks
        for index in range(num_tasks):
            mask[
                index * block : (index + 1) * block,
                index * block : (index + 1) * block,
            ] = 1
        dc_dict[key] = torch.linalg.multi_dot((cover_u, aggregate * mask, cover_vt))
    return OrderedDict(sorted(dc_dict.items()))


class TaskScaleCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator = torch.Generator().manual_seed(20260924)
        cls.matrices = [torch.randn(48, 48, generator=generator) for _ in range(3)]

    def test_default_and_explicit_ones_match_legacy(self):
        grouped = OrderedDict([("module", self.matrices)])
        legacy = legacy_dc_merge(grouped, smoothing_strategy="linear", rho=5.0)
        default = dc_merge(grouped, smoothing_strategy="linear", rho=5.0)
        explicit = dc_merge(
            grouped,
            smoothing_strategy="linear",
            rho=5.0,
            task_scales=[1.0, 1.0, 1.0],
        )
        self.assertTrue(torch.equal(legacy["module"], default["module"]))
        self.assertTrue(torch.equal(legacy["module"], explicit["module"]))

    def test_intervention_changes_amplitude_not_directions_or_cover_basis(self):
        _, baseline = _dc_merge_matrix_group(
            self.matrices,
            smoothing_strategy="linear",
            rho=5.0,
            task_scales=[1.0, 1.0, 1.0],
            return_intermediates=True,
        )
        _, scaled = _dc_merge_matrix_group(
            self.matrices,
            smoothing_strategy="linear",
            rho=5.0,
            task_scales=[1.0, 2.0, 0.5],
            return_intermediates=True,
        )
        for key in ("singular_directions_u", "singular_directions_v"):
            for before, after in zip(baseline[key], scaled[key]):
                self.assertTrue(torch.equal(before, after))
        for before, after in zip(
            baseline["normalized_smoothed_distributions"],
            scaled["normalized_smoothed_distributions"],
        ):
            self.assertTrue(torch.equal(before, after))
        self.assertTrue(torch.equal(baseline["cover_space_u"], scaled["cover_space_u"]))
        self.assertTrue(torch.equal(baseline["cover_space_vT"], scaled["cover_space_vT"]))
        expected = (1.0, 2.0, 0.5)
        for index, factor in enumerate(expected):
            torch.testing.assert_close(
                scaled["smoothed_singular_values"][index],
                baseline["smoothed_singular_values"][index] * factor,
                rtol=0.0,
                atol=0.0,
            )
            self.assertTrue(
                torch.equal(
                    torch.argsort(
                        baseline["smoothed_singular_values"][index], descending=True
                    ),
                    torch.argsort(
                        scaled["smoothed_singular_values"][index], descending=True
                    ),
                )
            )

    def test_invalid_scales_rejected(self):
        grouped = OrderedDict([("module", self.matrices)])
        for scales in ([1.0, 1.0], [1.0, 0.0, 1.0], [1.0, float("nan"), 1.0]):
            with self.assertRaises(ValueError):
                dc_merge(grouped, task_scales=scales)

    @unittest.skipUnless(
        (REPO / "artifacts" / "selftrained_b32_r16_8task" / "checkpoints").is_dir(),
        "local downloaded adapters are not present",
    )
    def test_real_adapter_module_all_ones_matches_legacy_exactly(self):
        root = REPO / "artifacts" / "selftrained_b32_r16_8task" / "checkpoints"
        states = {task: load_state(root / task / "adapter_model.bin") for task in TASKS}
        prefix = adapter_prefixes(states[TASKS[0]])[0]
        matrices = [
            states[task][prefix + ".lora_B.weight"]
            @ states[task][prefix + ".lora_A.weight"]
            for task in TASKS
        ]
        grouped = OrderedDict([(prefix, matrices)])
        legacy = legacy_dc_merge(grouped, smoothing_strategy="linear", rho=5.0)
        explicit = dc_merge(
            grouped,
            smoothing_strategy="linear",
            rho=5.0,
            task_scales=[1.0] * len(TASKS),
        )
        self.assertTrue(torch.equal(legacy[prefix], explicit[prefix]))

    @unittest.skipUnless(
        (REPO / "artifacts" / "selftrained_b32_r16_8task" / "checkpoints").is_dir(),
        "local downloaded adapters are not present",
    )
    def test_all_real_modules_preserve_positive_scaled_topk_support(self):
        root = REPO / "artifacts" / "selftrained_b32_r16_8task" / "checkpoints"
        cache = prepare_projection_cache(
            {task: root / task for task in TASKS}, TASKS, progress=False
        )
        positive_scales = (0.37, 0.53, 0.79, 1.1, 1.7, 2.3, 3.1, 4.2)
        for module in cache["modules"]:
            for task_entry, scale in zip(module["tasks"], positive_scales):
                scaled_projected = (
                    task_entry["left"]
                    @ torch.diag(task_entry["smoothed_singular"] * scale)
                    @ task_entry["right"]
                )
                torch.testing.assert_close(
                    scaled_projected,
                    task_entry["projected"] * scale,
                    rtol=5e-12,
                    atol=1e-12,
                    msg=f"projection linearity failed for {module['module']}",
                )
                scaled_filtered = keep_topk_percent([scaled_projected], 1e-3)[0]
                flat = scaled_projected.abs().reshape(-1)
                k = max(1, int(1e-3 * flat.numel()))
                scaled_threshold = torch.topk(flat, k).values.min()
                scaled_support = scaled_projected.abs() >= scaled_threshold
                self.assertTrue(
                    torch.equal(task_entry["topk_support"], scaled_support),
                    msg=f"top-k support changed for {module['module']}",
                )
                torch.testing.assert_close(
                    scaled_filtered,
                    task_entry["filtered"] * scale,
                    rtol=5e-12,
                    atol=1e-12,
                    msg=f"retained values failed to scale for {module['module']}",
                )
        primary = load_scale_configs(
            REPO / "task_scale_configs_primary_calibrated.json"
        )
        for entry in primary["configs"]:
            realized = effective_weights_from_projection_cache(
                cache, entry["task_scales"]
            )
            expected = entry["metadata"]["achieved_effective_weight_percent"]
            for task in ("eurosat", "sun397"):
                self.assertAlmostEqual(realized[task], expected[task], places=10)


class ConfigurationTests(unittest.TestCase):
    def test_model_checkpoint_head_dataset_and_scale_order_mapping(self):
        module = importlib.import_module(
            "configs.vitB32_r16_8task_selftrained"
        )
        config = module.config
        names = [item["name"] for item in config["dataset"]]
        self.assertEqual(names, list(TASKS))
        self.assertEqual(
            [Path(path).name for path in config["model"]["bases"]], list(TASKS)
        )
        self.assertEqual(
            [Path(item["clip_encodings"]).stem for item in config["dataset"]],
            [f"{task}_head" for task in TASKS],
        )

    def test_preregistered_configurations_are_exact_and_full(self):
        loaded = load_scale_configs(REPO / "task_scale_configs.json")
        self.assertEqual(loaded["series_role"], "secondary_heuristic_stress_test")
        expected_pairs = {
            "baseline": (1.0, 1.0),
            "partial_balance": (2.0, 0.8),
            "full_balance_initial": (3.07, 0.6),
            "strong_reallocation": (4.0, 0.42),
            "strong_reversal": (4.5, 0.32),
        }
        self.assertEqual([item["run_id"] for item in loaded["configs"]], list(expected_pairs))
        for item in loaded["configs"]:
            self.assertEqual(set(item["task_scales"]), set(TASKS))
            pair = expected_pairs[item["run_id"]]
            self.assertEqual(item["task_scales"]["eurosat"], pair[0])
            self.assertEqual(item["task_scales"]["sun397"], pair[1])
            for task in set(TASKS) - {"eurosat", "sun397"}:
                self.assertEqual(item["task_scales"][task], 1.0)

    def test_primary_calibrated_series_is_frozen_complete_and_pair_controlled(self):
        loaded = load_scale_configs(
            REPO / "task_scale_configs_primary_calibrated.json"
        )
        self.assertEqual(
            loaded["series_role"], "primary_controlled_redistribution"
        )
        self.assertTrue(
            loaded["series_metadata"]["frozen_before_accuracy_evaluation"]
        )
        self.assertFalse(
            loaded["series_metadata"]["calibration"]["uses_accuracy"]
        )
        expected_ids = [
            "baseline",
            "controlled_moderate",
            "controlled_balanced",
            "controlled_reversal",
        ]
        self.assertEqual(
            [entry["run_id"] for entry in loaded["configs"]], expected_ids
        )
        baseline_pair = loaded["series_metadata"]["calibration"][
            "baseline_pair_total_percent"
        ]
        for entry in loaded["configs"]:
            metadata = entry["metadata"]
            achieved = metadata["achieved_effective_weight_percent"]
            self.assertAlmostEqual(
                achieved["eurosat"] + achieved["sun397"], baseline_pair, places=11
            )
            self.assertLess(abs(metadata["pair_total_error_percent"]), 1e-10)
            self.assertLess(metadata["target_rmse_percent"], 1e-10)
            for task in set(TASKS) - {"eurosat", "sun397"}:
                self.assertEqual(entry["task_scales"][task], 1.0)

    def test_alpha_selection_is_strict_and_uses_normalized_mean(self):
        results = [
            {"alpha": 0.1, "average_normalized_accuracy_percent": 70.0},
            {"alpha": 0.2, "average_normalized_accuracy_percent": 71.0},
            {"alpha": 0.3, "average_normalized_accuracy_percent": 71.0},
        ]
        self.assertEqual(select_best_alpha(results)["alpha"], 0.2)

    def test_metric_definitions(self):
        single = {task: 80.0 for task in TASKS}
        merged = {task: 60.0 for task in TASKS}
        summary = summarize_split(merged, single)
        self.assertEqual(summary["average_accuracy_percent"], 60.0)
        self.assertEqual(summary["average_retention_percent"], 75.0)
        self.assertTrue(all(value == 20.0 for value in summary["absolute_drop_pp"].values()))

    def test_runner_contains_no_optimizer_or_training_entrypoint(self):
        source = (REPO / "run_task_scale_ablation.py").read_text(encoding="utf-8")
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("train_task", source)
        calibration = (REPO / "calibrate_primary_task_scales.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("torch.optim", calibration)
        self.assertNotIn("train_task", calibration)
        self.assertNotIn("selftrained_val_acc", calibration)
        self.assertNotIn("selftrained_test_acc", calibration)
        self.assertNotIn("evaluate_cliphead", calibration)


if __name__ == "__main__":
    unittest.main()
