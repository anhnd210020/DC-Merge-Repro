import builtins
import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path, PureWindowsPath
from unittest import mock

import torch

from state_dependent_task_drift.anchor import (
    anchor_definition,
    anchor_definition_hash,
    assert_optimizer_matches,
    configure_fresh_lora_trainability,
    materialize_anchor,
    validate_anchor_hash,
)
from state_dependent_task_drift.cli import main
from state_dependent_task_drift.config import PILOT_TASKS, load_config
from state_dependent_task_drift.aggregation import build_pair_level_join
from state_dependent_task_drift.analysis import (
    _conditional_adapter,
    _seed_drift_summaries,
    _symmetric_state_drift,
)
from state_dependent_task_drift.lora_delta import (
    adapter_checksum,
    adapter_report,
    effective_deltas,
    load_adapter,
    lora_scaling,
    paired_factors,
    validate_saved_adapter,
)
from state_dependent_task_drift.metrics import (
    aggregate_layer_rows,
    compare_adapter_subspaces,
    numerical_tolerance,
    safe_metric_aggregate,
)
from state_dependent_task_drift.pairwise_merge import (
    capture_clean_target_weights,
    materialize_from_clean,
    pairwise_run_definition,
    pairwise_run_definition_hash,
    validate_complete_pairwise_run,
)
from state_dependent_task_drift.run_validation import (
    effective_training_recipe,
    expected_anchor_definition,
    expected_run_definition,
    run_definition_hash,
    validate_complete_run,
    validate_reuse_adapter,
)
from state_dependent_task_drift.run_plan import (
    baseline_run_dir,
    baseline_run_id,
    conditional_run_dir,
    conditional_run_id,
    ordered_task_pairs,
    run_pilot_baseline_dependencies,
    seed_pairs,
    unordered_task_pairs,
)
from state_dependent_task_drift.state_io import atomic_write_json, read_json, run_action, set_status
from state_dependent_task_drift.training import (
    apply_reproduction_scheduler_step,
    git_dirty,
    grad_scaler_enabled,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "state_dependent_task_drift" / "pilot_config.json"


def adapter_state(a, b):
    return OrderedDict((
        ("module.lora_A.weight", a.clone()),
        ("module.lora_B.weight", b.clone()),
    ))


def tiny_adapter_state(value=1.0, rank=2):
    a = torch.arange(1, rank * 4 + 1, dtype=torch.float32).reshape(rank, 4) * value
    b = torch.arange(1, 4 * rank + 1, dtype=torch.float32).reshape(4, rank) * value
    prefix = "base_model.model.encoder.layers.0.self_attn.q_proj"
    return OrderedDict(((prefix + ".lora_A.weight", a), (prefix + ".lora_B.weight", b)))


def write_tiny_adapter(directory, value=1.0, rank=2, alpha=6.0):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = {"r": rank, "lora_alpha": alpha, "target_modules": ["q_proj"]}
    (directory / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    torch.save(tiny_adapter_state(value, rank), directory / "adapter_model.bin")
    return config, tiny_adapter_state(value, rank)


def tiny_cfg(root):
    cfg = load_config(CONFIG)
    cfg = dict(cfg)
    cfg.update({
        "output_root": Path(root) / "outputs",
        "baseline_adapter_root": Path(root) / "reuse",
        "lora_rank": 2,
        "lora_alpha": 6.0,
        "target_modules": ("q_proj",),
        "precision": "fp32",
    })
    return cfg


def write_complete_run(cfg, run_dir, target, seed, anchor=None, anchor_adapter=None):
    run_dir = Path(run_dir)
    _config, selected = write_tiny_adapter(run_dir / "adapter", value=1.0)
    report = adapter_report(run_dir / "adapter")
    report["disk_reload_succeeded"] = True
    report["fresh_peft_reload"] = {"succeeded": True}
    if anchor:
        report["conditional_anchor_relative_round_trip"] = {"succeeded": True}
    definition = expected_run_definition(
        cfg, target_task=target, target_seed=seed,
        anchor_task=anchor, anchor_adapter=anchor_adapter,
    )
    metadata = {
        "status": "COMPLETE", "completion_status": "COMPLETE",
        "target_task": target, "target_training_seed": seed,
        "anchor_task": anchor,
        "anchor_seed": cfg["anchor_seed"] if anchor else None,
        "anchor_alpha": cfg["anchor_alpha"] if anchor else None,
        "checkpoint_validation": report,
        "run_definition": definition,
        "run_definition_hash": run_definition_hash(definition),
    }
    if anchor:
        anchor_identity = expected_anchor_definition(cfg, anchor, Path(anchor_adapter))
        metadata.update({
            "anchor_adapter_path": str(anchor_adapter),
            "anchor_adapter_sha256": adapter_checksum(Path(anchor_adapter)),
            "anchor_definition": anchor_identity,
            "anchor_definition_hash": definition["anchor_definition_hash"],
        })
    atomic_write_json(run_dir / "metadata.json", metadata)
    set_status(run_dir, "COMPLETE", checkpoint_sha256=report["sha256"])
    return metadata, selected


def write_complete_pairwise(run_dir, definition):
    run_dir = Path(run_dir)
    definition_hash = pairwise_run_definition_hash(definition)
    task_a, task_b = definition["task_a"], definition["task_b"]
    sweep = [{"alpha": alpha} for alpha in definition["merge_configuration"]["alpha_candidates"]]
    selected_alpha = sweep[0]["alpha"]
    common = {
        "pairwise_run_definition": definition,
        "pairwise_run_definition_hash": definition_hash,
    }
    frozen = {
        **common,
        "selection_tasks": [task_a, task_b],
        "selection_split": "validation",
        "selection_objective": definition["evaluation_configuration"]["selection_objective"],
        "selected_alpha": selected_alpha,
        "task_scales": None,
        "validation_sweep": sweep,
    }
    result = {
        **common,
        "task_a": task_a,
        "task_b": task_b,
        "canonical_seed": definition["canonical_seed"],
        "dc_merge_smoothing": "linear",
        "dc_merge_rho": 5.0,
        "selected_alpha": selected_alpha,
        "task_scales": None,
        "validation_sweep": sweep,
        "alpha_evaluation_base_state": definition["merge_configuration"]["alpha_evaluation_base_state"],
        "test_read_only_after_selection_frozen": True,
    }
    atomic_write_json(run_dir / "frozen_selection.json", frozen)
    atomic_write_json(run_dir / "result.json", result)
    set_status(
        run_dir, "COMPLETE", selected_alpha=selected_alpha,
        pairwise_run_definition=definition,
        pairwise_run_definition_hash=definition_hash,
    )
    return result


class PairAndConfigTests(unittest.TestCase):
    def test_pair_enumeration(self):
        ordered = ordered_task_pairs(PILOT_TASKS)
        unordered = unordered_task_pairs(PILOT_TASKS)
        self.assertEqual(len(ordered), 12)
        self.assertEqual(len(set(ordered)), 12)
        self.assertTrue(all(a != b for a, b in ordered))
        self.assertEqual(len(unordered), 6)
        self.assertEqual(seed_pairs((420, 421, 422)), ((420, 421), (420, 422), (421, 422)))

    def test_deterministic_ids_are_platform_independent(self):
        self.assertEqual(baseline_run_id("svhn", 420), baseline_run_id("svhn", 420))
        run_id = conditional_run_id("svhn", "dtd", 0.5, 421, 420)
        self.assertNotIn("\\", run_id)
        self.assertNotIn("/", run_id)

    def test_windows_paths_parse_as_paths_without_polluting_ids(self):
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        raw["repository_root"] = str(REPO)
        raw["pretrained_model"] = r"C:\models\clip"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            cfg = load_config(path)
        self.assertEqual(PureWindowsPath(str(cfg["pretrained_model"])).name, "clip")


class LoRAAndAnchorTests(unittest.TestCase):
    def test_explicit_scaling_and_factor_shapes(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        cfg = {"r": 2, "lora_alpha": 6}
        self.assertEqual(lora_scaling(cfg), 3.0)
        reconstructed = effective_deltas(cfg, adapter_state(a, b))["module"]
        torch.testing.assert_close(reconstructed, 3.0 * (b @ a))
        self.assertEqual(tuple(paired_factors(adapter_state(a, b))["module"][0].shape), (2, 2))

    def test_anchor_zero_half_one_and_linearity_catches_alpha_squared(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2, bias=False)

        base = Tiny()
        initial = base.linear.weight.detach().clone()
        delta = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
        source = delta.clone()
        outputs = {}
        for alpha in (0.0, 0.5, 1.0):
            model = Tiny()
            model.linear.weight.data.copy_(initial)
            materialize_anchor(model, {"base_model.model.linear": delta}, alpha)
            outputs[alpha] = model.linear.weight.detach().clone()
        torch.testing.assert_close(outputs[0.0], initial)
        torch.testing.assert_close(outputs[1.0], initial + delta)
        torch.testing.assert_close(outputs[0.5] - initial, 0.5 * (outputs[1.0] - initial))
        self.assertFalse(torch.allclose(outputs[0.5] - initial, 0.25 * delta))
        torch.testing.assert_close(delta, source)

    def test_only_fresh_lora_is_trainable_and_optimizer_matches(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = torch.nn.Linear(2, 2)
                self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 1, bias=False)})
                self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(1, 2, bias=False)})

        model = Tiny()
        trainable = configure_fresh_lora_trainability(model)
        names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertEqual(names, {"lora_A.default.weight", "lora_B.default.weight"})
        optimizer = torch.optim.AdamW(trainable, lr=1e-3)
        assert_optimizer_matches(optimizer, trainable)
        wrong = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with self.assertRaises(AssertionError):
            assert_optimizer_matches(wrong, trainable)
        self.assertFalse(model.base.weight.requires_grad)
        self.assertFalse(model.base.bias.requires_grad)

    def test_conditional_round_trip_identity(self):
        generator = torch.Generator().manual_seed(7)
        m0 = torch.randn(5, 5, generator=generator)
        delta_b = torch.randn(5, 5, generator=generator)
        delta_cond = torch.randn(5, 5, generator=generator)
        alpha = 0.5
        anchor = m0 + alpha * delta_b
        trained = anchor + delta_cond
        torch.testing.assert_close(trained, m0 + alpha * delta_b + delta_cond)
        torch.testing.assert_close(trained - anchor, delta_cond)


class SubspaceMetricTests(unittest.TestCase):
    cfg = {"r": 2, "lora_alpha": 4}

    def compare(self, a1, b1, a2, b2, rank=2):
        return compare_adapter_subspaces(self.cfg, adapter_state(a1, b1), self.cfg, adapter_state(a2, b2), rank)

    def test_invariance_to_sign_permutation_and_rotation(self):
        a = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        b = a.T
        transforms = [
            torch.diag(torch.tensor([-1.0, 1.0])),
            torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            torch.tensor([[0.0, -1.0], [1.0, 0.0]]),
        ]
        for transform in transforms:
            rows, summary = self.compare(a, b, transform @ a, b @ transform.T)
            self.assertAlmostEqual(summary["macro_mean_drift"], 0.0, places=6)
            self.assertFalse(rows[0]["degenerate"])

    def test_orthogonal_left_and_right_subspaces(self):
        left_a = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        left_b = left_a.T
        right_a = torch.tensor([[0, 0, 1.0, 0], [0, 0, 0, 1.0]])
        right_b = right_a.T
        rows, summary = self.compare(left_a, left_b, right_a, right_b)
        self.assertAlmostEqual(rows[0]["S_U"], 0.0, places=6)
        self.assertAlmostEqual(rows[0]["S_V"], 0.0, places=6)
        self.assertAlmostEqual(summary["macro_mean_drift"], 1.0, places=6)

    def test_numerical_rank_minimum_tolerance_and_degenerate(self):
        full_a = torch.tensor([[1.0, 0, 0], [0, 2.0, 0]])
        full_b = torch.tensor([[1.0, 0], [0, 1.0], [0, 0]])
        deficient_a = torch.tensor([[1.0, 0, 0], [0, 0, 0]])
        rows, _ = self.compare(full_a, full_b, deficient_a, full_b, rank=2)
        self.assertEqual(rows[0]["baseline_numerical_rank"], 2)
        self.assertEqual(rows[0]["conditional_numerical_rank"], 1)
        self.assertEqual(rows[0]["comparison_rank"], 1)
        rows_reverse, _ = self.compare(deficient_a, full_b, full_a, full_b, rank=2)
        self.assertEqual(rows_reverse[0]["baseline_numerical_rank"], 1)
        self.assertEqual(rows_reverse[0]["conditional_numerical_rank"], 2)
        self.assertEqual(rows_reverse[0]["comparison_rank"], 1)
        self.assertGreater(rows[0]["baseline_tolerance"], 0.0)
        self.assertGreater(rows[0]["conditional_tolerance"], 0.0)
        singular = torch.tensor([2.0, 1.0], dtype=torch.float32)
        self.assertEqual(numerical_tolerance(singular, 3, 3, singular.dtype), torch.finfo(torch.float32).eps * 3 * 2)
        zeros = torch.zeros_like(full_b)
        rows, summary = self.compare(full_a, zeros, full_a, zeros)
        self.assertTrue(rows[0]["degenerate"])
        self.assertEqual(rows[0]["comparison_rank"], 0)
        self.assertIsNone(rows[0]["D"])
        self.assertIsNone(summary["macro_mean_drift"])


class IdentityAndStateTests(unittest.TestCase):
    def definition(self, alpha=0.5, checksum="a" * 64):
        return anchor_definition(
            base_model_id="clip-local", adapter_sha256=checksum, anchor_task="svhn",
            anchor_seed=420, alpha=alpha, rank=16, lora_alpha=16, scaling=1.0,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )

    def test_anchor_hash_deterministic_and_sensitive(self):
        first = anchor_definition_hash(self.definition())
        self.assertEqual(first, anchor_definition_hash(self.definition()))
        self.assertNotEqual(first, anchor_definition_hash(self.definition(alpha=1.0)))
        self.assertNotEqual(first, anchor_definition_hash(self.definition(checksum="b" * 64)))
        validate_anchor_hash({"anchor_definition_hash": first}, self.definition())
        with self.assertRaises(RuntimeError):
            validate_anchor_hash({"anchor_definition_hash": first}, self.definition(alpha=1.0))

    def test_atomic_json_and_resume_state_machine(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            atomic_write_json(run / "value.json", {"x": 1})
            self.assertEqual(read_json(run / "value.json"), {"x": 1})
            self.assertEqual(run_action(run, resume=False), "run")
            set_status(run, "COMPLETE")
            self.assertEqual(run_action(run, resume=True), "skip")
            with self.assertRaises(FileExistsError):
                run_action(run, resume=False)
            set_status(run, "FAILED")
            self.assertEqual(run_action(run, resume=True), "retry")
            with self.assertRaises(FileExistsError):
                run_action(run, resume=False)


class SchedulerAndScalerTests(unittest.TestCase):
    def test_zero_based_post_update_scheduler_matches_original_lr_sequence(self):
        base_lr, warmup, total = 0.3, 3, 6
        indices, learning_rates = [], []

        def original_schedule(index):
            indices.append(index)
            if index < warmup:
                learning_rates.append(base_lr * (index + 1) / warmup)
            else:
                progress = (index - warmup) / (total - warmup)
                learning_rates.append(0.5 * (1 + math.cos(math.pi * progress)) * base_lr)

        completed = 0
        for _ in range(5):
            completed = apply_reproduction_scheduler_step(original_schedule, completed)

        self.assertEqual(indices, [0, 1, 2, 3, 4])
        self.assertEqual(completed, 5)
        expected = [0.1, 0.2, 0.3, 0.3, 0.225]
        for actual, wanted in zip(learning_rates, expected):
            self.assertAlmostEqual(actual, wanted)

    def test_fp32_recipe_retains_original_cuda_scaler_policy(self):
        cfg = load_config(CONFIG)
        recipe = effective_training_recipe(cfg)
        self.assertFalse(grad_scaler_enabled("cpu"))
        self.assertTrue(grad_scaler_enabled("cuda"))
        self.assertTrue(grad_scaler_enabled("cuda:0"))
        self.assertEqual(recipe["precision"], "fp32")
        self.assertEqual(
            recipe["gradient_scaler"],
            "enabled_on_cuda_matching_original_reproduction",
        )
        self.assertEqual(recipe["scheduler_index_origin"], 0)


class RepairRunValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.cfg = tiny_cfg(self.root)
        self.anchor_task = "eurosat"
        self.target_task = "stanford_cars"
        self.anchor_run = baseline_run_dir(
            self.cfg["output_root"], self.anchor_task, self.cfg["anchor_seed"]
        )
        write_complete_run(
            self.cfg, self.anchor_run, self.anchor_task, self.cfg["anchor_seed"]
        )
        self.anchor_adapter = self.anchor_run / "adapter"
        self.conditional_run = conditional_run_dir(
            self.cfg["output_root"], self.target_task, self.anchor_task,
            self.cfg["anchor_alpha"], 420,
        )
        write_complete_run(
            self.cfg, self.conditional_run, self.target_task, 420,
            self.anchor_task, self.anchor_adapter,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def validate_conditional(self, cfg=None, seed=420, anchor_adapter=None):
        return validate_complete_run(
            cfg or self.cfg, self.conditional_run,
            target_task=self.target_task, target_seed=seed,
            anchor_task=self.anchor_task,
            anchor_adapter=anchor_adapter or self.anchor_adapter,
        )

    def test_compute_drift_loader_rejects_failed_and_incomplete_status(self):
        for status in ("FAILED", "RUNNING"):
            set_status(self.conditional_run, status)
            with self.assertRaisesRegex(RuntimeError, "not COMPLETE"):
                _conditional_adapter(self.cfg, self.target_task, self.anchor_task, 420)
        report = adapter_report(self.conditional_run / "adapter")
        set_status(self.conditional_run, "COMPLETE", checkpoint_sha256=report["sha256"])

    def test_complete_checksum_mismatch_rejected_and_valid_accepted(self):
        metadata = read_json(self.conditional_run / "metadata.json")
        metadata["checkpoint_validation"]["sha256"] = "0" * 64
        atomic_write_json(self.conditional_run / "metadata.json", metadata)
        with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            self.validate_conditional()
        write_complete_run(
            self.cfg, self.conditional_run, self.target_task, 420,
            self.anchor_task, self.anchor_adapter,
        )
        validated = self.validate_conditional()
        self.assertEqual(validated["status"]["status"], "COMPLETE")

    def test_resume_identity_same_config_accepts_changes_reject(self):
        self.validate_conditional()
        changed_alpha = dict(self.cfg, anchor_alpha=0.75)
        with self.assertRaisesRegex(RuntimeError, "incompatible|anchor_definition"):
            self.validate_conditional(changed_alpha)
        with self.assertRaisesRegex(RuntimeError, "seed mismatch"):
            self.validate_conditional(seed=421)
        changed_rank = dict(self.cfg, lora_rank=3)
        with self.assertRaises(RuntimeError):
            self.validate_conditional(changed_rank)
        write_tiny_adapter(self.anchor_adapter, value=2.0)
        with self.assertRaisesRegex(RuntimeError, "incompatible|anchor.*mismatch"):
            self.validate_conditional()

    def test_wrong_anchor_hash_rejected(self):
        metadata = read_json(self.conditional_run / "metadata.json")
        metadata["anchor_definition_hash"] = "f" * 64
        atomic_write_json(self.conditional_run / "metadata.json", metadata)
        with self.assertRaisesRegex(RuntimeError, "anchor_definition_hash"):
            self.validate_conditional()


class RepairSerializationTests(unittest.TestCase):
    def test_saved_adapter_reload_retains_effective_delta(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "adapter"
            _config, selected = write_tiny_adapter(directory)
            validation = validate_saved_adapter(directory, selected, 2, 6.0, ["q_proj"])
            self.assertTrue(validation["disk_reload_succeeded"])
            self.assertEqual(validation["effective_delta_max_absolute_difference"], 0.0)

    def test_conditional_relative_delta_survives_serialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            anchor_dir, conditional_dir = Path(temporary) / "anchor", Path(temporary) / "conditional"
            anchor_cfg, _ = write_tiny_adapter(anchor_dir, value=0.5)
            conditional_cfg, selected = write_tiny_adapter(conditional_dir, value=1.5)
            validate_saved_adapter(conditional_dir, selected, 2, 6.0, ["q_proj"])
            loaded_cfg, loaded_state = load_adapter(conditional_dir)
            anchor_config, anchor_state = load_adapter(anchor_dir)
            expected = 0.5 * effective_deltas(anchor_cfg, anchor_state)[next(iter(effective_deltas(anchor_cfg, anchor_state)))]
            expected = expected + effective_deltas(conditional_cfg, selected)[next(iter(effective_deltas(conditional_cfg, selected)))]
            actual = 0.5 * effective_deltas(anchor_config, anchor_state)[next(iter(effective_deltas(anchor_config, anchor_state)))]
            actual = actual + effective_deltas(loaded_cfg, loaded_state)[next(iter(effective_deltas(loaded_cfg, loaded_state)))]
            torch.testing.assert_close(actual, expected)


class RepairAggregateTests(unittest.TestCase):
    def test_pilot_precision_default_is_fp32(self):
        self.assertEqual(load_config(CONFIG)["precision"], "fp32")

    def test_per_layer_aggregation_excludes_degenerate_module(self):
        base = {
            "category": "cross_state_matched_seed_drift", "target_task": "stanford_cars",
            "anchor_task": "eurosat", "target_seed": 420, "anchor_seed": 420,
            "anchor_alpha": 0.5, "seed_left": 420, "seed_right": 420,
            "baseline_numerical_rank": 2, "conditional_numerical_rank": 2,
            "comparison_rank": 2, "baseline_frobenius_norm": 2.0,
            "conditional_frobenius_norm": 2.0, "baseline_energy": 4.0,
            "conditional_energy": 4.0,
        }
        rows = [
            {**base, "module": "base_model.model.encoder.layers.3.self_attn.q_proj", "degenerate": False, "S_U": 0.8, "S_V": 0.6, "D": 0.3},
            {**base, "module": "base_model.model.encoder.layers.3.self_attn.k_proj", "degenerate": False, "S_U": 0.6, "S_V": 0.4, "D": 0.5},
            {**base, "module": "base_model.model.encoder.layers.3.self_attn.v_proj", "degenerate": True, "S_U": None, "S_V": None, "D": None},
        ]
        result = aggregate_layer_rows(rows)[0]
        self.assertEqual(result["valid_module_count"], 2)
        self.assertEqual(result["degenerate_module_count"], 1)
        self.assertAlmostEqual(result["macro_mean_D"], 0.4)
        self.assertAlmostEqual(result["macro_mean_S_U"], 0.7)

    def test_seed_drift_three_pairs_mean_and_sample_std(self):
        observations = [
            {"category": "baseline_same_anchor_seed_drift", "target_task": "svhn", "anchor_task": None,
             "seed_left": left, "seed_right": right, "macro_mean_drift": value}
            for (left, right), value in zip(seed_pairs((420, 421, 422)), (1.0, 2.0, 3.0))
        ]
        summary = _seed_drift_summaries(observations)[0]
        self.assertEqual(summary["count"], 3)
        self.assertEqual(len(summary["seed_pair_observations"]), 3)
        self.assertEqual(summary["mean_macro_drift"], 2.0)
        self.assertEqual(summary["sample_std_macro_drift"], 1.0)

    def test_symmetric_state_drift_formula(self):
        directed = [
            {"target_task": "stanford_cars", "anchor_task": "eurosat", "macro_drift_mean": 0.2, "macro_drift_sample_std": 0.01},
            {"target_task": "eurosat", "anchor_task": "stanford_cars", "macro_drift_mean": 0.6, "macro_drift_sample_std": 0.03},
        ]
        result = _symmetric_state_drift(("stanford_cars", "eurosat"), directed)[0]
        self.assertAlmostEqual(result["D_AB_state"], 0.4)

    def test_pair_level_join_all_six_without_orientation_errors(self):
        states, merges = [], []
        for index, (task_a, task_b) in enumerate(unordered_task_pairs(PILOT_TASKS)):
            states.append({
                "task_a": task_a, "task_b": task_b, "D_AB_state": index / 10,
                "D_A_from_B_mean": index + 0.1, "D_B_from_A_mean": index + 0.2,
            })
            if index % 2:
                merges.append({"task_a": task_b, "task_b": task_a, "G_AB": index,
                               "Retention_A": 0.7, "Retention_B": 0.8,
                               "selected_alpha": 1.0, "canonical_seed": 420})
            else:
                merges.append({"task_a": task_a, "task_b": task_b, "G_AB": index,
                               "Retention_A": 0.8, "Retention_B": 0.7,
                               "selected_alpha": 1.0, "canonical_seed": 420})
        joined = build_pair_level_join(PILOT_TASKS, states, merges)
        self.assertEqual(len(joined), 6)
        self.assertEqual(len({frozenset((row["task_a"], row["task_b"])) for row in joined}), 6)
        self.assertTrue(all(row["canonical_pairwise_baseline_seed"] == 420 for row in joined))

    def test_static_null_safety(self):
        rows = [
            {"metric": None, "degenerate": False},
            {"metric": float("nan"), "degenerate": False},
            {"metric": 0.5, "degenerate": True},
        ]
        result = safe_metric_aggregate(rows, "metric")
        self.assertIsNone(result["value"])
        self.assertEqual(result["valid_count"], 0)
        self.assertEqual(result["skipped_count"], 2)
        self.assertEqual(result["degenerate_count"], 1)


class RepairPlanningAndIsolationTests(unittest.TestCase):
    def test_git_dirty_provenance_is_graceful(self):
        self.assertIn(git_dirty(REPO), (True, False, None))

    def test_git_dirty_ignores_outputs_but_detects_source_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repo), "config", "user.name", "State Drift Test"],
                check=True,
            )
            (repo / ".gitignore").write_text(
                "analysis_outputs/state_dependent_task_drift/\n", encoding="utf-8"
            )
            (repo / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "commit", "-q", "-m", "fixture"], check=True
            )
            output = repo / "analysis_outputs" / "state_dependent_task_drift"
            output.mkdir(parents=True)
            (output / "result.json").write_text("{}\n", encoding="utf-8")
            self.assertFalse(git_dirty(repo))

            (repo / "untracked.py").write_text("VALUE = 2\n", encoding="utf-8")
            self.assertTrue(git_dirty(repo))
            subprocess.run(["git", "-C", str(repo), "add", "untracked.py"], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "commit", "-q", "-m", "add source"], check=True
            )
            (repo / "tracked.py").write_text("VALUE = 3\n", encoding="utf-8")
            self.assertTrue(git_dirty(repo))

    def test_pairwise_resume_identity_accepts_exact_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = tiny_cfg(root)
            task_a, task_b = PILOT_TASKS[:2]
            adapters = {
                task_a: root / "adapters" / task_a,
                task_b: root / "adapters" / task_b,
            }
            write_tiny_adapter(adapters[task_a], value=1.0)
            write_tiny_adapter(adapters[task_b], value=2.0)
            definition = pairwise_run_definition(cfg, task_a, task_b, adapters, None)
            run_dir = root / "pair"
            write_complete_pairwise(run_dir, definition)
            self.assertEqual(run_action(run_dir, resume=True), "skip")
            validated = validate_complete_pairwise_run(run_dir, definition)
            self.assertEqual(validated["status"]["status"], "COMPLETE")

            changed_precision = pairwise_run_definition(
                dict(cfg, precision="bf16"), task_a, task_b, adapters, None
            )
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                validate_complete_pairwise_run(run_dir, changed_precision)

            changed_limit = pairwise_run_definition(cfg, task_a, task_b, adapters, 1)
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                validate_complete_pairwise_run(run_dir, changed_limit)

            write_tiny_adapter(adapters[task_a], value=3.0)
            changed_adapter = pairwise_run_definition(cfg, task_a, task_b, adapters, None)
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                validate_complete_pairwise_run(run_dir, changed_adapter)

            write_complete_pairwise(run_dir, changed_adapter)
            write_tiny_adapter(adapters[task_b], value=4.0)
            changed_adapter = pairwise_run_definition(cfg, task_a, task_b, adapters, None)
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                validate_complete_pairwise_run(run_dir, changed_adapter)

    def test_pairwise_resume_rejects_tampered_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = tiny_cfg(root)
            task_a, task_b = PILOT_TASKS[:2]
            adapters = {
                task_a: root / "adapters" / task_a,
                task_b: root / "adapters" / task_b,
            }
            write_tiny_adapter(adapters[task_a], value=1.0)
            write_tiny_adapter(adapters[task_b], value=2.0)
            definition = pairwise_run_definition(cfg, task_a, task_b, adapters, None)
            run_dir = root / "pair"
            write_complete_pairwise(run_dir, definition)
            status = read_json(run_dir / "status.json")
            status["pairwise_run_definition_hash"] = "0" * 64
            atomic_write_json(run_dir / "status.json", status)
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                validate_complete_pairwise_run(run_dir, definition)

            write_complete_pairwise(run_dir, definition)
            frozen = read_json(run_dir / "frozen_selection.json")
            frozen["selected_alpha"] = definition["merge_configuration"]["alpha_candidates"][1]
            atomic_write_json(run_dir / "frozen_selection.json", frozen)
            with self.assertRaisesRegex(RuntimeError, "inconsistent"):
                validate_complete_pairwise_run(run_dir, definition)

    def test_pairwise_alpha_order_isolation(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2, bias=False)

        model = Tiny()
        delta = {"base_model.model.linear": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
        clean = capture_clean_target_weights(model, delta)
        results = {}
        for alpha in (0.1, 0.5, 1.0):
            materialize_from_clean(model, clean, delta, alpha)
            results[alpha] = model.linear.weight.detach().clone()
        for alpha in (1.0, 0.5, 0.1):
            materialize_from_clean(model, clean, delta, alpha)
            torch.testing.assert_close(model.linear.weight, results[alpha], rtol=0, atol=0)

    def test_targeted_pair_plans_canonical_anchor_dependency(self):
        dependencies = run_pilot_baseline_dependencies(
            PILOT_TASKS, (420, 421, 422), 420,
            target_task="stanford_cars", anchor_task="eurosat", target_seed=421,
        )
        self.assertEqual(
            dependencies,
            (("stanford_cars", 421), ("stanford_cars", 420), ("eurosat", 420)),
        )

    def test_reuse_validation_missing_mismatch_and_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = tiny_cfg(temporary)
            directory = Path(temporary) / "reuse" / "eurosat"
            write_tiny_adapter(directory)
            with self.assertRaisesRegex(RuntimeError, "missing baseline reuse metadata"):
                validate_reuse_adapter(cfg, directory, "eurosat", 420)
            report = adapter_report(directory)
            metadata = {
                "target_task": "eurosat", "target_training_seed": 420,
                "base_model_identifier": cfg["base_model_identifier"],
                "lora_rank": 2, "lora_alpha": 6.0, "effective_lora_scaling": 3.0,
                "lora_target_modules": ["q_proj"], "completion_status": "COMPLETE",
                "adapter_sha256": "0" * 64,
                "training_configuration": effective_training_recipe(cfg),
            }
            atomic_write_json(directory / "state_drift_reuse_metadata.json", metadata)
            with self.assertRaisesRegex(RuntimeError, "adapter_sha256"):
                validate_reuse_adapter(cfg, directory, "eurosat", 420)
            metadata["adapter_sha256"] = report["sha256"]
            atomic_write_json(directory / "state_drift_reuse_metadata.json", metadata)
            self.assertTrue(validate_reuse_adapter(cfg, directory, "eurosat", 420)["adapter_report"])


class DryRunIsolationTests(unittest.TestCase):
    def test_dry_run_does_not_import_heavy_runtime_or_touch_cuda(self):
        forbidden = ("transformers", "peft", "torchvision", "dataset")
        real_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name in forbidden or name.startswith(tuple(item + "." for item in forbidden)):
                raise AssertionError(f"forbidden dry-run import: {name}")
            return real_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "must-not-exist"
            with mock.patch("builtins.__import__", side_effect=guarded), mock.patch(
                "torch.cuda.is_available", side_effect=AssertionError("CUDA queried")
            ), mock.patch("sys.stdout"):
                code = main(["dry-run", "--config", str(CONFIG), "--output-root", str(output)])
            self.assertEqual(code, 0)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
