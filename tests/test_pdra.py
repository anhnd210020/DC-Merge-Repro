import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from task_demand_prediction import pdra
from task_demand_prediction.storage import atomic_json, hash_file, output_lock, seal


ROOT = Path(__file__).resolve().parents[1]


class AllocationTests(unittest.TestCase):
    def test_templates_follow_descending_predicted_demand(self):
        tasks = ("stanford_cars", "dtd", "eurosat", "gtsrb")
        predicted = {"stanford_cars": 1.0, "dtd": 4.0, "eurosat": 3.0, "gtsrb": 2.0}
        ranks = pdra.assign_ranks(tasks, predicted, method="PDRA")
        self.assertEqual(ranks, {"dtd": 12, "eurosat": 8, "gtsrb": 8, "stanford_cars": 4})

        six = ("stanford_cars", "dtd", "eurosat", "gtsrb", "mnist", "svhn")
        six_demand = dict(zip(six, (1.0, 6.0, 5.0, 4.0, 3.0, 2.0)))
        six_ranks = pdra.assign_ranks(six, six_demand, method="PDRA")
        self.assertEqual([six_ranks[t] for t in ("dtd", "eurosat", "gtsrb", "mnist", "svhn", "stanford_cars")],
                         [12, 12, 8, 8, 4, 4])

    def test_ties_are_resolved_by_registered_task_order(self):
        tasks = ("stanford_cars", "dtd", "eurosat", "gtsrb")
        tied = {task: 1.0 for task in tasks}
        self.assertEqual(pdra.assign_ranks(tasks, tied, method="PDRA"),
                         {"stanford_cars": 12, "dtd": 8, "eurosat": 8, "gtsrb": 4})
        self.assertEqual(pdra.assign_ranks(tasks, tied, method="Reverse-PDRA"),
                         {"stanford_cars": 12, "dtd": 8, "eurosat": 8, "gtsrb": 4})

    def test_oracle_values_cannot_change_predicted_allocation(self):
        tasks = ("stanford_cars", "dtd", "eurosat", "gtsrb")
        predicted = {"stanford_cars": 1.0, "dtd": 4.0, "eurosat": 3.0, "gtsrb": 2.0}
        predicted_ranks = pdra.assign_ranks(tasks, predicted, method="PDRA")
        pdra.assign_ranks(tasks, predicted, method="Oracle-Ranked",
                          oracle_demand={task: float(i) for i, task in enumerate(tasks)})
        self.assertEqual(pdra.assign_ranks(tasks, predicted, method="PDRA"), predicted_ranks)

    def test_all_methods_keep_the_same_budget(self):
        for n, budget in ((4, 32), (6, 48)):
            tasks = pdra.TASKS[:n]
            values = {task: float(i) for i, task in enumerate(tasks)}
            allocations = [pdra.assign_ranks(tasks, values, method=m)
                           if m != "Oracle-Ranked" else
                           pdra.assign_ranks(tasks, values, method=m, oracle_demand=values)
                           for m in ("Uniform", "PDRA", "Reverse-PDRA", "Oracle-Ranked")]
            self.assertTrue(all(sum(row.values()) == budget for row in allocations))
            self.assertTrue(all(set(row.values()).issubset({2, 4, 8, 12, 16}) for row in allocations))

    def test_oracle_ranking_needs_explicit_oracle_values(self):
        with self.assertRaises(ValueError):
            pdra.assign_ranks(("a", "b", "c", "d"), {"a": 1, "b": 2, "c": 3, "d": 4},
                              method="Oracle-Ranked")


class ManifestAndProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = pdra.load_context_manifest(ROOT / "task_demand_prediction" / "pdra_context_manifest.json")
        cls.predictor = pdra.load_predictor_artifact(ROOT / "task_demand_prediction" / "pdra_predictor.json")

    def test_manifest_contains_only_new_exact_memberships_and_covers_tasks(self):
        pdra.validate_context_manifest(self.manifest)
        payload = self.manifest["payload"]
        self.assertEqual(len(payload["fresh_contexts"]["four_task"]), 45)
        self.assertEqual(len(payload["fresh_contexts"]["six_task"]), 19)
        self.assertEqual(payload["prior_reused_context_counts"], {"four_task": 25, "six_task": 9})
        self.assertTrue(all(value > 0 for value in payload["task_coverage"]["four_task"].values()))
        self.assertTrue(all(value > 0 for value in payload["task_coverage"]["six_task"].values()))

    def test_predictor_records_validation_rows_and_never_test_labels(self):
        pdra.validate_predictor_artifact(self.predictor)
        payload = self.predictor["payload"]
        self.assertEqual(payload["fit_split"], "validation")
        self.assertEqual(len(payload["training_rows"]), 118)
        self.assertTrue(all(row["split"] == "validation" for row in payload["training_rows"]))
        self.assertEqual(payload["source"]["test_label_files_read"], [])
        self.assertFalse(any("final_test" in name for name in payload["source"]["read_members"]))
        self.assertFalse(payload["final_test_labels_used"])
        self.assertEqual(set(payload["task_features"]), set(pdra.TASKS))
        self.assertEqual(set(payload["source"]["source_head_sha256"]), set(pdra.TASKS))
        self.assertTrue(payload["source"]["source_model_files"])
        self.assertTrue(payload["source"]["source_dataset_identity"]["required_file_sha256"])

    def test_predictor_rejects_an_unsealed_edit(self):
        tampered = json.loads(json.dumps(self.predictor))
        tampered["payload"]["estimator"]["alpha"] = 4.0
        with self.assertRaises(ValueError):
            pdra.validate_predictor_artifact(tampered)

    def test_preflight_is_static_and_does_not_load_evaluator(self):
        report = pdra.static_preflight(ROOT)
        self.assertEqual(report["inference_started"], False)
        self.assertEqual(report["context_count"], 64)
        self.assertEqual(report["oracle_sweep_units_per_split"], 1240)


class MetricTests(unittest.TestCase):
    def test_context_and_paired_metrics_include_worst_task_and_gain_status(self):
        task_rows = [
            {"method": "Uniform", "retention_percent": 90.0},
            {"method": "Uniform", "retention_percent": 70.0},
            {"method": "PDRA", "retention_percent": 95.0},
            {"method": "PDRA", "retention_percent": 75.0},
            {"method": "Reverse-PDRA", "retention_percent": 80.0},
            {"method": "Reverse-PDRA", "retention_percent": 80.0},
            {"method": "Oracle-Ranked", "retention_percent": 82.0},
            {"method": "Oracle-Ranked", "retention_percent": 82.0},
        ]
        result = pdra.summarize_context(task_rows)
        self.assertEqual(result["mean_retention_percent"]["Uniform"], 80.0)
        self.assertEqual(result["worst_task_retention_percent"]["PDRA"], 75.0)
        self.assertEqual(result["pdra_minus_uniform_pp"], 5.0)
        self.assertEqual(result["pdra_minus_reverse_pp"], 5.0)
        self.assertEqual(result["oracle_headroom_pp"], 2.0)
        self.assertEqual(result["captured_gain"], 2.5)

        no_headroom = pdra.captured_gain(1.0, 1.0, 1.0)
        negative_headroom = pdra.captured_gain(2.0, 1.0, 0.0)
        self.assertIsNone(no_headroom["value"])
        self.assertEqual(no_headroom["status"], "zero_headroom")
        self.assertIsNone(negative_headroom["value"])
        self.assertEqual(negative_headroom["status"], "negative_headroom")

    def test_paired_summary_counts_wins(self):
        contexts = [
            {"context_id": "a", "pdra_minus_uniform_pp": 2.0},
            {"context_id": "b", "pdra_minus_uniform_pp": 0.0},
            {"context_id": "c", "pdra_minus_uniform_pp": -1.0},
        ]
        result = pdra.summarize_paired(contexts, "pdra_minus_uniform_pp")
        self.assertAlmostEqual(result["mean_paired_improvement_pp"], 1.0 / 3.0)
        self.assertEqual(result["win_rate"], 1.0 / 3.0)


class OracleSweepTests(unittest.TestCase):
    def test_oracle_sweep_cost_and_demand_use_frozen_unit_rank_axis(self):
        four = {"context_id": "four", "tasks": list(pdra.TASKS[:4])}
        six = {"context_id": "six", "tasks": list(pdra.TASKS[:6])}
        self.assertEqual(len(pdra.make_oracle_sweep_units([four])), 17)
        self.assertEqual(len(pdra.make_oracle_sweep_units([six])), 25)
        self.assertEqual(len(pdra.make_oracle_sweep_units([four, six])), 42)

        units = pdra.make_oracle_sweep_units([four])
        references = {task: 100.0 for task in four["tasks"]}
        sweep_results = {}
        for unit in units:
            result = {task: {"accuracy_percent": 80.0} for task in four["tasks"]}
            if unit["focal_task"] is not None and unit["level"] == 2:
                result[unit["focal_task"]]["accuracy_percent"] = 78.0
            sweep_results[unit["run_id"]] = {"per_task": result}
        demands = pdra.oracle_demands_from_sweep([four], sweep_results, references)
        self.assertAlmostEqual(demands["four"][four["tasks"][0]], 2.0 / 14.0)

    def test_allocation_units_slice_global_predictions_to_context(self):
        context = {"context_id": "fresh", "tasks": list(pdra.TASKS[:4])}
        predictions = {task: float(i) for i, task in enumerate(pdra.TASKS)}
        oracle = {"fresh": {task: float(7 - i) for i, task in enumerate(context["tasks"])} }
        units = pdra.allocation_units([context], predictions, oracle)
        self.assertEqual(len(units), 4)
        self.assertEqual({unit["method"] for unit in units}, set(pdra.METHODS))


class PathTests(unittest.TestCase):
    def test_output_must_be_external_and_have_existing_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            outside = Path(temp) / "outputs" / "pdra"
            outside.parent.mkdir()
            self.assertEqual(pdra.validate_output_path(ROOT, outside), outside.resolve())
            with self.assertRaises(ValueError):
                pdra.validate_output_path(ROOT, ROOT / "outputs" / "pdra")


class ResumeTests(unittest.TestCase):
    def test_run_unit_refuses_unsealed_partial_and_skips_verified_result(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calls = []
            unit = {"run_id": "ctx__Uniform", "context_id": "ctx", "method": "Uniform"}
            evaluator = lambda _: calls.append(1) or {"accuracy": 90}
            pdra.run_unit(root, "validation", unit, evaluator, identity_sha256="identity")
            original = (root / "validation" / "runs" / "ctx__Uniform.json").read_bytes()
            pdra.run_unit(root, "validation", unit, evaluator, identity_sha256="identity", resume=True)
            self.assertEqual(len(calls), 1)
            self.assertEqual((root / "validation" / "runs" / "ctx__Uniform.json").read_bytes(), original)

            partial = {"run_id": "ctx__PDRA", "context_id": "ctx", "method": "PDRA"}
            partial_path = root / "validation" / "runs" / "ctx__PDRA.json"
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path.write_text("partial", encoding="utf-8")
            with self.assertRaises(ValueError):
                pdra.run_unit(root, "validation", partial, evaluator, identity_sha256="identity")

    def test_complete_without_done_recovers_only_after_all_hashes_verify(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage_dir = root / "validation"
            runs_dir = stage_dir / "runs"
            runs_dir.mkdir(parents=True)
            manifest = pdra.load_context_manifest(ROOT / "task_demand_prediction" / pdra.MANIFEST_FILENAME)
            predictor = pdra.load_predictor_artifact(ROOT / "task_demand_prediction" / pdra.PREDICTOR_FILENAME)
            identity_sha = pdra._register_output(root, {"test_identity": "resume-recovery"}, manifest, predictor)
            expected_ids = pdra._expected_run_ids(root)
            run_hashes = {}
            for run_id in sorted(expected_ids):
                unit = {"run_id": run_id}
                result = {"run_id": run_id, "unit": unit, "unit_sha256": pdra.sha256_json(unit),
                          "identity_sha256": identity_sha, "result": {}}
                result_path = runs_dir / f"{run_id}.json"
                atomic_json(result_path, result)
                result_hash = hash_file(result_path)
                run_hashes[run_id] = result_hash
                atomic_json(runs_dir / f"{run_id}.unit.json",
                            seal({"result_sha256": result_hash, "identity_sha256": identity_sha}))

            task_results = stage_dir / "task_results.csv"
            task_results.write_text("run_id\n", encoding="utf-8")
            summary = stage_dir / "summary.json"
            atomic_json(summary, {"summary": "sealed by completion hash"})
            oracle = stage_dir / "oracle_demand_same_split.json"
            atomic_json(oracle, {"oracle": "sealed by completion hash"})
            completion = {"stage": "validation", "identity_sha256": identity_sha,
                "context_manifest_sha256": manifest["sha256"], "predictor_sha256": predictor["sha256"],
                "oracle_policy": pdra.ORACLE_POLICY,
                "configuration_count": len(expected_ids), "allocation_unit_count": 256,
                "oracle_sweep_unit_count": 1240, "run_result_sha256": run_hashes,
                "task_results_sha256": hash_file(task_results), "summary_sha256": hash_file(summary),
                "oracle_demand_sha256": hash_file(oracle),
                "final_test_labels_used_for_pdra_prediction_or_assignment": False,
                "test_labels_used_for_oracle_ranked_diagnostic": False, "status": "complete"}
            complete_path = stage_dir / "COMPLETE.json"
            atomic_json(complete_path, seal(completion))
            status_dir = root / "status"
            status_dir.mkdir()
            running_path = status_dir / "validation.RUNNING.json"
            atomic_json(running_path, seal({"stage": "validation", "identity_sha256": identity_sha,
                                            "inference_started": True}))

            with output_lock(root):
                pdra._verify_completed_stage(root, "validation", identity_sha, recover_missing_done=True)
            done_path = root / "status" / "validation.DONE.json"
            self.assertEqual(pdra.unseal(json.loads(done_path.read_text(encoding="utf-8"))),
                             {"identity_sha256": identity_sha, "completion_sha256": hash_file(complete_path)})
            self.assertFalse(running_path.exists())

            done_path.unlink()
            damaged_run = runs_dir / f"{sorted(expected_ids)[0]}.json"
            damaged_run.write_text("damaged", encoding="utf-8")
            with output_lock(root), self.assertRaisesRegex(ValueError, "result changed"):
                pdra._verify_completed_stage(root, "validation", identity_sha, recover_missing_done=True)
            self.assertFalse(done_path.exists())


class FinalTestGateTests(unittest.TestCase):
    def _run_cli_final_test(self, output: Path, identity: dict):
        from task_demand_prediction.__main__ import main

        paths = SimpleNamespace(output=output)
        predictor = pdra.load_predictor_artifact(ROOT / "task_demand_prediction" / pdra.PREDICTOR_FILENAME)
        preflight_result = (paths, identity, {}, {task: 80.0 for task in pdra.TASKS}, predictor)
        evaluator_calls = []
        argv = ["pdra-run", "--repo-root", str(ROOT), "--output-dir", str(output), "--device", "cpu",
                "--stage", "final-test", "--confirm-oracle-policy", pdra.ORACLE_POLICY_TOKEN]
        with patch.object(pdra, "_load_server_preflight", return_value=preflight_result), \
                patch.object(pdra, "_make_real_evaluator", side_effect=lambda *_: evaluator_calls.append("called")):
            with self.assertRaisesRegex(ValueError, "final-test blocked"):
                main(argv)
        self.assertEqual(evaluator_calls, [])
        self.assertFalse((output / "final_test").exists())
        self.assertFalse((output / "status" / "final_test.RUNNING.json").exists())

    def test_cli_final_test_without_validation_completion_stops_before_evaluator_or_run_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "run"
            self._run_cli_final_test(output, {"test_identity": "missing-validation"})

    def test_cli_final_test_with_tampered_validation_done_marker_stops_before_evaluator(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "run"
            output.mkdir()
            identity = {"test_identity": "tampered-validation"}
            identity_sha = pdra.sha256_json(identity)
            validation = output / "validation"
            validation.mkdir()
            atomic_json(validation / "COMPLETE.json", seal({"stage": "validation", "identity_sha256": identity_sha}))
            status = output / "status"
            status.mkdir()
            atomic_json(status / "validation.DONE.json", {"payload": {"identity_sha256": identity_sha}, "sha256": "tampered"})
            self._run_cli_final_test(output, identity)

    def test_resume_rejects_wrong_identity_running_marker_without_modifying_or_running(self):
        from task_demand_prediction.__main__ import main

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "run"
            status = output / "status"
            status.mkdir(parents=True)
            marker_path = status / "validation.RUNNING.json"
            atomic_json(marker_path, seal({"stage": "validation", "identity_sha256": "different-identity",
                                           "inference_started": True}))
            original_marker = marker_path.read_bytes()

            identity = {"test_identity": "current-validation"}
            paths = SimpleNamespace(output=output)
            predictor = pdra.load_predictor_artifact(ROOT / "task_demand_prediction" / pdra.PREDICTOR_FILENAME)
            preflight_result = (paths, identity, {}, {task: 80.0 for task in pdra.TASKS}, predictor)
            evaluator_calls = []
            argv = ["pdra-run", "--repo-root", str(ROOT), "--output-dir", str(output), "--device", "cpu",
                    "--stage", "validation", "--resume", "--confirm-oracle-policy", pdra.ORACLE_POLICY_TOKEN]
            with patch.object(pdra, "_load_server_preflight", return_value=preflight_result), \
                    patch.object(pdra, "_make_real_evaluator", side_effect=lambda *_: evaluator_calls.append("called")):
                with self.assertRaisesRegex(ValueError, "RUNNING marker does not match"):
                    main(argv)

            self.assertEqual(evaluator_calls, [])
            self.assertEqual(marker_path.read_bytes(), original_marker)
            self.assertFalse((output / "validation").exists())
            self.assertFalse((status / "validation.DONE.json").exists())


if __name__ == "__main__":
    unittest.main()
