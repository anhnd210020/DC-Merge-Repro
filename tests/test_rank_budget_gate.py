"""CPU checks for the separate rank-tied Uniform validation gate."""
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from task_demand_prediction import pdra, rank_budget_gate as gate


class GatePlanTests(unittest.TestCase):
    def test_only_fresh_four_task_uniform_contexts_are_planned(self):
        from pathlib import Path
        repo = Path(__file__).resolve().parents[1]
        manifest = pdra.load_context_manifest(repo / "task_demand_prediction" / pdra.MANIFEST_FILENAME)
        units = gate._gate_units(manifest["payload"]["fresh_contexts"]["four_task"])
        self.assertEqual(len(units), 45)
        self.assertEqual(len({unit["run_id"] for unit in units}), 45)
        for unit in units:
            self.assertEqual(len(unit["tasks"]), 4)
            self.assertEqual(set(unit["rank_by_task"].values()), {8})
            self.assertEqual(sum(unit["rank_by_task"].values()), 32)
            self.assertEqual(set(unit["density_by_task"].values()), {1.0})

    def test_paired_summary_preserves_context_and_worst_task(self):
        units = gate._gate_units([{"context_id": "x", "tasks": ["a", "b", "c", "d"]}])
        old = {"per_task": {t: {"accuracy_percent": x} for t, x in
                            zip(units[0]["tasks"], (80, 70, 60, 50))}}
        new = {"per_task": {t: {"accuracy_percent": x} for t, x in
                            zip(units[0]["tasks"], (80, 70, 60, 54))}}
        result = gate._summarize(units, {units[0]["run_id"]: new}, {"x": old},
                                 {t: 100 for t in units[0]["tasks"]})
        self.assertEqual(result["mean_paired_difference_pp"], 1.0)
        self.assertEqual(result["mean_worst_task_difference_pp"], 4.0)

    def test_separate_runner_writes_and_verifies_only_gate_results(self):
        tasks = ["a", "b", "c", "d"]
        context = {"context_id": "fresh", "tasks": tasks}
        manifest = {"payload": {"fresh_contexts": {"four_task": [context]}}}
        old = {"per_task": {task: {"accuracy_percent": 70} for task in tasks}}
        new = {"per_task": {task: {"accuracy_percent": 72} for task in tasks}}
        baseline = {"completion_sha256": "old", "archived_identity_sha256": "old-id",
                    "results": {"fresh": old}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            repo.mkdir()
            output = root / "gate-output"
            args = ["run", "--repo-root", str(repo), "--baseline-archive", str(root / "archive.tar.gz"),
                    "--output-dir", str(output)]
            with (patch.object(pdra, "load_context_manifest", return_value=manifest),
                  patch.object(pdra, "load_predictor_artifact", return_value={"sha256": "p"}),
                  patch.object(gate, "load_baseline", return_value=baseline),
                  patch.object(pdra, "_load_server_preflight", return_value=(
                      SimpleNamespace(output=output), {"asset": "identity"}, {},
                      {task: 100 for task in tasks}, {})),
                  patch.object(gate, "_registration", return_value={"protocol": gate.GATE_NAME}),
                  patch("task_demand_prediction.runtime.RealEvaluator", return_value=lambda *_: new)):
                self.assertEqual(gate.main(args), 0)
                summary = json.loads((output / "validation" / "summary.json").read_text())
                self.assertEqual(summary["mean_paired_difference_pp"], 2.0)
                self.assertEqual(gate.main(["verify", *args[1:]]), 0)
                self.assertEqual(len(list((output / "validation" / "runs").glob("*.unit.json"))), 1)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for matrix operator checks")
class RankBudgetOperatorTests(unittest.TestCase):
    def test_rank_policy_reaches_global_density_one_path(self):
        import sys
        from pathlib import Path
        repo = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo / "vision_lora_merge"))
        from merging_functions import dc_merge

        generator = torch.Generator().manual_seed(741)
        matrices = [torch.randn(8, 8, generator=generator) for _ in range(4)]
        grouped = {"module": matrices}
        options = dict(smoothing_strategy="linear", rho=5.0,
                       task_ranks=(2, 2, 2, 2), task_densities=(1., 1., 1., 1.),
                       return_intermediates=True)
        legacy_merged, legacy = dc_merge(grouped, **options)
        explicit_merged, explicit = dc_merge(grouped, topk_policy="legacy", **options)
        _, rank = dc_merge(grouped, topk_policy="rank", **options)

        self.assertTrue(torch.equal(legacy_merged["module"], explicit_merged["module"]))
        for i in range(4):
            self.assertTrue(torch.equal(legacy["module"]["filtered_matrices"][i],
                                        explicit["module"]["filtered_matrices"][i]))
            self.assertEqual(legacy["module"]["density"][i]["retained_nonzero_count"], 1)
            self.assertEqual(rank["module"]["density"][i]["retained_nonzero_count"], 2)
            self.assertEqual(torch.count_nonzero(rank["module"]["filtered_matrices"][i]).item(), 2)
        self.assertEqual(rank["module"]["topk_policy"], "rank")

    def test_eight_task_default_topk_count_is_unchanged(self):
        import sys
        from pathlib import Path
        repo = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo / "vision_lora_merge"))
        from merging_functions import _keep_topk_count, keep_topk_percent
        matrix = torch.arange(1, 128 * 128 + 1, dtype=torch.float32).reshape(128, 128)
        self.assertTrue(torch.equal(_keep_topk_count(matrix, 16),
                                    keep_topk_percent([matrix], 1e-3)[0]))


if __name__ == "__main__":
    unittest.main()
