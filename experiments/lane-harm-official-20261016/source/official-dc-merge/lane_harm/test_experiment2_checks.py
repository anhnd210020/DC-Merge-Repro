"""Small CPU-independent checks for the Experiment 2 result gates."""

import json
import unittest
from pathlib import Path

import run_week1 as week1


REFERENCE = json.loads((Path(__file__).parent / "experiment1_reference.json").read_text())


class Experiment2Checks(unittest.TestCase):
    def test_official_reference_rejects_validation_and_test_drift(self):
        cid = "all8"
        acc = REFERENCE["validation_accuracy_by_context"][cid].copy()
        row = {"context": cid, "method": "dc_official", "selected_coef": 0.4,
               "test_mean_norm": REFERENCE["all8_test_mean_normalized"],
               "val_sweep": [{"coef": 0.4, "acc": acc}]}
        week1.check_experiment1_reference(row, REFERENCE)
        row["val_sweep"][0]["acc"] = {**acc, "eurosat": acc["eurosat"] + 1.0}
        with self.assertRaisesRegex(AssertionError, "validation at alpha"):
            week1.check_experiment1_reference(row, REFERENCE)
        row["val_sweep"][0]["acc"] = acc
        row["test_mean_norm"] += 1.0
        with self.assertRaisesRegex(AssertionError, "normalized test"):
            week1.check_experiment1_reference(row, REFERENCE)

    def test_mobius_recovers_main_effects_and_interaction(self):
        f = {"none": 100.0, "gtsrb": 95.0, "svhn": 96.0, "gtsrb+svhn": 88.0}
        h = week1.mobius(f, ["gtsrb", "svhn"])
        self.assertEqual(h, {"gtsrb": 5.0, "svhn": 4.0, "gtsrb+svhn": 3.0})
        self.assertEqual(sum(h.values()), f["none"] - f["gtsrb+svhn"])

    def test_decomposition_rejects_incomplete_subsets_and_baseline_drift(self):
        cid = week1.EUROSAT_CORE[0]
        tasks = list(week1.ro.CONTEXTS[cid])
        comps = [t for t in tasks if t != week1.TARGET]
        baseline = REFERENCE["rank_validation_accuracy_by_context"][cid]
        rows = []
        for kind, name, scales in week1.configurations(tasks)[1]:
            size = len(comps) - sum(scales.get(c, 1.0) == 0.0 for c in comps)
            eurosat = baseline["eurosat"] + (3 - size) * 2
            rows.append({"kind": kind, "name": name, "scales": scales,
                         "acc": {**baseline, "eurosat": eurosat}})
        row = {"context": cid, "operator": "rank", "alpha": 0.4,
               "tasks": tasks, "companions": comps,
               "solo_acc": baseline["eurosat"] + 8,
               "lane_checks": {"max_rel_sum_error": 0.0, "max_abs_diff_vs_official": None},
               "results": rows}
        week1.check_decomposition_row(row, REFERENCE)
        row["results"] = rows[:-1]
        with self.assertRaisesRegex(AssertionError, "every lane configuration"):
            week1.check_decomposition_row(row, REFERENCE)
        row["results"] = rows
        full_index = next(i for i, r in enumerate(rows) if r["kind"] == "subset"
                          and r["name"] == "+".join(comps))
        rows[full_index] = {**rows[full_index],
                            "acc": {**rows[full_index]["acc"], "eurosat": baseline["eurosat"] + 1}}
        with self.assertRaisesRegex(AssertionError, "full-context validation"):
            week1.check_decomposition_row(row, REFERENCE)


if __name__ == "__main__":
    unittest.main()
