"""Statistical references and orchestration failure cases; no full mining campaign."""
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import stats

from experiments.analyze_repeated_search import analyze, friedman, holm, load_panel, mean_ci, plots, report
from experiments.repeat_random_search import command_for, run_plan, write_json, precise_metrics, ROOT


def fixture_plan():
    return dict(protocol="mease-repetitions-v1", hashes={}, config_ids=[9, 55, 172],
                configs={str(c): dict(score_metric="fast_logrank", redundancy_penalty=.5,
                                     redundancy_similarity_threshold=.5) for c in [9, 55, 172]},
                datasets=[dict(name=d, path="unused", time_col="time", event_col="status") for d in ["a", "b", "c"]],
                seeds=[1, 2], repetitions=2, baseline="complement", plot_rank=0,
                fixed=dict(archive_selection="greedy", rate_policy="adaptive", km_time_bins=512),
                jobs=[list(j) for j in itertools.product([9, 55, 172], ["a", "b", "c"], [1, 2])],
                selection="synthetic fixture")


def fixture_runs(n=6, r=30):
    rng = np.random.default_rng(7)
    rows = []
    for cid, dataset, seed in itertools.product([9, 55, 172], range(n), range(1, r + 1)):
        level = {9: .2, 55: .5, 172: .8}[cid]
        value = level + rng.uniform(-.02, .02)
        rows.append(dict(config_id=cid, dataset=str(dataset), seed=seed,
                         exceptionality=value, **{"model redundancy": 1-value, "setCov": value, "#sg": 5}))
    return pd.DataFrame(rows)


class StatisticsTests(unittest.TestCase):
    def test_tied_friedman_matches_scipy_and_enumeration(self):
        matrix = np.array([[1, 2, 2], [3, 2, 1], [2, 4, 4]])
        ranks, result = friedman(matrix, True, 39999, np.random.default_rng(7))
        reference = stats.friedmanchisquare(*matrix.T)
        self.assertAlmostEqual(result["friedman_chi2"], reference.statistic)
        self.assertAlmostEqual(result["p_chi2"], reference.pvalue)
        observed = np.sum((ranks.sum(axis=0) - 6)**2)
        null = [sum((np.sum(rows, axis=0) - 6)**2) for rows in
                itertools.product(*[list(itertools.permutations(row)) for row in ranks])]
        exact = np.mean(np.asarray(null) >= observed)
        self.assertAlmostEqual(result["p_permutation"], exact, delta=.015)

    def test_all_ties_and_perfect_ordering(self):
        _, tied = friedman(np.ones((6, 3)), True, 999, np.random.default_rng(0))
        self.assertEqual(tied["p_permutation"], 1)
        self.assertEqual(tied["friedman_chi2"], 0)
        _, ordered = friedman(np.tile([1, 2, 3], (6, 1)), True, 999, np.random.default_rng(0))
        self.assertEqual(ordered["friedman_chi2"], 12)
        self.assertEqual(ordered["p_iman_davenport"], 0)
        self.assertLess(ordered["p_permutation"], .01)

    def test_holm_and_paired_intervals(self):
        np.testing.assert_allclose(holm([.03, .001, .04]), [.06, .003, .06])
        values = np.array([.1, .2, .3, .4])
        mean, sd, low, high = mean_ci(values, .05)
        ref = stats.t.interval(.95, 3, loc=values.mean(), scale=stats.sem(values))
        np.testing.assert_allclose([low, high], ref)
        self.assertAlmostEqual(mean, .25)
        self.assertGreater(sd, 0)

    def test_dataset_blocks_directions_and_nemenyi(self):
        runs = fixture_runs()
        tables = analyze(runs, [9, 55, 172], [str(d) for d in range(6)], list(range(1, 31)), permutations=1999)
        self.assertTrue(tables["friedman"].datasets.eq(6).all())
        self.assertEqual(int(tables["ranking"].iloc[0].config_id), 172)
        for _, group in tables["metric_ranks"].groupby("metric"):
            self.assertEqual(group.sort_values("mean_rank").config_id.tolist(), [172, 55, 9])
        row = tables["nemenyi"].query("config_a == 9 and config_b == 172").iloc[0]
        expected_p = stats.studentized_range.sf(2 / np.sqrt(3*4/(12*6)), 3, np.inf)
        self.assertAlmostEqual(row.p_nemenyi, expected_p)
        self.assertEqual(bool(row.significant), True)
        paired = tables["paired_seed_intervals"]
        self.assertTrue((paired.simultaneous_ci_low <= paired.ci_low).all())
        self.assertTrue((paired.simultaneous_ci_high >= paired.ci_high).all())
        # Duplicating seeds does not change the number of Friedman blocks or statistic.
        more = pd.concat([runs, runs.assign(seed=runs.seed + 30)])
        other = analyze(more, [9, 55, 172], [str(d) for d in range(6)], list(range(1, 61)), permutations=999)
        np.testing.assert_allclose(tables["friedman"].friedman_chi2, other["friedman"].friedman_chi2)

    def test_report_and_plot_artifacts(self):
        runs = fixture_runs()
        plan = fixture_plan()
        plan.update(datasets=[dict(name=str(d)) for d in range(6)], seeds=list(range(1, 31)), repetitions=30)
        tables = analyze(runs, plan["config_ids"], [str(d) for d in range(6)], plan["seeds"], permutations=999)
        result = report(plan, tables, .05, 999, 2026)
        self.assertIn("não demonstra equivalência", result)
        self.assertIn("pós-seleção", result)
        with tempfile.TemporaryDirectory() as temp:
            plots(tables, Path(temp))
            self.assertTrue((Path(temp) / "mean_ranks.pdf").stat().st_size > 1000)


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mease repetitions test ")
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.plan = fixture_plan()
        write_json(self.output / "plan.json", self.plan)
        self.metrics = {"exceptionality": .8, "model redundancy": .2, "setCov": .9, "#sg": 5}
        self.calls = []

    def fake_mease(self, command, **kwargs):
        self.calls.append(command)
        def arg(key): return command[command.index(key) + 1]
        folder = Path(arg("--output_dir")) / arg("--dataset_name") / "complement"
        prefix = arg("--dataset_name") + "_0_complement_"
        info = {**self.plan["fixed"], **self.plan["configs"]["9"]}
        pd.DataFrame([info]).to_csv(folder / (prefix + "Info.csv"), index=False)
        pd.DataFrame([self.metrics]).to_csv(folder / (prefix + "RulesMetricsResult.csv"), index=False)
        pd.DataFrame({"Rule_Obj": ["unused"]}).to_csv(folder / (prefix + "DetailedRules.csv"), index=False)
        self.assertEqual(arg("--executions"), "1")
        self.assertEqual(kwargs["env"]["PYTHONHASHSEED"], "0")
        return subprocess.CompletedProcess(command, 0)

    def run_mocked(self, max_runs=None):
        with patch("experiments.repeat_random_search.subprocess.run", side_effect=self.fake_mease), \
                patch("experiments.repeat_random_search.precise_metrics", return_value=self.metrics):
            run_plan(self.plan, self.output, max_runs=max_runs)

    def test_resume_hash_repair_and_balanced_panel(self):
        self.run_mocked(max_runs=1)
        self.assertEqual(len(self.calls), 1)
        with self.assertRaisesRegex(ValueError, "Incomplete/invalid panel"):
            load_panel(self.output)
        self.run_mocked()
        self.assertEqual(len(self.calls), 18)
        self.run_mocked()
        self.assertEqual(len(self.calls), 18)
        plan, rows, hashes = load_panel(self.output)
        self.assertEqual(len(rows), 18)
        self.assertEqual(len(hashes), 18)
        _, folder = command_for(self.plan, self.output, *self.plan["jobs"][0])
        (folder / "metrics_full.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Incomplete/invalid panel"):
            load_panel(self.output)
        self.run_mocked()
        self.assertEqual(len(self.calls), 19)
        self.assertEqual(len(load_panel(self.output)[1]), 18)
        result = subprocess.run([sys.executable, str(ROOT / "experiments/analyze_repeated_search.py"),
                                 "--input", str(self.output), "--permutations", "999"],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.output / "analysis/relatorio.md").exists())
        tests = pd.read_csv(self.output / "analysis/friedman.csv")
        self.assertTrue(tests.p_permutation.eq(1).all())
        self.assertFalse(tests.reject_global.any())

    def test_failure_never_marks_complete(self):
        with patch("experiments.repeat_random_search.subprocess.run", return_value=subprocess.CompletedProcess([], 7)):
            with self.assertRaisesRegex(RuntimeError, "MEASE exited 7"):
                run_plan(self.plan, self.output)
        self.assertEqual(len(list(self.output.rglob(".complete.json"))), 0)

    def test_code_drift_and_duplicate_jobs_refused(self):
        self.plan["hashes"] = {"main.py": "wrong"}
        write_json(self.output / "plan.json", self.plan)
        with self.assertRaisesRegex(ValueError, "fingerprint differs"):
            load_panel(self.output)
        self.plan["hashes"] = {}
        self.plan["jobs"][1] = self.plan["jobs"][0]
        write_json(self.output / "plan.json", self.plan)
        with self.assertRaisesRegex(ValueError, "balanced panel"):
            load_panel(self.output)

    def test_recompute_real_exported_rules(self):
        folder = ROOT / "experiments/random_search_shell/config_9/mgus2/complement"
        if not folder.exists():
            self.skipTest("Local Random Search artifacts unavailable")
        self.plan["datasets"] = [dict(name="mgus2", path="datasets/files/mgus2.parquet", time_col="futime", event_col="status")]
        values = precise_metrics(self.plan, folder, "mgus2", {})
        self.assertAlmostEqual(values["setCov"], 1312/1338)
        self.assertAlmostEqual(values["model redundancy"], .1)


if __name__ == "__main__":
    unittest.main()
