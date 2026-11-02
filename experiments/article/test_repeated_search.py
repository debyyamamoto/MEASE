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

from experiments.analyze_repeated_search import analyze, friedman, load_panel, load_single_panel, plots, report
from experiments.repeat_random_search import command_for, run_plan, write_json, precise_metrics, sha256, ROOT


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
    def test_tied_friedman_matches_scipy(self):
        matrix = np.array([[1, 2, 2], [3, 2, 1], [2, 4, 4]])
        ranks, result = friedman(matrix, True)
        reference = stats.friedmanchisquare(*matrix.T)
        np.testing.assert_allclose(ranks[0], [1, 2.5, 2.5])
        self.assertAlmostEqual(result["friedman_chi2"], reference.statistic)
        self.assertAlmostEqual(result["p_chi2"], reference.pvalue)
        expected_f = 2 * reference.statistic / (6 - reference.statistic)
        self.assertAlmostEqual(result["iman_davenport_f"], expected_f)
        self.assertAlmostEqual(result["p_value"], stats.f.sf(expected_f, 2, 4))

    def test_all_ties_and_perfect_ordering(self):
        _, tied = friedman(np.ones((6, 3)), True)
        self.assertEqual(tied["p_value"], 1)
        self.assertEqual(tied["friedman_chi2"], 0)
        _, ordered = friedman(np.tile([1, 2, 3], (6, 1)), True)
        self.assertEqual(ordered["friedman_chi2"], 12)
        self.assertEqual(ordered["p_value"], 0)
        self.assertEqual(ordered["iman_davenport_f"], np.inf)

    def test_dataset_blocks_directions_and_nemenyi(self):
        runs = fixture_runs()
        tables = analyze(runs, [9, 55, 172], [str(d) for d in range(6)], list(range(1, 31)))
        self.assertTrue(tables["friedman"].datasets.eq(6).all())
        self.assertEqual(int(tables["ranking"].iloc[0].config_id), 172)
        np.testing.assert_allclose(tables["ranking"].mean_rank, [1, 2, 3])
        for _, group in tables["metric_ranks"].groupby("metric"):
            self.assertEqual(group.sort_values("mean_rank").config_id.tolist(), [172, 55, 9])
        row = tables["nemenyi"].query("config_a == 9 and config_b == 172").iloc[0]
        expected_p = stats.studentized_range.sf(2 / np.sqrt(3*4/(12*6)), 3, np.inf)
        self.assertAlmostEqual(row.p_nemenyi, expected_p)
        self.assertTrue(row.significant)
        # Table 5(a), Demšar: q(.05, K=3)=2.343, with no alpha/3 adjustment.
        self.assertAlmostEqual(row.nemenyi_cd, 2.343 * np.sqrt(3*4/(6*6)), delta=.001)
        self.assertEqual(len(tables["nemenyi_significant"]), 4)
        self.assertEqual(len(tables["nemenyi_not_significant"]), 8)
        self.assertTrue((tables["nemenyi"].p_nemenyi.lt(.05) == tables["nemenyi"].significant).all())
        # Duplicating seeds does not increase the number of independent blocks.
        more = pd.concat([runs, runs.assign(seed=runs.seed + 30)])
        other = analyze(more, [9, 55, 172], [str(d) for d in range(6)], list(range(1, 61)))
        np.testing.assert_allclose(tables["friedman"].friedman_chi2, other["friedman"].friedman_chi2)
        np.testing.assert_allclose(tables["nemenyi"].p_nemenyi, other["nemenyi"].p_nemenyi)

    def test_average_repetitions_before_ranking(self):
        rows = []
        # Mean(A)=.6 > mean(B)=.5 although B wins in two of the three seeds.
        for dataset, seed, cid in itertools.product(["a", "b", "c"], range(3), [1, 2, 3]):
            value = {1: [.4, .4, 1], 2: [.5, .5, .5], 3: [.1, .1, .1]}[cid][seed]
            rows.append(dict(config_id=cid, dataset=dataset, seed=seed, exceptionality=value,
                             setCov=value, **{"model redundancy": 1-value, "#sg": 5}))
        tables = analyze(pd.DataFrame(rows), [1, 2, 3], ["a", "b", "c"], [0, 1, 2])
        self.assertEqual(tables["ranking"].config_id.tolist(), [1, 2, 3])
        np.testing.assert_allclose(tables["ranking"].mean_rank, [1, 2, 3])

    def test_demsar_published_critical_values(self):
        # Table 5(a), plus the K=4, N=14 example's CD=1.25.
        for k, published in [(3, 2.343), (4, 2.569), (10, 3.164)]:
            q_alpha = stats.studentized_range.ppf(.95, k, np.inf) / np.sqrt(2)
            self.assertAlmostEqual(q_alpha, published, delta=.001)
        cd = stats.studentized_range.ppf(.95, 4, np.inf) * np.sqrt(4*5/(12*14))
        self.assertAlmostEqual(cd, 1.25, delta=.005)

    def test_composite_is_reranked_with_ties_in_both_modes(self):
        # Metric ranks average to [2, 2, 3, 3], whose test ranks are [1.5, 1.5, 3.5, 3.5].
        ids, datasets = [7, 2, 19, 4], [str(d) for d in range(6)]
        for repetitions in (1, 2):
            with self.subTest(repetitions=repetitions):
                seeds = list(range(repetitions))
                rows = []
                for dataset, seed, i in itertools.product(datasets, seeds, range(4)):
                    rows.append(dict(
                        config_id=ids[i], dataset=dataset, seed=seed,
                        exceptionality=[.9, .7, .5, .3][i], setCov=[.2, .6, .4, .8][i],
                        **{"model redundancy": [.1, .3, .5, .7][i], "#sg": 5},
                    ))
                tables = analyze(pd.DataFrame(rows), ids, datasets, seeds)
                cells = tables["ranks_per_dataset"].query("dataset == '0'").set_index("config_id").loc[ids]
                np.testing.assert_allclose(cells.mean_metric_rank, [2, 2, 3, 3])
                np.testing.assert_allclose(cells.rank_mean_metric_rank, [1.5, 1.5, 3.5, 3.5])
                omnibus = tables["friedman"].set_index("metric").loc["mean_metric_rank"]
                self.assertEqual(omnibus.datasets, 6)
                self.assertAlmostEqual(omnibus.friedman_chi2, 18)
                self.assertTrue(omnibus.reject_global)
                pairs = tables["nemenyi"].query("metric == 'mean_metric_rank'")
                pair = pairs.query("config_a == 7 and config_b == 19").iloc[0]
                self.assertEqual(pair.rank_difference_a_minus_b, -2)
                self.assertAlmostEqual(pair.p_nemenyi, stats.studentized_range.sf(2 / np.sqrt(20/72), 4, np.inf))
                self.assertTrue(pair.significant)
                self.assertEqual(pairs.query("config_a == 7 and config_b == 2").iloc[0].p_nemenyi, 1)

    def test_omnibus_gates_posthoc(self):
        runs = fixture_runs()
        runs[["exceptionality", "model redundancy", "setCov"]] = .5
        tables = analyze(runs, [9, 55, 172], [str(d) for d in range(6)], list(range(1, 31)))
        self.assertFalse(tables["friedman"].reject_global.any())
        self.assertTrue(tables["nemenyi"].p_nemenyi.isna().all())
        self.assertTrue(tables["nemenyi"].decision.eq("not_tested").all())
        self.assertFalse(tables["nemenyi"].significant.any())
        self.assertTrue(tables["nemenyi_not_significant"].empty)

    def test_invalid_panel_refused(self):
        runs = fixture_runs()
        args = ([9, 55, 172], [str(d) for d in range(6)], list(range(1, 31)))
        for bad in [runs.iloc[1:], pd.concat([runs, runs.iloc[:1]]), runs.assign(setCov=np.nan)]:
            with self.assertRaises(ValueError):
                analyze(bad, *args)

    def test_report_and_plot_artifacts(self):
        runs = fixture_runs()
        plan = fixture_plan()
        plan.update(datasets=[dict(name=str(d)) for d in range(6)], seeds=list(range(1, 31)), repetitions=30)
        tables = analyze(runs, plan["config_ids"], [str(d) for d in range(6)], plan["seeds"])
        result = report(plan, tables, .05)
        self.assertIn("não demonstra equivalência", result)
        self.assertIn("pós-seleção", result)
        self.assertIn("sem Holm ou Bonferroni adicional", result)
        with tempfile.TemporaryDirectory() as temp:
            plots(tables, Path(temp))
            self.assertTrue((Path(temp) / "mean_ranks.pdf").stat().st_size > 1000)


class SingleModeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mease single test ")
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name)
        plan = fixture_plan()
        config = plan["configs"]["9"]
        (self.source / "configs.txt").write_text((repr(config) + "\n") * 12)
        (self.source / "protocol.txt").write_text(
            "protocol=mease-random-search-shell-v1\nnum_configs=12\n"
            "executions=1\nrun_seed=37\nbaseline=complement\n"
            + "".join(f"dataset={d}|unused|time|status\n" for d in ["a", "b", "c"])
        )
        for cid, dataset in itertools.product(range(12), ["a", "b", "c"]):
            folder = self.source / f"config_{cid}" / dataset / "complement"
            folder.mkdir(parents=True)
            value = .2 + cid * .05
            prefix = f"{dataset}_0_complement_"
            pd.DataFrame([dict(exceptionality=value, setCov=value,
                               **{"model redundancy": 1-value, "#sg": 5})]).to_csv(
                folder / (prefix + "RulesMetricsResult.csv"), index=False)
            pd.DataFrame([{**plan["fixed"], **config}]).to_csv(folder / (prefix + "Info.csv"), index=False)
            (folder / (prefix + "DetailedRules.csv")).write_text("unused\n")
            (folder / "command.sh").write_text("# fixture\n")
            (folder / ".complete").write_text("".join(
                f"{sha256(path)}  {path.name}\n" for path in sorted(folder.iterdir())
            ))

    def test_original_top_ten_and_one_observation_per_dataset(self):
        plan, runs, hashes = load_single_panel(self.source)
        self.assertEqual(plan["config_ids"], list(range(11, 1, -1)))
        self.assertEqual(plan["seeds"], [37])
        self.assertEqual(plan["repetitions"], 1)
        self.assertEqual(len(runs), 30)
        self.assertIn("protocol.txt", hashes)
        self.assertIn("config_0/a/complement/.complete", hashes)
        tables = analyze(runs, plan["config_ids"], ["a", "b", "c"], plan["seeds"])
        self.assertTrue(tables["friedman"].repetitions.eq(1).all())
        self.assertTrue(tables["friedman"].datasets.eq(3).all())
        self.assertEqual(len(tables["nemenyi"]), 180)
        # A single observation is used unchanged, rather than drawn from repeated runs.
        self.assertAlmostEqual(tables["dataset_means"].query("config_id == 11").iloc[0].setCov, .75)

    def test_missing_entire_dataset_refused(self):
        for path in self.source.glob("config_*/c/complement/*RulesMetricsResult.csv"):
            path.unlink()
        with self.assertRaisesRegex(ValueError, "complete single-run"):
            load_single_panel(self.source)

    def test_extra_execution_refused(self):
        folder = self.source / "config_0/a/complement"
        for suffix in ["RulesMetricsResult", "Info"]:
            (folder / f"a_1_complement_{suffix}.csv").write_bytes(
                (folder / f"a_0_complement_{suffix}.csv").read_bytes())
        with self.assertRaisesRegex(ValueError, "complete single-run"):
            load_single_panel(self.source)

    def test_modified_output_refused_even_outside_top_ten(self):
        (self.source / "config_0/a/complement/command.sh").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "completion checksums"):
            load_single_panel(self.source)

    def test_single_cli_without_repetition_plan(self):
        command = [sys.executable, str(ROOT / "experiments/analyze_repeated_search.py"),
                   "--mode", "single", "--input", str(self.source)]
        lock_path = self.source / ".random_search.lock"
        lock_path.mkdir()
        result = subprocess.run(command, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Experiment locked", result.stderr)
        lock_path.rmdir()
        result = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = self.source / "analysis"
        protocol = json.loads((output / "analysis_protocol.json").read_text())
        self.assertEqual(protocol["mode"], "single")
        self.assertIsNone(protocol["plan_sha256"])
        self.assertIn("Modo: `single`", (output / "relatorio.md").read_text())
        self.assertTrue(pd.read_csv(output / "friedman.csv").repetitions.eq(1).all())
        self.assertFalse(lock_path.exists())


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
                                 "--input", str(self.output)],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.output / "analysis/relatorio.md").exists())
        protocol = json.loads((self.output / "analysis/analysis_protocol.json").read_text())
        self.assertEqual(protocol["mode"], "repeated")
        tests = pd.read_csv(self.output / "analysis/friedman.csv")
        self.assertTrue(tests.p_value.eq(1).all())
        self.assertFalse(tests.reject_global.any())

    def test_failure_never_marks_complete(self):
        with patch("experiments.repeat_random_search.subprocess.run", return_value=subprocess.CompletedProcess([], 7)):
            with self.assertRaisesRegex(RuntimeError, "MEASE exited 7"):
                run_plan(self.plan, self.output)
        self.assertEqual(len(list(self.output.rglob(".complete.json"))), 0)

    def test_duplicate_jobs_refused(self):
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
