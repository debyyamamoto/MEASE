import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from easd.metrics import compute_run_metrics, output_metrics
from easd.survival_curves import KaplanMeierGrid, weighted_rms_distance
from experiments.collect_results import compile_metric_runs


class SurvivalMetricsTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(dict(group=[0, 0, 1, 1], time=[1., 3., 2., 3.], event=[1, 0, 1, 0]))
        self.rules = [(["group"], [(0, 0)]), (["group"], [(1, 1)])]

    def metrics(self, df=None, rules=None, **kwargs):
        return compute_run_metrics(self.df if df is None else df, self.rules if rules is None else rules,
                                   "time", "event", **kwargs)

    def test_hand_calculated_time_and_risk_weighting(self):
        result = self.metrics(km_time_bins=0)
        # At t=1: difference .5, width 1, pooled risk 1.
        # At t=2: difference 0, width 1, pooled risk 3/4.
        self.assertAlmostEqual(result.mean_pairwise_rms, np.sqrt(.25 / 1.75))
        self.assertEqual(result.rms_valid_pairs, 1)
        changed = self.df.copy()
        changed.time *= 365
        self.assertAlmostEqual(self.metrics(changed).mean_pairwise_rms, result.mean_pairwise_rms)
        self.assertAlmostEqual(self.metrics(baseline="complement").mean_pairwise_rms, result.mean_pairwise_rms)
        shifted = self.df.set_axis([10, 20, 30, 40])
        self.assertAlmostEqual(self.metrics(shifted).mean_pairwise_rms, result.mean_pairwise_rms)

    def test_exact_grid_matches_independent_kaplan_meier(self):
        times = np.array([1., 2., 2., 3., 4., 6., 7.])
        events = np.array([1, 1, 0, 1, 0, 1, 0])
        grid = KaplanMeierGrid(times, events, time_bins=0)
        for mask in [np.ones(7, dtype=bool), np.array([True, False, True, True, False, True, True])]:
            km = KaplanMeierFitter().fit(times[mask], events[mask])
            survival, risk = grid.curve(mask)
            np.testing.assert_allclose(survival, km.predict(grid.grid_times))
            np.testing.assert_array_equal(risk, [(times[mask] >= t).sum() for t in grid.grid_times])

    def test_crossings_do_not_cancel_and_bounds(self):
        risk = np.ones(2)
        first, second = (np.array([.8, .2]), risk), (np.array([.4, .6]), risk)
        self.assertAlmostEqual(weighted_rms_distance(first, second, risk, risk), .4)
        self.assertEqual(weighted_rms_distance(first, first, risk, risk), 0.)
        self.assertEqual(weighted_rms_distance((risk, risk), (np.zeros(2), risk), risk, risk), 1.)
        self.assertEqual(weighted_rms_distance(first, second, risk, risk),
                         weighted_rms_distance(second, first, risk, risk))

    def test_full_subgroups_including_nested_and_duplicate_rules(self):
        nested = self.rules + [(["group"], [(0, 1)])]
        grid = KaplanMeierGrid(self.df.time, self.df.event)
        curves = [grid.curve(mask) for mask in [self.df.group == 0, self.df.group == 1, np.ones(4, dtype=bool)]]
        distances = [weighted_rms_distance(a, b, grid.widths, grid.weights)
                     for a, b in itertools.combinations(curves, 2)]
        result = self.metrics(rules=nested)
        self.assertEqual(result.rms_valid_pairs, 3)
        self.assertAlmostEqual(result.mean_pairwise_rms, np.mean(distances))
        self.assertNotAlmostEqual(result.mean_pairwise_rms, np.sqrt(np.mean(np.square(distances))))
        duplicate = self.metrics(rules=[self.rules[0], self.rules[0]])
        self.assertEqual(duplicate.mean_pairwise_rms, 0.)
        self.assertEqual(duplicate.rms_valid_pairs, 1)

    def test_undefined_pairs_are_not_maximum_distance(self):
        for rules in [[], self.rules[:1], [self.rules[0], (["group"], [(3, 4)])]]:
            result = self.metrics(rules=rules)
            self.assertTrue(np.isnan(result.mean_pairwise_rms))
            self.assertEqual(result.rms_valid_pairs, 0)
        for df in [self.df.assign(event=0), self.df.assign(time=1)]:
            result = self.metrics(df)
            self.assertTrue(np.isnan(result.mean_pairwise_rms))
            self.assertEqual(result.rms_valid_pairs, 0)
        partial = self.metrics(rules=self.rules + [(["group"], [(3, 4)])])
        self.assertEqual(partial.rms_valid_pairs, 1)
        self.assertEqual(partial.mean_pairwise_rms, self.metrics().mean_pairwise_rms)

    def test_aggregation_new_and_historical_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out = root / "toy/complement"
            out.mkdir(parents=True)
            metrics = self.metrics().as_dict()
            pd.DataFrame([metrics]).to_csv(out / "toy_0_complement_RulesMetricsResult.csv", index=False)
            historical = {k: v for k, v in metrics.items() if k not in ("mean_pairwise_rms", "rms_valid_pairs")}
            pd.DataFrame([historical]).to_csv(out / "toy_1_complement_RulesMetricsResult.csv", index=False)
            compiled = compile_metric_runs(root, ["toy"], 2, "complement", root / "compiled.csv")
            self.assertAlmostEqual(compiled.iloc[0].mean_pairwise_rms, metrics["mean_pairwise_rms"])
            self.assertTrue(np.isnan(compiled.iloc[1].mean_pairwise_rms))
            # Exercise the aggregate fallback also used with short execution batches.
            with patch("easd.metrics.autorank", side_effect=ValueError("small batch")):
                output_metrics([metrics, {**metrics, "mean_pairwise_rms": .8}], str(root / "summary.csv"))
            summary = pd.read_csv(root / "summary.csv", index_col=0)
            self.assertAlmostEqual(summary.loc["mean_pairwise_rms", "mean"],
                                   (metrics["mean_pairwise_rms"] + .8) / 2, places=4)

    def test_config9_matches_original_diagnostic(self):
        root = Path(__file__).resolve().parents[1]
        diagnostic = root / "experiments/config9_score_diagnostic"
        if not (diagnostic / "summary.csv").exists():
            self.skipTest("Local config9 diagnostic not available")
        from experiments.analyze_random_search import parse_config
        df = pd.read_parquet(root / "datasets/files/mgus2.parquet")
        for _, expected in pd.read_csv(diagnostic / "summary.csv").iterrows():
            detail = pd.read_csv(diagnostic / expected.score_metric / "mgus2/complement/mgus2_0_complement_DetailedRules.csv")
            rules = [parse_config("{'rule': " + text + "}")["rule"] for text in detail.Rule_Obj]
            result = compute_run_metrics(df, rules, "futime", "status", baseline="complement", km_time_bins=512)
            self.assertAlmostEqual(result.mean_pairwise_rms, expected.mean_pairwise_rms, places=12)
            self.assertEqual(result.rms_valid_pairs, 10)


if __name__ == "__main__":
    unittest.main()
