from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from easd.runner import _AggregateState, _save_aggregate_outputs, _save_run_outputs


class RunnerAggregationTests(unittest.TestCase):
    def aggregate(self, scores, debug_performance=False):
        with tempfile.TemporaryDirectory() as folder:
            output_dir = Path(folder)
            state = _AggregateState()
            metrics = []
            for run, run_scores in enumerate(scores):
                run_metrics = {"mean_rule_score": float(np.mean(run_scores)) if run_scores else 0.0}
                metrics.append(run_metrics)
                info = pd.DataFrame({
                    name: [2.0 * run]
                    for name in (
                        "cpu_mean_percent", "cpu_peak_percent", "ram_mean_mb",
                        "ram_peak_mb", "ram_incremental_peak_mb", "ram_baseline_mb",
                    )
                })
                _save_run_outputs(
                    dataset_name="toy", baseline="complement", run=run,
                    detailed_rules=pd.DataFrame({"Rule_Score": run_scores}),
                    runtime=2.0 * run, mean_rule_size=2.0 * run,
                    info=info, metrics=run_metrics, output_dir=output_dir,
                    aggregate_state=state, debug_performance=debug_performance,
                )

            with patch("easd.metrics.autorank", side_effect=ValueError("use basic statistics")):
                stats_path, metrics_path = _save_aggregate_outputs(
                    dataset_name="toy", baseline="complement", metrics_list=metrics,
                    output_dir=output_dir, aggregate_state=state,
                    debug_performance=debug_performance,
                )
            self.assertTrue(metrics_path.is_file())
            self.assertEqual(len(list(output_dir.glob("toy_*_complement_DetailedRules.csv"))), len(scores))
            return pd.read_csv(stats_path).iloc[0]

    def test_thirty_runs_with_different_rule_counts_have_equal_weight(self):
        # Per-run means alternate between 2 and 6; pooling rules would give 5.
        result = self.aggregate([[2.0], [4.0, 6.0, 8.0]] * 15)
        self.assertEqual(result["mean_score"], "4.0+/-2.0")

    def test_equal_rule_counts_measure_variation_between_runs(self):
        result = self.aggregate([[1.0, 3.0], [5.0, 7.0]])
        self.assertEqual(result["mean_score"], "4.0+/-2.0")

    def test_runs_without_rules_use_zero_like_run_metrics(self):
        for scores, expected in [([[], [4.0, 8.0]], "3.0+/-3.0"), ([[], []], "0.0+/-0.0")]:
            with self.subTest(scores=scores):
                self.assertEqual(self.aggregate(scores)["mean_score"], expected)

    def test_scalar_statistics_include_variation_between_runs(self):
        result = self.aggregate([[1.0], [1.0]], debug_performance=True)
        for name in (
            "mean_runtime", "mean_rule_size", "cpu_mean_percent", "cpu_peak_percent",
            "ram_mean_mb", "ram_peak_mb", "ram_incremental_peak_mb", "ram_baseline_mb",
        ):
            with self.subTest(metric=name):
                self.assertEqual(result[name], "1.0+/-1.0")


if __name__ == "__main__":
    unittest.main()
