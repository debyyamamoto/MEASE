from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import stats

from experiments.factorial_design import factorial_analysis as analysis


def response_rows(groups, metric="score"):
    return pd.DataFrame([
        {"config": str(level), "factor_level": level, "metric": metric,
         "execution": run, "y_raw": value, "y": value, "response_transform": "identity"}
        for level, values in enumerate(groups)
        for run, value in enumerate(values)
    ])


class OneFactorAnovaTests(unittest.TestCase):
    def test_matches_scipy_for_balanced_and_unbalanced_samples(self):
        for groups in (
            [[1, 2, 4], [2, 5, 6], [7, 8, 10]],
            [[1, 2], [2, 5, 6, 9], [7, 8, 10]],
            [[1], [2, 5, 6, 9], [7, 8]],
        ):
            with self.subTest(groups=groups):
                info, treatments = analysis.build_one_factor_summary(response_rows(groups), 0.05)
                result = info["score"]
                reference = stats.f_oneway(*groups)
                self.assertAlmostEqual(result["F_calc"], reference.statistic)
                self.assertAlmostEqual(result["p_value"], reference.pvalue)
                self.assertAlmostEqual(result["SSA"] + result["SSE"], result["SST"])
                self.assertAlmostEqual(result["mean"], np.concatenate(groups).mean())
                self.assertEqual(result["df_error"], sum(map(len, groups)) - len(groups))
                self.assertEqual(result["significant"], result["F_calc"] > result["F_critical"])
                np.testing.assert_allclose(treatments["y_j_mean"], [np.mean(g) for g in groups])

    def test_hand_computed_sums_of_squares(self):
        info, treatments = analysis.build_one_factor_summary(response_rows([[1, 3], [5, 7]]), 0.05)
        result = info["score"]
        for name, expected in {"mean": 4, "SSA": 16, "SSE": 4, "SST": 20, "MSE": 2, "F_calc": 8,
                               "Explained Var": 0.8}.items():
            self.assertAlmostEqual(result[name], expected)
        np.testing.assert_allclose(treatments["alpha_j"], [-2, 2])

    def test_centered_sums_are_stable_with_large_offset(self):
        groups = [[1, 3], [5, 7]]
        shifted = [[1e12 + value for value in group] for group in groups]
        info, _ = analysis.build_one_factor_summary(response_rows(shifted), 0.05)
        self.assertEqual(info["score"]["SST"], 20)
        self.assertEqual(info["score"]["F_calc"], 8)

    def test_zero_residual_variance_with_different_means_is_significant(self):
        for groups in ([[1, 1], [2, 2]], [[0.1] * 30, [0.3] * 30, [0.7] * 30]):
            with self.subTest(groups=groups):
                info, _ = analysis.build_one_factor_summary(response_rows(groups), 0.05)
                self.assertEqual(info["score"]["F_calc"], np.inf)
                self.assertEqual(info["score"]["p_value"], 0)
                self.assertTrue(info["score"]["significant"])

    def test_identical_values_have_undefined_test_and_are_not_significant(self):
        info, _ = analysis.build_one_factor_summary(response_rows([[1, 1], [1, 1]]), 0.05)
        self.assertTrue(np.isnan(info["score"]["F_calc"]))
        self.assertTrue(np.isnan(info["score"]["p_value"]))
        self.assertTrue(np.isnan(info["score"]["Explained Var"]))
        self.assertFalse(info["score"]["significant"])

    def test_insufficient_or_nonfinite_data_and_invalid_alpha_are_rejected(self):
        for groups in ([[1, 2]], [[1], [2]], [[1, np.nan], [3, 4]], [[1, np.inf], [3, 4]]):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                analysis.build_one_factor_summary(response_rows(groups), 0.05)
        for alpha in (0, 1, -0.05):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                analysis.build_one_factor_summary(response_rows([[1, 2], [3, 4]]), alpha)

    def test_only_anova_is_used(self):
        rows = response_rows([[1, 2, 3], [3.5, 4.5, 5.5]])
        with (
            patch.object(analysis.stats, "shapiro", side_effect=AssertionError("Unexpected Shapiro test")),
            patch.object(analysis.stats, "levene", side_effect=AssertionError("Unexpected Levene test")),
            patch.object(analysis.stats.t, "ppf", side_effect=AssertionError("Unexpected t interval")),
        ):
            info, treatments = analysis.build_one_factor_summary(rows, 0.05)
        self.assertTrue(info["score"]["significant"])
        self.assertNotIn("significant_holm", info["score"])
        self.assertNotIn("ci_low", treatments)


class OneFactorPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.design = pd.DataFrame({
            "experiment": ["population", "population"], "factor": ["low", "high"],
            "population": [10, 100], "dataset": ["toy|toy.csv|time|event|"] * 2,
            "response": ["all", "all"],
        })
        for config, shift in (("low", 0), ("high", 10)):
            directory = self.root / "population" / "all" / config / "toy" / "complement"
            directory.mkdir(parents=True)
            for run in range(3):
                pd.DataFrame({
                    "unchanged": [run + 1], "last_metric": [run + shift + 1],
                    "excluded_1": [100 * shift], "excluded_2": [100 * shift], "excluded_3": [100 * shift],
                }).to_csv(
                    directory / f"toy_{run}_complement_RulesMetricsResult.csv", index=False
                )
            # Use the aggregate's metric order, but calculate means from the runs.
            pd.DataFrame({
                "metric": ["unchanged", "last_metric", "excluded_1", "excluded_2", "excluded_3"],
                "mean": [999] * 5,
            }).to_csv(
                directory / "toy_complement_RulesMetricsResult.csv", index=False
            )

    def collect(self, transforms=None):
        return analysis.collect_one_factor_rows(self.design, self.root, "complement", transforms)

    def test_last_three_metrics_are_excluded_and_means_come_from_runs(self):
        rows = self.collect()
        self.assertEqual(len(rows), 12)
        self.assertEqual(set(rows["metric"]), {"unchanged", "last_metric"})
        self.assertEqual(rows["y"].max(), 13)
        # Reordering raw columns must not change the original aggregate-based selection.
        for path in self.root.rglob("toy_?_complement_RulesMetricsResult.csv"):
            frame = pd.read_csv(path)
            frame[frame.columns[::-1]].to_csv(path, index=False)
        pd.testing.assert_frame_equal(rows, self.collect())

    def test_metric_specific_transform_overrides_all(self):
        rows = self.collect({"all": "log", "unchanged": "identity"})
        changed = rows[rows["metric"] == "last_metric"]
        np.testing.assert_allclose(changed["y"], np.log(changed["y_raw"]))
        unchanged = rows[rows["metric"] == "unchanged"]
        np.testing.assert_allclose(unchanged["y"], unchanged["y_raw"])

    def test_missing_treatment_and_inconsistent_schema_fail_explicitly(self):
        self.design.loc[0, "factor"] = "missing"
        with self.assertRaises(FileNotFoundError):
            self.collect()
        self.design.loc[0, "factor"] = "low"
        path = next(self.root.rglob("toy_0_complement_RulesMetricsResult.csv"))
        pd.DataFrame({"unchanged": [1]}).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "Inconsistent metric"):
            self.collect()

    def test_duplicate_levels_or_multiple_datasets_are_rejected(self):
        self.design.loc[1, "population"] = 10
        with self.assertRaises(ValueError):
            self.collect()
        self.design.loc[1, "population"] = 100
        self.design.loc[1, "dataset"] = "other|other.csv|time|event|"
        with self.assertRaisesRegex(ValueError, "one dataset"):
            self.collect()

    def test_pipeline_saves_summary_and_plots_only_significant_metric(self):
        design_path = self.root / "design.csv"
        self.design.to_csv(design_path, index=False)
        output = self.root / "out"
        original_subplots = analysis.plt.subplots
        with patch.object(analysis.plt, "subplots", wraps=original_subplots) as subplots:
            analysis.analyze_design(design_path, self.root, output, "complement", 0.05, None)
        self.assertEqual(subplots.call_args.args[:2], (1, 1))
        self.assertEqual([path.name for path in output.iterdir()], ["sensitivity_analysis"])
        output = output / "sensitivity_analysis"
        summary = pd.read_csv(output / "population_all_anova.csv").set_index("metric")
        self.assertTrue(summary.loc["last_metric", "significant"])
        self.assertFalse(summary.loc["unchanged", "significant"])
        self.assertEqual(set(summary.index), {"unchanged", "last_metric"})
        self.assertEqual(set(summary.columns), {
            "factor", "df_factor", "df_error", "F_calc", "F_critical", "p_value", "alpha_sig",
            "significant", "Explained Var", "response_transform",
        })
        self.assertEqual(set(path.name for path in output.iterdir()), {
            "population_all_anova.csv", "population_all_significant_impact.png",
        })
        self.assertGreater((output / "population_all_significant_impact.png").stat().st_size, 100)

    def test_no_significant_metrics_removes_previous_plot(self):
        rows = response_rows([[1, 2, 3], [1, 2, 3]])
        info, summary = analysis.build_one_factor_summary(rows, 0.05)
        prefix = self.root / "previous"
        stale = Path(f"{prefix}_significant_impact.png")
        stale.write_bytes(b"old plot")
        self.assertEqual(analysis.save_one_factor_impact_plot(summary, info, "factor", prefix), [])
        self.assertFalse(stale.exists())

    def test_categorical_pipeline_reports_significance_and_variance_without_plot(self):
        self.design = self.design.rename(columns={"population": "score_metric"})
        self.design["score_metric"] = ["fast_logrank", "km_abc"]
        design_path = self.root / "categorical.csv"
        self.design.to_csv(design_path, index=False)
        output = self.root / "out"
        sensitivity_dir = output / "sensitivity_analysis"
        sensitivity_dir.mkdir(parents=True)
        stale = sensitivity_dir / "population_all_significant_impact.png"
        stale.write_bytes(b"old categorical plot")
        with patch.object(analysis.plt, "subplots", side_effect=AssertionError("Unexpected categorical plot")):
            analysis.analyze_design(design_path, self.root, output, "complement", 0.05, None)
        summary = pd.read_csv(sensitivity_dir / "population_all_anova.csv").set_index("metric")
        self.assertTrue((summary["factor"] == "score_metric").all())
        self.assertTrue(summary.loc["last_metric", "significant"])
        self.assertFalse(summary.loc["unchanged", "significant"])
        # Group means are 2 and 12: SSA=150, SSE=4, SST=154.
        self.assertAlmostEqual(summary.loc["last_metric", "Explained Var"], 150 / 154)
        self.assertEqual(summary.loc["unchanged", "Explained Var"], 0)
        self.assertAlmostEqual(summary.loc["last_metric", "F_calc"], 150)
        self.assertEqual([path.name for path in sensitivity_dir.iterdir()], ["population_all_anova.csv"])


class ResponseTransformTests(unittest.TestCase):
    def test_supported_power_transforms_match_scipy(self):
        values = pd.Series([1.0, 2.0, 4.0, 8.0])
        for name, reference in (("box_cox", stats.boxcox), ("yeo_johnson", stats.yeojohnson)):
            with self.subTest(transform=name):
                actual, metadata = analysis.apply_response_transform(values, name)
                expected, expected_lambda = reference(values)
                np.testing.assert_allclose(actual, expected)
                self.assertAlmostEqual(metadata["transform_lambda"], expected_lambda)

    def test_invalid_domain_is_rejected(self):
        for transform, values in (("log", [0, 1]), ("sqrt", [-1, 2]), ("identity", [1, np.nan])):
            with self.subTest(transform=transform), self.assertRaises(ValueError):
                analysis.apply_response_transform(pd.Series(values), transform)


if __name__ == "__main__":
    unittest.main()
