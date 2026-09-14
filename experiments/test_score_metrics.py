from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import main
from easd.dataset import Dataset
from easd.evaluation import SCORE_METRICS, RuleEvaluator


class ScoreMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        n_subgroup = 40
        n_population = 160

        subgroup_feature = rng.uniform(0.10, 0.20, size=n_subgroup)
        population_feature = rng.uniform(0.65, 0.95, size=n_population)
        feature = np.r_[subgroup_feature, population_feature]
        noise = rng.uniform(0.0, 1.0, size=n_subgroup + n_population)

        subgroup_time = rng.weibull(1.5, size=n_subgroup) * 1.0
        population_time = rng.weibull(1.5, size=n_population) * 5.0
        time = np.maximum(np.r_[subgroup_time, population_time], np.finfo(float).tiny)
        event = np.ones(n_subgroup + n_population, dtype=int)

        self.df = pd.DataFrame(
            {
                "feature_0": feature,
                "feature_1": noise,
                "time": time,
                "event": event,
            }
        )
        self.dataset = Dataset(self.df.copy(), "time", "event")
        self.rule = [[0], [[0.0, 0.3]]]

    def test_all_score_metrics_return_finite_positive_fitness(self) -> None:
        for score_metric in SCORE_METRICS:
            with self.subTest(score_metric=score_metric):
                evaluator = RuleEvaluator(
                    self.dataset,
                    "complement",
                    alpha=0.5,
                    score_metric=score_metric,
                    km_time_bins=64,
                )

                fitness = evaluator.fitness(self.rule, self.dataset.data)

                self.assertTrue(np.isfinite(fitness))
                self.assertGreater(fitness, 0.0)

    def test_support_filter_runs_before_score_metric(self) -> None:
        evaluator = RuleEvaluator(
            self.dataset,
            "complement",
            alpha=0.5,
            score_metric="km_cvm",
            km_time_bins=64,
        )
        tiny_rule = [[0], [[0.10, 0.101]]]

        self.assertEqual(evaluator.fitness(tiny_rule, self.dataset.data), 0.0)

    def test_get_fitness_keeps_scores_aligned_with_sorted_population(self) -> None:
        evaluator = RuleEvaluator(
            self.dataset,
            "complement",
            alpha=0.5,
            score_metric="km_cvm",
            km_time_bins=64,
        )
        invalid_rule = [[0], [[0.0, 1.0]]]

        scores, population = evaluator.get_fitness([invalid_rule, self.rule], self.dataset.data)

        self.assertIs(population[0], self.rule)
        self.assertGreater(scores[0], scores[1])
        self.assertEqual(scores[0], evaluator.fitness(population[0], self.dataset.data))
        self.assertEqual(scores[1], evaluator.fitness(population[1], self.dataset.data))

    def test_redundancy_penalty_is_bounded_and_penalizes_identical_curve(self) -> None:
        evaluator = RuleEvaluator(
            self.dataset,
            "complement",
            alpha=0.5,
            score_metric="km_cvm",
            km_time_bins=64,
            redundancy_penalty=1.0,
            redundancy_similarity_threshold=0.9,
        )
        distinct_rule = [[0], [[0.70, 0.80]]]
        population = [self.rule, distinct_rule]
        raw_scores = [evaluator.fitness(rule, self.dataset.data) for rule in population]
        top_mask = evaluator.get_covered_mask(self.rule, self.dataset.data)
        equivalent_rule = [[0, 1], [[0.0, 0.3], [0.0, 1.0]]]

        penalized = evaluator.penalize_score(
            raw_scores,
            [(raw_scores[0], equivalent_rule, top_mask)],
            population,
            self.dataset.data,
            candidate_count=2,
        )

        self.assertEqual(penalized[0], 0.0)
        self.assertGreater(penalized[1], 0.0)
        self.assertTrue(all(0.0 <= score <= raw for score, raw in zip(penalized, raw_scores)))

    def test_replacement_is_compared_to_other_archive_entries(self) -> None:
        evaluator = RuleEvaluator(
            self.dataset, "complement", alpha=0.1, score_metric="km_abc",
            redundancy_penalty=1.0,
        )
        mask = evaluator.get_covered_mask(self.rule, self.dataset.data)
        equivalent = [[0, 1], [[0.0, 0.3], [0.0, 1.0]]]
        archive = [(0.2, self.rule, mask), (0.2, equivalent, mask)]
        self.assertEqual(evaluator.penalize_score([0.2], archive, [self.rule], self.dataset.data), [0.0])
        self.assertEqual(evaluator.penalize_score([0.2], archive[:1], [self.rule], self.dataset.data), [0.2])

    def test_penalty_applies_beyond_archive_size_and_is_scale_invariant(self) -> None:
        evaluator = RuleEvaluator(
            self.dataset, "complement", alpha=0.1, score_metric="fast_logrank",
            redundancy_penalty=0.5,
        )
        equivalent = [[0, 1], [[0.0, 0.3], [0.0, 1.0]]]
        mask = evaluator.get_covered_mask(equivalent, self.dataset.data)
        archive = [(0.2, equivalent, mask)]
        result = evaluator.penalize_score([0.1, 0.9], archive, [self.rule, self.rule], self.dataset.data)
        np.testing.assert_allclose(result, [0.05, 0.45])
        self.assertEqual(evaluator.penalize_score([], archive, [], self.dataset.data), [])

    def test_cli_accepts_score_metric_arguments(self) -> None:
        args = main.build_parser().parse_args(
            [
                "datasets/files/cancer.parquet",
                "-time",
                "time",
                "-event",
                "status",
                "--score_metric",
                "mdir4",
                "--km_time_bins",
                "128",
                "--redundancy_penalty",
                "0.75",
                "--redundancy_similarity_threshold",
                "0.85",
            ]
        )
        config = main.config_from_args(args)

        self.assertEqual(config.score_metric, "mdir4")
        self.assertEqual(config.km_time_bins, 128)
        self.assertEqual(config.redundancy_penalty, 0.75)
        self.assertEqual(config.redundancy_similarity_threshold, 0.85)


if __name__ == "__main__":
    unittest.main()
