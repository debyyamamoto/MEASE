import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.factorial_design import investigate_transforms as investigation


def rows(groups):
    return pd.DataFrame([
        dict(experiment="example", metric="score", config=str(j), y_raw=value)
        for j, group in enumerate(groups) for value in group
    ])


class TransformationInvestigationTests(unittest.TestCase):
    def test_uses_within_level_sample_variance(self):
        info, fitted = investigation.diagnose(rows([[1, 3], [10, 12, 14]]), "identity")
        self.assertEqual(info["variance_ratio"], 2)
        np.testing.assert_allclose(fitted.fitted, [2, 2, 12, 12, 12])
        np.testing.assert_allclose(fitted.residual, [-1, 1, -2, 0, 2])

    def test_refits_after_transforming_individual_observations(self):
        info, fitted = investigation.diagnose(rows([[1, 4], [4, 16]]), "log")
        np.testing.assert_allclose(fitted.fitted, [np.log(2)] * 2 + [np.log(8)] * 2)
        np.testing.assert_allclose(fitted.groupby("config").residual.sum(), 0, atol=1e-15)
        self.assertAlmostEqual(info["variance_ratio"], 1)

    def test_exact_constants_are_not_misclassified_as_homoscedastic(self):
        for transform in investigation.TRANSFORMS:
            with self.subTest(transform=transform):
                info, _ = investigation.diagnose(rows([[.1] * 30, [.3] * 30]), transform)
                self.assertEqual(info["status"], "all_constant")
                self.assertTrue(np.isnan(info["variance_ratio"]))
                info, _ = investigation.diagnose(rows([[.1] * 30, [.2, .4] * 15]), transform)
                self.assertEqual(info["status"], "some_constant")
                self.assertEqual(info["variance_ratio"], np.inf)

    def test_no_offsets_clipping_or_tests(self):
        with patch.object(investigation.stats, "shapiro", side_effect=AssertionError), \
             patch.object(investigation.stats, "levene", side_effect=AssertionError), \
             patch.object(investigation.stats, "f_oneway", side_effect=AssertionError):
            result = investigation.compare(rows([[0, .2], [.8, 1]])).set_index("transform")
        self.assertEqual(result.loc["log", "status"], "domain")
        self.assertEqual(result.loc["logit", "status"], "domain")
        self.assertEqual(result.loc["arcsin_sqrt", "status"], "variable")
        with self.assertRaises(ValueError):
            investigation.compare(rows([[1], [2, 3]]))
        with self.assertRaises(ValueError):
            investigation.compare(rows([[1, np.nan], [2, 3]]))


if __name__ == "__main__":
    unittest.main()
