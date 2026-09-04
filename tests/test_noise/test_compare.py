"""Tests for nolitisea.noise.compare (error metrics and similarity measures)."""

import unittest
import warnings

import numpy as np

from nolitisea.noise.compare import (
    _dtw,
    cosine_similarity,
    dtw_distance,
    mae,
    max_cross_correlation,
    mse,
    nrmse_ptp,
    pearson_corr,
    r2_score,
    rmse,
)


def _brute_dtw(x, y):
    """Independent transcription of the classic DTW recursion."""
    n, m = len(x), len(y)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = abs(x[i - 1] - y[j - 1]) + min(
                acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]
            )
    return acc[n, m]


class TestErrorMetrics(unittest.TestCase):
    def test_known_values(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(mae(x, y), 1.0)
        self.assertAlmostEqual(mse(x, y), 1.0)
        self.assertAlmostEqual(rmse(x, y), 1.0)
        self.assertAlmostEqual(nrmse_ptp(x, y), 1.0 / 3.0)

    def test_mixed_residuals(self):
        x = np.array([0.0, 0.0, 4.0])
        y = np.array([1.0, 0.0, 1.0])
        self.assertAlmostEqual(mae(x, y), 4.0 / 3.0)
        self.assertAlmostEqual(mse(x, y), 10.0 / 3.0)
        self.assertAlmostEqual(rmse(x, y), np.sqrt(10.0 / 3.0))

    def test_rmse_consistent_with_mse(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal(50)
        y = x + rng.standard_normal(50) * 0.1
        self.assertAlmostEqual(rmse(x, y), np.sqrt(mse(x, y)), places=15)

    def test_r2_score(self):
        x = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(r2_score(x, x), 1.0)
        # Predicting the mean gives exactly zero.
        self.assertAlmostEqual(r2_score(x, np.full(3, 2.0)), 0.0)
        # Worse than the mean gives a negative score.
        self.assertAlmostEqual(r2_score(x, np.array([3.0, 2.0, 1.0])), -3.0)

    def test_constant_reference_returns_nan(self):
        x = np.full(10, 2.5)
        y = np.arange(10.0)
        self.assertTrue(np.isnan(nrmse_ptp(x, y)))
        self.assertTrue(np.isnan(r2_score(x, y)))


class TestInputValidation(unittest.TestCase):
    ALL_FUNCS = (
        mae, mse, rmse, nrmse_ptp, r2_score,
        pearson_corr, max_cross_correlation, cosine_similarity, dtw_distance,
    )
    LENGTH_CHECKED = (
        mae, mse, rmse, nrmse_ptp, r2_score,
        pearson_corr, max_cross_correlation, cosine_similarity,
    )

    def test_2d_input_raises_everywhere(self):
        x2 = np.zeros((4, 2))
        y1 = np.zeros(8)
        for func in self.ALL_FUNCS:
            with self.assertRaises(ValueError, msg=func.__name__):
                func(x2, y1)
            with self.assertRaises(ValueError, msg=func.__name__):
                func(y1, x2)

    def test_length_mismatch_raises(self):
        x = np.zeros(5)
        y = np.zeros(6)
        for func in self.LENGTH_CHECKED:
            with self.assertRaises(ValueError, msg=func.__name__):
                func(x, y)

    def test_dtw_allows_unequal_lengths(self):
        self.assertIsInstance(dtw_distance(np.zeros(5), np.zeros(6)), float)

    def test_scalar_inputs_accepted(self):
        self.assertEqual(mae(1.0, 2.0), 1.0)


class TestPearsonAndSimilarity(unittest.TestCase):
    def test_perfect_correlation(self):
        x = np.arange(10.0)
        r, p = pearson_corr(x, x)
        self.assertAlmostEqual(r, 1.0, places=12)
        self.assertAlmostEqual(p, 0.0, places=12)
        r, _ = pearson_corr(x, -x)
        self.assertAlmostEqual(r, -1.0, places=12)

    def test_pearson_matches_numpy(self):
        rng = np.random.default_rng(11)
        x = rng.standard_normal(100)
        y = 0.3 * x + rng.standard_normal(100)
        r, _ = pearson_corr(x, y)
        self.assertAlmostEqual(r, np.corrcoef(x, y)[0, 1], places=12)

    def test_pearson_constant_input_returns_nan(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, _ = pearson_corr(np.ones(5), np.arange(5.0))
        self.assertTrue(np.isnan(r))

    def test_cosine_similarity(self):
        x = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(cosine_similarity(x, x), 1.0)
        self.assertAlmostEqual(cosine_similarity(x, -x), -1.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_cosine_zero_vector_returns_nan(self):
        self.assertTrue(np.isnan(cosine_similarity(np.zeros(4), np.ones(4))))


class TestMaxCrossCorrelation(unittest.TestCase):
    def test_periodic_shift_recovers_lag(self):
        # Zero pad the edges so the shifted copy loses no signal
        # energy: only then does the peak reach exactly 1.0.  The sine
        # spans an integer number of periods so its mean is exactly 0.
        t = np.arange(100.0)
        x = np.zeros(100)
        x[10:90] = np.sin(2.0 * np.pi * t[10:90] / 20.0)
        d = 10
        y = np.roll(x, d)  # y[n] = x[n - d]
        max_r, lag = max_cross_correlation(x, y)
        self.assertAlmostEqual(max_r, 1.0, places=12)
        self.assertEqual(lag, -d)

    def test_anticorrelation_gives_negative_peak(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(100)
        max_r, lag = max_cross_correlation(x, -x)
        self.assertAlmostEqual(max_r, -1.0, places=12)
        self.assertEqual(lag, 0)

    def test_max_lag_restricts_search_window(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal(200)
        y = np.roll(x, 40)  # true peak at lag -40
        _, global_lag = max_cross_correlation(x, y)
        self.assertEqual(global_lag, -40)
        max_r, lag = max_cross_correlation(x, y, max_lag=5)
        self.assertLessEqual(abs(lag), 5)
        self.assertLess(abs(max_r), 0.999)

    def test_constant_series_returns_nan(self):
        max_r, lag = max_cross_correlation(np.ones(10), np.ones(10))
        self.assertTrue(np.isnan(max_r))
        self.assertEqual(lag, 0)


class TestDtw(unittest.TestCase):
    def test_identical_series_zero_distance(self):
        x = np.sin(np.arange(50.0))
        self.assertEqual(dtw_distance(x, x), 0.0)

    def test_known_small_values(self):
        self.assertEqual(dtw_distance([0.0, 0.0], [1.0, 1.0]), 2.0)
        self.assertEqual(dtw_distance([0.0, 2.0], [1.0, 1.0]), 2.0)
        # Unequal lengths: every x element matched against the single y
        # element.
        self.assertEqual(dtw_distance([0.0, 1.0, 2.0], [1.0]), 2.0)

    def test_appended_duplicate_costs_nothing(self):
        # A duplicated final sample can be matched twice against the
        # last x element, so the warping absorbs it exactly.
        x = np.sin(np.arange(30.0))
        self.assertEqual(dtw_distance(x, np.append(x, x[-1])), 0.0)

    def test_matches_brute_force(self):
        rng = np.random.default_rng(21)
        for n, m in ((1, 1), (5, 5), (8, 3), (3, 8), (20, 20)):
            x = np.round(rng.standard_normal(n), 2)
            y = np.round(rng.standard_normal(m), 2)
            self.assertAlmostEqual(dtw_distance(x, y), _brute_dtw(x, y))

    def test_path_is_valid_and_costs_distance(self):
        rng = np.random.default_rng(13)
        x = rng.standard_normal(12)
        y = rng.standard_normal(9)
        dist, path = _dtw(x, y)
        # Monotone path from (0, 0) to (n - 1, m - 1) with unit steps.
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (x.size - 1, y.size - 1))
        for (i0, j0), (i1, j1) in zip(path[:-1], path[1:]):
            self.assertIn((i1 - i0, j1 - j0), ((0, 1), (1, 0), (1, 1)))
        # The distance equals the accumulated cost along the path.
        cost = sum(abs(x[i] - y[j]) for i, j in path)
        self.assertAlmostEqual(dist, cost, places=12)


if __name__ == "__main__":
    unittest.main()
