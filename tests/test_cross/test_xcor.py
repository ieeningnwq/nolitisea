"""Tests for the TISEAN ``xcor`` rewrite in nolitisea.cross.xcor."""

import unittest

import numpy as np

from nolitisea.cross.xcor import cross_correlation


def _c_transcription(a, b, max_lag):
    """Literal transcription of xcor.c on raw arrays."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    n = x.size
    if max_lag >= n:
        max_lag = n - 1

    av1 = np.sum(x) / n
    av2 = np.sum(y) / n
    var1 = np.sqrt(abs(np.sum(x * x) / n - av1 * av1))
    var2 = np.sqrt(abs(np.sum(y * y) / n - av2 * av2))

    arr1 = x - av1
    arr2 = y - av2
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.empty(lags.size)
    for k, lag in enumerate(lags):
        count = 0
        c = 0.0
        for j in range(n):
            hi = j + int(lag)
            if 0 <= hi < n:
                count += 1
                c += arr1[j] * arr2[hi]
        corr[k] = c / count / var1 / var2
    return lags, corr, av1, var1, av2, var2


def _demo_series(n=300, seed=3):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 8.0 * np.pi, n)
    a = np.sin(t) + 0.3 * np.sin(2.3 * t) + 0.01 * rng.standard_normal(n)
    b = np.sin(t + 0.4) + 0.3 * np.cos(2.1 * t) + 0.01 * rng.standard_normal(n)
    return a, b


class TestCrossCorrelation(unittest.TestCase):
    def test_matches_c_transcription(self):
        a, b = _demo_series()
        lags, corr, av1, var1, av2, var2 = _c_transcription(a, b, 40)
        result = cross_correlation(a, b, max_lag=40)
        np.testing.assert_allclose(result["lags"], lags)
        np.testing.assert_allclose(result["corr"], corr, atol=1e-12)
        self.assertAlmostEqual(result["mean_a"], av1, places=12)
        self.assertAlmostEqual(result["std_a"], var1, places=12)
        self.assertAlmostEqual(result["mean_b"], av2, places=12)
        self.assertAlmostEqual(result["std_b"], var2, places=12)

    def test_zero_lag_of_self_is_one(self):
        a, _ = _demo_series()
        result = cross_correlation(a, a, max_lag=5)
        # (1/n) sum a'^2 divided by the biased rms deviation of a: exactly 1
        self.assertAlmostEqual(float(result["corr"][5]), 1.0, places=12)

    def test_shift_invariance(self):
        a, b = _demo_series()
        base = cross_correlation(a, b, max_lag=10)["corr"]
        shifted = cross_correlation(a + 1000.0, b - 250.0, max_lag=10)["corr"]
        np.testing.assert_allclose(shifted, base, atol=1e-12)

    def test_max_lag_clamped_to_length_minus_one(self):
        a, b = _demo_series(n=50)
        result = cross_correlation(a, b, max_lag=10_000)
        self.assertEqual(result["lags"].size, 2 * (50 - 1) + 1)
        self.assertEqual(int(result["lags"][0]), -(50 - 1))
        self.assertEqual(int(result["lags"][-1]), 50 - 1)

    def test_length_mismatch_raises(self):
        a, b = _demo_series(n=40)
        with self.assertRaises(ValueError):
            cross_correlation(a, b[:20])

    def test_negative_max_lag_raises(self):
        a, b = _demo_series(n=40)
        with self.assertRaises(ValueError):
            cross_correlation(a, b, max_lag=-1)

    def test_constant_series_raises(self):
        a, _ = _demo_series(n=40)
        with self.assertRaises(ValueError):
            cross_correlation(a, np.ones(40))

if __name__ == "__main__":
    unittest.main()
