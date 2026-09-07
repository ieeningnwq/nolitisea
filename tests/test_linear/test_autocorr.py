"""Tests for ``nolitisea.linear.autocorr``."""

import unittest

import numpy as np

from nolitisea.linear.autocorr import autocorrelation


def _brute_autocorr(x, max_lag=None, norm=True, detrend=True):
    """Direct O(N^2) reference implementing the documented contract.

    ``r[k] = sum_n x[n] * x[n + k]`` (unnormalised),
    optionally divided by ``r[0]``, for lags ``0 .. max_lag`` with
    ``max_lag`` clipped to ``N - 1``.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if max_lag is None:
        max_lag = n - 1
    else:
        max_lag = min(n - 1, max_lag)
    if detrend:
        x = x - x.mean()
    r = np.array([np.sum(x[: n - k] * x[k:]) for k in range(max_lag + 1)])
    return r / r[0] if norm else r


class TestNormalizedPeak(unittest.TestCase):
    """With ``norm=True`` the lag-0 value is exactly 1."""

    def test_peak_is_one_detrended(self):
        rng = np.random.default_rng(0)
        r = autocorrelation(rng.standard_normal(500), max_lag=10)
        self.assertEqual(r[0], 1.0)

    def test_peak_is_one_with_offset(self):
        """The peak stays 1 even for data far from zero (mean removed)."""
        rng = np.random.default_rng(1)
        r = autocorrelation(rng.standard_normal(500) + 123.0, max_lag=10)
        self.assertEqual(r[0], 1.0)


class TestBruteForceAgreement(unittest.TestCase):
    """FFT result matches the direct O(N^2) definition."""

    def test_norm_true_detrend_true(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(1000) * 3.0 + 7.0
        r = autocorrelation(x, max_lag=50)
        np.testing.assert_allclose(r, _brute_autocorr(x, 50), rtol=1e-10)

    def test_norm_false(self):
        """Unnormalised output returns raw lagged sums."""
        rng = np.random.default_rng(3)
        x = rng.standard_normal(400)
        r = autocorrelation(x, max_lag=20, norm=False)
        np.testing.assert_allclose(r, _brute_autocorr(x, 20, norm=False),
                                   rtol=1e-10)

    def test_detrend_false(self):
        """``detrend=False`` keeps the mean^2 pedestal in every lag."""
        rng = np.random.default_rng(4)
        x = rng.standard_normal(600) + 5.0
        r = autocorrelation(x, max_lag=30, detrend=False)
        np.testing.assert_allclose(
            r, _brute_autocorr(x, 30, detrend=False), rtol=1e-10)

    def test_no_circular_wraparound(self):
        """The zero padding must suppress periodic aliases: an odd-length
        ramp has strongly lag-dependent overlap that a circular
        correlation would get wrong."""
        x = np.arange(101.0)
        r = autocorrelation(x, norm=False)
        np.testing.assert_allclose(r, _brute_autocorr(x, norm=False),
                                   rtol=1e-10)


class TestHandComputed(unittest.TestCase):
    """Exact small cases worked out by hand."""

    def test_alternating_series(self):
        """x = [1, -1, 1, -1] (zero mean): r = [4, -3, 2, -1] unnormalised,
        [1, -0.75, 0.5, -0.25] normalised."""
        x = [1.0, -1.0, 1.0, -1.0]
        r = autocorrelation(x, norm=False)
        np.testing.assert_allclose(r, [4.0, -3.0, 2.0, -1.0], atol=1e-12)
        r = autocorrelation(x)
        np.testing.assert_allclose(r, [1.0, -0.75, 0.5, -0.25], atol=1e-12)

    def test_alternating_series_odd_length(self):
        """x = [1, -1, 1]: demeaned to [1, -1, 1] - 1/3, checked against
        the brute reference (mean no longer exactly zero)."""
        x = [1.0, -1.0, 1.0]
        r = autocorrelation(x)
        np.testing.assert_allclose(r, _brute_autocorr(x), rtol=1e-12)

    def test_constant_series_unnormalised(self):
        """A constant series with detrend=True is all zeros; the
        unnormalised result is exactly zero."""
        r = autocorrelation(np.full(8, 3.5), max_lag=4, norm=False)
        np.testing.assert_array_equal(r, np.zeros(5))


class TestDetrendInvariance(unittest.TestCase):
    """``detrend=True`` makes the result invariant to constant shifts."""

    def test_shift_invariance(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal(800)
        r0 = autocorrelation(x, max_lag=30)
        r1 = autocorrelation(x + 1e6, max_lag=30)
        np.testing.assert_allclose(r1, r0, atol=1e-9)


class TestMaxLagConvention(unittest.TestCase):
    """The result covers lags ``0 .. max_lag`` (max_lag + 1 points)."""

    def test_explicit_max_lag_length(self):
        r = autocorrelation(np.arange(50.0), max_lag=7)
        self.assertEqual(len(r), 8)
        self.assertEqual(r[0], 1.0)

    def test_default_covers_all_lags(self):
        """Default ``max_lag = N - 1`` yields N points (lags 0..N-1)."""
        x = np.arange(37.0)
        self.assertEqual(len(autocorrelation(x)), 37)

    def test_max_lag_clipped_to_n_minus_one(self):
        x = np.arange(20.0)
        self.assertEqual(len(autocorrelation(x, max_lag=100)), 20)

    def test_zero_max_lag(self):
        r = autocorrelation([1.0, 2.0, 3.0], max_lag=0)
        np.testing.assert_allclose(r, [1.0], atol=1e-15)

    def test_last_point_matches_brute_force(self):
        """The requested endpoint lag max_lag is actually present."""
        rng = np.random.default_rng(6)
        x = rng.standard_normal(300)
        r = autocorrelation(x, max_lag=17)
        self.assertAlmostEqual(r[-1], _brute_autocorr(x, 17)[-1], places=12)


class TestWhiteNoise(unittest.TestCase):
    """IID noise has vanishing off-peak autocorrelation."""

    def test_off_peak_within_confidence_band(self):
        rng = np.random.default_rng(7)
        r = autocorrelation(rng.standard_normal(20000), max_lag=30)
        band = 4.0 / np.sqrt(20000)  # ~4 sigma
        self.assertTrue(np.all(np.abs(r[1:]) < band))


if __name__ == "__main__":
    unittest.main()
