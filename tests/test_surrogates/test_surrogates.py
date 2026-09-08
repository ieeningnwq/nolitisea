"""Tests for ``nolitisea.surrogates.surrogates`` (ft / aaft / iaaft).

The generators use the legacy global ``np.random`` stream, so every
test seeds it explicitly for reproducibility.
"""

import unittest

import numpy as np

from nolitisea.surrogates.surrogates import aaft, ft, iaaft

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ar1(n=512, a=0.9, seed=1):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = a * x[i - 1] + rng.standard_normal()
    return x


def _logistic(n=512, r=3.7, x0=0.3):
    x = np.empty(n)
    x[0] = x0
    for i in range(n - 1):
        x[i + 1] = r * x[i] * (1.0 - x[i])
    return x


def _skew(v):
    v = v - np.mean(v)
    m2 = np.mean(v ** 2)
    m3 = np.mean(v ** 3)
    return m3 / m2 ** 1.5


def _ac1(v):
    v = v - np.mean(v)
    return float(np.dot(v[:-1], v[1:]) / np.dot(v, v))


# ---------------------------------------------------------------------------
# ft
# ---------------------------------------------------------------------------

class TestFt(unittest.TestCase):
    def test_power_spectrum_preserved(self):
        x = _ar1()
        np.random.seed(42)
        y = ft(x)
        amp_x = np.abs(np.fft.rfft(x))
        amp_y = np.abs(np.fft.rfft(y))
        np.testing.assert_allclose(amp_y, amp_x, rtol=1e-10, atol=1e-12)

    def test_power_spectrum_preserved_odd_length(self):
        x = _ar1(511)
        np.random.seed(42)
        y = ft(x)
        amp_x = np.abs(np.fft.rfft(x))
        amp_y = np.abs(np.fft.rfft(y))
        np.testing.assert_allclose(amp_y, amp_x, rtol=1e-10, atol=1e-12)

    def test_mean_preserved(self):
        # phi[0] = 0 keeps the DC bin (and hence the mean) unchanged.
        x = _ar1()
        np.random.seed(42)
        y = ft(x)
        self.assertAlmostEqual(np.mean(y), np.mean(x), places=12)

    def test_variance_preserved(self):
        # Parseval: preserving |rfft| preserves the variance exactly.
        x = _ar1()
        np.random.seed(42)
        y = ft(x)
        np.testing.assert_allclose(np.var(y), np.var(x), rtol=1e-10)

    def test_output_shape_real_finite(self):
        x = _ar1(300)
        np.random.seed(42)
        y = ft(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertTrue(np.allclose(y, y.real))

    def test_differs_from_original(self):
        x = _logistic()
        np.random.seed(42)
        y = ft(x)
        self.assertTrue(np.any(y != x))

    def test_distribution_not_preserved(self):
        # FT surrogates destroy the amplitude distribution: the skewness
        # of a skewed series is (almost surely) not retained.
        x = np.random.default_rng(3).exponential(1.0, 512)
        np.random.seed(42)
        y = ft(x)
        self.assertGreater(abs(_skew(y) - _skew(x)), 0.1)

    def test_deterministic_given_seed(self):
        x = _ar1()
        np.random.seed(7)
        y1 = ft(x)
        np.random.seed(7)
        y2 = ft(x)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seed_different_surrogate(self):
        x = _ar1()
        np.random.seed(7)
        y1 = ft(x)
        np.random.seed(8)
        y2 = ft(x)
        self.assertFalse(np.array_equal(y1, y2))


# ---------------------------------------------------------------------------
# aaft
# ---------------------------------------------------------------------------

class TestAaft(unittest.TestCase):
    def test_distribution_preserved_exactly(self):
        # Rank ordering maps the surrogate onto the sorted original
        # values, so the amplitude distribution is retained exactly.
        x = _logistic()  # clearly non-Gaussian
        np.random.seed(42)
        y = aaft(x)
        np.testing.assert_array_equal(np.sort(y), np.sort(x))

    def test_distribution_preserved_gaussian_input(self):
        x = _ar1()
        np.random.seed(42)
        y = aaft(x)
        np.testing.assert_array_equal(np.sort(y), np.sort(x))

    def test_correlation_crudely_preserved(self):
        # Rank ordering + FT phase randomisation keeps the lag-1
        # autocorrelation of a strongly correlated series approximately.
        x = _ar1()
        np.random.seed(42)
        y = aaft(x)
        self.assertLess(abs(_ac1(y) - _ac1(x)), 0.05)

    def test_output_shape_real_finite(self):
        x = _ar1(300)
        np.random.seed(42)
        y = aaft(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertTrue(np.allclose(y, y.real))

    def test_odd_length(self):
        x = _ar1(511)
        np.random.seed(42)
        y = aaft(x)
        self.assertEqual(y.shape, x.shape)
        np.testing.assert_array_equal(np.sort(y), np.sort(x))

    def test_deterministic_given_seed(self):
        x = _ar1()
        np.random.seed(7)
        y1 = aaft(x)
        np.random.seed(7)
        y2 = aaft(x)
        np.testing.assert_array_equal(y1, y2)

    def test_differs_from_original(self):
        x = _ar1()
        np.random.seed(42)
        y = aaft(x)
        self.assertTrue(np.any(y != x))


# ---------------------------------------------------------------------------
# iaaft
# ---------------------------------------------------------------------------

class TestIaaft(unittest.TestCase):
    def test_return_types_and_shapes(self):
        x = _ar1()
        np.random.seed(42)
        y, i, e = iaaft(x)
        self.assertEqual(y.shape, x.shape)
        self.assertIsInstance(i, (int, np.integer))
        self.assertIsInstance(e, float)
        self.assertGreaterEqual(i, 0)
        self.assertLess(i, 1000)  # converged well before maxiter
        self.assertTrue(np.isfinite(e))
        self.assertGreaterEqual(e, 0.0)

    def test_distribution_preserved_exactly(self):
        x = _logistic()
        np.random.seed(42)
        y, _, _ = iaaft(x)
        np.testing.assert_array_equal(np.sort(y), np.sort(x))

    def test_mean_preserved(self):
        x = _ar1()
        np.random.seed(42)
        y, _, _ = iaaft(x)
        np.testing.assert_allclose(np.mean(y), np.mean(x), rtol=1e-12)

    def test_spectrum_converged(self):
        # The returned error is the final RMSD between |rfft|^2 of the
        # surrogate and of the original, normalised by mean(ampl^2).
        x = _ar1()
        np.random.seed(42)
        y, _, e = iaaft(x)
        self.assertLess(e, 0.05)
        amp_x = np.abs(np.fft.rfft(x))
        amp_y = np.abs(np.fft.rfft(y))
        rel_rmsd = np.sqrt(np.mean((amp_x ** 2 - amp_y ** 2) ** 2))
        rel_rmsd /= np.mean(amp_x ** 2)
        self.assertAlmostEqual(e, rel_rmsd, places=10)

    def test_maxiter_respected(self):
        x = _ar1()
        np.random.seed(42)
        y, i, e = iaaft(x, maxiter=5)
        self.assertEqual(i, 4)  # loop index of the last performed iteration
        self.assertTrue(np.isfinite(e))
        self.assertEqual(y.shape, x.shape)

    def test_odd_length(self):
        x = _ar1(511)
        np.random.seed(42)
        y, i, e = iaaft(x)
        self.assertEqual(y.shape, x.shape)
        self.assertLess(e, 0.05)
        self.assertLess(i, 1000)
        np.testing.assert_array_equal(np.sort(y), np.sort(x))

    def test_deterministic_given_seed(self):
        x = _ar1()
        np.random.seed(7)
        y1, i1, e1 = iaaft(x)
        np.random.seed(7)
        y2, i2, e2 = iaaft(x)
        np.testing.assert_array_equal(y1, y2)
        self.assertEqual(i1, i2)
        self.assertEqual(e1, e2)

    def test_differs_from_original(self):
        x = _ar1()
        np.random.seed(42)
        y, _, _ = iaaft(x)
        self.assertTrue(np.any(y != x))


if __name__ == "__main__":
    unittest.main()
