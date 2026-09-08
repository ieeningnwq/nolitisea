"""Tests for nolitisea.embedding.mutual."""

import unittest

import numpy as np

from nolitisea.embedding.mutual import (
    _build_gaussian_kernel,
    _matrix_renyi_entropy,
    embedded_mutual_information,
    first_minimum,
    matrix_renyi_mutual_information,
    mutual_information,
)


class TestMutualInformation(unittest.TestCase):
    """Tests for the scalar mutual_information function."""

    def test_self_information_equals_entropy(self):
        # I(X; X) = H(X) in bits.  Finite-sample histogram estimation
        # introduces a small bias, so we allow 3 decimal places.
        rng = np.random.default_rng(0)
        x = rng.integers(0, 4, size=20000).astype(float)
        mi = mutual_information(x, x, bins=4)
        # Manual Shannon entropy with equal probabilities -> 2 bits.
        self.assertAlmostEqual(mi, 2.0, places=3)

    def test_independent_variables_near_zero(self):
        # Two independent uniform series should have MI close to 0.
        rng = np.random.default_rng(1)
        x = rng.random(5000)
        y = rng.random(5000)
        mi = mutual_information(x, y, bins=16)
        self.assertLess(abs(mi), 0.05)

    def test_perfectly_correlated_equals_self_mi(self):
        # Y = a*X + b preserves the binning (a>0, monotonic) -> MI = H(X).
        rng = np.random.default_rng(2)
        x = rng.integers(0, 5, size=2000).astype(float)
        y = 2.0 * x + 3.0
        self.assertAlmostEqual(
            mutual_information(x, y, bins=5),
            mutual_information(x, x, bins=5),
            places=6,
        )

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            mutual_information(np.arange(10.0), np.arange(11.0))

    def test_empty_returns_zero(self):
        self.assertEqual(mutual_information(np.array([]), np.array([])), 0.0)

    def test_normalize_in_unit_interval(self):
        rng = np.random.default_rng(3)
        x = rng.random(2000)
        y = 0.5 * x + rng.random(2000)
        nmi = mutual_information(x, y, bins=16, normalize=True)
        self.assertGreaterEqual(nmi, 0.0)
        self.assertLessEqual(nmi, 1.0)

    def test_constant_series_normalize_returns_zero(self):
        # H(X) + H(Y) = 0 -> returns 0.0 to avoid division by zero.
        x = np.full(100, 2.5)
        y = np.full(100, 1.0)
        self.assertEqual(mutual_information(x, y, bins=4, normalize=True), 0.0)


class TestEmbeddedMutualInformation(unittest.TestCase):
    """Tests for embedded_mutual_information."""

    def test_output_shape(self):
        x = np.arange(100.0)
        mi = embedded_mutual_information(x, max_lag=10, n_bins=8)
        self.assertEqual(mi.shape, (11,))
        self.assertTrue(np.all(np.isfinite(mi)))

    def test_lag0_equals_self_mi(self):
        # mi[0] = I(x; x) = H(x).
        rng = np.random.default_rng(0)
        x = rng.integers(0, 4, size=1000).astype(float)
        mi = embedded_mutual_information(x, max_lag=5, n_bins=4)
        self.assertAlmostEqual(mi[0], mutual_information(x, x, bins=4), places=12)

    def test_decreases_initially_for_correlated_series(self):
        # For a smooth correlated series, MI(x(t); x(t+tau)) drops as
        # tau grows from 0 (the series decorrelates).
        t = np.linspace(0, 50 * np.pi, 4000)
        x = np.sin(t)
        mi = embedded_mutual_information(x, max_lag=20, n_bins=16)
        self.assertGreater(mi[0], mi[1])
        self.assertGreater(mi[1], mi[2])

    def test_white_noise_drops_sharply(self):
        # White noise: MI(0) = H, MI(tau>0) ~ 0.
        rng = np.random.default_rng(1)
        x = rng.standard_normal(4000)
        mi = embedded_mutual_information(x, max_lag=8, n_bins=16)
        self.assertGreater(mi[0], 1.0)        # substantial entropy
        self.assertLess(abs(mi[1]), 0.1)      # near zero after one lag
        self.assertLess(abs(mi[4]), 0.1)

    def test_deterministic_output(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(500)
        mi1 = embedded_mutual_information(x, max_lag=15, n_bins=8)
        mi2 = embedded_mutual_information(x, max_lag=15, n_bins=8)
        np.testing.assert_array_equal(mi1, mi2)

    def test_invalid_series_ndim(self):
        with self.assertRaises(ValueError):
            embedded_mutual_information(np.zeros((5, 5)), max_lag=1)

    def test_invalid_max_lag_negative(self):
        with self.assertRaises(ValueError):
            embedded_mutual_information(np.arange(10.0), max_lag=-1)

    def test_invalid_max_lag_too_large(self):
        with self.assertRaises(ValueError):
            embedded_mutual_information(np.arange(10.0), max_lag=10)

    def test_invalid_n_bins(self):
        with self.assertRaises(ValueError):
            embedded_mutual_information(np.arange(20.0), max_lag=5, n_bins=0)


class TestFirstMinimum(unittest.TestCase):
    """Tests for first_minimum."""

    def test_lorenz_returns_reasonable_lag(self):
        # Lorenz x-component (dt=0.03): the first MI minimum is
        # typically around lag 5-20.
        from nolitisea.generate.lorenz import lorenz

        np.random.seed(0)
        _, data = lorenz(length=3000)
        x = data[:, 0]
        tau = first_minimum(x, max_lag=50, n_bins=16)
        self.assertGreaterEqual(tau, 2)
        self.assertLessEqual(tau, 50)

    def test_returns_int(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(500)
        tau = first_minimum(x, max_lag=20, n_bins=8)
        self.assertIsInstance(tau, int)
        self.assertGreaterEqual(tau, 1)

    def test_max_lag_too_small_raises(self):
        with self.assertRaises(ValueError):
            first_minimum(np.arange(10.0), max_lag=1)

    def test_fallback_to_global_min_when_no_local_min(self):
        # Strictly monotonic MI curve (e.g. linear ramp) has no local
        # minimum; first_minimum should fall back to argmin.
        # A linear ramp x = 0,1,2,...,N-1 with bins creates a monotonic
        # MI curve, so no interior local minimum exists.
        x = np.arange(200.0)
        tau = first_minimum(x, max_lag=30, n_bins=8)
        # Should not raise and should be within range.
        self.assertGreaterEqual(tau, 1)
        self.assertLessEqual(tau, 30)

    def test_deterministic_output(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal(800)
        tau1 = first_minimum(x, max_lag=30, n_bins=16)
        tau2 = first_minimum(x, max_lag=30, n_bins=16)
        self.assertEqual(tau1, tau2)


class TestMatrixRenyiMutualInformation(unittest.TestCase):
    """Tests for the kernel-based matrix_renyi_mutual_information."""

    def test_self_information_nonneg(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(200)
        mi = matrix_renyi_mutual_information(x, x)
        self.assertGreaterEqual(mi, 0.0)

    def test_independent_near_zero(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(500)
        y = rng.standard_normal(500)
        mi = matrix_renyi_mutual_information(x, y)
        self.assertLess(mi, 0.2)

    def test_correlated_higher_than_independent(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(500)
        y_indep = rng.standard_normal(500)
        y_corr = x + 0.1 * rng.standard_normal(500)
        mi_indep = matrix_renyi_mutual_information(x, y_indep)
        mi_corr = matrix_renyi_mutual_information(x, y_corr)
        self.assertGreater(mi_corr, mi_indep)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            matrix_renyi_mutual_information(np.arange(10.0), np.arange(11.0))

    def test_alpha_general_path(self):
        # alpha != 2 exercises the eigenvalue decomposition branch.
        rng = np.random.default_rng(3)
        x = rng.standard_normal(100)
        mi = matrix_renyi_mutual_information(x, x, alpha=1.5)
        self.assertTrue(np.isfinite(mi))


class TestGaussianKernel(unittest.TestCase):
    """Tests for the _build_gaussian_kernel helper."""

    def test_diagonal_is_one(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(20)
        K = _build_gaussian_kernel(x)
        np.testing.assert_array_almost_equal(np.diag(K), np.ones(20))

    def test_symmetric(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(15)
        K = _build_gaussian_kernel(x)
        np.testing.assert_array_almost_equal(K, K.T)

    def test_explicit_sigma(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(10)
        K1 = _build_gaussian_kernel(x, sigma=1.0)
        K2 = _build_gaussian_kernel(x, sigma=2.0)
        # Larger sigma -> wider kernel -> higher off-diagonal values.
        self.assertGreater(K2[0, 1], K1[0, 1])


class TestMatrixRenyiEntropy(unittest.TestCase):
    """Tests for the _matrix_renyi_entropy helper."""

    def test_alpha2_fast_path(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(30)
        K = _build_gaussian_kernel(x)
        h = _matrix_renyi_entropy(K, alpha=2.0)
        self.assertTrue(np.isfinite(h))

    def test_identical_points_low_entropy(self):
        # All-identical points -> kernel is all ones -> minimal entropy.
        x = np.ones(10)
        K = _build_gaussian_kernel(x)
        h = _matrix_renyi_entropy(K, alpha=2.0)
        self.assertLessEqual(h, 1e-10)


if __name__ == "__main__":
    unittest.main()
