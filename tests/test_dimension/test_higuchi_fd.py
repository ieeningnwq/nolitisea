"""Tests for nolitisea.dimension.higuchi_fd."""

import unittest

import numpy as np

from nolitisea.dimension.higuchi_fd import higuchi_fd


def _brute_higuchi_fd(x, kmax):
    """Reference HFD implementation for cross-checking.

    Implements the exact formulas from Higuchi (1988), without any
    vectorisation tricks, so the math is as close to the paper as
    possible.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    N, n_series = x.shape
    L = np.empty((kmax, n_series))
    for k in range(1, kmax + 1):
        Lm_all = np.empty((k, n_series))
        for m in range(k):
            n_int = int((N - m - 1) / k)
            sub = x[m:m + n_int * k + 1:k, :]
            length = np.sum(np.abs(np.diff(sub, axis=0)), axis=0)
            Lm_all[m] = length * (N - 1) / (n_int * k ** 2)
        L[k - 1] = np.mean(Lm_all, axis=0)
    k_arr = np.arange(1, kmax + 1)
    coeffs = np.polyfit(np.log(k_arr), np.log(L), 1)
    return -coeffs[0]


class TestHiguchiFDBruteForce(unittest.TestCase):
    """Cross-check against a paper-faithful reference implementation."""

    def test_single_series_matches_brute(self):
        rng = np.random.default_rng(42)
        s = np.cumsum(rng.standard_normal(500)).reshape(-1, 1)
        for kmax in (5, 10, 20):
            with self.subTest(kmax=kmax):
                got = higuchi_fd(s, kmax)
                ref = _brute_higuchi_fd(s, kmax)
                np.testing.assert_allclose(got, ref, rtol=1e-12)

    def test_multivariate_matches_brute(self):
        rng = np.random.default_rng(7)
        data = np.column_stack([
            np.cumsum(rng.standard_normal(300)),
            rng.standard_normal(300),
            np.sin(np.linspace(0, 20 * np.pi, 300)),
        ])
        got = higuchi_fd(data, kmax=15)
        ref = _brute_higuchi_fd(data, kmax=15)
        np.testing.assert_allclose(got, ref, rtol=1e-12)

    def test_different_kmax_values(self):
        rng = np.random.default_rng(99)
        s = rng.standard_normal((400, 2))
        for kmax in (2, 3, 8, 16):
            with self.subTest(kmax=kmax):
                got = higuchi_fd(s, kmax)
                ref = _brute_higuchi_fd(s, kmax)
                np.testing.assert_allclose(got, ref, rtol=1e-12)


class TestHiguchiFDTheoreticalValues(unittest.TestCase):
    """Check known fractal dimensions for canonical series."""

    def test_random_walk_near_1_5(self):
        rng = np.random.default_rng(42)
        rw = np.cumsum(rng.standard_normal((5000, 5)), axis=0)
        hfd = higuchi_fd(rw, kmax=20)
        for val in hfd:
            self.assertGreater(val, 1.4)
            self.assertLess(val, 1.6)

    def test_white_noise_near_2(self):
        rng = np.random.default_rng(123)
        wn = rng.standard_normal((5000, 5))
        hfd = higuchi_fd(wn, kmax=20)
        for val in hfd:
            self.assertGreater(val, 1.85)
            self.assertLess(val, 2.15)

    def test_sine_wave_near_1(self):
        t = np.linspace(0, 50 * np.pi, 5000)
        sine = np.column_stack([np.sin(t), np.cos(t), np.sin(0.3 * t)])
        hfd = higuchi_fd(sine, kmax=20)
        for val in hfd:
            self.assertGreater(val, 0.9)
            self.assertLess(val, 1.15)

    def test_linear_ramp_is_1(self):
        ramp = np.column_stack([
            np.linspace(0, 10, 2000),
            np.linspace(-5, 5, 2000),
        ])
        hfd = higuchi_fd(ramp, kmax=10)
        np.testing.assert_allclose(hfd, 1.0, atol=1e-10)


class TestHiguchiFDProperties(unittest.TestCase):
    """Mathematical properties of the estimator."""

    def test_multi_column_consistency(self):
        rng = np.random.default_rng(42)
        s = np.cumsum(rng.standard_normal(3000))
        single = higuchi_fd(s.reshape(-1, 1), kmax=15)
        triple = higuchi_fd(np.column_stack([s, s, s]), kmax=15)
        self.assertEqual(triple.shape, (3,))
        np.testing.assert_allclose(triple, np.full(3, single[0]))

    def test_reversed_series_same_hfd(self):
        rng = np.random.default_rng(7)
        s = np.cumsum(rng.standard_normal(3000)).reshape(-1, 1)
        fwd = higuchi_fd(s, kmax=15)
        rev = higuchi_fd(s[::-1], kmax=15)
        np.testing.assert_allclose(fwd, rev)

    def test_shape_output(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((500, 7))
        hfd = higuchi_fd(data, kmax=10)
        self.assertEqual(hfd.shape, (7,))
        self.assertEqual(hfd.dtype, np.float64)


class TestHiguchiFDErrors(unittest.TestCase):
    """Input validation."""

    def test_1d_input_supported(self):
        rng = np.random.default_rng(42)
        s = np.cumsum(rng.standard_normal(500))
        hfd_1d = higuchi_fd(s, kmax=10)
        hfd_2d = higuchi_fd(s.reshape(-1, 1), kmax=10)
        self.assertEqual(hfd_1d.shape, (1,))
        np.testing.assert_allclose(hfd_1d, hfd_2d)

    def test_1d_list_input(self):
        s = list(range(100))
        hfd = higuchi_fd(s, kmax=5)
        self.assertEqual(hfd.shape, (1,))

    def test_3d_input_raises(self):
        with self.assertRaises(ValueError):
            higuchi_fd(np.zeros((10, 10, 3)), kmax=5)

    def test_kmax_below_2_raises(self):
        with self.assertRaises(ValueError):
            higuchi_fd(np.zeros((100, 2)), kmax=1)
        with self.assertRaises(ValueError):
            higuchi_fd(np.zeros((100, 2)), kmax=0)

    def test_series_too_short_raises(self):
        with self.assertRaises(ValueError):
            higuchi_fd(np.array([[1.0, 2.0, 3.0]]), kmax=2)
        with self.assertRaises(ValueError):
            higuchi_fd(np.zeros((3, 2)), kmax=10)

    def test_boundary_length_2kmax_works(self):
        rng = np.random.default_rng(5)
        data = rng.standard_normal((20, 2))
        hfd = higuchi_fd(data, kmax=10)
        self.assertEqual(hfd.shape, (2,))


if __name__ == "__main__":
    unittest.main()
