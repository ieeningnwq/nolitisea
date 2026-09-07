"""Tests for ``nolitisea.linear.mem_spec.mem_spectrum``."""

import unittest

import numpy as np

from nolitisea.linear.mem_spec import _burg_coefs, _power_spectrum, mem_spectrum


class TestBurgCoefficientsExact(unittest.TestCase):
    """Hand-computed exact cases for Burg's method."""

    def test_order1_zero_reflection(self):
        """x = [1, 2, -1]: forward/backward dot product vanishes -> k = 0."""
        coeffs, sigma2 = _burg_coefs(np.array([1.0, 2.0, -1.0]), 1)
        np.testing.assert_allclose(coeffs, [0.0], atol=1e-15)
        self.assertAlmostEqual(sigma2, 2.0, places=12)

    def test_order1_nonzero_reflection(self):
        """x = [1, 2, 3]: k = 8/9, sigma2 = (14/3) * (1 - 64/81) = 238/243."""
        coeffs, sigma2 = _burg_coefs(np.array([1.0, 2.0, 3.0]), 1)
        self.assertAlmostEqual(coeffs[0], 8.0 / 9.0, places=12)
        self.assertAlmostEqual(sigma2, 238.0 / 243.0, places=12)

    def test_order2_recursion(self):
        """x = [1, 2, -1], order 2: after the first (k=0) sweep the
        residual pair is ([1], [-1]), giving k = -1 and sigma2 = 0."""
        coeffs, sigma2 = _burg_coefs(np.array([1.0, 2.0, -1.0]), 2)
        np.testing.assert_allclose(coeffs, [0.0, -1.0], atol=1e-15)
        self.assertAlmostEqual(sigma2, 0.0, places=12)

    def test_sigma2_bounded_by_mean_square(self):
        """Each reflection factor (1 - k^2) lies in [0, 1], so the residual
        variance never exceeds the initial mean square and stays
        non-negative.  For mean-subtracted input this is the variance."""
        rng = np.random.default_rng(21)
        for _ in range(5):
            x = rng.standard_normal(300)
            x = x - x.mean()
            for order in (1, 3, 8):
                _, sigma2 = _burg_coefs(x, order)
                self.assertGreaterEqual(sigma2, 0.0)
                self.assertLessEqual(sigma2, np.dot(x, x) / x.size + 1e-12)


class TestBurgCoefficientsStatistical(unittest.TestCase):
    """Burg estimates recover the parameters of synthetic AR processes."""

    def test_recovers_ar1_coefficient(self):
        rho = 0.8
        rng = np.random.default_rng(42)
        n = 4096
        e = rng.standard_normal(n)
        x = np.empty(n)
        x[0] = e[0]
        for t in range(1, n):
            x[t] = rho * x[t - 1] + e[t]
        coeffs, _ = _burg_coefs(x - x.mean(), 1)
        self.assertAlmostEqual(coeffs[0], rho, delta=0.05)

    def test_recovers_ar2_coefficients(self):
        a1, a2 = 0.6, -0.3
        rng = np.random.default_rng(43)
        n = 20000
        e = rng.standard_normal(n)
        x = np.empty(n)
        x[:2] = e[:2]
        for t in range(2, n):
            x[t] = a1 * x[t - 1] + a2 * x[t - 2] + e[t]
        coeffs, _ = _burg_coefs(x - x.mean(), 2)
        self.assertAlmostEqual(coeffs[0], a1, delta=0.05)
        self.assertAlmostEqual(coeffs[1], a2, delta=0.05)

    def test_white_noise_coeffs_small(self):
        """Burg coefficients of white noise should be near zero."""
        rng = np.random.default_rng(44)
        x = rng.standard_normal(4096)
        coeffs, _ = _burg_coefs(x, 4)
        self.assertTrue(np.all(np.abs(coeffs) < 0.2))


class TestPowerSpectrumFormula(unittest.TestCase):
    """``_power_spectrum`` must evaluate sigma2 / |A(f)|^2 / sqrt(n)."""

    def test_matches_direct_evaluation(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(500)
        order = 6
        n_freq = 128
        coeffs, sigma2 = _burg_coefs(x, order)
        freqs, power = _power_spectrum(coeffs, sigma2, n_freq, x.size)

        idx = np.arange(1, order + 1)
        phase = 2.0 * np.pi * freqs[:, None] * idx[None, :]
        A = 1.0 - np.sum(coeffs[None, :] * np.exp(1j * phase), axis=1)
        expected = sigma2 / np.abs(A) ** 2 / np.sqrt(x.size)
        np.testing.assert_allclose(power, expected, rtol=1e-12)

    def test_grid_spacing(self):
        coeffs = np.array([0.5, -0.2])
        freqs, _ = _power_spectrum(coeffs, 1.0, 64, 100)
        self.assertEqual(freqs[0], 0.0)
        np.testing.assert_allclose(np.diff(freqs), 1.0 / 128.0, rtol=1e-15)
        self.assertLess(freqs[-1], 0.5)


class TestMemSpectrumOutputStructure(unittest.TestCase):
    """Validate output shapes and the frequency grid."""

    def test_returns_two_arrays_of_nfreq(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(256)
        freqs, power = mem_spectrum(x, order=4, n_freq=512)
        self.assertEqual(freqs.shape, (512,))
        self.assertEqual(power.shape, (512,))
        self.assertEqual(freqs.ndim, 1)

    def test_frequency_grid(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(128)
        for n_freq in (1, 2, 7, 512):
            freqs, _ = mem_spectrum(x, order=3, n_freq=n_freq)
            self.assertEqual(freqs[0], 0.0)
            self.assertLess(freqs[-1], 0.5)
            expected = np.arange(n_freq) / (2.0 * n_freq)
            np.testing.assert_allclose(freqs, expected, rtol=1e-15)

    def test_power_positive_and_finite(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(300)
        _, power = mem_spectrum(x, order=5)
        self.assertTrue(np.all(np.isfinite(power)))
        self.assertTrue(np.all(power > 0.0))


class TestMemSpectrumPhysical(unittest.TestCase):
    """Validate behaviour on signals with known spectra."""

    def test_pure_sine_peak_at_signal_frequency(self):
        n = 1000
        f0 = 0.1
        t = np.arange(n)
        x = np.sin(2.0 * np.pi * f0 * t)
        freqs, power = mem_spectrum(x, order=20, n_freq=1024)
        peak = freqs[np.argmax(power)]
        # MEM is a super-resolution method: the peak should be sharp and
        # within a few bins of the true frequency (bin width ~ 4.9e-4).
        self.assertAlmostEqual(peak, f0, delta=0.005)

    def test_ar1_spectrum_shape(self):
        """Fitted MEM spectrum matches the theoretical AR(1) PSD shape."""
        rho = 0.8
        rng = np.random.default_rng(42)
        n = 4096
        e = rng.standard_normal(n)
        x = np.empty(n)
        x[0] = e[0]
        for t in range(1, n):
            x[t] = rho * x[t - 1] + e[t]

        freqs, power = mem_spectrum(x, order=1, n_freq=512)
        grid = dict(zip(freqs, power))

        # Theoretical PSD ratio P(0) / P(0.25) for AR(1) with a = 0.8.
        theo = (1.0 + rho ** 2) / (1.0 - rho) ** 2
        actual = grid[0.0] / grid[0.25]
        self.assertAlmostEqual(actual, theo, delta=0.25 * theo)

    def test_white_noise_spectrum_approximately_flat(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal(4096)
        _, power = mem_spectrum(x, order=4, n_freq=256)
        interior = power[1:]  # skip the DC bin
        self.assertLess(interior.max() / interior.min(), 3.0)


class TestMemSpectrumInputHandling(unittest.TestCase):
    """Mean removal and determinism."""

    def test_mean_subtracted_internally(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal(400) + 100.0
        f1, p1 = mem_spectrum(x, order=4)
        f2, p2 = mem_spectrum(x - x.mean(), order=4)
        np.testing.assert_allclose(f1, f2, rtol=0)
        np.testing.assert_allclose(p1, p2, rtol=1e-10)

    def test_accepts_list_input(self):
        x = [1.0, -1.0, 0.5, 2.0, -0.5, 1.5, 0.0, -2.0]
        freqs, power = mem_spectrum(x, order=2, n_freq=16)
        self.assertEqual(freqs.shape, (16,))
        self.assertTrue(np.all(np.isfinite(power)))

    def test_deterministic(self):
        rng = np.random.default_rng(99)
        x = rng.standard_normal(200)
        r1 = mem_spectrum(x, order=4, n_freq=64)
        r2 = mem_spectrum(x, order=4, n_freq=64)
        np.testing.assert_array_equal(r1[0], r2[0])
        np.testing.assert_array_equal(r1[1], r2[1])


class TestMemSpectrumValidation(unittest.TestCase):
    """Parameter validation and error handling."""

    def test_2d_input_raises(self):
        with self.assertRaises(ValueError):
            mem_spectrum(np.zeros((4, 10)), order=2)

    def test_order_below_one_raises(self):
        with self.assertRaises(ValueError):
            mem_spectrum(np.ones(10), order=0)

    def test_order_equal_to_length_raises(self):
        with self.assertRaises(ValueError):
            mem_spectrum(np.ones(10), order=10)

    def test_order_above_length_raises(self):
        with self.assertRaises(ValueError):
            mem_spectrum(np.ones(10), order=11)

    def test_nfreq_below_one_raises(self):
        with self.assertRaises(ValueError):
            mem_spectrum(np.ones(10), order=2, n_freq=0)


if __name__ == "__main__":
    unittest.main()
