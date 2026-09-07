"""Tests for ``nolitisea.linear.spectrum.power_spectrum``."""

import unittest

import numpy as np
from scipy.signal import periodogram, welch

from nolitisea.linear.spectrum import power_spectrum


class TestPowerSpectrumOutputStructure(unittest.TestCase):
    """Validate output shapes, types and frequency spacing."""

    def test_returns_two_arrays_same_length(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(256)
        freq, psd = power_spectrum(x, method="welch")
        self.assertEqual(freq.shape, psd.shape)
        self.assertEqual(freq.ndim, 1)

    def test_frequency_starts_at_zero(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(128)
        for method in ("periodogram", "welch", "fft"):
            freq, _ = power_spectrum(x, method=method)
            self.assertEqual(freq[0], 0.0, f"method={method}")

    def test_onesided_nyquist_for_even_length(self):
        """For even real input the one-sided spectrum includes Nyquist."""
        rng = np.random.default_rng(2)
        x = rng.standard_normal(128)
        freq, _ = power_spectrum(x, method="periodogram", fs=1.0)
        self.assertAlmostEqual(freq[-1], 0.5, places=10)

    def test_fs_scales_frequencies(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal(64)
        f1, _ = power_spectrum(x, method="fft", fs=1.0)
        f10, _ = power_spectrum(x, method="fft", fs=10.0)
        np.testing.assert_allclose(f10, f1 * 10.0, rtol=1e-12)


class TestPowerSpectrumMethods(unittest.TestCase):
    """Cross-check each method against SciPy reference."""

    def test_periodogram_matches_scipy(self):
        rng = np.random.default_rng(10)
        x = rng.standard_normal(200)
        f_ref, p_ref = periodogram(x, fs=1.0, detrend="constant",
                                   scaling="density", return_onesided=True)
        f, p = power_spectrum(x, method="periodogram")
        np.testing.assert_allclose(f, f_ref, rtol=1e-12)
        np.testing.assert_allclose(p, p_ref, rtol=1e-12)

    def test_welch_matches_scipy(self):
        rng = np.random.default_rng(11)
        x = rng.standard_normal(500)
        f_ref, p_ref = welch(x, fs=1.0, nperseg=128, detrend="constant",
                             scaling="density", return_onesided=True)
        f, p = power_spectrum(x, method="welch", nperseg=128)
        np.testing.assert_allclose(f, f_ref, rtol=1e-12)
        np.testing.assert_allclose(p, p_ref, rtol=1e-12)

    def test_fft_matches_manual(self):
        rng = np.random.default_rng(12)
        x = rng.standard_normal(64)
        x = x - x.mean()
        n = x.size
        fft_vals = np.fft.rfft(x)
        # spectrum scaling: double non-DC/Nyquist, then / N²
        expected = np.abs(fft_vals) ** 2
        if len(expected) >= 3:
            expected[1:-1] *= 2.0
        expected /= n * n
        f, p = power_spectrum(x, method="fft", detrend=None,
                              scaling="spectrum")
        np.testing.assert_allclose(f, np.fft.rfftfreq(n, d=1.0), rtol=1e-12)
        np.testing.assert_allclose(p, expected, rtol=1e-12)

    def test_fft_density_scaling(self):
        rng = np.random.default_rng(13)
        x = rng.standard_normal(64)
        fs = 2.5
        n = len(x)
        _, p_dens = power_spectrum(x, method="fft", scaling="density", fs=fs)
        _, p_spec = power_spectrum(x, method="fft", scaling="spectrum", fs=fs)
        # density / spectrum ratio is N / fs for every bin (including DC
        # and Nyquist): density = doubling / (N·fs), spectrum = doubling / N²
        ratio = n / fs
        np.testing.assert_allclose(p_dens / p_spec, ratio, rtol=1e-12)
        # density also integrates to mean(x²) via ∑p·df with df = fs/N
        f = np.fft.rfftfreq(n, d=1.0 / fs)
        df = f[1] - f[0]
        x_mean0 = x - x.mean()
        np.testing.assert_allclose(np.sum(p_dens * df),
                                   np.mean(x_mean0 ** 2), rtol=1e-10)


class TestPowerSpectrumPhysical(unittest.TestCase):
    """Validate physical behaviour on signals with known spectra."""

    def test_pure_sine_peak_at_signal_frequency(self):
        fs = 100.0
        t = np.arange(1024) / fs
        f0 = 10.0
        x = np.sin(2 * np.pi * f0 * t)
        freq, psd = power_spectrum(x, method="welch", fs=fs,
                                    nperseg=256, detrend=None)
        peak_idx = np.argmax(psd)
        self.assertAlmostEqual(freq[peak_idx], f0, delta=fs / 256)

    def test_white_noise_flat_spectrum(self):
        """Welch PSD of white Gaussian noise should be approximately flat."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(4096)
        _, psd = power_spectrum(x, method="welch", nperseg=512,
                                detrend="constant")
        # Exclude DC and Nyquist bins, check spread is modest.
        interior = psd[1:-1]
        mean = interior.mean()
        # Standard deviation of a chi2(2) variate is equal to its mean;
        # Welch averages segments so the spread is smaller.  Allow 3x.
        self.assertLess(interior.std() / mean, 0.5)

    def test_dc_peak_removed_by_constant_detrend(self):
        """A constant offset should not create a DC peak after detrend."""
        t = np.arange(256)
        x = 5.0 * np.ones_like(t, dtype=float)
        for method in ("periodogram", "welch", "fft"):
            _, psd = power_spectrum(x, method=method, detrend="constant")
            # After removing the mean the signal is zero -> zero power.
            self.assertTrue(np.allclose(psd, 0.0, atol=1e-25),
                            f"method={method} not zeroed by detrend")


class TestPowerSpectrumValidation(unittest.TestCase):
    """Parameter validation and error handling."""

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            power_spectrum(np.array([]), method="welch")

    def test_single_sample_raises(self):
        with self.assertRaises(ValueError):
            power_spectrum(np.array([1.0]), method="welch")

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            power_spectrum(np.zeros(10), method="invalid")

    def test_fft_onesided_false_raises(self):
        with self.assertRaises(ValueError):
            power_spectrum(np.zeros(10), method="fft",
                           return_onesided=False)

    def test_fft_invalid_detrend_raises(self):
        with self.assertRaises(ValueError):
            power_spectrum(np.zeros(10), method="fft", detrend="quadratic")

    def test_flattens_2d_input(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal((4, 64))
        f1, p1 = power_spectrum(x, method="fft")
        f2, p2 = power_spectrum(x.ravel(), method="fft")
        np.testing.assert_allclose(f1, f2, rtol=0)
        np.testing.assert_allclose(p1, p2, rtol=0)


class TestPowerSpectrumDeterminism(unittest.TestCase):
    """Same input -> same output (no hidden randomness)."""

    def test_repeat_same_result(self):
        rng = np.random.default_rng(99)
        x = rng.standard_normal(200)
        r1 = power_spectrum(x, method="welch", nperseg=64)
        r2 = power_spectrum(x, method="welch", nperseg=64)
        np.testing.assert_array_equal(r1[0], r2[0])
        np.testing.assert_allclose(r1[1], r2[1], rtol=0)

    def test_linear_detrend_matches_manual(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(128) + np.linspace(0, 10, 128)
        trend = np.polyval(np.polyfit(np.arange(128), x, 1),
                           np.arange(128))
        x_det = x - trend
        _, p_ref = power_spectrum(x_det, method="fft", detrend=None)
        _, p = power_spectrum(x, method="fft", detrend="linear")
        np.testing.assert_allclose(p, p_ref, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
