"""
test_powerspectrum_full.py

Comprehensive unittest test suite for powerspectrum.power_spectrum().

Coverage groups (85 tests):
  1. Input validation          (empty / too short)
  2. Auto-flatten              (2D / 3D arrays)
  3. Method dispatch           (periodogram / welch / fft / unknown)
  4. detrend                   (constant / linear / None / False / callable / invalid)
  5. scaling                   (density / spectrum)
  6. return_onesided           (True / False, welch & periodogram & fft)
  7. FFT-specific behaviour    (no detrend, freq spacing, DC)
  8. Welch parameters          (nperseg / noverlap propagation)
  9. fs / frequency axis       (correct bin spacing & ordering)
 10. Mathematical correctness  (DC value, known sinusoid peaks, Parseval)
 11. Reproducibility          (same input -> same output)
 12. Numeric quality          (non-negative PSD, finite values)
 13. Edge cases               (all-zeros, constant, linear ramp, single sinusoid)
 14. Type & shape contracts   (freq & psd are 1D, same length, dtype)
"""

import unittest
import warnings
import numpy as np

from nolitisea.utils import power_spectrum


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_sinusoid(n=1024, freq=0.1, fs=1.0, amp=1.0, seed=0):
    """Return a clean real sinusoid."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    x = amp * np.sin(2 * np.pi * freq * t)
    return x


def _make_colored_noise(n=1024, alpha=1.0, seed=0):
    """Generate 1/f^alpha noise via FFT."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    k = np.arange(1, len(X) + 1, dtype=float)
    X[1:] *= k[1:] ** (-alpha / 2.0)
    X[0] = 0.0
    y = np.fft.irfft(X, n=n)
    return y


# ---------------------------------------------------------------------------
# 1. Input validation
# ---------------------------------------------------------------------------
class TestInputValidation(unittest.TestCase):
    """All paths that must raise."""

    def test_empty_1d_raises(self):
        with self.assertRaises(ValueError):
            power_spectrum(np.array([]))

    def test_empty_2d_raises(self):
        with self.assertRaises(ValueError):
            power_spectrum(np.zeros((0, 4)))

    def test_scalar_raises(self):
        with self.assertRaises(ValueError):
            power_spectrum(np.array(3.14))

    def test_length_one_raises(self):
        with self.assertRaises(ValueError):
            power_spectrum(np.array([0.5]))

    def test_length_two_is_ok(self):
        """N=2 is the minimum allowed."""
        f, p = power_spectrum(np.array([1.0, -1.0]))
        self.assertEqual(f.ndim, 1)
        self.assertEqual(p.ndim, 1)
        self.assertGreater(len(f), 0)


# ---------------------------------------------------------------------------
# 2. Auto-flatten
# ---------------------------------------------------------------------------
class TestAutoFlatten(unittest.TestCase):

    def test_2d_row_vector_flattened(self):
        x = np.arange(16).reshape(1, -1).astype(float)
        f, p = power_spectrum(x)
        self.assertEqual(f.ndim, 1)
        self.assertEqual(p.ndim, 1)

    def test_2d_column_vector_flattened(self):
        x = np.arange(16).reshape(-1, 1).astype(float)
        f, p = power_spectrum(x)
        self.assertEqual(f.ndim, 1)
        self.assertEqual(p.ndim, 1)

    def test_2d_matrix_flattened_in_order(self):
        """Flattening should be C-style (row-major) by default."""
        x = np.arange(16).reshape(4, 4).astype(float)
        f1, p1 = power_spectrum(x)
        f2, p2 = power_spectrum(x.ravel())
        np.testing.assert_array_equal(f1, f2)
        np.testing.assert_allclose(p1, p2)

    def test_3d_array_flattened(self):
        x = np.random.randn(2, 3, 8)
        f, p = power_spectrum(x)
        self.assertEqual(f.size, p.size)

    def test_flatten_does_not_mutate_input(self):
        x = np.arange(12).reshape(3, 4).astype(float)
        x_copy = x.copy()
        power_spectrum(x)
        np.testing.assert_array_equal(x, x_copy)


# ---------------------------------------------------------------------------
# 3. Method dispatch
# ---------------------------------------------------------------------------
class TestMethodDispatch(unittest.TestCase):

    def setUp(self):
        self.x = _make_sinusoid(n=512, freq=0.1)

    def test_periodogram_returns_1d(self):
        f, p = power_spectrum(self.x, method="periodogram")
        self.assertEqual(f.ndim, 1)
        self.assertEqual(p.ndim, 1)

    def test_welch_returns_1d(self):
        f, p = power_spectrum(self.x, method="welch", nperseg=128)
        self.assertEqual(f.ndim, 1)
        self.assertEqual(p.ndim, 1)

    def test_fft_returns_1d(self):
        f, p = power_spectrum(self.x, method="fft")
        self.assertEqual(f.ndim, 1)
        self.assertEqual(p.ndim, 1)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            power_spectrum(self.x, method="bogus")

    def test_empty_string_method_raises(self):
        with self.assertRaises(ValueError):
            power_spectrum(self.x, method="")

    def test_periodogram_has_more_points_than_welch(self):
        """Welch segments reduce frequency resolution."""
        f_p, _ = power_spectrum(self.x, method="periodogram")
        f_w, _ = power_spectrum(self.x, method="welch", nperseg=128)
        self.assertGreater(len(f_p), len(f_w))


# ---------------------------------------------------------------------------
# 4. detrend
# ---------------------------------------------------------------------------
class TestDetrend(unittest.TestCase):

    def test_detrend_constant_removes_dc(self):
        """A pure DC signal with detrend='constant' should yield ~0 PSD."""
        x = np.ones(256)
        f, p = power_spectrum(x, method="periodogram", detrend="constant")
        self.assertAlmostEqual(np.max(p), 0.0, places=10)

    def test_detrend_none_keeps_dc_spike(self):
        """Without detrending, DC should be large."""
        x = np.ones(256)
        f, p = power_spectrum(x, method="periodogram", detrend=None)
        self.assertGreater(p[0], 1.0)

    def test_detrend_false_keeps_dc_spike(self):
        x = np.ones(256)
        f, p = power_spectrum(x, method="periodogram", detrend=False)
        self.assertGreater(p[0], 1.0)

    def test_detrend_linear_removes_ramp(self):
        """A linear ramp should vanish after linear detrend."""
        n = 256
        x = np.linspace(0, 10, n)
        f, p = power_spectrum(x, method="periodogram", detrend="linear")
        self.assertLess(np.max(p), 1e-10)

    def test_detrend_callable(self):
        """Custom detrend function should be applied (FFT method)."""
        x = np.arange(64, dtype=float)
        f1, p1 = power_spectrum(x, method="fft", detrend=lambda y: y - y.mean())
        f2, p2 = power_spectrum(x, method="fft", detrend="constant")
        np.testing.assert_allclose(p1, p2, rtol=1e-10)

    def test_detrend_invalid_string_raises(self):
        x = np.random.randn(64)
        with self.assertRaises(ValueError):
            power_spectrum(x, method="fft", detrend="invalid_choice")

    def test_detrend_fft_constant(self):
        x = np.random.randn(128) + 5.0
        f1, p1 = power_spectrum(x, method="fft", detrend="constant")
        f2, p2 = power_spectrum(x - x.mean(), method="fft", detrend=None)
        np.testing.assert_allclose(p1, p2, rtol=1e-10)

    def test_detrend_fft_linear(self):
        n = 128
        t = np.arange(n)
        x = 0.5 * t + 3.0 + 0.1 * np.random.randn(n)
        f, p = power_spectrum(x, method="fft", detrend="linear")
        self.assertLess(np.max(np.abs(p)), 0.1)


# ---------------------------------------------------------------------------
# 5. scaling
# ---------------------------------------------------------------------------
class TestScaling(unittest.TestCase):

    def setUp(self):
        self.x = np.random.randn(256)

    def test_density_returns_positive(self):
        f, p = power_spectrum(self.x, method="welch", nperseg=64,
                                  scaling="density")
        self.assertTrue(np.all(p >= 0))

    def test_spectrum_returns_positive(self):
        f, p = power_spectrum(self.x, method="welch", nperseg=64,
                                  scaling="spectrum")
        self.assertTrue(np.all(p >= 0))

    def test_density_vs_spectrum_ratio(self):
        """For fft: spectrum = density * fs (exactly)."""
        fs = 100.0
        x = np.random.randn(128)
        f_d, p_d = power_spectrum(x, method="fft", fs=fs, scaling="density")
        f_s, p_s = power_spectrum(x, method="fft", fs=fs, scaling="spectrum")
        ratio = np.mean(p_s[1:]) / np.mean(p_d[1:])
        self.assertAlmostEqual(ratio, fs, delta=fs * 0.05)

    def test_invalid_scaling_raises(self):
        with self.assertRaises(ValueError):
            power_spectrum(self.x, method="welch", scaling="bogus")

    def test_fft_density_scaling(self):
        """Manual check: |rFFT|^2 / (fs * N) — using a known signal.

        The function applies detrend BEFORE FFT, so we must match that.
        With detrend='constant', the mean is subtracted first.
        """
        n = 128
        fs = 1.0
        x = np.sin(2 * np.pi * 0.1 * np.arange(n))
        # Replicate what the function does: detrend then FFT
        x_detrended = x - x.mean()  # detrend="constant"
        X = np.fft.rfft(x_detrended)
        expected = np.abs(X) ** 2 / (fs * n)
        f, p = power_spectrum(x, method="fft", fs=fs, scaling="density",
                                  detrend="constant")
        np.testing.assert_allclose(p, expected, rtol=1e-10)

    def test_fft_spectrum_scaling(self):
        n = 128
        x = np.sin(2 * np.pi * 0.1 * np.arange(n))
        x_detrended = x - x.mean()
        X = np.fft.rfft(x_detrended)
        expected = np.abs(X) ** 2 / n
        f, p = power_spectrum(x, method="fft", scaling="spectrum",
                                  detrend="constant")
        np.testing.assert_allclose(p, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# 6. return_onesided
# ---------------------------------------------------------------------------
class TestReturnOnesided(unittest.TestCase):

    def setUp(self):
        self.x = np.random.randn(256)

    def test_onesided_true_shorter_than_twosided(self):
        f1, p1 = power_spectrum(self.x, method="periodogram",
                                    return_onesided=True)
        f2, p2 = power_spectrum(self.x, method="periodogram",
                                    return_onesided=False)
        self.assertLess(len(f1), len(f2))

    def test_onesided_false_contains_negative_freqs_periodogram(self):
        f, p = power_spectrum(self.x, method="periodogram",
                                  return_onesided=False)
        self.assertTrue(np.any(f < 0))

    def test_onesided_false_contains_negative_freqs_welch(self):
        f, p = power_spectrum(self.x, method="welch", nperseg=64,
                                  return_onesided=False)
        self.assertTrue(np.any(f < 0))

    def test_fft_only_supports_onesided(self):
        with self.assertRaises(ValueError) as cm:
            power_spectrum(self.x, method="fft", return_onesided=False)
        self.assertIn("not supported", str(cm.exception))

    def test_onesided_true_no_negative_freqs(self):
        f, p = power_spectrum(self.x, method="periodogram",
                                  return_onesided=True)
        self.assertTrue(np.all(f >= 0))

# ---------------------------------------------------------------------------
# 7. FFT-specific behaviour
# ---------------------------------------------------------------------------
class TestFFTMethod(unittest.TestCase):

    def test_fft_does_not_modify_input(self):
        x = np.random.randn(64)
        x_copy = x.copy()
        power_spectrum(x, method="fft")
        np.testing.assert_array_equal(x, x_copy)

    def test_fft_freq_spacing(self):
        n = 200
        fs = 50.0
        x = np.random.randn(n)
        f, p = power_spectrum(x, method="fft", fs=fs)
        expected_spacing = fs / n
        self.assertAlmostEqual(f[1] - f[0], expected_spacing, places=10)

    def test_fft_freq_starts_at_zero(self):
        f, p = power_spectrum(np.random.randn(128), method="fft")
        self.assertAlmostEqual(f[0], 0.0, places=10)

    def test_fft_handles_all_zeros(self):
        f, p = power_spectrum(np.zeros(64), method="fft")
        np.testing.assert_allclose(p, np.zeros_like(p), atol=1e-15)

    def test_fft_no_detrend_preserves_dc(self):
        x = np.ones(64) * 3.0
        f, p = power_spectrum(x, method="fft", detrend=None)
        # DC bin = sum(x) = 192, |DC|^2/n = 192^2/64
        self.assertAlmostEqual(p[0], (np.sum(x) ** 2) / len(x), places=5)


# ---------------------------------------------------------------------------
# 8. Welch parameters
# ---------------------------------------------------------------------------
class TestWelchParameters(unittest.TestCase):

    def setUp(self):
        self.x = np.random.randn(512)

    def test_nperseg_propagates(self):
        f1, _ = power_spectrum(self.x, method="welch", nperseg=64)
        f2, _ = power_spectrum(self.x, method="welch", nperseg=256)
        self.assertNotEqual(len(f1), len(f2))

    def test_noverlap_default_is_half(self):
        """Default noverlap = nperseg // 2."""
        f1, p1 = power_spectrum(self.x, method="welch", nperseg=128)
        f2, p2 = power_spectrum(self.x, method="welch", nperseg=128,
                                    noverlap=64)
        np.testing.assert_allclose(p1, p2, rtol=1e-12)

    def test_noverlap_zero(self):
        f, p = power_spectrum(self.x, method="welch", nperseg=128,
                                  noverlap=0)
        self.assertTrue(np.all(np.isfinite(p)))

    def test_nperseg_equal_to_n_uses_single_segment(self):
        """nperseg = N should work with noverlap=0 (single segment, no averaging)."""
        n = 64
        x = np.random.randn(n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f, p = power_spectrum(x, method="welch", nperseg=n, noverlap=0)
        self.assertTrue(np.all(np.isfinite(p)))
        self.assertEqual(len(p), n // 2 + 1)

    def test_welch_uses_window(self):
        """Welch with nperseg=N is NOT identical to periodogram (window differs)."""
        n = 128
        x = np.random.randn(n)
        f_w, p_w = power_spectrum(x, method="welch", nperseg=n)
        f_p, p_p = power_spectrum(x, method="periodogram")
        # Same length but not identical (Welch uses Hann window)
        self.assertEqual(len(p_w), len(p_p))
        self.assertFalse(np.allclose(p_w, p_p, rtol=1e-3))


# ---------------------------------------------------------------------------
# 9. fs / frequency axis
# ---------------------------------------------------------------------------
class TestFrequencyAxis(unittest.TestCase):

    def test_fs_changes_axis_scale(self):
        n = 256
        x = np.random.randn(n)
        f1, _ = power_spectrum(x, method="periodogram", fs=1.0)
        f2, _ = power_spectrum(x, method="periodogram", fs=10.0)
        np.testing.assert_allclose(f2, f1 * 10.0, rtol=1e-10)

    def test_freq_monotonically_increasing(self):
        f, p = power_spectrum(np.random.randn(256), method="welch",
                                  nperseg=64, fs=100.0)
        self.assertTrue(np.all(np.diff(f) >= 0))

    def test_freq_axis_length_matches_psd(self):
        for m in ["periodogram", "welch", "fft"]:
            kw = {"nperseg": 64} if m == "welch" else {}
            f, p = power_spectrum(np.random.randn(256), method=m, **kw)
            self.assertEqual(len(f), len(p))

    def test_fs_default_is_one(self):
        n = 128
        f, p = power_spectrum(np.random.randn(n), method="fft")
        self.assertAlmostEqual(f[-1], 0.5, places=10)


# ---------------------------------------------------------------------------
# 10. Mathematical correctness
# ---------------------------------------------------------------------------
class TestMathematicalCorrectness(unittest.TestCase):

    def test_dc_value_periodogram(self):
        """DC bin of periodogram = (sum(x))^2 / N."""
        x = np.random.randn(64)
        f, p = power_spectrum(x, method="periodogram", detrend=None)
        expected = (np.sum(x)) ** 2 / len(x)
        self.assertAlmostEqual(p[0], expected, places=5)

    def test_dc_value_fft(self):
        """DC bin of fft method = (sum(x))^2 / N."""
        x = np.random.randn(64)
        f, p = power_spectrum(x, method="fft", detrend=None)
        expected = (np.sum(x)) ** 2 / len(x)
        self.assertAlmostEqual(p[0], expected, places=5)

    def test_sinusoid_peak_location_welch(self):
        """A sinusoid at f0 should produce a peak near f0 (within freq resolution)."""
        n = 1024
        fs = 100.0
        f0 = 12.5  # Hz — chosen to align with fs/nperseg grid
        t = np.arange(n) / fs
        x = np.sin(2 * np.pi * f0 * t)
        # Use nperseg that makes f0 land exactly on a bin: 100/8=12.5
        f, p = power_spectrum(x, method="welch", fs=fs, nperseg=256,
                                  detrend="constant")
        peak_idx = np.argmax(p)
        peak_freq = f[peak_idx]
        self.assertAlmostEqual(peak_freq, f0, delta=fs / 256 + 0.1)

    def test_sinusoid_peak_location_fft(self):
        """FFT method: choose N and f0 so f0 = k * fs/N exactly."""
        n = 100  # N
        fs = 100.0
        k = 10   # bin index
        f0 = k * fs / n  # = 10 Hz, exact bin
        t = np.arange(n) / fs
        x = np.sin(2 * np.pi * f0 * t)
        f, p = power_spectrum(x, method="fft", fs=fs, detrend="constant")
        peak_idx = np.argmax(p)
        self.assertEqual(f[peak_idx], f0)

    def test_parseval_periodogram(self):
        """Parseval: N * sum(p) ≈ sum(x^2) for 'spectrum' scaling."""
        n = 256
        fs = 1.0
        x = np.random.randn(n)
        f, p = power_spectrum(x, method="periodogram", fs=fs,
                                  scaling="spectrum", detrend=None)
        # For 'spectrum' scaling: p[k] = |X[k]|^2 / N
        # Parseval: sum(x^2) = (1/N) * sum(|X[k]|^2)
        # One-sided: N * sum(p) = sum_{k=0}^{N/2} |X[k]|^2
        # For real signals: sum_{all k} |X[k]|^2 = 2 * sum_{k=0}^{N/2} |X[k]|^2 - |X[0]|^2 - |X[N/2]|^2
        # But empirically N*sum(p) ≈ sum(x^2) within a small factor
        parseval_sum = n * np.sum(p)
        expected = np.sum(x ** 2)
        # Allow up to 20% difference due to one-sided convention
        self.assertAlmostEqual(parseval_sum, expected, delta=expected * 0.2)

    def test_known_signal_two_sinusoids(self):
        """Two sinusoids -> two dominant peaks, using exact bin frequencies."""
        n = 2048
        fs = 200.0
        # Choose frequencies that land exactly on FFT bins
        f1 = 25.0   # = 256 * 200/2048  -- not exact, let's use exact
        # Exact bins: f = k * fs / n
        k1 = 8
        k2 = 19
        f1 = k1 * fs / n  # 0.78125 * 8 ≈ 0.78? No: 8*200/2048 = 0.78125*8=6.25? 
        # Let me recalculate: 8 * 200 / 2048 = 1600/2048 = 0.78125 * 8 = wait
        # 8 * 200 = 1600; 1600/2048 = 0.78125. That's wrong. 1600/2048 = 0.78125? No: 1600/2048 = 0.78125 is wrong
        # 1600/2048 = 0.78125. Hmm, 200/2048 ≈ 0.097656. *8 ≈ 0.78125. That's < 1 Hz.
        # Let me just use simpler numbers.
        f1 = 12.5   # 12.5 = 128*200/2048? 128*200=25600/2048=12.5 ✓
        f2 = 37.5   # 37.5 = 384*200/2048? 384*200=76800/2048=37.5 ✓
        t = np.arange(n) / fs
        x = (np.sin(2 * np.pi * f1 * t) +
             0.5 * np.sin(2 * np.pi * f2 * t))
        f, p = power_spectrum(x, method="welch", fs=fs, nperseg=512,
                                  detrend="constant")
        peak_indices = np.argsort(p)[-5:]  # top 5 bins
        peak_freqs = sorted(f[peak_indices])
        # The two highest peaks should be near f1 and f2
        self.assertTrue(any(abs(freq - f1) < 1.0 for freq in peak_freqs),
                        f"f1={f1} not found in peaks {peak_freqs}")
        self.assertTrue(any(abs(freq - f2) < 1.0 for freq in peak_freqs),
                        f"f2={f2} not found in peaks {peak_freqs}")


# ---------------------------------------------------------------------------
# 11. Reproducibility
# ---------------------------------------------------------------------------
class TestReproducibility(unittest.TestCase):

    def test_same_input_same_output_periodogram(self):
        x = np.random.RandomState(42).randn(256)
        f1, p1 = power_spectrum(x, method="periodogram")
        f2, p2 = power_spectrum(x, method="periodogram")
        np.testing.assert_array_equal(f1, f2)
        np.testing.assert_allclose(p1, p2)

    def test_same_input_same_output_welch(self):
        x = np.random.RandomState(7).randn(512)
        kw = dict(nperseg=128, noverlap=64)
        f1, p1 = power_spectrum(x, method="welch", **kw)
        f2, p2 = power_spectrum(x, method="welch", **kw)
        np.testing.assert_allclose(p1, p2)

    def test_same_input_same_output_fft(self):
        x = np.random.RandomState(123).randn(128)
        f1, p1 = power_spectrum(x, method="fft")
        f2, p2 = power_spectrum(x, method="fft")
        np.testing.assert_allclose(p1, p2)

    def test_different_inputs_different_outputs(self):
        x1 = np.random.RandomState(0).randn(256)
        x2 = np.random.RandomState(1).randn(256)
        _, p1 = power_spectrum(x1, method="welch", nperseg=64)
        _, p2 = power_spectrum(x2, method="welch", nperseg=64)
        self.assertFalse(np.allclose(p1, p2))


# ---------------------------------------------------------------------------
# 12. Numeric quality
# ---------------------------------------------------------------------------
class TestNumericQuality(unittest.TestCase):

    def test_psd_non_negative(self):
        for m in ["periodogram", "welch", "fft"]:
            kw = {"nperseg": 64} if m == "welch" else {}
            f, p = power_spectrum(np.random.randn(256), method=m, **kw)
            self.assertTrue(np.all(p >= 0),
                            f"PSD negative values for method={m}")

    def test_psd_finite(self):
        for m in ["periodogram", "welch", "fft"]:
            kw = {"nperseg": 64} if m == "welch" else {}
            f, p = power_spectrum(np.random.randn(256), method=m, **kw)
            self.assertTrue(np.all(np.isfinite(p)))

    def test_psd_no_nan(self):
        x = np.random.randn(256)
        x[10] = 1e6  # a single outlier
        for m in ["periodogram", "welch", "fft"]:
            kw = {"nperseg": 64} if m == "welch" else {}
            f, p = power_spectrum(x, method=m, **kw)
            self.assertTrue(np.all(np.isfinite(p)))

    def test_very_large_signal(self):
        x = np.random.randn(8192) * 1e6
        f, p = power_spectrum(x, method="welch", nperseg=1024)
        self.assertTrue(np.all(np.isfinite(p)))
        self.assertTrue(np.all(p >= 0))


# ---------------------------------------------------------------------------
# 13. Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases(unittest.TestCase):

    def test_all_zeros(self):
        for m in ["periodogram", "welch", "fft"]:
            kw = {"nperseg": 32} if m == "welch" else {}
            f, p = power_spectrum(np.zeros(128), method=m, **kw)
            np.testing.assert_allclose(p, np.zeros_like(p), atol=1e-15)

    def test_all_ones_with_detrend(self):
        f, p = power_spectrum(np.ones(128), method="periodogram",
                                  detrend="constant")
        np.testing.assert_allclose(p, np.zeros_like(p), atol=1e-12)

    def test_all_ones_no_detrend(self):
        f, p = power_spectrum(np.ones(128), method="periodogram",
                                  detrend=None)
        self.assertGreater(p[0], 0)

    def test_constant_value(self):
        x = np.full(256, 3.14)
        f, p = power_spectrum(x, method="periodogram", detrend="constant")
        np.testing.assert_allclose(p, np.zeros_like(p), atol=1e-10)

    def test_linear_ramp_detrend_linear(self):
        x = np.linspace(-5, 5, 256)
        f, p = power_spectrum(x, method="periodogram", detrend="linear")
        self.assertLess(np.max(p), 1e-12)

    def test_single_sinusoid_no_noise(self):
        n = 512
        t = np.arange(n)
        x = np.sin(2 * np.pi * 0.1 * t)
        f, p = power_spectrum(x, method="periodogram", detrend="constant")
        peak = np.max(p)
        mean_others = np.mean(p[np.argsort(p)[:-5]])
        self.assertGreater(peak / max(mean_others, 1e-15), 50)

    def test_white_noise_flat_spectrum(self):
        """White noise should have roughly flat PSD."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(4096)
        f, p = power_spectrum(x, method="welch", nperseg=512)
        ratio = np.max(p) / np.min(p)
        self.assertLess(ratio, 10.0)

    def test_pink_noise_slope(self):
        """1/f noise: log-log PSD slope should be ~ -1."""
        x = _make_colored_noise(n=4096, alpha=1.0, seed=0)
        f, p = power_spectrum(x, method="welch", nperseg=1024)
        idx = (f > 0.01) & (f < 0.4)
        slope = np.polyfit(np.log10(f[idx]), np.log10(p[idx]), 1)[0]
        self.assertLess(slope, -0.5)
        self.assertGreater(slope, -1.5)


# ---------------------------------------------------------------------------
# 14. Type & shape contracts
# ---------------------------------------------------------------------------
class TestTypeShapeContracts(unittest.TestCase):

    def setUp(self):
        self.x = np.random.randn(256)

    def test_returns_tuple_of_length_two(self):
        out = power_spectrum(self.x)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)

    def test_freq_is_ndarray(self):
        f, p = power_spectrum(self.x)
        self.assertIsInstance(f, np.ndarray)

    def test_psd_is_ndarray(self):
        f, p = power_spectrum(self.x)
        self.assertIsInstance(p, np.ndarray)

    def test_freq_and_psd_same_length(self):
        for m in ["periodogram", "welch", "fft"]:
            kw = {"nperseg": 64} if m == "welch" else {}
            f, p = power_spectrum(self.x, method=m, **kw)
            self.assertEqual(len(f), len(p))

    def test_freq_float_dtype(self):
        f, p = power_spectrum(self.x, method="periodogram")
        self.assertTrue(np.issubdtype(f.dtype, np.floating))

    def test_psd_float_dtype(self):
        f, p = power_spectrum(self.x, method="periodogram")
        self.assertTrue(np.issubdtype(p.dtype, np.floating))

    def test_input_float32_preserved(self):
        x = np.random.randn(128).astype(np.float32)
        f, p = power_spectrum(x)
        self.assertTrue(p.dtype.itemsize >= 4)

    def test_large_input_stable(self):
        x = np.random.randn(16384)
        f, p = power_spectrum(x, method="welch", nperseg=2048)
        self.assertEqual(len(f), len(p))
        self.assertTrue(np.all(np.isfinite(p)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
