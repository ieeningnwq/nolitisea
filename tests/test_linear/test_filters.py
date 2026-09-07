"""Tests for ``nolitisea.linear.filters``."""

import unittest
from math import factorial

import numpy as np

from nolitisea.linear.filters import (
    _savgol_weights,
    low121,
    notch_filter,
    savitzky_golay,
    wiener_filter,
)


def _savgol_brute(series, window, order, deriv=0):
    """Independent reference: per-window least-squares polynomial fit.

    Edge windows follow interpolation mode (fit on the first/last window,
    evaluate the polynomial at the sample's local coordinate).
    """
    x = np.asarray(series, dtype=float)
    n = x.size
    half = (window - 1) // 2
    pos = np.arange(-half, half + 1, dtype=float)
    design = np.power(pos[:, None], np.arange(order + 1)[None, :])

    out = np.empty(n)
    for i in range(n):
        lo = max(0, min(i - half, n - window))
        coefs, *_ = np.linalg.lstsq(design, x[lo : lo + window], rcond=None)
        p = i - lo - half
        out[i] = sum(
            factorial(j) // factorial(j - deriv) * coefs[j] * p ** (j - deriv)
            for j in range(deriv, order + 1)
        )
    return out


def _bin_amplitude(series, freq, fs):
    """Magnitude of the FFT bin nearest ``freq``."""
    n = series.size
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    spectrum = np.abs(np.fft.rfft(series))
    return spectrum[np.argmin(np.abs(freqs - freq))]


class TestNotchFilter(unittest.TestCase):
    """Notch: rejects the target tone, keeps everything else."""

    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.fs = 200.0
        self.n = 4096
        self.t = np.arange(self.n) / self.fs
        self.f0 = 50.0  # rejected frequency
        self.f1 = 10.0  # frequency that must pass
        self.x = np.sin(2 * np.pi * self.f1 * self.t) + 0.8 * np.sin(
            2 * np.pi * self.f0 * self.t
        )

    def test_rejects_target_frequency(self):
        y = notch_filter(self.x, self.f0, q=30, fs=self.fs)
        ratio = _bin_amplitude(y, self.f0, self.fs) / _bin_amplitude(
            self.x, self.f0, self.fs
        )
        self.assertLess(ratio, 0.02)

    def test_preserves_other_frequencies(self):
        y = notch_filter(self.x, self.f0, q=30, fs=self.fs)
        ratio = _bin_amplitude(y, self.f1, self.fs) / _bin_amplitude(
            self.x, self.f1, self.fs
        )
        self.assertGreater(ratio, 0.98)

    def test_constant_signal_preserved(self):
        x = np.full(200, 3.5)
        y = notch_filter(x, 0.25)
        np.testing.assert_allclose(y, x, atol=1e-10)

    def test_zero_phase_no_lag(self):
        # filtfilt applies the biquad forward and backward: a surviving
        # sinusoid must stay in phase with the input.
        y = notch_filter(self.x, self.f0, q=30, fs=self.fs)
        ref = np.sin(2 * np.pi * self.f1 * self.t)
        interior = slice(100, -100)
        self.assertGreater(np.corrcoef(y[interior], ref[interior])[0, 1], 0.999)

    def test_higher_q_gives_narrower_notch(self):
        # One bin away from the notch centre, a high-Q filter attenuates
        # less than a low-Q filter.
        offset = 2.0
        tone = np.sin(2 * np.pi * (self.f0 + offset) * self.t)
        wide = notch_filter(tone, self.f0, q=5, fs=self.fs)
        narrow = notch_filter(tone, self.f0, q=100, fs=self.fs)
        amp_wide = _bin_amplitude(wide, self.f0 + offset, self.fs)
        amp_narrow = _bin_amplitude(narrow, self.f0 + offset, self.fs)
        self.assertGreater(amp_narrow, amp_wide)

    def test_shape_and_finite(self):
        y = notch_filter(self.x, self.f0, fs=self.fs)
        self.assertEqual(y.shape, self.x.shape)
        self.assertTrue(np.all(np.isfinite(y)))

    def test_accepts_list_input(self):
        y = notch_filter([0.0, 1.0, -1.0, 0.5, 2.0] * 20, 0.1)
        self.assertEqual(y.size, 100)
        self.assertTrue(np.all(np.isfinite(y)))

    def test_short_input_returned_as_copy(self):
        for n in (1, 2):
            x = np.array([1.0, -0.5][:n])
            y = notch_filter(x, 0.25)
            np.testing.assert_array_equal(y, x)

    def test_freq_above_nyquist_raises(self):
        with self.assertRaises(ValueError):
            notch_filter(self.x, self.fs / 2.0, fs=self.fs)

    def test_freq_nonpositive_raises(self):
        with self.assertRaises(ValueError):
            notch_filter(self.x, 0.0, fs=self.fs)

    def test_nonpositive_q_raises(self):
        with self.assertRaises(ValueError):
            notch_filter(self.x, self.f0, q=0.0, fs=self.fs)

    def test_nonpositive_fs_raises(self):
        with self.assertRaises(ValueError):
            notch_filter(self.x, self.f0, fs=-1.0)

    def test_2d_input_raises(self):
        with self.assertRaises(ValueError):
            notch_filter(np.zeros((4, 10)), 0.25)

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            notch_filter(np.array([]), 0.25)


class TestWienerFilter(unittest.TestCase):
    """Frequency-domain Wiener filter."""

    def setUp(self):
        self.rng = np.random.default_rng(11)
        self.n = 4096
        t = np.arange(self.n)
        self.clean = np.sin(2 * np.pi * 0.05 * t)
        self.noise = 0.3 * self.rng.standard_normal(self.n)
        self.noisy = self.clean + self.noise

    def test_zero_noise_var_is_identity(self):
        y = wiener_filter(self.noisy, 0.0)
        np.testing.assert_allclose(y, self.noisy, atol=1e-12)

    def test_large_noise_var_drives_output_to_zero(self):
        y = wiener_filter(self.noisy, 1e6)
        self.assertLess(np.max(np.abs(y)), 1e-6)

    def test_denoises_sine_plus_white_noise(self):
        y = wiener_filter(self.noisy, noise_var=np.var(self.noise))
        mse_noisy = np.mean((self.noisy - self.clean) ** 2)
        mse_filtered = np.mean((y - self.clean) ** 2)
        self.assertLess(mse_filtered, 0.4 * mse_noisy)

    def test_mean_nearly_preserved(self):
        # The DC bin carries far more power than n * noise_var and keeps
        # a gain of ~1 (relative error O(noise_var / (n * mean^2))).
        x = self.noisy + 5.0
        y = wiener_filter(x, noise_var=0.09)
        self.assertAlmostEqual(y.mean(), x.mean(), places=4)

    def test_output_real_same_length_finite(self):
        y = wiener_filter(self.noisy, 0.09)
        self.assertEqual(y.shape, self.noisy.shape)
        self.assertTrue(np.all(np.isreal(y)))
        self.assertTrue(np.all(np.isfinite(y)))

    def test_gain_between_zero_and_one(self):
        # Pure noise: the transfer function clips to [0, 1] and the output
        # energy cannot exceed the input energy.
        x = self.rng.standard_normal(2048)
        y = wiener_filter(x, noise_var=1.0)
        self.assertLessEqual(np.dot(y, y), np.dot(x, x) + 1e-12)

    def test_deterministic(self):
        y1 = wiener_filter(self.noisy, 0.09)
        y2 = wiener_filter(self.noisy, 0.09)
        np.testing.assert_array_equal(y1, y2)

    def test_accepts_list_input(self):
        y = wiener_filter([1.0, -1.0, 0.5, 2.0, -0.5, 1.0, 0.0, -2.0], 0.1)
        self.assertEqual(y.size, 8)
        self.assertTrue(np.all(np.isfinite(y)))

    def test_negative_noise_var_raises(self):
        with self.assertRaises(ValueError):
            wiener_filter(self.noisy, -1.0)

    def test_2d_input_raises(self):
        with self.assertRaises(ValueError):
            wiener_filter(np.zeros((3, 8)), 0.1)

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            wiener_filter(np.array([]), 0.1)


class TestLow121(unittest.TestCase):
    """The 1-2-1 binomial low-pass smoother."""

    def test_constant_signal_preserved(self):
        x = np.full(50, -2.25)
        np.testing.assert_allclose(low121(x), x, atol=1e-14)

    def test_linear_sequence_preserved(self):
        # A three-tap kernel with weights summing to 1 reproduces affine
        # sequences exactly (interior; edges use edge padding).
        x = np.arange(20.0) * 0.7 - 3.0
        y = low121(x)
        np.testing.assert_allclose(y[1:-1], x[1:-1], atol=1e-12)

    def test_matches_direct_formula(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal(30)
        y = low121(x)
        expected = np.empty_like(x)
        padded = np.pad(x, 1, mode="edge")
        for i in range(x.size):
            expected[i] = 0.25 * (padded[i] + 2.0 * padded[i + 1] + padded[i + 2])
        np.testing.assert_allclose(y, expected, atol=1e-14)

    def test_white_noise_variance_reduced_by_3_8(self):
        rng = np.random.default_rng(4)
        x = rng.standard_normal(200000)
        y = low121(x)
        interior = y[100:-100]
        # Interior variance factor: (1/4)^2 + (1/2)^2 + (1/4)^2 = 3/8.
        self.assertAlmostEqual(interior.var() / x.var(), 0.375, delta=0.02)

    def test_low_frequencies_kept_high_frequencies_suppressed(self):
        # Interior gain of the [1, 2, 1]/4 kernel is cos^2(omega/2):
        # ~1 at omega -> 0 and ~0.024 at omega = 2 pi * 0.45.  Interior
        # samples only, since edge padding produces boundary transients.
        n = 4096
        t = np.arange(n)
        low = np.sin(2 * np.pi * 0.01 * t)
        high = np.sin(2 * np.pi * 0.45 * t)
        rms_low = np.sqrt(np.mean(low121(low)[100:-100] ** 2))
        rms_high = np.sqrt(np.mean(low121(high)[100:-100] ** 2))
        self.assertGreater(rms_low / np.sqrt(np.mean(low**2)), 0.99)
        self.assertLess(rms_high / np.sqrt(np.mean(high**2)), 0.05)

    def test_length_preserved_for_short_inputs(self):
        for n in (1, 2, 3):
            x = np.arange(1.0, n + 1.0)
            y = low121(x)
            self.assertEqual(y.size, n)
            self.assertTrue(np.all(np.isfinite(y)))

    def test_accepts_list_input(self):
        y = low121([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(y.size, 5)

    def test_2d_input_raises(self):
        with self.assertRaises(ValueError):
            low121(np.zeros((3, 8)))

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            low121(np.array([]))


class TestSavitzkyGolayWeights(unittest.TestCase):
    """Closed-form Savitzky-Golay convolution coefficients."""

    def test_smoothing_kernel_window5_order2(self):
        # Classic coefficients: [-3, 12, 17, 12, -3] / 35.
        w = _savgol_weights(5, 2, 0)
        np.testing.assert_allclose(
            w[2], np.array([-3.0, 12.0, 17.0, 12.0, -3.0]) / 35.0, atol=1e-12
        )

    def test_first_derivative_kernel_window5_order2(self):
        w = _savgol_weights(5, 2, 1)
        np.testing.assert_allclose(
            w[2], np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) / 10.0, atol=1e-12
        )

    def test_weights_symmetric_for_smoothing(self):
        w = _savgol_weights(7, 3, 0)
        np.testing.assert_allclose(w[3], w[3][::-1], atol=1e-12)

    def test_weights_antisymmetric_for_odd_derivative(self):
        w = _savgol_weights(7, 3, 1)
        np.testing.assert_allclose(w[3], -w[3][::-1], atol=1e-12)


class TestSavitzkyGolayFilter(unittest.TestCase):
    """Behaviour of the public ``savitzky_golay``."""

    def setUp(self):
        self.rng = np.random.default_rng(13)
        self.x = self.rng.standard_normal(60)

    def test_matches_brute_force_reference(self):
        for window, order, deriv in [
            (5, 2, 0),
            (7, 3, 0),
            (11, 4, 0),
            (5, 2, 1),
            (9, 2, 1),
            (11, 4, 1),
            (5, 2, 2),
            (7, 4, 3),
        ]:
            with self.subTest(window=window, order=order, deriv=deriv):
                got = savitzky_golay(self.x, window, order, deriv)
                ref = _savgol_brute(self.x, window, order, deriv)
                np.testing.assert_allclose(got, ref, atol=1e-10)

    def test_matches_reference_on_minimum_length(self):
        # n == window: every sample comes from edge/interpolation slots.
        x = self.rng.standard_normal(7)
        got = savitzky_golay(x, 7, 3, deriv=1)
        ref = _savgol_brute(x, 7, 3, deriv=1)
        np.testing.assert_allclose(got, ref, atol=1e-10)

    def test_constant_signal_preserved(self):
        x = np.full(40, 1.7)
        np.testing.assert_allclose(savitzky_golay(x, 7, 3), x, atol=1e-12)

    def test_polynomials_reproduced_up_to_order(self):
        t = np.arange(-15.0, 16.0)
        linear = 2.0 * t - 4.0
        quadratic = 0.3 * t**2 - t + 1.0
        np.testing.assert_allclose(savitzky_golay(linear, 7, 2), linear, atol=1e-9)
        np.testing.assert_allclose(
            savitzky_golay(quadratic, 7, 2), quadratic, atol=1e-8
        )

    def test_first_derivative_of_linear_is_slope(self):
        t = np.arange(30.0)
        x = 2.5 * t + 1.0
        d = savitzky_golay(x, 7, 3, deriv=1)
        np.testing.assert_allclose(d, 2.5, atol=1e-9)

    def test_second_derivative_of_quadratic(self):
        t = np.arange(-12.0, 13.0)
        x = 3.0 * t**2 - 2.0 * t + 5.0
        d2 = savitzky_golay(x, 7, 3, deriv=2)
        np.testing.assert_allclose(d2[3:-3], 6.0, atol=1e-8)

    def test_smoothing_reduces_noise_variance(self):
        x = self.rng.standard_normal(10000)
        y = savitzky_golay(x, 11, 3)
        self.assertLess(y[20:-20].var(), 0.5 * x.var())

    def test_sine_derivative_amplitude(self):
        # Per-sample derivative of sin(2 pi f t): amplitude 2 pi f.
        n = 4096
        t = np.arange(n)
        f = 0.05
        d = savitzky_golay(np.sin(2 * np.pi * f * t), 11, 3, deriv=1)
        amplitude = np.sqrt(np.mean(d[50:-50] ** 2)) * np.sqrt(2)
        self.assertAlmostEqual(amplitude, 2 * np.pi * f, delta=0.05)

    def test_length_preserved_and_finite(self):
        y = savitzky_golay(self.x, 7, 3)
        self.assertEqual(y.shape, self.x.shape)
        self.assertTrue(np.all(np.isfinite(y)))

    def test_deterministic(self):
        a = savitzky_golay(self.x, 7, 3, deriv=1)
        b = savitzky_golay(self.x, 7, 3, deriv=1)
        np.testing.assert_array_equal(a, b)

    def test_accepts_list_input(self):
        y = savitzky_golay(list(range(15)), 5, 2)
        self.assertEqual(y.size, 15)

    def test_even_window_raises(self):
        with self.assertRaises(ValueError):
            savitzky_golay(self.x, 6, 2)

    def test_nonpositive_window_raises(self):
        with self.assertRaises(ValueError):
            savitzky_golay(self.x, 0, 2)

    def test_order_not_smaller_than_window_raises(self):
        with self.assertRaises(ValueError):
            savitzky_golay(self.x, 5, 5)
        with self.assertRaises(ValueError):
            savitzky_golay(self.x, 5, 6)

    def test_negative_order_raises(self):
        with self.assertRaises(ValueError):
            savitzky_golay(self.x, 5, -1)

    def test_deriv_above_order_raises(self):
        with self.assertRaises(ValueError):
            savitzky_golay(self.x, 7, 2, deriv=3)

    def test_negative_deriv_raises(self):
        with self.assertRaises(ValueError):
            savitzky_golay(self.x, 7, 2, deriv=-1)

    def test_input_shorter_than_window_raises(self):
        with self.assertRaises(ValueError):
            savitzky_golay(np.ones(4), 7, 3)

    def test_2d_input_raises(self):
        with self.assertRaises(ValueError):
            savitzky_golay(np.zeros((3, 8)), 7, 3)

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            savitzky_golay(np.array([]), 7, 3)


if __name__ == "__main__":
    unittest.main()
