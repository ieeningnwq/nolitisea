import unittest

import numpy as np

from nolitisea.noise.add_noise import add_noise


class TestAddNoiseGaussian(unittest.TestCase):
    """Gaussian noise tests."""

    def setUp(self):
        rng = np.random.default_rng(0)
        # Distinct scales per column to expose per-column normalisation.
        self.data = rng.normal(size=(50_000, 3)) * [1.0, 10.0, 100.0]

    def test_shape_and_dtype(self):
        noisy = add_noise(self.data, noise_type="gaussian", level=0.1, seed=0)
        self.assertEqual(noisy.shape, self.data.shape)
        self.assertEqual(noisy.dtype, np.float64)

    def test_input_not_modified(self):
        snapshot = self.data.copy()
        add_noise(self.data, noise_type="gaussian", level=0.1, seed=0)
        np.testing.assert_array_equal(self.data, snapshot)

    def test_relative_level_per_column(self):
        """Noise std of column j is level * std(column j)."""
        level = 0.1
        noisy = add_noise(self.data, noise_type="gaussian", level=level, seed=42)
        resid = noisy - self.data
        np.testing.assert_allclose(
            resid.std(axis=0), level * self.data.std(axis=0), rtol=0.03
        )

    def test_absolute_level(self):
        noisy = add_noise(
            self.data, noise_type="gaussian", level=2.0, absolute=True, seed=42
        )
        resid = noisy - self.data
        np.testing.assert_allclose(
            resid.std(axis=0), np.full(3, 2.0), rtol=0.03
        )

    def test_seed_reproducibility(self):
        a = add_noise(self.data, noise_type="gaussian", level=0.1, seed=7)
        b = add_noise(self.data, noise_type="gaussian", level=0.1, seed=7)
        c = add_noise(self.data, noise_type="gaussian", level=0.1, seed=8)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_accepts_1d_input(self):
        x = np.arange(1000.0)
        noisy = add_noise(x, noise_type="gaussian", level=0.0, seed=0)
        self.assertEqual(noisy.shape, x.shape)
        np.testing.assert_array_equal(noisy, x)


class TestAddNoiseUniform(unittest.TestCase):
    """Uniform noise tests (level is the half-width of [-a, a])."""

    def setUp(self):
        rng = np.random.default_rng(1)
        self.data = rng.normal(size=(50_000, 2)) * [1.0, 5.0]

    def test_bounded_by_level_times_std(self):
        level = 0.2
        resid = (
            add_noise(self.data, noise_type="uniform", level=level, seed=0)
            - self.data
        )
        bound = level * self.data.std(axis=0)
        self.assertTrue(np.all(np.abs(resid) <= bound))
        # A strict bound check alone could pass trivially; verify the
        # noise actually reaches into the outer part of the support.
        self.assertTrue(np.all(np.abs(resid).max(axis=0) > 0.98 * bound))

    def test_relative_std_is_level_times_std_over_sqrt3(self):
        """std of U(-a, a) is a / sqrt(3)."""
        level = 0.3
        resid = (
            add_noise(self.data, noise_type="uniform", level=level, seed=42)
            - self.data
        )
        expected = level * self.data.std(axis=0) / np.sqrt(3.0)
        np.testing.assert_allclose(resid.std(axis=0), expected, rtol=0.03)

    def test_absolute_bounds(self):
        level = 1.5
        resid = (
            add_noise(
                self.data, noise_type="uniform", level=level,
                absolute=True, seed=42,
            )
            - self.data
        )
        self.assertTrue(np.all(np.abs(resid) <= level))
        np.testing.assert_allclose(
            resid.std(axis=0), np.full(2, level / np.sqrt(3.0)), rtol=0.03
        )

    def test_zero_mean(self):
        resid = (
            add_noise(self.data, noise_type="uniform", level=0.1, seed=5)
            - self.data
        )
        self.assertTrue(np.all(np.abs(resid.mean(axis=0)) < 0.02))


class TestAddNoiseColored(unittest.TestCase):
    """Colored (1/f^alpha) noise tests."""

    def setUp(self):
        self.data = np.random.default_rng(2).normal(size=40_000)

    def test_relative_std(self):
        level = 0.1
        resid = (
            add_noise(self.data, noise_type="colored", level=level, seed=42)
            - self.data
        )
        self.assertAlmostEqual(
            resid.std(), level * self.data.std(), delta=0.02 * level * self.data.std()
        )

    def test_psd_slope_matches_alpha(self):
        """PSD of the added noise decays as 1/f^alpha."""
        n = 2 ** 15
        w = np.random.default_rng(3).normal(size=n)
        alpha = 2.0
        resid = (
            add_noise(w, noise_type="colored", level=5.0, alpha=alpha, seed=5)
            - w
        )
        freq = np.fft.rfftfreq(n)[1:]
        psd = np.abs(np.fft.rfft(resid)[1:]) ** 2
        slope = np.polyfit(np.log(freq), np.log(psd), 1)[0]
        self.assertAlmostEqual(slope, -alpha, delta=0.2)


class TestAddNoiseHarmonic(unittest.TestCase):
    """Harmonic (sum-of-sinusoids) noise tests."""

    def setUp(self):
        t = np.arange(10_000)
        self.t = t
        self.signal = np.sin(2.0 * np.pi * 0.01 * t)

    def test_amplitude_scaling_relative(self):
        """Peak amplitude of the added tone is level * std(signal)."""
        level = 0.5
        resid = (
            add_noise(
                self.signal, noise_type="harmonic", level=level,
                amplitudes=(1.0,), frequencies=(0.05,), seed=2,
            )
            - self.signal
        )
        expected_amp = level * self.signal.std()
        self.assertAlmostEqual(resid.max(), expected_amp, delta=0.05 * expected_amp)
        # std of a sine is amplitude / sqrt(2).
        self.assertAlmostEqual(
            resid.std(), expected_amp / np.sqrt(2.0), delta=0.02 * expected_amp
        )

    def test_frequency_content(self):
        resid = (
            add_noise(
                self.signal, noise_type="harmonic", level=0.5,
                amplitudes=(1.0,), frequencies=(0.05,), seed=2,
            )
            - self.signal
        )
        freqs = np.fft.rfftfreq(len(self.t))
        peak = freqs[np.argmax(np.abs(np.fft.rfft(resid)))]
        self.assertAlmostEqual(peak, 0.05, places=3)

    def test_explicit_phases_respected(self):
        resid = (
            add_noise(
                self.signal, noise_type="harmonic", level=0.5,
                amplitudes=(1.0,), frequencies=(0.05,), phases=(0.0,), seed=2,
            )
            - self.signal
        )
        # A sine with zero initial phase is ~0 at t = 0.
        self.assertAlmostEqual(resid[0], 0.0, delta=1e-9)


class TestAddNoiseImpulse(unittest.TestCase):
    """Impulse (salt-and-pepper) noise tests."""

    def setUp(self):
        rng = np.random.default_rng(4)
        self.data = rng.normal(size=(20_000, 2)) * [1.0, 10.0]

    def test_corruption_fraction(self):
        prob = 0.1
        noisy = add_noise(
            self.data, noise_type="impulse",
            impulse_prob=prob, impulse_amplitude=3.0, seed=4,
        )
        frac = (noisy != self.data).mean(axis=0)
        np.testing.assert_allclose(frac, prob, atol=0.01)

    def test_impulse_values(self):
        amp = 3.0
        noisy = add_noise(
            self.data, noise_type="impulse",
            impulse_prob=0.2, impulse_amplitude=amp, seed=4,
        )
        for j, s in enumerate(self.data.std(axis=0)):
            mask = noisy[:, j] != self.data[:, j]
            col_diffs = set(
                np.round((noisy[:, j] - self.data[:, j])[mask], 6).tolist()
            )
            self.assertEqual(col_diffs, {round(-amp * s, 6), round(amp * s, 6)})

    def test_salt_noise_positive_only(self):
        noisy = add_noise(
            self.data[:, 0], noise_type="impulse",
            impulse_prob=0.2, impulse_amplitude=3.0,
            impulse_bipolar=False, seed=4,
        )
        diffs = (noisy - self.data[:, 0])[noisy != self.data[:, 0]]
        self.assertTrue(np.all(diffs > 0))

    def test_level_is_unused_for_impulse(self):
        a = add_noise(
            self.data, noise_type="impulse", level=0.05,
            impulse_prob=0.1, impulse_amplitude=3.0, seed=4,
        )
        b = add_noise(
            self.data, noise_type="impulse", level=0.95,
            impulse_prob=0.1, impulse_amplitude=3.0, seed=4,
        )
        np.testing.assert_array_equal(a, b)

    def test_constant_signal_requires_absolute(self):
        with self.assertRaises(ValueError) as cm:
            add_noise(np.ones(100), noise_type="impulse")
        self.assertIn("absolute=True", str(cm.exception))

    def test_constant_signal_with_absolute_amplitude(self):
        noisy = add_noise(
            np.ones(1000), noise_type="impulse",
            impulse_prob=0.5, level=2.0, absolute=True, seed=0,
        )
        mask = noisy != 1.0
        self.assertTrue(mask.any())
        self.assertTrue(np.all(np.isin(noisy[mask], [1.0 + 2.0, 1.0 - 2.0])))


class TestAddNoiseErrors(unittest.TestCase):
    """Error-condition tests."""

    def setUp(self):
        self.data = np.ones((10, 2))

    def test_none_input(self):
        with self.assertRaises(ValueError):
            add_noise(None)

    def test_empty_input(self):
        with self.assertRaises(ValueError):
            add_noise(np.array([]))

    def test_3d_input(self):
        with self.assertRaises(ValueError):
            add_noise(np.zeros((2, 2, 2)))

    def test_unknown_noise_type(self):
        with self.assertRaises(ValueError):
            add_noise(self.data, noise_type="pink")

    def test_negative_level(self):
        with self.assertRaises(ValueError):
            add_noise(self.data, level=-0.1)


if __name__ == "__main__":
    unittest.main()
