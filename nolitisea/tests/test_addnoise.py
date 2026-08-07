# nolitisea/nolitisea/tests/test_addnoise.py
import unittest
import numpy as np
from nolitisea.utils import (
    add_noise,
    _make_colored_noise,
    _make_harmonic_noise,
    _make_impulse_noise
)


class TestNoiseHelpers(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(seed=42)
        self.n = 1000

    def test_make_colored_noise_basic(self):
        noise = _make_colored_noise(self.rng, n=self.n, alpha=0.0, sigma=2.0)
        self.assertEqual(noise.shape, (self.n,))
        self.assertAlmostEqual(np.mean(noise), 0.0, delta=0.1)
        self.assertAlmostEqual(np.std(noise), 2.0, delta=0.15)

    def test_make_colored_noise_short_n(self):
        noise = _make_colored_noise(self.rng, n=1, alpha=1.0, sigma=1.0)
        np.testing.assert_allclose(noise, np.zeros(1))

    def test_make_harmonic_noise(self):
        amps = [2.0]
        freqs = [0.1]
        noise = _make_harmonic_noise(
            self.rng, n=self.n, amplitudes=amps, frequencies=freqs, phases=[0.0], sigma=None
        )
        self.assertEqual(noise.shape, (self.n,))
        rms = np.sqrt(np.mean(np.square(noise)))
        expected_rms = 2.0 / np.sqrt(2.0)
        self.assertAlmostEqual(rms, expected_rms, delta=0.05)

    def test_make_harmonic_shape_mismatch_raise(self):
        with self.assertRaises(ValueError):
            _make_harmonic_noise(
                self.rng, n=100, amplitudes=[1, 2], frequencies=[0.1], phases=None
            )

    def test_make_impulse_noise_bipolar(self):
        noise = _make_impulse_noise(
            self.rng, n=2000, prob=0.1, amplitude=5.0, bipolar=True)
        self.assertEqual(noise.shape, (2000,))
        impulses = noise[np.abs(noise) > 1e-9]
        unique_vals = np.unique(np.round(impulses, decimals=6))
        self.assertTrue(set(unique_vals).issubset({-5.0, 5.0}))
        frac = np.mean(np.abs(noise) > 1e-9)
        self.assertAlmostEqual(frac, 0.1, delta=0.025)

    def test_make_impulse_noise_unipolar(self):
        noise = _make_impulse_noise(
            self.rng, n=2000, prob=0.1, amplitude=4.0, bipolar=False)
        impulses = noise[noise > 1e-9]
        self.assertTrue(np.allclose(impulses, 4.0))
        frac = np.mean(noise > 1e-9)
        self.assertAlmostEqual(frac, 0.1, delta=0.025)

    def test_make_impulse_zero_prob(self):
        noise = _make_impulse_noise(self.rng, n=100, prob=0.0, amplitude=10)
        np.testing.assert_allclose(noise, np.zeros(100))


class TestAddNoise(unittest.TestCase):
    def setUp(self):
        self.seed = 1234
        self.signal_1d = np.ones(1000, dtype=float)
        self.signal_2d = np.ones((1000, 2), dtype=float)

    def test_input_validation_none(self):
        with self.assertRaises(ValueError):
            add_noise(None)

    def test_input_validation_empty(self):
        with self.assertRaises(ValueError):
            add_noise(np.array([]))

    def test_input_validation_3d(self):
        with self.assertRaises(ValueError):
            add_noise(np.zeros((2, 2, 2)))

    def test_input_validation_bad_noise_type(self):
        with self.assertRaises(ValueError):
            add_noise(self.signal_1d, noise_type="badtype")

    def test_input_validation_negative_level(self):
        with self.assertRaises(ValueError):
            add_noise(self.signal_1d, level=-0.1)

    def test_add_gaussian_1d_relative_constant_signal(self):
        noisy = add_noise(
            self.signal_1d, noise_type="gaussian", level=0.2, absolute=False, seed=self.seed
        )
        self.assertEqual(noisy.shape, self.signal_1d.shape)
        self.assertAlmostEqual(np.std(noisy), 0.0, delta=1e-9)

    def test_add_gaussian_absolute(self):
        noisy = add_noise(
            self.signal_1d, noise_type="gaussian", level=0.5, absolute=True, seed=self.seed
        )
        self.assertEqual(noisy.shape, (1000,))
        self.assertAlmostEqual(np.std(noisy), 0.5, delta=0.08)

    def test_add_uniform_absolute(self):
        noisy = add_noise(
            self.signal_1d, noise_type="uniform", level=1.0, absolute=True, seed=self.seed
        )
        self.assertAlmostEqual(np.std(noisy), 1.0 / np.sqrt(3), delta=0.08)

    def test_add_colored_noise(self):
        noisy = add_noise(
            self.signal_1d, noise_type="colored", level=0.4, absolute=True, alpha=1.0, seed=self.seed
        )
        self.assertEqual(noisy.shape, (1000,))
        self.assertAlmostEqual(np.mean(noisy), 1.0, delta=0.15)

    def test_add_harmonic_noise(self):
        noisy = add_noise(
            self.signal_1d,
            noise_type="harmonic",
            level=1.0,
            absolute=True,
            amplitudes=(2.0,),
            frequencies=(0.05,),
            seed=self.seed
        )
        self.assertEqual(noisy.shape, (1000,))

    def test_impulse_relative_constant_signal_raises(self):
        """impulse absolute=False，常数信号(std≈0)，应当抛出ValueError"""
        with self.assertRaises(ValueError):
            add_noise(
                self.signal_1d,
                noise_type="impulse",
                level=0.0,
                absolute=False,
                impulse_prob=0.08,
                impulse_amplitude=5.0,
                impulse_bipolar=True,
                seed=self.seed
            )

    def test_add_impulse_noise_statistic(self):
        rng_test = np.random.default_rng(self.seed)
        orig_signal = rng_test.normal(loc=0.0, scale=2.0, size=20000)
        orig_copy = orig_signal.copy()
        noisy = add_noise(
            orig_signal,
            noise_type="impulse",
            level=0.0,
            absolute=False,
            impulse_prob=0.08,
            impulse_amplitude=5.0,
            impulse_bipolar=True,
            seed=self.seed
        )
        self.assertEqual(noisy.shape, orig_signal.shape)
        np.testing.assert_allclose(orig_signal, orig_copy)
        mask = np.abs(noisy - orig_signal) > 1e-6
        frac = np.mean(mask)
        self.assertAlmostEqual(frac, 0.08, delta=0.02)

    def test_2d_input_two_columns(self):
        noisy_2d = add_noise(
            self.signal_2d, noise_type="gaussian", level=0.3, absolute=True, seed=self.seed
        )
        self.assertEqual(noisy_2d.shape, self.signal_2d.shape)
        col0, col1 = noisy_2d[:, 0], noisy_2d[:, 1]
        self.assertFalse(np.allclose(col0, col1))

    def test_noise_reproducible_same_seed(self):
        out1 = add_noise(self.signal_1d, noise_type="gaussian",
                         level=0.2, absolute=True, seed=999)
        out2 = add_noise(self.signal_1d, noise_type="gaussian",
                         level=0.2, absolute=True, seed=999)
        np.testing.assert_allclose(out1, out2)


if __name__ == "__main__":
    unittest.main()
