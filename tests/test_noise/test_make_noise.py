import unittest

import numpy as np

from nolitisea.noise.make_noise import make_noise


class TestMakeNoiseGaussian(unittest.TestCase):
    """Gaussian noise tests."""

    def setUp(self):
        rng = np.random.default_rng(0)
        # Distinct scales per column to expose per-column normalisation.
        self.data = rng.normal(size=(100_000, 3)) * [1.0, 10.0, 100.0]

    def test_shape_and_dtype(self):
        noisy = make_noise(self.data, level=0.1, seed=0)
        self.assertEqual(noisy.shape, self.data.shape)
        self.assertEqual(noisy.dtype, np.float64)

    def test_input_not_modified(self):
        snapshot = self.data.copy()
        make_noise(self.data, level=0.1, seed=0)
        np.testing.assert_array_equal(self.data, snapshot)

    def test_relative_level_per_column(self):
        """Noise std of column j is level * std(column j)."""
        level = 0.05
        noisy = make_noise(self.data, level=level, seed=42)
        resid = noisy - self.data
        expected = level * self.data.std(axis=0)
        np.testing.assert_allclose(
            resid.std(axis=0), expected, rtol=0.02
        )

    def test_absolute_level(self):
        level = 2.0
        noisy = make_noise(self.data, level=level, absolute=True, seed=42)
        resid = noisy - self.data
        np.testing.assert_allclose(
            resid.std(axis=0), np.full(3, level), rtol=0.02
        )

    def test_seed_reproducibility(self):
        a = make_noise(self.data, level=0.1, seed=7)
        b = make_noise(self.data, level=0.1, seed=7)
        c = make_noise(self.data, level=0.1, seed=8)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_zero_level_is_identity(self):
        noisy = make_noise(self.data, level=0.0, seed=0)
        np.testing.assert_array_equal(noisy, self.data)

    def test_1d_input_stays_1d(self):
        x = np.arange(1000.0)
        noisy = make_noise(x, level=0.1, seed=0)
        self.assertEqual(noisy.ndim, 1)
        self.assertEqual(noisy.shape, x.shape)

    def test_accepts_noise_type_case_insensitive(self):
        a = make_noise(self.data, level=0.1, noise_type="GAUSSIAN", seed=3)
        b = make_noise(self.data, level=0.1, noise_type="gaussian", seed=3)
        np.testing.assert_array_equal(a, b)


class TestMakeNoiseUniform(unittest.TestCase):
    """Uniform noise tests."""

    def setUp(self):
        rng = np.random.default_rng(1)
        self.data = rng.normal(size=(100_000, 2)) * [1.0, 5.0]

    def test_relative_level_per_column(self):
        """Noise std of column j is level * std(column j)."""
        level = 0.1
        noisy = make_noise(
            self.data, level=level, noise_type="uniform", seed=42
        )
        resid = noisy - self.data
        expected = level * self.data.std(axis=0)
        np.testing.assert_allclose(
            resid.std(axis=0), expected, rtol=0.02
        )

    def test_noise_bounded_by_support(self):
        """Uniform noise never exceeds sqrt(3) * sigma in magnitude."""
        level = 0.2
        noise = (
            make_noise(self.data, level=level, noise_type="uniform", seed=0)
            - self.data
        )
        bound = np.sqrt(3.0) * level * self.data.std(axis=0)
        self.assertTrue(np.all(np.abs(noise) <= bound))

    def test_absolute_level(self):
        level = 1.5
        noise = (
            make_noise(
                self.data, level=level, noise_type="uniform",
                absolute=True, seed=42,
            )
            - self.data
        )
        np.testing.assert_allclose(
            noise.std(axis=0), np.full(2, level), rtol=0.02
        )
        self.assertTrue(np.all(np.abs(noise) <= np.sqrt(3.0) * level))

    def test_zero_mean(self):
        noise = (
            make_noise(self.data, level=0.1, noise_type="uniform", seed=5)
            - self.data
        )
        self.assertTrue(np.all(np.abs(noise.mean(axis=0)) < 0.01))

    def test_seed_reproducibility(self):
        a = make_noise(self.data, level=0.1, noise_type="uniform", seed=7)
        b = make_noise(self.data, level=0.1, noise_type="uniform", seed=7)
        c = make_noise(self.data, level=0.1, noise_type="uniform", seed=8)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_types_differ(self):
        """Gaussian and uniform draws with the same seed differ."""
        a = make_noise(self.data, level=0.1, noise_type="gaussian", seed=7)
        b = make_noise(self.data, level=0.1, noise_type="uniform", seed=7)
        self.assertFalse(np.array_equal(a, b))


class TestMakeNoiseErrors(unittest.TestCase):
    """Error-condition tests."""

    def setUp(self):
        self.data = np.ones((10, 2))

    def test_negative_level(self):
        with self.assertRaises(ValueError) as cm:
            make_noise(self.data, level=-0.1)
        self.assertIn(">= 0", str(cm.exception))

    def test_unknown_noise_type(self):
        with self.assertRaises(ValueError) as cm:
            make_noise(self.data, noise_type="pink")
        self.assertIn("gaussian", str(cm.exception))

    def test_3d_input(self):
        with self.assertRaises(ValueError) as cm:
            make_noise(np.zeros((2, 2, 2)))
        self.assertIn("1-D", str(cm.exception))

    def test_empty_input(self):
        with self.assertRaises(ValueError) as cm:
            make_noise(np.array([]))
        self.assertIn("at least one sample", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
