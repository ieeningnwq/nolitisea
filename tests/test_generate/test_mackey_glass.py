import unittest

import numpy as np

from nolitisea.generate.mackey_glass import mackey_glass


class TestMackeyGlass(unittest.TestCase):
    def test_output_shape_default(self):
        x = mackey_glass(length=1000, discard=10)
        self.assertIsInstance(x, np.ndarray)
        self.assertEqual(x.ndim, 1)
        self.assertEqual(x.shape, (1000,))

    def test_output_shape_custom_length(self):
        for length in [1, 5, 100, 500]:
            x = mackey_glass(length=length, discard=5)
            self.assertEqual(x.shape, (length,))

    def test_initial_condition_used(self):
        n = 10
        x0 = np.linspace(0.4, 0.6, n)
        x = mackey_glass(length=n, x0=x0, discard=0,
                         sample=2.5, n=n)  # internal sample = 1
        # With discard=0 and sample=1, the first n points are the initial cond.
        np.testing.assert_allclose(x[:n], x0)

    def test_custom_parameters(self):
        a, b, c, tau = 0.3, 0.05, 8.0, 17.0
        n = 10
        x0 = np.full(n, 0.5)
        x = mackey_glass(length=200, x0=x0, a=a, b=b, c=c, tau=tau,
                         n=n, sample=2.0, discard=0)
        self.assertEqual(x.shape, (200,))
        self.assertFalse(np.any(np.isnan(x)))
        self.assertFalse(np.any(np.isinf(x)))

    def test_no_nan_or_inf(self):
        x = mackey_glass(length=5000, discard=10)
        self.assertFalse(np.any(np.isnan(x)))
        self.assertFalse(np.any(np.isinf(x)))

    def test_positive_values(self):
        # Mackey-Glass series is typically positive for these parameters
        x = mackey_glass(length=5000, discard=10)
        self.assertTrue(np.all(x > 0))

    def test_bounded_range(self):
        # For default chaotic parameters the series stays bounded
        x = mackey_glass(length=10000, discard=50)
        self.assertLess(x.max(), 10.0)
        self.assertGreater(x.min(), 0.0)

    def test_chaotic_fluctuations(self):
        # The chaotic Mackey-Glass series should have non-trivial variance
        x = mackey_glass(length=5000, discard=50)
        self.assertGreater(np.std(x), 0.1)
        # Consecutive values should differ (not a fixed point)
        self.assertGreater(np.mean(np.abs(np.diff(x))), 1e-3)

    def test_subsampling_reduces_length(self):
        """Coarser sampling should produce fewer output points."""
        n = 10
        x0 = np.full(n, 0.5)
        x_fine = mackey_glass(length=100, x0=x0, discard=0, sample=2.5, n=n)
        x_coarse = mackey_glass(length=50, x0=x0, discard=0, sample=5.0, n=n)
        self.assertEqual(x_fine.shape, (100,))
        self.assertEqual(x_coarse.shape, (50,))
        self.assertFalse(np.any(np.isnan(x_fine)))
        self.assertFalse(np.any(np.isnan(x_coarse)))


if __name__ == "__main__":
    unittest.main()
