import unittest

import numpy as np

from nolitisea.generate.logistic import logistic


class TestLogisticMap(unittest.TestCase):
    def test_output_shape_default(self):
        X = logistic()
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual(X.ndim, 1)
        self.assertEqual(X.shape, (10000,))

    def test_output_shape_custom_n(self):
        for n in [0, 1, 5, 100, 5000]:
            X = logistic(n=n, discard=10)
            self.assertEqual(X.shape, (n,))

    def test_recurrence_relation(self):
        r = 3.7
        X = logistic(r=r, discard=100, n=50)
        for i in range(len(X) - 1):
            expected = r * X[i] * (1.0 - X[i])
            self.assertAlmostEqual(X[i + 1], expected, places=12)

    def test_reproducibility_same_params(self):
        X1 = logistic(r=3.7, x0=0.3, discard=500, n=2000)
        X2 = logistic(r=3.7, x0=0.3, discard=500, n=2000)
        np.testing.assert_allclose(X1, X2)

    def test_custom_r_parameter(self):
        r = 3.5
        X = logistic(r=r, discard=50, n=30)
        for i in range(len(X) - 1):
            self.assertAlmostEqual(X[i + 1], r * X[i] * (1.0 - X[i]), places=12)

    def test_custom_initial_condition(self):
        X = logistic(r=3.7, x0=0.5, discard=0, n=1)
        expected = 3.7 * 0.5 * (1.0 - 0.5)
        self.assertAlmostEqual(X[0], expected, places=12)

    def test_zero_discard(self):
        r, x0 = 3.7, 0.1
        X = logistic(r=r, x0=x0, discard=0, n=3)
        self.assertAlmostEqual(X[0], r * x0 * (1.0 - x0), places=12)
        self.assertAlmostEqual(X[1], r * X[0] * (1.0 - X[0]), places=12)
        self.assertAlmostEqual(X[2], r * X[1] * (1.0 - X[1]), places=12)

    def test_transient_is_discarded(self):
        X_short = logistic(discard=0, n=105)
        X_discard = logistic(discard=100, n=5)
        np.testing.assert_allclose(X_short[100:105], X_discard)

    def test_bounded_in_unit_interval(self):
        for r in [3.5, 3.7, 3.9]:
            X = logistic(r=r, discard=500, n=2000)
            self.assertTrue(np.all(X > 0.0), f"r={r}: all values should be > 0")
            self.assertTrue(np.all(X < 1.0), f"r={r}: all values should be < 1")

    def test_no_nan_or_inf(self):
        X = logistic(r=3.7, discard=1000, n=10000)
        self.assertFalse(np.any(np.isnan(X)))
        self.assertFalse(np.any(np.isinf(X)))

    def test_n_zero_returns_empty_array(self):
        X = logistic(n=0, discard=10)
        self.assertEqual(X.size, 0)
        self.assertEqual(X.shape, (0,))

    def test_periodic_r_parameter(self):
        # r = 3.2 gives a period-2 orbit
        X = logistic(r=3.2, discard=2000, n=1000)
        # For period-2: x_{n+2} == x_n (approximately)
        diffs = np.abs(X[2:] - X[:-2])
        self.assertLess(np.max(diffs), 1e-6)

    def test_chaotic_r_parameter(self):
        # r = 3.7 is chaotic; consecutive values should differ
        X = logistic(r=3.7, discard=500, n=1000)
        diffs = np.abs(np.diff(X))
        self.assertGreater(np.mean(diffs), 1e-3)


if __name__ == "__main__":
    unittest.main()
