import unittest

import numpy as np

from nolitisea.generate import henon


class TestHenonMap(unittest.TestCase):

    def test_output_shapes_default(self):
        X, Y = henon()
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(Y, np.ndarray)
        self.assertEqual(X.ndim, 1)
        self.assertEqual(Y.ndim, 1)
        self.assertEqual(X.shape, (10000,))
        self.assertEqual(Y.shape, (10000,))
        self.assertEqual(X.shape, Y.shape)

    def test_output_shapes_custom_n(self):
        for n in [0, 1, 5, 100, 5000]:
            X, Y = henon(n=n, discard=10)
            self.assertEqual(X.shape, (n,))
            self.assertEqual(Y.shape, (n,))

    def test_recurrence_relation(self):
        a, b = 1.4, 0.3
        X, Y = henon(a=a, b=b, discard=100, n=50)
        for i in range(len(X) - 1):
            expected_x_next = a - X[i] ** 2 + b * Y[i]
            expected_y_next = X[i]
            self.assertAlmostEqual(X[i + 1], expected_x_next, places=12)
            self.assertAlmostEqual(Y[i + 1], expected_y_next, places=12)

    def test_reproducibility_same_params(self):
        X1, Y1 = henon(a=1.4, b=0.3, x0=0.0, y0=0.0, discard=500, n=2000)
        X2, Y2 = henon(a=1.4, b=0.3, x0=0.0, y0=0.0, discard=500, n=2000)
        np.testing.assert_allclose(X1, X2)
        np.testing.assert_allclose(Y1, Y2)

    def test_custom_ab_parameters(self):
        a, b = 1.2, 0.2
        X, Y = henon(a=a, b=b, discard=50, n=30)
        for i in range(len(X) - 1):
            self.assertAlmostEqual(X[i + 1], a - X[i] ** 2 + b * Y[i], places=12)
            self.assertAlmostEqual(Y[i + 1], X[i], places=12)

    def test_custom_initial_conditions(self):
        X1, Y1 = henon(x0=0.5, y0=-0.2, discard=0, n=1)
        a, b = 1.4, 0.3
        expected_x1 = a - 0.5 ** 2 + b * (-0.2)
        expected_y1 = 0.5
        self.assertAlmostEqual(X1[0], expected_x1, places=12)
        self.assertAlmostEqual(Y1[0], expected_y1, places=12)

    def test_zero_discard(self):
        a, b = 1.4, 0.3
        x0, y0 = 0.1, 0.2
        X, Y = henon(a=a, b=b, x0=x0, y0=y0, discard=0, n=3)
        self.assertAlmostEqual(X[0], a - x0 ** 2 + b * y0, places=12)
        self.assertAlmostEqual(Y[0], x0, places=12)
        self.assertAlmostEqual(X[1], a - X[0] ** 2 + b * Y[0], places=12)
        self.assertAlmostEqual(Y[1], X[0], places=12)

    def test_transient_is_discarded(self):
        X_short, Y_short = henon(discard=0, n=105)
        X_discard, Y_discard = henon(discard=100, n=5)
        np.testing.assert_allclose(X_short[100:105], X_discard)
        np.testing.assert_allclose(Y_short[100:105], Y_discard)

    def test_no_nan_or_inf(self):
        X, Y = henon(a=1.4, b=0.3, discard=1000, n=10000)
        self.assertFalse(np.any(np.isnan(X)))
        self.assertFalse(np.any(np.isnan(Y)))
        self.assertFalse(np.any(np.isinf(X)))
        self.assertFalse(np.any(np.isinf(Y)))

    def test_n_zero_returns_empty_arrays(self):
        X, Y = henon(n=0, discard=10)
        self.assertEqual(X.size, 0)
        self.assertEqual(Y.size, 0)
        self.assertEqual(X.shape, (0,))
        self.assertEqual(Y.shape, (0,))

    def test_alternate_parameters_chaotic_region(self):
        for a, b in [(1.4, 0.3), (1.3, 0.35), (1.45, 0.25)]:
            X, Y = henon(a=a, b=b, discard=500, n=1000)
            self.assertEqual(X.shape, (1000,))
            self.assertEqual(Y.shape, (1000,))
            self.assertFalse(np.any(np.isnan(X)))
            self.assertFalse(np.any(np.isnan(Y)))


if __name__ == "__main__":
    unittest.main()
