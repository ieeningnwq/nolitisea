import unittest

import numpy as np

from nolitisea.generate.ikeda import ikeda


class TestIkedaMap(unittest.TestCase):
    def test_output_shape_default(self):
        x = ikeda()
        self.assertIsInstance(x, np.ndarray)
        self.assertEqual(x.ndim, 2)
        self.assertEqual(x.shape, (10000, 2))

    def test_output_shape_custom_length(self):
        for length in [0, 1, 5, 100, 5000]:
            x = ikeda(length=length, discard=10)
            self.assertEqual(x.shape, (length, 2))

    def test_recurrence_relation(self):
        alpha, beta, gamma, mu = 6.0, 0.4, 1.0, 0.9
        x = ikeda(alpha=alpha, beta=beta, gamma=gamma, mu=mu,
                  discard=100, length=50)
        for i in range(len(x) - 1):
            xi, yi = x[i]
            phi = beta - alpha / (1 + xi ** 2 + yi ** 2)
            expected_x = gamma + mu * (xi * np.cos(phi) - yi * np.sin(phi))
            expected_y = mu * (xi * np.sin(phi) + yi * np.cos(phi))
            self.assertAlmostEqual(x[i + 1, 0], expected_x, places=12)
            self.assertAlmostEqual(x[i + 1, 1], expected_y, places=12)

    def test_reproducibility_same_x0(self):
        x0 = np.array([0.1, -0.1])
        x1 = ikeda(x0=x0, discard=100, length=500)
        x2 = ikeda(x0=x0, discard=100, length=500)
        np.testing.assert_allclose(x1, x2)

    def test_custom_initial_condition(self):
        x0 = np.array([0.5, -0.3])
        x = ikeda(x0=x0, discard=0, length=1)
        # x[0] is the initial condition itself (not one iteration from it)
        np.testing.assert_allclose(x[0], x0)

    def test_zero_discard(self):
        x0 = np.array([0.2, 0.1])
        x = ikeda(x0=x0, discard=0, length=3)
        alpha, beta, gamma, mu = 6.0, 0.4, 1.0, 0.9
        # x[0] is the initial condition
        np.testing.assert_allclose(x[0], x0)
        # x[1] is one iteration from x0
        phi0 = beta - alpha / (1 + x0[0] ** 2 + x0[1] ** 2)
        expected_x1 = gamma + mu * (x0[0] * np.cos(phi0) - x0[1] * np.sin(phi0))
        expected_y1 = mu * (x0[0] * np.sin(phi0) + x0[1] * np.cos(phi0))
        self.assertAlmostEqual(x[1, 0], expected_x1, places=12)
        self.assertAlmostEqual(x[1, 1], expected_y1, places=12)
        # x[2] follows from x[1]
        xi, yi = x[1]
        phi1 = beta - alpha / (1 + xi ** 2 + yi ** 2)
        expected_x2 = gamma + mu * (xi * np.cos(phi1) - yi * np.sin(phi1))
        expected_y2 = mu * (xi * np.sin(phi1) + yi * np.cos(phi1))
        self.assertAlmostEqual(x[2, 0], expected_x2, places=12)
        self.assertAlmostEqual(x[2, 1], expected_y2, places=12)

    def test_transient_is_discarded(self):
        x0 = np.array([0.1, -0.1])
        x_short = ikeda(x0=x0, discard=0, length=105)
        x_discard = ikeda(x0=x0, discard=100, length=5)
        np.testing.assert_allclose(x_short[100:105], x_discard)

    def test_custom_parameters(self):
        alpha, beta, gamma, mu = 5.0, 0.5, 0.8, 0.85
        x = ikeda(alpha=alpha, beta=beta, gamma=gamma, mu=mu,
                  discard=50, length=30)
        for i in range(len(x) - 1):
            xi, yi = x[i]
            phi = beta - alpha / (1 + xi ** 2 + yi ** 2)
            expected_x = gamma + mu * (xi * np.cos(phi) - yi * np.sin(phi))
            expected_y = mu * (xi * np.sin(phi) + yi * np.cos(phi))
            self.assertAlmostEqual(x[i + 1, 0], expected_x, places=12)
            self.assertAlmostEqual(x[i + 1, 1], expected_y, places=12)

    def test_no_nan_or_inf(self):
        x = ikeda(discard=500, length=10000)
        self.assertFalse(np.any(np.isnan(x)))
        self.assertFalse(np.any(np.isinf(x)))

    def test_zero_length_returns_empty(self):
        x = ikeda(length=0, discard=10)
        self.assertEqual(x.size, 0)
        self.assertEqual(x.shape, (0, 2))

    def test_random_initial_conditions(self):
        # Two calls with x0=None may differ due to randomness,
        # but both should produce valid bounded output
        np.random.seed(0)
        x1 = ikeda(discard=100, length=100)
        np.random.seed(1)
        x2 = ikeda(discard=100, length=100)
        self.assertEqual(x1.shape, (100, 2))
        self.assertEqual(x2.shape, (100, 2))
        self.assertFalse(np.any(np.isnan(x1)))
        self.assertFalse(np.any(np.isnan(x2)))


if __name__ == "__main__":
    unittest.main()
