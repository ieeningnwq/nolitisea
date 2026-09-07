import unittest

import numpy as np

from nolitisea.generate.roessler import roessler


def _roessler_rhs(x, a, b, c):
    """Rossler ODE right-hand side."""
    return np.array([
        -(x[1] + x[2]),
        x[0] + a * x[1],
        b + x[2] * (x[0] - c),
    ])


class TestRoesslerOscillator(unittest.TestCase):
    def test_output_shape_default(self):
        t, x = roessler(length=100, discard=10)
        self.assertIsInstance(t, np.ndarray)
        self.assertIsInstance(x, np.ndarray)
        self.assertEqual(t.shape, (100,))
        self.assertEqual(x.shape, (100, 3))

    def test_output_shape_custom_length(self):
        for length in [1, 5, 50, 200]:
            t, x = roessler(length=length, discard=5)
            self.assertEqual(t.shape, (length,))
            self.assertEqual(x.shape, (length, 3))

    def test_custom_initial_conditions(self):
        x0 = [1.0, 2.0, 3.0]
        t, x = roessler(length=50, x0=x0, discard=0,
                        step=0.001, sample=0.001)
        self.assertEqual(x.shape, (50, 3))
        # With discard=0 and sample=step, first point should be close to x0
        np.testing.assert_allclose(x[0], x0, atol=1e-6)

    def test_ode_consistency(self):
        """Finite-difference check that the trajectory satisfies the Rossler ODE.

        The finite difference (x[i+1] - x[i]) / dt approximates the time
        derivative at the midpoint of the interval, so the analytical RHS is
        evaluated at (x[i] + x[i+1]) / 2 for a second-order comparison.
        """
        a, b, c = 0.2, 0.2, 5.7
        x0 = [-9.0, 0.0, 0.0]
        step = 0.001
        # sample=step so internal sample = 1 (every integration point)
        t, x = roessler(length=1000, x0=x0, a=a, b=b, c=c,
                        step=step, sample=step, discard=0)

        dt = t[1] - t[0]
        for i in range(len(x) - 1):
            dxdt_num = (x[i + 1] - x[i]) / dt
            midpoint = (x[i] + x[i + 1]) / 2.0
            dxdt_ana = _roessler_rhs(midpoint, a, b, c)
            np.testing.assert_allclose(dxdt_num, dxdt_ana, rtol=1e-4, atol=1e-4)

    def test_reproducibility_same_x0(self):
        x0 = (-9.0, 0.0, 0.0)
        t1, x1 = roessler(length=100, x0=x0, discard=5)
        t2, x2 = roessler(length=100, x0=x0, discard=5)
        np.testing.assert_allclose(x1, x2)
        np.testing.assert_allclose(t1, t2)

    def test_custom_parameters(self):
        a, b, c = 0.15, 0.25, 6.0
        try:
            t, x = roessler(length=50, a=a, b=b, c=c, discard=5)
        except Exception as e:
            self.fail(f"roessler raised with custom parameters: {e}")
        self.assertEqual(x.shape, (50, 3))

    def test_no_nan_or_inf(self):
        t, x = roessler(length=1000, discard=20)
        self.assertFalse(np.any(np.isnan(x)))
        self.assertFalse(np.any(np.isinf(x)))

    def test_time_axis_monotonic(self):
        t, x = roessler(length=100, discard=10)
        self.assertTrue(np.all(np.diff(t) > 0))

    def test_random_initial_condition_no_crash(self):
        np.random.seed(0)
        t, x = roessler(length=50, discard=5)
        self.assertEqual(x.shape, (50, 3))
        self.assertFalse(np.any(np.isnan(x)))

    def test_subsampling(self):
        """With sample=10*step, the returned time step should be ~10*step."""
        x0 = (-9.0, 0.0, 0.0)
        step = 0.001
        sample = 0.01  # internal sample = int(0.01 / 0.001) = 10
        t, x = roessler(length=50, x0=x0, step=step, sample=sample, discard=0)
        expected_dt = 10 * step
        actual_dt = t[1] - t[0]
        # linspace spacing is not exactly uniform; use a generous rtol
        np.testing.assert_allclose(actual_dt, expected_dt, rtol=5e-3)


if __name__ == "__main__":
    unittest.main()
