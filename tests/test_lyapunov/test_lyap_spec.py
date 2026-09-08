"""Tests for nolitisea.lyapunov.lyap_spec."""

import unittest

import numpy as np

from nolitisea.generate.henon import henon
from nolitisea.lyapunov.lyap_spec import lyap_spec


# Direct Henon Lyapunov spectrum via true Jacobian iteration (reference).
def _henon_direct(n=50000, a=1.4, b=0.3, seed=0):
    rng = np.random.default_rng(seed)
    x0, y0 = rng.standard_normal(2)
    for _ in range(2000):
        x0, y0 = 1 - a * x0 * x0 + b * y0, x0
    delta = np.eye(2)
    lf = np.zeros(2)
    for _ in range(n):
        J = np.array([[-2 * a * x0, b], [1.0, 0.0]])
        x0, y0 = 1 - a * x0 * x0 + b * y0, x0
        delta = J @ delta
        Q, R = np.linalg.qr(delta)
        lf += np.log(np.abs(np.diag(R)))
        delta = Q
    return lf / n  # QR returns largest stretch first → already descending


class TestLyapSpecOutputs(unittest.TestCase):
    def test_output_structure(self):
        x, _ = henon(n=5000)
        series = np.column_stack([x, x + 0.1])  # trivial 2-component
        res = lyap_spec(series, embed=1, n_iter=1000, n_neighbors=30)
        self.assertEqual(set(res), {
            "exponents", "times", "running", "ky_dim",
            "ave_neighbors", "ave_eps", "forecast_err",
        })
        self.assertEqual(res["exponents"].shape, (2,))
        self.assertEqual(res["times"].shape[0], res["running"].shape[0])
        self.assertEqual(res["running"].shape, (res["times"].size, 2))
        self.assertTrue(np.all(np.isfinite(res["exponents"])))

    def test_exponents_descending(self):
        x, _ = henon(n=10000)
        res = lyap_spec(x, embed=2, n_iter=2000, n_neighbors=30)
        for i in range(res["exponents"].size - 1):
            self.assertGreaterEqual(
                res["exponents"][i], res["exponents"][i + 1]
            )

    def test_sum_equal_log_det(self):
        # For a deterministic map, sum(lambda_j) = ln(|det(J)|).
        # Henon: det(J) = -b = -0.3, so ln(|det|) = ln(0.3) ≈ -1.204.
        x, y = henon(n=20000)
        series = np.column_stack([x, y])
        res = lyap_spec(series, embed=1, n_iter=10000, n_neighbors=30)
        expected_sum = np.log(0.3)
        self.assertAlmostEqual(res["exponents"].sum(), expected_sum, delta=0.08)

    def test_invalid_parameters(self):
        s = np.arange(50.0)
        with self.assertRaises(ValueError):
            lyap_spec(s, embed=0)
        with self.assertRaises(ValueError):
            lyap_spec(s, embed=2, n_neighbors=2)
        # Series too short for the requested embed.
        s_short = np.arange(10.0)
        with self.assertRaises(ValueError):
            lyap_spec(s_short, embed=5, n_neighbors=30)
        with self.assertRaises(ValueError):
            lyap_spec(np.zeros((10, 3, 3)), embed=1)

    def test_constant_series_raises(self):
        with self.assertRaises(RuntimeError):
            lyap_spec(np.ones(100), embed=2, n_neighbors=30)


class TestLyapSpecHenon(unittest.TestCase):
    def test_scalar_embed2_matches_direct(self):
        x, _ = henon(n=20000)
        # Direct computation (2D tangent space from scalar embed=2).
        direct = _henon_direct(n=50000)

        res = lyap_spec(x, embed=2, n_iter=10000, n_neighbors=30)
        # The largest exponent should be close to the direct lambda.
        self.assertAlmostEqual(res["exponents"][0], direct[0], delta=0.05)
        # Sum should match ln(|det|) within tolerance.
        self.assertAlmostEqual(res["exponents"].sum(), np.log(0.3), delta=0.08)

    def test_2d_components_embed1(self):
        x, y = henon(n=20000)
        series = np.column_stack([x, y])
        direct = _henon_direct(n=50000)

        res = lyap_spec(series, embed=1, n_iter=10000, n_neighbors=30)
        self.assertAlmostEqual(res["exponents"][0], direct[0], delta=0.05)
        self.assertAlmostEqual(res["exponents"].sum(), np.log(0.3), delta=0.08)

    def test_ky_dim_reasonable(self):
        # Henon attractor has KY dimension around 1.26.
        x, y = henon(n=20000)
        series = np.column_stack([x, y])
        res = lyap_spec(series, embed=1, n_iter=10000, n_neighbors=30)
        # Positive exponent contributes a fractional part.
        self.assertGreaterEqual(res["ky_dim"], 1.0)
        self.assertLessEqual(res["ky_dim"], 2.0)

    def test_deterministic_initial_condition(self):
        x, y = henon(n=10000)
        series = np.column_stack([x, y])
        res1 = lyap_spec(series, embed=1, n_iter=5000, n_neighbors=30, seed=42)
        res2 = lyap_spec(series, embed=1, n_iter=5000, n_neighbors=30, seed=42)
        np.testing.assert_array_equal(res1["exponents"], res2["exponents"])

    def test_seed_affects_result(self):
        x, y = henon(n=5000)
        series = np.column_stack([x, y])
        res1 = lyap_spec(series, embed=1, n_iter=500, n_neighbors=30, seed=0)
        res2 = lyap_spec(series, embed=1, n_iter=500, n_neighbors=30, seed=99)
        self.assertFalse(np.array_equal(res1["exponents"], res2["exponents"]))


if __name__ == "__main__":
    unittest.main()
