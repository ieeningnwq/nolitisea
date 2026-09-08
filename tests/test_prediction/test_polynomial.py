"""Tests for ``nolitisea.prediction.polynomial``."""

import unittest

import numpy as np

from nolitisea.prediction.polynomial import (
    fit_polynom,
    fit_polynom_terms,
    monomial_value,
    polyback,
    polypar,
)


class TestPolypar(unittest.TestCase):
    """Tests for the monomial generator (replaces TISEAN polypar)."""

    def test_count_matches_binomial(self):
        from math import comb
        for dim in [1, 2, 3, 4]:
            for degree in [0, 1, 2, 3]:
                terms = polypar(dim, degree)
                self.assertEqual(len(terms), comb(dim + degree, dim),
                                 f"polypar({dim}, {degree})")

    def test_all_exponents_valid(self):
        for dim in [2, 3, 4]:
            for degree in [0, 1, 3]:
                terms = polypar(dim, degree)
                for e in terms:
                    self.assertEqual(len(e), dim)
                    self.assertTrue(sum(e) <= degree)
                    for d in range(dim):
                        self.assertGreaterEqual(e[d], 0)

    def test_order_matches_tisean_c(self):
        self.assertEqual(
            polypar(3, 2),
            [(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0),
             (0, 2, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (0, 0, 2)],
        )
        self.assertEqual(polypar(2, 1), [(0, 0), (1, 0), (0, 1)])
        self.assertEqual(polypar(1, 3), [(0,), (1,), (2,), (3,)])
        self.assertEqual(polypar(2, 0), [(0, 0)])

    def test_dim_one_degree_zero(self):
        self.assertEqual(polypar(1, 0), [(0,)])

    def test_validation_dim(self):
        with self.assertRaises(ValueError):
            polypar(0, 2)
        with self.assertRaises(ValueError):
            polypar(-1, 2)

    def test_validation_degree(self):
        with self.assertRaises(ValueError):
            polypar(2, -1)


class TestMonomialValue(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(monomial_value(np.array([3.0, 4.0]), (0, 0)), 1.0)

    def test_linear(self):
        self.assertEqual(monomial_value(np.array([2.0, 3.0]), (1, 0)), 2.0)
        self.assertEqual(monomial_value(np.array([2.0, 3.0]), (0, 1)), 3.0)

    def test_quadratic(self):
        self.assertEqual(monomial_value(np.array([2.0, 3.0]), (2, 1)),
                         4.0 * 3.0)

    def test_mixed(self):
        self.assertEqual(monomial_value(np.array([2.0, 3.0]), (3, 0)), 8.0)


class TestFitPolynomPerfectModel(unittest.TestCase):
    """On noise-free polynomial recursions the OLS fit must recover
    the generating coefficients exactly (up to floating-point noise)."""

    def test_linear_map_dim1_stable(self):
        # x_{n+1} = 0.5 * x_n + 3   (stable fixed point x* = 6)
        x = np.zeros(500)
        x[0] = 0.0
        for i in range(499):
            x[i + 1] = 0.5 * x[i] + 3.0
        res = fit_polynom(x, dim=1, delay=1, degree=1)
        self.assertEqual(res["terms"], [(0,), (1,)])
        np.testing.assert_allclose(res["coeffs"], [3.0, 0.5], atol=1e-8)
        self.assertLess(res["in_sample_rmse"], 1e-8)

    def test_quadratic_map_dim1(self):
        # x_{n+1} = x_n^2 - 0.5   (dim=1, degree=2)
        x = np.zeros(200)
        x[0] = 0.5
        for i in range(199):
            x[i + 1] = x[i] ** 2 - 0.5
        res = fit_polynom(x, dim=1, delay=1, degree=2)
        # Terms: (0,), (1,), (2,)
        np.testing.assert_allclose(res["coeffs"], [-0.5, 0.0, 1.0], atol=1e-9)
        self.assertLess(res["in_sample_rmse"], 1e-9)

    def test_linear_dim2(self):
        # x_{n+1} = 0.5*x_n - 0.3*x_{n-1} + 1.0   (dim=2, degree=1)
        x = np.zeros(300)
        x[0] = 1.0
        x[1] = 0.5
        for i in range(298):
            x[i + 2] = 0.5 * x[i + 1] - 0.3 * x[i] + 1.0
        res = fit_polynom(x, dim=2, delay=1, degree=1)
        # Terms: [(0,0), (1,0), (0,1)]
        np.testing.assert_allclose(
            res["coeffs"], [1.0, 0.5, -0.3], atol=1e-9
        )
        self.assertLess(res["in_sample_rmse"], 1e-8)


class TestFitPolynomTerms(unittest.TestCase):
    """fit_polynom_terms with all terms from polypar(...) must agree
    with fit_polynom on the **unnormalised** coefficient values."""

    def test_equivalence_to_fit_polynom(self):
        rng = np.random.default_rng(0)
        x = np.zeros(500)
        x[0] = 1.0
        for i in range(499):
            x[i + 1] = 0.7 * x[i] + rng.normal(scale=0.1)

        terms = polypar(2, 1)
        res1 = fit_polynom(x, dim=2, delay=1, degree=1)
        res2 = fit_polynom_terms(x, dim=2, delay=1, terms=terms)

        np.testing.assert_allclose(res1["coeffs"], res2["coeffs"], atol=1e-10)
        np.testing.assert_allclose(
            res1["in_sample_rmse"] * res1["std_dev"],
            res2["in_sample_rmse"],
            atol=1e-10,
        )

    def test_subset_of_terms(self):
        # x_{n+1} = 0.7*x_n  → fit only (1,) term
        x = np.zeros(300)
        x[0] = 1.0
        for i in range(299):
            x[i + 1] = 0.7 * x[i]
        res = fit_polynom_terms(x, dim=1, delay=1, terms=[(1,)])
        self.assertEqual(len(res["coeffs"]), 1)
        self.assertAlmostEqual(res["coeffs"][0], 0.7, places=8)
        self.assertLess(res["in_sample_rmse"], 1e-6)

    def test_validation(self):
        with self.assertRaises(ValueError):
            fit_polynom_terms([1, 2, 3], dim=2, delay=1, terms=polypar(2, 1))
        with self.assertRaises(ValueError):
            fit_polynom_terms(np.zeros(20), dim=2, delay=1, terms=[])


class TestCastSteps(unittest.TestCase):
    def test_perfect_forecast_linear(self):
        # x_{n+1} = x_n + 1   (pure linear trend)
        x = np.arange(100, dtype=float)
        res = fit_polynom(x, dim=1, delay=1, degree=1, cast_steps=10)
        self.assertIn("casted", res)
        casted = res["casted"]
        expected = np.arange(100, 110, dtype=float)
        np.testing.assert_allclose(casted, expected, atol=1e-9)

    def test_fit_polynom_terms_cast(self):
        x = np.arange(100, dtype=float)
        res = fit_polynom_terms(
            x, dim=1, delay=1, terms=polypar(1, 1), cast_steps=5
        )
        expected = np.arange(100, 105, dtype=float)
        np.testing.assert_allclose(res["casted"], expected, atol=1e-9)


class TestPolyback(unittest.TestCase):
    def test_trivial_case(self):
        rng = np.random.default_rng(1)
        x = np.zeros(2000)
        x[0] = 1.0
        for i in range(1999):
            x[i + 1] = 0.5 * x[i] + rng.normal(scale=0.05)

        r = polyback(x, dim=2, delay=1, degree=2, down_to=2, step=1,
                    insample=1500)
        self.assertEqual(len(r["path"]), 6 - 2)  # 6 terms → down_to 2
        self.assertEqual(len(r["final"]["terms"]), 2)
        self.assertIn((1, 0), r["final"]["terms"])

    def test_decision_error_field(self):
        x = np.random.default_rng(2).normal(size=500)
        r = polyback(x, dim=2, delay=1, degree=1, down_to=1)
        self.assertEqual(r["decision_error"], "in_sample")
        r = polyback(x, dim=2, delay=1, degree=1, down_to=1, insample=400)
        self.assertEqual(r["decision_error"], "out_of_sample")

    def test_down_to_validation(self):
        x = np.random.default_rng(3).normal(size=500)
        with self.assertRaises(ValueError):
            polyback(x, dim=2, delay=1, degree=2, down_to=0)
        with self.assertRaises(ValueError):
            polyback(x, dim=2, delay=1, degree=1, down_to=10)


class TestCrossDimDelay(unittest.TestCase):
    """Regression checks: varying dim/delay/step must still produce
    sensible coefficients on a known dynamical system."""

    def test_dim3_delay2_sparse(self):
        # x_{n+1} = 0.8*x_n + 0.1*x_{n-2}   (dim=3, delay=2, degree=1)
        # Embedding: [x_n, x_{n-2}, x_{n-4}]
        # Zero noise, properly initialised so every row satisfies the rule.
        x = np.zeros(200)
        x[0] = 1.0
        x[1] = 0.5
        x[2] = -0.3
        x[3] = 0.8 * x[2] + 0.1 * x[0]   # = 0.8*(-0.3) + 0.1*1 = -0.14
        for i in range(4, 199):
            x[i + 1] = 0.8 * x[i] + 0.1 * x[i - 2]
        res = fit_polynom(x, dim=3, delay=2, degree=1)
        self.assertEqual(res["terms"],
                         [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
        np.testing.assert_allclose(res["coeffs"][1], 0.8, atol=1e-9)
        np.testing.assert_allclose(res["coeffs"][2], 0.1, atol=1e-9)
        np.testing.assert_allclose(res["coeffs"][3], 0.0, atol=1e-9)

    def test_step_2_deterministic(self):
        # x_{n+2} = 0.9*x_n + 0.1  (dim=1, degree=1, step=2)
        x = np.zeros(500)
        x[0] = 0.0
        for i in range(498):
            x[i + 2] = 0.9 * x[i] + 0.1
        res = fit_polynom(x, dim=1, delay=1, degree=1, step=2)
        self.assertEqual(res["terms"], [(0,), (1,)])
        np.testing.assert_allclose(res["coeffs"], [0.1, 0.9], atol=1e-6)
        self.assertLess(res["in_sample_rmse"], 1e-6)

    def test_dim2_delay2(self):
        # x_{n+1} = 0.6*x_n + 0.3*x_{n-2}  (dim=2, delay=2)
        # Embedding: [x_n, x_{n-2}]
        # Zero noise, properly initialised.
        x = np.zeros(200)
        x[0] = 1.0
        x[1] = 2.0
        x[2] = 3.0
        x[3] = 0.6 * x[2] + 0.3 * x[0]   # = 2.1
        for i in range(3, 199):
            x[i + 1] = 0.6 * x[i] + 0.3 * x[i - 2]
        res = fit_polynom(x, dim=2, delay=2, degree=1)
        # Terms: [(0,0), (1,0), (0,1)]
        np.testing.assert_allclose(res["coeffs"][1], 0.6, atol=1e-9)
        np.testing.assert_allclose(res["coeffs"][2], 0.3, atol=1e-9)
