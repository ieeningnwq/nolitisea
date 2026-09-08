"""Tests for ``nolitisea.prediction.esn``.

Covers ESN (Echo State Network) — pure numpy, no torch needed.
"""

import unittest

import numpy as np

from nolitisea.prediction.esn import (
    esn_forecast_error,
    fit_esn,
    predict_esn,
)
from nolitisea.utils.rescale import rescale_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _sine_series(n=500):
    t = np.linspace(0, 20, n)
    return np.sin(t)


def _ar1_series(n=500, a=0.9, seed=42):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = a * x[i - 1] + 0.1 * rng.standard_normal()
    return x


def _linear_map_series(n=300, a=-0.5, b=3.0, x0=10.0):
    x = np.zeros(n)
    x[0] = x0
    for i in range(n - 1):
        x[i + 1] = a * x[i] + b
    return x


def _logistic_map_series(n=500, r=3.7, x0=0.3):
    x = np.zeros(n)
    x[0] = x0
    for i in range(n - 1):
        x[i + 1] = r * x[i] * (1 - x[i])
    return x


def _two_component_sine_cosine(n=500):
    t = np.linspace(0, 20, n)
    return np.column_stack([np.sin(t), np.cos(t)])


# ---------------------------------------------------------------------------
# Basic fitting tests
# ---------------------------------------------------------------------------

class TestFitEsnBasic(unittest.TestCase):
    def test_sine_low_error(self):
        s = _sine_series(500)
        m = fit_esn(s, dim=3, n_reservoir=200, seed=42)
        self.assertLess(m["in_sample_fce"], 0.3)

    def test_returns_expected_keys(self):
        s = _sine_series(200)
        m = fit_esn(s, dim=2, n_reservoir=50, seed=42)
        for key in ("W_in", "W_res", "W_out", "minv", "interval",
                     "dim", "delay", "step", "n_vars", "n_reservoir",
                     "in_sample_rmse", "in_sample_fce", "seed", "last_state"):
            self.assertIn(key, m, f"missing key: {key}")

    def test_w_out_shape(self):
        s = _sine_series(200)
        m = fit_esn(s, dim=2, n_reservoir=50, seed=42)
        self.assertEqual(m["W_in"].shape, (50, 2))
        self.assertEqual(m["W_res"].shape, (50, 50))
        self.assertEqual(m["W_out"].shape, (1, 51))

    def test_rescale_params_consistent(self):
        s = _sine_series(200) * 10 + 3
        m = fit_esn(s, dim=2, n_reservoir=50, seed=42)
        _, minv, interval = rescale_data(s)
        self.assertAlmostEqual(m["minv"][0], minv)
        self.assertAlmostEqual(m["interval"][0], interval)


class TestFitEsnReproducibility(unittest.TestCase):
    def test_same_seed_same_weights(self):
        s = _sine_series(200)
        m1 = fit_esn(s, dim=2, n_reservoir=50, seed=42)
        m2 = fit_esn(s, dim=2, n_reservoir=50, seed=42)
        np.testing.assert_allclose(m1["W_in"], m2["W_in"])
        np.testing.assert_allclose(m1["W_res"], m2["W_res"])
        np.testing.assert_allclose(m1["W_out"], m2["W_out"])

    def test_different_seed_different_weights(self):
        s = _sine_series(200)
        m1 = fit_esn(s, dim=2, n_reservoir=50, seed=42)
        m2 = fit_esn(s, dim=2, n_reservoir=50, seed=43)
        self.assertGreater(
            np.max(np.abs(m1["W_res"] - m2["W_res"])), 1e-10
        )


class TestFitEsnMultivariate(unittest.TestCase):
    def test_two_component_shapes(self):
        s = _two_component_sine_cosine(200)
        m = fit_esn(s, dim=2, n_reservoir=50, seed=42)
        self.assertEqual(m["n_vars"], 2)
        self.assertEqual(m["W_in"].shape, (50, 4))
        self.assertEqual(m["W_out"].shape, (2, 51))

    def test_two_component_low_error(self):
        s = _two_component_sine_cosine(500)
        m = fit_esn(s, dim=3, n_reservoir=200, seed=42)
        self.assertLess(m["in_sample_fce"], 0.3)


class TestFitEsnInsample(unittest.TestCase):
    def test_out_of_sample_when_insample_lt_n(self):
        s = _sine_series(500)
        m = fit_esn(s, dim=3, n_reservoir=100, insample=400, seed=42)
        self.assertIsNotNone(m["out_of_sample_rmse"])
        self.assertIsNotNone(m["out_of_sample_fce"])

    def test_out_of_sample_matches_forecast_error(self):
        """Held-out RMSE must pair state t with target t + step (step=1)."""
        s = _sine_series(500)
        m = fit_esn(s, dim=3, n_reservoir=100, insample=400, step=1, seed=42)
        s_resc, _, _ = rescale_data(s)
        valid_start = (3 - 1) * 1
        err = esn_forecast_error(s_resc, m, dim=3, delay=1, step=1,
                                 i0=400 - 1 - valid_start, i1=500)
        self.assertAlmostEqual(err, m["out_of_sample_rmse"], places=8)

    def test_out_of_sample_matches_forecast_error_step2(self):
        """Same pairing check with step=2."""
        s = _sine_series(500)
        m = fit_esn(s, dim=3, n_reservoir=100, insample=400, step=2, seed=42)
        s_resc, _, _ = rescale_data(s)
        valid_start = (3 - 1) * 1
        err = esn_forecast_error(s_resc, m, dim=3, delay=1, step=2,
                                 i0=400 - 2 - valid_start, i1=500)
        self.assertAlmostEqual(err, m["out_of_sample_rmse"], places=8)

    def test_out_of_sample_low_error_on_sine(self):
        s = _sine_series(500)
        m = fit_esn(s, dim=3, n_reservoir=200, insample=400, seed=42)
        self.assertLess(m["out_of_sample_fce"], 0.3)

    def test_out_of_sample_none_when_full(self):
        s = _sine_series(200)
        m = fit_esn(s, dim=2, n_reservoir=50, seed=42)
        self.assertIsNone(m["out_of_sample_rmse"])
        self.assertIsNone(m["out_of_sample_fce"])


class TestFitEsnValidation(unittest.TestCase):
    def test_series_too_short(self):
        with self.assertRaises(ValueError):
            fit_esn(np.array([1.0, 2.0]), dim=3, n_reservoir=10)

    def test_n_reservoir_zero(self):
        with self.assertRaises(ValueError):
            fit_esn(_sine_series(200), dim=2, n_reservoir=0)

    def test_insample_too_short(self):
        with self.assertRaises(ValueError):
            fit_esn(_sine_series(200), dim=3, n_reservoir=10, insample=3)

    def test_bad_spectral_radius(self):
        with self.assertRaises(ValueError):
            fit_esn(_sine_series(200), dim=2, spectral_radius=0)


class TestPredictEsn(unittest.TestCase):
    def test_forecast_finite(self):
        s = _sine_series(500)
        m = fit_esn(s, dim=3, n_reservoir=100, seed=42)
        fc = predict_esn(m, s, n_steps=50)
        self.assertEqual(len(fc), 50)
        self.assertTrue(np.all(np.isfinite(fc)))

    def test_forecast_shape_univariate(self):
        s = _sine_series(300)
        m = fit_esn(s, dim=2, n_reservoir=50, seed=42)
        fc = predict_esn(m, s, n_steps=20)
        self.assertEqual(fc.shape, (20,))

    def test_n_steps_zero(self):
        s = _sine_series(100)
        m = fit_esn(s, dim=2, n_reservoir=50, seed=42)
        fc = predict_esn(m, s, n_steps=0)
        self.assertEqual(len(fc), 0)

    def test_deterministic(self):
        s = _sine_series(300)
        m = fit_esn(s, dim=2, n_reservoir=50, seed=42)
        fc1 = predict_esn(m, s, 10)
        fc2 = predict_esn(m, s, 10)
        np.testing.assert_allclose(fc1, fc2)


class TestEsnForecastError(unittest.TestCase):
    def test_matches_fit_esn_insample(self):
        s = _sine_series(300)
        m = fit_esn(s, dim=3, n_reservoir=100, seed=42)
        s_resc, _, _ = rescale_data(s)
        err = esn_forecast_error(s_resc, m, dim=3, delay=1, step=1,
                                 i0=0, i1=300)
        self.assertAlmostEqual(err, m["in_sample_rmse"], places=8)

    def test_empty_interval(self):
        s = _sine_series(100)
        m = fit_esn(s, dim=2, n_reservoir=50, seed=42)
        s_resc, _, _ = rescale_data(s)
        err = esn_forecast_error(s_resc, m, dim=2, delay=1, step=1,
                                 i0=50, i1=50)
        self.assertTrue(np.isnan(err))

    def test_subinterval(self):
        s = _sine_series(300)
        m = fit_esn(s, dim=3, n_reservoir=100, seed=42)
        s_resc, _, _ = rescale_data(s)
        err = esn_forecast_error(s_resc, m, dim=3, delay=1, step=1,
                                 i0=100, i1=200)
        self.assertTrue(np.isfinite(err))
        self.assertGreater(err, 0)


class TestBruteForceAgreement(unittest.TestCase):
    def test_linear_map_recovery(self):
        """Linear map x_{t+1} = a*x_t + b should be learnable by ESN."""
        s = _linear_map_series(300, a=-0.5, b=3.0)
        m = fit_esn(s, dim=2, n_reservoir=100, ridge_alpha=1e-8, seed=42)
        self.assertLess(m["in_sample_rmse"], 0.5)

    def test_logistic_map_one_step(self):
        """One-step prediction of logistic map should be close."""
        s = _logistic_map_series(400, r=3.7)
        m = fit_esn(s, dim=2, n_reservoir=200, ridge_alpha=1e-8, seed=42)
        fc = predict_esn(m, s, n_steps=1)
        expected = 3.7 * s[-1] * (1 - s[-1])
        self.assertLess(abs(fc[0] - expected), 0.1)


if __name__ == "__main__":
    unittest.main()
