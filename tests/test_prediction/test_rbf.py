"""Tests for ``nolitisea.prediction.rbf``.

Covers:

* :func:`fit_rbf` — Gaussian RBF model fitting (centre init, drift,
  width estimation, normal equations, in/out-of-sample errors)
* :func:`predict_rbf` — iterated multi-step forecasting
* :func:`rbf_forecast_error` — RMSE helper
* Internal helpers: ``_init_centers``, ``_avdistance``, ``_rbf``,
  ``_delay_vector``, ``_drift_centers``
"""

import unittest

import numpy as np

from nolitisea.prediction.rbf import (
    _avdistance,
    _delay_vector,
    _drift_centers,
    _init_centers,
    _rbf,
    fit_rbf,
    predict_rbf,
    rbf_forecast_error,
)
from nolitisea.utils.rescale import rescale_data

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _sine_series(n=500, seed=42):
    """Deterministic sine series — perfectly predictable by RBF."""
    t = np.linspace(0, 20, n)
    return np.sin(t)


def _ar1_series(n=500, a=0.9, seed=42):
    """AR(1) process — smooth, near-linear dynamics."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = a * x[i - 1] + 0.1 * rng.standard_normal()
    return x


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestInitCenters(unittest.TestCase):
    """Tests for ``_init_centers``."""

    def test_first_and_last_center_correct(self):
        s = np.linspace(0, 1, 100)
        dim, delay = 3, 1
        nc = 5
        centers = _init_centers(s, dim, delay, nc)
        self.assertEqual(centers.shape, (nc, dim))
        # First center: offset = 2, columns = [s[2], s[1], s[0]]
        offset = (dim - 1) * delay
        expected_first = np.array([s[offset - j * delay] for j in range(dim)])
        np.testing.assert_allclose(centers[0], expected_first)
        # Last center: index = 99
        expected_last = np.array([s[99 - j * delay] for j in range(dim)])
        np.testing.assert_allclose(centers[-1], expected_last)

    def test_integer_division_spacing(self):
        """Centre indices must follow the ``i*cstep//(nc-1)`` pattern."""
        n, dim, delay, nc = 200, 2, 1, 7
        s = np.random.default_rng(0).random(n)
        offset = (dim - 1) * delay
        cstep = n - 1 - offset
        centers = _init_centers(s, dim, delay, nc)
        for i in range(nc):
            expected_idx = offset + (i * cstep) // (nc - 1)
            expected_vec = np.array([s[expected_idx - j * delay] for j in range(dim)])
            np.testing.assert_allclose(centers[i], expected_vec)

    def test_single_dim_delay1(self):
        s = np.arange(50, dtype=float)
        centers = _init_centers(s, dim=1, delay=1, n_centers=3)
        self.assertEqual(centers.shape, (3, 1))

    def test_delay_gt1(self):
        s = np.arange(100, dtype=float)
        dim, delay, nc = 3, 2, 4
        centers = _init_centers(s, dim, delay, nc)
        self.assertEqual(centers.shape, (nc, dim))
        offset = (dim - 1) * delay  # 4
        cstep = 100 - 1 - offset   # 95
        for i in range(nc):
            idx = offset + (i * cstep) // (nc - 1)
            for j in range(dim):
                self.assertAlmostEqual(centers[i, j], s[idx - j * delay])


class TestAvdistance(unittest.TestCase):
    """Tests for ``_avdistance``."""

    def test_two_centers_1d(self):
        c = np.array([[0.0], [1.0]])
        d = _avdistance(c)
        # total = 1^2 = 1; 2*1 / (1*2*1) = 1; sqrt(1) = 1
        self.assertAlmostEqual(d, 1.0)

    def test_two_centers_2d(self):
        c = np.array([[0.0, 0.0], [3.0, 4.0]])
        d = _avdistance(c)
        # total = 9+16 = 25; 2*25 / (1*2*2) = 25/2 = 12.5; sqrt(12.5)
        self.assertAlmostEqual(d, np.sqrt(12.5))

    def test_symmetric_pattern(self):
        """Centres at corners of unit square."""
        c = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        d = _avdistance(c)
        self.assertGreater(d, 0)
        self.assertLess(d, 2.0)

    def test_positive(self):
        rng = np.random.default_rng(7)
        c = rng.random((10, 3))
        self.assertGreater(_avdistance(c), 0)


class TestRbfKernel(unittest.TestCase):
    """Tests for ``_rbf`` (Gaussian kernel)."""

    def test_self_distance_is_one(self):
        x = np.array([0.5, 0.3, 0.1])
        self.assertAlmostEqual(_rbf(x, x, eps=0.1), 1.0)

    def test_decays_with_distance(self):
        x = np.array([0.0])
        c = np.array([0.1])
        eps = 0.1
        r = 0.1
        expected = np.exp(-r ** 2 / (2 * eps ** 2))
        self.assertAlmostEqual(_rbf(x, c, eps), expected)

    def test_large_distance_near_zero(self):
        x = np.array([0.0])
        c = np.array([10.0])
        self.assertLess(_rbf(x, c, eps=0.1), 1e-100)

    def test_eps_scales_decay(self):
        """Larger eps → slower decay."""
        x = np.array([0.0])
        c = np.array([0.5])
        v_small = _rbf(x, c, eps=0.05)
        v_large = _rbf(x, c, eps=1.0)
        self.assertGreater(v_large, v_small)


class TestDelayVector(unittest.TestCase):
    """Tests for ``_delay_vector``."""

    def test_basic(self):
        s = np.arange(10, dtype=float)
        v = _delay_vector(s, n=5, dim=3, delay=1)
        np.testing.assert_allclose(v, [5, 4, 3])

    def test_delay2(self):
        s = np.arange(20, dtype=float)
        v = _delay_vector(s, n=10, dim=3, delay=2)
        np.testing.assert_allclose(v, [10, 8, 6])


class TestDriftCenters(unittest.TestCase):
    """Tests for ``_drift_centers``."""

    def test_separates_overlapping(self):
        c = np.array([[0.5, 0.5], [0.5001, 0.5], [0.5, 0.5001]])
        c_orig = c.copy()
        _drift_centers(c, n_iter=5, step0=0.01)
        # Centres should have moved apart
        moved = np.linalg.norm(c - c_orig)
        self.assertGreater(moved, 1e-6)

    def test_stays_in_bounds(self):
        rng = np.random.default_rng(3)
        c = rng.random((8, 2))
        _drift_centers(c, n_iter=20)
        self.assertTrue(np.all(c >= -0.1 - 1e-10))
        self.assertTrue(np.all(c <= 1.1 + 1e-10))

    def test_noop_with_single_center(self):
        c = np.array([[0.5, 0.5]])
        c_orig = c.copy()
        _drift_centers(c, n_iter=5)
        np.testing.assert_allclose(c, c_orig)


# ---------------------------------------------------------------------------
# fit_rbf
# ---------------------------------------------------------------------------

class TestFitRbfBasic(unittest.TestCase):
    """Basic fitting tests."""

    def test_sine_low_error(self):
        s = _sine_series(500)
        m = fit_rbf(s, dim=3, delay=1, n_centers=20)
        self.assertLess(m["in_sample_fce"], 0.1)

    def test_returns_expected_keys(self):
        s = _sine_series(200)
        m = fit_rbf(s, dim=2, delay=1, n_centers=5)
        for key in ("centers", "coeffs", "eps", "minv", "interval",
                     "dim", "delay", "step", "in_sample_rmse",
                     "out_of_sample_rmse", "in_sample_fce",
                     "out_of_sample_fce"):
            self.assertIn(key, m)

    def test_coeffs_shape(self):
        s = _sine_series(200)
        m = fit_rbf(s, dim=2, delay=1, n_centers=7)
        self.assertEqual(m["coeffs"].shape, (8,))  # bias + 7 weights

    def test_centers_shape(self):
        s = _sine_series(200)
        m = fit_rbf(s, dim=3, delay=2, n_centers=5)
        self.assertEqual(m["centers"].shape, (5, 3))

    def test_rescale_params_consistent(self):
        s = _sine_series(200) * 10 + 3  # shift and scale
        m = fit_rbf(s, dim=2, delay=1, n_centers=5)
        _, minv, interval = rescale_data(s)
        self.assertAlmostEqual(m["minv"], minv)
        self.assertAlmostEqual(m["interval"], interval)


class TestFitRbfInsample(unittest.TestCase):
    """In-sample / out-of-sample split tests."""

    def test_out_of_sample_when_insample_lt_n(self):
        s = _sine_series(500)
        m = fit_rbf(s, dim=3, delay=1, n_centers=15, insample=400)
        self.assertIsNotNone(m["out_of_sample_rmse"])
        self.assertIsNotNone(m["out_of_sample_fce"])

    def test_out_of_sample_none_when_full(self):
        s = _sine_series(200)
        m = fit_rbf(s, dim=2, delay=1, n_centers=5)
        self.assertIsNone(m["out_of_sample_rmse"])
        self.assertIsNone(m["out_of_sample_fce"])

    def test_insample_truncation(self):
        s = _sine_series(500)
        m1 = fit_rbf(s, dim=3, delay=1, n_centers=15, insample=300)
        m2 = fit_rbf(s, dim=3, delay=1, n_centers=15, insample=500)
        # In-sample error should be finite for both
        self.assertTrue(np.isfinite(m1["in_sample_fce"]))
        self.assertTrue(np.isfinite(m2["in_sample_fce"]))


class TestFitRbfDrift(unittest.TestCase):
    """Drift on/off comparison."""

    def test_drift_changes_centers(self):
        s = _sine_series(300)
        m_drift = fit_rbf(s, dim=3, delay=1, n_centers=10, drift=True)
        m_nodrift = fit_rbf(s, dim=3, delay=1, n_centers=10, drift=False)
        diff = np.linalg.norm(m_drift["centers"] - m_nodrift["centers"])
        self.assertGreater(diff, 1e-10)

    def test_nodrift_still_works(self):
        s = _sine_series(500)
        m = fit_rbf(s, dim=3, delay=1, n_centers=10, drift=False)
        self.assertLess(m["in_sample_fce"], 0.2)


class TestFitRbfCustomEps(unittest.TestCase):
    """Custom epsilon vs auto-epsilon."""

    def test_custom_eps_used(self):
        s = _sine_series(300)
        m = fit_rbf(s, dim=2, delay=1, n_centers=8, eps=0.05)
        self.assertAlmostEqual(m["eps"], 0.05)

    def test_auto_eps_positive(self):
        s = _sine_series(300)
        m = fit_rbf(s, dim=2, delay=1, n_centers=8)
        self.assertGreater(m["eps"], 0)


class TestFitRbfStep(unittest.TestCase):
    """Multi-step forecast horizon."""

    def test_step2(self):
        s = _sine_series(500)
        m = fit_rbf(s, dim=3, delay=1, n_centers=15, step=2)
        self.assertEqual(m["step"], 2)
        self.assertTrue(np.isfinite(m["in_sample_fce"]))


class TestFitRbfValidation(unittest.TestCase):
    """Parameter validation."""

    def test_series_too_short(self):
        with self.assertRaises(ValueError):
            fit_rbf(np.array([1.0, 2.0]), dim=3, delay=1, n_centers=2)

    def test_n_centers_lt_2(self):
        with self.assertRaises(ValueError):
            fit_rbf(_sine_series(200), dim=2, delay=1, n_centers=1)

    def test_n_centers_too_large(self):
        with self.assertRaises(ValueError):
            fit_rbf(np.arange(10, dtype=float), dim=2, delay=1, n_centers=20)

    def test_insample_too_short(self):
        with self.assertRaises(ValueError):
            fit_rbf(_sine_series(200), dim=3, delay=1, n_centers=5, insample=3)

    def test_2d_input_flattened(self):
        """2-D input is ravelled to 1-D (no ValueError, just flattened)."""
        s2d = np.column_stack([np.arange(50.0), np.arange(50.0)])
        m = fit_rbf(s2d, dim=2, delay=1, n_centers=5)
        self.assertEqual(m["centers"].shape, (5, 2))


# ---------------------------------------------------------------------------
# predict_rbf
# ---------------------------------------------------------------------------

class TestPredictRbf(unittest.TestCase):
    """Iterated forecasting tests."""

    def test_sine_forecast_finite(self):
        s = _sine_series(500)
        m = fit_rbf(s, dim=3, delay=1, n_centers=20)
        fc = predict_rbf(m, s, n_steps=50)
        self.assertEqual(len(fc), 50)
        self.assertTrue(np.all(np.isfinite(fc)))

    def test_forecast_shape(self):
        s = _sine_series(300)
        m = fit_rbf(s, dim=2, delay=1, n_centers=8)
        fc = predict_rbf(m, s, n_steps=20)
        self.assertEqual(fc.shape, (20,))

    def test_n_steps_zero(self):
        s = _sine_series(100)
        m = fit_rbf(s, dim=2, delay=1, n_centers=5)
        fc = predict_rbf(m, s, n_steps=0)
        self.assertEqual(len(fc), 0)

    def test_deterministic(self):
        """Same seed → same forecast."""
        s = _sine_series(300)
        m = fit_rbf(s, dim=2, delay=1, n_centers=8)
        fc1 = predict_rbf(m, s, 10)
        fc2 = predict_rbf(m, s, 10)
        np.testing.assert_allclose(fc1, fc2)

    def test_ar1_short_term(self):
        """AR(1) forecast should track near-linear dynamics briefly."""
        s = _ar1_series(500)
        m = fit_rbf(s, dim=3, delay=1, n_centers=15)
        fc = predict_rbf(m, s, n_steps=5)
        # AR(1) with a=0.9 decays, predictions should be finite
        self.assertTrue(np.all(np.isfinite(fc)))

    def test_uses_last_values_as_seed(self):
        """Buffer should seed from the tail of the series."""
        s = _sine_series(200)
        m = fit_rbf(s, dim=2, delay=1, n_centers=5)
        # The first prediction depends on the last dim delay+1 values.
        # If we pass a different tail, the forecast changes.
        s_trunc = s[:-1]
        fc_full = predict_rbf(m, s, 1)
        fc_trunc = predict_rbf(m, s_trunc, 1)
        self.assertFalse(np.allclose(fc_full, fc_trunc))

    def test_short_series_padding(self):
        """Series shorter than buffer should pad without crashing."""
        s = _sine_series(100)
        m = fit_rbf(s, dim=3, delay=1, n_centers=5)
        short = s[-2:]  # only 2 values, buffer needs 3
        fc = predict_rbf(m, short, n_steps=3)
        self.assertEqual(len(fc), 3)
        self.assertTrue(np.all(np.isfinite(fc)))


# ---------------------------------------------------------------------------
# rbf_forecast_error
# ---------------------------------------------------------------------------

class TestRbfForecastError(unittest.TestCase):
    """Tests for the RMSE helper."""

    def test_matches_fit_rbf_insample(self):
        """``rbf_forecast_error`` should agree with ``fit_rbf``'s in-sample RMSE."""
        s = _sine_series(300)
        m = fit_rbf(s, dim=3, delay=1, n_centers=10)
        err = rbf_forecast_error(
            rescale_data(s)[0], m["coeffs"], m["centers"], m["eps"],
            dim=3, delay=1, step=1, i0=0, i1=300,
        )
        self.assertAlmostEqual(err, m["in_sample_rmse"], places=10)

    def test_empty_interval(self):
        s = _sine_series(100)
        m = fit_rbf(s, dim=2, delay=1, n_centers=5)
        # i0 == i1 → no samples
        err = rbf_forecast_error(
            rescale_data(s)[0], m["coeffs"], m["centers"], m["eps"],
            dim=2, delay=1, step=1, i0=50, i1=50,
        )
        self.assertTrue(np.isnan(err))

    def test_interval_too_short(self):
        s = _sine_series(100)
        m = fit_rbf(s, dim=2, delay=1, n_centers=5)
        # offset=1, step=1 → need at least 3 points
        err = rbf_forecast_error(
            rescale_data(s)[0], m["coeffs"], m["centers"], m["eps"],
            dim=2, delay=1, step=1, i0=0, i1=2,
        )
        self.assertTrue(np.isnan(err))

    def test_subinterval(self):
        """Error over [100, 200) should be finite and positive."""
        s = _sine_series(300)
        m = fit_rbf(s, dim=3, delay=1, n_centers=10)
        err = rbf_forecast_error(
            rescale_data(s)[0], m["coeffs"], m["centers"], m["eps"],
            dim=3, delay=1, step=1, i0=100, i1=200,
        )
        self.assertTrue(np.isfinite(err))
        self.assertGreater(err, 0)


# ---------------------------------------------------------------------------
# Brute-force cross-check
# ---------------------------------------------------------------------------

class TestBruteForceAgreement(unittest.TestCase):
    """O(N^2) brute-force reference for the RBF normal equations."""

    def _brute_fit(self, series, dim, delay, n_centers, step=1, drift=True):
        """Independent reimplementation of fit_rbf's core math."""
        s, minv, interval = rescale_data(np.asarray(series, dtype=float).ravel())
        n = len(s)
        offset = (dim - 1) * delay

        # Centres (same init, optional drift)
        centers = _init_centers(s, dim, delay, n_centers)
        if drift:
            _drift_centers(centers)

        # Width
        eps = _avdistance(centers)

        # Design matrix
        n_start = offset
        n_end = n - step
        ns = n_end - n_start
        X = np.ones((ns, n_centers + 1))
        for k, ni in enumerate(range(n_start, n_end)):
            xv = _delay_vector(s, ni, dim, delay)
            for j in range(n_centers):
                X[k, j + 1] = _rbf(xv, centers[j], eps)
        y = s[n_start + step: n_end + step]

        # Solve least-squares (independent of numpy.linalg.solve path)
        coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return coefs, centers, eps, minv, interval

    def test_coeffs_match_brute_force(self):
        s = _sine_series(300)
        m = fit_rbf(s, dim=3, delay=1, n_centers=8, drift=False)
        coefs_brute, _, _, _, _ = self._brute_fit(
            s, dim=3, delay=1, n_centers=8, drift=False
        )
        np.testing.assert_allclose(m["coeffs"], coefs_brute, atol=1e-10)

    def test_eps_matches(self):
        s = _sine_series(300)
        m = fit_rbf(s, dim=3, delay=1, n_centers=8, drift=False)
        _, _, eps_brute, _, _ = self._brute_fit(
            s, dim=3, delay=1, n_centers=8, drift=False
        )
        self.assertAlmostEqual(m["eps"], eps_brute, places=12)

    def test_centers_match(self):
        s = _sine_series(300)
        m = fit_rbf(s, dim=3, delay=1, n_centers=8, drift=False)
        _, centers_brute, _, _, _ = self._brute_fit(
            s, dim=3, delay=1, n_centers=8, drift=False
        )
        np.testing.assert_allclose(m["centers"], centers_brute, atol=1e-12)


# ---------------------------------------------------------------------------
# Numerical robustness
# ---------------------------------------------------------------------------

class TestNumericalRobustness(unittest.TestCase):
    """Edge-case robustness."""

    def test_singular_matrix_raises(self):
        """Two identical centres → singular normal matrix."""
        # With n_centers=2 and drift=False, centres can be very close on
        # constant data, leading to a singular Gram matrix.
        s = np.ones(50) * 0.5
        with self.assertRaises((RuntimeError, np.linalg.LinAlgError)):
            fit_rbf(s, dim=2, delay=1, n_centers=3, drift=False)

    def test_constant_series_drift_off(self):
        """Constant series with drift off should either fit or raise cleanly."""
        s = np.ones(100) * 0.5
        try:
            m = fit_rbf(s, dim=2, delay=1, n_centers=3, drift=False)
            # If it doesn't raise, error should be ~0 (constant is trivially fit)
            self.assertAlmostEqual(m["in_sample_rmse"], 0.0, places=6)
        except (RuntimeError, np.linalg.LinAlgError):
            pass  # singular is acceptable

    def test_large_n_centers(self):
        s = _sine_series(1000)
        m = fit_rbf(s, dim=3, delay=1, n_centers=50)
        self.assertTrue(np.isfinite(m["in_sample_fce"]))


if __name__ == "__main__":
    unittest.main()
