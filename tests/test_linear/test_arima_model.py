"""Tests for ``nolitisea.linear.arima_model``."""

import unittest

import numpy as np

from nolitisea.linear.ar_model import fit_ar_model
from nolitisea.linear.arima_model import (
    difference,
    fit_arima,
    iterate_arima_model,
)


def _arma11_sample(n, phi, theta, seed):
    """ARMA(1,1) realisation: y(t) = phi*y(t-1) + e(t) + theta*e(t-1)."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    y = np.empty(n)
    y[0] = e[0]
    e_hist = e[0]
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t] + theta * e_hist
        e_hist = e[t]
    return y


class TestDifference(unittest.TestCase):
    """The public ``difference`` helper (I step)."""

    def test_first_difference_values(self):
        np.testing.assert_array_equal(
            difference([0.0, 1.0, 1.0, 2.0], 1), [1.0, 0.0, 1.0]
        )

    def test_matches_np_diff_multivariate(self):
        rng = np.random.default_rng(0)
        s = rng.standard_normal((50, 3))
        for d in (1, 2, 4):
            np.testing.assert_array_equal(difference(s, d), np.diff(s, n=d, axis=0))

    def test_zero_order_returns_copy(self):
        s = np.array([1.0, 2.0, 3.0])
        out = difference(s, 0)
        np.testing.assert_array_equal(out, s)
        out[0] = 99.0
        self.assertEqual(s[0], 1.0)

    def test_shapes(self):
        self.assertEqual(difference(np.ones(10), 3).shape, (7,))
        self.assertEqual(difference(np.ones((10, 2)), 3).shape, (7, 2))

    def test_negative_d_raises(self):
        with self.assertRaises(ValueError):
            difference(np.ones(5), -1)

    def test_too_short_raises(self):
        with self.assertRaises(ValueError):
            difference(np.ones(3), 3)
        with self.assertRaises(ValueError):
            difference(np.ones((3, 2)), 4)

    def test_3d_input_raises(self):
        with self.assertRaises(ValueError):
            difference(np.zeros((3, 2, 2)), 1)


class TestPureArHandComputed(unittest.TestCase):
    """Hand-verifiable pure-AR cases (no refinement)."""

    def test_three_points(self):
        """x = [-1, 1, 0], poles=1: a = -1/2, residuals [1/2, 1/2]."""
        fit = fit_arima([-1.0, 1.0, 0.0], poles=1)
        self.assertAlmostEqual(fit["coeffs"][0, 0], -0.5, places=15)
        np.testing.assert_allclose(fit["residuals"], [[0.5], [0.5]], atol=1e-15)
        self.assertAlmostEqual(fit["forecast_errors"][0], 0.5, places=15)
        self.assertAlmostEqual(fit["average_forecast_error"], 0.5, places=15)
        np.testing.assert_array_equal(fit["regressors"], [[0, 0, 0]])
        self.assertEqual(fit["mean"][0], 0.0)
        self.assertEqual(fit["n_iterations"], 0)
        self.assertEqual(fit["residual_rms_change"].shape, (0, 1))
        self.assertEqual(fit["coeff_rms_change"].shape, (0,))

    def test_differenced_hand_case(self):
        """[0, 1, 1, 2] with d=1 -> [1, 0, 1], demeaned [1/3, -2/3, 1/3]:
        a = -4/5, residuals [-2/5, -1/5], fe = sqrt(1/10)."""
        fit = fit_arima([0.0, 1.0, 1.0, 2.0], poles=1, diff_order=1)
        self.assertAlmostEqual(fit["coeffs"][0, 0], -0.8, places=14)
        np.testing.assert_allclose(fit["residuals"], [[-0.4], [-0.2]], atol=1e-14)
        self.assertAlmostEqual(fit["forecast_errors"][0], np.sqrt(0.1), places=14)
        self.assertAlmostEqual(fit["mean"][0], 2.0 / 3.0, places=14)

    def test_loglikelihood_and_aic_pure_ar(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(200) + 5.0
        fit = fit_arima(x, poles=3)
        length = 200
        expected_ll = (
            -length * np.sum(np.log(fit["forecast_errors"]))
            - length * (1.0 + np.log(2.0 * np.pi)) / 2.0
        )
        self.assertAlmostEqual(fit["log_likelihood"], expected_ll, places=10)
        self.assertAlmostEqual(fit["aic"], 2.0 * 3 - 2.0 * expected_ll, places=10)

    def test_univariate_shapes(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(100)
        fit = fit_arima(x, poles=4)
        self.assertEqual(fit["coeffs"].shape, (1, 4))
        self.assertEqual(fit["regressors"].shape, (4, 3))
        self.assertEqual(fit["residuals"].shape, (96, 1))
        self.assertEqual(fit["forecast_errors"].shape, (1,))
        self.assertEqual(fit["mean"].shape, (1,))


class TestMatchesArModel(unittest.TestCase):
    """Without refinement the fit must coincide with ``fit_ar_model``."""

    def test_matches_fit_ar_model(self):
        rng = np.random.default_rng(3)
        for poles in (1, 4):
            x = rng.standard_normal(400)
            f_arima = fit_arima(x, poles=poles)
            f_ar = fit_ar_model(x, poles)
            # fit_ar_model stores coeffs as (dim, dim, order); flatten the
            # (component, lag) pairs to fit_arima's (dim, size) layout.
            np.testing.assert_allclose(
                f_arima["coeffs"],
                f_ar["coeffs"].reshape(f_arima["coeffs"].shape),
                rtol=1e-10,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                f_arima["residuals"], f_ar["residuals"], rtol=1e-10, atol=1e-12
            )
            np.testing.assert_allclose(
                f_arima["forecast_errors"],
                f_ar["forecast_errors"],
                rtol=1e-10,
                atol=1e-12,
            )
            self.assertAlmostEqual(
                f_arima["average_forecast_error"],
                f_ar["average_forecast_error"],
                places=12,
            )

    def test_regressor_layout_pure_ar(self):
        """dim=2, poles=3: (component, lag) with lag fastest, all kind 0."""
        fit = fit_arima(np.random.default_rng(4).standard_normal((100, 2)), poles=3)
        expected = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 2],
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 2],
            ]
        )
        np.testing.assert_array_equal(fit["regressors"], expected)

    def test_mean_subtracted_internally(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal(300)
        f1 = fit_arima(x, poles=2)
        f2 = fit_arima(x + 100.0, poles=2)
        np.testing.assert_allclose(f2["coeffs"], f1["coeffs"], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            f2["forecast_errors"], f1["forecast_errors"], rtol=1e-10
        )
        self.assertAlmostEqual(f2["mean"][0], f1["mean"][0] + 100.0, places=12)

    def test_deterministic(self):
        rng = np.random.default_rng(6)
        x = rng.standard_normal((150, 2))
        f1 = fit_arima(x, poles=2)
        f2 = fit_arima(x, poles=2)
        for key in ("coeffs", "regressors", "residuals", "forecast_errors", "mean"):
            np.testing.assert_array_equal(f1[key], f2[key])


class TestArimaRefinement(unittest.TestCase):
    """Iterative ARMA refinement (-P) behaviour."""

    def test_regressor_layout_arima(self):
        """dim=2, ar=2, ma=1: AR block then MA block (sources dim..2dim-1)."""
        fit = fit_arima(
            np.random.default_rng(7).standard_normal((200, 2)),
            poles=3,
            ar_order=2,
            ma_order=1,
            max_iter=2,
        )
        expected = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 2, 0],
                [1, 3, 0],
            ]
        )
        np.testing.assert_array_equal(fit["regressors"], expected)
        self.assertEqual(fit["coeffs"].shape, (2, 6))
        self.assertEqual(fit["poles"], 2)

    def test_arma11_recovery(self):
        phi, theta = 0.6, 0.4
        y = _arma11_sample(40000, phi, theta, seed=20)
        fit = fit_arima(y, poles=10, ar_order=1, ma_order=1, convergence=1e-4)
        self.assertAlmostEqual(fit["coeffs"][0, 0], phi, delta=0.02)
        self.assertAlmostEqual(fit["coeffs"][0, 1], theta, delta=0.02)
        self.assertLess(fit["n_iterations"], 50)
        self.assertLess(fit["residual_rms_change"][-1].max(), 1e-4)
        self.assertAlmostEqual(fit["forecast_errors"][0], 1.0, delta=0.05)

    def test_ma1_recovery_without_ar(self):
        """ar_order=0, ma_order=1 recovers the MA coefficient."""
        n = 30000
        theta = 0.5
        rng = np.random.default_rng(21)
        e = rng.standard_normal(n)
        y = np.empty(n)
        y[0] = e[0]
        for t in range(1, n):
            y[t] = e[t] + theta * e[t - 1]
        fit = fit_arima(y, poles=10, ar_order=0, ma_order=1, convergence=1e-4)
        self.assertAlmostEqual(fit["coeffs"][0, 0], theta, delta=0.05)

    def test_trace_shapes_and_max_iter(self):
        y = _arma11_sample(3000, 0.5, 0.3, seed=22)
        fit = fit_arima(
            y, poles=5, ar_order=1, ma_order=1, max_iter=5, convergence=1e-12
        )
        self.assertEqual(fit["n_iterations"], 5)
        self.assertEqual(fit["residual_rms_change"].shape, (5, 1))
        self.assertEqual(fit["coeff_rms_change"].shape, (5,))
        self.assertTrue(np.all(fit["coeff_rms_change"] > 0.0))

    def test_arima_aic_beats_pure_ar_on_arma_data(self):
        y = _arma11_sample(20000, 0.6, 0.4, seed=23)
        f_arima = fit_arima(y, poles=10, ar_order=1, ma_order=1, convergence=1e-4)
        # AR(2) has the same parameter count k=2 as ARIMA(1,1) but is
        # genuinely misspecified for ARMA(1,1) data, so the comparison is
        # robust (an AR(10) rival would win/lose by finite-sample noise).
        f_ar = fit_arima(y, poles=2)
        self.assertLess(f_arima["aic"], f_ar["aic"])
        # k = ar + ma for the refinement, k = poles for the pure AR fit.
        self.assertAlmostEqual(
            f_arima["aic"], 2.0 * 2 - 2.0 * f_arima["log_likelihood"], places=10
        )
        self.assertAlmostEqual(
            f_ar["aic"], 2.0 * 2 - 2.0 * f_ar["log_likelihood"], places=10
        )

    def test_multivariate_independent_components(self):
        """Two independent ARMA(1,1) columns stay uncoupled (layout check)."""
        n = 40000
        y1 = _arma11_sample(n, 0.7, 0.3, seed=24)
        # (-0.6, -0.4) is strongly identifiable (rho_1 ~ -0.76); a
        # near-white combination like (-0.4, 0.5) has estimate noise
        # ~0.04 and would make the tolerances a seed lottery.
        y2 = _arma11_sample(n, -0.6, -0.4, seed=25)
        s = np.column_stack([y1, y2])
        fit = fit_arima(s, poles=10, ar_order=1, ma_order=1, convergence=1e-4)
        # Column order: [AR x0, AR x1, MA e0, MA e1].
        self.assertAlmostEqual(fit["coeffs"][0, 0], 0.7, delta=0.03)
        self.assertAlmostEqual(fit["coeffs"][1, 1], -0.6, delta=0.03)
        self.assertAlmostEqual(fit["coeffs"][0, 2], 0.3, delta=0.03)
        self.assertAlmostEqual(fit["coeffs"][1, 3], -0.4, delta=0.03)
        self.assertLess(abs(fit["coeffs"][0, 1]), 0.05)
        self.assertLess(abs(fit["coeffs"][1, 0]), 0.05)
        self.assertLess(abs(fit["coeffs"][0, 3]), 0.05)
        self.assertLess(abs(fit["coeffs"][1, 2]), 0.05)

    def test_arima11_on_differenced_data(self):
        """ARIMA(1,1,0): random walk with AR(1) increments recovers rho."""
        n = 30000
        rho = 0.7
        rng = np.random.default_rng(26)
        e = rng.standard_normal(n)
        w = np.empty(n)
        w[0] = e[0]
        for t in range(1, n):
            w[t] = rho * w[t - 1] + e[t]
        walk = np.cumsum(w)
        fit = fit_arima(walk, poles=10, ar_order=1, diff_order=1, convergence=1e-4)
        self.assertAlmostEqual(fit["coeffs"][0, 0], rho, delta=0.03)
        self.assertEqual(fit["poles"], 1)


class TestLogLikelihoodAic(unittest.TestCase):
    """Log-likelihood / AIC bookkeeping."""

    def test_multivariate_formula(self):
        rng = np.random.default_rng(8)
        s = rng.standard_normal((300, 3))
        fit = fit_arima(s, poles=2)
        length = 300
        expected_ll = (
            -length * np.sum(np.log(fit["forecast_errors"]))
            - length * 3 * (1.0 + np.log(2.0 * np.pi)) / 2.0
        )
        self.assertAlmostEqual(fit["log_likelihood"], expected_ll, places=9)
        self.assertAlmostEqual(fit["aic"], 2.0 * 2 - 2.0 * expected_ll, places=9)

    def test_average_forecast_error_is_rms(self):
        rng = np.random.default_rng(9)
        s = rng.standard_normal((200, 4))
        fit = fit_arima(s, poles=2)
        expected = np.sqrt(np.mean(fit["forecast_errors"] ** 2))
        self.assertAlmostEqual(fit["average_forecast_error"], expected, places=14)

    def test_zero_forecast_error_raises(self):
        """Two points always fit an AR(1) exactly -> vanishing residual."""
        with self.assertRaises(ValueError):
            fit_arima([-2.0, 1.0], poles=1)


class TestIterate(unittest.TestCase):
    """Iteration with bootstrapped innovations."""

    @staticmethod
    def _fake_fit(coeffs, regressors, pool):
        return {
            "coeffs": np.asarray(coeffs, dtype=float),
            "regressors": np.asarray(regressors, dtype=int),
            "residuals": np.asarray(pool, dtype=float).reshape(-1, 1),
        }

    def test_ar1_recursion_exact(self):
        """Pool [1.0] makes every bootstrap draw 1: x(t) = 0.5x(t-1) + 1."""
        fit = self._fake_fit([[0.5]], [[0, 0, 0]], [1.0])
        out = iterate_arima_model(fit, 6, seed=0, initial=[3.0])
        expected = []
        h = 3.0
        for _ in range(6):
            h = 0.5 * h + 1.0
            expected.append(h)
        np.testing.assert_allclose(out[:, 0], expected, rtol=1e-15)

    def test_arma11_recursion_exact(self):
        """x(t) = 0.5x(t-1) + 0.3e(t-1) + e(t) with e-history warm start."""
        fit = self._fake_fit([[0.5, 0.3]], [[0, 0, 0], [1, 1, 0]], [1.0])
        out = iterate_arima_model(fit, 5, seed=0, initial=[2.0])
        xh, eh = 2.0, 1.0
        expected = []
        for _ in range(5):
            xh = 0.5 * xh + 0.3 * eh + 1.0
            eh = 1.0
            expected.append(xh)
        np.testing.assert_allclose(out[:, 0], expected, rtol=1e-15)

    def test_ar2_lag_recursion_exact(self):
        """Lag-1 and lag-2 terms exercise the circular history buffer."""
        fit = self._fake_fit([[0.6, -0.2]], [[0, 0, 0], [0, 0, 1]], [2.0])
        # initial is chronological: [oldest, most recent] = [1.0, 3.0]
        out = iterate_arima_model(fit, 20, seed=0, initial=[1.0, 3.0])
        h1, h2 = 1.0, 3.0  # h2 = most recent sample
        expected = []
        for _ in range(20):
            new = 0.6 * h2 - 0.2 * h1 + 2.0
            expected.append(new)
            h1, h2 = h2, new
        np.testing.assert_allclose(out[:, 0], expected, rtol=1e-14)

    def test_seed_reproducibility(self):
        y = _arma11_sample(5000, 0.6, 0.4, seed=27)
        fit = fit_arima(y, poles=10, ar_order=1, ma_order=1, max_iter=3)
        r1 = iterate_arima_model(fit, 100, seed=30)
        r2 = iterate_arima_model(fit, 100, seed=30)
        r3 = iterate_arima_model(fit, 100, seed=31)
        np.testing.assert_array_equal(r1, r2)
        self.assertFalse(np.array_equal(r1, r3))

    def test_round_trip_recovers_parameters(self):
        phi, theta = 0.6, 0.4
        y = _arma11_sample(40000, phi, theta, seed=28)
        fit = fit_arima(y, poles=10, ar_order=1, ma_order=1, convergence=1e-4)
        sim = iterate_arima_model(fit, n_steps=40000, seed=29)
        refit = fit_arima(sim[:, 0], poles=10, ar_order=1, ma_order=1, convergence=1e-4)
        self.assertAlmostEqual(refit["coeffs"][0, 0], phi, delta=0.05)
        self.assertAlmostEqual(refit["coeffs"][0, 1], theta, delta=0.05)

    def test_output_shape(self):
        y = _arma11_sample(5000, 0.6, 0.4, seed=30)
        fit = fit_arima(y, poles=10, ar_order=1, ma_order=1, max_iter=3)
        out = iterate_arima_model(fit, 50, seed=31)
        self.assertEqual(out.shape, (50, 1))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_initial_1d_accepted_for_dim1(self):
        """A 1-D initial of shape (poles,) is accepted when dim == 1."""
        fit = self._fake_fit([[1.0]], [[0, 0, 0]], [1.0])
        out = iterate_arima_model(fit, 3, seed=0, initial=[7.0])
        # x(t) = x(t-1) + 1 with bootstrap pool [1.0]
        np.testing.assert_allclose(out[:, 0], [8.0, 9.0, 10.0], rtol=1e-15)

    def test_initial_wrong_shape_raises(self):
        fit = self._fake_fit([[0.6, -0.2]], [[0, 0, 0], [0, 0, 1]], [1.0])
        with self.assertRaises(ValueError):
            iterate_arima_model(fit, 5, initial=np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            iterate_arima_model(fit, 5, initial=np.zeros(3))

    def test_inconsistent_fit_raises(self):
        fit = self._fake_fit([[0.5]], [[0, 0, 0]], [1.0])
        bad = dict(fit)
        bad["regressors"] = np.array([[0, 0, 0], [1, 1, 0]])
        with self.assertRaises(ValueError):
            iterate_arima_model(bad, 5)
        bad = dict(fit)
        bad["residuals"] = np.zeros((0, 1))
        with self.assertRaises(ValueError):
            iterate_arima_model(bad, 5)
        bad = dict(fit)
        bad["coeffs"] = np.array([0.5])
        with self.assertRaises(ValueError):
            iterate_arima_model(bad, 5)
        with self.assertRaises(ValueError):
            iterate_arima_model(fit, 0)
        with self.assertRaises(KeyError):
            iterate_arima_model({"coeffs": fit["coeffs"]}, 5)


class TestRestoreLevel(unittest.TestCase):
    """``restore_level=True`` undifferences the simulation."""

    @staticmethod
    def _differenced_initial(fit, series):
        """Last ``poles`` demeaned differenced samples as model history."""
        s = np.diff(series, n=fit["diff_order"], axis=0)
        return (s - fit["mean"])[-fit["poles"]:]

    def test_fit_stores_diff_information(self):
        y = _arma11_sample(2000, 0.6, 0.4, seed=70)
        fit0 = fit_arima(y, poles=6)
        self.assertEqual(fit0["diff_order"], 0)
        self.assertEqual(fit0["diff_anchors"].shape, (0, 1))

        fit1 = fit_arima(y, poles=6, ar_order=1, diff_order=1, ma_order=1)
        self.assertEqual(fit1["diff_order"], 1)
        self.assertEqual(fit1["diff_anchors"].shape, (1, 1))
        np.testing.assert_allclose(fit1["diff_anchors"][0], y[-1])

        fit2 = fit_arima(y, poles=6, ar_order=1, diff_order=2, ma_order=1)
        self.assertEqual(fit2["diff_anchors"].shape, (2, 1))
        np.testing.assert_allclose(fit2["diff_anchors"][0], y[-1])
        np.testing.assert_allclose(fit2["diff_anchors"][1], np.diff(y)[-1])

    def test_restore_d0_adds_mean(self):
        y = _arma11_sample(3000, 0.6, 0.4, seed=71)
        fit = fit_arima(y, poles=8, ar_order=1, ma_order=1, max_iter=3)
        raw = iterate_arima_model(fit, 40, seed=72)
        restored = iterate_arima_model(fit, 40, seed=72, restore_level=True)
        np.testing.assert_allclose(restored, raw + fit["mean"])

    def test_restore_d1_matches_manual_cumsum(self):
        rng = np.random.default_rng(73)
        y = 0.05 * np.arange(3000) + rng.standard_normal(3000)
        fit = fit_arima(y, poles=8, ar_order=1, diff_order=1, ma_order=1,
                        max_iter=3)
        init = self._differenced_initial(fit, y)
        raw = iterate_arima_model(fit, 50, seed=74, initial=init)
        restored = iterate_arima_model(fit, 50, seed=74, initial=init,
                                       restore_level=True)
        manual = y[-1] + np.cumsum(raw[:, 0] + fit["mean"][0])
        np.testing.assert_allclose(restored[:, 0], manual)
        self.assertEqual(restored.shape, (50, 1))

    def test_restore_d2_matches_manual_double_cumsum(self):
        rng = np.random.default_rng(75)
        y = 0.0002 * np.arange(4000) ** 2 + rng.standard_normal(4000)
        fit = fit_arima(y, poles=8, ar_order=1, diff_order=2, ma_order=1,
                        max_iter=3)
        init = self._differenced_initial(fit, y)
        raw = iterate_arima_model(fit, 30, seed=76, initial=init)
        restored = iterate_arima_model(fit, 30, seed=76, initial=init,
                                       restore_level=True)
        level1 = np.diff(y)[-1] + np.cumsum(raw[:, 0] + fit["mean"][0])
        manual = y[-1] + np.cumsum(level1)
        np.testing.assert_allclose(restored[:, 0], manual)

    def test_restore_multivariate(self):
        rng = np.random.default_rng(77)
        y = np.column_stack([np.cumsum(rng.standard_normal(2000)),
                             np.cumsum(rng.standard_normal(2000))])
        fit = fit_arima(y, poles=8, ar_order=1, diff_order=1, ma_order=1,
                        max_iter=3)
        init = self._differenced_initial(fit, y)
        raw = iterate_arima_model(fit, 25, seed=78, initial=init)
        restored = iterate_arima_model(fit, 25, seed=78, initial=init,
                                       restore_level=True)
        manual = y[-1] + np.cumsum(raw + fit["mean"], axis=0)
        np.testing.assert_allclose(restored, manual)

    def test_restore_without_diff_info_raises(self):
        fit = {
            "coeffs": np.array([[0.5]]),
            "regressors": np.array([[0, 0, 0]]),
            "residuals": np.ones((10, 1)),
        }
        with self.assertRaises(ValueError):
            iterate_arima_model(fit, 5, restore_level=True)
        # Default behaviour is unaffected by the missing entries.
        out = iterate_arima_model(fit, 5, seed=0, initial=[1.0])
        self.assertEqual(out.shape, (5, 1))


class TestValidation(unittest.TestCase):
    """Parameter validation and error handling."""

    def test_poles_below_one_raises(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                fit_arima(np.ones(10), poles=bad)

    def test_poles_ge_length_raises(self):
        with self.assertRaises(ValueError):
            fit_arima(np.ones(10), poles=10)
        with self.assertRaises(ValueError):
            fit_arima(np.ones(10), poles=11)

    def test_3d_input_raises(self):
        with self.assertRaises(ValueError):
            fit_arima(np.zeros((4, 3, 2)), poles=1)

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            fit_arima(np.zeros(0), poles=1)
        with self.assertRaises(ValueError):
            fit_arima(np.zeros((0, 2)), poles=1)

    def test_zero_width_input_raises(self):
        with self.assertRaises(ValueError):
            fit_arima(np.zeros((5, 0)), poles=1)

    def test_diff_order_validation(self):
        with self.assertRaises(ValueError):
            fit_arima(np.ones(10), diff_order=-1)
        with self.assertRaises(ValueError):
            fit_arima(np.ones(10), diff_order=10)
        with self.assertRaises(ValueError):
            fit_arima(np.ones(10), diff_order=11)

    def test_negative_ar_ma_orders_raise(self):
        with self.assertRaises(ValueError):
            fit_arima(np.ones(20), ar_order=-1)
        with self.assertRaises(ValueError):
            fit_arima(np.ones(20), ma_order=-1)

    def test_ar_ma_ge_length_raises(self):
        with self.assertRaises(ValueError):
            fit_arima(np.ones(20), ar_order=20, ma_order=0)
        with self.assertRaises(ValueError):
            fit_arima(np.ones(20), ar_order=0, ma_order=21)

    def test_max_iter_below_one_raises(self):
        with self.assertRaises(ValueError):
            fit_arima(np.ones(50), ar_order=1, ma_order=1, max_iter=0)

    def test_convergence_validation(self):
        for bad in (0.0, -1e-3, np.nan, np.inf):
            with self.assertRaises(ValueError):
                fit_arima(np.ones(50), ar_order=1, ma_order=1, convergence=bad)

    def test_sample_range_exhaustion_raises(self):
        """offset grows by max(ar, ma) per iteration; a short series runs
        out of regression samples and must fail loudly."""
        rng = np.random.default_rng(10)
        with self.assertRaises(ValueError):
            fit_arima(
                rng.standard_normal(30), poles=10, ar_order=5, ma_order=5, max_iter=50
            )

    def test_singular_normal_equations_raise(self):
        """Constant differenced series -> singular regression matrix."""
        with self.assertRaises(np.linalg.LinAlgError):
            fit_arima(np.ones(20), poles=1)


if __name__ == "__main__":
    unittest.main()
