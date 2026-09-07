"""Tests for ``nolitisea.linear.ar_model``."""

import unittest

import numpy as np

from nolitisea.linear.ar_model import fit_ar_model, iterate_ar_model


def _build_design(series, order):
    """Test-side regressor/target construction with explicit loops.

    ``ar-model`` indexing: row ``k`` holds
    ``x_i(t - j)`` at column ``i * order + j`` for ``t = order - 1 + k``.
    """
    s = np.asarray(series, dtype=np.float64)
    length, dim = s.shape
    n_valid = length - order
    design = np.zeros((n_valid, dim * order))
    target = np.zeros((n_valid, dim))
    for k in range(n_valid):
        t = order - 1 + k
        for i in range(dim):
            for j in range(order):
                design[k, i * order + j] = s[t - j, i]
        target[k] = s[t + 1]
    return design, target


class TestUnivariateHandComputed(unittest.TestCase):
    """Hand-verifiable exact cases (zero-mean input, no demeaning shift)."""

    def test_two_points_exact_fit(self):
        """x = [-2, 1]: demeaned y = [-1.5, 1.5], a = (1.5*-1.5)/2.25 = -1,
        single equation fits exactly."""
        fit = fit_ar_model([-2.0, 1.0], 1)
        self.assertAlmostEqual(fit["coeffs"][0, 0, 0], -1.0, places=15)
        np.testing.assert_allclose(fit["residuals"], [[0.0]], atol=1e-15)
        self.assertAlmostEqual(fit["forecast_errors"][0], 0.0, places=15)
        self.assertAlmostEqual(fit["average_forecast_error"], 0.0, places=15)
        self.assertAlmostEqual(fit["mean"][0], -0.5, places=15)

    def test_three_points_hand_case(self):
        """x = [-1, 1, 0]: a = -1/2, residuals [1/2, 1/2], fe = 1/2."""
        fit = fit_ar_model([-1.0, 1.0, 0.0], 1)
        self.assertAlmostEqual(fit["coeffs"][0, 0, 0], -0.5, places=15)
        np.testing.assert_allclose(fit["residuals"], [[0.5], [0.5]], atol=1e-15)
        self.assertAlmostEqual(fit["forecast_errors"][0], 0.5, places=15)
        self.assertAlmostEqual(fit["average_forecast_error"], 0.5, places=15)

    def test_matches_closed_form_ar1(self):
        """AR(1) estimate equals the closed-form demeaned OLS solution."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(500)
        fit = fit_ar_model(x, 1)
        y = x - x.mean()
        expected = np.dot(y[1:], y[:-1]) / np.dot(y[:-1], y[:-1])
        self.assertAlmostEqual(fit["coeffs"][0, 0, 0], expected, places=12)

        resid = y[1:] - expected * y[:-1]
        np.testing.assert_allclose(
            fit["residuals"][:, 0], resid, rtol=1e-10, atol=1e-12
        )
        expected_fe = np.sqrt(np.sum(resid**2) / (x.size - 1))
        self.assertAlmostEqual(fit["forecast_errors"][0], expected_fe, places=12)
        self.assertAlmostEqual(fit["average_forecast_error"], expected_fe, places=12)


class TestMatchesTiseanLayout(unittest.TestCase):
    """Structural checks against a literal transcription of the C loops."""

    def test_matches_c_loop_transcription(self):
        """Normal equations built with the exact C indexing agree."""
        rng = np.random.default_rng(3)
        n, dim, order = 120, 2, 3
        s = rng.standard_normal((n, dim))
        fit = fit_ar_model(s, order)

        y = s - s.mean(axis=0)
        norm = 1.0 / (n - order)
        mat = np.zeros((dim * order, dim * order))
        for i1 in range(dim):
            for i2 in range(order):
                for j1 in range(dim):
                    for j2 in range(order):
                        acc = 0.0
                        for t in range(order - 1, n - 1):
                            acc += y[t - i2, i1] * y[t - j2, j1]
                        mat[i1 * order + i2, j1 * order + j2] = acc * norm

        for comp in range(dim):
            vec = np.zeros(dim * order)
            for i1 in range(dim):
                for i2 in range(order):
                    acc = 0.0
                    for t in range(order - 1, n - 1):
                        acc += y[t + 1, comp] * y[t - i2, i1]
                    vec[i1 * order + i2] = acc * norm
            expected = np.linalg.solve(mat, vec)
            np.testing.assert_allclose(
                fit["coeffs"][comp].ravel(),
                expected,
                rtol=1e-10,
                atol=1e-12,
            )

    def test_matches_lstsq_on_design_matrix(self):
        """Solution equals OLS on the explicitly built design matrix."""
        rng = np.random.default_rng(4)
        for dim, order in ((1, 1), (1, 4), (3, 1), (2, 2)):
            s = rng.standard_normal((300, dim))
            fit = fit_ar_model(s, order)
            y = s - s.mean(axis=0)
            design, target = _build_design(y, order)
            expected, *_ = np.linalg.lstsq(design, target, rcond=None)
            # expected rows are (i, j, d); coeffs are (d, i, j)
            expected_coeffs = expected.reshape(dim, order, dim).transpose(2, 0, 1)
            np.testing.assert_allclose(
                fit["coeffs"], expected_coeffs, rtol=1e-9, atol=1e-11
            )


class TestExactRecovery(unittest.TestCase):
    """Data generated from an exact linear recursion is recovered exactly."""

    def test_univariate_ar2_unit_sum(self):
        """a1 + a2 = 1 keeps any constant offset invariant, so the demeaned
        series still satisfies the recursion exactly."""
        a1, a2 = 0.6, 0.4
        n = 200
        x = np.empty(n)
        x[0], x[1] = 1.0, -0.5
        for t in range(2, n):
            x[t] = a1 * x[t - 1] + a2 * x[t - 2]
        fit = fit_ar_model(x, 2)
        np.testing.assert_allclose(fit["coeffs"][0, 0, :], [a1, a2], atol=1e-10)
        np.testing.assert_allclose(fit["residuals"], 0.0, atol=1e-9)
        np.testing.assert_allclose(fit["forecast_errors"], 0.0, atol=1e-9)

    def test_multivariate_ar2_exact(self):
        """A 2-D second-order system is recovered exactly (vanishing
        residuals, including cross-component couplings) when the sample
        mean of the realisation is exactly zero."""
        a1 = np.array([[0.5, 0.1], [-0.2, 0.7]])
        a2 = np.array([[0.2, 0.0], [0.1, -0.3]])
        n = 300
        y1 = np.array([1.0, -2.0])
        y0 = np.array([0.3, 0.7])

        def generate(y0):
            s = np.empty((n, 2))
            s[0] = y0
            s[1] = y1
            for t in range(2, n):
                s[t] = a1 @ s[t - 1] + a2 @ s[t - 2]
            return s

        # The trajectory is linear in the initial state y0, hence so is
        # the sample mean: one Newton step tunes y0 so that the mean
        # vanishes exactly and demeaning keeps the recursion intact.
        m0 = generate(y0).mean(axis=0)
        cols = []
        for k in range(2):
            step = np.zeros(2)
            step[k] = 1.0
            cols.append(generate(y0 + step).mean(axis=0) - m0)
        y0 = y0 - np.linalg.solve(np.column_stack(cols), m0)

        s = generate(y0)
        np.testing.assert_allclose(s.mean(axis=0), 0.0, atol=1e-12)
        fit = fit_ar_model(s, 2)
        np.testing.assert_allclose(fit["coeffs"][:, :, 0], a1, atol=1e-9)
        np.testing.assert_allclose(fit["coeffs"][:, :, 1], a2, atol=1e-9)
        np.testing.assert_allclose(fit["residuals"], 0.0, atol=1e-8)
        self.assertLess(fit["average_forecast_error"], 1e-8)

    def test_residual_definition_pointwise(self):
        """residual[k] = y(t) - sum coeffs[d, i, j] * y_i(t-1-j) exactly."""
        rng = np.random.default_rng(6)
        n, dim, order = 80, 3, 2
        s = rng.standard_normal((n, dim))
        fit = fit_ar_model(s, order)
        y = s - s.mean(axis=0)
        for k in range(n - order):
            t = order + k
            for d in range(dim):
                pred = 0.0
                for i in range(dim):
                    for j in range(order):
                        pred += fit["coeffs"][d, i, j] * y[t - 1 - j, i]
                self.assertAlmostEqual(
                    fit["residuals"][k, d], y[t, d] - pred, places=12
                )


class TestForecastErrors(unittest.TestCase):
    """Normalisation of the forecast errors."""

    def test_population_normalisation(self):
        """fe_d = sqrt(sum(resid^2) / (N - p)) with no dof correction."""
        rng = np.random.default_rng(7)
        n, dim, order = 250, 2, 3
        s = rng.standard_normal((n, dim))
        fit = fit_ar_model(s, order)
        expected = np.sqrt(np.sum(fit["residuals"] ** 2, axis=0) / (n - order))
        np.testing.assert_allclose(fit["forecast_errors"], expected, rtol=1e-12)

    def test_average_is_rms_across_components(self):
        rng = np.random.default_rng(8)
        s = rng.standard_normal((200, 4))
        fit = fit_ar_model(s, 2)
        expected = np.sqrt(np.mean(fit["forecast_errors"] ** 2))
        self.assertIsInstance(fit["average_forecast_error"], float)
        self.assertAlmostEqual(fit["average_forecast_error"], expected, places=14)

    def test_white_noise_coeffs_small(self):
        """Fitting white noise yields near-zero coefficients."""
        rng = np.random.default_rng(9)
        s = rng.standard_normal(5000)
        fit = fit_ar_model(s, 4)
        self.assertTrue(np.all(np.abs(fit["coeffs"][0, 0, :]) < 0.1))

    def test_independent_components_uncoupled(self):
        """Columns of independent AR(1) processes stay uncoupled and the
        diagonal recovers the true coefficients (verifies the (t, dim)
        column orientation)."""
        n = 20000
        rng = np.random.default_rng(10)
        e = rng.standard_normal((n, 2))
        s = np.empty((n, 2))
        s[0] = e[0]
        for t in range(1, n):
            s[t, 0] = 0.9 * s[t - 1, 0] + e[t, 0]
            s[t, 1] = 0.1 * s[t - 1, 1] + e[t, 1]
        fit = fit_ar_model(s, 1)
        self.assertAlmostEqual(fit["coeffs"][0, 0, 0], 0.9, delta=0.02)
        self.assertAlmostEqual(fit["coeffs"][1, 1, 0], 0.1, delta=0.02)
        self.assertLess(abs(fit["coeffs"][0, 1, 0]), 0.05)
        self.assertLess(abs(fit["coeffs"][1, 0, 0]), 0.05)


class TestFitInputHandling(unittest.TestCase):
    """Shapes, mean removal and determinism of the fit."""

    def test_univariate_shapes(self):
        rng = np.random.default_rng(11)
        x = rng.standard_normal(150)
        fit = fit_ar_model(x, 3)
        self.assertEqual(fit["coeffs"].shape, (1, 1, 3))
        self.assertEqual(fit["residuals"].shape, (147, 1))
        self.assertEqual(fit["forecast_errors"].shape, (1,))
        self.assertEqual(fit["mean"].shape, (1,))

    def test_multivariate_shapes(self):
        rng = np.random.default_rng(12)
        s = rng.standard_normal((150, 5))
        fit = fit_ar_model(s, 2)
        self.assertEqual(fit["coeffs"].shape, (5, 5, 2))
        self.assertEqual(fit["residuals"].shape, (148, 5))
        self.assertEqual(fit["forecast_errors"].shape, (5,))
        self.assertEqual(fit["mean"].shape, (5,))

    def test_mean_subtracted_internally(self):
        """Adding a constant offset does not change the fitted model."""
        rng = np.random.default_rng(13)
        s = rng.standard_normal((200, 2)) * [1.0, 3.0]
        fit_raw = fit_ar_model(s, 2)
        fit_off = fit_ar_model(s + 100.0, 2)
        np.testing.assert_allclose(
            fit_off["coeffs"], fit_raw["coeffs"], rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            fit_off["forecast_errors"],
            fit_raw["forecast_errors"],
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(fit_off["mean"], fit_raw["mean"] + 100.0)

    def test_accepts_list_input(self):
        fit = fit_ar_model([1.0, -1.0, 0.5, 2.0, -0.5, 1.5], 2)
        self.assertEqual(fit["coeffs"].shape, (1, 1, 2))
        self.assertTrue(np.all(np.isfinite(fit["residuals"])))

    def test_deterministic(self):
        rng = np.random.default_rng(14)
        s = rng.standard_normal((100, 2))
        f1 = fit_ar_model(s, 2)
        f2 = fit_ar_model(s, 2)
        for key in ("coeffs", "residuals", "forecast_errors", "mean"):
            np.testing.assert_array_equal(f1[key], f2[key])
        self.assertEqual(f1["average_forecast_error"], f2["average_forecast_error"])


class TestIterateDeterministic(unittest.TestCase):
    """Noiseless iteration with explicit initial history."""

    def test_matches_manual_recursion_dim2_order1(self):
        a = np.array([[[0.6], [-0.1]], [[0.2], [0.5]]])
        init = np.array([[1.0, -2.0]])
        out = iterate_ar_model(a, 0.0, 25, seed=0, initial=init)
        h = init[0].copy()
        expected = []
        for _ in range(25):
            h = np.array([0.6 * h[0] - 0.1 * h[1], 0.2 * h[0] + 0.5 * h[1]])
            expected.append(h.copy())
        np.testing.assert_allclose(out, np.array(expected), rtol=1e-13, atol=1e-15)

    def test_matches_manual_recursion_dim2_order2(self):
        """Lag-2 terms exercise the circular-history bookkeeping."""
        c = np.zeros((2, 2, 2))
        c[0, 0, 0], c[0, 1, 1], c[1, 0, 1], c[1, 1, 0] = 0.5, -0.2, 0.3, 0.4
        init = np.array([[2.0, 1.0], [-1.0, 0.5]])
        out = iterate_ar_model(c, 0.0, 30, seed=0, initial=init)
        hist = init.copy()
        expected = []
        for _ in range(30):
            new = np.array(
                [
                    0.5 * hist[-1, 0] - 0.2 * hist[-2, 1],
                    0.4 * hist[-1, 1] + 0.3 * hist[-2, 0],
                ]
            )
            expected.append(new.copy())
            hist = np.vstack([hist, new[None, :]])
        np.testing.assert_allclose(out, np.array(expected), rtol=1e-13, atol=1e-15)

    def test_univariate_halving(self):
        """a = 0.5, x0 = 2: exact geometric decay 1, 0.5, 0.25, ..."""
        out = iterate_ar_model(np.array([[[0.5]]]), 0.0, 4, seed=0, initial=[2.0])
        np.testing.assert_allclose(out[:, 0], [1.0, 0.5, 0.25, 0.125], rtol=1e-15)

    def test_output_shape(self):
        a = np.zeros((3, 3, 2))
        out = iterate_ar_model(a, 0.0, 7, seed=0, initial=np.zeros((2, 3)))
        self.assertEqual(out.shape, (7, 3))


class TestIterateStatistical(unittest.TestCase):
    """Noise-driven iteration behaves like the underlying process."""

    def test_round_trip_recovers_coefficients(self):
        """Fit an AR(1) process, re-synthesise, refit: coefficients and
        innovation level are recovered."""
        n = 30000
        a_true = 0.8
        rng = np.random.default_rng(20)
        e = rng.standard_normal(n)
        x = np.empty(n)
        x[0] = e[0]
        for t in range(1, n):
            x[t] = a_true * x[t - 1] + e[t]
        fit = fit_ar_model(x, 1)
        self.assertAlmostEqual(fit["coeffs"][0, 0, 0], a_true, delta=0.02)

        sim = iterate_ar_model(
            fit["coeffs"], fit["forecast_errors"], n_steps=n, seed=21
        )
        refit = fit_ar_model(sim[:, 0], 1)
        self.assertAlmostEqual(refit["coeffs"][0, 0, 0], a_true, delta=0.05)
        self.assertAlmostEqual(
            refit["forecast_errors"][0], fit["forecast_errors"][0], delta=0.1
        )

    def test_stationary_variance(self):
        """Simulated variance matches sigma^2 / (1 - a^2) within 10%."""
        a_true = 0.9
        n = 60000
        rng = np.random.default_rng(22)
        e = rng.standard_normal(n)
        x = np.empty(n)
        x[0] = e[0]
        for t in range(1, n):
            x[t] = a_true * x[t - 1] + e[t]
        fit = fit_ar_model(x, 1)
        sim = iterate_ar_model(
            fit["coeffs"], fit["forecast_errors"], n_steps=n, seed=23
        )
        theo_var = fit["forecast_errors"][0] ** 2 / (1.0 - a_true**2)
        self.assertLess(abs(sim[:, 0].var() - theo_var) / theo_var, 0.1)

    def test_zero_sigma_matches_noise_free_rng_stream(self):
        """sigma = 0 produces exactly the deterministic recursion."""
        a = np.array([[[0.5]]])
        out = iterate_ar_model(a, 0.0, 10, seed=5, initial=[1.0])
        np.testing.assert_allclose(out[:, 0], 0.5 ** np.arange(1, 11), rtol=1e-15)

    def test_scalar_sigma_broadcast_bitwise(self):
        """Scalar sigma equals an all-equal vector sigma bit for bit."""
        a = np.zeros((2, 2, 1))
        a[0, 0, 0], a[1, 1, 0] = 0.5, 0.3
        s1 = iterate_ar_model(a, 0.5, 100, seed=31)
        s2 = iterate_ar_model(a, [0.5, 0.5], 100, seed=31)
        np.testing.assert_array_equal(s1, s2)

    def test_seed_reproducibility(self):
        a = np.zeros((2, 2, 2))
        r1 = iterate_ar_model(a, 1.0, 50, seed=32)
        r2 = iterate_ar_model(a, 1.0, 50, seed=32)
        r3 = iterate_ar_model(a, 1.0, 50, seed=33)
        np.testing.assert_array_equal(r1, r2)
        self.assertFalse(np.array_equal(r1, r3))

    def test_default_initial_is_noise_warm_start(self):
        """Default history is drawn from N(0, sigma): finite output with a
        seeded generator, and the warm start consumes randomness."""
        a = np.array([[[0.0]]])  # y(t) = e(t): output is pure noise
        out1 = iterate_ar_model(a, 1.0, 100, seed=34)
        out2 = iterate_ar_model(a, 1.0, 100, seed=34)
        np.testing.assert_array_equal(out1, out2)
        self.assertTrue(np.all(np.isfinite(out1)))
        self.assertGreater(np.std(out1), 0.5)


class TestRestoreMean(unittest.TestCase):
    """``mean=`` shifts the iteration back to original units."""

    def test_mean_added_univariate_deterministic(self):
        """sigma = 0: restored output is the demeaned recursion plus mean."""
        a = np.array([[[0.5]]])
        raw = iterate_ar_model(a, 0.0, 4, seed=0, initial=[2.0])
        restored = iterate_ar_model(a, 0.0, 4, seed=0, initial=[2.0], mean=3.0)
        np.testing.assert_allclose(restored[:, 0], raw[:, 0] + 3.0, rtol=1e-15)

    def test_mean_vector_multivariate(self):
        a = np.zeros((2, 2, 1))
        a[0, 0, 0], a[1, 1, 0] = 0.5, 0.3
        raw = iterate_ar_model(a, 0.0, 10, seed=0,
                               initial=np.zeros((1, 2)))
        restored = iterate_ar_model(a, 0.0, 10, seed=0,
                                    initial=np.zeros((1, 2)),
                                    mean=[5.0, -2.0])
        np.testing.assert_allclose(restored, raw + np.array([5.0, -2.0]))

    def test_scalar_mean_broadcast_bitwise(self):
        """Scalar mean equals an all-equal mean vector bit for bit."""
        a = np.zeros((2, 2, 1))
        a[0, 0, 0], a[1, 1, 0] = 0.5, 0.3
        s1 = iterate_ar_model(a, 0.5, 50, seed=35, mean=4.0)
        s2 = iterate_ar_model(a, 0.5, 50, seed=35, mean=[4.0, 4.0])
        np.testing.assert_array_equal(s1, s2)

    def test_end_to_end_continuation_in_levels(self):
        """Offset AR(1) data: deterministic level forecast continues from
        the data in original units, with the initial history demeaned."""
        rng = np.random.default_rng(36)
        offset = 7.5
        n = 2000
        sig = np.empty(n)
        e = rng.standard_normal(n)
        sig[0] = e[0]
        for t in range(1, n):
            sig[t] = 0.8 * sig[t - 1] + e[t]
        x = sig + offset
        fit = fit_ar_model(x, 1)
        init = (x - fit["mean"])[-1:]  # model-domain (demeaned) history
        fc = iterate_ar_model(fit["coeffs"], 0.0, 3, seed=0,
                              initial=init, mean=fit["mean"])[:, 0]
        # deterministic forecast decays geometrically from the last
        # observation towards the level mean: fc[k] - m = a^(k+1)*(x[-1] - m)
        m = fit["mean"][0]
        a = fit["coeffs"][0, 0, 0]
        expect = m + a ** np.arange(1, 4) * (x[-1] - m)
        np.testing.assert_allclose(fc, expect, rtol=1e-12)
        # fitted mean and levels are in original (offset) units, not
        # demeaned units (which would be centred at zero)
        self.assertAlmostEqual(m, offset, delta=0.1)
        self.assertAlmostEqual(fc[0], a * x[-1] + (1 - a) * m, places=12)

    def test_restored_realisation_has_fit_mean(self):
        """A restored realisation of a zero-mean process plus offset has
        the offset as its sample mean."""
        a = np.array([[[0.5]]])
        out = iterate_ar_model(a, 1.0, 20000, seed=37, mean=10.0)
        self.assertAlmostEqual(float(out[:, 0].mean()), 10.0, delta=0.1)

    def test_default_is_still_mean_subtracted(self):
        """mean=None keeps the legacy demeaned output."""
        a = np.array([[[0.0]]])
        out = iterate_ar_model(a, 1.0, 20000, seed=38)
        self.assertLess(abs(float(out[:, 0].mean())), 0.1)


class TestValidation(unittest.TestCase):
    """Parameter validation and error handling."""

    def test_order_zero_raises(self):
        with self.assertRaises(ValueError):
            fit_ar_model(np.ones(10), 0)

    def test_order_negative_raises(self):
        with self.assertRaises(ValueError):
            fit_ar_model(np.ones(10), -1)

    def test_order_equal_length_raises(self):
        with self.assertRaises(ValueError):
            fit_ar_model(np.ones(10), 10)

    def test_order_above_length_raises(self):
        with self.assertRaises(ValueError):
            fit_ar_model(np.ones(10), 11)

    def test_3d_input_raises(self):
        with self.assertRaises(ValueError):
            fit_ar_model(np.zeros((4, 3, 2)), 1)

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            fit_ar_model(np.zeros(0), 1)
        with self.assertRaises(ValueError):
            fit_ar_model(np.zeros((0, 2)), 1)

    def test_zero_width_input_raises(self):
        with self.assertRaises(ValueError):
            fit_ar_model(np.zeros((5, 0)), 1)

    def test_constant_series_raises_linalgerror(self):
        """Demeaned constant series gives a singular normal matrix."""
        with self.assertRaises(np.linalg.LinAlgError):
            fit_ar_model(np.ones(20), 1)

    def test_iterate_coeffs_wrong_ndim_raises(self):
        with self.assertRaises(ValueError):
            iterate_ar_model(np.zeros((2, 2)), 1.0, 5)

    def test_iterate_coeffs_nonsquare_raises(self):
        with self.assertRaises(ValueError):
            iterate_ar_model(np.zeros((1, 2, 1)), 1.0, 5)

    def test_iterate_coeffs_zero_order_raises(self):
        with self.assertRaises(ValueError):
            iterate_ar_model(np.zeros((2, 2, 0)), 1.0, 5)

    def test_iterate_sigma_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            iterate_ar_model(np.zeros((2, 2, 1)), [1.0, 2.0, 3.0], 5)

    def test_iterate_sigma_negative_raises(self):
        with self.assertRaises(ValueError):
            iterate_ar_model(np.zeros((2, 2, 1)), -0.5, 5)

    def test_iterate_sigma_nan_raises(self):
        with self.assertRaises(ValueError):
            iterate_ar_model(np.zeros((2, 2, 1)), np.nan, 5)

    def test_iterate_mean_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            iterate_ar_model(np.zeros((2, 2, 1)), 1.0, 5, mean=[1.0, 2.0, 3.0])

    def test_iterate_mean_nan_raises(self):
        with self.assertRaises(ValueError):
            iterate_ar_model(np.zeros((2, 2, 1)), 1.0, 5, mean=[1.0, np.nan])

    def test_iterate_nsteps_below_one_raises(self):
        for bad in (0, -3):
            with self.assertRaises(ValueError):
                iterate_ar_model(np.zeros((2, 2, 1)), 1.0, bad)

    def test_iterate_initial_wrong_shape_raises(self):
        a = np.zeros((2, 2, 2))
        with self.assertRaises(ValueError):
            iterate_ar_model(a, 0.0, 5, initial=np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            iterate_ar_model(a, 0.0, 5, initial=np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            iterate_ar_model(np.array([[[0.5]]]), 0.0, 5, initial=[1.0, 2.0])

    def test_iterate_initial_1d_accepted_for_dim1(self):
        out = iterate_ar_model(np.array([[[0.5]]]), 0.0, 2, seed=0, initial=[2.0])
        self.assertEqual(out.shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
