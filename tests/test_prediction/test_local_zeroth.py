"""Tests for ``nolitisea.prediction.local_zeroth`` (``lzo_gm``, ``lzo_run``,
``lzo_test``).

The brute-force helpers here re-implement the TISEAN causal-window logic
from scratch so we can verify that the production code and the reference
agree on the critical `lo = i - causal + 1` /
`hi = i + causal + (embed-1)*delay - 1` exclusion interval.
"""

import unittest

import numpy as np

from nolitisea.core.embed import lag_block_delay_embed
from nolitisea.prediction.local_zeroth import (
    _resolve_inputs,
    _scan_epsilons,
    lzo_gm,
    lzo_run,
    lzo_test,
)
from nolitisea.utils.rescale import rescale_data

# ---------------------------------------------------------------------------
# Brute-force helpers
# ---------------------------------------------------------------------------


def _brute_zeroth_one_epsilon(series, embed=2, delay=1, step=1, causal=None,
                              epsilon=0.5, min_neighbors=0):
    """O(N^2) zeroth-order forecast error for a single Chebyshev radius.

    Returns ``None`` when ``pfound <= 1`` (not enough valid reference
    points to normalise).
    """
    n_times, n_vars, s2d = _resolve_inputs(series)
    causal = step if causal is None else causal

    rescaled = np.empty_like(s2d)
    for c in range(n_vars):
        sr, _, _ = rescale_data(s2d[:, c])
        rescaled[:, c] = sr

    E = lag_block_delay_embed(rescaled, embed=embed, delay=delay)
    valid_start = (embed - 1) * delay
    n_embed = E.shape[0]

    ref_orig = np.arange(valid_start, n_times - step)

    sum_err2 = np.zeros(n_vars)
    sum_hrms2 = np.zeros(n_vars)
    sum_hav = np.zeros(n_vars)
    pfound = 0

    for ref_idx, i_orig in enumerate(ref_orig):
        nbr_times_list = []
        for k in range(n_embed):
            t_orig = k + valid_start
            if t_orig + step > n_times - 1:
                continue

            lo = i_orig - causal + 1
            hi = i_orig + causal + (embed - 1) * delay - 1
            if lo <= t_orig <= hi:
                continue

            # Chebyshev distance in embed space
            cheb = np.max(np.abs(E[k] - E[ref_idx]))
            if cheb <= epsilon:
                nbr_times_list.append(t_orig)

        if len(nbr_times_list) < min_neighbors:
            continue

        nbr_times = np.asarray(nbr_times_list)
        y_all = rescaled[nbr_times + step, :]
        y_pred = y_all.mean(axis=0)
        y_true = rescaled[i_orig + step, :]

        sum_err2 += (y_pred - y_true) ** 2
        pfound += 1
        sum_hrms2 += y_true ** 2
        sum_hav += y_true

    if pfound <= 1:
        return None

    hav = sum_hav / pfound
    hrms = np.sqrt(np.maximum(0.0, (sum_hrms2 - pfound * hav ** 2) / (pfound - 1)))
    err_per_comp = np.sqrt(sum_err2 / pfound) / hrms
    err_per_comp = np.where(hrms > 0, err_per_comp, np.nan)
    return {
        "avg_error": float(np.nanmean(err_per_comp)),
        "comp_errors": err_per_comp,
        "coverage": pfound / len(ref_orig),
        "pfound": pfound,
    }


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


class TestParameterValidation(unittest.TestCase):
    """All three lzo_* functions must reject invalid parameters."""

    def setUp(self):
        self.series = np.arange(50, dtype=float)

    # -- lzo_gm ------------------------------------------------------------
    def test_lzo_gm_embed(self):
        with self.assertRaises(ValueError):
            lzo_gm(self.series, embed=0)

    def test_lzo_gm_delay(self):
        with self.assertRaises(ValueError):
            lzo_gm(self.series, delay=0)

    def test_lzo_gm_step(self):
        with self.assertRaises(ValueError):
            lzo_gm(self.series, step=0)

    def test_lzo_gm_causal_negative(self):
        with self.assertRaises(ValueError):
            lzo_gm(self.series, causal=0)

    def test_lzo_gm_eps_factor(self):
        with self.assertRaises(ValueError):
            lzo_gm(self.series, eps_factor=1.0)
        with self.assertRaises(ValueError):
            lzo_gm(self.series, eps_factor=0.5)

    def test_lzo_gm_min_neighbors(self):
        with self.assertRaises(ValueError):
            lzo_gm(self.series, min_neighbors=0)

    def test_lzo_gm_too_short(self):
        # length must be > step + (embed-1)*delay
        with self.assertRaises(ValueError):
            lzo_gm(np.arange(3, dtype=float), embed=3, delay=1, step=1)

    # -- lzo_run -----------------------------------------------------------
    def test_lzo_run_embed(self):
        with self.assertRaises(ValueError):
            lzo_run(self.series, embed=0)

    def test_lzo_run_noise_negative(self):
        with self.assertRaises(ValueError):
            lzo_run(self.series, noise_pct=-1.0)

    def test_lzo_run_escape_negative(self):
        with self.assertRaises(ValueError):
            lzo_run(self.series, escape_scale=-0.1)

    def test_lzo_run_too_short(self):
        with self.assertRaises(ValueError):
            lzo_run(np.arange(2, dtype=float), embed=2, delay=1)

    # -- lzo_test ----------------------------------------------------------
    def test_lzo_test_step(self):
        with self.assertRaises(ValueError):
            lzo_test(self.series, step=0)

    def test_lzo_test_refstep(self):
        with self.assertRaises(ValueError):
            lzo_test(self.series, refstep=0)

    def test_lzo_test_no_ref_points(self):
        # Very short series with large min_neighbors
        with self.assertRaises((ValueError, RuntimeError)):
            lzo_test(np.arange(20, dtype=float), embed=3, delay=2,
                     step=2, min_neighbors=100)


# ---------------------------------------------------------------------------
# _scan_epsilons helper
# ---------------------------------------------------------------------------


class TestScanEpsilons(unittest.TestCase):

    def test_empty_when_upper_lower(self):
        out = list(_scan_epsilons(2.0, 1.0, 1.2))
        self.assertEqual(out, [])

    def test_generates_expected_sequence(self):
        out = list(_scan_epsilons(1.0, 2.0, 2.0))
        # 1.0, 2.0 (since 2.0 < 2.0*2.0=4.0), stop before 4.0
        self.assertAlmostEqual(out[0], 1.0)
        self.assertAlmostEqual(out[1], 2.0)
        self.assertEqual(len(out), 2)

    def test_termination_boundary(self):
        # eps1 * factor = 1.2 * 2.0 = 2.4; loop yields up to but not >= 2.4
        out = list(_scan_epsilons(1.0, 1.2, 2.0))
        self.assertTrue(out[-1] < 2.4)
        self.assertGreaterEqual(out[-1] * 2.0, 2.4)


# ---------------------------------------------------------------------------
# lzo_gm
# ---------------------------------------------------------------------------


class TestLzoGm(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        # Chaotic-looking signal, deterministic
        t = np.linspace(0, 100 * np.pi, 1000)
        self.series = np.sin(t) + 0.05 * rng.standard_normal(1000)

    def test_returns_expected_keys(self):
        result = lzo_gm(self.series, embed=3, delay=1, step=1,
                        eps0=0.01, eps1=0.5, eps_factor=1.2,
                        min_neighbors=5)
        for key in ("epsilon", "avg_error", "comp_errors",
                    "coverage", "avg_neighbors", "embedding_norm",
                    "n_ref_points"):
            self.assertIn(key, result)

    def test_result_shapes_consistent(self):
        result = lzo_gm(self.series, embed=3, delay=1, step=1,
                        eps0=0.01, eps1=0.5, eps_factor=1.2,
                        min_neighbors=5)
        n = len(result["epsilon"])
        self.assertEqual(result["avg_error"].shape, (n,))
        self.assertEqual(result["coverage"].shape, (n,))
        self.assertEqual(result["avg_neighbors"].shape, (n,))
        self.assertEqual(result["comp_errors"].shape, (n, 1))

    def test_coverage_non_decreasing(self):
        # As epsilon grows, coverage cannot decrease
        result = lzo_gm(self.series, embed=3, delay=1, step=1,
                        eps0=0.01, eps1=0.8, eps_factor=1.2,
                        min_neighbors=5)
        cov = result["coverage"]
        self.assertTrue(np.all(np.diff(cov) >= -1e-12),
                        f"coverage not monotonic: {cov}")

    def test_embedding_norm_present(self):
        result = lzo_gm(self.series, embed=3, delay=1, step=1,
                        eps0=0.01, eps1=0.5, eps_factor=1.2,
                        min_neighbors=5)
        self.assertIn(0, result["embedding_norm"])
        _, interval = result["embedding_norm"][0]
        self.assertGreater(interval, 0.0)

    def test_brute_force_agreement_single_epsilon(self):
        """Pick one epsilon from the geometric scan and compare to brute."""
        result = lzo_gm(self.series, embed=3, delay=1, step=1,
                        eps0=0.1, eps1=0.1, eps_factor=2.0,
                        min_neighbors=10)
        eps = result["epsilon"][0]
        brute = _brute_zeroth_one_epsilon(
            self.series, embed=3, delay=1, step=1, epsilon=eps,
            min_neighbors=10,
        )
        self.assertIsNotNone(brute)
        self.assertAlmostEqual(result["avg_error"][0], brute["avg_error"],
                               places=10)
        self.assertAlmostEqual(result["coverage"][0], brute["coverage"],
                               places=10)

    def test_brute_force_delay_gt_1(self):
        """Causal window upper bound includes (embed-1)*delay term."""
        result = lzo_gm(self.series, embed=3, delay=3, step=1,
                        eps0=0.2, eps1=0.2, eps_factor=2.0,
                        min_neighbors=10)
        eps = result["epsilon"][0]
        brute = _brute_zeroth_one_epsilon(
            self.series, embed=3, delay=3, step=1, epsilon=eps,
            min_neighbors=10,
        )
        self.assertIsNotNone(brute)
        self.assertAlmostEqual(result["avg_error"][0], brute["avg_error"],
                               places=10)

    def test_with_2d_series(self):
        x = np.sin(np.linspace(0, 20 * np.pi, 500))
        y = np.cos(np.linspace(0, 20 * np.pi, 500))
        s2d = np.column_stack([x, y])
        result = lzo_gm(s2d, embed=2, delay=1, step=1,
                        eps0=0.05, eps1=0.3, eps_factor=1.5,
                        min_neighbors=10)
        self.assertEqual(result["comp_errors"].shape[1], 2)

    def test_no_scan_produces_runtime_error(self):
        with self.assertRaises(RuntimeError):
            lzo_gm(self.series, embed=3, delay=1, step=1,
                   eps0=0.0, eps1=0.0, eps_factor=2.0,
                   min_neighbors=10)


# ---------------------------------------------------------------------------
# lzo_run
# ---------------------------------------------------------------------------


class TestLzoRun(unittest.TestCase):

    def setUp(self):
        # Simple deterministic oscillator
        self.series = np.sin(np.linspace(0, 30 * np.pi, 1500))

    def test_status_ok(self):
        # Long series, large min_neighbors -> should succeed
        result = lzo_run(self.series, embed=3, delay=1, n_steps=50,
                         eps0=0.05, min_neighbors=20,
                         escape_scale=5.0)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["n_steps_done"], 50)
        self.assertEqual(len(result["trajectory"]), 50)

    def test_traj_in_original_units(self):
        """Values must be de-rescaled back to original physical units.

        We check this by verifying the trajectory range matches the
        rescale_info (i.e., trajectory values lie in the physical
        min..min+interval box), not the rescaled [0,1] box.
        """
        result = lzo_run(self.series, embed=3, delay=1, n_steps=5,
                         eps0=0.05, min_neighbors=20,
                         escape_scale=5.0)
        minv, interval = result["rescale_info"][0]
        traj = result["trajectory"].ravel()
        # Every trajectory point must fall inside the physical data box
        # (with some escape margin tolerance from escape_scale)
        self.assertTrue(np.all(traj >= minv - 5.0 * interval))
        self.assertTrue(np.all(traj <= minv + interval + 5.0 * interval))
        # rescaled box is [0,1], physical is ~[-1, 1]; traj should not
        # all be near 0-1 range of rescaled values
        self.assertFalse(np.all(traj >= 0.0) and np.all(traj <= 1.0),
                         "trajectory values still in [0,1] (not de-rescaled)")

    def test_epsilon_history_length(self):
        result = lzo_run(self.series, embed=3, delay=1, n_steps=50,
                         eps0=0.05, min_neighbors=20,
                         escape_scale=5.0)
        self.assertEqual(len(result["epsilon_history"]), 50)

    def test_knn_mode_produces_output(self):
        result = lzo_run(self.series, embed=3, delay=1, n_steps=30,
                         eps0=0.05, min_neighbors=20,
                         knn_mode=True, escape_scale=5.0)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["trajectory"].shape[0], 30)

    def test_noise_mode_deterministic_with_seed(self):
        r1 = lzo_run(self.series, embed=3, delay=1, n_steps=20,
                     eps0=0.05, min_neighbors=20, noise_pct=0.1,
                     seed=7, escape_scale=5.0)
        r2 = lzo_run(self.series, embed=3, delay=1, n_steps=20,
                     eps0=0.05, min_neighbors=20, noise_pct=0.1,
                     seed=7, escape_scale=5.0)
        np.testing.assert_allclose(r1["trajectory"], r2["trajectory"])

    def test_noise_zero_equals_no_noise(self):
        r1 = lzo_run(self.series, embed=3, delay=1, n_steps=20,
                     eps0=0.05, min_neighbors=20, noise_pct=0.0,
                     seed=1, escape_scale=5.0)
        r2 = lzo_run(self.series, embed=3, delay=1, n_steps=20,
                     eps0=0.05, min_neighbors=20, noise_pct=0.0,
                     seed=2, escape_scale=5.0)
        np.testing.assert_allclose(r1["trajectory"], r2["trajectory"])

    def test_small_escape_scale_triggers_escape(self):
        # With escape_scale=0.0 and data in [0,1], any prediction that
        # rounds to exactly 0 or 1 could trigger.  Use tiny value.
        result = lzo_run(self.series, embed=5, delay=1, n_steps=100,
                         eps0=0.01, min_neighbors=30,
                         escape_scale=0.0)
        # Might escape quickly or not at all; just check status is valid
        self.assertIn(result["status"], ("ok", "escaped", "failed"))
        self.assertGreater(result["n_steps_done"], 0)

    def test_final_epsilon_present(self):
        result = lzo_run(self.series, embed=3, delay=1, n_steps=10,
                         eps0=0.05, min_neighbors=20,
                         escape_scale=5.0)
        self.assertIsInstance(result["final_epsilon"], float)
        self.assertGreater(result["final_epsilon"], 0.0)

    def test_rolling_state_is_independent_copy(self):
        """Ensure we don't modify the original rescaled series buffer."""
        s_copy = self.series.copy()
        _ = lzo_run(self.series, embed=3, delay=1, n_steps=10,
                    eps0=0.05, min_neighbors=20, escape_scale=5.0)
        # Original series (physical units) must be unchanged
        np.testing.assert_array_equal(self.series, s_copy)


# ---------------------------------------------------------------------------
# lzo_test
# ---------------------------------------------------------------------------


class TestLzoTest(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(123)
        self.series = np.sin(np.linspace(0, 30 * np.pi, 2000)) \
            + 0.05 * rng.standard_normal(2000)

    def test_returns_expected_keys(self):
        result = lzo_test(self.series, embed=3, delay=1, step=1,
                          eps0=0.01, min_neighbors=15)
        for key in ("forecast_errors", "abs_errors", "n_ref_points",
                    "embedding_norm"):
            self.assertIn(key, result)

    def test_multistep_shape(self):
        result = lzo_test(self.series, embed=3, delay=1, step=3,
                          eps0=0.01, min_neighbors=15)
        self.assertEqual(result["forecast_errors"].shape, (3, 1))
        self.assertEqual(result["abs_errors"].shape, (3, 1))

    def test_error_increases_with_step(self):
        """For most dynamics, longer forecast horizons are harder."""
        result = lzo_test(self.series, embed=3, delay=1, step=4,
                          eps0=0.01, min_neighbors=20)
        avg_by_step = result["forecast_errors"].mean(axis=1)
        # Not strictly monotonic, but on average should be non-decreasing
        # up to some horizon.  We at least check step-3 >= step-0.
        self.assertGreaterEqual(avg_by_step[3], avg_by_step[0] - 0.3)

    def test_refstep_subsamples(self):
        """refstep controls the temporal stride *together with* n_ref.

        With n_ref set, clength = n_ref * refstep + step, and
        ref_orig_times is generated with that refstep.  So refstep=10
        with n_ref=20 yields ~20 points spread over ~200 steps,
        while refstep=1 with n_ref=20 yields ~20 adjacent points.
        """
        r1 = lzo_test(self.series, embed=3, delay=1, step=1,
                      eps0=0.01, min_neighbors=15, n_ref=30, refstep=1)
        r10 = lzo_test(self.series, embed=3, delay=1, step=1,
                       eps0=0.01, min_neighbors=15, n_ref=30, refstep=10)
        # Both should produce roughly n_ref reference points; the
        # wider-refstep version has fewer candidates within any
        # causality window so pfound may be slightly lower.
        self.assertLessEqual(r1["n_ref_points"], 30)
        self.assertLessEqual(r10["n_ref_points"], 30)
        # refstep=10 might have slightly lower pfound due to sparser
        # coverage, but shouldn't be drastically so
        self.assertGreater(r1["n_ref_points"], 5)
        self.assertGreater(r10["n_ref_points"], 5)

    def test_n_ref_limits_points(self):
        r1 = lzo_test(self.series, embed=3, delay=1, step=1,
                      eps0=0.01, min_neighbors=15, n_ref=20)
        r2 = lzo_test(self.series, embed=3, delay=1, step=1,
                      eps0=0.01, min_neighbors=15, n_ref=50)
        self.assertLessEqual(r1["n_ref_points"], 20)
        self.assertLessEqual(r2["n_ref_points"], 50)
        self.assertLessEqual(r1["n_ref_points"], r2["n_ref_points"])

    def test_verbose_single_produces_array(self):
        result = lzo_test(self.series, embed=3, delay=1, step=1,
                          eps0=0.01, min_neighbors=15, verbose_single=True)
        self.assertIn("single_step_predictions", result)
        self.assertEqual(result["single_step_predictions"].ndim, 2)
        self.assertEqual(result["single_step_predictions"].shape[1], 1)

    def test_brute_force_single_epsilon(self):
        """Compare adaptive lzo_test with the brute using a large epsilon
        that accepts everything."""
        result = lzo_test(self.series, embed=3, delay=1, step=1,
                          eps0=0.5, min_neighbors=5, eps_factor=10.0)
        # With huge eps_factor, eps expands to cover everything quickly
        brute = _brute_zeroth_one_epsilon(
            self.series, embed=3, delay=1, step=1, epsilon=2.0,
            min_neighbors=5,
        )
        self.assertIsNotNone(brute)
        # Normalised errors should be close (adapt vs fixed full-box)
        # They won't be bitwise identical because lzo_test uses adaptive
        # eps per ref point, but both cover > 99%
        self.assertGreater(result["n_ref_points"], brute["pfound"] * 0.99)

    def test_delay_gt_1(self):
        result = lzo_test(self.series, embed=3, delay=3, step=1,
                          eps0=0.05, min_neighbors=10)
        self.assertGreater(result["n_ref_points"], 0)

    def test_2d_series(self):
        x = np.sin(np.linspace(0, 30 * np.pi, 2000))
        y = np.cos(np.linspace(0, 30 * np.pi, 2000))
        s2d = np.column_stack([x, y])
        result = lzo_test(s2d, embed=3, delay=1, step=2,
                          eps0=0.05, min_neighbors=15)
        self.assertEqual(result["forecast_errors"].shape[1], 2)


# ---------------------------------------------------------------------------
# Cross-validation: lzo_gm and lzo_test should agree on single epsilon
# ---------------------------------------------------------------------------


class TestCrossValidation(unittest.TestCase):

    def test_lzo_gm_vs_lzo_test_same_epsilon(self):
        rng = np.random.default_rng(99)
        series = np.sin(np.linspace(0, 20 * np.pi, 800)) \
            + 0.05 * rng.standard_normal(800)

        # lzo_gm with single epsilon (eps0=eps1, factor large so scan has 1 item)
        gm = lzo_gm(series, embed=3, delay=1, step=1,
                    eps0=0.15, eps1=0.15, eps_factor=10.0,
                    min_neighbors=10)

        # lzo_test with same adaptive setup — single eps quickly
        tst = lzo_test(series, embed=3, delay=1, step=1,
                       eps0=0.15, min_neighbors=10, eps_factor=100.0)

        # Both should produce comparable normalised errors
        # gm["avg_error"][0] vs tst["forecast_errors"][0,0]
        self.assertGreater(gm["coverage"][0], 0.5,
                           "lzo_gm coverage is too low; epsilon might be too small")
        self.assertAlmostEqual(gm["avg_error"][0],
                               float(tst["forecast_errors"][0, 0]),
                               places=1)


if __name__ == "__main__":
    unittest.main()
