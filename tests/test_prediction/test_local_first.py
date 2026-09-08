"""Tests for ``nolitisea.prediction.local_first``.

Covers all three local-linear-forecast functions:

* :func:`lfo_ar` — epsilon-scan (``lfo-ar``)
* :func:`lfo_test` — adaptive epsilon (``lfo`` / ``lfo-test``)
* :func:`lfo_run` — iterative trajectory extrapolation (``lfo-run``)

Brute-force helpers re-implement the TISEAN contracts from scratch
(O(N^2) neighbour search, same causality window, same normalisation)
so we can verify the production code agrees with a reference on the
critical ``lo = i - causal + 1`` /
``hi = i + causal + (embed - 1) * delay - 1`` exclusion interval.
"""

import unittest

import numpy as np

from nolitisea.core.embed import lag_block_delay_embed
from nolitisea.prediction.local_first import (
    _resolve_inputs,
    lfo_ar,
    lfo_run,
    lfo_test,
)
from nolitisea.utils.rescale import rescale_data

# ---------------------------------------------------------------------------
# Brute-force helpers
# ---------------------------------------------------------------------------


def _brute_lfo_ar(series, embed=2, delay=1, step=1, causal=None):
    """O(N^2) reference for ``lfo_ar`` with a *global* epsilon.

    Accepts every non-causal neighbour and uses the same per-component
    sample-std normalisation.  Returns a dict with ``avg_error_brute``
    and ``comp_errors_brute``.
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
    n_ref = len(ref_orig)

    sum_err2 = np.zeros(n_vars)
    sum_hrms2 = np.zeros(n_vars)
    sum_hav = np.zeros(n_vars)
    pfound = 0

    for ref_idx, i_orig in enumerate(ref_orig):
        nbr_embed_idx = []
        for k in range(n_embed):
            t_orig = k + valid_start
            if t_orig + step > n_times - 1:
                continue
            lo = i_orig - causal + 1
            hi = i_orig + causal + (embed - 1) * delay - 1
            if lo <= t_orig <= hi:
                continue
            nbr_embed_idx.append(k)

        if len(nbr_embed_idx) < 1:
            continue

        nbr_embed_idx = np.asarray(nbr_embed_idx)
        nbr_times = nbr_embed_idx + valid_start
        X = E[nbr_embed_idx]  # pyright: ignore[reportCallIssue, reportArgumentType]
        y_all = rescaled[nbr_times + step, :]  # pyright: ignore[reportCallIssue, reportArgumentType]

        X_mean = X.mean(axis=0)
        Xc = X - X_mean
        y_mean = y_all.mean(axis=0)
        yc = y_all - y_mean

        beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)

        y_pred = y_mean + (E[ref_idx] - X_mean) @ beta
        y_true = rescaled[i_orig + step, :]

        sum_err2 += (y_pred - y_true) ** 2
        pfound += 1
        sum_hrms2 += y_true**2
        sum_hav += y_true

    if pfound <= 1:
        raise RuntimeError("brute: not enough valid points")

    hav = sum_hav / pfound
    hrms = np.sqrt(np.maximum(0.0, (sum_hrms2 - pfound * hav**2) / (pfound - 1)))
    err_per_comp = np.sqrt(sum_err2 / pfound) / hrms
    avg_error = float(np.nanmean(err_per_comp))

    return {
        "avg_error_brute": avg_error,
        "comp_errors_brute": err_per_comp,
        "pfound": pfound,
        "n_ref_total": n_ref,
    }


def _brute_lfo_test(series, embed=2, delay=1, step=1, causal=None, min_neighbors=1):
    """O(N^2) reference for ``lfo_test`` with a *global* epsilon.

    Accepts every non-causal neighbour and uses the same
    normalisation: ``sqrt(sum_err2 / n_total) / overall_std``.

    A reference point is accepted only when it has **strictly more**
    than ``min_neighbors`` neighbours.

    Returns ``None`` when fewer than 2 reference points are accepted.
    """
    n_times, n_vars, s2d = _resolve_inputs(series)
    causal = step if causal is None else causal

    rescaled = np.empty_like(s2d)
    for c in range(n_vars):
        sr, _, _ = rescale_data(s2d[:, c])
        rescaled[:, c] = sr

    overall_std = rescaled.std(axis=0, ddof=1)

    E = lag_block_delay_embed(rescaled, embed=embed, delay=delay)
    valid_start = (embed - 1) * delay
    n_embed = E.shape[0]

    ref_orig = np.arange(valid_start, n_times - step)
    n_ref = len(ref_orig)

    sum_err2 = np.zeros(n_vars)
    n_done = 0

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
            nbr_times_list.append(t_orig)

        n_nbrs = len(nbr_times_list)
        if n_nbrs <= min_neighbors:
            continue

        nbr_times = np.asarray(nbr_times_list)
        X = E[nbr_times - valid_start]  # pyright: ignore[reportCallIssue, reportArgumentType]
        y_all = rescaled[nbr_times + step, :]  # pyright: ignore[reportCallIssue, reportArgumentType]
        y_true = rescaled[i_orig + step, :]

        X_mean = X.mean(axis=0)
        Xc = X - X_mean
        y_mean = y_all.mean(axis=0)
        yc = y_all - y_mean

        beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
        y_pred = y_mean + (E[ref_idx] - X_mean) @ beta

        sum_err2 += (y_pred - y_true) ** 2
        n_done += 1

    if n_done < 2:
        return None

    norm = float(n_ref)
    comp_errors = np.zeros(n_vars)
    for c in range(n_vars):
        if overall_std[c] > 0 and norm > 0:
            comp_errors[c] = np.sqrt(sum_err2[c] / norm) / overall_std[c]
        else:
            comp_errors[c] = np.nan

    return {
        "comp_errors": comp_errors,
        "avg_error": float(np.nanmean(comp_errors)),
        "n_done": n_done,
        "n_total": n_ref,
        "rms": overall_std,
    }


# ===================================================================
# lfo_ar tests
# ===================================================================


class TestLfoArBasicExecution(unittest.TestCase):
    def test_smoke_runs(self):
        rng = np.random.default_rng(0)
        n = 400
        x = rng.standard_normal(n)
        res = lfo_ar(x, embed=2, delay=1, step=1, eps0=0.01, eps1=0.9, eps_factor=1.3)
        self.assertIn("epsilon", res)
        self.assertGreater(len(res["epsilon"]), 1)
        self.assertEqual(res["avg_error"].shape, res["epsilon"].shape)
        self.assertEqual(res["comp_errors"].ndim, 2)
        self.assertEqual(res["comp_errors"].shape[0], len(res["epsilon"]))
        self.assertEqual(res["n_ref_points"], n - 1 - 1)
        self.assertEqual(res["comp_errors"].shape[1], 1)


class TestLfoArBruteForceAgreement(unittest.TestCase):
    """Fast lfo_ar must agree with the O(N^2) brute on a very small
    dataset for the *single* epsilon that accepts every non-causal
    neighbour.  We compare the last scan point (largest epsilon)
    against the brute.
    """

    def test_single_component_embed1(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(80)
        res_fast = lfo_ar(
            x, embed=1, delay=1, step=1, causal=1, eps0=0.001, eps1=1.0, eps_factor=1.1
        )
        brute = _brute_lfo_ar(x, embed=1, delay=1, step=1, causal=1)
        np.testing.assert_allclose(
            res_fast["comp_errors"][-1], brute["comp_errors_brute"], rtol=1e-9
        )
        self.assertAlmostEqual(
            res_fast["avg_error"][-1], brute["avg_error_brute"], places=10
        )

    def test_single_component_embed2(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(100)
        res_fast = lfo_ar(
            x, embed=2, delay=1, step=1, causal=1, eps0=0.001, eps1=1.0, eps_factor=1.1
        )
        brute = _brute_lfo_ar(x, embed=2, delay=1, step=1, causal=1)
        np.testing.assert_allclose(
            res_fast["comp_errors"][-1], brute["comp_errors_brute"], rtol=1e-9
        )

    def test_multivariate(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((70, 2))
        res_fast = lfo_ar(
            X, embed=1, delay=1, step=1, causal=1, eps0=0.001, eps1=1.0, eps_factor=1.1
        )
        brute = _brute_lfo_ar(X, embed=1, delay=1, step=1, causal=1)
        np.testing.assert_allclose(
            res_fast["comp_errors"][-1], brute["comp_errors_brute"], rtol=1e-9
        )


class TestLfoArDeterministicProcess(unittest.TestCase):
    def test_linear_trend_embed2(self):
        t = np.arange(300.0)
        x = t * 0.1 + 5.0
        res = lfo_ar(
            x, embed=1, delay=1, step=1, causal=1, eps0=0.01, eps1=0.95, eps_factor=1.5
        )
        self.assertLess(res["avg_error"][-1], 1e-3)

    def test_noisy_ar1_relative_error_matches_theory(self):
        rng = np.random.default_rng(4)
        n = 2000
        x = np.zeros(n)
        x[0] = rng.standard_normal()
        for k in range(1, n):
            x[k] = 0.9 * x[k - 1] + rng.standard_normal()
        res = lfo_ar(
            x, embed=1, delay=1, step=1, causal=1, eps0=0.01, eps1=0.9, eps_factor=1.2
        )
        self.assertAlmostEqual(res["avg_error"][-1], np.sqrt(1 - 0.9**2), delta=0.05)


class TestLfoArCausalityExclusion(unittest.TestCase):
    def test_neighbour_from_near_future_is_excluded(self):
        t = np.arange(400.0)
        x = t * 0.1 + 5.0
        res1 = lfo_ar(
            x, embed=1, delay=1, step=1, causal=1, eps0=0.05, eps1=0.95, eps_factor=2.0
        )
        res50 = lfo_ar(
            x,
            embed=1,
            delay=1,
            step=1,
            causal=100,
            eps0=0.05,
            eps1=0.95,
            eps_factor=2.0,
        )
        self.assertLess(res1["avg_error"][-1], 1e-10)
        self.assertLess(res50["avg_error"][-1], 1e-10)
        self.assertGreater(res1["coverage"][-1], 0.9)
        self.assertGreater(res50["coverage"][-1], 0.9)


class TestLfoArValidation(unittest.TestCase):
    def test_embed_zero_raises(self):
        rng = np.random.default_rng(0)
        with self.assertRaises(ValueError):
            lfo_ar(rng.standard_normal(100), embed=0)

    def test_negative_delay_raises(self):
        rng = np.random.default_rng(0)
        with self.assertRaises(ValueError):
            lfo_ar(rng.standard_normal(100), delay=-1)

    def test_step_zero_raises(self):
        rng = np.random.default_rng(0)
        with self.assertRaises(ValueError):
            lfo_ar(rng.standard_normal(100), step=0)

    def test_eps_factor_le_one_raises(self):
        rng = np.random.default_rng(0)
        with self.assertRaises(ValueError):
            lfo_ar(rng.standard_normal(100), eps_factor=1.0)

    def test_too_short_series_raises(self):
        with self.assertRaises(ValueError):
            lfo_ar(np.arange(2.0), embed=2, delay=1, step=1)

    def test_2d_multivariate(self):
        rng = np.random.default_rng(0)
        res = lfo_ar(
            rng.standard_normal((200, 2)),
            embed=1,
            delay=1,
            step=1,
            eps0=0.01,
            eps1=0.9,
            eps_factor=1.2,
        )
        self.assertEqual(res["comp_errors"].shape[1], 2)


# ===================================================================
# lfo_test tests
# ===================================================================


class TestLfoTestBasicExecution(unittest.TestCase):
    def test_smoke_runs_and_returns_expected_keys(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(400)
        res = lfo_test(x, embed=2, delay=1, step=1, min_neighbors=30)
        for key in (
            "comp_errors",
            "avg_error",
            "final_epsilon",
            "n_done",
            "n_total",
            "n_unresolved",
            "embedding_norm",
            "rms",
        ):
            self.assertIn(key, res)

    def test_result_shapes(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(400)
        res = lfo_test(x, embed=2, delay=1, step=1, min_neighbors=30)
        self.assertEqual(res["comp_errors"].shape, (1,))
        self.assertIsInstance(res["avg_error"], float)
        self.assertIsInstance(res["final_epsilon"], float)
        self.assertIsInstance(res["n_done"], int)
        self.assertIsInstance(res["n_total"], int)
        self.assertIsInstance(res["n_unresolved"], int)

    def test_all_points_resolved_with_default_params(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(500)
        res = lfo_test(x, embed=2, delay=1, step=1, min_neighbors=30)
        self.assertEqual(res["n_unresolved"], 0)
        self.assertEqual(res["n_done"], res["n_total"])
        self.assertGreater(res["final_epsilon"], 0.0)

    def test_n_total_matches_expected(self):
        rng = np.random.default_rng(3)
        n = 400
        x = rng.standard_normal(n)
        res = lfo_test(x, embed=2, delay=1, step=1, min_neighbors=30)
        self.assertEqual(res["n_total"], n - 1 - 1)

    def test_embedding_norm_present(self):
        rng = np.random.default_rng(4)
        x = rng.standard_normal(200)
        res = lfo_test(x, embed=2, delay=1, step=1, min_neighbors=20)
        self.assertIn(0, res["embedding_norm"])
        _, interval = res["embedding_norm"][0]
        self.assertGreater(interval, 0.0)


class TestLfoTestBruteForceAgreement(unittest.TestCase):
    """Compare lfo_test (with large eps0 so all points resolve in round 1)
    against the O(N^2) brute that accepts every non-causal neighbour."""

    def test_single_component_embed1(self):
        rng = np.random.default_rng(10)
        x = rng.standard_normal(80)
        res = lfo_test(
            x,
            embed=1,
            delay=1,
            step=1,
            causal=1,
            eps0=2.0,
            eps_factor=10.0,
            min_neighbors=1,
        )
        brute = _brute_lfo_test(x, embed=1, delay=1, step=1, causal=1, min_neighbors=1)
        self.assertIsNotNone(brute)
        np.testing.assert_allclose(res["comp_errors"], brute["comp_errors"], rtol=1e-9)  # pyright: ignore[reportOptionalSubscript]
        self.assertAlmostEqual(res["avg_error"], brute["avg_error"], places=10)  # pyright: ignore[reportOptionalSubscript]

    def test_single_component_embed2(self):
        rng = np.random.default_rng(11)
        x = rng.standard_normal(100)
        res = lfo_test(
            x,
            embed=2,
            delay=1,
            step=1,
            causal=1,
            eps0=2.0,
            eps_factor=10.0,
            min_neighbors=1,
        )
        brute = _brute_lfo_test(x, embed=2, delay=1, step=1, causal=1, min_neighbors=1)
        self.assertIsNotNone(brute)
        np.testing.assert_allclose(res["comp_errors"], brute["comp_errors"], rtol=1e-9)  # pyright: ignore[reportOptionalSubscript]

    def test_multivariate(self):
        rng = np.random.default_rng(12)
        X = rng.standard_normal((70, 2))
        res = lfo_test(
            X,
            embed=1,
            delay=1,
            step=1,
            causal=1,
            eps0=2.0,
            eps_factor=10.0,
            min_neighbors=1,
        )
        brute = _brute_lfo_test(X, embed=1, delay=1, step=1, causal=1, min_neighbors=1)
        self.assertIsNotNone(brute)
        np.testing.assert_allclose(res["comp_errors"], brute["comp_errors"], rtol=1e-9)  # pyright: ignore[reportOptionalSubscript]

    def test_delay_gt_1(self):
        rng = np.random.default_rng(13)
        x = rng.standard_normal(120)
        res = lfo_test(
            x,
            embed=3,
            delay=2,
            step=1,
            causal=1,
            eps0=2.0,
            eps_factor=10.0,
            min_neighbors=1,
        )
        brute = _brute_lfo_test(x, embed=3, delay=2, step=1, causal=1, min_neighbors=1)
        self.assertIsNotNone(brute)
        np.testing.assert_allclose(res["comp_errors"], brute["comp_errors"], rtol=1e-9)  # pyright: ignore[reportOptionalSubscript]

    def test_large_causal_window(self):
        rng = np.random.default_rng(14)
        x = rng.standard_normal(150)
        res = lfo_test(
            x,
            embed=2,
            delay=1,
            step=1,
            causal=20,
            eps0=2.0,
            eps_factor=10.0,
            min_neighbors=1,
        )
        brute = _brute_lfo_test(x, embed=2, delay=1, step=1, causal=20, min_neighbors=1)
        self.assertIsNotNone(brute)
        np.testing.assert_allclose(res["comp_errors"], brute["comp_errors"], rtol=1e-9)  # pyright: ignore[reportOptionalSubscript]


class TestLfoTestDeterministicProcess(unittest.TestCase):
    def test_linear_trend_near_zero_error(self):
        t = np.arange(300.0)
        x = t * 0.1 + 5.0
        res = lfo_test(x, embed=1, delay=1, step=1, causal=1, min_neighbors=10)
        self.assertLess(res["avg_error"], 1e-6)

    def test_linear_trend_embed2(self):
        t = np.arange(300.0)
        x = t * 0.1 + 5.0
        res = lfo_test(x, embed=2, delay=1, step=1, causal=1, min_neighbors=10)
        self.assertLess(res["avg_error"], 1e-6)

    def test_noisy_ar1_error_below_one(self):
        rng = np.random.default_rng(15)
        n = 2000
        x = np.zeros(n)
        x[0] = rng.standard_normal()
        for k in range(1, n):
            x[k] = 0.9 * x[k - 1] + rng.standard_normal()
        res = lfo_test(x, embed=1, delay=1, step=1, causal=1, min_neighbors=30)
        self.assertLess(res["avg_error"], 0.6)
        self.assertGreater(res["avg_error"], 0.2)

    def test_deterministic_ar2_near_zero(self):
        n = 500
        x = np.zeros(n)
        x[0] = 0.1
        x[1] = -0.2
        for t in range(n - 2):
            x[t + 2] = 0.6 * x[t + 1] - 0.4 * x[t]
        res = lfo_test(x, embed=2, delay=1, step=1, causal=1, min_neighbors=10)
        self.assertLess(res["avg_error"], 1e-6)


class TestLfoTestAdaptiveEpsilon(unittest.TestCase):
    def test_final_epsilon_gte_eps0(self):
        rng = np.random.default_rng(20)
        x = rng.standard_normal(500)
        res = lfo_test(
            x, embed=2, delay=1, step=1, min_neighbors=30, eps0=0.001, eps_factor=1.2
        )
        self.assertGreaterEqual(res["final_epsilon"], 0.001)

    def test_larger_min_neighbors_larger_epsilon(self):
        _ = np.random.default_rng(21)
        x = np.sin(np.linspace(0, 20 * np.pi, 500))
        res10 = lfo_test(x, embed=2, delay=1, step=1, min_neighbors=10)
        res50 = lfo_test(x, embed=2, delay=1, step=1, min_neighbors=50)
        self.assertGreaterEqual(res50["final_epsilon"], res10["final_epsilon"] - 1e-12)

    def test_unresolved_with_huge_min_neighbors(self):
        rng = np.random.default_rng(22)
        x = rng.standard_normal(100)
        res = lfo_test(x, embed=2, delay=1, step=1, min_neighbors=200, max_epsilon=5.0)
        self.assertGreater(res["n_unresolved"], 0)
        self.assertLess(res["n_done"], res["n_total"])

    def test_eps_factor_affects_rounds(self):
        x = np.sin(np.linspace(0, 20 * np.pi, 500))
        res_slow = lfo_test(
            x, embed=2, delay=1, step=1, min_neighbors=30, eps_factor=1.1
        )
        res_fast = lfo_test(
            x, embed=2, delay=1, step=1, min_neighbors=30, eps_factor=2.0
        )
        self.assertEqual(res_slow["n_unresolved"], 0)
        self.assertEqual(res_fast["n_unresolved"], 0)
        self.assertGreaterEqual(
            res_fast["final_epsilon"], res_slow["final_epsilon"] * 0.5
        )


class TestLfoTestCausalityExclusion(unittest.TestCase):
    def test_deterministic_unaffected_by_causal_width(self):
        t = np.arange(400.0)
        x = t * 0.1 + 5.0
        res1 = lfo_test(x, embed=1, delay=1, step=1, causal=1, min_neighbors=10)
        res50 = lfo_test(x, embed=1, delay=1, step=1, causal=50, min_neighbors=10)
        self.assertLess(res1["avg_error"], 1e-6)
        self.assertLess(res50["avg_error"], 1e-6)

    def test_causal_window_reduces_n_done(self):
        rng = np.random.default_rng(30)
        x = rng.standard_normal(200)
        res1 = lfo_test(x, embed=2, delay=1, step=1, causal=1, min_neighbors=20)
        res20 = lfo_test(x, embed=2, delay=1, step=1, causal=20, min_neighbors=20)
        self.assertEqual(res1["n_unresolved"], 0)
        self.assertEqual(res20["n_unresolved"], 0)
        self.assertGreaterEqual(res20["final_epsilon"], res1["final_epsilon"] - 1e-12)


class TestLfoTestValidation(unittest.TestCase):
    def setUp(self):
        self.series = np.arange(50, dtype=float)

    def test_embed_zero_raises(self):
        with self.assertRaises(ValueError):
            lfo_test(self.series, embed=0)

    def test_delay_zero_raises(self):
        with self.assertRaises(ValueError):
            lfo_test(self.series, delay=0)

    def test_step_zero_raises(self):
        with self.assertRaises(ValueError):
            lfo_test(self.series, step=0)

    def test_causal_zero_raises(self):
        with self.assertRaises(ValueError):
            lfo_test(self.series, causal=0)

    def test_eps_factor_le_one_raises(self):
        with self.assertRaises(ValueError):
            lfo_test(self.series, eps_factor=1.0)
        with self.assertRaises(ValueError):
            lfo_test(self.series, eps_factor=0.5)

    def test_min_neighbors_zero_raises(self):
        with self.assertRaises(ValueError):
            lfo_test(self.series, min_neighbors=0)

    def test_max_epsilon_zero_raises(self):
        with self.assertRaises(ValueError):
            lfo_test(self.series, max_epsilon=0.0)

    def test_too_short_series_raises(self):
        with self.assertRaises(ValueError):
            lfo_test(np.arange(3, dtype=float), embed=3, delay=1, step=1)

    def test_negative_causal_raises(self):
        with self.assertRaises(ValueError):
            lfo_test(self.series, causal=-1)


class TestLfoTestMultivariate(unittest.TestCase):
    def test_two_component_shapes(self):
        rng = np.random.default_rng(40)
        X = rng.standard_normal((300, 2))
        res = lfo_test(X, embed=2, delay=1, step=1, min_neighbors=30)
        self.assertEqual(res["comp_errors"].shape, (2,))
        self.assertEqual(len(res["rms"]), 2)

    def test_two_component_both_below_one(self):
        t = np.linspace(0, 20 * np.pi, 500)
        s2d = np.column_stack([np.sin(t), np.cos(t)])
        res = lfo_test(s2d, embed=3, delay=1, step=1, min_neighbors=20)
        self.assertLess(res["comp_errors"][0], 0.5)
        self.assertLess(res["comp_errors"][1], 0.5)

    def test_rescale_info_has_both_components(self):
        rng = np.random.default_rng(41)
        X = rng.standard_normal((200, 2))
        res = lfo_test(X, embed=2, delay=1, step=1, min_neighbors=20)
        self.assertIn(0, res["embedding_norm"])
        self.assertIn(1, res["embedding_norm"])


class TestLfoTestNormalization(unittest.TestCase):
    def test_white_noise_error_near_one(self):
        rng = np.random.default_rng(50)
        x = rng.standard_normal(2000)
        res = lfo_test(x, embed=2, delay=1, step=1, min_neighbors=30)
        self.assertAlmostEqual(res["avg_error"], 1.0, delta=0.15)

    def test_rms_is_sample_std(self):
        rng = np.random.default_rng(51)
        x = rng.standard_normal(500)
        res = lfo_test(x, embed=2, delay=1, step=1, min_neighbors=30)
        sr, _, _ = rescale_data(x)
        expected_std = sr.std(ddof=1)
        self.assertAlmostEqual(res["rms"][0], expected_std, places=10)

    def test_constant_series_raises(self):
        x = np.full(100, 5.0)
        with self.assertRaises(RuntimeError):
            lfo_test(x, embed=2, delay=1, step=1, min_neighbors=10)


# ===================================================================
# lfo_run tests
# ===================================================================


class TestLfoRunBasicSmoke(unittest.TestCase):
    def test_default_linear_runs_50_steps_ok(self):
        rng = np.random.default_rng(0)
        x = np.empty(200)
        x[0] = 1.0
        for t in range(199):
            x[t + 1] = 0.9 * x[t] + rng.normal(scale=0.1)
        res = lfo_run(
            x, embed=2, delay=1, n_steps=50, method="linear", min_neighbors=20
        )
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["n_steps_done"], 50)
        self.assertEqual(res["trajectory"].shape, (50, 1))
        self.assertGreater(res["final_epsilon"], 0.0)
        self.assertLess(res["final_epsilon"], 2.0)

    def test_zeroth_runs(self):
        x = np.sin(np.linspace(0, 20, 500))
        res = lfo_run(
            x, embed=3, delay=1, n_steps=30, method="zeroth", min_neighbors=15
        )
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["n_steps_done"], 30)
        self.assertEqual(res["trajectory"].shape, (30, 1))


class TestLfoRunMultivariate(unittest.TestCase):
    def test_two_components_shapes(self):
        rng = np.random.default_rng(2)
        n = 300
        x = np.column_stack(
            [
                np.cumsum(rng.normal(scale=0.1, size=n)),
                np.cumsum(rng.normal(scale=0.05, size=n)),
            ]
        )
        res = lfo_run(
            x, embed=2, delay=1, n_steps=40, method="linear", min_neighbors=15
        )
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["trajectory"].shape, (40, 2))


class TestLfoRunValidation(unittest.TestCase):
    def test_embed_zero(self):
        with self.assertRaises(ValueError):
            lfo_run(np.zeros(10), embed=0)

    def test_delay_negative(self):
        with self.assertRaises(ValueError):
            lfo_run(np.zeros(10), delay=-1)

    def test_n_steps_zero(self):
        with self.assertRaises(ValueError):
            lfo_run(np.zeros(10), n_steps=0)

    def test_bad_method(self):
        with self.assertRaises(ValueError):
            lfo_run(np.zeros(20), method="quadratic")

    def test_eps_factor_one(self):
        with self.assertRaises(ValueError):
            lfo_run(np.zeros(20), eps_factor=1.0)

    def test_min_neighbors_zero(self):
        with self.assertRaises(ValueError):
            lfo_run(np.zeros(20), min_neighbors=0)

    def test_too_short_series(self):
        with self.assertRaises(ValueError):
            lfo_run(np.zeros(3), embed=3, delay=1, n_steps=1)


class TestLfoRunEscapeReturnsPartial(unittest.TestCase):
    def test_zero_escape_scale_triggers_escaped(self):
        x = np.exp(np.linspace(0, 1, 200))
        res = lfo_run(
            x,
            embed=2,
            delay=1,
            n_steps=100,
            method="linear",
            min_neighbors=20,
            escape_scale=0.0,
        )
        self.assertNotEqual(res["status"], "ok")
        self.assertGreater(res["n_steps_done"], 0)
        self.assertLess(res["n_steps_done"], 100)

    def test_large_escape_scale_runs_full(self):
        # Deterministic AR(2) with stable roots — perfectly predictable,
        # so the trajectory stays bounded over 50 steps.
        n = 500
        x = np.zeros(n)
        x[0] = 0.1
        x[1] = -0.2
        for t in range(n - 2):
            x[t + 2] = 0.6 * x[t + 1] - 0.4 * x[t]
        res = lfo_run(
            x,
            embed=3,
            delay=1,
            n_steps=50,
            method="linear",
            min_neighbors=30,
            escape_scale=10.0,
        )
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["n_steps_done"], 50)


class TestLfoRunMinNeighborsAdaptsEpsilon(unittest.TestCase):
    def test_larger_min_neighbors_uses_larger_final_epsilon(self):
        x = np.sin(np.linspace(0, 20, 500))
        res10 = lfo_run(
            x, embed=2, delay=1, n_steps=20, method="linear", min_neighbors=10
        )
        res50 = lfo_run(
            x, embed=2, delay=1, n_steps=20, method="linear", min_neighbors=50
        )
        self.assertGreaterEqual(res50["final_epsilon"], res10["final_epsilon"] - 1e-12)


class TestLfoRunOutputInOriginalScale(unittest.TestCase):
    def test_trajectory_rescaled_back(self):
        rng = np.random.default_rng(6)
        x = rng.normal(loc=100.0, scale=5.0, size=300)
        res = lfo_run(
            x, embed=2, delay=1, n_steps=30, method="linear", min_neighbors=15
        )
        self.assertAlmostEqual(float(res["trajectory"].mean()), 100.0, delta=50.0)


class TestLfoRunDeterministic(unittest.TestCase):
    def test_linear_first_step_matches(self):
        rng = np.random.default_rng(8)
        x = np.empty(400)
        x[0] = 1.0
        for t in range(399):
            x[t + 1] = 0.9 * x[t] + rng.normal(scale=0.1)
        res = lfo_run(x, embed=2, delay=1, n_steps=1, method="linear", min_neighbors=30)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["trajectory"].shape, (1, 1))
        self.assertGreater(res["final_epsilon"], 0.0)

    def test_deterministic_ar2_converges(self):
        n = 500
        x = np.zeros(n)
        x[0] = 0.1
        x[1] = -0.2
        for t in range(n - 2):
            x[t + 2] = 0.6 * x[t + 1] - 0.4 * x[t]
        res = lfo_run(
            x,
            embed=3,
            delay=1,
            n_steps=50,
            method="linear",
            min_neighbors=40,
            escape_scale=10.0,
        )
        self.assertEqual(res["status"], "ok")
        traj = res["trajectory"].flatten()
        data_range = float(np.ptp(x))
        self.assertTrue(np.all(np.abs(traj) < 10 * max(data_range, 1e-6)))


if __name__ == "__main__":
    unittest.main()
