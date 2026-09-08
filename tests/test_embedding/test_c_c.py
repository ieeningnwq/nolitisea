"""Tests for nolitisea.embedding.c_c (C-C method).

The brute-force helper transcribes ``_cc_single_t`` with an explicit
O(N^2) Chebyshev pair count (vectorized distance matrix), so the
cKDTree-based production code can be validated on continuous data with
exact agreement.
"""

import unittest

import numpy as np
from scipy.signal import argrelextrema
from scipy.spatial import cKDTree

from nolitisea.embedding.c_c import (
    _cc_single_t,
    _integral_from_tree,
    cc_method,
)
from nolitisea.generate.henon import henon

# ---------------------------------------------------------------------------
# Brute-force helpers
# ---------------------------------------------------------------------------


def _brute_cc_single_t(ts, t, m_values, r_values):
    """O(N^2) transcription of _cc_single_t.

    Counts unordered i<j pairs with Chebyshev distance <= r by building
    the full distance matrix, matching cKDTree.count_neighbors semantics
    exactly (self-pairs excluded, each unordered pair counted once).
    """
    ts = np.asarray(ts, dtype=np.float64)
    S_mr = np.zeros((len(m_values), len(r_values)))
    for s in range(t):
        sub = ts[s::t]
        n = sub.size
        if n <= max(m_values):
            continue

        total_pairs_1d = n * (n - 1) // 2
        # 1-D pairwise Chebyshev (= |diff|) distances
        d1 = np.abs(sub[:, None] - sub[None, :])
        c1_vals = []
        for r in r_values:
            raw = int(np.count_nonzero(d1 <= r))
            count = (raw - n) // 2
            c1_vals.append(
                count / total_pairs_1d if total_pairs_1d > 0 else 0.0
            )

        for a, m in enumerate(m_values):
            n_emb = n - (m - 1)
            total_pairs_m = n_emb * (n_emb - 1) // 2
            emb = np.empty((n_emb, m))
            for k in range(m):
                emb[:, k] = sub[k : k + n_emb]
            d_md = np.max(
                np.abs(emb[:, None, :] - emb[None, :, :]), axis=2
            )
            for b, r in enumerate(r_values):
                raw = int(np.count_nonzero(d_md <= r))
                count = (raw - n_emb) // 2
                C_m = count / total_pairs_m if total_pairs_m > 0 else 0.0
                S_mr[a, b] += C_m - c1_vals[b] ** m

    S_mr /= max(t, 1)
    s_mean = S_mr.mean()
    delta_s = (S_mr.max(axis=1) - S_mr.min(axis=1)).mean()
    s_cor = delta_s + abs(s_mean)
    return s_mean, delta_s, s_cor


# ---------------------------------------------------------------------------
# _integral_from_tree
# ---------------------------------------------------------------------------


class TestIntegralFromTree(unittest.TestCase):
    """Tests for the _integral_from_tree helper."""

    def test_single_point_returns_zero(self):
        # valid_total = 0 -> return 0.0 (avoid division by zero).
        tree = cKDTree(np.array([[1.0]]))
        self.assertEqual(_integral_from_tree(tree, 1, 0.5), 0.0)

    def test_two_points_close(self):
        tree = cKDTree(np.array([[0.0], [1.0]]))
        # eps=2: pair (0,1) has distance 1 <= 2 -> C = 1/1 = 1.0.
        self.assertEqual(_integral_from_tree(tree, 2, 2.0), 1.0)

    def test_two_points_far(self):
        tree = cKDTree(np.array([[0.0], [1.0]]))
        # eps=0.5: only self-pairs -> C = 0/1 = 0.0.
        self.assertEqual(_integral_from_tree(tree, 2, 0.5), 0.0)

    def test_three_points_hand_computed(self):
        # Points 0, 1, 3; eps=1.0 (Chebyshev = |diff|):
        # pairs i<j: (0,1) d=1 <= 1 yes, (0,2) d=3 no, (1,2) d=2 no -> C = 1/3.
        tree = cKDTree(np.array([[0.0], [1.0], [3.0]]))
        self.assertAlmostEqual(
            _integral_from_tree(tree, 3, 1.0), 1.0 / 3.0, places=15
        )

    def test_large_eps_all_pairs(self):
        # eps large enough -> every unordered pair counts -> C = 1.0.
        pts = np.array([[0.0], [1.0], [2.0], [5.0]])
        tree = cKDTree(pts)
        self.assertEqual(_integral_from_tree(tree, 4, 100.0), 1.0)

    def test_matches_brute_force_random(self):
        rng = np.random.default_rng(0)
        pts = rng.standard_normal(50).reshape(-1, 1)
        tree = cKDTree(pts)
        n = pts.size
        d1 = np.abs(pts[:, None] - pts[None, :])
        for eps in [0.1, 0.5, 1.0, 2.0]:
            with self.subTest(eps=eps):
                raw = int(np.count_nonzero(d1 <= eps))
                count = (raw - n) // 2
                expected = count / (n * (n - 1) // 2)
                self.assertAlmostEqual(
                    _integral_from_tree(tree, n, eps),
                    expected,
                    places=15,
                )


# ---------------------------------------------------------------------------
# _cc_single_t - brute-force agreement
# ---------------------------------------------------------------------------


class TestCcSingleTBruteForce(unittest.TestCase):
    """Compare _cc_single_t against an O(N^2) brute-force transcription."""

    @classmethod
    def setUpClass(cls):
        cls.x, _ = henon(n=300)
        cls.std = float(np.std(cls.x))
        cls.m_values = [2, 3, 4, 5]
        cls.r_values = [0.5 * cls.std, 1.0 * cls.std,
                        1.5 * cls.std, 2.0 * cls.std]

    def test_brute_force_match_multiple_t(self):
        for t in [1, 2, 3, 5]:
            with self.subTest(t=t):
                got = _cc_single_t(self.x, t, self.m_values, self.r_values)
                ref = _brute_cc_single_t(
                    self.x, t, self.m_values, self.r_values
                )
                for g, r in zip(got, ref):
                    self.assertAlmostEqual(float(g), float(r), places=12)

    def test_brute_force_match_lorenz(self):
        from nolitisea.generate.lorenz import lorenz

        _, data = lorenz(length=400)
        x = data[:, 0]
        std = float(np.std(x))
        m_values = [2, 3, 4, 5]
        r_values = [0.5 * std, 1.0 * std, 1.5 * std, 2.0 * std]
        for t in [1, 2, 4]:
            with self.subTest(t=t):
                got = _cc_single_t(x, t, m_values, r_values)
                ref = _brute_cc_single_t(x, t, m_values, r_values)
                for g, r in zip(got, ref):
                    self.assertAlmostEqual(float(g), float(r), places=12)

    def test_short_sub_series_skipped(self):
        # With t > len(ts), every sub-series has <= 1 point and is skipped;
        # S_mr stays zeros -> all statistics are 0.0.
        x = np.arange(10.0)
        s_mean, delta_s, s_cor = _cc_single_t(x, 20, [2, 3, 4, 5], [0.5, 1.0])
        self.assertEqual(s_mean, 0.0)
        self.assertEqual(delta_s, 0.0)
        self.assertEqual(s_cor, 0.0)


# ---------------------------------------------------------------------------
# cc_method - output structure and properties
# ---------------------------------------------------------------------------


class TestCcMethodOutput(unittest.TestCase):
    """Tests for the cc_method public API (output structure and values)."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        cls.x, _ = henon(n=2000)

    def test_output_structure(self):
        res = cc_method(self.x, max_t=15)
        self.assertEqual(
            set(res), {"tau", "t_w", "m", "S_mean", "delta_S", "S_cor"}
        )

    def test_statistic_array_lengths(self):
        res = cc_method(self.x, max_t=15)
        for key in ("S_mean", "delta_S", "S_cor"):
            self.assertEqual(len(res[key]), 15)

    def test_tau_and_t_w_in_range(self):
        res = cc_method(self.x, max_t=15)
        self.assertGreaterEqual(res["tau"], 1)
        self.assertLessEqual(res["tau"], 15)
        self.assertGreaterEqual(res["t_w"], 1)
        self.assertLessEqual(res["t_w"], 15)

    def test_m_at_least_two(self):
        res = cc_method(self.x, max_t=15)
        self.assertGreaterEqual(res["m"], 2)

    def test_m_consistency_with_tau_and_t_w(self):
        # m = max(round(t_w / tau) + 1, 2).
        res = cc_method(self.x, max_t=15)
        expected_m = max(
            int(np.round(res["t_w"] / res["tau"])) + 1, 2
        )
        self.assertEqual(res["m"], expected_m)

    def test_t_w_is_global_min_of_s_cor(self):
        res = cc_method(self.x, max_t=15)
        s_cor = np.asarray(res["S_cor"])
        # t_w is 1-indexed (t_range = 1..max_t).
        self.assertEqual(res["t_w"], int(np.argmin(s_cor)) + 1)

    def test_tau_is_first_local_min_of_delta_s(self):
        res = cc_method(self.x, max_t=15)
        delta_s = np.asarray(res["delta_S"])
        local_minima = argrelextrema(delta_s, np.less)[0]
        if len(local_minima) > 0:
            # tau is 1-indexed (t_range = 1..max_t).
            self.assertEqual(res["tau"], int(local_minima[0]) + 1)

    def test_statistics_finite(self):
        res = cc_method(self.x, max_t=15)
        for key in ("S_mean", "delta_S", "S_cor"):
            arr = np.asarray(res[key])
            self.assertTrue(np.all(np.isfinite(arr)))

    def test_deterministic_output(self):
        r1 = cc_method(self.x, max_t=10)
        r2 = cc_method(self.x, max_t=10)
        for key in ("tau", "t_w", "m"):
            self.assertEqual(r1[key], r2[key])
        for key in ("S_mean", "delta_S", "S_cor"):
            np.testing.assert_array_equal(r1[key], r2[key])


# ---------------------------------------------------------------------------
# cc_method - input validation
# ---------------------------------------------------------------------------


class TestCcMethodValidation(unittest.TestCase):
    """Tests for input validation in cc_method."""

    def test_zero_variance_raises(self):
        with self.assertRaises(ValueError):
            cc_method(np.ones(100), max_t=5)

    def test_invalid_backend_raises(self):
        with self.assertRaises(ValueError):
            cc_method(np.arange(100.0), max_t=5, backend="invalid")


# ---------------------------------------------------------------------------
# cc_method - integration on dynamical systems
# ---------------------------------------------------------------------------


class TestCcMethodLorenz(unittest.TestCase):
    """Integration test on the Lorenz x-component."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        from nolitisea.generate.lorenz import lorenz

        _, data = lorenz(length=2000)
        cls.x = data[:, 0]

    def test_returns_reasonable_tau(self):
        # Lorenz x (dt=0.03): the C-C delay is typically in 5-25.
        res = cc_method(self.x, max_t=30)
        self.assertGreaterEqual(res["tau"], 2)
        self.assertLessEqual(res["tau"], 30)
        self.assertGreaterEqual(res["m"], 2)


class TestCcMethodWhiteNoise(unittest.TestCase):
    """White noise has no structure; cc_method should still run cleanly."""

    def test_runs_without_crash(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(1000)
        res = cc_method(x, max_t=10)
        self.assertGreaterEqual(res["tau"], 1)
        self.assertGreaterEqual(res["m"], 2)
        arr = np.asarray(res["S_cor"])
        self.assertTrue(np.all(np.isfinite(arr)))


# ---------------------------------------------------------------------------
# Parallel bitwise consistency
# ---------------------------------------------------------------------------


class TestCcMethodParallel(unittest.TestCase):
    """Serial vs parallel results must be bitwise identical."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        cls.x, _ = henon(n=1000)
        cls.serial = cc_method(cls.x, max_t=10)

    def test_thread_backend_bitwise(self):
        par = cc_method(self.x, max_t=10, n_jobs=-1, backend="thread")
        self.assertEqual(par["tau"], self.serial["tau"])
        self.assertEqual(par["t_w"], self.serial["t_w"])
        self.assertEqual(par["m"], self.serial["m"])
        for key in ("S_mean", "delta_S", "S_cor"):
            np.testing.assert_array_equal(
                np.asarray(par[key]), np.asarray(self.serial[key])
            )

    def test_process_backend_bitwise(self):
        par = cc_method(self.x, max_t=10, n_jobs=2, backend="process")
        self.assertEqual(par["tau"], self.serial["tau"])
        self.assertEqual(par["t_w"], self.serial["t_w"])
        self.assertEqual(par["m"], self.serial["m"])
        for key in ("S_mean", "delta_S", "S_cor"):
            np.testing.assert_array_equal(
                np.asarray(par[key]), np.asarray(self.serial[key])
            )


if __name__ == "__main__":
    unittest.main()
