"""Tests for the TISEAN ``recurr`` rewrite in nolitisea.stationarity.recurrence."""

import unittest

import numpy as np

from nolitisea.core.embed import lag_block_delay_embed
from nolitisea.stationarity.recurrence import recurr


def _recurr_brute(series, embed, delay, eps=None):
    """Brute-force recurrence pairs matching C ``recurr.c`` logic.

    Uses a strict ``< eps`` comparison (as in C) and a pure-NumPy
    pairwise Chebyshev distance, independent of the cKDTree code
    path.
    """
    arr = np.asarray(series, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    _, n_vars = arr.shape

    # Rescale each component to [0, 1]
    maxmax = 0.0
    scaled = arr.copy()
    for c in range(n_vars):
        comp = arr[:, c]
        rng = np.ptp(comp)
        if rng < 1e-30:
            raise RuntimeError(f"component {c} is constant (zero range)")
        scaled[:, c] = (comp - comp.min()) / rng
        maxmax = max(maxmax, rng)

    eps_rescaled = 1.0e-3 if eps is None else abs(float(eps)) / maxmax

    # Build embedding (interleaved, matching C coordinate order)
    E = lag_block_delay_embed(scaled, embed, delay)
    n_points = E.shape[0]

    # Pairwise Chebyshev distance via broadcasting (strict < eps)
    diff = np.abs(E[:, None, :] - E[None, :, :])
    D = diff.max(axis=2)
    mask = np.triu(D < eps_rescaled, k=1)
    i_idx, j_idx = np.where(mask)
    pairs = np.column_stack([i_idx, j_idx]) if i_idx.size > 0 \
        else np.empty((0, 2), dtype=np.intp)

    return {"pairs": pairs, "eps": eps_rescaled, "n_points": n_points}


def _sort_pairs(pairs):
    """Sort pairs lexicographically for order-independent comparison."""
    if pairs.shape[0] == 0:
        return pairs
    order = np.lexsort(pairs.T[::-1])
    return pairs[order]


class TestRecurrBruteForce(unittest.TestCase):
    def test_matches_brute_force_1d(self):
        rng = np.random.default_rng(11)
        y = rng.standard_normal(200)
        res = recurr(y, embed=3, delay=2, eps=1.5)
        brute = _recurr_brute(y, embed=3, delay=2, eps=1.5)
        np.testing.assert_array_equal(
            _sort_pairs(res["pairs"]), _sort_pairs(brute["pairs"])
        )

    def test_matches_brute_force_2d(self):
        rng = np.random.default_rng(23)
        y = rng.standard_normal((200, 2))
        res = recurr(y, embed=2, delay=1, eps=2.0)
        brute = _recurr_brute(y, embed=2, delay=1, eps=2.0)
        np.testing.assert_array_equal(
            _sort_pairs(res["pairs"]), _sort_pairs(brute["pairs"])
        )

    def test_matches_brute_force_default_eps(self):
        rng = np.random.default_rng(5)
        y = rng.standard_normal(150)
        res = recurr(y, embed=2, delay=1)
        brute = _recurr_brute(y, embed=2, delay=1)
        np.testing.assert_array_equal(
            _sort_pairs(res["pairs"]), _sort_pairs(brute["pairs"])
        )


class TestRecurrOutputs(unittest.TestCase):
    def test_output_structure(self):
        rng = np.random.default_rng(1)
        y = rng.standard_normal(100)
        res = recurr(y, embed=2, delay=1, eps=2.0)
        self.assertEqual(set(res), {"pairs", "eps", "n_points"})
        self.assertEqual(res["pairs"].shape[1], 2)
        self.assertEqual(res["n_points"], 99)  # 100 - (2-1)*1

    def test_pairs_upper_triangle(self):
        rng = np.random.default_rng(2)
        y = rng.standard_normal(100)
        res = recurr(y, embed=2, delay=1, eps=3.0)
        pairs = res["pairs"]
        if pairs.shape[0] > 0:
            self.assertTrue(np.all(pairs[:, 0] < pairs[:, 1]))

    def test_eps_rescaling(self):
        rng = np.random.default_rng(3)
        y = 10.0 * rng.standard_normal(100)
        res = recurr(y, embed=2, delay=1, eps=1.0)
        # User eps=1.0, data range ~60-70, eps_rescaled should be small
        self.assertLess(res["eps"], 0.05)

    def test_default_eps_value(self):
        rng = np.random.default_rng(3)
        y = rng.standard_normal(100)
        res = recurr(y, embed=2, delay=1)
        # Default eps = 1e-3 in rescaled units
        self.assertAlmostEqual(res["eps"], 1.0e-3, places=10)

    def test_fraction_subsample(self):
        rng = np.random.default_rng(4)
        y = rng.standard_normal(200)
        full = recurr(y, embed=2, delay=1, eps=3.0, fraction=1.0)
        sub = recurr(y, embed=2, delay=1, eps=3.0, fraction=0.5, seed=42)
        self.assertLess(sub["pairs"].shape[0], full["pairs"].shape[0])
        self.assertGreater(sub["pairs"].shape[0], 0)

    def test_fraction_reproducible(self):
        rng = np.random.default_rng(4)
        y = rng.standard_normal(200)
        a = recurr(y, embed=2, delay=1, eps=3.0, fraction=0.5, seed=99)
        b = recurr(y, embed=2, delay=1, eps=3.0, fraction=0.5, seed=99)
        np.testing.assert_array_equal(a["pairs"], b["pairs"])

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            recurr([1, 2, 3], embed=0, delay=1)
        with self.assertRaises(ValueError):
            recurr([1, 2, 3], embed=2, delay=0)
        with self.assertRaises(ValueError):
            recurr([1, 2, 3], embed=2, delay=1, fraction=0.0)
        with self.assertRaises(ValueError):
            recurr([1, 2, 3], embed=2, delay=1, fraction=1.5)

    def test_constant_series_raises(self):
        with self.assertRaises(RuntimeError):
            recurr(np.ones(50), embed=2, delay=1)

    def test_empty_pairs_small_eps(self):
        # Very small eps -> no pairs
        rng = np.random.default_rng(6)
        y = rng.standard_normal(50)
        res = recurr(y, embed=2, delay=1, eps=0.001)
        self.assertEqual(res["pairs"].shape, (0, 2))


if __name__ == "__main__":
    unittest.main()
