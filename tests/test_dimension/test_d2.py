"""Tests for nolitisea.dimension.d2."""

import unittest

import numpy as np

from nolitisea.core.embed import lag_block_delay_embed
from nolitisea.dimension.d2 import d2


def _embed_brute(data, embed, delay, m):
    """Prefix order-``m`` embedding in the C interleaved column order."""
    return lag_block_delay_embed(data, embed, delay)[:, :m]


def _brute_c2(data, embed, delay, theiler, m, eps_arr):
    """Brute-force correlation sums for one embedding order."""
    E = _embed_brute(data, embed, delay, m)
    n = E.shape[0]
    norm = n * (n - 1) // 2
    for k in range(1, theiler + 1):
        norm -= max(n - k, 0)
    dists = []
    for i in range(n):
        lo = min(i + theiler + 1, n)
        if lo >= n:
            continue
        dists.append(np.max(np.abs(E[lo:] - E[i]), axis=1))
    dists = np.concatenate(dists) if dists else np.empty(0)
    return np.array([np.sum(dists <= eps) for eps in eps_arr]) / norm


class TestD2BruteForce(unittest.TestCase):
    def test_scalar_matches_brute_force(self):
        rng = np.random.default_rng(123)
        s = np.cumsum(rng.standard_normal(220))
        embed, delay, theiler = 4, 2, 3
        eps_arr = np.array([4.0, 2.0, 1.0, 0.5, 0.25, 0.1])

        res = d2(s, embed=embed, delay=delay, theiler=theiler, eps=eps_arr)
        self.assertEqual(res["c2"].shape, (embed, eps_arr.size))
        for m in range(1, embed + 1):
            brute = _brute_c2(s, embed, delay, theiler, m, eps_arr)
            np.testing.assert_allclose(
                res["c2"][m - 1], brute, rtol=1e-12, atol=1e-15
            )

    def test_multivariate_interleaved_matches_brute_force(self):
        rng = np.random.default_rng(7)
        data = np.column_stack([
            rng.standard_normal(160),
            np.cumsum(rng.standard_normal(160)),
        ])
        embed, delay, theiler = 3, 2, 1
        eps_arr = np.array([3.0, 1.5, 0.7, 0.3])

        res = d2(data, embed=embed, delay=delay, theiler=theiler, eps=eps_arr)
        m_full = data.shape[1] * embed
        self.assertEqual(res["c2"].shape, (m_full, eps_arr.size))
        for m in range(1, m_full + 1):
            brute = _brute_c2(data, embed, delay, theiler, m, eps_arr)
            np.testing.assert_allclose(
                res["c2"][m - 1], brute, rtol=1e-12, atol=1e-15
            )

    def test_theiler_norm_accounting(self):
        rng = np.random.default_rng(11)
        s = rng.standard_normal(100)
        embed, delay, theiler = 2, 1, 3
        n = 100 - (embed - 1) * delay
        expected = n * (n - 1) // 2
        for k in range(1, theiler + 1):
            expected -= max(n - k, 0)

        res = d2(s, embed=embed, delay=delay, theiler=theiler,
                 eps=[1.0, 0.5])
        self.assertEqual(res["norm"], expected)

    def test_c2_monotone_in_eps(self):
        rng = np.random.default_rng(5)
        s = rng.standard_normal(300)
        res = d2(s, embed=3, delay=1, theiler=2)
        # eps descends, so every c2 row must be non-increasing.
        self.assertTrue(np.all(np.diff(res["c2"], axis=1) <= 1e-12))


class TestD2Outputs(unittest.TestCase):
    def test_h2_and_slope_relations(self):
        rng = np.random.default_rng(21)
        s = rng.standard_normal(400)
        res = d2(s, embed=3, delay=2, theiler=1, eps=[2.0, 1.0, 0.5, 0.25])
        c2 = res["c2"]
        eps = res["eps"]

        # h2 row 0 = -log(c2 row 0); higher rows = log(c2[m-1]/c2[m]).
        pos = c2[0] > 0
        np.testing.assert_allclose(res["h2"][0, pos], -np.log(c2[0, pos]))
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = np.log(c2[:-1] / c2[1:])
        pos = (c2[:-1] > 0) & (c2[1:] > 0)
        np.testing.assert_allclose(res["h2"][1:][pos], expected[pos])

        # Slopes = log-ratio of c2 divided by log-ratio of eps.
        expected_slope = np.log(c2[:, :-1] / c2[:, 1:]) / np.log(
            eps[:-1] / eps[1:]
        )
        pos = (c2[:, :-1] > 0) & (c2[:, 1:] > 0)
        np.testing.assert_allclose(res["d2"][pos], expected_slope[pos])

    def test_limit_cycle_dimension_is_one(self):
        t = np.arange(4000)
        s = np.sin(2.0 * np.pi * t / 50.0)
        res = d2(s, embed=5, delay=1, theiler=5)
        eps = res["eps"]
        c2_last = res["c2"][-1]
        slopes = res["d2"][-1]
        # Scaling regime of a smooth 1-D curve in embedding space.
        mask = (eps[:-1] > 0.15) & (eps[:-1] < 1.0)
        mask &= (c2_last[:-1] > 0) & (c2_last[1:] > 0)
        self.assertGreater(np.count_nonzero(mask), 10)
        median_slope = np.median(slopes[mask])
        self.assertGreater(median_slope, 0.75)
        self.assertLess(median_slope, 1.3)

    def test_default_ladder_bounds(self):
        rng = np.random.default_rng(3)
        s = rng.standard_normal(500)
        res = d2(s, embed=2, delay=1, howoften=50)
        self.assertEqual(res["eps"].size, 50)
        # eps_max defaults to the data range, eps_min to its 1/1000.
        self.assertAlmostEqual(res["eps"][0], s.max() - s.min())
        self.assertAlmostEqual(
            res["eps"][-1], (s.max() - s.min()) / 1000.0, places=12
        )

class TestD2Errors(unittest.TestCase):
    def test_series_too_short(self):
        with self.assertRaises(ValueError):
            d2(np.zeros(5), embed=10, delay=1)

    def test_three_dimensional_input(self):
        with self.assertRaises(ValueError):
            d2(np.zeros((2, 10, 3)), embed=2)

    def test_invalid_parameters(self):
        s = np.random.default_rng(0).standard_normal(100)
        with self.assertRaises(ValueError):
            d2(s, embed=0)
        with self.assertRaises(ValueError):
            d2(s, delay=0)
        with self.assertRaises(ValueError):
            d2(s, theiler=-1)
        with self.assertRaises(ValueError):
            d2(s, howoften=1)
        with self.assertRaises(ValueError):
            d2(s, eps=[1.0])
        with self.assertRaises(ValueError):
            d2(s, eps=[1.0, 0.0])

    def test_constant_series_default_eps(self):
        with self.assertRaises(ValueError):
            d2(np.ones(50), embed=2)

    def test_theiler_excludes_all_pairs(self):
        with self.assertRaises(ValueError):
            d2(np.random.default_rng(1).standard_normal(50),
               embed=2, theiler=60)


if __name__ == "__main__":
    unittest.main()
