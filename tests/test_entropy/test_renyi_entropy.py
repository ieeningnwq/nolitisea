"""Tests for the nolitisea.entropy.renyi_entropy."""

import unittest

import numpy as np

from nolitisea.entropy.renyi_entropy import (
    _epsilon_ladder,
    renyi_entropy,
)


def _rescale(data):
    """Per-component min-max rescale to [0, 1] (independent path)."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    scaled = np.empty_like(data)
    for c in range(data.shape[1]):
        lo = data[:, c].min()
        scaled[:, c] = (data[:, c] - lo) / (data[:, c].max() - lo)
    return scaled


def _brute_entropies(series, embed, delay, q, epsi):
    """Brute-force order-``q`` entropies of every prefix partition.

    Independent implementation of the documented rule: forward
    interleaved embedding (``wd = e * n_vars + c`` holds component
    ``c`` at time ``t + e * delay``), box index ``int(x * epsi)``
    clipped to ``epsi - 1`` (documented guard for
    values rescaled exactly to ``1.0``), joint box occupation counted
    with a dict and normalized by the total number of points.
    """
    scaled = _rescale(series)
    n_times, n_vars = scaled.shape
    n_points = n_times - (embed - 1) * delay
    m_full = n_vars * embed
    h = np.empty(m_full)
    for wd in range(m_full):
        boxes = {}
        for t in range(n_points):
            key = []
            for w in range(wd + 1):
                e, c = divmod(w, n_vars)
                b = int(scaled[t + e * delay, c] * epsi)
                if b >= epsi:
                    b = epsi - 1
                key.append(b)
            key = tuple(key)
            boxes[key] = boxes.get(key, 0) + 1
        p = np.array(list(boxes.values()), dtype=np.float64) / n_points
        if q == 1.0:
            h[wd] = -np.sum(p * np.log(p))
        else:
            h[wd] = np.log(np.sum(p ** q)) / (1.0 - q)
    return h


def _ref_ladder(eps_min, eps_max, count):
    """Reference implementation of the epsilon ladder."""
    factor = (eps_max / eps_min) ** (1.0 / (count - 1)) if count > 1 else 1.0
    heps = eps_max * factor
    old = 0
    deps, epsis = [], []
    for _ in range(count):
        while True:
            heps /= factor
            test = int(1.0 / heps)
            if test > old:
                break
        old = test
        deps.append(heps)
        epsis.append(test)
    return np.array(deps), np.array(epsis, dtype=np.int64)


class TestRenyiEntropy(unittest.TestCase):
    def test_distinct_values_all_orders_equal_log_n(self):
        # 50 distinct values, tiny box -> every point in its own box,
        # uniform occupation -> every Renyi order equals log(50).
        s = np.arange(1.0, 51.0)
        for q in (0.5, 1.0, 2.0, 3.7):
            res = renyi_entropy(
                s, embed=1, delay=1, q=q,
                eps_min=5e-5, eps_max=1e-4, eps_count=1,
            )
            self.assertAlmostEqual(res["hq"][0, 0], np.log(50.0), places=10)

    def test_scalar_matches_brute_force(self):
        rng = np.random.default_rng(123)
        s = np.cumsum(rng.standard_normal(150))
        for eps in (2.0, 0.7, 0.25):
            for q in (1.0, 2.0):
                # eps_count=1 makes the ladder report eps_max alone.
                res = renyi_entropy(
                    s, embed=3, delay=2, q=q,
                    eps_min=eps / 2.0, eps_max=eps, eps_count=1,
                )
                lo, hi = s.min(), s.max()
                epsi = int(1.0 / (eps / (hi - lo)))
                brute = _brute_entropies(s, embed=3, delay=2, q=q, epsi=epsi)
                self.assertAlmostEqual(
                    res["hq"][-1, 0], brute[-1], places=12,
                )

    def test_multivariate_matches_brute_force(self):
        rng = np.random.default_rng(7)
        data = np.column_stack(
            [rng.standard_normal(120), np.cumsum(rng.standard_normal(120))]
        )
        res = renyi_entropy(
            data, embed=2, delay=1, q=2.0,
            eps_min=0.5, eps_max=1.0, eps_count=1,
        )
        ranges = np.ptp(data, axis=0)
        epsi = int(1.0 / (1.0 / ranges.max()))
        brute = _brute_entropies(data, embed=2, delay=1, q=2.0, epsi=epsi)
        self.assertAlmostEqual(res["hq"][-1, 0], brute[-1], places=12)

    def test_full_interval_single_box(self):
        # Box size equal to the full data range -> epsi = 1 -> a single
        # joint box -> every Renyi order gives 0.
        s = np.arange(1.0, 51.0)
        res = renyi_entropy(
            s, embed=2, delay=1, q=2.0,
            eps_min=24.5, eps_max=49.0, eps_count=1,
        )
        np.testing.assert_allclose(res["hq"], 0.0, atol=0.0)


class TestBoxcountLadder(unittest.TestCase):
    def test_ladder_matches_c_replication(self):
        for lo, hi, count in ((0.05, 3.0, 6), (1e-3, 1.0, 20), (0.4, 1.0, 1)):
            deps, epsis = _epsilon_ladder(lo, hi, count)
            ref_deps, ref_epsis = _ref_ladder(lo, hi, count)
            np.testing.assert_array_equal(deps, ref_deps)
            np.testing.assert_array_equal(epsis, ref_epsis)

    def test_ladder_descends_and_epsis_increase(self):
        deps, epsis = _epsilon_ladder(1e-3, 1.0, 20)
        self.assertTrue(np.all(np.diff(deps) < 0.0))
        self.assertTrue(np.all(np.diff(epsis) > 0))
        self.assertAlmostEqual(deps[0], 1.0, places=12)

    def test_invalid_bounds_raise(self):
        with self.assertRaises(ValueError):
            _epsilon_ladder(1.0, 1.0, 5)
        with self.assertRaises(ValueError):
            _epsilon_ladder(2.0, 1.0, 5)
        with self.assertRaises(ValueError):
            _epsilon_ladder(0.5, 1.0, 0)


class TestBoxcountCore(unittest.TestCase):
    def test_scalar_matches_brute_force(self):
        rng = np.random.default_rng(42)
        s = np.cumsum(rng.standard_normal(200))
        embed, delay, q = 4, 2, 2.0
        res = renyi_entropy(s, embed=embed, delay=delay, q=q,
                            eps_min=0.05, eps_max=3.0, eps_count=6)
        n_vars = 1
        m_full = n_vars * embed
        self.assertEqual(res["hq"].shape, (m_full, 6))
        self.assertEqual(res["components"].tolist(), [1] * m_full)
        self.assertEqual(res["embeddings"].tolist(), [1, 2, 3, 4])

        interval = np.ptp(s)
        ref_deps, ref_epsis = _ref_ladder(0.05 / interval, 3.0 / interval, 6)
        np.testing.assert_allclose(res["eps"], ref_deps * interval,
                                   rtol=1e-15, atol=0.0)
        for k, epsi in enumerate(ref_epsis):
            brute = _brute_entropies(s, embed=embed, delay=delay, q=q,
                                     epsi=int(epsi))
            np.testing.assert_allclose(res["hq"][:, k], brute,
                                       rtol=1e-12, atol=1e-14)

    def test_multivariate_interleaved_forward_order(self):
        rng = np.random.default_rng(11)
        data = np.column_stack([
            rng.standard_normal(160),
            np.cumsum(rng.standard_normal(160)),
        ])
        embed, delay, q = 3, 2, 1.0
        res = renyi_entropy(data, embed=embed, delay=delay, q=q,
                            eps_min=0.1, eps_max=2.0, eps_count=4)
        self.assertEqual(res["components"].tolist(), [1, 2] * embed)
        self.assertEqual(res["embeddings"].tolist(), [1, 1, 2, 2, 3, 3])

        interval = np.ptp(data, axis=0).max()
        _, ref_epsis = _ref_ladder(0.1 / interval, 2.0 / interval, 4)
        for k, epsi in enumerate(ref_epsis):
            brute = _brute_entropies(data, embed=embed, delay=delay, q=q,
                                     epsi=int(epsi))
            np.testing.assert_allclose(res["hq"][:, k], brute,
                                       rtol=1e-12, atol=1e-14)

    def test_dhq_increments(self):
        rng = np.random.default_rng(5)
        s = rng.standard_normal(120)
        res = renyi_entropy(s, embed=3, delay=1, q=2.0,
                            eps_min=0.02, eps_max=0.9, eps_count=5)
        np.testing.assert_allclose(res["dhq"][0], res["hq"][0], rtol=0,
                                   atol=0)
        np.testing.assert_allclose(res["dhq"][1:], np.diff(res["hq"], axis=0),
                                   rtol=0, atol=0)

    def test_rows_monotone_for_positive_q(self):
        rng = np.random.default_rng(21)
        s = rng.standard_normal(300)
        for q in (0.5, 2.0):
            res = renyi_entropy(s, embed=4, delay=1, q=q,
                                eps_min=0.01, eps_max=0.9, eps_count=4)
            # Refining the partition cannot decrease the entropy.
            self.assertTrue(np.all(np.diff(res["hq"], axis=0) >= -1e-12))

    def test_single_ladder_step(self):
        rng = np.random.default_rng(9)
        s = np.cumsum(rng.standard_normal(100))
        res = renyi_entropy(s, embed=2, delay=1, q=2.0,
                            eps_min=0.1, eps_max=0.5, eps_count=1)
        self.assertEqual(res["eps"].shape, (1,))
        self.assertAlmostEqual(res["eps"][0], 0.5, places=12)

    def test_default_bounds_match_c_defaults(self):
        rng = np.random.default_rng(31)
        s = rng.standard_normal(100)
        res = renyi_entropy(s, embed=2, delay=1, q=2.0, eps_count=3)
        interval = np.ptp(s)
        # Default relative bounds are 1e-3 and 1.0.
        ref_deps, _ = _ref_ladder(1e-3, 1.0, 3)
        np.testing.assert_allclose(res["eps"], ref_deps * interval,
                                   rtol=1e-15, atol=0.0)


class TestBoxcountErrors(unittest.TestCase):
    def test_constant_series_raises(self):
        with self.assertRaises(RuntimeError):
            renyi_entropy(np.full(50, 3.0), embed=2, delay=1)

    def test_too_short_series_raises(self):
        with self.assertRaises(ValueError):
            renyi_entropy(np.arange(10.0), embed=10, delay=1)

    def test_eps_max_above_range_raises(self):
        with self.assertRaises(ValueError):
            renyi_entropy(np.arange(50.0), embed=2, delay=1, eps_max=1e6)

    def test_eps_min_above_eps_max_raises(self):
        with self.assertRaises(ValueError):
            renyi_entropy(np.arange(50.0), embed=2, delay=1,
                          eps_min=5.0, eps_max=1.0)

    def test_non_finite_q_raises(self):
        with self.assertRaises(ValueError):
            renyi_entropy(np.arange(50.0), embed=2, delay=1, q=np.nan)

    def test_invalid_embed_delay_raises(self):
        with self.assertRaises(ValueError):
            renyi_entropy(np.arange(50.0), embed=0, delay=1)
        with self.assertRaises(ValueError):
            renyi_entropy(np.arange(50.0), embed=2, delay=0)

    def test_3d_input_raises(self):
        with self.assertRaises(ValueError):
            renyi_entropy(np.zeros((10, 2, 2)), embed=2, delay=1)

    def test_renyi_entropy_invalid_eps_raises(self):
        with self.assertRaises(ValueError):
            renyi_entropy(np.arange(50.0), embed=2, delay=1, q=2.0,
                          eps_min=0.0)
        with self.assertRaises(ValueError):
            renyi_entropy(np.arange(50.0), embed=2, delay=1, q=2.0,
                          eps_min=-1.0)


if __name__ == "__main__":
    unittest.main()
