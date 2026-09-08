"""Tests for nolitisea.cross.xrecur."""

import unittest

import numpy as np

from nolitisea.core.embed import lag_block_delay_embed
from nolitisea.cross.xrecur import cross_recurrence
from nolitisea.generate.henon import henon


def _henon_series(n=200, seed=0):
    x, _ = henon(n=n, discard=200 + 1000 * seed)
    return x


def _rescale_9999(data):
    out = np.array(data, dtype=np.float64, copy=True)
    for j in range(out.shape[1]):
        lo, hi = out[:, j].min(), out[:, j].max()
        out[:, j] = (out[:, j] - lo) * (0.9999 / (hi - lo))
    return out


def _bruteforce_fixed(a, b, dim, delay, eps, step_a=1, step_b=1,
                      normalize=True, metric="chebyshev"):
    da = np.asarray(a, dtype=np.float64)
    db = np.asarray(b, dtype=np.float64)
    if da.ndim == 1:
        da = da[:, None]
    if db.ndim == 1:
        db = db[:, None]
    if normalize:
        da = _rescale_9999(da)
        db = _rescale_9999(db)
    Ea = lag_block_delay_embed(da, dim, delay)
    Eb = lag_block_delay_embed(db, dim, delay)
    if metric == "chebyshev":
        dist = np.abs(Ea[:, None, :] - Eb[None, :, :]).max(axis=2)
    else:
        dist = np.sqrt(((Ea[:, None, :] - Eb[None, :, :]) ** 2).sum(axis=2))
    pairs = []
    for i in range(0, Ea.shape[0], step_a):
        for j in range(0, Eb.shape[0], step_b):
            if dist[i, j] < eps:
                pairs.append((i, j))
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64)


def _bruteforce_kmin(a, b, dim, delay, eps, kmin, step_a=1, step_b=1):
    da = np.asarray(a, dtype=np.float64)
    db = np.asarray(b, dtype=np.float64)
    if da.ndim == 1:
        da = da[:, None]
    if db.ndim == 1:
        db = db[:, None]
    da = _rescale_9999(da)
    db = _rescale_9999(db)
    Ea = lag_block_delay_embed(da, dim, delay)
    Eb = lag_block_delay_embed(db, dim, delay)
    dist = np.abs(Ea[:, None, :] - Eb[None, :, :]).max(axis=2)

    pairs = []
    done = np.zeros(Ea.shape[0], dtype=bool)
    eps_cur = eps / 1.1
    for _ in range(100):
        eps_cur *= 1.1
        ntodo = 0
        for i in range(0, Ea.shape[0], step_a):
            if done[i]:
                continue
            ntodo += 1
            cols = [j for j in range(0, Eb.shape[0], step_b)
                    if dist[i, j] < eps_cur]
            if len(cols) < kmin:
                continue
            done[i] = True
            ntodo -= 1
            pairs.extend((i, j) for j in cols)
        if ntodo == 0:
            break
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(sorted(pairs), dtype=np.int64)


class TestCrossRecurrence(unittest.TestCase):
    def test_fixed_eps_matches_bruteforce(self):
        a = _henon_series(seed=0, n=120)
        b = _henon_series(seed=1, n=120)
        pairs = cross_recurrence(a, b, dim=3, delay=1, eps=0.2)
        expected = _bruteforce_fixed(a, b, 3, 1, 0.2)
        np.testing.assert_array_equal(pairs, expected)

    def test_fixed_eps_multivariate_matches_bruteforce(self):
        rng = np.random.default_rng(5)
        a = np.column_stack([_henon_series(seed=0, n=100),
                             np.cos(np.arange(100) / 5.0)])
        b = a + 0.01 * rng.standard_normal(a.shape)
        pairs = cross_recurrence(a, b, dim=2, delay=2, eps=0.3)
        expected = _bruteforce_fixed(a, b, 2, 2, 0.3)
        np.testing.assert_array_equal(pairs, expected)

    def test_euclidean_metric_matches_bruteforce(self):
        a = _henon_series(seed=0, n=100)
        b = _henon_series(seed=1, n=100)
        pairs = cross_recurrence(a, b, dim=2, delay=1, eps=0.5,
                                 metric="euclidean")
        expected = _bruteforce_fixed(a, b, 2, 1, 0.5, metric="euclidean")
        np.testing.assert_array_equal(pairs, expected)

    def test_steps_match_bruteforce(self):
        a = _henon_series(seed=0, n=150)
        b = _henon_series(seed=1, n=150)
        pairs = cross_recurrence(a, b, dim=2, delay=1, eps=0.2,
                                 step_a=2, step_b=3)
        expected = _bruteforce_fixed(a, b, 2, 1, 0.2, step_a=2, step_b=3)
        np.testing.assert_array_equal(pairs, expected)
        if pairs.size:
            self.assertTrue((pairs[:, 0] % 2 == 0).all())
            self.assertTrue((pairs[:, 1] % 3 == 0).all())

    def test_output_sorted_lexicographically(self):
        a = _henon_series(seed=0, n=150)
        b = _henon_series(seed=1, n=150)
        pairs = cross_recurrence(a, b, dim=2, delay=1, eps=0.25)
        order = np.lexsort((pairs[:, 1], pairs[:, 0]))
        np.testing.assert_array_equal(pairs, pairs[order])

    def test_thinning_is_subset_and_deterministic(self):
        a = _henon_series(seed=0, n=150)
        b = _henon_series(seed=1, n=150)
        full = cross_recurrence(a, b, dim=2, delay=1, eps=0.25)
        thin1 = cross_recurrence(a, b, dim=2, delay=1, eps=0.25,
                                 percentage=40.0)
        thin2 = cross_recurrence(a, b, dim=2, delay=1, eps=0.25,
                                 percentage=40.0)
        np.testing.assert_array_equal(thin1, thin2)
        full_set = {tuple(p) for p in full}
        for p in thin1:
            self.assertIn(tuple(p), full_set)
        self.assertLess(thin1.shape[0], full.shape[0])
        self.assertGreater(thin1.shape[0], 0)

    def test_kmin_matches_bruteforce(self):
        a = _henon_series(seed=0, n=100)
        b = _henon_series(seed=1, n=100)
        pairs = cross_recurrence(a, b, dim=2, delay=1, eps=0.1, kmin=4)
        expected = _bruteforce_kmin(a, b, 2, 1, 0.1, 4)
        np.testing.assert_array_equal(pairs, expected)
        # every processed reference contributed at least kmin pairs
        for i in np.unique(pairs[:, 0]):
            self.assertGreaterEqual(int((pairs[:, 0] == i).sum()), 4)

    def test_kmin_covers_all_references(self):
        a = _henon_series(seed=0, n=80)
        b = _henon_series(seed=1, n=80)
        pairs = cross_recurrence(a, b, dim=2, delay=1, eps=0.05, kmin=1)
        rows = np.arange(0, a.size - 1)  # delay-embedded row count = n - 1
        np.testing.assert_array_equal(np.unique(pairs[:, 0]), rows)

    def test_normalize_matches_manual_rescaling(self):
        a = _henon_series(seed=0, n=120)
        b = _henon_series(seed=1, n=120)
        default = cross_recurrence(a, b, dim=2, delay=1, eps=0.2)
        manual = cross_recurrence(
            _rescale_9999(a[:, None]), _rescale_9999(b[:, None]),
            dim=2, delay=1, eps=0.2, normalize=False,
        )
        np.testing.assert_array_equal(default, manual)

    def test_disjoint_ranges_empty_without_normalization(self):
        a = _henon_series(seed=0, n=80) * 100.0 + 50.0
        b = _henon_series(seed=1, n=80) * 100.0 - 50.0
        pairs = cross_recurrence(a, b, dim=2, delay=1, eps=0.1,
                                 normalize=False)
        self.assertEqual(pairs.shape, (0, 2))

    def test_constant_component_raises(self):
        a = _henon_series(n=80)
        with self.assertRaises(ValueError):
            cross_recurrence(a, np.ones(80), dim=2, delay=1, eps=0.1)

    def test_invalid_parameters_raise(self):
        a = _henon_series(n=80)
        b = _henon_series(n=80)
        with self.assertRaises(ValueError):
            cross_recurrence(a, b, dim=0)
        with self.assertRaises(ValueError):
            cross_recurrence(a, b, dim=2, delay=0)
        with self.assertRaises(ValueError):
            cross_recurrence(a, b, dim=2, eps=0.0)
        with self.assertRaises(ValueError):
            cross_recurrence(a, b, dim=2, percentage=0.0)
        with self.assertRaises(ValueError):
            cross_recurrence(a, b, dim=2, metric="manhattan")
        with self.assertRaises(ValueError):
            cross_recurrence(
                np.column_stack([a, a]), b, dim=2
            )
        # unequal lengths are allowed
        mixed = cross_recurrence(a, b[:40], dim=2, eps=0.6)
        self.assertEqual(mixed.ndim, 2)
        self.assertEqual(mixed.shape[1], 2)
        self.assertTrue((mixed[:, 0] < a.size - 1).all())
        self.assertTrue((mixed[:, 1] < 40 - 1).all())

    def test_series_too_short_raises(self):
        a = _henon_series(n=10)
        with self.assertRaises(ValueError):
            cross_recurrence(a, a, dim=20, delay=1)


if __name__ == "__main__":
    unittest.main()
