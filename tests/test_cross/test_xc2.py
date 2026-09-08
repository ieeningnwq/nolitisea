"""Tests for nolitisea.cross.xc2."""

import unittest

import numpy as np

from nolitisea.core.embed import lag_block_delay_embed
from nolitisea.cross.xc2 import cross_correlation_integral
from nolitisea.generate.henon import henon


def _bruteforce_counts(a, b, embed, delay, eps_list, n_center, n_pairs):
    """Reference per-order pair counts with the documented conventions.

    References are rows of ``Eb``; neighbours are rows of ``Ea``; a pair
    is counted at order ``m`` when the maximum over the first ``m``
    coordinates is strictly below ``eps``.  The scan stops after a
    reference once ``scanned >= n_center`` and the full-order count
    reached ``n_pairs``.
    """
    da = np.asarray(a, dtype=np.float64)
    db = np.asarray(b, dtype=np.float64)
    if da.ndim == 1:
        da = da[:, None]
    if db.ndim == 1:
        db = db[:, None]
    Ea = lag_block_delay_embed(da, embed, delay)
    Eb = lag_block_delay_embed(db, embed, delay)
    m_full = Ea.shape[1]

    out = []
    for eps in eps_list:
        counts = np.zeros(m_full, dtype=np.int64)
        scanned = 0
        for r in range(Eb.shape[0]):
            diffs = np.abs(Ea - Eb[r])
            within = np.maximum.accumulate(diffs, axis=1) < eps
            counts += within.sum(axis=0)
            scanned += 1
            if scanned >= n_center and counts[m_full - 1] >= n_pairs:
                break
        c = counts / (scanned * Ea.shape[0])
        out.append((eps, c, scanned))
    return out


def _henon_series(n=250, seed=0):
    x, _ = henon(n=n, discard=200 + 1000 * seed)
    return x


class TestCrossCorrelationIntegral(unittest.TestCase):
    def test_scalar_matches_bruteforce(self):
        a = _henon_series(seed=0)
        b = _henon_series(seed=1)
        eps_list = [0.5, 0.25, 0.1, 0.05]
        result = cross_correlation_integral(
            a, b, embed=3, delay=1, eps=eps_list,
            n_center=10 ** 9, n_pairs=10 ** 9,
        )
        np.testing.assert_allclose(result["eps"], sorted(eps_list)[::-1])
        np.testing.assert_allclose(result["orders"], [2, 3])
        for col, (_, c_ref, scanned) in zip(result["c"].T,
                                            _bruteforce_counts(
                                                a, b, 3, 1, eps_list,
                                                10 ** 9, 10 ** 9)):
            np.testing.assert_allclose(col, c_ref[1:], atol=1e-14)
            self.assertEqual(col.size, 2)

    def test_multivariate_matches_bruteforce(self):
        rng = np.random.default_rng(11)
        a = np.column_stack([_henon_series(seed=0),
                             np.sin(np.arange(250) / 7.0)])
        b = a + 0.01 * rng.standard_normal(a.shape)
        eps_list = [0.4, 0.2]
        result = cross_correlation_integral(
            a, b, embed=2, delay=2, eps=eps_list,
            n_center=10 ** 9, n_pairs=10 ** 9,
        )
        np.testing.assert_allclose(result["orders"], [2, 3, 4])
        for col, (_, c_ref, _) in zip(result["c"].T,
                                      _bruteforce_counts(
                                          a, b, 2, 2, eps_list,
                                          10 ** 9, 10 ** 9)):
            np.testing.assert_allclose(col, c_ref[1:], atol=1e-14)

    def test_counts_nonincreasing_along_orders(self):
        a = _henon_series(seed=0)
        b = _henon_series(seed=1)
        result = cross_correlation_integral(
            a, b, embed=5, delay=2, eps=[0.3, 0.1],
            n_center=10 ** 9, n_pairs=10 ** 9,
        )
        for col in result["c"].T:
            diffs = np.diff(col)
            self.assertLessEqual(diffs.max(), 0.0 + 1e-15)

    def test_early_stopping_matches_bruteforce(self):
        a = _henon_series(seed=0, n=400)
        b = _henon_series(seed=1, n=400)
        eps_list = [0.4, 0.2]
        result = cross_correlation_integral(
            a, b, embed=2, delay=1, eps=eps_list,
            n_center=5, n_pairs=10,
        )
        for col, (_, c_ref, scanned) in zip(result["c"].T,
                                            _bruteforce_counts(
                                                a, b, 2, 1, eps_list, 5, 10)):
            np.testing.assert_allclose(col, c_ref[1:], atol=1e-14)
        np.testing.assert_allclose(
            result["scanned"],
            [scanned for _, _, scanned in _bruteforce_counts(
                a, b, 2, 1, eps_list, 5, 10)],
        )
        self.assertTrue((result["scanned"] < 400 - 1).any())

    def test_automatic_ladder_descends_by_octave_factor(self):
        a = _henon_series(seed=0, n=150)
        b = _henon_series(seed=1, n=150)
        resolution = 2.0
        result = cross_correlation_integral(
            a, b, embed=2, delay=1, eps=None, resolution=resolution,
            n_center=10 ** 9, n_pairs=10 ** 9,
        )
        eps = result["eps"]
        self.assertGreater(eps.size, 3)
        factors = eps[:-1] / eps[1:]
        np.testing.assert_allclose(
            factors, 2.0 ** (1.0 / resolution), rtol=1e-12
        )
        self.assertLessEqual(eps[0], 1.001 * max(np.ptp(a), np.ptp(b)))

    def test_explicit_eps_evaluates_even_when_empty(self):
        a = _henon_series(seed=0, n=120)
        b = _henon_series(seed=1, n=120)
        result = cross_correlation_integral(
            a, b, embed=2, delay=1, eps=[1e-12, 1.0],
            n_center=10 ** 9, n_pairs=10 ** 9,
        )
        np.testing.assert_allclose(result["eps"], [1.0, 1e-12])
        self.assertEqual(result["c"][0, 1], 0.0)  # order 2 at tiny radius
        self.assertGreater(result["c"][0, 0], 0.0)

    def test_mismatched_components_raise(self):
        a = np.column_stack([_henon_series(n=100), _henon_series(n=100)])
        b = _henon_series(n=100)
        with self.assertRaises(ValueError):
            cross_correlation_integral(a, b, embed=2)

    def test_invalid_parameters_raise(self):
        a = _henon_series(n=100)
        b = _henon_series(n=100)
        with self.assertRaises(ValueError):
            cross_correlation_integral(a, b, embed=0)
        with self.assertRaises(ValueError):
            cross_correlation_integral(a, b, embed=2, delay=0)
        with self.assertRaises(ValueError):
            cross_correlation_integral(a, b, embed=2, resolution=-1.0)
        with self.assertRaises(ValueError):
            cross_correlation_integral(a, b, embed=2, eps=[0.0])
        with self.assertRaises(ValueError):
            cross_correlation_integral(a, b, embed=1)  # order < 2

    def test_series_too_short_raises(self):
        a = _henon_series(n=10)
        with self.assertRaises(ValueError):
            cross_correlation_integral(a, a, embed=20, delay=1)


if __name__ == "__main__":
    unittest.main()
