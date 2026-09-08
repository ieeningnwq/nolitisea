"""Tests for the ``c1``."""

import unittest

import numpy as np
from scipy.special import psi as digamma

from nolitisea.dimension.c1 import _psi, c1


def _brute_c1(data, embed_min, embed_max, delay, theiler, ncmin,
              resolution, kmax, seed):
    """Independent naive reimplementation of the Fortran c1/d1 pair.

    Uses no tree, no parallelism and builds every order-``m`` embedding
    directly in the Fortran coordinate order (oldest delay block first,
    components in file order, coordinates capped at the total dimension
    ``m``).  Only ``_psi`` is shared with the implementation; the table
    itself is covered by :class:`TestPsi`.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    n, nv = data.shape

    col = data[:, 0]
    sd = np.sqrt(((col - col.mean()) ** 2).mean())
    resl = np.log(2.0) / resolution
    rng = np.random.default_rng(seed)

    results = []
    for m in range(embed_min, embed_max + 1):
        mt = (m - 1) // nv + 1
        ncomp = n - (mt - 1) * delay
        E = np.empty((ncomp, m))
        for r in range(ncomp):
            t = r + (mt - 1) * delay
            coords = []
            for i in range(mt - 1, -1, -1):
                for c in range(nv):
                    coords.append(data[t - i * delay, c])
            E[r] = coords[:m]

        eps_list = []
        mass_list = []
        pr = 0.0
        pl_start = np.log(1.0 / (n - (m - 1) * delay))
        idx = np.arange(ncomp)
        i = 0
        while True:
            pl = pl_start + i * resl
            if pl > 0.0:
                break
            N0 = ncomp - 2 * theiler - 1
            kpr = int(np.exp(pr) * N0) + 1
            k = int(np.exp(pl) * N0) + 1
            if k > kmax:
                N = int(N0 * kmax / k)
                k = kmax
                ncomp_eff = N + 2 * theiler + 1
            else:
                N = N0
                ncomp_eff = ncomp
                k = min(k, N)
            pln = _psi(k) - np.log(N)
            if k != kpr:
                centers = rng.permutation(ncomp)[:ncmin]
                eps = np.exp(pln / m) * sd
                queue = list(centers)
                contribs = []
                while queue:
                    next_queue = []
                    for nn in queue:
                        diffs = np.abs(E - E[nn]).max(axis=1)
                        md = np.abs(idx - nn) % ncomp_eff
                        mask = (
                            (md > theiler)
                            & (md < ncomp_eff - theiler)
                            & (diffs < eps)
                        )
                        dists = diffs[mask]
                        if dists.size < k:
                            next_queue.append(nn)
                        else:
                            contribs.append(np.log(
                                max(np.partition(dists, k - 1)[k - 1], 1e-20)
                            ))
                    queue = next_queue
                    eps = eps * np.sqrt(2.0)
                eln = np.sum(contribs) / (ncmin - (mt - 1) * delay)
                if pln != pr:
                    eps_list.append(np.exp(eln))
                    mass_list.append(np.exp(pln))
                    pr = pln
            i += 1
        results.append((m, np.asarray(eps_list), np.asarray(mass_list)))
    return results


def _fit_slope(eps, mass, m_lo=5e-4, m_hi=1.5e-2):
    """Least-squares slope of ``log(mass)`` vs ``log(eps)``.

    Uses only pairs whose corrected mass lies in ``[m_lo, m_hi]``: below
    the window the per-step estimates are noisy, above it the ``kmax``
    clamp distorts the ladder.
    """
    eps = np.asarray(eps, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    sel = (mass >= m_lo) & (mass <= m_hi)
    x = np.log(eps[sel])
    y = np.log(mass[sel])
    return np.polyfit(x, y, 1)[0]


class TestPsi(unittest.TestCase):
    def test_table_values(self):
        self.assertEqual(_psi(0), 0.0)
        self.assertAlmostEqual(_psi(1), -0.57721566490, places=10)
        self.assertAlmostEqual(_psi(2), 0.42278433509, places=10)
        self.assertAlmostEqual(_psi(20), 2.97052399224, places=10)

    def test_large_k_matches_digamma(self):
        for k in (21, 25, 100, 5000):
            self.assertAlmostEqual(
                _psi(k), digamma(k), delta=2.0 / (12.0 * k * k) + 1e-12
            )


class TestC1BruteForce(unittest.TestCase):
    def test_scalar_matches_brute_force(self):
        rng = np.random.default_rng(3)
        s = np.cumsum(rng.standard_normal(90))
        res = c1(
            s, embed_min=1, embed_max=2, delay=1, theiler=2,
            n_centers=30, resolution=4, kmax=10, seed=3,
        )
        brute = _brute_c1(
            s, 1, 2, 1, 2, 30, 4, 10, 3,
        )
        self.assertEqual(res["embed"].tolist(), [1, 2])
        for (m, b_eps, b_mass), eps_arr, mass_arr in zip(
            brute, res["eps"], res["mass"]
        ):
            self.assertIn(m, (1, 2))
            np.testing.assert_array_equal(eps_arr, b_eps)
            np.testing.assert_array_equal(mass_arr, b_mass)

    def test_multivariate_partial_cap_matches_brute_force(self):
        rng = np.random.default_rng(11)
        data = np.column_stack([
            rng.standard_normal(90),
            np.cumsum(rng.standard_normal(90)),
        ])
        res = c1(
            data, embed_min=2, embed_max=3, delay=2, theiler=2,
            n_centers=30, resolution=2, kmax=10, seed=5,
        )
        brute = _brute_c1(
            data, 2, 3, 2, 2, 30, 2, 10, 5,
        )
        for (m, b_eps, b_mass), eps_arr, mass_arr in zip(
            brute, res["eps"], res["mass"]
        ):
            np.testing.assert_array_equal(eps_arr, b_eps)
            np.testing.assert_array_equal(mass_arr, b_mass)


class TestC1DimensionEstimates(unittest.TestCase):
    def test_uniform_noise_recovers_embedding_dimension(self):
        rng = np.random.default_rng(42)
        s = rng.uniform(size=6000)
        res = c1(
            s, embed_min=1, embed_max=3, delay=1, theiler=0,
            n_centers=200, kmax=100, seed=1,
        )
        for j, m in enumerate(res["embed"]):
            slope = _fit_slope(res["eps"][j], res["mass"][j])
            self.assertAlmostEqual(
                slope, float(m), delta=0.3,
                msg=f"m={m}: slope {slope:.3f} != {m}",
            )

    def test_limit_cycle_has_dimension_one(self):
        t = np.arange(4000.0)
        s = np.sin(2.0 * np.pi * t / 500.0)
        res = c1(
            s, embed_min=3, embed_max=3, delay=10, theiler=2,
            n_centers=200, kmax=80, seed=2,
        )
        # A perfectly periodic signal has ~n/period exact phase-space
        # copies at distance ~1e-16, so only fit above their mass range
        # (k <= n/period).
        slope = _fit_slope(
            res["eps"][0], res["mass"][0], m_lo=2e-3, m_hi=1.5e-2,
        )
        self.assertAlmostEqual(slope, 1.0, delta=0.25)


class TestC1Behaviour(unittest.TestCase):
    def _small(self, **kwargs):
        rng = np.random.default_rng(7)
        s = np.cumsum(rng.standard_normal(500))
        params = {
            "embed_min": 1, "embed_max": 3, "delay": 1, "theiler": 1,
            "n_centers": 40, "kmax": 20, "seed": 7,
        }
        params.update(kwargs)
        return c1(s, **params)

    def test_output_structure(self):
        res = self._small()
        self.assertEqual(res["embed"].tolist(), [1, 2, 3])
        for eps_arr, mass_arr in zip(res["eps"], res["mass"]):
            self.assertEqual(eps_arr.shape, mass_arr.shape)
            self.assertGreaterEqual(eps_arr.size, 1)
            self.assertTrue(np.all(eps_arr > 0.0))
            self.assertTrue(np.all(mass_arr > 0.0))
            # The dedup guarantees strictly increasing corrected masses.
            self.assertTrue(np.all(np.diff(mass_arr) > 0.0))

    def test_seed_reproducibility(self):
        res_a = self._small()
        res_b = self._small()
        for eps_a, eps_b in zip(res_a["eps"], res_b["eps"]):
            np.testing.assert_array_equal(eps_a, eps_b)

        res_c = self._small(seed=8)
        differing = any(
            not np.array_equal(eps_a, eps_c)
            for eps_a, eps_c in zip(res_a["eps"], res_c["eps"])
        )
        self.assertTrue(differing)

    def test_parallel_bitwise_consistency(self):
        res_serial = self._small(n_jobs=1)
        res_thread = self._small(n_jobs=3, backend="thread")
        res_process = self._small(n_jobs=2, backend="process")
        for res in (res_thread, res_process):
            np.testing.assert_array_equal(res["embed"], res_serial["embed"])
            for eps_a, eps_b in zip(res_serial["eps"], res["eps"]):
                np.testing.assert_array_equal(eps_a, eps_b)
            for mass_a, mass_b in zip(res_serial["mass"], res["mass"]):
                np.testing.assert_array_equal(mass_a, mass_b)


class TestC1Errors(unittest.TestCase):
    def test_short_series(self):
        with self.assertRaises(ValueError):
            c1(np.arange(5.0), embed_min=10, embed_max=10, delay=1)

    def test_embed_range(self):
        with self.assertRaises(ValueError):
            c1(np.arange(50.0), embed_min=3, embed_max=2)

    def test_bad_delay(self):
        with self.assertRaises(ValueError):
            c1(np.arange(50.0), embed_min=1, embed_max=2, delay=0)

    def test_bad_theiler(self):
        with self.assertRaises(ValueError):
            c1(np.arange(50.0), embed_min=1, embed_max=2, delay=1,
               theiler=30)

    def test_bad_kmax(self):
        with self.assertRaises(ValueError):
            c1(np.arange(50.0), embed_min=1, embed_max=2, kmax=0)

    def test_bad_resolution(self):
        with self.assertRaises(ValueError):
            c1(np.arange(50.0), embed_min=1, embed_max=2, resolution=0.0)

    def test_bad_seed(self):
        with self.assertRaises(ValueError):
            c1(np.arange(50.0), embed_min=1, embed_max=2, seed=-1)

    def test_n_centers_too_large(self):
        with self.assertRaises(ValueError):
            c1(np.arange(50.0), embed_min=1, embed_max=1, delay=1,
               n_centers=60)

    def test_n_centers_below_embedding_loss(self):
        with self.assertRaises(ValueError):
            c1(np.arange(100.0), embed_min=5, embed_max=5, delay=10,
               n_centers=30)

    def test_constant_series(self):
        with self.assertRaises(ValueError):
            c1(np.full(100, 1.0), embed_min=1, embed_max=2)




if __name__ == "__main__":
    unittest.main()
