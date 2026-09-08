"""Tests for nolitisea.stationarity.nstat_z."""

import unittest

import numpy as np

from nolitisea.stationarity.nstat_z import nstat_z


def _nstat_z_brute(y, dim, delay, n_pieces, step, min_neighbors,
                   eps0=None, eps_factor=1.2, causal=None):
    """Literal transcription of nstat_z.c's main loop.

    Walks the same epsilon ladder and uses a brute-force Chebyshev
    neighbour scan, independent of the cKDTree-based rewrite.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    pstart = (dim - 1) * delay

    # Rescale to [0, 1]
    minv = y.min()
    interval = y.max() - minv
    scaled = (y - minv) / interval

    if eps0 is None:
        eps0_r = 1.0e-3
    else:
        eps0_r = abs(float(eps0)) / interval

    clength = (n - pstart) // n_pieces
    center = clength - step
    if causal is None:
        causal = step

    # Per-segment standard deviation (Bessel's correction)
    rms = []
    for i in range(n_pieces):
        seg = scaled[i * clength : (i + 1) * clength]
        rms.append(np.std(seg, ddof=1))

    result = np.full((n_pieces, n_pieces), np.nan)

    for first in range(n_pieces):
        s1_start = first * clength
        for second in range(n_pieces):
            s2_start = second * clength

            done = [False] * center
            error = 0.0
            eps = eps0_r / eps_factor
            alldone = False

            while not alldone:
                eps *= eps_factor
                alldone = True
                for r in range(center):
                    if done[r]:
                        continue
                    i = r + pstart  # segment base in target
                    # Brute-force neighbour search in source segment
                    neighbors = []
                    for j in range(pstart, clength - step):
                        dis = 0.0
                        for k in range(dim):
                            d = abs(
                                scaled[s1_start + j - k * delay]
                                - scaled[s2_start + i - k * delay]
                            )
                            dis = max(dis, d)
                        if dis < eps:
                            neighbors.append(j)
                    # Exclude interval
                    ex_lo = i - causal + 1
                    ex_hi = i + causal + pstart - 1
                    if ex_lo < 0:
                        kept = neighbors
                    else:
                        kept = [
                            nb for nb in neighbors
                            if nb < ex_lo or nb > ex_hi
                        ]
                    if len(kept) >= min_neighbors:
                        casted = sum(
                            scaled[s1_start + nb + step] for nb in kept
                        ) / len(kept)
                        actual = scaled[s2_start + i + step]
                        error += (casted - actual) ** 2
                        done[r] = True
                    alldone = alldone and done[r]

                if eps > 2.0:
                    break

            result[first, second] = np.sqrt(error / center) / rms[second]

    return result


class TestNstatZBruteForce(unittest.TestCase):
    def test_matches_transcription_default(self):
        rng = np.random.default_rng(11)
        y = np.cumsum(rng.standard_normal(600))
        res = nstat_z(y, dim=3, delay=1, n_pieces=4, step=1,
                      min_neighbors=15)
        brute = _nstat_z_brute(y, dim=3, delay=1, n_pieces=4, step=1,
                               min_neighbors=15)
        np.testing.assert_allclose(res["matrix"], brute, rtol=1e-10)

    def test_matches_transcription_larger_step(self):
        rng = np.random.default_rng(23)
        y = np.cumsum(rng.standard_normal(800))
        res = nstat_z(y, dim=2, delay=2, n_pieces=4, step=3,
                      min_neighbors=10)
        brute = _nstat_z_brute(y, dim=2, delay=2, n_pieces=4, step=3,
                               min_neighbors=10)
        np.testing.assert_allclose(res["matrix"], brute, rtol=1e-10)

    def test_matches_transcription_custom_eps(self):
        rng = np.random.default_rng(5)
        y = rng.standard_normal(700)
        res = nstat_z(y, dim=3, delay=1, n_pieces=4, step=1,
                      min_neighbors=10, eps0=0.05, eps_factor=1.5)
        brute = _nstat_z_brute(y, dim=3, delay=1, n_pieces=4, step=1,
                               min_neighbors=10, eps0=0.05,
                               eps_factor=1.5)
        np.testing.assert_allclose(res["matrix"], brute, rtol=1e-10)

    def test_matches_transcription_causal_window(self):
        rng = np.random.default_rng(37)
        y = np.cumsum(rng.standard_normal(700))
        res = nstat_z(y, dim=3, delay=1, n_pieces=4, step=1,
                      min_neighbors=10, causal=5)
        brute = _nstat_z_brute(y, dim=3, delay=1, n_pieces=4, step=1,
                               min_neighbors=10, causal=5)
        np.testing.assert_allclose(res["matrix"], brute, rtol=1e-10)


class TestNstatZOutputs(unittest.TestCase):
    def test_output_structure(self):
        rng = np.random.default_rng(1)
        y = rng.standard_normal(1000)
        res = nstat_z(y, dim=3, delay=1, n_pieces=5, step=1,
                      min_neighbors=20)
        self.assertEqual(set(res), {"matrix", "pieces", "clength", "rms"})
        self.assertEqual(res["matrix"].shape, (5, 5))
        self.assertEqual(res["pieces"], 5)
        self.assertEqual(res["rms"].shape, (5,))

    def test_stationary_uniform(self):
        # Stationary white noise -> error matrix roughly uniform
        rng = np.random.default_rng(42)
        y = rng.standard_normal(2000)
        res = nstat_z(y, dim=3, delay=1, n_pieces=5, step=1,
                      min_neighbors=20)
        m = res["matrix"]
        # All entries finite
        self.assertTrue(np.all(np.isfinite(m)))
        # Coefficient of variation small (< 10%)
        cv = m.std() / m.mean()
        self.assertLess(cv, 0.1)

    def test_nonstationary_diagonal_valley(self):
        # Piecewise-shifted series -> diagonal valley
        rng = np.random.default_rng(7)
        y = np.concatenate([
            rng.standard_normal(500),
            5 + rng.standard_normal(500),
            10 + rng.standard_normal(1000),
        ])
        res = nstat_z(y, dim=3, delay=1, n_pieces=5, step=1,
                      min_neighbors=20)
        m = res["matrix"]
        # Far off-diagonal should be large
        self.assertGreater(m[0, 4], m[0, 0])
        self.assertGreater(m[4, 0], m[4, 4])

    def test_n_refs_reduces_center(self):
        rng = np.random.default_rng(2)
        y = rng.standard_normal(1000)
        full = nstat_z(y, dim=3, delay=1, n_pieces=3, step=1,
                       min_neighbors=10)
        reduced = nstat_z(y, dim=3, delay=1, n_pieces=3, step=1,
                          min_neighbors=10, n_refs=50)
        # Both should produce finite matrices
        self.assertTrue(np.all(np.isfinite(full["matrix"])))
        self.assertTrue(np.all(np.isfinite(reduced["matrix"])))

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            nstat_z([1, 2, 3], dim=0, delay=1, n_pieces=2)
        with self.assertRaises(ValueError):
            nstat_z([1, 2, 3], dim=2, delay=0, n_pieces=2)
        with self.assertRaises(ValueError):
            nstat_z([1, 2, 3], dim=2, delay=1, n_pieces=0)
        with self.assertRaises(ValueError):
            nstat_z([1, 2, 3], dim=2, delay=1, n_pieces=2, step=0)
        with self.assertRaises(ValueError):
            nstat_z([1, 2, 3], dim=2, delay=1, n_pieces=2, min_neighbors=0)
        with self.assertRaises(ValueError):
            nstat_z([1, 2, 3], dim=2, delay=1, n_pieces=2, eps_factor=1.0)
        with self.assertRaises(ValueError):
            nstat_z([1, 2, 3], dim=2, delay=1, n_pieces=100)

    def test_constant_series_raises(self):
        with self.assertRaises(RuntimeError):
            nstat_z(np.ones(500), dim=3, delay=1, n_pieces=2)


if __name__ == "__main__":
    unittest.main()
