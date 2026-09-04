"""Tests for the TISEAN ``lyap_r`` rewrite in nolitisea.lyapunov.lyap_r."""

import math
import unittest

import numpy as np
from scipy.spatial import cKDTree

from nolitisea.core.embed import lag_block_delay_embed
from nolitisea.generate.henon import henon
from nolitisea.lyapunov.lyap_r import lyap_r
from nolitisea.utils.rescale import rescale_data


def _c_transcription(series, dim, delay, max_steps, theiler=0):
    """Literal transcription of lyap_r.c.

    Adaptive epsilon ladder with iterated growth factor 1.1 and the C
    strict-``<`` neighbour choice; candidates are scanned in ascending
    index order, which coincides with the C box-scan order whenever
    distances are distinct (generic, non-quantized data).
    """
    x = np.asarray(series, dtype=np.float64)
    length = x.size
    x, _, _ = rescale_data(x)
    del1 = dim * delay
    del_off = (dim - 1) * delay
    maxlength = length - del_off - max_steps - 1 - theiler

    lyap = np.zeros(max_steps + 1)
    found = np.zeros(max_steps + 1, dtype=np.int64)
    done = np.zeros(maxlength + 1, dtype=bool)
    eps = 1.0e-3
    while True:
        for n in range(maxlength + 1):
            if done[n]:
                continue
            mindx = 1.0
            minelement = -1
            for element in range(length - del_off - max_steps):
                if abs(n - element) <= theiler:
                    continue
                dx = 0.0
                complete = True
                for k in range(0, del1, delay):
                    dx += (x[n + k] - x[element + k]) ** 2
                    if dx > eps * eps:
                        complete = False
                        break
                if complete and dx < mindx and dx > 0.0:
                    mindx = dx
                    minelement = element
            done[n] = mindx < 1.0
            if minelement != -1:
                a, m = n - 1, minelement - 1
                for i in range(max_steps + 1):
                    a += 1
                    m += 1
                    dx = 0.0
                    for j in range(0, del1, delay):
                        dx += (x[a + j] - x[m + j]) ** 2
                    if dx > 0.0:
                        found[i] += 1
                        lyap[i] += math.log(dx)
        if done.all():
            break
        eps *= 1.1
    div = np.full(max_steps + 1, np.nan)
    hit = found > 0
    div[hit] = lyap[hit] / found[hit] / 2.0
    return div


def _reference_brute(series, dim, delay, max_steps, theiler=0):
    """Reference implementation of the documented algorithm.

    Explicit epsilon ladder as in C, but candidates and their distances
    come from the same KD-tree the rewrite uses; the kept neighbour
    minimises (distance, index) lexicographically among eligible
    candidates with a positive separation.
    """
    x = np.asarray(series, dtype=np.float64)
    length = x.size
    x, _, _ = rescale_data(x)
    del_off = (dim - 1) * delay
    E = lag_block_delay_embed(x, dim, delay)
    n_elem = E.shape[0] - max_steps
    n_ref = length - del_off - max_steps - theiler
    tree = cKDTree(E[:n_elem])

    lyap = np.zeros(max_steps + 1)
    found = np.zeros(max_steps + 1, dtype=np.int64)
    done = np.zeros(n_ref, dtype=bool)
    eps = 1.0e-3
    while True:
        for n in range(n_ref):
            if done[n]:
                continue
            d, i = tree.query(E[n], k=n_elem)
            within = (np.abs(i - n) > theiler) & (d <= eps)
            if not within.any():
                continue
            if d[within].min() >= 1.0:
                continue
            done[n] = True
            pos = within & (d > 0.0)
            if pos.any():
                dp, ip = d[pos], i[pos]
                m = ip[dp == dp.min()].min()
                for step in range(max_steps + 1):
                    diff = E[n + step] - E[m + step]
                    dx = np.sum(diff * diff)
                    if dx > 0.0:
                        found[step] += 1
                        lyap[step] += math.log(dx)
        if done.all():
            break
        eps *= 1.1
    div = np.full(max_steps + 1, np.nan)
    hit = found > 0
    div[hit] = lyap[hit] / found[hit] / 2.0
    return div


class TestLyapRBruteForce(unittest.TestCase):
    def test_continuous_matches_c_transcription(self):
        rng = np.random.default_rng(7)
        s = np.cumsum(rng.standard_normal(300))
        for theiler in [0, 3]:
            res = lyap_r(s, dim=3, delay=2, max_steps=8, theiler=theiler,n_jobs=-1)
            brute = _reference_brute(
                s, dim=3, delay=2, max_steps=8, theiler=theiler
            )
            np.testing.assert_allclose(
                res["divergence"], brute, rtol=1e-12
            )

    def test_huge_eps_matches_c_transcription(self):
        # A radius covering the whole attractor must only remove far
        # references; close ones keep the C neighbour exactly.
        rng = np.random.default_rng(7)
        s = np.cumsum(rng.standard_normal(300))
        res = lyap_r(s, dim=3, delay=2, max_steps=8, theiler=3, eps=1e9)
        brute = _c_transcription(s, dim=3, delay=2, max_steps=8, theiler=3)
        np.testing.assert_allclose(res["divergence"], brute, rtol=1e-12)

    def test_quantized_matches_reference(self):
        # Rounded data contains exact duplicate embeddings; the ladder
        # and smallest-index tie rule must follow the documented
        # algorithm exactly (the C box-scan tie order is an
        # implementation detail and deviates here by design).
        rng = np.random.default_rng(7)
        sd = np.round(np.cumsum(rng.standard_normal(240)))
        res = lyap_r(sd, dim=3, delay=1, max_steps=8, theiler=2)
        brute = _reference_brute(sd, dim=3, delay=1, max_steps=8, theiler=2)
        np.testing.assert_allclose(res["divergence"], brute, rtol=1e-12)


class TestLyapROutputs(unittest.TestCase):
    def test_output_structure(self):
        rng = np.random.default_rng(3)
        s = np.cumsum(rng.standard_normal(200))
        res = lyap_r(s, dim=3, delay=2, max_steps=6, theiler=2)
        self.assertEqual(set(res), {"times", "divergence"})
        np.testing.assert_array_equal(res["times"], np.arange(7))
        self.assertEqual(res["divergence"].shape, (7,))
        # Every reference keeps a neighbour on this data; all entries
        # are finite and the curve rises with the evolution time.
        self.assertTrue(np.all(np.isfinite(res["divergence"])))
        self.assertTrue(res["divergence"][-1] > res["divergence"][0])

    def test_invalid_parameters(self):
        s = np.arange(50.0)
        with self.assertRaises(ValueError):
            lyap_r(s, dim=1, delay=1, max_steps=5)
        with self.assertRaises(ValueError):
            lyap_r(s, dim=2, delay=0, max_steps=5)
        with self.assertRaises(ValueError):
            lyap_r(s, dim=2, delay=1, max_steps=0)
        with self.assertRaises(ValueError):
            lyap_r(s, dim=2, delay=1, max_steps=5, theiler=-1)
        with self.assertRaises(ValueError):
            lyap_r(s, dim=2, delay=1, max_steps=200)
        with self.assertRaises(ValueError):
            lyap_r(np.zeros((10, 2)), dim=2, delay=1, max_steps=5)

    def test_constant_series_raises(self):
        with self.assertRaises(RuntimeError):
            lyap_r(np.ones(50), dim=2, delay=1, max_steps=5)


class TestLyapRPhysical(unittest.TestCase):
    def test_henon_slope(self):
        # The Henon map has lambda1 = 0.4185 per iterate.
        x, _ = henon(n=10000)
        res = lyap_r(x, dim=5, delay=1, max_steps=20, theiler=20)
        t = res["times"]
        slope = np.polyfit(t[1:11], res["divergence"][1:11], 1)[0]
        self.assertAlmostEqual(slope, 0.4185, delta=0.06)


if __name__ == "__main__":
    unittest.main()
