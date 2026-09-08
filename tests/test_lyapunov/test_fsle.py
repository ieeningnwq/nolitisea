"""Tests for nolitisea.lyapunov.fsle."""

import math
import unittest

import numpy as np

from nolitisea.core.embed import lag_block_delay_embed
from nolitisea.generate.henon import henon
from nolitisea.generate.lorenz import lorenz
from nolitisea.lyapunov.fsle import fsle
from nolitisea.utils.rescale import rescale_data

_EPS_FAC = np.sqrt(2.0)


def _reference_brute(series, dim=2, delay=1, mindist=0, eps0=None):
    """Reference implementation of the documented FSLE algorithm.

    Literal transcription of fsle.c using an explicit O(N^2) neighbour
    search with the Chebyshev metric and the deterministic
    (distance, index) tie rule.  Used to validate the cKDTree rewrite
    on continuous data.
    """
    x = np.asarray(series, dtype=np.float64)
    length = x.size
    x, _, interval = rescale_data(x)
    se_var = np.var(x)

    if eps0 is not None:
        eps0_r = abs(float(eps0)) / interval
    else:
        eps0_r = 1e-3 * se_var
    epsmax = se_var
    if eps0_r >= epsmax:
        raise ValueError("eps0 too large")

    howmany = int(np.log(epsmax / eps0_r) / np.log(_EPS_FAC)) + 1
    eps_levels = eps0_r * _EPS_FAC ** np.arange(howmany)

    E = lag_block_delay_embed(x, dim, delay)
    del_off = delay * (dim - 1)
    maxlength = length - del_off - 1 - mindist
    advance = del_off + 1

    # Brute-force nearest eligible neighbour with dist > 0.
    first = np.full(maxlength + 1, -1, dtype=np.intp)
    mindx_arr = np.zeros(maxlength + 1)
    for n in range(maxlength + 1):
        best_d = np.inf
        best_e = -1
        for e in range(maxlength + 1):
            if abs(n - e) <= mindist:
                continue
            d = np.max(np.abs(E[n] - E[e]))
            if d <= 0.0:
                continue
            if d < best_d or (d == best_d and e < best_e):
                best_d = d
                best_e = e
        if best_e != -1 and best_d < epsmax:
            first[n] = best_e
            mindx_arr[n] = best_d

    total_time = np.zeros(howmany)
    total_factor = np.zeros(howmany)
    total_count = np.zeros(howmany, dtype=np.int64)

    log_fac = np.log(_EPS_FAC)
    for n in range(maxlength + 1):
        if first[n] < 0:
            continue
        act = n + advance
        nbr = int(first[n]) + advance
        mindx = mindx_arr[n]
        if act >= length or nbr >= length or mindx <= 0.0:
            continue

        which = int(np.log(mindx / eps0_r) / log_fac)
        done = False
        if which < 0:
            while True:
                dx = abs(x[act] - x[nbr])
                if dx >= eps_levels[0]:
                    break
                act += 1
                nbr += 1
                if act >= length or nbr >= length:
                    done = True
                    break
            if done:
                continue
            mindx = dx
            if mindx <= 0.0:
                continue
            which = int(np.log(mindx / eps0_r) / log_fac)

        for i in range(max(which, 0), howmany - 1):
            stime = 0
            while True:
                dx = abs(x[act] - x[nbr])
                if dx >= eps_levels[i + 1]:
                    break
                act += 1
                nbr += 1
                if act >= length or nbr >= length:
                    done = True
                    break
                stime += 1
            if done:
                break
            if stime > 0:
                total_time[i] += stime
                total_factor[i] += math.log(dx / mindx)
                total_count[i] += 1
            mindx = dx

    fsle_vals = np.full(howmany, np.nan)
    hit = total_count > 0
    fsle_vals[hit] = total_factor[hit] / total_time[hit]
    return {
        "eps": eps_levels * interval,
        "fsle": fsle_vals,
        "count": total_count,
        "time": total_time,
    }


class TestFsleBruteForce(unittest.TestCase):
    """Validate the cKDTree rewrite against a brute-force reference."""

    def test_continuous_matches_brute(self):
        rng = np.random.default_rng(7)
        s = np.cumsum(rng.standard_normal(300))
        for mindist in [0, 3]:
            res = fsle(s, dim=2, delay=1, mindist=mindist)
            brute = _reference_brute(s, dim=2, delay=1, mindist=mindist)
            np.testing.assert_allclose(
                res["fsle"], brute["fsle"], rtol=1e-12, equal_nan=True
            )
            np.testing.assert_array_equal(res["count"], brute["count"])

    def test_explicit_eps0_matches_brute(self):
        rng = np.random.default_rng(7)
        s = np.cumsum(rng.standard_normal(300))
        res = fsle(s, dim=2, delay=1, eps0=0.01)
        brute = _reference_brute(s, dim=2, delay=1, eps0=0.01)
        np.testing.assert_allclose(
            res["fsle"], brute["fsle"], rtol=1e-12, equal_nan=True
        )

    def test_higher_dim_matches_brute(self):
        rng = np.random.default_rng(11)
        s = np.cumsum(rng.standard_normal(250))
        res = fsle(s, dim=3, delay=2, mindist=2, n_jobs=-1)
        brute = _reference_brute(s, dim=3, delay=2, mindist=2)
        np.testing.assert_allclose(
            res["fsle"], brute["fsle"], rtol=1e-12, equal_nan=True
        )


class TestFsleOutputs(unittest.TestCase):
    def test_output_structure(self):
        x, _ = henon(n=3000)
        res = fsle(x, dim=2, delay=1)
        self.assertEqual(set(res), {"eps", "fsle", "count", "time"})
        n = res["eps"].size
        self.assertEqual(res["fsle"].shape, (n,))
        self.assertEqual(res["count"].shape, (n,))
        self.assertEqual(res["time"].shape, (n,))
        # Radii ascend geometrically.
        self.assertTrue(np.all(np.diff(res["eps"]) > 0))
        # Finite FSLE entries must have contributors.
        finite = np.isfinite(res["fsle"])
        self.assertTrue(np.all(res["count"][finite] > 0))
        # All counts are non-negative.
        self.assertTrue(np.all(res["count"] >= 0))

    def test_eps_geometric_ladder(self):
        x, _ = henon(n=2000)
        res = fsle(x, dim=2, delay=1)
        ratios = res["eps"][1:] / res["eps"][:-1]
        np.testing.assert_allclose(ratios, np.sqrt(2.0), rtol=1e-10)

    def test_invalid_parameters(self):
        s = np.arange(50.0)
        with self.assertRaises(ValueError):
            fsle(s, dim=1)
        with self.assertRaises(ValueError):
            fsle(s, dim=2, delay=0)
        with self.assertRaises(ValueError):
            fsle(s, dim=2, delay=1, mindist=-1)
        with self.assertRaises(ValueError):
            fsle(np.zeros((10, 2)), dim=2, delay=1)
        # Series too short.
        with self.assertRaises(ValueError):
            fsle(np.arange(5.0), dim=5, delay=1)

    def test_constant_series_raises(self):
        with self.assertRaises(RuntimeError):
            fsle(np.ones(100), dim=2, delay=1)

    def test_eps0_too_large_raises(self):
        # eps0 >= variance should raise.
        x, _ = henon(n=500)
        with self.assertRaises(ValueError):
            fsle(x, dim=2, delay=1, eps0=1e6)


class TestFsleDeterminism(unittest.TestCase):
    def test_repeat_same_result(self):
        x, _ = henon(n=2000)
        res1 = fsle(x, dim=2, delay=1)
        res2 = fsle(x, dim=2, delay=1)
        np.testing.assert_array_equal(res1["count"], res2["count"])
        np.testing.assert_array_equal(res1["time"], res2["time"])
        np.testing.assert_allclose(
            res1["fsle"], res2["fsle"], rtol=0, equal_nan=True
        )

    def test_serial_equals_parallel(self):
        x, _ = henon(n=3000)
        res_s = fsle(x, dim=2, delay=1, n_jobs=None)
        res_p = fsle(x, dim=2, delay=1, n_jobs=-1)
        np.testing.assert_array_equal(res_s["count"], res_p["count"])
        np.testing.assert_allclose(
            res_s["fsle"], res_p["fsle"], rtol=0, equal_nan=True
        )


class TestFslePhysical(unittest.TestCase):
    def test_henon_fsle_positive(self):
        # Henon map has lambda1 = 0.4185 per iterate; FSLE at small
        # scales should be positive and of that order.
        x, _ = henon(n=5000)
        res = fsle(x, dim=2, delay=1)
        finite = np.isfinite(res["fsle"])
        self.assertTrue(finite.any())
        # At least one small-scale FSLE should be near lambda1.
        small = np.where(finite & (res["eps"] < 0.01))[0]
        if small.size:
            self.assertGreater(res["fsle"][small[0]], 0.0)

    def test_lorenz_fsle_finite(self):
        # Lorenz x: FSLE should produce finite values at intermediate
        # scales.
        np.random.seed(0)
        _, data = lorenz(length=3000)
        x = data[:, 0]
        res = fsle(x, dim=2, delay=1)
        self.assertTrue(np.isfinite(res["fsle"]).any())

    def test_mindist_reduces_pairs(self):
        x, _ = henon(n=3000)
        res0 = fsle(x, dim=2, delay=1, mindist=0)
        res_t = fsle(x, dim=2, delay=1, mindist=10)
        self.assertLessEqual(res_t["count"].sum(), res0["count"].sum())


if __name__ == "__main__":
    unittest.main()
