"""Tests for nolitisea.stationarity.stp."""

import unittest

import numpy as np

from nolitisea.stationarity.stp import stp


def _stp_brute(y, dim, delay, ndt, idt, perc, meps=1000):
    """Literal transcription of the ``stplot`` subroutine in stp.f.

    Bins the Chebyshev distance between each embedding vector and its
    time-shifted copy, then reads the distance at each cumulative
    fraction.  Independent of the vectorised rewrite so it can validate
    it.
    """
    y = np.asarray(y, dtype=np.float64)
    nmax = y.size
    epsmax = y.max() - y.min()
    nfrac = min(100, int(1.0 / perc))
    perc = 1.0 / nfrac
    out = np.full((nfrac, ndt), np.nan)
    for it in range(1, ndt + 1):
        lag = it * idt
        start = lag + (dim - 1) * delay  # first valid n (0-based)
        if start >= nmax:
            continue
        ihist = [0] * (meps + 1)  # 1-based, index 1..meps
        for n in range(start, nmax):
            dis = 0.0
            for me in range(dim):
                dis = max(dis, abs(y[n - me * delay] - y[n - me * delay - lag]))
            ih = min(int(meps * dis / epsmax) + 1, meps)
            ihist[ih] += 1
        n_pairs = nmax - lag - (dim - 1) * delay
        for ifrac in range(1, nfrac + 1):
            need = n_pairs * ifrac / float(nfrac)
            s = 0
            ieps = meps  # if never reached, the Fortran loop leaves ieps=meps
            for k in range(1, meps + 1):
                s += ihist[k]
                if s >= need:
                    ieps = k
                    break
            out[ifrac - 1, it - 1] = ieps * epsmax / meps
    return out


class TestStpBruteForce(unittest.TestCase):
    def test_matches_transcription_default_params(self):
        rng = np.random.default_rng(11)
        y = np.cumsum(rng.standard_normal(400))
        res = stp(y, dim=3, delay=2, max_time=40, resolution=1, fraction=0.05)
        brute = _stp_brute(y, dim=3, delay=2, ndt=40, idt=1, perc=0.05)
        np.testing.assert_allclose(res["stp"], brute, rtol=1e-12)

    def test_matches_transcription_resolution(self):
        rng = np.random.default_rng(23)
        y = np.cumsum(rng.standard_normal(500))
        res = stp(y, dim=2, delay=3, max_time=20, resolution=2, fraction=0.1)
        brute = _stp_brute(y, dim=2, delay=3, ndt=20, idt=2, perc=0.1)
        np.testing.assert_allclose(res["stp"], brute, rtol=1e-12)

    def test_matches_transcription_single_fraction(self):
        rng = np.random.default_rng(5)
        y = rng.standard_normal(300)
        # fraction=0.5 -> n_frac = min(100, 2) = 2
        res = stp(y, dim=4, delay=1, max_time=15, fraction=0.5)
        brute = _stp_brute(y, dim=4, delay=1, ndt=15, idt=1, perc=0.5)
        np.testing.assert_allclose(res["stp"], brute, rtol=1e-12)


class TestStpOutputs(unittest.TestCase):
    def test_output_structure(self):
        rng = np.random.default_rng(1)
        y = np.cumsum(rng.standard_normal(200))
        res = stp(y, dim=2, delay=1, max_time=30)
        self.assertEqual(set(res), {"times", "fractions", "stp"})
        self.assertEqual(res["times"].shape, (30,))
        self.assertEqual(res["fractions"].size, res["stp"].shape[0])
        self.assertEqual(res["stp"].shape, (res["fractions"].size, 30))

    def test_fractions_and_times(self):
        rng = np.random.default_rng(2)
        y = rng.standard_normal(150)
        res = stp(y, dim=2, delay=1, max_time=10, resolution=3, fraction=0.1)
        np.testing.assert_array_equal(res["times"], np.arange(1, 11) * 3)
        # fraction=0.1 -> n_frac = min(100, 10) = 10, levels 0.1..1.0
        np.testing.assert_allclose(
            res["fractions"], np.arange(1, 11) / 10.0, rtol=1e-12
        )

    def test_monotone_in_fraction(self):
        # For a fixed time shift, larger fraction -> larger distance.
        rng = np.random.default_rng(7)
        y = np.cumsum(rng.standard_normal(300))
        res = stp(y, dim=3, delay=2, max_time=25)
        col = res["stp"][:, 10]
        finite = np.isfinite(col)
        col = col[finite]
        self.assertTrue(np.all(np.diff(col) >= -1e-12))

    def test_short_shift_nan_for_too_large(self):
        # When the shift exceeds the series, the column stays nan.
        rng = np.random.default_rng(9)
        y = rng.standard_normal(20)
        res = stp(y, dim=2, delay=1, max_time=50)
        # dim=2, delay=1 -> pstart=1; lag + pstart >= 20 for lag >= 19
        # so columns for t >= 19 (times[t-1] >= 19) are nan.
        for j in range(res["stp"].shape[1]):
            t = res["times"][j]
            col = res["stp"][:, j]
            if t + 1 >= 20:  # lag + pstart >= nmax
                self.assertTrue(np.all(~np.isfinite(col)))

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            stp([1, 2, 3], dim=0, delay=1)
        with self.assertRaises(ValueError):
            stp([1, 2, 3], dim=2, delay=0)
        with self.assertRaises(ValueError):
            stp([1, 2, 3], dim=2, delay=1, resolution=0)
        with self.assertRaises(ValueError):
            stp([1, 2, 3], dim=2, delay=1, max_time=0)
        with self.assertRaises(ValueError):
            stp([1, 2, 3], dim=2, delay=1, fraction=0.0)
        with self.assertRaises(ValueError):
            stp(np.zeros((5, 2)), dim=2, delay=1)

    def test_constant_series_raises(self):
        with self.assertRaises(RuntimeError):
            stp(np.ones(50), dim=2, delay=1)


if __name__ == "__main__":
    unittest.main()
