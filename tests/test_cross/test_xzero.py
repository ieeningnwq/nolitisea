"""Tests for nolitisea.cross.xzero."""

import unittest

import numpy as np

from nolitisea.cross.xzero import cross_zeroth
from nolitisea.generate.henon import henon
from nolitisea.utils.rescale import rescale_data


def _c_transcription(a, b, dim, delay, eps, minn, epsf, step, n_refs=None):
    """Literal transcription of xzero.c with a brute neighbour search."""
    x = np.asarray(a, dtype=np.float64).ravel().copy()
    y = np.asarray(b, dtype=np.float64).ravel().copy()
    n = x.size
    x, _, range_a = rescale_data(x)
    y, _, range_b = rescale_data(y)
    interval = (range_a + range_b) / 2.0
    av2 = y.mean()
    rms2 = np.sqrt(abs(np.mean(y * y) - av2 * av2))
    eps0 = 1e-3 if eps is None else eps / interval
    clength = (n_refs if n_refs is not None and n_refs <= n else n) - step
    emb_off = (dim - 1) * delay

    error = np.zeros(step)
    done = np.zeros(n, dtype=bool)
    epsilon = eps0 / epsf
    alldone = False
    while not alldone:
        alldone = True
        epsilon *= epsf
        for i in range(emb_off, clength):
            if done[i]:
                continue
            found = []
            for element in range(emb_off, n - step):
                ok = True
                for k in range(dim):
                    if abs(y[i - k * delay] - x[element - k * delay]) > epsilon:
                        ok = False
                        break
                if ok:
                    found.append(element)
            if len(found) >= minn:
                for j in range(1, step + 1):
                    casted = sum(x[e + j] for e in found) / len(found)
                    error[j - 1] += (casted - y[i + j]) ** 2
                done[i] = True
            alldone = alldone and done[i]
    return np.sqrt(error / (clength - emb_off)) / rms2


def _henon_series(n=300, seed=0):
    x, _ = henon(n=n, discard=200 + 1000 * seed)
    return x


class TestCrossZeroth(unittest.TestCase):
    def test_matches_c_transcription_default(self):
        a = _henon_series(seed=0, n=250)
        b = _henon_series(seed=1, n=250)
        result = cross_zeroth(a, b)
        expected = _c_transcription(a, b, 3, 1, None, 30, 1.2, 1)
        np.testing.assert_allclose(result["error"], expected, atol=1e-12)
        np.testing.assert_array_equal(result["steps"], [1])

    def test_matches_c_transcription_multi_step(self):
        a = _henon_series(seed=0, n=250)
        b = _henon_series(seed=1, n=250)
        result = cross_zeroth(a, b, dim=4, delay=2, eps=0.4,
                              n_neighbors=10, eps_factor=1.5, n_steps=3)
        expected = _c_transcription(a, b, 4, 2, 0.4, 10, 1.5, 3)
        np.testing.assert_allclose(result["error"], expected, atol=1e-12)
        np.testing.assert_array_equal(result["steps"], [1, 2, 3])

    def test_matches_c_transcription_n_refs(self):
        a = _henon_series(seed=0, n=250)
        b = _henon_series(seed=1, n=250)
        result = cross_zeroth(a, b, n_refs=150)
        expected = _c_transcription(a, b, 3, 1, None, 30, 1.2, 1, n_refs=150)
        np.testing.assert_allclose(result["error"], expected, atol=1e-12)

    def test_eps_interpreted_in_original_units(self):
        a = _henon_series(seed=0, n=250)
        b = _henon_series(seed=1, n=250)
        _, _, range_a = rescale_data(np.asarray(a, dtype=np.float64))
        _, _, range_b = rescale_data(np.asarray(b, dtype=np.float64))
        interval = (range_a + range_b) / 2.0
        explicit = cross_zeroth(a, b, eps=interval * 1e-3)
        default = cross_zeroth(a, b)
        np.testing.assert_allclose(explicit["error"], default["error"],
                                   rtol=1e-10)

    def test_self_prediction_beats_independent(self):
        a = _henon_series(seed=0, n=300)
        b = _henon_series(seed=1, n=300)
        same = cross_zeroth(a, a, dim=3, delay=1)["error"][0]
        crossed = cross_zeroth(a, b, dim=3, delay=1)["error"][0]
        # same attractor dynamics: cross prediction works well
        self.assertLess(crossed, 0.5)
        # predicting a series from itself is at least as good
        self.assertLessEqual(same, crossed + 1e-9)

    def test_shuffled_target_gives_error_near_one(self):
        a = _henon_series(seed=0, n=400)
        b = _henon_series(seed=1, n=400)
        rng = np.random.default_rng(17)
        shuffled = rng.permutation(b)
        crossed = cross_zeroth(a, b, dim=3, delay=1)["error"][0]
        broken = cross_zeroth(a, shuffled, dim=3, delay=1)["error"][0]
        self.assertGreater(broken, crossed)
        self.assertGreater(broken, 0.9)

    def test_missing_neighbors_raises_runtime_error(self):
        a = _henon_series(seed=0, n=120)
        with self.assertRaises(RuntimeError):
            cross_zeroth(a, a, n_neighbors=10 ** 6)

    def test_length_mismatch_raises(self):
        a = _henon_series(n=120)
        with self.assertRaises(ValueError):
            cross_zeroth(a, a[:60])

    def test_invalid_parameters_raise(self):
        a = _henon_series(n=120)
        b = _henon_series(n=120)
        with self.assertRaises(ValueError):
            cross_zeroth(a, b, dim=0)
        with self.assertRaises(ValueError):
            cross_zeroth(a, b, delay=0)
        with self.assertRaises(ValueError):
            cross_zeroth(a, b, n_steps=0)
        with self.assertRaises(ValueError):
            cross_zeroth(a, b, n_neighbors=0)
        with self.assertRaises(ValueError):
            cross_zeroth(a, b, eps_factor=1.0)
        with self.assertRaises(ValueError):
            cross_zeroth(a, b, eps=0.0)
        with self.assertRaises(ValueError):
            cross_zeroth(a, b, n_refs=1)  # n_refs <= n_steps
        with self.assertRaises(ValueError):
            cross_zeroth(a, b, dim=200)  # no reference points left

    def test_constant_series_raises(self):
        a = _henon_series(n=120)
        with self.assertRaises(RuntimeError):
            cross_zeroth(a, np.ones(120))

if __name__ == "__main__":
    unittest.main()
