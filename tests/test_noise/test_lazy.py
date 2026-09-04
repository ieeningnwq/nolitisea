"""Tests for simple nonlinear noise reduction."""

import unittest

import numpy as np

from nolitisea.generate.lorenz import lorenz
from nolitisea.noise.lazy import lazy

_METRIC_P = {"cityblock": 1, "euclidean": 2, "chebyshev": np.inf}


def _reference_lazy(x, dim, delay, r, metric, repeat):
    """Straightforward brute-force reimplementation of ``lazy``.

    Builds delay vectors by plain slicing, finds neighbours with a linear
    scan, and mimics the radius update (``r`` becomes the RMS correction
    of the previous pass, stopping when the series stops changing).
    """
    p = _METRIC_P[metric]
    n = x.size
    if dim % 2 == 0:
        mid = delay * dim // 2
    else:
        mid = delay * (dim - 1) // 2
    coord = mid // delay

    y = x.astype(float).copy()
    n_vec = n - (dim - 1) * delay
    for _ in range(repeat):
        z = y.copy()
        ps = np.column_stack([y[j * delay: j * delay + n_vec]
                              for j in range(dim)])
        for i in range(n_vec):
            diff = np.abs(ps - ps[i])
            if p == np.inf:
                dist = diff.max(axis=1)
            elif p == 1:
                dist = diff.sum(axis=1)
            else:
                dist = np.sqrt((diff ** 2).sum(axis=1))
            nb = np.where(dist <= r)[0]
            y[i + mid] = ps[nb, coord].mean()
        r = np.sqrt(np.mean((y - z) ** 2))
        if r == 0:
            break
    return y


class TestLazyOutputStructure(unittest.TestCase):
    """Shape preservation, input immutability and the r=0 identity."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.x = np.sin(np.arange(200.0) * 0.1) + 0.05 * rng.standard_normal(200)

    def test_shape_and_finiteness(self):
        for kwargs in ({}, {"dim": 3, "delay": 2}, {"dim": 4, "delay": 1},
                       {"dim": 5, "delay": 2, "repeat": 3}):
            y = lazy(self.x, r=0.5, **kwargs)
            self.assertEqual(y.shape, self.x.shape, msg=str(kwargs))
            self.assertTrue(np.all(np.isfinite(y)), msg=str(kwargs))

    def test_input_not_modified(self):
        x = self.x.copy()
        lazy(x, dim=3, delay=1, r=0.5, repeat=2)
        np.testing.assert_array_equal(x, self.x)

    def test_zero_radius_is_identity(self):
        """With r=0 each neighbourhood is the point itself (continuous
        data), so the series must come back unchanged."""
        for kwargs in ({"dim": 1}, {"dim": 3, "delay": 2}, {"dim": 4, "delay": 1}):
            y = lazy(self.x, **kwargs)
            np.testing.assert_array_equal(y, self.x, err_msg=str(kwargs))

    def test_zero_radius_stops_early(self):
        """repeat must not matter when the first pass changes nothing."""
        y1 = lazy(self.x, dim=3, delay=1, r=0, repeat=1)
        y5 = lazy(self.x, dim=3, delay=1, r=0, repeat=5)
        np.testing.assert_array_equal(y1, y5)

    def test_accepts_list_input(self):
        y = lazy([0.0, 1.0, 2.0, 3.0, 4.0, 2.5, 1.5], dim=3, delay=1, r=0.5)
        self.assertEqual(y.shape, (7,))
        self.assertTrue(np.all(np.isfinite(y)))


class TestLazyDim1HandComputed(unittest.TestCase):
    """Exact hand-computed result for the 1-D embedding path.

    x = [0, 1, 2, 3, 4], r = 1: point i averages all values within
    distance 1, giving [0.5, 1, 2, 3, 3.5].
    """

    def test_chebyshev(self):
        y = lazy(np.array([0.0, 1.0, 2.0, 3.0, 4.0]), dim=1, delay=1, r=1.0)
        np.testing.assert_allclose(y, [0.5, 1.0, 2.0, 3.0, 3.5], rtol=1e-15)

    def test_metrics_agree_in_1d(self):
        """In a 1-D embedding all three metrics reduce to |x_j - x_i|."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ref = lazy(x, dim=1, delay=1, r=1.0)
        for metric in ("cityblock", "euclidean"):
            y = lazy(x, dim=1, delay=1, r=1.0, metric=metric)
            np.testing.assert_allclose(y, ref, rtol=1e-15)


class TestLazyMatchesReference(unittest.TestCase):
    """Cross-check against a brute-force reference implementation."""

    def test_matches_reference(self):
        rng = np.random.default_rng(7)
        x = np.sin(np.arange(60.0) * 0.3) + 0.1 * rng.standard_normal(60)
        cases = [
            {"dim": 3, "delay": 1, "r": 0.5, "metric": "chebyshev"},
            {"dim": 4, "delay": 1, "r": 0.6, "metric": "euclidean"},
            {"dim": 2, "delay": 2, "r": 0.8, "metric": "cityblock"},
            {"dim": 5, "delay": 2, "r": 0.5, "metric": "chebyshev"},
        ]
        for case in cases:
            y = lazy(x, repeat=2, **case)
            ref = _reference_lazy(x, repeat=2, **case)
            np.testing.assert_allclose(
                y, ref, rtol=1e-10, err_msg=str(case))


class TestLazyDenoising(unittest.TestCase):
    """Physical behaviour on noisy Lorenz data."""

    @classmethod
    def setUpClass(cls):
        _, states = lorenz(length=700, step=0.001, sample=0.03,
                           discard=3000)
        rng = np.random.default_rng(42)
        cls.clean = states[:, 0].copy()
        cls.noisy = cls.clean + rng.normal(
            0.0, 0.05 * cls.clean.std(), cls.clean.shape
        )

    @staticmethod
    def _rms(a, ref):
        return np.sqrt(np.mean((a - ref) ** 2))

    def test_reduces_noise(self):
        y = lazy(self.noisy, dim=3, delay=1, r=1.2, repeat=2)
        self.assertLess(self._rms(y, self.clean),
                        0.85 * self._rms(self.noisy, self.clean))

    def test_deterministic(self):
        a = lazy(self.noisy, dim=3, delay=1, r=1.2, repeat=2)
        b = lazy(self.noisy, dim=3, delay=1, r=1.2, repeat=2)
        np.testing.assert_array_equal(a, b)

    def test_repeat_changes_result(self):
        """After the first pass the radius becomes the RMS correction, so
        a second pass must generally differ from a single one."""
        y1 = lazy(self.noisy, dim=3, delay=1, r=1.2, repeat=1)
        y2 = lazy(self.noisy, dim=3, delay=1, r=1.2, repeat=2)
        self.assertFalse(np.array_equal(y1, y2))


class TestLazyValidation(unittest.TestCase):
    """Parameter validation and error handling."""

    def test_unknown_metric_raises(self):
        x = np.arange(20.0)
        with self.assertRaises(ValueError):
            lazy(x, dim=2, delay=1, r=0.5, metric="manhattan")

    def test_series_too_short_raises(self):
        with self.assertRaises(ValueError):
            lazy(np.arange(5.0), dim=10, delay=1, r=0.5)


if __name__ == "__main__":
    unittest.main()
